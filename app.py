# Purpose: Streamlit main app – lean version
# How to change later:
# - To adjust page title/icon/layout: edit st.set_page_config below.
# - To add/remove pages: edit `menu` and the page blocks.
# - To change default data folder: edit DATA_DIR.

from __future__ import annotations
import re
import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np

from data_loader import (
    load_bookmark,             # bookmark.csv (optional but preferred)
    load_bookmark_map,         # -> dict, standard_key -> actual column
    load_population_agg,       # population.csv
    load_party_labels,         # party_labels.csv
    load_vote_trend,           # vote_trend.csv
    load_results_2024,         # 5_na_dis_results.csv
    load_current_info,         # current_info.csv
    load_index_sample,         # index_sample.csv
)

from charts import (
    render_population_box,
    render_vote_trend_chart,
    render_results_2024_card,
    render_incumbent_card,
    render_prg_party_box,
    render_region_detail_layout,
)

# ===== Absolute Maximum Scores for Bar Chart Scaling (CRITICAL: Needs User Input) =====
# 4. The user requested to set the bar chart max length based on an absolute max score
#    instead of the data's max value. These are placeholder values (100.0) and MUST be 
#    updated with the actual max scores provided by the user.
ABSOLUTE_MAX_SCORES = {
    # Main Scores (Using 100.0 as a temporary placeholder)
    "합계": 100.0,
    "유권자환경": 100.0,
    "정치지형": 100.0,
    "주체역량": 100.0,
    "상대역량": 100.0,
    # Detailed Indices (Using 100.0 as a temporary placeholder for all detailed indices)
    # If the user specifies different max scores per detailed index, they should be added here.
}
# Fallback Max Score if specific max is not defined for a column
FALLBACK_MAX = 100.0 
# ======================================================================================

# ===== Style Configurations =====
# 2. Consistent width for the region label column
REGION_COL_WIDTH = "150px"
# 3. Highlight color for top 3 cells
TOP3_HIGHLIGHT_COLOR = "#FFF9C4" # Light Gold/Yellow background
# Colors for the main scoring bars
BAR_COLORS_MAIN = {
    "합계": "#3498DB", # Blue
    "유권자환경": "#48C9B0", # Light Cyan
    "정치지형": "#1ABC9C", # Green
    "주체역량": "#76D7C4", # Very Light Green
    "상대역량": "#2ECC71", # Emerald Green
}
# Colors for the detailed index bars
BAR_COLORS_DETAIL = {
    "유권자환경": "#00CC99", # Green
    "정치지형": "#3498DB", # Blue
    "주체역량": "#E74C3C", # Red
    "상대역량": "#F39C12", # Orange
}
# ==============================


# ===== Add these small cached helpers =====
@st.cache_data(show_spinner=False)
def _read_scoring_cached(path_str: str) -> pd.DataFrame:
    """How to change later: if your scoring file is TSV, set sep='\\t' directly below."""
    p = Path(path_str)
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, encoding="utf-8-sig")
        if df.shape[1] == 1:  # possible TSV
            df = pd.read_csv(p, encoding="utf-8-sig", sep="\t")
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def _read_markdown_cached(path_str: str) -> str | None:
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8-sig")
    except Exception:
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return None
# ===== end helpers =====


# --------------------------------
# Page Config
# --------------------------------
APP_TITLE = "지역구 선정 1단계 조사 결과"
st.set_page_config(page_title=APP_TITLE, page_icon="🗳️", layout="wide")

# --------------------------------
# Sidebar (Navigation)
# --------------------------------
st.sidebar.header("메뉴 선택")
menu = st.sidebar.radio("페이지", ["종합", "지역별 분석", "데이터 설명"], index=0)

DATA_DIR = Path("data")

# --------------------------------
# Small utils (kept minimal)
# --------------------------------
CODE_CANDIDATES = ["코드", "지역구코드", "선거구코드", "지역코드", "code", "CODE"]
NAME_CANDIDATES = ["지역구", "선거구", "선거구명", "지역명", "district", "지역구명", "region", "지역"]
SIDO_CANDIDATES = ["시/도", "시도", "광역", "sido", "province"]

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df
    out = df.copy()
    out.columns = [str(c).strip().replace("\n", "").replace("\r", "") for c in out.columns]
    return out

def _detect_col(df: pd.DataFrame, candidates: list) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    cols = [str(c).strip().replace("\n", "").replace("\r", "") for c in df.columns]
    for cand in candidates:
        if cand in cols:
            return df.columns[cols.index(cand)]
    return None

def _canon_code(x: object) -> str:
    s = str(x).strip()
    s = re.sub(r"[^0-9A-Za-z]", "", s)
    s = s.lstrip("0")
    return s.lower()

def ensure_code_col(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df
    df2 = _normalize_columns(df)
    if "코드" not in df2.columns:
        found = _detect_col(df2, CODE_CANDIDATES)
        if found:
            df2 = df2.rename(columns={found: "코드"})
    if "코드" not in df2.columns:
        idx_name = df2.index.name
        if idx_name and idx_name in CODE_CANDIDATES + ["코드"]:
            df2 = df2.reset_index().rename(columns={idx_name: "코드"})
    if "코드" in df2.columns:
        df2["코드"] = df2["코드"].astype(str)
    return df2

def get_by_code(df: pd.DataFrame, code: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df2 = _normalize_columns(df)
    code_col = "코드" if "코드" in df2.columns else _detect_col(df2, CODE_CANDIDATES)
    if not code_col:
        return pd.DataFrame()
    key = _canon_code(code)
    try:
        sub = df2[df2[code_col].astype(str).map(_canon_code) == key]
        return sub if not sub.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def build_regions(primary_df: pd.DataFrame, *fallback_dfs: pd.DataFrame, bookmark_map: dict | None = None) -> pd.DataFrame:
    """
    Build region list for selectbox. bookmark_map can specify 'code', 'region', 'sido'.
    """
    base = next((d for d in (primary_df, *fallback_dfs) if isinstance(d, pd.DataFrame) and not d.empty), None)
    if base is None:
        return pd.DataFrame(columns=["코드", "라벨"])
    dfp = ensure_code_col(_normalize_columns(base))

    # bookmark-first
    name_col = None
    sido_col = None
    if bookmark_map:
        name_col = bookmark_map.get("region") if bookmark_map.get("region") in dfp.columns else None
        sido_col = bookmark_map.get("sido") if bookmark_map.get("sido") in dfp.columns else None

    # fallback autodetect
    if not name_col:
        name_col = _detect_col(dfp, NAME_CANDIDATES)
    if not name_col:
        return dfp.loc[:, ["코드"]].assign(라벨=dfp["코드"]).drop_duplicates().sort_values("라벨").reset_index(drop=True)

    if not sido_col:
        sido_col = _detect_col(dfp, SIDO_CANDIDATES)

    def _label(row):
        nm = str(row[name_col]).strip()
        if sido_col and sido_col in row.index and pd.notna(row[sido_col]):
            sido = str(row[sido_col]).strip()
            return nm if nm.startswith(sido) else f"{sido} {nm}"
        return nm

    out = (
        dfp.assign(라벨=dfp.apply(_label, axis=1))
           .loc[:, ["코드", "라벨"]]
           .drop_duplicates()
           .sort_values("라벨")
           .reset_index(drop=True)
    )
    return out

# --------------------------------
# Core Logic for Table Rendering (Refactored to support absolute max and top 3 highlight)
# --------------------------------

# 5. Helper to format numeric values based on assumed type (Count vs. Ratio/Score)
def _format_value(val: float | object, col_name: str) -> str:
    """
    Formats the value: comma for counts, two decimals for scores/ratios.
    """
    try:
        v = float(val)
        if np.isnan(v):
            return ""
    except Exception:
        return str(val)

    # Heuristic for Count-like data (should use comma)
    count_names = ["유권자 수", "유동인구", "진보당 당원수"]
    if col_name in count_names:
        return f"{int(round(v)):,d}" # Round to nearest int and apply comma
    
    # Default for Scores/Ratios (should use two decimal places)
    return f"{v:.2f}"

def _bar_cell_factory(score_df: pd.DataFrame, score_cols: list[str], bar_colors: dict) -> callable:
    """
    Generates an HTML bar cell function, incorporating absolute max scaling, top 3 highlighting,
    and number formatting.
    """
    
    # 3. Pre-calculate top 3 values for each column
    top3_values = {}
    for col in score_cols:
        # Use abs() to handle negative scores correctly if necessary, then sort descending
        top3_values[col] = set(score_df.nlargest(3, col, keep='all')[col].tolist())
    
    def _bar_cell(val, col):
        """
        Renders a single cell with an HTML bar, absolute scaling, value formatting, and highlight.
        """
        try:
            v = float(val)
        except Exception:
            return f"<span style='font-size:12px;font-weight:600;'>{val}</span>"

        if np.isnan(v):
             return ""
        
        # 4. Absolute Scaling: Use ABSOLUTE_MAX_SCORES
        max_score = ABSOLUTE_MAX_SCORES.get(col, FALLBACK_MAX)
        max_score = max(1.0, max_score) # Ensure max_score is at least 1.0 to prevent division by zero

        # Calculate percentage based on absolute max
        pct = max(0.0, min(100.0, (v / max_score) * 100.0))
        
        # 3. Check for Top 3 highlight
        is_top3 = col in top3_values and v in top3_values[col]
        cell_style = f"padding:6px 8px; {'background:' + TOP3_HIGHLIGHT_COLOR + ';' if is_top3 else ''}"
        
        # Determine bar color
        color = bar_colors.get(col, "#6B7280")
        
        # 5. Format value display
        formatted_value = _format_value(v, col)

        # HTML Structure
        return (
            f'<div style="{cell_style}">'
            f'<div style="position:relative;width:100%;background:#F3F4F6;height:18px;border-radius:4px;overflow:hidden;min-width:50px;">'
            f'  <div style="width:{pct:.2f}%;height:100%;background:{color};"></div>'
            f'  <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;'
            f'font-size:12px;font-weight:600;color:#111827;">{formatted_value}</div>'
            f'</div>'
            f'</div>'
        )
    return _bar_cell

# --------------------------------
# Load Data (single pass, bookmark-first)
# --------------------------------
with st.spinner("데이터 불러오는 중..."):
    df_bookmark = load_bookmark(DATA_DIR)              # may be empty
    bookmark_map = load_bookmark_map(df_bookmark)      # dict or {}

    df_pop   = ensure_code_col(load_population_agg(DATA_DIR))
    df_party = ensure_code_col(load_party_labels(DATA_DIR))
    df_trend = ensure_code_col(load_vote_trend(DATA_DIR))
    df_24    = ensure_code_col(load_results_2024(DATA_DIR))
    df_curr  = ensure_code_col(load_current_info(DATA_DIR))
    df_idx   = ensure_code_col(load_index_sample(DATA_DIR))

# --------------------------------
# Page: 종합
# --------------------------------
if menu == "종합":
    st.title("🗳️ 지역구 선정 1단계 조사 결과")
    
    # 1. Add spacing and divider after the main title
    st.write("")
    st.divider()

    csv_path = Path("data/scoring.csv")
    if not csv_path.exists():
        st.error("`data/scoring.csv`를 찾을 수 없습니다. (경로 고정)")
        st.stop()

    # --- Load CSV (utf-8-sig 우선, tsv fallback) ---
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as e:
        st.error(f"`data/scoring.csv` 읽기 실패: {e}")
        st.stop()

    if df.shape[1] == 1:  # TSV fallback
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig", sep="\t")
        except Exception as e:
            st.error(f"`data/scoring.csv` 구분자 문제: {e}")
            st.stop()

    # --- Always treat first column as region ---
    df = _normalize_columns(df)
    df.rename(columns={df.columns[0]: "region"}, inplace=True)
    label_col = "region"

    # --- Detect numeric columns (all except first) ---
    score_cols = [c for c in df.columns if c != label_col]
    df[score_cols] = df[score_cols].apply(pd.to_numeric, errors="coerce")
    
    # Instantiate the bar cell factory for the main table
    _bar_cell = _bar_cell_factory(df, score_cols, BAR_COLORS_MAIN)

    # --- Build HTML table for main scoring ---
    headers = [label_col] + score_cols
    
    # 2. Set width style for region column header
    thead = (
        f"<th style='text-align:left;padding:6px 8px;white-space:nowrap;width:{REGION_COL_WIDTH};'>"
        f"{label_col}</th>"
    )
    # Remaining headers should be equally wide
    remaining_cols_count = len(score_cols)
    col_width_pct = f"{100 / remaining_cols_count}%" if remaining_cols_count > 0 else "auto"
    
    thead += "".join(
        [
            f"<th style='text-align:left;padding:6px 8px;white-space:nowrap;width:{col_width_pct};'>{h}</th>" 
            for h in score_cols
        ]
    )

    rows_html = []
    for _, row in df.iterrows():
        # 2. Set width style for region column cell
        cells = [
            f"<td style='padding:6px 8px;white-space:nowrap;width:{REGION_COL_WIDTH};'>"
            f"<span style='font-size:13px;'>{row[label_col]}</span>"
            f"</td>"
        ]
        
        for c in score_cols:
            cells.append(f"<td style='padding:0px;width:{col_width_pct};'>{_bar_cell(row[c], c)}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    table_html = (
        "<div style='overflow-x:auto;'>"
        "<table style='border-collapse:separate;border-spacing:0;width:100%;font-size:13px;'>"
        f"<thead><tr>{thead}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>"
        "</div>"
    )

    st.markdown(table_html, unsafe_allow_html=True)
    
    # ====================================================================
    # [NEW SECTION] 세부 지표별 상세 분석 (Index Sample)
    # ====================================================================
    st.divider()
    st.subheader("세부 지표별 상세 분석")

    # 1. 지표 그룹 정의
    INDICATOR_GROUPS = {
        "유권자환경": ["유권자 수", "유동인구", "고령층 비율", "청년층 비율", "4-50대 비율", "2030여성 비율"],
        "정치지형": ["유동성A", "경합도A", "유동성B", "경합도B"],
        "주체역량": ["진보정당 득표력", "진보당 당원수", "진보당 지방선거 후보 수"],
        "상대역량": ["현직 득표력", "민주당 득표력", "보수 득표력"],
    }
    
    # 2. 그룹 선택 Tabs (시각적으로 더 깔끔하게 구성)
    tab_titles = list(INDICATOR_GROUPS.keys())
    tabs = st.tabs(tab_titles)
    
    # 탭별로 내용 렌더링
    for selected_group, tab in zip(tab_titles, tabs):
        with tab:
            target_cols = INDICATOR_GROUPS.get(selected_group, [])
            
            if not df_idx.empty and target_cols:
                df_idx_norm = _normalize_columns(df_idx)
                
                # 3. 데이터 준비 및 필터링
                regions_map = build_regions(df_idx_norm, bookmark_map=bookmark_map)
                
                df_display = pd.merge(
                    df_idx_norm, 
                    regions_map.rename(columns={"라벨": "region_display"}), 
                    on="코드", 
                    how="left"
                )
                
                label_col_new = "region_display"
                present_cols = [c for c in target_cols if c in df_display.columns]
                
                if not present_cols:
                    st.info(f"선택된 그룹 ({selected_group})에 해당하는 컬럼이 데이터에 없습니다.")
                else:
                    # 4. 차트 데이터 준비 (모든 지역 표시 - .head(10) 제거)
                    df_final = df_display.loc[:, [label_col_new, "코드"] + present_cols].copy()
                    
                    df_final[present_cols] = df_final[present_cols].apply(pd.to_numeric, errors="coerce")
                    df_final = df_final.dropna(subset=[label_col_new]).dropna(subset=present_cols, how='all')
                    
                    # 지역 제한 없이 모든 데이터를 표시
                    df_final = df_final.reset_index(drop=True)
                    
                    # Determine bar color for this group
                    bar_color_new = BAR_COLORS_DETAIL.get(selected_group, "#6B7280")
                    
                    # Instantiate the bar cell factory for the detailed table
                    # Note: We pass the color explicitly as a list for _bar_cell_factory to handle
                    color_dict = {col: bar_color_new for col in present_cols}
                    _bar_cell_detail = _bar_cell_factory(df_final, present_cols, color_dict)


                    # 6. 상세 분석 HTML 테이블 생성
                    headers_new = [label_col_new] + present_cols
                    
                    # 2. Set width style for region column header
                    thead_new = (
                        f"<th style='text-align:left;padding:6px 8px;white-space:nowrap;font-weight:700;width:{REGION_COL_WIDTH};'>"
                        f"{label_col_new}</th>"
                    )
                    # Remaining headers should be equally wide
                    remaining_cols_count_new = len(present_cols)
                    col_width_pct_new = f"{100 / remaining_cols_count_new}%" if remaining_cols_count_new > 0 else "auto"

                    thead_new += "".join(
                        [
                            f"<th style='text-align:left;padding:6px 8px;white-space:nowrap;width:{col_width_pct_new};'>{h}</th>" 
                            for h in present_cols
                        ]
                    )

                    rows_html_new = []
                    for _, row in df_final.iterrows():
                        # 2. Set width style for region column cell
                        cells = [
                            f"<td style='padding:6px 8px;white-space:nowrap;width:{REGION_COL_WIDTH};'>"
                            f"<span style='font-size:13px;'>{row[label_col_new]}</span>"
                            f"</td>"
                        ]
                        for c in present_cols:
                            cells.append(f"<td style='padding:0px;width:{col_width_pct_new};'>{_bar_cell_detail(row[c], c)}</td>") 
                        rows_html_new.append("<tr>" + "".join(cells) + "</tr>")

                    table_html_new = (
                        "<div style='overflow-x:auto;'>"
                        "<table style='border-collapse:separate;border-spacing:0;width:100%;font-size:13px;'>"
                        f"<thead><tr>{thead_new}</tr></thead>"
                        f"<tbody>{''.join(rows_html_new)}</tbody>"
                        "</table>"
                        "</div>"
                    )

                    st.markdown(table_html_new, unsafe_allow_html=True)

            else:
                if df_idx.empty:
                    st.info("`index_sample.csv` 데이터프레임이 비어 있어 상세 분석을 렌더링할 수 없습니다.")
                else:
                     st.info(f"선택된 그룹 ({selected_group})에 표시할 컬럼이 없습니다. (데이터 컬럼명 확인 필요)")

# --------------------------------
# Page: 지역별 분석
# --------------------------------
elif menu == "지역별 분석":
    regions = build_regions(df_pop, df_trend, df_24, df_curr, bookmark_map=bookmark_map)
    if regions.empty:
        st.subheader("지역 목록을 만들 수 없습니다. (코드/지역명 컬럼 미탐지)")
        st.stop()

    PLACEHOLDER = "— 지역을 선택하세요 —"
    options = [PLACEHOLDER] + regions["라벨"].tolist()

    st.sidebar.header("지역 선택")
    sel_label = st.sidebar.selectbox("선거구를 선택하세요", options, index=0)
    if sel_label == PLACEHOLDER:
        st.subheader("지역을 선택하세요")
        st.stop()

    sel_code = regions.loc[regions["라벨"] == sel_label, "코드"].iloc[0]

    # Selected vs All (pass both – charts do no I/O)
    pop_sel   = get_by_code(df_pop, sel_code)
    trend_sel = get_by_code(df_trend, sel_code) if "코드" in df_trend.columns else df_trend
    res_sel   = get_by_code(df_24, sel_code)
    cur_sel   = get_by_code(df_curr, sel_code)
    prg_sel   = get_by_code(df_idx, sel_code)   # index_sample as "prg data"

    render_region_detail_layout(
        df_pop_sel=pop_sel,
        df_pop_all=df_pop,
        df_trend_sel=trend_sel,
        df_trend_all=df_trend,
        df_24_sel=res_sel,
        df_24_all=df_24,
        df_cur_sel=cur_sel,
        df_idx_sel=prg_sel,
        df_idx_all=df_idx,
        bookmark_map=bookmark_map,
        page_title=sel_label,
        app_title=APP_TITLE,
    )

# --------------------------------
# Page: 데이터 설명
# --------------------------------
elif menu == "데이터 설명":
    # 좌: 큰 제목 / 우: 앱 제목 (오른쪽 상단 고정)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.title("📘 지표별 구성 및 해설")
    with c2:
        st.markdown(
            """
            <div style="text-align:right; font-weight:700; font-size:1.05rem;">
                🗳️ 지역구 선정 1단계 조사 결과
            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------
    # 지표별 구성 및 해설 (외부 MD 파일 렌더)
    # -----------------------------
    st.divider()

    md_candidates = [
        Path("sti") / "지표별 구성 및 해설.md",
        Path("지표별 구성 및 해설.md"),
        Path("/mnt/data/sti/지표별 구성 및 해설.md"),
    ]
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr"]

    md_text = None
    md_path_used = None
    for p in md_candidates:
        if p.exists():
            for enc in encodings:
                try:
                    md_text = p.read_text(encoding=enc)
                    md_path_used = p
                    break
                except Exception:
                    continue
            if md_text is not None:
                break

    if md_text:
        st.markdown(md_text)
    else:
        st.info("`sti/지표별 구성 및 해설.md` 파일을 찾지 못했습니다. 경로 또는 파일명을 확인해 주세요.")
        
# --------------------------------
# Footer
# --------------------------------
st.write("")
st.caption("© 2025 전략지역구 조사 · 에스티아이")
