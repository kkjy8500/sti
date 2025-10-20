# app.py
# Purpose: Streamlit main app - displays political analysis data in interactive tables and charts.
# Key Features: Two main tables on the 'Summary' page (종합).
# 1. Main Summary Table: Features score bars, dynamic Top 3 highlights, and fixed row highlights.
# 2. Detailed Index Table: Displays raw numeric data without bars or highlights for detailed inspection.

from __future__ import annotations
import re
import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np

# Data Loader Imports (Korean function names are maintained as per the original structure)
from data_loader import (
    load_bookmark,              # bookmark.csv (optional but preferred)
    load_bookmark_map,          # -> dict, standard_key -> actual column
    load_population_agg,        # population.csv
    load_party_labels,          # party_labels.csv
    load_vote_trend,            # vote_trend.csv
    load_results_2024,          # 5_na_dis_results.csv
    load_current_info,          # current_info.csv
    load_index_sample,          # index_sample.csv
)

# Chart Renderer Imports (Korean function names are maintained as per the original structure)
from charts import (
    render_population_box,
    render_vote_trend_chart,
    render_results_2024_card,
    render_incumbent_card,
    render_prg_party_box,
    render_region_detail_layout,
)

# ====================================================================
# CONFIGURATION CONSTANTS (MUST BE DEFINED EARLY)
# ====================================================================
APP_TITLE = "지역구 선정 1단계 조사 결과"
DATA_DIR = Path("data")  # [File root] change here if your relative data folder moves.

# Absolute Scaling: Columns listed here will use the specified value as the 100% max score
ABSOLUTE_MAX_SCORES = {
    "합계": 100.0,
    "유권자환경": 20.0,
    "정치지형": 20.0,
    "주체역량": 30.0,
    "상대역량": 30.0,
    "고령층 비율": 1.0,
    "청년층 비율": 1.0,
    "4-50대 비율": 1.0,
    "2030여성 비율": 1.0,
    "진보정당 득표력": 10.0,
    "현직 득표력": 100.0,
    "민주당 득표력": 100.0,
    "보수 득표력": 100.0,
}

# ===== Style Configurations (English Comments for Maintainability) =====
REGION_COL_WIDTH = "150px"  # [Spacing] Fixed width for region name column.

# Fixed highlight rows in Summary table
FIXED_HIGHLIGHT_REGIONS = ["서울 서대문구갑", "경기 평택시을", "경기 화성시을"]
FIXED_HIGHLIGHT_ROW_BG = "#FFF9C4"  # [Color] Light yellow row highlight.
DYNAMIC_HIGHLIGHT_CELL_BG = "#E0F2FE"  # [Color] Light sky-blue for Top 3 cells.

# [Color] Score bar colors in the Summary table
BAR_COLORS_MAIN = {
    "합계": "#3498DB",
    "유권자환경": "#48C9B0",
    "정치지형": "#1ABC9C",
    "주체역량": "#76D7C4",
    "상대역량": "#2ECC71",
}

# ====================================================================
# Utilities
# ====================================================================
@st.cache_data(show_spinner=False)
def _read_scoring_cached(path_str: str) -> pd.DataFrame:
    """Read scoring CSV/TSV with caching."""
    p = Path(path_str)
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, encoding="utf-8-sig")
        if df.shape[1] == 1:
            df = pd.read_csv(p, encoding="utf-8-sig", sep="\t")
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def _read_markdown_cached(path_str: str) -> str | None:
    """Read Markdown with caching."""
    p = Path(path_str)
    if not p.exists():
        return None
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return p.read_text(encoding=enc)
        except Exception:
            continue
    return None

CODE_CANDIDATES = ["코드", "지역구코드", "선거구코드", "지역코드", "code", "CODE"]
NAME_CANDIDATES = ["지역구", "선거구", "선거구명", "지역명", "district", "지역구명", "region", "지역"]
SIDO_CANDIDATES = ["시/도", "시도", "광역", "sido", "province"]

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespaces/newlines from headers."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c).strip().replace("\n", "").replace("\r", "") for c in out.columns]
    return out

def _detect_col(df: pd.DataFrame, candidates: list) -> str | None:
    """Return first matching header."""
    cols = [str(c).strip().replace("\n", "").replace("\r", "") for c in df.columns]
    for cand in candidates:
        if cand in cols:
            return df.columns[cols.index(cand)]
    return None

def _canon_code(x: object) -> str:
    """Standardize code (strip non-alnum, trim leading zeros)."""
    s = str(x).strip()
    s = re.sub(r"[^0-9A-Za-z]", "", s)
    s = s.lstrip("0")
    return s.lower()

def ensure_code_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure '코드' column exists by renaming detected code column."""
    if df is None or df.empty:
        return pd.DataFrame()
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
    """Filter DF by standardized '코드'."""
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
    """Build (코드, 라벨) list for region selector."""
    base = next((d for d in (primary_df, *fallback_dfs) if isinstance(d, pd.DataFrame) and not d.empty), None)
    if base is None:
        return pd.DataFrame(columns=["코드", "라벨"])
    dfp = ensure_code_col(_normalize_columns(base))

    name_col = None
    sido_col = None
    if bookmark_map:
        name_col = bookmark_map.get("region") if bookmark_map.get("region") in dfp.columns else None
        sido_col = bookmark_map.get("sido") if bookmark_map.get("sido") in dfp.columns else None

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

# ---------- [Bar & Text cell renderers for tables] ----------
def _format_value(val: float | object, col_name: str) -> str:
    """Comma for counts; 2 decimals for ratios/scores."""
    try:
        v = float(val)
        if np.isnan(v):
            return ""
    except Exception:
        return str(val)
    count_names = ["유권자 수", "유동인구", "진보당 당원수", "진보당 지방선거 후보 수", "유동성A", "유동성B"]
    if col_name in count_names:
        return f"{int(round(v)):,d}"
    return f"{v:.2f}"

def _bar_cell_factory(score_df: pd.DataFrame, score_cols: list[str], bar_colors: dict) -> callable:
    """HTML bar cell with Top3 highlight & absolute/dynamic scaling."""
    top3_values = {}
    for col in score_cols:
        try:
            top3_values[col] = set(score_df.nlargest(3, col, keep='all')[col].tolist())
        except KeyError:
            top3_values[col] = set()
    dynamic_maxes = {col: score_df[col].max() for col in score_cols if col not in ABSOLUTE_MAX_SCORES}

    def _bar_cell(val, col):
        try:
            v = float(val)
        except Exception:
            return f"<span style='font-size:12px;font-weight:600;'>{val}</span>"
        if np.isnan(v):
            return ""
        max_score = ABSOLUTE_MAX_SCORES.get(col, dynamic_maxes.get(col, 1.0))
        max_score = max(1.0, max_score)
        pct = max(0.0, min(100.0, (v / max_score) * 100.0))
        is_top3 = col in top3_values and v in top3_values[col]
        color = bar_colors.get(col, "#6B7280")
        container_bg = DYNAMIC_HIGHLIGHT_CELL_BG if is_top3 else "#F3F4F6"  # [Color] change highlight bg later

        formatted_value = _format_value(v, col)
        return (
            f'<div style="padding:6px 8px; height:100%; box-sizing:border-box;">'
            f'<div style="position:relative;width:100%;background:{container_bg};height:18px;border-radius:4px;overflow:hidden;min-width:50px;">'
            f'  <div style="width:{pct:.2f}%;height:100%;background:{color}; border-radius:4px 0 0 4px;"></div>'
            f'  <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;'
            f'font-size:12px;font-weight:600;color:#111827; text-shadow: 0 0 1px #fff;">{formatted_value}</div>'
            f'</div>'
            f'</div>'
        )
    return _bar_cell

def _text_only_cell(val: float | object, col_name: str) -> str:
    """Plain numeric cell for detailed table (no bars)."""
    formatted_value = _format_value(val, col_name)
    return (
        f'<div style="text-align:center; padding: 6px 8px; font-size:13px; font-weight:600; color:#1F2937;">'
        f'{formatted_value}'
        f'</div>'
    )

# ---------- [UPDATED] Minimal indicator description loader ----------
@st.cache_data(show_spinner=False)
def _read_index_desc_csv() -> pd.DataFrame:
    """
    Minimal loader for indicator descriptions.
    - Primary path: data/index.csv
    - Fallback path: /mnt/data/index.csv  (uploaded file location)
    - Encodings: utf-8-sig -> utf-8 -> cp949
    - If only 1 column is detected, try TSV fallback once.
    """
    # [Paths] Keep it minimal: just two explicit candidates (no other assumptions)
    candidates = [DATA_DIR / "index.csv", Path("/mnt/data/index.csv")]
    encodings = ("utf-8-sig", "utf-8", "cp949")  # [Encoding] extend if you later standardize differently

    for p in candidates:
        if not p.exists():
            continue
        for enc in encodings:
            try:
                df = pd.read_csv(p, encoding=enc)
                if df.shape[1] == 1:
                    # Possible TSV saved as .csv
                    try:
                        df = pd.read_csv(p, encoding=enc, sep="\t")
                    except Exception:
                        pass
                return _normalize_columns(df)
            except Exception:
                continue
    return pd.DataFrame()

def _find_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Pick the first matching column header by exact string match."""
    if df is None or df.empty:
        return None
    cols = [str(c).strip() for c in df.columns]
    for key in candidates:
        if key in cols:
            return df.columns[cols.index(key)]
    return None

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🗳️", layout="wide")

# --------------------------------
# Sidebar (Navigation)
# --------------------------------
st.sidebar.header("메뉴 선택")
menu = st.sidebar.radio("페이지", ["종합", "지역별 분석", "데이터 설명"], index=0)

# --------------------------------
# Data Loading (Uses DATA_DIR defined above)
# --------------------------------
with st.spinner("데이터 불러오는 중..."):
    df_bookmark = load_bookmark(DATA_DIR)
    bookmark_map = load_bookmark_map(df_bookmark)

    df_pop   = ensure_code_col(load_population_agg(DATA_DIR))
    df_party = ensure_code_col(load_party_labels(DATA_DIR))
    df_trend = ensure_code_col(load_vote_trend(DATA_DIR))
    df_24    = ensure_code_col(load_results_2024(DATA_DIR))
    df_curr  = ensure_code_col(load_current_info(DATA_DIR))
    df_idx   = ensure_code_col(load_index_sample(DATA_DIR))

# --------------------------------
# Page: 종합 (Summary Dashboard)
# --------------------------------
if menu == "종합":
    st.title("🗳️ 지역구 선정 1단계 조사 결과")
    st.write("")
    st.divider()

    # --- Load Scoring Data ---
    csv_path = Path("data/scoring.csv")
    if not csv_path.exists():
        st.error("`data/scoring.csv`를 찾을 수 없습니다. (경로 고정)")
        st.stop()

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if df.shape[1] == 1:
            df = pd.read_csv(csv_path, encoding="utf-8-sig", sep="\t")
    except Exception as e:
        st.error(f"`data/scoring.csv` 읽기 실패: {e}")
        st.stop()

    # --- Data Cleaning and Prep ---
    df = _normalize_columns(df)
    df.rename(columns={df.columns[0]: "지역"}, inplace=True)
    label_col = "지역"

    score_cols = [c for c in df.columns if c != label_col]
    df[score_cols] = df[score_cols].apply(pd.to_numeric, errors="coerce")

    _bar_cell = _bar_cell_factory(df, score_cols, BAR_COLORS_MAIN)

    st.subheader("결과 요약")

    # --- Build HTML table for main scoring ('결과 요약') ---
    headers = [label_col] + score_cols
    thead = (
        f"<th style='text-align:left;padding:6px 8px;white-space:nowrap;width:{REGION_COL_WIDTH};'>지역</th>"
    )
    remaining_cols_count = len(score_cols)
    col_width_pct = f"{100 / remaining_cols_count}%" if remaining_cols_count > 0 else "auto"
    thead += "".join(
        [f"<th style='text-align:center;padding:6px 8px;white-space:nowrap;width:{col_width_pct};'>{h}</th>" for h in score_cols]
    )

    rows_html = []
    for _, row in df.iterrows():
        is_fixed_highlight = row[label_col] in FIXED_HIGHLIGHT_REGIONS
        row_style = f"background-color:{FIXED_HIGHLIGHT_ROW_BG};" if is_fixed_highlight else ""
        cells = [
            f"<td style='padding:6px 8px;white-space:nowrap;width:{REGION_COL_WIDTH};'>"
            f"<span style='font-size:13px; font-weight:{'700' if is_fixed_highlight else '600'};'>{row[label_col]}</span>"
            f"</td>"
        ]
        for c in score_cols:
            cells.append(f"<td style='padding:0px;width:{col_width_pct};'>{_bar_cell(row[c], c)}</td>")
        rows_html.append(f"<tr style='{row_style}'>" + "".join(cells) + "</tr>")

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
    # 세부 지표별 상세 분석 (Detailed Index Analysis) - Text Only
    # ====================================================================
    st.divider()
    st.subheader("세부 지표별 상세 분석")

    INDICATOR_GROUPS = {
        "유권자환경": ["유권자 수", "유동인구", "고령층 비율", "청년층 비율", "4-50대 비율", "2030여성 비율"],
        "정치지형": ["유동성A", "경합도A", "유동성B", "경합도B"],
        "주체역량": ["진보정당 득표력", "진보당 당원수", "진보당 지방선거 후보 수"],
        "상대역량": ["현직 득표력", "민주당 득표력", "보수 득표력"],
    }

    tab_titles = list(INDICATOR_GROUPS.keys())
    tabs = st.tabs(tab_titles)

    for selected_group, tab in zip(tab_titles, tabs):
        with tab:
            target_cols = INDICATOR_GROUPS.get(selected_group, [])

            if not df_idx.empty and target_cols:
                df_idx_norm = _normalize_columns(df_idx)

                regions_map = build_regions(df_idx_norm, bookmark_map=bookmark_map)

                df_display = pd.merge(
                    df_idx_norm,
                    regions_map.rename(columns={"라벨": "지역"}),
                    on="코드",
                    how="left"
                )

                label_col_new = "지역"
                present_cols = [c for c in target_cols if c in df_display.columns]

                if not present_cols:
                    st.info(f"선택된 그룹 ({selected_group})에 해당하는 컬럼이 데이터에 없습니다.")
                else:
                    df_final = df_display.loc[:, [label_col_new, "코드"] + present_cols].copy()
                    df_final[present_cols] = df_final[present_cols].apply(pd.to_numeric, errors="coerce")
                    df_final = df_final.dropna(subset=[label_col_new]).dropna(subset=present_cols, how='all').reset_index(drop=True)

                    # --- Detailed table (text only)
                    thead_new = (
                        f"<th style='text-align:left;padding:6px 8px;white-space:nowrap;font-weight:700;width:{REGION_COL_WIDTH};'>지역</th>"
                    )
                    remaining_cols_count_new = len(present_cols)
                    col_width_pct_new = f"{100 / remaining_cols_count_new}%" if remaining_cols_count_new > 0 else "auto"
                    thead_new += "".join(
                        [f"<th style='text-align:center;padding:6px 8px;white-space:nowrap;width:{col_width_pct_new};'>{h}</th>" for h in present_cols]
                    )

                    rows_html_new = []
                    for _, row in df_final.iterrows():
                        cells = [
                            f"<td style='padding:6px 8px;white-space:nowrap;width:{REGION_COL_WIDTH};'>"
                            f"<span style='font-size:13px; font-weight:600;'>{row[label_col_new]}</span>"
                            f"</td>"
                        ]
                        for c in present_cols:
                            cells.append(f"<td style='padding:0;width:{col_width_pct_new};'>{_text_only_cell(row[c], c)}</td>")
                        rows_html_new.append(f"<tr>" + "".join(cells) + "</tr>")

                    table_html_new = (
                        "<div style='overflow-x:auto;'>"
                        "<table style='border-collapse:separate;border-spacing:0;width:100%;font-size:13px;'>"
                        f"<thead><tr>{thead_new}</tr></thead>"
                        f"<tbody>{''.join(rows_html_new)}</tbody>"
                        "</table>"
                        "</div>"
                    )
                    st.markdown(table_html_new, unsafe_allow_html=True)

                    # ============================
                    # Descriptions under the table
                    # ============================
                    st.divider()  # [Spacing] separator between data table & descriptions

                    # Load index descriptions (data/index.csv or /mnt/data/index.csv)
                    desc_df = _read_index_desc_csv()
                    name_col = _find_first_col(desc_df, ["지표", "지표명", "항목", "지표명칭", "indicator", "name"]) if not desc_df.empty else None
                    desc_col = _find_first_col(desc_df, ["설명", "정의", "지표설명", "description", "desc"]) if not desc_df.empty else None
                    src_col  = _find_first_col(desc_df, ["출처", "source"]) if not desc_df.empty else None

                    st.subheader(f"지표 설명 · {selected_group}")

                    if desc_df.empty or not name_col:
                        st.info("`index.csv`에서 지표 설명을 불러오지 못했습니다. (경로/인코딩/구분자 확인)")
                    else:
                        present_set = set(present_cols)
                        df_desc = desc_df.copy()
                        df_desc[name_col] = df_desc[name_col].astype(str).str.strip()
                        matched = df_desc[df_desc[name_col].isin(present_set)].copy()

                        if matched.empty:
                            st.info("현재 탭의 컬럼명과 `index.csv`의 지표명이 일치하지 않습니다. 표기 통일이 필요합니다.")
                            st.caption(f"탭 컬럼: {', '.join(present_cols)}")
                        else:
                            show_cols = [name_col] + ([desc_col] if desc_col else []) + ([src_col] if src_col else [])
                            try:
                                matched = matched.loc[:, show_cols].sort_values(by=[name_col]).reset_index(drop=True)
                            except Exception:
                                matched = matched.loc[:, show_cols].reset_index(drop=True)

                            head_src = '<th style="text-align:left;padding:8px 10px;">출처</th>' if src_col else ''
                            rows_html = []
                            for _, r in matched.iterrows():
                                nm = str(r.get(name_col, "")).strip()
                                ds = str(r.get(desc_col, "" if desc_col else "")).strip()
                                sc = str(r.get(src_col, "" if src_col else "")).strip()
                                rows_html.append(
                                    f"""
                                    <tr>
                                        <td style="padding:8px 10px;vertical-align:top;font-weight:700;white-space:nowrap;">{nm}</td>
                                        <td style="padding:8px 10px;vertical-align:top;">{ds}</td>
                                        { f'<td style="padding:8px 10px;vertical-align:top;color:#4B5563;">{sc}</td>' if src_col else '' }
                                    </tr>
                                    """
                                )

                            html_desc = f"""
                            <div style="overflow-x:auto;">
                                <table style="border-collapse:separate;border-spacing:0;width:100%;font-size:13px;">
                                    <thead>
                                        <tr>
                                            <th style="text-align:left;padding:8px 10px;white-space:nowrap;">지표</th>
                                            <th style="text-align:left;padding:8px 10px;">설명</th>
                                            {head_src}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {''.join(rows_html)}
                                    </tbody>
                                </table>
                            </div>
                            """
                            st.markdown(html_desc, unsafe_allow_html=True)

            else:
                if df_idx.empty:
                    st.info("`index_sample.csv` 데이터프레임이 비어 있어 상세 분석을 렌더링할 수 없습니다.")
                else:
                    st.info(f"선택된 그룹 ({selected_group})에 표시할 컬럼이 없습니다. (데이터 컬럼명 확인 필요)")

# --------------------------------
# Page: 지역별 분석 (Regional Detail Analysis)
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

    pop_sel   = get_by_code(df_pop, sel_code)
    trend_sel = get_by_code(df_trend, sel_code) if "코드" in df_trend.columns else df_trend
    res_sel   = get_by_code(df_24, sel_code)
    cur_sel   = get_by_code(df_curr, sel_code)
    prg_sel   = get_by_code(df_idx, sel_code)

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
# Page: 데이터 설명 (Data Explanation)
# --------------------------------
elif menu == "데이터 설명":
    c1, c2 = st.columns([1, 1])
    with c1:
        st.title("📘 지표별 구성 및 해설")
    with c2:
        st.markdown(
            f"""
            <div style="text-align:right; font-weight:700; font-size:1.05rem;">
                🗳️ {APP_TITLE}
            </div>
            """,
            unsafe_allow_html=True
        )
    st.divider()

    md_text = None
    for p in [Path("sti") / "지표별 구성 및 해설.md", Path("지표별 구성 및 해설.md"), Path("/mnt/data/sti/지표별 구성 및 해설.md")]:
        s = _read_markdown_cached(str(p))
        if s:
            md_text = s
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
