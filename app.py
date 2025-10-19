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

    # --- Colors & scaling ---
    # Global colors for the main scoring table
    bar_colors = {
        "합계": "#3498DB",
        "유권자환경": "#48C9B0",
        "정치지형": "#1ABC9C",
        "주체역량": "#76D7C4",
        "상대역량": "#2ECC71",
    }
    vmax = {c: (float(df[c].max()) if df[c].notna().any() else 0.0) for c in score_cols}

    # --- HTML bar cell for main table ---
    def _bar_cell(val, col):
        try:
            v = float(val)
        except Exception:
            return f"{val}"
        mx = vmax.get(col, 0.0) or 1.0
        pct = max(0.0, min(100.0, (v / mx) * 100.0))
        color = bar_colors.get(col, "#6B7280")
        return (
            f'<div style="position:relative;width:100%;background:#F3F4F6;height:18px;border-radius:4px;overflow:hidden;">'
            f'  <div style="width:{pct:.2f}%;height:100%;background:{color};"></div>'
            f'  <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;'
            f'font-size:12px;font-weight:600;color:#111827;">{v:.1f}</div>'
            f'</div>'
        )

    # --- Build HTML table for main scoring ---
    headers = [label_col] + score_cols
    thead = "".join(
        [f"<th style='text-align:left;padding:6px 8px;white-space:nowrap;'>{h}</th>" for h in headers]
    )

    rows_html = []
    for _, row in df.iterrows():
        cells = [f"<td style='padding:6px 8px;white-space:nowrap;'>{row[label_col]}</td>"]
        for c in score_cols:
            cells.append(f"<td style='padding:6px 8px;'>{_bar_cell(row[c], c)}</td>")
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
    # 세부 지표별 상세 분석 (Index Sample)
    # ====================================================================
    st.divider()
    st.subheader("세부 지표별 상세 분석 (Index Sample)")

    # 1. 지표 그룹 정의
    INDICATOR_GROUPS = {
        "유권자환경": ["유권자 수", "유동인구", "고령층 비율", "청년층 비율", "4-50대 비율", "2030여성 비율"],
        "정치지형": ["유동성A", "경합도A", "유동성B", "경합도B"],
        "주체역량": ["진보정당 득표력", "진보당 당원수", "진보당 지방선거 후보 수"],
        "상대역량": ["현직 득표력", "민주당 득표력", "보수 득표력"],
    }
    
    # 지표 그룹별 색상 정의 (선택된 그룹의 모든 지표에 동일 색상 적용)
    DETAIL_GROUP_COLORS = {
        "유권자환경": "#00CC99", # Green
        "정치지형": "#3498DB", # Blue
        "주체역량": "#E74C3C", # Red
        "상대역량": "#F39C12", # Yellow/Orange
    }
    
    # 2. 그룹 선택 Tabs 
    tab_titles = list(INDICATOR_GROUPS.keys())
    tabs = st.tabs(tab_titles)
    
    # 5. 상세 테이블용 HTML Bar Cell 함수 
    def _bar_cell_detail_factory(vmax_data, color):
        """클로저를 사용하여 vmax_data와 color를 캡처하는 Bar Cell 함수를 생성"""
        def _bar_cell_detail(val, col):
            try:
                v = float(val)
            except Exception:
                return ""
            mx = vmax_data.get(col, 0.0) or 1.0
            pct = max(0.0, min(100.0, (v / mx) * 100.0))
            
            return (
                f'<div style="position:relative;width:100%;background:#F3F4F6;height:18px;border-radius:4px;overflow:hidden;">'
                f'  <div style="width:{pct:.2f}%;height:100%;background:{color};"></div>'
                f'  <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;'
                f'font-size:12px;font-weight:600;color:#111827;">{v:.3f}</div>'
                f'</div>'
            )
        return _bar_cell_detail

    # 탭별로 내용 렌더링
    for selected_group, tab in zip(tab_titles, tabs):
        with tab:
            target_cols = INDICATOR_GROUPS.get(selected_group, [])
            bar_color_new = DETAIL_GROUP_COLORS.get(selected_group, "#6B7280")
            
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
                    
                    # 막대 차트 스케일링을 위한 컬럼별 최댓값 계산
                    vmax_new = {c: (float(df_final[c].max()) if df_final[c].notna().any() else 0.0) for c in present_cols}
                    
                    # Bar Cell 함수 인스턴스화
                    _bar_cell_detail = _bar_cell_detail_factory(vmax_new, bar_color_new)

                    # 6. 상세 분석 HTML 테이블 생성
                    headers_new = [label_col_new] + present_cols
                    thead_new = "".join(
                        [f"<th style='text-align:left;padding:6px 8px;white-space:nowrap;font-weight:700;'>{h}</th>" for h in headers_new]
                    )

                    rows_html_new = []
                    for _, row in df_final.iterrows():
                        cells = [f"<td style='padding:6px 8px;white-space:nowrap;'>{row[label_col_new]}</td>"]
                        for c in present_cols:
                            # 캡처된 함수 사용
                            cells.append(f"<td style='padding:6px 8px;'>{_bar_cell_detail(row[c], c)}</td>") 
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
