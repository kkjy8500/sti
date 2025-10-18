# app.py
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
    load_bookmark,            # bookmark.csv (optional but preferred)
    load_bookmark_map,        # -> dict, standard_key -> actual column
    load_population_agg,      # population.csv
    load_party_labels,        # party_labels.csv
    load_vote_trend,          # vote_trend.csv
    load_results_2024,        # 5_na_dis_results.csv
    load_current_info,        # current_info.csv
    load_index_sample,        # index_sample1012.csv (optional)
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
    df_bookmark = load_bookmark(DATA_DIR)                  # may be empty
    bookmark_map = load_bookmark_map(df_bookmark)          # dict or {}

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
    st.title("🗳️ 지역별 스코어 표 (막대 포함)")

    # How to change later:
    # - Change CSV path/encoding: edit the read_csv line below.
    # - To change bar colors or max scaling, move to charts? (kept simple here.)

    csv_path = Path("data/scoring.csv")  # simplified (Req #2)
    if not csv_path.exists():
        st.error("`data/scoring.csv`를 찾을 수 없습니다. (경로 고정)"); st.stop()

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as e:
        st.error(f"`data/scoring.csv` 읽기 실패: {e}"); st.stop()

    if df.shape[1] == 1:
        # If accidental TSV, force tab parsing once
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig", sep="\t")
        except Exception as e:
            st.error(f"`data/scoring.csv` 구분자 문제: {e}"); st.stop()

    label_col = "region"
    score_cols = [c for c in df.columns if c != label_col]
    df[score_cols] = df[score_cols].apply(pd.to_numeric, errors="coerce")

    bar_colors = {"합계": "#2563EB", "유권자환경": "#059669", "정치지형": "#F59E0B", "주체역량": "#DC2626", "상대역량": "#7C3AED"}
    vmax = {c: (float(df[c].max()) if df[c].notna().any() else 0.0) for c in score_cols}

    def _bar_cell(val, col):
        # How to change later: change bar height (18px) or track color (#F3F4F6)
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

    headers = [label_col] + score_cols
    thead = "".join([f"<th style='text-align:left;padding:6px 8px;white-space:nowrap;'>{h}</th>" for h in headers])

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
    c1, c2 = st.columns([1, 1])
    with c1:
        st.title("📘 지표별 구성 및 해설")
    with c2:
        st.markdown(f"<div style='text-align:right;font-weight:700;font-size:1.05rem;'>🗳️ {APP_TITLE}</div>", unsafe_allow_html=True)

    st.divider()
    md_path = Path("sti/지표별 구성 및 해설.md")  # simplified (Req #7)
    if md_path.exists():
        try:
            st.markdown(md_path.read_text(encoding="utf-8-sig"))
        except Exception:
            st.markdown(md_path.read_text(encoding="utf-8"))
    else:
        st.info("`sti/지표별 구성 및 해설.md` 파일을 찾지 못했습니다.")

# --------------------------------
# Footer
# --------------------------------
st.write("")
st.caption("© 2025 전략지역구 조사 · 에스티아이")
