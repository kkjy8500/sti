from __future__ import annotations

import re
import streamlit as st
import pandas as pd
from pathlib import Path

import importlib
import charts
importlib.reload(charts)
from charts import *

from data_loader import (
    load_population_agg,     # population.csv (구 단위 합계본)
    load_party_labels,       # party_labels.csv
    load_vote_trend,         # vote_trend.csv
    load_results_2024,       # 5_na_dis_results.csv
    load_current_info,       # current_info.csv
    load_index_sample,       # index_sample.csv
)

from metrics import (
    compute_trend_series,
    compute_summary_metrics,
    compute_24_gap,
)

from charts import (
    render_population_box,
    render_vote_trend_chart,
    render_results_2024_card,
    render_incumbent_card,
    render_prg_party_box,
    render_region_detail_layout,
)


# -----------------------------
# Page Config
# -----------------------------
APP_TITLE = "지역구 선정 1단계 조사 결과"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🗳️",
    layout="wide",
)

# ---------- Sidebar Navigation ----------
st.sidebar.header("메뉴 선택")
menu = st.sidebar.radio(
    "페이지",
    ["종합", "지역별 분석", "데이터 설명"],
    index=0
)

DATA_DIR = Path("data")

# -----------------------------
# 공통 유틸
# -----------------------------
CODE_CANDIDATES = ["코드", "지역구코드", "선거구코드", "지역코드", "code", "CODE"]
NAME_CANDIDATES = ["지역구", "선거구", "선거구명", "지역명", "district", "지역구명", "region", "지역"]
SIDO_CANDIDATES = ["시/도", "시도", "광역", "sido", "province"]

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if len(df) == 0:
        return df
    df2 = df.copy()
    df2.columns = [str(c).strip().replace("\n", "").replace("\r", "") for c in df2.columns]
    return df2

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
    """하이픈/공백 제거, 대소문자 무시, 선행 0 제거 → 코드 표준화"""
    s = str(x).strip()
    s = re.sub(r"[^0-9A-Za-z]", "", s)
    s = s.lstrip("0")
    return s.lower()

def ensure_code_col(df: pd.DataFrame) -> pd.DataFrame:
    """여러 이름의 코드 컬럼을 '코드'(str)로 표준화."""
    if df is None:
        return pd.DataFrame()
    if len(df) == 0:
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
    else:
        df2["__NO_CODE__"] = True
    return df2

def get_by_code(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """코드 컬럼 자동 탐지 + 표준화 비교로 해당 code 행만 반환(없으면 빈 DF)."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df2 = _normalize_columns(df)
    code_col = "코드" if "코드" in df2.columns else _detect_col(df2, CODE_CANDIDATES)
    if not code_col:
        return pd.DataFrame()
    try:
        key = _canon_code(code)
        sub = df2[df2[code_col].astype(str).map(_canon_code) == key]
        return sub if len(sub) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def _first_nonempty(*dfs: pd.DataFrame) -> pd.DataFrame | None:
    for d in dfs:
        if isinstance(d, pd.DataFrame) and len(d) > 0:
            return d
    return None

def build_regions(primary_df: pd.DataFrame, *fallback_dfs: pd.DataFrame) -> pd.DataFrame:
    """
    사이드바 선택용 지역 목록: 코드 + 라벨(시/도 + 지역구).
    primary_df가 비어있으면 fallback들(df_24, df_trend, df_curr 등)에서 생성.
    """
    base = _first_nonempty(primary_df, *fallback_dfs)
    if base is None:
        return pd.DataFrame(columns=["코드", "라벨"])
    dfp = ensure_code_col(_normalize_columns(base))

    name_col = _detect_col(dfp, NAME_CANDIDATES)
    if not name_col:
        return (
            dfp.loc[:, ["코드"]]
               .assign(라벨=dfp["코드"])
               .drop_duplicates()
               .sort_values("라벨")
               .reset_index(drop=True)
        )

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

# -----------------------------
# 상단 바 렌더링 (지역별 분석에서만 사용)
# -----------------------------
def render_topbar(page_title: str | None):
    """좌: 페이지별 동적 제목 / 우: 앱 제목(오른쪽 상단 고정)."""
    c1, c2 = st.columns([1, 1])
    with c1:
        if page_title:
            st.title(page_title)
        else:
            st.write("")
    with c2:
        st.markdown(
            f"""
            <div style="text-align:right; font-weight:700; font-size:1.05rem;">
                🗳️ {APP_TITLE}
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------
# Load Data
# -----------------------------
with st.spinner("데이터 불러오는 중..."):
    df_pop   = load_population_agg(DATA_DIR)       # population.csv
    df_party = load_party_labels(DATA_DIR)         # party_labels.csv
    df_trend = load_vote_trend(DATA_DIR)           # vote_trend.csv
    df_24    = load_results_2024(DATA_DIR)         # 5_na_dis_results.csv
    df_curr  = load_current_info(DATA_DIR)         # current_info.csv
    df_idx   = load_index_sample(DATA_DIR)         # index_sample.csv (선택)

# 표준화
df_pop   = ensure_code_col(df_pop)
df_party = ensure_code_col(df_party)
df_trend = ensure_code_col(df_trend)
df_24    = ensure_code_col(df_24)
df_curr  = ensure_code_col(df_curr)
df_idx   = ensure_code_col(df_idx)

# -----------------------------
# Page: 종합
# -----------------------------
if menu == "종합":
    st.title("🗳️ 지역구 선정 1단계 조사 결과")
    st.caption("에스티아이")

    # --- Load data
    def _find_scoring_csv() -> Path | None:
        # Try common project-relative paths first
        candidates = [
            Path(__file__).resolve().parent / "data" / "scoring.csv",  # ./data/scoring.csv next to app.py
            Path.cwd() / "data" / "scoring.csv",                       # working dir fallback
            Path("data/scoring.csv"),                                  # relative fallback
            Path("/mount/src/sti/data/scoring.csv"),                   # streamlit cloud repo path (common)
            Path("/mnt/data/sti/data/scoring.csv"),                    # mounted data (less common)
            Path("/mnt/data/scoring.csv"),                             # last resort
        ]
        for p in candidates:
            if p.exists():
                return p
        return None
    
    CSV_PATH = _find_scoring_csv()
    if not CSV_PATH:
        st.error("`scoring.csv`를 찾을 수 없습니다. 경로를 확인하세요: `sti/data/scoring.csv` 위치에 두는 것을 권장합니다.")
        st.stop()
    
    # Robust read: encoding & delimiter inference (comma or tab)
    tried = []
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(CSV_PATH, encoding=enc)
            # If it collapsed into a single column, try tab-delimited
            if df.shape[1] == 1:
                df = pd.read_csv(CSV_PATH, encoding=enc, sep="\t")
            break
        except Exception as e:
            tried.append(f"{enc}: {e}")
    else:
        st.error("`scoring.csv` 읽기에 실패했습니다. 인코딩/구분자 확인이 필요합니다.")
        st.stop()

    # --- Ensure numeric dtypes for score columns
    label_col = "region"
    score_cols = [c for c in df.columns if c != label_col]
    df[score_cols] = df[score_cols].apply(pd.to_numeric, errors="coerce")
    
    # --- Per-column colors
    bar_colors = {
        "합계": "#2563EB",
        "유권자환경": "#059669",
        "정치지형": "#F59E0B",
        "주체역량": "#DC2626",
        "상대역량": "#7C3AED",
    }
    
    # --- Per-column max (0~max 스케일; 고정 100이면 vmax=100.0로 바꾸세요)
    vmax = {c: (float(df[c].max()) if df[c].notna().any() else 0.0) for c in score_cols}
    
    # --- Tiny HTML table with in-cell bars (left-aligned)
    def _bar_cell(val, col):
        try:
            v = float(val)
        except Exception:
            return f"{val}"
        mx = vmax.get(col, 0.0) or 1.0
        pct = max(0.0, min(100.0, (v / mx) * 100.0))
        color = bar_colors.get(col, "#6B7280")
        # outer: light track, inner: colored bar
        return (
            f'<div style="position:relative;width:100%;background:#F3F4F6;'
            f'height:18px;border-radius:4px;overflow:hidden;">'
            f'  <div style="width:{pct:.2f}%;height:100%;background:{color};"></div>'
            f'  <div style="position:absolute;inset:0;display:flex;align-items:center;'
            f'justify-content:center;font-size:12px;font-weight:600;color:#111827;">{v:.1f}</div>'
            f'</div>'
        )
    
    # Build header
    headers = [label_col] + score_cols
    thead = "".join([f"<th style='text-align:left;padding:6px 8px;white-space:nowrap;'>{h}</th>" for h in headers])
    
    # Build body
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
    
    st.subheader("지역별 스코어 표 (막대 포함)")
    st.markdown(table_html, unsafe_allow_html=True)
    

# -----------------------------
# Page: 지역별 분석
# -----------------------------
elif menu == "지역별 분석":
    regions = build_regions(df_pop, df_trend, df_24, df_curr)
    if regions.empty:
        render_topbar(None)
        st.error("지역 목록을 만들 수 없습니다. (어느 데이터셋에도 '코드' 및 지역명 컬럼이 없음)")
        st.stop()

    PLACEHOLDER = "— 지역을 선택하세요 —"
    options = [PLACEHOLDER] + regions["라벨"].tolist()

    st.sidebar.header("지역 선택")
    sel_label = st.sidebar.selectbox("선거구를 선택하세요", options, index=0)

    if sel_label == PLACEHOLDER:
        render_topbar(None)
        st.subheader("지역을 선택하세요")
        st.stop()

    # 선택된 코드 찾기
    sel_code = regions.loc[regions["라벨"] == sel_label, "코드"].iloc[0]

    # 상단 고정 헤더
    render_topbar(sel_label)

    # 선택된 지역 데이터만 필터링
    pop_sel   = get_by_code(df_pop, sel_code)
    trend_sel = get_by_code(df_trend, sel_code) if "코드" in df_trend.columns else df_trend
    res_sel   = get_by_code(df_24, sel_code)
    cur_sel   = get_by_code(df_curr, sel_code)
    prg_sel   = get_by_code(df_party, sel_code)

    # 상세 레이아웃 렌더링 (charts.py 내부에서 각 카드/차트 호출)
    render_region_detail_layout(
        df_pop=pop_sel,
        df_trend=trend_sel,
        df_24=res_sel,
        df_cur=cur_sel,
        df_prg=prg_sel
    )

# -----------------------------
# Page: 데이터 설명
# -----------------------------
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


# -----------------------------
# Footer (모든 페이지 공통)
# -----------------------------
st.write("")
st.caption("© 2025 전략지역구 조사 · 에스티아이")

