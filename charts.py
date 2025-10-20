# charts.py
# Purpose: All visuals. No file I/O here. Bookmark-first column resolution.
# How to change later:
# - Colors/sizes per chart: search "TUNE:" comments near each chart.
# - To tweak axis order for vote-trend: edit ORDER_LABELS in render_vote_trend_chart.

from __future__ import annotations
import re, math
import pandas as pd
import streamlit as st
import altair as alt
from metrics import compute_24_gap


# Preferred brand-ish blues/greys for KPI/mini elements
COLOR_BLUE = "#3498DB"   # TUNE: main accent blue
COLOR_GRAY = "#95A5A6"   # TUNE: neutral grey

alt.data_transformers.disable_max_rows()

# -----------------------------
# Small utils
# -----------------------------
def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None: return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c).strip().replace("\n","").replace("\r","") for c in out.columns]
    return out

def _to_pct_float(v, default=None):
    if v is None or (isinstance(v, float) and pd.isna(v)): return default
    try:
        s = str(v).strip().replace(",", "")
        m = re.match(r"^\s*([+-]?\d+(\.\d+)?)\s*%?\s*$", s)
        if not m: return default
        x = float(m.group(1))
        if "%" in s: return x
        return x * 100.0 if 0 <= x <= 1 else x
    except Exception:
        return default

def _to_num(x):
    if pd.isna(x): return 0.0
    if isinstance(x,(int,float)): return float(x)
    try: return float(str(x).replace(",","").strip())
    except: return 0.0

def _fmt_pct(x):
    return f"{x:.2f}%" if isinstance(x, (int, float)) else "N/A"

def _col(df: pd.DataFrame, bookmark_map: dict | None, std_key: str, candidates: list[str], required: bool = True) -> str | None:
    """
    Column resolver: bookmark first (if provided), then candidate scan.
    - std_key: standard key in bookmark_map (e.g., 'total_voters')
    """
    if df is None or df.empty: return None
    if bookmark_map:
        cand = bookmark_map.get(std_key)
        if cand and cand in df.columns:
            return cand
    for c in candidates:
        if c in df.columns:
            return c
    # try normalized compare
    cols = [str(c).strip().replace("\n","").replace("\r","") for c in df.columns]
    for c in candidates:
        if c in cols:
            return df.columns[cols.index(c)]
    if required:
        raise ValueError(f"Required column not found (std_key='{std_key}', candidates={candidates})")
    return None

# -----------------------------
# Top bar (lightweight)
# -----------------------------
def _render_topbar(page_title: str | None, app_title: str | None):
    c1, c2 = st.columns([1, 1])
    with c1:
        if page_title: st.title(page_title)
        else: st.write("")
    with c2:
        if app_title:
            st.markdown(f"<div style='text-align:right;font-weight:700;font-size:1.05rem;'>🗳️ {app_title}</div>", unsafe_allow_html=True)

# =========================================================
# --- Dependencies/Constants for Runnability ---
# Placeholder function for numeric conversion (used in original code)
def _to_num(v):
    """Converts value to float, handling common non-numeric formats."""
    if isinstance(v, (int, float)):
        return v
    try:
        if isinstance(v, str):
            v = v.replace(",", "").replace("%", "")
        return float(v)
    except (ValueError, TypeError):
        return float('nan')

# TUNE: Bar Colors (ensure they contrast well with the background)
COLOR_BLUE = "#3B82F6"  # Blue color for the selected region bar
COLOR_GRAY = "#D1D5DB"  # Gray color for the average/comparison bar
# =========================================================

# =========================================================
# Population Box – KPI + two-bars (Region vs 10-avg)
# Key modifications:
# - Implemented explicit grouping by 'region_key' (Gu level) for selected data (df_s) 
#   before calculating region_total, as requested for Dong-level input.
# - Ensures chart is responsive and X-axis labels are horizontal.
# =========================================================
def render_population_box(
    pop_sel: pd.DataFrame,
    *,
    df_pop_all: pd.DataFrame,
    bookmark_map: dict | None = None,
    box_height_px: int = 170, # TUNE: Chart container height reference (pixels)
    SHOW_DEBUG: bool = False, # set True when diagnosing
):
    if pop_sel is None or pop_sel.empty:
        st.info("인구 데이터가 없습니다."); return
    df_s = pop_sel.copy()
    df_a = df_pop_all.copy() if df_pop_all is not None else pd.DataFrame()

    # ---------- Resolve total/floating columns (bookmark first, then common aliases) ----------
    def _norm(s: str) -> str:
        return re.sub(r"\s+", "", str(s)).lower()

    def _find_total_col_in(df: pd.DataFrame, bm: dict | None):
        if bm:
            for k in ["total_voters", "총유권자", "전체유권자"]:
                v = bm.get(k)
                if v and v in df.columns: return v
        aliases = ["전체 유권자 수","전체유권자수","전체 유권자","전체유권자","총유권자","유권자수","total_voters","voters","totalvoters"]
        norm_cols = [_norm(c) for c in df.columns]
        for a in aliases:
            if _norm(a) in norm_cols:
                return df.columns[norm_cols.index(_norm(a))]
        for c in df.columns:
            n = _norm(c)
            if ("유권자" in n) or ("voter" in n): return c
        return None

    def _find_region_key(df: pd.DataFrame, bm: dict | None):
        if bm:
            for k in ["region_code","선거구코드","지역코드","구코드","code","region"]:
                v = bm.get(k)
                if v and v in df.columns: return v
        aliases = ["선거구코드","지역구코드","지역코드","구코드","자치구코드","선거구","지역구","지역","자치구","구","행정구역","code","region_code","region","district","gu"]
        norm_cols = [_norm(c) for c in df.columns]
        for a in aliases:
            na = _norm(a)
            for i, nc in enumerate(norm_cols):
                if (na == nc) or (na in nc): return df.columns[i]
        for c in df.columns:
            s = df[c]
            if s.dtype == "O":
                u = s.dropna().nunique()
                if 1 < u < len(df): return c
        return None

    total_col = _find_total_col_in(df_s, bookmark_map)
    region_key_s = _find_region_key(df_s, bookmark_map) # Find region key for selected data (df_s)

    if not total_col or total_col not in df_s.columns:
        st.info("⚠️ 총유권자 컬럼을 찾지 못했습니다.")
        if SHOW_DEBUG: st.write({"DF_S_cols": list(df_s.columns)})
        return

    float_col = None
    for c in df_s.columns:
        if any(k in str(c) for k in ["유동", "전입", "전출", "유출입", "floating"]):
            float_col = c; break

    # ---------- Numeric casting and Region Total calculation (for KPI) ----------
    df_s[total_col] = pd.to_numeric(df_s[total_col].apply(_to_num), errors="coerce")
    if float_col:
        df_s[float_col] = pd.to_numeric(df_s[float_col].apply(_to_num), errors="coerce")

    # Calculate region_total (for selected area KPI)
    region_total = 0.0
    if region_key_s and region_key_s in df_s.columns:
        # Aggregation by region code (Gu level) as requested for the KPI total.
        grp_s = (
            df_s[[region_key_s, total_col]]
            .groupby(region_key_s, dropna=False)[total_col]
            .sum(min_count=1)
        )
        if grp_s.notna().any():
            region_total = float(grp_s.sum(skipna=True))
    else:
        # Fallback to simple sum if region key is missing
        if df_s[total_col].notna().any():
             region_total = float(df_s[total_col].sum(skipna=True))
             if SHOW_DEBUG: st.warning(f"Region key not found in df_s. Falling back to simple sum.")

    # ---------- Compute avg_total & regional data for chart (from all data df_a) ----------
    avg_total = None
    grp = None # Initialize grp for scope
    total_col_a = _find_total_col_in(df_a, bookmark_map)
    region_key_a = _find_region_key(df_a, bookmark_map)

    if not df_a.empty and total_col_a and region_key_a:
        a_vals = pd.to_numeric(df_a[total_col_a].apply(_to_num), errors="coerce")
        if region_key_a in df_a.columns:
            # Group all data (df_a) by region key (Gu level) and sum the voters
            grp = (
                df_a[[region_key_a, total_col_a]]
                .assign(**{total_col_a: a_vals})
                .groupby(region_key_a, dropna=False)[total_col_a]
                .sum(min_count=1)
            )
            # Calculate the overall mean across the regional sums (10 regions' average)
            avg_total = float(grp.mean(skipna=True)) if grp.notna().any() else None

    # ---------- KPI (top) rendering ----------
    st.markdown("<!-- TUNE: Top section padding/margin -->", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div style="text-align:center;">
              <div style="color:#6B7280; font-weight:600; margin-bottom:4px;">전체 유권자 수</div>
              <div style="font-weight:800; color:#111827;">{int(round(max(region_total,0))):,}명</div>
            </div>
            """, unsafe_allow_html=True
        )
    with c2:
        if float_col and df_s[float_col].notna().any():
            floating_value_txt = f"{int(round(float(df_s[float_col].sum()))):,}명"
        else:
            floating_value_txt = "N/A"
        st.markdown(
            f"""
            <div style="text-align:center;">
              <div style="color:#6B7280; font-weight:600; margin-bottom:4px;">유동인구</div>
              <div style="font-weight:800; color:#111827;">{floating_value_txt}</div>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ---------- Vertical Bar Chart (Two-Bar Comparison) ----------
    if grp is not None and avg_total is not None and region_total > 0:
        current_selected_gu_name = "선택 지역 (합산)"
        if region_key_s and region_key_s in df_s.columns:
            u = df_s[region_key_s].dropna().unique()
            if len(u) == 1:
                current_selected_gu_name = str(u[0])

        df_chart = pd.DataFrame({
            "Category": [str(current_selected_gu_name), "10개 평균"],
            "VoterCount": [float(region_total), float(avg_total)],
            "Highlight": ["Selected", "Average"],
        }).replace([pd.NA, float("inf"), float("-inf")], 0.0)

        x_enc = alt.X("Category:N", title=None, axis=alt.Axis(labelAngle=0))
        y_enc = alt.Y(
            "VoterCount:Q",
            title="유권자 수 (명)",
            scale=alt.Scale(domain=[0, 250000]),
            axis=alt.Axis(format="~s"),
        )
        color_enc = alt.Color(
            "Highlight:N",
            scale=alt.Scale(domain=["Selected", "Average"], range=[COLOR_BLUE, COLOR_GRAY]),
            legend=None,
        )

        base = alt.Chart(df_chart)

        bars = base.mark_bar(cornerRadiusEnd=3, width=50).encode(
            x=x_enc, y=y_enc, color=color_enc,
            tooltip=[
                alt.Tooltip("Category:N", title="구분"),
                alt.Tooltip("VoterCount:Q", title="유권자 수", format=",.0f"),
            ],
        )

        text = base.mark_text(
            align="center", baseline="bottom", dy=-8, fontWeight="bold", fontSize=14, color="#1F2937"
        ).encode(
            x=x_enc, y=y_enc, text=alt.Text("VoterCount:Q", format=",.0f")
        )

        chart = alt.layer(bars, text, data=df_chart).properties(
            height=int(box_height_px * 1.5),
            padding={"top": 8, "right": 0, "bottom": 0, "left": 0},
        ).configure_view(stroke=None)

        st.altair_chart(chart, use_container_width=True)

# =========================================================
# Age Composition (Half donut)
# TUNE: inner/outer radius, fonts, center offsets, chart width/height.
# =========================================================
def render_age_highlight_chart(pop_sel: pd.DataFrame, *, bookmark_map: dict | None = None, box_height_px: int = 240):
    df = _norm_cols(pop_sel.copy()) if pop_sel is not None else pd.DataFrame()
    if df.empty:
        st.info("연령 구성 데이터가 없습니다.")
        return

    # --- Key labels ---
    Y, M, O = "청년층(18~39세)", "중년층(40~59세)", "고령층(65세 이상)"

    # --- total column auto detect ---
    total_col = None
    try:
        total_col = _col(df, bookmark_map, "total_voters",
                         ["전체 유권자", "유권자수", "선거인수", "total_voters"], required=False)
    except Exception:
        total_col = None

    for c in (Y, M, O):
        if c not in df.columns:
            st.info(f"연령대 컬럼이 없습니다: {c}")
            return

    # --- numeric cast ---
    def _num(v):
        try:
            return float(str(v).replace(",", "").replace("%", "").strip())
        except Exception:
            return 0.0

    for c in [Y, M, O] + ([total_col] if total_col else []):
        df[c] = df[c].apply(_num).fillna(0.0)

    y, m, o = df[Y].sum(), df[M].sum(), df[O].sum()
    tot = df[total_col].sum() if total_col else y + m + o
    if tot <= 0:
        st.info("유효한 전체 유권자 수가 없습니다.")
        return

    mid_60_64 = max(0.0, tot - (y + m + o))
    labels = [Y, M, "60–64세", O]
    values = [y, m, mid_60_64, o]
    ratios = [v / tot for v in values]
    ratios100 = [r * 100 for r in ratios]

    focus = st.radio("강조", [Y, M, O], index=0, horizontal=True, label_visibility="collapsed")
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    df_vis = pd.DataFrame({
        "연령": labels,
        "비율": ratios,
        "표시비율": ratios100,
        "강조": [l == focus for l in labels],
        "순서": [1, 2, 3, 4],
    })

    # =======================================================
    # 🟦 반원 도넛 (상단 유지, 전체적으로 살짝 아래 배치)
    # =======================================================
    donut = (
        alt.Chart(df_vis)
        .mark_arc(innerRadius=70, outerRadius=110, cornerRadius=6, stroke="white", strokeWidth=1)
        .encode(
            theta=alt.Theta("비율:Q", stack=True, sort=None,
                            scale=alt.Scale(range=[-math.pi/2, math.pi/2])),  # 상단 반원
            order=alt.Order("순서:Q"),
            color=alt.Color("강조:N",
                            scale=alt.Scale(domain=[True, False], range=["#1E6BFF", "#E5E7EB"]),
                            legend=None),
            tooltip=[
                alt.Tooltip("연령:N", title="연령대"),
                alt.Tooltip("표시비율:Q", title="비율(%)", format=".2f"),
            ],
        )
        .properties(
            height=box_height_px,
            padding={"top": 0, "bottom": 0, "left": 0, "right": 0},
        )
        .transform_calculate(
            # ⚙️ 도넛 자체를 약간 아래로 내리기 위한 보정 (height * 0.48)
            cy="height * 0.48"
        )
    )

    # =======================================================
    # 🟨 텍스트: 도넛 중심 기준 아래 정렬
    # =======================================================
    label_map = {Y: "청년층(18~39세)", M: "중년층(40~59세)", O: "고령층(65세 이상)"}
    pct_txt = f"{ratios100[labels.index(focus)]:.2f}%"
    lbl_txt = label_map.get(focus, focus)
    text_df = pd.DataFrame({"pct": [pct_txt], "lbl": [lbl_txt]})

    # 숫자: 도넛 중심 기준 (height*0.48)
    num_layer = (
        alt.Chart(text_df)
        .transform_calculate(cx="width/2", cy="height*0.48")
        .mark_text(fontSize=28, fontWeight="bold", color="#0f172a",
                   align="center", baseline="middle")
        .encode(x="cx:Q", y="cy:Q", text="pct:N")
    )

    # 라벨: 숫자 바로 아래 (height*0.48 + 24)
    lbl_layer = (
        alt.Chart(text_df)
        .transform_calculate(cx="width/2", cy="height*0.48 + 24")
        .mark_text(fontSize=14, color="#475569", align="center", baseline="top")
        .encode(x="cx:Q", y="cy:Q", text="lbl:N")
    )

    # =======================================================
    # 🎯 Combine (격자, 뷰프레임 제거)
    # =======================================================
    final_chart = (
        alt.layer(donut, num_layer, lbl_layer)
        .configure_view(stroke=None)  # ✅ 격자/테두리 제거
        .properties(autosize=alt.AutoSizeParams(type="fit", contains="padding"))
    )

    st.altair_chart(final_chart, use_container_width=True, theme=None)

# =========================================================
# Sex ratio by age – horizontal bars
# TUNE: x-axis ticks at every 10%, bar_size, legend position/colors.
# =========================================================
def render_sex_ratio_bar(pop_sel: pd.DataFrame, *, bookmark_map: dict | None = None, box_height_px: int = 340):
    if pop_sel is None or pop_sel.empty:
        st.info("성비 데이터를 표시할 수 없습니다.")
        return

    df = _norm_cols(pop_sel.copy())
    age_buckets = ["20대","30대","40대","50대","60대","70대 이상"]
    expect = [f"{a} 남성" for a in age_buckets] + [f"{a} 여성" for a in age_buckets]
    miss = [c for c in expect if c not in df.columns]
    if miss:
        st.info("성비용 컬럼이 부족합니다: " + ", ".join(miss))
        return

    df_num = df[expect].applymap(_to_num).fillna(0.0)
    sums = df_num.sum(axis=0)
    grand_total = float(sums.sum())
    if grand_total <= 0:
        st.info("성비 데이터(연령×성별)가 모두 0입니다.")
        return

    rows = []
    for a in age_buckets:
        m, f = float(sums[f"{a} 남성"]), float(sums[f"{a} 여성"])
        rows += [{"연령대":a,"성별":"남성","명":m}, {"연령대":a,"성별":"여성","명":f}]
    tidy = pd.DataFrame(rows)
    tidy["전체비중"] = tidy["명"] / grand_total
    age_tot = tidy.groupby("연령대")["명"].transform("sum").replace(0, 1.0)
    tidy["연령대내비중"] = tidy["명"] / age_tot

    label_map = {"20대":"18–29세","30대":"30대","40대":"40대","50대":"50대","60대":"60대","70대 이상":"70대 이상"}
    tidy["연령대표시"] = tidy["연령대"].map(label_map)

    bar_size = 30  # TUNE: bar thickness (px)
    tick_values = [0.0, 0.1, 0.2, 0.3]  # TUNE: 10% ticks

    bars = (
        alt.Chart(tidy)
        .mark_bar(size=bar_size)
        .encode(
            y=alt.Y("연령대표시:N", sort=[label_map[a] for a in age_buckets], title=None),
            x=alt.X(
                "전체비중:Q",
                scale=alt.Scale(domain=[0, 0.30]),
                axis=alt.Axis(format=".0%", title="전체 기준 구성비(%)", values=tick_values, tickMinStep=0.1, grid=True),  # TUNE: 10% ticks
            ),
            color=alt.Color(
                "성별:N",
                scale=alt.Scale(domain=["남성","여성"], range=["#4DA6B7", "#85C1E9"]),  # TUNE: male/female colors
                legend=alt.Legend(title=None, orient="top")
            ),
            tooltip=[
                alt.Tooltip("연령대표시:N", title="연령대"),
                alt.Tooltip("성별:N", title="성별"),
                alt.Tooltip("명:Q", title="인원", format=",.0f"),
                alt.Tooltip("전체비중:Q", title="전체 기준 비중", format=".1%"),
                alt.Tooltip("연령대내비중:Q", title="연령대 내부 비중", format=".1%"),
            ],
        )
        .properties(height=box_height_px)
        .configure_view(stroke=None)
    )
    st.altair_chart(bars, use_container_width=True, theme=None)

# =========================================================
# Vote trend (keep interactions)
# TUNE: ORDER_LABELS, legend orientation, line width, point size.
# =========================================================
def render_vote_trend_chart(ts_sel: pd.DataFrame, ts_all: pd.DataFrame | None = None, *, box_height_px: int = 420):
    with st.container(border=True):
        if ts_sel is None or ts_sel.empty:
            st.info("득표 추이 데이터가 없습니다."); return
        df = _norm_cols(ts_sel.copy())

        label_col = next((c for c in ["계열","성향","정당성향","party_label","label"] if c in df.columns), None)
        value_col = next((c for c in ["득표율","비율","share","ratio","pct","prop"] if c in df.columns), None)
        wide_cols = [c for c in ["민주","보수","진보","기타"] if c in df.columns]
        id_col    = next((c for c in ["선거명","election","분류","연도","year"] if c in df.columns), None)
        year_col  = next((c for c in ["연도","year"] if c in df.columns), None)

        if wide_cols:
            if not id_col: st.warning("선거명을 식별할 컬럼이 필요합니다."); return
            long_df = df.melt(id_vars=id_col, value_vars=wide_cols, var_name="계열", value_name="득표율")
            base_e  = long_df[id_col].astype(str)
        else:
            if not (label_col and value_col): st.warning("정당 성향(계열)과 득표율 컬럼이 필요합니다."); return
            long_df = df.rename(columns={label_col:"계열", value_col:"득표율"}).copy()
            if id_col:     base_e = long_df[id_col].astype(str)
            elif year_col: base_e = long_df[year_col].astype(str)
            else: st.warning("선거명을 식별할 컬럼이 필요합니다."); return

        long_df["득표율"] = pd.to_numeric(long_df["득표율"].astype(str).str.replace("%","", regex=False).str.strip(), errors="coerce")

        def _norm_token(s: str) -> str:
            s = str(s).strip().replace("-","_").replace(" ","_").upper()
            return re.sub(r"_+","_", s)
        CODE = re.compile(r"^(20\d{2})(?:_([SG]))?_(NA|LOC|PRESIDENT)(?:_(PRO|GOV))?$")
        def to_kr(s: str) -> str:
            key = _norm_token(s); m = CODE.fullmatch(key)
            if not m: return str(s)
            year, _rg, lvl, kind = m.group(1), m.group(2), m.group(3), m.group(4)
            if lvl=="PRESIDENT": return f"{year} 대선"
            if lvl=="NA"  and kind=="PRO": return f"{year} 총선 비례"
            if lvl=="LOC" and kind=="PRO": return f"{year} 광역 비례"
            if lvl=="LOC" and kind=="GOV": return f"{year} 광역단체장"
            return s

        long_df["선거명_표시"] = base_e.apply(to_kr)
        long_df = long_df.dropna(subset=["선거명_표시","계열","득표율"]).copy()

        ORDER_LABELS = [
            "2016 총선 비례","2017 대선",
            "2018 광역단체장","2018 광역 비례",
            "2020 총선 비례",
            "2022 대선","2022 광역단체장","2022 광역 비례",
            "2024 총선 비례",
            "2025 대선",  # keep
        ]

        long_df = long_df[long_df["선거명_표시"].isin(ORDER_LABELS)].copy()
        long_df["__xorder__"] = pd.Categorical(long_df["선거명_표시"], categories=ORDER_LABELS, ordered=True)
        long_df = long_df.sort_values(["계열","__xorder__"]).reset_index(drop=True)

        party_order = ["민주","보수","진보","기타"]
        colors      = ["#152484", "#E61E2B", "#7B2CBF", "#6C757D"]  # TUNE: series colors

        x_shared = alt.X(
            "선거명_표시:N",
            sort=None,
            scale=alt.Scale(domain=ORDER_LABELS),
            axis=alt.Axis(labelAngle=-32, labelOverlap=False, labelPadding=20, labelLimit=280, title="선거명")
        )

        base = alt.Chart(long_df)
        lines = base.mark_line(point=False, strokeWidth=2).encode(
            x=x_shared,
            y=alt.Y("득표율:Q", axis=alt.Axis(title="득표율(%)")),
            color=alt.Color("계열:N",
                            scale=alt.Scale(domain=party_order, range=colors),
                            legend=alt.Legend(title=None, orient="top", direction="horizontal", columns=4)),
            detail="계열:N"
        )

        sel = alt.selection_point(fields=["선거명_표시","계열"], nearest=True, on="pointerover", empty=False)

        hit = base.mark_circle(size=650, opacity=0).encode(
            x=x_shared, y="득표율:Q",
            color=alt.Color("계열:N", scale=alt.Scale(domain=party_order, range=colors), legend=None),
            detail="계열:N"
        ).add_params(sel)

        pts = base.mark_circle(size=120).encode(
            x=x_shared, y="득표율:Q",
            color=alt.Color("계열:N", scale=alt.Scale(domain=party_order, range=colors), legend=None),
            opacity=alt.condition(sel, alt.value(1), alt.value(0)),
            detail="계열:N",
            tooltip=[alt.Tooltip("선거명_표시:N", title="선거명"),
                     alt.Tooltip("계열:N", title="계열"),
                     alt.Tooltip("득표율:Q", title="득표율(%)", format=".2f")]
        ).transform_filter(sel)

        zoomX = alt.selection_interval(bind='scales', encodings=['x'])
        chart = (lines + hit + pts).properties(height=box_height_px).add_params(zoomX).configure_view(stroke=None)
        st.altair_chart(chart, use_container_width=True, theme=None)

# =========================================================
# 2024 Results (card)
# TUNE: html_component height, chip colors in _party_chip_color.
# =========================================================
def _party_chip_color(name: str) -> tuple[str, str]:
    s = (name or "").strip()
    mapping = [
        ("더불어민주당", ("#152484", "rgba(21,36,132,.08)")),
        ("국민의힘", ("#E61E2B", "rgba(230,30,43,.10)")),
        ("개혁신당", ("#798897", "rgba(121,136,151,.12)")),
    ]
    for key, col in mapping:
        if key in s:
            return col
    return ("#334155", "rgba(51,65,85,.08)")

def render_results_2024_card(res_sel: pd.DataFrame | None, *, df_24_all: pd.DataFrame | None = None, code: str | None = None):
    with st.container(border=True, height="stretch"):
        st.markdown("**24년 총선결과**")
        if res_sel is None or res_sel.empty:
            st.info("해당 선거구의 24년 결과 데이터가 없습니다.")
            return

        res_row = _norm_cols(res_sel)
        try:
            if "연도" in res_row.columns:
                c = res_row.dropna(subset=["연도"]).copy()
                c["__y__"] = pd.to_numeric(c["연도"], errors="coerce")
                if (c["__y__"] == 2024).any():
                    r = c[c["__y__"] == 2024].iloc[0]
                else:
                    r = c.loc[c["__y__"].idxmax()]
            else:
                r = res_row.iloc[0]
        except Exception:
            r = res_row.iloc[0]

        def share_col(n):
            for cand in (f"후보{n}_득표율", f"후보{n}_득표율(%)"):
                if cand in res_row.columns: return cand
            return None

        name_cols = [c for c in res_row.columns if re.match(r"^후보\d+_이름$", c)]
        pairs = []
        for nc in name_cols:
            n = re.findall(r"\d+", nc)[0]
            sc = share_col(n)
            if not sc: continue
            nm = str(r.get(nc)) if pd.notna(r.get(nc)) else None
            sh = _to_pct_float(r.get(sc))
            if nm and isinstance(sh, (int, float)):
                pairs.append((nm, sh))

        if pairs:
            pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
            top2 = pairs[:2] if len(pairs) >= 2 else [pairs[0], ("2위", None)]
        else:
            c1n = next((c for c in ["후보1_이름", "1위이름", "1위 후보"] if c in res_row.columns), None)
            c1v = next((c for c in ["후보1_득표율", "1위득표율", "1위득표율(%)"] if c in res_row.columns), None)
            c2n = next((c for c in ["후보2_이름", "2위이름", "2위 후보"] if c in res_row.columns), None)
            c2v = next((c for c in ["후보2_득표율", "2위득표율", "2위득표율(%)"] if c in res_row.columns), None)
            top2 = [
                (str(r.get(c1n)) if c1n else "1위", _to_pct_float(r.get(c1v))),
                (str(r.get(c2n)) if c2n else "2위", _to_pct_float(r.get(c2v))),
            ]

        name1, share1 = top2[0][0] or "1위", top2[0][1]
        if len(top2) > 1:
            name2, share2 = top2[1][0] or "2위", top2[1][1]
        else:
            name2, share2 = "2위", None

        if isinstance(share1, (int, float)) and isinstance(share2, (int, float)):
            gap = round(share1 - share2, 2)
        else:
            try:
                gap = compute_24_gap(df_24_all, code) if (df_24_all is not None and code is not None) else None
            except Exception:
                gap = None

        c1_fg, c1_bg = _party_chip_color(name1)
        c2_fg, c2_bg = _party_chip_color(name2)

        def split(nm: str):
            parts = (nm or "").split()
            return (parts[0], " ".join(parts[1:])) if len(parts) >= 2 else (nm, "")

        p1, cand1 = split(name1)
        p2, cand2 = split(name2)
        gap_txt = f"{gap:.2f} %p" if isinstance(gap,(int,float)) else "N/A"

        html = f"""
        <div style="display:grid; grid-template-columns: 1fr 1fr; align-items:center; gap:0; padding:12px 8px 4px 8px;">
          <div style="text-align:center; padding:10px 6px;">
            <div style="display:inline-flex; padding:6px 10px; border-radius:14px; font-weight:600; color:{c1_fg}; background:{c1_bg};">{p1}</div>
            <div style="font-weight:700; margin-top:8px; color:#111827;">{_fmt_pct(share1)}</div>
            <div style="opacity:.8;">{cand1}</div>
          </div>
          <div style="text-align:center; padding:10px 6px; border-left:1px solid #EEF2F7;">
            <div style="display:inline-flex; padding:6px 10px; border-radius:14px; font-weight:600; color:{c2_fg}; background:{c2_bg};">{p2}</div>
            <div style="font-weight:700; margin-top:8px; color:#111827;">{_fmt_pct(share2)}</div>
            <div style="opacity:.8;">{cand2}</div>
          </div>
          <div style="grid-column: 1 / -1; text-align:center; padding:12px 8px 8px; border-top:1px solid #EEF2F7;">
            <div style="color:#6B7280; font-weight:600; margin-bottom:4px;">1~2위 격차</div>
            <div style="font-weight:700; color:#111827;">{gap_txt}</div>
          </div>
        </div>
        """
        from streamlit.components.v1 import html as html_component
        html_component(html, height=250, scrolling=False)

# =========================================================
# Incumbent card
# TUNE: chip colors via _party_chip_color; list bullets via CSS.
# =========================================================
def render_incumbent_card(cur_sel: pd.DataFrame | None):
    with st.container(border=True, height="stretch"):
        st.markdown("**현직정보**")
        if cur_sel is None or cur_sel.empty:
            st.info("현직 정보 데이터가 없습니다.")
            return

        cur_row = _norm_cols(cur_sel)
        r = cur_row.iloc[0]

        name_col = next((c for c in ["의원명", "이름", "성명"] if c in cur_row.columns), None)
        party_col = next((c for c in ["정당", "소속정당"] if c in cur_row.columns), None)
        term_col = next((c for c in ["선수", "당선횟수"] if c in cur_row.columns), None)
        age_col = next((c for c in ["연령", "나이"] if c in cur_row.columns), None)
        gender_col = next((c for c in ["성별"] if c in cur_row.columns), None)

        career_cols = [c for c in ["최근경력", "주요경력", "경력", "이력", "최근 활동"] if c in cur_row.columns]
        raw = None
        for c in career_cols:
            v = str(r.get(c))
            if v and v.lower() not in ("nan", "none"):
                raw = v; break

        def _split(s: str) -> list[str]:
            if not s: return []
            return [p.strip() for p in re.split(r"[;\n•·/]+", s) if p.strip()]

        items = _split(raw)

        name = str(r.get(name_col, "정보없음")) if name_col else "정보없음"
        party = str(r.get(party_col, "정당미상")) if party_col else "정당미상"
        term = str(r.get(term_col, "N/A")) if term_col else "N/A"
        gender = str(r.get(gender_col, "N/A")) if gender_col else "N/A"
        age = str(r.get(age_col, "N/A")) if age_col else "N/A"

        fg, bg = ("#334155", "rgba(51,65,85,.08)")
        if party:
            fg, bg = _party_chip_color(party)

        items_html = "".join([f"<li>{p}</li>" for p in items])
        html = f"""
        <div style="display:flex; flex-direction:column; gap:8px; margin-top:2px; padding:0 8px;">
          <div style="display:flex; flex-wrap:wrap; align-items:center; gap:8px;">
            <div style="font-weight:700; color:#111827;">{name}</div>
            <div style="display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:999px; font-weight:600; color:{fg}; background:{bg};">
              {party}
            </div>
          </div>
          <ul style="margin:0; padding-left:1.1rem; color:#374151;">
            <li>선수: {term}</li><li>성별: {gender}</li><li>연령: {age}</li>
            {"<li>최근 경력</li><ul style='margin:.2rem 0 0 0.1rem;'>"+items_html+"</ul>" if items_html else ""}
          </ul>
        </div>
        """
        from streamlit.components.v1 import html as html_component
        html_component(html, height=250, scrolling=False)

# =========================================================
# Progressive party box (KPI + mini two-bar)
# TUNE: KPI font sizes, mini-bar height, tick step, colors.
# =========================================================
def render_prg_party_box(prg_sel: pd.DataFrame | None, *, df_idx_all: pd.DataFrame | None = None):
    with st.container(border=True, height="stretch"):
        st.markdown("**진보당 현황**")
        if prg_sel is None or prg_sel.empty:
            st.info("지표 소스(index_sample.csv)에서 행을 찾지 못했습니다."); return

        df = prg_sel.copy()
        df.columns = [" ".join(str(c).replace("\n"," ").replace("\r"," ").strip().split()) for c in df.columns]
        r = df.iloc[0]

        def find_col_exact_or_compact(df, prefer_name, compact_key):
            if prefer_name in df.columns: return prefer_name
            for c in df.columns:
                if compact_key in str(c).replace(" ",""): return c
            return None

        col_strength = find_col_exact_or_compact(df, "진보정당 득표력", "진보정당득표력")
        col_members  = find_col_exact_or_compact(df, "진보당 당원수", "진보당당원수")

        strength = _to_pct_float(r.get(col_strength)) if col_strength else None
        members  = int(float(r.get(col_members))) if (col_members and pd.notna(r.get(col_members))) else None

        html = f"""
        <div style="display:grid; grid-template-columns: 1fr 1fr; align-items:center; gap:12px; margin:0; padding:0 8px;">
            <div style="text-align:center;">
                <div style="color:#6B7280; font-weight:600; margin-bottom:2px;">진보 득표력</div>
                <div style="font-weight:800; color:#111827;">{_fmt_pct(strength) if strength is not None else 'N/A'}</div>
            </div>
            <div style="text-align:center;">
                <div style="color:#6B7280; font-weight:600; margin-bottom:2px;">진보당 당원수</div>
                <div style="font-weight:800; color:#111827;">{(f"{members:,}명" if isinstance(members,int) else "N/A")}</div>
            </div>
        </div>
        """
        from streamlit.components.v1 import html as html_component
        html_component(html, height=100, scrolling=False)

        try:
            avg_strength = None
            if df_idx_all is not None and not df_idx_all.empty:
                cols_norm = [" ".join(str(c).replace("\n"," ").replace("\r"," ").strip().split()) for c in df_idx_all.columns]
                key_cs = col_strength if (col_strength and col_strength in df.columns) else None
                if not key_cs:
                    key_cs = next((c for c in cols_norm if "진보정당득표력" in c), None)
                    key_cs = df_idx_all.columns[cols_norm.index(key_cs)] if key_cs else None
                if key_cs and key_cs in df_idx_all.columns:
                    vals = pd.to_numeric(df_idx_all[key_cs].astype(str).str.replace("%","", regex=False), errors="coerce")
                    avg_strength = float(vals.mean()) if vals.notna().any() else None

            if strength is not None and avg_strength is not None:
                bar_df = pd.DataFrame({
                    "항목": ["해당 지역", "10개 평균"],
                    "값": [strength/100.0 if strength>1 else strength,
                          (avg_strength/100.0 if avg_strength>1 else avg_strength)]
                })
                bar_df["색상"] = bar_df["항목"].map(lambda x: "#1E6BFF" if x == "해당 지역" else "#9CA3AF")

                mini = (
                    alt.Chart(bar_df)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "값:Q",
                            axis=alt.Axis(
                                title=None,
                                format=".0%",
                                values=[v / 100 for v in range(0, 101, 2)]
                            ),
                            scale=alt.Scale(domain=[0, 0.1], nice=False),
                        ),
                        y=alt.Y("항목:N", title=None, sort=["해당 지역", "10개 평균"]),
                        color=alt.condition(
                            alt.datum["항목"] == "해당 지역",
                            alt.value(COLOR_BLUE),
                            alt.value(COLOR_GRAY),
                        ),
                        tooltip=[
                            alt.Tooltip("항목:N", title="구분"),
                            alt.Tooltip("값:Q", title="비율", format=".2%"),
                        ],
                    )
                    .properties(
                        height=110,
                        padding={"top": 0, "bottom": 0, "left": 0, "right": 0},
                    )
                    .configure_view(stroke=None)
                )
                st.altair_chart(mini, use_container_width=True, theme=None)
        except Exception:
            pass

# =========================================================
# Region detail layout
# (Do not change structure; only calls simplified functions above.)
# =========================================================
def render_region_detail_layout(
    *,
    df_pop_sel: pd.DataFrame | None,
    df_pop_all: pd.DataFrame | None,
    df_trend_sel: pd.DataFrame | None,
    df_trend_all: pd.DataFrame | None,
    df_24_sel: pd.DataFrame | None,
    df_24_all: pd.DataFrame | None,
    df_cur_sel: pd.DataFrame | None,
    df_idx_sel: pd.DataFrame | None,
    df_idx_all: pd.DataFrame | None,
    bookmark_map: dict | None,
    page_title: str | None,
    app_title: str | None,
):
    _render_topbar(page_title, app_title)

    st.markdown("### 👥 인구 정보")
    with st.container():
        col1, col2, col3 = st.columns([1.4, 1.5, 2.6], gap="small")
        with col1.container(border=True, height="stretch"):
            render_population_box(df_pop_sel, df_pop_all=df_pop_all, bookmark_map=bookmark_map)
        with col2.container(border=True, height="stretch"):
            st.markdown("**연령 구성**")
            render_age_highlight_chart(df_pop_sel, bookmark_map=bookmark_map)
        with col3.container(border=True, height="stretch"):
            st.markdown("**연령별, 성별 인구분포**")
            render_sex_ratio_bar(df_pop_sel, bookmark_map=bookmark_map)

    st.markdown("### 📈 정당성향별 득표추이")
    render_vote_trend_chart(df_trend_sel, ts_all=df_trend_all, box_height_px=420)

    st.markdown("### 🗳️ 선거 결과 및 정치지형")
    with st.container():
        c1, c2, c3 = st.columns(3, gap="small")
        with c1.container(height="stretch"):
            render_results_2024_card(df_24_sel, df_24_all=df_24_all)
        with c2.container(height="stretch"):
            render_incumbent_card(df_cur_sel)
        with c3.container(height="stretch"):
            render_prg_party_box(df_idx_sel, df_idx_all=df_idx_all)





















