# charts.py
# Purpose: All visuals. No file I/O here. Bookmark-first column resolution.
# How to change later:
# - Colors/sizes per chart: search "HOW TO CHANGE LATER" inside each function.
# - To tweak axis order for vote-trend: edit ORDER_LABELS in render_vote_trend_chart.

from __future__ import annotations
import re, math
import pandas as pd
import streamlit as st
import altair as alt
from metrics import compute_24_gap

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
    return f"{x:.1f}%" if isinstance(x, (int, float)) else "N/A"

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
# Population Box – KPI + two-bars (Region vs 10-avg)
# HOW TO CHANGE LATER:
# - Change bar height: set bar_h (e.g., 140~190).
# - Change blue/gray: edit `COLOR_BLUE` / `COLOR_GRAY` below.
# - Change x-axis number format: edit axis format in alt.Axis.
# =========================================================
def render_population_box(pop_sel: pd.DataFrame, *, df_pop_all: pd.DataFrame, bookmark_map: dict | None = None, box_height_px: int = 240):
    if pop_sel is None or pop_sel.empty:
        st.info("인구 데이터가 없습니다."); return
    df_s = _norm_cols(pop_sel.copy())
    df_a = _norm_cols(df_pop_all.copy()) if df_pop_all is not None else pd.DataFrame()

    COLOR_BLUE = "#1E6BFF"
    COLOR_GRAY = "#9CA3AF"

    total_col = _col(df_s, bookmark_map, "total_voters", ["전체 유권자 수","전체 유권자","전체유권자","total_voters"])
    float_col = _col(df_s, bookmark_map, "floating", ["유동인구","전입전출","전입+전출","유출입","floating_pop"], required=False)

    df_s[total_col] = df_s[total_col].apply(_to_num)
    if float_col: df_s[float_col] = df_s[float_col].apply(_to_num)

    region_total = float(df_s[total_col].sum())

    avg_total = None
    if df_a is not None and not df_a.empty:
        if total_col in df_a.columns:
            avg_total = float(pd.to_numeric(df_a[total_col].apply(_to_num)).groupby(0).mean()) if df_a.shape[0] == 1 else float(pd.to_numeric(df_a[total_col].apply(_to_num)).mean())
        else:
            # fallback: compute mean of per-code totals if same column exists under different casing
            if total_col in [str(c) for c in df_a.columns]:
                avg_total = float(pd.to_numeric(df_a[total_col].apply(_to_num)).mean())

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
        floating_value_txt = (f"{int(round(float(df_s[float_col].sum()))):,}명" if float_col else "N/A")
        st.markdown(
            f"""
            <div style="text-align:center;">
              <div style="color:#6B7280; font-weight:600; margin-bottom:4px;">유동인구</div>
              <div style="font-weight:800; color:#111827;">{floating_value_txt}</div>
            </div>
            """, unsafe_allow_html=True
        )

    bar_h = 170
    if isinstance(avg_total,(int,float)) and avg_total and avg_total>0:
        bar_df = pd.DataFrame({"항목": ["해당 지역", "10개 평균"], "값": [float(region_total), float(avg_total)]})
        x_max = max(float(region_total), float(avg_total)) * 1.1
    else:
        bar_df = pd.DataFrame({"항목": ["해당 지역"], "값": [float(region_total)]})
        x_max = float(region_total) * 1.1 if region_total > 0 else 1.0

    chart = (
        alt.Chart(bar_df)
        .mark_bar()
        .encode(
            y=alt.Y("항목:N", title=None, axis=alt.Axis(labels=True, ticks=False)),
            x=alt.X("값:Q", title=None, axis=alt.Axis(format="~,"), scale=alt.Scale(domain=[0, x_max])),
            color=alt.condition(alt.datum.항목 == "해당 지역", alt.value(COLOR_BLUE), alt.value(COLOR_GRAY)),
            tooltip=[alt.Tooltip("항목:N", title="구분"), alt.Tooltip("값:Q", title="유권자수", format=",.0f")],
        )
    ).properties(height=bar_h, padding={"left":0, "right":0, "top":4, "bottom":2}).configure_view(stroke=None)
    st.altair_chart(chart, use_container_width=True, theme=None)

# =========================================================
# Age Composition (Half donut)
# HOW TO CHANGE LATER:
# - Change donut size: tweak inner_r/outer_r.
# - Change label positions: change TXT_NUM_Y / TXT_LBL_Y.
# =========================================================
def render_age_highlight_chart(pop_sel: pd.DataFrame, *, bookmark_map: dict | None = None, box_height_px: int = 240):
    df = _norm_cols(pop_sel.copy()) if pop_sel is not None else pd.DataFrame()
    if df.empty: st.info("연령 구성 데이터가 없습니다."); return

    Y, M, O = "청년층(18~39세)", "중년층(40~59세)", "고령층(65세 이상)"
    total_col = _col(df, bookmark_map, "total_voters", ["전체 유권자 수","전체 유권자","전체유권자","total_voters"])
    for c in (Y, M, O):
        if c not in df.columns:
            st.info(f"연령대 컬럼이 없습니다: {c}"); return

    for c in [Y, M, O, total_col]:
        df[c] = pd.to_numeric(df[c].astype(str).replace(",","", regex=False).str.strip(), errors="coerce").fillna(0)

    y, m, o = float(df[Y].sum()), float(df[M].sum()), float(df[O].sum())
    tot = float(df[total_col].sum())
    if tot <= 0: st.info("전체 유권자 수(분모)가 0입니다."); return

    mid_60_64 = max(0.0, tot - (y + m + o))
    labels_order = [Y, M, "60–64세", O]
    values = [y, m, mid_60_64, o]
    ratios01  = [v/tot for v in values]
    ratios100 = [r*100 for r in ratios01]

    focus = st.radio("강조", [Y, M, O], index=0, horizontal=True, label_visibility="collapsed")
    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    inner_r, outer_r = 68, 106
    W = 320
    H = max(220, int(box_height_px))

    df_vis = pd.DataFrame({
        "연령": labels_order, "명": values, "비율": ratios01, "표시비율": ratios100,
        "강조": [l == focus for l in labels_order], "순서": [1, 2, 3, 4],
    })

    base = (
        alt.Chart(df_vis, width=W, height=H)
        .mark_arc(innerRadius=inner_r, outerRadius=outer_r, cornerRadius=6, stroke="white", strokeWidth=1)
        .encode(
            theta=alt.Theta("비율:Q", stack=True, sort=None, scale=alt.Scale(range=[-math.pi/2, math.pi/2])),
            order=alt.Order("순서:Q"),
            color=alt.Color("강조:N", scale=alt.Scale(domain=[True, False], range=["#1E6BFF", "#E5E7EB"]), legend=None),
            tooltip=[alt.Tooltip("연령:N", title="연령대"),
                     alt.Tooltip("명:Q", title="인원", format=",.0f"),
                     alt.Tooltip("표시비율:Q", title="비율(%)", format=".1f")],
        )
    )

    label_map = {Y: "청년층(18~39세)", M: "중년층(40~59세)", O: "고령층(65세 이상)"}
    idx = labels_order.index(focus)
    pct_txt = f"{(ratios100[idx]):.1f}%"

    NUM_FONT, LBL_FONT = 28, 14
    TXT_NUM_Y = outer_r + 30
    TXT_LBL_Y = outer_r + 54

    num_text = (
        alt.Chart(pd.DataFrame({"t":[pct_txt]}), width=W, height=H)
        .mark_text(fontWeight="bold", fontSize=NUM_FONT, color="#0f172a")
        .encode(text="t:N", x=alt.value(W/2), y=alt.value(TXT_NUM_Y))
    )
    lbl_text = (
        alt.Chart(pd.DataFrame({"t":[label_map.get(focus, focus)]}), width=W, height=H)
        .mark_text(fontSize=LBL_FONT, color="#475569")
        .encode(text="t:N", x=alt.value(W/2), y=alt.value(TXT_LBL_Y))
    )

    st.altair_chart((base + num_text + lbl_text).configure_view(stroke=None), use_container_width=True, theme=None)

# =========================================================
# Sex ratio by age – horizontal bars
# HOW TO CHANGE LATER:
# - Change x-axis max (0.30 -> 0.35 etc.).
# - Change bar size: bar_size.
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

    bar_size = 30
    bars = (
        alt.Chart(tidy)
        .mark_bar(size=bar_size)
        .encode(
            y=alt.Y("연령대표시:N", sort=[label_map[a] for a in age_buckets], title=None),
            x=alt.X("전체비중:Q", scale=alt.Scale(domain=[0, 0.30]), axis=alt.Axis(format=".0%", title="전체 기준 구성비(%)", grid=True)),
            color=alt.Color("성별:N", scale=alt.Scale(domain=["남성","여성"]), legend=alt.Legend(title=None, orient="top")),
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
# HOW TO CHANGE LATER:
# - To change global x order: edit ORDER_LABELS (keep "2025 대선").
# - To change legend orientation: edit legend params.
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
        colors      = ["#152484", "#E61E2B", "#7B2CBF", "#6C757D"]

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
                     alt.Tooltip("득표율:Q", title="득표율(%)", format=".1f")]
        ).transform_filter(sel)

        zoomX = alt.selection_interval(bind='scales', encodings=['x'])
        chart = (lines + hit + pts).properties(height=box_height_px).add_params(zoomX).configure_view(stroke=None)
        st.altair_chart(chart, use_container_width=True, theme=None)

# =========================================================
# 2024 Results (card)
# HOW TO CHANGE LATER:
# - Card height: html_component(..., height=250)
# - Chip colors: _party_chip_color mapping.
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
        gap_txt = f"{gap:.1f} %p" if isinstance(gap,(int,float)) else "N/A"

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
            # reuse from results card
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
            {"<li>최근 경력</li><ul style='margin:.2rem 0 0 0.8rem;'>"+items_html+"</ul>" if items_html else ""}
          </ul>
        </div>
        """
        from streamlit.components.v1 import html as html_component
        html_component(html, height=250, scrolling=False)

# =========================================================
# Progressive party box (KPI + mini two-bar)
# HOW TO CHANGE LATER:
# - KPI font sizes: inline HTML <div> font-weight/size.
# - Mini bar ticks: values step (2%) and chart height (110).
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
                            axis=alt.Axis(title=None, format=".0%", values=[v/100 for v in range(0, 101, 2)]),
                            scale=alt.Scale(domain=[0, float(bar_df["값"].max())*1.1], nice=False)
                        ),
                        y=alt.Y("항목:N", title=None, sort=["해당 지역", "10개 평균"]),
                        color=alt.Color("색상:N", scale=None, legend=None),
                        tooltip=[alt.Tooltip("항목:N"), alt.Tooltip("값:Q", format=".2%")]
                    )
                    .properties(height=110, padding={"top":0,"bottom":0,"left":0,"right":0})
                    .configure_view(stroke=None)
                )
                st.altair_chart(mini, use_container_width=True, theme=None)
        except Exception:
            pass

# =========================================================
# Region detail layout
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
        col1, col2, col3 = st.columns([1.25, 1.35, 2.85], gap="small")
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

