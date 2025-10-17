# =============================
# charts.py
# =============================
from __future__ import annotations

import re, math
import pandas as pd
import streamlit as st
import altair as alt
from metrics import compute_24_gap  # Safe import: used only if available

# ---------------------------------------------
# [Altair Base Setup]
# - Disable max_rows to avoid down-sampling.
# - No Streamlit theme overrides or CSS hacks.
# ---------------------------------------------
alt.data_transformers.disable_max_rows()

# ---------------------------------------------
# [Utils]
# ---------------------------------------------
def _norm(x) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^\w]+", "", s)
    return s

def _to_pct_float(v, default=None):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    try:
        s = str(v).strip().replace(",", "")
        m = re.match(r"^\s*([+-]?\d+(\.\d+)?)\s*%?\s*$", s)
        if not m:
            return default
        x = float(m.group(1))
        if "%" in s:
            return x
        return x * 100.0 if 0 <= x <= 1 else x
    except Exception:
        return default

def _to_float(v, default=None):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        s = str(v).replace(",", "").strip()
        return float(s) if s not in ("", "nan", "None") else default
    except Exception:
        return default

def _to_int(v, default=None):
    f = _to_float(v, default=None)
    try:
        return int(f) if f is not None else default
    except Exception:
        return default

def _fmt_pct(x):
    return f"{x:.2f}%" if isinstance(x, (int, float)) else "N/A"

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c).strip().replace("\n", "").replace("\r", "") for c in out.columns]
    return out

def _load_index_df() -> pd.DataFrame | None:
    paths = [
        "sti/data/index_sample.csv", "./sti/data/index_sample.csv",
        "data/index_sample.csv", "./data/index_sample.csv",
        "index_sample.csv",
    ]
    for path in paths:
        try:
            return _norm_cols(pd.read_csv(path))
        except FileNotFoundError:
            continue
        except UnicodeDecodeError:
            try:
                return _norm_cols(pd.read_csv(path, encoding="cp949"))
            except Exception:
                continue
        except Exception:
            continue
    return None

def _load_population_master() -> pd.DataFrame | None:
    candidates = [
        "sti/data/population.csv", "./sti/data/population.csv",
        "data/population.csv", "./data/population.csv",
        "population.csv",
    ]
    for p in candidates:
        try:
            return _norm_cols(pd.read_csv(p))
        except FileNotFoundError:
            continue
        except UnicodeDecodeError:
            try:
                return _norm_cols(pd.read_csv(p, encoding="cp949"))
            except Exception:
                continue
        except Exception:
            continue
    return None

# ---------------------------------------------
# [Colors]
# ---------------------------------------------
COLOR_TEXT_DARK = "#111827"
COLOR_BLUE      = "#1E6BFF"

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

# =========================================================
# [Population Box] KPI + two-bars (Region vs 10-avg)
# HOW TO CHANGE LATER:
#  - To switch back to ratio bar, set use_two_bars=False (kept logic).
#  - To adjust bar height, tweak bar_h (e.g., 150~190).
# (REQ 1) Two bars: blue = region, gray = 10-avg.
# =========================================================
def render_population_box(pop_df: pd.DataFrame, *, box_height_px: int = 240):
    if pop_df is None or pop_df.empty:
        st.info("유동인구/연령/성비 차트를 위한 데이터가 없습니다.")
        return

    df = _norm_cols(pop_df.copy())
    CAND_CODE  = ["지역구코드","선거구코드","코드","code","CODE"]
    CAND_TOTAL = ["전체 유권자 수","전체 유권자","전체유권자","total_voters","TOTAL_VOTERS"]
    CAND_FLOAT = ["유동인구","전입전출","전입+전출","유출입","floating_pop","FLOATING"]

    code_col  = next((c for c in CAND_CODE  if c in df.columns), None)
    total_col = next((c for c in CAND_TOTAL if c in df.columns), None)
    float_col = next((c for c in CAND_FLOAT if c in df.columns), None)
    if not total_col:
        st.error("population.csv에서 '전체 유권자' 컬럼을 찾지 못했습니다."); return

    def _to_num(x):
        if pd.isna(x): return 0.0
        if isinstance(x,(int,float)): return float(x)
        try: return float(str(x).replace(",","").strip())
        except: return 0.0

    df[total_col] = df[total_col].apply(_to_num)
    if float_col: df[float_col] = df[float_col].apply(_to_num)

    # --- Region aggregate (robust to multi-rows) ---
    if code_col:
        grp = df.groupby(code_col, dropna=False)[[total_col]].sum(min_count=1).reset_index()
        region_total = float(grp[total_col].iloc[0]); region_cnt = int(grp.shape[0])
    else:
        region_total = float(df[total_col].sum());    region_cnt = 1

    # --- 10-avg fallback (when single region given) ---
    avg_total = None
    if region_cnt >= 2:
        avg_total = float(df.groupby(code_col, dropna=False)[total_col].sum().mean())
    else:
        pop_all = _load_population_master()
        if pop_all is not None:
            tcol = next((c for c in CAND_TOTAL if c in pop_all.columns), None)
            ccol = next((c for c in CAND_CODE  if c in pop_all.columns), None)
            if tcol:
                pop_all[tcol] = pop_all[tcol].apply(_to_num)
                avg_total = float(pop_all.groupby(ccol, dropna=False)[tcol].sum().mean()) if ccol else float(pop_all[tcol].mean())

    # --- KPIs (simple vertical stack; keep typography consistent) ---
    floating_value_txt = (f"{int(round(float(df[float_col].sum()))):,}명" if float_col else "N/A")
    st.markdown(
        f"""
        <div style="display:flex; flex-direction:column; gap:6px; margin-top:2px;">
          <div style="text-align:center;">
            <div style="color:#6B7280; font-weight:600; margin-bottom:4px;">전체 유권자 수</div>
            <div style="font-weight:800; color:{COLOR_TEXT_DARK};">{int(round(region_total)):,}명</div>
          </div>
          <div style="text-align:center; margin-top:2px;">
            <div style="color:#6B7280; font-weight:600; margin-bottom:4px;">유동인구</div>
            <div style="font-weight:800; color:{COLOR_TEXT_DARK};">{floating_value_txt}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Two-bar chart (match Progressive mini chart style; numbers instead of %) ---
    use_two_bars = True
    bar_h = 110
    if use_two_bars:
        if isinstance(avg_total, (int, float)) and avg_total and avg_total > 0:
            bar_df = pd.DataFrame({"항목": ["해당 지역", "10개 평균"], "값": [float(region_total), float(avg_total)]})
            x_max = max(float(region_total), float(avg_total)) * 1.1
        else:
            bar_df = pd.DataFrame({"항목": ["해당 지역"], "값": [float(region_total)]})
            x_max = float(region_total) * 1.1 if region_total > 0 else 1.0

        # ✅ Precomputed color column (robust; avoids predicates/expressions)
        bar_df["색상"] = bar_df["항목"].map(lambda x: COLOR_BLUE if x == "해당 지역" else "#9CA3AF")

        chart = (
            alt.Chart(bar_df)
            .mark_bar()
            .encode(
                x=alt.X("값:Q",
                        axis=alt.Axis(format="~,", title=None),
                        scale=alt.Scale(domain=[0, x_max])),
                y=alt.Y("항목:N", title=None, sort=["해당 지역","10개 평균"]),
                color=alt.Color("색상:N", scale=None, legend=None),
                tooltip=[
                    alt.Tooltip("항목:N", title="구분"),
                    alt.Tooltip("값:Q", title="유권자수", format=",.0f")
                ]
            )
            .properties(height=bar_h, padding={"top":0,"bottom":0,"left":0,"right":0})
            .configure_view(stroke=None)
        )
        st.altair_chart(chart, use_container_width=True, theme=None)

# =========================================================
# [Age Composition: Half Donut]
# HOW TO CHANGE LATER:
#  - To move the number/label closer/farther, change text_y_num/text_y_lbl.
#  - To change sizes, adjust num_font_px / lbl_font_px.
# (REQ 2) Place emphasis number+label INSIDE the same chart (just below donut),
#         not below the container; implemented as layered text.
# =========================================================
def render_age_highlight_chart(pop_df: pd.DataFrame, *, box_height_px: int = 240):
    df = _norm_cols(pop_df.copy()) if pop_df is not None else pd.DataFrame()
    if df.empty:
        st.info("연령 구성 데이터가 없습니다."); return

    Y, M, O = "청년층(18~39세)", "중년층(40~59세)", "고령층(65세 이상)"
    T_CANDS = ["전체 유권자 수","전체 유권자","전체유권자","total_voters"]
    for c in (Y, M, O):
        if c not in df.columns:
            st.info(f"연령대 컬럼이 없습니다: {c}"); return
    total_col = next((c for c in T_CANDS if c in df.columns), None)
    if total_col is None:
        st.info("'전체 유권자 수' 컬럼이 없습니다."); return

    for c in [Y, M, O, total_col]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce").fillna(0)

    y, m, o = float(df[Y].sum()), float(df[M].sum()), float(df[O].sum())
    tot = float(df[total_col].sum())
    if tot <= 0: st.info("전체 유권자 수(분모)가 0입니다."); return

    mid_60_64 = max(0.0, tot - (y + m + o))
    labels_order = [Y, M, "60–64세", O]
    values = [y, m, mid_60_64, o]
    ratios01  = [v/tot for v in values]
    ratios100 = [r*100 for r in ratios01]

    focus = st.radio("강조", [Y, M, O], index=0, horizontal=True, label_visibility="collapsed")

    # Donut base
    inner_r, outer_r = 68, 106
    df_vis = pd.DataFrame({
        "연령": labels_order, "명": values, "비율": ratios01, "표시비율": ratios100,
        "강조": [l == focus for l in labels_order], "순서": [1, 2, 3, 4],
    })
    W = 320
    H = max(220, int(box_height_px))  # a bit taller to host texts under donut

    # Donut base (⚠️ NOTE: do NOT call .configure_* here; apply it AFTER vconcat)
    base = (
        alt.Chart(df_vis)
        .mark_arc(innerRadius=inner_r, outerRadius=outer_r, cornerRadius=6, stroke="white", strokeWidth=1)
        .encode(
            theta=alt.Theta("비율:Q", stack=True, sort=None, scale=alt.Scale(range=[-math.pi/2, math.pi/2])),
            order=alt.Order("순서:Q"),
            color=alt.Color("강조:N", scale=alt.Scale(domain=[True, False], range=[COLOR_BLUE, "#E5E7EB"]), legend=None),
            tooltip=[alt.Tooltip("연령:N", title="연령대"),
                     alt.Tooltip("명:Q", title="인원", format=",.0f"),
                     alt.Tooltip("표시비율:Q", title="비율(%)", format=".1f")],
        )
        .properties(width=W, height=H)
        # .configure_view(stroke=None)  # <-- ❌ remove here: config on subchart causes TypeError in vconcat
    )

    # Robust caption inside the chart area using a tiny vconcat text panel
    # English footnotes:
    # - Use 0~1 normalized coords to avoid pixel math & clipping on resize.
    # - Change panel_h to adjust spacing under the donut; num_font_px/lbl_font_px to resize texts.
    label_map = {Y: "청년층(18~39세)", M: "중년층(40~59세)", O: "고령층(65세 이상)"}
    idx = labels_order.index(focus)
    pct_txt = f"{(ratios100[idx]):.1f}%"
    num_font_px = 28
    lbl_font_px = 14
    panel_h = 46

    txt_df = pd.DataFrame({"x":[0.5], "y":[0.5], "num":[pct_txt], "lbl":[label_map.get(focus, focus)]})

    num = (
        alt.Chart(txt_df)
        .mark_text(fontWeight="bold", fontSize=num_font_px, color="#0f172a")
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[0,1]), axis=None),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[0,1]), axis=None),
            text="num:N"
        )
    )
    lbl = (
        alt.Chart(txt_df)
        .mark_text(fontSize=lbl_font_px, color="#475569", dy=23)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[0,1]), axis=None),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[0,1]), axis=None),
            text="lbl:N"
        )
    )
    text_panel = (num + lbl).properties(height=panel_h)

    # Concat first, THEN apply any configure_* at top-level
    chart_all = (
        alt.vconcat(base, text_panel)
        .resolve_scale(x="independent", y="independent")
        .properties(spacing=0)
        .configure_view(stroke=None)  # ✅ top-level config is safe here
    )

    st.altair_chart(chart_all, use_container_width=True, theme=None)

# =========================================================
# [Sex Composition by Age: Horizontal Bars]
# HOW TO CHANGE LATER:
#  - To adjust visual density, tweak bar_size and box_height_px.
#  - To widen the x-range headroom, increase domain max (e.g., 0.30 → 0.35).
# =========================================================
def render_sex_ratio_bar(pop_df: pd.DataFrame, *, box_height_px: int = 340):
    if pop_df is None or pop_df.empty:
        st.info("성비 데이터를 표시할 수 없습니다. (population.csv 없음)")
        return

    df = _norm_cols(pop_df.copy())
    age_buckets = ["20대","30대","40대","50대","60대","70대 이상"]
    expect = [f"{a} 남성" for a in age_buckets] + [f"{a} 여성" for a in age_buckets]
    miss = [c for c in expect if c not in df.columns]
    if miss:
        st.info("성비용 컬럼이 부족합니다: " + ", ".join(miss))
        return

    def _num(x):
        if pd.isna(x): return 0.0
        if isinstance(x,(int,float)): return float(x)
        try: return float(str(x).replace(",","").strip())
        except: return 0.0

    df_num = df[expect].applymap(_num).fillna(0.0)
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

    male_color = "#1E40AF"
    female_color = "#60A5FA"

    bar_size = 30
    bars = (
        alt.Chart(tidy)
        .mark_bar(size=bar_size)
        .encode(
            y=alt.Y("연령대표시:N", sort=[label_map[a] for a in age_buckets], title=None),
            x=alt.X(
                "전체비중:Q",
                scale=alt.Scale(domain=[0, 0.30]),
                axis=alt.Axis(format=".0%", title="전체 기준 구성비(%)", grid=True)
            ),
            color=alt.Color(
                "성별:N",
                scale=alt.Scale(domain=["남성","여성"], range=[male_color, female_color]),
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
# [Vote Trend by Ideology]
# HOW TO CHANGE LATER:
#  - Adjust hover hitbox via HIT_SIZE.
#  - Move legend by Legend(orient=...).
#  - Zoom/pan: interval selection bound to scales on X (robust).
# (REQ 4) Scroll (wheel) & drag zoom on X kept via selection_interval.
# =========================================================
def render_vote_trend_chart(ts: pd.DataFrame, *, box_height_px: int = 420):
    with st.container(border=True):  # outer border (Streamlit >= 1.31)
        import re
        if ts is None or ts.empty:
            st.info("득표 추이 데이터가 없습니다."); return
        df = _norm_cols(ts.copy())

        label_col = next((c for c in ["계열","성향","정당성향","party_label","label"] if c in df.columns), None)
        value_col = next((c for c in ["득표율","비율","share","ratio","pct","prop"] if c in df.columns), None)
        wide_cols = [c for c in ["민주","보수","진보","기타"] if c in df.columns]

        id_col   = next((c for c in ["선거명","election","분류","연도","year"] if c in df.columns), None)
        year_col = next((c for c in ["연도","year"] if c in df.columns), None)

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

        order = long_df["선거명_표시"].astype(str).dropna().unique().tolist()
        party_order = ["민주","보수","진보","기타"]
        color_map   = {"민주":"#152484", "보수":"#E61E2B", "진보":"#7B2CBF", "기타":"#6C757D"}
        colors      = [color_map[p] for p in party_order]

        x_shared = alt.X(
            "선거명_표시:N",
            sort=None,
            scale=alt.Scale(domain=order),
            axis=alt.Axis(labelAngle=-32, labelOverlap=False, labelPadding=6, labelLimit=280, title="선거명")
        )

        base = alt.Chart(long_df)
        lines = base.mark_line(point=False, strokeWidth=2).encode(
            x=x_shared,
            y=alt.Y("득표율:Q", axis=alt.Axis(title="득표율(%)")),
            color=alt.Color(
                "계열:N",
                scale=alt.Scale(domain=party_order, range=colors),
                legend=alt.Legend(title=None, orient="top", direction="horizontal", columns=4)
            )
        )

        sel = alt.selection_point(fields=["선거명_표시","계열"], nearest=True, on="pointerover", empty=False)
        HIT_SIZE = 650
        hit = base.mark_circle(size=HIT_SIZE, opacity=0).encode(
            x=x_shared,
            y="득표율:Q",
            color=alt.Color("계열:N", scale=alt.Scale(domain=party_order, range=colors), legend=None)
        ).add_params(sel)

        pts = base.mark_circle(size=120).encode(
            x=x_shared,
            y="득표율:Q",
            color=alt.Color("계열:N", scale=alt.Scale(domain=party_order, range=colors), legend=None),
            opacity=alt.condition(sel, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("선거명_표시:N", title="선거명"),
                alt.Tooltip("계열:N", title="계열"),
                alt.Tooltip("득표율:Q", title="득표율(%)", format=".1f")
            ]
        ).transform_filter(sel)

        # (REQ 4) Robust scroll/drag zoom on X
        zoomX = alt.selection_interval(bind='scales', encodings=['x'])
        chart = (lines + hit + pts).properties(height=box_height_px).add_params(zoomX).configure_view(stroke=None)

        st.altair_chart(chart, use_container_width=True, theme=None)

# =========================================================
# [2024 Results Card]
# HOW TO CHANGE LATER:
#  - To keep equal card heights, keep html_component height=260.
#  - Inner paddings are only inline CSS below.
# (REQ 5) Unified card size + comfortable inner paddings.
# =========================================================
def render_results_2024_card(res_row_or_df: pd.DataFrame | None, df_24: pd.DataFrame | None = None, code: str | None = None):
    with st.container(border=True):
        st.markdown("**24년 총선결과**")

        if res_row_or_df is None or res_row_or_df.empty:
            st.info("해당 선거구의 24년 결과 데이터가 없습니다.")
            return

        res_row = _norm_cols(res_row_or_df)
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

        name_cols = [c for c in res_row.columns if re.match(r"^후보\d+_이름$", c)]
        def share_col(n):
            for cand in (f"후보{n}_득표율", f"후보{n}_득표율(%)"):
                if cand in res_row.columns: return cand
            return None

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
                gap = compute_24_gap(df_24, code) if (df_24 is not None and code is not None) else None
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
        <div style="display:grid; grid-template-columns: 1fr 1fr; align-items:center; gap:0; padding:6px 8px 2px 8px;">
          <div style="text-align:center; padding:8px 6px;">
            <div style="display:inline-flex; padding:6px 10px; border-radius:14px; font-weight:600; color:{c1_fg}; background:{c1_bg};">{p1}</div>
            <div style="font-weight:700; margin-top:6px; color:{COLOR_TEXT_DARK};">{_fmt_pct(share1)}</div>
            <div style="opacity:.8;">{cand1}</div>
          </div>
          <div style="text-align:center; padding:8px 6px; border-left:1px solid #EEF2F7;">
            <div style="display:inline-flex; padding:6px 10px; border-radius:14px; font-weight:600; color:{c2_fg}; background:{c2_bg};">{p2}</div>
            <div style="font-weight:700; margin-top:6px; color:{COLOR_TEXT_DARK};">{_fmt_pct(share2)}</div>
            <div style="opacity:.8;">{cand2}</div>
          </div>
          <div style="grid-column: 1 / -1; text-align:center; padding:10px 8px 6px; border-top:1px solid #EEF2F7;">
            <div style="color:#6B7280; font-weight:600; margin-bottom:4px;">1~2위 격차</div>
            <div style="font-weight:700; color:{COLOR_TEXT_DARK};">{gap_txt}</div>
          </div>
        </div>
        """
        from streamlit.components.v1 import html as html_component
        html_component(html, height=260, scrolling=False)

# =========================================================
# [Incumbent Card]
# (REQ 5) Keep unified height and tidy paddings.
# =========================================================
def render_incumbent_card(cur_row: pd.DataFrame | None):
    with st.container(border=True):
        st.markdown("**현직정보**")
        if cur_row is None or cur_row.empty:
            st.info("현직 정보 데이터가 없습니다.")
            return

        cur_row = _norm_cols(cur_row)
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
                raw = v
                break

        def _split(s: str) -> list[str]:
            if not s: return []
            return [p.strip() for p in re.split(r"[;\n•·/]+", s) if p.strip()]

        items = _split(raw)

        name = str(r.get(name_col, "정보없음")) if name_col else "정보없음"
        party = str(r.get(party_col, "정당미상")) if party_col else "정당미상"
        term = str(r.get(term_col, "N/A")) if term_col else "N/A"
        gender = str(r.get(gender_col, "N/A")) if gender_col else "N/A"
        age = str(r.get(age_col, "N/A")) if age_col else "N/A"
        fg, bg = _party_chip_color(party)

        items_html = "".join([f"<li>{p}</li>" for p in items])
        html = f"""
        <div style="display:flex; flex-direction:column; gap:8px; margin-top:2px; padding:0 8px;">
          <div style="display:flex; flex-wrap:wrap; align-items:center; gap:8px;">
            <div style="font-weight:700; color:{COLOR_TEXT_DARK};">{name}</div>
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
        html_component(html, height=260, scrolling=False)

# =========================================================
# [Progressive Party Box]
# (REQ 5) Match height by splitting KPI (~140) + mini chart (~110).
# =========================================================
def render_prg_party_box(prg_row: pd.DataFrame|None=None, pop_row: pd.DataFrame|None=None, *, code: str|int|None=None, region: str|None=None, debug: bool=False):
    with st.container(border=True):
        st.markdown("**진보당 현황**")
        if prg_row is None or prg_row.empty:
            df_all = _load_index_df()
            if df_all is None or df_all.empty:
                st.info("지표 소스(index_sample.csv)를 찾을 수 없습니다."); return
            df_all.columns = [_norm(c) for c in df_all.columns]
            code_col = "code" if "code" in df_all.columns else None
            name_col = "region" if "region" in df_all.columns else None
            prg_row = pd.DataFrame()
            if code is not None and code_col:
                key = _norm(code); prg_row = df_all[df_all[code_col].astype(str).map(_norm)==key].head(1)
            if (prg_row is None or prg_row.empty) and region and name_col:
                key = _norm(region)
                prg_row = df_all[df_all[name_col].astype(str).map(_norm)==key].head(1)
                if prg_row.empty:
                    prg_row = df_all[df_all[name_col].astype(str).str.contains(key, na=False)].head(1)
            if prg_row is None or prg_row.empty:
                prg_row = df_all.head(1)
        else:
            df_all = _load_index_df()

        df = prg_row.copy(); df.columns = [_norm(c) for c in df.columns]; r = df.iloc[0]

        def find_col_exact_or_compact(df, prefer_name, compact_key):
            if prefer_name in df.columns: return prefer_name
            for c in df.columns:
                if compact_key in str(c).replace(" ",""): return c
            return None

        col_strength = find_col_exact_or_compact(df, "진보정당 득표력", "진보정당득표력")
        col_members  = find_col_exact_or_compact(df, "진보당 당원수", "진보당당원수")

        strength = _to_pct_float(r.get(col_strength)) if col_strength else None
        members  = _to_int(r.get(col_members)) if col_members else None

        # --- KPI (kept typography consistent with population box) ---
        html = f"""
        <div style="display:grid; grid-template-columns: 1fr 1fr; align-items:center; gap:12px; margin-top:2px; margin-bottom:0; padding:0 8px;">
            <div style="text-align:center; padding:8px 6px;">
                <div style="color:#6B7280; font-weight:600; margin-bottom:6px;">진보 득표력</div>
                <div style="font-weight:800; color:#111827;">{_fmt_pct(strength) if strength is not None else 'N/A'}</div>
            </div>
            <div style="text-align:center; padding:8px 6px;">
                <div style="color:#6B7280; font-weight:600; margin-bottom:6px;">진보당 당원수</div>
                <div style="font-weight:800; color:#111827;">{(f"{members:,}명" if isinstance(members,int) else "N/A")}</div>
            </div>
        </div>
        """
        from streamlit.components.v1 import html as html_component
        html_component(html, height=140, scrolling=False)

        # --- Mini two-bar (Region vs 10-avg) ---
        try:
            avg_strength = None
            if df_all is not None:
                cols_norm = [_norm(c) for c in df_all.columns]
                if col_strength and col_strength in df.columns:
                    key_cs = col_strength
                else:
                    key_cs = next((c for c in cols_norm if "진보정당득표력" in c), None)
                    if key_cs: key_cs = df_all.columns[cols_norm.index(key_cs)]
                if key_cs and key_cs in df_all.columns:
                    vals = pd.to_numeric(df_all[key_cs].astype(str).str.replace("%","", regex=False), errors="coerce")
                    avg_strength = float(vals.mean()) if vals.notna().any() else None

            if strength is not None and avg_strength is not None:
                bar_df = pd.DataFrame({
                    "항목": ["해당 지역", "10개 평균"],
                    "값": [strength/100.0 if strength>1 else strength,
                          (avg_strength/100.0 if avg_strength>1 else avg_strength)]
                })
                # ✅ Precomputed color column (robust; avoids predicates/expressions)
                bar_df["색상"] = bar_df["항목"].map(lambda x: COLOR_BLUE if x == "해당 지역" else "#9CA3AF")

                mini = (
                    alt.Chart(bar_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("값:Q",
                                axis=alt.Axis(format=".0%"),
                                scale=alt.Scale(domain=[0, float(bar_df["값"].max())*1.1])),
                        y=alt.Y("항목:N", title=None, sort=["해당 지역","10개 평균"]),
                        color=alt.Color("색상:N", scale=None, legend=None),
                        tooltip=[alt.Tooltip("항목:N"), alt.Tooltip("값:Q", format=".1%")]
                    )
                    .properties(height=110, padding={"top":0,"bottom":0,"left":0,"right":0})
                    .configure_view(stroke=None)
                )
                st.altair_chart(mini, use_container_width=True, theme=None)
        except Exception:
            # Keep silent to avoid breaking the whole page
            pass


# =========================================================
# [Region Detail Layout]
# HOW TO CHANGE LATER:
#  - To reduce gaps and enlarge right box only, tweak ratios below.
# (REQ 3) Reduce spacing by changing column ratios; enlarge only the last column.
# =========================================================
def render_region_detail_layout(
    df_pop: pd.DataFrame | None = None,
    df_trend: pd.DataFrame | None = None,
    df_24: pd.DataFrame | None = None,
    df_cur: pd.DataFrame | None = None,
    df_prg: pd.DataFrame | None = None,
):
    # Version tag (optional)
    st.caption(f"charts.py tag: v2025-10-17d | Streamlit {getattr(st, '__version__', 'unknown')} | Altair {getattr(alt, '__version__', 'unknown')}")

    st.markdown("### 👥 인구 정보")
    # ratios tweaked: shrink 1st/2nd a bit, enlarge 3rd; gap small
    col1, col2, col3 = st.columns([1.0, 1.35, 2.85], gap="small")

    with col1.container(border=True, height="stretch"):
        render_population_box(df_pop, box_height_px=240)

    with col2.container(border=True, height="stretch"):
        st.markdown("**연령 구성**")
        render_age_highlight_chart(df_pop, box_height_px=240)

    with col3.container(border=True, height="stretch"):
        st.markdown("**연령별, 성별 인구분포**")
        render_sex_ratio_bar(df_pop, box_height_px=340)

    st.markdown("### 📈 정당성향별 득표추이")
    render_vote_trend_chart(df_trend, box_height_px=420)

    st.markdown("### 🗳️ 선거 결과 및 정치지형")
    c1, c2, c3 = st.columns(3)
    with c1:
        render_results_2024_card(df_24)
    with c2:
        render_incumbent_card(df_cur)
    with c3:
        render_prg_party_box(df_prg, df_pop)





