# =============================
# File: charts.py
# =============================
from __future__ import annotations
import re
import math
import pandas as pd
import streamlit as st
import altair as alt
from metrics import compute_24_gap


# =============================
# 기본 유틸
# =============================
def _to_pct_float(v, default=None):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    s = str(v).strip().replace(",", "")
    m = re.match(r"^\s*([+-]?\d+(\.\d+)?)\s*%?\s*$", s)
    if not m:
        return default
    x = float(m.group(1))
    if "%" in s:
        return x
    return x * 100.0 if 0 <= x <= 1 else x

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

def _fmt_gap(x):
    return f"{x:.2f}p" if isinstance(x, (int, float)) else "N/A"

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    out.columns = [str(c).strip().replace("\n", "").replace("\r", "") for c in out.columns]
    return out

def _load_index_df() -> pd.DataFrame | None:
    candidate_paths = [
        "sti/data/index_sample.csv",
        "./sti/data/index_sample.csv",
        "data/index_sample.csv",
        "./data/index_sample.csv",
        "index_sample.csv",
        "/mnt/data/index_sample.csv",
        "/mnt/data/index_sample1012.csv",
    ]
    for path in candidate_paths:
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


CARD_HEIGHT = 180


# =============================
# 색상/스타일 공통
# =============================
COLOR_TEXT_DARK = "#111827"
COLOR_TEXT_MID  = "#6B7280"
COLOR_MUTED     = "#334155"
COLOR_BLUE      = "#1E6BFF"
COLOR_GRAY_BAR  = "#E5E7EB"


# =============================
# 24년 총선 결과 칩 색
# =============================
def _party_chip_color(name: str) -> tuple[str, str]:
    s = (name or "").strip()
    MAP = [
        ("더불어민주당", ("#152484", "rgba(21,36,132,0.08)")),
        ("국민의힘",     ("#E61E2B", "rgba(230,30,43,0.10)")),
        ("개혁신당",     ("#798897", "rgba(121,136,151,0.12)")),
    ]
    for key, col in MAP:
        if key in s:
            return col
    return ("#334155", "rgba(51,65,85,0.08)")


# =============================
# 인구 정보 (유동/전체)
# =============================
def render_population_box(pop_df: pd.DataFrame):
    """
    [REQ-1] 첫 컨테이너 중앙 정렬 + '진보당 현황' 스타일 맞춤
    [REQ-2] 유동비율 그래프는 x축 최대 10% 고정
    """
    import numpy as np

    with st.container(border=True):
        if pop_df is None or pop_df.empty:
            st.info("유동인구/연령/성비 차트를 위한 데이터가 없습니다.")
            return

        df = _norm_cols(pop_df.copy())
        code_col  = next((c for c in ["지역구코드","선거구코드","코드","code","CODE"] if c in df.columns), None)
        total_col = next((c for c in ["전체 유권자","전체유권자","total_voters"] if c in df.columns), None)
        float_col = next((c for c in ["유동인구","유권자 이동","floating","mobility"] if c in df.columns), None)

        if not total_col or not float_col:
            st.error("population.csv에서 '전체 유권자' 또는 '유동인구' 컬럼을 찾지 못했습니다.")
            return

        def _to_num(x):
            if pd.isna(x): return np.nan
            if isinstance(x, (int, float)): return float(x)
            s = str(x).strip().replace(",", "")
            try: return float(s)
            except: return np.nan

        df[total_col] = df[total_col].apply(_to_num)
        df[float_col] = df[float_col].apply(_to_num)

        if code_col:
            agg = df.groupby(code_col, dropna=False)[[total_col, float_col]].sum(min_count=1).reset_index(drop=True)
            total_voters = float(agg[total_col].sum())
            floating_pop = float(agg[float_col].sum())
        else:
            total_voters = float(df[total_col].sum())
            floating_pop = float(df[float_col].sum())

        if np.isnan(total_voters) and np.isnan(floating_pop):
            st.info("표시할 합계 수치가 없습니다.")
            return

        total_voters = 0.0 if np.isnan(total_voters) else total_voters
        floating_pop = 0.0 if np.isnan(floating_pop) else floating_pop
        mobility_rate = floating_pop / total_voters if total_voters > 0 else float("nan")

        # 중앙 정렬 + '진보당 현황' KPI 스타일
        st.markdown(
            f"""
            <div style="display:flex; flex-direction:column; align-items:center; gap:8px; text-align:center; padding-top:2px;">
              <div style="color:{COLOR_TEXT_MID}; font-weight:600; font-size:0.95rem;">전체 유권자 수</div>
              <div style="font-weight:800; font-size:1.20rem; color:{COLOR_TEXT_DARK}; letter-spacing:-0.2px;">
                {int(round(total_voters)):,}명
              </div>
              <div style="height:6px;"></div>
              <div style="color:{COLOR_TEXT_MID}; font-weight:600; font-size:0.95rem;">유동인구</div>
              <div style="font-weight:800; font-size:1.20rem; color:{COLOR_TEXT_DARK}; letter-spacing:-0.2px;">
                {int(round(floating_pop)):,}명
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        # 유동비율 바(최대 10% 고정)  [REQ-2]
        if mobility_rate == mobility_rate:
            bar_df = pd.DataFrame({"항목": ["유동비율"], "값": [mobility_rate]})
            x_max = 0.10  # 10% 고정
            chart = (
                alt.Chart(bar_df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "값:Q",
                        axis=alt.Axis(title=None, format=".0%"),
                        scale=alt.Scale(domain=[0, x_max]),
                    ),
                    y=alt.Y("항목:N", axis=alt.Axis(title=None, labels=False, ticks=False)),
                    tooltip=[alt.Tooltip("값:Q", title="유동비율", format=".1%")],
                )
                .properties(height=80)
            )
            text = (
                alt.Chart(bar_df)
                .mark_text(align="left", dx=4)
                .encode(
                    x=alt.X("값:Q", scale=alt.Scale(domain=[0, x_max])),
                    y=alt.Y("항목:N"),
                    text=alt.Text("값:Q", format=".1%"),
                )
            )
            st.altair_chart(chart + text, use_container_width=True)
            st.caption("유동비율 = (전입 + 전출) ÷ 전체 유권자 (동일 기간 기준, 그래프 최대치 10%)")
        else:
            st.info("유동비율을 계산할 수 없습니다.")


# =============================
# 연령 구성 (반원 도넛)
# =============================
def render_age_highlight_chart(pop_df: pd.DataFrame, *, box_height_px: int = 280, width_px: int = 320):
    """
    [REQ-3]
    - 순서 고정: 청년층 → 중년층 → 고령층
    - 검은 테두리 하이라이트 삭제
    - 기본은 옅은 회색, 선택(버튼)한 연령대만 파란색으로 표시
    """
    import numpy as np

    df = _norm_cols(pop_df.copy()) if pop_df is not None else pd.DataFrame()
    if df is None or df.empty:
        st.info("연령 구성 데이터가 없습니다.")
        return

    Y_COL, M_COL, O_COL = "청년층(18~39세)", "중년층(40~59세)", "고령층(65세 이상)"
    TOTAL_CANDIDATES = ["전체 유권자 수", "전체 유권자", "전체유권자", "total_voters"]
    required = [Y_COL, M_COL, O_COL]
    for c in required:
        if c not in df.columns:
            st.error(f"필수 컬럼이 없습니다: {c}")
            return
    total_col = next((c for c in TOTAL_CANDIDATES if c in df.columns), None)
    if total_col is None:
        st.error("'전체 유권자 수' 컬럼을 찾지 못했습니다.")
        return

    for c in [Y_COL, M_COL, O_COL, total_col]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce").fillna(0)

    y, m, o = float(df[Y_COL].sum()), float(df[M_COL].sum()), float(df[O_COL].sum())
    total_v = float(df[total_col].sum())
    if total_v <= 0:
        st.info("전체 유권자 수(분모)가 0입니다.")
        return

    labels = [Y_COL, M_COL, O_COL]  # 순서 고정 [REQ-3]
    values = [y, m, o]
    ratios01 = [v / total_v for v in values]
    ratios100 = [r * 100.0 for r in ratios01]

    # 버튼(라디오)
    focus = st.radio("강조", labels, index=0, horizontal=True, label_visibility="collapsed")

    df_vis = pd.DataFrame({
        "연령": labels,
        "명": values,
        "비율": ratios01,
        "표시비율": ratios100,
    })

    width  = max(300, int(width_px))
    height = max(220, int(box_height_px))
    inner_r, outer_r = 70, 110
    cx = width / 2
    cy = height * 0.65

    base = alt.Chart(df_vis).properties(width=width, height=height)
    theta_enc = alt.Theta("비율:Q", stack=True, scale=alt.Scale(range=[-math.pi/2, math.pi/2]))

    # 기본: 옅은 회색, 선택만 파란색  [REQ-3]
    arcs = (
        base
        .mark_arc(innerRadius=inner_r, outerRadius=outer_r, cornerRadius=6, stroke="white", strokeWidth=1)
        .encode(
            theta=theta_enc,
            color=alt.condition(
                alt.datum.연령 == focus,
                alt.value(COLOR_BLUE),
                alt.value("#E5E7EB"),  # light gray
            ),
            tooltip=[
                alt.Tooltip("연령:N"),
                alt.Tooltip("명:Q", format=",.0f"),
                alt.Tooltip("표시비율:Q", title="비율(%)", format=".1f"),
            ],
        )
    )

    # 중앙 텍스트(선택된 연령만 표기)
    idx = labels.index(focus)
    big_txt_val = f"{df_vis.loc[idx, '표시비율']:.1f}%"
    center_big = (
        alt.Chart(pd.DataFrame({"_":[0]}))
        .mark_text(fontSize=38, fontWeight="bold", color="#0f172a")
        .encode(x=alt.value(cx), y=alt.value(cy - 6), text=alt.value(big_txt_val))
    )
    center_small = (
        alt.Chart(pd.DataFrame({"_":[0]}))
        .mark_text(fontSize=12, color="#475569")
        .encode(x=alt.value(cx), y=alt.value(cy + 16), text=alt.value(focus))
    )

    st.altair_chart(arcs + center_big + center_small, use_container_width=False)


# =============================
# 성비 (연령대×성별 가로 막대)
# =============================
def render_sex_ratio_bar(pop_df: pd.DataFrame, *, box_height_px: int = 320):
    import numpy as np

    if pop_df is None or pop_df.empty:
        st.info("성비 데이터를 표시할 수 없습니다. (population.csv 없음)")
        return

    df = _norm_cols(pop_df.copy())

    age_buckets = ["20대", "30대", "40대", "50대", "60대", "70대 이상"]
    col_pairs = [(f"{a} 남성", f"{a} 여성") for a in age_buckets]
    expect_cols = [c for pair in col_pairs for c in pair]

    missing = [c for c in expect_cols if c not in df.columns]
    if missing:
        st.error("population.csv에 다음 컬럼이 필요합니다: " + ", ".join(missing))
        return

    def _to_num(x):
        if pd.isna(x):
            return 0.0
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace(",", "")
        try:
            return float(s)
        except:
            return 0.0

    df_num = df[expect_cols].applymap(_to_num).fillna(0.0)
    sums = df_num.sum(axis=0)

    if float(sums.sum()) <= 0:
        st.info("성비 데이터(연령×성별)가 모두 0입니다.")
        return

    tidy_rows = []
    for a in age_buckets:
        m_col, f_col = f"{a} 남성", f"{a} 여성"
        m_val, f_val = float(sums[m_col]), float(sums[f_col])
        age_total = m_val + f_val
        if age_total <= 0:
            m_ratio = f_ratio = 0.0
        else:
            m_ratio = m_val / age_total
            f_ratio = f_val / age_total

        tidy_rows.append({"연령대": a, "성별": "남성", "명": m_val, "비율": m_ratio, "연령대총합": age_total})
        tidy_rows.append({"연령대": a, "성별": "여성", "명": f_val, "비율": f_ratio, "연령대총합": age_total})

    tidy = pd.DataFrame(tidy_rows)

    age_label_map = {"20대": "18–29세", "30대": "30대", "40대": "40대", "50대": "50대", "60대": "60대", "70대 이상": "70대 이상"}
    tidy["연령대표시"] = tidy["연령대"].map(age_label_map)

    color_domain = ["남성", "여성"]
    color_range = ["#3B82F6", "#EF4444"]

    n_items = tidy["연령대표시"].nunique()
    per_item_px = 50
    height_px = max(box_height_px, n_items * per_item_px + 40)

    chart = (
        alt.Chart(tidy)
        .mark_bar(size=20)
        .encode(
            y=alt.Y(
                "연령대표시:N",
                sort=[age_label_map[a] for a in age_buckets],
                title=None,
                axis=alt.Axis(labelLimit=160),
            ),
            x=alt.X(
                "비율:Q",
                stack="normalize",
                title="구성비(%)",
                axis=alt.Axis(format=".0%"),
            ),
            color=alt.Color(
                "성별:N",
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("연령대표시:N", title="연령대"),
                alt.Tooltip("성별:N", title="성별"),
                alt.Tooltip("연령대총합:Q", title="해당 연령대 인원", format=",.0f"),
                alt.Tooltip("명:Q", title="성별 인원", format=",.0f"),
                alt.Tooltip("비율:Q", title="해당 연령대 내 비중", format=".1%"),
            ],
        )
        .properties(height=height_px)
    )

    st.altair_chart(chart, use_container_width=True)


# =============================
# 정당성향별 득표추이 (라인)
# =============================
def render_vote_trend_chart(ts: pd.DataFrame):
    """
    [REQ-6] 2022년도 정렬: 2022 대선 → 2022 광역 비례 → 2022 광역단체장
    """
    if ts is None or ts.empty:
        st.info("득표 추이 데이터가 없습니다.")
        return

    df = _norm_cols(ts.copy())

    label_col = next((c for c in ["계열","성향","정당성향","party_label","label"] if c in df.columns), None)
    value_col = next((c for c in ["득표율","비율","share","ratio","pct","prop"] if c in df.columns), None)
    wide_value_cols = [c for c in ["민주","보수","진보","기타"] if c in df.columns]

    prefer_ids = ["선거명","election","선거","분류","연도","year"]
    fallback_ids = ["코드","code"]
    id_col = next((c for c in prefer_ids if c in df.columns), None)
    if id_col is None:
        id_col = next((c for c in fallback_ids if c in df.columns), None)
    year_col = next((c for c in ["연도","year"] if c in df.columns), None)

    if wide_value_cols:
        if not id_col:
            st.warning("선거명을 식별할 컬럼이 필요합니다. (선거명/election/연도/코드)")
            return
        long_df = df.melt(id_vars=id_col, value_vars=wide_value_cols, var_name="계열", value_name="득표율")
        base_elec = long_df[id_col].astype(str)
    else:
        if not (label_col and value_col):
            st.warning("정당 성향(계열)과 득표율 컬럼이 필요합니다.")
            return
        long_df = df.rename(columns={label_col:"계열", value_col:"득표율"}).copy()
        if id_col:
            base_elec = long_df[id_col].astype(str)
        elif year_col:
            base_elec = long_df[year_col].astype(str)
        else:
            st.warning("선거명을 식별할 컬럼이 필요합니다. (선거명/election/연도/코드)")
            return

    def _norm_token(s: str) -> str:
        s = str(s).strip().replace("-", "_").replace(" ", "_").upper()
        s = re.sub(r"_+", "_", s)
        return s

    CODE_RE = re.compile(r"^(20\d{2})(?:_([SG]))?_(NA|LOC|PRESIDENT)(?:_(PRO|GOV))?$")
    KR_REGION_RE = re.compile(r"^(20\d{2})\s+(서울|경기)\s+(.*)$")

    def to_kr_label(raw: str) -> str:
        s = str(raw)
        key = _norm_token(s)
        m = CODE_RE.fullmatch(key)
        if m:
            year, region_tag, lvl, kind = m.group(1), m.group(2), m.group(3), m.group(4)
            region_txt = f" {region_tag} " if region_tag else " "
            if lvl == "PRESIDENT": return f"{year}{region_txt}대선".strip()
            if lvl == "NA" and (kind == "PRO"): return f"{year}{region_txt}총선 비례".strip()
            if lvl == "LOC" and (kind == "PRO"): return f"{year}{region_txt}광역 비례".strip()
            if lvl == "LOC" and (kind == "GOV"): return f"{year}{region_txt}광역단체장".strip()
        km = KR_REGION_RE.match(s)
        if km: return f"{km.group(1)} {km.group(2)} {km.group(3)}"
        if re.match(r"^\s*20\d{2}", s): return s.strip()
        return s

    long_df["선거명_표시"] = base_elec.apply(to_kr_label)
    long_df = long_df.dropna(subset=["선거명_표시","계열","득표율"])

    # 정렬용 컬럼
    long_df["연도"] = long_df["선거명_표시"].str.extract(r"^(20\d{2})").astype(int)
    # 타입(대선/광역 비례/광역단체장) 추출
    def _etype(s: str) -> str:
        if "대선" in s: return "대선"
        if "광역 비례" in s: return "광역 비례"
        if "광역단체장" in s: return "광역단체장"
        if "총선 비례" in s: return "총선 비례"
        return "기타"
    long_df["선거타입"] = long_df["선거명_표시"].map(_etype)

    long_df = long_df.sort_values(["연도","선거타입","선거명_표시","계열"]).drop_duplicates(subset=["선거명_표시","계열","득표율"])

    # 최종 x축 순서 만들기 (2022년만 커스텀)  [REQ-6]
    order_by_year = []
    for y in sorted(long_df["연도"].unique()):
        sub = long_df[long_df["연도"] == y]
        if y == 2022:
            preferred = ["대선", "광역 비례", "광역단체장"]
            for t in preferred:
                labels = sub[sub["선거타입"] == t]["선거명_표시"].unique().tolist()
                order_by_year.extend(labels)
            # 그 외 타입이 있다면 뒤에
            others = sub[~sub["선거타입"].isin(preferred)]["선거명_표시"].unique().tolist()
            order_by_year.extend(others)
        else:
            labels = sub["선거명_표시"].unique().tolist()
            order_by_year.extend(labels)

    party_order = ["민주","보수","진보","기타"]
    color_map = {"민주":"#152484", "보수":"#E61E2B", "진보":"#7B2CBF", "기타":"#6C757D"}
    present = [p for p in party_order if p in long_df["계열"].unique().tolist()]
    colors  = [color_map[p] for p in present]

    selector = alt.selection_point(fields=["선거명_표시","계열"], nearest=True, on="mouseover", empty=False)

    line = (
        alt.Chart(long_df)
        .mark_line(point=False, strokeWidth=3)
        .encode(
            x=alt.X("선거명_표시:N", sort=order_by_year, title="선거명",
                    axis=alt.Axis(labelAngle=-35, labelOverlap=False, labelPadding=6, labelLimit=280)),
            y=alt.Y("득표율:Q", title="득표율(%)"),
            color=alt.Color("계열:N", scale=alt.Scale(domain=present, range=colors),
                            legend=alt.Legend(title=None, orient="top")),
        )
    )

    hit = (
        alt.Chart(long_df)
        .mark_circle(size=600, opacity=0)
        .encode(x="선거명_표시:N", y="득표율:Q", color=alt.Color("계열:N", legend=None))
        .add_params(selector)
    )

    points = (
        alt.Chart(long_df)
        .mark_circle(size=140)
        .encode(
            x="선거명_표시:N",
            y="득표율:Q",
            color=alt.Color("계열:N", scale=alt.Scale(domain=present, range=colors), legend=None),
            opacity=alt.condition(selector, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("선거명_표시:N", title="선거명"),
                alt.Tooltip("계열:N", title="계열"),
                alt.Tooltip("득표율:Q", title="득표율(%)", format=".1f"),
            ],
        )
        .transform_filter(selector)
    )

    chart = (line + hit + points).properties(height=360).interactive()

    with st.container(border=True):
        st.altair_chart(chart, use_container_width=True)


# =============================
# 24년 총선 결과 카드
# =============================
def render_results_2024_card(res_row: pd.DataFrame, df_24: pd.DataFrame = None, code: str = None):
    """
    [REQ-7] 반응형 개선: 작은 화면에서는 세로 스택, 넓은 화면에서는 3열 그리드
    """
    if res_row is None or res_row.empty:
        st.info("해당 선거구의 24년 결과 데이터가 없습니다.")
        return

    res_row = _norm_cols(res_row)

    if "연도" in res_row.columns:
        try:
            cands = res_row.dropna(subset=["연도"]).copy()
            cands["__year__"] = pd.to_numeric(cands["연도"], errors="coerce")
            if (cands["__year__"] == 2024).any():
                r = cands[cands["__year__"] == 2024].iloc[0]
            else:
                r = cands.loc[cands["__year__"].idxmax()]
        except Exception:
            r = res_row.iloc[0]
    else:
        r = res_row.iloc[0]

    name_cols = [c for c in res_row.columns if re.match(r"^후보\d+_이름$", c)]
    def share_col_for(n: str) -> str | None:
        for cand in (f"후보{n}_득표율", f"후보{n}_득표율(%)"):
            if cand in res_row.columns:
                return cand
        return None

    pairs = []
    for nc in name_cols:
        n = re.findall(r"\d+", nc)[0]
        sc = share_col_for(n)
        if not sc: continue
        nm = str(r.get(nc)) if pd.notna(r.get(nc)) else None
        sh = _to_pct_float(r.get(sc))
        if nm and isinstance(sh, (int, float)):
            pairs.append((nm, sh))

    if pairs:
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
        top2 = pairs_sorted[:2] if len(pairs_sorted) >= 2 else [pairs_sorted[0], ("2위", None)]
    else:
        c1n = next((c for c in ["후보1_이름","1위이름","1위 후보"] if c in res_row.columns), None)
        c1v = next((c for c in ["후보1_득표율","1위득표율","1위득표율(%)"] if c in res_row.columns), None)
        c2n = next((c for c in ["후보2_이름","2위이름","2위 후보"] if c in res_row.columns), None)
        c2v = next((c for c in ["후보2_득표율","2위득표율","2위득표율(%)"] if c in res_row.columns), None)
        name1 = str(r.get(c1n)) if c1n else "1위"; share1 = _to_pct_float(r.get(c1v))
        name2 = str(r.get(c2n)) if c2n else "2위"; share2 = _to_pct_float(r.get(c2v))
        top2 = [(name1, share1), (name2, share2)]

    name1, share1 = top2[0][0] or "1위", top2[0][1]
    name2, share2 = (top2[1][0] or "2위", top2[1][1]) if len(top2) > 1 else ("2위", None)
    gap = round(share1 - share2, 2) if isinstance(share1,(int,float)) and isinstance(share2,(int,float)) \
          else (compute_24_gap(df_24, code) if (df_24 is not None and code is not None) else None)

    with st.container(border=True):
        st.markdown("**24년 총선결과**")

        c1_fg, c1_bg = _party_chip_color(name1)
        c2_fg, c2_bg = _party_chip_color(name2)

        def split_name(nm: str):
            parts = (nm or "").strip().split()
            if len(parts) >= 2:
                return parts[0], " ".join(parts[1:])
            return nm, ""

        p1, cand1 = split_name(name1)
        p2, cand2 = split_name(name2)

        # 반응형 CSS  [REQ-7]
        html = f"""
        <style>
        .grid-24 {{
          display:grid; grid-template-columns: repeat(3, 1fr); align-items:center; gap:0; margin-top:6px;
        }}
        @media (max-width: 900px) {{
          .grid-24 {{ grid-template-columns: 1fr; gap:8px; }}
          .cell {{ border-left:none !important; }}
        }}
        .chip {{
          display:inline-flex; flex-direction:column; align-items:center; padding:6px 10px; border-radius:14px;
          font-weight:600; font-size:.95rem; line-height:1.2;
        }}
        .kpi {{
          font-weight:700; font-size:1.05rem; margin-top:8px; font-variant-numeric:tabular-nums; letter-spacing:-0.2px; color:{COLOR_TEXT_DARK};
        }}
        .cell {{ padding:10px 8px; text-align:center; }}
        .divider {{ border-left:1px solid #EEF2F7; }}
        </style>

        <div class="grid-24">
            <div class="cell">
                <div class="chip" style="color:{c1_fg}; background:{c1_bg};">
                    <span style="opacity:0.9;">{p1}</span><span style="color:{COLOR_TEXT_DARK};">{cand1}</span>
                </div>
                <div class="kpi">{_fmt_pct(share1)}</div>
            </div>
            <div class="cell divider">
                <div class="chip" style="color:{c2_fg}; background:{c2_bg};">
                    <span style="opacity:0.9;">{p2}</span><span style="color:{COLOR_TEXT_DARK};">{cand2}</span>
                </div>
                <div class="kpi">{_fmt_pct(share2)}</div>
            </div>
            <div class="cell divider">
                <div style="color:#6B7280; font-weight:600;">1~2위 격차</div>
                <div class="kpi">{_fmt_gap(gap)}</div>
            </div>
        </div>
        """
        from streamlit.components.v1 import html as html_component
        html_component(html, height=CARD_HEIGHT, scrolling=False)


# =============================
# 현직 정보
# =============================
def render_incumbent_card(cur_row: pd.DataFrame):
    """
    [REQ-8]
    - 이름 앞 원형 이니셜 제거
    - 이름/정당 배지는 유지
    - 나머지(선수/성별/연령 + 최근경력) 불릿 텍스트로 표기
    """
    if cur_row is None or cur_row.empty:
        with st.container(border=True):
            st.markdown("**현직정보**")
            st.info("현직 정보 데이터가 없습니다.")
        return

    cur_row = _norm_cols(cur_row)
    r = cur_row.iloc[0]

    name_col   = next((c for c in ["의원명","이름","성명"] if c in cur_row.columns), None)
    party_col  = next((c for c in ["정당","소속정당"] if c in cur_row.columns), None)
    term_col   = next((c for c in ["선수","당선횟수"] if c in cur_row.columns), None)
    age_col    = next((c for c in ["연령","나이"] if c in cur_row.columns), None)
    gender_col = next((c for c in ["성별"] if c in cur_row.columns), None)

    # 최근 경력 후보 컬럼
    career_cols = [c for c in ["최근경력","주요경력","경력","이력","최근 활동"] if c in cur_row.columns]
    recent_career = None
    for c in career_cols:
        v = str(r.get(c))
        if v and v.lower() not in ("nan","none"):
            recent_career = v
            break

    name   = str(r.get(name_col, "정보없음")) if name_col else "정보없음"
    party  = str(r.get(party_col, "정당미상")) if party_col else "정당미상"
    term   = str(r.get(term_col, "N/A")) if term_col else "N/A"
    gender = str(r.get(gender_col, "N/A")) if gender_col else "N/A"
    age    = str(r.get(age_col, "N/A")) if age_col else "N/A"

    try:
        fg, bg = _party_chip_color(party)
    except Exception:
        fg, bg = "#334155", "rgba(51,65,85,0.08)"

    with st.container(border=True):
        st.markdown("**현직정보**")
        html = f"""
        <div style="display:flex; flex-direction:column; gap:10px; margin-top:6px;">
          <div style="display:flex; flex-wrap:wrap; align-items:center; gap:8px;">
            <div style="font-size:1.05rem; font-weight:700; color:{COLOR_TEXT_DARK};">{name}</div>
            <div style="display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:999px;
                        font-weight:600; font-size:.92rem; color:{fg}; background:{bg};">
              {party}
            </div>
          </div>
          <ul style="margin:0; padding-left:1.1rem; color:#374151; font-size:.92rem; line-height:1.5;">
            <li>선수: {term}</li>
            <li>성별: {gender}</li>
            <li>연령: {age}</li>
            {"<li>최근 경력: " + recent_career + "</li>" if recent_career else ""}
          </ul>
        </div>
        """
        from streamlit.components.v1 import html as html_component
        html_component(html, height=CARD_HEIGHT, scrolling=False)


# =============================
# 진보당 현황 (개선 시각화)
# =============================
def render_prg_party_box(prg_row: pd.DataFrame | None, pop_row: pd.DataFrame | None = None, *, code: str | int | None = None, region: str | None = None, debug: bool = False):
    """
    [REQ-9] 대시보드형 요약:
      - KPI 2개: 진보 득표력, 당원수(있을 때)
      - 막대 게이지: 진보 득표력 (0~40% 스케일, 상한 40% 고정으로 상대 감 지각)
      - 보조지표: 당원/유권자(천명당) 표시 (가능할 때)
    """
    def _norm(s: str) -> str:
        s = str(s).replace("\n", " ").replace("\r", " ").strip()
        return " ".join(s.split())

    with st.container(border=True):
        st.markdown("**진보당 현황**")
        st.markdown("<div style='padding-top:4px;'></div>", unsafe_allow_html=True)

        # 안전 로딩
        if prg_row is None or prg_row.empty:
            df_all = _load_index_df()
            if df_all is None or df_all.empty:
                st.info("지표 소스(index_sample.csv)를 찾을 수 없습니다. (sti/data/index_sample.csv 경로 확인)")
                return
            df_all.columns = [_norm(c) for c in df_all.columns]
            code_col = "code" if "code" in df_all.columns else None
            name_col = "region" if "region" in df_all.columns else None

            prg_row = pd.DataFrame()
            if code is not None and code_col:
                key = _norm(code); prg_row = df_all[df_all[code_col].astype(str).map(_norm) == key].head(1)
            if (prg_row is None or prg_row.empty) and region and name_col:
                key = _norm(region)
                prg_row = df_all[df_all[name_col].astype(str).map(_norm) == key].head(1)
                if prg_row.empty:
                    prg_row = df_all[df_all[name_col].astype(str).str.contains(key, na=False)].head(1)
            if prg_row is None or prg_row.empty:
                prg_row = df_all.head(1)

        df = prg_row.copy()
        df.columns = [_norm(c) for c in df.columns]
        r = df.iloc[0]

        # 컬럼 탐색
        col_strength = "진보정당 득표력" if "진보정당 득표력" in df.columns else next((c for c in df.columns if "진보정당득표력" in c.replace(" ", "")), None)
        col_members  = "진보당 당원수"   if "진보당 당원수"   in df.columns else next((c for c in df.columns if "진보당당원수"   in c.replace(" ", "")), None)

        strength = _to_pct_float(r.get(col_strength)) if col_strength else None  # % 단위 값으로 가정
        members  = _to_int(r.get(col_members)) if col_members else None

        # 보조지표: 당원/유권자 (천명당)
        voters_total = None
        if pop_row is not None and not pop_row.empty:
            p = _norm_cols(pop_row.copy())
            tcol = next((c for c in ["전체 유권자","전체유권자","total_voters"] if c in p.columns), None)
            if tcol:
                try:
                    voters_total = int(pd.to_numeric(p[tcol].astype(str).str.replace(",","",regex=False), errors="coerce").sum())
                except Exception:
                    voters_total = None
        per_1000 = None
        if members is not None and voters_total and voters_total > 0:
            per_1000 = round(members / (voters_total/1000), 2)

        # 게이지 막대용 데이터 (0~40%) 고정 스케일
        s_val = (strength if isinstance(strength,(int,float)) else None)
        s01 = (s_val/100.0) if s_val is not None else None
        gauge_df = pd.DataFrame({"항목":["진보 득표력"], "값":[s01 if s01 is not None else 0.0]})

        left_html = f"""
        <div style="display:grid; grid-template-columns: 1fr 1fr; align-items:center; gap:12px; margin-top:2px;">
          <div style="text-align:center; padding:6px 4px;">
            <div style="color:{COLOR_TEXT_MID}; font-weight:600; font-size:0.95rem; margin-bottom:6px;">진보 득표력</div>
            <div style="font-weight:800; font-size:1.20rem; color:{COLOR_TEXT_DARK}; letter-spacing:-0.2px;">
              {_fmt_pct(s_val) if s_val is not None else "N/A"}
            </div>
          </div>
          <div style="text-align:center; padding:6px 4px;">
            <div style="color:{COLOR_TEXT_MID}; font-weight:600; font-size:0.95rem; margin-bottom:6px;">진보당 당원수</div>
            <div style="font-weight:800; font-size:1.20rem; color:{COLOR_TEXT_DARK}; letter-spacing:-0.2px;">
              {(f"{members:,}명" if members is not None else "N/A")}
            </div>
          </div>
        </div>
        """
        from streamlit.components.v1 import html as html_component
        html_component(left_html, height=90, scrolling=False)

        # 게이지 막대 (0~40%)
        if s01 is not None:
            g = (
                alt.Chart(gauge_df)
                .mark_bar()
                .encode(
                    x=alt.X("값:Q", axis=alt.Axis(title=None, format=".0%"), scale=alt.Scale(domain=[0, 0.40])),
                    y=alt.Y("항목:N", axis=alt.Axis(title=None, labels=False, ticks=False)),
                    color=alt.value(COLOR_BLUE),
                    tooltip=[alt.Tooltip("값:Q", title="진보 득표력", format=".1%")],
                )
                .properties(height=70)
            )
            g_text = (
                alt.Chart(gauge_df)
                .mark_text(align="left", dx=4)
                .encode(
                    x=alt.X("값:Q", scale=alt.Scale(domain=[0, 0.40])),
                    y=alt.Y("항목:N"),
                    text=alt.Text("값:Q", format=".1%"),
                )
            )
            st.altair_chart(g + g_text, use_container_width=True)
            st.caption("진보 득표력 게이지 (스케일 0–40%)")
        else:
            st.info("진보 득표력 지표가 없습니다.")

        # 보조: 천명당 당원
        if per_1000 is not None:
            st.markdown(
                f"<div style='color:{COLOR_MUTED}; font-size:.90rem;'>보조지표: 유권자 1,000명당 당원 {per_1000}명</div>",
                unsafe_allow_html=True
            )


# =============================
# 레이아웃
# =============================
def render_region_detail_layout(
    df_pop: pd.DataFrame | None = None,
    df_trend: pd.DataFrame | None = None,
    df_24: pd.DataFrame | None = None,
    df_cur: pd.DataFrame | None = None,
    df_prg: pd.DataFrame | None = None
):
    """
    [REQ-4] 인구 정보 밑(오른쪽 두 박스) 높이 맞춤 & 여백 자연스럽게
    [REQ-5] 연령 구성 컨테이너 더 좁게 (오른쪽 성비 더 넓게)
    """
    st.markdown("### 👥 인구 정보")

    # 바깥 비율: 첫 박스(유동·전체) 좁게, 오른쪽(연령·성비) 넓게
    left_col, right_col = st.columns([1, 5])

    with left_col:
        render_population_box(df_pop)

    with right_col:
        # 오른쪽 내부: 연령은 더 좁게, 성비는 더 넓게  [REQ-5]
        subcol_age, subcol_sex = st.columns([1.2, 2.8])
        with subcol_age.container(border=True):
            st.markdown("**연령 구성**")
            render_age_highlight_chart(df_pop, box_height_px=320, width_px=300)  # 높이 통일 [REQ-4]
        with subcol_sex.container(border=True):
            st.markdown("**연령별, 성별 인구분포**")
            render_sex_ratio_bar(df_pop, box_height_px=320)  # 높이 통일 [REQ-4]

    # 자연스러운 여백  [REQ-4]
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    st.markdown("### 📈 정당성향별 득표추이")
    render_vote_trend_chart(df_trend)

    st.markdown("### 🗳️ 선거 결과 및 정치지형")
    col1, col2, col3 = st.columns(3)
    with col1:
        render_results_2024_card(df_24)
    with col2:
        render_incumbent_card(df_cur)
    with col3:
        render_prg_party_box(df_prg, df_pop)
