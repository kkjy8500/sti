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
    load_bookmark,             # bookmark.csv (optional but preferred)
    load_bookmark_map,         # -> dict, standard_key -> actual column
    load_population_agg,       # population.csv
    load_party_labels,         # party_labels.csv
    load_vote_trend,           # vote_trend.csv
    load_results_2024,         # 5_na_dis_results.csv
    load_current_info,         # current_info.csv
    load_index_sample,         # index_sample.csv
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
DATA_DIR = Path("data") # FIX: Moved up to ensure initialization before use.

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
# Fixed width for the Region Name column (to ensure stable alignment)
REGION_COL_WIDTH = "150px"  

# FIXED HIGHLIGHT: List of regions for permanent row highlighting in the '결과 요약' table.
FIXED_HIGHLIGHT_REGIONS = ["서울 서대문구갑", "경기 평택시을", "경기 화성시을"]  

# FIXED HIGHLIGHT: Background color for the entire row of the fixed highlight regions (Summary Table only)
FIXED_HIGHLIGHT_ROW_BG = "#FFF9C4" # Light yellow background color for fixed highlighted rows

# DYNAMIC HIGHLIGHT: Background color for the bar container when a score is in the Top 3  
DYNAMIC_HIGHLIGHT_CELL_BG = "#E0F2FE" # Light sky blue for dynamic Top 3 scores

# Colors for the main scoring bars (Used in the '종합' tab - '결과 요약')
BAR_COLORS_MAIN = {
    "합계": "#3498DB",           # Total Score Bar Color (Blue)
    "유권자환경": "#48C9B0", # Electorate Environment Bar Color (Light Cyan)
    "정치지형": "#1ABC9C",     # Political Landscape Bar Color (Green)
    "주체역량": "#76D7C4",     # Subjective Capacity Bar Color (Very Light Green)
    "상대역량": "#2ECC71",     # Opponent Capacity Bar Color (Emerald Green)
}
# ====================================================================


# ===== Utility Functions (Kept concise and unchanged, added comments where necessary) =====
@st.cache_data(show_spinner=False)
def _read_scoring_cached(path_str: str) -> pd.DataFrame:
    """Reads scoring CSV/TSV data with caching."""
    p = Path(path_str)
    if not p.exists():
        return pd.DataFrame()
    try:
        # Tries CSV first, then TSV fallback
        df = pd.read_csv(p, encoding="utf-8-sig")
        if df.shape[1] == 1:
            df = pd.read_csv(p, encoding="utf-8-sig", sep="\t")
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def _read_markdown_cached(path_str: str) -> str | None:
    """Reads Markdown file content with caching."""
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

# ====================================================================
# NEW FUNCTION: Load Index Description from data/index.csv
# ====================================================================
@st.cache_data(show_spinner=False)
def load_index_descriptions(data_dir: Path) -> pd.DataFrame:
    """
    Loads data/index.csv (expected header: 지표명, 설명, 상관관계, 가중치) 
    and returns a normalized DataFrame.
    """
    path = data_dir / "index.csv"
    if not path.exists():
        return pd.DataFrame(columns=["지표명", "설명"])
    
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = _normalize_columns(df)
        
        # Ensure the required columns exist
        if "지표명" in df.columns and "설명" in df.columns:
            return df.loc[:, ["지표명", "설명"]].dropna(subset=["지표명", "설명"]).copy()
        else:
            # Handle case where column names are different but predictable
            if df.columns.shape[0] >= 2:
                df.rename(columns={df.columns[0]: "지표명", df.columns[1]: "설명"}, inplace=True)
                return df.loc[:, ["지표명", "설명"]].dropna(subset=["지표명", "설명"]).copy()
                
            return pd.DataFrame(columns=["지표명", "설명"])
    except Exception:
        return pd.DataFrame(columns=["지표명", "설명"])
# ====================================================================
# Rest of Utility Functions
# ====================================================================
CODE_CANDIDATES = ["코드", "지역구코드", "선거구코드", "지역코드", "code", "CODE"]
NAME_CANDIDATES = ["지역구", "선거구", "선거구명", "지역명", "district", "지역구명", "region", "지역"]
SIDO_CANDIDATES = ["시/도", "시도", "광역", "sido", "province"]

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Removes leading/trailing whitespace and newline characters from column names."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c).strip().replace("\n", "").replace("\r", "") for c in out.columns]
    return out

def _detect_col(df: pd.DataFrame, candidates: list) -> str | None:
    """Finds the first matching column name from a list of candidates."""
    cols = [str(c).strip().replace("\n", "").replace("\r", "") for c in df.columns]
    for cand in candidates:
        if cand in cols:
            return df.columns[cols.index(cand)]
    return None

def _canon_code(x: object) -> str:
    """Standardizes code format for robust lookup (strip non-alphanumeric, remove leading zeros)."""
    s = str(x).strip()
    s = re.sub(r"[^0-9A-Za-z]", "", s)
    s = s.lstrip("0")
    return s.lower()

def ensure_code_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures a '코드' (Code) column exists, standardizing various code column names."""
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
    """Filters a DataFrame by the standardized code."""
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
    """Builds a unique list of regions (Code, Label) for the sidebar selectbox."""
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
            # Prepends Sido name if it's not already part of the Region name
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
# Core Logic for Table Rendering
# --------------------------------

# Helper to format numeric values (Count vs. Ratio/Score)
def _format_value(val: float | object, col_name: str) -> str:
    """
    Formats the value: comma for counts (and specific integer metrics like 유동성), two decimals for scores/ratios.
    """
    try:
        v = float(val)
        if np.isnan(v):
            return ""
    except Exception:
        return str(val)

    # Heuristic for Count-like data (should use comma)
    # MODIFICATION 3: Added "유동성A", "유동성B" to count_names for integer formatting.
    # Also includes "진보당 지방선거 후보 수" for integer formatting.
    count_names = ["유권자 수", "유동인구", "진보당 당원수", "진보당 지방선거 후보 수", "유동성A", "유동성B"]
    if col_name in count_names:
        # Rounds to nearest int and applies comma format
        # Using int(round(v)) to handle float values that should be treated as integers
        return f"{int(round(v)):,d}" 
    
    # Default for Scores/Ratios
    return f"{v:.2f}" # Two decimal places

def _bar_cell_factory(score_df: pd.DataFrame, score_cols: list[str], bar_colors: dict) -> callable:
    """
    Generates an HTML bar cell function for the main Summary Table ('결과 요약').
    Includes: Absolute/Dynamic Scaling, Top 3 Highlighting, and value formatting.
    """
    
    # 1. Pre-calculate top 3 values for dynamic highlighting
    top3_values = {}
    for col in score_cols:
        try:
            top3_values[col] = set(score_df.nlargest(3, col, keep='all')[col].tolist())
        except KeyError:
            top3_values[col] = set()
    
    # 2. Pre-calculate dynamic maxes for scaling unlisted columns
    dynamic_maxes = {
        col: score_df[col].max() for col in score_cols 
        if col not in ABSOLUTE_MAX_SCORES
    }
    
    def _bar_cell(val, col):
        """Renders a single cell with an HTML bar."""
        try:
            v = float(val)
        except Exception:
            # Non-numeric or missing data handling
            return f"<span style='font-size:12px;font-weight:600;'>{val}</span>"

        if np.isnan(v):
             return ""
        
        # Determine the maximum score for scaling (ABSOLUTE_MAX_SCORES takes precedence)
        max_score = ABSOLUTE_MAX_SCORES.get(col)
        
        if max_score is None:
            max_score = dynamic_maxes.get(col, 1.0)
        
        # Ensure max_score is at least 1.0
        max_score = max(1.0, max_score) 

        # Calculate bar percentage
        pct = max(0.0, min(100.0, (v / max_score) * 100.0))
        
        # Check for Top 3 dynamic highlight
        is_top3 = col in top3_values and v in top3_values[col]
        
        # Get bar color
        color = bar_colors.get(col, "#6B7280") 
        
        # Set background for the bar container (DYNAMIC_HIGHLIGHT_CELL_BG for Top 3)
        container_bg = DYNAMIC_HIGHLIGHT_CELL_BG if is_top3 else "#F3F4F6" # #F3F4F6: light gray background for score bar container
        
        # Format the number displayed on the bar
        formatted_value = _format_value(v, col)

        # HTML Structure for the bar
        return (
            # Outer Div: Handles padding for the cell
            f'<div style="padding:6px 8px; height:100%; box-sizing:border-box;">' # 6px vertical, 8px horizontal padding for cell content
            # Inner Container Div: Holds the bar and background color for Top 3 highlight
            f'<div style="position:relative;width:100%;background:{container_bg};height:18px;border-radius:4px;overflow:hidden;min-width:50px; transition: background-color 0.2s ease-in-out;">'
            # Bar Div: The colored bar that represents the score
            f'  <div style="width:{pct:.2f}%;height:100%;background:{color}; border-radius:4px 0 0 4px;"></div>' 
            # Value Text Overlay Div: Displays the formatted score on top of the bar
            f'  <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;'
            f'font-size:12px;font-weight:600;color:#111827; text-shadow: 0 0 1px #fff;">{formatted_value}</div>' 
            f'</div>'
            f'</div>'
        )
    return _bar_cell

def _text_only_cell(val: float | object, col_name: str) -> str:
    """
    Renders a single cell with only the formatted score (NO BARS, NO HIGHLIGHTS).
    Used for the lower detailed index table ('세부 지표별 상세 분석').
    """
    
    formatted_value = _format_value(val, col_name)
    
    # Text-only styling, centered and using standard padding
    # Styling note: This is a plain data cell.
    return (
        f'<div style="text-align:center; padding: 6px 8px; font-size:13px; font-weight:600; color:#1F2937;">'
        f'{formatted_value}'
        f'</div>'
    )
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
    # Load all required dataframes
    df_bookmark = load_bookmark(DATA_DIR)
    bookmark_map = load_bookmark_map(df_bookmark)

    df_pop   = ensure_code_col(load_population_agg(DATA_DIR))
    df_party = ensure_code_col(load_party_labels(DATA_DIR))
    df_trend = ensure_code_col(load_vote_trend(DATA_DIR))
    df_24    = ensure_code_col(load_results_2024(DATA_DIR))
    df_curr  = ensure_code_col(load_current_info(DATA_DIR))
    df_idx   = ensure_code_col(load_index_sample(DATA_DIR))
    
    # NEW: Load index description data
    df_desc = load_index_descriptions(DATA_DIR)

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
    # Assumes the first column is the region name for display
    df.rename(columns={df.columns[0]: "지역"}, inplace=True) 
    label_col = "지역"

    score_cols = [c for c in df.columns if c != label_col]
    df[score_cols] = df[score_cols].apply(pd.to_numeric, errors="coerce")
    
    # Instantiate the bar cell factory for the main table (using BAR_COLORS_MAIN)
    _bar_cell = _bar_cell_factory(df, score_cols, BAR_COLORS_MAIN)

    st.subheader("결과 요약")

    # --- Build HTML table for main scoring ('결과 요약') ---
    headers = [label_col] + score_cols
    
    # Region Column Header (Uses fixed width, left-aligned)
    thead = (
        f"<th style='text-align:left;padding:6px 8px;white-space:nowrap;width:{REGION_COL_WIDTH};'>"
        f"지역</th>" 
    )
    # Remaining headers (Calculates equal width based on column count, CENTER-ALIGNED)
    # MODIFICATION 1: Changed text-align to 'center' for score headers
    remaining_cols_count = len(score_cols)
    col_width_pct = f"{100 / remaining_cols_count}%" if remaining_cols_count > 0 else "auto"
    
    thead += "".join(
        [
            f"<th style='text-align:center;padding:6px 8px;white-space:nowrap;width:{col_width_pct};'>{h}</th>" 
            for h in score_cols
        ]
    )

    rows_html = []
    for _, row in df.iterrows():
        # LOGIC: Check for FIXED_HIGHLIGHT_REGIONS using the actual region name in the data
        is_fixed_highlight = row[label_col] in FIXED_HIGHLIGHT_REGIONS
        # CSS: Apply row background color if highlighted
        row_style = f"background-color:{FIXED_HIGHLIGHT_ROW_BG};" if is_fixed_highlight else "" # FIXED_HIGHLIGHT_ROW_BG: light yellow row highlight color

        # Region Column Cell (Fixed width)
        cells = [
            f"<td style='padding:6px 8px;white-space:nowrap;width:{REGION_COL_WIDTH};'>" # Standard padding for region column
            # CSS: Apply bold font-weight (700) to the region name if the row is highlighted
            f"<span style='font-size:13px; font-weight:{'700' if is_fixed_highlight else '600'};'>{row[label_col]}</span>"
            f"</td>"
        ]
        
        for c in score_cols:
            # Bar Cell (Bar logic handles dynamic highlight/scaling)
            cells.append(f"<td style='padding:0px;width:{col_width_pct};'>{_bar_cell(row[c], c)}</td>") # 0px padding as the bar cell factory handles inner padding
        
        # Stitch all cells together for the row
        rows_html.append(f"<tr style='{row_style}'>" + "".join(cells) + "</tr>")

    table_html = (
        "<div style='overflow-x:auto;'>"
        # CSS: table-layout auto (default) or fixed is fine, separate border collapse
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

    # Indicator Groups Definition
    INDICATOR_GROUPS = {
        "유권자환경": ["유권자 수", "유동인구", "고령층 비율", "청년층 비율", "4-50대 비율", "2030여성 비율"],
        "정치지형": ["유동성A", "경합도A", "유동성B", "경합도B"],
        "주체역량": ["진보정당 득표력", "진보당 당원수", "진보당 지방선거 후보 수"],
        "상대역량": ["현직 득표력", "민주당 득표력", "보수 득표력"],
    }
    
    # --------------------------------------------------------------------
    # NEW LOGIC START: Index Descriptions Tab
    # --------------------------------------------------------------------
    st.markdown("##### 🔍 지표 설명") # Sub-heading for the description section
    
    if df_desc.empty:
        st.info("`data/index.csv`에서 지표 설명을 불러오지 못했습니다. 파일 경로 및 형식을 확인해 주세요.")
    else:
        # Create a dictionary for easy lookup: Indicator Name -> Description
        desc_map = df_desc.set_index("지표명")["설명"].to_dict()

        tab_titles = list(INDICATOR_GROUPS.keys())
        tabs = st.tabs(tab_titles)
        
        # Render content for each description tab
        for selected_group, tab in zip(tab_titles, tabs):
            with tab:
                target_cols = INDICATOR_GROUPS.get(selected_group, [])
                markdown_list = []
                
                for indicator in target_cols:
                    description = desc_map.get(indicator, f"'{indicator}' 지표에 대한 설명이 없습니다.")
                    # Format as a bullet point list
                    markdown_list.append(f"**- {indicator}:** {description}")
                    
                st.markdown("\n".join(markdown_list))

    st.divider() # New divider to separate Description from Detailed Table
    st.markdown("##### 📈 지표별 데이터 (Text-Only Table)") # Sub-heading for the detailed data table
    
    # --------------------------------------------------------------------
    # EXISTING LOGIC: Detailed Index Table (Numerical Data)
    # --------------------------------------------------------------------
    tab_titles_table = list(INDICATOR_GROUPS.keys())
    tabs_table = st.tabs([t + " 데이터" for t in tab_titles_table]) # Changed tab names for the table view
    
    # Render content for each data table tab
    for selected_group, tab in zip(tab_titles_table, tabs_table):
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
                # Filter to only columns that actually exist in the dataframe
                present_cols = [c for c in target_cols if c in df_display.columns]
                
                if not present_cols:
                    # NOTE: This message will appear if a column is missing from index_sample.csv
                    st.info(f"선택된 그룹 ({selected_group})에 해당하는 컬럼이 데이터에 없습니다.")
                else:
                    df_final = df_display.loc[:, [label_col_new, "코드"] + present_cols].copy()
                    
                    df_final[present_cols] = df_final[present_cols].apply(pd.to_numeric, errors="coerce")
                    df_final = df_final.dropna(subset=[label_col_new]).dropna(subset=present_cols, how='all')
                    
                    df_final = df_final.reset_index(drop=True)
                    
                    # Detailed Analysis HTML Table Generation (Text-Only)
                    headers_new = [label_col_new] + present_cols
                    
                    # Region Column Header (Fixed width, left-aligned)
                    thead_new = (
                        f"<th style='text-align:left;padding:6px 8px;white-space:nowrap;font-weight:700;width:{REGION_COL_WIDTH};'>"
                        f"지역</th>" 
                    )
                    # Remaining headers (Equal width, CENTER-ALIGNED)
                    remaining_cols_count_new = len(present_cols)
                    col_width_pct_new = f"{100 / remaining_cols_count_new}%" if remaining_cols_count_new > 0 else "auto"

                    thead_new += "".join(
                        [
                            f"<th style='text-align:center;padding:6px 8px;white-space:nowrap;width:{col_width_pct_new};'>{h}</th>" 
                            for h in present_cols
                        ]
                    )

                    rows_html_new = []
                    for _, row in df_final.iterrows():
                        # LOGIC: Fixed row highlight is explicitly REMOVED for this detailed table.
                        row_style = "" 
                        
                        # Region Column Cell (Fixed width) - Standard font weight (600)
                        cells = [
                            f"<td style='padding:6px 8px;white-space:nowrap;width:{REGION_COL_WIDTH};'>"
                            f"<span style='font-size:13px; font-weight:600;'>{row[label_col_new]}</span>"
                            f"</td>"
                        ]
                        for c in present_cols:
                            # Text-only Cell (Uses _text_only_cell function for integer formatting)
                            cells.append(f"<td style='padding:0;width:{col_width_pct_new};'>{_text_only_cell(row[c], c)}</td>") 
                        
                        # Stitch all cells together for the row
                        rows_html_new.append(f"<tr style='{row_style}'>" + "".join(cells) + "</tr>")

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

    # Prepare data subset for the selected region
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
    # Layout for Title and App name
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

    st.divider()

    # Logic to find and render external Markdown file
    md_candidates = [
        Path("sti") / "지표별 구성 및 해설.md",
        Path("지표별 구성 및 해설.md"),
        Path("/mnt/data/sti/지표별 구성 및 해설.md"),
    ]
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr"]

    md_text = None
    for p in md_candidates:
        if p.exists():
            for enc in encodings:
                try:
                    md_text = p.read_text(encoding=enc)
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
