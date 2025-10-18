# metrics.py
# Purpose: Only what's used now (compute_24_gap) – trimmed per request.
# How to change later:
# - If you need more metrics, add new functions here.

from __future__ import annotations
import re
import pandas as pd

_CODE_CANDIDATES = ["코드", "지역구코드", "선거구코드", "지역코드", "code", "CODE"]

def _canon_code(x: object) -> str:
    s = str(x).strip()
    s = re.sub(r"[^0-9A-Za-z]", "", s)
    s = s.lstrip("0")
    return s.lower()

def _pct_float(v) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().replace(",", "")
    m = re.match(r"^\s*([+-]?\d+(\.\d+)?)\s*%?\s*$", s)
    if not m:
        return None
    x = float(m.group(1))
    if "%" in s:
        return x
    return x * 100.0 if 0 <= x <= 1 else x

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df
    df2 = df.copy()
    df2.columns = [str(c).strip().replace("\n", "").replace("\r", "") for c in df2.columns]
    return df2

def _detect_code_col(df: pd.DataFrame) -> str | None:
    for c in _CODE_CANDIDATES:
        if c in df.columns:
            return c
    cols = [str(c).strip().replace("\n", "").replace("\r", "") for c in df.columns]
    for cand in _CODE_CANDIDATES:
        if cand in cols:
            return df.columns[cols.index(cand)]
    return None

def _get_by_code_local(df: pd.DataFrame, code: str) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df2 = _normalize_columns(df)
    col = "코드" if "코드" in df2.columns else _detect_code_col(df2)
    if not col:
        return pd.DataFrame()
    key = _canon_code(code)
    try:
        sub = df2[df2[col].astype(str).map(_canon_code) == key]
        return sub if len(sub) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def compute_24_gap(df_24: pd.DataFrame, code: str) -> float | None:
    """
    Return (1st - 2nd) share gap in percentage points for 2024 (or latest year).
    """
    try:
        sub = _get_by_code_local(df_24, code)
        if sub.empty:
            return None

        if "연도" in sub.columns:
            tmp = sub.dropna(subset=["연도"]).copy()
            tmp["__year__"] = pd.to_numeric(tmp["연도"], errors="coerce")
            if (tmp["__year__"] == 2024).any():
                row = tmp[tmp["__year__"] == 2024].iloc[0]
            else:
                row = tmp.loc[tmp["__year__"].idxmax()]
        else:
            row = sub.iloc[0]

        c1v = next((c for c in ["후보1_득표율","1위득표율","1위 득표율","1st_share","득표율_1위","1위득표율(%)"] if c in sub.columns), None)
        c2v = next((c for c in ["후보2_득표율","2위득표율","2위 득표율","2nd_share","득표율_2위","2위득표율(%)"] if c in sub.columns), None)

        if not (c1v and c2v):
            return None

        v1 = _pct_float(row[c1v]); v2 = _pct_float(row[c2v])
        if v1 is None or v2 is None:
            return None
        return round(v1 - v2, 2)
    except Exception:
        return None
