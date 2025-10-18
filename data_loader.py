# data_loader.py (drop-in replace for the whole file)
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Union
import streamlit as st

# How to change later:
# - Add/remove encodings below if your files use a new encoding.
ENCODINGS = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]

# ---------- Cached readers (avoid re-opening files each run) ----------
@st.cache_data(show_spinner=False)
def _read_csv_impl(path_str: str, mtime_ns: int, dtype: Optional[Dict[str, Union[str, type]]]) -> pd.DataFrame:
    """Low-level cached reader keyed by (path, mtime)."""
    p = Path(path_str)
    if not p.exists():
        return pd.DataFrame()
    for enc in ENCODINGS:
        try:
            return pd.read_csv(p, encoding=enc, dtype=dtype)
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    return pd.DataFrame()

def _read_csv_safe(path: Path, dtype: Optional[Dict[str, Union[str, type]]] = None) -> pd.DataFrame:
    """One open per path; subsequent runs served from cache until file changes."""
    if not path.exists():
        return pd.DataFrame()
    mtime = path.stat().st_mtime_ns
    return _read_csv_impl(str(path), mtime, dtype)

def _read_csv_safe_any(paths: List[Path], dtype: Optional[Dict[str, Union[str, type]]] = None) -> pd.DataFrame:
    """Try candidates; first successful read is cached by its own (path, mtime)."""
    for p in paths:
        df = _read_csv_safe(p, dtype=dtype)
        if not df.empty:
            return df
    return pd.DataFrame()

# ---------- Light post-processing helpers ----------
def _tidy_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cols, seen = [], {}
    for c in df.columns:
        base = str(c).strip()
        if base not in seen:
            seen[base] = 0; cols.append(base)
        else:
            seen[base] += 1; cols.append(f"{base}.{seen[base]}")
    df.columns = cols
    return df

def _ensure_str(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

# ---------- Bookmark ----------
def load_bookmark(data_dir: Path) -> pd.DataFrame:
    df = _read_csv_safe_any([data_dir / "bookmark.csv", Path("/mnt/data") / "bookmark.csv"])
    return _tidy_columns(df)

def load_bookmark_map(df_bookmark: pd.DataFrame) -> dict:
    """Return dict standard_key -> actual_column. (Two formats supported)"""
    if df_bookmark is None or df_bookmark.empty:
        return {}
    lower_cols = [str(c).strip().lower() for c in df_bookmark.columns]
    if "std" in lower_cols and "actual" in lower_cols:
        std_idx = lower_cols.index("std"); act_idx = lower_cols.index("actual")
        out = {}
        for _, row in df_bookmark.iterrows():
            std = str(row.iloc[std_idx]).strip()
            act = str(row.iloc[act_idx]).strip()
            if std and act:
                out[std] = act
        return out
    try:
        first = df_bookmark.iloc[0].to_dict()
        return {str(k).strip(): str(v).strip() for k, v in first.items() if isinstance(v, str) and v.strip()}
    except Exception:
        return {}

# ---------- Public loaders ----------
def load_population_agg(data_dir: Path) -> pd.DataFrame:
    df = _read_csv_safe_any([data_dir / "population.csv", Path("/mnt/data") / "population.csv"])
    df = _tidy_columns(df)
    return _ensure_str(df, ["코드", "지역구", "구", "동"])

def load_party_labels(data_dir: Path) -> pd.DataFrame:
    df = _read_csv_safe_any([data_dir / "party_labels.csv", Path("/mnt/data") / "party_labels.csv"])
    df = _tidy_columns(df)
    return _ensure_str(df, ["정당코드", "정당", "party", "code"])

def load_vote_trend(data_dir: Path) -> pd.DataFrame:
    df = _read_csv_safe_any([data_dir / "vote_trend.csv", Path("/mnt/data") / "vote_trend.csv"])
    df = _tidy_columns(df)
    return _ensure_str(df, ["코드", "선거구명", "지역구", "district", "label"])

def load_results_2024(data_dir: Path) -> pd.DataFrame:
    df = _read_csv_safe_any([data_dir / "5_na_dis_results.csv", Path("/mnt/data") / "5_na_dis_results.csv"])
    df = _tidy_columns(df)
    return _ensure_str(df, ["코드", "선거구명", "지역구", "구", "동"])

def load_current_info(data_dir: Path) -> pd.DataFrame:
    df = _read_csv_safe_any([data_dir / "current_info.csv", Path("/mnt/data") / "current_info.csv"])
    df = _tidy_columns(df)
    return _ensure_str(df, ["코드", "선거구명", "지역구", "이름", "정당"])

def load_index_sample(data_dir: Path) -> pd.DataFrame:
    df = _read_csv_safe_any([data_dir / "index_sample.csv", Path("/mnt/data") / "index_sample.csv"])
    df = _tidy_columns(df)
    return _ensure_str(df, ["코드", "선거구명", "지역구"])

def load_all(data_dir: Union[str, Path]) -> dict:
    data_dir = Path(data_dir)
    return {
        "bookmark": load_bookmark(data_dir),
        "population": load_population_agg(data_dir),
        "party_labels": load_party_labels(data_dir),
        "vote_trend": load_vote_trend(data_dir),
        "results_2024": load_results_2024(data_dir),
        "current_info": load_current_info(data_dir),
        "index_sample": load_index_sample(data_dir),
    }
