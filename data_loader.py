# data_loader.py
# Purpose: File reads only. No visualization. Bookmark-first helpers.
# How to change later:
# - To change default encodings: edit ENCODINGS.
# - To change default data directory: pass a different Path from app.py.

from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Union

ENCODINGS = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]

def _read_csv_safe(path: Path, dtype: Optional[Dict[str, Union[str, type]]] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, dtype=dtype)
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    return pd.DataFrame()

def _read_csv_safe_any(paths: List[Path], dtype: Optional[Dict[str, Union[str, type]]] = None) -> pd.DataFrame:
    for p in paths:
        df = _read_csv_safe(p, dtype=dtype)
        if not df.empty:
            return df
    return pd.DataFrame()

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

# -------- Bookmark --------
def load_bookmark(data_dir: Path) -> pd.DataFrame:
    df = _read_csv_safe_any([data_dir / "bookmark.csv", Path("/mnt/data") / "bookmark.csv"])
    return _tidy_columns(df)

def load_bookmark_map(df_bookmark: pd.DataFrame) -> dict:
    """
    Return dict standard_key -> actual_column.
    How to change later:
    - If your bookmark has different schema, edit this parser.
    Supported simple formats:
    1) Two columns: 'std','actual'
    2) Wide: first row contains actual names keyed by std headers
    """
    if df_bookmark is None or df_bookmark.empty:
        return {}

    # Case 1: explicit pairs
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

    # Case 2: wide header row – take first row as values
    # e.g., columns: code, region, sido, total_voters, youth, middle, old, ...
    try:
        first = df_bookmark.iloc[0].to_dict()
        # Only include non-empty strings
        out = {str(k).strip(): str(v).strip() for k, v in first.items() if isinstance(v, str) and v.strip()}
        return out
    except Exception:
        return {}

# -------- Public loaders --------
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
    df = _read_csv_safe_any([data_dir / "index_sample1012.csv", Path("/mnt/data") / "index_sample1012.csv"])
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
