"""lib/name_match.py — Player lookup and safe-get utilities."""

from typing import Any, Optional
import pandas as pd
from lib.utils import _norm


def safe_get(row: pd.Series, *col_names, default: Any = None, as_pct: bool = False) -> Any:
    """Return the first non-null value found across col_names in a row."""
    for col in col_names:
        if col in row.index:
            val = row[col]
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                try:
                    result = float(val)
                    if as_pct and result > 1.5:
                        result /= 100.0
                    return result
                except (ValueError, TypeError):
                    return val
    return default


def prepare_lookup_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-process a DataFrame for fast player lookup.
    Adds _norm_name. Converts xMLBAMID/MLBAMID to clean string int.
    Call once before the scoring loop.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    name_col = "_name" if "_name" in df.columns else next(
        (c for c in df.columns if c.lower() in ("name", "playername")), None
    )
    if name_col and "_norm_name" not in df.columns:
        df["_norm_name"] = df[name_col].apply(_norm)
    for _id_col in ("xMLBAMID", "MLBAMID"):
        if _id_col in df.columns:
            df[_id_col] = df[_id_col].apply(
                lambda x: str(int(float(x))) if pd.notna(x) and str(x) not in ("", "nan", "None") else ""
            )
    return df


def find_player_row(df: pd.DataFrame, player_name: str, mlb_id: str = "") -> Optional[pd.Series]:
    """
    Fast player lookup. Priority:
      1. xMLBAMID / MLBAMID exact match
      2. mlbam_id / _mlb_id match
      3. Normalized name match via _norm_name index
    Requires prepare_lookup_df() to have been called first.
    """
    if df is None or df.empty or not player_name:
        return None

    for _id_col in ("xMLBAMID", "MLBAMID"):
        if mlb_id and _id_col in df.columns:
            try:
                m = df[df[_id_col].astype(str).str.split(".").str[0] == str(mlb_id)]
                if not m.empty:
                    return m.iloc[0]
            except Exception:
                pass

    for _id_col in ("mlbam_id", "_mlb_id"):
        if mlb_id and _id_col in df.columns:
            try:
                m = df[df[_id_col].astype(str) == str(mlb_id)]
                if not m.empty:
                    return m.iloc[0]
            except Exception:
                pass

    norm = _norm(player_name)
    parts = norm.split()
    last  = parts[-1] if parts else ""
    first = parts[0]  if len(parts) > 1 else ""

    nc = "_norm_name" if "_norm_name" in df.columns else None
    if nc:
        m = df[df[nc] == norm]
        if not m.empty:
            return m.iloc[0]

        if last:
            cands = df[df[nc].str.contains(last, na=False, regex=False)]
            if not cands.empty:
                if first:
                    refined = cands[cands[nc].str.contains(first[:3], na=False, regex=False)]
                    if not refined.empty:
                        return refined.iloc[0]
                if len(cands) == 1:
                    return cands.iloc[0]

    return None
