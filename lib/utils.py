"""lib/utils.py — Text normalization and DataFrame cleaning utilities."""

import re
import unicodedata
import pandas as pd


def _norm(s: str) -> str:
    """Normalize a player name for fuzzy matching."""
    if not s:
        return ""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_fangraphs_df(df: pd.DataFrame) -> pd.DataFrame:
    """Strip HTML tags from FanGraphs Name column and normalize player ID."""
    if df.empty:
        return df
    df = df.copy()
    html_re = re.compile(r"<[^>]+>")
    if "Name" in df.columns:
        df["_name"] = df["Name"].astype(str).apply(lambda s: html_re.sub("", s).strip())
        # FG embeds playerid in href — extract it
        def _extract_fg_id(cell: str) -> str:
            m = re.search(r'playerid=(\d+)', str(cell))
            return m.group(1) if m else ""
        df["_fg_id"] = df["Name"].astype(str).apply(_extract_fg_id)
    return df
