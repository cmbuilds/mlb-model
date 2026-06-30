"""scoring/pitcher.py — Pitcher vulnerability sub-score and bullpen scores."""

from typing import Dict, Tuple
import pandas as pd


def compute_pitcher_score(statcast: Dict, fg_stats: Dict = None,
                          bullpen_vuln: float = 42.0) -> Tuple[float, str]:
    """
    Pitcher VULNERABILITY score 0–100.
    High = pitcher is hittable (good for batter TB). Low = ace.
    Blends SP stats (60%) with per-team bullpen vulnerability (40%).
    """
    def f(key, default):
        try: return float(statcast.get(key, default))
        except Exception: return float(default)

    k_rate   = f("k_rate_allowed",   0.220)
    hard_hit = f("hard_hit_allowed", 0.370)
    barrel   = f("barrel_allowed",   0.070)
    era      = f("era",              4.20)
    fip      = f("fip",              4.20)
    whip     = f("whip",             1.30)

    k_vuln      = max(0, min(100, 50 - (k_rate   - 0.220) / 0.050 * 25))
    hh_vuln     = max(0, min(100, 50 + (hard_hit - 0.370) / 0.055 * 25))
    barrel_vuln = max(0, min(100, 50 + (barrel   - 0.070) / 0.030 * 25))
    era_use     = fip if fip > 0 else era
    era_vuln    = max(0, min(100, 50 + (era_use  - 4.20)  / 0.80  * 25))
    whip_vuln   = max(0, min(100, 50 + (whip     - 1.30)  / 0.20  * 25))

    LEAGUE_AVG_HH     = 0.360
    LEAGUE_AVG_BARREL = 0.070
    HH_IS_DEFAULT     = abs(hard_hit - LEAGUE_AVG_HH)     < 0.003
    BARREL_IS_DEFAULT = abs(barrel   - LEAGUE_AVG_BARREL)  < 0.003

    if HH_IS_DEFAULT and BARREL_IS_DEFAULT:
        sp_score = k_vuln * 0.50 + era_vuln * 0.35 + whip_vuln * 0.15
    else:
        sp_score = (
            k_vuln      * 0.40 +
            hh_vuln     * 0.28 +
            era_vuln    * 0.18 +
            barrel_vuln * 0.09 +
            whip_vuln   * 0.05
        )

    blended  = sp_score * 0.60 + float(bullpen_vuln) * 0.40
    mode_tag = "FIP-only" if (HH_IS_DEFAULT and BARREL_IS_DEFAULT) else "full"
    label    = f"K%: {k_rate*100:.0f}% | WHIP: {whip:.2f} | FIP: {era_use:.2f} | BP vuln: {bullpen_vuln:.0f} | mode: {mode_tag}"
    return max(0, min(100, blended)), label


def compute_team_bullpen_scores(pitching_df: pd.DataFrame) -> Dict[str, float]:
    """
    Build per-team bullpen vulnerability 0–100 from the loaded pitching DataFrame.
    Filters to relievers (GS==0 or GS/G < 30%), groups by team.
    Returns dict keyed by UPPERCASE team abbreviation. Missing teams → 42.0 at scoring time.
    """
    if pitching_df is None or pitching_df.empty:
        return {}

    df = pitching_df.copy()
    has_gs = "GS" in df.columns
    has_g  = "G"  in df.columns

    if has_gs and has_g:
        df["_GS"] = pd.to_numeric(df["GS"], errors="coerce").fillna(0)
        df["_G"]  = pd.to_numeric(df["G"],  errors="coerce").fillna(0)
        df["_IP"] = pd.to_numeric(df.get("IP", pd.Series(dtype=float)), errors="coerce").fillna(0)
        relievers = df[
            (df["_GS"] == 0) |
            ((df["_G"] > 0) & (df["_GS"] / df["_G"].replace(0, 1) < 0.30))
        ].copy()
        if not relievers.empty and len(relievers) / max(1, len(df)) > 0.70:
            df["_ipg"] = df["_IP"] / df["_G"].replace(0, 1)
            relievers = df[df["_ipg"] < 2.0].copy()
    elif has_gs:
        df["_GS"] = pd.to_numeric(df["GS"], errors="coerce").fillna(0)
        relievers = df[df["_GS"] == 0].copy()
    else:
        if "IP" in df.columns and "G" in df.columns:
            df["_IP"] = pd.to_numeric(df["IP"], errors="coerce").fillna(0)
            df["_G"]  = pd.to_numeric(df["G"],  errors="coerce").fillna(0)
            df["_ipg"] = df["_IP"] / df["_G"].replace(0, 1)
            relievers = df[df["_ipg"] < 2.0].copy()
        else:
            relievers = df.copy()

    if relievers.empty:
        relievers = df.copy()
    if relievers.empty:
        return {}

    team_col = next((c for c in ["Team", "team", "Tm", "tm", "TEAM"] if c in relievers.columns), None)
    if team_col is None:
        return {}

    def to_rate(series: pd.Series, thresh: float = 1.0) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        if s.dropna().median() > thresh:
            s = s / 100.0
        return s

    relievers = relievers.copy()
    relievers["_k_rate"] = to_rate(relievers["K%"],    1.0).values if "K%"    in relievers.columns else float("nan")
    relievers["_fip"]    = (pd.to_numeric(relievers["FIP"],  errors="coerce") if "FIP"  in relievers.columns else
                            pd.to_numeric(relievers["xFIP"], errors="coerce") if "xFIP" in relievers.columns else
                            pd.Series(dtype=float, index=relievers.index)).values
    relievers["_whip"]   = pd.to_numeric(relievers["WHIP"],  errors="coerce").values if "WHIP"  in relievers.columns else float("nan")
    relievers["_hh"]     = to_rate(relievers["Hard%"], 1.0).values if "Hard%" in relievers.columns else float("nan")

    try:
        group = relievers.groupby(team_col).agg(
            k_rate=("_k_rate", "median"),
            fip   =("_fip",    "median"),
            whip  =("_whip",   "median"),
            hh    =("_hh",     "median"),
        ).reset_index()
    except Exception:
        return {}

    team_scores: Dict[str, float] = {}
    for _, row in group.iterrows():
        team = str(row[team_col]).strip().upper()
        if not team or team in ("", "---", "TOT", "2TM", "3TM"):
            continue
        k_rate = max(0.08, min(0.42, float(row["k_rate"]) if pd.notna(row["k_rate"]) else 0.228))
        fip    = max(2.0,  min(8.0,  float(row["fip"])    if pd.notna(row["fip"])    else 4.50))
        whip   = max(0.80, min(2.20, float(row["whip"])   if pd.notna(row["whip"])   else 1.35))
        hh     = max(0.20, min(0.55, float(row["hh"])     if pd.notna(row["hh"])     else 0.340))
        k_vuln    = max(0.0, min(100.0, (0.35 - k_rate) / (0.35 - 0.10) * 100))
        era_vuln  = max(0.0, min(100.0, (fip  - 2.0)    / (7.0  - 2.0)  * 100))
        whip_vuln = max(0.0, min(100.0, (whip - 0.90)   / (1.80 - 0.90) * 100))
        hh_vuln   = max(0.0, min(100.0, (hh   - 0.28)   / (0.50 - 0.28) * 100))
        team_scores[team] = round(max(0.0, min(100.0,
            k_vuln * 0.38 + hh_vuln * 0.22 + era_vuln * 0.22 + whip_vuln * 0.18
        )), 1)

    return team_scores
