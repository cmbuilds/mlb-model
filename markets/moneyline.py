"""
markets/moneyline.py — Per-game win probability scorer.

Pure: no Streamlit, no network calls, no market comparison.
Returns win probabilities and driver breakdown for both teams.
No picks, no edge, no confidence tiers — the tab user compares to live odds directly.
"""

from typing import Dict, Optional, Tuple

from config import ML_MIN_BATTERS_PER_TEAM
from data.provenance import check_bettable_ml
from scoring.moneyline import compute_win_probability
from scoring.pitcher import compute_sp_vuln_pure


def score_game_ml(
    *,
    home_team: str,
    away_team: str,
    home_sp_name: str,
    away_sp_name: str,
    home_sp_stats: Dict,
    away_sp_stats: Dict,
    home_wrc_plus: float,
    away_wrc_plus: float,
    home_n_batters: int,
    away_n_batters: int,
    home_bp_vuln: float,
    away_bp_vuln: float,
    home_bp_is_estimate: bool = False,
    away_bp_is_estimate: bool = False,
    home_sp_matched: bool = False,
    away_sp_matched: bool = False,
    home_sp_prov: Optional[Dict] = None,
    away_sp_prov: Optional[Dict] = None,
    park_key: str = "",
    game_time: str = "",
) -> Dict:
    """
    Score one game for win probability.

    Returns a full game record with win probs, driver breakdown, data flags,
    and bettable gate result. No market odds required or used.

    home_sp_stats / away_sp_stats: pitcher stat dicts from get_pitcher_stats().
    home_bp_is_estimate / away_bp_is_estimate: True when bullpen score is the
        league-average fallback (42.0) — shown as a soft flag, not a hard block.
    """
    home_sp_prov = home_sp_prov or {}
    away_sp_prov = away_sp_prov or {}

    home_sp_tbd = not home_sp_name or home_sp_name.strip() in ("TBD", "")
    away_sp_tbd = not away_sp_name or away_sp_name.strip() in ("TBD", "")

    # Pure SP vulnerability (no BP blend) so SP and BP are weighted independently
    home_sp_vuln, home_sp_label = compute_sp_vuln_pure(home_sp_stats)
    away_sp_vuln, away_sp_label = compute_sp_vuln_pure(away_sp_stats)

    # Hard gate: TBD SP → no projection
    if home_sp_tbd or away_sp_tbd:
        missing = []
        if home_sp_tbd: missing.append(f"{home_team} SP")
        if away_sp_tbd: missing.append(f"{away_team} SP")
        return {
            "home_team":         home_team,
            "away_team":         away_team,
            "home_sp_name":      home_sp_name or "TBD",
            "away_sp_name":      away_sp_name or "TBD",
            "home_sp_tbd":       home_sp_tbd,
            "away_sp_tbd":       away_sp_tbd,
            "blocked":           True,
            "block_reasons":     [f"SP not confirmed: {', '.join(missing)}"],
            "home_win_prob":     None,
            "away_win_prob":     None,
            "drivers":           {},
            "home_n_batters":    home_n_batters,
            "away_n_batters":    away_n_batters,
            "home_wrc_plus":     home_wrc_plus,
            "away_wrc_plus":     away_wrc_plus,
            "home_sp_vuln":      None,
            "away_sp_vuln":      None,
            "home_bp_vuln":      home_bp_vuln,
            "away_bp_vuln":      away_bp_vuln,
            "home_bp_estimate":  home_bp_is_estimate,
            "away_bp_estimate":  away_bp_is_estimate,
            "park_key":          park_key,
            "game_time":         game_time,
            "bettable":          False,
            "non_bettable_reasons": [f"SP not confirmed: {', '.join(missing)}"],
        }

    hwp, drivers = compute_win_probability(
        home_sp_vuln, away_sp_vuln,
        home_wrc_plus, away_wrc_plus,
        home_bp_vuln, away_bp_vuln,
    )
    awp = round(1.0 - hwp, 4)

    # Data quality flags (soft — don't block output, but surface in UI)
    data_flags = []
    if home_n_batters < ML_MIN_BATTERS_PER_TEAM:
        data_flags.append(f"{home_team}: only {home_n_batters} batters with measured wRC+")
    if away_n_batters < ML_MIN_BATTERS_PER_TEAM:
        data_flags.append(f"{away_team}: only {away_n_batters} batters with measured wRC+")
    if home_bp_is_estimate:
        data_flags.append(f"{home_team} bullpen: estimated (league avg)")
    if away_bp_is_estimate:
        data_flags.append(f"{away_team} bullpen: estimated (league avg)")

    is_bettable, bet_reasons = check_bettable_ml(
        home_sp_matched=home_sp_matched,
        away_sp_matched=away_sp_matched,
        home_sp_tbd=home_sp_tbd,
        away_sp_tbd=away_sp_tbd,
        home_sp_prov=home_sp_prov,
        away_sp_prov=away_sp_prov,
        home_n_batters=home_n_batters,
        away_n_batters=away_n_batters,
        min_batters=ML_MIN_BATTERS_PER_TEAM,
    )

    return {
        "home_team":         home_team,
        "away_team":         away_team,
        "home_sp_name":      home_sp_name,
        "away_sp_name":      away_sp_name,
        "home_sp_tbd":       home_sp_tbd,
        "away_sp_tbd":       away_sp_tbd,
        "blocked":           False,
        "block_reasons":     [],
        "home_win_prob":     hwp,
        "away_win_prob":     awp,
        "drivers":           drivers,
        "data_flags":        data_flags,
        "home_n_batters":    home_n_batters,
        "away_n_batters":    away_n_batters,
        "home_wrc_plus":     home_wrc_plus,
        "away_wrc_plus":     away_wrc_plus,
        "home_sp_vuln":      round(home_sp_vuln, 1),
        "away_sp_vuln":      round(away_sp_vuln, 1),
        "home_bp_vuln":      home_bp_vuln,
        "away_bp_vuln":      away_bp_vuln,
        "home_bp_estimate":  home_bp_is_estimate,
        "away_bp_estimate":  away_bp_is_estimate,
        "park_key":          park_key,
        "game_time":         game_time,
        "bettable":          is_bettable,
        "non_bettable_reasons": bet_reasons,
        "home_sp_label":     home_sp_label,
        "away_sp_label":     away_sp_label,
    }
