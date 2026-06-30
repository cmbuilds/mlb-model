"""data/provenance.py — Data quality score and bettable gate logic."""

from typing import Dict, List, Tuple


# The 11 key inputs checked for data quality scoring
_DQ_CHECKS = [
    ("batter", "k_rate"),
    ("batter", "woba"),
    ("batter", "slg_proxy"),
    ("batter", "hard_hit_rate"),
    ("batter", "barrel_rate"),
    ("batter", "iso_proxy"),
    ("pitcher", "k_rate_allowed"),
    ("pitcher", "hard_hit_allowed"),
    "lineup_confirmed",
    "sp_known",
    "hand_real",
]


def compute_data_quality_score(
    batter_prov: Dict,
    pitcher_prov: Dict,
    lineup_confirmed: bool,
    sp_known: bool,
    hand_real: bool,
) -> int:
    """Return 0–100: % of 11 key scoring inputs that are measured."""
    checks = [
        batter_prov.get("k_rate",        "league_avg") == "measured",
        batter_prov.get("woba",          "league_avg") == "measured",
        batter_prov.get("slg_proxy",     "league_avg") == "measured",
        batter_prov.get("hard_hit_rate", "league_avg") == "measured",
        batter_prov.get("barrel_rate",   "league_avg") == "measured",
        batter_prov.get("iso_proxy",     "league_avg") == "measured",
        pitcher_prov.get("k_rate_allowed",  "league_avg") == "measured",
        pitcher_prov.get("hard_hit_allowed","league_avg") == "measured",
        lineup_confirmed,
        sp_known,
        hand_real,
    ]
    return round(sum(checks) / len(checks) * 100)


def check_bettable_tb(
    batter_prov: Dict,
    pitcher_prov: Dict,
    batter_matched: bool,
    pitcher_matched: bool,
    lineup_confirmed: bool,
    sp_known: bool,
    hand_real: bool,
    batter_pa: int = 0,
    min_pa: int = 50,
) -> Tuple[bool, List[str]]:
    """
    9-condition bettable gate for TB / O0.5 props.
    Returns (is_bettable, reasons_list). All conditions must be True to bet.

    batter_pa / min_pa: player blocked if PA known and below threshold.
    50 PA is the floor for rate stats (barrel%, HH%, wRC+) to carry any signal.
    Below that, a 1-for-2 game inflates wRC+ to 800+ and barrel% to 50% —
    noise that passes the provenance check but means nothing.
    """
    reasons: List[str] = []
    if batter_pa == 0:
        reasons.append("plate appearances unknown — cannot verify sufficient sample")
    elif batter_pa < min_pa:
        reasons.append(f"insufficient sample ({batter_pa} PA < {min_pa} minimum)")
    if not batter_matched:
        reasons.append("player not matched to real stats")
    if not hand_real:
        reasons.append("batter handedness defaulted (not confirmed)")
    if not sp_known:
        reasons.append("opposing SP unknown (TBD)")
    if not pitcher_matched:
        reasons.append("SP has no real stats")
    if not lineup_confirmed:
        reasons.append("lineup not confirmed")
    if batter_prov.get("k_rate",        "league_avg") != "measured":
        reasons.append("K% not measured")
    if batter_prov.get("slg_proxy",     "league_avg") != "measured":
        reasons.append("xSLG/SLG not measured")
    if batter_prov.get("woba",          "league_avg") != "measured":
        reasons.append("wOBA not measured")
    if batter_prov.get("hard_hit_rate", "league_avg") != "measured":
        reasons.append("hard-hit% not measured")
    return (len(reasons) == 0, reasons)


def check_bettable_hr(
    batter_prov: Dict,
    batter_matched: bool,
    pitcher_matched: bool,
    lineup_confirmed: bool,
    sp_known: bool,
    hand_real: bool,
    batter_pa: int = 0,
    min_pa: int = 50,
) -> Tuple[bool, List[str]]:
    """
    7-condition bettable gate for Home Run props.
    barrel% is non-negotiable — it's 35%+ of the HR score weight.
    ISO and HH% also required; K%/xSLG/wOBA are NOT required (not in HR formula).
    """
    reasons: List[str] = []
    if batter_pa == 0:
        reasons.append("plate appearances unknown — cannot verify sufficient sample")
    elif batter_pa < min_pa:
        reasons.append(f"insufficient sample ({batter_pa} PA < {min_pa} minimum)")
    if not batter_matched:
        reasons.append("player not matched to real stats")
    if not hand_real:
        reasons.append("batter handedness defaulted (not confirmed)")
    if not sp_known:
        reasons.append("opposing SP unknown (TBD)")
    if not pitcher_matched:
        reasons.append("SP has no real stats")
    if not lineup_confirmed:
        reasons.append("lineup not confirmed")
    if batter_prov.get("barrel_rate", "league_avg") != "measured":
        reasons.append("barrel% not measured — required for HR model")
    if batter_prov.get("hard_hit_rate", "league_avg") != "measured":
        reasons.append("hard-hit% not measured")
    if batter_prov.get("iso_proxy", "league_avg") != "measured":
        reasons.append("ISO not measured")
    return (len(reasons) == 0, reasons)


def check_bettable_k_prop(
    sp_prov: Dict,
    sp_matched: bool,
    n_batters_with_k: int,
    min_batters: int = 5,
) -> Tuple[bool, List[str]]:
    """
    Bettable gate for Pitcher K props.
    Requires: SP matched to real stats, SP K% measured, ≥5 opposing batters with measured K%.
    SwStr% is nice-to-have (scored but not required — proxy is acceptable).
    """
    reasons: List[str] = []
    if not sp_matched:
        reasons.append("SP not matched to real stats")
    if sp_prov.get("k_rate_allowed", "league_avg") != "measured":
        reasons.append("SP K% not measured")
    if n_batters_with_k < min_batters:
        reasons.append(f"only {n_batters_with_k}/{min_batters} opposing batters have measured K%")
    return (len(reasons) == 0, reasons)


def check_bettable_o05(
    batter_prov: Dict,
    pitcher_prov: Dict,
    batter_matched: bool,
    pitcher_matched: bool,
    lineup_confirmed: bool,
    sp_known: bool,
    hand_real: bool,
    batter_pa: int = 0,
    min_pa: int = 50,
) -> Tuple[bool, List[str]]:
    """
    9-condition bettable gate for O0.5 Any Hit props.
    Differs from TB gate: xSLG and barrel% removed (power not required for a hit);
    SP K% required (strikeout = direct no-hit outcome).
    """
    reasons: List[str] = []
    if batter_pa == 0:
        reasons.append("plate appearances unknown — cannot verify sufficient sample")
    elif batter_pa < min_pa:
        reasons.append(f"insufficient sample ({batter_pa} PA < {min_pa} minimum)")
    if not batter_matched:
        reasons.append("player not matched to real stats")
    if not hand_real:
        reasons.append("batter handedness defaulted (not confirmed)")
    if not sp_known:
        reasons.append("opposing SP unknown (TBD)")
    if not pitcher_matched:
        reasons.append("SP has no real stats")
    if not lineup_confirmed:
        reasons.append("lineup not confirmed")
    if batter_prov.get("k_rate",           "league_avg") != "measured":
        reasons.append("K% not measured")
    if batter_prov.get("woba",             "league_avg") != "measured":
        reasons.append("wOBA not measured")
    if batter_prov.get("hard_hit_rate",    "league_avg") != "measured":
        reasons.append("hard-hit% not measured")
    if pitcher_prov.get("k_rate_allowed",  "league_avg") != "measured":
        reasons.append("SP K% not measured")
    return (len(reasons) == 0, reasons)


def check_bettable_ml(
    home_sp_matched: bool,
    away_sp_matched: bool,
    home_sp_tbd: bool,
    away_sp_tbd: bool,
    home_sp_prov: Dict,
    away_sp_prov: Dict,
    home_n_batters: int,
    away_n_batters: int,
    has_odds: bool,
    min_batters: int = 5,
) -> Tuple[bool, List[str]]:
    """
    Bettable gate for moneyline.

    Requires:
    - Both SPs known (not TBD) and matched to real stats
    - Both SP vulnerability stats measured (k_rate_allowed as proxy)
    - Both lineups have ≥5 batters with measured wRC+
    - ML odds present
    """
    reasons: List[str] = []
    if home_sp_tbd:
        reasons.append("home SP unknown (TBD)")
    if away_sp_tbd:
        reasons.append("away SP unknown (TBD)")
    if not home_sp_matched:
        reasons.append("home SP not matched to real stats")
    if not away_sp_matched:
        reasons.append("away SP not matched to real stats")
    if home_sp_prov.get("k_rate_allowed", "league_avg") != "measured":
        reasons.append("home SP vulnerability stats not measured")
    if away_sp_prov.get("k_rate_allowed", "league_avg") != "measured":
        reasons.append("away SP vulnerability stats not measured")
    if home_n_batters < min_batters:
        reasons.append(f"home lineup: only {home_n_batters}/{min_batters} batters with measured wRC+")
    if away_n_batters < min_batters:
        reasons.append(f"away lineup: only {away_n_batters}/{min_batters} batters with measured wRC+")
    if not has_odds:
        reasons.append("ML odds not available")
    return (len(reasons) == 0, reasons)
