"""
markets/hr.py — Home Run prop market.

Pure per-batter HR prop scorer. No Streamlit imports.
Caller resolves all data before calling here.

Key design choices:
- barrel% drives 35%+ of the score and is NON-NEGOTIABLE in the bettable gate.
  The bettable flag is False unless barrel% is measured (not proxied from HR/PA).
- compute_hr_score (scoring/hr.py) already encodes the full weight structure;
  this module wraps it with market-facing output: tier, prob, bettable gate, edge.
- No BvP or TTO signals (HR rate is more stable than TB across at-bats).
- Platoon IS included: LHB vs RHP advantage is real for HR (pull-side power).

Calibration status: deferred — backtest requires per-game HR outcome data
(game_results.hit_o15 tracks TB≥2 not HR specifically). Collect via
game_results.hr column when available.
"""

import math
from typing import Dict, List, Optional, Tuple

from config import HR_PROB_MAX, HR_PROB_MIDPOINT, HR_PROB_MIN, HR_PROB_SLOPE, HR_TIERS
from data.provenance import check_bettable_hr, compute_data_quality_score
from lib.constants import PARK_HR_FACTORS
from scoring.hr import compute_hr_score
from scoring.park import compute_lineup_score, compute_park_score, compute_platoon_score
from scoring.pitcher import compute_pitcher_score
from scoring.vegas import compute_vegas_score


def hr_score_to_prob(score: float) -> float:
    """Convert HR score 0–100 to P(batter hits HR today)."""
    prob = 1 / (1 + math.exp(-HR_PROB_SLOPE * (score - HR_PROB_MIDPOINT)))
    prob = HR_PROB_MIN + prob * (HR_PROB_MAX - HR_PROB_MIN)
    return round(min(HR_PROB_MAX, max(HR_PROB_MIN, prob)), 3)


def hr_get_tier(score: float) -> str:
    """Tier label for HR score."""
    for label, floor in HR_TIERS.items():
        if score >= floor:
            return label
    return "➖ NO PLAY"


def hr_market_edge(
    model_prob: float,
    prop_implied: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Edge for HR props.

    prop_implied: market's implied probability from the specific HR line
    (e.g. +280 → 26.3%; +350 → 22.2%). If None, returns tier-only label.

    Without Odds API data, pass None — the tier label conveys value direction.
    """
    if prop_implied is None:
        if model_prob >= 0.18:
            return 0.0, "💣 Elite HR spot — verify line"
        elif model_prob >= 0.14:
            return 0.0, "🔥 Strong HR upside — check odds"
        elif model_prob >= 0.11:
            return 0.0, "📊 Lean HR — verify line value"
        return 0.0, "➖ No play"

    if not (0.05 < prop_implied < 0.65):
        return 0.0, "⚠️ Invalid market implied"

    edge = model_prob - prop_implied
    if edge >= 0.06:
        label = f"🔥 +{edge*100:.0f}% EDGE"
    elif edge >= 0.03:
        label = f"✅ +{edge*100:.0f}% edge"
    elif edge >= 0:
        label = f"~{edge*100:.0f}% (thin)"
    else:
        label = f"❌ {edge*100:.0f}%"
    return edge, label


def score_one_batter_hr(
    *,
    name: str,
    player_id: str,
    team: str,
    opponent: str,
    game_pk: str,
    batter_hand: str,
    hand_real: bool,
    sp_hand: str,
    sp_name: str,
    sp_id: str,
    lineup_slot: int,
    lineup_confirmed: bool,
    batter_position: str,
    park_team: str,
    batter_stats: Dict,
    pitcher_stats: Dict,
    weather: Dict,
    implied: float,
    prop_implied: Optional[float],
    team_bullpen_scores: Dict[str, float],
    proxy_mode: bool = False,
) -> Dict:
    """
    Pure per-batter HR prop scorer.

    No network calls or Streamlit imports. Returns dict with score,
    tier, probability, bettable gate, and all sub-scores.

    barrel% source is surfaced explicitly — if it's 'proxy' (derived from
    HR/PA ratio), the bettable flag is False regardless of other signals.
    """
    sp_tbd = not sp_name or sp_name == "TBD"

    # ── Extract batter stats ──────────────────────────────────────────────────
    def f(key, default):
        try: return float(batter_stats.get(key, default) or default)
        except Exception: return float(default)

    barrel_rate   = f("barrel_rate",     0.070)
    hard_hit      = f("hard_hit_rate",   0.370)
    iso           = f("iso_proxy",       0.165)
    ev50          = f("ev50",            0.0)
    bat_speed     = f("bat_speed",       0.0)
    blast_rate    = f("blast_rate",      0.0)
    sweet_spot    = f("sweet_spot_rate", 0.30)
    exit_velocity = f("exit_velocity_avg", 88.5)

    # ── Pitch-type matchup for matchup_score param ────────────────────────────
    from scoring.park import compute_pitch_matchup_score
    matchup_sc, matchup_label = compute_pitch_matchup_score(batter_stats, pitcher_stats)

    # ── Park HR factor ────────────────────────────────────────────────────────
    park_hr_factor = PARK_HR_FACTORS.get(park_team, 1.0)
    park_sc, park_label = compute_park_score(park_team, True)

    # ── Vegas signal ──────────────────────────────────────────────────────────
    vegas_sc, _ = (0.0, "no lines") if not implied else compute_vegas_score(implied)

    # ── Platoon ───────────────────────────────────────────────────────────────
    plat_score, plat_label = compute_platoon_score(batter_hand, sp_hand)

    # ── Lineup slot ───────────────────────────────────────────────────────────
    lineup_sc, lineup_label = compute_lineup_score(lineup_slot)

    # ── HR score (barrel%-first, weather-adjusted) ────────────────────────────
    hr_score_val = compute_hr_score(
        barrel_rate=barrel_rate,
        sweet_spot=sweet_spot,
        park_hr_factor=park_hr_factor,
        implied_total=implied,
        weather=weather,
        hard_hit=hard_hit,
        exit_velocity=exit_velocity,
        iso=iso,
        ev50=ev50,
        bat_speed=bat_speed,
        blast_rate=blast_rate,
        pitch_matchup_score=matchup_sc,
    )

    # Small platoon and lineup adjustments (they affect opportunity, not raw power)
    platoon_adj = (plat_score - 50) * 0.03   # ±3 pts max platoon effect on HR
    lineup_adj  = (lineup_sc  - 50) * 0.02   # ±2 pts max lineup-slot effect
    final_score = max(0, min(100, round(hr_score_val + platoon_adj + lineup_adj, 1)))

    if sp_tbd:
        final_score = min(final_score, 74.0)
    if not lineup_confirmed:
        final_score = min(final_score, 70.0)

    prob = hr_score_to_prob(final_score)
    tier = hr_get_tier(final_score)
    edge_val, edge_label = hr_market_edge(prob, prop_implied)

    # ── Bettable gate ─────────────────────────────────────────────────────────
    bat_prov = batter_stats.get("_provenance", {})
    pit_prov = pitcher_stats.get("_provenance", {})
    bat_matched = batter_stats.get("data_source", "league_avg") != "league_avg"
    pit_matched = pitcher_stats.get("data_source", "league_avg") != "league_avg"
    sp_known = not sp_tbd
    dq_score = compute_data_quality_score(bat_prov, pit_prov, lineup_confirmed, sp_known, hand_real)
    is_bettable, bet_reasons = check_bettable_hr(
        bat_prov, bat_matched, pit_matched, lineup_confirmed, sp_known, hand_real
    )

    barrel_source = bat_prov.get("barrel_rate", "league_avg")

    return {
        "name":              name,
        "player_id":         player_id,
        "team":              team,
        "opponent":          opponent,
        "game_id":           str(game_pk),
        "lineup_slot":       lineup_slot,
        "lineup_confirmed":  lineup_confirmed,
        "batter_hand":       batter_hand,
        "batter_position":   batter_position,
        "sp_name":           sp_name,
        "sp_hand":           sp_hand,
        "sp_tbd":            sp_tbd,
        "score":             final_score,
        "prob":              prob,
        "tier":              tier,
        "park":              park_team,
        "park_label":        park_label,
        "park_hr_factor":    park_hr_factor,
        "weather":           weather,
        "wind_effect":       weather.get("wind_effect", "neutral"),
        "wind_speed":        weather.get("wind_speed", 0),
        "is_dome":           weather.get("is_dome", False),
        "temperature":       weather.get("temperature", 70),
        "implied_total":     implied,
        "market_edge":       round(edge_val * 100, 1),
        "edge_label":        edge_label,
        "platoon_label":     plat_label,
        "lineup_label":      lineup_label,
        "matchup_label":     matchup_label,
        "barrel_rate":       barrel_rate,
        "barrel_source":     barrel_source,   # "measured" | "proxy" | "league_avg"
        "hard_hit_rate":     hard_hit,
        "iso":               iso,
        "ev50":              ev50,
        "bat_speed":         bat_speed,
        "blast_rate":        blast_rate,
        "sub_hr_score":      round(hr_score_val, 1),
        "sub_platoon":       round(plat_score, 1),
        "sub_lineup":        round(lineup_sc, 1),
        "sub_matchup":       round(matchup_sc, 1),
        "dq_score":          dq_score,
        "bettable":          is_bettable,
        "non_bettable_reasons": bet_reasons,
        "_batter_prov":      bat_prov,
        "_pitcher_prov":     pit_prov,
    }
