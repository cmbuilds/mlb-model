"""
markets/hits_o05.py — Over 0.5 Hits market.

Pure scoring kernel and parlay builder. No Streamlit imports.
Caller resolves all data before calling here.

Key difference from O1.5 TB:
- SP K% dominates (35% of composite) — each K is a guaranteed no-hit outcome
- Batter K-avoidance is the #1 batter stat (35% of batter sub-score)
- barrel%, ISO removed from batter score — power irrelevant for any-hit
- Lineup slot carries more weight (10%) — more AB = more chances
- No TTO bonus, no BvP signal (not enough sample size for single-hit prediction)
"""

import math
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from config import (
    O05_OFFSET_FULL, O05_OFFSET_PROXY, O05_PROB_MAX, O05_PROB_MIDPOINT,
    O05_PROB_MIN, O05_PROB_SLOPE, O05_TIERS_FULL, O05_TIERS_PROXY,
    O05_WEIGHTS, PARLAY_CORR_SAME_GAME, PARLAY_CORR_SAME_TEAM,
)
from data.provenance import compute_data_quality_score, check_bettable_o05
from scoring.hits import compute_hits_batter_score, compute_hits_pitcher_score
from scoring.park import compute_lineup_score, compute_park_score, compute_platoon_score
from scoring.streak import compute_streak_score
from scoring.vegas import compute_vegas_score


def o05_score_to_prob(score: float) -> float:
    """Convert O0.5 model score to P(any hit). Higher floor/ceiling than TB."""
    prob = 1 / (1 + math.exp(-O05_PROB_SLOPE * (score - O05_PROB_MIDPOINT)))
    prob = O05_PROB_MIN + prob * (O05_PROB_MAX - O05_PROB_MIN)
    return round(min(O05_PROB_MAX, max(O05_PROB_MIN, prob)), 3)


def o05_get_tier(score: float, proxy_mode: bool = False) -> str:
    """Tier label for O0.5 score."""
    thresholds = O05_TIERS_PROXY if proxy_mode else O05_TIERS_FULL
    for label, floor in thresholds.items():
        if score >= floor:
            return label
    return "❌ NO PLAY"


def _compute_final_score_o05(
    batter_score: float,
    pitcher_score: float,
    platoon_score: float,
    lineup_score: float,
    vegas_score: float,
    streak_score: float,
    park_score: float,
    proxy_mode: bool = False,
) -> float:
    """Final weighted composite for O0.5. Weights sum to 1.00 — no normalization."""
    W = O05_WEIGHTS
    raw = (
        batter_score  * W["batter"]  +
        pitcher_score * W["pitcher"] +
        platoon_score * W["platoon"] +
        lineup_score  * W["lineup"]  +
        vegas_score   * W["vegas"]   +
        streak_score  * W["streak"]  +
        park_score    * W["park"]
    )
    _offset = O05_OFFSET_PROXY if proxy_mode else O05_OFFSET_FULL
    return max(0, min(100, round(raw + _offset, 1)))


def o05_market_edge(
    model_prob: float,
    implied: float,
    prop_implied: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Edge vs market for O0.5 Any Hit props.
    Standard O0.5 lines run -160 to -200 (63–67% implied).
    """
    if prop_implied and 0.45 < prop_implied < 0.90:
        market_implied = prop_implied
    elif implied and implied > 0:
        if implied >= 5.5:
            market_implied = 0.67
        elif implied >= 4.5:
            market_implied = 0.64
        else:
            market_implied = 0.61
    else:
        market_implied = 0.636  # standard -175 line: 63.6% implied

    edge = model_prob - market_implied
    if edge >= 0.08:
        label = f"🔥 +{edge*100:.0f}% EDGE"
    elif edge >= 0.04:
        label = f"✅ +{edge*100:.0f}% edge"
    elif edge >= 0:
        label = f"~{edge*100:.0f}% (thin)"
    else:
        label = f"❌ {edge*100:.0f}%"
    return edge, label


def score_one_batter_o05(
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
    recent_form: Dict,
    implied: float,
    prop_implied: Optional[float],
    team_bullpen_scores: Dict[str, float],
    proxy_mode: bool = False,
) -> Dict:
    """
    Pure per-batter scoring kernel for O0.5 Any Hit market.

    No network calls or Streamlit imports — testable in isolation.
    Returns a dict with all scores, labels, provenance flags, and metadata.
    """
    sp_tbd = not sp_name or sp_name == "TBD"

    # ── Sub-scores ────────────────────────────────────────────────────────────
    bat_score, _, bat_details = compute_hits_batter_score(batter_stats)

    opp_team = opponent.strip().upper()
    bp_vuln = team_bullpen_scores.get(opp_team, 42.0)
    pit_score, pit_label = compute_hits_pitcher_score(pitcher_stats, bullpen_vuln=bp_vuln)

    plat_score, plat_label  = compute_platoon_score(batter_hand, sp_hand)
    lineup_sc, lineup_label = compute_lineup_score(lineup_slot)
    park_sc, park_label     = compute_park_score(park_team, True)
    vegas_sc, _             = (0.0, "no lines") if not implied else compute_vegas_score(implied)

    season_slg = batter_stats.get("slg_proxy", 0.398)
    streak_sc, streak_label = compute_streak_score(recent_form, season_slg)

    final_score = _compute_final_score_o05(
        batter_score=bat_score,
        pitcher_score=pit_score,
        platoon_score=plat_score,
        lineup_score=lineup_sc,
        vegas_score=vegas_sc,
        streak_score=streak_sc,
        park_score=park_sc,
        proxy_mode=proxy_mode,
    )

    # Score caps
    if sp_tbd:
        final_score = min(final_score, 70.0)
    if not lineup_confirmed:
        final_score = min(final_score, 68.0)

    prob = o05_score_to_prob(final_score)
    tier = o05_get_tier(final_score, proxy_mode=proxy_mode)
    market_edge_val, edge_label = o05_market_edge(prob, implied, prop_implied)

    # ── Data quality + bettable gate ─────────────────────────────────────────
    bat_prov = batter_stats.get("_provenance", {})
    pit_prov = pitcher_stats.get("_provenance", {})
    bat_matched = batter_stats.get("data_source", "league_avg") != "league_avg"
    pit_matched = pitcher_stats.get("data_source", "league_avg") != "league_avg"
    sp_known = not sp_tbd
    dq_score = compute_data_quality_score(bat_prov, pit_prov, lineup_confirmed, sp_known, hand_real)
    is_bettable, bet_reasons = check_bettable_o05(
        bat_prov, pit_prov, bat_matched, pit_matched,
        lineup_confirmed, sp_known, hand_real,
    )

    return {
        "name":             name,
        "player_id":        player_id,
        "team":             team,
        "opponent":         opponent,
        "game_id":          str(game_pk),
        "lineup_slot":      lineup_slot,
        "lineup_confirmed": lineup_confirmed,
        "batter_hand":      batter_hand,
        "batter_position":  batter_position,
        "sp_name":          sp_name,
        "sp_hand":          sp_hand,
        "sp_tbd":           sp_tbd,
        "score":            final_score,
        "prob":             prob,
        "tier":             tier,
        "park":             park_team,
        "park_label":       park_label,
        "implied_total":    implied,
        "market_edge":      round(market_edge_val * 100, 1),
        "edge_label":       edge_label,
        "platoon_label":    plat_label,
        "lineup_label":     lineup_label,
        "pitcher_label":    pit_label,
        "streak_label":     streak_label,
        "k_rate":           batter_stats.get("k_rate", 0),
        "woba":             batter_stats.get("woba", 0),
        "wrc_plus":         batter_stats.get("wrc_plus", 100.0),
        "hard_hit_rate":    batter_stats.get("hard_hit_rate", 0),
        "sub_batter":       round(bat_score, 1),
        "sub_pitcher":      round(pit_score, 1),
        "sub_platoon":      round(plat_score, 1),
        "sub_lineup":       round(lineup_sc, 1),
        "sub_vegas":        round(vegas_sc, 1),
        "sub_streak":       round(streak_sc, 1),
        "sub_park":         round(park_sc, 1),
        "bullpen_vuln":     round(bp_vuln, 1),
        "recent_h_per_game": recent_form.get("h_per_game"),
        "recent_games":     recent_form.get("games", 0),
        "sp_id":            sp_id,
        "dq_score":         dq_score,
        "bettable":         is_bettable,
        "non_bettable_reasons": bet_reasons,
        "_batter_prov":     bat_prov,
        "_pitcher_prov":    pit_prov,
        "_pitcher_k_rate":  float(pitcher_stats.get("k_rate_allowed", 0.220) or 0.220),
    }


def build_parlays_o05(
    plays: List[Dict],
    num_legs: int = 3,
    max_same_team: int = 2,
    min_score: float = 70.0,
) -> List[Dict]:
    """
    Build optimal O0.5 parlays from bettable plays.
    Same correlation-adjustment logic as TB parlay builder.
    Returns top-10 by combined model probability.
    """
    eligible = [p for p in plays if p["score"] >= min_score and not p["sp_tbd"]]
    if len(eligible) < num_legs:
        return []

    best: List[Dict] = []
    for combo in combinations(eligible[:20], num_legs):
        teams = [p["team"] for p in combo]
        games = [p["game_id"] for p in combo]
        team_counts: Dict[str, int] = {}
        for t in teams:
            team_counts[t] = team_counts.get(t, 0) + 1
        if max(team_counts.values()) > max_same_team:
            continue

        corr = 1.0
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                if combo[i]["team"] == combo[j]["team"]:
                    corr *= PARLAY_CORR_SAME_TEAM
                elif combo[i]["game_id"] == combo[j]["game_id"]:
                    corr *= PARLAY_CORR_SAME_GAME

        combined_raw = 1.0
        for p in combo:
            combined_raw *= p["prob"]
        combined_prob = combined_raw * corr

        fair_payout = 1.0 / combined_prob if combined_prob > 0 else 999.0
        ev = (combined_prob * fair_payout) - 1.0
        avg_score = sum(p["score"] for p in combo) / num_legs

        best.append({
            "players":           [p["name"] for p in combo],
            "teams":             teams,
            "games":             list(set(games)),
            "num_legs":          num_legs,
            "combined_prob":     round(combined_prob * 100, 1),
            "combined_prob_raw": round(combined_raw * 100, 1),
            "fair_payout":       round(fair_payout, 2),
            "ev":                round(ev * 100, 1),
            "avg_score":         round(avg_score, 1),
            "min_score":         round(min(p["score"] for p in combo), 1),
            "corr_factor":       round(corr, 3),
            "combo":             combo,
            "notes":             "SGP ⭐" if len(set(games)) == 1 else f"{len(set(games))} games",
        })

    best.sort(key=lambda x: x["combined_prob"], reverse=True)
    return best[:10]
