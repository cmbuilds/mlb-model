"""
markets/tb_o15.py — Over 1.5 Total Bases market.

Pure scoring kernel and parlay builder. No Streamlit imports.
Caller (run_model in the monolith) resolves all data before calling here.
"""

from itertools import combinations
from typing import Dict, List, Optional, Tuple

from config import PARLAY_CORR_SAME_GAME, PARLAY_CORR_SAME_TEAM
from data.provenance import compute_data_quality_score, check_bettable_tb
from lib.constants import PARK_HR_FACTORS
from scoring.batter import compute_batter_score
from scoring.final import compute_final_score
from scoring.hr import compute_hr_score
from scoring.park import (
    compute_lineup_score, compute_park_score,
    compute_pitch_matchup_score, compute_platoon_score,
)
from scoring.pitcher import compute_pitcher_score
from scoring.streak import compute_bvp_score, compute_streak_score, compute_tto_bonus
from scoring.vegas import compute_vegas_score, get_tier, score_to_prob
from scoring.weather import compute_weather_score


def tb_market_edge(
    model_prob: float,
    implied: float,
    prop_implied: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Edge vs market for O1.5 TB props.
    Prefer prop-specific implied probability; fall back to team-total estimate.
    Returns (edge_pct_float, label_str).
    """
    if prop_implied and 0.30 < prop_implied < 0.85:
        market_implied = prop_implied
    elif implied and implied > 0:
        if implied >= 5.5:
            market_implied = 0.56
        elif implied >= 4.5:
            market_implied = 0.53
        else:
            market_implied = 0.50
    else:
        market_implied = 0.515  # historical MLB O1.5 base rate

    edge = model_prob - market_implied
    if edge >= 0.10:
        label = f"🔥 +{edge*100:.0f}% EDGE"
    elif edge >= 0.05:
        label = f"✅ +{edge*100:.0f}% edge"
    elif edge >= 0:
        label = f"~{edge*100:.0f}% (thin)"
    else:
        label = f"❌ {edge*100:.0f}%"
    return edge, label


def score_one_batter(
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
    bvp_data: Dict,
    weather: Dict,
    implied: float,
    prop_implied: Optional[float],
    team_bullpen_scores: Dict[str, float],
    proxy_mode: bool = False,
) -> Dict:
    """
    Pure per-batter scoring kernel for O1.5 TB market.

    All data must be fully resolved by the caller. No network calls or
    Streamlit imports here — keeps this function testable in isolation.

    Returns a dict with all scores, labels, provenance flags, and metadata
    ready to append to the results list.
    """
    sp_tbd = not sp_name or sp_name == "TBD"

    # ── Sub-scores ────────────────────────────────────────────────────────────
    bat_score, _, bat_details = compute_batter_score(batter_stats)

    opp_team = opponent.strip().upper()
    bp_vuln = team_bullpen_scores.get(opp_team, 42.0)
    pit_score, pit_label = compute_pitcher_score(pitcher_stats, bullpen_vuln=bp_vuln)

    matchup_sc, matchup_label = compute_pitch_matchup_score(batter_stats, pitcher_stats)
    plat_score, plat_label   = compute_platoon_score(batter_hand, sp_hand)
    lineup_sc, lineup_label  = compute_lineup_score(lineup_slot)
    park_sc, park_label      = compute_park_score(park_team, True)
    weather_sc, weather_label = compute_weather_score(weather)
    vegas_missing = not bool(implied)
    vegas_sc, vegas_label = (0.0, "no lines") if vegas_missing else compute_vegas_score(implied)
    tto_sc, tto_label        = compute_tto_bonus(lineup_slot)

    season_slg = batter_stats.get("slg_proxy", 0.398)
    streak_sc, streak_label = compute_streak_score(recent_form, season_slg)
    bvp_sc, bvp_label, bvp_sig = compute_bvp_score(bvp_data, season_slg)

    _bvp_boost = 0.08 if bvp_sig == "owns" else 0.0

    final_score = compute_final_score(
        batter_score=bat_score,
        pitcher_vuln_score=pit_score,
        platoon_score=plat_score,
        lineup_score=lineup_sc,
        park_score=park_sc,
        weather_score=weather_sc,
        vegas_score=vegas_sc,
        tto_bonus=tto_sc,
        pitch_matchup_score=matchup_sc,
        streak_score=streak_sc,
        bvp_score=bvp_sc,
        bvp_weight_boost=_bvp_boost,
        proxy_mode=proxy_mode,
        vegas_missing=vegas_missing,
    )

    # Caps
    if sp_tbd:
        final_score = min(final_score, 72.0)
    if not lineup_confirmed:
        final_score = min(final_score, 70.0)

    prob = score_to_prob(final_score)
    tier = get_tier(final_score, proxy_mode=proxy_mode)
    market_edge_val, edge_label = tb_market_edge(prob, implied, prop_implied)

    # ── HR sub-score (for HR tab reference, not used in TB final score) ──────
    hr_score = compute_hr_score(
        barrel_rate=batter_stats.get("barrel_rate", 0.07),
        sweet_spot=batter_stats.get("sweet_spot_rate", 0.30),
        park_hr_factor=PARK_HR_FACTORS.get(park_team, 1.0),
        implied_total=implied,
        weather=weather,
        hard_hit=batter_stats.get("hard_hit_rate", 0.37),
        exit_velocity=batter_stats.get("exit_velocity_avg", 88.5),
        iso=batter_stats.get("iso_proxy", 0.165),
        ev50=batter_stats.get("ev50", 0.0),
        bat_speed=batter_stats.get("bat_speed", 0.0),
        blast_rate=batter_stats.get("blast_rate", 0.0),
        pitch_matchup_score=matchup_sc,
    )

    # ── Data quality + bettable gate ─────────────────────────────────────────
    bat_prov = batter_stats.get("_provenance", {})
    pit_prov = pitcher_stats.get("_provenance", {})
    bat_matched = batter_stats.get("data_source", "league_avg") != "league_avg"
    pit_matched = pitcher_stats.get("data_source", "league_avg") != "league_avg"
    sp_known = not sp_tbd
    batter_pa = int(batter_stats.get("pa", 0) or 0)
    dq_score = compute_data_quality_score(bat_prov, pit_prov, lineup_confirmed, sp_known, hand_real)
    is_bettable, bet_reasons = check_bettable_tb(
        bat_prov, pit_prov, bat_matched, pit_matched,
        lineup_confirmed, sp_known, hand_real,
        batter_pa=batter_pa,
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
        "weather":          weather,
        "weather_label":    weather_label,
        "implied_total":    implied,
        "market_edge":      round(market_edge_val * 100, 1),
        "edge_label":       edge_label,
        "tto_label":        tto_label,
        "platoon_label":    plat_label,
        "lineup_label":     lineup_label,
        "pitcher_label":    pit_label,
        "hr_score":         hr_score,
        "xslg":             bat_details.get("xSLG", 0),
        "barrel_rate":      batter_stats.get("barrel_rate", 0),
        "hard_hit_rate":    batter_stats.get("hard_hit_rate", 0),
        "k_rate":           batter_stats.get("k_rate", 0),
        "bb_rate":          batter_stats.get("bb_rate", 0.082),
        "wrc_plus":         batter_stats.get("wrc_plus", 100.0),
        "iso":              bat_details.get("ISO", 0),
        "exit_velocity":    batter_stats.get("exit_velocity_avg", 0),
        "sweet_spot_rate":  batter_stats.get("sweet_spot_rate", 0),
        "sub_batter":       round(bat_score, 1),
        "sub_pitcher":      round(pit_score, 1),
        "sub_matchup":      round(matchup_sc, 1),
        "matchup_label":    matchup_label,
        "sub_streak":       round(streak_sc, 1),
        "streak_label":     streak_label,
        "recent_tb_per_game": recent_form.get("tb_per_game"),
        "recent_games":     recent_form.get("games", 0),
        "_hr_last7":        recent_form.get("hr_last_7", 0),
        "game_total":       0.0,  # caller sets this after scoring (needs both sides)
        "_h_last7":         recent_form.get("h_last_7", 0),
        "_ab_last7":        recent_form.get("ab_last_7", 0),
        "sub_bvp":          round(bvp_sc, 1),
        "bvp_label":        bvp_label,
        "bvp_sig":          bvp_sig,
        "bvp_ab":           bvp_data.get("ab", 0),
        "bvp_slg":          bvp_data.get("slg"),
        "bvp_hr":           bvp_data.get("hr", 0),
        "bvp_xbh":          bvp_data.get("xbh", 0),
        "sp_id":            sp_id,
        "sub_platoon":      round(plat_score, 1),
        "sub_lineup":       round(lineup_sc, 1),
        "sub_park":         round(park_sc, 1),
        "sub_weather":      round(weather_sc, 1),
        "sub_vegas":        round(vegas_sc, 1),
        "vegas_missing":    vegas_missing,
        "batter_pa":        batter_pa,
        "bullpen_vuln":     round(bp_vuln, 1),
        "platoon_edge":     plat_label,
        "bat_speed":        batter_stats.get("bat_speed", 0),
        "blast_rate":       batter_stats.get("blast_rate", 0),
        "ev50":             batter_stats.get("ev50", 0),
        "sprint_speed":     batter_stats.get("sprint_speed", 0),
        "dq_score":         dq_score,
        "bettable":         is_bettable,
        "non_bettable_reasons": bet_reasons,
        "_batter_prov":     bat_prov,
        "_pitcher_prov":    pit_prov,
        "temperature":      weather.get("temperature", 70),
        "wind_speed":       weather.get("wind_speed", 0),
        "wind_dir":         weather.get("wind_dir_label", ""),
        "wind_effect":      weather.get("wind_effect", "neutral"),
        "is_dome":          weather.get("is_dome", False),
        "_pitcher_k_rate":  float(pitcher_stats.get("k_rate_allowed", 0.228) or 0.228),
        "_pitcher_swstr":   float(pitcher_stats.get("swstr_pct", 0.0) or 0.0),
    }


def build_parlays(
    plays: List[Dict],
    num_legs: int = 3,
    max_same_team: int = 2,
    min_score: float = 70.0,
) -> List[Dict]:
    """
    Build optimal parlays from bettable plays.
    Correlation-adjusted: same-team legs discounted 10%, same-game 5%.
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
