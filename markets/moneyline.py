"""
markets/moneyline.py — Moneyline game scorer.

Pure: no Streamlit imports. No network calls. Caller resolves all data.

Per-game (not per-batter): takes both teams' SP stats, lineup wRC+ averages,
bullpen vulnerabilities, run differentials, and optional Vegas ML odds.

Returns a full game record with win probabilities, edges, confidence scores,
best pick, tier, and the bettable gate result.
"""

from typing import Dict, Optional, Tuple

from config import ML_EDGE_LEAN, ML_EDGE_STRONG, ML_MIN_BATTERS_PER_TEAM
from data.provenance import check_bettable_ml
from scoring.moneyline import compute_ml_confidence, compute_win_probability


def ml_market_edge(
    model_prob: float,
    market_implied: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Edge for one ML side.

    model_prob:      home or away win probability (0–1) from compute_win_probability
    market_implied:  vig-removed implied probability from the book (0–1)

    Without market_implied: returns qualitative tier label.
    With market_implied:    returns (edge_pct, label) where edge_pct is percentage points.
    """
    if market_implied is None:
        if model_prob >= 0.60:
            return 0.0, "🔥 Favorite territory — verify line"
        elif model_prob >= 0.52:
            return 0.0, "📊 Slight lean — check odds"
        return 0.0, "➖ No lean"

    if not (0.20 <= market_implied <= 0.80):
        return 0.0, "⚠️ Invalid market implied"

    edge = (model_prob - market_implied) * 100
    if edge >= ML_EDGE_STRONG:
        label = f"🔥 +{edge:.1f}% STRONG EDGE"
    elif edge >= ML_EDGE_LEAN:
        label = f"✅ +{edge:.1f}% Lean"
    elif edge >= 0:
        label = f"~{edge:.1f}% (thin)"
    else:
        label = f"❌ {edge:.1f}%"
    return round(edge, 1), label


def score_game_ml(
    *,
    home_team: str,
    away_team: str,
    home_sp_name: str,
    away_sp_name: str,
    home_sp_id: str,
    away_sp_id: str,
    home_sp_stats: Dict,
    away_sp_stats: Dict,
    home_off_wrc: float,
    away_off_wrc: float,
    home_n_batters: int,
    away_n_batters: int,
    home_bp_vuln: float,
    away_bp_vuln: float,
    home_run_diff: float,
    away_run_diff: float,
    home_implied_runs: float,
    away_implied_runs: float,
    home_market_implied: Optional[float] = None,
    away_market_implied: Optional[float] = None,
    has_odds: bool = False,
    home_sp_matched: bool = False,
    away_sp_matched: bool = False,
    home_sp_prov: Optional[Dict] = None,
    away_sp_prov: Optional[Dict] = None,
) -> Dict:
    """
    Per-game moneyline scorer.

    home_sp_stats / away_sp_stats must carry "_sp_vuln" (0–100 pitcher vulnerability
    composite) pre-computed by the caller via compute_pitcher_score.

    home_market_implied / away_market_implied: vig-removed. If None, edge = None.
    has_odds: True if book ML odds were fetched for this game.

    Returns full game record: win probs, edges, confidence, best pick, tier,
    bettable gate, and all sub-scores for display.
    """
    home_sp_prov = home_sp_prov or {}
    away_sp_prov = away_sp_prov or {}

    home_sp_tbd = not home_sp_name or home_sp_name == "TBD"
    away_sp_tbd = not away_sp_name or away_sp_name == "TBD"

    hwp, detail_label = compute_win_probability(
        home_sp_stats, away_sp_stats,
        home_off_wrc, away_off_wrc,
        home_bp_vuln, away_bp_vuln,
        home_run_diff, away_run_diff,
        home_implied_runs, away_implied_runs,
    )
    awp = round(1.0 - hwp, 4)

    hedge_pct = (
        round((hwp - home_market_implied) * 100, 1)
        if home_market_implied is not None else None
    )
    aedge_pct = (
        round((awp - away_market_implied) * 100, 1)
        if away_market_implied is not None else None
    )

    _, hedge_label = ml_market_edge(hwp, home_market_implied)
    _, aedge_label = ml_market_edge(awp, away_market_implied)

    home_sp_vuln = float(home_sp_stats.get("_sp_vuln", 50.0))
    away_sp_vuln = float(away_sp_stats.get("_sp_vuln", 50.0))

    hconf = compute_ml_confidence(hedge_pct, home_sp_name, home_n_batters, home_sp_vuln, has_odds)
    aconf = compute_ml_confidence(aedge_pct, away_sp_name, away_n_batters, away_sp_vuln, has_odds)

    # Select best side (≥ ML_EDGE_LEAN to qualify as a pick)
    if hedge_pct is not None and aedge_pct is not None:
        if hedge_pct >= ML_EDGE_LEAN and hedge_pct >= aedge_pct:
            pick       = home_team;   pick_edge = hedge_pct; pick_conf = hconf
            pick_wp    = hwp;         pick_mkt  = home_market_implied
            pick_sp    = home_sp_name
        elif aedge_pct >= ML_EDGE_LEAN and aedge_pct > hedge_pct:
            pick       = away_team;   pick_edge = aedge_pct; pick_conf = aconf
            pick_wp    = awp;         pick_mkt  = away_market_implied
            pick_sp    = away_sp_name
        else:
            pick       = None;        pick_edge = max(hedge_pct or 0, aedge_pct or 0)
            pick_conf  = 0;           pick_wp   = None
            pick_mkt   = None;        pick_sp   = "—"
    else:
        pick       = None; pick_edge = 0; pick_conf = 0
        pick_wp    = None; pick_mkt  = None; pick_sp = "—"

    if pick and pick_edge >= ML_EDGE_STRONG:
        pick_tier = "🔥 Strong Edge"
    elif pick and pick_edge >= ML_EDGE_LEAN:
        pick_tier = "📊 Lean"
    else:
        pick_tier = "➖ No Play"

    # Bettable gate
    is_bettable, bet_reasons = check_bettable_ml(
        home_sp_matched=home_sp_matched,
        away_sp_matched=away_sp_matched,
        home_sp_tbd=home_sp_tbd,
        away_sp_tbd=away_sp_tbd,
        home_sp_prov=home_sp_prov,
        away_sp_prov=away_sp_prov,
        home_n_batters=home_n_batters,
        away_n_batters=away_n_batters,
        has_odds=has_odds,
        min_batters=ML_MIN_BATTERS_PER_TEAM,
    )

    return {
        "home_team":             home_team,
        "away_team":             away_team,
        "home_sp_name":          home_sp_name,
        "away_sp_name":          away_sp_name,
        "home_sp_tbd":           home_sp_tbd,
        "away_sp_tbd":           away_sp_tbd,
        "home_win_prob":         hwp,
        "away_win_prob":         awp,
        "home_market_implied":   home_market_implied,
        "away_market_implied":   away_market_implied,
        "home_edge_pct":         hedge_pct,
        "away_edge_pct":         aedge_pct,
        "home_edge_label":       hedge_label,
        "away_edge_label":       aedge_label,
        "home_confidence":       hconf,
        "away_confidence":       aconf,
        "home_n_batters":        home_n_batters,
        "away_n_batters":        away_n_batters,
        "home_wrc_avg":          home_off_wrc,
        "away_wrc_avg":          away_off_wrc,
        "home_run_diff":         home_run_diff,
        "away_run_diff":         away_run_diff,
        "home_bp_vuln":          home_bp_vuln,
        "away_bp_vuln":          away_bp_vuln,
        "home_sp_vuln":          home_sp_vuln,
        "away_sp_vuln":          away_sp_vuln,
        "pick":                  pick,
        "pick_edge":             pick_edge,
        "pick_confidence":       pick_conf,
        "pick_tier":             pick_tier,
        "pick_wp":               pick_wp,
        "pick_market_implied":   pick_mkt,
        "pick_sp":               pick_sp,
        "has_odds":              has_odds,
        "detail_label":          detail_label,
        "bettable":              is_bettable,
        "non_bettable_reasons":  bet_reasons,
    }
