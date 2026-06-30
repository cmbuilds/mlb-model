"""
markets/k_props.py — Pitcher Strikeout prop market (e.g. Crochet O7.5 Ks).

Pure SP-level K prop scorer. No Streamlit imports.
Caller resolves all data (SP stats, opposing lineup) before calling here.

Model predicts K upside for a starting pitcher in a specific game context.
Score 80+ = elite K spot (bet the over). Score <60 = no play.

Limitation: Calibration requires SP game K total data (game_results only has
per-batter outcomes). Run data/backtest_fetch_sp_ks.py when available.
"""

from typing import Dict, List, Optional, Tuple

from data.provenance import check_bettable_k_prop
from scoring.strikeout import (
    compute_batter_k_propensity, compute_sp_k_score,
    k_get_tier, k_score_to_prob,
)


def k_prop_market_edge(
    model_score: float,
    market_implied: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Edge for K prop.

    market_implied: the book's implied probability for the specific K line
    (e.g. -120 → 54.5%). If None, returns score-relative label only.

    Note: K prop lines are SP-specific (e.g. O7.5 for Crochet, O5.5 for Steele),
    so model_score represents K upside ranking. Pass market_implied from the
    specific odds line to get a meaningful edge number.
    """
    prob = k_score_to_prob(model_score)
    if market_implied is None:
        if model_score >= 80:
            return 0.0, "⚡ Elite K spot — check specific line"
        elif model_score >= 70:
            return 0.0, "🔥 Strong K upside — check specific line"
        elif model_score >= 60:
            return 0.0, "📊 Lean K — verify line value"
        return 0.0, "➖ No play"

    edge = prob - market_implied
    if edge >= 0.08:
        label = f"🔥 +{edge*100:.0f}% EDGE"
    elif edge >= 0.04:
        label = f"✅ +{edge*100:.0f}% edge"
    elif edge >= 0:
        label = f"~{edge*100:.0f}% (thin)"
    else:
        label = f"❌ {edge*100:.0f}%"
    return edge, label


def score_sp_k_prop(
    *,
    sp_name: str,
    sp_id: str,
    sp_team: str,
    opp_team: str,
    game_pk: str,
    sp_stats: Dict,
    opp_batter_stats: List[Dict],
    implied_total: float,
    ump_k_adj: float = 0.0,
    ump_name: str = "—",
    market_implied: Optional[float] = None,
) -> Dict:
    """
    Pure SP-level K prop scorer.

    sp_stats: pitcher_stats dict (same format as scoring/pitcher.py inputs)
    opp_batter_stats: list of batter_stats dicts for the opposing starting lineup

    Returns dict with score, tier, all sub-scores, and bettable flag.
    """
    # ── Opposing lineup K% average ────────────────────────────────────────────
    batter_k_rates = []
    batter_k_scores = []
    for b in opp_batter_stats:
        k_score, _, _ = compute_batter_k_propensity(b, sp_stats)
        k_rate = float(b.get("k_rate", 0.228) or 0.228)
        batter_k_scores.append(k_score)
        batter_k_rates.append(k_rate)

    opp_lineup_k_avg = (sum(batter_k_rates) / len(batter_k_rates)
                        if batter_k_rates else 0.228)
    opp_batter_k_avg_score = (sum(batter_k_scores) / len(batter_k_scores)
                               if batter_k_scores else 50.0)

    # ── SP K score ────────────────────────────────────────────────────────────
    score, label, details = compute_sp_k_score(
        sp_stats, opp_lineup_k_avg, implied_total, ump_k_adj
    )

    prob = k_score_to_prob(score)
    tier = k_get_tier(score)
    edge_val, edge_label = k_prop_market_edge(score, market_implied)

    # ── Bettable gate ─────────────────────────────────────────────────────────
    sp_prov = sp_stats.get("_provenance", {})
    sp_matched = sp_stats.get("data_source", "league_avg") != "league_avg"
    n_batters_with_k = sum(
        1 for b in opp_batter_stats
        if b.get("_provenance", {}).get("k_rate", "league_avg") == "measured"
    )
    is_bettable, bet_reasons = check_bettable_k_prop(
        sp_prov, sp_matched, n_batters_with_k
    )

    return {
        "sp_name":             sp_name,
        "sp_id":               sp_id,
        "sp_team":             sp_team,
        "opp_team":            opp_team,
        "game_id":             str(game_pk),
        "score":               score,
        "prob":                prob,
        "tier":                tier,
        "label":               label,
        "edge":                round(edge_val * 100, 1),
        "edge_label":          edge_label,
        "sp_k_pct":            details["sp_k_pct"],
        "sp_swstr_pct":        details["sp_swstr_pct"],
        "swstr_real":          details["swstr_real"],
        "opp_lineup_k_avg":    details["opp_lineup_k_avg"],
        "n_batters":           len(opp_batter_stats),
        "n_batters_with_k":    n_batters_with_k,
        "opp_batter_k_score":  round(opp_batter_k_avg_score, 1),
        "implied_total":       implied_total,
        "ump_name":            ump_name,
        "ump_k_adj":           details["ump_k_adj"],
        "sub_sp_k":            details["sub_sp_k"],
        "sub_swstr":           details["sub_swstr"],
        "sub_opp_lineup":      details["sub_opp_lineup"],
        "sub_context":         details["sub_context"],
        "bettable":            is_bettable,
        "non_bettable_reasons": bet_reasons,
        "_sp_prov":            sp_prov,
    }
