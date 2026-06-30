"""scoring/strikeout.py — K scoring functions for Pitcher Strikeout props.

Two entry points:
  compute_sp_k_score     — SP-level K upside (0-100, used by markets/k_props.py)
  compute_batter_k_propensity — per-batter K tendency (0-100, aggregated into lineup K score)

SwStr% from Savant whiff_percent column (in pitcher_stats if pipeline was re-run after
Phase 4.3); falls back to K%*0.49 proxy (r≈0.85 correlation, 2-5% error) automatically.
"""

import math
from typing import Dict, List, Tuple

from config import K_PROB_MAX, K_PROB_MIDPOINT, K_PROB_MIN, K_PROB_SLOPE, K_TIERS, K_WEIGHTS


def compute_sp_k_score(
    pitcher_stats: Dict,
    opp_lineup_k_avg: float,
    implied_total: float,
    ump_k_adj: float = 0.0,
) -> Tuple[float, str, Dict]:
    """
    SP-level K prop score 0–100. HIGH = elite K spot, bet the OVER.

    Weights live in config.K_WEIGHTS:
      sp_k (40%)      — SP K%: primary strikeout signal
      sp_swstr (20%)  — SP SwStr%: predictive of K rate, independent of BABIP
      opp_lineup (25%)— opposing lineup K%: high K lineups give SP more K chances
      context (15%)   — low implied total (pitcher's duel) + tight umpire zone

    SwStr%: uses pitcher_stats["swstr_pct"] if > 0.02 (real Savant value);
    otherwise holds neutral 50.0 (true neutral, not proxy-boosted).
    """
    def f(key, default):
        try: return float(pitcher_stats.get(key, default) or default)
        except Exception: return float(default)

    pit_k  = f("k_rate_allowed", 0.228)
    swstr  = f("swstr_pct", 0.0)
    swstr_real = swstr > 0.02

    # Component 1: SP K% (primary)
    sp_k_score = max(0, min(100, 50 + (pit_k - 0.228) / 0.060 * 25))

    # Component 2: SP SwStr% (independent strikeout signal)
    if swstr_real:
        swstr_score = max(0, min(100, 50 + (swstr - 0.110) / 0.040 * 25))
    else:
        swstr_score = 50.0  # neutral when Savant data not yet populated

    # Component 3: Opposing lineup K%
    opp_k_score = max(0, min(100, 50 + (opp_lineup_k_avg - 0.228) / 0.060 * 25))

    # Component 4: Game context
    if implied_total <= 3.5:
        game_score = 75.0   # pitcher's duel → deep IP, max K chances
    elif implied_total <= 4.5:
        game_score = 62.0
    elif implied_total <= 5.5:
        game_score = 50.0
    else:
        game_score = 35.0   # high-scoring game → early hook, fewer K chances
    ump_pts = max(-15.0, min(15.0, ump_k_adj * 400.0))  # +0.02 k_rate_added → +8 pts
    context_score = max(0, min(100, game_score + ump_pts))

    W = K_WEIGHTS
    raw = (
        sp_k_score    * W["sp_k"]       +
        swstr_score   * W["sp_swstr"]   +
        opp_k_score   * W["opp_lineup"] +
        context_score * W["context"]
    )

    label = (
        f"SP K%: {pit_k*100:.0f}% | "
        f"SwStr%: {swstr*100:.0f}%{'(real)' if swstr_real else '(N/A)'} | "
        f"Opp lineup K%: {opp_lineup_k_avg*100:.0f}% | "
        f"Context: {context_score:.0f}"
    )
    details = {
        "sp_k_pct":         round(pit_k * 100, 1),
        "sp_swstr_pct":     round(swstr * 100, 1) if swstr_real else "N/A",
        "swstr_real":       swstr_real,
        "opp_lineup_k_avg": round(opp_lineup_k_avg * 100, 1),
        "implied_total":    implied_total,
        "ump_k_adj":        round(ump_k_adj * 100, 2),
        "sub_sp_k":         round(sp_k_score, 1),
        "sub_swstr":        round(swstr_score, 1),
        "sub_opp_lineup":   round(opp_k_score, 1),
        "sub_context":      round(context_score, 1),
    }
    return max(0, min(100, round(raw, 1))), label, details


def compute_batter_k_propensity(
    batter_stats: Dict,
    pitcher_stats: Dict,
    lineup_slot: int = 5,
) -> Tuple[float, str, Dict]:
    """
    Per-batter K propensity 0–100. HIGH = batter likely to strike out.

    Used to build the opposing lineup K score for compute_sp_k_score.

    Weights:
      Batter K%:   40% (most stable batter K signal)
      Pitcher K%:  35% (K%, SwStr% blend when both available)
      Whiff/chase: 15% (batter SwStr% + O-Swing%; falls back to K%-proxy)
      Lineup slot: 10% (PA opportunity; middle order sees TTO2+ = more Ks)
    """
    def fb(key, default):
        try: return float(batter_stats.get(key, default) or default)
        except Exception: return float(default)

    def fp(key, default):
        try: return float(pitcher_stats.get(key, default) or default)
        except Exception: return float(default)

    # Batter K%
    batter_k = fb("k_rate", 0.228)
    batter_k_score = max(0, min(100, 50 + (batter_k - 0.228) / 0.060 * 25))

    # Pitcher K% + SwStr% blend
    pit_k   = fp("k_rate_allowed", 0.228)
    swstr   = fp("swstr_pct", 0.0)
    pit_raw = max(0, min(100, 50 + (pit_k - 0.228) / 0.060 * 25))
    if swstr > 0.02:
        swstr_sc  = max(0, min(100, 50 + (swstr - 0.110) / 0.040 * 25))
        pit_k_score = pit_raw * 0.70 + swstr_sc * 0.30
    else:
        pit_k_score = pit_raw
    pit_k_score = max(0, min(100, pit_k_score))

    # Batter whiff/chase — real values from FanGraphs (often unavailable); K%-proxy fallback
    batter_swstr = fb("batter_swstr_pct", 0.0)
    o_swing      = fb("o_swing_pct", 0.0)
    if batter_swstr > 0.02 and o_swing > 0.02:
        swstr_b_sc  = max(0, min(100, 50 + (batter_swstr - 0.110) / 0.035 * 25))
        oswing_sc   = max(0, min(100, 50 + (o_swing - 0.310) / 0.060 * 25))
        whiff_score = swstr_b_sc * 0.55 + oswing_sc * 0.45
    elif batter_swstr > 0.02:
        whiff_score = max(0, min(100, 50 + (batter_swstr - 0.110) / 0.035 * 25))
    else:
        whiff_score = max(0, min(100, 50 + (batter_k - 0.228) / 0.060 * 15))

    # Lineup slot → PA opportunity (middle order gets TTO2+ matchup = more Ks)
    slot_pa_score = {1: 55, 2: 58, 3: 60, 4: 62, 5: 63,
                     6: 52, 7: 48, 8: 45, 9: 40}.get(lineup_slot, 50)

    raw = (
        batter_k_score * 0.40 +
        pit_k_score    * 0.35 +
        whiff_score    * 0.15 +
        slot_pa_score  * 0.10
    )

    label = f"Batter K%: {batter_k*100:.1f}% | SP K%: {pit_k*100:.1f}%"
    details = {
        "batter_k_pct":  round(batter_k * 100, 1),
        "batter_k_score": round(batter_k_score, 1),
        "pit_k_pct":     round(pit_k * 100, 1),
        "pit_k_score":   round(pit_k_score, 1),
        "whiff_score":   round(whiff_score, 1),
        "slot_pa_score": slot_pa_score,
    }
    return max(0, min(100, round(raw, 1))), label, details


def k_score_to_prob(score: float) -> float:
    """Convert K upside score to P(SP exceeds a typical K line). General — not line-specific."""
    prob = 1 / (1 + math.exp(-K_PROB_SLOPE * (score - K_PROB_MIDPOINT)))
    prob = K_PROB_MIN + prob * (K_PROB_MAX - K_PROB_MIN)
    return round(min(K_PROB_MAX, max(K_PROB_MIN, prob)), 3)


def k_get_tier(score: float) -> str:
    """Tier label for SP K prop score."""
    for label, floor in K_TIERS.items():
        if score >= floor:
            return label
    return "➖ NO PLAY"
