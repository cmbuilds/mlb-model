"""scoring/vegas.py — Implied-total, edge, and probability conversion."""

import math
from typing import Tuple

from config import (
    TB_PROB_MAX, TB_PROB_MIDPOINT, TB_PROB_MIN, TB_PROB_SLOPE,
    TIERS_FULL, TIERS_PROXY,
)


def compute_vegas_score(implied_total: float) -> Tuple[float, str]:
    """Vegas signal 0–100. Returns 0 when no lines (don't fake neutrality)."""
    if not implied_total or implied_total <= 0:
        return 0.0, "No lines ⚠️"
    score = (implied_total - 3.0) / (6.5 - 3.0) * 100
    score = max(0, min(100, score))
    if implied_total >= 5.5:
        flag = " 🔥"
    elif implied_total >= 4.5:
        flag = " ✅"
    elif implied_total < 3.5:
        flag = " ❄️"
    else:
        flag = ""
    return score, f"{implied_total:.1f} implied runs{flag}"


def compute_market_edge(model_prob: float, implied_prob: float) -> float:
    """Edge = model probability minus market implied probability."""
    if implied_prob <= 0:
        return 0.0
    return round((model_prob - implied_prob) * 100, 1)


def score_to_prob(score: float) -> float:
    """
    Map 0–100 score to O1.5 TB probability.
    Research-calibrated: score 50 → ~52%, 70 → ~64%, 80 → ~70%.
    Curve parameters live in config.py (TB_PROB_*).
    """
    prob = 1 / (1 + math.exp(-TB_PROB_SLOPE * (score - TB_PROB_MIDPOINT)))
    prob = TB_PROB_MIN + prob * (TB_PROB_MAX - TB_PROB_MIN)
    return round(min(TB_PROB_MAX, max(TB_PROB_MIN, prob)), 3)


def get_tier(score: float, proxy_mode: bool = False) -> str:
    """Map composite score to tier label. Thresholds live in config.py."""
    thresholds = TIERS_PROXY if proxy_mode else TIERS_FULL
    for label, floor in thresholds.items():
        if score >= floor:
            return label
    return "❌ NO PLAY"
