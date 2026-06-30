"""
config.py — Single source of truth for weights, thresholds, and calibration.

To retune: change values here; no hunting across modules.
Phase 3 backtest will drive recalibration of these numbers.
"""

# ── O1.5 Total Bases: final composite weights (sum ≈ 1.03, normalized /1.03) ─
TB_WEIGHTS = {
    "pitcher":    0.30,
    "batter_base": 0.28,   # reduced by bvp_weight_boost when batter "owns" SP
    "platoon":    0.12,
    "vegas":      0.08,
    "park":       0.07,
    "streak":     0.05,
    "tto":        0.04,
    "weather":    0.04,
    "matchup":    0.02,
    "bvp_base":   0.02,    # rises to 0.10 when bvp_sig == "owns"
    "lineup":     0.01,
    "bvp_owns":   0.10,    # weight when batter dominates SP
    "batter_min": 0.20,    # floor for batter weight when bvp boost is active
}

# ── Score offset (added after normalization, before clamping 0–100) ───────────
TB_OFFSET_FULL  = 7.0    # Savant + FanGraphs data present
TB_OFFSET_PROXY = 9.5    # MLB API only / stale disk cache (compressed score range)

# ── Tier thresholds ──────────────────────────────────────────────────────────
TIERS_FULL = {
    "🔒 TIER 1": 80,
    "✅ TIER 2": 70,
    "📊 TIER 3": 60,
    "❌ NO PLAY": 0,
}
TIERS_PROXY = {   # shifted -5 pts to account for proxy score compression
    "🔒 TIER 1": 75,
    "✅ TIER 2": 65,
    "📊 TIER 3": 55,
    "❌ NO PLAY": 0,
}

# ── TB probability calibration (score → P(O1.5 TB hit)) ─────────────────────
TB_PROB_SLOPE     = 0.07    # logistic curve steepness
TB_PROB_MIDPOINT  = 62      # score where P ≈ base rate
TB_PROB_MIN       = 0.42    # floor probability
TB_PROB_MAX       = 0.78    # ceiling probability

# ── Data-quality score: inputs counted (denominator = 11) ────────────────────
DQ_SCORED_INPUTS = 11

# ── Freshness guard ──────────────────────────────────────────────────────────
DB_FRESHNESS_HOURS = 8

# ── Parlay builder defaults ──────────────────────────────────────────────────
PARLAY_MIN_SCORE   = 70.0
PARLAY_MAX_SAME_TEAM = 2
PARLAY_CORR_SAME_TEAM = 0.90
PARLAY_CORR_SAME_GAME = 0.95

# ── O0.5 Any Hit: final composite weights (sum = 1.00) ───────────────────────
O05_WEIGHTS = {
    "pitcher": 0.35,   # K%-dominant SP vulnerability (K = guaranteed no-hit)
    "batter":  0.30,   # contact-first batter score (K-avoidance #1)
    "platoon": 0.10,   # handedness: batter vs SP
    "lineup":  0.10,   # more AB = more chances for a hit
    "vegas":   0.07,   # team total → scoring context / PA frequency
    "streak":  0.05,   # recent form
    "park":    0.03,   # contact park factor (minimal for O0.5)
}

O05_OFFSET_FULL  = 5.0    # Savant + FanGraphs data present
O05_OFFSET_PROXY = 7.5    # MLB API only / stale disk cache

O05_TIERS_FULL  = {"🔒 TIER 1": 80, "✅ TIER 2": 70, "📊 TIER 3": 60, "❌ NO PLAY": 0}
O05_TIERS_PROXY = {"🔒 TIER 1": 75, "✅ TIER 2": 65, "📊 TIER 3": 55, "❌ NO PLAY": 0}

# ── O0.5 probability calibration (score → P(any hit)) ────────────────────────
# Higher floor/ceiling than TB: league avg batter gets ≥1 hit ~65% of games.
O05_PROB_SLOPE    = 0.07   # logistic curve steepness
O05_PROB_MIDPOINT = 55     # score where P ≈ base rate (~68%)
O05_PROB_MIN      = 0.52   # floor probability (~55%: weakest hitters)
O05_PROB_MAX      = 0.85   # ceiling probability (~85%: elite contact hitters)

# ── Pitcher K Props: SP-level strikeout prop weights (sum = 1.00) ─────────────
K_WEIGHTS = {
    "sp_k":       0.40,   # SP K% (direct strikeout rate)
    "sp_swstr":   0.20,   # SP SwStr% (real from Savant; proxy K%*0.49 if absent)
    "opp_lineup": 0.25,   # opposing lineup average K% (how strikeout-prone)
    "context":    0.15,   # game total (low total = pitcher's duel) + umpire zone
}

K_TIERS = {"⚡ ELITE K": 80, "🔥 STRONG K": 70, "📊 LEAN K": 60, "➖ NO PLAY": 0}

# ── K prop probability calibration (general K-upside; not line-specific) ──────
K_PROB_SLOPE    = 0.08
K_PROB_MIDPOINT = 65
K_PROB_MIN      = 0.45
K_PROB_MAX      = 0.75

# ── Home Run prop: tier thresholds ────────────────────────────────────────────
HR_TIERS = {"💣 ELITE HR": 80, "🔥 STRONG HR": 70, "📊 LEAN HR": 60, "➖ NO PLAY": 0}

# ── HR probability calibration (score → P(batter hits HR today)) ──────────────
# Lower range than TB/O0.5: league-avg power hitter hits HR ~11-13% of games.
# Elite (Judge/Alonso) in Coors+wind: ~20-22%.
HR_PROB_SLOPE    = 0.09
HR_PROB_MIDPOINT = 65    # score where P ≈ 15% (qualifying power hitter)
HR_PROB_MIN      = 0.08  # floor: weakest qualifying power hitter
HR_PROB_MAX      = 0.22  # ceiling: elite power in best park/wind conditions

# ── Moneyline: edge thresholds (as % points above vig-removed market implied) ──
ML_EDGE_STRONG = 7.0    # ≥7% model edge → Strong Edge
ML_EDGE_LEAN   = 4.0    # ≥4% model edge → Lean (min threshold to flag)

# ── Moneyline: bettable gate ──────────────────────────────────────────────────
ML_MIN_BATTERS_PER_TEAM = 5   # both teams need ≥5 batters with measured wRC+
