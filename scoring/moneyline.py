"""scoring/moneyline.py — Pure win-probability model for MLB games.

No Streamlit imports. No network calls. No market comparison.
The only job: given our data about both teams, return a win probability.

Formula (approved 2026-07-08):
    home_win_pct = HFA + W_SP×sp_edge + W_OFF×off_edge + W_BP×bp_edge
    clamped to [0.15, 0.85]

Weights are PROVISIONAL — tune to calibration after backtest runs.
"""

from typing import Dict, List, Tuple


# ── Model constants (provisional — tune with backtest) ────────────────────────
HFA        = 0.535   # home-field baseline: MLB historical home win rate
W_SP       = 0.40    # SP is the dominant single-game input
W_OFF      = 0.20    # team run-scoring ability (wRC+ lineup average)
W_BP       = 0.10    # bullpen quality
CLAMP_LOW  = 0.15
CLAMP_HIGH = 0.85


def compute_win_probability(
    home_sp_vuln:  float,    # pure SP vulnerability 0–100 (from compute_sp_vuln_pure)
    away_sp_vuln:  float,
    home_wrc_plus: float,    # avg measured wRC+ for home lineup
    away_wrc_plus: float,
    home_bp_vuln:  float,    # per-team bullpen vulnerability 0–100
    away_bp_vuln:  float,
) -> Tuple[float, Dict]:
    """
    Return (home_win_prob, drivers_dict).

    home_win_prob: probability home team wins, clamped to [CLAMP_LOW, CLAMP_HIGH].
    drivers_dict:  per-component breakdown for display, all in probability-point units.

    SP edge: positive = home SP better (lower vuln). Range ±W_SP.
    Off edge: positive = home offense better (higher wRC+). Range ±W_OFF.
    BP edge: positive = home bullpen better (lower vuln). Range ±W_BP.
    """
    # Positive values = home team advantage on this dimension
    sp_edge  = (away_sp_vuln  - home_sp_vuln)  / 100.0
    off_edge = (home_wrc_plus - away_wrc_plus) / 100.0
    bp_edge  = (away_bp_vuln  - home_bp_vuln)  / 100.0

    sp_pts  = round(W_SP  * sp_edge,  4)
    off_pts = round(W_OFF * off_edge, 4)
    bp_pts  = round(W_BP  * bp_edge,  4)

    raw = HFA + sp_pts + off_pts + bp_pts
    home_win_prob = max(CLAMP_LOW, min(CLAMP_HIGH, raw))

    drivers = {
        "sp_edge_pts":   sp_pts,
        "off_edge_pts":  off_pts,
        "bp_edge_pts":   bp_pts,
        "hfa":           HFA,
        "raw":           round(raw, 4),
        "home_sp_vuln":  round(home_sp_vuln, 1),
        "away_sp_vuln":  round(away_sp_vuln, 1),
        "home_wrc":      round(home_wrc_plus, 1),
        "away_wrc":      round(away_wrc_plus, 1),
        "home_bp_vuln":  round(home_bp_vuln, 1),
        "away_bp_vuln":  round(away_bp_vuln, 1),
        "clamped":       home_win_prob != raw,
    }
    return round(home_win_prob, 4), drivers


def compute_team_offense_score(plays: List[Dict], team: str) -> Tuple[float, int]:
    """
    Aggregate wRC+ for a team's confirmed lineup.
    Returns (avg_wrc_plus, n_batters_with_measured_wrc).
    Zero wRC+ = league-avg fallback, not a real measurement — excluded from count.
    """
    tp = [p for p in plays if p.get("team", "") == team]
    vals = [float(p.get("wrc_plus", 0)) for p in tp
            if p.get("wrc_plus") and float(p.get("wrc_plus", 0)) > 0]
    if not vals:
        return 100.0, 0
    return round(sum(vals) / len(vals), 1), len(vals)
