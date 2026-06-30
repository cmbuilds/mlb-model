"""
scoring/moneyline.py — Pure win-probability functions for moneyline.

No Streamlit imports. No network calls. Caller resolves all data before calling here.

Extracted from mlb_tb_analyzer.py compute_win_probability (line ~5680) and
display_moneyline_tab._conf (line ~5806) with no logic changes.

Model: modified Log5-style.
  - SP vulnerability drives opposing team's scoring potential.
  - Home-field: +3.5% multiplicative boost to home scoring.
  - Vegas blend: when both implied runs are present, 70% model / 30% Vegas.
  - Run-diff nudge: ±0.02 max based on recent 7-day run differential.
  - Output clamped to [0.30, 0.75] — no absolute locks.
"""

from typing import Dict, List, Optional, Tuple


def compute_team_offense_score(plays: List[Dict], team: str) -> Tuple[float, int]:
    """Aggregate wRC+ for a team's confirmed lineup. Returns (avg_wrc_plus, n_batters)."""
    tp = [p for p in plays if p.get("team", "") == team]
    vals = [p.get("wrc_plus", 100.0) for p in tp if p.get("wrc_plus", 100.0) > 0]
    return (round(sum(vals) / len(vals), 1), len(vals)) if vals else (100.0, 0)


def compute_win_probability(
    home_sp_stats: Dict,
    away_sp_stats: Dict,
    home_off_wrc: float,
    away_off_wrc: float,
    home_bp_vuln: float,
    away_bp_vuln: float,
    home_run_diff: float,
    away_run_diff: float,
    home_implied_runs: float,
    away_implied_runs: float,
) -> Tuple[float, str]:
    """
    Log5-style win probability.

    _sp_vuln in stats dicts: composite pitcher vulnerability (0=unhittable, 100=BP-equivalent).
    Pitcher factor: vuln/50 scales how many runs the opposing offense scores.
    Home field advantage: +3.5% multiplicative on home scoring.
    Vegas blend: 70% model / 30% market when implied runs are present.
    Nudge: capped ±0.02 from recent run differential.

    Returns (home_win_prob 0-1, label).
    """
    home_off = max(0.5, home_off_wrc / 100.0)
    away_off = max(0.5, away_off_wrc / 100.0)

    # Combined pitcher vulnerability: 60% SP + 40% bullpen
    hpv = float(home_sp_stats.get("_sp_vuln", 50.0)) * 0.60 + home_bp_vuln * 0.40
    apv = float(away_sp_stats.get("_sp_vuln", 50.0)) * 0.60 + away_bp_vuln * 0.40

    # away pitcher vuln → scales how much home offense scores
    h_pit_factor = max(0.10, apv / 50.0)
    # home pitcher vuln → scales how much away offense scores
    a_pit_factor = max(0.10, hpv / 50.0)

    hs  = home_off * h_pit_factor * 1.035   # +3.5% home field
    aws = away_off * a_pit_factor
    tot = hs + aws
    if tot <= 0:
        return 0.52, "Log5 zero — home-field default"

    raw   = hs / tot
    nudge = max(-0.02, min(0.02, (home_run_diff - away_run_diff) * 0.01))

    if home_implied_runs > 0 and away_implied_runs > 0:
        ti  = home_implied_runs + away_implied_runs
        vwp = home_implied_runs / ti if ti > 0 else 0.52
        final = raw * 0.70 + vwp * 0.30 + nudge
    else:
        final = raw + nudge

    final = max(0.30, min(0.75, final))

    pqh = "Elite" if hpv < 30 else "Good" if hpv < 45 else "Average" if hpv < 58 else "Weak"
    pqa = "Elite" if apv < 30 else "Good" if apv < 45 else "Average" if apv < 58 else "Weak"
    label = (
        f"Home pit: {pqh} (v={hpv:.0f}) | Away pit: {pqa} (v={apv:.0f}) | "
        f"H wRC+: {home_off_wrc:.0f} | A wRC+: {away_off_wrc:.0f} | "
        f"7d RD H/A: {home_run_diff:+.1f}/{away_run_diff:+.1f}"
    )
    return round(final, 4), label


def compute_ml_confidence(
    side_edge: Optional[float],
    sp_name: Optional[str],
    n_batters: int,
    sp_vuln: float,
    has_odds: bool,
) -> float:
    """
    0-100 confidence score for one ML side.

    Components (max 100):
      - Edge magnitude:  0-50 pts  (edge_pct / 7% × 50; caps at 7%)
      - SP quality:      0-20 pts  ((60 - sp_vuln) / 60 × 20; better SP = more pts)
      - Lineup data:     0-15 pts  (n_batters / 9 × 15)
      - Odds available:  0-10 pts  (binary)
      - SP known:        0-5 pts   (binary)

    Returns 0.0 when edge is None or negative (no value, no confidence).
    """
    if side_edge is None or side_edge < 0:
        return 0.0
    e_pts  = min(50.0, side_edge / 7.0 * 50.0)
    sp_pts = max(0.0, min(20.0, (60.0 - sp_vuln) / 60.0 * 20.0))
    l_pts  = min(15.0, n_batters / 9.0 * 15.0)
    o_pts  = 10.0 if has_odds else 0.0
    s_pts  = 5.0 if sp_name not in ("TBD", "", None) else 0.0
    return round(min(100.0, e_pts + sp_pts + l_pts + o_pts + s_pts), 1)
