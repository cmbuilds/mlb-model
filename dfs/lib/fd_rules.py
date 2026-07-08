"""
dfs/lib/fd_rules.py — FanDuel MLB Classic roster rules and scoring.

FD scoring verified from:
  https://www.fanduel.com/rules  (MLB section)

Hitter scoring:
  Single   = 3 pts    Double  = 6 pts    Triple  = 9 pts    HR = 12 pts
  RBI      = 3.5 pts  Run     = 3.2 pts  BB      = 3 pts
  HBP      = 3 pts    SB      = 6 pts

Pitcher scoring:
  IP       = 3 pts/IP    K = 3 pts     Win = 6 pts    QS = 4 pts
  ER       = -3 pts      H = -0.6 pts  BB  = -0.6 pts
  CG       = 3 pts bonus CGSO = 3 pts additional   NH = 6 pts

Roster (MLB, 9 spots):
  P  × 1   C/1B × 1   2B × 1   3B × 1   SS × 1   OF × 3   UTIL × 1
Salary cap: $35,000
"""

from __future__ import annotations
from typing import Dict, Optional

# ── Hitter scoring ─────────────────────────────────────────────────────────────
FD_HITTER_SCORING: Dict[str, float] = {
    "single":  3.0,
    "double":  6.0,
    "triple":  9.0,
    "hr":     12.0,
    "rbi":     3.5,
    "run":     3.2,
    "bb":      3.0,
    "hbp":     3.0,
    "sb":      6.0,
}

# ── Pitcher scoring ─────────────────────────────────────────────────────────────
FD_PITCHER_SCORING: Dict[str, float] = {
    "ip":      3.0,    # per inning pitched
    "k":       3.0,
    "win":     6.0,
    "qs":      4.0,    # quality start bonus (FD only; DK doesn't award QS)
    "er":     -3.0,
    "h":      -0.6,
    "bb":     -0.6,
    "cg":      3.0,
    "cgso":    3.0,    # additional on top of cg
    "nh":      6.0,
}

FD_SALARY_CAP = 35_000


def compute_fd_pitcher_projection(p: Dict) -> Optional[Dict]:
    """
    Project FD fantasy points for a SP from a batter play dict (batter-centric
    view of the opposing SP). Groups of batter plays for the same SP are averaged
    by the caller; this function processes aggregated SP stats.

    Expected keys in p (all optional with safe defaults):
      _pitcher_k_rate    : SP K rate allowed (float 0-1)
      _pitcher_fip       : SP FIP (float, ~2.0-6.0)
      _pitcher_whip      : SP WHIP (float)
      implied_total      : batting team's implied total (SP's opponent)
      _pitcher_prov      : provenance dict

    Returns dict with fd_pts, ceiling, floor, win_prob, qs_prob, proj_ip,
    or None if inputs are insufficient.
    """
    try:
        k_rate = float(p.get("_pitcher_k_rate") or 0.228)
        fip    = float(p.get("_pitcher_fip")    or p.get("sp_fip", 4.10))
        whip   = float(p.get("_pitcher_whip")   or 1.30)
        # implied_total here is the BATTER team's implied (= SP's opponent's runs)
        opp_implied = float(p.get("implied_total") or 4.5)
        if opp_implied < 0.5:
            opp_implied = 4.5
    except (TypeError, ValueError):
        return None

    # IP estimate from FIP
    if fip < 3.0:   proj_ip = 6.5
    elif fip < 3.5: proj_ip = 6.2
    elif fip < 4.0: proj_ip = 5.8
    elif fip < 4.5: proj_ip = 5.4
    else:           proj_ip = 4.8

    # Adjust for opponent run environment
    proj_ip = proj_ip * max(0.85, 1.0 - (opp_implied - 4.5) * 0.04)

    bf       = proj_ip * 3.3
    proj_k   = k_rate * bf
    proj_er  = max(0.0, (fip / 9.0) * proj_ip)
    proj_hbb = whip * proj_ip
    proj_h   = proj_hbb * 0.70
    proj_bb  = proj_hbb * 0.30

    win_prob = 0.50
    if fip < 3.5 and opp_implied < 4.0:
        win_prob = 0.65
    elif fip > 4.5 or opp_implied > 5.0:
        win_prob = 0.35

    qs_prob = 0.70 if proj_ip >= 5.8 and proj_er <= 3.0 else 0.35

    fd_pts = (
        proj_k   * FD_PITCHER_SCORING["k"] +
        proj_ip  * FD_PITCHER_SCORING["ip"] +
        proj_er  * FD_PITCHER_SCORING["er"] +
        proj_h   * FD_PITCHER_SCORING["h"] +
        proj_bb  * FD_PITCHER_SCORING["bb"] +
        win_prob * FD_PITCHER_SCORING["win"] +
        qs_prob  * FD_PITCHER_SCORING["qs"]
    )

    variance = fd_pts * 0.40
    ceiling  = round(fd_pts + variance, 1)
    floor    = round(max(0.0, fd_pts - variance * 0.6), 1)

    return {
        "fd_pts":    round(fd_pts, 1),
        "ceiling":   ceiling,
        "floor":     floor,
        "win_prob":  round(win_prob, 2),
        "qs_prob":   round(qs_prob, 2),
        "proj_ip":   round(proj_ip, 1),
        "proj_k":    round(proj_k, 1),
    }
