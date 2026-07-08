"""
dfs/lib/dk_rules.py — DraftKings MLB Classic roster rules and scoring.

Scoring verified against DraftKings' official MLB scoring page:
  https://www.draftkings.com/help/rules/4/21

Hitter scoring (verified):
  Single   =  3 pts
  Double   =  5 pts
  Triple   =  8 pts
  Home Run = 10 pts
  RBI      =  2 pts
  Run      =  2 pts
  BB       =  2 pts
  HBP      =  2 pts
  SB       =  5 pts

Pitcher scoring (verified):
  IP      =  2.25 pts per IP
  K       =  2 pts
  Win     =  4 pts
  ER      = -2 pts
  H       = -0.6 pts
  BB      = -0.6 pts
  HBP     = -0.6 pts
  CG      =  2.5 pts bonus
  CGSO    =  2.5 pts bonus (on top of CG)
  NH      =  5 pts bonus (no-hitter)

  Note: DK does NOT award QS bonus (FD does). Pitcher is SP-heavy.

Roster (MLB Classic, 10 spots):
  SP  × 2   (can use P slot for SP or RP)
  C   × 1
  1B  × 1
  2B  × 1
  3B  × 1
  SS  × 1
  OF  × 3
  FLEX × 1  (any hitter position, no P)

Salary cap: $50,000
"""

from __future__ import annotations
from typing import Dict, Optional

# ── Scoring tables ─────────────────────────────────────────────────────────────
DK_HITTER_SCORING: Dict[str, float] = {
    "single":  3.0,
    "double":  5.0,
    "triple":  8.0,
    "hr":     10.0,
    "rbi":     2.0,
    "run":     2.0,
    "bb":      2.0,
    "hbp":     2.0,
    "sb":      5.0,
}

DK_PITCHER_SCORING: Dict[str, float] = {
    "ip":      2.25,   # per inning pitched
    "k":       2.0,
    "win":     4.0,
    "er":     -2.0,
    "h":      -0.6,
    "bb":     -0.6,
    "hbp":    -0.6,
    "cg":      2.5,
    "cgso":    2.5,    # additional on top of cg
    "nh":      5.0,
}

# ── Roster config ──────────────────────────────────────────────────────────────
DK_LINEUP_SLOTS = {
    "SP":   2,
    "C":    1,
    "1B":   1,
    "2B":   1,
    "3B":   1,
    "SS":   1,
    "OF":   3,
    "FLEX": 1,   # any hitter, no P
}

DK_SALARY_CAP = 50_000

# pydfs position names for DK MLB (how pydfs expects positions)
DK_PYDFS_POSITIONS = {
    "SP": "SP", "RP": "RP",
    "C": "C", "1B": "1B", "2B": "2B", "3B": "3B", "SS": "SS",
    "OF": "OF", "LF": "OF", "CF": "OF", "RF": "OF",
    "DH": "1B",   # DH typically fills 1B/UTIL slot
}


def compute_dk_projection(p: Dict) -> Optional[float]:
    """
    Compute DK projected fantasy points for a hitter from a plays dict.
    Returns None if required inputs are absent.
    Uses same PA-estimate and rate-to-outcome math as compute_fd_projection,
    applied to DK's point values.
    """
    try:
        k_rate    = float(p.get("k_rate")    or 0.228)
        bb_rate   = float(p.get("bb_rate")   or 0.082)
        woba      = float(p.get("xslg")      or 0.398) * 0.85   # rough AVG proxy
        iso       = float(p.get("iso")       or 0.165)
        barrel    = float(p.get("barrel_rate") or 0.070)
        slot      = int(p.get("lineup_slot") or 5)
        implied   = float(p.get("implied_total") or 0.0)
        park      = p.get("park", "")
        weather   = p.get("weather") or {}
    except (TypeError, ValueError):
        return None

    # PA estimate by slot (same as FD)
    pa_by_slot = {1:4.8,2:4.7,3:4.6,4:4.5,5:4.3,6:4.2,7:4.1,8:3.9,9:3.8}
    est_pa = pa_by_slot.get(slot, 4.2)
    if implied > 0:
        est_pa += (implied - 4.5) * 0.08

    # Hit types
    hit_rate      = max(0.180, woba)
    hr_per_pa     = barrel * 0.35
    xbh_per_pa    = (iso * 0.6) * (1 - k_rate)
    single_per_pa = max(0, hit_rate - hr_per_pa - xbh_per_pa)
    sb_rate       = 0.05

    # Park / weather adjustments (import from lib.constants)
    try:
        from lib.constants import PARK_HR_FACTORS, PARK_TB_FACTORS
        hr_per_pa     *= PARK_HR_FACTORS.get(park, 1.0)
        single_per_pa *= PARK_TB_FACTORS.get(park, 1.0)
    except ImportError:
        pass

    if not weather.get("is_dome"):
        we = weather.get("wind_effect", "neutral")
        if we == "strong_out": hr_per_pa *= 1.25
        elif we == "out":      hr_per_pa *= 1.15
        elif we == "in":       hr_per_pa *= 0.80

    # SP quality adjustment
    pitcher = p.get("sub_pitcher", 50.0)
    q_adj   = 0.85 + (pitcher / 100.0) * 0.30   # range ~0.85–1.15

    proj_singles = single_per_pa * est_pa * q_adj
    proj_hr      = hr_per_pa     * est_pa * q_adj
    proj_xbh     = xbh_per_pa    * est_pa * q_adj
    proj_doubles = proj_xbh * 0.65
    proj_triples = proj_xbh * 0.05
    proj_bb      = bb_rate  * est_pa
    proj_sb      = sb_rate  * est_pa
    proj_hits    = proj_singles + proj_doubles + proj_triples + proj_hr

    rbi_rate = 0.32 if slot <= 4 else 0.22
    run_rate = 0.38 if slot <= 3 else (0.28 if slot <= 6 else 0.20)
    if implied > 0:
        rbi_rate *= implied / 4.5
        run_rate *= implied / 4.5

    proj_rbi  = proj_hits * rbi_rate + proj_hr
    proj_runs = proj_hits * run_rate + proj_hr

    dk_pts = (
        proj_singles * DK_HITTER_SCORING["single"] +
        proj_doubles * DK_HITTER_SCORING["double"] +
        proj_triples * DK_HITTER_SCORING["triple"] +
        proj_hr      * DK_HITTER_SCORING["hr"] +
        proj_rbi     * DK_HITTER_SCORING["rbi"] +
        proj_runs    * DK_HITTER_SCORING["run"] +
        proj_bb      * DK_HITTER_SCORING["bb"] +
        proj_sb      * DK_HITTER_SCORING["sb"]
    )
    return round(dk_pts, 1)


def dk_salary_csv_to_players(rows):
    """
    Normalize DraftKings salary CSV rows (list of dicts from csv.DictReader)
    into the format consumed by dfs/sources/salaries.py.

    DK CSV headers (MLB Classic):
      Position, Name + ID, Name, ID, Roster Position, Salary, Game Info, TeamAbbrev, AvgPointsPerGame
    """
    out = []
    for row in rows:
        name = row.get("Name") or row.get("Name + ID", "").split("(")[0].strip()
        if not name:
            continue
        try:
            salary = int(str(row.get("Salary", "0")).replace(",", "").replace("$", ""))
        except (ValueError, TypeError):
            salary = 0
        pos_raw  = row.get("Position") or row.get("Roster Position", "UTIL")
        pos_norm = DK_PYDFS_POSITIONS.get(pos_raw.strip().upper(), pos_raw.strip().upper())
        out.append({
            "name":     name.strip(),
            "position": pos_norm,
            "salary":   salary,
            "team":     (row.get("TeamAbbrev") or row.get("Team", "")).strip().upper(),
            "game":     row.get("Game Info", ""),
            "avg_pts":  float(row.get("AvgPointsPerGame") or 0.0),
            "site":     "dk",
        })
    return out
