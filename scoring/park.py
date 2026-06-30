"""scoring/park.py — Park factor, platoon, lineup, and pitch-matchup scores."""

import math
from typing import Dict, Tuple

from lib.constants import PARK_TB_FACTORS, PARK_HR_FACTORS, STADIUM_COORDS


def compute_park_score(team: str, is_home: bool) -> Tuple[float, str]:
    """Compute park factor sub-score 0–100."""
    tb_factor = PARK_TB_FACTORS.get(team, 1.00)
    hr_factor = PARK_HR_FACTORS.get(team, 1.00)
    composite = tb_factor * 0.4 + hr_factor * 0.6
    if team == "COL":
        composite = 1.30
    score = (composite - 0.80) / (1.30 - 0.80) * 100
    score = max(0, min(100, score))
    park_name = STADIUM_COORDS.get(team, (0, 0, "Unknown", False))[2]
    flag = "🏟️" if STADIUM_COORDS.get(team, (0, 0, "", False))[3] else ""
    return score, f"{flag}{park_name} ({tb_factor:.2f}x TB | {hr_factor:.2f}x HR)"


def compute_platoon_score(batter_hand: str, pitcher_hand: str) -> Tuple[float, str]:
    """Compute platoon matchup sub-score 0–100. Handles L/R/B/S batter codes."""
    bh = batter_hand.upper() if batter_hand else "R"
    ph = pitcher_hand.upper() if pitcher_hand else "R"
    if bh in ("B", "S"):
        if ph == "R":
            return 75.0, "Switch hitter vs RHP (bats L, +56 SLG)"
        elif ph == "L":
            return 65.0, "Switch hitter vs LHP (bats R, +33 SLG)"
        return 60.0, "Switch hitter (platoon adv)"
    if bh == "L" and ph == "R":
        return 75.0, "LHB vs RHP (+56 SLG)"
    if bh == "R" and ph == "L":
        return 65.0, "RHB vs LHP (+33 SLG)"
    if bh == "L" and ph == "L":
        return 30.0, "LHB vs LHP (-35 wOBA)"
    return 50.0, "RHB vs RHP (neutral)"


def compute_lineup_score(lineup_slot: int) -> Tuple[float, str]:
    """Sub-score 0–100 for lineup position expected PA."""
    pa_by_slot = {1: 4.8, 2: 4.7, 3: 4.6, 4: 4.5, 5: 4.3,
                  6: 4.2, 7: 4.1, 8: 3.9, 9: 3.8}
    expected_pa = pa_by_slot.get(lineup_slot, 4.2)
    score = (expected_pa - 3.8) / (4.8 - 3.8) * 80 + 20
    slot_labels = {1: "Leadoff", 2: "2-Hole", 3: "3-Hole", 4: "Cleanup",
                   5: "5th", 6: "6th", 7: "7th", 8: "8th", 9: "9th"}
    bonus = " ⭐" if lineup_slot <= 4 else ""
    return score, f"#{lineup_slot} {slot_labels.get(lineup_slot, str(lineup_slot))}{bonus} ({expected_pa:.1f} PA/g)"


def compute_pitch_matchup_score(batter_stats: Dict, pitcher_stats: Dict) -> Tuple[float, str]:
    """
    Pitch-type matchup score 0–100.
    Crosses pitcher's arsenal mix with batter's run value vs each pitch type.
    """
    PITCH_TYPES = ("FF", "SI", "SL", "CH", "CU", "FC")
    LEAGUE_AVG_RV = {"FF": 0.0, "SI": -0.1, "SL": -0.2, "CH": -0.1, "CU": -0.2, "FC": -0.1}
    try:
        total_pct = sum(float(pitcher_stats.get(f"pct_{pt}", 0) or 0) for pt in PITCH_TYPES)
        if total_pct < 0.01:
            return 50.0, "Pitch mix: no data"
        weighted_rv = 0.0
        coverage = 0.0
        pitch_details = []
        for pt in PITCH_TYPES:
            pct = float(pitcher_stats.get(f"pct_{pt}", 0) or 0)
            if pct < 0.01:
                continue
            pct_norm = pct / total_pct
            rv = float(batter_stats.get(f"rv_vs_{pt}", LEAGUE_AVG_RV.get(pt, 0.0)))
            rv_relative = rv - LEAGUE_AVG_RV.get(pt, 0.0)
            weighted_rv += pct_norm * rv_relative
            coverage += pct_norm
            if pct_norm > 0.15:
                sign = "+" if rv > 0 else ""
                pitch_details.append(f"{pt}({pct_norm*100:.0f}%): {sign}{rv:.2f}")
        score = 50.0 + (weighted_rv / 1.5) * 40.0
        score = max(10.0, min(90.0, score))
        if coverage < 0.5:
            score = score * (coverage / 0.5) + 50.0 * (1 - coverage / 0.5)
        label = "Pitch mix: " + " | ".join(pitch_details) if pitch_details else "Pitch mix: avg splits"
        return round(score, 1), label
    except Exception:
        return 50.0, "Pitch mix: error"
