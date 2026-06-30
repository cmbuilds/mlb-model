"""scoring/streak.py — Recent form (streak) and batter-vs-pitcher scores."""

import math
from typing import Dict, Tuple


def compute_streak_score(recent: Dict, season_slg: float = 0.398) -> Tuple[float, str]:
    """
    Convert recent form data into a 0–100 streak score.
    Compares recent TB/game to expected TB/game from season SLG.
    Score 50 = on pace with season average. 70+ = hot. 30- = cold.
    """
    if not recent or recent.get("games", 0) < 3 or recent.get("tb_per_game") is None:
        return 50.0, "Form: no data"
    tb_recent = recent["tb_per_game"]
    g         = recent["games"]
    h         = recent.get("h_last_7", 0)
    ab        = recent.get("ab_last_7", max(1, g * 3))
    hr        = recent.get("hr_last_7", 0)
    # Cap season_slg at 0.420 so elite hitters aren't penalized vs their own ceiling
    baseline_slg = min(season_slg, 0.420)
    season_exp   = max(0.01, baseline_slg * 3.7)
    log_ratio = math.log(max(0.05, tb_recent) / season_exp)
    raw_score = max(10.0, min(90.0, 50 + log_ratio * 36))
    if ab > 0:
        hit_rate = h / ab
        if hit_rate >= 0.350:
            raw_score += 8
        elif hit_rate >= 0.300:
            raw_score += 4
        elif hit_rate < 0.150 and g >= 5:
            raw_score -= 6
    if hr >= 3:
        raw_score += 5
    elif hr >= 2:
        raw_score += 3
    raw_score = max(10.0, min(90.0, raw_score))
    if g < 5:
        weight    = g / 5
        raw_score = raw_score * weight + 50.0 * (1 - weight)
    raw_score = round(raw_score, 1)
    ratio    = tb_recent / season_exp
    hit_rate = h / max(1, ab)
    is_cold  = ratio <= 0.65 and hit_rate < 0.200 and g >= 4
    if ratio >= 1.4 or (hit_rate >= 0.320 and g >= 4):
        label = f"🔥 Hot ({tb_recent:.2f} TB/g last {g}g | {h}/{ab}" + (f" {hr}HR" if hr else "") + ")"
    elif is_cold:
        label = f"❄️ Cold ({tb_recent:.2f} TB/g last {g}g | {h}/{ab})"
    else:
        label = f"{tb_recent:.2f} TB/g last {g}g | {h}/{ab}" + (f" {hr}HR" if hr else "")
    return raw_score, label


def compute_bvp_score(bvp: Dict, batter_slg: float = 0.398) -> Tuple[float, str, str]:
    """
    Career batter-vs-pitcher score 0–100.
    Returns (score, label, significance_flag).
    significance_flag: owns / edge / neutral / dominated / fade / no_data
    """
    ab  = bvp.get("ab", 0) if bvp else 0
    h   = bvp.get("h",  0) if bvp else 0
    hr  = bvp.get("hr", 0) if bvp else 0
    xbh = bvp.get("xbh", 0) if bvp else 0
    so  = bvp.get("so", 0) if bvp else 0
    tb  = bvp.get("tb", 0) if bvp else 0
    if not bvp or ab < 5 or bvp.get("slg") is None:
        if 0 < ab < 5:
            return 50.0, f"BvP: {h}/{ab} ({ab} AB — need 5+)", "no_data"
        return 50.0, "BvP: no history", "no_data"
    career_slg = float(bvp["slg"])
    career_avg = float(bvp.get("avg") or 0)
    slg_delta  = career_slg - batter_slg
    hr_rate    = hr / ab if ab > 0 else 0
    xbh_rate   = xbh / ab if ab > 0 else 0
    k_rate_bvp = so / ab if ab > 0 else 0
    base = max(15.0, min(90.0, 50.0 + slg_delta * 100.0))
    hr_bonus  = min(8.0, hr * 2.5)
    xbh_bonus = min(5.0, xbh_rate * 20.0)
    k_penalty = min(10.0, k_rate_bvp * 25.0) if k_rate_bvp > 0.30 else 0
    score = base + hr_bonus + xbh_bonus - k_penalty
    score = max(10.0, min(95.0, score))
    # Sample size confidence — dampen toward 50 with <10 AB
    if ab < 10:
        confidence = 0.5 + (ab - 5) / 10.0
        score = score * confidence + 50.0 * (1 - confidence)
    score = round(score, 1)
    # Significance flag
    is_owns = (career_avg >= 0.400 or career_slg >= 0.800) and ab >= 10
    is_edge = slg_delta >= 0.150 and career_slg >= 0.450 and ab >= 8
    is_fade = career_slg <= 0.250 and slg_delta <= -0.150 and ab >= 6
    is_dominated = slg_delta <= -0.100 and k_rate_bvp >= 0.35 and ab >= 6
    if is_owns:
        flag = "owns"
        label = f"🔥 OWNS ({h}/{ab} career | AVG {career_avg:.3f} | SLG {career_slg:.3f})"
    elif is_fade:
        flag = "fade"
        label = f"🔴 FADE ({h}/{ab} career | SLG {career_slg:.3f})"
    elif is_dominated:
        flag = "dominated"
        label = f"⚠️ DOMINATED ({h}/{ab} | K%: {k_rate_bvp:.0%})"
    elif is_edge:
        flag = "edge"
        label = f"🟢 EDGE ({h}/{ab} | SLG {career_slg:.3f})"
    else:
        flag = "neutral"
        label = f"BvP: {h}/{ab} | SLG {career_slg:.3f}"
    return score, label, flag


def compute_tto_bonus(lineup_slot: int, sp_ip_estimate: float = 6.0) -> Tuple[float, str]:
    """
    Times Through Order bonus 0–15. Research-backed: 3rd TTO +20 wOBA.
    Estimate TTO from lineup slot and typical SP workload.
    """
    if lineup_slot <= 3:
        return 0.60 * 25, "3rd TTO boost (+20 wOBA)"
    elif lineup_slot <= 6:
        return 0.45 * 25, "2-3rd TTO (~+12 wOBA)"
    else:
        return 0.25 * 25, "2nd TTO (+8 wOBA)"
