"""scoring/hr.py — HR upside score (separate from O1.5 TB batter score)."""

from typing import Dict


def compute_hr_score(
    barrel_rate: float,
    sweet_spot: float,
    park_hr_factor: float,
    implied_total: float,
    weather: Dict,
    hard_hit: float = 0.37,
    exit_velocity: float = 88.5,
    iso: float = 0.165,
    ev50: float = 95.0,
    bat_speed: float = 71.0,
    blast_rate: float = 0.21,
    pitch_matchup_score: float = 50.0,
) -> float:
    """
    HR upside score 0–100. V2.1: no fabrication — bat-tracking weight
    redistributes to barrel% when signals are absent.

    Weights (research-backed):
      Barrel%     35%+ — r=0.93 with HR rate (#1 predictor)
      Park factor 15%  — Coors/GABP vs pitcher's parks
      EV50        10%  — hardest 50% EV (only when measured)
      Bat speed    8%  — mechanical HR ceiling (only when measured)
      Hard hit%    8%  — contact quality
      ISO          7%  — raw power
      Blast rate   6%  — fast + squared-up (only when measured)
      Vegas        5%  — high-total games = more HR opps
      Pitch match  4%  — favorable pitch-type RV
      Wind/weather — dynamic
    """
    has_ev50      = ev50 >= 50
    has_bat_speed = bat_speed >= 30
    has_blast     = blast_rate >= 0.01

    _ev50_w   = 0.10 if has_ev50      else 0.0
    _speed_w  = 0.08 if has_bat_speed else 0.0
    _blast_w  = 0.06 if has_blast     else 0.0
    _barrel_w = 0.35 + (0.10 - _ev50_w) + (0.08 - _speed_w) + (0.06 - _blast_w)

    barrel_score    = max(0, min(100, barrel_rate / 0.20 * 100))
    ev50_score      = max(0, min(100, (ev50       - 88.0) / (104.0 - 88.0) * 100))
    bat_speed_score = max(0, min(100, (bat_speed  - 65.0) / (78.0  - 65.0) * 100))
    blast_score     = max(0, min(100, (blast_rate - 0.10) / (0.35  - 0.10) * 100))
    hh_score        = max(0, min(100, (hard_hit   - 0.28) / (0.56  - 0.28) * 100))
    iso_score       = max(0, min(100, (iso        - 0.080)/ (0.320 - 0.080)* 100))
    park_score      = max(0, min(100, (park_hr_factor - 0.85) / (1.35 - 0.85) * 100))
    vegas_score     = max(0, min(100, (implied_total - 3.0) / (6.5 - 3.0) * 100)) if implied_total > 0 else 40.0
    matchup_score   = max(0, min(100, pitch_matchup_score))

    wind_raw = 0.0
    wind_weight = 0.0
    temp_adj = 0.0

    if not weather.get("is_dome"):
        effect = weather.get("wind_effect", "neutral")
        speed  = float(weather.get("wind_speed", 0))
        temp   = float(weather.get("temperature", 70))
        if speed >= 8:
            speed_factor = min(1.0, (speed - 8.0) / (25.0 - 8.0))
            if effect == "strong_out":
                wind_raw    = 80.0 * speed_factor
                wind_weight = 0.08 + 0.07 * speed_factor
            elif effect == "out":
                wind_raw    = 55.0 * speed_factor
                wind_weight = 0.05 + 0.07 * speed_factor
            elif effect == "in":
                wind_raw    = -70.0 * speed_factor
                wind_weight = 0.05 + 0.07 * speed_factor
        if temp < 45:
            temp_adj = -8.0
        elif temp < 55:
            temp_adj = -4.0
        elif temp > 92:
            temp_adj = +7.0
        elif temp > 85:
            temp_adj = +4.0

    adjusted_park_weight = max(0.06, 0.15 - wind_weight)

    base = (
        barrel_score  * _barrel_w +
        park_score    * adjusted_park_weight +
        hh_score      * 0.08 +
        iso_score     * 0.07 +
        vegas_score   * 0.05 +
        matchup_score * 0.04 +
        (ev50_score      * _ev50_w  if has_ev50      else 0) +
        (bat_speed_score * _speed_w if has_bat_speed else 0) +
        (blast_score     * _blast_w if has_blast     else 0)
    )

    wind_contribution = wind_raw * wind_weight if wind_weight > 0 else 0.0
    composite = base + wind_contribution + temp_adj
    return max(0, min(100, round(composite, 1)))
