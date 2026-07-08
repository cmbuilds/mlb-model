"""scoring/weather.py — Wind classification and weather sub-score."""

from typing import Dict, Optional, Tuple


def classify_wind(
    direction: float, speed: float, park_team: Optional[str] = None
) -> Tuple[str, str]:
    """Classify wind: returns (direction_label, effect). Effect: strong_out/out/in/neutral.

    When park_team is supplied, wind effect is relative to that park's CF bearing
    from lib.constants.CF_BEARINGS (fail-safe: falls back to fixed range if absent).
    """
    if speed < 8:
        return "Calm", "neutral"
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    label = dirs[int((direction + 22.5) / 45) % 8]

    cf_bearing: Optional[float] = None
    if park_team:
        try:
            from lib.constants import CF_BEARINGS
            cf_bearing = CF_BEARINGS.get(park_team)
        except Exception:
            cf_bearing = None

    if cf_bearing is not None:
        # Wind "out" = blowing toward CF (from home plate direction = cf_bearing ± 45°)
        # Wind "in"  = blowing from CF toward home plate (opposite direction ± 45°)
        diff = (direction - cf_bearing + 360) % 360
        if diff <= 45 or diff >= 315:
            effect = "strong_out" if speed >= 12 else "out"
        elif 135 <= diff <= 225:
            effect = "in" if speed >= 10 else "neutral"
        else:
            effect = "neutral"
    else:
        # Park-agnostic fallback (used only when park_team is absent or not in CF_BEARINGS)
        if 157.5 <= direction <= 292.5:
            effect = "strong_out" if speed >= 12 else "out"
        elif direction <= 67.5 or direction >= 337.5:
            effect = "in" if speed >= 10 else "neutral"
        else:
            effect = "neutral"

    return label, effect


def compute_weather_score(weather: Dict) -> Tuple[float, str]:
    """Weather sub-score 0–100. Wind out = boost, wind in = suppress, dome = neutral."""
    if not weather or weather.get("is_dome"):
        return 50.0, "🏟️ Dome"
    score = 50.0
    notes = []
    wind_effect = weather.get("wind_effect", "neutral")
    wind_speed  = weather.get("wind_speed", 0)
    temp        = weather.get("temperature", 70)
    wind_label  = weather.get("wind_dir_label", "")
    if wind_effect == "strong_out":
        score += 25
        notes.append(f"💨 {wind_speed}mph Out (+25)")
    elif wind_effect == "out":
        score += 15
        notes.append(f"💨 {wind_speed}mph Out (+15)")
    elif wind_effect == "in":
        score -= 20
        notes.append(f"💨 {wind_speed}mph In (-20)")
    else:
        notes.append(f"💨 {wind_speed}mph {wind_label}" if wind_speed else "💨 Calm")
    if temp < 50:
        adj = max(-15, -8 * (50 - temp) / 10)
        score += adj
        notes.append(f"🌡️ {temp:.0f}°F Cold ({adj:.0f})")
    elif temp > 83:
        adj = min(10, 5 * (temp - 83) / 10)
        score += adj
        notes.append(f"🌡️ {temp:.0f}°F Hot (+{adj:.0f})")
    else:
        notes.append(f"🌡️ {temp:.0f}°F")
    return max(0, min(100, score)), " | ".join(notes)
