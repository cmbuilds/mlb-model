"""scoring/weather.py — Wind classification and weather sub-score."""

from typing import Dict, Tuple


def classify_wind(direction: float, speed: float) -> Tuple[str, str]:
    """Classify wind: returns (direction_label, effect). Effect: strong_out/out/in/neutral."""
    if speed < 8:
        return "Calm", "neutral"
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    label = dirs[int((direction + 22.5) / 45) % 8]
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
