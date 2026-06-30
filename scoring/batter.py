"""scoring/batter.py — Batter quality sub-score for O1.5 TB props."""

from typing import Dict, Tuple


def compute_batter_score(statcast: Dict, fg_stats: Dict = None) -> Tuple[float, str, Dict]:
    """
    Batter profile sub-score 0–100.
    League avg = ~50. Elite (Judge/Soto tier) = 85+.
    Research-calibrated for O1.5 TB prediction (not HR rate).
    """
    details: Dict = {}

    def f(key, default):
        try: return float(statcast.get(key, default))
        except Exception: return float(default)

    xslg        = f("slg_proxy",     0.398)
    barrel_rate = f("barrel_rate",   0.070)
    hard_hit    = f("hard_hit_rate", 0.370)
    k_rate      = f("k_rate",        0.228)
    iso         = f("iso_proxy",     0.165)
    wrc_plus    = f("wrc_plus",      100.0)
    ev50_raw    = f("ev50",          0.0)
    bat_speed_raw = f("bat_speed",   0.0)
    blast_raw   = f("blast_rate",    0.0)

    details["xSLG"]     = round(xslg, 3)
    details["Barrel%"]  = f"{barrel_rate*100:.1f}%"
    details["HardHit%"] = f"{hard_hit*100:.1f}%"
    details["K%"]       = f"{k_rate*100:.1f}%"
    details["ISO"]      = round(iso, 3)
    details["wRC+"]     = int(wrc_plus)
    if ev50_raw >= 50:
        details["EV50"]   = round(ev50_raw, 1)
    if bat_speed_raw >= 30:
        details["BatSpd"] = round(bat_speed_raw, 1)
    if blast_raw >= 0.01:
        details["Blast%"] = f"{blast_raw*100:.1f}%"

    # Z-score sub-scores (league avg batter = 50 on each)
    xslg_score     = max(0, min(100, 50 + (xslg - 0.398)      / 0.080 * 25))
    wrc_score      = max(0, min(100, 50 + (wrc_plus - 100)    / 35.0  * 25))
    barrel_score   = max(0, min(100, 50 + (barrel_rate - 0.070)/ 0.040 * 25))
    hard_hit_score = max(0, min(100, 50 + (hard_hit - 0.370)  / 0.055 * 25))
    k_score        = max(0, min(100, 50 + (0.228 - k_rate)     / 0.060 * 25))
    iso_score      = max(0, min(100, 50 + (iso - 0.165)        / 0.065 * 25))

    woba_raw = float(statcast.get("woba", 0.0) or 0.0)
    if woba_raw < 0.200:
        woba_raw = max(0.240, min(0.420, 0.245 + xslg * 0.22 - k_rate * 0.15))
    woba_score = max(0, min(100, 50 + (woba_raw - 0.315) / 0.040 * 25))
    details["wOBA"] = f"{woba_raw:.3f}"

    has_bat_tracking = ev50_raw >= 50 and bat_speed_raw >= 30 and blast_raw >= 0.01

    if has_bat_tracking:
        ev50_score      = max(0, min(100, 50 + (ev50_raw      - 95.0) / 3.0   * 25))
        bat_speed_score = max(0, min(100, 50 + (bat_speed_raw - 71.0) / 3.0   * 25))
        blast_score     = max(0, min(100, 50 + (blast_raw     - 0.21) / 0.050 * 25))
        composite = (
            k_score         * 0.18 +
            woba_score      * 0.16 +
            xslg_score      * 0.14 +
            hard_hit_score  * 0.12 +
            wrc_score       * 0.10 +
            ev50_score      * 0.10 +
            blast_score     * 0.08 +
            bat_speed_score * 0.06 +
            barrel_score    * 0.04 +
            iso_score       * 0.02
        )
    else:
        composite = (
            k_score        * 0.24 +
            woba_score     * 0.20 +
            xslg_score     * 0.18 +
            hard_hit_score * 0.16 +
            wrc_score      * 0.12 +
            barrel_score   * 0.06 +
            iso_score      * 0.04
        )

    return max(0, min(100, composite)), "Contact quality", details
