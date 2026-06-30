"""scoring/hits.py — Batter and pitcher sub-scores for O0.5 Any Hit props.

Contact-first weighting: K-avoidance dominates since any hit counts.
ISO and barrel% are excluded — power quality is irrelevant for O0.5.
"""

from typing import Dict, Tuple


def compute_hits_batter_score(statcast: Dict) -> Tuple[float, str, Dict]:
    """
    Contact-first batter sub-score 0–100 for O0.5 Any Hit.

    K% receives 35% weight (vs 24% for O1.5 TB) — a strikeout is a
    guaranteed no-hit outcome, making K-avoidance the primary predictor.
    barrel% and ISO excluded: we only need any hit, not extra bases.

    League avg = ~50. Elite contact hitter (Betts/Arraez tier) = 80+.
    """
    details: Dict = {}

    def f(key, default):
        try: return float(statcast.get(key, default))
        except Exception: return float(default)

    k_rate      = f("k_rate",        0.228)
    hard_hit    = f("hard_hit_rate", 0.370)
    wrc_plus    = f("wrc_plus",      100.0)
    ev50_raw    = f("ev50",          0.0)
    bat_speed_raw = f("bat_speed",   0.0)
    blast_raw   = f("blast_rate",    0.0)
    xslg        = f("slg_proxy",     0.398)

    woba_raw = float(statcast.get("woba", 0.0) or 0.0)
    if woba_raw < 0.200:
        woba_raw = max(0.240, min(0.420, 0.245 + xslg * 0.22 - k_rate * 0.15))

    details["K%"]       = f"{k_rate*100:.1f}%"
    details["wOBA"]     = f"{woba_raw:.3f}"
    details["wRC+"]     = int(wrc_plus)
    details["HardHit%"] = f"{hard_hit*100:.1f}%"
    if ev50_raw >= 50:
        details["EV50"]   = round(ev50_raw, 1)
    if bat_speed_raw >= 30:
        details["BatSpd"] = round(bat_speed_raw, 1)
    if blast_raw >= 0.01:
        details["Blast%"] = f"{blast_raw*100:.1f}%"

    k_score        = max(0, min(100, 50 + (0.228 - k_rate)     / 0.060 * 25))
    woba_score     = max(0, min(100, 50 + (woba_raw - 0.315)   / 0.040 * 25))
    wrc_score      = max(0, min(100, 50 + (wrc_plus - 100)     / 35.0  * 25))
    hard_hit_score = max(0, min(100, 50 + (hard_hit - 0.370)   / 0.055 * 25))
    xslg_score     = max(0, min(100, 50 + (xslg - 0.398)       / 0.080 * 25))

    has_bat_tracking = ev50_raw >= 50 and bat_speed_raw >= 30 and blast_raw >= 0.01

    if has_bat_tracking:
        ev50_score      = max(0, min(100, 50 + (ev50_raw      - 95.0) / 3.0   * 25))
        bat_speed_score = max(0, min(100, 50 + (bat_speed_raw - 71.0) / 3.0   * 25))
        blast_score     = max(0, min(100, 50 + (blast_raw     - 0.21) / 0.050 * 25))
        composite = (
            k_score         * 0.28 +
            woba_score      * 0.20 +
            wrc_score       * 0.15 +
            hard_hit_score  * 0.12 +
            ev50_score      * 0.10 +
            bat_speed_score * 0.07 +
            blast_score     * 0.05 +
            xslg_score      * 0.03
        )
    else:
        # No bat tracking — weight wOBA+wRC+ higher; no barrel%/ISO for O0.5
        composite = (
            k_score        * 0.35 +
            woba_score     * 0.25 +
            wrc_score      * 0.20 +
            hard_hit_score * 0.12 +
            xslg_score     * 0.08
        )

    return max(0, min(100, composite)), "Contact quality (O0.5)", details


def compute_hits_pitcher_score(
    statcast: Dict, bullpen_vuln: float = 42.0
) -> Tuple[float, str]:
    """
    Pitcher vulnerability 0–100 for O0.5 Any Hit.

    K% receives 72% weight (vs 40-60% for O1.5 TB): a SP strikeout is a
    direct no-hit outcome for that AB, making K% the dominant signal.
    barrel% removed from formula — extra-base quality less relevant for
    a batter who just needs to reach base safely.

    Blends SP stats (60%) with team bullpen vulnerability (40%).
    High score = SP/BP is vulnerable (good for batter O0.5).
    """
    def f(key, default):
        try: return float(statcast.get(key, default))
        except Exception: return float(default)

    k_rate   = f("k_rate_allowed",   0.220)
    hard_hit = f("hard_hit_allowed", 0.370)
    whip     = f("whip",             1.30)
    era      = f("era",              4.20)
    fip      = f("fip",              4.20)

    # Higher SP K% → lower vulnerability for batter → lower score
    k_vuln    = max(0, min(100, 50 - (k_rate   - 0.220) / 0.050 * 25))
    hh_vuln   = max(0, min(100, 50 + (hard_hit - 0.370) / 0.055 * 25))
    whip_vuln = max(0, min(100, 50 + (whip     - 1.30)  / 0.20  * 25))
    era_use   = fip if fip > 0 else era
    era_vuln  = max(0, min(100, 50 + (era_use  - 4.20)  / 0.80  * 25))

    K_IS_DEFAULT  = abs(k_rate   - 0.220) < 0.003
    HH_IS_DEFAULT = abs(hard_hit - 0.370) < 0.003

    if K_IS_DEFAULT and HH_IS_DEFAULT:
        # No Statcast data: fall back to WHIP + ERA as proxies
        sp_score = whip_vuln * 0.60 + era_vuln * 0.40
        mode_tag = "WHIP/ERA-only"
    else:
        sp_score = (
            k_vuln    * 0.72 +
            hh_vuln   * 0.18 +
            whip_vuln * 0.10
        )
        mode_tag = "full"

    blended = sp_score * 0.60 + float(bullpen_vuln) * 0.40
    label = (
        f"K%: {k_rate*100:.0f}% | WHIP: {whip:.2f} | FIP: {era_use:.2f} | "
        f"BP vuln: {bullpen_vuln:.0f} | mode: {mode_tag}"
    )
    return max(0, min(100, blended)), label
