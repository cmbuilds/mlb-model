"""scoring/final.py — Final weighted composite score for O1.5 TB props."""

from config import TB_OFFSET_FULL, TB_OFFSET_PROXY, TB_WEIGHTS


def compute_final_score(
    batter_score: float,
    pitcher_vuln_score: float,
    platoon_score: float,
    lineup_score: float,
    park_score: float,
    weather_score: float,
    vegas_score: float,
    tto_bonus: float = 0.0,
    pitch_matchup_score: float = 50.0,
    streak_score: float = 50.0,
    bvp_score: float = 50.0,
    bvp_weight_boost: float = 0.0,
    proxy_mode: bool = False,
    vegas_missing: bool = False,
) -> float:
    """
    Final weighted composite 0–100. Weights live in config.py.

    proxy_mode: True when running on MLB API proxies only (no Savant).
    vegas_missing: True when no odds data is available. Vegas is excluded from
        the weighted sum entirely and the remaining weights are renormalized so
        an absent input doesn't penalize the score vs a neutral-50 baseline.
    Caller must determine and pass these flags — no Streamlit import here.
    """
    W = TB_WEIGHTS
    _bvp_w = W["bvp_base"] + bvp_weight_boost
    _bat_w = max(W["batter_min"], W["batter_base"] - bvp_weight_boost)

    if vegas_missing:
        # Exclude vegas from the weighted sum and renormalize.
        # Denominator drops from 1.03 → (1.03 - W["vegas"]) so all remaining
        # weights scale up proportionally. Passing implied=0 and keeping it in
        # the sum would floor every score ~4–6 pts below a real-lines baseline.
        raw = (
            batter_score        * _bat_w         +
            pitcher_vuln_score  * W["pitcher"]   +
            platoon_score       * W["platoon"]   +
            park_score          * W["park"]      +
            streak_score        * W["streak"]    +
            tto_bonus           * W["tto"]       +
            weather_score       * W["weather"]   +
            pitch_matchup_score * W["matchup"]   +
            bvp_score           * _bvp_w         +
            lineup_score        * W["lineup"]
        ) / (1.03 - W["vegas"])
    else:
        raw = (
            batter_score        * _bat_w         +
            pitcher_vuln_score  * W["pitcher"]   +
            platoon_score       * W["platoon"]   +
            vegas_score         * W["vegas"]     +
            park_score          * W["park"]      +
            streak_score        * W["streak"]    +
            tto_bonus           * W["tto"]       +
            weather_score       * W["weather"]   +
            pitch_matchup_score * W["matchup"]   +
            bvp_score           * _bvp_w         +
            lineup_score        * W["lineup"]
        ) / 1.03

    _offset = TB_OFFSET_PROXY if proxy_mode else TB_OFFSET_FULL
    return max(0, min(100, round(raw + _offset, 1)))
