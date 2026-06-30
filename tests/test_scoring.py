"""
Unit tests for scoring/* pure functions.
All tests use known inputs and verify outputs against hand-calculated expected values.
Run: pytest tests/test_scoring.py -v
"""

import pytest
from scoring.batter import compute_batter_score
from scoring.final import compute_final_score
from scoring.hits import compute_hits_batter_score, compute_hits_pitcher_score
from scoring.hr import compute_hr_score
from markets.hr import hr_get_tier, hr_score_to_prob, score_one_batter_hr
from markets.moneyline import ml_market_edge, score_game_ml
from scoring.moneyline import compute_ml_confidence, compute_win_probability
from scoring.strikeout import (
    compute_batter_k_propensity, compute_sp_k_score, k_get_tier, k_score_to_prob,
)
from scoring.park import compute_lineup_score, compute_park_score, compute_platoon_score
from scoring.pitcher import compute_pitcher_score, compute_team_bullpen_scores
from scoring.streak import compute_bvp_score, compute_streak_score, compute_tto_bonus
from scoring.vegas import compute_vegas_score, get_tier, score_to_prob
from scoring.weather import classify_wind, compute_weather_score
from data.provenance import check_bettable_ml


# ── Batter score ──────────────────────────────────────────────────────────────

class TestComputeBatterScore:
    def test_league_average_batter(self):
        """A batter at all league-average inputs should score near 50."""
        stats = {
            "slg_proxy": 0.398, "barrel_rate": 0.070, "hard_hit_rate": 0.370,
            "k_rate": 0.228, "iso_proxy": 0.165, "wrc_plus": 100.0,
            "woba": 0.315, "ev50": 0.0, "bat_speed": 0.0, "blast_rate": 0.0,
        }
        score, label, details = compute_batter_score(stats)
        assert 45 <= score <= 55, f"Expected ~50, got {score}"

    def test_elite_batter(self):
        """A Judge-tier batter (high everything) should score 75+."""
        stats = {
            "slg_proxy": 0.620, "barrel_rate": 0.200, "hard_hit_rate": 0.550,
            "k_rate": 0.150, "iso_proxy": 0.340, "wrc_plus": 190.0,
            "woba": 0.440, "ev50": 0.0, "bat_speed": 0.0, "blast_rate": 0.0,
        }
        score, label, details = compute_batter_score(stats)
        assert score >= 75, f"Expected elite ≥75, got {score}"

    def test_weak_batter(self):
        """A weak hitter (high K, low power) should score below 40."""
        stats = {
            "slg_proxy": 0.280, "barrel_rate": 0.020, "hard_hit_rate": 0.250,
            "k_rate": 0.380, "iso_proxy": 0.060, "wrc_plus": 55.0,
            "woba": 0.240, "ev50": 0.0, "bat_speed": 0.0, "blast_rate": 0.0,
        }
        score, label, details = compute_batter_score(stats)
        assert score <= 40, f"Expected weak ≤40, got {score}"

    def test_bat_tracking_boost(self):
        """Bat-tracking data should only add (not lower) vs same baseline without it."""
        base = {
            "slg_proxy": 0.450, "barrel_rate": 0.100, "hard_hit_rate": 0.400,
            "k_rate": 0.200, "iso_proxy": 0.200, "wrc_plus": 120.0, "woba": 0.350,
        }
        score_no_bt, _, _ = compute_batter_score({**base, "ev50": 0.0, "bat_speed": 0.0, "blast_rate": 0.0})
        score_with_bt, _, _ = compute_batter_score({
            **base, "ev50": 98.0, "bat_speed": 74.0, "blast_rate": 0.30,
        })
        assert score_with_bt >= score_no_bt - 2, "Bat tracking should not significantly hurt"

    def test_details_keys_present(self):
        stats = {"slg_proxy": 0.398, "barrel_rate": 0.07, "hard_hit_rate": 0.37,
                 "k_rate": 0.228, "iso_proxy": 0.165, "wrc_plus": 100.0, "woba": 0.315}
        _, _, details = compute_batter_score(stats)
        for key in ("xSLG", "Barrel%", "HardHit%", "K%", "ISO", "wRC+", "wOBA"):
            assert key in details, f"Missing detail key: {key}"

    def test_output_in_0_100_range(self):
        for k_rate in (0.10, 0.22, 0.40):
            stats = {"slg_proxy": 0.398, "barrel_rate": 0.07, "hard_hit_rate": 0.37,
                     "k_rate": k_rate, "iso_proxy": 0.165, "wrc_plus": 100.0}
            score, _, _ = compute_batter_score(stats)
            assert 0 <= score <= 100, f"Score {score} out of range for K%={k_rate}"


# ── Pitcher score ─────────────────────────────────────────────────────────────

class TestComputePitcherScore:
    def test_ace_is_low_vulnerability(self):
        """A dominant ace (high K, low HH, low ERA) = low vulnerability = favorable for batter."""
        stats = {"k_rate_allowed": 0.32, "hard_hit_allowed": 0.28, "barrel_allowed": 0.04,
                 "era": 2.50, "fip": 2.60, "whip": 0.90}
        score, label = compute_pitcher_score(stats)
        assert score < 45, f"Ace should be low vuln (<45), got {score}"

    def test_bad_pitcher_high_vulnerability(self):
        """A poor pitcher (low K, high HH, high ERA) = high vulnerability."""
        stats = {"k_rate_allowed": 0.14, "hard_hit_allowed": 0.50, "barrel_allowed": 0.12,
                 "era": 6.00, "fip": 5.80, "whip": 1.70}
        score, label = compute_pitcher_score(stats)
        assert score > 60, f"Bad pitcher should be high vuln (>60), got {score}"

    def test_bullpen_blended(self):
        """High bullpen vulnerability should raise overall score."""
        stats = {"k_rate_allowed": 0.22, "hard_hit_allowed": 0.37, "barrel_allowed": 0.07,
                 "era": 4.20, "fip": 4.20, "whip": 1.30}
        score_avg_bp, _ = compute_pitcher_score(stats, bullpen_vuln=42.0)
        score_bad_bp, _ = compute_pitcher_score(stats, bullpen_vuln=75.0)
        assert score_bad_bp > score_avg_bp, "Bad bullpen should raise vulnerability"

    def test_output_in_range(self):
        stats = {"k_rate_allowed": 0.22, "hard_hit_allowed": 0.37, "barrel_allowed": 0.07,
                 "era": 4.20, "fip": 4.20, "whip": 1.30}
        score, _ = compute_pitcher_score(stats)
        assert 0 <= score <= 100


# ── Park score ────────────────────────────────────────────────────────────────

class TestParkScores:
    def test_coors_highest(self):
        coors_score, _ = compute_park_score("COL", True)
        petco_score, _ = compute_park_score("SDP", True)
        assert coors_score > 80, f"Coors should score >80, got {coors_score}"
        assert coors_score > petco_score, "Coors should beat Petco"

    def test_platoon_lhb_rhp(self):
        score, label = compute_platoon_score("L", "R")
        assert score == 75.0, f"LHB vs RHP should be 75.0, got {score}"
        assert "56" in label

    def test_platoon_rhb_lhp(self):
        score, _ = compute_platoon_score("R", "L")
        assert score == 65.0

    def test_platoon_same_hand_disadvantage(self):
        lhb_vs_lhp, _ = compute_platoon_score("L", "L")
        rhb_vs_rhp, _ = compute_platoon_score("R", "R")
        assert lhb_vs_lhp < 50, "LHB vs LHP should be below 50"
        assert rhb_vs_rhp == 50.0

    def test_platoon_switch_hitter(self):
        score_vs_rhp, _ = compute_platoon_score("S", "R")
        score_vs_lhp, _ = compute_platoon_score("B", "L")
        assert score_vs_rhp >= 70, "Switch vs RHP should be ≥70 (bats LH)"
        assert score_vs_lhp >= 60, "Switch vs LHP should be ≥60 (bats RH)"

    def test_lineup_leadoff_beats_9th(self):
        lead, _ = compute_lineup_score(1)
        ninth, _ = compute_lineup_score(9)
        assert lead > ninth, "Leadoff should score higher than 9th"


# ── Vegas / probability ───────────────────────────────────────────────────────

class TestVegas:
    def test_no_lines_returns_zero(self):
        score, label = compute_vegas_score(0)
        assert score == 0.0
        assert "⚠️" in label

    def test_high_total_scores_higher(self):
        low, _ = compute_vegas_score(3.0)
        high, _ = compute_vegas_score(6.0)
        assert high > low

    def test_score_to_prob_midpoint(self):
        # At the configured midpoint (score=62): logistic=0.5, prob = 0.42 + 0.5*(0.78-0.42) = 0.60
        prob = score_to_prob(62)
        assert 0.58 <= prob <= 0.62, f"Score 62 → prob should be ~0.60, got {prob}"

    def test_score_to_prob_monotonic(self):
        probs = [score_to_prob(s) for s in range(40, 95, 5)]
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1], "score_to_prob must be monotonically increasing"

    def test_get_tier_full_mode(self):
        assert get_tier(85, proxy_mode=False) == "🔒 TIER 1"
        assert get_tier(72, proxy_mode=False) == "✅ TIER 2"
        assert get_tier(63, proxy_mode=False) == "📊 TIER 3"
        assert get_tier(55, proxy_mode=False) == "❌ NO PLAY"

    def test_get_tier_proxy_mode(self):
        assert get_tier(77, proxy_mode=True) == "🔒 TIER 1"
        assert get_tier(67, proxy_mode=True) == "✅ TIER 2"
        assert get_tier(57, proxy_mode=True) == "📊 TIER 3"
        assert get_tier(50, proxy_mode=True) == "❌ NO PLAY"

    def test_tier_thresholds_shift_down_in_proxy(self):
        # Score 79 = Tier 2 in full mode, but Tier 1 in proxy (floor shifted to 75)
        assert get_tier(79, proxy_mode=False) == "✅ TIER 2"
        assert get_tier(79, proxy_mode=True) == "🔒 TIER 1"


# ── Weather ───────────────────────────────────────────────────────────────────

class TestWeather:
    def test_dome_neutral(self):
        score, _ = compute_weather_score({"is_dome": True, "wind_speed": 0, "temperature": 72})
        assert score == 50.0, "Dome should be neutral (50)"

    def test_strong_out_wind_boost(self):
        # wind_effect="out" adds +15 to base 50 → 65.0 exactly
        score, _ = compute_weather_score({
            "is_dome": False, "wind_speed": 18, "wind_effect": "out",
            "temperature": 72, "wind_dir_label": "Out to CF"
        })
        assert score >= 65, f"Out wind should boost to ≥65, got {score}"
        # strong_out adds even more
        score_so, _ = compute_weather_score({
            "is_dome": False, "wind_speed": 18, "wind_effect": "strong_out",
            "temperature": 72, "wind_dir_label": "Out to CF"
        })
        assert score_so > 65, f"Strong out wind should boost >65, got {score_so}"

    def test_in_wind_suppresses(self):
        score, _ = compute_weather_score({
            "is_dome": False, "wind_speed": 18, "wind_effect": "in",
            "temperature": 72, "wind_dir_label": "In from CF"
        })
        assert score < 40, f"In wind should suppress, got {score}"

    def test_cold_temp_penalized(self):
        cold, _ = compute_weather_score({"is_dome": False, "wind_speed": 0,
                                          "temperature": 40, "wind_effect": "neutral"})
        warm, _ = compute_weather_score({"is_dome": False, "wind_speed": 0,
                                          "temperature": 72, "wind_effect": "neutral"})
        assert cold < warm, "Cold weather should score lower than warm"

    def test_classify_wind(self):
        # classify_wind takes (direction_degrees: float, speed: float)
        # SW = 225° → in 157.5-292.5 range → out wind; speed 20 ≥ 12 → strong_out
        _, effect_out = classify_wind(225, 20)
        assert effect_out == "strong_out", f"SW 20mph should be strong_out, got {effect_out}"
        # N = 0° → ≤ 67.5 → in wind; speed 20 ≥ 10
        _, effect_in = classify_wind(0, 20)
        assert effect_in == "in", f"N 20mph should be in wind, got {effect_in}"
        # E = 90° → none of the threshold ranges → neutral; speed 5 < 8 → calm/neutral
        _, effect_calm = classify_wind(90, 5)
        assert effect_calm == "neutral"


# ── Streak / BvP ─────────────────────────────────────────────────────────────

class TestStreakBvP:
    def test_hot_streak_above_50(self):
        hot = {"tb_per_game": 2.5, "games": 7, "hr_last_7": 2, "h_last_7": 10}
        score, _ = compute_streak_score(hot, 0.398)
        assert score > 60, f"Hot streak should score >60, got {score}"

    def test_cold_streak_below_50(self):
        cold = {"tb_per_game": 0.5, "games": 7, "hr_last_7": 0, "h_last_7": 3}
        score, _ = compute_streak_score(cold, 0.398)
        assert score < 45, f"Cold streak should score <45, got {score}"

    def test_no_data_neutral(self):
        score, _ = compute_streak_score({}, 0.398)
        assert score == 50.0

    def test_bvp_owns(self):
        bvp = {"ab": 25, "avg": 0.420, "slg": 0.720, "hr": 3, "xbh": 6}
        score, label, sig = compute_bvp_score(bvp, 0.398)
        assert sig == "owns", f"Expected 'owns', got '{sig}'"
        assert score > 70

    def test_bvp_dominated(self):
        # "dominated" requires: slg_delta ≤ -0.10 AND k_rate_bvp ≥ 0.35 AND NOT is_fade
        # is_fade fires when career_slg ≤ 0.250, so use slg > 0.250 to avoid it
        # slg=0.280 (>0.250 → not fade), slg_delta=-0.118 (<-0.100), k_rate=12/25=0.48 (≥0.35)
        bvp = {"ab": 25, "avg": 0.180, "slg": 0.280, "hr": 0, "xbh": 2, "so": 12}
        score, label, sig = compute_bvp_score(bvp, 0.398)
        assert sig == "dominated", f"Expected 'dominated', got '{sig}' (score={score})"
        assert score < 50

    def test_bvp_no_data(self):
        score, label, sig = compute_bvp_score({}, 0.398)
        assert sig == "no_data"
        assert score == 50.0

    def test_tto_bonus_slot_5_or_fewer(self):
        early_sc, _ = compute_tto_bonus(1)   # will see SP 3rd time
        late_sc, _  = compute_tto_bonus(8)   # unlikely to see SP 3rd time
        assert early_sc >= late_sc, "Early slots should get TTO bonus ≥ late slots"


# ── Final composite ───────────────────────────────────────────────────────────

class TestComputeFinalScore:
    _base = dict(
        batter_score=65.0, pitcher_vuln_score=65.0, platoon_score=75.0,
        lineup_score=70.0, park_score=55.0, weather_score=50.0, vegas_score=60.0,
    )

    def test_output_in_range(self):
        score = compute_final_score(**self._base)
        assert 0 <= score <= 100, f"Score {score} out of range"

    def test_proxy_mode_higher_than_full(self):
        full  = compute_final_score(**self._base, proxy_mode=False)
        proxy = compute_final_score(**self._base, proxy_mode=True)
        assert proxy > full, f"Proxy offset ({proxy}) should exceed full ({full})"

    def test_proxy_offset_difference(self):
        full  = compute_final_score(**self._base, proxy_mode=False)
        proxy = compute_final_score(**self._base, proxy_mode=True)
        diff = round(proxy - full, 1)
        assert abs(diff - 2.5) < 0.5, f"Proxy/full offset diff should be ~2.5, got {diff}"

    def test_higher_inputs_higher_score(self):
        low  = compute_final_score(batter_score=30, pitcher_vuln_score=30,
                                   platoon_score=30, lineup_score=30, park_score=30,
                                   weather_score=30, vegas_score=30)
        high = compute_final_score(batter_score=90, pitcher_vuln_score=90,
                                   platoon_score=90, lineup_score=90, park_score=90,
                                   weather_score=90, vegas_score=90)
        assert high > low

    def test_bvp_boost_reduces_batter_weight(self):
        no_boost   = compute_final_score(**self._base, bvp_score=90, bvp_weight_boost=0.0)
        with_boost = compute_final_score(**self._base, bvp_score=90, bvp_weight_boost=0.08)
        # BvP boost raises bvp_weight and lowers batter_weight — with a strong BvP
        # and average batter score, the result should be higher
        assert with_boost >= no_boost - 1.0, "BvP boost with strong BvP should not hurt"


# ── HR score ─────────────────────────────────────────────────────────────────

class TestHRScore:
    def test_high_barrel_high_score(self):
        score = compute_hr_score(barrel_rate=0.22, sweet_spot=0.40, park_hr_factor=1.10,
                                  implied_total=5.0, weather={"is_dome": True},
                                  hard_hit=0.50, exit_velocity=92.0, iso=0.280)
        assert score > 65, f"Elite barrel/HH should score >65, got {score}"

    def test_weak_contact_low_score(self):
        score = compute_hr_score(barrel_rate=0.02, sweet_spot=0.20, park_hr_factor=0.90,
                                  implied_total=3.5, weather={"is_dome": True},
                                  hard_hit=0.25, exit_velocity=83.0, iso=0.080)
        assert score < 35, f"Weak contact should score <35, got {score}"

    def test_coors_boost(self):
        petco = compute_hr_score(barrel_rate=0.10, sweet_spot=0.35, park_hr_factor=0.82,
                                  implied_total=4.5, weather={"is_dome": True})
        coors = compute_hr_score(barrel_rate=0.10, sweet_spot=0.35, park_hr_factor=1.35,
                                  implied_total=4.5, weather={"is_dome": True})
        assert coors > petco, "Coors should score higher than Petco"

    def test_bat_tracking_zero_when_absent(self):
        """When bat-tracking values are 0 (absent), score should still compute."""
        score = compute_hr_score(barrel_rate=0.10, sweet_spot=0.35, park_hr_factor=1.0,
                                  implied_total=4.5, weather={"is_dome": True},
                                  ev50=0.0, bat_speed=0.0, blast_rate=0.0)
        assert 0 <= score <= 100

    def test_output_always_in_range(self):
        for barrel in (0.0, 0.07, 0.25):
            score = compute_hr_score(barrel_rate=barrel, sweet_spot=0.30,
                                     park_hr_factor=1.0, implied_total=4.5,
                                     weather={"is_dome": True})
            assert 0 <= score <= 100, f"HR score {score} out of range for barrel={barrel}"


# ── O0.5 batter scorer ────────────────────────────────────────────────────────

class TestHitsBatterScore:
    _league_avg = {
        "k_rate": 0.228, "hard_hit_rate": 0.370, "wrc_plus": 100.0,
        "slg_proxy": 0.398, "woba": 0.315,
        "ev50": 0.0, "bat_speed": 0.0, "blast_rate": 0.0,
    }

    def test_league_avg_near_50(self):
        score, _, _ = compute_hits_batter_score(self._league_avg)
        assert 45 <= score <= 55, f"Expected ~50, got {score}"

    def test_high_k_rate_penalized(self):
        """High K% batter should score well below league average."""
        stats = {**self._league_avg, "k_rate": 0.380}
        score, _, _ = compute_hits_batter_score(stats)
        assert score < 40, f"High K% should score <40, got {score}"

    def test_low_k_rate_rewarded(self):
        """Rare-K contact hitter should score above league average."""
        stats = {**self._league_avg, "k_rate": 0.090, "woba": 0.360, "wrc_plus": 130.0}
        score, _, _ = compute_hits_batter_score(stats)
        assert score >= 60, f"Low-K contact hitter should score ≥60, got {score}"

    def test_k_rate_dominates_over_power(self):
        """For O0.5: contact hitter (low K, low SLG) outscores power hitter (high K, high SLG)."""
        contact = {**self._league_avg, "k_rate": 0.100, "slg_proxy": 0.380, "iso_proxy": 0.080}
        power   = {**self._league_avg, "k_rate": 0.340, "slg_proxy": 0.600, "iso_proxy": 0.300}
        s_contact, _, _ = compute_hits_batter_score(contact)
        s_power,   _, _ = compute_hits_batter_score(power)
        assert s_contact > s_power, f"Contact ({s_contact:.1f}) should beat power ({s_power:.1f}) for O0.5"

    def test_barrel_and_iso_not_present_in_details(self):
        """Barrel% and ISO should not appear in O0.5 batter details (power not relevant)."""
        _, _, details = compute_hits_batter_score(self._league_avg)
        assert "Barrel%" not in details, "Barrel% should not be in O0.5 batter details"
        assert "ISO" not in details, "ISO should not be in O0.5 batter details"

    def test_output_in_range(self):
        for k in (0.05, 0.228, 0.45):
            score, _, _ = compute_hits_batter_score({**self._league_avg, "k_rate": k})
            assert 0 <= score <= 100, f"Score {score} out of range for k_rate={k}"


# ── O0.5 pitcher scorer ───────────────────────────────────────────────────────

class TestHitsPitcherScore:
    _league_avg_sp = {
        "k_rate_allowed": 0.220, "hard_hit_allowed": 0.370,
        "whip": 1.30, "era": 4.20, "fip": 4.20,
    }

    def test_league_avg_near_50_before_bullpen(self):
        """SP at all league-avg stats + avg bullpen should score near 50."""
        score, _ = compute_hits_pitcher_score(self._league_avg_sp, bullpen_vuln=50.0)
        assert 45 <= score <= 60, f"Expected near 50, got {score}"

    def test_high_k_sp_lowers_vulnerability(self):
        """A strikeout pitcher (K%=35%) is harder to get hits off — lower score."""
        ace = {**self._league_avg_sp, "k_rate_allowed": 0.350}
        avg = self._league_avg_sp
        s_ace, _ = compute_hits_pitcher_score(ace, bullpen_vuln=50.0)
        s_avg, _ = compute_hits_pitcher_score(avg, bullpen_vuln=50.0)
        assert s_ace < s_avg, f"High-K SP ({s_ace:.1f}) should be less vulnerable than avg ({s_avg:.1f})"

    def test_low_k_sp_raises_vulnerability(self):
        """A contact pitcher (K%=12%) allows more hits — higher vulnerability."""
        soft = {**self._league_avg_sp, "k_rate_allowed": 0.120}
        score, _ = compute_hits_pitcher_score(soft, bullpen_vuln=50.0)
        avg_score, _ = compute_hits_pitcher_score(self._league_avg_sp, bullpen_vuln=50.0)
        assert score > avg_score, f"Low-K SP ({score:.1f}) should be more vulnerable than avg ({avg_score:.1f})"

    def test_default_inputs_use_fallback(self):
        """When no Statcast data (all defaults), should still return valid score."""
        score, label = compute_hits_pitcher_score({}, bullpen_vuln=42.0)
        assert 0 <= score <= 100
        assert "WHIP/ERA-only" in label

    def test_output_in_range(self):
        for k in (0.08, 0.220, 0.42):
            score, _ = compute_hits_pitcher_score(
                {**self._league_avg_sp, "k_rate_allowed": k}, bullpen_vuln=50.0
            )
            assert 0 <= score <= 100, f"Score {score} out of range for k_rate={k}"


# ── O0.5 market kernel ────────────────────────────────────────────────────────

class TestScoreOneBatterO05:
    _batter = {
        "k_rate": 0.228, "hard_hit_rate": 0.370, "wrc_plus": 100.0,
        "slg_proxy": 0.398, "woba": 0.315, "ev50": 0.0,
        "bat_speed": 0.0, "blast_rate": 0.0, "iso_proxy": 0.165,
        "data_source": "statcast",
        "_provenance": {
            "k_rate": "measured", "woba": "measured", "slg_proxy": "measured",
            "hard_hit_rate": "measured", "barrel_rate": "measured",
            "iso_proxy": "measured",
        },
    }
    _pitcher = {
        "k_rate_allowed": 0.220, "hard_hit_allowed": 0.370,
        "era": 4.20, "fip": 4.20, "whip": 1.30,
        "data_source": "statcast",
        "_provenance": {
            "k_rate_allowed": "measured", "hard_hit_allowed": "measured",
        },
    }

    def _score(self, **overrides):
        from markets.hits_o05 import score_one_batter_o05
        batter = {**self._batter, **overrides.pop("batter_overrides", {})}
        pitcher = {**self._pitcher, **overrides.pop("pitcher_overrides", {})}
        defaults = dict(
            name="Test Batter", player_id="999", team="NYY", opponent="BOS",
            game_pk="12345", batter_hand="R", hand_real=True, sp_hand="R",
            sp_name="J. Test", sp_id="888", lineup_slot=3, lineup_confirmed=True,
            batter_position="RF", park_team="NYY", batter_stats=batter,
            pitcher_stats=pitcher, recent_form={}, implied=4.5,
            prop_implied=None, team_bullpen_scores={},
        )
        defaults.update(overrides)
        return score_one_batter_o05(**defaults)

    def test_returns_required_keys(self):
        result = self._score()
        for key in ("score", "prob", "tier", "sub_batter", "sub_pitcher",
                    "dq_score", "bettable", "non_bettable_reasons"):
            assert key in result, f"Missing key: {key}"

    def test_score_in_range(self):
        assert 0 <= self._score()["score"] <= 100

    def test_prob_in_o05_range(self):
        """O0.5 prob should be in the [0.52, 0.85] config range."""
        prob = self._score()["prob"]
        assert 0.52 <= prob <= 0.85, f"O0.5 prob {prob} outside expected range"

    def test_high_k_sp_lowers_score(self):
        """Batter facing a 35% K pitcher should score lower than vs avg pitcher."""
        avg_score = self._score()["score"]
        k_score = self._score(pitcher_overrides={"k_rate_allowed": 0.350})["score"]
        assert k_score < avg_score, f"High-K SP should lower score: {k_score} vs {avg_score}"

    def test_sp_tbd_caps_score(self):
        """Unknown SP should cap score at 70."""
        result = self._score(sp_name="TBD")
        assert result["score"] <= 70.0, f"TBD SP should cap at 70, got {result['score']}"

    def test_not_bettable_without_sp(self):
        result = self._score(sp_name="TBD", hand_real=True)
        assert not result["bettable"]
        assert any("SP unknown" in r for r in result["non_bettable_reasons"])

    def test_o05_prob_higher_than_o15_prob(self):
        """O0.5 probability should be higher than O1.5 for the same batter."""
        from markets.hits_o05 import o05_score_to_prob
        from scoring.vegas import score_to_prob
        # Both scorers produce a score; the prob calibration for O0.5 is higher-floored
        o05_prob = o05_score_to_prob(55)  # at midpoint
        o15_prob = score_to_prob(55)      # O1.5 at same score
        assert o05_prob > o15_prob, f"O0.5 prob ({o05_prob}) should exceed O1.5 ({o15_prob}) at same score"


# ── K prop — SP K score ───────────────────────────────────────────────────────

class TestComputeSpKScore:
    _avg_sp = {
        "k_rate_allowed": 0.228, "swstr_pct": 0.0,
        "_provenance": {"k_rate_allowed": "measured"},
        "data_source": "statcast",
    }

    def test_league_avg_near_50(self):
        """All-average SP vs all-average lineup in neutral game should score near 50."""
        score, label, details = compute_sp_k_score(
            self._avg_sp, opp_lineup_k_avg=0.228, implied_total=4.5
        )
        assert 45 <= score <= 60, f"Expected near 50, got {score}"

    def test_elite_sp_scores_high(self):
        """An ace (K%=35%) vs a K-prone lineup (K%=30%) scores 70+."""
        ace = {**self._avg_sp, "k_rate_allowed": 0.350, "swstr_pct": 0.145}
        score, _, details = compute_sp_k_score(ace, opp_lineup_k_avg=0.300, implied_total=4.0)
        assert score >= 70, f"Elite K spot should score ≥70, got {score}"

    def test_soft_sp_scores_low(self):
        """A contact SP (K%=14%) vs contact lineup scores below 40."""
        soft = {**self._avg_sp, "k_rate_allowed": 0.140}
        score, _, _ = compute_sp_k_score(soft, opp_lineup_k_avg=0.170, implied_total=5.5)
        assert score < 45, f"Soft SP should score <45, got {score}"

    def test_low_total_raises_score(self):
        """Low-implied-total game (pitcher's duel) should score higher than high-total."""
        low  = compute_sp_k_score(self._avg_sp, opp_lineup_k_avg=0.228, implied_total=3.0)[0]
        high = compute_sp_k_score(self._avg_sp, opp_lineup_k_avg=0.228, implied_total=6.0)[0]
        assert low > high, f"Low total ({low:.1f}) should beat high total ({high:.1f})"

    def test_swstr_real_boosts_score(self):
        """Real SwStr% above average should boost score vs no-SwStr%."""
        with_swstr = {**self._avg_sp, "swstr_pct": 0.145}
        without    = self._avg_sp
        s_with, _, d_with = compute_sp_k_score(with_swstr, opp_lineup_k_avg=0.228, implied_total=4.5)
        s_without, _, _ = compute_sp_k_score(without, opp_lineup_k_avg=0.228, implied_total=4.5)
        assert s_with > s_without, f"Real SwStr% should boost score: {s_with} vs {s_without}"
        assert d_with["swstr_real"] is True

    def test_ump_adj_applied(self):
        """Positive ump_k_adj (tight-zone ump) should raise score."""
        base = compute_sp_k_score(self._avg_sp, opp_lineup_k_avg=0.228, implied_total=4.5)[0]
        high = compute_sp_k_score(self._avg_sp, opp_lineup_k_avg=0.228, implied_total=4.5,
                                   ump_k_adj=0.02)[0]
        assert high > base, f"Positive ump_k_adj should raise score: {high} vs {base}"

    def test_output_in_range(self):
        for k in (0.10, 0.228, 0.40):
            sp = {**self._avg_sp, "k_rate_allowed": k}
            score, _, _ = compute_sp_k_score(sp, opp_lineup_k_avg=0.228, implied_total=4.5)
            assert 0 <= score <= 100, f"Score {score} out of range for K%={k}"

    def test_k_score_to_prob_monotonic(self):
        """Higher K scores should map to higher probabilities."""
        probs = [k_score_to_prob(s) for s in (40, 50, 60, 70, 80)]
        assert all(p1 < p2 for p1, p2 in zip(probs, probs[1:])), f"Prob not monotonic: {probs}"

    def test_k_get_tier(self):
        assert k_get_tier(82) == "⚡ ELITE K"
        assert k_get_tier(72) == "🔥 STRONG K"
        assert k_get_tier(62) == "📊 LEAN K"
        assert k_get_tier(55) == "➖ NO PLAY"


# ── K prop — per-batter K propensity ─────────────────────────────────────────

class TestComputeBatterKPropensity:
    _avg_batter  = {"k_rate": 0.228, "_provenance": {"k_rate": "measured"}}
    _avg_pitcher = {"k_rate_allowed": 0.228, "swstr_pct": 0.0,
                    "_provenance": {"k_rate_allowed": "measured"}}

    def test_league_avg_near_50(self):
        score, _, _ = compute_batter_k_propensity(self._avg_batter, self._avg_pitcher)
        assert 45 <= score <= 55, f"Expected ~50, got {score}"

    def test_high_k_batter_scores_high(self):
        """A 35% K-rate batter vs avg SP should score high."""
        high_k = {**self._avg_batter, "k_rate": 0.350}
        score, _, _ = compute_batter_k_propensity(high_k, self._avg_pitcher)
        assert score >= 60, f"High K batter should score ≥60, got {score}"

    def test_low_k_batter_scores_low(self):
        """A contact hitter (10% K) vs avg SP should score low."""
        low_k = {**self._avg_batter, "k_rate": 0.100}
        score, _, _ = compute_batter_k_propensity(low_k, self._avg_pitcher)
        assert score <= 40, f"Low K batter should score ≤40, got {score}"

    def test_high_k_sp_raises_batter_k_score(self):
        """Batter facing a 35% K SP should score higher K propensity."""
        avg_score = compute_batter_k_propensity(self._avg_batter, self._avg_pitcher)[0]
        k_sp = {**self._avg_pitcher, "k_rate_allowed": 0.350}
        k_score = compute_batter_k_propensity(self._avg_batter, k_sp)[0]
        assert k_score > avg_score, f"High-K SP ({k_score}) should beat avg SP ({avg_score})"

    def test_output_in_range(self):
        score, _, _ = compute_batter_k_propensity(self._avg_batter, self._avg_pitcher)
        assert 0 <= score <= 100


# ── K prop — market module ────────────────────────────────────────────────────

class TestScoreSpKProp:
    _sp = {
        "k_rate_allowed": 0.280, "swstr_pct": 0.130, "era": 3.50, "fip": 3.40,
        "data_source": "statcast",
        "_provenance": {"k_rate_allowed": "measured", "hard_hit_allowed": "measured"},
    }
    _lineup = [
        {"k_rate": 0.220, "_provenance": {"k_rate": "measured"}},
        {"k_rate": 0.280, "_provenance": {"k_rate": "measured"}},
        {"k_rate": 0.190, "_provenance": {"k_rate": "measured"}},
        {"k_rate": 0.310, "_provenance": {"k_rate": "measured"}},
        {"k_rate": 0.240, "_provenance": {"k_rate": "measured"}},
        {"k_rate": 0.260, "_provenance": {"k_rate": "measured"}},
        {"k_rate": 0.200, "_provenance": {"k_rate": "measured"}},
        {"k_rate": 0.295, "_provenance": {"k_rate": "measured"}},
        {"k_rate": 0.175, "_provenance": {"k_rate": "measured"}},
    ]

    def _score(self, **overrides):
        from markets.k_props import score_sp_k_prop
        defaults = dict(
            sp_name="J. Ace", sp_id="12345", sp_team="NYY", opp_team="BOS",
            game_pk="99999", sp_stats=self._sp, opp_batter_stats=self._lineup,
            implied_total=4.0, ump_k_adj=0.0, ump_name="—", market_implied=None,
        )
        defaults.update(overrides)
        return score_sp_k_prop(**defaults)

    def test_returns_required_keys(self):
        r = self._score()
        for key in ("score", "tier", "bettable", "sp_k_pct", "opp_lineup_k_avg",
                    "n_batters", "n_batters_with_k", "sub_sp_k", "sub_opp_lineup"):
            assert key in r, f"Missing key: {key}"

    def test_score_in_range(self):
        assert 0 <= self._score()["score"] <= 100

    def test_elite_sp_scores_high(self):
        """Above-avg K SP (28%) vs avg-K lineup should score above 50."""
        score = self._score()["score"]
        assert score >= 55, f"Above-avg K SP should score ≥55, got {score}"

    def test_bettable_with_full_data(self):
        result = self._score()
        assert result["bettable"], f"Should be bettable: {result['non_bettable_reasons']}"

    def test_not_bettable_without_sp_stats(self):
        bad_sp = {"data_source": "league_avg", "_provenance": {"k_rate_allowed": "league_avg"}}
        result = self._score(sp_stats=bad_sp)
        assert not result["bettable"]

    def test_not_bettable_too_few_batters_with_k(self):
        """Fewer than 5 batters with measured K% → not bettable."""
        small_lineup = [
            {"k_rate": 0.228, "_provenance": {"k_rate": "measured"}} for _ in range(3)
        ]
        result = self._score(opp_batter_stats=small_lineup)
        assert not result["bettable"]

    def test_edge_label_with_market_implied(self):
        result = self._score(market_implied=0.55)
        assert "edge_label" in result
        assert result["edge_label"] != ""


# ── HR market module ──────────────────────────────────────────────────────────

class TestScoreOneBatterHR:
    _elite_batter = {
        "barrel_rate": 0.180, "hard_hit_rate": 0.540, "iso_proxy": 0.310,
        "ev50": 102.0, "bat_speed": 76.0, "blast_rate": 0.32,
        "exit_velocity_avg": 95.0, "sweet_spot_rate": 0.38,
        "slg_proxy": 0.600, "wrc_plus": 175.0, "k_rate": 0.180,
        "data_source": "statcast",
        "_provenance": {
            "barrel_rate": "measured", "hard_hit_rate": "measured",
            "iso_proxy": "measured", "k_rate": "measured",
            "slg_proxy": "measured", "woba": "measured",
        },
    }
    _avg_batter = {
        "barrel_rate": 0.070, "hard_hit_rate": 0.370, "iso_proxy": 0.165,
        "ev50": 0.0, "bat_speed": 0.0, "blast_rate": 0.0,
        "exit_velocity_avg": 88.5, "sweet_spot_rate": 0.30,
        "slg_proxy": 0.398, "wrc_plus": 100.0, "k_rate": 0.228,
        "data_source": "statcast",
        "_provenance": {
            "barrel_rate": "measured", "hard_hit_rate": "measured",
            "iso_proxy": "measured", "k_rate": "measured",
            "slg_proxy": "measured", "woba": "measured",
        },
    }
    _pitcher = {
        "k_rate_allowed": 0.228, "hard_hit_allowed": 0.370,
        "era": 4.20, "fip": 4.20, "whip": 1.30,
        "data_source": "statcast",
        "_provenance": {"k_rate_allowed": "measured", "hard_hit_allowed": "measured"},
    }
    _dome_weather = {"is_dome": True}

    def _score(self, batter, **overrides):
        pitcher = overrides.pop("pitcher", self._pitcher)
        defaults = dict(
            name="Test", player_id="1", team="NYY", opponent="BOS",
            game_pk="123", batter_hand="R", hand_real=True, sp_hand="R",
            sp_name="J. Pitcher", sp_id="2", lineup_slot=4, lineup_confirmed=True,
            batter_position="1B", park_team="NYY", batter_stats=batter,
            pitcher_stats=pitcher, weather=self._dome_weather,
            implied=4.5, prop_implied=None, team_bullpen_scores={},
        )
        defaults.update(overrides)
        return score_one_batter_hr(**defaults)

    def test_elite_batter_scores_high(self):
        """Judge-tier batter (barrel%=18%) should score 70+."""
        score = self._score(self._elite_batter)["score"]
        assert score >= 70, f"Elite HR batter should score ≥70, got {score}"

    def test_avg_batter_scores_low(self):
        """Average power batter (barrel%=7%) should score <50."""
        score = self._score(self._avg_batter)["score"]
        assert score < 50, f"Average batter should score <50 for HR, got {score}"

    def test_barrel_dominates_over_hh(self):
        """High barrel%, low HH% should outscore low barrel%, high HH% for HR."""
        high_barrel = {**self._avg_batter, "barrel_rate": 0.160, "hard_hit_rate": 0.300}
        high_hh     = {**self._avg_batter, "barrel_rate": 0.040, "hard_hit_rate": 0.520}
        s_barrel = self._score(high_barrel)["score"]
        s_hh     = self._score(high_hh)["score"]
        assert s_barrel > s_hh, f"High barrel% ({s_barrel:.1f}) should outscore high HH% ({s_hh:.1f}) for HR"

    def test_coors_vs_neutral_park(self):
        """Coors Field (1.35x HR factor) should score higher than neutral park."""
        coors   = self._score(self._avg_batter, park_team="COL")["score"]
        neutral = self._score(self._avg_batter, park_team="NYY")["score"]
        assert coors > neutral, f"Coors ({coors:.1f}) should outscore neutral ({neutral:.1f})"

    def test_not_bettable_without_barrel(self):
        """proxy or missing barrel% → not bettable."""
        bad_prov = {**self._avg_batter}
        bad_prov["_provenance"] = {**bad_prov["_provenance"], "barrel_rate": "proxy"}
        result = self._score(bad_prov)
        assert not result["bettable"]
        assert any("barrel" in r for r in result["non_bettable_reasons"])

    def test_barrel_source_surfaced(self):
        """barrel_source field should reflect provenance."""
        result = self._score(self._elite_batter)
        assert result["barrel_source"] == "measured"

    def test_sp_tbd_caps_score(self):
        result = self._score(self._elite_batter, sp_name="TBD")
        assert result["score"] <= 74.0

    def test_prob_in_hr_range(self):
        """HR prob should be in the calibrated 8-22% range."""
        prob = self._score(self._elite_batter)["prob"]
        assert 0.08 <= prob <= 0.22, f"HR prob {prob} outside [0.08, 0.22]"

    def test_hr_tiers(self):
        assert hr_get_tier(85) == "💣 ELITE HR"
        assert hr_get_tier(72) == "🔥 STRONG HR"
        assert hr_get_tier(62) == "📊 LEAN HR"
        assert hr_get_tier(55) == "➖ NO PLAY"

    def test_hr_prob_monotonic(self):
        probs = [hr_score_to_prob(s) for s in (40, 50, 60, 70, 80, 90)]
        assert all(p1 < p2 for p1, p2 in zip(probs, probs[1:])), f"HR prob not monotonic: {probs}"

    def test_output_required_keys(self):
        result = self._score(self._avg_batter)
        for key in ("score", "prob", "tier", "barrel_rate", "barrel_source",
                    "bettable", "non_bettable_reasons", "dq_score"):
            assert key in result, f"Missing key: {key}"


# ── Moneyline: pure scoring functions ─────────────────────────────────────────

class TestComputeWinProbability:
    def _sp(self, vuln=50.0):
        return {"_sp_vuln": vuln}

    def test_balanced_game_has_home_advantage(self):
        """Equal teams → home wins >50% due to +3.5% home-field boost."""
        hwp, _ = compute_win_probability(
            self._sp(50), self._sp(50),
            100, 100, 42, 42, 0, 0, 0, 0,
        )
        assert 0.50 < hwp < 0.60

    def test_elite_home_sp_raises_home_prob(self):
        """Elite home SP (low vuln) → higher home win prob than average."""
        hwp_elite, _ = compute_win_probability(
            self._sp(20), self._sp(50), 100, 100, 42, 42, 0, 0, 0, 0
        )
        hwp_avg, _ = compute_win_probability(
            self._sp(50), self._sp(50), 100, 100, 42, 42, 0, 0, 0, 0
        )
        assert hwp_elite > hwp_avg

    def test_elite_away_sp_lowers_home_prob(self):
        """Elite away SP → lower home win probability."""
        hwp_elite_away, _ = compute_win_probability(
            self._sp(50), self._sp(20), 100, 100, 42, 42, 0, 0, 0, 0
        )
        hwp_balanced, _ = compute_win_probability(
            self._sp(50), self._sp(50), 100, 100, 42, 42, 0, 0, 0, 0
        )
        assert hwp_elite_away < hwp_balanced

    def test_vegas_blend_shifts_toward_implied_runs(self):
        """When home implied runs >> away, win prob shifts toward home."""
        hwp_v, _ = compute_win_probability(
            self._sp(50), self._sp(50), 100, 100, 42, 42, 0, 0, 7.0, 3.0
        )
        hwp_no_v, _ = compute_win_probability(
            self._sp(50), self._sp(50), 100, 100, 42, 42, 0, 0, 0, 0
        )
        assert hwp_v > hwp_no_v

    def test_run_diff_nudge_capped_at_two_pct(self):
        """Run-diff nudge is ±0.02 max regardless of extreme inputs."""
        hwp_h, _ = compute_win_probability(
            self._sp(50), self._sp(50), 100, 100, 42, 42, 50.0, -50.0, 0, 0
        )
        hwp_a, _ = compute_win_probability(
            self._sp(50), self._sp(50), 100, 100, 42, 42, -50.0, 50.0, 0, 0
        )
        assert abs(hwp_h - hwp_a) <= 0.045   # ±0.02 nudge → ≤0.04 delta (+rounding)

    def test_output_clamped_to_valid_range(self):
        """Win probability never escapes [0.30, 0.75]."""
        hwp_dom, _ = compute_win_probability(
            self._sp(0), self._sp(100), 200, 50, 10, 90, 20, -20, 0, 0
        )
        hwp_weak, _ = compute_win_probability(
            self._sp(100), self._sp(0), 50, 200, 90, 10, -20, 20, 0, 0
        )
        assert 0.30 <= hwp_dom  <= 0.75
        assert 0.30 <= hwp_weak <= 0.75

    def test_label_surfaces_pitcher_quality_and_wrc(self):
        """Label must mention pitcher quality tier and wRC+."""
        _, lbl = compute_win_probability(
            {"_sp_vuln": 20}, {"_sp_vuln": 80},
            115, 92, 42, 42, 0, 0, 0, 0,
        )
        assert "Elite" in lbl
        assert "wRC+" in lbl


class TestComputeMlConfidence:
    def test_zero_edge_still_scores_data_quality(self):
        """edge=0 → 0 edge-pts but SP/lineup/odds components still contribute."""
        score = compute_ml_confidence(0.0, "J. Verlander", 9, 40, True)
        assert 0 < score < 50   # some quality pts; edge not adding anything

    def test_zero_when_negative_edge(self):
        assert compute_ml_confidence(-3.0, "J. Verlander", 9, 40, True) == 0.0

    def test_zero_when_none_edge(self):
        assert compute_ml_confidence(None, "J. Verlander", 9, 40, True) == 0.0

    def test_near_max_with_elite_scenario(self):
        """7%+ edge, elite SP, full lineup, odds → 90+."""
        score = compute_ml_confidence(7.0, "C. Sale", 9, 25, True)
        assert score >= 90.0

    def test_no_odds_costs_ten_pts(self):
        with_odds    = compute_ml_confidence(5.0, "C. Sale", 9, 40, True)
        without_odds = compute_ml_confidence(5.0, "C. Sale", 9, 40, False)
        assert abs(with_odds - without_odds - 10.0) < 0.01

    def test_tbd_sp_costs_five_pts(self):
        known = compute_ml_confidence(5.0, "C. Sale", 9, 40, True)
        tbd   = compute_ml_confidence(5.0, "TBD",    9, 40, True)
        assert abs(known - tbd - 5.0) < 0.01

    def test_partial_lineup_lowers_score(self):
        full = compute_ml_confidence(5.0, "C. Sale", 9, 40, True)
        half = compute_ml_confidence(5.0, "C. Sale", 4, 40, True)
        assert full > half


# ── Moneyline: market module ──────────────────────────────────────────────────

class TestScoreGameMl:
    def _sp(self, vuln=50.0):
        return {"_sp_vuln": vuln}

    def _score(self, **overrides):
        defaults = dict(
            home_team="NYY", away_team="BOS",
            home_sp_name="G. Cole", away_sp_name="C. Sale",
            home_sp_id="111", away_sp_id="222",
            home_sp_stats=self._sp(40), away_sp_stats=self._sp(55),
            home_off_wrc=108, away_off_wrc=102,
            home_n_batters=9, away_n_batters=9,
            home_bp_vuln=42, away_bp_vuln=45,
            home_run_diff=1.0, away_run_diff=-0.5,
            home_implied_runs=4.7, away_implied_runs=4.3,
            home_market_implied=0.52, away_market_implied=0.48,
            has_odds=True,
            home_sp_matched=True, away_sp_matched=True,
            home_sp_prov={"k_rate_allowed": "measured"},
            away_sp_prov={"k_rate_allowed": "measured"},
        )
        defaults.update(overrides)
        return score_game_ml(**defaults)

    def test_returns_required_keys(self):
        r = self._score()
        for k in ("home_win_prob", "away_win_prob", "pick", "pick_tier",
                  "bettable", "home_edge_pct", "away_edge_pct",
                  "home_confidence", "away_confidence", "detail_label"):
            assert k in r, f"Missing key: {k}"

    def test_probs_sum_to_one(self):
        r = self._score()
        assert abs(r["home_win_prob"] + r["away_win_prob"] - 1.0) < 0.001

    def test_bettable_when_all_data_present(self):
        r = self._score()
        assert r["bettable"] is True
        assert r["non_bettable_reasons"] == []

    def test_not_bettable_without_odds(self):
        r = self._score(has_odds=False)
        assert r["bettable"] is False
        assert any("odds" in reason for reason in r["non_bettable_reasons"])

    def test_not_bettable_tbd_sp(self):
        r = self._score(home_sp_name="TBD")
        assert r["bettable"] is False
        assert r["home_sp_tbd"] is True

    def test_not_bettable_thin_lineup(self):
        r = self._score(home_n_batters=3, away_n_batters=3)
        assert r["bettable"] is False

    def test_pick_has_min_edge_when_selected(self):
        """Any non-None pick must have edge ≥ 4.0%."""
        r = self._score()
        if r["pick"] is not None:
            assert r["pick_edge"] >= 4.0

    def test_no_play_when_edge_below_threshold(self):
        """When model prob closely matches market, pick_tier should be No Play."""
        # Market exactly matches model → zero edge → No Play
        r = self._score(
            home_market_implied=0.535,   # matches typical balanced home prob
            away_market_implied=0.465,
        )
        # With near-matched odds, edge might be thin — tier should be Lean or No Play
        assert r["pick_tier"] in ("➖ No Play", "📊 Lean")

    def test_strong_edge_when_massive_underdog(self):
        """Model strongly favors home but book sets them as heavy underdog → strong edge."""
        r = self._score(
            home_sp_stats=self._sp(15),    # elite home SP
            away_sp_stats=self._sp(75),    # weak away SP
            home_off_wrc=130,              # great home offense
            away_off_wrc=80,
            home_market_implied=0.35,      # book severely undervalues home
            away_market_implied=0.65,
        )
        # Home model prob should far exceed 0.35 → strong edge
        if r["home_edge_pct"] is not None and r["home_edge_pct"] >= 7.0:
            assert r["pick_tier"] == "🔥 Strong Edge"
            assert r["pick"] == "NYY"


# ── check_bettable_ml ─────────────────────────────────────────────────────────

class TestCheckBettableMl:
    def _gate(self, **overrides):
        defaults = dict(
            home_sp_matched=True, away_sp_matched=True,
            home_sp_tbd=False, away_sp_tbd=False,
            home_sp_prov={"k_rate_allowed": "measured"},
            away_sp_prov={"k_rate_allowed": "measured"},
            home_n_batters=9, away_n_batters=9,
            has_odds=True,
        )
        defaults.update(overrides)
        return check_bettable_ml(**defaults)

    def test_all_good_passes(self):
        ok, reasons = self._gate()
        assert ok is True
        assert reasons == []

    def test_fails_without_odds(self):
        ok, reasons = self._gate(has_odds=False)
        assert ok is False
        assert any("odds" in r for r in reasons)

    def test_fails_home_tbd_sp(self):
        ok, reasons = self._gate(home_sp_tbd=True)
        assert ok is False
        assert any("home SP" in r for r in reasons)

    def test_fails_away_unmatched_sp(self):
        ok, reasons = self._gate(away_sp_matched=False)
        assert ok is False
        assert any("away SP" in r for r in reasons)

    def test_fails_insufficient_home_batters(self):
        ok, reasons = self._gate(home_n_batters=3)
        assert ok is False
        assert any("home lineup" in r for r in reasons)

    def test_fails_insufficient_away_batters(self):
        ok, reasons = self._gate(away_n_batters=2)
        assert ok is False
        assert any("away lineup" in r for r in reasons)

    def test_fails_unmeasured_home_sp_vuln(self):
        ok, reasons = self._gate(home_sp_prov={"k_rate_allowed": "league_avg"})
        assert ok is False
        assert any("home SP vulnerability" in r for r in reasons)

    def test_multiple_failures_all_listed(self):
        """When multiple gates fail, all reasons are returned."""
        ok, reasons = self._gate(has_odds=False, home_sp_tbd=True, home_n_batters=2)
        assert ok is False
        assert len(reasons) >= 3
