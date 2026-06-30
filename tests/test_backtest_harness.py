"""
Tests for 3.3: backtest harness infrastructure.
Tests the result-fetching schema and scoring loop without network calls.
"""

import os
import sqlite3
import tempfile
import pytest
from unittest.mock import patch


class TestGameResultsSchema:
    """Verify the game_results table schema can be created and written to."""

    def test_schema_creation(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            from data.backtest_fetcher import _ensure_schema
            conn = sqlite3.connect(db_path)
            _ensure_schema(conn)
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            assert "game_results" in tables
            conn.close()
        finally:
            os.unlink(db_path)

    def test_schema_columns(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            from data.backtest_fetcher import _ensure_schema
            conn = sqlite3.connect(db_path)
            _ensure_schema(conn)
            cols = [d[0] for d in conn.execute(
                "SELECT * FROM game_results LIMIT 0"
            ).description]
            for required in ("game_pk", "player_id", "tb", "hit_o15", "hit_o05", "ab"):
                assert required in cols, f"Missing column: {required}"
            conn.close()
        finally:
            os.unlink(db_path)

    def test_write_results(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            from data.backtest_fetcher import _ensure_schema, write_results
            conn = sqlite3.connect(db_path)
            _ensure_schema(conn)
            rows = [
                {"game_pk": 1001, "game_date": "2026-04-01", "player_id": 12345,
                 "player_name": "Test Batter", "team": "NYY", "opponent": "BOS",
                 "lineup_slot": 3, "ab": 4, "h": 2, "doubles": 1, "triples": 0,
                 "hr": 0, "tb": 3, "bb": 0, "k": 1, "rbi": 1,
                 "hit_o15": 1, "hit_o05": 1, "hit_hr": 0},
            ]
            n = write_results(conn, rows)
            assert n == 1
            count = conn.execute("SELECT COUNT(*) FROM game_results").fetchone()[0]
            assert count == 1
            row = conn.execute("SELECT tb, hit_o15 FROM game_results WHERE player_id=12345").fetchone()
            assert row[0] == 3   # tb
            assert row[1] == 1   # hit_o15
            conn.close()
        finally:
            os.unlink(db_path)

    def test_upsert_on_conflict(self):
        """Duplicate game_pk/player_id should replace, not duplicate."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            from data.backtest_fetcher import _ensure_schema, write_results
            conn = sqlite3.connect(db_path)
            _ensure_schema(conn)
            base_row = {"game_pk": 1001, "game_date": "2026-04-01", "player_id": 99,
                        "player_name": "X", "team": "A", "opponent": "B",
                        "lineup_slot": 1, "ab": 3, "h": 1, "doubles": 0, "triples": 0,
                        "hr": 0, "tb": 1, "bb": 0, "k": 1, "rbi": 0,
                        "hit_o15": 0, "hit_o05": 1, "hit_hr": 0}
            write_results(conn, [base_row])
            # Update same key
            updated = {**base_row, "tb": 4, "hit_o15": 1}
            write_results(conn, [updated])
            count = conn.execute("SELECT COUNT(*) FROM game_results").fetchone()[0]
            tb = conn.execute("SELECT tb FROM game_results WHERE player_id=99").fetchone()[0]
            assert count == 1, "Should have exactly 1 row after upsert"
            assert tb == 4, f"Should have updated tb to 4, got {tb}"
            conn.close()
        finally:
            os.unlink(db_path)


class TestBacktestScoringKernel:
    """Test that the backtest scoring loop produces valid scores."""

    def _make_batter_stat_row(self, **overrides):
        base = {
            "mlbam_id": 1001,
            "xSLG": 0.450, "barrel_batted_rate": 0.100, "hard_hit_percent": 0.400,
            "K%": 0.200, "ISO": 0.200, "wRC+": 120.0, "xwOBA": 0.360,
            "ev50": 0.0, "bat_speed": 0.0, "blast_rate": 0.0,
            "prov_xslg": "measured", "prov_krate": "measured", "prov_hh": "measured",
            "prov_woba": "measured", "prov_barrel": "measured", "prov_slg": "measured",
            "prov_iso": "measured", "prov_xwoba": "measured",
            "fetch_season": 2026,
        }
        base.update(overrides)
        return base

    def test_stat_row_to_scoring_input(self):
        from backtest import _stat_row_to_scoring_input
        row = self._make_batter_stat_row()
        inp = _stat_row_to_scoring_input(row)
        assert "slg_proxy" in inp
        assert "barrel_rate" in inp
        assert "_provenance" in inp
        assert isinstance(inp["k_rate"], float)
        assert 0.0 <= inp["k_rate"] <= 1.0

    def test_scoring_produces_valid_score(self):
        from backtest import _stat_row_to_scoring_input
        from markets.tb_o15 import score_one_batter

        stat_row = self._make_batter_stat_row()
        batter_input = _stat_row_to_scoring_input(stat_row)
        pitcher_input = {
            "k_rate_allowed": 0.228, "hard_hit_allowed": 0.370,
            "barrel_allowed": 0.070, "era": 4.20, "fip": 4.20, "whip": 1.30,
            "data_source": "league_avg",
            "_provenance": {"k_rate_allowed": "league_avg", "hard_hit_allowed": "league_avg"},
        }

        result = score_one_batter(
            name="Test Player", player_id="1001",
            team="NYY", opponent="BOS", game_pk="1001",
            batter_hand="R", hand_real=False,
            sp_hand="R", sp_name="", sp_id="",
            lineup_slot=3, lineup_confirmed=True, batter_position="",
            park_team="NYY",
            batter_stats=batter_input, pitcher_stats=pitcher_input,
            recent_form={}, bvp_data={},
            weather={"is_dome": True},
            implied=0.0, prop_implied=None,
            team_bullpen_scores={},
            proxy_mode=False,
        )
        assert "score" in result
        assert 0 <= result["score"] <= 100
        assert "prob" in result
        assert 0 < result["prob"] < 1
        assert "tier" in result
        assert "bettable" in result

    def test_calibration_report_format(self):
        """Verify the calibration report runs without error on synthetic data."""
        from backtest import print_calibration_report
        import io
        import sys

        fake_results = {
            "scored": 100,
            "unmatched": 10,
            "proxy_count": 20,
            "buckets": {
                "<50":   [0, 0, 1, 0, 1],
                "50-59": [0, 1, 1, 0, 1, 0, 1, 0],
                "60-69": [1, 1, 0, 1, 1, 0],
                "70-79": [1, 1, 1, 0, 1],
                "80+":   [1, 1, 1, 1],
            },
        }
        # Should not raise
        captured = io.StringIO()
        sys.stdout = captured
        try:
            print_calibration_report(fake_results)
        finally:
            sys.stdout = sys.__stdout__
        output = captured.getvalue()
        assert "CALIBRATION REPORT" in output
        assert "80+" in output
        assert "OVERALL" in output
