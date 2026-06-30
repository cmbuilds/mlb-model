"""
Data-contract tests (3.2): verify that the SQLite DB and provenance module
produce expected columns, types, and provenance tags.

Skipped when the DB doesn't exist (e.g. CI without a fetch run).
Run: pytest tests/test_data_contract.py -v
"""

import os
import sqlite3
import pytest

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "mlb_stats.db")
DB_EXISTS = os.path.exists(DB_PATH)

skip_no_db = pytest.mark.skipif(not DB_EXISTS, reason="mlb_stats.db not present — run fetch_pipeline.py first")


def _table_exists(name: str) -> bool:
    if not DB_EXISTS:
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        found = bool(conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone())
        conn.close()
        return found
    except Exception:
        return False


BATTER_REQUIRED_COLS = [
    "mlbam_id", "_name",
    "xSLG", "xwOBA", "barrel_batted_rate", "avg_exit_velocity",
    "hard_hit_percent", "sweet_spot_percent", "ev50",
    "SLG", "K%", "bat_speed", "blast_rate", "wRC+",
    "prov_barrel", "prov_ev", "prov_xslg", "prov_xwoba",
    "prov_hh", "prov_bat_speed", "prov_blast", "prov_krate",
    "prov_woba", "prov_slg", "prov_iso",
    "fetch_season", "fetched_at",
]
PITCHER_REQUIRED_COLS = [
    "mlbam_id", "_name", "swstr_pct",
]
GAME_RESULTS_REQUIRED_COLS = [
    "game_pk", "game_date", "player_id", "player_name", "team", "opponent",
    "lineup_slot", "ab", "h", "hr", "tb", "k",
    "hit_o15", "hit_o05", "hit_hr",
]
GAME_SP_REQUIRED_COLS = [
    "game_pk", "game_date", "team", "sp_id", "sp_name",
    "sp_k", "sp_ip", "hit_k5", "hit_k6",
]
PROVENANCE_VALUES = {"measured", "proxy", "league_avg"}


@skip_no_db
class TestBatterTable:
    @pytest.fixture(scope="class")
    @classmethod
    def conn(cls):
        c = sqlite3.connect(DB_PATH)
        yield c
        c.close()

    def test_table_exists(self, conn):
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        assert "batter_stats" in tables

    def test_required_columns_present(self, conn):
        cur = conn.execute("SELECT * FROM batter_stats LIMIT 1")
        cols = [d[0] for d in cur.description]
        missing = [c for c in BATTER_REQUIRED_COLS if c not in cols]
        assert not missing, f"Missing batter columns: {missing}"

    def test_minimum_row_count(self, conn):
        n = conn.execute("SELECT COUNT(*) FROM batter_stats").fetchone()[0]
        assert n >= 200, f"Expected ≥200 batters, got {n}"

    def test_provenance_values_valid(self, conn):
        for col in ("prov_barrel", "prov_ev", "prov_xslg", "prov_krate"):
            rows = conn.execute(f"SELECT DISTINCT [{col}] FROM batter_stats WHERE [{col}] IS NOT NULL").fetchall()
            for (val,) in rows:
                assert val in PROVENANCE_VALUES, f"Invalid provenance '{val}' in {col}"

    def test_mlbam_id_not_null(self, conn):
        nulls = conn.execute("SELECT COUNT(*) FROM batter_stats WHERE mlbam_id IS NULL").fetchone()[0]
        assert nulls == 0, f"{nulls} rows have NULL mlbam_id"

    def test_slg_in_sane_range(self, conn):
        cur = conn.execute("SELECT xSLG FROM batter_stats WHERE xSLG IS NOT NULL")
        slgs = [r[0] for r in cur.fetchall()]
        assert slgs, "No xSLG values found"
        assert all(0.0 <= s <= 1.5 for s in slgs), \
            f"xSLG out of range: min={min(slgs):.3f} max={max(slgs):.3f}"

    def test_krate_measured_coverage(self, conn):
        total = conn.execute("SELECT COUNT(*) FROM batter_stats").fetchone()[0]
        measured = conn.execute(
            "SELECT COUNT(*) FROM batter_stats WHERE prov_krate = 'measured'"
        ).fetchone()[0]
        pct = measured / total if total > 0 else 0
        # MLB API returns only qualified batters (~150/580 = 26%). Document, don't fail.
        assert pct >= 0.15, f"K% measured coverage {pct:.0%} below 15% floor — data source may have failed"

    def test_xslg_measured_coverage(self, conn):
        total = conn.execute("SELECT COUNT(*) FROM batter_stats").fetchone()[0]
        measured = conn.execute(
            "SELECT COUNT(*) FROM batter_stats WHERE prov_xslg = 'measured'"
        ).fetchone()[0]
        pct = measured / total if total > 0 else 0
        assert pct >= 0.70, f"xSLG measured coverage {pct:.0%} below 70% — Savant fetch may have failed"


@skip_no_db
class TestPitcherTable:
    @pytest.fixture(scope="class")
    @classmethod
    def conn(cls):
        c = sqlite3.connect(DB_PATH)
        yield c
        c.close()

    def test_table_exists(self, conn):
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        assert "pitcher_stats" in tables

    def test_minimum_row_count(self, conn):
        n = conn.execute("SELECT COUNT(*) FROM pitcher_stats").fetchone()[0]
        assert n >= 100, f"Expected ≥100 pitchers, got {n}"

    def test_required_columns_present(self, conn):
        cur = conn.execute("SELECT * FROM pitcher_stats LIMIT 1")
        cols = [d[0] for d in cur.description]
        missing = [c for c in PITCHER_REQUIRED_COLS if c not in cols]
        assert not missing, f"Missing pitcher columns: {missing}"

    def test_swstr_pct_coverage(self, conn):
        """SwStr% should be populated for ≥25% of pitchers (pitch-arsenal endpoint)."""
        total = conn.execute("SELECT COUNT(*) FROM pitcher_stats").fetchone()[0]
        measured = conn.execute(
            "SELECT COUNT(*) FROM pitcher_stats WHERE swstr_pct > 0.02"
        ).fetchone()[0]
        pct = measured / total if total > 0 else 0
        assert pct >= 0.25, f"swstr_pct coverage {pct:.0%} below 25% — pitch-arsenal fetch may have failed"

    def test_swstr_pct_in_sane_range(self, conn):
        rows = conn.execute(
            "SELECT swstr_pct FROM pitcher_stats WHERE swstr_pct IS NOT NULL AND swstr_pct > 0"
        ).fetchall()
        vals = [r[0] for r in rows]
        if vals:
            assert all(0.0 < v < 1.0 for v in vals), \
                f"swstr_pct out of [0,1]: min={min(vals):.3f} max={max(vals):.3f}"


@skip_no_db
class TestGameResultsTable:
    @pytest.fixture(scope="class")
    @classmethod
    def conn(cls):
        c = sqlite3.connect(DB_PATH)
        yield c
        c.close()

    def test_table_exists(self, conn):
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        assert "game_results" in tables

    def test_required_columns_present(self, conn):
        cur = conn.execute("SELECT * FROM game_results LIMIT 1")
        cols = [d[0] for d in cur.description]
        missing = [c for c in GAME_RESULTS_REQUIRED_COLS if c not in cols]
        assert not missing, f"Missing game_results columns: {missing}"

    def test_hit_hr_column_populated(self, conn):
        """hit_hr must be 0 or 1, matching hr > 0 for every row."""
        n_total  = conn.execute("SELECT COUNT(*) FROM game_results").fetchone()[0]
        n_hr_pos = conn.execute("SELECT COUNT(*) FROM game_results WHERE hr > 0").fetchone()[0]
        n_hit_hr = conn.execute("SELECT COUNT(*) FROM game_results WHERE hit_hr = 1").fetchone()[0]
        assert n_hit_hr == n_hr_pos, \
            f"hit_hr mismatch: {n_hit_hr} flagged but {n_hr_pos} have hr>0 (total={n_total})"

    def test_hit_hr_binary(self, conn):
        invalid = conn.execute(
            "SELECT COUNT(*) FROM game_results WHERE hit_hr NOT IN (0, 1)"
        ).fetchone()[0]
        assert invalid == 0, f"{invalid} rows with non-binary hit_hr"

    def test_minimum_row_count(self, conn):
        n = conn.execute("SELECT COUNT(*) FROM game_results").fetchone()[0]
        assert n >= 10000, f"Expected ≥10k game_results rows, got {n}"


skip_no_sp_results = pytest.mark.skipif(
    not DB_EXISTS, reason="mlb_stats.db not present"
)


@skip_no_db
class TestGameSpResultsTable:
    @pytest.fixture(scope="class")
    @classmethod
    def conn(cls):
        c = sqlite3.connect(DB_PATH)
        yield c
        c.close()

    def test_table_exists(self, conn):
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        assert "game_sp_results" in tables, \
            "game_sp_results missing — run data/backtest_sp_k_fetcher.py"

    def test_required_columns_present(self, conn):
        cur = conn.execute("SELECT * FROM game_sp_results LIMIT 1")
        cols = [d[0] for d in cur.description]
        missing = [c for c in GAME_SP_REQUIRED_COLS if c not in cols]
        assert not missing, f"Missing game_sp_results columns: {missing}"

    def test_minimum_row_count(self, conn):
        n = conn.execute("SELECT COUNT(*) FROM game_sp_results").fetchone()[0]
        assert n >= 1000, f"Expected ≥1k SP-game rows, got {n}"

    def test_hit_k5_matches_sp_k(self, conn):
        """hit_k5 must be 1 exactly when sp_k >= 5."""
        mismatch = conn.execute(
            "SELECT COUNT(*) FROM game_sp_results "
            "WHERE hit_k5 != CASE WHEN sp_k >= 5 THEN 1 ELSE 0 END"
        ).fetchone()[0]
        assert mismatch == 0, f"{mismatch} rows where hit_k5 doesn't match sp_k >= 5"

    def test_hit_k6_matches_sp_k(self, conn):
        """hit_k6 must be 1 exactly when sp_k >= 6."""
        mismatch = conn.execute(
            "SELECT COUNT(*) FROM game_sp_results "
            "WHERE hit_k6 != CASE WHEN sp_k >= 6 THEN 1 ELSE 0 END"
        ).fetchone()[0]
        assert mismatch == 0, f"{mismatch} rows where hit_k6 doesn't match sp_k >= 6"

    def test_sp_k_in_sane_range(self, conn):
        max_k = conn.execute("SELECT MAX(sp_k) FROM game_sp_results").fetchone()[0]
        assert max_k is not None and max_k <= 20, f"Implausible sp_k max: {max_k}"


GAME_OUTCOMES_REQUIRED_COLS = [
    "game_pk", "game_date", "home_team", "away_team",
    "home_score", "away_score", "winning_team",
]

skip_no_outcomes = pytest.mark.skipif(
    not _table_exists("game_outcomes"),
    reason="game_outcomes table not present — run data/backtest_game_outcomes_fetcher.py first",
)


@skip_no_outcomes
class TestGameOutcomesTable:
    @pytest.fixture(scope="class")
    @classmethod
    def conn(cls):
        c = sqlite3.connect(DB_PATH)
        yield c
        c.close()

    def test_table_exists(self, conn):
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        assert "game_outcomes" in tables, \
            "game_outcomes missing — run data/backtest_game_outcomes_fetcher.py"

    def test_required_columns_present(self, conn):
        cur = conn.execute("SELECT * FROM game_outcomes LIMIT 1")
        cols = [d[0] for d in cur.description]
        missing = [c for c in GAME_OUTCOMES_REQUIRED_COLS if c not in cols]
        assert not missing, f"Missing game_outcomes columns: {missing}"

    def test_minimum_row_count(self, conn):
        n = conn.execute("SELECT COUNT(*) FROM game_outcomes").fetchone()[0]
        assert n >= 500, f"Expected ≥500 game outcomes, got {n}"

    def test_winning_team_values(self, conn):
        invalid = conn.execute(
            "SELECT COUNT(*) FROM game_outcomes WHERE winning_team NOT IN ('home','away','tie')"
        ).fetchone()[0]
        assert invalid == 0, f"{invalid} rows with invalid winning_team value"

    def test_score_consistency(self, conn):
        """home_score > away_score ↔ winning_team = 'home', and vice versa."""
        bad_home = conn.execute(
            "SELECT COUNT(*) FROM game_outcomes "
            "WHERE home_score > away_score AND winning_team != 'home'"
        ).fetchone()[0]
        bad_away = conn.execute(
            "SELECT COUNT(*) FROM game_outcomes "
            "WHERE away_score > home_score AND winning_team != 'away'"
        ).fetchone()[0]
        assert bad_home == 0 and bad_away == 0, \
            f"Score/winner mismatch: {bad_home} home, {bad_away} away"


@skip_no_db
class TestFetchLog:
    @pytest.fixture(scope="class")
    @classmethod
    def conn(cls):
        c = sqlite3.connect(DB_PATH)
        yield c
        c.close()

    def test_fetch_log_exists(self, conn):
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        assert "fetch_log" in tables

    def test_fetch_log_has_entries(self, conn):
        n = conn.execute("SELECT COUNT(*) FROM fetch_log").fetchone()[0]
        assert n > 0, "fetch_log is empty — pipeline may not have run"

    def test_last_fetch_recorded(self, conn):
        row = conn.execute(
            "SELECT fetched_at, batter_rows, pitcher_rows FROM fetch_log ORDER BY fetched_at DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        _, batter_rows, pitcher_rows = row
        assert batter_rows > 100, f"fetch_log shows only {batter_rows} batter rows"
        assert pitcher_rows > 50, f"fetch_log shows only {pitcher_rows} pitcher rows"


class TestProvenanceFunctions:
    """Test the pure provenance functions — no DB needed."""

    def test_dq_score_all_measured(self):
        from data.provenance import compute_data_quality_score
        # Keys must match what compute_data_quality_score actually reads (see provenance.py)
        bat = {k: "measured" for k in ("k_rate", "woba", "slg_proxy", "hard_hit_rate", "barrel_rate", "iso_proxy")}
        pit = {k: "measured" for k in ("k_rate_allowed", "hard_hit_allowed")}
        score = compute_data_quality_score(bat, pit, lineup_confirmed=True, sp_known=True, hand_real=True)
        assert score == 100, f"All measured should be 100, got {score}"

    def test_dq_score_all_proxy(self):
        from data.provenance import compute_data_quality_score
        bat = {k: "proxy" for k in ("barrel", "ev", "xslg", "xwoba", "hh", "bat_speed", "blast", "krate")}
        pit = {k: "proxy" for k in ("k_rate", "hard_hit", "era")}
        score = compute_data_quality_score(bat, pit, lineup_confirmed=False, sp_known=False, hand_real=False)
        assert score == 0, f"All proxy/unknown should be 0, got {score}"

    def test_dq_score_partial(self):
        from data.provenance import compute_data_quality_score
        bat = {"barrel": "measured", "ev": "measured", "xslg": "measured",
               "xwoba": "proxy", "hh": "proxy", "bat_speed": "league_avg",
               "blast": "league_avg", "krate": "measured"}
        pit = {"k_rate": "measured", "hard_hit": "proxy", "era": "measured"}
        score = compute_data_quality_score(bat, pit, lineup_confirmed=True, sp_known=True, hand_real=True)
        assert 0 < score < 100

    def test_bettable_gate_all_measured(self):
        from data.provenance import check_bettable_tb
        # Keys must match what check_bettable_tb actually reads (see provenance.py)
        bat = {k: "measured" for k in ("k_rate", "slg_proxy", "woba", "hard_hit_rate")}
        pit = {}  # pitcher_matched=True is what matters for gate; no prov keys checked
        ok, reasons = check_bettable_tb(bat, pit, True, True, True, True, True)
        assert ok, f"All measured should be bettable, reasons: {reasons}"
        assert reasons == []

    def test_bettable_gate_missing_sp(self):
        from data.provenance import check_bettable_tb
        bat = {k: "measured" for k in ("barrel", "ev", "xslg", "xwoba", "hh", "krate")}
        pit = {k: "measured" for k in ("k_rate", "hard_hit", "era")}
        ok, reasons = check_bettable_tb(bat, pit, True, True, True, sp_known=False, hand_real=True)
        assert not ok, "Unknown SP should block bettable gate"
        assert any("SP" in r or "pitcher" in r.lower() for r in reasons), f"Reason missing SP context: {reasons}"

    def test_bettable_gate_unmatched_batter(self):
        from data.provenance import check_bettable_tb
        bat = {k: "league_avg" for k in ("barrel", "ev", "xslg", "xwoba", "hh", "krate")}
        pit = {k: "measured" for k in ("k_rate", "hard_hit", "era")}
        ok, reasons = check_bettable_tb(bat, pit, batter_matched=False, pitcher_matched=True,
                                        lineup_confirmed=True, sp_known=True, hand_real=True)
        assert not ok
