"""Tests for dfs/lineup_log.py — SQLite persistence."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tempfile
import pytest
from dfs.lineup_log import log_lineups, get_lineups, record_result


def _fake_lineup(site="fd", proj=38.5, salary=32_400):
    return {
        "site": site,
        "contest": "test",
        "total_proj": proj,
        "total_salary": salary,
        "players": [
            {"name": "Aaron Judge", "position": "OF", "team": "NYY",
             "salary": 3700, "fppg": 18.5},
            {"name": "Gerrit Cole", "position": "P", "team": "NYY",
             "salary": 9800, "fppg": 32.1},
        ],
    }


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "test_lineups.db")


# ── log_lineups ───────────────────────────────────────────────────────────────
def test_log_returns_row_id(db):
    row_id = log_lineups([_fake_lineup()], site="fd",
                         contest_label="single-entry", slate_date="2026-07-01", db_path=db)
    assert isinstance(row_id, int)
    assert row_id >= 1


def test_log_empty_raises(db):
    with pytest.raises(ValueError, match="empty"):
        log_lineups([], site="fd", contest_label="test", slate_date="2026-07-01", db_path=db)


def test_log_persists_count(db):
    lineups = [_fake_lineup(proj=40.0), _fake_lineup(proj=38.0)]
    log_lineups(lineups, site="fd", contest_label="multi-entry GPP",
                slate_date="2026-07-01", db_path=db)
    rows = get_lineups(db_path=db)
    assert len(rows) == 1
    assert rows[0]["n_lineups"] == 2


def test_log_avg_proj_stored(db):
    lineups = [_fake_lineup(proj=40.0), _fake_lineup(proj=38.0)]
    log_lineups(lineups, site="fd", contest_label="gpp",
                slate_date="2026-07-01", db_path=db)
    rows = get_lineups(db_path=db)
    assert rows[0]["total_proj"] == pytest.approx(39.0, abs=0.1)


def test_log_players_json_decoded(db):
    log_lineups([_fake_lineup()], site="fd", contest_label="se",
                slate_date="2026-07-01", db_path=db)
    rows = get_lineups(db_path=db)
    players = rows[0]["players_json"]
    assert isinstance(players, list)
    assert len(players) == 1      # 1 lineup → 1 inner list
    assert len(players[0]) == 2   # 2 players in the lineup


def test_log_notes_stored(db):
    log_lineups([_fake_lineup()], site="fd", contest_label="se",
                slate_date="2026-07-01", notes="$40K Battery", db_path=db)
    rows = get_lineups(db_path=db)
    assert rows[0]["notes"] == "$40K Battery"


# ── get_lineups filters ───────────────────────────────────────────────────────
def test_get_filter_by_site(db):
    log_lineups([_fake_lineup(site="fd")], site="fd",
                contest_label="se", slate_date="2026-07-01", db_path=db)
    log_lineups([_fake_lineup(site="dk")], site="dk",
                contest_label="se", slate_date="2026-07-01", db_path=db)
    fd_rows = get_lineups(site="fd", db_path=db)
    assert len(fd_rows) == 1
    assert fd_rows[0]["site"] == "fd"


def test_get_filter_by_date(db):
    log_lineups([_fake_lineup()], site="fd", contest_label="se",
                slate_date="2026-07-01", db_path=db)
    log_lineups([_fake_lineup()], site="fd", contest_label="se",
                slate_date="2026-07-02", db_path=db)
    rows = get_lineups(slate_date="2026-07-01", db_path=db)
    assert len(rows) == 1
    assert rows[0]["slate_date"] == "2026-07-01"


def test_get_newest_first(db):
    for i in range(3):
        log_lineups([_fake_lineup(proj=30.0 + i)], site="fd",
                    contest_label="se", slate_date="2026-07-01", db_path=db)
    rows = get_lineups(db_path=db)
    assert rows[0]["id"] > rows[1]["id"] > rows[2]["id"]


# ── record_result ─────────────────────────────────────────────────────────────
def test_record_result_updates_row(db):
    row_id = log_lineups([_fake_lineup()], site="fd", contest_label="se",
                         slate_date="2026-07-01", db_path=db)
    record_result(row_id, result_pts=152.4, result_rank=234, db_path=db)
    rows = get_lineups(db_path=db)
    assert rows[0]["result_pts"] == pytest.approx(152.4)
    assert rows[0]["result_rank"] == 234


def test_result_starts_null(db):
    log_lineups([_fake_lineup()], site="fd", contest_label="se",
                slate_date="2026-07-01", db_path=db)
    rows = get_lineups(db_path=db)
    assert rows[0]["result_pts"] is None
    assert rows[0]["result_rank"] is None
