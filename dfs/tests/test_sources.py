"""Tests for dfs/sources/ — salary parsing, model adapter, DK rules."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from dfs.sources.salaries import parse_fd_salary_csv, parse_dk_salary_csv
from dfs.sources.api_external import SourceError, BluecollarDFSProjections, SourceKind
from dfs.lib.dk_rules import compute_dk_projection, dk_salary_csv_to_players, DK_SALARY_CAP


# ── FD salary CSV ─────────────────────────────────────────────────────────────
FD_CSV = """FPPG,Nickname,First Name,Last Name,ID,Position,Team,Salary,Game,Opponent,Weather,Injury Indicator
18.5,Aaron Judge,Aaron,Judge,111,OF,NYY,3700,NYY@BOS,BOS,Clear,,
12.3,Jose Ramirez,Jose,Ramirez,222,3B,CLE,3200,CLE@DET,DET,Clear,,
"""

def test_parse_fd_csv_basic():
    rows = parse_fd_salary_csv(FD_CSV)
    assert len(rows) == 2
    assert rows[0]["name"] == "Aaron Judge"
    assert rows[0]["salary"] == 3700
    assert rows[0]["position"] == "OF"
    assert rows[0]["site"] == "fd"

def test_parse_fd_csv_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        parse_fd_salary_csv("")

def test_parse_fd_csv_wrong_format_raises():
    with pytest.raises(ValueError):
        parse_fd_salary_csv("col1,col2\nfoo,bar\n")


# ── DK salary CSV ─────────────────────────────────────────────────────────────
DK_CSV = """Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame
SP,Gerrit Cole (111),Gerrit Cole,111,SP,9800,NYY@BOS 07:05PM ET,NYY,22.5
OF,Aaron Judge (222),Aaron Judge,222,OF,6800,NYY@BOS 07:05PM ET,NYY,18.5
"""

def test_parse_dk_csv_basic():
    rows = parse_dk_salary_csv(DK_CSV)
    assert len(rows) == 2
    assert rows[0]["name"] == "Gerrit Cole"
    assert rows[0]["salary"] == 9800
    assert rows[0]["site"] == "dk"

def test_parse_dk_csv_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        parse_dk_salary_csv("")

def test_dk_salary_cap():
    assert DK_SALARY_CAP == 50_000


# ── DK rules ─────────────────────────────────────────────────────────────────
def _play(slot=3, implied=5.0, barrel=0.10, k_rate=0.228, iso=0.165, xslg=0.400):
    return {
        "name": "Test Player", "team": "HOU", "opponent": "NYY",
        "lineup_slot": slot, "implied_total": implied,
        "barrel_rate": barrel, "k_rate": k_rate, "bb_rate": 0.082,
        "iso": iso, "xslg": xslg,
        "park": "HOU", "weather": {"is_dome": True},
        "sub_pitcher": 55.0,
    }

def test_dk_projection_returns_float():
    pts = compute_dk_projection(_play())
    assert isinstance(pts, float)
    assert pts > 0

def test_dk_projection_slot_effect():
    pts_leadoff = compute_dk_projection(_play(slot=1))
    pts_bottom  = compute_dk_projection(_play(slot=8))
    assert pts_leadoff > pts_bottom   # more PAs in leadoff

def test_dk_projection_implied_effect():
    pts_low  = compute_dk_projection(_play(implied=3.0))
    pts_high = compute_dk_projection(_play(implied=6.0))
    assert pts_high > pts_low

def test_dk_projection_missing_fields_graceful():
    # Should not crash on empty play dict
    pts = compute_dk_projection({})
    # Either None or 0 — not a crash
    assert pts is None or pts >= 0

def test_dk_csv_to_players():
    rows = [{"Position": "OF", "Name": "Aaron Judge", "Name + ID": "Aaron Judge (1)",
             "Salary": "6800", "TeamAbbrev": "NYY", "Game Info": "NYY@BOS", "AvgPointsPerGame": "18.5", "ID": "1"}]
    players = dk_salary_csv_to_players(rows)
    assert len(players) == 1
    assert players[0]["name"] == "Aaron Judge"
    assert players[0]["salary"] == 6800


# ── BluecollarDFS error path (no key) ────────────────────────────────────────
def test_bluecollardfs_no_key_raises():
    src = BluecollarDFSProjections({})
    with pytest.raises(SourceError, match="missing api_key"):
        src.fetch_projections(site="fd", date="2026-06-30")


# ── model_proj ────────────────────────────────────────────────────────────────
def test_plays_to_fd_projections_basic():
    from dfs.sources.model_proj import plays_to_fd_projections
    play = {
        "name": "Aaron Judge", "team": "NYY", "opponent": "BOS",
        "batter_position": "OF", "batter_hand": "R",
        "lineup_slot": 3, "lineup_confirmed": True,
        "implied_total": 5.0, "park": "NYY",
        "fd_proj": 18.5, "fd_ceiling": 28.0, "fd_floor": 9.0,
        "weather": {"is_dome": False, "wind_effect": "neutral"},
        "_batter_prov": {k: "measured" for k in
                         ("k_rate","slg_proxy","woba","hard_hit_rate","barrel_rate")},
    }
    rows = plays_to_fd_projections([play])
    assert len(rows) == 1
    assert rows[0].player == "Aaron Judge"
    assert rows[0].proj_pts == pytest.approx(18.5)
    assert rows[0].source == "model"

def test_plays_to_fd_empty_raises():
    from dfs.sources.model_proj import plays_to_fd_projections
    from dfs.sources.api_external import SourceError
    with pytest.raises(SourceError):
        plays_to_fd_projections([])


# ── pitchers_from_salary_csv ──────────────────────────────────────────────────
from dfs.sources.salaries import pitchers_from_salary_csv
from dfs.contracts import ConfidenceState, Provenance

_FD_PITCHERS_CSV = """FPPG,Nickname,First Name,Last Name,ID,Position,Team,Salary,Game,Opponent,Weather,Injury Indicator
38.5,Gerrit Cole,Gerrit,Cole,1001,P,NYY,10400,NYY@BOS,BOS,Clear,
22.1,Luis Castillo,Luis,Castillo,1002,P,SEA,7200,SEA@LAA,LAA,Clear,
6.2,Spot Starter,Spot,Starter,1003,P,MIA,4800,MIA@PHI,PHI,Clear,
18.5,Aaron Judge,Aaron,Judge,2001,OF,NYY,3700,NYY@BOS,BOS,Clear,
"""

def test_pitchers_from_fd_csv_returns_only_pitchers():
    salary_rows = parse_fd_salary_csv(_FD_PITCHERS_CSV)
    pitchers = pitchers_from_salary_csv(salary_rows, site="fd")
    assert len(pitchers) == 3
    assert all(p.position == "P" for p in pitchers)

def test_pitchers_confident_above_threshold():
    salary_rows = parse_fd_salary_csv(_FD_PITCHERS_CSV)
    pitchers = pitchers_from_salary_csv(salary_rows, site="fd")
    # Cole (38.5 fppg, $10400) and Castillo (22.1 fppg, $7200) should be CONFIDENT
    by_name = {p.name: p for p in pitchers}
    assert by_name["Gerrit Cole"].state == ConfidenceState.CONFIDENT
    assert by_name["Luis Castillo"].state == ConfidenceState.CONFIDENT

def test_pitcher_flagged_below_threshold():
    salary_rows = parse_fd_salary_csv(_FD_PITCHERS_CSV)
    pitchers = pitchers_from_salary_csv(salary_rows, site="fd")
    by_name = {p.name: p for p in pitchers}
    # Spot Starter: $4800 (below $5K floor) → FLAGGED
    assert by_name["Spot Starter"].state == ConfidenceState.FLAGGED
    assert by_name["Spot Starter"].flagged_reason != ""

def test_pitcher_sources_tagged_site_fppg():
    salary_rows = parse_fd_salary_csv(_FD_PITCHERS_CSV)
    pitchers = pitchers_from_salary_csv(salary_rows, site="fd")
    for p in pitchers:
        assert p.sources_used == ["site_fppg"]
        assert p.source_count == 1

def test_pitcher_ownership_modeled():
    salary_rows = parse_fd_salary_csv(_FD_PITCHERS_CSV)
    pitchers = pitchers_from_salary_csv(salary_rows, site="fd")
    for p in pitchers:
        assert p.own_provenance == Provenance.MODELED

def test_pitcher_opponent_parsed_from_game_string():
    salary_rows = parse_fd_salary_csv(_FD_PITCHERS_CSV)
    pitchers = pitchers_from_salary_csv(salary_rows, site="fd")
    by_name = {p.name: p for p in pitchers}
    # Cole is NYY, game is "NYY@BOS" — opponent should be BOS
    assert by_name["Gerrit Cole"].opponent == "BOS"

def test_pitcher_dk_position_is_sp():
    # DK uses SP not P
    dk_salary_rows = [
        {"name": "Gerrit Cole", "position": "SP", "salary": 9800,
         "fppg": 0.0, "avg_pts": 22.5, "team": "NYY", "opponent": "BOS",
         "game": "NYY@BOS", "site": "dk"},
    ]
    pitchers = pitchers_from_salary_csv(dk_salary_rows, site="dk")
    assert len(pitchers) == 1
    assert pitchers[0].position == "SP"
