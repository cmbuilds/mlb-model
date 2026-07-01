"""Tests for dfs/optimize.py — pydfs wrapper, contest config, guard rails."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from dfs.contracts import ConsensusRow, ConfidenceState, Provenance
from dfs.optimize import (
    build_fd_lineups, CONTEST_SINGLE_ENTRY, CONTEST_MULTI_ENTRY_GPP,
    export_fd_csv,
)
import tempfile


def _row(name, pos="OF", team="NYY", pts=18.5, salary=3500, state=ConfidenceState.CONFIDENT):
    _opp_map = {"NYY": "BOS", "BOS": "NYY", "HOU": "TEX"}
    opp = _opp_map.get(team, "OPP")
    return ConsensusRow(
        name=name, team=team, opponent=opp,
        position=pos, site="fd",
        salary=salary, lineup_slot=3,
        consensus_pts=pts,
        consensus_value=round(pts / (salary/1000), 2),
        model_pts=pts, model_ceiling=pts*1.45, model_floor=pts*0.4,
        state=state, source_count=1, sources_used=["model"],
        flagged_reason="" if state == ConfidenceState.CONFIDENT else "proxy data",
        own_pct=15.0, own_provenance=Provenance.MODELED,
        divergence=0.0, divergence_flag=False,
        bettable=(state == ConfidenceState.CONFIDENT),
        batter_hand="R", sp_name="Gerrit Cole", sp_hand="R",
        park="NYY", implied_total=5.0, game_id="g1",
        score=72.0, hr_score=45.0, dq_score=82,
    )


def _full_board():
    """
    Build a 20-player CONFIDENT board that satisfies FD constraints:
      - 9 roster spots: P, C/1B, 2B, 3B, SS, OF×3, UTIL
      - Max 4 hitters from any one team (FanduelBaseballRosterRule)
      - Three teams (NYY, BOS, HOU) — pydfs FD Baseball requires min 3 teams in lineup
      - All salaries $2500 so any 9-player combo clears $35K cap ($22,500 total)
    """
    # (name, position, team)
    specs = [
        ("Pitcher One",    "P",  "NYY"),
        ("Pitcher Two",    "P",  "BOS"),
        ("Pitcher Three",  "P",  "HOU"),
        ("Catcher One",    "C",  "NYY"),
        ("First One",      "1B", "BOS"),
        ("Second One",     "2B", "HOU"),
        ("Third One",      "3B", "NYY"),
        ("Short One",      "SS", "BOS"),
        ("Outfield One",   "OF", "HOU"),
        ("Outfield Two",   "OF", "NYY"),
        ("Outfield Three", "OF", "BOS"),
        ("Outfield Four",  "OF", "HOU"),
        ("Short Two",      "SS", "NYY"),
        ("Second Two",     "2B", "BOS"),
        ("First Two",      "1B", "HOU"),
        ("Catcher Two",    "C",  "BOS"),
        ("Third Two",      "3B", "HOU"),
        ("Outfield Five",  "OF", "NYY"),
        ("Second Three",   "2B", "NYY"),
        ("Third Three",    "3B", "BOS"),
    ]
    return [
        _row(name, pos, team=team, salary=2500, pts=15.0 + i * 0.5)
        for i, (name, pos, team) in enumerate(specs)
    ]


# ── Guard: too few CONFIDENT players ─────────────────────────────────────────
def test_build_fd_requires_10_confident():
    small_board = [_row(f"P{i}") for i in range(5)]
    with pytest.raises(ValueError, match="need ≥10"):
        build_fd_lineups(small_board)


def test_build_fd_excludes_flagged():
    """FLAGGED players must not enter the pool."""
    board = _full_board()
    board.append(_row("Flagged Player", state=ConfidenceState.FLAGGED, salary=4000, pts=25.0))
    # If FLAGGED were included they'd dominate; this just checks we don't crash
    # and that FLAGGED player may or may not appear (pydfs decides)
    lineups = build_fd_lineups(board)
    assert len(lineups) == 1
    player_names = [p["name"] for p in lineups[0]["players"]]
    assert "Flagged Player" not in player_names


# ── Single-entry build ────────────────────────────────────────────────────────
def test_build_fd_single_entry_returns_one_lineup():
    board = _full_board()
    lineups = build_fd_lineups(board, contest=CONTEST_SINGLE_ENTRY)
    assert len(lineups) == 1


def test_build_fd_lineup_under_salary_cap():
    board = _full_board()
    lineups = build_fd_lineups(board, contest=CONTEST_SINGLE_ENTRY)
    for lu in lineups:
        assert lu["total_salary"] <= 35_000


def test_build_fd_lineup_has_required_keys():
    board = _full_board()
    lineups = build_fd_lineups(board, contest=CONTEST_SINGLE_ENTRY)
    lu = lineups[0]
    assert "players" in lu
    assert "total_salary" in lu
    assert "total_proj" in lu
    assert len(lu["players"]) >= 9


# ── Pitcher-augmented board ───────────────────────────────────────────────────
def test_build_fd_with_pitcher_from_salary_csv():
    """
    Verify that pitchers_from_salary_csv + batter board produces a valid lineup.
    This simulates the real app flow: batter plays → consensus board,
    then pitchers appended from salary CSV before build.
    """
    from dfs.sources.salaries import pitchers_from_salary_csv, parse_fd_salary_csv
    # Batter-only board (no P)
    board = [r for r in _full_board() if r.position != "P"]
    assert all(r.position != "P" for r in board), "sanity: no pitchers in batter board"

    # Simulate salary CSV with 3 pitchers (2 CONFIDENT, 1 FLAGGED)
    fd_csv = (
        "FPPG,Nickname,First Name,Last Name,ID,Position,Team,Salary,"
        "Game,Opponent,Weather,Injury Indicator\n"
        "35.0,Ace One,Ace,One,501,P,NYY,9800,NYY@BOS,BOS,Clear,\n"
        "28.5,Ace Two,Ace,Two,502,P,BOS,8200,NYY@BOS,NYY,Clear,\n"
        "25.0,Ace Three,Ace,Three,503,P,HOU,7600,HOU@TEX,TEX,Clear,\n"
    )
    salary_rows = parse_fd_salary_csv(fd_csv)
    pitchers = pitchers_from_salary_csv(salary_rows, site="fd")
    assert len(pitchers) == 3
    assert all(p.state.value == "CONFIDENT" for p in pitchers)

    augmented = board + pitchers
    lineups = build_fd_lineups(augmented, contest=CONTEST_SINGLE_ENTRY)
    assert len(lineups) == 1
    player_positions = [p["position"] for p in lineups[0]["players"]]
    assert "P" in player_positions, "lineup must contain a pitcher"


# ── CSV export ────────────────────────────────────────────────────────────────
def test_export_fd_csv_creates_file():
    board = _full_board()
    lineups = build_fd_lineups(board, contest=CONTEST_SINGLE_ENTRY)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        path = tmp.name
    try:
        export_fd_csv(lineups, path)
        with open(path) as f:
            lines = f.readlines()
        # header + 1 data row
        assert len(lines) >= 2
        assert "P" in lines[0]   # FD slots header
    finally:
        os.unlink(path)
