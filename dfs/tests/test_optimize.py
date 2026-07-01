"""Tests for dfs/optimize.py — pydfs wrapper, contest config, guard rails."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from dfs.contracts import ConsensusRow, ConfidenceState, Provenance
from dfs.optimize import (
    build_fd_lineups, build_fd_lineups_diverse, CONTEST_SINGLE_ENTRY, CONTEST_MULTI_ENTRY_GPP,
    export_fd_csv, allocate_lineups_by_stack, with_n_lineups, check_lineup_diversity,
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
# ── allocate_lineups_by_stack ─────────────────────────────────────────────────
def test_allocate_sums_to_n():
    scores = {"NYY": 120.0, "BOS": 110.0, "HOU": 100.0, "LAD": 90.0}
    alloc = allocate_lineups_by_stack(scores, n_lineups=10)
    assert sum(alloc.values()) == 10


def test_allocate_all_teams_get_at_least_one():
    scores = {"NYY": 120.0, "BOS": 110.0, "HOU": 100.0}
    alloc = allocate_lineups_by_stack(scores, n_lineups=10, max_stacks=3)
    for team in ["NYY", "BOS", "HOU"]:
        assert alloc.get(team, 0) >= 1


def test_allocate_top_team_gets_most():
    scores = {"NYY": 150.0, "BOS": 100.0, "HOU": 80.0}
    alloc = allocate_lineups_by_stack(scores, n_lineups=12, max_stacks=3)
    assert alloc["NYY"] >= alloc["BOS"] >= alloc.get("HOU", 0)


def test_allocate_n_less_than_teams():
    scores = {"NYY": 120.0, "BOS": 110.0, "HOU": 100.0, "LAD": 90.0}
    alloc = allocate_lineups_by_stack(scores, n_lineups=2, max_stacks=4)
    assert sum(alloc.values()) == 2
    assert len(alloc) == 2


def test_allocate_empty_scores_returns_empty():
    assert allocate_lineups_by_stack({}, n_lineups=5) == {}


# ── build_fd_lineups_diverse ──────────────────────────────────────────────────
def test_build_fd_diverse_multi_team_stacks():
    """Diverse build produces valid lineups split across multiple teams."""
    board = _full_board()
    # Use deterministic single-entry clones per team (no randomness) to guarantee
    # feasibility on the small 20-player test board.
    from dfs.optimize import ContestConfig
    det_gpp = ContestConfig(
        n_lineups=3, max_exposure=1.0, min_teams=3,
        stack_size=3, randomness=0.0, label="test-gpp",
    )
    lineups = build_fd_lineups_diverse(board, contest=det_gpp, max_stacks=3)
    assert len(lineups) == 3
    for lu in lineups:
        assert lu["total_salary"] <= 35_000
        assert len(lu["players"]) >= 9


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
