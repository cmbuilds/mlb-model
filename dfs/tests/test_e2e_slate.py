"""
End-to-end integration test: realistic slate mock → consensus board → build lineups → log → retrieve.

Uses a realistic 3-game slate with 18 batters + 3 SPs per team, full provenance,
pitcher play dicts with _pitcher_k_rate, and FD salary CSV with all positions.

This test validates the full chain that runs on contest night:
  run_model() plays → build_consensus_board() → build_fd_lineups_diverse() → log_lineups() → get_lineups()
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tempfile
import pytest

from dfs.consensus import build_consensus_board, compute_stack_scores
from dfs.contracts import ConfidenceState, Provenance
from dfs.optimize import build_fd_lineups, build_fd_lineups_diverse, with_n_lineups, ContestConfig
from dfs.sources.salaries import parse_fd_salary_csv, merge_salaries_into_board, pitchers_from_salary_csv
from dfs.lineup_log import log_lineups, get_lineups, record_result


# ── Realistic slate fixtures ───────────────────────────────────────────────────
GAMES = [
    ("NYY", "BOS"),
    ("HOU", "TEX"),
    ("LAD", "SDP"),
]
PROV_MEASURED = {k: "measured" for k in
                 ("k_rate", "slg_proxy", "woba", "hard_hit_rate", "barrel_rate")}
PROV_PITCHER = {"k_rate_allowed": "measured", "hard_hit_allowed": "measured"}


def _make_plays(teams=GAMES) -> list:
    """Generate a realistic set of batter plays across 3 games."""
    plays = []
    slot = 1
    for home, away in teams:
        sp_home = f"SP_{home}"
        sp_away = f"SP_{away}"
        for batting_team, sp_name in [(home, sp_away), (away, sp_home)]:
            opp_team = away if batting_team == home else home
            implied = 5.0 if batting_team == home else 4.5
            for i in range(6):    # 6 batters per team in lineup
                pos_map = ["C", "1B", "2B", "3B", "SS", "OF"]
                plays.append({
                    "name": f"{batting_team}_Batter{i+1}",
                    "team": batting_team,
                    "opponent": opp_team,
                    "batter_position": pos_map[i],
                    "batter_hand": "R",
                    "sp_name": sp_name,
                    "sp_hand": "R",
                    "park": home,
                    "implied_total": implied,
                    "game_id": f"{home}_{away}",
                    "lineup_slot": i + 1,
                    "lineup_confirmed": True,
                    "bettable": True,
                    "k_rate": 0.220,
                    "barrel_rate": 0.095,
                    "hard_hit_rate": 0.44,
                    "bb_rate": 0.090,
                    "woba": 0.340,
                    "xslg": 0.430,
                    "iso": 0.165,
                    "score": 65.0 + i * 2,
                    "hr_score": 45.0,
                    "dq_score": 80,
                    "sub_pitcher": 55.0,
                    "sub_streak": 48.0,
                    "fd_proj": 14.0 + i * 0.8,
                    "fd_ceiling": 22.0 + i,
                    "fd_floor": 7.0 + i * 0.3,
                    "weather": {"is_dome": False, "wind_effect": "neutral"},
                    "_batter_prov": PROV_MEASURED,
                    "_pitcher_prov": PROV_PITCHER,
                    "_pitcher_k_rate": 0.255,
                    "sp_fip": 3.85,
                    "sp_whip": 1.18,
                })
    return plays


# FD salary CSV with all positions for 3 games
_FD_SALARY_CSV = (
    "FPPG,Nickname,First Name,Last Name,ID,Position,Team,Salary,Game,Opponent,Weather,Injury Indicator\n"
    # Pitchers
    "38.5,SP_BOS,SP,BOS,901,P,BOS,10800,NYY@BOS,NYY,Clear,\n"
    "35.2,SP_NYY,SP,NYY,902,P,NYY,10200,NYY@BOS,BOS,Clear,\n"
    "32.1,SP_TEX,SP,TEX,903,P,TEX,9600,HOU@TEX,HOU,Clear,\n"
    "30.5,SP_HOU,SP,HOU,904,P,HOU,9200,HOU@TEX,TEX,Clear,\n"
    "29.8,SP_SDP,SP,SDP,905,P,SDP,8800,LAD@SDP,LAD,Clear,\n"
    "28.3,SP_LAD,SP,LAD,906,P,LAD,8400,LAD@SDP,SDP,Clear,\n"
    # Batters — 6 per team = 36 total
    "18.0,NYY_Batter1,NYY,Batter1,101,C,NYY,4200,NYY@BOS,BOS,Clear,\n"
    "17.2,NYY_Batter2,NYY,Batter2,102,1B,NYY,3900,NYY@BOS,BOS,Clear,\n"
    "16.8,NYY_Batter3,NYY,Batter3,103,2B,NYY,3700,NYY@BOS,BOS,Clear,\n"
    "15.5,NYY_Batter4,NYY,Batter4,104,3B,NYY,3500,NYY@BOS,BOS,Clear,\n"
    "14.9,NYY_Batter5,NYY,Batter5,105,SS,NYY,3300,NYY@BOS,BOS,Clear,\n"
    "14.5,NYY_Batter6,NYY,Batter6,106,OF,NYY,3100,NYY@BOS,BOS,Clear,\n"
    "17.5,BOS_Batter1,BOS,Batter1,111,C,BOS,4000,NYY@BOS,NYY,Clear,\n"
    "16.8,BOS_Batter2,BOS,Batter2,112,1B,BOS,3800,NYY@BOS,NYY,Clear,\n"
    "16.2,BOS_Batter3,BOS,Batter3,113,2B,BOS,3600,NYY@BOS,NYY,Clear,\n"
    "15.0,BOS_Batter4,BOS,Batter4,114,3B,BOS,3400,NYY@BOS,NYY,Clear,\n"
    "14.5,BOS_Batter5,BOS,Batter5,115,SS,BOS,3200,NYY@BOS,NYY,Clear,\n"
    "14.0,BOS_Batter6,BOS,Batter6,116,OF,BOS,3000,NYY@BOS,NYY,Clear,\n"
    "18.5,HOU_Batter1,HOU,Batter1,121,C,HOU,4300,HOU@TEX,TEX,Clear,\n"
    "17.8,HOU_Batter2,HOU,Batter2,122,1B,HOU,4000,HOU@TEX,TEX,Clear,\n"
    "17.2,HOU_Batter3,HOU,Batter3,123,2B,HOU,3800,HOU@TEX,TEX,Clear,\n"
    "16.0,HOU_Batter4,HOU,Batter4,124,3B,HOU,3600,HOU@TEX,TEX,Clear,\n"
    "15.5,HOU_Batter5,HOU,Batter5,125,SS,HOU,3400,HOU@TEX,TEX,Clear,\n"
    "15.0,HOU_Batter6,HOU,Batter6,126,OF,HOU,3200,HOU@TEX,TEX,Clear,\n"
    "16.5,TEX_Batter1,TEX,Batter1,131,C,TEX,3800,HOU@TEX,HOU,Clear,\n"
    "15.8,TEX_Batter2,TEX,Batter2,132,1B,TEX,3600,HOU@TEX,HOU,Clear,\n"
    "15.2,TEX_Batter3,TEX,Batter3,133,2B,TEX,3400,HOU@TEX,HOU,Clear,\n"
    "14.0,TEX_Batter4,TEX,Batter4,134,3B,TEX,3200,HOU@TEX,HOU,Clear,\n"
    "13.5,TEX_Batter5,TEX,Batter5,135,SS,TEX,3000,HOU@TEX,HOU,Clear,\n"
    "13.0,TEX_Batter6,TEX,Batter6,136,OF,TEX,2800,HOU@TEX,HOU,Clear,\n"
    "19.0,LAD_Batter1,LAD,Batter1,141,C,LAD,4500,LAD@SDP,SDP,Clear,\n"
    "18.2,LAD_Batter2,LAD,Batter2,142,1B,LAD,4200,LAD@SDP,SDP,Clear,\n"
    "17.8,LAD_Batter3,LAD,Batter3,143,2B,LAD,4000,LAD@SDP,SDP,Clear,\n"
    "16.5,LAD_Batter4,LAD,Batter4,144,3B,LAD,3700,LAD@SDP,SDP,Clear,\n"
    "16.0,LAD_Batter5,LAD,Batter5,145,SS,LAD,3500,LAD@SDP,SDP,Clear,\n"
    "15.5,LAD_Batter6,LAD,Batter6,146,OF,LAD,3300,LAD@SDP,SDP,Clear,\n"
    "16.0,SDP_Batter1,SDP,Batter1,151,C,SDP,3700,LAD@SDP,LAD,Clear,\n"
    "15.3,SDP_Batter2,SDP,Batter2,152,1B,SDP,3500,LAD@SDP,LAD,Clear,\n"
    "14.8,SDP_Batter3,SDP,Batter3,153,2B,SDP,3300,LAD@SDP,LAD,Clear,\n"
    "13.5,SDP_Batter4,SDP,Batter4,154,3B,SDP,3100,LAD@SDP,LAD,Clear,\n"
    "13.0,SDP_Batter5,SDP,Batter5,155,SS,SDP,2900,LAD@SDP,LAD,Clear,\n"
    "12.5,SDP_Batter6,SDP,Batter6,156,OF,SDP,2700,LAD@SDP,LAD,Clear,\n"
)


@pytest.fixture
def slate_plays():
    return _make_plays()


@pytest.fixture
def salary_rows():
    return parse_fd_salary_csv(_FD_SALARY_CSV)


@pytest.fixture
def full_board(slate_plays, salary_rows):
    """Build a board with salaries merged and pitchers augmented."""
    board = build_consensus_board(slate_plays, site="fd")
    board, matched = merge_salaries_into_board(board, salary_rows)
    assert matched >= 30, f"Expected >=30 salary matches, got {matched}"

    # Add pitcher rows from salary CSV for any SP not already on board
    on_board_pitchers = {r.name.lower() for r in board if r.position in ("P", "SP", "RP")}
    pitcher_rows = pitchers_from_salary_csv(salary_rows, site="fd")
    new_pitchers = [p for p in pitcher_rows if p.name.lower() not in on_board_pitchers]
    return board + new_pitchers


# ── Board sanity ──────────────────────────────────────────────────────────────
def test_board_has_batters_from_all_teams(full_board):
    batter_teams = {r.team for r in full_board if r.position not in ("P", "SP", "RP")}
    for team, _ in GAMES:
        assert team in batter_teams, f"Missing batters for {team}"
    for _, team in GAMES:
        assert team in batter_teams, f"Missing batters for {team}"


def test_board_has_pitchers(full_board):
    pitchers = [r for r in full_board if r.position in ("P", "SP", "RP")]
    assert len(pitchers) >= 4


def test_board_confident_majority(full_board):
    confident = [r for r in full_board if r.state == ConfidenceState.CONFIDENT]
    assert len(confident) / len(full_board) >= 0.7, (
        f"Expected >=70% CONFIDENT, got {len(confident)}/{len(full_board)}"
    )


def test_board_salaries_populated(full_board):
    with_salary = [r for r in full_board if r.salary >= 1000]
    assert len(with_salary) >= 30


def test_board_sorted_by_pts(full_board):
    pts = [r.consensus_pts for r in full_board]
    assert pts == sorted(pts, reverse=True)


# ── Stack scores ──────────────────────────────────────────────────────────────
def test_stack_scores_all_teams(full_board):
    scores = compute_stack_scores(full_board)
    for team, _ in GAMES:
        assert team in scores
    for _, team in GAMES:
        assert team in scores


def test_stack_scores_sorted(full_board):
    scores = compute_stack_scores(full_board)
    vals = list(scores.values())
    assert vals == sorted(vals, reverse=True)


# ── Optimizer (single-entry) ──────────────────────────────────────────────────
def test_single_entry_lineup_valid(full_board):
    from dfs.optimize import CONTEST_SINGLE_ENTRY
    lineups = build_fd_lineups(full_board, contest=CONTEST_SINGLE_ENTRY)
    assert len(lineups) == 1
    lu = lineups[0]
    assert lu["total_salary"] <= 35_000
    assert len(lu["players"]) >= 9
    positions = [p["position"] for p in lu["players"]]
    assert "P" in positions


def test_single_entry_uses_confident_only(full_board):
    from dfs.optimize import CONTEST_SINGLE_ENTRY
    lineups = build_fd_lineups(full_board, contest=CONTEST_SINGLE_ENTRY)
    flagged_names = {r.name for r in full_board if r.state == ConfidenceState.FLAGGED}
    for lu in lineups:
        for p in lu["players"]:
            assert p["name"] not in flagged_names


# ── Diverse multi-entry ───────────────────────────────────────────────────────
def test_diverse_build_returns_n_lineups(full_board):
    # max_exposure=1.0 so deterministic (randomness=0) build can reuse players
    # across stack groups without triggering per-player exposure caps.
    contest = ContestConfig(
        n_lineups=6, max_exposure=1.0, min_teams=3,
        stack_size=3, randomness=0.0, label="e2e-gpp",
    )
    lineups = build_fd_lineups_diverse(full_board, contest=contest, max_stacks=3)
    assert len(lineups) == 6


def test_diverse_build_each_under_cap(full_board):
    contest = ContestConfig(
        n_lineups=4, max_exposure=1.0, min_teams=3,
        stack_size=3, randomness=0.0, label="e2e-gpp",
    )
    lineups = build_fd_lineups_diverse(full_board, contest=contest, max_stacks=2)
    for lu in lineups:
        assert lu["total_salary"] <= 35_000


def test_diverse_build_multiple_primary_teams(full_board):
    """Lineups should span at least 2 different primary (most-represented) stack teams."""
    contest = ContestConfig(
        n_lineups=4, max_exposure=1.0, min_teams=3,
        stack_size=3, randomness=0.0, label="e2e-gpp",
    )
    lineups = build_fd_lineups_diverse(full_board, contest=contest, max_stacks=2)

    def _primary_team(lu):
        from collections import Counter
        non_p = [p for p in lu["players"] if p["position"] != "P"]
        return Counter(p["team"] for p in non_p).most_common(1)[0][0]

    primary_teams = {_primary_team(lu) for lu in lineups}
    assert len(primary_teams) >= 2, f"Expected diverse stacks, got: {primary_teams}"


# ── Log pipeline ──────────────────────────────────────────────────────────────
def test_log_and_retrieve_pipeline(full_board, tmp_path):
    from dfs.optimize import CONTEST_SINGLE_ENTRY
    db = str(tmp_path / "e2e.db")

    lineups = build_fd_lineups(full_board, contest=CONTEST_SINGLE_ENTRY)
    row_id = log_lineups(
        lineups, site="fd", contest_label="single-entry",
        slate_date="2026-07-01", notes="e2e test", db_path=db,
    )
    assert isinstance(row_id, int)

    rows = get_lineups(slate_date="2026-07-01", db_path=db)
    assert len(rows) == 1
    assert rows[0]["n_lineups"] == 1
    assert rows[0]["notes"] == "e2e test"
    assert isinstance(rows[0]["players_json"], list)

    # Record result and verify
    record_result(row_id, result_pts=141.25, result_rank=88, db_path=db)
    rows_after = get_lineups(db_path=db)
    assert rows_after[0]["result_pts"] == pytest.approx(141.25)
    assert rows_after[0]["result_rank"] == 88
