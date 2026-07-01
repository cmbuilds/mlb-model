"""Tests for dfs/consensus.py — weighted consensus, state logic, ownership tagging."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from dfs.contracts import ConfidenceState, Provenance
from dfs.consensus import (
    _norm, _weighted_consensus, build_consensus_board,
    compute_stack_scores, DIVERGENCE_THRESHOLD,
)
from dfs.sources.api_external import ProjectionRow, SourceKind


# ── Fixtures ──────────────────────────────────────────────────────────────────
def _play(name="Aaron Judge", team="NYY", opp="BOS", bettable=True,
          lineup_confirmed=True, k_rate=0.228, barrel=0.12, hard_hit=0.45,
          woba=0.380, slg=0.580, iso=0.280, xslg=0.580,
          score=72.0, hr_score=55.0, dq_score=85, slot=3, implied=5.2,
          all_measured=True):
    prov = {k: "measured" for k in ("k_rate","slg_proxy","woba","hard_hit_rate","barrel_rate")}
    if not all_measured:
        prov["barrel_rate"] = "league_avg"
    return {
        "name": name, "team": team, "opponent": opp,
        "batter_position": "OF", "batter_hand": "R",
        "sp_name": "Gerrit Cole", "sp_hand": "R",
        "park": "NYY", "implied_total": implied,
        "game_id": "test_game",
        "lineup_slot": slot, "lineup_confirmed": lineup_confirmed,
        "bettable": bettable,
        "k_rate": k_rate, "barrel_rate": barrel, "hard_hit_rate": hard_hit,
        "bb_rate": 0.120, "woba": woba, "xslg": xslg, "iso": iso,
        "score": score, "hr_score": hr_score, "dq_score": dq_score,
        "sub_pitcher": 58.0, "sub_streak": 50.0,
        "fd_proj": 18.5, "fd_ceiling": 28.0, "fd_floor": 9.0,
        "weather": {"is_dome": False, "wind_effect": "neutral"},
        "_batter_prov": prov,
        "_pitcher_prov": {"k_rate_allowed": "measured", "hard_hit_allowed": "measured"},
    }


def _proj_row(name="Aaron Judge", pts=18.5, kind=SourceKind.MODEL,
              prov=Provenance.MEASURED, source="model"):
    return ProjectionRow(
        player=name, team="NYY", position="OF", site="fd",
        proj_pts=pts, source=source, kind=kind, provenance=prov,
    )


# ── _norm ─────────────────────────────────────────────────────────────────────
def test_norm_strips_accents():
    assert _norm("José Ramírez") == _norm("Jose Ramirez")

def test_norm_case_insensitive():
    assert _norm("Aaron Judge") == _norm("aaron judge")


# ── _weighted_consensus ───────────────────────────────────────────────────────
def test_single_source_consensus():
    rows = [_proj_row(pts=18.5)]
    pts, div = _weighted_consensus(rows)
    assert pts == pytest.approx(18.5)
    assert div == 0.0

def test_two_source_equal_weight():
    rows = [
        _proj_row(pts=18.0, kind=SourceKind.MODEL, prov=Provenance.MEASURED),
        _proj_row(pts=20.0, kind=SourceKind.MARKET, prov=Provenance.MARKET, source="odds"),
    ]
    pts, div = _weighted_consensus(rows)
    # both weight 1.0 → simple mean
    assert pts == pytest.approx(19.0)
    assert div == pytest.approx(2.0)

def test_external_model_discounted():
    rows = [
        _proj_row(pts=18.0, kind=SourceKind.MODEL, prov=Provenance.MEASURED, source="model"),
        _proj_row(pts=22.0, kind=SourceKind.EXTERNAL_MODEL,
                  prov=Provenance.MODEL_EXTERNAL, source="ext"),
    ]
    pts, div = _weighted_consensus(rows)
    # model weight=1.0, ext weight=0.5 → (18*1 + 22*0.5) / 1.5 = 38/1.5 ≈ 19.33
    expected = (18.0 * 1.0 + 22.0 * 0.5) / 1.5
    assert pts == pytest.approx(expected, abs=0.01)

def test_proxy_row_excluded_from_consensus():
    rows = [
        _proj_row(pts=18.0, prov=Provenance.MEASURED),
        _proj_row(pts=99.0, prov=Provenance.PROXY, source="bad"),  # must be excluded
    ]
    pts, div = _weighted_consensus(rows)
    assert pts == pytest.approx(18.0)
    assert div == 0.0

def test_all_proxy_returns_zero():
    rows = [_proj_row(pts=18.0, prov=Provenance.PROXY)]
    pts, div = _weighted_consensus(rows)
    assert pts == 0.0


# ── build_consensus_board ─────────────────────────────────────────────────────
def test_confident_player_on_measured_data():
    plays = [_play("Aaron Judge")]
    board = build_consensus_board(plays, site="fd")
    # Board now includes SP pitcher rows in addition to batters
    batters = [r for r in board if r.position not in ("P", "SP", "RP")]
    assert len(batters) == 1
    row = batters[0]
    assert row.state == ConfidenceState.CONFIDENT
    assert row.flagged_reason == ""
    assert row.source_count == 1
    assert row.sources_used == ["model"]

def test_flagged_on_proxy_data():
    plays = [_play("Bench Player", all_measured=False)]
    board = build_consensus_board(plays, site="fd")
    batters = [r for r in board if r.position not in ("P", "SP", "RP")]
    assert len(batters) == 1
    assert batters[0].state == ConfidenceState.FLAGGED
    assert batters[0].flagged_reason != ""

def test_flagged_on_unconfirmed_lineup():
    plays = [_play("Utility Man", lineup_confirmed=False)]
    board = build_consensus_board(plays, site="fd")
    # Filter to batters — the sp_name pitcher (Gerrit Cole) appears as CONFIDENT
    batters = [r for r in board if r.position not in ("P", "SP", "RP")]
    assert len(batters) == 1
    assert batters[0].state == ConfidenceState.FLAGGED
    assert "lineup" in batters[0].flagged_reason

def test_ownership_always_modeled():
    plays = [_play("Aaron Judge")]
    board = build_consensus_board(plays, site="fd")
    row = board[0]
    assert row.own_provenance == Provenance.MODELED
    assert row.own_pct > 0

def test_board_sorted_by_pts_desc():
    plays = [
        _play("Low Scorer",  score=50.0),
        _play("High Scorer", score=75.0, team="HOU"),
    ]
    board = build_consensus_board(plays, site="fd")
    assert board[0].consensus_pts >= board[-1].consensus_pts

def test_no_plays_raises():
    with pytest.raises(Exception):
        build_consensus_board([], site="fd")


# ── divergence flag ───────────────────────────────────────────────────────────
def test_divergence_flag_above_threshold():
    rows = [
        _proj_row(pts=10.0, kind=SourceKind.MODEL, prov=Provenance.MEASURED, source="model"),
        _proj_row(pts=10.0 + DIVERGENCE_THRESHOLD + 0.1,
                  kind=SourceKind.MARKET, prov=Provenance.MARKET, source="mkt"),
    ]
    pts, div = _weighted_consensus(rows)
    assert div > DIVERGENCE_THRESHOLD

def test_divergence_no_flag_single_source():
    rows = [_proj_row(pts=18.0)]
    pts, div = _weighted_consensus(rows)
    assert div == 0.0


# ── stack scores ──────────────────────────────────────────────────────────────
def test_stack_scores_sorted_desc():
    plays = [
        _play("A", team="NYY", score=75.0),
        _play("B", team="NYY", score=70.0),
        _play("C", team="HOU", score=55.0),
    ]
    board = build_consensus_board(plays, site="fd")
    scores = compute_stack_scores(board)
    teams = list(scores.keys())
    if len(teams) >= 2:
        assert scores[teams[0]] >= scores[teams[1]]
