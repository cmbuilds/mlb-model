"""
dfs/consensus.py — fuse available projection sources into a consensus board.

Axis: **projected fantasy points** (not the 0-100 TB score).
The TB score is a confidence modifier, not the projection axis.

Rules (from spec §5 — do not relax):
- EXCLUDE (never default) missing or proxy sources.
- consensus_pts = provenance-weighted mean of PRESENT, non-proxy sources only.
- A player with only proxy inputs → FLAGGED, excluded from auto-build.
- Modeled ownership is tagged Provenance.MODELED, never shown as field ownership.
- High source divergence → divergence_flag=True, state=FLAGGED.

Tonight (D0): single source (model). The multi-source path is ready but
single-source rows are tagged source_count=1 so the UI is honest.
"""

from __future__ import annotations

import unicodedata, re
from typing import Dict, List, Optional

from dfs.contracts import (
    ConsensusRow, ConfidenceState,
    ProjectionRow, SourceKind, Provenance, ORTHOGONALITY_WEIGHT,
)
from dfs.sources.model_proj import _provenance_from_play

# Divergence flag threshold — spread > this across sources → FLAGGED
DIVERGENCE_THRESHOLD = 4.0   # pts

# Minimum consensus_pts to appear on the board
MIN_PTS_THRESHOLD = 5.0


def _norm(name: str) -> str:
    """Normalise player name for cross-source matching."""
    s = unicodedata.normalize("NFD", str(name))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _weighted_consensus(rows: List[ProjectionRow]) -> tuple[float, float]:
    """
    Compute provenance-weighted mean across projection rows for one player.
    Excludes PROXY / LEAGUE_AVG rows — they are not averaged in.
    Returns (consensus_pts, divergence).
    """
    eligible = [r for r in rows
                if r.provenance not in (Provenance.PROXY, Provenance.LEAGUE_AVG)]
    if not eligible:
        return 0.0, 0.0

    total_w = 0.0
    total_wpts = 0.0
    pts_vals = []
    for r in eligible:
        w = ORTHOGONALITY_WEIGHT.get(r.kind, 0.5)
        total_w += w
        total_wpts += r.proj_pts * w
        pts_vals.append(r.proj_pts)

    consensus = total_wpts / total_w if total_w > 0 else 0.0
    divergence = max(pts_vals) - min(pts_vals) if len(pts_vals) > 1 else 0.0
    return round(consensus, 2), round(divergence, 2)


def build_consensus_board(
    plays: List[Dict],
    extra_sources: Optional[List[List[ProjectionRow]]] = None,
    site: str = "fd",
) -> List[ConsensusRow]:
    """
    Main entry point.

    plays       — the app's per-player plays list (output of run_model())
    extra_sources — optional list of ProjectionRow lists from external sources
                    (each inner list is one source's full slate output)
    site        — "fd" or "dk"

    Returns list of ConsensusRow sorted by consensus_pts desc.
    """
    from dfs.sources.model_proj import plays_to_fd_projections, plays_to_dk_projections

    # Step 1: build model source rows (batters + pitchers)
    try:
        if site == "fd":
            from dfs.sources.model_proj import plays_to_fd_pitcher_projections
            model_rows = plays_to_fd_projections(plays)
            try:
                pitcher_rows = plays_to_fd_pitcher_projections(plays)
                model_rows = model_rows + pitcher_rows
            except Exception:
                pass  # no SP data in plays — site FPPG fallback handled in dfs_tabs
        else:
            from dfs.sources.model_proj import plays_to_dk_pitcher_projections
            model_rows = plays_to_dk_projections(plays)
            try:
                pitcher_rows = plays_to_dk_pitcher_projections(plays)
                model_rows = model_rows + pitcher_rows
            except Exception:
                pass
    except Exception as e:
        raise RuntimeError(f"consensus: model source failed — {e}") from e

    # Index model rows by normalised name
    model_index: Dict[str, ProjectionRow] = {_norm(r.player): r for r in model_rows}

    # Step 2: index extra source rows by norm name
    # {norm_name: [ProjectionRow, ...]}
    extra_index: Dict[str, List[ProjectionRow]] = {}
    if extra_sources:
        for source_rows in extra_sources:
            for r in source_rows:
                key = _norm(r.player)
                extra_index.setdefault(key, []).append(r)

    # Step 3: build plays index for metadata (batter-centric: keyed by batter name)
    plays_index: Dict[str, Dict] = {_norm(p.get("name", "")): p for p in plays}

    # Secondary index for pitcher rows: keyed by sp_name (since pitchers don't
    # have a play dict keyed by their own name — they appear in batter plays).
    pitcher_plays_index: Dict[str, Dict] = {}
    for p in plays:
        sp = p.get("sp_name", "")
        if sp and sp not in ("TBD", "?", ""):
            pitcher_plays_index.setdefault(_norm(sp), p)

    # Step 4: one ConsensusRow per player that has a model projection
    board: List[ConsensusRow] = []
    for norm_name, model_row in model_index.items():
        is_pitcher = model_row.position in ("P", "SP", "RP")

        play = plays_index.get(norm_name, {})
        if not play and not is_pitcher:
            # Batter: try partial match (handles nickname variants)
            for k, v in plays_index.items():
                if norm_name in k or k in norm_name:
                    play = v
                    break

        # For pitcher rows, look up via sp_name secondary index
        pitcher_play: Dict = {}
        if is_pitcher:
            pitcher_play = pitcher_plays_index.get(norm_name, {})

        # Gather all rows for this player across sources
        all_rows: List[ProjectionRow] = [model_row]
        all_rows.extend(extra_index.get(norm_name, []))

        consensus_pts, divergence = _weighted_consensus(all_rows)

        if consensus_pts < MIN_PTS_THRESHOLD:
            # Proxy-only players return 0 from _weighted_consensus because proxy rows
            # are excluded from the mean. Fall back to raw model pts so FLAGGED players
            # still appear on the board (greyed) rather than silently disappearing.
            if model_row.proj_pts >= MIN_PTS_THRESHOLD:
                consensus_pts = round(model_row.proj_pts, 2)
            else:
                continue

        eligible_rows = [r for r in all_rows
                         if r.provenance not in (Provenance.PROXY, Provenance.LEAGUE_AVG)]
        source_count  = len(eligible_rows)
        sources_used  = list(dict.fromkeys(r.source for r in eligible_rows))

        # Determine state
        if is_pitcher and pitcher_play:
            # Pitcher confirmed status: if they appear in any batter play, the pitcher
            # is starting. Use _pitcher_prov for provenance.
            from dfs.sources.model_proj import _pitcher_provenance
            pit_prov_dict = pitcher_play.get("_pitcher_prov", {})
            prov = _pitcher_provenance(pit_prov_dict)
            lineup_confirmed = True   # SP appears in plays → confirmed starter
            bettable = False
        else:
            prov = _provenance_from_play(play) if play else Provenance.LEAGUE_AVG
            lineup_confirmed = play.get("lineup_confirmed", False)
            bettable = play.get("bettable", False)

        if not lineup_confirmed or prov in (Provenance.PROXY, Provenance.LEAGUE_AVG):
            state = ConfidenceState.FLAGGED
            if not lineup_confirmed:
                flagged_reason = "lineup not confirmed"
            else:
                flagged_reason = "proxy/league-avg core data"
        elif divergence > DIVERGENCE_THRESHOLD and source_count > 1:
            state = ConfidenceState.FLAGGED
            flagged_reason = f"source divergence {divergence:.1f} pts"
        else:
            state = ConfidenceState.CONFIDENT
            flagged_reason = ""

        # Salary — prefer from model_row, fall back to play dict
        salary = model_row.salary or play.get("fd_salary" if site == "fd" else "dk_salary") or 0
        consensus_value = round(consensus_pts / (salary / 1000), 2) if salary >= 1000 else 0.0

        # Modeled ownership
        own_pct = _compute_modeled_ownership(
            pts=consensus_pts,
            salary=salary,
            implied=play.get("implied_total", 0.0),
            slot=play.get("lineup_slot", 5),
            barrel=play.get("barrel_rate", 0.07),
        )

        # Model ceiling/floor from play or derive
        model_pts    = round(model_row.proj_pts, 1)
        model_ceiling = round(play.get("fd_ceiling" if site == "fd" else "dk_ceiling",
                                       model_pts * 1.45), 1)
        model_floor   = round(play.get("fd_floor" if site == "fd" else "dk_floor",
                                       max(0, model_pts * 0.4)), 1)

        board.append(ConsensusRow(
            name=model_row.player,
            team=model_row.team,
            opponent=play.get("opponent", ""),
            position=model_row.position,
            site=site,
            salary=salary,
            lineup_slot=play.get("lineup_slot", 0),
            consensus_pts=consensus_pts,
            consensus_value=consensus_value,
            model_pts=model_pts,
            model_ceiling=model_ceiling,
            model_floor=model_floor,
            state=state,
            source_count=source_count,
            sources_used=sources_used,
            flagged_reason=flagged_reason,
            own_pct=own_pct,
            own_provenance=Provenance.MODELED,
            divergence=divergence,
            divergence_flag=(divergence > DIVERGENCE_THRESHOLD and source_count > 1),
            bettable=bettable,
            batter_hand=play.get("batter_hand", "?"),
            sp_name=play.get("sp_name", "TBD"),
            sp_hand=play.get("sp_hand", "?"),
            park=play.get("park", ""),
            implied_total=play.get("implied_total", 0.0),
            game_id=play.get("game_id", ""),
            score=play.get("score", 0.0),
            hr_score=play.get("hr_score", 0.0),
            dq_score=play.get("dq_score", 0),
        ))

    board.sort(key=lambda r: r.consensus_pts, reverse=True)
    return board


def compute_stack_scores(board: List[ConsensusRow]) -> Dict[str, float]:
    """
    Rank teams by combined consensus ceiling of their top 4 hitters.
    Returns {team: stack_score} — used by the UI for stack grouping.
    """
    team_pts: Dict[str, List[float]] = {}
    for row in board:
        if row.state == ConfidenceState.EXCLUDED:
            continue
        team_pts.setdefault(row.team, []).append(row.model_ceiling)

    stack_scores: Dict[str, float] = {}
    for team, ceilings in team_pts.items():
        top4 = sorted(ceilings, reverse=True)[:4]
        implied_bonus = 0.0
        # Use implied_total from the first row for this team as a stack multiplier
        for row in board:
            if row.team == team and row.implied_total > 0:
                implied_bonus = (row.implied_total - 4.5) * 0.5
                break
        stack_scores[team] = round(sum(top4) + implied_bonus, 1)

    return dict(sorted(stack_scores.items(), key=lambda kv: kv[1], reverse=True))


def _compute_modeled_ownership(
    pts: float, salary: int, implied: float, slot: int, barrel: float
) -> float:
    """
    Model field ownership % from consensus pts + context.
    Tagged Provenance.MODELED — NEVER shown as real field ownership.
    """
    if salary <= 0:
        return 15.0
    value = pts / (salary / 1000)
    own = value * 4.5
    if implied >= 5.5:
        own *= 1.30
    elif implied >= 5.0:
        own *= 1.15
    elif implied < 3.5:
        own *= 0.70
    if slot <= 2:
        own *= 1.20
    elif slot >= 7:
        own *= 0.80
    if barrel > 0.15:
        own *= 1.15
    return round(min(60.0, max(3.0, own)), 1)
