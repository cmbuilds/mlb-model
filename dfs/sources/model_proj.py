"""
dfs/sources/model_proj.py

Adapter: converts the app's per-player `plays` dicts (output of run_model())
into ProjectionRows with per-field provenance, reusing the monolith's
compute_fd_projection / compute_pp_projection functions for points math.

Design contract:
- NEVER re-derive Statcast values — only package what the model already computed.
- Tag every row with provenance derived from the play's _batter_prov dict.
- A player on proxy core data (any field league_avg / proxy) → Provenance.PROXY.
- A player on measured core → Provenance.MEASURED.
- Missing / pa=0 / unmatched → excluded (SourceError raised upstream).
"""

from __future__ import annotations

import sys
import os
from typing import Dict, List

# Allow import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dfs.sources.api_external import (
    ProjectionRow, SourceKind, Provenance, SourceError
)


SOURCE_ID = "model"
_CORE_FIELDS = ("k_rate", "slg_proxy", "woba", "hard_hit_rate", "barrel_rate")


def _provenance_from_play(p: Dict) -> Provenance:
    """
    Determine provenance of this play's core batter stats.
    Any field not 'measured' → PROXY (cannot present as measured).
    """
    prov = p.get("_batter_prov", {})
    if all(prov.get(f, "league_avg") == "measured" for f in _CORE_FIELDS):
        return Provenance.MEASURED
    return Provenance.PROXY


def plays_to_fd_projections(plays: List[Dict]) -> List[ProjectionRow]:
    """
    Convert plays list → FanDuel ProjectionRows.

    Uses fd_proj already computed by run_model() (via compute_fd_projection).
    If fd_proj is missing (model ran before DFS tab existed), falls back to
    recomputing — but never invents a value for a missing stat.
    """
    rows: List[ProjectionRow] = []
    for p in plays:
        name = p.get("name", "")
        if not name:
            continue

        # fd_proj may already be on the play dict (run_model computes it)
        fd_pts = p.get("fd_proj")
        fd_ceil = p.get("fd_ceiling")
        fd_floor = p.get("fd_floor")

        if fd_pts is None:
            # Recompute from play dict fields using the proven monolith function.
            # Import lazily to avoid circular import / streamlit dependency.
            fd_pts, fd_ceil, fd_floor = _recompute_fd(p)

        if fd_pts is None or fd_pts <= 0:
            continue  # no projection available — skip, never fabricate

        prov = _provenance_from_play(p)
        salary = p.get("fd_salary") or 0

        rows.append(ProjectionRow(
            player=name,
            team=p.get("team", ""),
            position=p.get("batter_position", "OF"),
            site="fd",
            proj_pts=float(fd_pts),
            source=SOURCE_ID,
            kind=SourceKind.MODEL,
            provenance=prov,
            salary=int(salary) if salary else None,
            updated=None,
        ))

    if not rows:
        raise SourceError("model_proj (fd): no projectable plays in plays list")
    return rows


def plays_to_dk_projections(plays: List[Dict]) -> List[ProjectionRow]:
    """
    Convert plays list → DraftKings ProjectionRows using DK scoring rules.
    DK points math live in dfs/lib/dk_rules.py.
    """
    from dfs.lib.dk_rules import compute_dk_projection

    rows: List[ProjectionRow] = []
    for p in plays:
        name = p.get("name", "")
        if not name:
            continue

        dk_pts = compute_dk_projection(p)
        if dk_pts is None or dk_pts <= 0:
            continue

        prov = _provenance_from_play(p)
        salary = p.get("dk_salary") or 0

        rows.append(ProjectionRow(
            player=name,
            team=p.get("team", ""),
            position=p.get("batter_position", "OF"),
            site="dk",
            proj_pts=float(dk_pts),
            source=SOURCE_ID,
            kind=SourceKind.MODEL,
            provenance=prov,
            salary=int(salary) if salary else None,
            updated=None,
        ))

    if not rows:
        raise SourceError("model_proj (dk): no projectable plays in plays list")
    return rows


# ─── lazy recompute fallback ─────────────────────────────────────────────────

def _recompute_fd(p: Dict):
    """
    Recompute FD projection from a plays dict if fd_proj is absent.
    Returns (pts, ceiling, floor) or (None, None, None) if inputs insufficient.
    """
    try:
        from mlb_tb_analyzer import compute_fd_projection
    except ImportError:
        return None, None, None

    batter_stats = {
        "k_rate":        p.get("k_rate", 0.228),
        "bb_rate":       p.get("bb_rate", 0.082),
        "slg_proxy":     p.get("xslg", 0.398),
        "iso_proxy":     p.get("iso", 0.165),
        "woba":          p.get("woba", 0.315),
        "hard_hit_rate": p.get("hard_hit_rate", 0.370),
        "barrel_rate":   p.get("barrel_rate", 0.070),
    }
    pitcher_stats = {
        "k_rate_allowed":   0.228,
        "hard_hit_allowed": 0.370,
        "fip":              4.10,
    }
    weather = p.get("weather", {})
    try:
        result = compute_fd_projection(
            statcast=batter_stats,
            pitcher_statcast=pitcher_stats,
            lineup_slot=p.get("lineup_slot", 5),
            implied_total=p.get("implied_total", 0.0),
            batter_hand=p.get("batter_hand", "R"),
            sp_hand=p.get("sp_hand", "R"),
            park_team=p.get("park", ""),
            weather=weather,
        )
        return result.get("fd_proj"), result.get("fd_ceiling"), result.get("fd_floor")
    except Exception:
        return None, None, None
