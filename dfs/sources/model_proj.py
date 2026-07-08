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
_PITCHER_CORE_FIELDS = ("k_rate_allowed", "hard_hit_allowed")


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


def _pitcher_provenance(pit_prov: Dict) -> Provenance:
    """MEASURED only if both core pitcher fields are measured."""
    if all(pit_prov.get(f, "proxy") == "measured" for f in _PITCHER_CORE_FIELDS):
        return Provenance.MEASURED
    return Provenance.PROXY


def plays_to_fd_pitcher_projections(plays: List[Dict]) -> List[ProjectionRow]:
    """
    Group batter plays by sp_name and project FD pitcher points using SP stats
    embedded in the batter plays (_pitcher_k_rate, _pitcher_prov, implied_total).

    This produces model-derived pitcher projections (not site FPPG), tagged with
    the appropriate provenance so the board shows measured vs proxy.

    Raises SourceError if no projectable SPs found.
    """
    from dfs.lib.fd_rules import compute_fd_pitcher_projection

    # Group plays by SP name
    sp_groups: Dict[str, List[Dict]] = {}
    for p in plays:
        sp = p.get("sp_name", "")
        if not sp or sp in ("TBD", "?", ""):
            continue
        sp_groups.setdefault(sp, []).append(p)

    rows: List[ProjectionRow] = []
    for sp_name, sp_plays in sp_groups.items():
        # Aggregate: use first play for team/hand meta, average k_rate / implied
        first = sp_plays[0]
        sp_team = first.get("opponent", "")   # SP's team is the batter's opponent
        sp_hand = first.get("sp_hand", "?")

        # Aggregate pitcher stats across batter plays for this SP
        k_rates = [p.get("_pitcher_k_rate", 0.0) for p in sp_plays
                   if p.get("_pitcher_k_rate", 0.0) > 0.05]
        avg_k_rate = sum(k_rates) / len(k_rates) if k_rates else 0.228

        # opp_implied: from batter plays' implied_total (the SP's opponent's runs)
        implied_vals = [p.get("implied_total", 0.0) for p in sp_plays
                        if p.get("implied_total", 0.0) > 0.5]
        opp_implied = sum(implied_vals) / len(implied_vals) if implied_vals else 4.5

        # Pitcher provenance from first play's _pitcher_prov
        pit_prov = first.get("_pitcher_prov", {})
        prov = _pitcher_provenance(pit_prov)

        agg = {
            "_pitcher_k_rate": avg_k_rate,
            "_pitcher_fip":    first.get("sp_fip") or first.get("pitcher_fip", 4.10),
            "_pitcher_whip":   first.get("sp_whip") or first.get("pitcher_whip", 1.30),
            "implied_total":   opp_implied,
        }
        result = compute_fd_pitcher_projection(agg)
        if result is None or result["fd_pts"] <= 0:
            continue

        rows.append(ProjectionRow(
            player=sp_name,
            team=sp_team,
            position="P",
            site="fd",
            proj_pts=result["fd_pts"],
            source=SOURCE_ID,
            kind=SourceKind.MODEL,
            provenance=prov,
            salary=None,
            updated=None,
        ))

    if not rows:
        raise SourceError("model_proj (fd pitchers): no projectable SPs in plays list")
    return rows


def plays_to_dk_pitcher_projections(plays: List[Dict]) -> List[ProjectionRow]:
    """
    Same as FD but applies DK pitcher scoring instead.
    DK scoring: IP=2.25, K=2, Win=4, ER=-2, H=-0.6, BB=-0.6 (no QS).
    """
    from dfs.lib.dk_rules import DK_PITCHER_SCORING

    sp_groups: Dict[str, List[Dict]] = {}
    for p in plays:
        sp = p.get("sp_name", "")
        if not sp or sp in ("TBD", "?", ""):
            continue
        sp_groups.setdefault(sp, []).append(p)

    rows: List[ProjectionRow] = []
    for sp_name, sp_plays in sp_groups.items():
        first = sp_plays[0]
        sp_team = first.get("opponent", "")
        k_rates = [p.get("_pitcher_k_rate", 0.0) for p in sp_plays
                   if p.get("_pitcher_k_rate", 0.0) > 0.05]
        avg_k_rate = sum(k_rates) / len(k_rates) if k_rates else 0.228
        implied_vals = [p.get("implied_total", 0.0) for p in sp_plays
                        if p.get("implied_total", 0.0) > 0.5]
        opp_implied = sum(implied_vals) / len(implied_vals) if implied_vals else 4.5
        fip = first.get("sp_fip") or first.get("pitcher_fip", 4.10)
        whip = first.get("sp_whip") or first.get("pitcher_whip", 1.30)

        # IP estimate from FIP (same as FD)
        if fip < 3.0:   proj_ip = 6.5
        elif fip < 3.5: proj_ip = 6.2
        elif fip < 4.0: proj_ip = 5.8
        elif fip < 4.5: proj_ip = 5.4
        else:           proj_ip = 4.8
        proj_ip = proj_ip * max(0.85, 1.0 - (opp_implied - 4.5) * 0.04)

        bf      = proj_ip * 3.3
        proj_k  = avg_k_rate * bf
        proj_er = max(0.0, (fip / 9.0) * proj_ip)
        proj_h  = whip * proj_ip * 0.70
        proj_bb = whip * proj_ip * 0.30

        win_prob = 0.50
        if fip < 3.5 and opp_implied < 4.0: win_prob = 0.65
        elif fip > 4.5 or opp_implied > 5.0: win_prob = 0.35

        dk_pts = (
            proj_k  * DK_PITCHER_SCORING["k"] +
            proj_ip * DK_PITCHER_SCORING["ip"] +
            proj_er * DK_PITCHER_SCORING["er"] +
            proj_h  * DK_PITCHER_SCORING["h"] +
            proj_bb * DK_PITCHER_SCORING["bb"] +
            win_prob * DK_PITCHER_SCORING["win"]
        )
        if dk_pts <= 0:
            continue

        pit_prov = first.get("_pitcher_prov", {})
        prov = _pitcher_provenance(pit_prov)

        rows.append(ProjectionRow(
            player=sp_name,
            team=sp_team,
            position="SP",
            site="dk",
            proj_pts=round(dk_pts, 1),
            source=SOURCE_ID,
            kind=SourceKind.MODEL,
            provenance=prov,
            salary=None,
            updated=None,
        ))

    if not rows:
        raise SourceError("model_proj (dk pitchers): no projectable SPs in plays list")
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
