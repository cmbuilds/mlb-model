"""
dfs/optimize.py — pydfs-lineup-optimizer wrapper for FanDuel and DraftKings.

Rules:
- Only CONFIDENT players enter the auto-pool. FLAGGED = manual override only.
- Proxy data never enters an auto-build without an explicit override flag.
- contest_type drives optimizer settings (single-entry vs multi-entry GPP).

FD auto-build: proven, runs tonight.
DK auto-build: greenfield — DO NOT call build_dk_lineups() into real money
  until validated (see spec §7 D0.4). Board-only tonight.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from pydfs_lineup_optimizer import get_optimizer, Site, Sport, Player
from pydfs_lineup_optimizer.player import GameInfo
from pydfs_lineup_optimizer.stacks import TeamStack

from dfs.contracts import ConsensusRow, ConfidenceState

logger = logging.getLogger("dfs.optimize")

# ── Contest presets ────────────────────────────────────────────────────────────
@dataclass
class ContestConfig:
    n_lineups: int
    max_exposure: float     # max % any player appears across n_lineups
    min_teams: int          # minimum unique teams across lineups
    stack_size: int         # primary stack size
    randomness: float       # 0 = deterministic, >0 = adds variance
    label: str


CONTEST_SINGLE_ENTRY = ContestConfig(
    n_lineups=1, max_exposure=1.0, min_teams=3,
    stack_size=3, randomness=0.0, label="single-entry"
)

CONTEST_MULTI_ENTRY_GPP = ContestConfig(
    n_lineups=10, max_exposure=0.4, min_teams=4,
    stack_size=4, randomness=0.5, label="multi-entry GPP"
)


# ── FanDuel ────────────────────────────────────────────────────────────────────
_FD_POS_MAP = {
    "SP": "P", "RP": "P",
    "OF": "OF", "LF": "OF", "CF": "OF", "RF": "OF",
    "DH": "C",   # rare; DH sometimes plays C/1B slot
}

def _fd_pos(pos: str) -> List[str]:
    mapped = _FD_POS_MAP.get(pos.upper(), pos.upper())
    return [mapped]


def build_fd_lineups(
    board: List[ConsensusRow],
    contest: ContestConfig = CONTEST_SINGLE_ENTRY,
    locked_names: Optional[List[str]] = None,
    excluded_names: Optional[List[str]] = None,
    stack_team: Optional[str] = None,
) -> List[Dict]:
    """
    Build FanDuel lineups from the CONFIDENT pool using pydfs.

    Returns list of lineup dicts (serialisable). Empty list = solver failed.
    FLAGGED players are never in the auto-pool.
    """
    confident_rows = [r for r in board
                      if r.state == ConfidenceState.CONFIDENT
                      and r.salary >= 1000
                      and r.consensus_pts > 0]

    if len(confident_rows) < 10:
        raise ValueError(
            f"FD build: only {len(confident_rows)} CONFIDENT players with salary — "
            "need ≥10. Upload salary CSV and rerun model."
        )

    opt = get_optimizer(Site.FANDUEL, Sport.BASEBALL)

    locked  = set(locked_names or [])
    excluded = set(excluded_names or [])

    # Build a GameInfo per unique game pairing so pydfs constraints resolve correctly.
    # Key: frozenset({team, opponent}) → GameInfo (order doesn't matter for grouping).
    _game_cache: Dict[frozenset, GameInfo] = {}

    def _game_info_for(team: str, opponent: str) -> GameInfo:
        key = frozenset({team, opponent})
        if key not in _game_cache:
            _game_cache[key] = GameInfo(home_team=opponent, away_team=team, starts_at=None)
        return _game_cache[key]

    players: List[Player] = []
    for row in confident_rows:
        pname = row.name.strip()
        parts  = pname.split(" ", 1)
        first  = parts[0] if parts else pname
        last   = parts[1] if len(parts) > 1 else ""

        is_locked   = pname in locked
        is_excluded = pname in excluded

        p = Player(
            player_id=pname.replace(" ", "_"),
            first_name=first,
            last_name=last,
            positions=_fd_pos(row.position),
            team=row.team,
            salary=float(row.salary),
            fppg=row.consensus_pts,
            max_exposure=1.0 if is_locked else contest.max_exposure,
            projected_ownership=row.own_pct / 100.0,
            fppg_floor=row.model_floor,
            fppg_ceil=row.model_ceiling,
            game_info=_game_info_for(row.team, row.opponent or "OPP"),
        )
        players.append(p)

    opt.load_players(players)

    # Lock / exclude
    for pname in locked:
        try:
            opt.add_player_to_lineup(opt.get_player_by_name(pname))
        except Exception:
            logger.warning("FD lock: player not found — %s", pname)

    for pname in excluded:
        try:
            opt.remove_player(opt.get_player_by_name(pname))
        except Exception:
            pass

    # Stack — only apply when the pool has enough teams; avoids infeasible constraints
    n_teams = len(set(r.team for r in confident_rows if r.position != "P"))
    if stack_team:
        opt.add_stack(TeamStack(contest.stack_size, for_teams=[stack_team]))
    elif contest.stack_size >= 3 and n_teams >= 2:
        opt.add_stack(TeamStack(contest.stack_size))

    if contest.randomness > 0:
        opt.set_deviation(0.0, contest.randomness)

    results = []
    try:
        for lineup in opt.optimize(n=contest.n_lineups):
            lp = lineup.players
            results.append({
                "site":      "fd",
                "contest":   contest.label,
                "players":   [
                    {
                        "name":     f"{p.first_name} {p.last_name}".strip(),
                        "position": p.positions[0] if p.positions else "?",
                        "team":     p.team,
                        "salary":   int(p.salary),
                        "fppg":     round(p.fppg, 1),
                    }
                    for p in lp
                ],
                "total_salary":  sum(int(p.salary) for p in lp),
                "total_proj":    round(sum(p.fppg for p in lp), 1),
                "total_ceiling": round(sum(p.fppg_ceil or p.fppg for p in lp), 1),
            })
    except Exception as e:
        logger.error("FD optimizer failed: %s", e)
        raise RuntimeError(f"FD optimizer failed: {e}") from e

    return results


def export_fd_csv(lineups: List[Dict], filepath: str) -> None:
    """Export lineups to FanDuel bulk-import CSV format."""
    fd_slots = ["P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF", "UTIL"]

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fd_slots)
        for lu in lineups:
            by_pos: Dict[str, List[str]] = {}
            for p in lu["players"]:
                pos = p["position"]
                by_pos.setdefault(pos, []).append(p["name"])

            row = []
            pos_cursors = {k: 0 for k in by_pos}
            for slot in fd_slots:
                avail = [k for k in by_pos if k == slot or
                          (slot == "OF" and k in ("LF","CF","RF")) or
                          (slot == "UTIL" and k != "P")]
                placed = False
                for pos in avail:
                    idx = pos_cursors.get(pos, 0)
                    names = by_pos.get(pos, [])
                    if idx < len(names):
                        row.append(names[idx])
                        pos_cursors[pos] = idx + 1
                        placed = True
                        break
                if not placed:
                    row.append("")
            writer.writerow(row)


# ── DraftKings — board only tonight ───────────────────────────────────────────
# DO NOT call this function into real money until the DK optimizer is validated
# on a test slate (D1). Tonight it is intentionally not called from the UI.

_DK_POS_MAP = {
    "OF": "OF", "LF": "OF", "CF": "OF", "RF": "OF",
    "DH": "1B",
}

def _dk_pos(pos: str) -> List[str]:
    return [_DK_POS_MAP.get(pos.upper(), pos.upper())]


def build_dk_lineups(
    board: List[ConsensusRow],
    contest: ContestConfig = CONTEST_SINGLE_ENTRY,
    locked_names: Optional[List[str]] = None,
    excluded_names: Optional[List[str]] = None,
    stack_team: Optional[str] = None,
) -> List[Dict]:
    """
    DraftKings auto-build — EXPERIMENTAL. NOT enabled in the UI tonight.
    Validate on a test slate before using in live contests.
    """
    confident_rows = [r for r in board
                      if r.state == ConfidenceState.CONFIDENT
                      and r.salary >= 3000
                      and r.consensus_pts > 0]

    if len(confident_rows) < 10:
        raise ValueError(
            f"DK build: only {len(confident_rows)} CONFIDENT players — need ≥10."
        )

    opt = get_optimizer(Site.DRAFTKINGS, Sport.BASEBALL)

    locked   = set(locked_names or [])
    excluded = set(excluded_names or [])

    players: List[Player] = []
    for row in confident_rows:
        pname = row.name.strip()
        parts  = pname.split(" ", 1)
        first  = parts[0] if parts else pname
        last   = parts[1] if len(parts) > 1 else ""

        p = Player(
            player_id=pname.replace(" ", "_"),
            first_name=first,
            last_name=last,
            positions=_dk_pos(row.position),
            team=row.team,
            salary=float(row.salary),
            fppg=row.consensus_pts,
            max_exposure=1.0 if pname in locked else contest.max_exposure,
            projected_ownership=row.own_pct / 100.0,
            fppg_floor=row.model_floor,
            fppg_ceil=row.model_ceiling,
        )
        players.append(p)

    opt.load_players(players)

    for pname in locked:
        try:
            opt.add_player_to_lineup(opt.get_player_by_name(pname))
        except Exception:
            logger.warning("DK lock: player not found — %s", pname)

    for pname in excluded:
        try:
            opt.remove_player(opt.get_player_by_name(pname))
        except Exception:
            pass

    if stack_team:
        opt.add_stack(TeamStack(contest.stack_size, for_teams=[stack_team]))
    elif contest.stack_size >= 3:
        opt.add_stack(TeamStack(contest.stack_size))

    if contest.randomness > 0:
        opt.set_deviation(0.0, contest.randomness)

    results = []
    try:
        for lineup in opt.optimize(n=contest.n_lineups):
            lp = lineup.players
            results.append({
                "site":      "dk",
                "contest":   contest.label,
                "players":   [
                    {
                        "name":     f"{p.first_name} {p.last_name}".strip(),
                        "position": p.positions[0] if p.positions else "?",
                        "team":     p.team,
                        "salary":   int(p.salary),
                        "fppg":     round(p.fppg, 1),
                    }
                    for p in lp
                ],
                "total_salary": sum(int(p.salary) for p in lp),
                "total_proj":   round(sum(p.fppg for p in lp), 1),
            })
    except Exception as e:
        logger.error("DK optimizer failed: %s", e)
        raise RuntimeError(f"DK optimizer failed: {e}") from e

    return results


def export_dk_csv(lineups: List[Dict], filepath: str) -> None:
    """Export lineups to DraftKings bulk-upload CSV format."""
    dk_slots = ["SP", "SP", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF", "FLEX"]
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(dk_slots)
        for lu in lineups:
            by_pos: Dict[str, List[str]] = {}
            for p in lu["players"]:
                by_pos.setdefault(p["position"], []).append(p["name"])
            row = []
            pos_cursors = {k: 0 for k in by_pos}
            for slot in dk_slots:
                avail = [k for k in by_pos if k == slot or
                          (slot == "OF" and k in ("LF","CF","RF")) or
                          (slot == "FLEX" and k not in ("SP","RP"))]
                placed = False
                for pos in avail:
                    idx = pos_cursors.get(pos, 0)
                    names = by_pos.get(pos, [])
                    if idx < len(names):
                        row.append(names[idx])
                        pos_cursors[pos] = idx + 1
                        placed = True
                        break
                if not placed:
                    row.append("")
            writer.writerow(row)
