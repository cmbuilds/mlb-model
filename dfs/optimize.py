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


def allocate_lineups_by_stack(
    stack_scores: Dict[str, float],
    n_lineups: int,
    max_stacks: int = 4,
) -> Dict[str, int]:
    """
    Distribute n_lineups across top teams by stack score using rank weighting.

    Guarantees: every team in the allocation gets >= 1 lineup.
    Remaining lineups after floor allocation go to highest-scored teams first.
    Returns {team: count}.
    """
    if not stack_scores or n_lineups <= 0:
        return {}

    ranked = sorted(stack_scores.items(), key=lambda kv: kv[1], reverse=True)
    teams  = [t for t, _ in ranked[:max_stacks]]

    n_teams = len(teams)
    if n_teams == 0:
        return {}

    # Start with 1 lineup per team, then distribute remainder by rank weight
    allocation: Dict[str, int] = {t: 1 for t in teams}
    remaining = n_lineups - n_teams
    if remaining <= 0:
        # Fewer lineups than teams — just take the top n_lineups teams
        return {t: 1 for t in teams[:n_lineups]}

    # Rank weights: 1st gets n_teams points, 2nd gets n_teams-1, …
    weights = {t: (n_teams - i) for i, t in enumerate(teams)}
    total_w = sum(weights.values())
    extras  = {t: round(w / total_w * remaining) for t, w in weights.items()}

    # Rounding may leave 1-2 lineups unallocated — give them to the top team
    shortfall = remaining - sum(extras.values())
    if shortfall != 0:
        extras[teams[0]] = extras.get(teams[0], 0) + shortfall

    for t in teams:
        allocation[t] += extras.get(t, 0)

    return {t: v for t, v in allocation.items() if v > 0}


def build_fd_lineups_diverse(
    board: List[ConsensusRow],
    contest: ContestConfig = CONTEST_MULTI_ENTRY_GPP,
    locked_names: Optional[List[str]] = None,
    excluded_names: Optional[List[str]] = None,
    max_stacks: int = 4,
) -> List[Dict]:
    """
    Build multi-entry FD lineups with enforced stack diversity.

    Distributes contest.n_lineups across the top `max_stacks` teams ranked by
    stack score, then builds each group with that team as the forced stack.
    Each group's lineups have contest randomness applied so they differ within
    the same primary-stack correlated universe.

    Returns combined list of lineup dicts.
    """
    from dfs.consensus import compute_stack_scores

    if contest.n_lineups <= 1:
        return build_fd_lineups(board, contest, locked_names, excluded_names)

    hitter_board = [r for r in board if r.position not in ("P", "SP", "RP")]
    stack_scores = compute_stack_scores(hitter_board)

    allocation = allocate_lineups_by_stack(stack_scores, contest.n_lineups, max_stacks)
    if not allocation:
        logger.warning("FD diverse build: no stack scores — falling back to single pool")
        return build_fd_lineups(board, contest, locked_names, excluded_names)

    logger.info("FD diverse allocation: %s", allocation)

    all_lineups: List[Dict] = []
    for team, k in allocation.items():
        sub_contest = with_n_lineups(contest, k)
        try:
            chunk = build_fd_lineups(
                board=board, contest=sub_contest,
                locked_names=locked_names, excluded_names=excluded_names,
                stack_team=team,
            )
            all_lineups.extend(chunk)
        except Exception as e:
            logger.warning("FD diverse: stack %s failed (%s) — skipping", team, e)

    if not all_lineups:
        raise RuntimeError("FD diverse build: all team stacks failed — check board and salary CSV")

    return all_lineups


def with_n_lineups(config: ContestConfig, n: int) -> ContestConfig:
    """Return a copy of config with n_lineups overridden."""
    return ContestConfig(
        n_lineups=n, max_exposure=config.max_exposure,
        min_teams=config.min_teams, stack_size=config.stack_size,
        randomness=config.randomness, label=config.label,
    )


def lineup_uniqueness_score(lu1: Dict, lu2: Dict) -> int:
    """Count shared players between two lineups. 7+ of 9 is degenerate for FD."""
    names1 = {p["name"] for p in lu1["players"]}
    names2 = {p["name"] for p in lu2["players"]}
    return len(names1 & names2)


def check_lineup_diversity(lineups: List[Dict], warn_threshold: int = 7) -> List[str]:
    """
    Return a list of warning strings for degenerate lineup pairs.
    warn_threshold: lineups sharing >= this many players are flagged.
    """
    warnings = []
    n = len(lineups)
    for i in range(n):
        for j in range(i + 1, n):
            shared = lineup_uniqueness_score(lineups[i], lineups[j])
            if shared >= warn_threshold:
                warnings.append(
                    f"Lineups {i+1} and {j+1} share {shared} players — may be too similar"
                )
    return warnings


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

    _game_cache_dk: Dict[frozenset, GameInfo] = {}

    def _game_info_for_dk(team: str, opponent: str) -> GameInfo:
        key = frozenset({team, opponent})
        if key not in _game_cache_dk:
            _game_cache_dk[key] = GameInfo(home_team=opponent, away_team=team, starts_at=None)
        return _game_cache_dk[key]

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
            game_info=_game_info_for_dk(row.team, row.opponent or "OPP"),
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
