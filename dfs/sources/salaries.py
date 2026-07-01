"""
dfs/sources/salaries.py — salary CSV ingest for FanDuel and DraftKings.

Both sites export a CSV from their lineup tool. User uploads it in the UI.
Normalises to a common dict format consumed by optimize.py and the board.

FanDuel CSV headers (MLB):
  FPPG, Nickname, First Name, Last Name, ID, Position, Team, Salary, Game, Opponent, Weather, Injury Indicator

DraftKings CSV headers (MLB Classic):
  Position, Name + ID, Name, ID, Roster Position, Salary, Game Info, TeamAbbrev, AvgPointsPerGame
"""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from typing import Dict, List, Optional


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFD", str(name))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _parse_salary(raw) -> int:
    try:
        return int(str(raw).replace(",", "").replace("$", "").strip())
    except (ValueError, TypeError):
        return 0


# ── FanDuel ────────────────────────────────────────────────────────────────────
def parse_fd_salary_csv(content: str) -> List[Dict]:
    """
    Parse a FanDuel MLB salary export (string or file content).
    Returns list of normalised player dicts.
    Raises ValueError if the content doesn't look like an FD salary file.
    """
    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    if not rows:
        raise ValueError("FD salary CSV is empty")

    # Detect FD format by header presence
    headers = set(reader.fieldnames or [])
    if not ({"Nickname", "Salary", "Position"} & headers or
            {"First Name", "Last Name", "Salary"} & headers):
        raise ValueError("File does not look like a FanDuel salary CSV")

    out = []
    for row in rows:
        # FD uses either "Nickname" or "First Name"/"Last Name"
        name = (row.get("Nickname") or
                f"{row.get('First Name','')} {row.get('Last Name','')}").strip()
        if not name:
            continue
        salary = _parse_salary(row.get("Salary", 0))
        if salary <= 0:
            continue

        pos_raw = row.get("Position", "OF").strip().upper()
        # FD sometimes uses "C/1B" — take first valid position
        pos = pos_raw.split("/")[0]

        out.append({
            "name":     name,
            "norm":     _norm(name),
            "position": pos,
            "salary":   salary,
            "team":     (row.get("Team") or "").strip().upper(),
            "opponent": (row.get("Opponent") or "").strip().upper(),
            "fppg":     float(row.get("FPPG") or 0.0),
            "game":     row.get("Game") or row.get("Game Info", ""),
            "injury":   (row.get("Injury Indicator") or "").strip(),
            "site":     "fd",
            "player_id": str(row.get("ID") or ""),
        })

    if not out:
        raise ValueError("FD salary CSV parsed but produced no valid rows")
    return out


# ── DraftKings ─────────────────────────────────────────────────────────────────
def parse_dk_salary_csv(content: str) -> List[Dict]:
    """
    Parse a DraftKings MLB Classic salary export.
    Returns list of normalised player dicts.
    """
    from dfs.lib.dk_rules import DK_PYDFS_POSITIONS

    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    if not rows:
        raise ValueError("DK salary CSV is empty")

    headers = set(reader.fieldnames or [])
    if not ({"Name", "Salary", "Position"} & headers or
            {"Name + ID", "Salary"} & headers):
        raise ValueError("File does not look like a DraftKings salary CSV")

    out = []
    for row in rows:
        # DK name may be "First Last (id)" in some exports
        raw_name = row.get("Name") or row.get("Name + ID", "")
        name = raw_name.split("(")[0].strip()
        if not name:
            continue
        salary = _parse_salary(row.get("Salary", 0))
        if salary <= 0:
            continue

        pos_raw  = (row.get("Position") or row.get("Roster Position", "UTIL")).strip().upper()
        pos_norm = DK_PYDFS_POSITIONS.get(pos_raw, pos_raw)

        out.append({
            "name":     name,
            "norm":     _norm(name),
            "position": pos_norm,
            "salary":   salary,
            "team":     (row.get("TeamAbbrev") or row.get("Team", "")).strip().upper(),
            "opponent": "",   # not in DK CSV; matched from game string if needed
            "avg_pts":  float(row.get("AvgPointsPerGame") or 0.0),
            "game":     row.get("Game Info", ""),
            "site":     "dk",
            "player_id": str(row.get("ID") or ""),
        })

    if not out:
        raise ValueError("DK salary CSV parsed but produced no valid rows")
    return out


# ── Salary merger ──────────────────────────────────────────────────────────────
def merge_salaries_into_board(board, salary_rows: List[Dict]) -> List:
    """
    Enrich ConsensusRow objects with salary from the uploaded CSV.
    Matches on normalised name. Unmatched rows keep salary=0.
    Returns the board in-place (mutates salary field).
    """
    salary_map = {r["norm"]: r for r in salary_rows}

    matched = 0
    for row in board:
        key = _norm(row.name)
        sal_row = salary_map.get(key)
        if sal_row is None:
            # fuzzy: check if any salary key is a subset of the name
            for k, v in salary_map.items():
                if k in key or key in k:
                    sal_row = v
                    break
        if sal_row:
            object.__setattr__(row, "salary", sal_row["salary"]) if hasattr(row, "__dataclass_fields__") else setattr(row, "salary", sal_row["salary"])
            # recalculate value
            sal = sal_row["salary"]
            if sal >= 1000:
                cv = round(row.consensus_pts / (sal / 1000), 2)
                try:
                    object.__setattr__(row, "consensus_value", cv)
                except AttributeError:
                    row.consensus_value = cv
            matched += 1

    return board, matched
