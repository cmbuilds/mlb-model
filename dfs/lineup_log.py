"""
dfs/lineup_log.py — SQLite persistence for DFS lineups.

Schema is append-only. Never update or delete rows — results layer on top
so we can audit every build and backtest any date.

Table: lineups
  id            INTEGER PRIMARY KEY
  built_at      TEXT    ISO-8601 UTC timestamp of the build
  slate_date    TEXT    YYYY-MM-DD game date
  site          TEXT    "fd" | "dk"
  contest_label TEXT    free-text (e.g. "single-entry", "multi-entry GPP")
  n_lineups     INTEGER number of lineups in this batch
  total_proj    REAL    sum of projected pts across all lineups / n_lineups
  players_json  TEXT    JSON: list of player dicts (name, team, pos, salary, fppg)
  result_pts    REAL    NULL until contest results populated
  result_rank   INTEGER NULL until contest results populated
  notes         TEXT    freeform (e.g. "DK $40K Battery" or manual notes)

DB location: DFS_LOG_PATH env var, or ./dfs_lineups.db in the working directory.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger("dfs.lineup_log")

_DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dfs_lineups.db")
DFS_LOG_PATH = os.environ.get("DFS_LOG_PATH", _DEFAULT_DB)

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS lineups (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    built_at      TEXT    NOT NULL,
    slate_date    TEXT    NOT NULL,
    site          TEXT    NOT NULL,
    contest_label TEXT    NOT NULL,
    n_lineups     INTEGER NOT NULL,
    total_proj    REAL,
    players_json  TEXT    NOT NULL,
    result_pts    REAL,
    result_rank   INTEGER,
    notes         TEXT
);
CREATE INDEX IF NOT EXISTS idx_slate ON lineups(slate_date, site);
"""


def _connect(db_path: str = DFS_LOG_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_CREATE_SQL)
    return conn


def log_lineups(
    lineups: List[Dict],
    site: str,
    contest_label: str,
    slate_date: str,
    notes: str = "",
    db_path: str = DFS_LOG_PATH,
) -> int:
    """
    Persist a batch of lineups to the log.

    Returns the row id of the inserted record.
    Raises on DB error — caller should surface this, not swallow it.
    """
    if not lineups:
        raise ValueError("log_lineups: lineups list is empty")

    built_at = datetime.now(timezone.utc).isoformat()
    total_proj = round(
        sum(lu.get("total_proj", 0.0) for lu in lineups) / len(lineups), 2
    )

    # Normalise players_json: each lineup is a list of player dicts
    players_payload = [lu.get("players", []) for lu in lineups]

    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO lineups
                (built_at, slate_date, site, contest_label, n_lineups,
                 total_proj, players_json, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                built_at,
                slate_date,
                site,
                contest_label,
                len(lineups),
                total_proj,
                json.dumps(players_payload),
                notes,
            ),
        )
        row_id = cur.lastrowid
        conn.commit()

    logger.info(
        "lineup_log: saved %d %s lineups for %s → row %d",
        len(lineups), site, slate_date, row_id,
    )
    return row_id


def get_lineups(
    slate_date: Optional[str] = None,
    site: Optional[str] = None,
    limit: int = 50,
    db_path: str = DFS_LOG_PATH,
) -> List[Dict]:
    """
    Retrieve logged lineups, newest first.

    slate_date: filter to a specific YYYY-MM-DD date (None = all)
    site:       filter to "fd" or "dk" (None = all)
    Returns list of dicts with all columns; players_json is decoded to list.
    """
    clauses = []
    params: list = []
    if slate_date:
        clauses.append("slate_date = ?")
        params.append(slate_date)
    if site:
        clauses.append("site = ?")
        params.append(site)

    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(limit)

    with _connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT * FROM lineups {where} ORDER BY id DESC LIMIT ?",
            params,
        ).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        try:
            d["players_json"] = json.loads(d["players_json"])
        except (json.JSONDecodeError, TypeError):
            pass
        result.append(d)
    return result


def record_result(
    row_id: int,
    result_pts: float,
    result_rank: Optional[int] = None,
    db_path: str = DFS_LOG_PATH,
) -> None:
    """
    Update a saved lineup batch with actual contest results.
    Call this after the contest closes.
    """
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE lineups SET result_pts = ?, result_rank = ? WHERE id = ?",
            (result_pts, result_rank, row_id),
        )
        conn.commit()
    logger.info("lineup_log: recorded result %.1f (rank %s) for row %d",
                result_pts, result_rank, row_id)
