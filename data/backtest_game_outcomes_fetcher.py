"""
data/backtest_game_outcomes_fetcher.py — Fetch final game scores for ML calibration.

Uses the MLB schedule API with scores (already present in the Final game objects).
Writes to `game_outcomes` table in mlb_stats.db:
  home_team, away_team, home_score, away_score, winning_team ("home"|"away"|"tie")

This is all that's needed for the ML backtest — we measure whether the model's
predicted winner (home_wp > 0.5 → home) matched the actual winner.
No Odds API needed for this calibration step.

Usage:
    python3 data/backtest_game_outcomes_fetcher.py --start 2026-04-01 --end 2026-06-29
    python3 data/backtest_game_outcomes_fetcher.py --start 2026-04-01 --end 2026-06-29 --dry-run

Prerequisite:
    game_results table must exist (run data/backtest_fetcher.py first) —
    used to determine which dates have data so we don't over-fetch.
"""

import argparse
import logging
import sqlite3
import time
from datetime import date, timedelta
from typing import Dict, List, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH       = "data/mlb_stats.db"
MLB_API       = "https://statsapi.mlb.com/api/v1"
REQUEST_DELAY = 0.25


def _get(url: str, params: dict = None, timeout: int = 15) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"GET {url} failed: {e}")
        return None


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_outcomes (
            game_pk       INTEGER PRIMARY KEY,
            game_date     TEXT    NOT NULL,
            home_team     TEXT    NOT NULL,
            away_team     TEXT    NOT NULL,
            home_score    INTEGER DEFAULT 0,
            away_score    INTEGER DEFAULT 0,
            winning_team  TEXT    NOT NULL    -- "home" | "away" | "tie"
        )
    """)
    conn.commit()


def fetch_outcomes_for_date(date_str: str) -> List[Dict]:
    """Return final scores for all completed games on date_str."""
    data = _get(f"{MLB_API}/schedule", params={
        "sportId": 1, "date": date_str, "gameType": "R",
    })
    if not data:
        return []

    rows = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            status = g.get("status", {}).get("detailedState", "")
            if status not in ("Final", "Game Over", "Completed Early"):
                continue

            teams     = g.get("teams", {})
            home_data = teams.get("home", {})
            away_data = teams.get("away", {})

            home_team  = home_data.get("team", {}).get("name", "?")
            away_team  = away_data.get("team", {}).get("name", "?")
            home_score = int(home_data.get("score", 0) or 0)
            away_score = int(away_data.get("score", 0) or 0)
            game_pk    = g["gamePk"]

            if home_score > away_score:
                winning_team = "home"
            elif away_score > home_score:
                winning_team = "away"
            else:
                winning_team = "tie"

            rows.append({
                "game_pk":      game_pk,
                "game_date":    date_str,
                "home_team":    home_team,
                "away_team":    away_team,
                "home_score":   home_score,
                "away_score":   away_score,
                "winning_team": winning_team,
            })
    return rows


def write_outcomes(conn: sqlite3.Connection, rows: List[Dict]) -> int:
    written = 0
    for row in rows:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO game_outcomes
                (game_pk, game_date, home_team, away_team, home_score, away_score, winning_team)
                VALUES
                (:game_pk, :game_date, :home_team, :away_team, :home_score, :away_score, :winning_team)
            """, row)
            written += 1
        except Exception as e:
            log.warning(f"DB write error game {row.get('game_pk')}: {e}")
    conn.commit()
    return written


def fetch_all(start: str, end: str, dry_run: bool = False) -> Dict:
    """Fetch game outcomes for all dates between start and end."""
    conn = sqlite3.connect(DB_PATH)
    _ensure_schema(conn)

    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    summary = {"dates": 0, "games": 0, "skipped": 0}

    cur = s
    while cur <= e:
        date_str = cur.isoformat()

        # Skip dates already fully fetched
        existing = conn.execute(
            "SELECT COUNT(*) FROM game_outcomes WHERE game_date = ?", (date_str,)
        ).fetchone()[0]
        if existing > 0:
            log.debug(f"{date_str}: {existing} outcomes already cached — skipping")
            summary["skipped"] += existing
            cur += timedelta(days=1)
            continue

        if dry_run:
            log.info(f"[DRY RUN] Would fetch outcomes for {date_str}")
            cur += timedelta(days=1)
            continue

        rows = fetch_outcomes_for_date(date_str)
        if rows:
            n = write_outcomes(conn, rows)
            log.info(f"{date_str}: {n} game outcomes  "
                     + ", ".join(f"{r['away_team']}@{r['home_team']} {r['away_score']}-{r['home_score']}" for r in rows[:3])
                     + ("..." if len(rows) > 3 else ""))
            summary["dates"] += 1
            summary["games"] += n
        else:
            log.info(f"{date_str}: no completed games")

        time.sleep(REQUEST_DELAY)
        cur += timedelta(days=1)

    conn.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Fetch game outcomes for ML backtest")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = fetch_all(args.start, args.end, dry_run=args.dry_run)
    print(f"\nDone. Dates: {summary['dates']} | Games: {summary['games']} | "
          f"Skipped (cached): {summary['skipped']}")


if __name__ == "__main__":
    main()
