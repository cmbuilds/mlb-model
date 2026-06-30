"""
data/backtest_sp_k_fetcher.py — Fetch SP K totals per game for K prop backtest.

Reads the same MLB Stats API live-feed endpoint used by backtest_fetcher.py.
The first pitcher in each team's pitching order is the SP. Pulls:
  - sp_k:   strikeouts by the SP in that game
  - sp_ip:  innings pitched by the SP (float, e.g. 6.1 → 6.333)
  - hit_k5: 1 if sp_k >= 5 (O4.5 K outcome)
  - hit_k6: 1 if sp_k >= 6 (O5.5 K outcome — the most common main K prop line)

Writes to game_sp_results table in mlb_stats.db.

Usage:
    python3 data/backtest_sp_k_fetcher.py --start 2026-04-01 --end 2026-06-29
    python3 data/backtest_sp_k_fetcher.py --start 2026-04-01 --end 2026-06-29 --dry-run

Prerequisite:
    data/backtest_fetcher.py must already have populated game_results for the same
    date range (uses game_results to get the list of game_pks to fetch).
"""

import argparse
import logging
import sqlite3
import time
from typing import Dict, List, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH      = "data/mlb_stats.db"
MLB_API      = "https://statsapi.mlb.com/api/v1"
REQUEST_DELAY = 0.3


def _get(url: str, timeout: int = 15) -> Optional[dict]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"GET {url} failed: {e}")
        return None


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_sp_results (
            game_pk   INTEGER NOT NULL,
            game_date TEXT    NOT NULL,
            team      TEXT    NOT NULL,
            sp_id     INTEGER,
            sp_name   TEXT,
            sp_k      INTEGER DEFAULT 0,
            sp_ip     REAL    DEFAULT 0.0,
            hit_k5    INTEGER DEFAULT 0,
            hit_k6    INTEGER DEFAULT 0,
            PRIMARY KEY (game_pk, team)
        )
    """)
    conn.commit()


def _ip_to_float(ip_str) -> float:
    """Convert MLB API inningsPitched string (e.g. '6.1') to float innings."""
    try:
        parts = str(ip_str).split(".")
        full  = int(parts[0])
        outs  = int(parts[1]) if len(parts) > 1 else 0
        return round(full + outs / 3.0, 3)
    except Exception:
        return 0.0


def fetch_sp_results(game_pk: int, game_date: str) -> List[Dict]:
    """
    Fetch starting pitcher K totals for both teams in a game.
    Returns up to 2 rows (one per team). Returns [] on any failure.
    """
    data = _get(f"{MLB_API}.1/game/{game_pk}/feed/live")
    if not data:
        return []

    rows = []
    box = data.get("liveData", {}).get("boxscore", {}).get("teams", {})

    for side in ("home", "away"):
        team_data = box.get(side, {})
        team_name = team_data.get("team", {}).get("name", side)
        pitching_order = team_data.get("pitchers", [])   # list of player IDs in appearance order
        players        = team_data.get("players", {})

        if not pitching_order:
            log.debug(f"  game {game_pk} {side}: no pitching order")
            continue

        sp_id   = pitching_order[0]   # first pitcher = starter
        pdata   = players.get(f"ID{sp_id}", {})
        person  = pdata.get("person", {})
        sp_name = person.get("fullName", f"ID{sp_id}")
        pstats  = pdata.get("stats", {}).get("pitching", {})

        sp_k  = int(pstats.get("strikeOuts",      0) or 0)
        sp_ip = _ip_to_float(pstats.get("inningsPitched", "0.0"))

        rows.append({
            "game_pk":   game_pk,
            "game_date": game_date,
            "team":      team_name,
            "sp_id":     sp_id,
            "sp_name":   sp_name,
            "sp_k":      sp_k,
            "sp_ip":     sp_ip,
            "hit_k5":    1 if sp_k >= 5 else 0,
            "hit_k6":    1 if sp_k >= 6 else 0,
        })

    return rows


def write_sp_results(conn: sqlite3.Connection, rows: List[Dict]) -> int:
    written = 0
    for row in rows:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO game_sp_results
                (game_pk, game_date, team, sp_id, sp_name, sp_k, sp_ip, hit_k5, hit_k6)
                VALUES
                (:game_pk, :game_date, :team, :sp_id, :sp_name, :sp_k, :sp_ip, :hit_k5, :hit_k6)
            """, row)
            written += 1
        except Exception as e:
            log.warning(f"DB write error for {row.get('sp_name')}: {e}")
    conn.commit()
    return written


def fetch_all(start: str, end: str, dry_run: bool = False) -> Dict:
    """
    Fetch SP K totals for all games in game_results between start and end.
    Uses game_results as the source of game_pks (avoids re-fetching the schedule).
    """
    conn = sqlite3.connect(DB_PATH)
    _ensure_schema(conn)

    # Pull distinct game_pks + dates from game_results for the target date range
    try:
        game_pks = conn.execute(
            "SELECT DISTINCT game_pk, game_date FROM game_results "
            "WHERE game_date >= ? AND game_date <= ? ORDER BY game_date",
            (start, end),
        ).fetchall()
    except Exception as e:
        log.error(f"Could not read game_results: {e}")
        conn.close()
        return {}

    summary = {"total": len(game_pks), "fetched": 0, "skipped": 0, "rows": 0}
    log.info(f"Found {len(game_pks)} distinct games in game_results ({start} to {end})")

    for game_pk, game_date in game_pks:
        # Skip if already in DB
        existing = conn.execute(
            "SELECT COUNT(*) FROM game_sp_results WHERE game_pk=?", (game_pk,)
        ).fetchone()[0]
        if existing >= 2:   # both teams already stored
            log.debug(f"  game {game_pk} already in game_sp_results — skipping")
            summary["skipped"] += 1
            continue

        if dry_run:
            log.info(f"  [DRY RUN] Would fetch SP Ks for game {game_pk} ({game_date})")
            summary["fetched"] += 1
            continue

        rows = fetch_sp_results(game_pk, game_date)
        if rows:
            n = write_sp_results(conn, rows)
            log.info(f"  game {game_pk} ({game_date}): {n} SP rows — "
                     + ", ".join(f"{r['sp_name']} {r['sp_k']}K" for r in rows))
            summary["fetched"] += 1
            summary["rows"] += n
        else:
            log.warning(f"  game {game_pk}: no SP data returned")
        time.sleep(REQUEST_DELAY)

    conn.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Fetch SP K totals for K prop backtest")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="List games without fetching")
    args = parser.parse_args()

    summary = fetch_all(args.start, args.end, dry_run=args.dry_run)
    print(f"\nDone. Games found: {summary.get('total', 0)} | "
          f"Fetched: {summary.get('fetched', 0)} | "
          f"Skipped (cached): {summary.get('skipped', 0)} | "
          f"Rows written: {summary.get('rows', 0)}")


if __name__ == "__main__":
    main()
