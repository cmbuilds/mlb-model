"""
data/backtest_fetcher.py — Fetch historical game results for backtesting.

Pulls actual per-batter outcomes (TB, H, AB, HR) from the MLB Stats API
and writes them to the `game_results` table in mlb_stats.db.

Usage:
    python3 data/backtest_fetcher.py --start 2026-04-01 --end 2026-06-01
    python3 data/backtest_fetcher.py --start 2026-04-01 --end 2026-06-01 --dry-run
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

DB_PATH = "data/mlb_stats.db"
MLB_API = "https://statsapi.mlb.com/api/v1"
REQUEST_DELAY = 0.3   # polite rate limiting (seconds between requests)


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
        CREATE TABLE IF NOT EXISTS game_results (
            game_pk     INTEGER NOT NULL,
            game_date   TEXT    NOT NULL,
            player_id   INTEGER NOT NULL,
            player_name TEXT,
            team        TEXT,
            opponent    TEXT,
            lineup_slot INTEGER,
            ab          INTEGER DEFAULT 0,
            h           INTEGER DEFAULT 0,
            doubles     INTEGER DEFAULT 0,
            triples     INTEGER DEFAULT 0,
            hr          INTEGER DEFAULT 0,
            tb          INTEGER DEFAULT 0,
            bb          INTEGER DEFAULT 0,
            k           INTEGER DEFAULT 0,
            rbi         INTEGER DEFAULT 0,
            hit_o15     INTEGER DEFAULT 0,
            hit_o05     INTEGER DEFAULT 0,
            hit_hr      INTEGER DEFAULT 0,
            PRIMARY KEY (game_pk, player_id)
        )
    """)
    # Migration: add hit_hr to existing tables that pre-date this column
    try:
        conn.execute("ALTER TABLE game_results ADD COLUMN hit_hr INTEGER DEFAULT 0")
        conn.execute("UPDATE game_results SET hit_hr = CASE WHEN hr >= 1 THEN 1 ELSE 0 END")
        log.info("Migrated game_results: added hit_hr column and backfilled from hr")
    except Exception:
        pass   # column already exists — skip silently
    conn.commit()


def fetch_games_for_date(date_str: str) -> List[Dict]:
    """Return list of {game_pk, home_team_id, away_team_id, home_name, away_name} for the date."""
    data = _get(f"{MLB_API}/schedule", params={
        "sportId": 1, "date": date_str, "gameType": "R",
    })
    if not data:
        return []
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            status = g.get("status", {}).get("detailedState", "")
            if status not in ("Final", "Game Over", "Completed Early"):
                continue
            home = g.get("teams", {}).get("home", {}).get("team", {})
            away = g.get("teams", {}).get("away", {}).get("team", {})
            games.append({
                "game_pk":       g["gamePk"],
                "home_team":     home.get("name", "?"),
                "away_team":     away.get("name", "?"),
                "home_team_id":  home.get("id"),
                "away_team_id":  away.get("id"),
            })
    return games


def fetch_batter_results(game_pk: int, home_team: str, away_team: str, game_date: str) -> List[Dict]:
    """Return per-batter stat rows for a completed game."""
    data = _get(f"{MLB_API}.1/game/{game_pk}/feed/live")
    if not data:
        return []

    rows = []
    box = data.get("liveData", {}).get("boxscore", {}).get("teams", {})
    for side, opp in (("home", away_team), ("away", home_team)):
        team_abbr = home_team if side == "home" else away_team
        batters = box.get(side, {}).get("batters", [])
        players = box.get(side, {}).get("players", {})
        batting_order = box.get(side, {}).get("battingOrder", [])

        for order_idx, pid in enumerate(batting_order[:9]):
            pdata = players.get(f"ID{pid}", {})
            stats = pdata.get("stats", {}).get("batting", {})
            person = pdata.get("person", {})
            name = person.get("fullName", f"ID{pid}")

            ab  = int(stats.get("atBats",        0) or 0)
            h   = int(stats.get("hits",           0) or 0)
            d   = int(stats.get("doubles",        0) or 0)
            t   = int(stats.get("triples",        0) or 0)
            hr  = int(stats.get("homeRuns",       0) or 0)
            bb  = int(stats.get("baseOnBalls",    0) or 0)
            k   = int(stats.get("strikeOuts",     0) or 0)
            rbi = int(stats.get("rbi",            0) or 0)
            tb  = int(stats.get("totalBases",     0) or (h - d - t - hr + 2*d + 3*t + 4*hr))

            rows.append({
                "game_pk":    game_pk,
                "game_date":  game_date,
                "player_id":  pid,
                "player_name": name,
                "team":       team_abbr,
                "opponent":   opp,
                "lineup_slot": order_idx + 1,
                "ab":  ab,
                "h":   h,
                "doubles": d,
                "triples": t,
                "hr":  hr,
                "tb":  tb,
                "bb":  bb,
                "k":   k,
                "rbi": rbi,
                "hit_o15": 1 if tb >= 2 else 0,   # O1.5 TB result
                "hit_o05": 1 if h  >= 1 else 0,   # O0.5 H result
                "hit_hr":  1 if hr >= 1 else 0,   # O0.5 HR result
            })
    return rows


def write_results(conn: sqlite3.Connection, rows: List[Dict]) -> int:
    written = 0
    for row in rows:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO game_results
                (game_pk, game_date, player_id, player_name, team, opponent,
                 lineup_slot, ab, h, doubles, triples, hr, tb, bb, k, rbi,
                 hit_o15, hit_o05, hit_hr)
                VALUES
                (:game_pk, :game_date, :player_id, :player_name, :team, :opponent,
                 :lineup_slot, :ab, :h, :doubles, :triples, :hr, :tb, :bb, :k, :rbi,
                 :hit_o15, :hit_o05, :hit_hr)
            """, row)
            written += 1
        except Exception as e:
            log.warning(f"DB write error for {row.get('player_name')}: {e}")
    conn.commit()
    return written


def fetch_date_range(start: str, end: str, dry_run: bool = False) -> Dict:
    """Fetch all game results between start and end (inclusive, YYYY-MM-DD)."""
    conn = sqlite3.connect(DB_PATH)
    _ensure_schema(conn)

    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    summary = {"dates": 0, "games": 0, "rows": 0, "skipped": 0}

    cur = s
    while cur <= e:
        date_str = cur.isoformat()
        games = fetch_games_for_date(date_str)
        if not games:
            log.info(f"{date_str}: no completed games")
            cur += timedelta(days=1)
            continue

        log.info(f"{date_str}: {len(games)} completed games")
        summary["dates"] += 1

        for g in games:
            pk = g["game_pk"]
            # Skip if already in DB
            existing = conn.execute(
                "SELECT COUNT(*) FROM game_results WHERE game_pk=?", (pk,)
            ).fetchone()[0]
            if existing > 0:
                log.debug(f"  game {pk} already fetched ({existing} rows) — skipping")
                summary["skipped"] += 1
                continue

            if dry_run:
                log.info(f"  [DRY RUN] Would fetch game {pk}: {g['away_team']} @ {g['home_team']}")
                summary["games"] += 1
                continue

            rows = fetch_batter_results(pk, g["home_team"], g["away_team"], date_str)
            if rows:
                n = write_results(conn, rows)
                log.info(f"  game {pk} ({g['away_team']}@{g['home_team']}): {n} batter rows written")
                summary["games"] += 1
                summary["rows"] += n
            else:
                log.warning(f"  game {pk}: no batter data returned")
            time.sleep(REQUEST_DELAY)

        cur += timedelta(days=1)

    conn.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Fetch historical batter results for backtest")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="List games without fetching")
    args = parser.parse_args()

    summary = fetch_date_range(args.start, args.end, dry_run=args.dry_run)
    print(f"\nDone. Dates processed: {summary['dates']} | Games: {summary['games']} | "
          f"Rows written: {summary['rows']} | Skipped (cached): {summary['skipped']}")


if __name__ == "__main__":
    main()
