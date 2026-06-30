"""
data/backtest_enrich_sp.py — Populate game_pitchers table from already-fetched game results.

Reads all game_pk values from game_results, fetches the SP for each game from the
live feed, and writes to game_pitchers. Skips games already present.

Usage: python3 data/backtest_enrich_sp.py
"""

import logging
import sqlite3
import time
from typing import Optional, Tuple

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH = "data/mlb_stats.db"
MLB_API = "https://statsapi.mlb.com/api/v1"
REQUEST_DELAY = 0.3


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_pitchers (
            game_pk      INTEGER PRIMARY KEY,
            game_date    TEXT,
            home_sp_id   INTEGER,
            home_sp_name TEXT,
            away_sp_id   INTEGER,
            away_sp_name TEXT,
            home_team    TEXT,
            away_team    TEXT
        )
    """)
    conn.commit()


def fetch_sp_for_game(game_pk: int) -> Optional[dict]:
    try:
        r = requests.get(
            f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
            timeout=15,
        )
        r.raise_for_status()
        d = r.json()
        pp = d.get("gameData", {}).get("probablePitchers", {})
        home = pp.get("home", {})
        away = pp.get("away", {})
        teams = d.get("gameData", {}).get("teams", {})
        home_team = teams.get("home", {}).get("abbreviation", "")
        away_team = teams.get("away", {}).get("abbreviation", "")
        return {
            "home_sp_id":   home.get("id"),
            "home_sp_name": home.get("fullName", ""),
            "away_sp_id":   away.get("id"),
            "away_sp_name": away.get("fullName", ""),
            "home_team":    home_team,
            "away_team":    away_team,
        }
    except Exception as e:
        log.warning(f"Failed to fetch SP for game {game_pk}: {e}")
        return None


def main():
    conn = sqlite3.connect(DB_PATH)
    _ensure_schema(conn)

    # Get distinct game_pks not yet in game_pitchers
    all_pks = [r[0] for r in conn.execute(
        "SELECT DISTINCT gr.game_pk, gr.game_date FROM game_results gr "
        "LEFT JOIN game_pitchers gp ON gr.game_pk = gp.game_pk "
        "WHERE gp.game_pk IS NULL"
    ).fetchall()]

    log.info(f"Found {len(all_pks)} games needing SP enrichment")
    written = 0

    for game_pk in all_pks:
        game_date = conn.execute(
            "SELECT game_date FROM game_results WHERE game_pk=? LIMIT 1", (game_pk,)
        ).fetchone()
        game_date = game_date[0] if game_date else ""

        sp_data = fetch_sp_for_game(game_pk)
        if sp_data:
            conn.execute("""
                INSERT OR REPLACE INTO game_pitchers
                (game_pk, game_date, home_sp_id, home_sp_name, away_sp_id, away_sp_name, home_team, away_team)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (game_pk, game_date, sp_data["home_sp_id"], sp_data["home_sp_name"],
                  sp_data["away_sp_id"], sp_data["away_sp_name"],
                  sp_data["home_team"], sp_data["away_team"]))
            conn.commit()
            written += 1
            if written % 50 == 0:
                log.info(f"  {written}/{len(all_pks)} games enriched...")
        time.sleep(REQUEST_DELAY)

    conn.close()
    log.info(f"Done. {written} games enriched with SP data.")


if __name__ == "__main__":
    main()
