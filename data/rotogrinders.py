"""
data/rotogrinders.py — Fetch MLB implied run totals from RotoGrinders.

Source: https://rotogrinders.com/lineups/mlb (server-rendered HTML, no API key needed)
Provenance tag: "rotogrinders_scrape"

Design contract (matches project non-negotiable principles):
- FAIL LOUD: if fetch fails, page structure changed, or a game can't parse
  → that game gets NO implied total (excluded). NEVER a default or stale value.
- ONE fetch per run: cached to disk with timestamp. Freshness window = 20 min.
- Returns {team_abbr: implied_runs, ...} matching the model's own abbreviations.
- Caller must check provenance: use rg_implied_totals.get(team) which returns None
  for any team not in the dict — never .get(team, 4.5).

Usage:
    from data.rotogrinders import fetch_rg_implied_totals
    implied = fetch_rg_implied_totals()   # {team_abbr: float} or {}

Fail paths:
    - requests timeout / non-200  → logs warning, returns {}
    - BeautifulSoup parse finds 0 game-cards → logs warning, returns {}
    - Individual game missing O/U or team → that game excluded (logged), rest proceed
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, Optional, Tuple

import requests

log = logging.getLogger("rotogrinders")

# ─── constants ────────────────────────────────────────────────────────────────

_URL     = "https://rotogrinders.com/lineups/mlb"
_TIMEOUT = 15
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "_rg_implied_cache.json")
_CACHE_MAX_AGE = 1200   # 20 minutes in seconds — fetch once per run, not per interaction

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "DNT": "1",
}

# RotoGrinders team abbreviations → our model's abbreviations
# RG uses ESPN-style codes; our model uses its own. Map every divergence.
_RG_TO_MODEL: Dict[str, str] = {
    "TBR": "TB",    # Tampa Bay Rays
    "SFG": "SF",    # San Francisco Giants
    "SDP": "SD",    # San Diego Padres
    "KCR": "KC",    # Kansas City Royals
    "CHW": "CWS",   # Chicago White Sox
    "ATH": "OAK",   # Oakland/Sacramento Athletics
    "WAS": "WSH",   # Washington Nationals
    # All others are identical (TOR, BAL, NYY, BOS, ATL, NYM, PHI, MIA,
    # CIN, PIT, CHC, STL, MIL, HOU, DET, MIN, CLE, SEA, LAA, LAD, COL,
    # ARI, TEX)
}


def _rg_abbr_to_model(rg: str) -> str:
    """Convert RotoGrinders team abbreviation to model abbreviation."""
    return _RG_TO_MODEL.get(rg, rg)


# ─── HTML parsing ─────────────────────────────────────────────────────────────

def _parse_page(html: str) -> Dict[str, float]:
    """
    Parse RotoGrinders lineup page HTML.

    Returns {model_team_abbr: implied_runs}.
    Games where any required field is missing are excluded (logged, never defaulted).
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        log.error("rotogrinders: BeautifulSoup not installed — run: pip install beautifulsoup4")
        return {}

    soup = BeautifulSoup(html, "html.parser")
    game_cards = soup.select("div.game-card")

    if not game_cards:
        log.warning("rotogrinders: found 0 game-card divs — page structure may have changed")
        return {}

    log.info(f"rotogrinders: found {len(game_cards)} game cards")

    result: Dict[str, float] = {}
    parsed = 0
    skipped = 0

    for card in game_cards:
        # ── Team abbreviations ───────────────────────────────────────────────
        nameplates = card.select("span.team-nameplate-title")
        if len(nameplates) < 2:
            log.warning("rotogrinders: game-card missing team nameplates — skipping")
            skipped += 1
            continue

        rg_away = nameplates[0].get("data-abbr", "").strip()
        rg_home = nameplates[1].get("data-abbr", "").strip()
        if not rg_away or not rg_home:
            log.warning(f"rotogrinders: game-card missing data-abbr (got '{rg_away}'@'{rg_home}') — skipping")
            skipped += 1
            continue

        away_abbr = _rg_abbr_to_model(rg_away)
        home_abbr = _rg_abbr_to_model(rg_home)
        matchup   = f"{away_abbr}@{home_abbr}"

        # ── Game total (O/U) ────────────────────────────────────────────────
        total_el = card.select_one("span.vegas-bar-total-points")
        if not total_el:
            log.warning(f"rotogrinders: {matchup} missing vegas-bar-total-points — skipping")
            skipped += 1
            continue

        try:
            game_total = float(total_el.text.strip())
        except (ValueError, TypeError):
            log.warning(f"rotogrinders: {matchup} bad O/U value '{total_el.text}' — skipping")
            skipped += 1
            continue

        # ── Per-team implied runs (pre-split by RG) ─────────────────────────
        vbar = card.select_one("div.vegas-bar")
        if not vbar:
            log.warning(f"rotogrinders: {matchup} missing vegas-bar — skipping")
            skipped += 1
            continue

        non_total = [
            d for d in vbar.find_all("div", recursive=False)
            if "vegas-bar-total" not in d.get("class", [])
        ]
        if len(non_total) < 2:
            log.warning(f"rotogrinders: {matchup} vegas-bar has <2 team divs — skipping")
            skipped += 1
            continue

        away_link = non_total[0].find("a")
        home_link = non_total[1].find("a")

        away_impl = home_impl = None
        try:
            away_impl = float(away_link.text.strip()) if away_link else None
        except (ValueError, TypeError):
            pass
        try:
            home_impl = float(home_link.text.strip()) if home_link else None
        except (ValueError, TypeError):
            pass

        if away_impl is None or home_impl is None:
            log.warning(
                f"rotogrinders: {matchup} missing implied split "
                f"(away={away_impl} home={home_impl}) — skipping"
            )
            skipped += 1
            continue

        # Sanity: implied runs should sum roughly to game total (allow ±0.5 rounding)
        impl_sum = away_impl + home_impl
        if abs(impl_sum - game_total) > 0.6:
            log.warning(
                f"rotogrinders: {matchup} implied sum {impl_sum:.2f} ≠ O/U {game_total} "
                f"(diff={abs(impl_sum-game_total):.2f}) — skipping (possible parse error)"
            )
            skipped += 1
            continue

        result[away_abbr] = round(away_impl, 2)
        result[home_abbr] = round(home_impl, 2)
        log.info(
            f"rotogrinders: {matchup}  O/U={game_total}  "
            f"{away_abbr}={away_impl:.2f}  {home_abbr}={home_impl:.2f}"
            f"  [provenance: rotogrinders_scrape]"
        )
        parsed += 1

    log.info(f"rotogrinders: {parsed} games parsed, {skipped} skipped")
    if parsed == 0:
        log.warning("rotogrinders: 0 games parsed — returning empty (no implied totals will be used)")

    return result


# ─── cache ────────────────────────────────────────────────────────────────────

def _cache_read() -> Optional[Dict]:
    """Return cached data if fresh, else None."""
    if not os.path.exists(_CACHE_PATH):
        return None
    try:
        with open(_CACHE_PATH) as f:
            cached = json.load(f)
        age = time.time() - cached.get("fetched_at", 0)
        if age <= _CACHE_MAX_AGE:
            log.info(f"rotogrinders: cache hit (age {age:.0f}s, max {_CACHE_MAX_AGE}s)")
            return cached["data"]
        log.info(f"rotogrinders: cache stale ({age:.0f}s old) — will refetch")
    except Exception as e:
        log.warning(f"rotogrinders: cache read failed ({e}) — will refetch")
    return None


def _cache_write(data: Dict) -> None:
    try:
        with open(_CACHE_PATH, "w") as f:
            json.dump({"fetched_at": time.time(), "data": data}, f)
    except Exception as e:
        log.warning(f"rotogrinders: cache write failed ({e})")


# ─── public API ───────────────────────────────────────────────────────────────

def fetch_rg_implied_totals(force_refresh: bool = False) -> Dict[str, float]:
    """
    Fetch per-team implied run totals from RotoGrinders.

    Returns {team_abbr: implied_runs} tagged provenance="rotogrinders_scrape".
    Returns {} (empty) if fetch or parse fails — never returns a default value.

    Args:
        force_refresh: bypass cache and re-fetch even if cache is fresh.
    """
    if not force_refresh:
        cached = _cache_read()
        if cached is not None:
            return cached

    log.info(f"rotogrinders: fetching {_URL}")
    try:
        r = requests.get(_URL, headers=_HEADERS, timeout=_TIMEOUT)
    except Exception as e:
        log.warning(f"rotogrinders: fetch failed ({e}) — returning empty implied totals")
        return {}

    if r.status_code != 200:
        log.warning(
            f"rotogrinders: HTTP {r.status_code} — returning empty implied totals. "
            f"Games with no implied total will be excluded/greyed."
        )
        return {}

    content_len = len(r.text)
    if content_len < 10_000:
        log.warning(
            f"rotogrinders: response suspiciously short ({content_len} chars) — "
            f"may be a bot block or redirect. Returning empty."
        )
        return {}

    data = _parse_page(r.text)
    if data:
        _cache_write(data)
    else:
        log.warning("rotogrinders: parse returned empty dict — cache not updated")

    return data


def fetch_rg_weather(html: Optional[str] = None) -> Dict[str, Dict]:
    """
    Extract per-game park-specific weather strings from RotoGrinders.

    Returns {away@home_key: {"direction": "SW", "speed_mph": 9, "temp_f": 61}}.
    Outdoor games only — dome games are not in the weather-bar.

    Not wired into scoring yet (evaluation mode only).
    """
    if html is None:
        try:
            r = requests.get(_URL, headers=_HEADERS, timeout=_TIMEOUT)
            if r.status_code != 200:
                return {}
            html = r.text
        except Exception:
            return {}

    try:
        from bs4 import BeautifulSoup
        import re
    except ImportError:
        return {}

    soup = BeautifulSoup(html, "html.parser")
    result = {}

    for card in soup.select("div.game-card"):
        nameplates = card.select("span.team-nameplate-title")
        if len(nameplates) < 2:
            continue
        away = _rg_abbr_to_model(nameplates[0].get("data-abbr", ""))
        home = _rg_abbr_to_model(nameplates[1].get("data-abbr", ""))
        key  = f"{away}@{home}"

        wx_el = card.select_one("div.weather-bar-temps span")
        if not wx_el:
            continue
        raw = wx_el.get_text(" ", strip=True)
        # Expected format: "SW at 9 mph 61°" or "SW at 16 mph 85°"
        m = re.match(
            r"([A-Z]+)\s+at\s+(\d+)\s+mph\s+([\d.]+)",
            raw, re.IGNORECASE
        )
        if m:
            result[key] = {
                "direction": m.group(1).upper(),
                "speed_mph": int(m.group(2)),
                "temp_f":    float(m.group(3)),
                "raw":       raw,
            }

    return result


# ─── standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("Fetching RotoGrinders implied totals...")
    data = fetch_rg_implied_totals(force_refresh=True)

    if not data:
        print("FAILED — no data returned")
    else:
        print(f"\n{len(data)} teams with implied totals:")
        teams_sorted = sorted(data.items(), key=lambda kv: -kv[1])
        for team, impl in teams_sorted:
            print(f"  {team:6s}  {impl:.2f}R  [provenance: rotogrinders_scrape]")

    print("\nFetching RotoGrinders park weather...")
    wx = fetch_rg_weather()
    if wx:
        print(f"\n{len(wx)} outdoor games with weather:")
        for game, w in wx.items():
            print(f"  {game:14s}  {w['direction']} {w['speed_mph']}mph  {w['temp_f']:.0f}°F")
    else:
        print("No weather data (all domes, or fetch failed)")
