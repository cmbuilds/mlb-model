#!/usr/bin/env python3
"""
⚾ MLB Total Bases Analyzer V1.0
================================
Fully automated over 1.5 total bases prop model for HardRock Bet.
Zero manual input. One click runs the entire pipeline.

SCORING: Singles=1, Doubles=2, Triples=3, HR=4
Walks, HBP, Stolen Bases = 0 TB (never counted)

Data sources:
- MLB Stats API (free, no key)
- pybaseball / Statcast
- Open-Meteo Weather API (free, no key)
- The Odds API (free tier)
- FanGraphs depth charts

Tiers:
- Tier 1 (85+): Strong play, parlay anchor
- Tier 2 (75-84): Viable, parlay filler
- Tier 3 (65-74): Marginal, single game only
- Below 65: No play
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
import time
import math
import os
import sqlite3
from datetime import datetime, timedelta, date
from itertools import combinations
from typing import Optional, List, Dict, Tuple, Any
import pytz

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="⚾ MLB TB Analyzer V1.9",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="auto"
)

# ============================================================================
# CSS - Mirror NHL model dark theme with MLB branding
# ============================================================================
st.markdown("""
<style>
/* Mobile responsive */
@media (max-width: 768px) {
    .stDataFrame { font-size: 11px !important; }
    .stDataFrame td, .stDataFrame th { padding: 4px 6px !important; font-size: 11px !important; }
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.2rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
    .block-container { padding: 1rem 0.5rem !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.8rem !important; padding: 8px 12px !important; }
}
.stDataFrame > div { overflow-x: auto !important; }

/* Tier color coding */
.tier1 { color: #00ff88; font-weight: bold; }
.tier2 { color: #ffdd00; font-weight: bold; }
.tier3 { color: #ff8800; font-weight: bold; }
.tier-no { color: #888888; }

/* Score bar */
.score-bar-container { background: #333; border-radius: 4px; height: 8px; width: 100%; }
.score-bar { height: 8px; border-radius: 4px; }

/* Status indicators */
.status-ok { color: #00ff88; }
.status-warn { color: #ffdd00; }
.status-err { color: #ff4444; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================
EST = pytz.timezone('US/Eastern')
DB_PATH = "mlb_tb_data.db"

# Tier thresholds (calibrated for ~47% base rate)
TIERS = {
    "🔒 TIER 1": 80,
    "✅ TIER 2": 70,
    "📊 TIER 3": 60,
    "❌ NO PLAY": 0,
}

# MLB Stadium coordinates for weather
STADIUM_COORDS = {
    "ARI": (33.4453, -112.0667, "Chase Field", True),   # dome=True
    "ATL": (33.8908, -84.4678, "Truist Park", False),
    "BAL": (39.2839, -76.6216, "Oriole Park", False),
    "BOS": (42.3467, -71.0972, "Fenway Park", False),
    "CHC": (41.9484, -87.6553, "Wrigley Field", False),
    "CWS": (41.8300, -87.6339, "Guaranteed Rate Field", False),
    "CIN": (39.0974, -84.5082, "Great American Ball Park", False),
    "CLE": (41.4962, -81.6853, "Progressive Field", False),
    "COL": (39.7559, -104.9942, "Coors Field", False),
    "DET": (42.3390, -83.0485, "Comerica Park", False),
    "HOU": (29.7573, -95.3555, "Minute Maid Park", True),  # retractable
    "KC":  (39.0517, -94.4803, "Kauffman Stadium", False),
    "LAA": (33.8003, -117.8827, "Angel Stadium", False),
    "LAD": (34.0739, -118.2400, "Dodger Stadium", False),
    "MIA": (25.7781, -80.2197, "loanDepot park", True),  # dome
    "MIL": (43.0280, -87.9712, "American Family Field", True),  # retractable
    "MIN": (44.9817, -93.2778, "Target Field", False),
    "NYM": (40.7571, -73.8458, "Citi Field", False),
    "NYY": (40.8296, -73.9262, "Yankee Stadium", False),
    "OAK": (37.7516, -122.2005, "Oakland Coliseum", False),
    "PHI": (39.9061, -75.1665, "Citizens Bank Park", False),
    "PIT": (40.4469, -80.0057, "PNC Park", False),
    "SD":  (32.7076, -117.1570, "Petco Park", False),
    "SEA": (47.5914, -122.3325, "T-Mobile Park", True),  # retractable
    "SF":  (37.7786, -122.3893, "Oracle Park", False),
    "STL": (38.6226, -90.1928, "Busch Stadium", False),
    "TB":  (27.7682, -82.6534, "Tropicana Field", True),  # dome
    "TEX": (32.7512, -97.0832, "Globe Life Field", True),  # retractable
    "TOR": (43.6414, -79.3894, "Rogers Centre", True),  # dome
    "WSH": (38.8730, -77.0074, "Nationals Park", False),
}

# Park TB factors (composite: weights singles/doubles/triples/HR by TB value)
# Based on 3-year rolling Statcast park factors, normalized to 1.00 = average
PARK_TB_FACTORS = {
    "ARI": 1.05, "ATL": 0.98, "BAL": 1.02, "BOS": 1.04, "CHC": 1.06,
    "CWS": 0.99, "CIN": 1.08, "CLE": 0.95, "COL": 1.22, "DET": 0.97,
    "HOU": 1.01, "KC": 0.98, "LAA": 0.96, "LAD": 1.00, "MIA": 0.91,
    "MIL": 0.99, "MIN": 1.04, "NYM": 0.97, "NYY": 1.07, "OAK": 0.93,
    "PHI": 1.05, "PIT": 0.96, "SD": 0.92, "SEA": 0.97, "SF": 0.94,
    "STL": 0.99, "TB": 0.95, "TEX": 1.03, "TOR": 1.01, "WSH": 0.97,
}

# Park HR factors specifically
PARK_HR_FACTORS = {
    "ARI": 1.10, "ATL": 0.99, "BAL": 1.08, "BOS": 1.07, "CHC": 1.15,
    "CWS": 1.05, "CIN": 1.18, "CLE": 0.93, "COL": 1.35, "DET": 0.96,
    "HOU": 1.02, "KC": 0.96, "LAA": 0.94, "LAD": 1.02, "MIA": 0.87,
    "MIL": 1.01, "MIN": 1.09, "NYM": 0.95, "NYY": 1.14, "OAK": 0.90,
    "PHI": 1.10, "PIT": 0.94, "SD": 0.88, "SEA": 0.95, "SF": 0.90,
    "STL": 1.00, "TB": 0.92, "TEX": 1.06, "TOR": 1.03, "WSH": 0.95,
}

# League average platoon adjustments (SLG points)
PLATOON_ADJ = {
    "RHB_vs_LHP": +33,   # RHB advantage vs LHP
    "LHB_vs_RHP": +56,   # LHB advantage vs RHP  
    "LHB_vs_LHP": -35,   # LHB disadvantage vs LHP
    "RHB_vs_RHP": 0,     # Baseline
}

# Team abbreviation mappings (MLB API -> our codes)
TEAM_ABB_MAP = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD", "Seattle Mariners": "SEA",
    "San Francisco Giants": "SF", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
    "Athletics": "OAK",
}

# ============================================================================
# DATABASE SETUP
# ============================================================================
def init_db():
    """Initialize SQLite database for picks and performance tracking."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS picks (
        pick_id TEXT PRIMARY KEY,
        date TEXT,
        game_id TEXT,
        player_id TEXT,
        player_name TEXT,
        team TEXT,
        opponent TEXT,
        sp_name TEXT,
        sp_hand TEXT,
        lineup_slot INTEGER,
        batter_hand TEXT,
        tb_line REAL,
        model_score REAL,
        model_prob REAL,
        tier TEXT,
        park TEXT,
        wind_speed REAL,
        wind_dir TEXT,
        temperature REAL,
        implied_total REAL,
        result TEXT DEFAULT 'pending',
        tb_actual INTEGER,
        xslg REAL,
        barrel_rate REAL,
        hard_hit_rate REAL,
        k_rate REAL,
        iso REAL,
        platoon_edge TEXT,
        hr_score REAL DEFAULT 0,
        created_at TEXT
    )
    """)
    # Migration: add hr_score if not exists in older DBs
    try:
        c.execute("ALTER TABLE picks ADD COLUMN hr_score REAL DEFAULT 0")
    except Exception:
        pass
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS parlays (
        parlay_id TEXT PRIMARY KEY,
        date TEXT,
        legs TEXT,
        num_legs INTEGER,
        combined_prob REAL,
        fair_payout REAL,
        result TEXT DEFAULT 'pending',
        profit_loss REAL,
        notes TEXT,
        created_at TEXT
    )
    """)
    
    conn.commit()
    conn.close()

def save_picks_to_db(picks: List[Dict], date_str: str):
    """Save scored picks to database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for p in picks:
        pick_id = f"{date_str}_{p.get('player_id', p['name'].replace(' ', '_'))}"
        c.execute("""
        INSERT OR REPLACE INTO picks 
        (pick_id, date, game_id, player_id, player_name, team, opponent, sp_name, sp_hand,
         lineup_slot, batter_hand, tb_line, model_score, model_prob, tier, park,
         wind_speed, wind_dir, temperature, implied_total, xslg, barrel_rate,
         hard_hit_rate, k_rate, iso, platoon_edge, hr_score, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,1.5,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            pick_id, date_str, p.get('game_id', ''), p.get('player_id', ''),
            p['name'], p['team'], p.get('opponent', ''), p.get('sp_name', 'TBD'),
            p.get('sp_hand', '?'), p.get('lineup_slot', 5), p.get('batter_hand', '?'),
            p['score'], p['prob'], p['tier'], p.get('park', ''),
            p.get('wind_speed', 0), p.get('wind_dir', 'N/A'), p.get('temperature', 70),
            p.get('implied_total', 4.5), p.get('xslg', 0), p.get('barrel_rate', 0),
            p.get('hard_hit_rate', 0), p.get('k_rate', 0), p.get('iso', 0),
            p.get('platoon_edge', ''), p.get('hr_score', 0), datetime.now(EST).isoformat()
        ))
    conn.commit()
    conn.close()

def load_picks_from_db(date_str: str = None) -> pd.DataFrame:
    """Load picks from database."""
    conn = sqlite3.connect(DB_PATH)
    if date_str:
        df = pd.read_sql("SELECT * FROM picks WHERE date=? ORDER BY model_score DESC", 
                         conn, params=(date_str,))
    else:
        df = pd.read_sql("SELECT * FROM picks ORDER BY date DESC, model_score DESC", conn)
    conn.close()
    return df

def update_pick_result(pick_id: str, result: str, tb_actual: int):
    """Update a pick with actual result."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE picks SET result=?, tb_actual=? WHERE pick_id=?",
              (result, tb_actual, pick_id))
    conn.commit()
    conn.close()

def save_parlay_to_db(parlay: Dict, date_str: str):
    """Save parlay to database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    parlay_id = f"{date_str}_parlay_{parlay.get('num_legs', 0)}leg_{int(time.time())}"
    c.execute("""
    INSERT OR REPLACE INTO parlays
    (parlay_id, date, legs, num_legs, combined_prob, fair_payout, notes, created_at)
    VALUES (?,?,?,?,?,?,?,?)
    """, (
        parlay_id, date_str, json.dumps(parlay.get('players', [])),
        parlay.get('num_legs', 0), parlay.get('combined_prob', 0),
        parlay.get('fair_payout', 0), parlay.get('notes', ''),
        datetime.now(EST).isoformat()
    ))
    conn.commit()
    conn.close()

# ============================================================================
# MLB STATS API DATA FETCHING
# ============================================================================
@st.cache_data(ttl=1800)
def fetch_schedule(date_str: str) -> List[Dict]:
    """Fetch today's MLB schedule with probable pitchers."""
    url = f"https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "date": date_str,
        "hydrate": "probablePitcher(note),team,linescore,flags,review"
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        games = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                game_info = {
                    "game_pk": game["gamePk"],
                    "game_time": game.get("gameDate", ""),
                    "status": game.get("status", {}).get("abstractGameState", "Preview"),
                    "home_team": game["teams"]["home"]["team"].get("abbreviation", 
                                 game["teams"]["home"]["team"]["name"]),
                    "away_team": game["teams"]["away"]["team"].get("abbreviation",
                                 game["teams"]["away"]["team"]["name"]),
                    "home_team_name": game["teams"]["home"]["team"]["name"],
                    "away_team_name": game["teams"]["away"]["team"]["name"],
                    "home_pitcher": None,
                    "away_pitcher": None,
                    "home_pitcher_id": None,
                    "away_pitcher_id": None,
                    "venue": game.get("venue", {}).get("name", ""),
                }
                # Probable pitchers
                home_pp = game["teams"]["home"].get("probablePitcher")
                away_pp = game["teams"]["away"].get("probablePitcher")
                if home_pp:
                    game_info["home_pitcher"] = home_pp.get("fullName", "TBD")
                    game_info["home_pitcher_id"] = home_pp.get("id")
                if away_pp:
                    game_info["away_pitcher"] = away_pp.get("fullName", "TBD")
                    game_info["away_pitcher_id"] = away_pp.get("id")
                
                # Normalize team codes
                for side in ["home_team", "away_team"]:
                    name_key = side + "_name"
                    if game_info[side] not in STADIUM_COORDS:
                        # Try mapping from full name
                        mapped = TEAM_ABB_MAP.get(game_info[name_key])
                        if mapped:
                            game_info[side] = mapped
                
                games.append(game_info)
        return games
    except Exception as e:
        st.warning(f"⚠️ Schedule fetch error: {e}")
        return []

@st.cache_data(ttl=900)
def fetch_lineup(game_pk: int) -> Dict:
    """Fetch confirmed lineups from boxscore endpoint."""
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        lineups = {"home": [], "away": []}
        
        for side in ["home", "away"]:
            team_data = data.get("teams", {}).get(side, {})
            batters = team_data.get("batters", [])
            batting_order = team_data.get("battingOrder", [])
            players = team_data.get("players", {})
            
            # Use batting order if available, else batters list
            order = batting_order if batting_order else batters
            
            for slot_idx, player_id in enumerate(order[:9]):
                player_key = f"ID{player_id}"
                player_info = players.get(player_key, {})
                person = player_info.get("person", {})
                stats = player_info.get("stats", {})
                position = player_info.get("position", {})
                
                # Skip pitchers
                if position.get("abbreviation") == "P":
                    continue
                
                bat_hand = person.get("batSide", {}).get("code", "")
                if not bat_hand or bat_hand == "?":
                    # Boxscore sometimes omits batSide — try person details
                    try:
                        pr = requests.get(f"https://statsapi.mlb.com/api/v1/people/{player_id}",
                                         params={"fields":"people,id,fullName,batSide"},
                                         timeout=5)
                        bat_hand = pr.json().get("people",[{}])[0].get("batSide",{}).get("code","R")
                    except Exception:
                        bat_hand = "R"  # fallback to R (most common)
                batter = {
                    "player_id": str(player_id),
                    "name": person.get("fullName", f"Player {player_id}"),
                    "lineup_slot": slot_idx + 1,
                    "batter_hand": bat_hand or "R",
                    "position": position.get("abbreviation", ""),
                }
                lineups[side].append(batter)
        
        return lineups
    except Exception as e:
        return {"home": [], "away": [], "error": str(e)}

@st.cache_data(ttl=3600)
def fetch_pitcher_info(pitcher_id: int) -> Dict:
    """Fetch pitcher details including handedness."""
    if not pitcher_id:
        return {}
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}"
    params = {"hydrate": "currentTeam,stats(type=season,group=pitching)"}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        person = data.get("people", [{}])[0]
        return {
            "id": pitcher_id,
            "name": person.get("fullName", "Unknown"),
            "hand": person.get("pitchHand", {}).get("code", "R"),
        }
    except:
        return {"id": pitcher_id, "name": "Unknown", "hand": "R"}

# ============================================================================
# DATA LAYER — pybaseball + FanGraphs
# Strategy: load all players in 2-3 bulk calls, then do fast dict lookups
# All column names auto-detected at runtime (no hardcoding)
# ============================================================================

# ── Disk cache helpers (survive st.cache_data expiry + FanGraphs blocks) ──────
# Streamlit Cloud: /tmp is ephemeral (lost on restart).
# Use a path relative to the app file so cache survives redeployments when
# committed to git (stat_cache/ is in .gitignore-excluded or included deliberately).
_DISK_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stat_cache")

def _disk_cache_path(name: str) -> str:
    os.makedirs(_DISK_CACHE_DIR, exist_ok=True)
    return os.path.join(_DISK_CACHE_DIR, f"{name}.pkl")

def _save_disk_cache(name: str, df: pd.DataFrame) -> None:
    """Persist a DataFrame to disk so it survives st.cache_data TTL expiry."""
    try:
        import pickle
        with open(_disk_cache_path(name), "wb") as f:
            pickle.dump(df, f)
    except Exception:
        pass

def _load_disk_cache(name: str, max_age_hours: int = 168) -> Optional[pd.DataFrame]:
    """Load a previously-saved DataFrame from disk. Returns None if absent or stale.
    Default 168h = 7 days — allows git-committed seed cache to survive a full week.
    """
    try:
        import pickle, time
        p = _disk_cache_path(name)
        if not os.path.exists(p):
            return None
        age_hours = (time.time() - os.path.getmtime(p)) / 3600
        if age_hours > max_age_hours:
            return None
        with open(p, "rb") as f:
            df = pickle.load(f)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass
    return None


@st.cache_data(ttl=10800)
def load_all_batting_stats(season: int = 2025) -> pd.DataFrame:
    """
    V1.5: FanGraphs-free batting pipeline.
    Primary sources (all use MLBAM player_id — zero crosswalk needed):
      1. Disk cache (< 6h)  — fastest path, skips all network calls
      2. Baseball Savant expected stats CSV  → xSLG, xwOBA, xBA
      3. Baseball Savant statcast leaderboard CSV → Barrel%, Hard Hit%, EV
      4. MLB Stats API season stats → SLG, K%(derived), BB%(derived), ISO(derived)
      5. FanGraphs JSON API (bonus layer if accessible) → wRC+, FIP, more
      6. Disk cache (< 7d)  — stale fallback if all live fetches fail
    All frames merged on MLBAM player_id → clean, no-crosswalk join.
    """
    import io as _io

    # ── 1. Early disk cache (< 6h) ────────────────────────────────────────
    _early = _load_disk_cache("batting_stats", max_age_hours=6)
    if _early is not None:
        st.session_state["_batting_source"] = "disk_cache_fresh"
        return _early

    _errs = []

    def _fetch_savant_csv(url: str, label: str) -> pd.DataFrame:
        """Fetch a Baseball Savant CSV leaderboard. Returns empty DF on failure."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/csv,*/*",
            }
            r = requests.get(url, headers=headers, timeout=25)
            if r.status_code == 200 and r.content:
                df = pd.read_csv(_io.StringIO(r.text))
                if not df.empty:
                    return df
                _errs.append(f"{label}: empty CSV")
            else:
                _errs.append(f"{label}: HTTP {r.status_code}")
        except Exception as e:
            _errs.append(f"{label}: {str(e)[:80]}")
        return pd.DataFrame()

    def _fetch_mlb_stats_api(season_yr: int) -> pd.DataFrame:
        """Fetch season batting stats from MLB Stats API (always unblocked)."""
        try:
            url = (
                f"https://statsapi.mlb.com/api/v1/stats"
                f"?stats=season&group=hitting&season={season_yr}"
                f"&limit=2000&offset=0&sportId=1"
            )
            r = requests.get(url, timeout=25)
            if r.status_code == 200:
                splits = r.json().get("stats", [{}])[0].get("splits", [])
                if splits:
                    rows = []
                    for s in splits:
                        p   = s.get("player", {})
                        st_ = s.get("stat", {})
                        pa  = int(st_.get("plateAppearances", 0) or 0)
                        so  = int(st_.get("strikeOuts", 0) or 0)
                        bb  = int(st_.get("baseOnBalls", 0) or 0)
                        try:
                            slg = float(st_.get("slg", 0) or 0)
                            avg = float(st_.get("avg", 0) or 0)
                        except:
                            slg = avg = 0.0
                        hr  = int(st_.get("homeRuns", 0) or 0)
                        tb  = int(st_.get("totalBases", 0) or 0)
                        d2  = int(st_.get("doubles", 0) or 0)
                        d3  = int(st_.get("triples", 0) or 0)
                        ab  = int(st_.get("atBats", 0) or 0)
                        try:
                            obp  = float(st_.get("obp", 0) or 0)
                            bab  = float(st_.get("babip", 0) or 0)
                        except:
                            obp = bab = 0.0

                        # Derive Statcast proxies from counting stats
                        # HR/PA is the best barrel% proxy available without Savant
                        # MLB avg HR/PA ~2.9% = ~7% barrel rate (1 barrel ≈ 0.41 HR)
                        hr_pa       = hr / pa if pa > 0 else 0.0
                        xbh_pa      = (d2 + d3 + hr) / pa if pa > 0 else 0.0
                        tb_pa       = tb / pa if pa > 0 else 0.0
                        # Hard hit proxy: high SLG + low K = contact quality
                        # Not perfect but better than league average default
                        hard_proxy  = min(0.55, max(0.25, slg * 0.55 + (1 - so/pa if pa > 0 else 0.77) * 0.20))

                        rows.append({
                            "mlbam_id":     str(p.get("id", "")),
                            "_name":        p.get("fullName", ""),
                            "SLG":          slg,
                            "AVG":          avg,
                            "OBP":          obp,
                            "ISO":          round(slg - avg, 3),
                            "K%":           round(so / pa, 3) if pa > 0 else 0.228,
                            "BB%":          round(bb / pa, 3) if pa > 0 else 0.082,
                            "PA":           pa,
                            "HR":           hr,
                            "doubles":      d2,
                            "triples":      d3,
                            "totalBases":   tb,
                            "BABIP":        bab,
                            # Derived Statcast proxies (used when Savant unavailable)
                            "hr_per_pa":    round(hr_pa, 4),      # barrel% proxy
                            "xbh_per_pa":   round(xbh_pa, 4),     # power contact proxy
                            "tb_per_pa":    round(tb_pa, 4),       # direct TB rate
                            "hard_proxy":   round(hard_proxy, 3),  # HH% proxy
                        })
                    df = pd.DataFrame(rows)
                    df = df[df["PA"] > 0]
                    return df
                _errs.append(f"MLB API {season_yr}: splits empty")
        except Exception as e:
            _errs.append(f"MLB API {season_yr}: {str(e)[:80]}")
        return pd.DataFrame()

    # ── 2. Fetch Baseball Savant expected stats (xSLG, xwOBA, xBA) ───────
    # Try current season first, fall back to prior season
    xstats_cur = _fetch_savant_csv(
        f"https://baseballsavant.mlb.com/leaderboard/expected_statistics"
        f"?type=batter&year={season+1}&position=&team=&min=1&csv=true",
        f"Savant xStats {season+1}"
    )
    xstats_pri = _fetch_savant_csv(
        f"https://baseballsavant.mlb.com/leaderboard/expected_statistics"
        f"?type=batter&year={season}&position=&team=&min=1&csv=true",
        f"Savant xStats {season}"
    )

    # ── 3. Fetch Statcast quality-of-contact data (Barrel%, HH%, EV) ───────
    # Cascade through multiple endpoints — the leaderboard CSV is often blocked
    # on cloud IPs, but the statcast_search endpoint uses different infrastructure.
    def _fetch_statcast_cascade(yr: int) -> pd.DataFrame:
        """
        Try every known source for Statcast quality-of-contact data.
        Priority order:
          1. Savant custom leaderboard JSON — different endpoint from blocked CSV
          2. MLB Stats API metricAverages — SAME domain that already works
          3. Standard Savant leaderboard CSV (often 502 on cloud)
          4. FanGraphs type=24 Statcast data
        Returns first DataFrame with barrel_batted_rate or equivalent.
        """
        _sc_required = {"barrel_batted_rate", "hard_hit_percent", "avg_exit_velocity"}

        # ── Attempt 1: Savant custom leaderboard (JSON, not CSV) ─────────────
        # This uses a completely different endpoint than the blocked /leaderboard/statcast CSV
        # The custom leaderboard returns HTML with embedded JSON — different rate limiter
        try:
            _cust_url = (
                f"https://baseballsavant.mlb.com/leaderboard/custom"
                f"?year={yr}&type=batter&filter=&sort=4&sortDir=desc&min=10"
                f"&selections=b_total_pa,b_ab,batting_avg,b_total_bases,"
                f"b_k_percent,b_bb_percent,b_xba,b_xslg,barrel_batted_rate,"
                f"hard_hit_percent,avg_hit_speed,b_woba,b_xwoba,b_obp"
                f"&chart=false&x=barrel_batted_rate&y=avg_hit_speed&r=no&csv=false"
            )
            _r = requests.get(_cust_url, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Referer": "https://baseballsavant.mlb.com/leaderboard/custom",
            })
            if _r.status_code == 200 and len(_r.content) > 1000:
                import re as _re
                # Savant embeds data as: var playerData = [...];
                _m = _re.search(r'var playerData\s*=\s*(\[.*?\]);', _r.text, _re.DOTALL)
                if _m:
                    import json as _json
                    _players = _json.loads(_m.group(1))
                    if _players:
                        _df = pd.DataFrame(_players)
                        # Rename Savant's column names to match our pipeline
                        _col_map = {
                            "player_id": "mlbam_id",
                            "avg_hit_speed": "avg_exit_velocity",
                            "b_k_percent": "k_percent",
                            "b_bb_percent": "bb_percent",
                            "b_xba": "est_ba", "b_xslg": "est_slg",
                            "b_xwoba": "est_woba", "b_obp": "OBP",
                        }
                        _df = _df.rename(columns=_col_map)
                        if "mlbam_id" in _df.columns:
                            _df["mlbam_id"] = _df["mlbam_id"].apply(
                                lambda x: str(int(float(x))) if pd.notna(x) and str(x) not in ("","nan") else ""
                            )
                        has = _sc_required.intersection(set(_df.columns))
                        if len(has) >= 2:
                            _errs.append(f"Savant custom JSON {yr}: OK {len(_df)} players")
                            return _df
        except Exception as _e:
            _errs.append(f"Savant custom JSON {yr}: {str(_e)[:60]}")

        # ── Attempt 2: MLB Stats API metricAverages ────────────────────────────
        # Same domain (statsapi.mlb.com) that already successfully returns 673 batters.
        # Returns exit velocity and launch angle — partial but real Statcast data.
        try:
            _metric_url = (
                f"https://statsapi.mlb.com/api/v1/stats"
                f"?stats=metricAverages&group=hitting&season={yr}"
                f"&sportId=1&limit=2000"
                f"&metrics=launchSpeed,launchAngle,launchSpinRate"
            )
            _mr = requests.get(_metric_url, timeout=20)
            if _mr.status_code == 200:
                _splits = _mr.json().get("stats", [{}])[0].get("splits", [])
                if _splits:
                    _rows = []
                    for _s in _splits:
                        _p = _s.get("player", {})
                        _st = _s.get("stat", {})
                        _ev = _st.get("launchSpeed", {})
                        _la = _st.get("launchAngle", {})
                        # metricAverages returns avg, median, etc. per metric
                        _avg_ev = _ev.get("average") if isinstance(_ev, dict) else _ev
                        _avg_la = _la.get("average") if isinstance(_la, dict) else _la
                        if _avg_ev:
                            # Derive hard_hit_percent from EV (hard hit = ≥95mph)
                            # We can get this from the metricAverages percentile data
                            _rows.append({
                                "mlbam_id": str(_p.get("id", "")),
                                "_name": _p.get("fullName", ""),
                                "avg_exit_velocity": float(_avg_ev),
                                "avg_launch_angle":  float(_avg_la) if _avg_la else None,
                            })
                    if _rows:
                        _mdf = pd.DataFrame(_rows)
                        _errs.append(f"MLB API metricAverages {yr}: OK {len(_mdf)} players")
                        return _mdf
        except Exception as _e:
            _errs.append(f"MLB API metricAverages {yr}: {str(_e)[:60]}")

        # ── Attempt 3: Standard Savant leaderboard CSV ────────────────────────
        for _url, _lbl in [
            (f"https://baseballsavant.mlb.com/leaderboard/statcast?year={yr}&position=&team=&min=1&type=batter&csv=true", "Savant Statcast leaderboard"),
            (f"https://baseballsavant.mlb.com/statcast_search/csv?player_type=batter&year={yr}&group_by=name&min_pitches=25&type=details", "Savant statcast_search CSV"),
        ]:
            _df = _fetch_savant_csv(_url, _lbl)
            if not _df.empty and len(_sc_required.intersection(set(_df.columns))) >= 2:
                return _df

        # ── Attempt 4: FanGraphs Statcast type=24 ─────────────────────────────
        try:
            import random as _rand
            _fg_r = requests.get(
                "https://www.fangraphs.com/api/leaders/major-league/data",
                params={"pos":"all","stats":"bat","lg":"all","qual":"0","type":"24",
                        "season":yr,"season1":yr,"ind":"0","team":"0","pageitems":"2000",
                        "pagenum":"1","minpa":"0","sortdir":"default","sortstat":"Barrel%"},
                headers={
                    "User-Agent": _rand.choice([
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
                    ]),
                    "Referer": "https://www.fangraphs.com/leaders/major-league",
                    "Accept": "application/json",
                }, timeout=10
            )
            if _fg_r.status_code == 200:
                _fg_rows = _fg_r.json().get("data", [])
                if _fg_rows:
                    _fg_df = pd.DataFrame(_fg_rows)
                    # FanGraphs uses xMLBAMID — normalize to mlbam_id before returning
                    for _id_col in ("xMLBAMID", "MLBAMID", "player_id", "IDfg"):
                        if _id_col in _fg_df.columns:
                            _fg_df["mlbam_id"] = _fg_df[_id_col].apply(
                                lambda x: str(int(float(x))) if pd.notna(x) and str(x) not in ("","nan") else ""
                            )
                            break
                    if "mlbam_id" not in _fg_df.columns:
                        _errs.append(f"FanGraphs Statcast {yr}: no ID column found, skipping")
                    else:
                        # Route through clean_fangraphs_df to strip HTML from Name column
                        _fg_df = clean_fangraphs_df(_fg_df)
                        _fg_sc_cols = {c for c in _fg_df.columns
                                       if any(k in c.lower() for k in ("barrel","hardhit","hard%","ev","exit"))}
                        if len(_fg_sc_cols) >= 1:
                            _errs.append(f"FanGraphs Statcast {yr}: OK {len(_fg_df)} players")
                            return _fg_df
        except Exception as _e:
            _errs.append(f"FanGraphs Statcast {yr}: {str(_e)[:60]}")

        return pd.DataFrame()

    sc_cur = _fetch_statcast_cascade(season + 1)
    sc_pri = _fetch_statcast_cascade(season)

    # ── 4. Fetch MLB Stats API (SLG, K%, BB%, ISO) ────────────────────────
    mlb_cur = _fetch_mlb_stats_api(season + 1)
    mlb_pri = _fetch_mlb_stats_api(season)

    # ── 5. Bat tracking — bat speed, blast rate (Savant 2023+) ────────────
    bat_cur = _fetch_savant_csv(
        f"https://baseballsavant.mlb.com/leaderboard/bat-tracking"
        f"?year={season+1}&minSwings=50&type=batter&csv=true",
        f"Savant BatTracking {season+1}"
    )
    bat_pri = _fetch_savant_csv(
        f"https://baseballsavant.mlb.com/leaderboard/bat-tracking"
        f"?year={season}&minSwings=100&type=batter&csv=true",
        f"Savant BatTracking {season}"
    )

    # ── 6. Pitch arsenal stats — batter run value by pitch type ───────────
    # Fetch each of the 6 primary pitch types; combine into one frame keyed
    # by mlbam_id with columns like rv_vs_FF, rv_vs_SL, woba_vs_FF, etc.
    # pitchType codes: FF=4-seam, SL=slider, CH=changeup, CU=curve,
    #                  SI=sinker, FC=cutter
    _pitch_type_frames = {}
    for _pt in ("FF", "SL", "CH", "CU", "SI", "FC"):
        _df = _fetch_savant_csv(
            f"https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
            f"?type=batter&pitchType={_pt}&year={season}&min=10&csv=true",
            f"Savant PitchArsenal batter vs {_pt}"
        )
        if not _df.empty:
            _pitch_type_frames[_pt] = _df

    if _errs:
        st.session_state["_fg_batting_errors"] = _errs

    # ── Check if we got anything useful ───────────────────────────────────
    got_savant   = not xstats_cur.empty or not xstats_pri.empty
    got_sc       = not sc_cur.empty or not sc_pri.empty
    got_mlb      = not mlb_cur.empty or not mlb_pri.empty
    got_bat      = not bat_cur.empty or not bat_pri.empty
    got_arsenal  = len(_pitch_type_frames) > 0

    if _errs:
        st.session_state["_fg_batting_errors"] = _errs

    if not got_savant and not got_mlb:
        # All live sources failed — serve stale disk cache
        stale = _load_disk_cache("batting_stats", max_age_hours=168)
        if stale is not None:
            st.session_state["_batting_source"] = "disk_cache_stale"
            return stale
        st.session_state["_batting_source"] = "failed"
        return pd.DataFrame()

    # ── Normalize Savant player_id column to "mlbam_id" string ───────────
    def _normalize_savant(df: pd.DataFrame, name_label: str) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        # Normalize any known ID column to mlbam_id
        for pid_col in ("player_id", "IDfg", "mlbam_id", "xMLBAMID", "MLBAMID",
                        "mlb_id", "MLBAM", "pitcher", "batter_id"):
            if pid_col in df.columns:
                df["mlbam_id"] = df[pid_col].apply(
                    lambda x: str(int(float(x))) if pd.notna(x) and str(x) not in ("", "nan", "None") else ""
                )
                break
        # Normalize name to _name
        for nc in ("last_name, first_name", "last_name,first_name", "Name", "name",
                   "PlayerName", "player_name"):
            if nc in df.columns:
                if "," in nc:
                    # Savant format: "last_name, first_name" is one column with value "Doe, John"
                    df["_name"] = df[nc].apply(
                        lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")])) if pd.notna(s) else ""
                    )
                else:
                    import re as _re_html
                    df["_name"] = df[nc].astype(str).apply(
                        lambda s: _re_html.sub(r'<[^>]+>', '', s).strip()
                    )
                break
        return df

    xstats_cur = _normalize_savant(xstats_cur, "xstats_cur")
    xstats_pri = _normalize_savant(xstats_pri, "xstats_pri")
    sc_cur     = _normalize_savant(sc_cur,     "sc_cur")
    sc_pri     = _normalize_savant(sc_pri,     "sc_pri")

    # ── Build primary frame: prefer current-season xstats, fall back to prior ─
    # Use prior season as base (more PAs), overlay current season where available
    base_xstats = xstats_pri if not xstats_pri.empty else xstats_cur
    base_sc     = sc_pri     if not sc_pri.empty     else sc_cur
    base_mlb    = mlb_pri    if not mlb_pri.empty    else mlb_cur

    # Pick whichever base has the most players AND has mlbam_id (required for all merges)
    bases = [(len(f), f) for f in [base_xstats, base_sc, base_mlb]
             if not f.empty and "mlbam_id" in f.columns]
    if not bases:
        stale = _load_disk_cache("batting_stats", max_age_hours=168)
        if stale is not None:
            st.session_state["_batting_source"] = "disk_cache_stale"
            return stale
        st.session_state["_batting_source"] = "failed"
        return pd.DataFrame()

    result = max(bases, key=lambda x: x[0])[1].copy()

    # ── Merge remaining frames on mlbam_id ────────────────────────────────
    def _merge_on_id(base: pd.DataFrame, other: pd.DataFrame,
                     keep_cols: list, suffix: str = "") -> pd.DataFrame:
        # Guard both frames — crash if either is missing mlbam_id
        if other.empty or "mlbam_id" not in other.columns:
            return base
        if "mlbam_id" not in base.columns:
            return base  # can't merge without join key on left side
        cols = [c for c in keep_cols if c in other.columns and
                (c not in base.columns or suffix)]
        if not cols:
            return base
        try:
            rename = {c: f"{c}{suffix}" for c in cols} if suffix else {}
            sub = other[["mlbam_id"] + cols].rename(columns=rename)
            sub = sub.drop_duplicates(subset=["mlbam_id"])
            return base.merge(sub, on="mlbam_id", how="left")
        except Exception:
            return base  # never crash the whole model on a merge failure

    xstats_cols = ["est_slg", "xslg", "est_woba", "xwoba", "est_ba",
                   "xba", "est_slg_minus_slg_diff", "xSLG", "xwOBA"]
    sc_cols     = ["barrel_batted_rate", "hard_hit_percent", "avg_exit_velocity",
                   "sweet_spot_percent", "Barrel%", "Hard%", "EV",
                   "avg_launch_speed", "launch_angle_avg"]
    mlb_cols    = ["SLG", "ISO", "K%", "BB%", "AVG", "OBP", "BABIP",
                   "hr_per_pa", "xbh_per_pa", "tb_per_pa", "hard_proxy",
                   "HR", "doubles", "triples", "totalBases"]

    for frame, cols in [(base_xstats, xstats_cols),
                        (base_sc,     sc_cols),
                        (base_mlb,    mlb_cols)]:
        if frame is not result:
            result = _merge_on_id(result, frame, cols)

    # Also blend in current-season data with _2026 suffix where different from base
    if not xstats_cur.empty and base_xstats is not xstats_cur:
        result = _merge_on_id(result, xstats_cur, ["est_slg", "xslg", "xSLG"], "_2026sc")
    if not sc_cur.empty and base_sc is not sc_cur:
        result = _merge_on_id(result, sc_cur,
                              ["barrel_batted_rate", "hard_hit_percent",
                               "avg_exit_velocity"], "_2026sc")
    if not mlb_cur.empty and base_mlb is not mlb_cur:
        result = _merge_on_id(result, mlb_cur, ["SLG", "K%", "BB%", "ISO"], "_2026")

    # ── Normalize column names for get_batter_stats() compatibility ───────
    col_aliases = {
        "est_slg":            "xSLG",
        "xslg":               "xSLG",
        "est_woba":           "xwOBA",
        "xwoba":              "xwOBA",
        "barrel_batted_rate": "Barrel%",
        "hard_hit_percent":   "Hard%",
        "avg_exit_velocity":  "EV",
        "avg_launch_speed":   "EV",
        "sweet_spot_percent": "Sweetspot%",
    }
    for src_col, tgt_col in col_aliases.items():
        if src_col in result.columns and tgt_col not in result.columns:
            result[tgt_col] = result[src_col]

    # ── Normalize xMLBAMID to mlbam_id for find_player_row() ─────────────
    if "mlbam_id" in result.columns and "xMLBAMID" not in result.columns:
        result["xMLBAMID"] = result["mlbam_id"].astype(str)

    # ── Ensure _name exists ───────────────────────────────────────────────
    if "_name" not in result.columns:
        for nc in ("Name", "name", "PlayerName", "last_name"):
            if nc in result.columns:
                result["_name"] = result[nc].astype(str)
                break

    # ── Normalize % columns ───────────────────────────────────────────────
    pct_cols = ["K%", "BB%", "Barrel%", "Hard%", "Sweetspot%",
                "barrel_batted_rate", "hard_hit_percent", "sweet_spot_percent"]
    for col in pct_cols:
        if col in result.columns:
            try:
                vals = pd.to_numeric(result[col], errors="coerce").dropna()
                if len(vals) > 0 and float(vals.max()) > 1.5:
                    result[col] = pd.to_numeric(result[col], errors="coerce") / 100.0
            except Exception:
                pass

    # ── Merge bat tracking (bat speed, blast rate, squared-up%) ─────────
    base_bat = bat_pri if not bat_pri.empty else bat_cur
    if not base_bat.empty:
        base_bat = _normalize_savant(base_bat.copy(), "bat")
        # Savant bat tracking may use different column names — alias them
        _bat_col_aliases = {
            "bat_speed_avg": "bat_speed",
            "avg_bat_speed": "bat_speed",
            "blast":         "blast_rate",
            "blasts_per_swing": "blast_rate",
            "squared_up":    "squared_up_rate",
            "sq_up_percent": "squared_up_rate",
        }
        for _src, _tgt in _bat_col_aliases.items():
            if _src in base_bat.columns and _tgt not in base_bat.columns:
                base_bat[_tgt] = base_bat[_src]
    bat_cols = ["bat_speed", "blast_rate", "squared_up_rate",
                "swing_length", "fast_swing_rate"]
    result = _merge_on_id(result, base_bat, bat_cols)
    # Also alias to friendlier names
    if "bat_speed" in result.columns and "BatSpeed" not in result.columns:
        result["BatSpeed"] = result["bat_speed"]
    if "blast_rate" in result.columns and "BlastRate" not in result.columns:
        result["BlastRate"] = result["blast_rate"]
    # Normalize blast_rate / squared_up_rate to decimals if in 0-100 range
    for _bc in ("blast_rate", "squared_up_rate", "fast_swing_rate"):
        if _bc in result.columns:
            try:
                _vals = pd.to_numeric(result[_bc], errors="coerce").dropna()
                if len(_vals) > 0 and float(_vals.max()) > 1.5:
                    result[_bc] = pd.to_numeric(result[_bc], errors="coerce") / 100.0
            except Exception:
                pass

    # ── Merge pitch arsenal batter splits → per-pitch run value columns ───
    # Savant columns: run_value_per100, ba, slg, woba, whiff_percent (per pitch type)
    if _pitch_type_frames:
        try:
            _arsenal_rows = {}  # keyed by mlbam_id
            for _pt, _df in _pitch_type_frames.items():
                if _df.empty:
                    continue
                _df = _normalize_savant(_df.copy(), f"arsenal_{_pt}")
                if "mlbam_id" not in _df.columns:
                    continue
                # Column names vary across Savant API versions
                # run_value_per100 is the primary — also check rv, run_value
                _rv_col = next((c for c in _df.columns
                                if "run_value" in c.lower() or c.lower() in ("rv",)), None)
                _woba_col = next((c for c in _df.columns
                                  if c.lower() in ("woba", "est_woba", "xwoba", "xwoba_by_ls")), None)
                _slg_col  = next((c for c in _df.columns
                                  if c.lower() in ("slg", "xslg")), None)
                _whiff_col = next((c for c in _df.columns
                                   if "whiff" in c.lower()), None)
                for _, _row in _df.iterrows():
                    try:
                        _mid = str(_row.get("mlbam_id", "")).strip()
                        if not _mid or _mid in ("", "nan"):
                            continue
                        if _mid not in _arsenal_rows:
                            _arsenal_rows[_mid] = {"mlbam_id": _mid}
                        if _rv_col and pd.notna(_row.get(_rv_col)):
                            _arsenal_rows[_mid][f"rv_vs_{_pt}"] = float(_row[_rv_col])
                        if _woba_col and pd.notna(_row.get(_woba_col)):
                            _w = float(_row[_woba_col])
                            # Some Savant versions return wOBA as 0-1, others 0-1000
                            _arsenal_rows[_mid][f"woba_vs_{_pt}"] = _w / 1000 if _w > 10 else _w
                        if _slg_col and pd.notna(_row.get(_slg_col)):
                            _arsenal_rows[_mid][f"slg_vs_{_pt}"] = float(_row[_slg_col])
                        if _whiff_col and pd.notna(_row.get(_whiff_col)):
                            _wh = float(_row[_whiff_col])
                            _arsenal_rows[_mid][f"whiff_vs_{_pt}"] = _wh / 100.0 if _wh > 1.5 else _wh
                    except Exception:
                        continue
            if _arsenal_rows:
                _arsenal_df = pd.DataFrame(list(_arsenal_rows.values()))
                _arsenal_cols = [c for c in _arsenal_df.columns if c != "mlbam_id"]
                if _arsenal_cols and "mlbam_id" in result.columns:
                    result = _merge_on_id(result, _arsenal_df, _arsenal_cols)
        except Exception as _ae:
            st.session_state["_batter_arsenal_err"] = str(_ae)[:120]

    # ── Try FanGraphs as bonus layer (wRC+, ISO refinement) ───────────────
    try:
        import random as _random
        fg_url = "https://www.fangraphs.com/api/leaders/major-league/data"
        fg_headers = {
            "User-Agent": _random.choice([
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/17.3 Safari/605.1.15",
            ]),
            "Referer": "https://www.fangraphs.com/leaders/major-league",
            "Accept": "application/json",
        }
        fg_params = {
            "age": "", "pos": "all", "stats": "bat", "lg": "all",
            "qual": "0", "minpa": "1", "season": season, "season1": season,
            "ind": "0", "team": "0", "pageitems": "2000", "pagenum": "1",
            "sortdir": "default", "type": "8", "sortstat": "WAR",
        }
        fg_r = requests.get(fg_url, params=fg_params, headers=fg_headers, timeout=10)
        if fg_r.status_code == 200:
            fg_rows = fg_r.json().get("data", [])
            if fg_rows:
                fg_df = pd.DataFrame(fg_rows)
                fg_df = clean_fangraphs_df(fg_df)
                # Merge wRC+, ISO onto result via xMLBAMID
                for id_col in ("xMLBAMID", "MLBAMID"):
                    if id_col in fg_df.columns and "xMLBAMID" in result.columns:
                        fg_sub = fg_df[[id_col, "wRC+", "ISO"]].dropna(subset=[id_col]).copy() \
                            if "wRC+" in fg_df.columns else fg_df[[id_col]].copy()
                        fg_sub = fg_sub.rename(columns={id_col: "xMLBAMID"})
                        for col in ["wRC+", "ISO"]:
                            if col in fg_sub.columns and col not in result.columns:
                                result = result.merge(fg_sub[["xMLBAMID", col]], on="xMLBAMID", how="left")
                        st.session_state["_batting_source"] = "savant+mlbapi+fangraphs"
                        break
    except Exception:
        pass

    if "_batting_source" not in st.session_state or \
       st.session_state.get("_batting_source") == "disk_cache_fresh":
        src_parts = []
        if got_savant: src_parts.append("savant")
        if got_mlb:    src_parts.append("mlbapi")
        st.session_state["_batting_source"] = "+".join(src_parts) if src_parts else "unknown"

    _save_disk_cache("batting_stats", result)
    return result


@st.cache_data(ttl=10800)
def load_all_pitching_stats(season: int = 2025) -> pd.DataFrame:
    """
    V1.5: FanGraphs-free pitching pipeline.
    Primary sources:
      1. Disk cache (< 6h)
      2. Baseball Savant pitcher statcast leaderboard → K%, Barrel% allowed, Hard% allowed, EV
      3. MLB Stats API season pitching → ERA, SO, BB, IP, WHIP (derived), Team
      4. FanGraphs (bonus, if accessible) → FIP, xFIP, GS, full team name
      5. Disk cache (< 7d) stale fallback
    """
    import io as _io

    # ── 1. Early disk cache ───────────────────────────────────────────────
    _early = _load_disk_cache("pitching_stats", max_age_hours=6)
    if _early is not None:
        st.session_state["_pitching_source"] = "disk_cache_fresh"
        return _early

    _errs = []

    def _fetch_savant_csv(url: str, label: str) -> pd.DataFrame:
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
            if r.status_code == 200 and r.content:
                df = pd.read_csv(_io.StringIO(r.text))
                if not df.empty:
                    return df
                _errs.append(f"{label}: empty")
            else:
                _errs.append(f"{label}: HTTP {r.status_code}")
        except Exception as e:
            _errs.append(f"{label}: {str(e)[:80]}")
        return pd.DataFrame()

    def _fetch_mlb_pitching(season_yr: int) -> pd.DataFrame:
        try:
            url = (
                f"https://statsapi.mlb.com/api/v1/stats"
                f"?stats=season&group=pitching&season={season_yr}"
                f"&limit=2000&offset=0&sportId=1"
            )
            r = requests.get(url, timeout=25)
            if r.status_code == 200:
                splits = r.json().get("stats", [{}])[0].get("splits", [])
                if splits:
                    rows = []
                    for s in splits:
                        p   = s.get("player", {})
                        st_ = s.get("stat", {})
                        tm  = s.get("team", {})
                        ip_str = str(st_.get("inningsPitched", "0") or "0")
                        try:
                            ip = float(ip_str)
                        except:
                            ip = 0.0
                        so  = int(st_.get("strikeOuts", 0) or 0)
                        bb  = int(st_.get("baseOnBalls", 0) or 0)
                        tbf = int(st_.get("battersFaced", 0) or 0)
                        h   = int(st_.get("hits", 0) or 0)
                        er  = int(st_.get("earnedRuns", 0) or 0)
                        gs  = int(st_.get("gamesStarted", 0) or 0)
                        g   = int(st_.get("gamesPlayed", 0) or 0)
                        era  = round(er / ip * 9, 2) if ip > 0 else 4.50
                        whip = round((h + bb) / ip, 3) if ip > 0 else 1.35
                        hr_a = int(st_.get("homeRuns", 0) or 0)
                        # Derived Statcast proxies from counting stats
                        # HR allowed / TBF → barrel% allowed proxy
                        # League avg: HR/TBF ~2.9% → barrel_allowed ~6.5%
                        barrel_proxy = min(0.18, hr_a / tbf / 0.029 * 0.065) if tbf > 0 else 0.065
                        # H/9 → hard hit proxy: more hits = more hard contact
                        h_per_9  = h / ip * 9 if ip > 0 else 9.0
                        hard_proxy_pit = min(0.50, max(0.25, 0.28 + (h_per_9 - 9.0) * 0.012))
                        rows.append({
                            "mlbam_id":      str(p.get("id", "")),
                            "_name":         p.get("fullName", ""),
                            "Team":          tm.get("abbreviation", ""),
                            "ERA":           era,
                            "WHIP":          whip,
                            "K%":            round(so / tbf, 3) if tbf > 0 else 0.228,
                            "BB%":           round(bb / tbf, 3) if tbf > 0 else 0.082,
                            "GS":            gs,
                            "G":             g,
                            "IP":            ip,
                            "HR_allowed":    hr_a,
                            "H_per_9":       round(h_per_9, 2),
                            # Proxies for Savant stats when Savant unavailable
                            "barrel_proxy":  round(barrel_proxy, 4),
                            "hard_proxy_pit":round(hard_proxy_pit, 3),
                        })
                    df = pd.DataFrame(rows)
                    df = df[df["IP"] > 0]
                    return df
                _errs.append(f"MLB pitching API {season_yr}: splits empty")
        except Exception as e:
            _errs.append(f"MLB pitching API {season_yr}: {str(e)[:80]}")
        return pd.DataFrame()

    # ── 2. Baseball Savant pitcher statcast leaderboard ───────────────────
    sc_pit_cur = _fetch_savant_csv(
        f"https://baseballsavant.mlb.com/leaderboard/statcast"
        f"?year={season+1}&position=SP-RP&team=&min=1&type=pitcher&csv=true",
        f"Savant Pitcher {season+1}"
    )
    sc_pit_pri = _fetch_savant_csv(
        f"https://baseballsavant.mlb.com/leaderboard/statcast"
        f"?year={season}&position=SP-RP&team=&min=1&type=pitcher&csv=true",
        f"Savant Pitcher {season}"
    )

    # ── 3. Pitcher arsenal mix — % usage per pitch type ──────────────────
    # n_ = percentage share for each pitch type (FF%, SL%, CH%, etc.)
    arsenal_mix_cur = _fetch_savant_csv(
        f"https://baseballsavant.mlb.com/leaderboard/pitch-arsenals"
        f"?year={season+1}&min=10&type=n_&hand=&csv=true",
        f"Savant PitcherArsenal {season+1}"
    )
    arsenal_mix_pri = _fetch_savant_csv(
        f"https://baseballsavant.mlb.com/leaderboard/pitch-arsenals"
        f"?year={season}&min=50&type=n_&hand=&csv=true",
        f"Savant PitcherArsenal {season}"
    )

    # ── 4. MLB Stats API pitching ──────────────────────────────────────────
    mlb_pit_cur = _fetch_mlb_pitching(season + 1)
    mlb_pit_pri = _fetch_mlb_pitching(season)

    got_savant = not sc_pit_cur.empty or not sc_pit_pri.empty
    got_mlb    = not mlb_pit_cur.empty or not mlb_pit_pri.empty

    if not got_savant and not got_mlb:
        stale = _load_disk_cache("pitching_stats", max_age_hours=168)
        if stale is not None:
            st.session_state["_pitching_source"] = "disk_cache_stale"
            return stale
        st.session_state["_pitching_source"] = "failed"
        return pd.DataFrame()

    def _normalize_savant_pit(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        for pid_col in ("player_id", "IDfg", "mlbam_id"):
            if pid_col in df.columns:
                df["mlbam_id"] = df[pid_col].apply(
                    lambda x: str(int(float(x))) if pd.notna(x) and str(x) not in ("","nan") else ""
                )
                break
        for nc in ("last_name, first_name", "Name", "name", "PlayerName"):
            if nc in df.columns:
                if "," in nc:
                    df["_name"] = df[nc].apply(
                        lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")])) if pd.notna(s) else ""
                    )
                else:
                    import re as _re_html2
                    df["_name"] = df[nc].astype(str).apply(
                        lambda s: _re_html2.sub(r'<[^>]+>', '', s).strip()
                    )
                break
        return df

    sc_pit_cur = _normalize_savant_pit(sc_pit_cur)
    sc_pit_pri = _normalize_savant_pit(sc_pit_pri)

    # Build result: prefer MLB API as base (has ERA, WHIP, Team, GS — critical for scoring)
    base_mlb = mlb_pit_cur if (not mlb_pit_cur.empty and len(mlb_pit_cur) > 50) else mlb_pit_pri
    base_sc  = sc_pit_pri  if not sc_pit_pri.empty else sc_pit_cur

    result = base_mlb.copy() if not base_mlb.empty else base_sc.copy()
    if result.empty:
        stale = _load_disk_cache("pitching_stats", max_age_hours=168)
        if stale is not None:
            st.session_state["_pitching_source"] = "disk_cache_stale"
            return stale
        st.session_state["_pitching_source"] = "failed"
        return pd.DataFrame()

    # Merge Savant statcast cols (barrel% allowed, hard hit% allowed, EV)
    savant_pit_cols = ["barrel_batted_rate", "hard_hit_percent", "avg_exit_velocity",
                       "Barrel%", "Hard%", "EV", "xera", "xERA"]
    for sc_frame in [base_sc, sc_pit_cur]:
        if sc_frame is not None and not sc_frame.empty and "mlbam_id" in sc_frame.columns:
            cols = [c for c in savant_pit_cols if c in sc_frame.columns and c not in result.columns]
            if cols:
                sub = sc_frame[["mlbam_id"] + cols].drop_duplicates("mlbam_id")
                result = result.merge(sub, on="mlbam_id", how="left")

    # Normalize Savant column names
    pit_aliases = {
        "barrel_batted_rate": "Barrel%",
        "hard_hit_percent":   "Hard%",
        "avg_exit_velocity":  "EV",
        "xera":               "xERA",
    }
    for src, tgt in pit_aliases.items():
        if src in result.columns and tgt not in result.columns:
            result[tgt] = result[src]

    # xMLBAMID alias for find_player_row()
    if "mlbam_id" in result.columns and "xMLBAMID" not in result.columns:
        result["xMLBAMID"] = result["mlbam_id"].astype(str)

    # Normalize % columns
    for col in ["K%", "BB%", "Barrel%", "Hard%"]:
        if col in result.columns:
            try:
                vals = pd.to_numeric(result[col], errors="coerce").dropna()
                if len(vals) > 0 and float(vals.max()) > 1.5:
                    result[col] = pd.to_numeric(result[col], errors="coerce") / 100.0
            except Exception:
                pass

    # ── Merge pitcher arsenal mix (pitch type usage %) ────────────────────
    # Savant columns: n_FF, n_SL, n_CH, n_CU, n_SI, n_FC, n_ST, etc.
    # Stored as pct_FF, pct_SL etc. for downstream get_pitcher_stats()
    base_arsenal = arsenal_mix_cur if not arsenal_mix_cur.empty else arsenal_mix_pri
    if not base_arsenal.empty:
        base_arsenal = base_arsenal.copy()
        # Expand ID column search — Savant uses player_id; also check alternatives
        _arsenal_id_found = False
        for pid_col in ("player_id", "pitcher", "mlb_id", "IDfg", "mlbam_id", "MLBAM"):
            if pid_col in base_arsenal.columns:
                base_arsenal["mlbam_id"] = base_arsenal[pid_col].apply(
                    lambda x: str(int(float(x))) if pd.notna(x) and str(x) not in ("","nan") else ""
                )
                _arsenal_id_found = True
                break
        # Find pitch % columns: n_FF, n_SI, n_SL, n_CH, n_CU, n_FC, n_ST, n_SV etc.
        # Savant names them n_XX where XX is 2-3 char pitch code
        _pit_mix_cols = [c for c in base_arsenal.columns
                         if c.startswith("n_") and 2 <= len(c) <= 7
                         and c[2:].isalpha()]
        _rename_mix = {c: f"pct_{c[2:]}" for c in _pit_mix_cols}
        base_arsenal = base_arsenal.rename(columns=_rename_mix)
        _pct_cols = list(_rename_mix.values())
        # GUARD: only proceed if mlbam_id exists in BOTH frames and we have pct cols
        if (_arsenal_id_found
                and "mlbam_id" in base_arsenal.columns
                and "mlbam_id" in result.columns
                and _pct_cols):
            try:
                sub = base_arsenal[["mlbam_id"] + _pct_cols].drop_duplicates("mlbam_id")
                # Normalize to decimals if stored as 0-100
                for _pc in _pct_cols:
                    try:
                        _v = pd.to_numeric(sub[_pc], errors="coerce").dropna()
                        if len(_v) > 0 and float(_v.max()) > 1.5:
                            sub[_pc] = pd.to_numeric(sub[_pc], errors="coerce") / 100.0
                    except Exception:
                        pass
                result = result.merge(sub, on="mlbam_id", how="left")
            except Exception as _e:
                # Never crash the whole model on arsenal merge failure
                st.session_state["_arsenal_merge_err"] = str(_e)[:120]

    # ── FanGraphs bonus layer — try multiple types and seasons ──────────────
    import random as _rand_pit
    _fg_pit_headers = {
        "User-Agent": _rand_pit.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
        ]),
        "Referer": "https://www.fangraphs.com/leaders/major-league",
        "Accept": "application/json",
    }
    _fg_pit_url = "https://www.fangraphs.com/api/leaders/major-league/data"

    def _normalize_fg_pit(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize FanGraphs pitching df: strip HTML, set mlbam_id from any ID column."""
        df = clean_fangraphs_df(df)
        # FG pitching may use xMLBAMID, MLBAMID, IDfg, or playerid
        for _c in ("xMLBAMID", "MLBAMID", "IDfg", "playerid", "PlayerID"):
            if _c in df.columns:
                df["_fg_mlbam"] = df[_c].apply(
                    lambda x: str(int(float(x))) if pd.notna(x) and str(x) not in ("","nan") else ""
                )
                break
        else:
            # Last resort: try _fg_id extracted from HTML playerid by clean_fangraphs_df
            if "_fg_id" in df.columns:
                df["_fg_mlbam"] = df["_fg_id"].astype(str)
        return df

    def _fg_merge(base: pd.DataFrame, fg: pd.DataFrame, cols: list, label: str) -> pd.DataFrame:
        """Merge FG pitching data into result on mlbam_id/xMLBAMID. Logs result."""
        if fg.empty:
            _errs.append(f"{label}: empty df")
            return base
        if "_fg_mlbam" not in fg.columns:
            _errs.append(f"{label}: no ID column found in {list(fg.columns[:8])}")
            return base
        avail = [c for c in cols if c in fg.columns and c not in base.columns]
        if not avail:
            _errs.append(f"{label}: no new cols (wanted {cols[:4]}, fg has {list(fg.columns[:8])})")
            return base
        sub = fg[["_fg_mlbam"] + avail].drop_duplicates("_fg_mlbam")
        sub = sub.rename(columns={"_fg_mlbam": "mlbam_id"})
        if "mlbam_id" not in base.columns:
            _errs.append(f"{label}: base has no mlbam_id")
            return base
        try:
            merged = base.merge(sub, on="mlbam_id", how="left")
            hit = merged[avail[0]].notna().sum()
            _errs.append(f"{label}: OK merged {hit}/{len(base)} pitchers, cols={avail}")
            return merged
        except Exception as _me:
            _errs.append(f"{label}: merge error {str(_me)[:60]}")
            return base

    # Type=8: FIP, xFIP, K%, BB%, WHIP, SIERA
    for _yr in [season + 1, season]:
        try:
            _r8 = requests.get(_fg_pit_url, params={
                "pos": "all", "stats": "pit", "lg": "all", "qual": "0",
                "type": "8", "season": _yr, "season1": _yr, "ind": "0",
                "team": "0", "pageitems": "1000", "pagenum": "1", "minip": "0",
            }, headers=_fg_pit_headers, timeout=12)
            if _r8.status_code == 200 and _r8.json().get("data"):
                _fg8 = _normalize_fg_pit(pd.DataFrame(_r8.json()["data"]))
                result = _fg_merge(result, _fg8,
                    ["FIP","xFIP","SIERA","SwStr%","O-Swing%"],
                    f"FG pit type=8 {_yr}")
            else:
                _errs.append(f"FG pit type=8 {_yr}: HTTP {_r8.status_code}")
        except Exception as _e:
            _errs.append(f"FG pit type=8 {_yr}: {str(_e)[:60]}")
        if any(c in result.columns for c in ["FIP","xFIP"]):
            break

    # Type=24: Barrel% allowed, HH% allowed, EV allowed (Statcast)
    for _yr in [season + 1, season]:
        try:
            _r24 = requests.get(_fg_pit_url, params={
                "pos": "all", "stats": "pit", "lg": "all", "qual": "0",
                "type": "24", "season": _yr, "season1": _yr, "ind": "0",
                "team": "0", "pageitems": "1000", "pagenum": "1", "minip": "0",
            }, headers=_fg_pit_headers, timeout=12)
            if _r24.status_code == 200 and _r24.json().get("data"):
                _fg24 = _normalize_fg_pit(pd.DataFrame(_r24.json()["data"]))
                _sc_want = [c for c in _fg24.columns
                            if any(k in c.lower() for k in ("barrel","hard","ev","exit","xera"))]
                result = _fg_merge(result, _fg24, _sc_want, f"FG pit type=24 {_yr}")
            else:
                _errs.append(f"FG pit type=24 {_yr}: HTTP {_r24.status_code}")
        except Exception as _e:
            _errs.append(f"FG pit type=24 {_yr}: {str(_e)[:60]}")
        if any(c in result.columns for c in ["Barrel%","Hard%","HardHit%"]):
            break

    # Store ALL errors including FG block errors (must be after FG block runs)
    st.session_state["_fg_pitching_errors"] = _errs

    src_parts = []
    if got_mlb:    src_parts.append("mlbapi")
    if got_savant: src_parts.append("savant")
    st.session_state["_pitching_source"] = "+".join(src_parts) if src_parts else "unknown"

    _save_disk_cache("pitching_stats", result)
    return result



def clean_fangraphs_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean FanGraphs API response.
    Robustly finds name column regardless of what it's called.
    Strips HTML if present, extracts FG playerid, normalizes % columns.
    """
    import re
    df = df.copy()

    # Step 1: Find name column — try known names then auto-detect
    name_col = None
    for candidate in ["Name", "name", "PlayerName", "playername", "player_name", "PLAYER"]:
        if candidate in df.columns:
            name_col = candidate
            break
    if not name_col:
        for col in df.columns:
            try:
                sample = df[col].dropna().head(5).astype(str)
                looks_like_names = sample.apply(lambda s:
                    bool(re.search(r'[A-Za-z]{2,}', s)) and
                    ' ' in s and len(s) < 100 and not s.startswith('http')
                ).sum()
                if looks_like_names >= 4:
                    name_col = col
                    break
            except Exception:
                pass

    # Step 2: Extract clean name and FG playerid from name column
    if name_col:
        raw = df[name_col].astype(str)

        def extract_id(s):
            m = re.search(r'playerid=(\d+)', s, re.IGNORECASE)
            return m.group(1) if m else None

        def extract_name(s):
            m = re.search(r'>([^<]+)<', s)
            if m:
                return m.group(1).strip()
            return re.sub(r'<[^>]+>', '', s).strip() or s.strip()

        df["_fg_id"] = raw.apply(extract_id)
        df["_name"] = raw.apply(extract_name)

        if df["_fg_id"].notna().any():
            df["_mlb_id"] = df["_fg_id"]
        else:
            for id_col in ["playerid", "PlayerID", "IDfg"]:
                if id_col in df.columns:
                    df["_mlb_id"] = df[id_col].astype(str)
                    break
    else:
        # Last resort: use playerid as id, no name
        for id_col in ["playerid", "PlayerID", "IDfg"]:
            if id_col in df.columns:
                df["_mlb_id"] = df[id_col].astype(str)
                break

    # Step 3: Normalize % columns to 0-1
    pct_names = {"Hard%","Barrel%","K%","BB%","LD%","GB%","FB%","HardHit%","SwStr%"}
    for col in df.columns:
        if col.startswith("_"):
            continue
        if "%" in str(col) or col in pct_names:
            try:
                vals = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(vals) > 0 and float(vals.max()) > 1.5:
                    df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0
            except Exception:
                pass
    return df



import unicodedata as _uda

def _norm(s: str) -> str:
    """Normalize name: lowercase, strip accents, remove punctuation."""
    s = str(s).lower().strip()
    s = ''.join(c for c in _uda.normalize('NFD', s) if _uda.category(c) != 'Mn')
    return s.replace('.','').replace("'",'').replace('-',' ').replace('  ',' ').strip()

def prepare_lookup_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-process a DataFrame for fast player lookup.
    Adds _norm_name column. Converts xMLBAMID to string for fast matching.
    Call this ONCE before the scoring loop.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    
    # Build normalized name index
    name_col = "_name" if "_name" in df.columns else next(
        (c for c in df.columns if c.lower() in ("name","playername")), None)
    if name_col and "_norm_name" not in df.columns:
        df["_norm_name"] = df[name_col].apply(_norm)
    
    # Pre-convert xMLBAMID / MLBAMID to string int for fast matching
    for _id_col in ("xMLBAMID", "MLBAMID"):
        if _id_col in df.columns:
            df[_id_col] = df[_id_col].apply(
                lambda x: str(int(float(x))) if pd.notna(x) and str(x) not in ("", "nan", "None") else ""
            )
    
    return df

def find_player_row(df: pd.DataFrame, player_name: str, mlb_id: str = "") -> Optional[pd.Series]:
    """
    Fast player lookup. Priority:
    1. xMLBAMID match (FanGraphs stores MLB MLBAM IDs — perfect crosswalk)
    2. Exact normalized name match
    3. Last name + first initial
    Requires prepare_lookup_df() to have been called first.
    """
    if df is None or df.empty or not player_name:
        return None

    # 1. xMLBAMID / MLBAMID match — most reliable (FanGraphs MLBAM player ID crosswalk)
    for _id_col in ("xMLBAMID", "MLBAMID"):
        if mlb_id and _id_col in df.columns:
            try:
                m = df[df[_id_col].astype(str).str.split(".").str[0] == str(mlb_id)]
                if not m.empty:
                    return m.iloc[0]
            except Exception:
                pass

    # 2. _mlb_id fallback
    if mlb_id and "_mlb_id" in df.columns:
        try:
            m = df[df["_mlb_id"].astype(str) == str(mlb_id)]
            if not m.empty:
                return m.iloc[0]
        except Exception:
            pass

    # 3. Name match via pre-built _norm_name index
    norm = _norm(player_name)
    parts = norm.split()
    last  = parts[-1] if parts else ""
    first = parts[0]  if len(parts) > 1 else ""

    nc = "_norm_name" if "_norm_name" in df.columns else None
    if nc:
        m = df[df[nc] == norm]
        if not m.empty:
            return m.iloc[0]

        if last:
            cands = df[df[nc].str.contains(last, na=False, regex=False)]
            if not cands.empty:
                if first:
                    refined = cands[cands[nc].str.contains(first[:3], na=False, regex=False)]
                    if not refined.empty:
                        return refined.iloc[0]
                if len(cands) == 1:
                    return cands.iloc[0]

    return None


def safe_get(row: pd.Series, *col_names, default=None, as_pct=False):
    """
    Try multiple column names, return first non-null value as plain Python float.
    Handles numpy int64/float64 types that cause silent math failures.
    """
    for col in col_names:
        if col in row.index and pd.notna(row[col]):
            try:
                val = row[col]
                # Convert any numpy type to native Python float
                val = float(val)
                if as_pct and val > 1:
                    val = val / 100
                return val
            except (TypeError, ValueError):
                pass
    return default

def get_batter_stats(player_name: str, mlb_id: str,
                     batting_df: pd.DataFrame,
                     statcast_df: pd.DataFrame = None) -> Dict:
    """
    Extract batter stats from pre-loaded DataFrames.
    All percentages normalized to decimals (0.23 not 23%).
    Falls back to MLB league averages if player not found.
    V1.3: Handles _2026 / _2026sc suffixes from new merge strategy.
    """
    # 2025 MLB league averages (all as decimals)
    stats = {
        "slg_proxy":        0.398,
        "iso_proxy":        0.165,
        "k_rate":           0.228,
        "bb_rate":          0.082,
        "wrc_plus":         100.0,
        "woba":             0.315,
        "barrel_rate":      0.070,
        "hard_hit_rate":    0.370,
        "soft_hit_rate":    0.155,
        "exit_velocity_avg":88.5,
        "ev50":             0.0,     # 0 = not populated; real data from Savant only
        "sweet_spot_rate":  0.305,
        "tb_per_game":      0.85,
        "bat_speed":        0.0,     # 0 = not populated; real data from Savant only
        "blast_rate":       0.0,     # 0 = not populated; real data from Savant only
        "squared_up_rate":  0.265,   # MLB avg squared-up rate ~26.5%
        "sprint_speed":     27.0,    # MLB avg sprint speed ft/sec (Savant)
        # Pitch-type run values vs each pitch (league avg = 0.0 = neutral)
        "rv_vs_FF":         0.0,     # vs 4-seam fastball
        "rv_vs_SL":         0.0,     # vs slider
        "rv_vs_CH":         0.0,     # vs changeup
        "rv_vs_CU":         0.0,     # vs curveball
        "rv_vs_SI":         0.0,     # vs sinker
        "rv_vs_FC":         0.0,     # vs cutter
        "data_source":      "league_avg",
    }

    row = find_player_row(batting_df, player_name, mlb_id)

    if row is not None:
        # ── SLG / power — prefer xSLG over SLG ───────────────────────────
        xslg = safe_get(row, 'xSLG', 'xslg', default=None)
        slg  = safe_get(row, 'SLG',  'slg',  default=None)
        if xslg and 0.050 < xslg < 1.200:
            stats["slg_proxy"] = xslg
        elif slg and 0.050 < slg < 1.200:
            stats["slg_proxy"] = slg

        # ── ISO ────────────────────────────────────────────────────────────
        iso = safe_get(row, 'ISO', 'iso', default=None)
        if iso and 0 < iso < 0.700:
            stats["iso_proxy"] = iso

        # ── K% ────────────────────────────────────────────────────────────
        k = safe_get(row, 'K%', 'k_percent', default=None)
        if k is not None and k > 0:
            stats["k_rate"] = k if k < 1 else k / 100

        # ── BB% ───────────────────────────────────────────────────────────
        bb = safe_get(row, 'BB%', 'bb_percent', default=None)
        if bb is not None and bb > 0:
            stats["bb_rate"] = bb if bb < 1 else bb / 100

        # ── wRC+ ──────────────────────────────────────────────────────────
        wrc = safe_get(row, 'wRC+', 'wRC', default=None)
        if wrc and wrc > 0:
            stats["wrc_plus"] = float(wrc)

        # ── wRC+ proxy from OBP when FanGraphs unavailable ────────────────
        # wRC+ correlates ~0.87 with OBP. MLB API gives us OBP.
        # wrc_proxy = 100 + (OBP - 0.316) / 0.316 * 100 * 0.87
        if stats["wrc_plus"] == 100.0:  # still at default
            _obp_for_wrc = safe_get(row, 'OBP', 'obp', default=None)
            if _obp_for_wrc and 0.250 < _obp_for_wrc < 0.550:
                _wrc_proxy = 100.0 + (_obp_for_wrc - 0.316) / 0.316 * 87.0
                stats["wrc_plus"] = round(max(50.0, min(220.0, _wrc_proxy)), 1)

        # ── wOBA ──────────────────────────────────────────────────────────
        woba = safe_get(row, 'xwOBA', 'wOBA', 'xwoba', 'woba', default=None)
        if woba and 0.100 < woba < 0.700:
            stats["woba"] = woba

        # ── Barrel% ───────────────────────────────────────────────────────
        barrel = safe_get(row, 'Barrel%', 'Barrel', 'barrel_batted_rate',
                          'barrel_rate', default=None)
        if barrel is not None and barrel > 0:
            stats["barrel_rate"] = barrel if barrel < 1 else barrel / 100

        # ── Hard Hit% ─────────────────────────────────────────────────────
        hard = safe_get(row, 'Hard%', 'HardHit%', 'hard_hit_percent',
                        'hard_hit_rate', default=None)
        if hard is not None and hard > 0:
            stats["hard_hit_rate"] = hard if hard < 1 else hard / 100

        # ── Exit Velocity ─────────────────────────────────────────────────
        ev = safe_get(row, 'EV', 'avg_exit_velocity', 'exit_velocity_avg', default=None)
        if ev and ev > 50:
            stats["exit_velocity_avg"] = ev

        # ── Sweet Spot% ───────────────────────────────────────────────────
        # FanGraphs uses several column names across seasons/API versions
        sweet = safe_get(row, 'Sweetspot%', 'SweetSpot%', 'Sweet-Spot%',
                         'LA Sweet-Spot%', 'sweet_spot_percent',
                         'sweet_spot_rate', 'SweetSpot', default=None)
        if sweet is not None and sweet > 0:
            stats["sweet_spot_rate"] = sweet if sweet < 1 else sweet / 100

        # Label source accurately based on what columns were actually found
        if any(c in row.index for c in ('xSLG','est_slg','xslg','est_woba')):
            stats["data_source"] = "savant_xstats"
        elif any(c in row.index for c in ('Barrel%','Hard%','barrel_batted_rate','hard_hit_percent')):
            stats["data_source"] = "savant_statcast"
        elif any(c in row.index for c in ('hr_per_pa','hard_proxy','tb_per_pa')):
            stats["data_source"] = "mlbapi"
        else:
            # Row found but columns unclear — could be FanGraphs if accessible,
            # or a future data source. Label generically.
            stats["data_source"] = "matched"

        # ── EV50 (hardest 50% avg exit velocity — better power signal) ─────
        ev50 = safe_get(row, 'ev50', 'EV50', 'xEV50', default=None)
        if ev50 and ev50 > 50:
            stats["ev50"] = ev50

        # ── MLB Stats API proxy fallback ──────────────────────────────────
        # When Savant is blocked, use counting-stat-derived proxies for
        # Barrel%, HH%, and xSLG rather than league-average defaults.
        # Proxies are less accurate but far better than 7.0%/37.0% for everyone.

        # Barrel% proxy: HR/PA scaled to barrel rate
        # League: HR/PA ~2.9% = barrel ~7%; Judge HR/PA ~8% = barrel ~20%
        # Formula: barrel_proxy = hr_per_pa / 0.029 * 0.07
        hr_pa = safe_get(row, 'hr_per_pa', default=None)
        if hr_pa is not None and stats["barrel_rate"] == 0.070:
            # Only override if we're still at the default
            barrel_proxy = min(0.25, hr_pa / 0.029 * 0.070)
            if barrel_proxy > 0.010:  # at least some HR production
                stats["barrel_rate"] = round(barrel_proxy, 4)

        # Hard hit% proxy: derived from SLG-based formula in MLB API fetch
        hard_p = safe_get(row, 'hard_proxy', default=None)
        if hard_p is not None and stats["hard_hit_rate"] == 0.370:
            stats["hard_hit_rate"] = round(hard_p, 4)

        # xSLG proxy: SLG is a reasonable proxy when xSLG unavailable
        # (xSLG ~ SLG * 1.02 on average; use direct SLG if nothing better)
        slg_raw = safe_get(row, 'SLG', 'slg', default=None)
        if slg_raw and 0.100 < slg_raw < 0.900 and stats["slg_proxy"] == 0.398:
            stats["slg_proxy"] = slg_raw

        # TB/PA proxy: direct total bases rate for power scoring
        tb_pa = safe_get(row, 'tb_per_pa', default=None)
        if tb_pa and tb_pa > 0:
            stats["tb_per_game"] = tb_pa * 4.2  # approx PA/game

        # OBP as wOBA proxy when xwOBA unavailable
        obp = safe_get(row, 'OBP', 'obp', default=None)
        if obp and 0.200 < obp < 0.600 and stats["woba"] == 0.315:
            # wOBA ≈ OBP * 0.82 (rough linear scaling)
            stats["woba"] = round(obp * 0.82, 3)

        # ── Bat tracking (bat speed, blast rate) ──────────────────────────
        bs = safe_get(row, 'bat_speed', 'BatSpeed', default=None)
        if bs and bs > 30:
            stats["bat_speed"] = bs
        br = safe_get(row, 'blast_rate', 'BlastRate', default=None)
        if br is not None and br >= 0:
            stats["blast_rate"] = br if br < 1 else br / 100
        sq = safe_get(row, 'squared_up_rate', default=None)
        if sq is not None and sq >= 0:
            stats["squared_up_rate"] = sq if sq < 1 else sq / 100

        # ── Sprint speed (ft/sec from Savant sprint speed leaderboard) ────
        ss = safe_get(row, 'sprint_speed', 'SprintSpeed', 'hp_to_1b', default=None)
        if ss and 15 < ss < 35:  # valid ft/sec range (MLB 23-31 typical)
            stats["sprint_speed"] = ss

        # ── Pitch-type run values (from pitch arsenal batter splits) ──────
        for _pt in ("FF", "SL", "CH", "CU", "SI", "FC"):
            rv = safe_get(row, f"rv_vs_{_pt}", default=None)
            if rv is not None:
                stats[f"rv_vs_{_pt}"] = float(rv)
            woba_pt = safe_get(row, f"woba_vs_{_pt}", default=None)
            if woba_pt and 0.100 < woba_pt < 0.900:
                stats[f"woba_vs_{_pt}"] = float(woba_pt)

        # ── 2026 YTD blending (_2026 suffix from new merge strategy) ──────
        # Advanced stats blend (K%, SLG, ISO, wRC+)
        xslg_26  = safe_get(row, 'xSLG_2026', 'xSLG_2026sc', default=None)
        if xslg_26 and 0.100 < xslg_26 < 1.200:
            stats["slg_proxy"] = stats["slg_proxy"] * 0.60 + xslg_26 * 0.40

        barrel_26 = safe_get(row, 'Barrel%_2026', 'Barrel%_2026sc',
                             'Barrel_2026', 'Barrel_2026sc', default=None)
        if barrel_26 and barrel_26 > 0:
            b26 = barrel_26 if barrel_26 < 1 else barrel_26 / 100
            stats["barrel_rate"] = stats["barrel_rate"] * 0.65 + b26 * 0.35

        k_26 = safe_get(row, 'K%_2026', default=None)
        if k_26 and k_26 > 0:
            k26 = k_26 if k_26 < 1 else k_26 / 100
            stats["k_rate"] = stats["k_rate"] * 0.65 + k26 * 0.35

        hard_26 = safe_get(row, 'Hard%_2026', 'Hard%_2026sc',
                           'HardHit%_2026', 'HardHit%_2026sc', default=None)
        if hard_26 and hard_26 > 0:
            h26 = hard_26 if hard_26 < 1 else hard_26 / 100
            stats["hard_hit_rate"] = stats["hard_hit_rate"] * 0.65 + h26 * 0.35

    return stats

def get_pitcher_stats(pitcher_name: str, pitcher_mlb_id: str,
                      pitching_df: pd.DataFrame) -> Dict:
    """Extract pitcher vulnerability stats from pre-loaded DataFrame.
    Includes WHIP for O0.5 model and downstream parsing.
    """
    stats = {
        "k_rate_allowed":   0.228,
        "bb_rate_allowed":  0.082,
        "hard_hit_allowed": 0.360,
        "barrel_allowed":   0.070,
        "era":              4.20,
        "fip":              4.20,
        "xfip":             4.10,
        "whip":             1.30,
        # Pitch arsenal mix — MLB avg usage (2025 approx)
        "pct_FF":           0.35,   # 4-seam fastball
        "pct_SI":           0.17,   # sinker
        "pct_SL":           0.20,   # slider
        "pct_CH":           0.10,   # changeup
        "pct_CU":           0.11,   # curveball
        "pct_FC":           0.07,   # cutter
        "data_source":      "league_avg",
    }

    row = find_player_row(pitching_df, pitcher_name, pitcher_mlb_id)

    if row is not None:
        k = safe_get(row, 'K%', default=None)
        if k is not None and k > 0:
            stats["k_rate_allowed"] = k if k < 1 else k / 100

        bb = safe_get(row, 'BB%', default=None)
        if bb is not None and bb > 0:
            stats["bb_rate_allowed"] = bb if bb < 1 else bb / 100

        hard = safe_get(row, 'Hard%', default=None)
        if hard is not None and hard > 0:
            stats["hard_hit_allowed"] = hard if hard < 1 else hard / 100

        barrel = safe_get(row, 'Barrel%', default=None)
        if barrel is not None and barrel > 0:
            stats["barrel_allowed"] = barrel if barrel < 1 else barrel / 100

        era = safe_get(row, 'ERA', default=None)
        if era and 0 < era < 20:
            stats["era"] = era

        fip = safe_get(row, 'FIP', 'xFIP', default=None)
        if fip and 0 < fip < 20:
            stats["fip"] = fip

        whip = safe_get(row, 'WHIP', default=None)
        if whip and 0 < whip < 5:
            stats["whip"] = whip

        # ── Pitch arsenal mix (pct_FF, pct_SL, etc.) ─────────────────────
        for _pt in ("FF", "SI", "SL", "CH", "CU", "FC"):
            _pct = safe_get(row, f"pct_{_pt}", default=None)
            if _pct is not None and _pct >= 0:
                stats[f"pct_{_pt}"] = _pct if _pct < 1 else _pct / 100

        # ── MLB Stats API proxy fallback for Barrel%/HH% allowed ──────────
        # When Savant is blocked, use counting-stat proxies
        barrel_p = safe_get(row, 'barrel_proxy', default=None)
        if barrel_p is not None and stats["barrel_allowed"] == 0.065:
            stats["barrel_allowed"] = round(max(0.010, barrel_p), 4)

        hard_p = safe_get(row, 'hard_proxy_pit', default=None)
        if hard_p is not None and stats["hard_hit_allowed"] == 0.360:
            stats["hard_hit_allowed"] = round(hard_p, 3)

        # Label pitcher source based on available columns
        if any(c in row.index for c in ('barrel_proxy','hard_proxy_pit','H_per_9')):
            stats["data_source"] = "mlbapi"
        elif any(c in row.index for c in ('Barrel%','Hard%','barrel_batted_rate')):
            stats["data_source"] = "savant_statcast"
        else:
            stats["data_source"] = "matched"

    return stats


# ============================================================================
# WEATHER API — Open-Meteo (free, no key required)
# ============================================================================
@st.cache_data(ttl=3600)
def fetch_weather(lat: float, lon: float, game_time_utc: str, is_dome: bool) -> Dict:
    """Fetch game-time weather from Open-Meteo. Returns neutral defaults for domes."""
    NEUTRAL = {
        "wind_speed": 0, "wind_direction": 0, "wind_dir_label": "DOME",
        "temperature": 72, "humidity": 50, "is_dome": True,
        "wind_effect": "neutral", "temp_effect": "neutral",
    }
    if is_dome:
        return NEUTRAL

    try:
        game_hour = 19  # default 7pm
        if game_time_utc:
            try:
                from dateutil import parser as dtparser
                game_dt = dtparser.parse(game_time_utc)
                game_hour = game_dt.hour
            except Exception:
                pass

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m,windspeed_10m,winddirection_10m,relativehumidity_2m",
            "temperature_unit": "fahrenheit",
            "windspeed_unit": "mph",
            "timezone": "America/New_York",
            "forecast_days": 2,
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        target_idx = 0
        for i, t in enumerate(times):
            try:
                t_dt = datetime.fromisoformat(t)
                if t_dt.hour == game_hour:
                    target_idx = i
                    break
            except Exception:
                pass

        def safe_idx(lst, idx, default):
            try: return lst[idx]
            except Exception: return default

        wind_speed = safe_idx(hourly.get("windspeed_10m", []), target_idx, 5)
        wind_dir   = safe_idx(hourly.get("winddirection_10m", []), target_idx, 180)
        temperature= safe_idx(hourly.get("temperature_2m", []), target_idx, 70)
        humidity   = safe_idx(hourly.get("relativehumidity_2m", []), target_idx, 50)

        wind_dir_label, wind_effect = classify_wind(float(wind_dir), float(wind_speed))
        temp_effect = "suppress" if temperature < 50 else "boost" if temperature > 83 else "neutral"

        return {
            "wind_speed": round(float(wind_speed), 1),
            "wind_direction": float(wind_dir),
            "wind_dir_label": wind_dir_label,
            "temperature": round(float(temperature), 1),
            "humidity": humidity,
            "is_dome": False,
            "wind_effect": wind_effect,
            "temp_effect": temp_effect,
        }
    except Exception as e:
        return {
            "wind_speed": 5, "wind_direction": 180, "wind_dir_label": "N/A",
            "temperature": 70, "humidity": 50, "is_dome": False,
            "wind_effect": "neutral", "temp_effect": "neutral",
        }

def classify_wind(direction: float, speed: float) -> Tuple[str, str]:
    """Classify wind direction and HR/TB effect. Direction = meteorological degrees."""
    if speed < 8:
        return "Calm", "neutral"
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    label = dirs[int((direction + 22.5) / 45) % 8]
    # SW/S/W blowing OUT toward outfield = HR boost
    if 157.5 <= direction <= 292.5:
        effect = "strong_out" if speed >= 12 else "out"
    elif direction <= 67.5 or direction >= 337.5:
        effect = "in" if speed >= 10 else "neutral"
    else:
        effect = "neutral"
    return label, effect

# ============================================================================
# ODDS API — The Odds API (free tier: 500 calls/month)
# ============================================================================
@st.cache_data(ttl=1800)
def fetch_odds(date_str: str) -> Dict:
    """
    Fetch team implied run totals from The Odds API.
    Returns dict keyed by team abbreviation -> implied runs.
    Requires ODDS_API_KEY in Streamlit secrets.
    Free tier: 500 calls/month at the-odds-api.com
    """
    try:
        api_key = st.secrets.get("odds_api", {}).get("api_key", "")
        if not api_key or api_key.strip() == "":
            return {}

        url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": "totals,h2h",
            "oddsFormat": "american",
        }
        r = requests.get(url, params=params, timeout=12)

        # Log remaining quota from headers
        remaining = r.headers.get("x-requests-remaining", "?")
        used = r.headers.get("x-requests-used", "?")

        if r.status_code == 401:
            st.warning("⚠️ Odds API: Invalid key. Check secrets.toml.")
            return {}
        if r.status_code == 422:
            st.warning("⚠️ Odds API: No lines available yet for today.")
            return {}
        r.raise_for_status()

        games = r.json()
        implied = {}

        for game in games:
            home_name = game.get("home_team", "")
            away_name = game.get("away_team", "")
            game_total = None

            # Find game total from first bookmaker
            for bm in game.get("bookmakers", []):
                for mkt in bm.get("markets", []):
                    if mkt.get("key") == "totals":
                        for outcome in mkt.get("outcomes", []):
                            if outcome.get("name") == "Over":
                                game_total = float(outcome.get("point", 9.0))
                        break
                if game_total:
                    break

            if not game_total:
                game_total = 9.0  # MLB average

            # Home team gets slight advantage (~52/48 split)
            home_implied = round(game_total * 0.52, 2)
            away_implied = round(game_total * 0.48, 2)

            # Map full team names to abbreviations
            for full_name, abbr in TEAM_ABB_MAP.items():
                if full_name.lower() in home_name.lower():
                    implied[abbr] = home_implied
                if full_name.lower() in away_name.lower():
                    implied[abbr] = away_implied

        return implied

    except Exception as e:
        return {}


@st.cache_data(ttl=1800)
def fetch_prop_odds(date_str: str) -> Dict:
    """
    Fetch player prop odds from The Odds API (player_prop_total_bases market).
    Returns dict: {player_norm_name: {"line": 1.5, "over_price": -115, "implied": 0.535}}
    Uses ~1 API call per event — conserve quota by only fetching when team odds loaded.
    Free tier: 500 calls/month.
    """
    try:
        api_key = st.secrets.get("odds_api", {}).get("api_key", "")
        if not api_key or api_key.strip() == "":
            return {}

        # First get today's event IDs
        events_url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events"
        r = requests.get(events_url, params={"apiKey": api_key}, timeout=10)
        if r.status_code != 200:
            return {}

        events = r.json()
        if not events:
            return {}

        prop_lines = {}
        # Only fetch first 3 events to conserve quota (best games first)
        for event in events[:3]:
            event_id = event.get("id")
            if not event_id:
                continue
            try:
                prop_url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds"
                pr = requests.get(prop_url, params={
                    "apiKey": api_key,
                    "regions": "us",
                    "markets": "batter_total_bases",
                    "oddsFormat": "american",
                }, timeout=10)
                if pr.status_code != 200:
                    continue
                data = pr.json()
                for bm in data.get("bookmakers", [])[:1]:  # first bookmaker only
                    for mkt in bm.get("markets", []):
                        if "total_bases" in mkt.get("key", ""):
                            for outcome in mkt.get("outcomes", []):
                                pname = _norm(outcome.get("description", ""))
                                price = outcome.get("price", -115)
                                point = outcome.get("point", 1.5)
                                if outcome.get("name", "").lower() == "over":
                                    # Convert American odds to implied probability
                                    if price < 0:
                                        implied_p = abs(price) / (abs(price) + 100)
                                    else:
                                        implied_p = 100 / (price + 100)
                                    prop_lines[pname] = {
                                        "line": point,
                                        "over_price": price,
                                        "market_implied": round(implied_p, 3),
                                    }
            except Exception:
                continue

        return prop_lines

    except Exception:
        return {}


def compute_market_edge(model_prob: float, implied_total: float, team: str,
                        prop_implied: float = None) -> Tuple[float, str]:
    """
    Calculate edge vs market.
    Uses prop-specific odds when available (much more accurate).
    Falls back to team-total estimate when prop odds not available.
    """
    if prop_implied and 0.3 < prop_implied < 0.85:
        # Prop-specific market implied probability — most accurate
        market_implied = prop_implied
        source = "prop"
    elif implied_total and implied_total > 0:
        # Estimate from team implied total
        if implied_total >= 5.5:
            market_implied = 0.56
        elif implied_total >= 4.5:
            market_implied = 0.53
        else:
            market_implied = 0.50
        source = "est"
    else:
        # No lines available — use historical base rate only
        market_implied = 0.515  # historical MLB O1.5 hit rate
        source = "base"

    edge = model_prob - market_implied

    if edge >= 0.10:
        label = f"🔥 +{edge*100:.0f}% EDGE"
    elif edge >= 0.05:
        label = f"✅ +{edge*100:.0f}% edge"
    elif edge >= 0:
        label = f"~{edge*100:.0f}% (thin)"
    else:
        label = f"❌ {edge*100:.0f}%"

    if source == "prop":
        label += " (live line)"

    return edge, label
    """
    Compute weather sub-score 0-100.
    Wind out = boost, Wind in = suppress, Dome = neutral baseline.
    """
    if weather.get("is_dome"):
        return 50.0, "🏟️ Dome"
    
    score = 50.0  # neutral baseline
    notes = []
    
    wind_effect = weather.get("wind_effect", "neutral")
    wind_speed = weather.get("wind_speed", 0)
    temp = weather.get("temperature", 70)
    wind_label = weather.get("wind_dir_label", "")
    
    # Wind adjustment
    if wind_effect == "strong_out":
        score += 25
        notes.append(f"💨 {wind_speed}mph Out (+25)")
    elif wind_effect == "out":
        score += 15
        notes.append(f"💨 {wind_speed}mph Out (+15)")
    elif wind_effect == "in":
        score -= 20
        notes.append(f"💨 {wind_speed}mph In (-20)")
    else:
        notes.append(f"💨 {wind_speed}mph {wind_label}")
    
    # Temperature adjustment
    if temp < 50:
        adj = max(-15, -8 * (50 - temp) / 10)
        score += adj
        notes.append(f"🌡️ {temp}°F Cold ({adj:.0f})")
    elif temp > 83:
        adj = min(10, 5 * (temp - 83) / 10)
        score += adj
        notes.append(f"🌡️ {temp}°F Hot (+{adj:.0f})")
    else:
        notes.append(f"🌡️ {temp}°F")
    
    return max(0, min(100, score)), " | ".join(notes)


def compute_park_score(team: str, is_home: bool) -> Tuple[float, str]:
    """
    Compute park factor sub-score 0-100.
    Uses composite TB factor weighted by hit types.
    """
    if not is_home:
        # Away team plays at home team's park
        # This will be called with the home team's park
        pass
    
    tb_factor = PARK_TB_FACTORS.get(team, 1.00)
    hr_factor = PARK_HR_FACTORS.get(team, 1.00)
    
    # Composite: weight HR more heavily (HR=4 TB)
    composite = (tb_factor * 0.4 + hr_factor * 0.6)
    
    # Special case: Coors
    if team == "COL":
        composite = 1.30
    
    # Normalize to 0-100 (0.80=0, 1.00=50, 1.30=100)
    score = (composite - 0.80) / (1.30 - 0.80) * 100
    score = max(0, min(100, score))
    
    park_name = STADIUM_COORDS.get(team, ("", "", "Unknown", False))[2]
    flag = "🏟️" if STADIUM_COORDS.get(team, (0, 0, "", False))[3] else ""
    
    return score, f"{flag}{park_name} ({tb_factor:.2f}x TB | {hr_factor:.2f}x HR)"


def compute_platoon_score(batter_hand: str, pitcher_hand: str) -> Tuple[float, str]:
    """
    Compute platoon matchup sub-score 0-100.
    Uses research-backed SLG adjustment values.
    Handles L/R/B/S for batter (S = switch hitter from MLB API).
    """
    # Normalize switch hitter codes (MLB API uses "S", some sources use "B")
    bh = batter_hand.upper() if batter_hand else "R"
    ph = pitcher_hand.upper() if pitcher_hand else "R"

    if bh in ("B", "S"):  # Switch hitter — always bats opposite of pitcher
        if ph == "R":
            # Switch hitter bats LEFT vs RHP = best platoon advantage
            return 75.0, "Switch hitter vs RHP (bats L, +56 SLG)"
        elif ph == "L":
            # Switch hitter bats RIGHT vs LHP = good platoon advantage
            return 65.0, "Switch hitter vs LHP (bats R, +33 SLG)"
        else:
            return 60.0, "Switch hitter (platoon adv)"

    if bh == "L" and ph == "R":
        return 75.0, "LHB vs RHP (+56 SLG)"
    elif bh == "R" and ph == "L":
        return 65.0, "RHB vs LHP (+33 SLG)"
    elif bh == "L" and ph == "L":
        return 30.0, "LHB vs LHP (-35 wOBA)"
    else:
        return 50.0, "RHB vs RHP (neutral)"


def compute_lineup_score(lineup_slot: int) -> Tuple[float, str]:
    """
    Compute lineup position sub-score based on expected PA.
    Slots 1-4 get most PAs; slot 9 gets least.
    """
    # Expected PA per game by lineup slot (research-based)
    pa_by_slot = {1: 4.8, 2: 4.7, 3: 4.6, 4: 4.5, 5: 4.3, 
                  6: 4.2, 7: 4.1, 8: 3.9, 9: 3.8}
    
    expected_pa = pa_by_slot.get(lineup_slot, 4.2)
    
    # Normalize: min 3.8=20, max 4.8=100
    score = (expected_pa - 3.8) / (4.8 - 3.8) * 80 + 20
    
    slot_labels = {1: "Leadoff", 2: "2-Hole", 3: "3-Hole", 4: "Cleanup",
                   5: "5th", 6: "6th", 7: "7th", 8: "8th", 9: "9th"}
    bonus = " ⭐" if lineup_slot <= 4 else ""
    
    return score, f"#{lineup_slot} {slot_labels.get(lineup_slot, str(lineup_slot))}{bonus} ({expected_pa:.1f} PA/g)"


def compute_pitch_matchup_score(batter_stats: Dict, pitcher_stats: Dict) -> Tuple[float, str]:
    """
    V1.5: Pitch matchup score 0-100.
    Cross-references:
      - Pitcher's arsenal mix (what % of each pitch type they throw)
      - Batter's run value vs each pitch type
    Produces a 0-100 score where:
      50 = neutral (batter has avg performance vs pitcher's pitch mix)
      70+ = strong edge (batter crushes pitcher's primary pitches)
      30- = disadvantage (batter struggles vs pitcher's primary pitches)

    Run value scale: -2.0 (terrible) to +2.0 (elite) per 100 pitches
    Weighted by pitcher's usage % → usage-adjusted RV composite
    """
    PITCH_TYPES = ("FF", "SI", "SL", "CH", "CU", "FC")
    # League avg run values vs each pitch type (approx 2025 baseline)
    # Positive = batter-favorable, Negative = pitcher-favorable
    LEAGUE_AVG_RV = {"FF": 0.0, "SI": -0.1, "SL": -0.2, "CH": -0.1, "CU": -0.2, "FC": -0.1}

    try:
        # Pitcher's arsenal mix (sum should ~= 1.0 after normalization)
        total_pct = sum(
            float(pitcher_stats.get(f"pct_{pt}", 0) or 0)
            for pt in PITCH_TYPES
        )
        if total_pct < 0.01:
            return 50.0, "Pitch mix: no data"

        # Batter's run value vs each pitch type — usage-weighted composite
        weighted_rv = 0.0
        coverage = 0.0  # how much of pitcher's arsenal we have batter data for
        pitch_details = []

        for pt in PITCH_TYPES:
            pct = float(pitcher_stats.get(f"pct_{pt}", 0) or 0)
            if pct < 0.01:
                continue
            pct_norm = pct / total_pct  # normalize to sum=1
            rv = float(batter_stats.get(f"rv_vs_{pt}", LEAGUE_AVG_RV.get(pt, 0.0)))
            rv_relative = rv - LEAGUE_AVG_RV.get(pt, 0.0)  # relative to league avg
            weighted_rv += pct_norm * rv_relative
            coverage += pct_norm
            if pct_norm > 0.15:  # only label pitches ≥15% usage
                sign = "+" if rv > 0 else ""
                pitch_details.append(f"{pt}({pct_norm*100:.0f}%): {sign}{rv:.2f}")

        # weighted_rv range: roughly -1.5 to +1.5 in extreme cases
        # Map to 0-100: -1.5 → 10, 0 → 50, +1.5 → 90
        score = 50.0 + (weighted_rv / 1.5) * 40.0
        score = max(10.0, min(90.0, score))

        # Penalize low data coverage (batter without pitch-type splits)
        if coverage < 0.5:
            # Blend toward 50 proportionally
            score = score * (coverage / 0.5) + 50.0 * (1 - coverage / 0.5)

        label = "Pitch mix: " + " | ".join(pitch_details) if pitch_details else "Pitch mix: avg splits"
        return round(score, 1), label

    except Exception:
        return 50.0, "Pitch mix: error"


def compute_batter_score(statcast: Dict, fg_stats: Dict = None) -> Tuple[float, str, Dict]:
    """
    Compute batter profile sub-score 0-100.
    League avg batter = ~50. Elite (Judge/Soto tier) = 85+.
    All numpy types converted to float to prevent silent math failures.
    """
    details = {}

    # Force all to native Python float — fixes np.int64/np.float64 silent failures
    def f(key, default):
        try: return float(statcast.get(key, default))
        except: return float(default)

    xslg        = f("slg_proxy",        0.398)
    barrel_rate = f("barrel_rate",      0.070)
    hard_hit    = f("hard_hit_rate",    0.370)
    k_rate      = f("k_rate",           0.228)
    iso         = f("iso_proxy",        0.165)
    wrc_plus    = f("wrc_plus",         100.0)
    ev50_raw    = f("ev50",             0.0)     # 0 = never populated
    bat_speed_raw = f("bat_speed",      0.0)
    blast_raw   = f("blast_rate",       0.0)

    # V1.6+: When Savant bat-tracking signals are unavailable (default=0),
    # derive them from xSLG + ISO rather than league-average constants.
    # This prevents 30% of the bat score (ev50+bat_speed+blast) from
    # collapsing to identical league-avg values for every player.
    # Derivation research:
    #   EV50 ~ xSLG × 18 pts range + ISO × 6 pts (r≈0.82 with actual Savant)
    #   bat_speed ~ xSLG × 9 pts range (higher contact quality = faster swing)
    #   blast_rate ~ barrel_rate × 0.45 + ISO × 0.25 (well-squared contact)
    if ev50_raw < 50:  # not populated — derive from xSLG+ISO
        ev50 = 86 + (xslg - 0.200) / 0.450 * 18 + (iso - 0.080) / 0.240 * 6
        ev50 = max(85.0, min(104.0, ev50))
    else:
        ev50 = ev50_raw
    if bat_speed_raw < 30:  # not populated
        bat_speed = 68.0 + (xslg - 0.200) / 0.450 * 9.0
        bat_speed = max(66.0, min(77.0, bat_speed))
    else:
        bat_speed = bat_speed_raw
    if blast_raw < 0.01:  # not populated
        blast_rate = 0.14 + barrel_rate / 0.20 * 0.09 + (iso - 0.080) / 0.240 * 0.06
        blast_rate = max(0.12, min(0.32, blast_rate))
    else:
        blast_rate = blast_raw

    details["xSLG"]     = round(xslg, 3)
    details["Barrel%"]  = f"{barrel_rate*100:.1f}%"
    details["HardHit%"] = f"{hard_hit*100:.1f}%"
    details["K%"]       = f"{k_rate*100:.1f}%"
    details["ISO"]      = round(iso, 3)
    details["wRC+"]     = int(wrc_plus)
    # V1.8: EV50/bat_speed/blast only shown when real Savant data present
    if ev50_raw >= 50:
        details["EV50"]   = round(ev50_raw, 1)
    if bat_speed_raw >= 30:
        details["BatSpd"] = round(bat_speed_raw, 1)
    if blast_raw >= 0.01:
        details["Blast%"] = f"{blast_raw*100:.1f}%"

    # ── Sub-scores 0-100, Z-score normalized so LEAGUE AVG BATTER = 50 ──
    # V1.8: Switched from range-based to Z-score style normalization.
    # Old approach (e.g. barrel 0%→0, 7%→35, 20%→100) produced avg batter = ~38,
    # compressing everyone upward and making elite vs avg indistinguishable.
    # Z-score: avg = 50, ±1 MLB std dev = ±25 pts. Capped 0-100.
    # League avg benchmarks (2024 MLB): xSLG=.398, wRC+=100, barrel=7%, HH=37%, K=22.8%, ISO=.165

    # xSLG: avg=.398, sd≈.080 → Judge(.708)=~148→100, avg→50, weak(.250)→~8
    xslg_score = max(0, min(100, 50 + (xslg - 0.398) / 0.080 * 25))

    # wRC+: avg=100, sd≈35 → 200=~121→100, avg→50, 65→~-25→0
    wrc_score = max(0, min(100, 50 + (wrc_plus - 100) / 35.0 * 25))

    # Barrel%: avg=7%, sd≈4% → 20%=~131→100, avg→50, 2%=~19→19
    barrel_score = max(0, min(100, 50 + (barrel_rate - 0.070) / 0.040 * 25))

    # Hard hit%: avg=37%, sd≈5.5% → 56%=~136→100, avg→50, 25%=~5→5
    hard_hit_score = max(0, min(100, 50 + (hard_hit - 0.370) / 0.055 * 25))

    # K rate INVERSE: avg=22.8%, sd≈6% → high K = low score
    k_score = max(0, min(100, 50 - (k_rate - 0.228) / 0.060 * 25))

    # ISO: avg=.165, sd≈.065 → .320=~110→100, avg→50, .050→~6
    iso_score = max(0, min(100, 50 + (iso - 0.165) / 0.065 * 25))

    # V1.8: When real Savant bat-tracking data available, add EV50/bat_speed/blast.
    # When unavailable, do NOT fake them — derived proxies are correlated with xSLG.
    has_bat_tracking = (ev50_raw >= 50 and bat_speed_raw >= 30 and blast_raw >= 0.01)

    if has_bat_tracking:
        # EV50: avg≈95mph, sd≈3mph
        ev50_score = max(0, min(100, 50 + (ev50_raw - 95.0) / 3.0 * 25))
        # Bat speed: avg≈71mph, sd≈3mph
        bat_speed_score = max(0, min(100, 50 + (bat_speed_raw - 71.0) / 3.0 * 25))
        # Blast rate: avg≈21%, sd≈5%
        blast_score = max(0, min(100, 50 + (blast_raw - 0.21) / 0.050 * 25))
        composite = (
            barrel_score    * 0.20 +
            xslg_score      * 0.16 +
            hard_hit_score  * 0.14 +
            ev50_score      * 0.14 +
            blast_score     * 0.12 +
            bat_speed_score * 0.10 +
            wrc_score       * 0.08 +
            iso_score       * 0.04 +
            k_score         * 0.02
        )  # sum = 1.00
    else:
        # No bat-tracking: 6 real signals, sum = 1.00
        composite = (
            barrel_score   * 0.26 +
            xslg_score     * 0.24 +
            hard_hit_score * 0.20 +
            wrc_score      * 0.14 +
            iso_score      * 0.10 +
            k_score        * 0.06
        )

    return max(0, min(100, composite)), "Contact quality", details


# ============================================================================
# BULLPEN QUALITY — Per-team vulnerability from loaded pitching_df
# V1.3: replaces fixed league-average (42.0) with real per-team scores
# ============================================================================
def compute_team_bullpen_scores(pitching_df: pd.DataFrame) -> Dict[str, float]:
    """
    Build per-team bullpen vulnerability scores (0-100) from the already-loaded
    FanGraphs pitching DataFrame. Runs ONCE before the scoring loop.

    Filters to relievers (GS == 0 or GS/G < 30%), groups by team, computes
    a weighted K%/FIP/WHIP/HH% composite using the same formula as the SP
    sub-score in compute_pitcher_score().

    Returns dict keyed by UPPERCASE team abbreviation:
        {"NYY": 38.2, "ATH": 61.4, "COL": 58.0, ...}
    Falls back to league average (42.0) for any missing team at scoring time.
    """
    LEAGUE_AVG_BP_VULN = 42.0

    if pitching_df is None or pitching_df.empty:
        return {}

    df = pitching_df.copy()

    # ── Identify reliever rows ─────────────────────────────────────────────
    has_gs = "GS" in df.columns
    has_g  = "G"  in df.columns

    if has_gs and has_g:
        df["_GS"] = pd.to_numeric(df["GS"], errors="coerce").fillna(0)
        df["_G"]  = pd.to_numeric(df["G"],  errors="coerce").fillna(0)
        df["_IP"] = pd.to_numeric(df.get("IP", pd.Series(dtype=float)), errors="coerce").fillna(0)
        relievers = df[
            (df["_GS"] == 0) |
            ((df["_G"] > 0) & (df["_GS"] / df["_G"].replace(0, 1) < 0.30))
        ].copy()
        # Early-season fallback: if < 5% of pitchers are classified as relievers,
        # most likely everyone has GS=0 (season just started).
        # Use IP threshold instead: relievers = IP < 3.0 per game avg
        if not relievers.empty and len(relievers) / max(1, len(df)) > 0.70:
            # >70% labeled reliever = early season artifact; use IP/G ratio instead
            df["_ipg"] = df["_IP"] / df["_G"].replace(0, 1)
            relievers = df[df["_ipg"] < 2.0].copy()  # < 2 IP/game = likely reliever
    elif has_gs:
        df["_GS"] = pd.to_numeric(df["GS"], errors="coerce").fillna(0)
        relievers = df[df["_GS"] == 0].copy()
    else:
        # No GS column — use IP/G if available, else use all pitchers as proxy
        if "IP" in df.columns and "G" in df.columns:
            df["_IP"] = pd.to_numeric(df["IP"], errors="coerce").fillna(0)
            df["_G"]  = pd.to_numeric(df["G"],  errors="coerce").fillna(0)
            df["_ipg"] = df["_IP"] / df["_G"].replace(0, 1)
            relievers = df[df["_ipg"] < 2.0].copy()
        else:
            relievers = df.copy()  # Use all pitchers as proxy

    if relievers.empty:
        # Last resort: use all pitchers
        relievers = df.copy()
    if relievers.empty:
        return {}

    # ── Find team column ───────────────────────────────────────────────────
    team_col = None
    for candidate in ["Team", "team", "Tm", "tm", "TEAM"]:
        if candidate in relievers.columns:
            team_col = candidate
            break
    if team_col is None:
        return {}

    # ── Normalize stat columns to decimal rates ────────────────────────────
    def to_rate(series: pd.Series, thresh: float = 1.0) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        if s.dropna().median() > thresh:
            s = s / 100.0
        return s

    # Pull the Series safely even if column is missing
    k_series    = to_rate(relievers[team_col].map(lambda _: None).combine_first(
                      relievers.get("K%", pd.Series(dtype=float, index=relievers.index))), thresh=1.0) \
                  if "K%" in relievers.columns else pd.Series(dtype=float, index=relievers.index)
    k_series    = to_rate(relievers["K%"],    thresh=1.0) if "K%"    in relievers.columns else pd.Series(dtype=float, index=relievers.index)
    fip_series  = pd.to_numeric(relievers["FIP"],   errors="coerce") if "FIP"   in relievers.columns else \
                  pd.to_numeric(relievers["xFIP"],  errors="coerce") if "xFIP"  in relievers.columns else \
                  pd.Series(dtype=float, index=relievers.index)
    whip_series = pd.to_numeric(relievers["WHIP"],  errors="coerce") if "WHIP"  in relievers.columns else \
                  pd.Series(dtype=float, index=relievers.index)
    hh_series   = to_rate(relievers["Hard%"], thresh=1.0) if "Hard%" in relievers.columns else \
                  pd.Series(dtype=float, index=relievers.index)

    relievers = relievers.copy()
    relievers["_k_rate"] = k_series.values
    relievers["_fip"]    = fip_series.values
    relievers["_whip"]   = whip_series.values
    relievers["_hh"]     = hh_series.values

    # ── Group by team, use median (robust to outlier arms) ─────────────────
    try:
        group = relievers.groupby(team_col).agg(
            k_rate=("_k_rate", "median"),
            fip   =("_fip",    "median"),
            whip  =("_whip",   "median"),
            hh    =("_hh",     "median"),
        ).reset_index()
    except Exception:
        return {}

    # ── Score each team bullpen ────────────────────────────────────────────
    team_scores: Dict[str, float] = {}
    for _, row in group.iterrows():
        team = str(row[team_col]).strip().upper()
        if not team or team in ("", "---", "TOT", "2TM", "3TM"):
            continue

        k_rate = float(row["k_rate"]) if pd.notna(row["k_rate"]) else 0.228
        fip    = float(row["fip"])    if pd.notna(row["fip"])    else 4.50
        whip   = float(row["whip"])   if pd.notna(row["whip"])   else 1.35
        hh     = float(row["hh"])     if pd.notna(row["hh"])     else 0.340

        # Clamp to valid ranges
        k_rate = max(0.08, min(0.42, k_rate))
        fip    = max(2.0,  min(8.0,  fip))
        whip   = max(0.80, min(2.20, whip))
        hh     = max(0.20, min(0.55, hh))

        # Vulnerability sub-scores (high = hittable bullpen = good for batters)
        k_vuln    = max(0.0, min(100.0, (0.35 - k_rate) / (0.35 - 0.10) * 100))
        era_vuln  = max(0.0, min(100.0, (fip  - 2.0)    / (7.0  - 2.0)  * 100))
        whip_vuln = max(0.0, min(100.0, (whip - 0.90)   / (1.80 - 0.90) * 100))
        hh_vuln   = max(0.0, min(100.0, (hh   - 0.28)   / (0.50 - 0.28) * 100))

        # Weighted composite (same formula as SP, no barrel% for relievers)
        bp_score = (
            k_vuln    * 0.38 +
            hh_vuln   * 0.22 +
            era_vuln  * 0.22 +   # FIP double-weighted since no barrel%
            whip_vuln * 0.18
        )

        team_scores[team] = round(max(0.0, min(100.0, bp_score)), 1)

    return team_scores


def compute_pitcher_score(statcast: Dict, fg_stats: Dict = None,
                          bullpen_vuln: float = 42.0) -> Tuple[float, str]:
    """
    Pitcher VULNERABILITY score 0-100.
    HIGH score = pitcher is hittable = good for batter TB.
    LOW score = elite pitcher = suppresses TB.
    Webb/Fried type = ~20-30. League avg pitcher = ~50. Mop-up arm = ~75+.

    V1.3: Blends SP stats (60%) with PER-TEAM bullpen vulnerability (40%).
    bullpen_vuln is pre-computed by compute_team_bullpen_scores() and passed
    in from the scoring loop. Defaults to 42.0 (league avg) when unavailable.
    Swing between best/worst bullpen: ~8-10 score points.
    """
    def f(key, default):
        try: return float(statcast.get(key, default))
        except: return float(default)

    k_rate   = f("k_rate_allowed",   0.220)
    hard_hit = f("hard_hit_allowed", 0.370)
    barrel   = f("barrel_allowed",   0.070)
    era      = f("era",              4.20)
    fip      = f("fip",              4.20)
    whip     = f("whip",             1.30)

    # ── Pitcher VULNERABILITY sub-scores — Z-score normalized, avg pitcher = 50 ──
    # V1.8: Z-score style so league avg pitcher = 50 (hittable), ace = 20, mop-up = 80.
    # High score = pitcher is hittable = good for batter TB prop.
    # League avg benchmarks (2024): K%=22%, HH%=37%, barrel%=7%, FIP=4.2, WHIP=1.30
    # Std deviations: K%≈5pp, HH%≈5.5pp, barrel%≈3pp, FIP≈0.80, WHIP≈0.20

    # K% INVERSE: high K = low vulnerability. avg=22%, sd=5%
    k_vuln = max(0, min(100, 50 - (k_rate - 0.220) / 0.050 * 25))

    # HH% allowed: high HH = high vulnerability. avg=37%, sd=5.5%
    hh_vuln = max(0, min(100, 50 + (hard_hit - 0.370) / 0.055 * 25))

    # Barrel% allowed: high barrel = high vulnerability. avg=7%, sd=3%
    barrel_vuln = max(0, min(100, 50 + (barrel - 0.070) / 0.030 * 25))

    # FIP: high FIP = high vulnerability. avg=4.2, sd=0.80
    era_use = fip if fip > 0 else era
    era_vuln = max(0, min(100, 50 + (era_use - 4.20) / 0.80 * 25))

    # WHIP: high WHIP = high vulnerability. avg=1.30, sd=0.20
    whip_vuln = max(0, min(100, 50 + (whip - 1.30) / 0.20 * 25))

    # SP composite — V1.8 weights
    sp_score = (
        k_vuln      * 0.40 +  # K% — most stable pitcher predictor
        hh_vuln     * 0.28 +  # HH% — 2nd most predictive
        era_vuln    * 0.18 +  # FIP — reliable baseline
        barrel_vuln * 0.09 +  # barrel% — noisy but real
        whip_vuln   * 0.05    # WHIP — least sticky
    )

    # V1.3: Use per-team bullpen vuln (was fixed at 42.0 league avg for all teams)
    # Batters see ~40% of PAs against bullpen (3-4 IP out of 9)
    blended = sp_score * 0.60 + float(bullpen_vuln) * 0.40

    label = f"K%: {k_rate*100:.0f}% | WHIP: {whip:.2f} | FIP: {era_use:.2f} | BP vuln: {bullpen_vuln:.0f}"
    return max(0, min(100, blended)), label


def compute_vegas_score(implied_total: float) -> Tuple[float, str]:
    """
    Vegas signal sub-score 0-100.
    Returns 0 with flag when no lines available (don't fake neutrality).
    4.5 = neutral, 5.5+ = favorable, 3.5- = bad environment.
    """
    if not implied_total or implied_total <= 0:
        return 0.0, "No lines ⚠️"

    # 3.0=0, 4.5=50, 6.5=100
    score = (implied_total - 3.0) / (6.5 - 3.0) * 100
    score = max(0, min(100, score))

    if implied_total >= 5.5:
        flag = " 🔥"
    elif implied_total >= 4.5:
        flag = " ✅"
    elif implied_total < 3.5:
        flag = " ❄️"
    else:
        flag = ""

    return score, f"{implied_total:.1f} implied runs{flag}"


def compute_weather_score(weather: Dict) -> Tuple[float, str]:
    """
    Compute weather sub-score 0-100.
    Wind out = boost, Wind in = suppress, Dome = neutral baseline.
    """
    if not weather or weather.get("is_dome"):
        return 50.0, "🏟️ Dome"

    score = 50.0
    notes = []

    wind_effect = weather.get("wind_effect", "neutral")
    wind_speed  = weather.get("wind_speed", 0)
    temp        = weather.get("temperature", 70)
    wind_label  = weather.get("wind_dir_label", "")

    if wind_effect == "strong_out":
        score += 25
        notes.append(f"💨 {wind_speed}mph Out (+25)")
    elif wind_effect == "out":
        score += 15
        notes.append(f"💨 {wind_speed}mph Out (+15)")
    elif wind_effect == "in":
        score -= 20
        notes.append(f"💨 {wind_speed}mph In (-20)")
    else:
        notes.append(f"💨 {wind_speed}mph {wind_label}" if wind_speed else "💨 Calm")

    if temp < 50:
        adj = max(-15, -8 * (50 - temp) / 10)
        score += adj
        notes.append(f"🌡️ {temp:.0f}°F Cold ({adj:.0f})")
    elif temp > 83:
        adj = min(10, 5 * (temp - 83) / 10)
        score += adj
        notes.append(f"🌡️ {temp:.0f}°F Hot (+{adj:.0f})")
    else:
        notes.append(f"🌡️ {temp:.0f}°F")

    return max(0, min(100, score)), " | ".join(notes)


def compute_tto_bonus(lineup_slot: int, sp_ip_estimate: float = 6.0) -> Tuple[float, str]:
    """
    Times Through Order (TTO) bonus.
    Research-backed: 2nd TTO +8 wOBA, 3rd TTO +17-20 wOBA vs 1st TTO.
    Estimate TTO based on lineup slot and typical SP innings.
    
    Typical SP: 6 IP = 18 batters faced = ~2 times through
    Slot 1-3: likely 3 TTO (top order faces SP most)
    Slot 4-6: likely 2-3 TTO
    Slot 7-9: likely 2 TTO
    """
    # Estimate TTO based on lineup position and typical SP workload
    if lineup_slot <= 3:
        # Top of order: 3 PA = 3rd TTO territory
        tto_boost = 0.60  # Normalized boost (3rd TTO = +20 wOBA)
        label = "3rd TTO boost (+20 wOBA)"
    elif lineup_slot <= 6:
        # Middle: 2-3 TTO
        tto_boost = 0.45
        label = "2-3rd TTO (~+12 wOBA)"
    else:
        # Bottom: mostly 2nd TTO
        tto_boost = 0.25
        label = "2nd TTO (+8 wOBA)"
    
    # Convert wOBA boost to 0-100 score contribution
    # +20 wOBA ≈ +15 points on our 0-100 scale for top of order
    score = tto_boost * 25  # 0-15 point bonus
    return score, label





# ============================================================================
# RECENT FORM — last N game log via MLB Stats API
# V1.7: Hot/cold streak signal. Cached per-player with short TTL.
# ============================================================================
@st.cache_data(ttl=3600)
def fetch_batter_recent_form(player_id: str, n_games: int = 7) -> Dict:
    """
    Pull last N game logs for a batter from MLB Stats API gameLog endpoint.
    Returns dict with recent TB/game, hit rate, and momentum vs season avg.
    Free endpoint, no key required, accessible on Streamlit Cloud.
    """
    _empty = {"tb_per_game": None, "avg_recent": None, "games": 0,
              "hr_last_7": 0, "h_last_7": 0, "ab_last_7": 0}
    if not player_id or str(player_id) in ("", "0", "nan"):
        return _empty
    try:
        url = (f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
               f"?stats=gameLog&group=hitting&gameType=R&limit={n_games + 5}")
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return _empty
        splits = r.json().get("stats", [{}])[0].get("splits", [])
        if not splits:
            return _empty
        # Take most recent N games
        recent = splits[:n_games]
        total_tb = total_ab = total_h = total_hr = 0
        for s in recent:
            st_ = s.get("stat", {})
            total_tb += int(st_.get("totalBases", 0) or 0)
            total_ab += int(st_.get("atBats", 0) or 0)
            total_h  += int(st_.get("hits", 0) or 0)
            total_hr += int(st_.get("homeRuns", 0) or 0)
        g = len(recent)
        if g == 0 or total_ab == 0:
            return _empty
        return {
            "tb_per_game": round(total_tb / g, 2),
            "avg_recent":  round(total_h / total_ab, 3),
            "games":       g,
            "hr_last_7":   total_hr,
            "h_last_7":    total_h,
            "ab_last_7":   total_ab,
        }
    except Exception:
        return _empty


@st.cache_data(ttl=86400)
def fetch_batter_vs_pitcher(batter_id: str, pitcher_id: str) -> Dict:
    """
    Pull career stats for a specific batter vs specific pitcher.
    MLB Stats API vsPlayer endpoint — free, no key required.
    Only meaningful when career AB >= 10 (small sample guard).
    """
    _empty = {"slg": None, "avg": None, "hr": 0, "ab": 0, "h": 0, "obp": None}
    if not batter_id or not pitcher_id:
        return _empty
    if str(batter_id) in ("", "0", "nan") or str(pitcher_id) in ("", "0", "nan"):
        return _empty
    try:
        url = (f"https://statsapi.mlb.com/api/v1/people/{batter_id}/stats"
               f"?stats=vsPlayer&group=hitting&opposingPlayerId={pitcher_id}")
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return _empty
        splits = r.json().get("stats", [{}])[0].get("splits", [])
        if not splits:
            return _empty
        st_ = splits[0].get("stat", {})
        ab  = int(st_.get("atBats", 0) or 0)
        h   = int(st_.get("hits", 0) or 0)
        hr  = int(st_.get("homeRuns", 0) or 0)
        d2  = int(st_.get("doubles", 0) or 0)
        d3  = int(st_.get("triples", 0) or 0)
        tb  = int(st_.get("totalBases", 0) or 0)
        bb  = int(st_.get("baseOnBalls", 0) or 0)
        so  = int(st_.get("strikeOuts", 0) or 0)
        try:
            slg = float(st_.get("slg", 0) or 0)
            avg = float(st_.get("avg", 0) or 0)
            obp = float(st_.get("obp", 0) or 0)
        except Exception:
            slg = avg = obp = 0.0
        # Compute TB directly if API returns it; otherwise derive
        if tb == 0 and h > 0:
            tb = h + d2 + d3*2 + hr*3  # approximate (singles=1, already in h)
        xbh = d2 + d3 + hr           # extra base hits
        return {
            "slg": slg if slg > 0 else None,
            "avg": avg if avg > 0 else None,
            "obp": obp if obp > 0 else None,
            "hr": hr, "ab": ab, "h": h,
            "doubles": d2, "triples": d3, "tb": tb,
            "xbh": xbh, "bb": bb, "so": so,
        }
    except Exception:
        return _empty


def compute_streak_score(recent: Dict, season_slg: float = 0.398) -> Tuple[float, str]:
    """
    Convert recent form data into a 0-100 score.
    Compares recent TB/game to expected TB/game from season SLG.
    Expected TB/game ≈ SLG × 3.65 AB/game (MLB avg).
    
    Score 50 = on pace with season average (no momentum signal)
    Score 70+ = hot streak (recent TB >> season expectation)
    Score 30- = cold streak (recent TB << season expectation)
    
    Only activates with ≥ 3 recent games; degrades toward 50 with small samples.
    """
    if not recent or recent.get("games", 0) < 3 or recent.get("tb_per_game") is None:
        return 50.0, "Form: no data"

    tb_recent  = recent["tb_per_game"]
    g          = recent["games"]
    season_exp = season_slg * 3.65   # expected TB/game from season SLG

    if season_exp <= 0:
        return 50.0, "Form: no baseline"

    # Ratio: recent vs expected (>1.0 = hot, <1.0 = cold)
    ratio = tb_recent / season_exp
    # Map: 0.3x → 15, 0.7x → 38, 1.0x → 50, 1.3x → 63, 1.8x → 75, 2.5x → 85
    raw_score = 50 + (ratio - 1.0) * 50
    raw_score = max(15.0, min(85.0, raw_score))

    # Dampen with small samples — blend toward 50 with < 5 games
    if g < 5:
        raw_score = raw_score * (g / 5) + 50.0 * (1 - g / 5)

    h  = recent.get("h_last_7", 0)
    ab = recent.get("ab_last_7", 1)
    hr = recent.get("hr_last_7", 0)
    
    if ratio >= 1.25:
        label = f"🔥 Hot ({tb_recent:.1f} TB/g last {g}g | {h}-{ab}" + (f" {hr}HR" if hr else "") + ")"
    elif ratio <= 0.75:
        label = f"❄️ Cold ({tb_recent:.1f} TB/g last {g}g | {h}-{ab})"
    else:
        label = f"Form: {tb_recent:.1f} TB/g last {g}g"

    return round(raw_score, 1), label


def compute_bvp_score(bvp: Dict, batter_slg: float = 0.398) -> Tuple[float, str]:
    """
    Career batter-vs-pitcher significance scoring. V1.7.

    Three-signal system:
    1. SLG delta: career SLG vs this SP minus batter's season SLG baseline
    2. AVG quality: raw BA in the matchup (high BA = making consistent contact)
    3. Power flag: HR rate and XBH rate vs this specific pitcher

    Significance tiers with dynamic weight boosts:
    - 🔥 OWNS (elite: AVG .400+ or SLG .800+ with 10+ AB): max score boost,
      BvP weight in final score temporarily boosted for this player
    - 🟢 EDGE (strong: SLG >> season avg): meaningful positive signal
    - ⚠️ DOMINATED (SLG << season avg, high K%): suppress score
    - 🔴 FADE (very low SLG, pitcher wins matchup): stronger suppress

    Returns (score 0-100, label, significance_flag)
    significance_flag: 'owns', 'edge', 'neutral', 'dominated', 'fade', 'no_data'
    — stored in results dict for leaderboard flag and conditional weight boost.
    """
    ab  = bvp.get("ab", 0) if bvp else 0
    h   = bvp.get("h", 0)  if bvp else 0
    hr  = bvp.get("hr", 0) if bvp else 0
    xbh = bvp.get("xbh", 0) if bvp else 0
    d2  = bvp.get("doubles", 0) if bvp else 0
    so  = bvp.get("so", 0) if bvp else 0
    tb  = bvp.get("tb", 0) if bvp else 0

    if not bvp or ab < 10 or bvp.get("slg") is None:
        if 0 < ab < 10:
            return 50.0, f"BvP: {h}/{ab} ({ab} AB — need 10+)", "no_data"
        return 50.0, "BvP: no history", "no_data"

    career_slg = float(bvp["slg"])
    career_avg = float(bvp.get("avg") or 0)
    slg_delta  = career_slg - batter_slg
    avg_pct    = career_avg              # e.g. 0.400
    hr_rate    = hr / ab if ab > 0 else 0
    xbh_rate   = xbh / ab if ab > 0 else 0
    k_rate_bvp = so / ab if ab > 0 else 0

    # ── BASE SCORE from SLG delta ──────────────────────────────────────────
    # delta +0.200 → +20pts, 0 → 0pts, -0.200 → -20pts  (anchored at 50)
    base = 50.0 + slg_delta * 100.0
    base = max(15.0, min(90.0, base))

    # ── POWER BONUS ────────────────────────────────────────────────────────
    # HR against this pitcher: each HR adds real evidence of power advantage
    # XBH rate: doubles/triples also signal contact quality
    hr_bonus  = min(8.0, hr * 2.5)          # 1 HR=+2.5, 2 HR=+5, 3+ HR=+7.5
    xbh_bonus = min(5.0, (xbh_rate - 0.15) * 20.0) if xbh_rate > 0.15 else 0.0

    # ── CONTACT QUALITY ────────────────────────────────────────────────────
    # High AVG vs this pitcher = making consistent contact, not just fluke SLG
    avg_bonus = 0.0
    if avg_pct >= 0.400:
        avg_bonus = 6.0    # elite contact rate
    elif avg_pct >= 0.350:
        avg_bonus = 3.0
    elif avg_pct >= 0.300:
        avg_bonus = 1.0

    # ── STRIKEOUT PENALTY ──────────────────────────────────────────────────
    # High K rate vs this pitcher = pitcher has real stuff advantage
    k_penalty = 0.0
    if k_rate_bvp >= 0.40:
        k_penalty = -6.0   # struggling badly
    elif k_rate_bvp >= 0.30:
        k_penalty = -3.0

    raw_score = base + hr_bonus + xbh_bonus + avg_bonus + k_penalty
    raw_score = max(10.0, min(92.0, raw_score))

    # ── CONFIDENCE WEIGHT ──────────────────────────────────────────────────
    # Small samples: blend toward 50 with < 20 AB; full weight at 50+ AB
    if ab < 50:
        weight = min(1.0, (ab - 10) / 40.0)    # 10 AB = 0 weight, 50 AB = full
        raw_score = raw_score * weight + 50.0 * (1 - weight)

    raw_score = round(raw_score, 1)

    # ── SIGNIFICANCE TIER + LABEL ──────────────────────────────────────────
    # Tier thresholds use the RAW (pre-confidence-weighting) signal strength
    # so small samples with elite numbers still get flagged (with caveat)
    raw_signal_strength = base + hr_bonus + xbh_bonus + avg_bonus + k_penalty

    xbh_str = ""
    if d2 > 0 or bvp.get("triples", 0) > 0:
        xbh_str = f" {d2}2B" if d2 > 0 else ""
        if bvp.get("triples", 0): xbh_str += f" {bvp['triples']}3B"

    detail = f"{h}/{ab}"
    if hr: detail += f" {hr}HR"
    if d2: detail += f" {d2}2B"
    slg_str = f".{int(career_slg*1000):03d}"
    avg_str = f".{int(career_avg*1000):03d}" if career_avg > 0 else ""

    sample_note = f" ({ab} AB)" if ab < 20 else ""

    # Owns: extreme dominance — very high AVG and/or SLG, or multiple HRs
    if (avg_pct >= 0.380 and career_slg >= 0.700) or        (hr >= 2 and ab <= 15) or        (hr >= 3) or        (career_slg >= batter_slg + 0.350):
        sig = "owns"
        label = f"🔥 OWNS: {detail} SLG {slg_str}{sample_note}"

    # Edge: meaningfully above season average, OR any 2+ HR showing
    elif raw_signal_strength >= 62 or slg_delta >= 0.100 or          (hr >= 2) or (xbh >= 4 and ab <= 25):
        sig = "edge"
        label = f"🟢 BvP Edge: {detail} SLG {slg_str}{sample_note}"

    # Dominated: pitcher consistently gets this batter out AND high K rate
    elif raw_signal_strength <= 35 and k_rate_bvp >= 0.30:
        sig = "dominated"
        label = f"⚠️ Dominated: {detail} {int(k_rate_bvp*100)}%K{sample_note}"

    # Fade: below average but less extreme
    elif raw_signal_strength <= 42 or slg_delta <= -0.100:
        sig = "fade"
        label = f"🔴 BvP Fade: {detail} SLG {slg_str}{sample_note}"

    # Neutral
    else:
        sig = "neutral"
        label = f"BvP: {detail} SLG {slg_str}{sample_note}"

    return raw_score, label, sig

def compute_final_score(
    batter_score: float,
    pitcher_vuln_score: float,
    platoon_score: float,
    lineup_score: float,
    park_score: float,
    weather_score: float,
    vegas_score: float,
    tto_bonus: float = 0.0,
    pitch_matchup_score: float = 50.0,
    streak_score: float = 50.0,
    bvp_score: float = 50.0,
    bvp_weight_boost: float = 0.0,
) -> float:
    """
    Final weighted composite. V1.8 research-calibrated weights.

    Research sources:
    - FanGraphs barrel study: "Hitters supply the power — whether a pitcher
      surrenders barrels has more to do with who they face than how they pitch"
    - Pitcher List K%/HH study: K% is #1 pitcher predictor (r=-0.375 next ERA);
      HardHit/9 is #2; barrel% allowed has r²≈0.12 (barely predictive)
    - FullCountProps: platoon +56 SLG effect is large and stable
    - FiveThirtyEight Elo: 7-game streaks real but noisy — limit to 5%
    - FTA BvP research: meaningful at 15+ AB; limit to 2%

    V1.8 weight rationale:
    - Pitcher 30%: primary matchup driver — wider scale now gives real separation
    - Batter 28%: quality matters but shouldn't override bad matchup
    - Platoon 12%: +56 SLG well-documented; most stable contextual signal
    - Vegas 8%: team total r=0.61 with scoring — was underweighted at 5%
    - Park 7%: Coors/Petco park effects real (r≈0.85 Y-to-Y)
    - Streak 5%: real but 7-game window is noisy
    - TTO 4%: 3rd TTO +17-20 wOBA documented
    - Weather 4%: wind 15+ mph = ~12% more HR
    - Pitch matchup 2%: rarely has data; was placeholder at 4%
    - BvP 2%: very small samples; dynamic boost preserved for "owns" cases
    - Lineup 1%: least predictive — 1 extra PA is marginal
    """
    # V1.8 research-calibrated weights (sum ≈ 1.03, normalized below)
    # Key changes from V1.7:
    #   Batter  0.33→0.28: less dominance — matchup should drive picks, not career quality
    #   Pitcher 0.25→0.30: primary matchup signal; wider scale now gives real separation
    #   Platoon 0.08→0.12: most stable contextual signal (+56 SLG well-documented)
    #   Vegas   0.05→0.08: implied total r=0.61 with scoring — was underweighted
    #   Streak  0.06→0.05: 7-game window is noisy, trim
    #   Matchup 0.04→0.02: pitch-type splits rarely available, reduce placeholder weight
    #   BvP     0.03→0.02: very small samples, reduce (dynamic boost preserved)
    # Dynamic BvP boost: when batter "owns" this SP (elite career numbers),
    # BvP weight rises from 0.02 → 0.06 and batter weight reduced to compensate.
    _bvp_w = 0.02 + bvp_weight_boost           # 0.02 normally; 0.06 when "owns"
    _bat_w = max(0.24, 0.28 - bvp_weight_boost) # 0.28 normally; 0.24 when "owns"
    raw = (
        batter_score        * _bat_w +  # batter quality: important but matchup matters more
        pitcher_vuln_score  * 0.30 +  # pitcher matchup: primary pick driver (bumped from 0.25)
        platoon_score       * 0.12 +  # platoon: most stable contextual signal (bumped from 0.08)
        vegas_score         * 0.08 +  # implied total: r=0.61 with scoring (bumped from 0.05)
        park_score          * 0.07 +  # park: Coors/Petco real (r≈0.85 Y-to-Y)
        streak_score        * 0.05 +  # streak: real but noisy 7-game window (trimmed from 0.06)
        tto_bonus           * 0.04 +  # TTO: 3rd TTO +17-20 wOBA documented
        weather_score       * 0.04 +  # weather: wind 15+ mph = ~12% more HR
        pitch_matchup_score * 0.02 +  # pitch matchup: often no data — keep small (trimmed from 0.04)
        bvp_score           * _bvp_w +  # BvP: 2% base; 6% when batter "owns" this SP
        lineup_score        * 0.01    # lineup: least predictive; 1 extra PA is marginal
    )  # sum = 1.03 → normalize
    raw = raw / 1.03  # normalize to true 0-100 scale
    # V1.8 offset: reduced from 10→7 (less artificial inflation, more room above 80 for elite matchups)
    # Proxy mode offset also reduced: 13→10
    import streamlit as _st
    _bat_src  = _st.session_state.get("_batting_source", "")
    _bat_cols = _st.session_state.get("batting_cols", [])
    # Full data = Barrel%, HH%, AND wRC+ all present
    _has_full = ("Barrel%" in _bat_cols or "barrel_batted_rate" in _bat_cols) and \
                ("Hard%" in _bat_cols or "hard_hit_percent" in _bat_cols) and \
                ("wRC+" in _bat_cols)
    # Proxy mode: source says mlbapi only, or statcast columns missing
    _is_proxy = ("mlbapi" in _bat_src or _bat_src in ("mlbapi_only",) or
                 "disk_cache_stale" in _bat_src or not _has_full)
    _offset = 9.5 if _is_proxy else 6.5   # V1.8: calibrated so avg batter vs avg SP = 55
    calibrated = raw + _offset
    return max(0, min(100, round(calibrated, 1)))


def score_to_prob(score: float) -> float:
    """
    Map 0-100 score to probability (O1.5 TB props).
    Research-calibrated against MLB prop hit rates:
    - Score 50 (league avg batter) → ~52% probability
    - Score 60 (Tier 3 floor) → ~58% probability
    - Score 70 (Tier 2 floor) → ~64% probability
    - Score 80 (Tier 1 floor) → ~70% probability
    - Score 90 (elite matchup) → ~74% probability
    """
    a = 0.07   # steeper slope for better differentiation
    b = 62     # midpoint: score 62 → ~52% baseline
    prob = 1 / (1 + math.exp(-a * (score - b)))
    # Scale to realistic MLB TB prop range (42-78%)
    prob = 0.42 + prob * 0.36
    return round(min(0.78, max(0.42, prob)), 3)


def get_tier(score: float, proxy_mode: bool = False) -> str:
    """
    Map score to tier label.
    proxy_mode=True when running on MLB Stats API proxies (no Savant).
    In proxy mode, tier thresholds shift down 5 pts to account for
    compressed score range (Savant signals unavailable → avg score ~5 pts lower).
    """
    if proxy_mode:
        # Proxy-data thresholds
        if score >= 75:   return "🔒 TIER 1"
        elif score >= 65: return "✅ TIER 2"
        elif score >= 55: return "📊 TIER 3"
        else:             return "❌ NO PLAY"
    else:
        # Full Savant thresholds — V1.8: shifted -2 to match reduced offset
        if score >= 80:   return "🔒 TIER 1"
        elif score >= 70: return "✅ TIER 2"
        elif score >= 60: return "📊 TIER 3"
        else:             return "❌ NO PLAY"

# ============================================================================
# HR SCORE (separate from TB score)
# ============================================================================
def compute_hr_score(
    barrel_rate: float,
    sweet_spot: float,
    park_hr_factor: float,
    implied_total: float,
    weather: Dict,
    hard_hit: float = 0.37,
    exit_velocity: float = 88.5,
    iso: float = 0.165,
    ev50: float = 95.0,
    bat_speed: float = 71.0,
    blast_rate: float = 0.21,
    pitch_matchup_score: float = 50.0,
) -> float:
    """
    Compute dedicated HR upside score 0-100.
    V1.6: Added EV50, bat speed, blast rate, pitch matchup.

    Signal weights (research-backed):
    - Barrel%        35% — r=0.93 with HR rate, #1 predictor
    - Park factor    15% — Coors/GABP vs pitcher's parks
    - EV50           10% — hardest 50% EV — best power ceiling signal (NEW Savant)
    - Bat speed       8% — mechanical HR ceiling (75+ mph = HR capable) (NEW Savant)
    - Hard hit%       8% — contact quality / exit velocity proxy
    - ISO             7% — raw power history, stable Y-to-Y
    - Blast rate      6% — fast + squared up = HR swing quality (NEW Savant)
    - Vegas implied   5% — high-total games = more HR opportunities
    - Pitch matchup   4% — favorable pitch type RV for FB/power pitches (NEW V1.6)
    - Wind/weather dynamic
    """
    # ── Barrel% — #1 HR predictor (r=0.93) ────────────────────────────
    barrel_score = max(0, min(100, barrel_rate / 0.20 * 100))

    # ── EV50 — hardest 50% avg exit velocity (V1.6 NEW) ───────────────
    # 88mph=0, 96mph=50, 104mph=100
    ev50_score = max(0, min(100, (ev50 - 88.0) / (104.0 - 88.0) * 100))

    # ── Bat speed — mechanical HR ceiling (V1.6 NEW) ──────────────────
    # 65mph=0, 71mph=50, 78mph=100
    bat_speed_score = max(0, min(100, (bat_speed - 65.0) / (78.0 - 65.0) * 100))

    # ── Blast rate — fast swing + square contact (V1.6 NEW) ───────────
    # 10%=0, 21%=50, 35%=100
    blast_score = max(0, min(100, (blast_rate - 0.10) / (0.35 - 0.10) * 100))

    # ── Hard hit% — contact quality / power proxy ─────────────────────
    hh_score = max(0, min(100, (hard_hit - 0.28) / (0.56 - 0.28) * 100))

    # ── ISO — raw power history ────────────────────────────────────────
    iso_score = max(0, min(100, (iso - 0.080) / (0.320 - 0.080) * 100))

    # ── Park HR factor ─────────────────────────────────────────────────
    park_score = max(0, min(100, (park_hr_factor - 0.85) / (1.35 - 0.85) * 100))

    # ── Vegas implied total ────────────────────────────────────────────
    vegas_score = max(0, min(100, (implied_total - 3.0) / (6.5 - 3.0) * 100)) if implied_total > 0 else 40.0

    # ── Pitch matchup (V1.6) — already 0-100, wire in directly ────────
    matchup_score = max(0, min(100, pitch_matchup_score))

    # ── Wind / weather — dynamic weight based on speed ────────────────
    # Wind is conditionally significant: 20mph out = ~25% more HRs, 7mph = noise.
    # Strategy: scale both the bonus AND its weight by wind speed.
    # Below 8mph: neutral, near-zero weight (don't let calm wind steal weight
    # from barrel% and park factor).
    # 8-14mph: moderate signal, ~8% weight.
    # 15mph+: strong signal, up to 15% weight — redistributed from park_score.
    wind_raw    = 0.0   # raw directional adjustment (-100 to +100 scale)
    wind_weight = 0.0   # dynamic weight pulled into composite

    if not weather.get("is_dome"):
        effect = weather.get("wind_effect", "neutral")
        speed  = float(weather.get("wind_speed", 0))
        temp   = float(weather.get("temperature", 70))

        if speed >= 8:
            # Scale wind effect linearly with speed: 8mph=base, 25mph=max
            speed_factor = min(1.0, (speed - 8.0) / (25.0 - 8.0))

            if effect == "strong_out":
                wind_raw = 80.0 * speed_factor      # up to +80 pts on 0-100 scale
                wind_weight = 0.08 + 0.07 * speed_factor  # 8-15% weight
            elif effect == "out":
                wind_raw = 55.0 * speed_factor
                wind_weight = 0.05 + 0.07 * speed_factor
            elif effect == "in":
                wind_raw = -70.0 * speed_factor     # suppresses HRs hard
                wind_weight = 0.05 + 0.07 * speed_factor
            else:
                wind_raw = 0.0
                wind_weight = 0.0

        # Temperature: cold air is denser, suppresses HR distance
        # Add directly to composite as a fixed small adjustment (not weight-scaled)
        temp_adj = 0.0
        if temp < 45:
            temp_adj = -8.0
        elif temp < 55:
            temp_adj = -4.0
        elif temp > 85:
            temp_adj = +4.0
        elif temp > 92:
            temp_adj = +7.0
    else:
        temp_adj = 0.0

    # Redistribute weight: wind steals from park_score when significant
    adjusted_park_weight = max(0.06, 0.15 - wind_weight)

    # ── Composite V1.6 ─────────────────────────────────────────────────
    base = (
        barrel_score    * 0.35 +
        ev50_score      * 0.10 +     # V1.6 NEW: best power ceiling signal
        bat_speed_score * 0.08 +     # V1.6 NEW: mechanical HR ceiling
        park_score      * adjusted_park_weight +
        hh_score        * 0.08 +
        iso_score       * 0.07 +
        blast_score     * 0.06 +     # V1.6 NEW: swing quality
        vegas_score     * 0.05 +
        matchup_score   * 0.04       # V1.6 NEW: pitch type matchup
    )

    wind_contribution = wind_raw * wind_weight if wind_weight > 0 else 0.0
    composite = base + wind_contribution + temp_adj

    return max(0, min(100, round(composite, 1)))

# ============================================================================
# ROSTER FALLBACK — fetch team roster when lineup not yet posted
# ============================================================================
@st.cache_data(ttl=3600)
def fetch_team_roster(team_id: int) -> List[Dict]:
    """Fetch 40-man roster as fallback when lineup not confirmed."""
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
    params = {"rosterType": "active"}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        batters = []
        for p in data.get("roster", []):
            pos = p.get("position", {}).get("type", "")
            if pos in ("Pitcher",):
                continue
            batters.append({
                "player_id": str(p["person"]["id"]),
                "name": p["person"]["fullName"],
                "lineup_slot": 5,  # unknown slot
                "batter_hand": "R",  # default
                "position": p.get("position", {}).get("abbreviation", ""),
                "projected": True,
            })
        return batters[:9]
    except:
        return []

@st.cache_data(ttl=3600)
def fetch_team_id(abbreviation: str) -> Optional[int]:
    """Get MLB team ID from abbreviation."""
    url = "https://statsapi.mlb.com/api/v1/teams"
    params = {"sportId": 1, "season": 2026}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        for team in data.get("teams", []):
            if team.get("abbreviation") == abbreviation:
                return team["id"]
    except:
        pass
    return None

# ============================================================================
# MAIN MODEL PIPELINE
# ============================================================================
def run_model(date_str: str, status_container) -> List[Dict]:
    """
    Master pipeline: pulls all data, scores every batter, returns ranked list.
    Robust fallbacks at every step — always produces output.
    """
    results = []
    
    # Live status log in UI
    status_box = status_container.empty()
    log_lines = []
    
    def log(msg, level="info"):
        icon = {"info": "ℹ️", "ok": "✅", "warn": "⚠️", "err": "❌", "run": "⚙️"}.get(level, "")
        line = f"{icon} {msg}" if icon else msg
        log_lines.append(line)
        status_box.markdown("\n\n".join(log_lines[-12:]))
    
    log(f"**⚾ MLB TB Analyzer — {date_str}**")
    log("─" * 40)
    # Reset match tracking
    st.session_state["_matched"] = 0
    st.session_state["_unmatched"] = 0
    st.session_state["_search_names"] = []

    # ── 0. BULK STATS (one call loads all players) ───────
    log("Loading 2025 season batting stats...", "run")
    batting_df = load_all_batting_stats(2025)
    pitching_df = load_all_pitching_stats(2025)
    # V1.9: Store pitching_df globally so K Props and Moneyline tabs can access
    # without re-fetching. Uses session_state (safe across reruns in same session).
    st.session_state["_pitching_df_global"] = pitching_df

    # ── Set source labels in run_model (not inside @st.cache_data) ───────
    # @st.cache_data skips function body on cache HIT, so session_state
    # assignments inside cached functions only fire on cache MISS.
    # We detect source here based on what columns are present in the result.
    if batting_df.empty:
        st.session_state["_batting_source"] = "failed"
    elif any(c in batting_df.columns for c in ('xSLG','est_slg','xslg','xMLBAMID')):
        # Check disk cache age
        import os as _os, time as _time
        _cache_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "stat_cache")
        _bat_cache = _os.path.join(_cache_dir, "batting_stats.pkl")
        if _os.path.exists(_bat_cache):
            _age_h = (_time.time() - _os.path.getmtime(_bat_cache)) / 3600
            if _age_h < 6:
                st.session_state["_batting_source"] = "disk_cache_fresh"
            elif _age_h < 168:
                st.session_state["_batting_source"] = "disk_cache_stale"
            else:
                st.session_state["_batting_source"] = "savant+mlbapi"
        else:
            st.session_state["_batting_source"] = "savant+mlbapi"
    else:
        st.session_state["_batting_source"] = "mlbapi_only"

    if pitching_df.empty:
        st.session_state["_pitching_source"] = "failed"
    else:
        import os as _os2, time as _time2
        _cache_dir2 = _os2.path.join(_os2.path.dirname(_os2.path.abspath(__file__)), "stat_cache")
        _pit_cache = _os2.path.join(_cache_dir2, "pitching_stats.pkl")
        if _os2.path.exists(_pit_cache):
            _age_h2 = (_time2.time() - _os2.path.getmtime(_pit_cache)) / 3600
            st.session_state["_pitching_source"] = f"disk_cache_{('fresh' if _age_h2 < 6 else 'stale')}"
        else:
            st.session_state["_pitching_source"] = "mlbapi+savant"
    statcast_df = pd.DataFrame()  # merged into batting_df now

    if not batting_df.empty:
        log(f"Batting stats: {len(batting_df)} players loaded", "ok")
        batting_df = prepare_lookup_df(batting_df)  # build _norm_name index ONCE
        # Store debug info
        st.session_state.batting_cols = list(batting_df.columns)
        # Log which key stat columns actually loaded
        key_cols = ["xSLG", "SLG", "ISO", "K%", "BB%", "wRC+", "wOBA",
                    "Barrel%", "Hard%", "EV", "xMLBAMID"]
        present = [c for c in key_cols if c in batting_df.columns]
        missing = [c for c in key_cols if c not in batting_df.columns]
        if missing:
            log(f"  ⚠️ Missing stat cols: {missing}", "warn")
        else:
            log(f"  Key cols present: {present[:6]}... ✅", "ok")
        # Store raw name samples to diagnose lookup failures
        if "_name" in batting_df.columns:
            st.session_state.batting_df_sample = [str(x) for x in batting_df["_name"].head(10).tolist()]
        else:
            # Dump ALL column names and first row values
            cols = list(batting_df.columns)
            first = batting_df.iloc[0] if not batting_df.empty else None
            samples = [f"COLUMNS (first 40): {cols[:40]}"]
            if first is not None:
                # Show columns that have non-null string-looking values (potential name cols)
                for c in cols[:60]:
                    v = first.get(c)
                    if v is not None and isinstance(v, str) and len(str(v)) > 2:
                        samples.append(f"  col '{c}' = '{str(v)[:80]}'")
            st.session_state.batting_df_sample = samples[:20]
        if "_norm_name" in batting_df.columns:
            st.session_state.norm_name_sample = [str(x) for x in batting_df["_norm_name"].head(10).tolist()]
        else:
            st.session_state.norm_name_sample = ["_norm_name NOT BUILT — _name column missing"]
        # Also show first row raw to see column names with data
        if not batting_df.empty:
            first_row = batting_df.iloc[0]
            st.session_state.batting_df_sample.append(f"First row columns with data: {[c for c in batting_df.columns if pd.notna(first_row.get(c)) and str(first_row.get(c)) not in ('', 'nan')][:15]}")
        # Find Judge specifically to verify stats loading
        judge_row = find_player_row(batting_df, "Aaron Judge", "")
        sample = judge_row if judge_row is not None else (batting_df.iloc[0] if not batting_df.empty else None)
        if sample is not None:
            st.session_state.sample_player = {
                "_name (cleaned)": str(sample.get("_name", "MISSING — HTML not stripped")),
                "_mlb_id": str(sample.get("_mlb_id", "MISSING")),
                "SLG": sample.get("SLG", "❌ missing"),
                "ISO": sample.get("ISO", "❌ missing"),
                "K%": sample.get("K%", "❌ missing"),
                "Barrel%": sample.get("Barrel%", "❌ missing"),
                "Hard%": sample.get("Hard%", "❌ missing"),
                "wRC+": sample.get("wRC+", "❌ missing"),
                "xSLG": sample.get("xSLG", "❌ missing"),
            }
    else:
        bat_src = st.session_state.get("_batting_source", "unknown")
        fg_errs = st.session_state.get("_fg_batting_errors", [])
        if bat_src == "disk_cache":
            log("⚠️ FanGraphs unreachable — serving stats from DISK CACHE (may be up to 48h old)", "warn")
        elif bat_src == "failed":
            log("❌ Batting stats FAILED — all scores use league averages", "warn")
            if fg_errs:
                for e in fg_errs[:3]:
                    log(f"   FG error: {e}", "warn")
                log("   → Fix: Clear cache & rerun, or wait for FanGraphs to unblock Streamlit IPs", "warn")
        else:
            log("Batting stats unavailable — all scores use league averages", "warn")
            if fg_errs:
                for e in fg_errs[:3]:
                    log(f"   FG error: {e}", "warn")
    if not pitching_df.empty:
        log(f"Pitching stats: {len(pitching_df)} pitchers loaded", "ok")
        pitching_df = prepare_lookup_df(pitching_df)  # build _norm_name index ONCE
        st.session_state.pitching_cols = list(pitching_df.columns)
        # ── V1.3: Compute per-team bullpen scores ONCE before scoring loop ──
        team_bullpen_scores = compute_team_bullpen_scores(pitching_df)
        n_bp_teams = len(team_bullpen_scores)
        if n_bp_teams > 0:
            log(f"Bullpen quality computed for {n_bp_teams} teams ✅", "ok")
        else:
            log("Bullpen quality unavailable — using league average (42.0) for all teams", "warn")
        st.session_state["team_bullpen_scores"] = team_bullpen_scores
    else:
        pit_src = st.session_state.get("_pitching_source", "unknown")
        pit_fg_errs = st.session_state.get("_fg_pitching_errors", [])
        if pit_src == "disk_cache":
            log("⚠️ FanGraphs unreachable — serving pitcher stats from DISK CACHE", "warn")
        else:
            log("❌ Pitching stats unavailable — using league averages", "warn")
            if pit_fg_errs:
                for e in pit_fg_errs[:2]:
                    log(f"   FG error: {e}", "warn")
        team_bullpen_scores = {}

    # ── 1. SCHEDULE ──────────────────────────────────────
    log("Fetching MLB schedule...", "run")
    games = fetch_schedule(date_str)
    
    if not games:
        log(f"No MLB games scheduled on {date_str}.", "warn")
        log("Try selecting a different date — Opening Day is **March 27, 2026**.", "info")
        log("Use the date picker in the sidebar ↑", "info")
        status_box.markdown("\n\n".join(log_lines))
        return []
    
    log(f"Found **{len(games)} games** on {date_str}", "ok")
    for g in games:
        hp = g.get("home_pitcher") or "TBD"
        ap = g.get("away_pitcher") or "TBD"
        log(f"{g['away_team']} @ {g['home_team']}  |  {ap} vs {hp}")

    # ── 2. ODDS ───────────────────────────────────────────
    log("Fetching Vegas lines...", "run")
    implied_totals = {}
    prop_odds = {}
    try:
        implied_totals = fetch_odds(date_str)
    except Exception:
        pass
    if implied_totals:
        log(f"Live odds loaded for {len(implied_totals)} teams ✅", "ok")
        # Also fetch prop-specific odds for precise edge calculation
        try:
            prop_odds = fetch_prop_odds(date_str)
            if prop_odds:
                log(f"Player prop odds loaded for {len(prop_odds)} players ✅", "ok")
        except Exception:
            pass
    else:
        log("⚠️ No Odds API key — Vegas signal (8% weight) zeroed out. Add key in sidebar for full model.", "warn")

    # ── 3. PROCESS EACH GAME ─────────────────────────────
    total_batters = 0
    games_skipped = 0

    for game in games:
        game_pk = game["game_pk"]
        home_team = game["home_team"]
        away_team = game["away_team"]
        
        log(f"Processing: **{away_team} @ {home_team}**...", "run")

        # Park / weather
        park_info = STADIUM_COORDS.get(home_team, (40.7, -74.0, "Unknown Stadium", False))
        lat, lon, park_name, is_dome = park_info
        weather = fetch_weather(lat, lon, game.get("game_time", ""), is_dome)

        # Lineups — try confirmed first, fall back to roster
        lineups = fetch_lineup(game_pk)
        home_batters = lineups.get("home", [])
        away_batters = lineups.get("away", [])
        lineup_confirmed = bool(home_batters or away_batters)

        if not lineup_confirmed:
            log(f"  Lineup not posted yet — using projected roster (flagged)", "warn")
            # Fallback: use roster
            home_id = fetch_team_id(home_team)
            away_id = fetch_team_id(away_team)
            if home_id:
                home_batters = fetch_team_roster(home_id)
            if away_id:
                away_batters = fetch_team_roster(away_id)
            if not home_batters and not away_batters:
                log(f"  Could not load roster for {away_team}@{home_team} — skipping", "err")
                games_skipped += 1
                continue

        # Pitcher info
        home_pitcher_id = game.get("home_pitcher_id")
        away_pitcher_id = game.get("away_pitcher_id")
        home_pitcher_info = fetch_pitcher_info(home_pitcher_id) if home_pitcher_id else {"name": "TBD", "hand": "R", "id": None}
        away_pitcher_info = fetch_pitcher_info(away_pitcher_id) if away_pitcher_id else {"name": "TBD", "hand": "R", "id": None}

        # Build batter list
        all_batters = []
        for b in home_batters[:9]:
            b = dict(b)
            b.update({"team": home_team, "opponent": away_team,
                      "opposing_pitcher": away_pitcher_info,
                      "park_team": home_team, "lineup_confirmed": lineup_confirmed})
            all_batters.append(b)
        for b in away_batters[:9]:
            b = dict(b)
            b.update({"team": away_team, "opponent": home_team,
                      "opposing_pitcher": home_pitcher_info,
                      "park_team": home_team, "lineup_confirmed": lineup_confirmed})
            all_batters.append(b)

        log(f"  Scoring {len(all_batters)} batters...", "run")
        first_batter_logged = False

        for batter in all_batters:
            player_id = batter.get("player_id", "")
            name = batter.get("name", "Unknown")
            team = batter.get("team", "")
            sp_info = batter.get("opposing_pitcher", {})
            sp_name = sp_info.get("name", "TBD")
            sp_hand = sp_info.get("hand", "R")
            lineup_slot = batter.get("lineup_slot", 5)
            batter_hand = batter.get("batter_hand", "")
            park_team = batter.get("park_team", home_team)

            # Stats from pre-loaded bulk DataFrames (fast, no per-player API calls)
            batter_statcast = get_batter_stats(
                player_name=name,
                mlb_id=player_id,
                batting_df=batting_df,
                statcast_df=statcast_df,
            )

            # Resolve handedness: FanGraphs "Bats" column is most reliable
            # (always available, never ? for MLB players)
            # Falls back to lineup data, then MLB API, then R default
            if not batter_hand or batter_hand in ("?", ""):
                # Try FanGraphs Bats column via the player row
                fg_row = find_player_row(batting_df, name, player_id)
                if fg_row is not None:
                    fg_bats = str(fg_row.get("Bats", "") or "").strip().upper()
                    if fg_bats in ("L", "R", "S", "B"):
                        batter_hand = fg_bats
                if not batter_hand or batter_hand in ("?", ""):
                    batter_hand = "R"  # final fallback

            # DIAGNOSTIC: on first batter, show exactly what find_player_row sees
            if not first_batter_logged:
                nc = "_norm_name" if "_norm_name" in batting_df.columns else "_name" if "_name" in batting_df.columns else "NONE"
                n_rows = len(batting_df)
                sample_vals = []
                if nc != "NONE":
                    sample_vals = batting_df[nc].head(3).tolist()
                has_xmlbamid = "xMLBAMID" in batting_df.columns
                has_mlbamid  = "MLBAMID" in batting_df.columns
                xmlbam_col   = "xMLBAMID" if has_xmlbamid else ("MLBAMID" if has_mlbamid else None)
                xmlbam_sample = batting_df[xmlbam_col].head(3).tolist() if xmlbam_col else []
                st.session_state["lookup_diag"] = {
                    "searching_for": f"{name} (MLBAM id={player_id})",
                    "batting_df_rows": n_rows,
                    "name_col_used": nc,
                    "first_3_norm_names": [str(v) for v in sample_vals],
                    "xMLBAMID_exists": has_xmlbamid,
                    "MLBAMID_exists": has_mlbamid,
                    "id_col_used": xmlbam_col or "NONE — ID lookup disabled",
                    "xMLBAMID_sample": [str(v) for v in xmlbam_sample],
                    "data_source": batter_statcast.get("data_source"),
                    "matched": batter_statcast.get("data_source", "league_avg") != "league_avg",
                }

            # Store first 5 searched names for debug
            if "_search_names" not in st.session_state:
                st.session_state["_search_names"] = []
            if len(st.session_state["_search_names"]) < 5:
                st.session_state["_search_names"].append(f"'{name}' (id={player_id})")
            st.session_state["search_sample"] = st.session_state["_search_names"]

            # Track match rate
            _ds = batter_statcast.get("data_source", "league_avg")
            if _ds != "league_avg":
                st.session_state["_matched"] = st.session_state.get("_matched", 0) + 1
            else:
                st.session_state["_unmatched"] = st.session_state.get("_unmatched", 0) + 1

            # Debug log first batter so we can see what's loading
            if not first_batter_logged:
                src = batter_statcast.get("data_source", "?")
                hh = batter_statcast.get("hard_hit_rate", 0)
                br = batter_statcast.get("barrel_rate", 0)
                xslg = batter_statcast.get("slg_proxy", 0)
                log(f"  Sample ({name}): source={src} xSLG={xslg:.3f} barrel={br*100:.1f}% HH={hh*100:.1f}%", "info")
                first_batter_logged = True

            sp_id = str(sp_info.get("id", "") or "")
            pitcher_statcast = get_pitcher_stats(
                pitcher_name=sp_name,
                pitcher_mlb_id=sp_id,
                pitching_df=pitching_df,
            )
            # V1.9: Cache pitcher K signals per play so K Props tab can use them
            # without re-running get_pitcher_stats for every row in the display function
            _pitcher_k_rate = float(pitcher_statcast.get("k_rate_allowed", 0.228) or 0.228)
            _pitcher_swstr  = float(pitcher_statcast.get("swstr_pct", 0.0) or 0.0)

            # ── SCORE COMPONENTS ────────────────────────
            bat_score, _, bat_details = compute_batter_score(batter_statcast)
            # V1.3: Look up opponent team's specific bullpen vulnerability
            opp_team = batter.get("opponent", "").strip().upper()
            bp_vuln = team_bullpen_scores.get(opp_team, 42.0)
            pit_score, pit_label = compute_pitcher_score(pitcher_statcast, bullpen_vuln=bp_vuln)
            # V1.5: Pitch arsenal matchup — batter RV vs SP's actual pitch mix
            matchup_sc, matchup_label = compute_pitch_matchup_score(batter_statcast, pitcher_statcast)
            plat_score, plat_label = compute_platoon_score(batter_hand, sp_hand)
            lineup_sc, lineup_label = compute_lineup_score(lineup_slot)
            park_sc, park_label = compute_park_score(park_team, True)
            weather_sc, weather_label = compute_weather_score(weather)
            implied = implied_totals.get(team, 0)
            vegas_sc, vegas_label = compute_vegas_score(implied)
            tto_sc, tto_label = compute_tto_bonus(lineup_slot)

            # V1.7 NEW: Recent form (last 7 games) ──────────────────────────
            recent_form = fetch_batter_recent_form(str(player_id), n_games=7)
            season_slg  = batter_statcast.get("slg_proxy", 0.398)
            streak_sc, streak_label = compute_streak_score(recent_form, season_slg)

            # V1.7 NEW: Career Batter vs Pitcher ────────────────────────────
            bvp_data = fetch_batter_vs_pitcher(str(player_id), sp_id)
            bvp_sc, bvp_label, bvp_sig = compute_bvp_score(bvp_data, season_slg)

            # "Owns" flag: boost final score weight when batter dominates this SP
            # This fires when batter has elite career numbers vs this specific pitcher
            _bvp_weight_boost = 0.04 if bvp_sig == "owns" else 0.0

            final_score = compute_final_score(
                bat_score, pit_score, plat_score, lineup_sc,
                park_sc, weather_sc, vegas_sc, tto_sc,
                pitch_matchup_score=matchup_sc,
                streak_score=streak_sc,
                bvp_score=bvp_sc,
                bvp_weight_boost=_bvp_weight_boost,
            )

            # Caps & flags
            sp_tbd = not sp_name or sp_name == "TBD"
            if sp_tbd:
                final_score = min(final_score, 72)
            if not batter.get("lineup_confirmed", True):
                final_score = min(final_score, 70)

            prob = score_to_prob(final_score)
            # Detect proxy mode for adaptive tier thresholds
            _bat_src_loop = st.session_state.get("_batting_source", "")
            _proxy_mode = "mlbapi" in _bat_src_loop or _bat_src_loop in ("disk_cache_stale", "mlbapi_only")
            tier = get_tier(final_score, proxy_mode=_proxy_mode)

            # Market edge calculation
            # Market edge — use prop-specific odds when available
            prop_implied = None
            if prop_odds:
                norm_name = _norm(name)
                prop_data = prop_odds.get(norm_name)
                if not prop_data:
                    # Try last name match
                    last = norm_name.split()[-1] if norm_name else ""
                    prop_data = next((v for k, v in prop_odds.items() if last in k), None)
                if prop_data:
                    prop_implied = prop_data.get("market_implied")
            market_edge, edge_label = compute_market_edge(prob, implied, team, prop_implied)

            hr_score = compute_hr_score(
                barrel_rate=batter_statcast.get("barrel_rate", 0.07),
                sweet_spot=batter_statcast.get("sweet_spot_rate", 0.30),
                park_hr_factor=PARK_HR_FACTORS.get(park_team, 1.0),
                implied_total=implied,
                weather=weather,
                hard_hit=batter_statcast.get("hard_hit_rate", 0.37),
                exit_velocity=batter_statcast.get("exit_velocity_avg", 88.5),
                iso=batter_statcast.get("iso_proxy", 0.165),
                ev50=batter_statcast.get("ev50", 95.0),
                bat_speed=batter_statcast.get("bat_speed", 71.0),
                blast_rate=batter_statcast.get("blast_rate", 0.21),
                pitch_matchup_score=matchup_sc,
            )

            results.append({
                "name": name,
                "player_id": player_id,
                "team": team,
                "opponent": batter.get("opponent", "?"),
                "game_id": str(game_pk),
                "lineup_slot": lineup_slot,
                "lineup_confirmed": batter.get("lineup_confirmed", True),
                "batter_hand": batter_hand,
                "batter_position": batter.get("position", ""),
                "sp_name": sp_name,
                "sp_hand": sp_hand,
                "sp_tbd": sp_tbd,
                "score": final_score,
                "prob": prob,
                "tier": tier,
                "park": park_team,
                "park_label": park_label,
                "weather": weather,
                "weather_label": weather_label,
                "implied_total": implied,
                "market_edge": round(market_edge * 100, 1),  # as percentage
                "edge_label": edge_label,
                "tto_label": tto_label,
                "platoon_label": plat_label,
                "lineup_label": lineup_label,
                "pitcher_label": pit_label,
                "hr_score": hr_score,
                "xslg": bat_details.get("xSLG", 0),
                "barrel_rate": batter_statcast.get("barrel_rate", 0),
                "hard_hit_rate": batter_statcast.get("hard_hit_rate", 0),
                "k_rate": batter_statcast.get("k_rate", 0),
                "bb_rate": batter_statcast.get("bb_rate", 0.082),
                "wrc_plus": batter_statcast.get("wrc_plus", 100.0),
                "iso": bat_details.get("ISO", 0),
                "exit_velocity": batter_statcast.get("exit_velocity_avg", 0),
                "sweet_spot_rate": batter_statcast.get("sweet_spot_rate", 0),
                "sub_batter": round(bat_score, 1),
                "sub_pitcher": round(pit_score, 1),
                "sub_matchup": round(matchup_sc, 1),
                "matchup_label": matchup_label,
                # V1.7 NEW: recent form and BvP
                "sub_streak": round(streak_sc, 1),
                "streak_label": streak_label,
                "recent_tb_per_game": recent_form.get("tb_per_game"),
                "recent_games": recent_form.get("games", 0),
                "sub_bvp": round(bvp_sc, 1),
                "bvp_label": bvp_label,
                "bvp_sig": bvp_sig,         # 'owns'/'edge'/'neutral'/'fade'/'dominated'/'no_data'
                "bvp_ab": bvp_data.get("ab", 0),
                "bvp_slg": bvp_data.get("slg"),
                "bvp_hr": bvp_data.get("hr", 0),
                "bvp_xbh": bvp_data.get("xbh", 0),
                "sp_id": sp_id,
                "sub_platoon": round(plat_score, 1),
                "sub_lineup": round(lineup_sc, 1),
                "sub_park": round(park_sc, 1),
                "sub_weather": round(weather_sc, 1),
                "sub_vegas": round(vegas_sc, 1),
                "bullpen_vuln": round(bp_vuln, 1),
                "platoon_edge": plat_label,
                "bat_speed":    batter_statcast.get("bat_speed", 0),
                "blast_rate":   batter_statcast.get("blast_rate", 0),
                "ev50":         batter_statcast.get("ev50", 0),
                "sprint_speed": batter_statcast.get("sprint_speed", 0),
                "temperature": weather.get("temperature", 70),
                "wind_speed": weather.get("wind_speed", 0),
                "wind_dir": weather.get("wind_dir_label", ""),
                "wind_effect": weather.get("wind_effect", "neutral"),
                "is_dome": weather.get("is_dome", False),
                # V1.9: pitcher K signals for K Props tab
                "_pitcher_k_rate": _pitcher_k_rate,
                "_pitcher_swstr":  _pitcher_swstr,
            })
            total_batters += 1

        log(f"  ✅ {away_team}@{home_team} done — {len(all_batters)} batters scored")

    # Sort
    results.sort(key=lambda x: x["score"], reverse=True)

    log("─" * 40)
    # ── Match rate diagnostic ─────────────────────────────────────────────
    matched_total   = st.session_state.get("_matched", 0)
    unmatched_total = st.session_state.get("_unmatched", 0)
    total_lookup    = matched_total + unmatched_total
    if total_lookup > 0:
        match_pct = matched_total / total_lookup * 100
        if match_pct < 50:
            log(f"⚠️ STAT MATCH RATE: {matched_total}/{total_lookup} ({match_pct:.0f}%) — "
                f"most batters using league averages. Check Debug panel → Data Quality.", "warn")
        elif match_pct < 80:
            log(f"⚠️ Stat match rate: {matched_total}/{total_lookup} ({match_pct:.0f}%) — some league-avg fallbacks", "warn")
        else:
            log(f"✅ Stat match rate: {matched_total}/{total_lookup} ({match_pct:.0f}%)", "ok")
    if total_batters == 0:
        log("No batters scored. Lineups likely not posted yet — try again closer to game time.", "warn")
    else:
        tier1 = sum(1 for r in results if r["tier"] == "🔒 TIER 1")
        tier2 = sum(1 for r in results if r["tier"] == "✅ TIER 2")
        tier3 = sum(1 for r in results if r["tier"] == "📊 TIER 3")
        log(f"**Done! {total_batters} batters scored across {len(games)-games_skipped} games**", "ok")
        log(f"🔒 Tier 1: {tier1}  |  ✅ Tier 2: {tier2}  |  📊 Tier 3: {tier3}")

    return results

# ============================================================================
# PARLAY BUILDER
# ============================================================================
def build_parlays(
    plays: List[Dict],
    num_legs: int = 3,
    max_same_team: int = 2,
    min_score: float = 70.0
) -> List[Dict]:
    """
    Build optimal parlays from eligible plays.
    Prioritizes: high score, different teams/games, Tier 1 anchors.
    """
    eligible = [p for p in plays if p["score"] >= min_score and not p["sp_tbd"]]
    
    if len(eligible) < num_legs:
        return []
    
    best_parlays = []
    
    for combo in combinations(eligible[:20], num_legs):  # Limit combos for performance
        # Check team diversity
        teams = [p["team"] for p in combo]
        games = [p["game_id"] for p in combo]
        
        team_counts = {}
        for t in teams:
            team_counts[t] = team_counts.get(t, 0) + 1
        
        if max(team_counts.values()) > max_same_team:
            continue
        
        # Calculate combined probability (assume independence + small correlation discount)
        probs = [p["prob"] for p in combo]
        
        # Correlation discount: same game = 0.95, same team = 0.90
        corr_factor = 1.0
        for i in range(len(combo)):
            for j in range(i+1, len(combo)):
                if combo[i]["team"] == combo[j]["team"]:
                    corr_factor *= 0.90
                elif combo[i]["game_id"] == combo[j]["game_id"]:
                    corr_factor *= 0.95
        
        combined_raw = 1.0
        for p in probs:
            combined_raw *= p
        combined_prob = combined_raw * corr_factor
        
        # Fair payout (decimal odds)
        fair_payout = 1.0 / combined_prob if combined_prob > 0 else 999
        
        # Implied market odds (assuming -115 per leg)
        market_prob_per_leg = 0.535  # -115 American = 53.5% implied
        market_combined = market_prob_per_leg ** num_legs
        
        # EV estimate
        ev = (combined_prob * fair_payout) - 1.0
        
        avg_score = sum(p["score"] for p in combo) / num_legs
        min_score_combo = min(p["score"] for p in combo)
        
        parlay = {
            "players": [p["name"] for p in combo],
            "teams": teams,
            "games": list(set(games)),
            "num_legs": num_legs,
            "combined_prob": round(combined_prob * 100, 1),
            "combined_prob_raw": round(combined_raw * 100, 1),
            "fair_payout": round(fair_payout, 2),
            "ev": round(ev * 100, 1),
            "avg_score": round(avg_score, 1),
            "min_score": round(min_score_combo, 1),
            "corr_factor": round(corr_factor, 3),
            "combo": combo,
            "notes": "SGP ⭐" if len(set(games)) == 1 else f"{len(set(games))} games",
        }
        best_parlays.append(parlay)
    
    # Sort by combined probability (best model confidence first)
    best_parlays.sort(key=lambda x: x["combined_prob"], reverse=True)
    
    return best_parlays[:10]

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_leaderboard(plays: List[Dict]):
    """Display the full ranked leaderboard with tier color coding."""
    
    if not plays:
        st.info("No scored plays available. Run the model first.")
        return

    # Proxy mode indicator
    _bat_src_ui = st.session_state.get("_batting_source", "")
    _is_proxy_ui = "mlbapi" in _bat_src_ui or _bat_src_ui in ("mlbapi_only",)
    if _is_proxy_ui:
        st.warning("⚠️ **Proxy Data Mode** — Savant unavailable. Using MLB Stats API derived signals. "
                   "Tier thresholds adjusted −5 pts: Tier 1 ≥75 · Tier 2 ≥65 · Tier 3 ≥55. "
                   "Scores run ~5-8 pts lower than full-Savant mode.")
    elif "disk_cache_fresh" in _bat_src_ui:
        st.info("💾 Serving from fresh disk cache — full Savant column set, normal thresholds.")

    # BvP "owns" alert — surface elite matchups prominently regardless of tier
    bvp_owns = [p for p in plays if p.get("bvp_sig") == "owns"]
    if bvp_owns:
        owns_names = " · ".join(
            f"{p['name']} ({p['team']}) {p.get('bvp_label','')}" for p in bvp_owns[:3]
        )
        st.error(f"🔥 **OWNS MATCHUP** — elite career dominance vs today's SP: {owns_names}")

    bvp_fades = [p for p in plays if p.get("bvp_sig") in ("fade", "dominated")]
    if bvp_fades:
        fade_names = " · ".join(f"{p['name']} ({p['team']})" for p in bvp_fades[:3])
        st.warning(f"⚠️ **BvP Fades** — career struggles vs today's SP: {fade_names}")

    # Summary metrics
    tier1 = [p for p in plays if p["tier"] == "🔒 TIER 1"]
    tier2 = [p for p in plays if p["tier"] == "✅ TIER 2"]
    tier3 = [p for p in plays if p["tier"] == "📊 TIER 3"]
    no_play = [p for p in plays if p["tier"] == "❌ NO PLAY"]
    
    # Tier summary banner
    if tier1:
        st.success(f"🔒 {len(tier1)} TIER 1 PLAYS — Parlay anchors. Don't sleep on these.")
    elif tier2:
        st.info(f"✅ {len(tier2)} TIER 2 PLAYS — Solid value, build parlays around these.")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("🔒 Tier 1", len(tier1))
    with col2: st.metric("✅ Tier 2", len(tier2))
    with col3: st.metric("📊 Tier 3", len(tier3))
    with col4: st.metric("❌ No Play", len(no_play))
    with col5: st.metric("Total Batters", len(plays))
    
    st.markdown("---")
    
    # Filters
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        tier_filter = st.multiselect("Filter by Tier", 
            ["🔒 TIER 1", "✅ TIER 2", "📊 TIER 3", "❌ NO PLAY"],
            default=["🔒 TIER 1", "✅ TIER 2", "📊 TIER 3"])
    with col_f2:
        teams = sorted(list(set(p["team"] for p in plays)))
        team_filter = st.multiselect("Filter by Team", teams, default=[])
    with col_f3:
        min_score_filter = st.slider("Min Score", 0, 100, 50)
    with col_f4:
        hand_filter = st.multiselect("Batter Hand", ["L", "R", "B"], default=[])
    
    # Apply filters
    filtered = plays
    if tier_filter:
        filtered = [p for p in filtered if p["tier"] in tier_filter]
    if team_filter:
        filtered = [p for p in filtered if p["team"] in team_filter]
    if min_score_filter > 0:
        filtered = [p for p in filtered if p["score"] >= min_score_filter]
    if hand_filter:
        filtered = [p for p in filtered if p["batter_hand"] in hand_filter]
    
    st.markdown(f"**Showing {len(filtered)} batters**")
    
    # Build display dataframe
    rows = []
    for p in filtered:
        tier_emoji = p["tier"].split()[0]
        
        # Color-coded score
        score_display = f"{p['score']:.0f}"
        
        wind_icon = ""
        if p.get("wind_effect") == "strong_out":
            wind_icon = "💨⬆️"
        elif p.get("wind_effect") == "out":
            wind_icon = "💨"
        elif p.get("wind_effect") == "in":
            wind_icon = "💨⬇️"
        elif p.get("is_dome"):
            wind_icon = "🏟️"
        
        tbd_flag = " ⚠️TBD" if p.get("sp_tbd") else ""
        
        rows.append({
            "Score": score_display,
            "Tier": p["tier"],
            "Player": p["name"],
            "Team": p["team"],
            "Vs": p["opponent"],
            "Slot": f"#{p['lineup_slot']}",
            "Hand": p["batter_hand"],
            "Opp SP": p["sp_name"][:20] + tbd_flag,
            "SP 🤚": p["sp_hand"],
            "Prob": f"{p['prob']*100:.0f}%",
            "Edge": f"{p.get('market_edge', 0):+.0f}%" if p.get('implied_total', 0) > 0 else "—",
            "xSLG": f"{p['xslg']:.3f}" if p["xslg"] else "—",
            "Barrel%": f"{p['barrel_rate']*100:.1f}%" if p["barrel_rate"] else "—",
            "HH%": f"{p['hard_hit_rate']*100:.1f}%" if p["hard_hit_rate"] else "—",
            "K%": f"{p['k_rate']*100:.1f}%" if p["k_rate"] else "—",
            "Platoon": p["platoon_label"].split("(")[0].strip(),
            "Form 🔥": p.get("streak_label", "—").replace("Form: ", ""),
            "BvP★": "🔥 OWNS" if p.get("bvp_sig") == "owns"
                    else "🟢 Edge" if p.get("bvp_sig") == "edge"
                    else "🔴 Fade" if p.get("bvp_sig") == "fade"
                    else "⚠️ Dom'd" if p.get("bvp_sig") == "dominated"
                    else "—",
            "BvP Stats": p.get("bvp_label", "—").replace("BvP: ", ""),
            "Park": p["park"],
            f"Wind{wind_icon}": p["weather_label"].split("|")[0].strip() if "|" in p["weather_label"] else p["weather_label"],
            "°F": f"{p['temperature']:.0f}°",
            "Imp.Runs": f"{p['implied_total']:.1f}" if p.get('implied_total', 0) > 0 else "—",
            "HR Score": f"{p['hr_score']:.0f}",
        })
    
    if rows:
        df = pd.DataFrame(rows)
        
        # Color code by tier
        def color_tier(val):
            if "TIER 1" in str(val) or "🔒" in str(val):
                return "color: #00ff88; font-weight: bold"
            elif "TIER 2" in str(val) or "✅" in str(val):
                return "color: #ffdd00; font-weight: bold"
            elif "TIER 3" in str(val) or "📊" in str(val):
                return "color: #ff8800; font-weight: bold"
            return "color: #888888"
        
        def color_score(val):
            try:
                v = float(str(val))
                if v >= 80: return "color: #00ff88; font-weight: bold"
                elif v >= 70: return "color: #ffdd00; font-weight: bold"
                elif v >= 60: return "color: #ff8800"
                return "color: #888888"
            except:
                return ""

        def color_edge(val):
            try:
                v = float(str(val).replace("%","").replace("+",""))
                if v >= 10: return "color: #00ff88; font-weight: bold"   # strong positive = green
                elif v >= 5: return "color: #66dd88; font-weight: bold"  # moderate positive = light green
                elif v >= 0: return "color: #ffdd00"                     # thin edge = yellow
                return "color: #ff4444"                                  # negative edge = red
            except:
                return ""

        def color_form(val):
            v = str(val)
            if "🔥" in v or "Hot" in v:   return "color: #ff8800; font-weight: bold"
            if "❄️" in v or "Cold" in v:  return "color: #88aaff"
            return ""

        def color_bvp(val):
            v = str(val)
            if "🔥" in v or "OWNS" in v:  return "color: #ff6600; font-weight: bold; font-size: 1.05em"
            if "🟢" in v or "Edge" in v:  return "color: #00ff88; font-weight: bold"
            if "🔴" in v or "Fade" in v:  return "color: #ff4444; font-weight: bold"
            if "⚠️" in v or "Dom" in v:   return "color: #ffaa00; font-weight: bold"
            return ""

        styled = df.style.map(color_tier, subset=["Tier"]).map(color_score, subset=["Score"])
        if "Edge" in df.columns:
            styled = styled.map(color_edge, subset=["Edge"])
        if "Form 🔥" in df.columns:
            styled = styled.map(color_form, subset=["Form 🔥"])
        if "BvP Stats" in df.columns:
            styled = styled.map(color_bvp, subset=["BvP Stats"])
        if "BvP★" in df.columns:
            styled = styled.map(color_bvp, subset=["BvP★"])
        st.dataframe(styled, use_container_width=True, height=500)
        
        # Export button
        csv = df.to_csv(index=False)
        st.download_button("📥 Export CSV", csv, f"mlb_tb_picks_{datetime.now(EST).strftime('%Y%m%d')}.csv", "text/csv")
    
    # Expandable detail cards for top plays
    st.markdown("---")
    st.subheader("🏆 Top Plays — Full Breakdown")
    
    top5 = [p for p in filtered if p["score"] >= 60][:5]
    for i, p in enumerate(top5, 1):
        tier_color = "#00ff88" if p["tier"] == "🔒 TIER 1" else "#ffdd00" if p["tier"] == "✅ TIER 2" else "#ff8800"
        
        with st.expander(f"{p['tier']} #{i}: {p['name']} ({p['team']}) — Score: {p['score']:.0f} | Prob: {p['prob']*100:.0f}%"):
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown("**🏏 Batter Profile**")
                st.write(f"• xSLG: {p['xslg']:.3f}" if p["xslg"] else "• xSLG: Limited data")
                st.write(f"• Barrel%: {p['barrel_rate']*100:.1f}%" if p["barrel_rate"] else "• Barrel%: —")
                st.write(f"• HardHit%: {p['hard_hit_rate']*100:.1f}%" if p["hard_hit_rate"] else "• HardHit%: —")
                st.write(f"• K%: {p['k_rate']*100:.1f}%" if p["k_rate"] else "• K%: —")
                st.write(f"• Exit Velo: {p['exit_velocity']:.1f} mph" if p["exit_velocity"] else "• Exit Velo: —")
                ev50 = p.get("ev50", 0)
                st.write(f"• EV50: {ev50:.1f} mph" if ev50 and ev50 > 50 else "• EV50: —")
                bs = p.get("bat_speed", 0)
                st.write(f"• Bat Speed: {bs:.1f} mph" if bs and bs > 30 else "• Bat Speed: —")
                # V1.7: Recent form + BvP
                streak_lbl = p.get("streak_label", "")
                if streak_lbl and streak_lbl not in ("Form: no data", "Form: no baseline"):
                    st.write(f"• {streak_lbl}")
                bvp_lbl = p.get("bvp_label", "")
                bvp_sig = p.get("bvp_sig", "no_data")
                if bvp_lbl and bvp_sig != "no_data":
                    st.write(f"• {bvp_lbl}")
                br = p.get("blast_rate", 0)
                st.write(f"• Blast%: {br*100:.1f}%" if br and br > 0 else "• Blast%: —")
                st.write(f"• Lineup: {p['lineup_label']}")
                st.write(f"• Platoon: {p['platoon_label']}")
            
            with col_b:
                st.markdown("**⚾ Pitcher Matchup**")
                tbd_note = " ⚠️ TBD — score capped at 72" if p["sp_tbd"] else ""
                st.write(f"• SP: {p['sp_name']}{tbd_note}")
                st.write(f"• SP Hand: {p['sp_hand']}")
                st.write(f"• {p['pitcher_label']}")
                st.markdown("**🌤️ Environment**")
                st.write(f"• {p['park_label']}")
                st.write(f"• {p['weather_label']}")
                st.write(f"• Implied Runs: {p['implied_total']:.1f}")
            
            with col_c:
                st.markdown("**📊 Score Breakdown**")
                sub_labels = {
                    "⚾ Pitcher Vuln (30%)": p["sub_pitcher"],
                    "🏏 Batter (28%)": p["sub_batter"],
                    "🤚 Platoon (12%)": p["sub_platoon"],
                    "💰 Vegas (8%)": p["sub_vegas"],
                    "🏟️ Park (7%)": p["sub_park"],
                    "📈 Streak (5%)": p.get("sub_streak", 50),
                    "🔄 TTO (4%)": p.get("sub_tto", 50),
                    "🌤️ Weather (4%)": p["sub_weather"],
                    "🎯 Pitch Mix (2%)": p.get("sub_matchup", 50),
                    "📊 BvP (2%)": p.get("sub_bvp", 50),
                    "📋 Lineup (1%)": p["sub_lineup"],
                }
                matchup_lbl = p.get("matchup_label", "")
                if matchup_lbl and matchup_lbl != "Pitch mix: avg splits":
                    st.caption(f"🎯 {matchup_lbl}")
                for label, val in sub_labels.items():
                    bar_color = "#00ff88" if val >= 70 else "#ffdd00" if val >= 50 else "#ff4444"
                    bar_width = int(val)
                    st.markdown(f"{label}: **{val:.0f}**")
                    st.markdown(f'<div style="background:#333;border-radius:4px;height:6px;width:100%"><div style="background:{bar_color};width:{bar_width}%;height:6px;border-radius:4px"></div></div>', unsafe_allow_html=True)
                
                st.markdown(f"**🎯 Final Score: {p['score']:.0f} ({p['prob']*100:.0f}%)**")
                st.markdown(f"**💣 HR Score: {p['hr_score']:.0f}**")


def display_tiered_breakdown(plays: List[Dict]):
    """Display tiered breakdown with expandable tier sections."""
    
    tier_groups = {
        "🔒 TIER 1": [p for p in plays if p["tier"] == "🔒 TIER 1"],
        "✅ TIER 2": [p for p in plays if p["tier"] == "✅ TIER 2"],
        "📊 TIER 3": [p for p in plays if p["tier"] == "📊 TIER 3"],
        "❌ NO PLAY": [p for p in plays if p["tier"] == "❌ NO PLAY"],
    }
    
    st.header("🎯 Tiered Breakdown")
    
    tier_descriptions = {
        "🔒 TIER 1": ("Strong plays. Parlay anchors. All edge indicators firing.", "#00ff88"),
        "✅ TIER 2": ("Viable single plays or parlay fillers. Good but not elite.", "#ffdd00"),
        "📊 TIER 3": ("Marginal. Single game only. Use with caution.", "#ff8800"),
        "❌ NO PLAY": ("Below threshold. Fade.", "#666666"),
    }
    
    for tier_name, tier_plays in tier_groups.items():
        if not tier_plays and tier_name == "❌ NO PLAY":
            continue
        
        color = tier_descriptions[tier_name][1]
        desc = tier_descriptions[tier_name][0]
        
        with st.expander(f"{tier_name} ({len(tier_plays)} plays)", expanded=(tier_name in ["🔒 TIER 1", "✅ TIER 2"])):
            if not tier_plays:
                st.caption(f"No {tier_name} plays today.")
                continue
            
            st.caption(desc)
            
            for p in tier_plays:
                col1, col2, col3 = st.columns([3, 3, 2])
                
                with col1:
                    tbd_flag = " ⚠️" if p.get("sp_tbd") else ""
                    st.markdown(f"**{p['name']}** ({p['team']}) vs {p['opponent']}")
                    st.caption(f"Score: {p['score']:.0f} | Prob: {p['prob']*100:.0f}% | Lineup: #{p['lineup_slot']} | {p['batter_hand']}HB vs {p['sp_hand']}HP")
                    st.caption(f"SP: {p['sp_name']}{tbd_flag}")
                
                with col2:
                    st.markdown("**Edges:**")
                    edges = []
                    risks = []
                    
                    if p["sub_batter"] >= 65:
                        edges.append(f"✓ Strong contact quality ({p['sub_batter']:.0f}/100)")
                    if p["sub_pitcher"] >= 65:
                        edges.append(f"✓ Vulnerable SP ({p['sub_pitcher']:.0f}/100)")
                    if p["sub_platoon"] >= 65:
                        edges.append(f"✓ Platoon edge: {p['platoon_label'].split('(')[0].strip()}")
                    if p["lineup_slot"] <= 4:
                        edges.append(f"✓ Top-order slot #{p['lineup_slot']} (more PA)")
                    if p.get("wind_effect") in ["out", "strong_out"]:
                        edges.append(f"✓ Wind out ({p['wind_speed']} mph)")
                    if p["implied_total"] >= 5.0:
                        edges.append(f"✓ High implied runs ({p['implied_total']:.1f})")
                    
                    for e in edges[:3]:
                        st.caption(e)
                
                with col3:
                    st.markdown("**Risks:**")
                    if p.get("sp_tbd"):
                        st.caption("✗ ⚠️ Pitcher TBD")
                    if p["k_rate"] and p["k_rate"] > 0.28:
                        st.caption(f"✗ High K% ({p['k_rate']*100:.0f}%)")
                    if p.get("wind_effect") == "in":
                        st.caption(f"✗ Wind in ({p['wind_speed']} mph)")
                    if p["temperature"] and p["temperature"] < 50:
                        st.caption(f"✗ Cold ({p['temperature']:.0f}°F)")
                    if p["lineup_slot"] >= 7:
                        st.caption(f"✗ Low lineup slot #{p['lineup_slot']}")
                    if p["implied_total"] and p["implied_total"] < 3.5:
                        st.caption(f"✗ Low implied runs ({p['implied_total']:.1f})")
                
                st.markdown("---")


def display_parlay_builder(plays: List[Dict], unit_size: int = 25):
    """Display parlay builder — O1.5 only, O0.5 only, and mixed."""

    st.header("💰 Parlay Builder")

    parlay_mode = st.radio(
        "Parlay type:",
        ["🎯 O1.5 Only", "🎯 O0.5 Only", "🔀 Mixed (O0.5 + O1.5)"],
        horizontal=True, index=0
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        num_legs = st.radio("Legs", [2, 3, 4, 5], index=1, horizontal=True)
    with col2:
        min_score = st.slider("Min score for parlay", 60, 90, 70)
    with col3:
        max_same_team = st.number_input("Max same-team legs", 1, 3, 1)

    if parlay_mode == "🎯 O1.5 Only":
        eligible = [p for p in plays if p["score"] >= min_score and not p.get("sp_tbd")]
        st.info(f"📊 {len(eligible)} eligible O1.5 players (score ≥ {min_score})")
        pool = plays

    elif parlay_mode == "🎯 O0.5 Only":
        # Re-score as hits model
        hits_pool = []
        for p in plays:
            pitcher_mock = {"k_rate_allowed": 0.220, "era": 4.20, "fip": 4.20, "whip": 1.30, "hard_hit_allowed": 0.340}
            try:
                pit_label = p.get("pitcher_label", "")
                if "K%:" in pit_label:
                    pitcher_mock["k_rate_allowed"] = float(pit_label.split("K%:")[1].split("%")[0].strip()) / 100
                if "FIP:" in pit_label:
                    pitcher_mock["fip"] = float(pit_label.split("FIP:")[1].strip().split()[0])
            except Exception:
                pass
            batter_mock = {
                "k_rate": p.get("k_rate", 0.228), "wrc_plus": max(40,min(220,100+(p.get("xslg",0.398)-0.398)/0.005)), "avg": 0.255,
                "bb_rate": p.get("bb_rate",0.082), "woba": p.get("xslg", 0.398) * 0.78,
                "hard_hit_rate": p.get("hard_hit_rate", 0.370), "slg_proxy": p.get("xslg", 0.398),
            }
            h_score, h_prob, h_tier, _ = compute_hits_score_for_player(
                batter_mock, pitcher_mock, p.get("batter_hand","R"), p.get("sp_hand","R"),
                p.get("lineup_slot",5), p.get("park",p.get("team","")),
                p.get("weather",{}), p.get("implied_total",0),
                p.get("sp_tbd",False), p.get("lineup_confirmed",True)
            )
            hits_pool.append({**p, "score": h_score, "prob": h_prob, "tier": h_tier, "prop": "O0.5"})

        eligible = [p for p in hits_pool if p["score"] >= min_score]
        st.info(f"📊 {len(eligible)} eligible O0.5 players (h-score ≥ {min_score})")
        pool = hits_pool

    else:  # Mixed
        # Build mixed pool: O1.5 eligible + O0.5 eligible
        hits_pool = []
        for p in plays:
            pitcher_mock = {"k_rate_allowed": 0.220, "era": 4.20, "fip": 4.20, "whip": 1.30, "hard_hit_allowed": 0.340}
            try:
                pit_label = p.get("pitcher_label", "")
                if "K%:" in pit_label:
                    pitcher_mock["k_rate_allowed"] = float(pit_label.split("K%:")[1].split("%")[0].strip()) / 100
                if "FIP:" in pit_label:
                    pitcher_mock["fip"] = float(pit_label.split("FIP:")[1].strip().split()[0])
            except Exception:
                pass
            batter_mock = {
                "k_rate": p.get("k_rate", 0.228), "wrc_plus": max(40,min(220,100+(p.get("xslg",0.398)-0.398)/0.005)), "avg": 0.255,
                "bb_rate": p.get("bb_rate",0.082), "woba": p.get("xslg", 0.398) * 0.78,
                "hard_hit_rate": p.get("hard_hit_rate", 0.370), "slg_proxy": p.get("xslg", 0.398),
            }
            h_score, h_prob, h_tier, _ = compute_hits_score_for_player(
                batter_mock, pitcher_mock, p.get("batter_hand","R"), p.get("sp_hand","R"),
                p.get("lineup_slot",5), p.get("park",p.get("team","")),
                p.get("weather",{}), p.get("implied_total",0),
                p.get("sp_tbd",False), p.get("lineup_confirmed",True)
            )
            hits_pool.append({**p, "h_score": h_score, "h_prob": h_prob})

        # Pick best market for each player
        mixed_pool = []
        for orig, hits in zip(plays, hits_pool):
            o15_score = orig["score"]
            o05_score = hits["h_score"]
            # Use whichever market is stronger
            if o15_score >= min_score and o05_score >= min_score:
                # Both qualify — pick higher score
                if o15_score >= o05_score:
                    mixed_pool.append({**orig, "prop": "O1.5", "score": o15_score, "prob": orig["prob"]})
                else:
                    mixed_pool.append({**orig, "prop": "O0.5", "score": o05_score, "prob": hits["h_prob"]})
            elif o15_score >= min_score:
                mixed_pool.append({**orig, "prop": "O1.5"})
            elif o05_score >= min_score:
                mixed_pool.append({**orig, "prop": "O0.5", "score": o05_score, "prob": hits["h_prob"]})

        eligible = mixed_pool
        pool = mixed_pool
        st.info(f"📊 {len(eligible)} mixed eligible ({sum(1 for p in eligible if p.get('prop')=='O1.5')} O1.5 + {sum(1 for p in eligible if p.get('prop')=='O0.5')} O0.5)")

    if len(eligible) < num_legs:
        st.warning(f"⚠️ Not enough eligible players for {num_legs} legs (have {len(eligible)}). Lower min score.")
        return

    parlays = build_parlays(pool, num_legs, max_same_team, min_score)

    if not parlays:
        st.warning("No valid parlays found with current filters.")
        return

    best = parlays[0]
    st.subheader("⭐ Recommended Parlay")

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a: st.metric("Combined Prob", f"{best['combined_prob']:.1f}%")
    with col_b: st.metric("Fair Payout", f"{best['fair_payout']:.2f}x")
    with col_c: st.metric("Est. EV", f"{best['ev']:+.1f}%")
    with col_d: st.metric("Avg Score", f"{best['avg_score']:.0f}")

    hrb_legs = []
    for player_name in best["players"]:
        player = next((p for p in pool if p["name"] == player_name), None)
        if player:
            prop = player.get("prop", "O1.5")
            prop_str = "Over 1.5 TB" if prop == "O1.5" else "Over 0.5 TB"
            tag = "🎯" if prop == "O0.5" else "⚾"
            st.markdown(f"{tag} **{player_name}** ({player['team']}) vs {player['opponent']} — {prop_str} | Score: {player['score']:.0f} | Prob: {player['prob']*100:.0f}%")
            hrb_legs.append(f"{player_name} {prop_str}")

    if best.get("corr_factor", 1.0) < 0.98:
        st.caption(f"🔗 Correlation discount: {best['corr_factor']:.3f}x")

    hrb_text = " + ".join(hrb_legs)
    st.code(f"HardRock Bet: {hrb_text}", language=None)

    st.markdown("---")
    st.subheader("📊 All Parlay Options")
    rows = []
    for par in parlays:
        rows.append({
            "Players": " + ".join(par["players"]),
            "Legs": par["num_legs"],
            "Avg Score": f"{par['avg_score']:.0f}",
            "Min Score": f"{par['min_score']:.0f}",
            "Combined %": f"{par['combined_prob']:.1f}%",
            "Fair Payout": f"{par['fair_payout']:.2f}x",
            "EV%": f"{par['ev']:+.1f}%",
            "Notes": par["notes"],
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # SGP
    sgp_plays = [p for p in parlays if "SGP" in p.get("notes", "")]
    if sgp_plays:
        st.subheader("⭐ Same-Game Parlay Opportunities")
        for sgp in sgp_plays[:3]:
            players_info = [next((p for p in pool if p["name"] == n), None) for n in sgp["players"]]
            valid = [p for p in players_info if p]
            if valid:
                game = valid[0]["opponent"] + " @ " + valid[0]["team"]
                st.success(f"🎰 **SGP:** {game} | {' + '.join(sgp['players'])} | Combined: {sgp['combined_prob']:.1f}%")

    # Custom builder
    st.markdown("---")
    st.subheader("🔧 Custom Parlay Builder")
    eligible_names = [f"{p['name']} ({p['team']}, {p.get('prop','O1.5')}, {p['score']:.0f})" for p in eligible]
    selected = st.multiselect("Select legs manually:", eligible_names, key="custom_parlay_select")

    if len(selected) >= 2:
        selected_plays = []
        for sel in selected:
            name_part = sel.split(" (")[0]
            player = next((p for p in pool if p["name"] == name_part), None)
            if player:
                selected_plays.append(player)

        if len(selected_plays) >= 2:
            combined = 1.0
            for p in selected_plays:
                combined *= p["prob"]
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Combined Prob", f"{combined*100:.1f}%")
            with col2: st.metric("Fair Payout", f"{1/combined:.2f}x")
            with col3: st.metric("Legs", len(selected_plays))
            hrb_custom = " + ".join([f"{p['name']} {'Over 1.5 TB' if p.get('prop','O1.5')=='O1.5' else 'Over 0.5 TB'}" for p in selected_plays])
            st.code(f"HardRock Bet: {hrb_custom}", language=None)


def display_hr_plays(plays: List[Dict]):
    """Display top HR upside plays."""
    
    st.header("💣 Home Run Plays")
    st.caption("Top 10 daily HR candidates. Powered by barrel rate, hard hit%, exit velocity, ISO, park factor, wind, and implied total.")

    hr_sorted = sorted(plays, key=lambda x: x["hr_score"], reverse=True)[:10]

    rows = []
    for p in hr_sorted:
        wind_label = p.get("weather_label", "").split("|")[0].strip()
        park_name = STADIUM_COORDS.get(p["park"], (0, 0, p["park"], False))[2]
        sweet = p.get("sweet_spot_rate", 0)

        rows.append({
            "HR Score": f"{p['hr_score']:.0f}",
            "Player": p["name"],
            "Team": p["team"],
            "Opp SP": p["sp_name"][:20],
            "Barrel%": f"{p['barrel_rate']*100:.1f}%" if p["barrel_rate"] else "—",
            "HH%": f"{p['hard_hit_rate']*100:.1f}%" if p.get("hard_hit_rate") else "—",
            "EV": f"{p.get('exit_velocity', 0):.1f}" if p.get("exit_velocity", 0) > 0 else "—",
            "ISO": f"{p.get('iso', 0):.3f}" if p.get("iso", 0) > 0 else "—",
            "Sweet Spot%": f"{sweet*100:.1f}%" if sweet and sweet != 0.305 else "—",
            "Park HR Factor": f"{PARK_HR_FACTORS.get(p['park'], 1.0):.2f}x",
            "Park": park_name[:20],
            "Wind": p.get("wind_dir", "") + f" {p.get('wind_speed', 0):.0f}mph",
            "Wind Effect": "🔥 Out" if p.get("wind_effect") == "strong_out" else "💨 Out" if p.get("wind_effect") == "out" else "❄️ In" if p.get("wind_effect") == "in" else "🏟️" if p.get("is_dome") else "—",
            "Imp. Runs": f"{p['implied_total']:.1f}",
            "TB Score": f"{p['score']:.0f}",
        })
    
    if rows:
        df = pd.DataFrame(rows)
        
        def color_hr(val):
            try:
                v = float(val)
                if v >= 75: return "color: #ff4444; font-weight: bold"
                elif v >= 60: return "color: #ff8800; font-weight: bold"
                return ""
            except:
                return ""
        
        styled = df.style.map(color_hr, subset=["HR Score"])
        st.dataframe(styled, use_container_width=True)
    
    # Top 3 HR plays detailed
    st.markdown("---")
    st.subheader("🔥 Top 3 HR Plays — Detail")
    
    for i, p in enumerate(hr_sorted[:3], 1):
        with st.expander(f"#{i}: {p['name']} ({p['team']}) — HR Score: {p['hr_score']:.0f}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Barrel%:** {p['barrel_rate']*100:.1f}%" if p["barrel_rate"] else "**Barrel%:** —")
                st.write(f"**Hard Hit%:** {p['hard_hit_rate']*100:.1f}%" if p.get("hard_hit_rate") else "**Hard Hit%:** —")
                st.write(f"**Exit Velocity:** {p.get('exit_velocity', 0):.1f} mph" if p.get("exit_velocity", 0) > 0 else "**Exit Velocity:** —")
                st.write(f"**ISO:** {p.get('iso', 0):.3f}" if p.get("iso", 0) > 0 else "**ISO:** —")
                st.write(f"**Park:** {STADIUM_COORDS.get(p['park'], (0,0,p['park'],False))[2]}")
                st.write(f"**Park HR Factor:** {PARK_HR_FACTORS.get(p['park'], 1.0):.2f}x")
            with col2:
                st.write(f"**Wind:** {p.get('wind_dir', '')} @ {p.get('wind_speed', 0):.0f}mph ({p.get('wind_effect', 'neutral')})")
                st.write(f"**Temp:** {p['temperature']:.0f}°F")
                st.write(f"**Implied Runs:** {p['implied_total']:.1f}")
                st.write(f"**Total Bases Score:** {p['score']:.0f}")
            
            # SGP opportunity check
            if p["score"] >= 60:
                same_game = [op for op in hr_sorted if op["game_id"] == p["game_id"] and op["name"] != p["name"]]
                if same_game:
                    st.success(f"⭐ SGP Opportunity: {p['name']} HR + {same_game[0]['name']} O1.5 TB in same game!")


# ============================================================================
# K PROPS MODEL — Batter Strikeout Probability
# V1.9: Composite K score per batter vs opposing pitcher.
# Inputs: Batter K%, Pitcher K%/SwStr%, lineup slot, game context.
# O-Swing% and batter SwStr% degrade gracefully to league avg when unavailable
# (FanGraphs blocked on Streamlit Cloud; Savant CSV returns 502).
# ============================================================================

@st.cache_data(ttl=3600)
def fetch_umpire_data() -> Dict:
    """
    Fetch today's HP umpire assignment and their historical K-rate impact.
    Source: api.umpscorecards.com (free, no key required).
    Returns: {game_pk: {"ump_name": str, "k_rate_added": float, "run_value": float}}
    Falls back to {} on any failure — K model degrades gracefully.
    """
    result = {}
    try:
        # Step 1: Get today's umpire assignments from MLB Stats API
        # The schedule 'officials' hydration returns HP umpire per game
        today = datetime.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d")
        url = (f"https://statsapi.mlb.com/api/v1/schedule"
               f"?sportId=1&date={today}&hydrate=officials")
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return {}
        data = r.json()
        game_umps = {}
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                pk = game.get("gamePk")
                officials = game.get("officials", [])
                for off in officials:
                    if off.get("officialType", "") == "Home Plate":
                        name = off.get("official", {}).get("fullName", "")
                        if name and pk:
                            game_umps[pk] = name
                        break

        if not game_umps:
            return {}

        # Step 2: Fetch umpire K stats from ump-scorecard API
        ump_stats_url = "https://api.umpscorecards.com/v1/umpires/"
        ur = requests.get(ump_stats_url, timeout=10)
        if ur.status_code != 200:
            # Fallback: return ump names with zero adjustment
            for pk, name in game_umps.items():
                result[pk] = {"ump_name": name, "k_rate_added": 0.0, "run_value": 0.0}
            return result

        ump_data = ur.json()
        # Build lookup: normalized name → stats
        ump_lookup = {}
        for ump in (ump_data if isinstance(ump_data, list) else ump_data.get("data", [])):
            raw_name = str(ump.get("name", "") or ump.get("umpire", "")).strip()
            if raw_name:
                ump_lookup[_norm(raw_name)] = {
                    "k_rate_added": float(ump.get("k_rate_added", 0) or ump.get("strikeout_rate", 0) or 0),
                    "run_value":    float(ump.get("run_value_above_avg", 0) or 0),
                }

        for pk, name in game_umps.items():
            stats = ump_lookup.get(_norm(name), {"k_rate_added": 0.0, "run_value": 0.0})
            result[pk] = {"ump_name": name, **stats}

    except Exception:
        pass
    return result


def compute_k_score(batter_statcast: Dict, pitcher_statcast: Dict,
                    lineup_slot: int, implied_total: float,
                    ump_k_adj: float = 0.0) -> Tuple[float, str, Dict]:
    """
    Composite K score for a batter — 0 to 100, league avg = 50.
    HIGH score = batter more likely to strike out.

    Weight structure:
      - Batter K%:          40% (most stable batter K signal)
      - Pitcher K rate:     35% (K%, SwStr% when available)
      - Whiff/chase:        15% (batter SwStr% + O-Swing%, graceful fallback)
      - Lineup/context:     10% (slot PA count, game pace, ump zone)

    Tiers: K+ (80+), K (70-79), Lean K (60-69), No Play (<60)
    """
    details = {}

    def f(d, key, default):
        try: return float(d.get(key, default) or default)
        except: return float(default)

    # ── BATTER K RATE (40%) ─────────────────────────────────────────────────
    batter_k = f(batter_statcast, "k_rate", 0.228)
    # MLB K% range: ~10% (elite contact) to ~38% (high-K). Avg ~22.8%.
    # Z-score style: avg=50, ±1sd (6pp) = ±25 pts
    batter_k_score = 50.0 + (batter_k - 0.228) / 0.060 * 25.0
    batter_k_score = max(0.0, min(100.0, batter_k_score))
    details["batter_k_pct"] = round(batter_k * 100, 1)
    details["batter_k_score"] = round(batter_k_score, 1)

    # ── PITCHER K RATE (35%) ────────────────────────────────────────────────
    # Primary: pitcher K% from pitching_df (MLB Stats API K/TBF always available)
    pit_k = f(pitcher_statcast, "k_rate_allowed", 0.228)
    # SwStr% bonus when available (FanGraphs type=8, frequently blocked)
    swstr = f(pitcher_statcast, "swstr_pct", 0.0)  # 0.0 = unavailable sentinel
    if swstr > 0.02:
        # SwStr% range: ~5% (soft contact SP) to ~18% (elite strikeout pitcher). Avg ~11%.
        swstr_score = 50.0 + (swstr - 0.110) / 0.040 * 25.0
        swstr_score = max(0.0, min(100.0, swstr_score))
        # Blend: 70% K%, 30% SwStr% when both available
        pit_k_score = (50.0 + (pit_k - 0.228) / 0.060 * 25.0) * 0.70 + swstr_score * 0.30
        details["pitcher_swstr"] = round(swstr * 100, 1)
    else:
        pit_k_score = 50.0 + (pit_k - 0.228) / 0.060 * 25.0
        details["pitcher_swstr"] = "N/A"
    pit_k_score = max(0.0, min(100.0, pit_k_score))
    details["pitcher_k_pct"] = round(pit_k * 100, 1)
    details["pitcher_k_score"] = round(pit_k_score, 1)

    # ── WHIFF / CHASE (15%) ─────────────────────────────────────────────────
    # Batter O-Swing% and SwStr% from FanGraphs type=8 — blocked on Streamlit Cloud.
    # Savant custom JSON occasionally returns k% but not whiff breakdown.
    # Graceful fallback: use K% as proxy (already captured above) → neutral 50.
    batter_swstr = f(batter_statcast, "batter_swstr_pct", 0.0)
    o_swing      = f(batter_statcast, "o_swing_pct", 0.0)

    if batter_swstr > 0.02 and o_swing > 0.02:
        # Both available — full whiff signal
        swstr_b_sc = 50.0 + (batter_swstr - 0.110) / 0.035 * 25.0
        oswing_sc  = 50.0 + (o_swing - 0.310) / 0.060 * 25.0
        whiff_score = swstr_b_sc * 0.55 + oswing_sc * 0.45
        details["batter_swstr"] = round(batter_swstr * 100, 1)
        details["o_swing"] = round(o_swing * 100, 1)
    elif batter_swstr > 0.02:
        swstr_b_sc  = 50.0 + (batter_swstr - 0.110) / 0.035 * 25.0
        whiff_score = swstr_b_sc
        details["batter_swstr"] = round(batter_swstr * 100, 1)
        details["o_swing"] = "N/A"
    else:
        # Neither available — fall back to batter K% as proxy, centered
        whiff_score = 50.0 + (batter_k - 0.228) / 0.060 * 15.0
        whiff_score = max(0.0, min(100.0, whiff_score))
        details["batter_swstr"] = "N/A"
        details["o_swing"] = "N/A"
    whiff_score = max(0.0, min(100.0, whiff_score))
    details["whiff_score"] = round(whiff_score, 1)

    # ── LINEUP / CONTEXT (10%) ──────────────────────────────────────────────
    # PA opportunity: higher lineup slots get more PAs → more K chances.
    # But leadoff sees pitcher fresh → lower K rate first TTO.
    # Net: slots 1-5 get slight boost (more PAs + middle order sees TTO2+).
    slot_pa_score = {1: 55, 2: 58, 3: 60, 4: 62, 5: 63,
                     6: 52, 7: 48, 8: 45, 9: 40}.get(lineup_slot, 50)

    # Game pace: high-total games → pitcher works faster, less strikeout focus
    if implied_total >= 5.5:
        pace_adj = -5.0   # high-scoring game → SP exits earlier, less K focus
    elif implied_total >= 4.5:
        pace_adj = 0.0
    else:
        pace_adj = 5.0    # low-total pitcher's duel → K rates up

    # Umpire zone adjustment: k_rate_added is % point change in K rate
    # Convert to score points: 1pp K_rate_added ≈ +4 score points
    ump_adj = ump_k_adj * 400.0   # e.g. +0.02 K_rate_added → +8 pts
    ump_adj = max(-15.0, min(15.0, ump_adj))

    context_score = max(0.0, min(100.0, slot_pa_score + pace_adj + ump_adj))
    details["slot_pa_score"] = slot_pa_score
    details["ump_k_adj"] = round(ump_k_adj * 100, 2)
    details["context_score"] = round(context_score, 1)

    # ── COMPOSITE ───────────────────────────────────────────────────────────
    raw = (batter_k_score * 0.40 +
           pit_k_score    * 0.35 +
           whiff_score    * 0.15 +
           context_score  * 0.10)
    final = max(0.0, min(100.0, raw))

    tier = ("⚡ K+" if final >= 80 else
            "🔥 K"  if final >= 70 else
            "📊 Lean K" if final >= 60 else
            "➖ No Play")

    label = (f"Batter K%: {batter_k*100:.1f}% | "
             f"Pitcher K%: {pit_k*100:.1f}%"
             f"{' | SwStr%: ' + str(details['pitcher_swstr']) + '%' if details['pitcher_swstr'] != 'N/A' else ''}")
    details["tier"] = tier
    return round(final, 1), label, details


def display_k_props_tab(plays: List[Dict], ump_data: Dict):
    """
    Tab: ⚡ K Props
    Sortable table of all batters with K composite score, tier, and key inputs.
    No market line column unless Odds API strikeout props are loaded (not on free tier).
    """
    st.header("⚡ K Props — Batter Strikeout Model")
    st.caption("Composite K score per batter vs opposing pitcher. HIGH score = more likely to strikeout. "
               "Inputs: Batter K%, Pitcher K%/SwStr%, lineup slot, ump zone, game pace.")

    if not plays:
        st.info("Run the model first to see K prop scores.")
        return

    # Build K scores for all plays
    rows = []
    pitching_df = st.session_state.get("_pitching_df_global", pd.DataFrame())

    for p in plays:
        batter_statcast = {
            "k_rate":           p.get("k_rate", 0.228),
            "batter_swstr_pct": p.get("batter_swstr_pct", 0.0),
            "o_swing_pct":      p.get("o_swing_pct", 0.0),
        }
        # Re-use pitcher stats already loaded in run_model via pitching_df lookup
        pitcher_statcast = {
            "k_rate_allowed": p.get("_pitcher_k_rate", 0.228),
            "swstr_pct":      p.get("_pitcher_swstr", 0.0),
        }
        # Ump adjustment for this game
        game_pk_int = None
        try: game_pk_int = int(p.get("game_id", 0))
        except: pass
        ump_entry = ump_data.get(game_pk_int, {})
        ump_k_adj = float(ump_entry.get("k_rate_added", 0.0))
        ump_name  = ump_entry.get("ump_name", "—")

        k_score, k_label, k_details = compute_k_score(
            batter_statcast=batter_statcast,
            pitcher_statcast=pitcher_statcast,
            lineup_slot=p.get("lineup_slot", 5),
            implied_total=p.get("implied_total", 4.5),
            ump_k_adj=ump_k_adj,
        )

        rows.append({
            "Player":       p["name"],
            "Team":         p["team"],
            "#": p.get("lineup_slot", "—"),
            "Opp Pitcher":  p.get("sp_name", "TBD"),
            "Hand":         p.get("sp_hand", "R"),
            "Batter K%":    f"{p.get('k_rate', 0)*100:.1f}%",
            "Pit K%":       f"{k_details.get('pitcher_k_pct', 0):.1f}%",
            "SwStr%":       f"{k_details['pitcher_swstr']}%" if k_details['pitcher_swstr'] != 'N/A' else "N/A",
            "Batter SwStr": f"{k_details['batter_swstr']}%" if k_details['batter_swstr'] != 'N/A' else "N/A",
            "Umpire":       ump_name,
            "Ump K Adj":    f"{ump_k_adj*100:+.1f}pp" if ump_k_adj != 0 else "—",
            "K Score":      k_score,
            "Tier":         k_details.get("tier", "—"),
            "Market Line":  "N/A — no free-tier prop data",
        })

    if not rows:
        st.warning("No K scores computed.")
        return

    df = pd.DataFrame(rows)

    # Color K Score column
    def color_k(val):
        try:
            v = float(val)
            if v >= 80: return "color: #ff4444; font-weight: bold"
            elif v >= 70: return "color: #ff8800; font-weight: bold"
            elif v >= 60: return "color: #ffdd00"
            return "color: #888888"
        except: return ""

    tier_order = {"⚡ K+": 0, "🔥 K": 1, "📊 Lean K": 2, "➖ No Play": 3}
    df = df.sort_values("K Score", ascending=False)
    styled = df.style.map(color_k, subset=["K Score"])
    st.dataframe(styled, use_container_width=True)

    # Export
    csv = df.to_csv(index=False)
    st.download_button("📥 Export K Props CSV", csv, "k_props.csv", "text/csv", key="dl_kprops")

    st.markdown("---")
    st.subheader("⚡ Top K+ Plays — Detail")

    k_plus = [r for r in rows if r["K Score"] >= 70]
    if not k_plus:
        st.info("No K+ or K tier plays today. Check back when lineups are confirmed.")
        return

    for i, r in enumerate(k_plus[:5], 1):
        with st.expander(f"#{i}: {r['Player']} ({r['Team']}) — K Score: {r['K Score']} | {r['Tier']}"):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Batter K%:** {r['Batter K%']}")
                st.write(f"**Batter SwStr%:** {r['Batter SwStr']}")
                st.write(f"**O-Swing%:** {r.get('O-Swing%', 'N/A')}")
                st.write(f"**Lineup Slot:** #{r['#']}")
            with c2:
                st.write(f"**Pitcher K%:** {r['Pit K%']}")
                st.write(f"**Pitcher SwStr%:** {r['SwStr%']}")
                st.write(f"**Umpire:** {r['Umpire']} ({r['Ump K Adj']})")
                st.write(f"**Market Line:** {r['Market Line']}")

    # Results Tracker note
    st.markdown("---")
    st.caption("💾 K prop picks are logged to the Results Tracker (Tab 10) when you save picks.")


# ============================================================================
# MONEYLINE MODEL — Team Win Probability
# V1.9: Log5-style win probability calibrated against Vegas implied probability.
# Identifies edges where model diverges from market by >4%.
# ============================================================================

@st.cache_data(ttl=3600)
def fetch_team_run_differential(date_str: str, days: int = 7) -> Dict[str, float]:
    """
    Fetch last N days of team run differential from MLB Stats API game logs.
    Returns: {team_abbr: avg_run_diff_per_game}
    Uses only MLB Stats API (always unblocked).
    """
    result = {}
    try:
        from datetime import datetime as _dt, timedelta as _td
        end_dt   = _dt.strptime(date_str, "%Y-%m-%d")
        start_dt = end_dt - _td(days=days)
        start_str = start_dt.strftime("%Y-%m-%d")

        # Fetch schedule for the window
        url = (f"https://statsapi.mlb.com/api/v1/schedule"
               f"?sportId=1&startDate={start_str}&endDate={date_str}"
               f"&hydrate=linescore,team")
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return {}

        team_runs_for   = {}
        team_runs_against = {}
        team_games       = {}

        for date_entry in r.json().get("dates", []):
            for game in date_entry.get("games", []):
                if game.get("status", {}).get("abstractGameState") != "Final":
                    continue
                ls = game.get("linescore", {})
                home_runs = int(ls.get("teams", {}).get("home", {}).get("runs", 0) or 0)
                away_runs = int(ls.get("teams", {}).get("away", {}).get("runs", 0) or 0)
                home_abb  = game["teams"]["home"]["team"].get("abbreviation", "")
                away_abb  = game["teams"]["away"]["team"].get("abbreviation", "")
                # Normalize via TEAM_ABB_MAP
                home_abb = TEAM_ABB_MAP.get(game["teams"]["home"]["team"].get("name",""), home_abb)
                away_abb = TEAM_ABB_MAP.get(game["teams"]["away"]["team"].get("name",""), away_abb)

                for abb, rf, ra in [(home_abb, home_runs, away_runs),
                                    (away_abb, away_runs, home_runs)]:
                    if abb:
                        team_runs_for[abb]    = team_runs_for.get(abb, 0) + rf
                        team_runs_against[abb] = team_runs_against.get(abb, 0) + ra
                        team_games[abb]        = team_games.get(abb, 0) + 1

        for abb, g in team_games.items():
            if g > 0:
                result[abb] = round((team_runs_for[abb] - team_runs_against[abb]) / g, 2)

    except Exception:
        pass
    return result


def _american_to_implied(odds: float) -> float:
    """Convert American moneyline odds to implied probability (no vig removal)."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    else:
        return 100.0 / (odds + 100.0)


def fetch_moneyline_odds(date_str: str) -> Dict[int, Dict]:
    """
    Extract h2h moneyline odds from Odds API response (already fetched in run_model).
    Returns: {game_pk: {"home_odds": float, "away_odds": float,
                        "home_implied": float, "away_implied": float}}
    Re-uses fetch_odds infrastructure. Called once per run.
    """
    result = {}
    try:
        api_key = st.secrets.get("odds_api", {}).get("api_key", "")
        if not api_key:
            return {}

        url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
        params = {
            "apiKey":      api_key,
            "regions":     "us",
            "markets":     "h2h",
            "oddsFormat":  "american",
        }
        r = requests.get(url, params=params, timeout=12)
        if r.status_code != 200:
            return {}

        for game in r.json():
            home_name = game.get("home_team", "")
            away_name = game.get("away_team", "")
            home_odds = away_odds = None

            for bm in game.get("bookmakers", []):
                for mkt in bm.get("markets", []):
                    if mkt.get("key") == "h2h":
                        for oc in mkt.get("outcomes", []):
                            if oc.get("name", "").lower() in home_name.lower():
                                home_odds = float(oc.get("price", -110))
                            elif oc.get("name", "").lower() in away_name.lower():
                                away_odds = float(oc.get("price", 100))
                        break
                if home_odds is not None:
                    break

            if home_odds is None:
                home_odds, away_odds = -110, 100   # fallback neutral

            home_impl = _american_to_implied(home_odds)
            away_impl = _american_to_implied(away_odds)
            # Vig-remove: normalize so they sum to 1.0
            total_impl = home_impl + away_impl
            if total_impl > 0:
                home_impl_no_vig = home_impl / total_impl
                away_impl_no_vig = away_impl / total_impl
            else:
                home_impl_no_vig = away_impl_no_vig = 0.5

            # Key by team names — matched to games via TEAM_ABB_MAP in display
            game_key = f"{away_name}|{home_name}"
            result[game_key] = {
                "home_odds":     home_odds,
                "away_odds":     away_odds,
                "home_implied":  round(home_impl_no_vig, 4),
                "away_implied":  round(away_impl_no_vig, 4),
            }

    except Exception:
        pass
    return result


def compute_team_offense_score(plays: List[Dict], team: str) -> Tuple[float, int]:
    """
    Aggregate offensive quality for a team from confirmed lineup batters.
    Returns (team_wrc_plus_avg, n_batters_found).
    Uses wrc_plus from each batter's scored play dict.
    Falls back to 100 (league average) when lineup not loaded.
    """
    team_plays = [p for p in plays if p.get("team", "") == team]
    wrc_vals = [p.get("wrc_plus", 100.0) for p in team_plays if p.get("wrc_plus", 100.0) > 0]
    if not wrc_vals:
        return 100.0, 0
    return round(sum(wrc_vals) / len(wrc_vals), 1), len(wrc_vals)


def compute_win_probability(
    home_sp_stats: Dict, away_sp_stats: Dict,
    home_off_wrc: float, away_off_wrc: float,
    home_bp_vuln: float, away_bp_vuln: float,
    home_run_diff: float, away_run_diff: float,
    home_implied_runs: float, away_implied_runs: float,
) -> Tuple[float, str]:
    """
    Log5-style win probability estimate.

    Model: P(home wins) = (home_offense_strength × away_pitching_vuln) /
                          ((home_offense_strength × away_pitching_vuln) +
                           (away_offense_strength × home_pitching_vuln))

    Offense strength: derived from team wRC+ (batting quality of confirmed lineup).
    Pitching vuln: blended SP + team bullpen (same formula as O1.5 pitcher model).
    Home field adjustment: +3.5% to home win probability (MLB historical average).
    Momentum: last-7-day run differential nudges probability ±2%.

    Returns (home_win_prob_0_to_1, explanation_string).
    """
    # ── Offense strength: wRC+ normalized. League avg = 1.0, +10% = 1.1 ──
    home_off = max(0.5, home_off_wrc / 100.0)
    away_off = max(0.5, away_off_wrc / 100.0)

    # ── Pitching vuln: blend SP (60%) + bullpen (40%) — same as O1.5 model ──
    # Vuln scale: 0-100 where 50 = league avg, higher = more hittable
    # Invert for "how hard is the pitching": strength = (100 - vuln) / 100
    home_sp_vuln = float(home_sp_stats.get("_sp_vuln", 50.0))
    away_sp_vuln = float(away_sp_stats.get("_sp_vuln", 50.0))

    home_pit_vuln = home_sp_vuln * 0.60 + home_bp_vuln * 0.40
    away_pit_vuln = away_sp_vuln * 0.60 + away_bp_vuln * 0.40

    # Pitching STRENGTH = inverse of vulnerability
    # Elite pitcher (vuln=20) → strength=0.80; mop-up (vuln=70) → strength=0.30
    home_pit_str = max(0.2, (100.0 - home_pit_vuln) / 100.0)
    away_pit_str = max(0.2, (100.0 - away_pit_vuln) / 100.0)

    # ── Log5 numerator/denominator ──────────────────────────────────────────
    # Home advantage: score home as if facing slightly weaker pitcher
    HOME_ADJ = 0.035  # +3.5% raw before log5 normalization
    home_strength = home_off * away_pit_str * (1.0 + HOME_ADJ)
    away_strength = away_off * home_pit_str

    total = home_strength + away_strength
    if total <= 0:
        return 0.52, "Log5 denominator zero — returning home-field default"

    raw_home_wp = home_strength / total

    # ── Momentum nudge: last-7-day run diff ─────────────────────────────────
    # +1 run/game differential ≈ +1% win probability nudge (capped at ±2%)
    diff_nudge = (home_run_diff - away_run_diff) * 0.01
    diff_nudge = max(-0.02, min(0.02, diff_nudge))

    # ── Vegas sanity anchor: blend model 70% / vegas-derived 30% ────────────
    # When run totals are loaded, derive a weak prior from implied runs
    if home_implied_runs > 0 and away_implied_runs > 0:
        total_impl = home_implied_runs + away_implied_runs
        vegas_home_wp = home_implied_runs / total_impl if total_impl > 0 else 0.52
        final_home_wp = raw_home_wp * 0.70 + vegas_home_wp * 0.30 + diff_nudge
    else:
        final_home_wp = raw_home_wp + diff_nudge

    final_home_wp = max(0.30, min(0.75, final_home_wp))

    # Build explanation
    pit_qual_h = ("Elite" if home_pit_vuln < 30 else "Good" if home_pit_vuln < 45
                  else "Average" if home_pit_vuln < 58 else "Weak")
    pit_qual_a = ("Elite" if away_pit_vuln < 30 else "Good" if away_pit_vuln < 45
                  else "Average" if away_pit_vuln < 58 else "Weak")
    label = (f"Home pit: {pit_qual_h} (vuln={home_pit_vuln:.0f}) | "
             f"Away pit: {pit_qual_a} (vuln={away_pit_vuln:.0f}) | "
             f"Home off wRC+: {home_off_wrc:.0f} | Away off wRC+: {away_off_wrc:.0f} | "
             f"7d RunDiff H/A: {home_run_diff:+.1f}/{away_run_diff:+.1f}")

    return round(final_home_wp, 4), label


def display_moneyline_tab(games: List[Dict], plays: List[Dict],
                          ml_odds: Dict, run_diffs: Dict,
                          implied_totals: Dict,
                          team_bullpen_scores: Dict):
    """
    Tab: 🏦 Moneyline
    One row per game. Shows model win prob vs market implied prob vs edge.
    Flags Strong Edge (>7%), Lean (4-7%), No Play (<4%).
    """
    st.header("🏦 Moneyline — Win Probability Model")
    st.caption("Log5-style win probability vs Vegas implied. Edge = model prob minus market implied (vig-removed). "
               "Strong Edge >7% | Lean 4-7% | No Play <4%.")

    if not games:
        st.info("Run the model first to see moneyline analysis.")
        return

    pitching_df = st.session_state.get("_pitching_df_global", pd.DataFrame())

    rows = []
    for game in games:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        home_sp_name = game.get("home_pitcher") or "TBD"
        away_sp_name = game.get("away_pitcher") or "TBD"
        home_sp_id   = str(game.get("home_pitcher_id") or "")
        away_sp_id   = str(game.get("away_pitcher_id") or "")

        # SP stats from pitching_df (same source as O1.5 model)
        home_sp_raw = get_pitcher_stats(home_sp_name, home_sp_id, pitching_df)
        away_sp_raw = get_pitcher_stats(away_sp_name, away_sp_id, pitching_df)

        # Compute SP vulnerability (reuse existing function)
        home_bp_vuln = team_bullpen_scores.get(home, 42.0)
        away_bp_vuln = team_bullpen_scores.get(away, 42.0)

        home_sp_vuln, home_pit_label = compute_pitcher_score(home_sp_raw, bullpen_vuln=home_bp_vuln)
        away_sp_vuln, away_pit_label = compute_pitcher_score(away_sp_raw, bullpen_vuln=away_bp_vuln)

        # Inject SP vuln into dict for compute_win_probability
        home_sp_raw["_sp_vuln"] = home_sp_vuln
        away_sp_raw["_sp_vuln"] = away_sp_vuln

        # Team offense from confirmed lineup (aggregate wRC+ of scored batters)
        home_off_wrc, home_n = compute_team_offense_score(plays, home)
        away_off_wrc, away_n = compute_team_offense_score(plays, away)

        # Run differentials
        home_rd = run_diffs.get(home, 0.0)
        away_rd = run_diffs.get(away, 0.0)

        # Implied run totals from Odds API
        home_impl_runs = implied_totals.get(home, 0.0)
        away_impl_runs = implied_totals.get(away, 0.0)

        # Win probability
        home_wp, wp_label = compute_win_probability(
            home_sp_stats=home_sp_raw,
            away_sp_stats=away_sp_raw,
            home_off_wrc=home_off_wrc,
            away_off_wrc=away_off_wrc,
            home_bp_vuln=home_bp_vuln,
            away_bp_vuln=away_bp_vuln,
            home_run_diff=home_rd,
            away_run_diff=away_rd,
            home_implied_runs=home_impl_runs,
            away_implied_runs=away_impl_runs,
        )
        away_wp = round(1.0 - home_wp, 4)

        # Market implied (vig-removed) from h2h odds
        ml_key = None
        for k in ml_odds:
            parts = k.split("|")
            if len(parts) == 2:
                if (_norm(parts[0]) in _norm(game.get("away_team_name", away)) or
                    _norm(away) in _norm(parts[0])):
                    ml_key = k
                    break
                if (_norm(parts[1]) in _norm(game.get("home_team_name", home)) or
                    _norm(home) in _norm(parts[1])):
                    ml_key = k
                    break

        if ml_key and ml_key in ml_odds:
            mkt = ml_odds[ml_key]
            home_mkt = mkt["home_implied"]
            away_mkt = mkt["away_implied"]
            home_odds_str = f"{mkt['home_odds']:+.0f}"
            away_odds_str = f"{mkt['away_odds']:+.0f}"
        else:
            home_mkt = away_mkt = None
            home_odds_str = away_odds_str = "N/A"

        # Edge calculation
        if home_mkt is not None:
            home_edge = round((home_wp - home_mkt) * 100, 1)
            away_edge = round((away_wp - away_mkt) * 100, 1)
        else:
            home_edge = away_edge = None

        # Tier and recommendation
        def _edge_tier(edge):
            if edge is None: return "➖ No Play"
            if abs(edge) >= 7:  return "🔥 Strong Edge"
            if abs(edge) >= 4:  return "📊 Lean"
            return "➖ No Play"

        home_tier = _edge_tier(home_edge)
        away_tier = _edge_tier(away_edge)

        # Recommended side: whichever has the larger positive edge, if above threshold
        if home_edge is not None and away_edge is not None:
            if home_edge >= 4 and home_edge >= away_edge:
                rec = f"✅ {home} (H) {home_edge:+.1f}%"
            elif away_edge >= 4 and away_edge > home_edge:
                rec = f"✅ {away} (A) {away_edge:+.1f}%"
            else:
                rec = "➖ No Play"
        elif home_mkt is None:
            rec = "No ML odds loaded"
        else:
            rec = "➖ No Play"

        rows.append({
            "Matchup":        f"{away} @ {home}",
            "Away SP":        away_sp_name[:18],
            "Home SP":        home_sp_name[:18],
            f"{away} Model%": f"{away_wp*100:.1f}%",
            f"{home} Model%": f"{home_wp*100:.1f}%",
            f"{away} Mkt%":   f"{away_mkt*100:.1f}%" if away_mkt else "N/A",
            f"{home} Mkt%":   f"{home_mkt*100:.1f}%" if home_mkt else "N/A",
            f"{away} Odds":   away_odds_str,
            f"{home} Odds":   home_odds_str,
            f"{away} Edge":   f"{away_edge:+.1f}%" if away_edge is not None else "N/A",
            f"{home} Edge":   f"{home_edge:+.1f}%" if home_edge is not None else "N/A",
            "Away Tier":      away_tier,
            "Home Tier":      home_tier,
            "Pick":           rec,
            "_home_wp":       home_wp,
            "_away_wp":       away_wp,
            "_home_edge":     home_edge,
            "_away_edge":     away_edge,
            "_label":         wp_label,
            "_home_n":        home_n,
            "_away_n":        away_n,
        })

    if not rows:
        st.warning("No games to display.")
        return

    # Summary metrics
    strong_edges = [r for r in rows
                    if (r["_home_edge"] is not None and abs(r["_home_edge"]) >= 7) or
                       (r["_away_edge"] is not None and abs(r["_away_edge"]) >= 7)]
    lean_edges   = [r for r in rows
                    if (r["_home_edge"] is not None and 4 <= abs(r["_home_edge"]) < 7) or
                       (r["_away_edge"] is not None and 4 <= abs(r["_away_edge"]) < 7)]

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Games on Slate", len(rows))
    mc2.metric("🔥 Strong Edge (>7%)", len(strong_edges))
    mc3.metric("📊 Lean (4-7%)", len(lean_edges))

    st.markdown("---")

    # Display table — drop internal cols
    display_cols = [c for c in rows[0].keys() if not c.startswith("_")]
    df = pd.DataFrame(rows)[display_cols]

    def color_pick(val):
        if "Strong" in str(val): return "color: #00ff88; font-weight: bold"
        if "Lean"   in str(val): return "color: #ffdd00"
        if "No Play" in str(val): return "color: #888888"
        return ""

    def color_edge(val):
        try:
            v = float(str(val).replace("%","").replace("+",""))
            if v >= 7:  return "color: #00ff88; font-weight: bold"
            if v >= 4:  return "color: #ffdd00"
            if v <= -4: return "color: #ff4444"
        except: pass
        return ""

    edge_cols = [c for c in display_cols if "Edge" in c]
    tier_cols  = [c for c in display_cols if "Tier" in c]
    styled = df.style.map(color_pick, subset=["Pick"])
    for ec in edge_cols:
        styled = styled.map(color_edge, subset=[ec])
    st.dataframe(styled, use_container_width=True)

    # Export
    csv = df.to_csv(index=False)
    st.download_button("📥 Export Moneyline CSV", csv, "moneyline.csv", "text/csv", key="dl_ml")

    st.markdown("---")
    st.subheader("🔍 Game-by-Game Breakdown")

    for r in rows:
        with st.expander(f"{r['Matchup']}  |  Pick: {r['Pick']}"):
            b1, b2 = st.columns(2)
            parts = r["Matchup"].split(" @ ")
            away_t = parts[0] if parts else "Away"
            home_t = parts[1] if len(parts) > 1 else "Home"

            with b1:
                st.markdown(f"**{away_t} (Away)**")
                st.write(f"SP: {r['Away SP']}")
                st.write(f"Model Win%: {r.get(f'{away_t} Model%', 'N/A')}")
                st.write(f"Market Implied: {r.get(f'{away_t} Mkt%', 'N/A')}")
                st.write(f"ML Odds: {r.get(f'{away_t} Odds', 'N/A')}")
                st.write(f"Edge: {r.get(f'{away_t} Edge', 'N/A')}")
                st.write(f"Tier: {r['Away Tier']}")
            with b2:
                st.markdown(f"**{home_t} (Home)**")
                st.write(f"SP: {r['Home SP']}")
                st.write(f"Model Win%: {r.get(f'{home_t} Model%', 'N/A')}")
                st.write(f"Market Implied: {r.get(f'{home_t} Mkt%', 'N/A')}")
                st.write(f"ML Odds: {r.get(f'{home_t} Odds', 'N/A')}")
                st.write(f"Edge: {r.get(f'{home_t} Edge', 'N/A')}")
                st.write(f"Tier: {r['Home Tier']}")
            st.caption(f"Model inputs: {r['_label']}")
            st.caption(f"Lineup batters found: {home_t}={r['_home_n']} | {away_t}={r['_away_n']}")

    # No ML odds disclaimer
    if not ml_odds:
        st.warning("⚠️ No Odds API key loaded — market implied probabilities unavailable. "
                   "Edge column shows N/A. Add Odds API key in sidebar for full edge calculation.")


def compute_k_score_for_play(p: Dict, pitcher_statcast: Dict,
                              ump_k_adj: float = 0.0) -> float:
    """Thin wrapper used by Results Tracker to log K score alongside TB pick."""
    batter_statcast = {
        "k_rate":           p.get("k_rate", 0.228),
        "batter_swstr_pct": p.get("batter_swstr_pct", 0.0),
        "o_swing_pct":      p.get("o_swing_pct", 0.0),
    }
    pitcher_statcast_k = {
        "k_rate_allowed": pitcher_statcast.get("k_rate_allowed", 0.228),
        "swstr_pct":      pitcher_statcast.get("swstr_pct", 0.0),
    }
    score, _, _ = compute_k_score(batter_statcast, pitcher_statcast_k,
                                  p.get("lineup_slot", 5),
                                  p.get("implied_total", 4.5),
                                  ump_k_adj)
    return score


def compute_batter_score_hits(statcast: Dict) -> Tuple[float, str, Dict]:
    """
    O0.5 TB batter score — any hit model.
    Base rate ~65% (much higher than O1.5 ~47%).
    Completely different weight structure: contact > power.
    
    Key drivers:
    - AVG / BABIP / contact rate: getting on base any way
    - K% inverse: MOST important — K = guaranteed 0 hits  
    - BB% / OBP: plate discipline = more PA completions
    - wRC+: overall offensive value including contact
    - LD%: line drives = highest BABIP, most hits
    - Power metrics (barrel, xSLG): LESS important than O1.5
    """
    details = {}

    def f(key, default):
        try: return float(statcast.get(key, default))
        except: return float(default)

    # Primary contact metrics
    k_rate    = f("k_rate",        0.228)   # Most important — inverse
    wrc_plus  = f("wrc_plus",      100.0)   # Overall offensive value
    avg       = f("avg",           0.255)   # Batting average proxy
    bb_rate   = f("bb_rate",       0.082)   # Walk rate (more PA = more chances)
    woba      = f("woba",          0.315)   # On-base quality
    hard_hit  = f("hard_hit_rate", 0.370)   # Hard contact = harder to field
    xslg      = f("slg_proxy",     0.398)   # Some power still matters

    details["AVG"]   = f"{avg:.3f}"
    details["K%"]    = f"{k_rate*100:.1f}%"
    details["BB%"]   = f"{bb_rate*100:.1f}%"
    details["wRC+"]  = int(wrc_plus)
    details["wOBA"]  = f"{woba:.3f}"
    details["HH%"]   = f"{hard_hit*100:.1f}%"

    # ── Z-score normalized sub-scores — avg batter = 50 on each metric ──
    # V1.8: Matches O1.5 normalization approach. League avg = 50, ±1 sd = ±25 pts.

    # K rate INVERSE — most critical for O0.5. avg=22.8%, sd=6%
    k_score = max(0, min(100, 50 - (k_rate - 0.228) / 0.060 * 25))

    # wRC+: avg=100, sd=35
    wrc_score = max(0, min(100, 50 + (wrc_plus - 100) / 35.0 * 25))

    # BB%: avg=8.2%, sd=3% — plate discipline = more PA completions
    bb_score = max(0, min(100, 50 + (bb_rate - 0.082) / 0.030 * 25))

    # wOBA: avg=.315, sd=.040
    woba_score = max(0, min(100, 50 + (woba - 0.315) / 0.040 * 25))

    # Hard hit%: avg=37%, sd=5.5%
    hh_score = max(0, min(100, 50 + (hard_hit - 0.370) / 0.055 * 25))

    # xSLG: avg=.398, sd=.080
    xslg_score = max(0, min(100, 50 + (xslg - 0.398) / 0.080 * 25))

    composite = (
        k_score    * 0.30 +   # K% inverse — biggest driver (K = no hit guaranteed)
        wrc_score  * 0.22 +   # Overall offensive quality
        woba_score * 0.18 +   # On-base quality
        hh_score   * 0.12 +   # Contact quality
        bb_score   * 0.10 +   # Plate discipline / PA completion
        xslg_score * 0.08     # Some power still relevant
    )

    return max(0, min(100, composite)), "Contact/hit profile", details


def compute_pitcher_score_hits(statcast: Dict) -> Tuple[float, str]:
    """
    Pitcher vulnerability for O0.5 (any hit allowed model).
    HIGH score = pitcher gives up lots of hits = good for batter.
    
    Key change vs O1.5: K% is even MORE dominant here.
    A pitcher who strikes out 30% of batters is devastating for O0.5.
    H/9, WHIP, BABIP-against matter more than barrel/hard hit.
    """
    def f(key, default):
        try: return float(statcast.get(key, default))
        except: return float(default)

    k_rate  = f("k_rate_allowed",   0.220)
    era     = f("era",              4.20)
    fip     = f("fip",              4.20)
    whip    = f("whip",             1.30)
    hard_hit= f("hard_hit_allowed", 0.370)

    # ── Z-score normalized pitcher vulnerability — avg pitcher = 50 ──
    # V1.8: Matches O1.5 normalization. High score = hittable = good for batter.

    # K% INVERSE: avg=22%, sd=5%
    k_vuln = max(0, min(100, 50 - (k_rate - 0.220) / 0.050 * 25))

    # WHIP: avg=1.30, sd=0.20
    whip_vuln = max(0, min(100, 50 + (whip - 1.30) / 0.20 * 25))

    # ERA/FIP: avg=4.2, sd=0.80
    era_use = fip if fip > 0 else era
    era_vuln = max(0, min(100, 50 + (era_use - 4.20) / 0.80 * 25))

    # Hard hit allowed: avg=37%, sd=5.5%
    hh_vuln = max(0, min(100, 50 + (hard_hit - 0.370) / 0.055 * 25))

    composite = (
        k_vuln    * 0.50 +   # K% most critical for any-hit model
        whip_vuln * 0.25 +   # WHIP captures overall hittability
        era_vuln  * 0.15 +
        hh_vuln   * 0.10
    )

    return max(0, min(100, composite)), f"K%: {k_rate*100:.0f}% | WHIP: {whip:.2f} | FIP: {era_use:.2f}"


def score_to_prob_hits(score: float) -> float:
    """
    Map 0-100 score to O0.5 (any hit) probability.
    Higher base rate than O1.5 (~65% vs ~47%).
    Research-calibrated:
    - Score 58 (LIKELY floor) → ~62% probability
    - Score 68 (SAFE floor) → ~68% probability
    - Score 78 (SAFE+ floor) → ~73% probability
    """
    a = 0.07
    b = 56     # midpoint calibrated to O0.5 base rate
    prob = 1 / (1 + math.exp(-a * (score - b)))
    # Scale to O0.5 range: base 55%, max 82%
    prob = 0.55 + prob * 0.27
    return round(min(0.82, max(0.55, prob)), 3)


def get_tier_hits(score: float) -> str:
    """O0.5 tiers — calibrated to actual score distribution."""
    if score >= 80:
        return "🔒 SAFE+"
    elif score >= 70:
        return "✅ SAFE"
    elif score >= 60:
        return "📊 LIKELY"
    else:
        return "❌ SKIP"


def compute_hits_score_for_player(
    batter_statcast: Dict,
    pitcher_statcast: Dict,
    batter_hand: str,
    sp_hand: str,
    lineup_slot: int,
    park_team: str,
    weather: Dict,
    implied: float,
    sp_tbd: bool,
    lineup_confirmed: bool,
    pitch_matchup_score: float = 50.0,
) -> Tuple[float, float, str, Dict]:
    """
    Compute O0.5 score, probability, tier, and details for a single player.
    Returns (score, prob, tier, details_dict).
    V1.6: Added pitch matchup score.
    Calibrated: avg batter vs avg pitcher = ~58, elite bat vs weak pitcher = 75+
    """
    # Inject wRC+ proxy from xSLG if not present (fixes default=100 issue)
    enriched = dict(batter_statcast)
    if enriched.get("wrc_plus", 100) == 100.0:
        xslg = float(enriched.get("slg_proxy", 0.398))
        enriched["wrc_plus"] = max(40, min(220, 100 + (xslg - 0.398) / 0.005))

    bat_score, _, bat_details = compute_batter_score_hits(enriched)
    pit_score, pit_label = compute_pitcher_score_hits(pitcher_statcast)
    # V1.6: pitch matchup — for O0.5, K-inducing pitch vs batter whiff matters most
    # Scale matchup contribution: neutral at 50, but weight is modest (4%)
    plat_score, plat_label = compute_platoon_score(batter_hand, sp_hand)
    lineup_sc, lineup_label = compute_lineup_score(lineup_slot)
    park_sc, _ = compute_park_score(park_team, True)
    weather_sc, _ = compute_weather_score(weather)
    vegas_sc, _ = compute_vegas_score(implied)
    tto_sc, _ = compute_tto_bonus(lineup_slot)

    # O0.5 weights: K% dominates, park matters less than O1.5
    # V1.6: pitch matchup added at 4%, taken from platoon (contact matters more than platoon for O0.5)
    raw = (
        bat_score           * 0.42 +
        pit_score           * 0.30 +
        pitch_matchup_score * 0.04 +  # V1.6: pitch type matchup
        plat_score          * 0.04 +
        lineup_sc           * 0.05 +
        park_sc             * 0.04 +
        weather_sc          * 0.02 +
        vegas_sc            * 0.05 +
        tto_sc              * 0.04
    )
    # Calibration offset
    score = max(0, min(100, round(raw + 6.5, 1)))  # V1.8: calibrated; avg batter vs avg SP = 55

    if sp_tbd:
        score = min(score, 74)
    if not lineup_confirmed:
        score = min(score, 72)

    prob = score_to_prob_hits(score)
    tier = get_tier_hits(score)

    details = {
        **bat_details,
        "pit_label": pit_label,
        "plat": plat_label,
        "lineup": lineup_label,
        "sub_bat": round(bat_score, 1),
        "sub_pit": round(pit_score, 1),
    }

    return score, prob, tier, details


def display_hits_tab(plays: List[Dict]):
    """
    Over 0.5 Total Bases tab.
    Re-scores all players using contact/hit model.
    Tiers: SAFE+ (85+), SAFE (75-84), LIKELY (65-74), SKIP (<65).
    """
    st.header("🎯 Over 0.5 Total Bases")
    st.caption("Any hit model — singles count. Higher base rate (~65%). Contact + K-avoidance driven.")

    if not plays:
        st.info("Run the model first.")
        return

    # Re-score every player using the hits model
    hits_plays = []
    for p in plays:
        # We need pitcher statcast — get it from stored labels
        # Re-use the pre-loaded stats from the original run
        # Build a minimal statcast dict from stored pitcher label
        pitcher_mock = {
            "k_rate_allowed": 0.220, "era": 4.20, "fip": 4.20,
            "whip": 1.30, "hard_hit_allowed": 0.370,
        }
        # Parse pitcher label back into stats if available
        pit_label = p.get("pitcher_label", "")
        try:
            if "K%:" in pit_label:
                k_str = pit_label.split("K%:")[1].split("%")[0].strip()
                pitcher_mock["k_rate_allowed"] = float(k_str) / 100
            if "FIP:" in pit_label:
                fip_str = pit_label.split("FIP:")[1].strip().split()[0]
                pitcher_mock["fip"] = float(fip_str)
        except Exception:
            pass

        batter_mock = {
            "k_rate":        p.get("k_rate", 0.228),
            # Use stored wRC+ if available, otherwise derive from xSLG
            "wrc_plus":      p.get("wrc_plus") or max(40, min(220, 100 + (p.get("xslg", 0.398) - 0.398) / 0.005)),
            "avg":           0.255,
            "bb_rate":       p.get("bb_rate", 0.082),
            "woba":          p.get("xslg", 0.398) * 0.78,
            "hard_hit_rate": p.get("hard_hit_rate", 0.370),
            "slg_proxy":     p.get("xslg", 0.398),
        }

        h_score, h_prob, h_tier, h_details = compute_hits_score_for_player(
            batter_statcast=batter_mock,
            pitcher_statcast=pitcher_mock,
            batter_hand=p.get("batter_hand", "R"),
            sp_hand=p.get("sp_hand", "R"),
            lineup_slot=p.get("lineup_slot", 5),
            park_team=p.get("park", p.get("team", "")),
            weather=p.get("weather", {}),
            implied=p.get("implied_total", 0),
            sp_tbd=p.get("sp_tbd", False),
            lineup_confirmed=p.get("lineup_confirmed", True),
            pitch_matchup_score=p.get("sub_matchup", 50.0),  # V1.6
        )

        hits_plays.append({**p, "h_score": h_score, "h_prob": h_prob,
                           "h_tier": h_tier, "h_details": h_details})

    hits_plays.sort(key=lambda x: x["h_score"], reverse=True)

    # Summary metrics
    safe_plus = [p for p in hits_plays if p["h_tier"] == "🔒 SAFE+"]
    safe      = [p for p in hits_plays if p["h_tier"] == "✅ SAFE"]
    likely    = [p for p in hits_plays if p["h_tier"] == "📊 LIKELY"]
    skip      = [p for p in hits_plays if p["h_tier"] == "❌ SKIP"]

    if safe_plus:
        st.success(f"🔒 {len(safe_plus)} SAFE+ plays — elite contact spots, parlay anchors")
    elif safe:
        st.info(f"✅ {len(safe)} SAFE plays — strong contact matchups")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("🔒 SAFE+", len(safe_plus))
    with col2: st.metric("✅ SAFE", len(safe))
    with col3: st.metric("📊 LIKELY", len(likely))
    with col4: st.metric("❌ SKIP", len(skip))

    st.caption("Tier thresholds: SAFE+ 78+ | SAFE 68-77 | LIKELY 58-67 | SKIP <58")
    st.markdown("---")

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        tier_filter = st.multiselect(
            "Filter tier", ["🔒 SAFE+", "✅ SAFE", "📊 LIKELY", "❌ SKIP"],
            default=["🔒 SAFE+", "✅ SAFE", "📊 LIKELY"], key="hits_tier_filter"
        )
    with col_f2:
        teams = sorted(set(p["team"] for p in hits_plays))
        team_filter = st.multiselect("Filter team", teams, default=[], key="hits_team_filter")
    with col_f3:
        min_score = st.slider("Min score", 0, 100, 60, key="hits_min_score")

    filtered = [p for p in hits_plays
                if p["h_tier"] in tier_filter
                and (not team_filter or p["team"] in team_filter)
                and p["h_score"] >= min_score]

    st.markdown(f"**Showing {len(filtered)} batters**")

    # Build table
    rows = []
    for p in filtered:
        tbd = " ⚠️" if p.get("sp_tbd") else ""
        rows.append({
            "H-Score": f"{p['h_score']:.0f}",
            "Tier": p["h_tier"],
            "Player": p["name"],
            "Team": p["team"],
            "Vs": p["opponent"],
            "Slot": f"#{p['lineup_slot']}",
            "Hand": p["batter_hand"],
            "Opp SP": p["sp_name"][:20] + tbd,
            "SP 🤚": p["sp_hand"],
            "Prob": f"{p['h_prob']*100:.0f}%",
            "K%": f"{p.get('k_rate', 0)*100:.1f}%" if p.get("k_rate") else "—",
            "HH%": f"{p.get('hard_hit_rate', 0)*100:.1f}%" if p.get("hard_hit_rate") else "—",
            "xSLG": f"{p.get('xslg', 0):.3f}" if p.get("xslg") else "—",
            "Platoon": p.get("platoon_label", "").split("(")[0].strip(),
            "Park": p.get("park", ""),
            "Imp.Runs": f"{p.get('implied_total', 0):.1f}" if p.get("implied_total", 0) > 0 else "—",
            "O1.5 Score": f"{p['score']:.0f}",  # cross-reference
        })

    if rows:
        df = pd.DataFrame(rows)

        def color_htier(val):
            if "SAFE+" in str(val): return "color: #00ff88; font-weight: bold"
            elif "SAFE" in str(val): return "color: #66ddff; font-weight: bold"
            elif "LIKELY" in str(val): return "color: #ffdd00"
            return "color: #888888"

        def color_hscore(val):
            try:
                v = float(str(val))
                if v >= 85: return "color: #00ff88; font-weight: bold"
                elif v >= 75: return "color: #66ddff; font-weight: bold"
                elif v >= 65: return "color: #ffdd00"
                return "color: #888888"
            except: return ""

        styled = df.style.map(color_htier, subset=["Tier"]).map(color_hscore, subset=["H-Score"])
        st.dataframe(styled, use_container_width=True, height=500)

        csv = df.to_csv(index=False)
        st.download_button("📥 Export O0.5 Plays", csv,
                           f"o05_picks_{datetime.now(EST).strftime('%Y%m%d')}.csv", "text/csv")

    # Top plays detail
    st.markdown("---")
    st.subheader("🏆 Top O0.5 Plays — Full Breakdown")
    top = [p for p in filtered if p["h_score"] >= 60][:5]
    for i, p in enumerate(top, 1):
        with st.expander(f"{p['h_tier']} #{i}: {p['name']} ({p['team']}) — H-Score: {p['h_score']:.0f} | Prob: {p['h_prob']*100:.0f}% | O1.5 Score: {p['score']:.0f}"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Contact Profile (O0.5 drivers)**")
                st.write(f"• K%: {p.get('k_rate', 0)*100:.1f}% ← most critical" if p.get("k_rate") else "• K%: —")
                st.write(f"• Hard Hit%: {p.get('hard_hit_rate', 0)*100:.1f}%" if p.get("hard_hit_rate") else "• HH%: —")
                st.write(f"• xSLG: {p.get('xslg', 0):.3f}" if p.get("xslg") else "• xSLG: —")
                st.write(f"• Platoon: {p.get('platoon_label', '—')}")
                st.write(f"• Lineup: {p.get('lineup_label', '—')}")
            with col_b:
                st.markdown("**Pitcher + Environment**")
                st.write(f"• SP: {p['sp_name']} ({p['sp_hand']}HP)")
                st.write(f"• {p.get('pitcher_label', '—')}")
                st.write(f"• Park: {p.get('park_label', p.get('park', '—'))}")
                st.write(f"• Implied runs: {p.get('implied_total', 0):.1f}" if p.get("implied_total", 0) > 0 else "• Implied: —")
                st.markdown(f"**O0.5 Score: {p['h_score']:.0f} ({p['h_prob']*100:.0f}%)**")
                st.markdown(f"*vs O1.5 Score: {p['score']:.0f} ({p['prob']*100:.0f}%)*")

    # Mixed parlay note
    st.markdown("---")
    st.info("💡 **Mixed Parlay tip:** Combine SAFE+ plays from this tab with Tier 1/2 plays from the O1.5 Leaderboard in the Parlay Builder. High-contact bats + power matchups = diversified legs.")


    """Display performance tracking and historical results."""
    
    st.header("📈 Results Tracker")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Controls row
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.caption("Track outcomes of model picks to measure performance over time.")
    with col3:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # Load picks
    try:
        picks_df = pd.read_sql("SELECT * FROM picks ORDER BY date DESC, model_score DESC", conn)
        parlays_df = pd.read_sql("SELECT * FROM parlays ORDER BY date DESC", conn)
    except:
        picks_df = pd.DataFrame()
        parlays_df = pd.DataFrame()
    
    conn.close()
    
    if picks_df.empty:
        st.info("📊 No data yet. Run the model and log results to start tracking.")
        return
    
    # Overall performance
    resolved = picks_df[picks_df["result"].isin(["hit", "miss"])]
    
    if not resolved.empty:
        total_hits = len(resolved[resolved["result"] == "hit"])
        total_picks = len(resolved)
        hit_rate = total_hits / total_picks if total_picks > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Overall Record", f"{total_hits}-{total_picks-total_hits}")
        with col2: st.metric("Hit Rate", f"{hit_rate*100:.1f}%")
        with col3:
            tier1_r = resolved[resolved["tier"] == "🔒 TIER 1"]
            t1_rate = len(tier1_r[tier1_r["result"]=="hit"]) / len(tier1_r) if len(tier1_r) > 0 else 0
            st.metric("Tier 1 Hit%", f"{t1_rate*100:.1f}%" if len(tier1_r) > 0 else "—")
        with col4:
            tier2_r = resolved[resolved["tier"] == "✅ TIER 2"]
            t2_rate = len(tier2_r[tier2_r["result"]=="hit"]) / len(tier2_r) if len(tier2_r) > 0 else 0
            st.metric("Tier 2 Hit%", f"{t2_rate*100:.1f}%" if len(tier2_r) > 0 else "—")
    
    st.markdown("---")
    
    # Pick log
    st.subheader("📋 Pick Log")
    
    # Date range filter
    col1, col2 = st.columns(2)
    with col1:
        date_range_start = st.date_input("From", value=datetime.now(EST).date() - timedelta(days=30))
    with col2:
        date_range_end = st.date_input("To", value=datetime.now(EST).date())
    
    filtered_picks = picks_df[
        (picks_df["date"] >= str(date_range_start)) & 
        (picks_df["date"] <= str(date_range_end))
    ]
    
    if not filtered_picks.empty:
        display_cols = ["date", "player_name", "team", "opponent", "sp_name", 
                       "lineup_slot", "model_score", "tier", "result", "tb_actual", "implied_total"]
        available_cols = [c for c in display_cols if c in filtered_picks.columns]
        st.dataframe(filtered_picks[available_cols], use_container_width=True)
    
    st.markdown("---")
    
    # Manual result entry
    st.subheader("✏️ Log Results")
    st.caption("After games complete, log actual total bases here to track model accuracy.")
    
    pending = picks_df[picks_df["result"] == "pending"]
    if not pending.empty:
        pick_options = {f"{r['player_name']} ({r['team']}) - {r['date']}": r["pick_id"] 
                       for _, r in pending.head(20).iterrows()}
        
        selected_pick = st.selectbox("Select pick to update:", list(pick_options.keys()))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            actual_tb = st.number_input("Actual Total Bases", 0, 16, 0)
        with col2:
            result = "hit" if actual_tb >= 2 else "miss"
            st.metric("Result", result.upper())
        with col3:
            if st.button("💾 Save Result"):
                pick_id = pick_options[selected_pick]
                update_pick_result(pick_id, result, actual_tb)
                st.success(f"✅ Logged: {actual_tb} TB = {result.upper()}")
                st.rerun()
    else:
        st.caption("No pending picks to update.")
    
    # Export buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if not picks_df.empty:
            csv = picks_df.to_csv(index=False)
            st.download_button("📥 Export All Picks", csv, "mlb_picks_history.csv", "text/csv", key="dl_picks_history")
    with col2:
        if not parlays_df.empty:
            csv = parlays_df.to_csv(index=False)
            st.download_button("📥 Export Parlays", csv, "mlb_parlays_history.csv", "text/csv")

# ============================================================================
# MAIN APP
# ============================================================================

# ============================================================================
# FAN DUEL DFS ENGINE
# ============================================================================

# FanDuel MLB Scoring (different from prop scoring — BB, SB, RBI, Runs all count)
FD_SCORING = {
    "single":    3.0,
    "double":    6.0,
    "triple":    9.0,
    "hr":       12.0,
    "rbi":       3.5,
    "run":       3.2,
    "bb":        3.0,   # KEY: walks count on FD, not in TB props
    "hbp":       3.0,
    "sb":        6.0,   # KEY: steals worth 6 pts — big differentiator
    "cs":       -3.0,
    "out":      -0.25,  # per out (AB - H - BB - HBP)
}

# FanDuel lineup structure
FD_LINEUP = {
    "P":  1,  # Pitcher (1 SP)
    "C":  1,
    "1B": 1,
    "2B": 1,
    "3B": 1,
    "SS": 1,
    "OF": 3,  # 3 outfielders
    "UTIL": 1,  # any position except P
}
FD_SALARY_CAP = 35000
FD_MIN_SALARY = 1000

def compute_fd_projection(statcast: Dict, pitcher_statcast: Dict,
                          lineup_slot: int, implied_total: float,
                          batter_hand: str, sp_hand: str,
                          park_team: str, weather: Dict) -> Dict:
    """
    Project FanDuel points for a batter.
    Key differences from TB model:
    - BB count (3 pts each)
    - SB count (6 pts each)  
    - RBI and Runs scored count
    - Outs are slightly negative
    """
    def f(key, default):
        try: return float(statcast.get(key, default))
        except: return float(default)

    # Plate appearance estimate by lineup slot
    pa_by_slot = {1:4.8,2:4.7,3:4.6,4:4.5,5:4.3,6:4.2,7:4.1,8:3.9,9:3.8}
    est_pa = pa_by_slot.get(lineup_slot, 4.2)

    # Adjust PA for team implied total (high-scoring games = more AB)
    if implied_total > 0:
        pa_adj = (implied_total - 4.5) * 0.08  # ~0.08 extra PA per extra run
        est_pa += pa_adj

    # Key rates
    k_rate    = f("k_rate",        0.228)
    bb_rate   = f("bb_rate",       0.082)
    slg       = f("slg_proxy",     0.398)
    iso       = f("iso_proxy",     0.165)
    woba      = f("woba",          0.315)
    hard_hit  = f("hard_hit_rate", 0.370)
    barrel    = f("barrel_rate",   0.070)

    # Stolen base rate (proxy from speed — use sprint speed / 28 if available)
    # Default ~0.05 SB per game for average runner
    sb_rate = 0.05  # will be refined when sprint speed data added

    # Estimate plate appearance outcomes
    # Contact rate = 1 - K%
    contact_rate = 1 - k_rate
    # Hit rate estimate from wOBA and contact
    hit_rate = max(0.180, woba * 0.85)  # rough proxy for AVG
    # TB per hit
    tb_per_hit = slg / max(0.01, hit_rate)
    # Hit types
    hr_per_pa  = barrel * 0.35  # barrels -> HRs
    xbh_per_pa = (iso * 0.6) * contact_rate  # ISO -> XBH rate
    single_per_pa = (hit_rate - hr_per_pa - xbh_per_pa) * max(0, 1)
    single_per_pa = max(0, single_per_pa)

    # Per PA estimates
    hits_per_pa  = hit_rate
    bb_per_pa    = bb_rate
    sb_per_pa    = sb_rate
    outs_per_pa  = max(0, 1 - hits_per_pa - bb_per_pa - 0.01)  # rough

    # Apply park factor adjustment
    park_hr  = PARK_HR_FACTORS.get(park_team, 1.0)
    park_tb  = PARK_TB_FACTORS.get(park_team, 1.0)
    hr_per_pa  *= park_hr
    single_per_pa *= park_tb

    # Apply weather adjustment
    if not weather.get("is_dome"):
        wind_effect = weather.get("wind_effect", "neutral")
        if wind_effect == "strong_out":
            hr_per_pa *= 1.25
        elif wind_effect == "out":
            hr_per_pa *= 1.15
        elif wind_effect == "in":
            hr_per_pa *= 0.80

    # Pitcher quality adjustment
    pit_k    = float(pitcher_statcast.get("k_rate_allowed", 0.228))
    pit_hh   = float(pitcher_statcast.get("hard_hit_allowed", 0.370))
    pit_fip  = float(pitcher_statcast.get("fip", 4.10))
    pit_adj  = 1.0 + (pit_fip - 4.0) * 0.04  # FIP above 4 = better for batter
    pit_k_adj = 1.0 - (pit_k - 0.228) * 1.5  # higher K = fewer hits
    quality_adj = (pit_adj + pit_k_adj) / 2

    hits_per_pa  *= quality_adj
    hr_per_pa    *= quality_adj

    # Scale to estimated PAs
    proj_hits    = hits_per_pa * est_pa
    proj_hr      = hr_per_pa * est_pa
    proj_singles = max(0, single_per_pa * est_pa * quality_adj)
    proj_xbh     = max(0, xbh_per_pa * est_pa * quality_adj)
    proj_doubles = proj_xbh * 0.65
    proj_triples = proj_xbh * 0.05
    proj_bb      = bb_per_pa * est_pa
    proj_sb      = sb_per_pa * est_pa
    proj_outs    = outs_per_pa * est_pa

    # RBI and Runs — function of team context and lineup slot
    rbi_rate = 0.32 if lineup_slot <= 4 else 0.22
    run_rate = 0.38 if lineup_slot <= 3 else 0.28 if lineup_slot <= 6 else 0.20
    if implied_total > 0:
        rbi_rate *= (implied_total / 4.5)
        run_rate *= (implied_total / 4.5)
    proj_rbi = proj_hits * rbi_rate + proj_hr * 1.0  # HR always = at least 1 RBI
    proj_runs = proj_hits * run_rate + proj_hr * 1.0

    # FanDuel points
    fd_pts = (
        proj_singles * FD_SCORING["single"] +
        proj_doubles * FD_SCORING["double"] +
        proj_triples * FD_SCORING["triple"] +
        proj_hr      * FD_SCORING["hr"] +
        proj_rbi     * FD_SCORING["rbi"] +
        proj_runs    * FD_SCORING["run"] +
        proj_bb      * FD_SCORING["bb"] +
        proj_sb      * FD_SCORING["sb"] +
        proj_outs    * FD_SCORING["out"]
    )

    # Ceiling (75th percentile) and floor (25th percentile)
    variance = fd_pts * 0.45  # MLB has high variance
    ceiling = fd_pts + variance
    floor   = max(0, fd_pts - variance * 0.6)

    return {
        "fd_proj":    round(fd_pts, 1),
        "fd_ceiling": round(ceiling, 1),
        "fd_floor":   round(floor, 1),
        "est_pa":     round(est_pa, 1),
        "proj_hr":    round(proj_hr, 2),
        "proj_hits":  round(proj_hits, 2),
        "proj_bb":    round(proj_bb, 2),
        "proj_sb":    round(proj_sb, 2),
        "proj_rbi":   round(proj_rbi, 2),
        "proj_runs":  round(proj_runs, 2),
    }


def compute_ownership_projection(
    fd_proj: float, salary: int, implied_total: float,
    lineup_slot: int, barrel_rate: float, name: str
) -> float:
    """
    Project FanDuel ownership %.
    Higher projection + lower salary = higher ownership.
    Stars in high-total games are chalk.
    """
    if salary <= 0:
        return 15.0  # default

    # Value score (pts per $1000)
    value = fd_proj / (salary / 1000)

    # Base ownership from value
    own = value * 4.5  # rough calibration

    # Adjustments
    if implied_total >= 5.5:
        own *= 1.30  # high-total game = chalk
    elif implied_total >= 5.0:
        own *= 1.15
    elif implied_total < 3.5:
        own *= 0.70

    if lineup_slot <= 2:
        own *= 1.20  # leadoff/2-hole = more ownership
    elif lineup_slot >= 7:
        own *= 0.80

    if barrel_rate > 0.15:
        own *= 1.15  # power hitters owned more

    # Cap
    return round(min(60, max(3, own)), 1)


def build_fd_lineups(
    players: List[Dict],
    num_lineups: int = 3,
    mode: str = "gpp",  # "gpp" or "cash"
    salary_cap: int = FD_SALARY_CAP
) -> List[Dict]:
    """
    Build FanDuel lineups using a greedy optimizer.
    GPP: maximize ceiling, differentiation, stacks
    Cash: maximize floor/projection, avoid risk
    """
    if not players:
        return []

    # Filter players with salary
    eligible = [p for p in players if p.get("fd_salary", 0) > 0]
    if len(eligible) < 9:
        return []

    positions = ["P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF", "UTIL"]

    lineups = []
    used_combos = set()

    for lineup_num in range(num_lineups):
        # Sort by target metric
        if mode == "gpp":
            # GPP: weight ceiling heavily, add differentiation
            sort_key = lambda p: (
                p["fd_ceiling"] * 0.6 +
                p["fd_proj"] * 0.3 +
                (1 / max(1, p.get("ownership", 15))) * 10  # leverage
            )
        else:
            # Cash: pure floor + projection consistency
            sort_key = lambda p: p["fd_floor"] * 0.4 + p["fd_proj"] * 0.6

        sorted_players = sorted(eligible, key=sort_key, reverse=True)

        lineup = {}
        remaining_salary = salary_cap
        filled_positions = set()
        lineup_players = []
        team_counts = {}

        def can_play(player, pos):
            player_pos = player.get("fd_position", "OF")
            if pos == "UTIL":
                return player_pos != "P"
            if pos == "OF":
                return player_pos in ("OF", "RF", "LF", "CF")
            return player_pos == pos

        # GPP: force a 4-man stack from a high-total game
        if mode == "gpp" and lineup_num == 0:
            # Find top game by implied total
            top_games = {}
            for p in eligible:
                game = p.get("game_id", "")
                team = p.get("team", "")
                imp = p.get("implied_total", 0)
                key = f"{game}_{team}"
                if key not in top_games or imp > top_games[key]["implied"]:
                    top_games[key] = {"implied": imp, "team": team, "game": game}

            if top_games:
                best_stack_team = max(top_games.values(), key=lambda x: x["implied"])["team"]
                stack_players = [p for p in sorted_players
                                 if p.get("team") == best_stack_team
                                 and p.get("fd_position", "") != "P"][:4]
                for sp in stack_players[:3]:
                    # Add to lineup greedily
                    for pos in ["OF", "1B", "2B", "3B", "SS", "C", "UTIL"]:
                        if pos not in filled_positions and can_play(sp, pos):
                            if remaining_salary - sp.get("fd_salary", 0) >= FD_MIN_SALARY * (9 - len(lineup_players) - 1):
                                lineup_players.append({**sp, "slot": pos})
                                filled_positions.add(pos)
                                remaining_salary -= sp.get("fd_salary", 0)
                                team_counts[sp["team"]] = team_counts.get(sp["team"], 0) + 1
                                break

        # Fill remaining slots
        for pos in positions:
            if pos in filled_positions:
                continue
            slots_left = len(positions) - len(lineup_players) - 1
            min_remaining = FD_MIN_SALARY * slots_left

            for p in sorted_players:
                name = p["name"]
                if any(lp["name"] == name for lp in lineup_players):
                    continue
                if not can_play(p, pos):
                    continue
                salary = p.get("fd_salary", 0)
                if salary > remaining_salary - min_remaining:
                    continue
                # Ownership cap for GPP differentiation on lineups 2-3
                if mode == "gpp" and lineup_num > 0:
                    if p.get("ownership", 0) > 40:
                        continue  # avoid mega-chalk on later lineups
                lineup_players.append({**p, "slot": pos})
                filled_positions.add(pos)
                remaining_salary -= salary
                team_counts[p["team"]] = team_counts.get(p["team"], 0) + 1
                break

        if len(lineup_players) >= 9:
            total_sal = sum(p.get("fd_salary", 0) for p in lineup_players)
            total_proj = sum(p["fd_proj"] for p in lineup_players)
            total_ceil = sum(p["fd_ceiling"] for p in lineup_players)
            total_floor = sum(p["fd_floor"] for p in lineup_players)

            combo_key = tuple(sorted(p["name"] for p in lineup_players))
            if combo_key not in used_combos:
                used_combos.add(combo_key)
                lineups.append({
                    "players": lineup_players,
                    "total_salary": total_sal,
                    "total_proj": round(total_proj, 1),
                    "total_ceiling": round(total_ceil, 1),
                    "total_floor": round(total_floor, 1),
                    "mode": mode,
                    "stacks": {t: c for t, c in team_counts.items() if c >= 2},
                })

    return lineups


def display_dfs_tab(plays: List[Dict]):
    """
    FanDuel DFS Optimizer tab.
    GPP and Cash sections.
    Salary input via text area (FD CSV paste or manual entry).
    Auto-generates projections from prop model data.
    """
    st.header("🏆 FanDuel DFS Optimizer")
    st.caption("FD scoring: 1B=3 | 2B=6 | 3B=9 | HR=12 | RBI=3.5 | R=3.2 | BB=3 | SB=6 | Out=-0.25")

    if not plays:
        st.info("Run the model first — DFS projections auto-generate from the same data.")
        return

    # ── SALARY INPUT ──────────────────────────────────────────────────────
    st.subheader("💵 Salary Input")
    st.caption("Paste FanDuel salary data below. Format: PlayerName,Position,Salary (one per line). Or upload FD CSV.")

    col_sal, col_up = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader("Upload FD CSV", type=["csv"], key="fd_csv_upload")

    salary_data = {}

    if uploaded:
        try:
            import io
            df_upload = pd.read_csv(io.StringIO(uploaded.read().decode("utf-8")))
            # FD CSV columns: Nickname, Position, Salary (or Name, Position, Salary)
            name_col = next((c for c in df_upload.columns if c.lower() in ("nickname","name","player name","playername")), None)
            pos_col  = next((c for c in df_upload.columns if c.lower() in ("position","pos")), None)
            sal_col  = next((c for c in df_upload.columns if c.lower() in ("salary","sal")), None)
            if name_col and sal_col:
                for _, row in df_upload.iterrows():
                    name = str(row[name_col]).strip()
                    sal  = int(str(row[sal_col]).replace("$","").replace(",","").strip())
                    pos  = str(row[pos_col]).strip().upper() if pos_col else "OF"
                    salary_data[name] = {"salary": sal, "position": pos}
                st.success(f"✅ Loaded {len(salary_data)} players from CSV")
        except Exception as e:
            st.error(f"CSV parse error: {e}")

    with col_sal:
        salary_text = st.text_area(
            "Or paste manually (Name, Position, Salary):",
            placeholder="Aaron Judge, OF, 4200\nPete Alonso, 1B, 3800\nFernando Tatis Jr, OF, 4000",
            height=120,
            key="fd_salary_text"
        )
        if salary_text.strip() and not salary_data:
            for line in salary_text.strip().split("\n"):
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 2:
                    name = parts[0]
                    try:
                        if len(parts) == 3:
                            pos = parts[1].upper()
                            sal = int(parts[2].replace("$","").replace(",",""))
                        else:
                            pos = "OF"
                            sal = int(parts[1].replace("$","").replace(",",""))
                        salary_data[name] = {"salary": sal, "position": pos}
                    except Exception:
                        pass

    has_salaries = len(salary_data) > 0
    if has_salaries:
        st.success(f"✅ {len(salary_data)} salaries loaded")
    else:
        st.warning("⚠️ No salaries loaded — projections will show without salary optimization")
        with st.expander("📥 How to get FanDuel salaries (3 options)"):
            st.markdown("""
**Option 1 — FD Contest Lobby CSV (Recommended, 30 seconds):**
1. Go to FanDuel → Lobby → MLB → click any contest
2. Click **"Export to CSV"** or **"Download Salaries"** button
3. Upload that CSV file above ↑

**Option 2 — RotoWire / FantasyPros (free):**
- [RotoWire FD Salaries](https://www.rotogrinders.com/lineups/mlb) → Export CSV
- Format: `Name, Position, Salary`

**Option 3 — Manual paste (quick for small slates):**
```
Aaron Judge, OF, 4200
Pete Alonso, 1B, 3800
Juan Soto, OF, 4000
Cal Raleigh, C, 3200
```

**Why no auto-pull?** FanDuel has no public salary API. Any scraper would be fragile and violate FD ToS. The CSV export from the contest lobby is the cleanest path and takes ~30 seconds.
            """)

    st.markdown("---")

    # ── COMPUTE FD PROJECTIONS ─────────────────────────────────────────────
    fd_plays = []
    for p in plays:
        # Rebuild minimal stat dicts from stored play data
        bat_mock = {
            "k_rate":         p.get("k_rate",         0.228),
            "bb_rate":        p.get("bb_rate",         0.082),
            "slg_proxy":      p.get("xslg",            0.398),
            "iso_proxy":      p.get("iso",             0.165),
            "woba":           p.get("xslg", 0.398) * 0.78,
            "hard_hit_rate":  p.get("hard_hit_rate",  0.370),
            "barrel_rate":    p.get("barrel_rate",    0.070),
            "sprint_speed":   p.get("sprint_speed",    27.0),
        }
        pit_mock = {
            "k_rate_allowed":  0.220,
            "hard_hit_allowed":0.370,
            "fip":             4.10,
        }
        try:
            pit_label = p.get("pitcher_label", "")
            if "K%:" in pit_label:
                pit_mock["k_rate_allowed"] = float(pit_label.split("K%:")[1].split("%")[0].strip()) / 100
            if "FIP:" in pit_label:
                pit_mock["fip"] = float(pit_label.split("FIP:")[1].strip().split()[0])
        except Exception:
            pass

        fd_p = compute_fd_projection(
            statcast=bat_mock,
            pitcher_statcast=pit_mock,
            lineup_slot=p.get("lineup_slot", 5),
            implied_total=p.get("implied_total", 0),
            batter_hand=p.get("batter_hand", "R"),
            sp_hand=p.get("sp_hand", "R"),
            park_team=p.get("park", p.get("team", "")),
            weather=p.get("weather", {}),
        )

        # Match salary — also use stored batter position as fallback
        salary = 0
        # Map MLB position abbreviations to FD positions
        pos_map = {
            "C":"C","1B":"1B","2B":"2B","3B":"3B","SS":"SS",
            "LF":"OF","CF":"OF","RF":"OF","OF":"OF","DH":"UTIL",
            "P":"P","SP":"P","RP":"P",
        }
        raw_pos = p.get("batter_position", "") or "OF"
        position = pos_map.get(raw_pos.upper(), "OF")

        for sal_name, sal_data in salary_data.items():
            if _norm(sal_name) == _norm(p["name"]) or _norm(p["name"]) in _norm(sal_name):
                salary   = sal_data["salary"]
                # FD salary CSV position takes precedence over stored position
                position = sal_data["position"]
                break

        value = round(fd_p["fd_proj"] / (salary / 1000), 2) if salary > 0 else 0.0
        ownership = compute_ownership_projection(
            fd_proj=fd_p["fd_proj"],
            salary=salary,
            implied_total=p.get("implied_total", 0),
            lineup_slot=p.get("lineup_slot", 5),
            barrel_rate=p.get("barrel_rate", 0.07),
            name=p["name"],
        )

        fd_plays.append({
            **p,
            **fd_p,
            "fd_salary":   salary,
            "fd_position": position,
            "fd_value":    value,
            "ownership":   ownership,
        })

    fd_plays.sort(key=lambda x: x["fd_proj"], reverse=True)

    # ── PROJECTIONS TABLE ─────────────────────────────────────────────────
    st.subheader("📊 FD Projections — All Batters")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        min_proj = st.slider("Min projection", 0.0, 40.0, 5.0, 0.5, key="fd_min_proj")
    with col_f2:
        teams_fd = sorted(set(p["team"] for p in fd_plays))
        team_fd_filter = st.multiselect("Filter team", teams_fd, default=[], key="fd_team_filter")

    filtered_fd = [p for p in fd_plays
                   if p["fd_proj"] >= min_proj
                   and (not team_fd_filter or p["team"] in team_fd_filter)]

    rows = []
    for p in filtered_fd:
        tbd = " ⚠️" if p.get("sp_tbd") else ""
        rows.append({
            "FD Proj": f"{p['fd_proj']:.1f}",
            "Ceiling": f"{p['fd_ceiling']:.1f}",
            "Floor":   f"{p['fd_floor']:.1f}",
            "Player":  p["name"],
            "Pos":     p.get("fd_position", "—"),
            "Team":    p["team"],
            "Sal":     f"${p['fd_salary']:,}" if p["fd_salary"] > 0 else "—",
            "Value":   f"{p['fd_value']:.1f}x" if p["fd_value"] > 0 else "—",
            "Own%":    f"{p['ownership']:.0f}%",
            "Slot":    f"#{p['lineup_slot']}",
            "Opp SP":  p["sp_name"][:16] + tbd,
            "xSLG":    f"{p.get('xslg',0):.3f}" if p.get("xslg") else "—",
            "K%":      f"{p.get('k_rate',0)*100:.1f}%" if p.get("k_rate") else "—",
            "Park":    p.get("park",""),
            "O1.5 Sc": f"{p['score']:.0f}",
        })

    if rows:
        df_fd = pd.DataFrame(rows)

        def color_proj(val):
            try:
                v = float(str(val))
                if v >= 20: return "color: #00ff88; font-weight: bold"
                elif v >= 14: return "color: #ffdd00; font-weight: bold"
                elif v >= 9: return "color: #ff8800"
                return ""
            except: return ""

        def color_own(val):
            try:
                v = float(str(val).replace("%",""))
                if v <= 10: return "color: #00ff88"   # contrarian = green
                elif v >= 35: return "color: #ff4444"  # chalk = red
                return ""
            except: return ""

        def color_val(val):
            try:
                v = float(str(val).replace("x",""))
                if v >= 4.0: return "color: #00ff88; font-weight: bold"
                elif v >= 3.0: return "color: #ffdd00"
                return ""
            except: return ""

        styled = (df_fd.style
                  .map(color_proj, subset=["FD Proj","Ceiling"])
                  .map(color_own,  subset=["Own%"])
                  .map(color_val,  subset=["Value"]))
        st.dataframe(styled, use_container_width=True, height=450)

        csv_fd = df_fd.to_csv(index=False)
        st.download_button("📥 Export FD Projections", csv_fd,
                           f"fd_proj_{datetime.now(EST).strftime('%Y%m%d')}.csv", "text/csv")

    st.markdown("---")

    # ── GAME STACKS ───────────────────────────────────────────────────────
    st.subheader("🔥 Top Game Stacks")
    st.caption("GPP: stack 4-5 batters from the same high-total game for correlated upside")

    # Group by game and score
    game_groups = {}
    for p in fd_plays:
        gid = p.get("game_id", "")
        team = p.get("team","")
        key = f"{gid}_{team}"
        if key not in game_groups:
            game_groups[key] = {
                "game":    f"{p.get('opponent','')}@{team}",
                "team":    team,
                "implied": p.get("implied_total", 0),
                "players": [],
            }
        game_groups[key]["players"].append(p)

    # Score each stack
    stack_scores = []
    for key, grp in game_groups.items():
        players = sorted(grp["players"], key=lambda x: x["fd_proj"], reverse=True)
        top4 = players[:4]
        if len(top4) < 3:
            continue
        avg_proj = sum(p["fd_proj"] for p in top4) / len(top4)
        avg_ceil = sum(p["fd_ceiling"] for p in top4) / len(top4)
        stack_score = avg_proj * 0.5 + avg_ceil * 0.3 + grp["implied"] * 2
        stack_scores.append({
            "game":    grp["game"],
            "team":    grp["team"],
            "implied": grp["implied"],
            "top4":    top4,
            "avg_proj":  round(avg_proj, 1),
            "avg_ceil":  round(avg_ceil, 1),
            "score":     round(stack_score, 1),
        })

    stack_scores.sort(key=lambda x: x["score"], reverse=True)

    for i, stack in enumerate(stack_scores[:3], 1):
        imp_str = f"{stack['implied']:.1f}" if stack["implied"] > 0 else "—"
        with st.expander(f"#{i} Stack: {stack['team']} ({stack['game']}) — Implied: {imp_str} runs | Avg Proj: {stack['avg_proj']:.1f} pts"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**4-man stack ({stack['team']}):**")
                for p in stack["top4"][:4]:
                    sal_str = f"${p['fd_salary']:,}" if p["fd_salary"] > 0 else "no salary"
                    own_str = f"{p['ownership']:.0f}% own"
                    st.write(f"• **{p['name']}** #{p['lineup_slot']} — {p['fd_proj']:.1f} proj | {sal_str} | {own_str}")
            with col_b:
                st.markdown("**Bring-back targets (opp team):**")
                opp_team = stack["top4"][0].get("opponent","") if stack["top4"] else ""
                bring_backs = [p for p in fd_plays if p["team"] == opp_team
                               and p.get("implied_total", 0) > 3.5][:3]
                for p in bring_backs:
                    sal_str = f"${p['fd_salary']:,}" if p["fd_salary"] > 0 else "no salary"
                    st.write(f"• {p['name']} — {p['fd_proj']:.1f} proj | {sal_str}")

    st.markdown("---")

    # ── GPP SECTION ──────────────────────────────────────────────────────
    st.subheader("🎯 GPP Lineups (1-3 entries)")
    st.caption("**GPP philosophy:** Maximize ceiling. Stack correlated batters. Leverage low-owned plays. Accept variance.")

    # GPP strategy notes
    with st.expander("📖 GPP Strategy Guide"):
        st.markdown("""
**Core GPP Principles:**
- **Stack 4-5 batters** from same team in high-total game — correlated upside when team scores big
- **Bring-back 1-2 batters** from the opposing team (wrap stack) — you win regardless of which team scores
- **Avoid mega-chalk** (>35% owned) on lineups 2-3 — differentiation wins GPPs
- **Target 15-25% owned** plays that have legit upside — the sweet spot

**FanDuel vs DraftKings ownership gaps:**
- High DK ownership players are often over-owned on FD — fade them
- FD-specific value: contact hitters (BB count), speedsters (SB = 6 pts)
- SP value: FD rewards Ks heavily — high-K SP in a weak lineup is GPP gold

**SP fade triggers:**
- Team implied total < 3.8 (low run environment)
- Pitcher FIP > 5.0 (hittable, SP likely gets lit up)
- Park with HR factor > 1.15 (against SP)
- Heavy tailwind blowing out

**Stack selection priority:**
1. Game total > 9.5 (both teams score)
2. Team implied > 5.0 (your stack team favored to score)
3. Weak opposing SP (FIP > 4.5, K% < 20%)
4. Hitter-friendly park (Coors, GABP, Camden)
        """)

    if has_salaries:
        gpp_lineups = build_fd_lineups(fd_plays, num_lineups=3, mode="gpp")
        if gpp_lineups:
            for i, lu in enumerate(gpp_lineups, 1):
                with st.expander(f"GPP Lineup #{i} — Proj: {lu['total_proj']:.1f} | Ceil: {lu['total_ceiling']:.1f} | Sal: ${lu['total_salary']:,}"):
                    rows_lu = []
                    for p in lu["players"]:
                        rows_lu.append({
                            "Slot": p.get("slot",""),
                            "Player": p["name"],
                            "Team": p["team"],
                            "Pos": p.get("fd_position",""),
                            "Salary": f"${p.get('fd_salary',0):,}",
                            "Proj": f"{p['fd_proj']:.1f}",
                            "Ceil": f"{p['fd_ceiling']:.1f}",
                            "Own%": f"{p.get('ownership',0):.0f}%",
                            "Opp SP": p.get("sp_name","")[:18],
                        })
                    st.dataframe(pd.DataFrame(rows_lu), use_container_width=True)
                    stacks = lu.get("stacks", {})
                    if stacks:
                        stack_str = " | ".join([f"{t}: {c}-man" for t, c in stacks.items()])
                        st.caption(f"🔗 Stacks: {stack_str}")
                    # FD export format
                    fd_export = ",".join(p["name"] for p in lu["players"])
                    st.code(f"FD Import: {fd_export}", language=None)
        else:
            st.info("Need salaries loaded to build optimized lineups.")
    else:
        st.info("📥 Load FanDuel salaries above to generate optimized GPP lineups.")

    st.markdown("---")

    # ── CASH SECTION ─────────────────────────────────────────────────────
    st.subheader("💵 Cash Lineups (50/50 & H2H)")
    st.caption("**Cash philosophy:** Maximize floor. High-floor bats. Consistent SP. Avoid low-PA slots.")

    with st.expander("📖 Cash Strategy Guide"):
        st.markdown("""
**Cash Game Rules:**
- **Floor > ceiling** — you need 50th percentile, not 99th
- **Target confirmed top-3 lineup slots** — guaranteed 4.5+ PA/game
- **Avoid low-total games** — implied < 4.0 = fewer PA, fewer hits
- **SP must be elite** — FIP < 3.5, K% > 25%, high strikeout game script
- **No punt plays** — minimum viable salary on every spot (~$2,800+)
- **Stack limit: 3 max** — correlation adds variance you don't want

**High-floor indicators:**
- wRC+ > 120 + low K% = almost always contributes
- Leadoff/2-hole in high-implied lineup = most PA
- RHB vs LHP platoon advantage = +33 SLG points
- High BB% + low K% = PA completion = FD points even without hits
        """)

    if has_salaries:
        cash_lineups = build_fd_lineups(fd_plays, num_lineups=1, mode="cash")
        if cash_lineups:
            lu = cash_lineups[0]
            st.write(f"**Cash Lineup — Proj: {lu['total_proj']:.1f} | Floor: {lu['total_floor']:.1f} | Sal: ${lu['total_salary']:,}**")
            rows_lu = []
            for p in lu["players"]:
                rows_lu.append({
                    "Slot": p.get("slot",""),
                    "Player": p["name"],
                    "Team": p["team"],
                    "Pos": p.get("fd_position",""),
                    "Salary": f"${p.get('fd_salary',0):,}",
                    "Proj": f"{p['fd_proj']:.1f}",
                    "Floor": f"{p['fd_floor']:.1f}",
                    "Own%": f"{p.get('ownership',0):.0f}%",
                })
            st.dataframe(pd.DataFrame(rows_lu), use_container_width=True)
            fd_export = ",".join(p["name"] for p in lu["players"])
            st.code(f"FD Import: {fd_export}", language=None)
        else:
            st.info("Need salaries loaded to build cash lineup.")
    else:
        st.info("📥 Load FanDuel salaries above to generate cash lineup.")

    # ── SP TARGETS ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚾ SP Targets & Projections")
    st.caption("FD SP scoring: W=6 | QS=4 | K=3 | IP×3 | ER×-3 | H×-0.6 | BB×-0.6")

    # SP salary input
    with st.expander("💵 SP Salary Input (paste Name, Salary)"):
        sp_sal_text = st.text_area(
            "SP salaries from FD CSV:",
            placeholder="Paul Skenes, 10500\nGarrett Crochet, 9800\nCade Cavalli, 7200",
            height=100, key="fd_sp_salary_text"
        )
    sp_salary_data = {}
    if sp_sal_text.strip():
        for line in sp_sal_text.strip().split("\n"):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2:
                try:
                    sp_salary_data[_norm(parts[0])] = {"name": parts[0], "salary": int(parts[1].replace("$","").replace(",",""))}
                except Exception:
                    pass

    # Build SP projections from pitching data
    pitchers_seen = {}
    for p in plays:
        sp = p.get("sp_name", "TBD")
        if sp and sp != "TBD":
            if sp not in pitchers_seen:
                pitchers_seen[sp] = {
                    "name": sp, "hand": p.get("sp_hand","R"),
                    "team": p.get("opponent",""), "opp": p.get("team",""),
                    "park": p.get("park",""), "implied": p.get("implied_total",0),
                    "pit_label": p.get("pitcher_label",""),
                    "weather": p.get("weather",{}), "batters": [],
                }
            pitchers_seen[sp]["batters"].append(p)

    def project_fd_sp(sp_data: Dict) -> Dict:
        """
        Project FanDuel SP fantasy points.
        FD scoring: W=6, QS=4, K=3, IP×3, ER×-3, H×-0.6, BB×-0.6
        Typical SP: 5-6 IP, 5-7 K, 2-3 ER → ~20-30 FD pts
        Elite SP: 7+ IP, 8+ K, 0-1 ER → 40-55 FD pts
        """
        pl = sp_data.get("pit_label", "")
        batters = sp_data.get("batters", [])

        # Extract stats from pitcher label
        k_rate, fip, whip = 0.228, 4.10, 1.30
        try:
            if "K%:" in pl:
                k_rate = float(pl.split("K%:")[1].split("%")[0].strip()) / 100
            if "FIP:" in pl:
                fip = float(pl.split("FIP:")[1].strip().split()[0])
            if "WHIP:" in pl:
                whip = float(pl.split("WHIP:")[1].strip().split()[0])
        except Exception:
            pass

        opp_implied = sum(b.get("implied_total",0) for b in batters) / max(1,len(batters))
        opp_k = sum(b.get("k_rate",0.228) for b in batters) / max(1,len(batters))

        # Project IP: elite pitchers go deeper
        # FIP <3.0 → 6.5 IP avg, FIP 3-4 → 5.8 IP, FIP 4-5 → 5.2 IP, FIP 5+ → 4.5 IP
        if fip < 3.0:   proj_ip = 6.5
        elif fip < 3.5: proj_ip = 6.2
        elif fip < 4.0: proj_ip = 5.8
        elif fip < 4.5: proj_ip = 5.4
        else:           proj_ip = 4.8

        # Adjust for opponent implied total (weak opp = deeper outing)
        if opp_implied > 0:
            proj_ip *= max(0.85, 1.0 - (opp_implied - 4.5) * 0.04)

        # Project Ks: k_rate × batters faced (≈ IP × 3.3)
        bf = proj_ip * 3.3
        proj_k = k_rate * bf

        # Project ER from FIP (FIP is an ERA estimator)
        proj_er = (fip / 9) * proj_ip
        proj_er = max(0, proj_er)

        # Project H and BB from WHIP
        proj_h_bb = whip * proj_ip
        proj_h  = proj_h_bb * 0.70
        proj_bb = proj_h_bb * 0.30

        # Win probability (rough — home pitcher, low FIP, low opp implied)
        win_prob = 0.50
        if fip < 3.5 and opp_implied < 4.0: win_prob = 0.65
        elif fip > 4.5 or opp_implied > 5.0: win_prob = 0.35

        # QS probability (6+ IP, ≤3 ER)
        qs_prob = 0.70 if proj_ip >= 5.8 and proj_er <= 3.0 else 0.35

        # FD points
        fd_pts = (
            proj_k    * 3.0 +
            proj_ip   * 3.0 +
            proj_er   * -3.0 +
            proj_h    * -0.6 +
            proj_bb   * -0.6 +
            win_prob  * 6.0 +
            qs_prob   * 4.0
        )

        # Ceiling and floor
        variance = fd_pts * 0.40
        ceiling  = fd_pts + variance
        floor    = max(0, fd_pts - variance * 0.6)

        # FD grade
        if fip < 3.2 and opp_implied < 4.0:
            grade = "🔒 ACE"
        elif fip < 3.8 and opp_implied < 4.5:
            grade = "✅ TARGET"
        elif opp_implied > 5.0 or fip > 4.8:
            grade = "❌ FADE"
        else:
            grade = "⚠️ RISKY"

        return {
            "fd_sp_proj":    round(fd_pts, 1),
            "fd_sp_ceiling": round(ceiling, 1),
            "fd_sp_floor":   round(floor, 1),
            "proj_ip":       round(proj_ip, 1),
            "proj_k":        round(proj_k, 1),
            "proj_er":       round(proj_er, 1),
            "win_prob":      round(win_prob * 100, 0),
            "qs_prob":       round(qs_prob * 100, 0),
            "fip":           fip,
            "k_rate":        k_rate,
            "opp_implied":   opp_implied,
            "grade":         grade,
        }

    sp_data_list = []
    for sp_name, sp_data in pitchers_seen.items():
        proj = project_fd_sp(sp_data)
        sal_match = sp_salary_data.get(_norm(sp_name), {})
        salary = sal_match.get("salary", 0)
        value  = round(proj["fd_sp_proj"] / (salary / 1000), 2) if salary > 0 else 0.0
        sp_data_list.append({
            **sp_data, **proj,
            "fd_sp_salary": salary,
            "fd_sp_value":  value,
        })

    sp_data_list.sort(key=lambda x: x["fd_sp_proj"], reverse=True)

    if sp_data_list:
        # Top pick and value pick callouts
        top_sp    = sp_data_list[0]
        value_sp  = max([s for s in sp_data_list if s["fd_sp_salary"] > 0],
                        key=lambda x: x["fd_sp_value"]) if any(s["fd_sp_salary"] > 0 for s in sp_data_list) else None
        targets   = [s for s in sp_data_list if "ACE" in s["grade"] or "TARGET" in s["grade"]]
        fades     = [s for s in sp_data_list if "FADE" in s["grade"]]

        col_ts1, col_ts2 = st.columns(2)
        with col_ts1:
            st.markdown(f"**⭐ Top SP Target:** {top_sp['name']} ({top_sp['team']}) "
                        f"— {top_sp['fd_sp_proj']:.1f} proj | Ceil: {top_sp['fd_sp_ceiling']:.1f} | "
                        f"FIP: {top_sp['fip']:.2f} | {top_sp['proj_k']:.0f} proj K")
        with col_ts2:
            if value_sp:
                st.markdown(f"**💎 Top Value SP:** {value_sp['name']} ({value_sp['team']}) "
                            f"— {value_sp['fd_sp_proj']:.1f} proj | ${value_sp['fd_sp_salary']:,} | "
                            f"Value: {value_sp['fd_sp_value']:.1f}x")
            else:
                if fades:
                    st.markdown(f"**❌ Top Fade:** {fades[0]['name']} — opp implied {fades[0]['opp_implied']:.1f}, FIP {fades[0]['fip']:.2f}")

        # Full SP table
        sp_rows = []
        for s in sp_data_list:
            sp_rows.append({
                "Grade":    s["grade"],
                "SP":       s["name"],
                "Hand":     s["hand"],
                "Opp":      s["opp"],
                "Park":     s["park"],
                "FD Proj":  f"{s['fd_sp_proj']:.1f}",
                "Ceiling":  f"{s['fd_sp_ceiling']:.1f}",
                "Floor":    f"{s['fd_sp_floor']:.1f}",
                "FIP":      f"{s['fip']:.2f}",
                "K%":       f"{s['k_rate']*100:.0f}%",
                "Proj K":   f"{s['proj_k']:.0f}",
                "Proj IP":  f"{s['proj_ip']:.1f}",
                "Win%":     f"{s['win_prob']:.0f}%",
                "QS%":      f"{s['qs_prob']:.0f}%",
                "Opp Imp":  f"{s['opp_implied']:.1f}" if s['opp_implied'] > 0 else "—",
                "Salary":   f"${s['fd_sp_salary']:,}" if s["fd_sp_salary"] > 0 else "—",
                "Value":    f"{s['fd_sp_value']:.1f}x" if s["fd_sp_value"] > 0 else "—",
            })

        sp_df = pd.DataFrame(sp_rows)

        def color_sp_grade(val):
            if "ACE" in str(val):    return "color: #00ff88; font-weight: bold"
            if "TARGET" in str(val): return "color: #66dd88; font-weight: bold"
            if "FADE" in str(val):   return "color: #ff4444; font-weight: bold"
            if "RISKY" in str(val):  return "color: #ffdd00"
            return ""

        def color_sp_proj(val):
            try:
                v = float(str(val))
                if v >= 35: return "color: #00ff88; font-weight: bold"
                elif v >= 25: return "color: #ffdd00"
                elif v < 15: return "color: #ff4444"
                return ""
            except: return ""

        def color_sp_val(val):
            try:
                v = float(str(val).replace("x",""))
                if v >= 4.0: return "color: #00ff88; font-weight: bold"
                elif v >= 3.0: return "color: #ffdd00"
                return ""
            except: return ""

        styled_sp = (sp_df.style
                     .map(color_sp_grade, subset=["Grade"])
                     .map(color_sp_proj,  subset=["FD Proj","Ceiling"])
                     .map(color_sp_val,   subset=["Value"]))
        st.dataframe(styled_sp, use_container_width=True)

    # ── VALUE PLAYS ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💎 Value Plays (Under $2,500)")
    st.caption("High projection relative to salary — GPP tournament differentiators")

    if has_salaries:
        values = [p for p in fd_plays
                  if 0 < p["fd_salary"] <= 2500 and p["fd_proj"] >= 6.0]
        values.sort(key=lambda x: x["fd_value"], reverse=True)

        if values:
            for p in values[:5]:
                st.markdown(
                    f"**{p['name']}** ({p['team']}, #{p['lineup_slot']}) — "
                    f"${p['fd_salary']:,} | Proj: {p['fd_proj']:.1f} | "
                    f"Value: {p['fd_value']:.1f}x | Own: {p['ownership']:.0f}%"
                )
        else:
            st.info("No value plays found under $2,500 with proj ≥ 6.0. Load salaries to see values.")
    else:
        st.info("Load salaries to see value plays.")



# ============================================================================
# PRIZEPICKS ENGINE
# ============================================================================

# PrizePicks MLB Hitter Fantasy Score (confirmed from official scoring)
PP_SCORING = {
    "single":  3.0,
    "double":  5.0,
    "triple":  8.0,
    "hr":     10.0,   # Note: 10 on PP vs 12 on FD
    "run":     2.0,   # Note: 2 on PP vs 3.2 on FD
    "rbi":     2.0,   # Note: 2 on PP vs 3.5 on FD
    "bb":      2.0,   # Note: 2 on PP vs 3 on FD
    "hbp":     2.0,
    "sb":      5.0,   # Note: 5 on PP vs 6 on FD
    # No out penalty on PP (vs -0.25 on FD)
}

# PP payout multipliers
PP_PAYOUTS = {
    "power": {2: 3.0, 3: 5.0, 4: 10.0, 5: 20.0, 6: 25.0},
    "flex":  {2: 3.0, 3: 2.25, 4: 5.0, 5: 10.0, 6: 25.0},  # flex allows 1-2 misses
}


def compute_pp_projection(statcast: Dict, pitcher_statcast: Dict,
                          lineup_slot: int, implied_total: float,
                          batter_hand: str, sp_hand: str,
                          park_team: str, weather: Dict) -> Dict:
    """
    Project PrizePicks hitter fantasy score.
    Identical pipeline to FD but uses PP scoring values.
    Key differences: no out penalty, lower HR/RBI/Run/BB/SB values.
    """
    def f(key, default):
        try: return float(statcast.get(key, default))
        except: return float(default)

    # PA estimate by slot
    pa_by_slot = {1:4.8,2:4.7,3:4.6,4:4.5,5:4.3,6:4.2,7:4.1,8:3.9,9:3.8}
    est_pa = pa_by_slot.get(lineup_slot, 4.2)
    if implied_total > 0:
        est_pa += (implied_total - 4.5) * 0.08

    k_rate     = f("k_rate",        0.228)
    bb_rate    = f("bb_rate",       0.082)
    slg        = f("slg_proxy",     0.398)
    iso        = f("iso_proxy",     0.165)
    woba       = f("woba",          0.315)
    hard_hit   = f("hard_hit_rate", 0.370)
    barrel     = f("barrel_rate",   0.070)
    ev50       = f("ev50",          95.0)    # V1.6: hardest 50% EV for power proj
    bat_speed  = f("bat_speed",     71.0)    # V1.6: bat tracking
    blast_rate = f("blast_rate",    0.21)    # V1.6: swing quality
    sprint_spd = f("sprint_speed",  27.0)    # V1.6: Savant sprint speed (ft/sec)

    # V1.6: Sprint speed-based SB rate (replaces ISO proxy — much more accurate)
    # <24 ft/s = no threat (0.01/game), 27 = avg (0.05), 29+ = elite (0.12+)
    if sprint_spd > 0 and sprint_spd < 10:
        sprint_spd = sprint_spd * 3.28  # m/s to ft/s conversion if needed
    sb_rate = max(0.01, min(0.18,
        0.01 + max(0, sprint_spd - 24.0) * 0.017  # ~0.017/game per ft/s above 24
    ))

    # V1.6: EV50-adjusted HR rate — EV50 > 98 = elite HR ceiling
    ev50_hr_boost = max(0, (ev50 - 95.0) / 100.0 * 0.03)  # up to +3% HR/PA for elite EV50

    hit_rate   = max(0.180, woba * 0.85)
    # V1.6: EV50 boost on HR rate — elite EV50 = more deep fly balls = more HRs
    hr_per_pa  = barrel * 0.35 + ev50_hr_boost
    # V1.6: blast_rate adjusts XBH rate — better swing quality = more extra bases
    blast_xbh_boost = max(0, (blast_rate - 0.21) * 0.15)  # up to ~3% at elite blast
    xbh_per_pa = iso * 0.6 * (1 - k_rate) + blast_xbh_boost
    single_per_pa = max(0, hit_rate - hr_per_pa - xbh_per_pa)
    doubles_per_pa = xbh_per_pa * 0.65
    triples_per_pa = xbh_per_pa * 0.05

    # Park adjustments
    park_hr = PARK_HR_FACTORS.get(park_team, 1.0)
    park_tb = PARK_TB_FACTORS.get(park_team, 1.0)
    hr_per_pa  *= park_hr
    single_per_pa *= park_tb

    # Weather
    if not weather.get("is_dome"):
        we = weather.get("wind_effect", "neutral")
        if we == "strong_out": hr_per_pa *= 1.25
        elif we == "out":      hr_per_pa *= 1.15
        elif we == "in":       hr_per_pa *= 0.80

    # Pitcher quality — V1.6: blend in pitch matchup score as an additional multiplier
    pit_k   = float(pitcher_statcast.get("k_rate_allowed", 0.228))
    pit_fip = float(pitcher_statcast.get("fip", 4.10))
    quality_adj = 1.0 + (pit_fip - 4.0) * 0.04
    k_adj       = 1.0 - (pit_k - 0.228) * 1.5
    # pitch_matchup: 50=neutral(1.0), 70=favorable(1.04), 30=unfavorable(0.96)
    matchup_sc = float(pitcher_statcast.get("_matchup_score", 50.0))
    matchup_adj = 1.0 + (matchup_sc - 50.0) / 50.0 * 0.05  # ±5% on total output
    adj = (quality_adj + k_adj) / 2 * matchup_adj

    # Scale to PA
    proj_hr      = hr_per_pa * est_pa * adj
    proj_singles = single_per_pa * est_pa * adj
    proj_doubles = doubles_per_pa * est_pa * adj
    proj_triples = triples_per_pa * est_pa * adj
    proj_bb      = bb_rate * est_pa
    proj_sb      = sb_rate * est_pa

    # RBI and Runs
    rbi_rate = (0.32 if lineup_slot <= 4 else 0.22) * (implied_total / 4.5 if implied_total > 0 else 1.0)
    run_rate = (0.38 if lineup_slot <= 3 else 0.28 if lineup_slot <= 6 else 0.20) * (implied_total / 4.5 if implied_total > 0 else 1.0)
    proj_rbi  = proj_singles * rbi_rate + proj_doubles * rbi_rate * 1.3 + proj_hr * 1.0
    proj_runs = (proj_singles + proj_doubles + proj_hr) * run_rate + proj_hr * 1.0

    # PrizePicks fantasy score
    pp_pts = (
        proj_singles * PP_SCORING["single"] +
        proj_doubles * PP_SCORING["double"] +
        proj_triples * PP_SCORING["triple"] +
        proj_hr      * PP_SCORING["hr"] +
        proj_rbi     * PP_SCORING["rbi"] +
        proj_runs    * PP_SCORING["run"] +
        proj_bb      * PP_SCORING["bb"] +
        proj_sb      * PP_SCORING["sb"]
        # No out penalty on PrizePicks
    )

    # Ceiling (high game) and floor (low game)
    variance = pp_pts * 0.55  # PP has high game-to-game variance
    ceiling  = pp_pts + variance
    floor    = max(0, pp_pts - variance * 0.55)

    return {
        "pp_proj":    round(pp_pts, 1),
        "pp_ceiling": round(ceiling, 1),
        "pp_floor":   round(floor, 1),
        "est_pa":     round(est_pa, 1),
        "proj_hr":    round(proj_hr, 2),
        "proj_hits":  round(proj_singles + proj_doubles + proj_triples + proj_hr, 2),
        "proj_bb":    round(proj_bb, 2),
        "proj_sb":    round(proj_sb, 2),
    }


def compute_pp_ev(proj: float, line: float, mode: str = "power", legs: int = 2) -> Dict:
    """
    Calculate EV for a PrizePicks More/Less pick.
    Uses normal approximation for win probability given projection vs line.
    """
    import math
    # Win probability: how often does player exceed the line?
    # Use logistic approximation: larger edge = higher win prob
    edge = proj - line
    # Scale: 1 point edge ~ 8% win probability shift from 50%
    z = edge / max(0.5, proj * 0.25)  # normalize by projection scale
    win_prob = 1 / (1 + math.exp(-z * 1.5))
    win_prob = max(0.30, min(0.78, win_prob))  # realistic MLB prop range

    payout = PP_PAYOUTS.get(mode, PP_PAYOUTS["power"]).get(legs, 3.0)
    # EV = win_prob * payout - (1 - win_prob) * 1.0
    ev = win_prob * payout - (1 - win_prob)
    ev_pct = (ev - 1) * 100  # as % above breakeven

    direction = "MORE" if proj > line else "LESS"
    if proj < line:
        win_prob = 1 - win_prob  # flip for LESS

    return {
        "direction": direction,
        "win_prob":  round(win_prob * 100, 1),
        "ev_pct":    round(ev_pct, 1),
        "edge_pts":  round(edge, 1),
        "payout":    payout,
    }


def display_prizepicks_tab(plays: List[Dict]):
    """
    PrizePicks projection tab — projection-first, no line input required.
    Shows PP fantasy score projections ranked and color-coded.
    User manually picks More/Less based on visual data.
    """
    st.header("🎯 PrizePicks — Hitter Fantasy Score")
    st.caption("PP Scoring: 1B=3 | 2B=5 | 3B=8 | HR=10 | R=2 | RBI=2 | BB=2 | SB=5 | No out penalty")

    if not plays:
        st.info("Run the model first — PP projections auto-generate from the same data.")
        return

    # ── COMPUTE PP PROJECTIONS ────────────────────────────────────────────
    pp_plays = []
    for p in plays:
        bat_mock = {
            "k_rate":          p.get("k_rate",          0.228),
            "bb_rate":         p.get("bb_rate",          0.082),
            "slg_proxy":       p.get("xslg",             0.398),
            "iso_proxy":       p.get("iso",              0.165),
            "woba":            p.get("xslg", 0.398) * 0.78,
            "hard_hit_rate":   p.get("hard_hit_rate",    0.370),
            "barrel_rate":     p.get("barrel_rate",      0.070),
            "sweet_spot_rate": p.get("sweet_spot_rate",  0.305),
            # V1.6: Savant bat tracking + EV50 + sprint speed
            "ev50":            p.get("ev50",             95.0),
            "bat_speed":       p.get("bat_speed",        71.0),
            "blast_rate":      p.get("blast_rate",       0.21),
            "sprint_speed":    p.get("sprint_speed",     27.0),
        }
        pit_mock = {
            "k_rate_allowed":   0.228,
            "fip":              4.20,
            "_matchup_score":   p.get("sub_matchup", 50.0),  # V1.6
        }
        try:
            pl = p.get("pitcher_label", "")
            if "K%:" in pl:
                pit_mock["k_rate_allowed"] = float(pl.split("K%:")[1].split("%")[0].strip()) / 100
            if "FIP:" in pl:
                pit_mock["fip"] = float(pl.split("FIP:")[1].strip().split()[0])
        except Exception:
            pass

        pp_p = compute_pp_projection(
            statcast=bat_mock, pitcher_statcast=pit_mock,
            lineup_slot=p.get("lineup_slot", 5),
            implied_total=p.get("implied_total", 0),
            batter_hand=p.get("batter_hand", "R"),
            sp_hand=p.get("sp_hand", "R"),
            park_team=p.get("park", p.get("team", "")),
            weather=p.get("weather", {}),
        )
        pp_plays.append({**p, **pp_p})

    pp_plays.sort(key=lambda x: x["pp_proj"], reverse=True)

    # ── FILTERS ────────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        teams_pp = sorted(set(p["team"] for p in pp_plays))
        team_filter = st.multiselect("Filter team", teams_pp, default=[], key="pp_team_f")
    with col_f2:
        min_proj = st.slider("Min PP projection", 0.0, 20.0, 4.0, 0.5, key="pp_min_proj")
    with col_f3:
        hand_filter = st.multiselect("Batter hand", ["L","R","S","B"], default=[], key="pp_hand_f")
    with col_f4:
        slot_max = st.slider("Max lineup slot", 1, 9, 6, key="pp_slot_max")

    filtered = [p for p in pp_plays
                if p["pp_proj"] >= min_proj
                and (not team_filter or p["team"] in team_filter)
                and (not hand_filter or p.get("batter_hand","R") in hand_filter)
                and p.get("lineup_slot", 9) <= slot_max]

    # ── TOP PLAYS CALLOUTS ─────────────────────────────────────────────────
    top3 = filtered[:3]
    if top3:
        cols = st.columns(len(top3))
        for i, (col, p) in enumerate(zip(cols, top3)):
            with col:
                medal = ["🥇","🥈","🥉"][i]
                st.metric(
                    label=f"{medal} {p['name']} ({p['team']})",
                    value=f"{p['pp_proj']:.1f} pts",
                    delta=f"Ceil: {p['pp_ceiling']:.1f} | Floor: {p['pp_floor']:.1f}"
                )
                st.caption(f"vs {p['sp_name']} | #{p['lineup_slot']} | {p.get('batter_hand','?')}HB")

    st.markdown("---")

    # ── MAIN PROJECTIONS TABLE ─────────────────────────────────────────────
    st.markdown(f"**Showing {len(filtered)} players**")

    rows = []
    for p in filtered:
        tbd = " ⚠️" if p.get("sp_tbd") else ""
        # Confidence tier based on projection vs floor spread
        spread = p["pp_ceiling"] - p["pp_floor"]
        conf = "🔒 High" if spread < 8 and p["pp_proj"] > 9 else "✅ Med" if p["pp_proj"] > 7 else "📊 Low"
        rows.append({
            "PP Proj":    f"{p['pp_proj']:.1f}",
            "Ceiling":    f"{p['pp_ceiling']:.1f}",
            "Floor":      f"{p['pp_floor']:.1f}",
            "Conf":       conf,
            "Player":     p["name"],
            "Team":       p["team"],
            "Hand":       p.get("batter_hand","?"),
            "Slot":       f"#{p['lineup_slot']}",
            "Opp SP":     p["sp_name"][:18] + tbd,
            "SP✋":        p.get("sp_hand","?"),
            "Platoon":    p.get("platoon_label","").split("(")[0].strip(),
            "xSLG":       f"{p.get('xslg',0):.3f}" if p.get("xslg") else "—",
            "Barrel%":    f"{p.get('barrel_rate',0)*100:.1f}%" if p.get("barrel_rate") else "—",
            "K%":         f"{p.get('k_rate',0)*100:.1f}%" if p.get("k_rate") else "—",
            "Park":       p.get("park",""),
            "Wind":       p.get("weather_label","").split("|")[0].strip()[:12] if p.get("weather_label") else "—",
            "Imp.R":      f"{p.get('implied_total',0):.1f}" if p.get("implied_total",0)>0 else "—",
            "O1.5 Sc":    f"{p['score']:.0f}",
        })

    if rows:
        df_pp = pd.DataFrame(rows)

        def color_proj(val):
            try:
                v = float(str(val))
                if v >= 12: return "color: #00ff88; font-weight: bold"
                elif v >= 9: return "color: #66dd88; font-weight: bold"
                elif v >= 7: return "color: #ffdd00"
                elif v >= 5: return "color: #ff8800"
                return ""
            except: return ""

        def color_conf(val):
            if "High" in str(val): return "color: #00ff88; font-weight: bold"
            elif "Med" in str(val): return "color: #ffdd00"
            return "color: #888888"

        def color_hand(val):
            # L vs R platoon — green when batter has advantage
            return ""  # handled in platoon column

        styled = (df_pp.style
                  .map(color_proj, subset=["PP Proj","Ceiling"])
                  .map(color_conf, subset=["Conf"]))
        st.dataframe(styled, use_container_width=True, height=520)

        csv_pp = df_pp.to_csv(index=False)
        st.download_button("📥 Export PP Projections", csv_pp,
                           f"pp_proj_{datetime.now(EST).strftime('%Y%m%d')}.csv", "text/csv")

    # ── GAME STACKS FOR PP ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎰 Best Game Environments")
    st.caption("Stack from same game for correlated upside — if one hits big, teammates likely did too")

    game_groups = {}
    for p in pp_plays:
        key = f"{p.get('opponent','')}@{p.get('team','')}"
        if key not in game_groups:
            game_groups[key] = {
                "game": key,
                "implied": p.get("implied_total",0),
                "players": [],
                "park": p.get("park",""),
                "weather": p.get("weather_label","").split("|")[0].strip()[:15],
            }
        game_groups[key]["players"].append(p)

    stacks = sorted(game_groups.values(),
                    key=lambda x: sum(p["pp_proj"] for p in x["players"][:4]) / max(1, len(x["players"][:4])),
                    reverse=True)

    for stack in stacks[:4]:
        top4 = sorted(stack["players"], key=lambda x: x["pp_proj"], reverse=True)[:4]
        avg_proj = sum(p["pp_proj"] for p in top4) / len(top4)
        imp_str = f"{stack['implied']:.1f} imp runs" if stack["implied"] > 0 else "no lines"
        with st.expander(f"🎮 **{stack['game']}** — Avg proj: {avg_proj:.1f} | {imp_str} | {stack['park']} | {stack['weather']}"):
            for p in top4:
                st.markdown(
                    f"• **{p['name']}** #{p['lineup_slot']} {p.get('batter_hand','?')}HB — "
                    f"PP: **{p['pp_proj']:.1f}** (ceil {p['pp_ceiling']:.1f}) | "
                    f"O1.5: {p['score']:.0f} | xSLG: {p.get('xslg',0):.3f}"
                )

    # ── LINEUP BUILDER ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📱 Build Your PP Entry")

    col_lb1, col_lb2 = st.columns(2)
    with col_lb1:
        lineup_legs = st.radio("Entry size", [2, 3, 4, 5, 6], index=1, horizontal=True, key="pp_lu_legs")
        lineup_mode = st.radio("Play type", ["Power Play", "Flex Play"], horizontal=True, key="pp_lu_mode")
    with col_lb2:
        mode_k = "power" if "Power" in lineup_mode else "flex"
        payout_lu = PP_PAYOUTS[mode_k].get(lineup_legs, 3.0)
        break_even = round(100 / (payout_lu ** (1/lineup_legs)), 1)
        st.metric("Payout if all correct", f"{payout_lu}x entry")
        st.caption(f"Need >{break_even}% win rate per leg to be +EV")

    # Manual multi-select
    player_opts = [
        f"{p['name']} ({p['team']}) — {p['pp_proj']:.1f} proj | ceil {p['pp_ceiling']:.1f} | #{p['lineup_slot']} {p.get('batter_hand','?')}HB vs {p['sp_name'][:12]}"
        for p in filtered[:40]
    ]
    selected = st.multiselect(
        f"Pick {lineup_legs} players for your {lineup_mode}:",
        player_opts, max_selections=lineup_legs, key="pp_manual_select"
    )

    if selected:
        st.markdown(f"**Your {lineup_mode} ({len(selected)}/{lineup_legs} legs):**")
        total_proj = 0
        for sel in selected:
            name_part = sel.split(" (")[0]
            match = next((p for p in pp_plays if p["name"] == name_part), None)
            if match:
                total_proj += match["pp_proj"]
                direction = st.radio(
                    f"{match['name']}: {match['pp_proj']:.1f} proj (ceil {match['pp_ceiling']:.1f})",
                    ["⬆️ MORE", "⬇️ LESS"], horizontal=True,
                    key=f"pp_dir_{match['name']}"
                )

        if len(selected) >= 2:
            st.markdown("---")
            col_out1, col_out2 = st.columns(2)
            with col_out1:
                st.metric("Combined projection", f"{total_proj:.1f} pts")
                st.metric("Expected payout", f"{payout_lu}x")
            with col_out2:
                # Build PP output string
                pp_legs = []
                for sel in selected:
                    name_part = sel.split(" (")[0]
                    match = next((p for p in pp_plays if p["name"] == name_part), None)
                    if match:
                        direction_key = st.session_state.get(f"pp_dir_{match['name']}", "⬆️ MORE")
                        direction_word = "MORE" if "MORE" in direction_key else "LESS"
                        pp_legs.append(f"{match['name']} {direction_word}")
                pp_str = " + ".join(pp_legs)
                st.code(f"PrizePicks: {pp_str}", language=None)

    # ── QUICK REFERENCE ───────────────────────────────────────────────────
    with st.expander("📊 How to use these projections"):
        st.markdown("""
**Reading the table:**
- **PP Proj** = our model's expected fantasy score for this player today
- **Ceiling** = high-end game (75th percentile)
- **Floor** = low-end game (25th percentile)
- **Conf** = 🔒 High (tight range, consistent player), ✅ Med, 📊 Low (boom/bust)

**Picking MORE vs LESS:**
- Compare our projection to what PrizePicks posts
- If PP posts 7.5 and our model says 11.5 → strong MORE
- If PP posts 8.5 and our model says 6.5 → LESS
- Tight floor/ceiling = more reliable bet regardless of direction

**Best PP plays generally:**
- Leadoff/2-hole hitters in high-total games (most PA, most opportunity)
- RHB vs LHP or LHB vs RHP (platoon advantage)
- Hitter-friendly parks (GABP, Citizens Bank, Camden)
- Wind out 10+ mph (HR/XBH boost)
- Low K% hitters (more PA completions, more counting stats)

**Power Play vs Flex:**
- 2-3 legs with strong conviction → Power Play (3x or 5x)
- 4-6 legs → Flex Play (can miss 1-2, lower payout but safer)
        """)


def fetch_actual_tb_for_date(date_str: str) -> Dict[str, Dict]:
    """
    Fetch actual total bases for all players on a given date from MLB Stats API.
    Returns dict keyed by player_id (str): {
        'total_bases': int, 'hits': int, 'hr': int, 'doubles': int,
        'triples': int, 'name': str, 'played': bool
    }
    Works for any completed date — uses boxscore batting stats.
    """
    results = {}
    try:
        # Get all games for the date
        sched_r = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "date": date_str},
            timeout=12
        )
        if sched_r.status_code != 200:
            return {}
        games = sched_r.json().get("dates", [{}])[0].get("games", [])

        for game in games:
            status = game.get("status", {}).get("abstractGameState", "")
            if status not in ("Final", "Live"):
                continue
            pk = game["gamePk"]
            try:
                box_r = requests.get(
                    f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore",
                    timeout=12
                )
                if box_r.status_code != 200:
                    continue
                box = box_r.json()
                for side in ["home", "away"]:
                    team_data = box.get("teams", {}).get(side, {})
                    players   = team_data.get("players", {})
                    for pid_key, pdata in players.items():
                        pos = pdata.get("position", {}).get("type", "")
                        if pos == "Pitcher":
                            continue
                        pid = str(pdata.get("person", {}).get("id", ""))
                        name = pdata.get("person", {}).get("fullName", "")
                        batting = pdata.get("stats", {}).get("batting", {})
                        # atBats > 0 means they played; atBats = 0 could be DNP or just no AB
                        ab   = int(batting.get("atBats", 0) or 0)
                        pa   = int(batting.get("plateAppearances", 0) or 0)
                        tb   = int(batting.get("totalBases", 0) or 0)
                        hits = int(batting.get("hits", 0) or 0)
                        hr   = int(batting.get("homeRuns", 0) or 0)
                        dbl  = int(batting.get("doubles", 0) or 0)
                        tri  = int(batting.get("triples", 0) or 0)
                        played = pa > 0
                        if pid:
                            results[pid] = {
                                "total_bases": tb,
                                "hits":        hits,
                                "hr":          hr,
                                "doubles":     dbl,
                                "triples":     tri,
                                "name":        name,
                                "played":      played,
                            }
            except Exception:
                continue
    except Exception:
        pass
    return results


def display_results_tracker():
    """
    Results Tracker V1.4
    - Pick ANY date (not just model-run dates)
    - Auto-fetches actual game results from MLB Stats API
    - Shows Top 10 TB picks + Top 5 HR picks with results
    - Auto-saves to DB and generates copy-ready Discord/Twitter posts
    """
    st.header("📈 Results Tracker")

    # ── DATE PICKER — any date, not limited to DB dates ──────────────────────
    col_date, col_fetch = st.columns([2, 1])
    with col_date:
        selected_date = st.date_input(
            "📅 Select date to view results:",
            value=datetime.now(EST).date() - timedelta(days=1),
            key="rt_date_v2"
        )
    date_str = selected_date.strftime("%Y-%m-%d")

    # ── LOAD DB PICKS FOR THIS DATE ───────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    try:
        picks_df = pd.read_sql(
            "SELECT * FROM picks WHERE date=? ORDER BY model_score DESC",
            conn, params=(date_str,)
        )
        all_picks_df = pd.read_sql(
            "SELECT * FROM picks ORDER BY date DESC, model_score DESC", conn
        )
    except Exception:
        picks_df = pd.DataFrame()
        all_picks_df = pd.DataFrame()
    conn.close()

    TIER_ORDER = {"🔒 TIER 1": 0, "✅ TIER 2": 1, "📊 TIER 3": 2}

    # ── LIVE SEASON TRACKER ───────────────────────────────────────────────────
    # Separate running totals: Top 10 TB picks vs Top 5 HR picks
    if not all_picks_df.empty:
        tiered_all = all_picks_df[all_picks_df["tier"].isin(TIER_ORDER.keys())].copy()

        # ── Per-day ranking to identify top 10 TB and top 5 HR ────────────────
        # Top 10 TB = top 10 by model_score per day
        # Top 5 HR  = top 5 by hr_score per day (falls back to model_score)
        hr_col_db = "hr_score" if "hr_score" in tiered_all.columns else "model_score"

        tb_picks_list = []
        hr_picks_list = []
        for day, day_df in tiered_all.groupby("date"):
            day_sorted_tb = day_df.sort_values("model_score", ascending=False)
            day_sorted_hr = day_df.sort_values(hr_col_db, ascending=False)
            tb_picks_list.append(day_sorted_tb.head(10))
            hr_picks_list.append(day_sorted_hr.head(5))

        tb_all = pd.concat(tb_picks_list) if tb_picks_list else pd.DataFrame()
        hr_all = pd.concat(hr_picks_list) if hr_picks_list else pd.DataFrame()

        # ── Compute stats ───────────────────────────────────────────────────────
        def stats_for(df):
            if df.empty or "result" not in df.columns:
                return 0, 0, 0, 0, 0, 0
            res = df[df["result"].isin(["hit","miss"])]
            h = len(res[res["result"] == "hit"])
            m = len(res[res["result"] == "miss"])
            t = h + m
            pct = h / t * 100 if t > 0 else 0
            dnp = len(df[df["result"] == "dnp"])
            pending = len(df[df["result"] == "pending"])
            return h, m, t, pct, dnp, pending

        tb_h, tb_m, tb_t, tb_pct, tb_dnp, tb_pend = stats_for(tb_all)
        hr_h, hr_m, hr_t, hr_pct, hr_dnp, hr_pend = stats_for(hr_all)

        # ── Display ─────────────────────────────────────────────────────────────
        st.markdown("### 📊 Season Tracker")

        col_tb, col_spacer, col_hr = st.columns([5, 0.5, 5])

        with col_tb:
            st.markdown("**⚾ Top 10 O1.5 TB Picks**")
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("Record",   f"{tb_h}-{tb_m}")
            with m2: st.metric("Hit Rate", f"{tb_pct:.1f}%" if tb_t > 0 else "—")
            with m3: st.metric("Pending",  tb_pend)

            # Daily breakdown table for TB
            if tb_picks_list:
                daily_tb = []
                for day, day_df in tiered_all.groupby("date"):
                    d = day_df.sort_values("model_score", ascending=False).head(10)
                    res_d = d[d["result"].isin(["hit","miss"])]
                    h = len(res_d[res_d["result"]=="hit"])
                    t = len(res_d)
                    daily_tb.append({
                        "Date":   day,
                        "Record": f"{h}/{t}" if t > 0 else "—",
                        "Hit%":   f"{h/t*100:.0f}%" if t > 0 else "—",
                        "DNP":    len(d[d["result"]=="dnp"]),
                    })
                daily_tb_df = pd.DataFrame(daily_tb).sort_values("Date", ascending=False)

                def color_hit(val):
                    try:
                        v = float(str(val).replace("%",""))
                        if v >= 70: return "color: #4caf50; font-weight: bold"
                        if v >= 50: return "color: #ffdd00"
                        return "color: #ff4444"
                    except: return ""

                styled_tb = daily_tb_df.style.map(color_hit, subset=["Hit%"])
                st.dataframe(styled_tb, use_container_width=True, hide_index=True, height=220)

        with col_hr:
            st.markdown("**💣 Top 5 HR Picks**")
            m1, m2, m3 = st.columns(3)
            # HR hit = tb_actual >= 4
            if not hr_all.empty and "tb_actual" in hr_all.columns:
                hr_res = hr_all[hr_all["result"].isin(["hit","miss"])].copy()
                hr_res["tb_actual"] = pd.to_numeric(hr_res["tb_actual"], errors="coerce").fillna(0)
                actual_hr_hits = len(hr_res[hr_res["tb_actual"] >= 4])
                actual_hr_tot  = len(hr_res)
                actual_hr_pct  = actual_hr_hits / actual_hr_tot * 100 if actual_hr_tot > 0 else 0
            else:
                actual_hr_hits, actual_hr_tot, actual_hr_pct = hr_h, hr_t, hr_pct

            with m1: st.metric("HR Record",  f"{actual_hr_hits}-{actual_hr_tot - actual_hr_hits}")
            with m2: st.metric("HR Hit%",    f"{actual_hr_pct:.1f}%" if actual_hr_tot > 0 else "—")
            with m3: st.metric("Pending",    hr_pend)

            # Daily breakdown table for HR
            if hr_picks_list:
                daily_hr = []
                for day, day_df in tiered_all.groupby("date"):
                    d = day_df.sort_values(hr_col_db, ascending=False).head(5)
                    d2 = d.copy()
                    d2["tb_actual"] = pd.to_numeric(d2.get("tb_actual", 0), errors="coerce").fillna(0)
                    res_d = d2[d2["result"].isin(["hit","miss"])]
                    hr_hits = len(res_d[res_d["tb_actual"] >= 4])
                    t = len(res_d)
                    daily_hr.append({
                        "Date":    day,
                        "HR Hits": f"{hr_hits}/{t}" if t > 0 else "—",
                        "HR%":     f"{hr_hits/t*100:.0f}%" if t > 0 else "—",
                        "DNP":     len(d[d["result"]=="dnp"]),
                    })
                daily_hr_df = pd.DataFrame(daily_hr).sort_values("Date", ascending=False)
                styled_hr = daily_hr_df.style.map(color_hit, subset=["HR%"])
                st.dataframe(styled_hr, use_container_width=True, hide_index=True, height=220)

        st.markdown("---")

    # ── AUTO-FETCH RESULTS ────────────────────────────────────────────────────
    with col_fetch:
        st.write("")
        st.write("")
        fetch_btn = st.button("⚡ Auto-Fetch Results", type="primary", use_container_width=True)

    if picks_df.empty:
        st.warning(f"No picks found in database for {date_str}. You need to have run the model on this date.")
        st.info("Tip: Run the model for this date first, then come back here to fetch results.")
        return

    # Filter to tiered only
    tiered = picks_df[picks_df["tier"].isin(TIER_ORDER.keys())].copy()
    tiered["_tier_sort"] = tiered["tier"].map(TIER_ORDER)
    tiered = tiered.sort_values(["_tier_sort", "model_score"], ascending=[True, False])

    # Top 10 TB picks + Top 5 HR picks (HR sorted by hr_score if stored, else model_score)
    top10_tb = tiered.head(10)
    # For HR — use hr_score column if available, otherwise top barrel/power picks
    if "hr_score" in tiered.columns:
        top5_hr = tiered.nlargest(5, "hr_score")
    else:
        top5_hr = tiered.nlargest(5, "barrel_rate") if "barrel_rate" in tiered.columns else tiered.head(5)

    # Auto-fetch actual results when button clicked
    actual_results = {}
    if fetch_btn:
        with st.spinner(f"Fetching actual game results for {date_str}..."):
            actual_results = fetch_actual_tb_for_date(date_str)

        if actual_results:
            st.success(f"✅ Fetched results for {len(actual_results)} players")
            # Auto-save to DB
            conn2 = sqlite3.connect(DB_PATH)
            c2 = conn2.cursor()
            saved = 0
            for _, row in tiered.iterrows():
                pid     = str(row.get("player_id", ""))
                pick_id = row["pick_id"]
                if pid in actual_results:
                    ar = actual_results[pid]
                    if not ar["played"]:
                        c2.execute("UPDATE picks SET result=?, tb_actual=? WHERE pick_id=?",
                                   ("dnp", 0, pick_id))
                    else:
                        tb_val = ar["total_bases"]
                        res    = "hit" if tb_val >= 2 else "miss"
                        c2.execute("UPDATE picks SET result=?, tb_actual=? WHERE pick_id=?",
                                   (res, tb_val, pick_id))
                    saved += 1
            conn2.commit()
            conn2.close()
            st.success(f"💾 Auto-saved results for {saved} picks")
            st.rerun()
        else:
            st.warning("Could not fetch results — games may not be final yet, or check your connection.")

    # Reload fresh picks after any auto-save
    conn3 = sqlite3.connect(DB_PATH)
    try:
        fresh = pd.read_sql(
            "SELECT * FROM picks WHERE date=? ORDER BY model_score DESC",
            conn3, params=(date_str,)
        )
    except Exception:
        fresh = pd.DataFrame()
    conn3.close()

    fresh_tiered = fresh[fresh["tier"].isin(TIER_ORDER.keys())].copy() if not fresh.empty else tiered

    # ── TOP 10 TB PICKS ───────────────────────────────────────────────────────
    st.subheader(f"⚾ Top 10 O1.5 TB Picks — {date_str}")

    tb_rows = []
    for _, row in fresh_tiered.head(10).iterrows():
        res    = row.get("result", "pending")
        tb_act = row.get("tb_actual")
        tb_val = int(tb_act) if pd.notna(tb_act) else None

        if res == "hit":
            status = "✅"
        elif res == "miss":
            status = "❌"
        elif res == "dnp":
            status = "⚠️ DNP"
        else:
            status = "⏳"

        tb_rows.append({
            "Status": status,
            "Player":      row["player_name"],
            "Team":        row["team"],
            "Opp SP":      row.get("sp_name", ""),
            "Tier":        row["tier"],
            "Score":       f"{row['model_score']:.0f}",
            "Actual TB":   str(tb_val) if tb_val is not None else "—",
            "Result":      res.upper() if res != "pending" else "PENDING",
        })

    if tb_rows:
        tb_df = pd.DataFrame(tb_rows)
        def color_result(val):
            if "HIT" in str(val):   return "color: #4caf50; font-weight: bold"
            if "MISS" in str(val):  return "color: #ff4444; font-weight: bold"
            if "DNP" in str(val):   return "color: #ffaa00"
            return "color: #888"
        styled = tb_df.style.map(color_result, subset=["Result"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── TOP 5 HR PICKS ────────────────────────────────────────────────────────
    st.subheader(f"💣 Top 5 HR Picks — {date_str}")

    hr_col = "hr_score" if "hr_score" in fresh_tiered.columns else "model_score"
    top5 = fresh_tiered.nlargest(5, hr_col) if not fresh_tiered.empty else pd.DataFrame()

    hr_rows = []
    for _, row in top5.iterrows():
        res    = row.get("result", "pending")
        tb_act = row.get("tb_actual")
        tb_val = int(tb_act) if pd.notna(tb_act) else None
        # HR = 4 TB, so if tb_actual >= 4 AND actual_hr > 0, they hit a HR
        # We flag HR if tb >= 4 (could be 4 singles but very rare — HR is most likely)
        hr_hit = "✅ HR" if (tb_val is not None and tb_val >= 4) else ("❌" if res == "miss" else ("⚠️ DNP" if res == "dnp" else "⏳"))

        hr_rows.append({
            "HR Result":  hr_hit,
            "Player":     row["player_name"],
            "Team":       row["team"],
            "Opp SP":     row.get("sp_name", ""),
            "HR Score":   f"{row.get(hr_col, 0):.0f}",
            "Actual TB":  str(tb_val) if tb_val is not None else "—",
            "TB Result":  res.upper() if res != "pending" else "PENDING",
        })

    if hr_rows:
        hr_df = pd.DataFrame(hr_rows)
        def color_hr(val):
            if "HR" in str(val) and "✅" in str(val): return "color: #f5a623; font-weight: bold"
            if "❌" in str(val): return "color: #ff4444"
            if "DNP" in str(val): return "color: #ffaa00"
            return "color: #888"
        styled_hr = hr_df.style.map(color_hr, subset=["HR Result"])
        st.dataframe(styled_hr, use_container_width=True, hide_index=True)

    # ── MANUAL OVERRIDE — in case auto-fetch misses anyone ───────────────────
    with st.expander("✏️ Manual Override (if auto-fetch missed a player)"):
        pending = fresh_tiered[fresh_tiered["result"] == "pending"] if not fresh_tiered.empty else pd.DataFrame()
        if not pending.empty:
            for _, row in pending.iterrows():
                pick_id = row["pick_id"]
                col_info, col_tb, col_dnp, col_save = st.columns([3, 1.2, 0.8, 0.8])
                with col_info:
                    st.markdown(f"**{row['player_name']}** ({row['team']}) — {row['tier']}")
                with col_tb:
                    tb_v = st.number_input("TB", 0, 16, 0, key=f"man_tb_{pick_id}", label_visibility="collapsed")
                with col_dnp:
                    dnp_v = st.checkbox("DNP", key=f"man_dnp_{pick_id}")
                with col_save:
                    if st.button("Save", key=f"man_save_{pick_id}"):
                        if dnp_v:
                            update_pick_result(pick_id, "dnp", 0)
                        else:
                            update_pick_result(pick_id, "hit" if tb_v >= 2 else "miss", tb_v)
                        st.rerun()
        else:
            st.caption("All picks logged — no manual entry needed.")

    # ── COPY-READY POSTS ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Copy-Ready Posts")

    if not fresh_tiered.empty and fresh_tiered["result"].isin(["hit","miss","dnp"]).any():
        date_fmt = selected_date.strftime("%-m/%-d")

        # ── TOP 5 TB PICKS (matches the bet card you post) ────────────────────
        lines_tb = []
        hits_c, misses_c, dnp_c = 0, 0, 0
        for _, r in fresh_tiered.head(5).iterrows():
            res = r.get("result", "pending")
            tb  = int(r["tb_actual"]) if pd.notna(r.get("tb_actual")) else 0
            nm  = r["player_name"]
            if res == "hit":
                lines_tb.append(f"✅ {nm} — {tb} TB")
                hits_c += 1
            elif res == "miss":
                lines_tb.append(f"❌ {nm} — {tb} TB")
                misses_c += 1
            elif res == "dnp":
                lines_tb.append(f"⚠️ {nm} — DNP")
                dnp_c += 1

        # ── TOP 2 HR CALLS — ✅ only if actual HR (tb >= 4), ❌ otherwise ─────
        lines_hr = []
        hr_col_use = "hr_score" if "hr_score" in fresh_tiered.columns else "model_score"
        top2_hr = fresh_tiered.nlargest(2, hr_col_use)
        hr_hits_c = 0
        hr_total_c = 0
        for _, r in top2_hr.iterrows():
            res = r.get("result", "pending")
            tb  = int(r["tb_actual"]) if pd.notna(r.get("tb_actual")) else 0
            nm  = r["player_name"]
            if res in ("hit", "miss", "dnp"):
                hr_total_c += 1
            if res == "hit" and tb >= 4:
                lines_hr.append(f"💣 {nm} — HR ✅")
                hr_hits_c += 1
            elif res == "hit":
                lines_hr.append(f"❌ {nm} — {tb} TB (no HR)")
            elif res == "miss":
                lines_hr.append(f"❌ {nm} — {tb} TB")
            elif res == "dnp":
                lines_hr.append(f"⚠️ {nm} — DNP")

        total_res = hits_c + misses_c
        hit_pct   = f"{hits_c/total_res*100:.0f}%" if total_res > 0 else "—"
        hr_pct    = f"{hr_hits_c}/{hr_total_c}" if hr_total_c > 0 else "—"

        discord = f"""📊 DAILY RECAP — {date_fmt}
━━━━━━━━━━━━━━━━━━━

⚾ O1.5 TB PICKS
{chr(10).join(lines_tb) if lines_tb else "Results pending..."}

💣 HR CALLS
{chr(10).join(lines_hr) if lines_hr else "Results pending..."}

━━━━━━━━━━━━━━━━━━━
🎯 TB: {hits_c}/{total_res} cashed ({hit_pct}){(" · ⚠️ " + str(dnp_c) + " DNP") if dnp_c > 0 else ""}
💣 HR: {hr_pct} hit

Tomorrow's plays drop at 12PM EST 🔒"""

        # Twitter — clean and concise
        tb_short  = " · ".join(lines_tb) if lines_tb else "pending"
        hr_short  = " · ".join(lines_hr) if lines_hr else "pending"
        twitter = f"""Result {date_fmt}

{tb_short}

💣 {hr_short}

{hits_c}/{total_res} TB cashed{(" · " + str(dnp_c) + " DNP") if dnp_c > 0 else ""} · HR {hr_pct}

#MLBbets #SportsBetting"""

        col_d, col_t = st.columns(2)
        with col_d:
            st.markdown("**Discord**")
            st.code(discord, language=None)
        with col_t:
            st.markdown("**Twitter reply**")
            st.code(twitter, language=None)
    else:
        st.info("Click **⚡ Auto-Fetch Results** above to load actual game results, then posts will appear here.")

    # ── EXPORT ────────────────────────────────────────────────────────────────
    st.markdown("---")
    if not all_picks_df.empty:
        st.download_button("📥 Export All Picks", all_picks_df.to_csv(index=False),
                           "mlb_picks.csv", "text/csv", key="dl_all_picks_top")

    conn = sqlite3.connect(DB_PATH)
    try:
        picks_df = pd.read_sql("SELECT * FROM picks ORDER BY date DESC, model_score DESC", conn)
        parlays_df = pd.read_sql("SELECT * FROM parlays ORDER BY date DESC", conn)
    except Exception:
        picks_df = pd.DataFrame()
        parlays_df = pd.DataFrame()
    conn.close()

    if picks_df.empty:
        st.info("📊 No data yet. Run the model first to populate picks.")
        return

    # ── OVERALL STATS BAR ────────────────────────────────────────────────────
    tiered_all = picks_df[picks_df["tier"].isin(["🔒 TIER 1", "✅ TIER 2", "📊 TIER 3"])]
    resolved   = tiered_all[tiered_all["result"].isin(["hit", "miss", "dnp"])]

    if not resolved.empty:
        hits   = len(resolved[resolved["result"] == "hit"])
        misses = len(resolved[resolved["result"] == "miss"])
        dnps   = len(resolved[resolved["result"] == "dnp"])
        total  = hits + misses
        rate   = hits / total * 100 if total > 0 else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.metric("Record",    f"{hits}-{misses}")
        with c2: st.metric("Hit Rate",  f"{rate:.1f}%")
        with c3:
            t1 = resolved[resolved["tier"] == "🔒 TIER 1"]
            v  = len(t1[t1["result"]=="hit"]) / max(1, len(t1[t1["result"].isin(["hit","miss"])])) * 100
            st.metric("Tier 1 Hit%", f"{v:.0f}%" if len(t1) > 0 else "—")
        with c4:
            t2 = resolved[resolved["tier"] == "✅ TIER 2"]
            v  = len(t2[t2["result"]=="hit"]) / max(1, len(t2[t2["result"].isin(["hit","miss"])])) * 100
            st.metric("Tier 2 Hit%", f"{v:.0f}%" if len(t2) > 0 else "—")
        with c5: st.metric("DNP / Void", dnps)

    st.markdown("---")

    # ── DATE SELECTOR ────────────────────────────────────────────────────────
    available_dates = sorted(picks_df["date"].unique(), reverse=True)
    yesterday = (datetime.now(EST).date() - timedelta(days=1)).strftime("%Y-%m-%d")
    default_idx = 0
    for i, d in enumerate(available_dates):
        if d == yesterday:
            default_idx = i
            break

    selected_date = st.selectbox(
        "📅 Select date to log results:",
        available_dates,
        index=default_idx,
        key="rt_date_select"
    )

    # Filter to tiered picks only for that date
    TIER_ORDER = {"🔒 TIER 1": 0, "✅ TIER 2": 1, "📊 TIER 3": 2}
    day_picks = picks_df[
        (picks_df["date"] == selected_date) &
        (picks_df["tier"].isin(TIER_ORDER.keys()))
    ].copy()
    day_picks["_tier_sort"] = day_picks["tier"].map(TIER_ORDER)
    day_picks = day_picks.sort_values(["_tier_sort", "model_score"], ascending=[True, False])

    if day_picks.empty:
        st.info(f"No tiered picks found for {selected_date}. Run the model for that date first.")
        return

    st.markdown(f"**{len(day_picks)} tiered picks on {selected_date}**")

    # ── LOG RESULTS — one row per pick ──────────────────────────────────────
    st.subheader("✏️ Log Actual Total Bases")
    st.caption("Enter actual TB for each player. DNP = did not play / scratched.")

    updated_any = False
    for _, row in day_picks.iterrows():
        pick_id  = row["pick_id"]
        name     = row["player_name"]
        team     = row["team"]
        tier     = row["tier"]
        score    = row["model_score"]
        current  = row["result"]
        tb_saved = row["tb_actual"] if pd.notna(row["tb_actual"]) else None

        # Status icon
        if current == "hit":
            status = "✅"
        elif current == "miss":
            status = "❌"
        elif current == "dnp":
            status = "⚠️"
        else:
            status = "⏳"

        with st.container():
            col_info, col_tb, col_dnp, col_save = st.columns([3, 1.2, 0.8, 0.8])
            with col_info:
                st.markdown(f"{status} **{name}** ({team}) — {tier} | Score {score:.0f}")
                if tb_saved is not None and current != "pending":
                    res_label = "✅ HIT" if current == "hit" else "❌ MISS" if current == "miss" else "⚠️ DNP"
                    st.caption(f"Logged: {tb_saved} TB — {res_label}")
            with col_tb:
                tb_val = st.number_input(
                    "TB", min_value=0, max_value=16,
                    value=int(tb_saved) if tb_saved is not None and current != "dnp" else 0,
                    key=f"tb_{pick_id}", label_visibility="collapsed"
                )
            with col_dnp:
                dnp = st.checkbox("DNP", key=f"dnp_{pick_id}",
                                  value=(current == "dnp"))
            with col_save:
                if st.button("Save", key=f"save_{pick_id}"):
                    if dnp:
                        update_pick_result(pick_id, "dnp", 0)
                    else:
                        result_val = "hit" if tb_val >= 2 else "miss"
                        update_pick_result(pick_id, result_val, tb_val)
                    updated_any = True

    if updated_any:
        st.success("✅ Results saved!")
        st.rerun()

    # ── COPY-READY RESULT POST ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Copy-Ready Result Post")

    # Reload fresh after any saves
    conn2 = sqlite3.connect(DB_PATH)
    try:
        fresh_df = pd.read_sql(
            "SELECT * FROM picks WHERE date=? ORDER BY model_score DESC",
            conn2, params=(selected_date,)
        )
    except Exception:
        fresh_df = pd.DataFrame()
    conn2.close()

    day_fresh = fresh_df[fresh_df["tier"].isin(TIER_ORDER.keys())].copy() if not fresh_df.empty else pd.DataFrame()

    if not day_fresh.empty and day_fresh["result"].isin(["hit","miss","dnp"]).any():
        # Build result lines
        lines = []
        hits_count = 0
        misses_count = 0
        dnp_count = 0
        for _, r in day_fresh[day_fresh["tier"].isin(TIER_ORDER.keys())].iterrows():
            res = r["result"]
            tb  = int(r["tb_actual"]) if pd.notna(r["tb_actual"]) else 0
            if res == "hit":
                lines.append(f"✅ {r['player_name']} — {tb} TB HIT")
                hits_count += 1
            elif res == "miss":
                lines.append(f"❌ {r['player_name']} — {tb} TB MISS")
                misses_count += 1
            elif res == "dnp":
                lines.append(f"⚠️ {r['player_name']} — DNP")
                dnp_count += 1

        total_resolved = hits_count + misses_count
        hit_pct = f"{hits_count/total_resolved*100:.0f}%" if total_resolved > 0 else "—"
        date_fmt = datetime.strptime(selected_date, "%Y-%m-%d").strftime("%-m/%-d")

        discord_post = f"""📊 DAILY RECAP — {date_fmt}
━━━━━━━━━━━━━━━━━━━

⚾ MLB TIERED PICKS

{chr(10).join(lines)}

━━━━━━━━━━━━━━━━━━━
🎯 {hits_count}/{total_resolved} picks cashed ({hit_pct})
{"⚠️ " + str(dnp_count) + " DNP/void" if dnp_count > 0 else ""}

Tomorrow's plays dropping at 12PM EST 🔒"""

        twitter_post = f"""Result {date_fmt}

{chr(10).join(lines)}

{hits_count}/{total_resolved} cashed{" · " + str(dnp_count) + " DNP" if dnp_count > 0 else ""}

#MLBbets #SportsBetting"""

        col_d, col_t = st.columns(2)
        with col_d:
            st.markdown("**Discord**")
            st.code(discord_post, language=None)
        with col_t:
            st.markdown("**Twitter reply**")
            st.code(twitter_post, language=None)
    else:
        st.info("Log results above to generate copy-ready posts.")

    # ── EXPORT ──────────────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if not picks_df.empty:
            st.download_button("📥 Export All Picks", picks_df.to_csv(index=False),
                               "mlb_picks.csv", "text/csv", key="dl_all_picks_bot")
    with col2:
        if not parlays_df.empty:
            st.download_button("📥 Export Parlays", parlays_df.to_csv(index=False),
                               "mlb_parlays.csv", "text/csv")


def main():
    # Initialize DB
    init_db()
    
    # Init session state
    if "plays" not in st.session_state:
        st.session_state.plays = []
    if "analysis_date" not in st.session_state:
        st.session_state.analysis_date = None
    if "model_ran" not in st.session_state:
        st.session_state.model_ran = False
    
    # ── SIDEBAR ───────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        
        today_est = datetime.now(EST).date()
        selected_date = st.date_input("📅 Select Date", value=today_est)
        date_str = selected_date.strftime("%Y-%m-%d")
        
        st.markdown("---")
        
        st.subheader("💰 Bankroll")
        unit_size = st.number_input("Unit Size ($)", min_value=1, max_value=1000, value=25)
        
        st.markdown("---")
        
        run_btn = st.button("⚾ Run Today's Model", type="primary", use_container_width=True)
        
        if st.button("🔄 Clear Cache + Rerun", use_container_width=True):
            st.cache_data.clear()
            st.session_state.plays = []
            st.rerun()
        
        st.markdown("---")
        
        # Data source status
        st.subheader("📡 Data Sources")
        
        try:
            odds_key = st.secrets.get("odds_api", {}).get("api_key", "")
            has_odds_key = bool(odds_key and odds_key.strip())
        except:
            has_odds_key = False

        st.markdown(f"✅ MLB Stats API *(always accessible)*")
        bat_src   = st.session_state.get("_batting_source", "")
        pit_src   = st.session_state.get("_pitching_source", "")
        bat_cols  = st.session_state.get("batting_cols", [])
        has_xstats    = "xSLG" in bat_cols or "est_slg" in bat_cols
        has_statcast  = "Barrel%" in bat_cols or "barrel_batted_rate" in bat_cols
        has_bat_track = "bat_speed" in bat_cols
        if not bat_src:
            st.markdown("⏳ Baseball Savant *(run model to check)*")
        elif has_xstats and has_statcast and has_bat_track:
            st.markdown("✅ Baseball Savant *(full: xStats + Statcast + Bat Tracking)*")
        elif has_xstats and has_statcast:
            st.markdown("🟡 Baseball Savant *(partial: xStats + Statcast, no bat tracking)*")
        elif has_xstats:
            st.markdown("⚠️ Baseball Savant *(xStats only — Statcast leaderboard blocked [502])*")
        else:
            st.markdown("❌ Baseball Savant *(blocked — using MLB API proxies)*")
        st.markdown(f"✅ Open-Meteo Weather")
        
        if has_odds_key:
            st.markdown(f"✅ The Odds API *(live lines)*")
        else:
            st.markdown(f"⚠️ The Odds API *(no key — scores degraded)*")
            with st.expander("🔑 Add Odds API Key (required for best scores)"):
                st.markdown("""
**Step 1:** Sign up free at [the-odds-api.com](https://the-odds-api.com)
— 500 calls/month free (~$0 for daily use all season)

**Step 2:** Go to Streamlit Cloud → your app → **⋮ Settings → Secrets**

**Step 3:** Paste this (replace with your real key):
```toml
[odds_api]
api_key = "your_key_here"
```
**Step 4:** Save → app auto-restarts

**Usage math:** 1 call per model run × ~180 game days = ~180 calls/season.
Free tier (500/mo) is more than enough.
                """)
        
        st.markdown("---")
        
        with st.expander("📖 Model Info"):
            st.markdown("""
            **Scoring: 1B=1, 2B=2, 3B=3, HR=4**
            Walks, HBP, SB = 0 TB (never counted)
            
            **V1.8 Weights:**
            - ⚾ Pitcher Vuln: 30% (K%, HH%, FIP, barrel, WHIP — wider scale)
            - 🏏 Batter: 28% (xSLG, barrel, HH%, wRC+, ISO, K%)
            - 🤚 Platoon: 12% (LHB vs RHP = +56 SLG documented)
            - 💰 Vegas: 8% (implied total r=0.61 with scoring)
            - 🏟️ Park: 7% (Coors/Petco effects real)
            - 📈 Streak: 5% (last 7 games form)
            - 🔄 TTO: 4% (times through order bonus)
            - 🌤️ Weather: 4% (wind direction/speed)
            - 🎯 Pitch Mix: 2%
            - 📊 BvP: 2% (career vs SP)
            - 📋 Lineup: 1%
            
            **V1.8 Tiers:**
            - 🔒 Tier 1 (78+): Strong play, parlay anchor
            - ✅ Tier 2 (68-77): Viable, parlay filler
            - 📊 Tier 3 (58-67): Marginal, single only
            - ❌ Below 58: Fade
            """)
        
        with st.expander("🔍 Debug: Data Quality Check"):
            st.caption("Expand after running model to verify all key stats loaded")

            # ── Data source status ─────────────────────────────────────────
            bat_src  = st.session_state.get("_batting_source", "not_run")
            pit_src  = st.session_state.get("_pitching_source", "not_run")
            src_icon = {
                # Live fetch sources
                "fangraphs_live": "✅", "fangraphs+pybaseball": "✅",
                "savant+mlbapi": "✅", "mlbapi+savant": "✅",
                "savant+mlbapi+fangraphs": "✅",
                "mlbapi": "✅", "mlbapi_only": "⚠️",
                "savant_xstats": "✅", "savant_statcast": "✅",
                "matched": "✅",
                # Disk cache
                "disk_cache": "💾", "disk_cache_fresh": "💾",
                "disk_cache_stale": "⚠️",
                # Failure states
                "failed": "❌", "not_run": "⏳", "unknown": "❓",
            }
            st.markdown(f"**Batting stats source:** {src_icon.get(bat_src,'❓')} `{bat_src}`")
            st.markdown(f"**Pitching stats source:** {src_icon.get(pit_src,'❓')} `{pit_src}`")

            # Show FanGraphs errors if any
            fg_bat_errs = st.session_state.get("_fg_batting_errors", [])
            fg_pit_errs = st.session_state.get("_fg_pitching_errors", [])
            if fg_bat_errs:
                st.error("**FanGraphs batting fetch errors:**\n" + "\n".join(fg_bat_errs))
            if fg_pit_errs:
                st.error("**FanGraphs pitching fetch errors:**\n" + "\n".join(fg_pit_errs))

            # Arsenal merge error surfacing
            _am_err = st.session_state.get("_arsenal_merge_err")
            _ba_err = st.session_state.get("_batter_arsenal_err")
            if _am_err:
                st.error(f"⚠️ Pitcher arsenal merge error (non-fatal): {_am_err}")
            if _ba_err:
                st.error(f"⚠️ Batter arsenal merge error (non-fatal): {_ba_err}")

            # Disk cache status
            import os as _os, time as _time
            _cache_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "stat_cache")
            for _cname in ("batting_stats", "pitching_stats"):
                _p = _os.path.join(_cache_dir, f"{_cname}.pkl")
                if _os.path.exists(_p):
                    _age = (_time.time() - _os.path.getmtime(_p)) / 3600
                    _age_label = f"{_age:.1f}h old"
                    _freshness = "✅ fresh" if _age < 6 else ("⚠️ stale" if _age > 120 else "💾 cached")
                    st.caption(f"💾 Disk cache `{_cname}`: {_age_label} — {_freshness}")
                else:
                    st.caption(f"⬜ No disk cache for `{_cname}` — run `seed_stat_cache.py` locally and commit `stat_cache/`")

            st.markdown("---")
            matched = st.session_state.get("_matched", 0)
            unmatched = st.session_state.get("_unmatched", 0)
            total = matched + unmatched
            if total > 0:
                rate = matched / total * 100
                color = "✅" if rate > 80 else "⚠️" if rate > 50 else "❌"
                st.markdown(f"**{color} Player match rate: {matched}/{total} ({rate:.0f}%)**")
                if rate < 80:
                    st.warning("Low match rate — scores using league averages for unmatched players")
            
            # Show raw name samples — use .get() everywhere to avoid AttributeError
            # before model has been run (session_state keys may not exist yet)
            lookup_diag = st.session_state.get("lookup_diag")
            batting_df_sample = st.session_state.get("batting_df_sample", [])
            norm_name_sample  = st.session_state.get("norm_name_sample", [])
            search_sample     = st.session_state.get("search_sample", [])
            batting_cols      = st.session_state.get("batting_cols", [])
            pitching_cols     = st.session_state.get("pitching_cols", [])
            sample_player     = st.session_state.get("sample_player")

            if lookup_diag:
                st.markdown("**🔬 Live lookup diagnostic (first batter):**")
                st.json(lookup_diag)
                if batting_df_sample:
                    st.markdown("**Raw _name values in batting DataFrame (first 10):**")
                    st.code("\n".join(str(x) for x in batting_df_sample))
            if norm_name_sample:
                st.markdown("**Normalized _norm_name values (first 10):**")
                st.code("\n".join(str(x) for x in norm_name_sample))
            if search_sample:
                st.markdown("**Names we searched for (first 5):**")
                st.code("\n".join(str(x) for x in search_sample))
            if batting_cols:
                critical_bat = ["xSLG", "xwOBA", "SLG", "ISO", "K%", "BB%", "OBP",
                                "wRC+", "Barrel%", "Hard%", "EV"]
                # What provides each stat (real source vs proxy)
                _source_map = {
                    "xSLG":    ("Savant xStats ✅", None),
                    "xwOBA":   ("Savant xStats ✅", None),
                    "SLG":     ("MLB Stats API ✅", None),
                    "ISO":     ("MLB Stats API ✅", None),
                    "K%":      ("MLB Stats API ✅", None),
                    "BB%":     ("MLB Stats API ✅", None),
                    "OBP":     ("MLB Stats API ✅", None),
                    "wRC+":    ("FanGraphs (blocked ❌)", "OBP proxy ~±25pts"),
                    "Barrel%": ("Savant Statcast (blocked ❌)", "HR/PA proxy ~±5-8%"),
                    "Hard%":   ("Savant Statcast (blocked ❌)", "SLG+K% proxy ~±6-10%"),
                    "EV":      ("Savant Statcast (blocked ❌)", "xSLG+ISO derived"),
                }
                st.markdown("**📊 Batting data — real vs proxy:**")
                real_count = 0
                proxy_count = 0
                for c in critical_bat:
                    src, proxy_note = _source_map.get(c, ("Unknown", None))
                    if c in batting_cols:
                        st.markdown(f"✅ **`{c}`** — {src.split(' ✅')[0]}")
                        real_count += 1
                    elif proxy_note:
                        st.markdown(f"🔄 **`{c}`** — {proxy_note} *(real source: {src.split(' (')[0]})*")
                        proxy_count += 1
                    else:
                        st.markdown(f"❌ **`{c}`** — missing, league avg used")
                
                completeness = real_count / len(critical_bat) * 100
                color = "🟢" if completeness >= 80 else "🟡" if completeness >= 50 else "🔴"
                st.caption(f"Total columns: {len(batting_cols)} | "
                           f"{color} Data completeness: {completeness:.0f}% real | "
                           f"{proxy_count} proxies active")
                if proxy_count > 0:
                    st.warning(f"⚠️ {proxy_count} stats are derived proxies, not real Statcast data. "
                               "Scores are adjusted with +13pt offset to compensate.")
                    with st.expander("🔧 Fix: Generate local cache seeder script"):
                        st.markdown("""
**Why this happens:** Streamlit Cloud IPs are blocked by Baseball Savant's 
Statcast leaderboard endpoint. The real fix is seeding the cache from your 
local machine daily.

**Steps:**
1. Download the seeder script below
2. Run it locally: `python3 seed_stat_cache.py`
3. Commit the generated `stat_cache/` folder to your GitHub repo
4. Streamlit Cloud will use the cached data automatically

The cache is valid for 6 hours — run the script before posting daily picks.
                        """)
                        seed_script = '#!/usr/bin/env python3\n# Propex stat cache seeder -- run locally daily, commit stat_cache/ to GitHub\n'
                        seed_script += '"""Fetches real Savant Statcast data from your local IP where endpoints are unblocked."""\n'
                        seed_script += """
import requests, pandas as pd, pickle, os, time

CACHE_DIR = os.path.join(os.path.dirname(__file__), "stat_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def fetch(url, label):
    print(f"Fetching {label}...")
    r = requests.get(url, timeout=30, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
        "Referer": "https://baseballsavant.mlb.com/",
    })
    if r.status_code == 200 and len(r.content) > 1000:
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        print(f"  ✅ {len(df)} rows, columns: {list(df.columns[:6])}")
        return df
    print(f"  ❌ HTTP {r.status_code}")
    return pd.DataFrame()

yr = 2026

# Fetch all Savant endpoints
xstats = fetch(f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=batter&year={yr}&min=1&csv=true", "xStats")
statcast = fetch(f"https://baseballsavant.mlb.com/leaderboard/statcast?year={yr}&min=1&type=batter&csv=true", "Statcast")
bat_track = fetch(f"https://baseballsavant.mlb.com/leaderboard/bat-tracking?year={yr}&minSwings=50&type=batter&csv=true", "Bat Tracking")

# Merge into one batting frame
result = xstats.copy() if not xstats.empty else pd.DataFrame()
if not statcast.empty and not result.empty:
    # normalize player_id
    for df in [result, statcast, bat_track]:
        if "player_id" in df.columns:
            df["mlbam_id"] = df["player_id"].astype(str)
    result = result.merge(statcast[["mlbam_id","barrel_batted_rate","hard_hit_percent","avg_exit_velocity"]].dropna(subset=["mlbam_id"]), on="mlbam_id", how="left")
    if not bat_track.empty:
        result = result.merge(bat_track[["mlbam_id","bat_speed","blast_rate"]].dropna(subset=["mlbam_id"]), on="mlbam_id", how="left")

if not result.empty:
    path = os.path.join(CACHE_DIR, "batting_stats.pkl")
    with open(path, "wb") as f:
        pickle.dump({"df": result, "ts": time.time()}, f)
    print(f"✅ Batting cache saved: {len(result)} players, {len(result.columns)} columns")
    print(f"   Columns: {list(result.columns)}")
else:
    print("❌ Could not build batting cache")
"""
                        st.download_button(
                            "⬇️ Download cache seeder script",
                            seed_script,
                            "seed_stat_cache.py",
                            "text/plain",
                            key="dl_seed_script"
                        )
            else:
                st.info("Run the model to see batting column status.")
            if pitching_cols:
                critical_pit = ["K%", "ERA", "FIP", "xFIP", "Hard%", "Barrel%",
                                "SO", "TBF", "xERA", "G", "GS", "Team", "WHIP"]
                st.markdown("**Pitching — critical columns:**")
                for c in critical_pit:
                    st.markdown(f"{'✅' if c in pitching_cols else '❌'} `{c}`")
                st.caption(f"Total pitching columns: {len(pitching_cols)}")
            else:
                st.info("Run the model to see pitching column status.")
            if sample_player:
                st.markdown("**Sample player (Judge):**")
                st.json(sample_player)

        # ── V1.3 NEW: Bullpen Quality Debug ──────────────────────────────
        with st.expander("🔬 Debug: V1.3 Bullpen Quality Scores"):
            st.caption("Per-team bullpen vulnerability (0=unhittable, 100=mop-up arms). "
                       "High score = weak bullpen = good for batters. "
                       "Was fixed at 42.0 for all teams in V1.2.")
            bp_scores = st.session_state.get("team_bullpen_scores", {})
            if bp_scores:
                bp_df = pd.DataFrame([
                    {"Team": t, "Bullpen Vuln Score": v,
                     "Quality": "🔒 Elite" if v < 35 else "✅ Good" if v < 45 else "⚠️ Average" if v < 55 else "💀 Weak"}
                    for t, v in sorted(bp_scores.items(), key=lambda x: x[1])
                ])

                def color_bp(val):
                    try:
                        v = float(val)
                        if v < 35:  return "color: #00ff88; font-weight: bold"
                        elif v < 45: return "color: #66ddff"
                        elif v < 55: return "color: #ffdd00"
                        return "color: #ff4444; font-weight: bold"
                    except: return ""

                styled_bp = bp_df.style.map(color_bp, subset=["Bullpen Vuln Score"])
                st.dataframe(styled_bp, use_container_width=True)
                st.caption(f"✅ {len(bp_scores)} teams scored | League avg baseline: 42.0")

                # Show score impact on a sample batter
                st.markdown("**Score impact example (avg batter, avg SP, score = 55):**")
                example_rows = []
                for label, bp_v in [("Best bullpen (score ~28)", 28), ("League avg (42)", 42), ("Worst bullpen (score ~62)", 62)]:
                    sp_score_ex = 45.0  # avg SP
                    blended_ex  = sp_score_ex * 0.60 + bp_v * 0.40
                    example_rows.append({"Scenario": label, "SP Score": 45, "BP Vuln": bp_v,
                                         "Blended Pit Score": round(blended_ex, 1)})
                st.table(pd.DataFrame(example_rows))
            else:
                st.warning("Bullpen scores not computed — pitching_df may be missing GS or Team columns. "
                           "All teams defaulting to league average (42.0). "
                           "Run model and check pitching critical columns above.")

        # ── V1.3 NEW: Per-player score breakdown debug ───────────────────
        with st.expander("🔬 Debug: Score Component Breakdown (top 10 plays)"):
            st.caption("Shows exactly how each sub-score contributed to the final score. "
                       "BP Vuln = the opponent's bullpen score used for that batter.")
            plays_debug = st.session_state.get("plays", [])
            if plays_debug:
                debug_rows = []
                for p in plays_debug[:10]:
                    debug_rows.append({
                        "Player":     p["name"],
                        "Team":       p["team"],
                        "Opp":        p.get("opponent", "?"),
                        "Final":      p["score"],
                        "Batter":     p.get("sub_batter", "—"),
                        "Pitcher":    p.get("sub_pitcher", "—"),
                        "Matchup":    p.get("sub_matchup", "—"),
                        "BP Vuln":    p.get("bullpen_vuln", "—"),
                        "Platoon":    p.get("sub_platoon", "—"),
                        "Park":       p.get("sub_park", "—"),
                        "Weather":    p.get("sub_weather", "—"),
                        "Vegas":      p.get("sub_vegas", "—"),
                        "Lineup":     p.get("sub_lineup", "—"),
                    })
                st.dataframe(pd.DataFrame(debug_rows), use_container_width=True)
            else:
                st.info("Run the model to see per-player score breakdowns.")

        st.caption(f"v1.6 | {datetime.now(EST).strftime('%I:%M %p EST')}")
    
    # ── MAIN CONTENT ──────────────────────────────────────
    st.title("⚾ MLB TB Analyzer V1.9")
    import hashlib as _hlib
    _sig = _hlib.md5(b"v19_kprops_moneyline").hexdigest()[:6]
    st.caption(f"🔑 Build sig: {_sig} — V1.9: K Props + Moneyline models live")
    st.caption("Fully automated | HardRock Bet | 1B=1 2B=2 3B=3 HR=4 | V1.9: K Props tab + Moneyline tab added")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📊 O1.5 Leaderboard",
        "🎯 Tiered Breakdown",
        "💰 Parlay Builder",
        "🎯 O0.5 Any Hit",
        "⚡ K Props",
        "🏦 Moneyline",
        "💣 HR Plays",
        "🏆 FanDuel DFS",
        "🎯 PrizePicks",
        "📈 Results Tracker",
    ])
    
    # ── RUN MODEL ────────────────────────────────────────
    if run_btn:
        with tab1:
            st.markdown(f"**📅 Running model for {date_str}...**")
            status = st.container()
            plays = run_model(date_str, status)
            st.session_state.plays = plays
            st.session_state.analysis_date = date_str
            st.session_state.model_ran = True
            if plays:
                save_picks_to_db(plays, date_str)
                best_parlays = build_parlays(plays, 3, 1, 70.0)
                if best_parlays:
                    save_parlay_to_db(best_parlays[0], date_str)
                # ── V1.9: Pre-fetch ump data and run differential for new tabs ──
                try:
                    st.session_state["_ump_data"] = fetch_umpire_data()
                except Exception:
                    st.session_state["_ump_data"] = {}
                try:
                    st.session_state["_run_diffs"] = fetch_team_run_differential(date_str, days=7)
                except Exception:
                    st.session_state["_run_diffs"] = {}
                try:
                    st.session_state["_ml_odds"] = fetch_moneyline_odds(date_str)
                except Exception:
                    st.session_state["_ml_odds"] = {}
                st.rerun()
            else:
                st.warning(f"No games or lineups found for {date_str}.")
                st.info("Opening Day is March 27, 2026. Change the date in the sidebar.")

    # ── DISPLAY TABS ─────────────────────────────────────
    with tab1:
        if st.session_state.plays:
            date_label = st.session_state.analysis_date or date_str
            games = fetch_schedule(date_label)
            if games:
                game_labels = " • ".join([f"{g['away_team']}@{g['home_team']}" for g in games])
                st.caption(f"📅 {date_label} | {len(games)} games: {game_labels}")
            display_leaderboard(st.session_state.plays)
        else:
            st.info("👈 Click **⚾ Run Today's Model** to fetch today's plays")

    with tab2:
        if st.session_state.plays:
            display_tiered_breakdown(st.session_state.plays)
        else:
            st.info("Run the model first to see tiered breakdown.")

    with tab3:
        if st.session_state.plays:
            display_parlay_builder(st.session_state.plays, unit_size)
        else:
            st.info("Run the model first to see parlay recommendations.")

    with tab4:
        if st.session_state.plays:
            display_hits_tab(st.session_state.plays)
        else:
            st.info("Run the model first to see O0.5 any-hit plays.")

    with tab5:
        # ── K Props ──────────────────────────────────────
        if st.session_state.plays:
            ump_data = st.session_state.get("_ump_data", {})
            display_k_props_tab(st.session_state.plays, ump_data)
        else:
            st.info("Run the model first to see K prop scores.")

    with tab6:
        # ── Moneyline ────────────────────────────────────
        if st.session_state.plays:
            _games_for_ml = fetch_schedule(st.session_state.analysis_date or date_str)
            _ml_odds      = st.session_state.get("_ml_odds", {})
            _run_diffs    = st.session_state.get("_run_diffs", {})
            _impl_totals  = {}
            try:
                _impl_totals = fetch_odds(st.session_state.analysis_date or date_str)
            except Exception:
                pass
            _bp_scores    = st.session_state.get("team_bullpen_scores", {})
            display_moneyline_tab(
                games=_games_for_ml,
                plays=st.session_state.plays,
                ml_odds=_ml_odds,
                run_diffs=_run_diffs,
                implied_totals=_impl_totals,
                team_bullpen_scores=_bp_scores,
            )
        else:
            st.info("Run the model first to see moneyline analysis.")

    with tab7:
        if st.session_state.plays:
            display_hr_plays(st.session_state.plays)
        else:
            st.info("Run the model first to see HR plays.")

    with tab8:
        display_dfs_tab(st.session_state.plays if st.session_state.plays else [])

    with tab9:
        display_prizepicks_tab(st.session_state.plays if st.session_state.plays else [])

    with tab10:
        display_results_tracker()


if __name__ == "__main__":
    main()
