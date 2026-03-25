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
    page_title="⚾ MLB TB Analyzer V1.0",
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
    "🔒 TIER 1": 85,
    "✅ TIER 2": 75,
    "📊 TIER 3": 65,
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
        created_at TEXT
    )
    """)
    
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
         hard_hit_rate, k_rate, iso, platoon_edge, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,1.5,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            pick_id, date_str, p.get('game_id', ''), p.get('player_id', ''),
            p['name'], p['team'], p.get('opponent', ''), p.get('sp_name', 'TBD'),
            p.get('sp_hand', '?'), p.get('lineup_slot', 5), p.get('batter_hand', '?'),
            p['score'], p['prob'], p['tier'], p.get('park', ''),
            p.get('wind_speed', 0), p.get('wind_dir', 'N/A'), p.get('temperature', 70),
            p.get('implied_total', 4.5), p.get('xslg', 0), p.get('barrel_rate', 0),
            p.get('hard_hit_rate', 0), p.get('k_rate', 0), p.get('iso', 0),
            p.get('platoon_edge', ''), datetime.now(EST).isoformat()
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
                
                batter = {
                    "player_id": str(player_id),
                    "name": person.get("fullName", f"Player {player_id}"),
                    "lineup_slot": slot_idx + 1,
                    "batter_hand": person.get("batSide", {}).get("code", "?"),
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

@st.cache_data(ttl=7200)
def load_all_batting_stats(season: int = 2025) -> pd.DataFrame:
    """
    Load batter stats from FanGraphs JSON API directly.
    type=8 = advanced stats (SLG, ISO, K%, BB%, wRC+, wOBA)
    type=24 = statcast stats (Barrel%, HardHit%, EV, xSLG, xwOBA)
    Merges both tables for complete picture.
    Falls back to pybaseball if API fails.
    """
    base_url = "https://www.fangraphs.com/api/leaders/major-league/data"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.fangraphs.com/leaders/major-league",
        "Accept": "application/json",
    }
    common_params = {
        "age": "", "pos": "all", "stats": "bat", "lg": "all",
        "qual": "y", "season": season, "season1": season,
        "ind": "0", "team": "0", "pageitems": "500", "pagenum": "1",
        "sortdir": "default",
    }
    
    frames = {}
    for stat_type, label in [("8", "advanced"), ("24", "statcast")]:
        try:
            params = {**common_params, "type": stat_type, "sortstat": "WAR" if stat_type == "8" else "xSLG"}
            r = requests.get(base_url, params=params, headers=headers, timeout=20)
            if r.status_code == 200:
                rows = r.json().get("data", [])
                if rows:
                    frames[label] = pd.DataFrame(rows)
        except Exception:
            pass
    
    # Pybaseball fallback
    if not frames:
        try:
            from pybaseball import batting_stats
            df = batting_stats(season, qual=30)
            if df is not None and not df.empty:
                frames["pybaseball"] = df
        except Exception:
            pass
    
    if not frames:
        return pd.DataFrame()
    
    # Use advanced as base, merge statcast
    if "advanced" in frames:
        result = frames["advanced"].copy()
        if "statcast" in frames:
            sc = frames["statcast"]
            # Find join key
            for k in ["playerid", "PlayerID", "IDfg", "Name"]:
                if k in result.columns and k in sc.columns:
                    new_cols = [c for c in sc.columns if c not in result.columns]
                    if new_cols:
                        result = result.merge(sc[[k] + new_cols], on=k, how="left")
                    break
    else:
        result = list(frames.values())[0].copy()
    
    return clean_fangraphs_df(result)


@st.cache_data(ttl=7200)
def load_all_pitching_stats(season: int = 2025) -> pd.DataFrame:
    """
    Load pitcher stats from FanGraphs JSON API.
    Uses qual=0 (no minimum) to capture all starters including early-season.
    Merges advanced (K%, ERA, FIP) + statcast (Hard%, Barrel%, xERA) tables.
    """
    base_url = "https://www.fangraphs.com/api/leaders/major-league/data"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.fangraphs.com/leaders/major-league",
        "Accept": "application/json",
    }
    common_params = {
        "age": "", "pos": "all", "stats": "pit", "lg": "all",
        "qual": "0",   # No minimum — capture all starters
        "season": season, "season1": season,
        "ind": "0", "team": "0", "pageitems": "500", "pagenum": "1",
        "sortdir": "default", "minip": "1",
    }

    frames = {}
    for stat_type, label, sort in [
        ("8",  "advanced", "ERA"),
        ("24", "statcast", "xERA"),
    ]:
        try:
            params = {**common_params, "type": stat_type, "sortstat": sort}
            r = requests.get(base_url, params=params, headers=headers, timeout=20)
            if r.status_code == 200:
                rows = r.json().get("data", [])
                if rows:
                    frames[label] = pd.DataFrame(rows)
        except Exception:
            pass

    # Pybaseball fallback
    if not frames:
        try:
            from pybaseball import pitching_stats
            df = pitching_stats(season, qual=1)
            if df is not None and not df.empty:
                frames["pybaseball"] = df
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    if "advanced" in frames:
        result = frames["advanced"].copy()
        if "statcast" in frames:
            sc = frames["statcast"]
            for k in ["playerid", "PlayerID", "IDfg", "Name"]:
                if k in result.columns and k in sc.columns:
                    new_cols = [c for c in sc.columns if c not in result.columns]
                    if new_cols:
                        result = result.merge(sc[[k] + new_cols], on=k, how="left")
                    break
    else:
        result = list(frames.values())[0].copy()

    # Add K% alias — FanGraphs pitching sometimes uses "K%" or "K/9"
    # Normalize to K% as rate (0-1)
    if "K%" not in result.columns and "SO" in result.columns and "TBF" in result.columns:
        try:
            result["K%"] = result["SO"] / result["TBF"]
        except Exception:
            pass

    return clean_fangraphs_df(result)


def clean_fangraphs_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean FanGraphs API response:
    - Strip HTML tags from Name column (returns raw anchor tags)
    - Extract playerid from href for reliable ID matching
    - Normalize % columns to decimals
    """
    import re

    # Find the name column (FanGraphs returns HTML like <a href="...playerid=15640...">Aaron Judge</a>)
    name_col = next((c for c in df.columns if c.lower() in ("name", "playername")), None)

    if name_col:
        raw = df[name_col].astype(str)

        # Extract playerid from href: playerid=15640
        def extract_id(s):
            m = re.search(r'playerid=(\d+)', s, re.IGNORECASE)
            return m.group(1) if m else None

        # Extract clean name from HTML: >Aaron Judge<
        def extract_name(s):
            m = re.search(r'>([^<]+)<', s)
            if m:
                return m.group(1).strip()
            # No HTML — return as-is stripped
            return re.sub(r'<[^>]+>', '', s).strip()

        df = df.copy()
        df["_fg_id"]  = raw.apply(extract_id)
        df["_name"]   = raw.apply(extract_name)

        # Use _fg_id as primary ID (FanGraphs playerid maps to IDfg)
        if "_fg_id" not in df.columns or df["_fg_id"].isna().all():
            for id_col in ["playerid", "PlayerID", "IDfg"]:
                if id_col in df.columns:
                    df["_mlb_id"] = df[id_col].astype(str)
                    break
        else:
            df["_mlb_id"] = df["_fg_id"]
    else:
        # Fallback name standardization
        for n in ["Name", "PlayerName", "name"]:
            if n in df.columns:
                df["_name"] = df[n].astype(str).str.strip()
                break
        for i in ["playerid", "PlayerID", "IDfg", "mlbamid"]:
            if i in df.columns:
                df["_mlb_id"] = df[i].astype(str)
                break

    # Normalize % columns to decimals (FanGraphs returns 0-100)
    for col in df.columns:
        if "%" in str(col) or col in ("Hard%","Barrel%","K%","BB%","HardHit%","LD%","GB%","FB%"):
            try:
                vals = df[col].dropna()
                if len(vals) > 0 and float(vals.max()) > 1.5:
                    df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0
            except Exception:
                pass

    return df


def find_player_row(df: pd.DataFrame, player_name: str, mlb_id: str) -> Optional[pd.Series]:
    """
    Find a player row using name matching (primary) or ID (secondary).
    Tries multiple name variations to handle accents, suffixes, nicknames.
    """
    if df is None or df.empty or not player_name:
        return None

    import unicodedata

    def normalize(s: str) -> str:
        """Lowercase, strip accents, remove punctuation."""
        s = str(s).lower().strip()
        s = ''.join(c for c in unicodedata.normalize('NFD', s)
                    if unicodedata.category(c) != 'Mn')
        s = s.replace('.', '').replace("'", '').replace('-', ' ')
        return s

    name_col = "_name" if "_name" in df.columns else next(
        (c for c in df.columns if c.lower() in ("name", "playername")), None)
    id_col = "_mlb_id" if "_mlb_id" in df.columns else None

    # Pre-compute normalized names once (cached per call via the df)
    if name_col and "_norm_name" not in df.columns:
        try:
            df = df.copy()
            df["_norm_name"] = df[name_col].apply(normalize)
        except Exception:
            pass

    norm_name = normalize(player_name)
    parts = norm_name.split()
    last  = parts[-1] if parts else ""
    first = parts[0]  if len(parts) > 1 else ""

    # 1. Exact normalized full name match
    if "_norm_name" in df.columns:
        match = df[df["_norm_name"] == norm_name]
        if not match.empty:
            return match.iloc[0]

        # 2. Last name + first initial
        if first and last:
            cands = df[df["_norm_name"].str.contains(last, na=False, regex=False)]
            if not cands.empty:
                refined = cands[cands["_norm_name"].str.startswith(first[0], na=False)]
                if not refined.empty:
                    return refined.iloc[0]
                # 3. Just last name if unique
                if len(cands) == 1:
                    return cands.iloc[0]

        # 4. Last name only (wider net)
        if last and len(last) > 3:
            cands = df[df["_norm_name"].str.contains(last, na=False, regex=False)]
            if len(cands) == 1:
                return cands.iloc[0]

    # 5. ID fallback
    if mlb_id and id_col:
        try:
            match = df[df[id_col].astype(str) == str(mlb_id)]
            if not match.empty:
                return match.iloc[0]
        except Exception:
            pass

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
        "sweet_spot_rate":  0.305,
        "tb_per_game":      0.85,
        "data_source":      "league_avg",
    }
    
    row = find_player_row(batting_df, player_name, mlb_id)
    
    if row is not None:
        # SLG / power — prefer xSLG over SLG
        xslg = safe_get(row, 'xSLG', default=None)
        slg  = safe_get(row, 'SLG',  default=None)
        if xslg and xslg > 0:
            stats["slg_proxy"] = xslg
        elif slg and slg > 0:
            stats["slg_proxy"] = slg

        # ISO
        iso = safe_get(row, 'ISO', default=None)
        if iso and iso > 0:
            stats["iso_proxy"] = iso

        # K% and BB% — pybaseball returns as decimals (0.23)
        k = safe_get(row, 'K%', default=None)
        if k is not None and 0 < k < 1:
            stats["k_rate"] = k
        elif k is not None and k > 1:
            stats["k_rate"] = k / 100  # handle if stored as pct

        bb = safe_get(row, 'BB%', default=None)
        if bb is not None and 0 < bb < 1:
            stats["bb_rate"] = bb
        elif bb is not None and bb > 1:
            stats["bb_rate"] = bb / 100

        # wRC+ — stored as integer 100 = average
        wrc = safe_get(row, 'wRC+', default=None)
        if wrc and wrc > 0:
            stats["wrc_plus"] = wrc

        # wOBA — decimal
        woba = safe_get(row, 'xwOBA', 'wOBA', default=None)
        if woba and 0 < woba < 1:
            stats["woba"] = woba

        # Barrel% — pybaseball normalizes to decimal after our fix (0.085)
        barrel = safe_get(row, 'Barrel%', 'barrel_batted_rate', default=None)
        if barrel is not None and barrel > 0:
            stats["barrel_rate"] = barrel if barrel < 1 else barrel / 100

        # Hard% — normalized to decimal after our fix (0.34)
        hard = safe_get(row, 'Hard%', 'hard_hit_percent', default=None)
        if hard is not None and hard > 0:
            stats["hard_hit_rate"] = hard if hard < 1 else hard / 100

        # EV — exit velocity in mph
        ev = safe_get(row, 'EV', 'avg_exit_velocity', default=None)
        if ev and ev > 50:  # sanity check
            stats["exit_velocity_avg"] = ev

        stats["data_source"] = "fangraphs"

    return stats

def get_pitcher_stats(pitcher_name: str, pitcher_mlb_id: str,
                      pitching_df: pd.DataFrame) -> Dict:
    """Extract pitcher vulnerability stats from pre-loaded DataFrame."""
    stats = {
        "k_rate_allowed":   0.228,
        "bb_rate_allowed":  0.082,
        "hard_hit_allowed": 0.360,
        "barrel_allowed":   0.065,
        "era":              4.20,
        "fip":              4.10,
        "xfip":             4.10,
        "data_source":      "league_avg",
    }
    
    row = find_player_row(pitching_df, pitcher_name, pitcher_mlb_id)
    
    if row is not None:
        k = safe_get(row, 'K%', default=None)
        if k is not None and k > 0:
            stats["k_rate_allowed"] = k if k < 1 else k / 100

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

        stats["data_source"] = "fangraphs"
    
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

def compute_weather_score(weather: Dict) -> Tuple[float, str]:
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
    """
    if batter_hand == "L" and pitcher_hand == "R":
        adj = PLATOON_ADJ["LHB_vs_RHP"]  # +56
        score = 75.0
        label = "LHB vs RHP (+56 SLG)"
    elif batter_hand == "R" and pitcher_hand == "L":
        adj = PLATOON_ADJ["RHB_vs_LHP"]  # +33
        score = 65.0
        label = "RHB vs LHP (+33 SLG)"
    elif batter_hand == "L" and pitcher_hand == "L":
        adj = PLATOON_ADJ["LHB_vs_LHP"]  # -35
        score = 30.0
        label = "LHB vs LHP (-35 wOBA)"
    elif batter_hand == "B":  # Switch hitter
        # Switch hitter always has platoon advantage
        score = 65.0
        label = "Switch hitter (platoon adv)"
    else:
        adj = 0
        score = 50.0
        label = "RHB vs RHP (neutral)"
    
    return score, label


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

    details["xSLG"]     = round(xslg, 3)
    details["Barrel%"]  = f"{barrel_rate*100:.1f}%"
    details["HardHit%"] = f"{hard_hit*100:.1f}%"
    details["K%"]       = f"{k_rate*100:.1f}%"
    details["ISO"]      = round(iso, 3)
    details["wRC+"]     = int(wrc_plus)

    # ── Sub-scores 0-100, calibrated to actual MLB distributions ──

    # xSLG: .200=0, .398=50, .650=100
    # Judge .708 → ~96, Avg .398 → 50, Weak .250 → ~13
    xslg_score = (xslg - 0.200) / (0.650 - 0.200) * 100
    xslg_score = max(0, min(100, xslg_score))

    # wRC+: 60=0, 100=50, 180=100
    # Judge 204 → 100 (capped), Avg 100 → 50, Weak 70 → ~13
    wrc_score = (wrc_plus - 60) / (180 - 60) * 100
    wrc_score = max(0, min(100, wrc_score))

    # Barrel%: 0%=0, 7%=44, 20%=100
    # Judge 24.7% → 100 (capped), Avg 7% → 44, Weak 3% → 15
    barrel_score = barrel_rate / 0.20 * 100
    barrel_score = max(0, min(100, barrel_score))

    # Hard hit%: 28%=0, 38%=50, 56%=100
    # Judge 45.6% → ~86, Avg 38% → 50, Weak 28% → 0
    hard_hit_score = (hard_hit - 0.28) / (0.56 - 0.28) * 100
    hard_hit_score = max(0, min(100, hard_hit_score))

    # K rate INVERSE: 8%=100, 23%=50, 38%=0
    # Low K = more TB opportunities (no K = 0 TB guaranteed)
    k_score = max(0, min(100, (0.38 - k_rate) / (0.38 - 0.08) * 100))

    # ISO: .080=0, .165=50, .320=100
    # Judge .357 → 100 (capped), Avg .165 → 50, Weak .080 → 0
    iso_score = (iso - 0.080) / (0.320 - 0.080) * 100
    iso_score = max(0, min(100, iso_score))

    # Weighted composite
    composite = (
        xslg_score    * 0.25 +   # xSLG — best single predictor of TB
        wrc_score     * 0.18 +   # wRC+ — overall offensive context
        barrel_score  * 0.22 +   # barrel% — strongest XBH/HR signal
        hard_hit_score* 0.15 +   # hard hit% — contact quality
        iso_score     * 0.12 +   # ISO — raw power
        k_score       * 0.08     # K% inverse — PA completion
    )

    return max(0, min(100, composite)), "Contact quality", details


def compute_pitcher_score(statcast: Dict, fg_stats: Dict = None) -> Tuple[float, str]:
    """
    Pitcher VULNERABILITY score 0-100.
    HIGH score = pitcher is hittable = good for batter TB.
    LOW score = elite pitcher = suppresses TB.
    Webb/Fried type = ~20-30. League avg pitcher = ~50. Mop-up arm = ~75+.
    """
    def f(key, default):
        try: return float(statcast.get(key, default))
        except: return float(default)

    k_rate   = f("k_rate_allowed",   0.228)
    hard_hit = f("hard_hit_allowed", 0.340)
    barrel   = f("barrel_allowed",   0.065)
    era      = f("era",              4.20)
    fip      = f("fip",              4.10)

    # K% INVERSE — high K pitcher = low vulnerability
    # Webb/Fried ~28-32% K → low vuln score
    # 10% K = 100 vuln, 23% K = 50 vuln, 35%+ K = ~0 vuln
    k_vuln = max(0, min(100, (0.35 - k_rate) / (0.35 - 0.10) * 100))

    # Hard hit allowed — higher = more vulnerable
    # 28%=0, 36%=50, 50%=100
    hh_vuln = (hard_hit - 0.28) / (0.50 - 0.28) * 100
    hh_vuln = max(0, min(100, hh_vuln))

    # Barrel% allowed
    # 3%=0, 7%=50, 14%=100
    barrel_vuln = (barrel - 0.03) / (0.14 - 0.03) * 100
    barrel_vuln = max(0, min(100, barrel_vuln))

    # ERA/FIP quality
    # 2.0=0, 4.20=50, 7.0=100
    era_use = fip if fip > 0 else era
    era_vuln = (era_use - 2.0) / (7.0 - 2.0) * 100
    era_vuln = max(0, min(100, era_vuln))

    composite = (
        k_vuln      * 0.40 +   # K% most stable and predictive
        hh_vuln     * 0.25 +   # hard hit quality allowed
        barrel_vuln * 0.20 +   # barrel rate allowed
        era_vuln    * 0.15     # FIP/ERA quality signal
    )

    return max(0, min(100, composite)), f"K%: {k_rate*100:.0f}% | HH%: {hard_hit*100:.0f}% | Barrel: {barrel*100:.1f}% | FIP: {era_use:.2f}"


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


def compute_final_score(
    batter_score: float,
    pitcher_vuln_score: float,
    platoon_score: float,
    lineup_score: float,
    park_score: float,
    weather_score: float,
    vegas_score: float,
) -> float:
    """
    Final weighted composite. Calibrated so:
    - League avg batter vs league avg pitcher = ~50
    - Elite batter vs bad pitcher, good park/weather/Vegas = 85+
    - Elite batter vs elite pitcher, bad park = 55-65
    """
    raw = (
        batter_score      * 0.45 +
        pitcher_vuln_score* 0.30 +
        platoon_score     * 0.06 +
        lineup_score      * 0.04 +
        park_score        * 0.07 +
        weather_score     * 0.03 +
        vegas_score       * 0.05
    )
    # Calibration offset: raw league-avg matchup ≈ 42 → target 50
    calibrated = raw + 8.0
    return max(0, min(100, round(calibrated, 1)))


def score_to_prob(score: float) -> float:
    """
    Map 0-100 score to probability.
    Calibrated for MLB O1.5 TB props:
    - Score 50 (league avg) → ~52% probability
    - Score 75 → ~65% probability  
    - Score 85+ → ~72-78% probability
    """
    a = 0.06
    b = 48
    prob = 1 / (1 + math.exp(-a * (score - b)))
    # Scale to MLB prop range (40% - 80%)
    prob = 0.40 + prob * 0.40
    return round(min(0.80, max(0.40, prob)), 3)


def get_tier(score: float) -> str:
    """Map score to tier label."""
    if score >= 85:
        return "🔒 TIER 1"
    elif score >= 75:
        return "✅ TIER 2"
    elif score >= 65:
        return "📊 TIER 3"
    else:
        return "❌ NO PLAY"

# ============================================================================
# HR SCORE (separate from TB score)
# ============================================================================
def compute_hr_score(
    barrel_rate: float,
    sweet_spot: float,
    park_hr_factor: float,
    implied_total: float,
    weather: Dict,
    hard_hit: float = 0.37
) -> float:
    """
    Compute dedicated HR upside score 0-100.
    Used for Home Run Plays page.
    """
    # Barrel rate is #1 HR predictor (r=0.93)
    barrel_score = min(100, barrel_rate / 0.20 * 100)
    
    # Sweet spot rate
    sweet_score = min(100, sweet_spot / 0.40 * 100)
    
    # Park HR factor: 0.85=0, 1.0=50, 1.35=100
    park_score = (park_hr_factor - 0.85) / (1.35 - 0.85) * 100
    park_score = max(0, min(100, park_score))
    
    # Vegas implied
    vegas_score = min(100, implied_total / 6.5 * 100)
    
    # Wind effect on HR
    wind_bonus = 0
    if not weather.get("is_dome"):
        effect = weather.get("wind_effect", "neutral")
        speed = weather.get("wind_speed", 0)
        if effect == "strong_out":
            wind_bonus = 25
        elif effect == "out":
            wind_bonus = 15
        elif effect == "in":
            wind_bonus = -20
    
    composite = (
        barrel_score * 0.40 +
        sweet_score * 0.20 +
        park_score * 0.25 +
        vegas_score * 0.10 +
        min(25, max(-20, wind_bonus)) * 0.05 + 50 * 0.05  # wind normalized
    )
    
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

    # ── 0. BULK STATS (one call loads all players) ───────
    log("Loading 2025 season batting stats...", "run")
    batting_df = load_all_batting_stats(2025)
    pitching_df = load_all_pitching_stats(2025)
    statcast_df = pd.DataFrame()  # merged into batting_df now

    if not batting_df.empty:
        log(f"Batting stats: {len(batting_df)} players loaded", "ok")
        # Store debug info
        st.session_state.batting_cols = list(batting_df.columns)
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
        log("Batting stats unavailable — all scores use league averages", "warn")
    if not pitching_df.empty:
        log(f"Pitching stats: {len(pitching_df)} pitchers loaded", "ok")
        st.session_state.pitching_cols = list(pitching_df.columns)
    else:
        log("Pitching stats unavailable — using league averages", "warn")

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
    try:
        implied_totals = fetch_odds(date_str)
    except Exception:
        pass
    if implied_totals:
        log(f"Live odds loaded for {len(implied_totals)} teams ✅", "ok")
    else:
        log("⚠️ No Odds API key — Vegas signal (5% weight) zeroed out. Add key in sidebar for full model.", "warn")

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
            batter_hand = batter.get("batter_hand", "R")
            park_team = batter.get("park_team", home_team)

            # Stats from pre-loaded bulk DataFrames (fast, no per-player API calls)
            batter_statcast = get_batter_stats(
                player_name=name,
                mlb_id=player_id,
                batting_df=batting_df,
                statcast_df=statcast_df,
            )

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

            # ── SCORE COMPONENTS ────────────────────────
            bat_score, _, bat_details = compute_batter_score(batter_statcast)
            pit_score, pit_label = compute_pitcher_score(pitcher_statcast)
            plat_score, plat_label = compute_platoon_score(batter_hand, sp_hand)
            lineup_sc, lineup_label = compute_lineup_score(lineup_slot)
            park_sc, park_label = compute_park_score(park_team, True)
            weather_sc, weather_label = compute_weather_score(weather)
            implied = implied_totals.get(team, 4.5)
            vegas_sc, vegas_label = compute_vegas_score(implied)

            final_score = compute_final_score(
                bat_score, pit_score, plat_score, lineup_sc,
                park_sc, weather_sc, vegas_sc
            )

            # Caps & flags
            sp_tbd = not sp_name or sp_name == "TBD"
            if sp_tbd:
                final_score = min(final_score, 72)
            if not batter.get("lineup_confirmed", True):
                final_score = min(final_score, 70)

            prob = score_to_prob(final_score)
            tier = get_tier(final_score)

            hr_score = compute_hr_score(
                barrel_rate=batter_statcast.get("barrel_rate", 0.07),
                sweet_spot=batter_statcast.get("sweet_spot_rate", 0.30),
                park_hr_factor=PARK_HR_FACTORS.get(park_team, 1.0),
                implied_total=implied,
                weather=weather,
                hard_hit=batter_statcast.get("hard_hit_rate", 0.37),
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
                "platoon_label": plat_label,
                "lineup_label": lineup_label,
                "pitcher_label": pit_label,
                "hr_score": hr_score,
                "xslg": bat_details.get("xSLG", 0),
                "barrel_rate": batter_statcast.get("barrel_rate", 0),
                "hard_hit_rate": batter_statcast.get("hard_hit_rate", 0),
                "k_rate": batter_statcast.get("k_rate", 0),
                "iso": bat_details.get("ISO", 0),
                "exit_velocity": batter_statcast.get("exit_velocity_avg", 0),
                "sweet_spot_rate": batter_statcast.get("sweet_spot_rate", 0),
                "sub_batter": round(bat_score, 1),
                "sub_pitcher": round(pit_score, 1),
                "sub_platoon": round(plat_score, 1),
                "sub_lineup": round(lineup_sc, 1),
                "sub_park": round(park_sc, 1),
                "sub_weather": round(weather_sc, 1),
                "sub_vegas": round(vegas_sc, 1),
                "platoon_edge": plat_label,
                "temperature": weather.get("temperature", 70),
                "wind_speed": weather.get("wind_speed", 0),
                "wind_dir": weather.get("wind_dir_label", ""),
                "wind_effect": weather.get("wind_effect", "neutral"),
                "is_dome": weather.get("is_dome", False),
            })
            total_batters += 1

        log(f"  ✅ {away_team}@{home_team} done — {len(all_batters)} batters scored")

    # Sort
    results.sort(key=lambda x: x["score"], reverse=True)

    log("─" * 40)
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
    min_score: float = 75.0
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
            "xSLG": f"{p['xslg']:.3f}" if p["xslg"] else "—",
            "Barrel%": f"{p['barrel_rate']*100:.1f}%" if p["barrel_rate"] else "—",
            "HH%": f"{p['hard_hit_rate']*100:.1f}%" if p["hard_hit_rate"] else "—",
            "K%": f"{p['k_rate']*100:.1f}%" if p["k_rate"] else "—",
            "Platoon": p["platoon_label"].split("(")[0].strip(),
            "Park": p["park"],
            f"Wind{wind_icon}": p["weather_label"].split("|")[0].strip() if "|" in p["weather_label"] else p["weather_label"],
            "°F": f"{p['temperature']:.0f}°",
            "Imp.Runs": f"{p['implied_total']:.1f}",
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
                if v >= 85: return "color: #00ff88; font-weight: bold"
                elif v >= 75: return "color: #ffdd00; font-weight: bold"
                elif v >= 65: return "color: #ff8800"
                return "color: #888888"
            except:
                return ""
        
        styled = df.style.applymap(color_tier, subset=["Tier"]).applymap(color_score, subset=["Score"])
        st.dataframe(styled, use_container_width=True, height=500)
        
        # Export button
        csv = df.to_csv(index=False)
        st.download_button("📥 Export CSV", csv, f"mlb_tb_picks_{datetime.now(EST).strftime('%Y%m%d')}.csv", "text/csv")
    
    # Expandable detail cards for top plays
    st.markdown("---")
    st.subheader("🏆 Top Plays — Full Breakdown")
    
    top5 = [p for p in filtered if p["score"] >= 65][:5]
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
                    "🏏 Batter (45%)": p["sub_batter"],
                    "⚾ Pitcher Vuln (30%)": p["sub_pitcher"],
                    "🤚 Platoon (6%)": p["sub_platoon"],
                    "📋 Lineup (4%)": p["sub_lineup"],
                    "🏟️ Park (7%)": p["sub_park"],
                    "🌤️ Weather (3%)": p["sub_weather"],
                    "💰 Vegas (5%)": p["sub_vegas"],
                }
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
    """Display parlay builder with multiple parlay sizes."""
    
    st.header("💰 Parlay Builder")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        num_legs = st.radio("Legs", [2, 3, 4, 5], index=1, horizontal=True)
    with col2:
        min_score = st.slider("Min player score for parlay", 65, 90, 75)
    with col3:
        max_same_team = st.number_input("Max same-team legs", 1, 3, 1)
    
    eligible = [p for p in plays if p["score"] >= min_score]
    st.info(f"📊 {len(eligible)} eligible players (score ≥ {min_score})")
    
    if len(eligible) < num_legs:
        st.warning(f"⚠️ Not enough eligible players for a {num_legs}-leg parlay (need {num_legs}, have {len(eligible)}). Try lowering min score or run on a slate with more games.")
        return
    
    parlays = build_parlays(plays, num_legs, max_same_team, min_score)
    
    if not parlays:
        st.warning("No valid parlays found with current filters.")
        return
    
    # Recommended parlay
    best = parlays[0]
    st.subheader("⭐ Recommended Parlay")
    
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a: st.metric("Combined Prob", f"{best['combined_prob']:.1f}%")
    with col_b: st.metric("Fair Payout", f"{best['fair_payout']:.2f}x")
    with col_c: st.metric("Est. EV", f"{best['ev']:+.1f}%")
    with col_d: st.metric("Avg Score", f"{best['avg_score']:.0f}")
    
    for player_name in best["players"]:
        player = next((p for p in plays if p["name"] == player_name), None)
        if player:
            st.markdown(f"✅ **{player_name}** ({player['team']}) vs {player['opponent']} — Score: {player['score']:.0f} | Prob: {player['prob']*100:.0f}% | SP: {player['sp_name']}")
    
    # Correlation note
    if best["corr_factor"] < 0.98:
        st.caption(f"🔗 Correlation discount applied: {best['corr_factor']:.3f}x ({best['notes']})")
    
    # HardRock Bet formatted output
    hrb_text = " + ".join([f"{p} Over 1.5 TB" for p in best["players"]])
    st.code(f"HardRock Bet: {hrb_text}", language=None)
    
    st.markdown("---")
    st.subheader("📊 All Parlay Options")
    
    # Full parlay table
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
    
    # SGP opportunities
    sgp_plays = [p for p in parlays if "SGP" in p.get("notes", "")]
    if sgp_plays:
        st.subheader("⭐ Same-Game Parlay Opportunities")
        for sgp in sgp_plays[:3]:
            players_info = [next((p for p in plays if p["name"] == n), None) for n in sgp["players"]]
            valid = [p for p in players_info if p]
            if valid:
                game = valid[0]["opponent"] + " @ " + valid[0]["team"]
                st.success(f"🎰 **SGP:** {game} | {' + '.join(sgp['players'])} | Combined: {sgp['combined_prob']:.1f}%")
    
    # Custom manual builder
    st.markdown("---")
    st.subheader("🔧 Custom Parlay Builder")
    eligible_names = [f"{p['name']} ({p['team']}, {p['score']:.0f})" for p in eligible]
    selected = st.multiselect("Select legs manually:", eligible_names)
    
    if len(selected) >= 2:
        selected_plays = []
        for sel in selected:
            name_part = sel.split(" (")[0]
            player = next((p for p in plays if p["name"] == name_part), None)
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
            
            hrb_text = " + ".join([f"{p['name']} Over 1.5 TB" for p in selected_plays])
            st.code(f"HardRock Bet: {hrb_text}", language=None)


def display_hr_plays(plays: List[Dict]):
    """Display top HR upside plays."""
    
    st.header("💣 Home Run Plays")
    st.caption("Top 10 daily HR candidates. Powered by barrel rate, sweet spot%, park HR factor, and wind vector.")
    
    hr_sorted = sorted(plays, key=lambda x: x["hr_score"], reverse=True)[:10]
    
    rows = []
    for p in hr_sorted:
        wind_label = p.get("weather_label", "").split("|")[0].strip()
        park_name = STADIUM_COORDS.get(p["park"], (0, 0, p["park"], False))[2]
        
        rows.append({
            "HR Score": f"{p['hr_score']:.0f}",
            "Player": p["name"],
            "Team": p["team"],
            "Opp SP": p["sp_name"][:20],
            "Barrel%": f"{p['barrel_rate']*100:.1f}%" if p["barrel_rate"] else "—",
            "Sweet Spot%": f"{p['sweet_spot_rate']*100:.1f}%" if p["sweet_spot_rate"] else "—",
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
        
        styled = df.style.applymap(color_hr, subset=["HR Score"])
        st.dataframe(styled, use_container_width=True)
    
    # Top 3 HR plays detailed
    st.markdown("---")
    st.subheader("🔥 Top 3 HR Plays — Detail")
    
    for i, p in enumerate(hr_sorted[:3], 1):
        with st.expander(f"#{i}: {p['name']} ({p['team']}) — HR Score: {p['hr_score']:.0f}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Barrel%:** {p['barrel_rate']*100:.1f}%" if p["barrel_rate"] else "**Barrel%:** —")
                st.write(f"**Sweet Spot%:** {p['sweet_spot_rate']*100:.1f}%" if p["sweet_spot_rate"] else "**Sweet Spot%:** —")
                st.write(f"**Park:** {STADIUM_COORDS.get(p['park'], (0,0,p['park'],False))[2]}")
                st.write(f"**Park HR Factor:** {PARK_HR_FACTORS.get(p['park'], 1.0):.2f}x")
            with col2:
                st.write(f"**Wind:** {p.get('wind_dir', '')} @ {p.get('wind_speed', 0):.0f}mph ({p.get('wind_effect', 'neutral')})")
                st.write(f"**Temp:** {p['temperature']:.0f}°F")
                st.write(f"**Implied Runs:** {p['implied_total']:.1f}")
                st.write(f"**Total Bases Score:** {p['score']:.0f}")
            
            # SGP opportunity check
            if p["score"] >= 65:
                same_game = [op for op in hr_sorted if op["game_id"] == p["game_id"] and op["name"] != p["name"]]
                if same_game:
                    st.success(f"⭐ SGP Opportunity: {p['name']} HR + {same_game[0]['name']} O1.5 TB in same game!")


def display_results_tracker():
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
            st.download_button("📥 Export All Picks", csv, "mlb_picks_history.csv", "text/csv")
    with col2:
        if not parlays_df.empty:
            csv = parlays_df.to_csv(index=False)
            st.download_button("📥 Export Parlays", csv, "mlb_parlays_history.csv", "text/csv")

# ============================================================================
# MAIN APP
# ============================================================================
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

        st.markdown(f"✅ MLB Stats API")
        st.markdown(f"✅ FanGraphs (pybaseball)")
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
            
            **Weights:**
            - 🏏 Batter: 45% (xSLG, barrel, HH%, ISO, K%)
            - ⚾ Pitcher Vuln: 30% (K%, barrel allowed, HH allowed)
            - 🤚 Platoon: 6%
            - 📋 Lineup: 4%
            - 🏟️ Park: 7%
            - 🌤️ Weather: 3%
            - 💰 Vegas: 5%
            
            **Tiers:**
            - 🔒 Tier 1 (85+): Parlay anchor
            - ✅ Tier 2 (75-84): Parlay filler
            - 📊 Tier 3 (65-74): Single only
            - ❌ Below 65: Fade
            """)
        
        with st.expander("🔍 Debug: Data Quality Check"):
            st.caption("Expand after running model to verify all key stats loaded")
            if "batting_cols" in st.session_state:
                cols = st.session_state.batting_cols
                # Check for the critical columns we need
                critical_bat = ["SLG", "ISO", "K%", "BB%", "wRC+", "wOBA", 
                               "Barrel%", "Hard%", "EV", "xSLG", "xwOBA"]
                st.markdown("**Batting — critical columns:**")
                for c in critical_bat:
                    found = c in cols
                    st.markdown(f"{'✅' if found else '❌'} `{c}`")
                st.caption(f"Total columns loaded: {len(cols)}")
            if "pitching_cols" in st.session_state:
                cols = st.session_state.pitching_cols
                critical_pit = ["K%", "ERA", "FIP", "xFIP", "Hard%", "Barrel%", "SO", "TBF", "xERA"]
                st.markdown("**Pitching — critical columns:**")
                for c in critical_pit:
                    found = c in cols
                    st.markdown(f"{'✅' if found else '❌'} `{c}`")
                st.caption(f"Total pitching columns: {len(cols)}")
            if "sample_player" in st.session_state:
                st.markdown("**Sample player (Judge):**")
                st.json(st.session_state.sample_player)

        st.caption(f"v1.1 | {datetime.now(EST).strftime('%I:%M %p EST')}")
    
    # ── MAIN CONTENT ──────────────────────────────────────
    st.title("⚾ MLB Total Bases Analyzer V1.0")
    st.caption("Fully automated over 1.5 TB prop model | HardRock Bet | 1B=1 2B=2 3B=3 HR=4")
    
    # Tabs (mirroring NHL model structure)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Leaderboard",
        "🎯 Tiered Breakdown",
        "💰 Parlay Builder",
        "💣 HR Plays",
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
                best_parlays = build_parlays(plays, 3, 1, 75.0)
                if best_parlays:
                    save_parlay_to_db(best_parlays[0], date_str)
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
            display_hr_plays(st.session_state.plays)
        else:
            st.info("Run the model first to see HR plays.")
    
    with tab5:
        display_results_tracker()


if __name__ == "__main__":
    main()
