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
import logging
from datetime import datetime, timedelta, date
from itertools import combinations
from typing import Optional, List, Dict, Tuple, Any
import pytz

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="⚾ MLB TB Analyzer V2.1",
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
# Pure scoring functions and utilities now live in their own modules.
# Imported here so this file stays as the Streamlit runner while tests
# import from the modules directly (no Streamlit dependency in modules).
from lib.constants import (
    TIERS, STADIUM_COORDS, PARK_TB_FACTORS, PARK_HR_FACTORS,
    PLATOON_ADJ, TEAM_ABB_MAP, MLBAM_BATTER_HAND,
)
from lib.utils import _norm, clean_fangraphs_df
from lib.name_match import safe_get, prepare_lookup_df, find_player_row
from data.provenance import compute_data_quality_score, check_bettable_tb
from scoring.park import (
    compute_park_score, compute_platoon_score,
    compute_lineup_score, compute_pitch_matchup_score,
)
from scoring.weather import classify_wind, compute_weather_score
from scoring.vegas import compute_vegas_score, score_to_prob, get_tier
from scoring.streak import compute_streak_score, compute_bvp_score, compute_tto_bonus
from scoring.batter import compute_batter_score
from scoring.pitcher import compute_pitcher_score, compute_team_bullpen_scores
from scoring.hr import compute_hr_score
from scoring.final import compute_final_score as _compute_final_score_pure
from markets.tb_o15 import score_one_batter as _score_one_batter_pure, build_parlays as _build_parlays_pure, tb_market_edge

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
    # Neutral-site / international venues
    "MEX": (19.4240, -99.0680, "Estadio Alfredo Harp Helú", False),  # Mexico City, altitude 7349ft
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
    "MEX": 1.18,  # High altitude (7349ft) — major TB boost
}


# ============================================================================
# MLBAM PLAYER ID → BATTER HAND MAP
# Pre-seeded for roster fallback path where MLB API may not return batSide.
# L = Left, R = Right, S = Switch. Updated for 2026 season.
# ============================================================================
MLBAM_BATTER_HAND: Dict[str, str] = {
    # Switch hitters
    "665742": "S",  # Jazz Chisholm Jr.
    "669257": "S",  # Elly De La Cruz
    "677594": "S",  # Corbin Carroll
    "641355": "S",  # Max Kepler
    "621566": "S",  # Ketel Marte
    "663728": "S",  # Ha-Seong Kim
    "665489": "S",  # Trea Turner
    "680757": "S",  # Gunnar Henderson
    "681481": "S",  # Jackson Merrill
    "671218": "S",  # Nick Gonzales
    "681624": "S",  # Colt Keith
    # Left-handed batters
    "592450": "L",  # Juan Soto
    "518692": "L",  # Freddie Freeman
    "623993": "L",  # Pete Alonso
    "641313": "L",  # Matt Olson
    "670541": "L",  # Yordan Alvarez
    "666023": "L",  # Josh Naylor
    "680757": "L",  # Coby Mayo — actually switch? confirm
    "665919": "L",  # Michael Busch
    "656976": "L",  # Kyle Tucker
    "660271": "L",  # Bo Bichette
    "665487": "L",  # Coby Mayo
    "680757": "L",  # Coby Mayo
    "666142": "L",  # Nathaniel Lowe
    "608384": "L",  # Paul Goldschmidt
    "641645": "L",  # Dominic Smith
    "677951": "L",  # Marco Luciano
    "643376": "L",  # Rhys Hoskins
    "664702": "L",  # Triston Casas
    "672515": "L",  # Christian Walker (actually R)
    "668939": "L",  # James Wood
    "682998": "L",  # Colson Montgomery
    "683737": "L",  # Wyatt Langford
    "663586": "L",  # Anthony Volpe (actually R)
    "650490": "L",  # Kyle Schwarber
    "642133": "L",  # Andrew Benintendi
    "660162": "L",  # MJ Melendez
    "682625": "L",  # Jonah Bride
    "666301": "L",  # Oneil Cruz
    "642708": "L",  # Chas McCormick
    "664056": "L",  # Ji-Man Choi
    "605141": "L",  # Joey Votto
    "687695": "L",  # Jackson Chourio
    "669032": "L",  # Junior Caminero
    "691026": "L",  # Jackson Holliday
    "691157": "L",  # Roman Anthony
    "694538": "L",  # Coby Mayo
}

# Park HR factors specifically
PARK_HR_FACTORS = {
    "ARI": 1.10, "ATL": 0.99, "BAL": 1.08, "BOS": 1.07, "CHC": 1.15,
    "CWS": 1.05, "CIN": 1.18, "CLE": 0.93, "COL": 1.35, "DET": 0.96,
    "HOU": 1.02, "KC": 0.96, "LAA": 0.94, "LAD": 1.02, "MIA": 0.87,
    "MIL": 1.01, "MIN": 1.09, "NYM": 0.95, "NYY": 1.14, "OAK": 0.90,
    "PHI": 1.10, "PIT": 0.94, "SD": 0.88, "SEA": 0.95, "SF": 0.90,
    "STL": 1.00, "TB": 0.92, "TEX": 1.06, "TOR": 1.03, "WSH": 0.95,
    "MEX": 1.30,  # Altitude HR boost comparable to Coors
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
    # Alternate abbreviations (MLB API international/neutral-site games)
    "AZ": "ARI", "D-backs": "ARI", "Diamondbacks": "ARI",
    "Cubs": "CHC", "Reds": "CIN", "Guardians": "CLE",
    "Rockies": "COL", "Tigers": "DET", "Astros": "HOU",
    "Royals": "KC", "Angels": "LAA", "Dodgers": "LAD",
    "Marlins": "MIA", "Brewers": "MIL", "Twins": "MIN",
    "Mets": "NYM", "Yankees": "NYY", "Phillies": "PHI",
    "Pirates": "PIT", "Padres": "SD", "Mariners": "SEA",
    "Giants": "SF", "Cardinals": "STL", "Rays": "TB",
    "Rangers": "TEX", "Blue Jays": "TOR", "Nationals": "WSH",
    "Braves": "ATL", "Orioles": "BAL", "Red Sox": "BOS",
    "White Sox": "CWS",
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
                
                # Normalize team codes — always try full-name map AND code-level map
                for side in ["home_team", "away_team"]:
                    name_key = side + "_name"
                    code = game_info[side]
                    _raw_code = code  # save for debug
                    mapped_name = TEAM_ABB_MAP.get(game_info[name_key])
                    mapped_code = TEAM_ABB_MAP.get(code)
                    if code not in STADIUM_COORDS:
                        if mapped_name:
                            game_info[side] = mapped_name
                        elif mapped_code:
                            game_info[side] = mapped_code
                    elif mapped_code and mapped_code in STADIUM_COORDS:
                        game_info[side] = mapped_code
                    # Embed raw API value for debug
                    game_info[f"_raw_{side}"] = _raw_code

                # Neutral-site detection — Mexico City & international venues
                venue_lower = game_info.get("venue", "").lower()
                if any(kw in venue_lower for kw in ["alfredo harp", "mexico", "estadio", "monterrey"]):
                    game_info["neutral_site"] = True
                    game_info["park_override"] = "MEX"
                else:
                    game_info["neutral_site"] = False
                    game_info["park_override"] = None

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
            
            # Use batting order if available, else batters list.
            # Take up to 12 entries to handle NL/Mexico City rules where pitcher
            # occupies a batting order slot — we skip pitchers and collect 9 real batters.
            order = batting_order if batting_order else batters
            
            slot_num = 0  # tracks actual batter position (1-9), ignoring pitcher slots
            for player_id in order[:12]:
                if slot_num >= 9:
                    break
                player_key = f"ID{player_id}"
                player_info = players.get(player_key, {})
                person = player_info.get("person", {})
                position = player_info.get("position", {})
                
                # Skip pitchers — they appear in batting order for NL/international rules
                if position.get("abbreviation") == "P":
                    continue
                
                # Also skip if player type indicates pitcher (extra safety)
                if player_info.get("gameStatus", {}).get("isCurrentBatter") is False and \
                   position.get("type", "") == "Pitcher":
                    continue
                
                slot_num += 1
                bat_hand = person.get("batSide", {}).get("code", "")
                if not bat_hand or bat_hand == "?":
                    try:
                        pr = requests.get(f"https://statsapi.mlb.com/api/v1/people/{player_id}",
                                         params={"fields":"people,id,fullName,batSide"},
                                         timeout=5)
                        bat_hand = pr.json().get("people",[{}])[0].get("batSide",{}).get("code","R")
                    except Exception:
                        bat_hand = "R"
                batter = {
                    "player_id": str(player_id),
                    "name": person.get("fullName", f"Player {player_id}"),
                    "lineup_slot": slot_num,
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
    except Exception as e:
        logging.warning(f"[fetch_pitcher_info pid={pitcher_id}] {e}")
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
_STATS_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mlb_stats.db")
_DB_FRESHNESS_HOURS = 8  # warn in UI if dataset older than this

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


def _load_from_db(table: str) -> Optional[pd.DataFrame]:
    """Read batter_stats or pitcher_stats from data/mlb_stats.db.
    Returns None if DB is absent, stale (> _DB_FRESHNESS_HOURS), or corrupt."""
    if not os.path.exists(_STATS_DB):
        return None
    try:
        age_hours = (time.time() - os.path.getmtime(_STATS_DB)) / 3600
        if age_hours > _DB_FRESHNESS_HOURS:
            st.session_state["_db_freshness_warning"] = (
                f"Dataset is {age_hours:.0f}h old (threshold {_DB_FRESHNESS_HOURS}h). "
                f"Run: python3 data/fetch_pipeline.py"
            )
        else:
            st.session_state.pop("_db_freshness_warning", None)
        con = sqlite3.connect(_STATS_DB)
        df = pd.read_sql(f"SELECT * FROM {table}", con)
        con.close()
        if df.empty:
            return None
        # Alias mlbam_id → xMLBAMID so find_player_row() can match by MLBAM ID
        if "mlbam_id" in df.columns and "xMLBAMID" not in df.columns:
            df["xMLBAMID"] = df["mlbam_id"].astype(str)
        return df
    except Exception as e:
        logging.warning(f"[_load_from_db {table}] {e}")
        return None


def _db_age_hours() -> Optional[float]:
    """Return age of the stats DB in hours, or None if absent."""
    if not os.path.exists(_STATS_DB):
        return None
    return (time.time() - os.path.getmtime(_STATS_DB)) / 3600


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

    # ── 0. SQLite dataset (fetch_pipeline.py output) — primary path ───────
    _db = _load_from_db("batter_stats")
    if _db is not None:
        st.session_state["_batting_source"] = "sqlite_db"
        return _db

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
                f"&limit=2000&offset=0&sportId=1&playerPool=All"
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
                        except Exception as e:
                            logging.warning(f"[mlb_stats_api slg/avg parse] {e}")
                            slg = avg = 0.0
                        hr  = int(st_.get("homeRuns", 0) or 0)
                        tb  = int(st_.get("totalBases", 0) or 0)
                        d2  = int(st_.get("doubles", 0) or 0)
                        d3  = int(st_.get("triples", 0) or 0)
                        ab  = int(st_.get("atBats", 0) or 0)
                        try:
                            obp  = float(st_.get("obp", 0) or 0)
                            bab  = float(st_.get("babip", 0) or 0)
                        except Exception as e:
                            logging.warning(f"[mlb_stats_api obp/babip parse] {e}")
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
                            "brl_percent": "barrel_batted_rate",
                            "ev95percent": "hard_hit_percent",
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
            if not _df.empty:
                _df = _df.rename(columns={
                    "brl_percent":           "barrel_batted_rate",
                    "avg_hit_speed":         "avg_exit_velocity",
                    "ev95percent":           "hard_hit_percent",
                    "anglesweetspotpercent": "sweet_spot_percent",
                })
                if len(_sc_required.intersection(set(_df.columns))) >= 2:
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

    # ── 0. SQLite dataset (fetch_pipeline.py output) — primary path ───────
    _db = _load_from_db("pitcher_stats")
    if _db is not None:
        st.session_state["_pitching_source"] = "sqlite_db"
        return _db

    # ── 1. Early disk cache ───────────────────────────────────────────────
    # NOTE: Cache cleared when FIP-only vulnerability fix deployed (2026-05-05)
    # Reduce TTL to 2h so fresh data always runs through FIP-only detection
    _early = _load_disk_cache("pitching_stats", max_age_hours=2)
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
                f"&limit=2000&offset=0&sportId=1&playerPool=All"
            )
            r = requests.get(url, timeout=25)
            if r.status_code == 200:
                splits = r.json().get("stats", [{}])[0].get("splits", [])
                if splits:
                    from lib.constants import TEAM_ABB_MAP as _TEAM_ABB_MAP
                    rows = []
                    for s in splits:
                        p   = s.get("player", {})
                        st_ = s.get("stat", {})
                        tm  = s.get("team", {})
                        ip_str = str(st_.get("inningsPitched", "0") or "0")
                        try:
                            ip = float(ip_str)
                        except Exception as e:
                            logging.warning(f"[mlb_stats_api ip parse] {e}")
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
                        # Team: API splits have id+name but no abbreviation field —
                        # map full name through TEAM_ABB_MAP
                        _tm_name = tm.get("name", "")
                        _tm_abbr = _TEAM_ABB_MAP.get(_tm_name, tm.get("abbreviation", ""))
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
                            "Team":          _tm_abbr,
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
        # Savant renamed columns — normalize to internal names used by merges
        df = df.rename(columns={
            "brl_percent":  "barrel_batted_rate",
            "avg_hit_speed": "avg_exit_velocity",
            "ev95percent":  "hard_hit_percent",
        })
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

    # Per-field provenance: 'measured' | 'proxy' | 'league_avg'
    prov = {k: "league_avg" for k in [
        "slg_proxy", "iso_proxy", "k_rate", "bb_rate", "wrc_plus",
        "woba", "barrel_rate", "hard_hit_rate", "exit_velocity_avg",
        "ev50", "sweet_spot_rate", "bat_speed", "blast_rate", "squared_up_rate",
    ]}
    stats["_provenance"] = prov

    row = find_player_row(batting_df, player_name, mlb_id)

    if row is not None:
        # ── SLG / power — prefer xSLG over SLG ───────────────────────────
        xslg = safe_get(row, 'xSLG', 'xslg', default=None)
        slg  = safe_get(row, 'SLG',  'slg',  default=None)
        if xslg and 0.050 < xslg < 1.200:
            stats["slg_proxy"] = xslg
            prov["slg_proxy"] = "measured"
        elif slg and 0.050 < slg < 1.200:
            stats["slg_proxy"] = slg
            prov["slg_proxy"] = "measured"

        # ── ISO ────────────────────────────────────────────────────────────
        iso = safe_get(row, 'ISO', 'iso', default=None)
        if iso and 0 < iso < 0.700:
            stats["iso_proxy"] = iso
            prov["iso_proxy"] = "measured"

        # ── K% ────────────────────────────────────────────────────────────
        k = safe_get(row, 'K%', 'k_percent', default=None)
        if k is not None and k > 0:
            stats["k_rate"] = k if k < 1 else k / 100
            prov["k_rate"] = "measured"

        # ── BB% ───────────────────────────────────────────────────────────
        bb = safe_get(row, 'BB%', 'bb_percent', default=None)
        if bb is not None and bb > 0:
            stats["bb_rate"] = bb if bb < 1 else bb / 100
            prov["bb_rate"] = "measured"

        # ── wRC+ ──────────────────────────────────────────────────────────
        wrc = safe_get(row, 'wRC+', 'wRC', default=None)
        if wrc and wrc > 0:
            stats["wrc_plus"] = float(wrc)
            prov["wrc_plus"] = "measured"

        # ── wRC+ proxy from OBP when FanGraphs unavailable ────────────────
        # wRC+ correlates ~0.87 with OBP. MLB API gives us OBP.
        # wrc_proxy = 100 + (OBP - 0.316) / 0.316 * 100 * 0.87
        if stats["wrc_plus"] == 100.0:  # still at default
            _obp_for_wrc = safe_get(row, 'OBP', 'obp', default=None)
            if _obp_for_wrc and 0.250 < _obp_for_wrc < 0.550:
                _wrc_proxy = 100.0 + (_obp_for_wrc - 0.316) / 0.316 * 87.0
                stats["wrc_plus"] = round(max(50.0, min(220.0, _wrc_proxy)), 1)
                prov["wrc_plus"] = "proxy"

        # ── wOBA ──────────────────────────────────────────────────────────
        woba = safe_get(row, 'xwOBA', 'wOBA', 'xwoba', 'woba', default=None)
        if woba and 0.100 < woba < 0.700:
            stats["woba"] = woba
            prov["woba"] = "measured"

        # ── Barrel% ───────────────────────────────────────────────────────
        barrel = safe_get(row, 'Barrel%', 'Barrel', 'barrel_batted_rate',
                          'barrel_rate', default=None)
        if barrel is not None and barrel > 0:
            stats["barrel_rate"] = barrel if barrel < 1 else barrel / 100
            prov["barrel_rate"] = "measured"

        # ── Hard Hit% ─────────────────────────────────────────────────────
        hard = safe_get(row, 'Hard%', 'HardHit%', 'hard_hit_percent',
                        'hard_hit_rate', default=None)
        if hard is not None and hard > 0:
            stats["hard_hit_rate"] = hard if hard < 1 else hard / 100
            prov["hard_hit_rate"] = "measured"

        # ── Exit Velocity ─────────────────────────────────────────────────
        ev = safe_get(row, 'EV', 'avg_exit_velocity', 'exit_velocity_avg', default=None)
        if ev and ev > 50:
            stats["exit_velocity_avg"] = ev
            prov["exit_velocity_avg"] = "measured"

        # ── Sweet Spot% ───────────────────────────────────────────────────
        # FanGraphs uses several column names across seasons/API versions
        sweet = safe_get(row, 'Sweetspot%', 'SweetSpot%', 'Sweet-Spot%',
                         'LA Sweet-Spot%', 'sweet_spot_percent',
                         'sweet_spot_rate', 'SweetSpot', default=None)
        if sweet is not None and sweet > 0:
            stats["sweet_spot_rate"] = sweet if sweet < 1 else sweet / 100
            prov["sweet_spot_rate"] = "measured"

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
            prov["ev50"] = "measured"

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
                prov["barrel_rate"] = "proxy"

        # Hard hit% proxy: derived from SLG-based formula in MLB API fetch
        hard_p = safe_get(row, 'hard_proxy', default=None)
        if hard_p is not None and stats["hard_hit_rate"] == 0.370:
            stats["hard_hit_rate"] = round(hard_p, 4)
            prov["hard_hit_rate"] = "proxy"

        # xSLG proxy: SLG is a reasonable proxy when xSLG unavailable
        # (xSLG ~ SLG * 1.02 on average; use direct SLG if nothing better)
        slg_raw = safe_get(row, 'SLG', 'slg', default=None)
        if slg_raw and 0.100 < slg_raw < 0.900 and stats["slg_proxy"] == 0.398:
            stats["slg_proxy"] = slg_raw
            prov["slg_proxy"] = "measured"

        # TB/PA proxy: direct total bases rate for power scoring
        tb_pa = safe_get(row, 'tb_per_pa', default=None)
        if tb_pa and tb_pa > 0:
            stats["tb_per_game"] = tb_pa * 4.2  # approx PA/game

        # OBP as wOBA proxy when xwOBA unavailable
        obp = safe_get(row, 'OBP', 'obp', default=None)
        if obp and 0.200 < obp < 0.600 and stats["woba"] == 0.315:
            # wOBA ≈ OBP * 0.82 (rough linear scaling)
            stats["woba"] = round(obp * 0.82, 3)
            prov["woba"] = "proxy"

        # ── Bat tracking (bat speed, blast rate) ──────────────────────────
        bs = safe_get(row, 'bat_speed', 'BatSpeed', default=None)
        if bs and bs > 30:
            stats["bat_speed"] = bs
            prov["bat_speed"] = "measured"
        br = safe_get(row, 'blast_rate', 'BlastRate', default=None)
        if br is not None and br >= 0:
            stats["blast_rate"] = br if br < 1 else br / 100
            prov["blast_rate"] = "measured"
        sq = safe_get(row, 'squared_up_rate', default=None)
        if sq is not None and sq >= 0:
            stats["squared_up_rate"] = sq if sq < 1 else sq / 100
            prov["squared_up_rate"] = "measured"

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

    # Per-field provenance: 'measured' | 'proxy' | 'league_avg'
    prov = {k: "league_avg" for k in [
        "k_rate_allowed", "bb_rate_allowed", "hard_hit_allowed",
        "barrel_allowed", "era", "fip", "whip", "swstr_pct",
    ]}
    stats["_provenance"] = prov

    row = find_player_row(pitching_df, pitcher_name, pitcher_mlb_id)

    if row is not None:
        k = safe_get(row, 'K%', default=None)
        if k is not None and k > 0:
            stats["k_rate_allowed"] = k if k < 1 else k / 100
            prov["k_rate_allowed"] = "measured"

        bb = safe_get(row, 'BB%', default=None)
        if bb is not None and bb > 0:
            stats["bb_rate_allowed"] = bb if bb < 1 else bb / 100
            prov["bb_rate_allowed"] = "measured"

        hard = safe_get(row, 'Hard%', default=None)
        if hard is not None and hard > 0:
            stats["hard_hit_allowed"] = hard if hard < 1 else hard / 100
            prov["hard_hit_allowed"] = "measured"

        barrel = safe_get(row, 'Barrel%', default=None)
        if barrel is not None and barrel > 0:
            stats["barrel_allowed"] = barrel if barrel < 1 else barrel / 100
            prov["barrel_allowed"] = "measured"

        era = safe_get(row, 'ERA', default=None)
        if era and 0 < era < 20:
            stats["era"] = era
            prov["era"] = "measured"

        fip = safe_get(row, 'FIP', 'xFIP', default=None)
        if fip and 0 < fip < 20:
            stats["fip"] = fip
            prov["fip"] = "measured"

        whip = safe_get(row, 'WHIP', default=None)
        if whip and 0 < whip < 5:
            stats["whip"] = whip
            prov["whip"] = "measured"

        # ── Pitch arsenal mix (pct_FF, pct_SL, etc.) ─────────────────────
        for _pt in ("FF", "SI", "SL", "CH", "CU", "FC"):
            _pct = safe_get(row, f"pct_{_pt}", default=None)
            if _pct is not None and _pct >= 0:
                stats[f"pct_{_pt}"] = _pct if _pct < 1 else _pct / 100

        # ── MLB Stats API proxy fallback for Barrel%/HH% allowed ──────────
        # When Savant is blocked, use counting-stat proxies
        barrel_p = safe_get(row, 'barrel_proxy', default=None)
        if barrel_p is not None and stats["barrel_allowed"] == 0.070:
            stats["barrel_allowed"] = round(max(0.010, barrel_p), 4)
            prov["barrel_allowed"] = "proxy"

        hard_p = safe_get(row, 'hard_proxy_pit', default=None)
        if hard_p is not None and stats["hard_hit_allowed"] == 0.360:
            stats["hard_hit_allowed"] = round(hard_p, 3)
            prov["hard_hit_allowed"] = "proxy"

        # ── SwStr% — read real value from pitching_df before falling back to proxy ──
        swstr_real = safe_get(row, 'swstr_pct', 'SwStr%', default=None)
        if swstr_real is not None and swstr_real > 0:
            stats["swstr_pct"] = swstr_real if swstr_real < 1 else swstr_real / 100
            stats["swstr_pct_is_proxy"] = False
            prov["swstr_pct"] = "measured"
        elif stats["k_rate_allowed"] > 0:
            stats["swstr_pct"] = round(stats["k_rate_allowed"] * 0.49, 4)
            stats["swstr_pct_is_proxy"] = True
            prov["swstr_pct"] = "proxy"

        # Label pitcher source based on available columns
        if any(c in row.index for c in ('barrel_proxy','hard_proxy_pit','H_per_9')):
            stats["data_source"] = "mlbapi"
        elif any(c in row.index for c in ('Barrel%','Hard%','barrel_batted_rate')):
            stats["data_source"] = "savant_statcast"
        else:
            stats["data_source"] = "matched"

    else:
        # No row found — still compute SwStr% proxy from default K% league avg
        if stats.get("swstr_pct", 0.0) == 0.0:
            stats["swstr_pct"] = round(stats["k_rate_allowed"] * 0.49, 4)
            stats["swstr_pct_is_proxy"] = True
            prov["swstr_pct"] = "proxy"

    return stats


# ============================================================================
# DATA QUALITY + BETTABLE GATE
# ============================================================================

def compute_data_quality_score(batter_prov: dict, pitcher_prov: dict,
                                lineup_confirmed: bool, sp_known: bool,
                                hand_real: bool) -> int:
    """
    0-100 score: % of key TB scoring inputs that are 'measured'.
    Inputs listed in importance order for TB / O0.5 market.
    """
    checks = [
        batter_prov.get("k_rate",          "league_avg") == "measured",
        batter_prov.get("woba",            "league_avg") == "measured",
        batter_prov.get("slg_proxy",       "league_avg") == "measured",
        batter_prov.get("hard_hit_rate",   "league_avg") == "measured",
        batter_prov.get("barrel_rate",     "league_avg") == "measured",
        batter_prov.get("iso_proxy",       "league_avg") == "measured",
        pitcher_prov.get("k_rate_allowed", "league_avg") == "measured",
        pitcher_prov.get("hard_hit_allowed","league_avg") == "measured",
        lineup_confirmed,
        sp_known,
        hand_real,
    ]
    return round(sum(checks) / len(checks) * 100)


def check_bettable_tb(batter_prov: dict, pitcher_prov: dict,
                       batter_matched: bool, pitcher_matched: bool,
                       lineup_confirmed: bool, sp_known: bool,
                       hand_real: bool) -> tuple:
    """
    Returns (is_bettable: bool, reasons: list[str]) for TB / O0.5 market.
    Required core must ALL be 'measured' — any gap = non-bettable.
    """
    reasons = []
    if not batter_matched:
        reasons.append("player not matched to real stats")
    if not hand_real:
        reasons.append("batter handedness defaulted (not confirmed)")
    if not sp_known:
        reasons.append("opposing SP unknown (TBD)")
    if not pitcher_matched:
        reasons.append("SP has no real stats")
    if not lineup_confirmed:
        reasons.append("lineup not confirmed")
    if batter_prov.get("k_rate",        "league_avg") != "measured":
        reasons.append("K% not measured")
    if batter_prov.get("slg_proxy",     "league_avg") != "measured":
        reasons.append("xSLG/SLG not measured")
    if batter_prov.get("woba",          "league_avg") != "measured":
        reasons.append("wOBA not measured (Savant or real wOBA needed)")
    if batter_prov.get("hard_hit_rate", "league_avg") != "measured":
        reasons.append("hard-hit% not measured (Savant column drift — fix in fetch pipeline)")
    return (len(reasons) == 0, reasons)


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

    # Bat-tracking signals — only use when real Savant data present (not derived)
    ev50       = ev50_raw
    bat_speed  = bat_speed_raw
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
    # V2.1: Rebalanced for O1.5 TB prediction accuracy.
    #
    # Problem: Previous model used barrel% (26%) + xSLG (24%) + HH% (20%) = 70% weight
    # on power metrics. This correctly predicts HR rate but NOT O1.5 TB hit rate.
    # O1.5 TB is cleared by 2 singles or 1 double. Contact hitters like Turang/Chourio
    # who hit .290+ with 15% K were scoring below power hitters who K 25%+.
    #
    # Research-backed O1.5 TB predictors (r = correlation with prop hit rate):
    #   wOBA / OBP:  r≈0.71  — getting on base = getting TBs
    #   K% inverse:  r≈0.68  — strikeout = 0 TB guaranteed, biggest miss source
    #   wRC+:        r≈0.65  — overall offensive context
    #   xSLG:        r≈0.61  — extra base quality
    #   Hard Hit%:   r≈0.55  — sustainable contact quality
    #   Barrel%:     r≈0.42  — predicts HR, not singles/doubles
    #   ISO:         r≈0.40  — raw power, less relevant for 1.5 TB threshold
    #
    # New weight structure: wOBA+K% = primary (48%), xSLG+HH% = secondary (34%), barrel+ISO = tertiary (18%)

    # xSLG: avg=.398, sd≈.080
    xslg_score = max(0, min(100, 50 + (xslg - 0.398) / 0.080 * 25))

    # wRC+: avg=100, sd≈35
    wrc_score = max(0, min(100, 50 + (wrc_plus - 100) / 35.0 * 25))

    # Barrel%: avg=7%, sd≈4%
    barrel_score = max(0, min(100, 50 + (barrel_rate - 0.070) / 0.040 * 25))

    # Hard hit%: avg=37%, sd≈5.5%
    hard_hit_score = max(0, min(100, 50 + (hard_hit - 0.370) / 0.055 * 25))

    # K rate INVERSE: avg=22.8%, sd≈6% — most important single metric for TB props
    # Turang 15% K → score = 50 + (0.228-0.15)/0.06 × 25 = 82.5 (correctly elite)
    # Judge 24% K → score = 50 + (0.228-0.24)/0.06 × 25 = 45 (slight negative)
    k_score = max(0, min(100, 50 + (0.228 - k_rate) / 0.060 * 25))

    # ISO: avg=.165, sd≈.065
    iso_score = max(0, min(100, 50 + (iso - 0.165) / 0.065 * 25))

    # wOBA: avg=.315, sd≈.040 — on-base quality, primary O1.5 predictor
    woba_raw   = statcast.get("woba", 0.0) or 0.0
    try: woba_raw = float(woba_raw)
    except: woba_raw = 0.0
    # If wOBA not available, proxy from xSLG and k_rate
    if woba_raw < 0.200:
        woba_raw = 0.245 + xslg * 0.22 - k_rate * 0.15  # proxy formula
        woba_raw = max(0.240, min(0.420, woba_raw))
    woba_score = max(0, min(100, 50 + (woba_raw - 0.315) / 0.040 * 25))
    details["wOBA"] = f"{woba_raw:.3f}"

    # V1.8: When real Savant bat-tracking data available, add EV50/bat_speed/blast.
    # When unavailable, do NOT fake them — derived proxies are correlated with xSLG.
    has_bat_tracking = (ev50_raw >= 50 and bat_speed_raw >= 30 and blast_raw >= 0.01)

    if has_bat_tracking:
        # With Savant bat-tracking: EV50 and blast rate added
        ev50_score = max(0, min(100, 50 + (ev50_raw - 95.0) / 3.0 * 25))
        bat_speed_score = max(0, min(100, 50 + (bat_speed_raw - 71.0) / 3.0 * 25))
        blast_score = max(0, min(100, 50 + (blast_raw - 0.21) / 0.050 * 25))
        composite = (
            k_score         * 0.18 +   # K% inverse — #1 O1.5 TB predictor
            woba_score      * 0.16 +   # wOBA — on-base quality
            xslg_score      * 0.14 +   # xSLG — extra base quality
            hard_hit_score  * 0.12 +   # HH% — contact quality
            wrc_score       * 0.10 +   # wRC+ — overall offensive value
            ev50_score      * 0.10 +   # EV50 — raw power ceiling
            blast_score     * 0.08 +   # Blast — squared-up contact
            bat_speed_score * 0.06 +   # Bat speed — mechanical ceiling
            barrel_score    * 0.04 +   # Barrel% — HR predictor (tertiary for O1.5)
            iso_score       * 0.02     # ISO — raw power
        )  # sum = 1.00
    else:
        # No bat-tracking — 7 signals, O1.5 optimized weights
        # Key insight: K% and wOBA predict O1.5 hit rate better than barrel%
        composite = (
            k_score        * 0.24 +   # K% inverse — most important: K = 0 TB guaranteed
            woba_score     * 0.20 +   # wOBA — best single on-base quality metric
            xslg_score     * 0.18 +   # xSLG — extra base potential
            hard_hit_score * 0.16 +   # HH% — contact quality / floor
            wrc_score      * 0.12 +   # wRC+ — overall offensive context
            barrel_score   * 0.06 +   # Barrel% — some weight but tertiary
            iso_score      * 0.04     # ISO — raw power signal
        )  # sum = 1.00

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

    # ── Data quality detection ──────────────────────────────────────────────
    # HH% and Barrel% allowed are only populated from Savant/FanGraphs.
    # When blocked (Streamlit Cloud), they default to league avg (0.360/0.070).
    # Using defaults contaminates elite SPs (Sanchez FIP=2.60 scores like avg).
    # FIP-only mode: when statcast contact data is missing, use only K%+FIP+WHIP.
    # This produces a 30-pt gap (Sanchez=25 vs Severino=56) vs 9-pt gap with defaults.
    LEAGUE_AVG_HH     = 0.360
    LEAGUE_AVG_BARREL = 0.070
    HH_IS_DEFAULT     = abs(hard_hit - LEAGUE_AVG_HH)     < 0.003
    BARREL_IS_DEFAULT = abs(barrel   - LEAGUE_AVG_BARREL)  < 0.003

    if HH_IS_DEFAULT and BARREL_IS_DEFAULT:
        # FIP-only mode: K%=50%, FIP=35%, WHIP=15%
        # No HH%/Barrel% contamination when data is unavailable
        sp_score = (
            k_vuln   * 0.50 +
            era_vuln * 0.35 +
            whip_vuln* 0.15
        )
    else:
        # Full mode: all five components with standard weights
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

    mode_tag = "FIP-only" if (HH_IS_DEFAULT and BARREL_IS_DEFAULT) else "full"
    label = f"K%: {k_rate*100:.0f}% | WHIP: {whip:.2f} | FIP: {era_use:.2f} | BP vuln: {bullpen_vuln:.0f} | mode: {mode_tag}"
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
@st.cache_data(ttl=1800)
def fetch_batter_recent_form(player_id: str, n_games: int = 7) -> Dict:
    """
    Pull last N game box scores for a batter using the MLB Stats API.

    Strategy: Use the player's recent game schedule to get game PKs,
    then compute per-game TB from the box score hitting line.
    This bypasses the gameLog endpoint which returns running season totals,
    not individual game stats.

    Fallback: If box score approach fails, use the gameLog endpoint and
    compute TB from component stats (H, 2B, 3B, HR).
    """
    _empty = {"tb_per_game": None, "avg_recent": None, "games": 0,
              "hr_last_7": 0, "h_last_7": 0, "ab_last_7": 0}
    if not player_id or str(player_id) in ("", "0", "nan"):
        return _empty

    import datetime as _dt
    today      = _dt.datetime.now()
    current_year = today.year

    # ── Method 1: gameLog endpoint (most reliable when it returns per-game data) ─
    def _fetch_gamelogs(year: int) -> Dict:
        url = (f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
               f"?stats=gameLog&group=hitting&gameType=R&season={year}&limit=20")
        try:
            r = requests.get(url, timeout=8)
            if r.status_code != 200:
                return {}
            splits = r.json().get("stats", [{}])[0].get("splits", [])
            if not splits:
                return {}

            # The gameLog endpoint CAN return running season totals instead of
            # per-game stats. Detect this by checking if totals increase monotonically.
            # If splits[0]["stat"]["hits"] > splits[1]["stat"]["hits"], it's per-game.
            # If they increase, it's cumulative — we need to diff consecutive rows.
            hits_vals = [int(s.get("stat",{}).get("hits",0) or 0) for s in splits[:5]]
            is_cumulative = len(hits_vals) >= 2 and all(
                hits_vals[i] >= hits_vals[i+1] for i in range(len(hits_vals)-1)
            )
            # Actually: gameLog is most-recent-first, so if it's cumulative,
            # hits would DECREASE as we go back (older games have lower cumulative totals)
            # If per-game, values are random (some games 0H, some 3H)
            # Cumulative detection: if sorted desc == original, it's cumulative
            is_cumulative = (sorted(hits_vals, reverse=True) == hits_vals and
                             max(hits_vals) > 5 and min(hits_vals) == 0)

            recent = splits[-n_games:]  # API returns oldest-first; take the tail for most recent games

            if is_cumulative:
                # Diff consecutive rows to get per-game stats
                # splits[0] = most recent cumulative, splits[1] = day before, etc.
                game_stats = []
                for i in range(len(recent) - 1):
                    cur  = recent[i].get("stat",{})
                    prev = recent[i+1].get("stat",{})
                    game_stats.append({
                        "ab": max(0, int((cur.get("atBats",0) or 0)) - int((prev.get("atBats",0) or 0))),
                        "h":  max(0, int((cur.get("hits",0) or 0)) - int((prev.get("hits",0) or 0))),
                        "d2": max(0, int((cur.get("doubles",0) or 0)) - int((prev.get("doubles",0) or 0))),
                        "d3": max(0, int((cur.get("triples",0) or 0)) - int((prev.get("triples",0) or 0))),
                        "hr": max(0, int((cur.get("homeRuns",0) or 0)) - int((prev.get("homeRuns",0) or 0))),
                    })
            else:
                game_stats = []
                for s in recent:
                    st_ = s.get("stat",{})
                    game_stats.append({
                        "ab": int(st_.get("atBats",0) or 0),
                        "h":  int(st_.get("hits",0) or 0),
                        "d2": int(st_.get("doubles",0) or 0),
                        "d3": int(st_.get("triples",0) or 0),
                        "hr": int(st_.get("homeRuns",0) or 0),
                    })

            total_tb = total_ab = total_h = total_hr = 0
            g = 0
            for gs in game_stats:
                singles = max(0, gs["h"] - gs["d2"] - gs["d3"] - gs["hr"])
                tb = singles + gs["d2"]*2 + gs["d3"]*3 + gs["hr"]*4
                total_tb += tb
                total_ab += gs["ab"]
                total_h  += gs["h"]
                total_hr += gs["hr"]
                g += 1

            if g == 0 or total_ab == 0:
                return {}

            return {
                "tb_per_game": round(total_tb / g, 2),
                "avg_recent":  round(total_h / total_ab, 3),
                "games":       g,
                "hr_last_7":   total_hr,
                "h_last_7":    total_h,
                "ab_last_7":   total_ab,
            }
        except Exception:
            return {}

    result = _fetch_gamelogs(current_year)
    if not result:
        result = _fetch_gamelogs(current_year - 1)
    if result:
        return result

    return _empty
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
    Convert recent form data into a 0-100 streak score.

    Compares recent TB/game to expected TB/game from season SLG.
    Expected TB/game ≈ SLG × 3.7 AB/game.

    Score 50 = on pace with season average
    Score 70+ = hot (recent TB well above expectation OR high hit rate)
    Score 30- = cold (recent TB AND hit rate well below expectation)

    IMPORTANT: Uses LOWER of (season_slg, 0.420) as baseline to prevent
    power hitters from being labeled Cold just because their recent TB/g
    is below their own elite-season expectation. League avg SLG ~0.398.
    A player with 0.480 season SLG but a 1.2 TB/g recent stretch is NOT cold.
    """
    if not recent or recent.get("games", 0) < 3 or recent.get("tb_per_game") is None:
        return 50.0, "Form: no data"

    tb_recent  = recent["tb_per_game"]
    g          = recent["games"]
    h          = recent.get("h_last_7", 0)
    ab         = recent.get("ab_last_7", max(1, g * 3))
    hr         = recent.get("hr_last_7", 0)

    # CAP season_slg at 0.420 (just above league avg 0.398) so elite hitters
    # don't get penalized for not matching their own power ceiling recently.
    # If a player has a 0.500 SLG but hits 1.2 TB/g in 7 games, that is NEUTRAL
    # form, not cold. We compare to a reasonable baseline, not their personal peak.
    baseline_slg = min(season_slg, 0.420)
    season_exp   = max(0.01, baseline_slg * 3.7)   # ~1.47 TB/g at 0.398 SLG

    import math as _math
    log_ratio = _math.log(max(0.05, tb_recent) / season_exp)

    # Scale: log_ratio 0 → 50, +0.69 (2x above baseline) → 75
    raw_score = 50 + log_ratio * 36
    raw_score = max(10.0, min(90.0, raw_score))

    # Hit-rate bonus: if player is getting hits consistently, boost score
    # even if TB/g is dragged down by singles (still good prop performance)
    if ab > 0:
        hit_rate = h / ab
        if hit_rate >= 0.350:
            raw_score += 8   # high contact rate over last 7 games
        elif hit_rate >= 0.300:
            raw_score += 4
        elif hit_rate < 0.150 and g >= 5:
            raw_score -= 6   # genuinely struggling with contact

    # HR bonus: recent power
    if hr >= 3:
        raw_score += 5
    elif hr >= 2:
        raw_score += 3

    raw_score = max(10.0, min(90.0, raw_score))

    # Dampen with small samples
    if g < 5:
        weight    = g / 5
        raw_score = raw_score * weight + 50.0 * (1 - weight)

    raw_score = round(raw_score, 1)

    ratio = tb_recent / season_exp
    # Only label Cold if BOTH TB/g is low AND hit rate is low
    hit_rate = h / max(1, ab)
    is_truly_cold = ratio <= 0.65 and hit_rate < 0.200 and g >= 4

    if ratio >= 1.4 or (hit_rate >= 0.320 and g >= 4):
        label = f"🔥 Hot ({tb_recent:.2f} TB/g last {g}g | {h}/{ab}" + (f" {hr}HR" if hr else "") + ")"
    elif is_truly_cold:
        label = f"❄️ Cold ({tb_recent:.2f} TB/g last {g}g | {h}/{ab})"
    else:
        label = f"{tb_recent:.2f} TB/g last {g}g | {h}/{ab}" + (f" {hr}HR" if hr else "")

    return raw_score, label


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

    if not bvp or ab < 5 or bvp.get("slg") is None:
        if 0 < ab < 5:
            return 50.0, f"BvP: {h}/{ab} ({ab} AB — need 5+)", "no_data"
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
    # Small samples: blend toward 50 with < 45 AB; full weight at 45+ AB
    # 5 AB = ~12% weight, 15 AB = 37% weight, 30 AB = 75% weight, 45 AB = full
    if ab < 45:
        weight = min(1.0, (ab - 5) / 40.0)    # 5 AB = 0 weight, 45 AB = full
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

    sample_note = f" ({ab} AB)" if ab < 15 else ""

    # Owns: elite career dominance — must show consistent contact (AVG >= .400)
    # OR pure power dominance (2+ HR in ≤15 AB, or 3+ HR total, or SLG+.350 above season avg)
    # Tightened AVG gate to .400 since owns now carries 10% weight
    if (avg_pct >= 0.400 and career_slg >= 0.700) or        (hr >= 2 and ab <= 15) or        (hr >= 3) or        (career_slg >= batter_slg + 0.350):
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
    """Thin Streamlit-aware wrapper; pure logic lives in scoring/final.py."""
    _bat_src  = st.session_state.get("_batting_source", "")
    _bat_cols = st.session_state.get("batting_cols", [])
    _has_full = (
        ("Barrel%" in _bat_cols or "barrel_batted_rate" in _bat_cols)
        and ("Hard%" in _bat_cols or "hard_hit_percent" in _bat_cols)
        and ("wRC+" in _bat_cols)
    )
    _is_proxy = (
        "mlbapi" in _bat_src
        or _bat_src in ("mlbapi_only",)
        or "disk_cache_stale" in _bat_src
        or not _has_full
    )
    return _compute_final_score_pure(
        batter_score=batter_score,
        pitcher_vuln_score=pitcher_vuln_score,
        platoon_score=platoon_score,
        lineup_score=lineup_score,
        park_score=park_score,
        weather_score=weather_score,
        vegas_score=vegas_score,
        tto_bonus=tto_bonus,
        pitch_matchup_score=pitch_matchup_score,
        streak_score=streak_score,
        bvp_score=bvp_score,
        bvp_weight_boost=bvp_weight_boost,
        proxy_mode=_is_proxy,
    )


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
    V2.1: When Savant bat-tracking unavailable (ev50=0), derive from ISO+barrel
          so scores differentiate elite power hitters vs avg hitters correctly.

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
    # ── Availability flags — only score what is measured ─────────────────
    # Do NOT derive ev50/bat_speed/blast_rate from other stats.
    # If Savant bat-tracking not available, omit those signals and
    # redistribute their weight to barrel% (strongest HR predictor).
    has_ev50      = ev50 >= 50          # real values are 85-105 mph
    has_bat_speed = bat_speed >= 30     # real values are 65-78 mph
    has_blast     = blast_rate >= 0.01  # real values are 0.10-0.35

    # Weights redistributed to barrel% when bat-tracking signals absent
    _ev50_w    = 0.10 if has_ev50      else 0.0
    _speed_w   = 0.08 if has_bat_speed else 0.0
    _blast_w   = 0.06 if has_blast     else 0.0
    _barrel_w  = 0.35 + (0.10 - _ev50_w) + (0.08 - _speed_w) + (0.06 - _blast_w)

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

    # ── Composite — only score signals that are measured ─────────────────
    # _barrel_w absorbs weight from any missing bat-tracking signals
    base = (
        barrel_score    * _barrel_w +
        park_score      * adjusted_park_weight +
        hh_score        * 0.08 +
        iso_score       * 0.07 +
        vegas_score     * 0.05 +
        matchup_score   * 0.04 +
        (ev50_score      * _ev50_w   if has_ev50      else 0) +
        (bat_speed_score * _speed_w  if has_bat_speed else 0) +
        (blast_score     * _blast_w  if has_blast     else 0)
    )

    wind_contribution = wind_raw * wind_weight if wind_weight > 0 else 0.0
    composite = base + wind_contribution + temp_adj

    return max(0, min(100, round(composite, 1)))

# ============================================================================
# ROSTER FALLBACK — fetch team roster when lineup not yet posted
# ============================================================================
def fetch_team_roster(team_id: int) -> List[Dict]:
    """
    Fetch active roster as lineup fallback.
    Resolves batter handedness via:
      1. MLBAM_BATTER_HAND constant map (instant, covers ~200 regulars)
      2. MLB API /people/{id}?fields=batSide (single call, 5s timeout)
      3. Hardcoded "R" default (last resort)
    NOT cached — cache of empty list is worse than a fresh call.
    """
    for roster_type in ["active", "fullRoster"]:
        try:
            r = requests.get(
                f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster",
                params={"rosterType": roster_type,
                        "hydrate": "person(batSide,pitchHand)"},
                timeout=15
            )
            if r.status_code != 200:
                continue
            data = r.json()
            batters = []
            for p in data.get("roster", []):
                pos_type = p.get("position", {}).get("type", "")
                pos_abbr = p.get("position", {}).get("abbreviation", "")
                if pos_type == "Pitcher" or pos_abbr == "P":
                    continue
                pid = str(p["person"]["id"])
                # 1. Try hydrated batSide from roster response
                bat_hand = p.get("person", {}).get("batSide", {}).get("code", "")
                # 2. Try constant map
                if not bat_hand or bat_hand == "?":
                    bat_hand = MLBAM_BATTER_HAND.get(pid, "")
                # 3. Single API call fallback
                if not bat_hand or bat_hand == "?":
                    try:
                        pr = requests.get(
                            f"https://statsapi.mlb.com/api/v1/people/{pid}",
                            params={"fields": "people,id,fullName,batSide"},
                            timeout=5
                        )
                        bat_hand = pr.json().get("people", [{}])[0].get("batSide", {}).get("code", "R")
                    except Exception:
                        bat_hand = "R"
                batters.append({
                    "player_id": pid,
                    "name": p["person"]["fullName"],
                    "lineup_slot": 5,
                    "batter_hand": bat_hand or "R",
                    "position": pos_abbr,
                    "lineup_confirmed": False,
                })
            if batters:
                return batters[:9]
        except Exception:
            continue
    return []

@st.cache_data(ttl=3600)
def fetch_team_id(abbreviation: str) -> Optional[int]:
    """Get MLB team ID. Hardcoded map first (never fails on API timeout), then API fallback."""
    TEAM_IDS = {
        "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
        "CWS": 145, "CIN": 113, "CLE": 114, "COL": 115, "DET": 116,
        "HOU": 117, "KC":  118, "LAA": 108, "LAD": 119, "MIA": 146,
        "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
        "PHI": 143, "PIT": 134, "SD":  135, "SEA": 136, "SF":  137,
        "STL": 138, "TB":  139, "TEX": 140, "TOR": 141, "WSH": 120,
    }
    team_id = TEAM_IDS.get(str(abbreviation).upper().strip())
    if team_id:
        return team_id
    try:
        r = requests.get("https://statsapi.mlb.com/api/v1/teams",
                         params={"sportId": 1, "season": 2026}, timeout=10)
        for team in r.json().get("teams", []):
            if team.get("abbreviation") == abbreviation:
                return team["id"]
    except Exception as e:
        logging.warning(f"[get_team_id abbr={abbreviation}] {e}")
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
    # Load current season first, fall back to prior season for veteran pitchers
    # This catches: rookies/returns (2026 only) AND veterans with 2025 history
    pitching_df_cur = load_all_pitching_stats(2026)
    pitching_df_pri = load_all_pitching_stats(2025)
    # Merge: 2026 data takes priority, fill gaps with 2025
    if not pitching_df_cur.empty and not pitching_df_pri.empty:
        id_col = "xMLBAMID" if "xMLBAMID" in pitching_df_cur.columns else "mlbam_id"
        if id_col in pitching_df_cur.columns and id_col in pitching_df_pri.columns:
            merged = pitching_df_cur.merge(
                pitching_df_pri[[c for c in pitching_df_pri.columns if c not in pitching_df_cur.columns or c == id_col]],
                on=id_col, how="outer", suffixes=("","_2025")
            )
            pitching_df = merged
        else:
            pitching_df = pitching_df_cur if not pitching_df_cur.empty else pitching_df_pri
    else:
        pitching_df = pitching_df_cur if not pitching_df_cur.empty else pitching_df_pri
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
        log(f"  DEBUG: home_team='{home_team}' away_team='{away_team}' venue='{game.get('venue','?')}' neutral={game.get('neutral_site',False)}", "info")
        log(f"  DEBUG: raw API codes: home='{game.get('_raw_home_team','?')}' away='{game.get('_raw_away_team','?')}'", "info")

        # Park / weather — use neutral-site override when applicable
        park_override = game.get("park_override")
        park_key = park_override if park_override else home_team
        park_info = STADIUM_COORDS.get(park_key, STADIUM_COORDS.get(home_team, (40.7, -74.0, "Unknown Stadium", False)))
        lat, lon, park_name, is_dome = park_info
        if game.get("neutral_site"):
            log(f"  🌎 Neutral site — using {park_key} park factors ({park_info[2]})", "info")
        weather = fetch_weather(lat, lon, game.get("game_time", ""), is_dome)

        # Lineups — try confirmed first, fall back to roster per side independently
        lineups = fetch_lineup(game_pk)
        home_batters = lineups.get("home", [])
        away_batters = lineups.get("away", [])

        # Per-side fallback: if one side has no batters, try roster for that side
        log(f"  DEBUG: fetch_lineup returned home={len(home_batters)} away={len(away_batters)} batters", "info")
        if not home_batters:
            home_id = fetch_team_id(home_team)
            log(f"  DEBUG: {home_team} lineup empty → fetch_team_id returned {home_id}", "info")
            if home_id:
                home_batters = fetch_team_roster(home_id)
                log(f"  DEBUG: {home_team} roster fetch returned {len(home_batters)} players", "info")
                if home_batters:
                    log(f"  {home_team} lineup not posted — using projected roster ({len(home_batters)} players)", "warn")
                else:
                    log(f"  ❌ {home_team} roster fetch also failed — team_id={home_id}", "err")
        if not away_batters:
            away_id = fetch_team_id(away_team)
            log(f"  DEBUG: {away_team} lineup empty → fetch_team_id returned {away_id}", "info")
            if away_id:
                away_batters = fetch_team_roster(away_id)
                log(f"  DEBUG: {away_team} roster fetch returned {len(away_batters)} players", "info")
                if away_batters:
                    log(f"  {away_team} lineup not posted — using projected roster ({len(away_batters)} players)", "warn")
                else:
                    log(f"  ❌ {away_team} roster fetch also failed — team_id={away_id}", "err")

        lineup_confirmed = bool(home_batters or away_batters)
        log(f"  DEBUG: after fallback home={len(home_batters)} away={len(away_batters)} | home_team={home_team} away_team={away_team}", "info")

        if not lineup_confirmed:
            log(f"  Could not load any batters for {away_team}@{home_team} — skipping", "err")
            games_skipped += 1
            continue

        # Pitcher info
        home_pitcher_id = game.get("home_pitcher_id")
        away_pitcher_id = game.get("away_pitcher_id")
        home_pitcher_info = fetch_pitcher_info(home_pitcher_id) if home_pitcher_id else {"name": "TBD", "hand": "R", "id": None}
        away_pitcher_info = fetch_pitcher_info(away_pitcher_id) if away_pitcher_id else {"name": "TBD", "hand": "R", "id": None}

        # Build batter list — use park_key so neutral-site games get correct factors
        all_batters = []
        _park_key = park_override if park_override else home_team
        for b in home_batters[:9]:
            b = dict(b)
            b.update({"team": home_team, "opponent": away_team,
                      "opposing_pitcher": away_pitcher_info,
                      "park_team": _park_key, "lineup_confirmed": lineup_confirmed})
            all_batters.append(b)
        for b in away_batters[:9]:
            b = dict(b)
            b.update({"team": away_team, "opponent": home_team,
                      "opposing_pitcher": home_pitcher_info,
                      "park_team": _park_key, "lineup_confirmed": lineup_confirmed})
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

            # Resolve handedness: cascade from best to least reliable
            # 1. Already set from boxscore batSide.code (confirmed lineups path)
            # 2. MLBAM constant map (fast, covers ~200 regulars, for roster fallback)
            # 3. FanGraphs Bats column (rarely available on Streamlit Cloud)
            # 4. MLB API people endpoint individual call
            # 5. "R" hardcoded default (last resort — will be scored as RHB)
            hand_real = bool(batter_hand and batter_hand not in ("?", ""))  # True if from boxscore
            if not batter_hand or batter_hand in ("?", ""):
                # 2. Constant map — instant, no network call
                bat_from_map = MLBAM_BATTER_HAND.get(str(player_id), "")
                if bat_from_map in ("L", "R", "S"):
                    batter_hand = bat_from_map
                    hand_real = True
            if not batter_hand or batter_hand in ("?", ""):
                # 3. FanGraphs Bats column (type=8 — may be blocked on cloud)
                fg_row = find_player_row(batting_df, name, player_id)
                if fg_row is not None:
                    fg_bats = str(fg_row.get("Bats", "") or "").strip().upper()
                    if fg_bats in ("L", "R", "S", "B"):
                        batter_hand = fg_bats
                        hand_real = True
            if not batter_hand or batter_hand in ("?", ""):
                # 4. MLB API people endpoint — individual call (adds ~50ms per unresolved player)
                try:
                    _pr = requests.get(
                        f"https://statsapi.mlb.com/api/v1/people/{player_id}",
                        params={"fields": "people,id,batSide"},
                        timeout=4
                    )
                    batter_hand = _pr.json().get("people", [{}])[0].get("batSide", {}).get("code", "R")
                    hand_real = True
                except Exception:
                    batter_hand = "R"
                    hand_real = False
            if not batter_hand or batter_hand in ("?", ""):
                batter_hand = "R"  # 5. absolute final fallback
                hand_real = False

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

            sp_tbd = not sp_name or sp_name == "TBD"
            implied = implied_totals.get(team, 0)

            # Recent form and BvP (network calls — stay in orchestrator)
            recent_form = fetch_batter_recent_form(str(player_id), n_games=7)
            bvp_data    = fetch_batter_vs_pitcher(str(player_id), sp_id)

            # Prop-specific implied probability (for edge calc)
            prop_implied = None
            if prop_odds:
                _norm_n = _norm(name)
                _pd = prop_odds.get(_norm_n)
                if not _pd:
                    _last = _norm_n.split()[-1] if _norm_n else ""
                    _pd = next((v for k, v in prop_odds.items() if _last in k), None)
                if _pd:
                    prop_implied = _pd.get("market_implied")

            # Detect proxy mode for tier thresholds and scoring offset
            _bat_src_loop = st.session_state.get("_batting_source", "")
            _proxy_mode = (
                "mlbapi" in _bat_src_loop
                or _bat_src_loop in ("disk_cache_stale", "mlbapi_only")
            )

            result = _score_one_batter_pure(
                name=name,
                player_id=player_id,
                team=team,
                opponent=batter.get("opponent", "?"),
                game_pk=str(game_pk),
                batter_hand=batter_hand,
                hand_real=hand_real,
                sp_hand=sp_hand,
                sp_name=sp_name,
                sp_id=sp_id,
                lineup_slot=lineup_slot,
                lineup_confirmed=batter.get("lineup_confirmed", True),
                batter_position=batter.get("position", ""),
                park_team=park_team,
                batter_stats=batter_statcast,
                pitcher_stats=pitcher_statcast,
                recent_form=recent_form,
                bvp_data=bvp_data,
                weather=weather,
                implied=implied,
                prop_implied=prop_implied,
                team_bullpen_scores=team_bullpen_scores,
                proxy_mode=_proxy_mode,
            )
            # game_total requires both sides' implied — set it here in the orchestrator
            result["game_total"] = round(
                implied_totals.get(home_team, 4.7) + implied_totals.get(away_team, 4.3), 1
            )
            results.append(result)
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
    min_score: float = 70.0,
) -> List[Dict]:
    """Delegates to markets/tb_o15.py — pure logic lives there."""
    return _build_parlays_pure(plays, num_legs=num_legs,
                               max_same_team=max_same_team, min_score=min_score)

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================


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



# ============================================================================
# HOT STREAKS TAB — Top 10 batters by recent form / hit streak
# ============================================================================
# ============================================================================
# MONEYLINE MODEL HELPERS
# ============================================================================

@st.cache_data(ttl=3600)
def fetch_team_run_differential(date_str: str, days: int = 7) -> Dict[str, float]:
    """Last N days team run differential. Returns {team_abbr: avg_run_diff/game}."""
    result = {}
    try:
        from datetime import datetime as _dt, timedelta as _td
        end_dt   = _dt.strptime(date_str, "%Y-%m-%d")
        start_dt = end_dt - _td(days=days)
        url = (f"https://statsapi.mlb.com/api/v1/schedule"
               f"?sportId=1&startDate={start_dt.strftime('%Y-%m-%d')}&endDate={date_str}"
               f"&hydrate=linescore,team")
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return {}
        trf = {}; tra = {}; tg = {}
        for de in r.json().get("dates", []):
            for g in de.get("games", []):
                if g.get("status", {}).get("abstractGameState") != "Final":
                    continue
                ls = g.get("linescore", {})
                hr = int(ls.get("teams", {}).get("home", {}).get("runs", 0) or 0)
                ar = int(ls.get("teams", {}).get("away", {}).get("runs", 0) or 0)
                ha = g["teams"]["home"]["team"].get("abbreviation", "")
                aa = g["teams"]["away"]["team"].get("abbreviation", "")
                ha = TEAM_ABB_MAP.get(g["teams"]["home"]["team"].get("name", ""), ha)
                aa = TEAM_ABB_MAP.get(g["teams"]["away"]["team"].get("name", ""), aa)
                for ab, rf, ra in [(ha, hr, ar), (aa, ar, hr)]:
                    if ab:
                        trf[ab] = trf.get(ab, 0) + rf
                        tra[ab] = tra.get(ab, 0) + ra
                        tg[ab]  = tg.get(ab, 0) + 1
        for ab, g in tg.items():
            if g > 0:
                result[ab] = round((trf[ab] - tra[ab]) / g, 2)
    except Exception:
        pass
    return result


def _american_to_implied(odds: float) -> float:
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def fetch_moneyline_odds(date_str: str) -> Dict:
    """Pull h2h moneyline from Odds API. Returns {'away|home': {...}} vig-removed."""
    result = {}
    try:
        api_key = st.secrets.get("odds_api", {}).get("api_key", "")
        if not api_key:
            return {}
        r = requests.get("https://api.the-odds-api.com/v4/sports/baseball_mlb/odds",
                         params={"apiKey": api_key, "regions": "us",
                                 "markets": "h2h", "oddsFormat": "american"}, timeout=12)
        if r.status_code != 200:
            return {}
        for game in r.json():
            hn = game.get("home_team", ""); an = game.get("away_team", "")
            ho = ao = None
            for bm in game.get("bookmakers", []):
                for mkt in bm.get("markets", []):
                    if mkt.get("key") == "h2h":
                        for oc in mkt.get("outcomes", []):
                            if oc.get("name", "").lower() in hn.lower():
                                ho = float(oc.get("price", -110))
                            elif oc.get("name", "").lower() in an.lower():
                                ao = float(oc.get("price", 100))
                        break
                if ho is not None:
                    break
            if ho is None:
                ho, ao = -110, 100
            hi = _american_to_implied(ho); ai = _american_to_implied(ao)
            tot = hi + ai
            result[f"{an}|{hn}"] = {
                "home_odds": ho, "away_odds": ao,
                "home_implied": round(hi / tot, 4) if tot > 0 else 0.5,
                "away_implied": round(ai / tot, 4) if tot > 0 else 0.5,
            }
    except Exception:
        pass
    return result


def compute_team_offense_score(plays: List[Dict], team: str) -> Tuple[float, int]:
    """Aggregate wRC+ for a team's confirmed lineup. Returns (avg_wrc_plus, n_batters)."""
    tp = [p for p in plays if p.get("team", "") == team]
    vals = [p.get("wrc_plus", 100.0) for p in tp if p.get("wrc_plus", 100.0) > 0]
    return (round(sum(vals) / len(vals), 1), len(vals)) if vals else (100.0, 0)


def compute_win_probability(
    home_sp_stats: Dict, away_sp_stats: Dict,
    home_off_wrc: float, away_off_wrc: float,
    home_bp_vuln: float, away_bp_vuln: float,
    home_run_diff: float, away_run_diff: float,
    home_implied_runs: float, away_implied_runs: float,
) -> Tuple[float, str]:
    """Log5-style win probability. Returns (home_win_prob 0-1, label)."""
    home_off = max(0.5, home_off_wrc / 100.0)
    away_off = max(0.5, away_off_wrc / 100.0)
    hpv = float(home_sp_stats.get("_sp_vuln", 50.0)) * 0.60 + home_bp_vuln * 0.40
    apv = float(away_sp_stats.get("_sp_vuln", 50.0)) * 0.60 + away_bp_vuln * 0.40
    # Pitcher factor: scales how many runs the OPPOSING offense scores.
    # High vulnerability = opponent scores more. Low vulnerability = opponent scores less.
    # Scale: vuln 0 → 0.0 factor, vuln 50 → 1.0 (avg), vuln 100 → 2.0
    # PHI scores against Severino (away pitcher = apv):  high apv = PHI scores more
    # OAK scores against Sanchez  (home pitcher = hpv):  high hpv = OAK scores more
    h_pit_factor = max(0.10, apv / 50.0)   # away pitcher vuln → home offense multiplier
    a_pit_factor = max(0.10, hpv / 50.0)   # home pitcher vuln → away offense multiplier
    hs  = home_off * h_pit_factor * 1.035  # home team scoring (home field +3.5%)
    aws = away_off * a_pit_factor           # away team scoring
    tot = hs + aws
    if tot <= 0:
        return 0.52, "Log5 zero — home-field default"
    raw = hs / tot
    nudge = max(-0.02, min(0.02, (home_run_diff - away_run_diff) * 0.01))
    if home_implied_runs > 0 and away_implied_runs > 0:
        ti = home_implied_runs + away_implied_runs
        vwp = home_implied_runs / ti if ti > 0 else 0.52
        final = raw * 0.70 + vwp * 0.30 + nudge
    else:
        final = raw + nudge
    final = max(0.30, min(0.75, final))
    pqh = "Elite" if hpv < 30 else "Good" if hpv < 45 else "Average" if hpv < 58 else "Weak"
    pqa = "Elite" if apv < 30 else "Good" if apv < 45 else "Average" if apv < 58 else "Weak"
    label = (f"Home pit: {pqh} (v={hpv:.0f}) | Away pit: {pqa} (v={apv:.0f}) | "
             f"H wRC+: {home_off_wrc:.0f} | A wRC+: {away_off_wrc:.0f} | "
             f"7d RD H/A: {home_run_diff:+.1f}/{away_run_diff:+.1f}")
    return round(final, 4), label


# ============================================================================
# MONEYLINE TAB — Professional ML picker with ranked confidence cards
# ============================================================================
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


# ============================================================================
# PRIZEPICKS SCORING CONSTANTS
# ============================================================================
PP_SCORING = {
    "single": 3.0, "double": 5.0, "triple": 8.0, "hr": 10.0,
    "run": 2.0, "rbi": 2.0, "bb": 2.0, "hbp": 2.0, "sb": 5.0,
}
PP_PAYOUTS = {
    "power": {2: 3.0, 3: 5.0, 4: 10.0, 5: 20.0, 6: 25.0},
    "flex":  {2: 3.0, 3: 2.25, 4: 5.0, 5: 10.0, 6: 25.0},
}

def compute_pp_projection(statcast: Dict, pitcher_statcast: Dict,
                          lineup_slot: int, implied_total: float,
                          batter_hand: str, sp_hand: str,
                          park_team: str, weather: Dict) -> Dict:
    """Project PrizePicks fantasy score using PP scoring rules."""
    def f(key, default):
        try: return float(statcast.get(key, default))
        except: return float(default)

    pa_by_slot = {1:4.8,2:4.7,3:4.6,4:4.5,5:4.3,6:4.2,7:4.1,8:3.9,9:3.8}
    est_pa = pa_by_slot.get(lineup_slot, 4.2)
    if implied_total > 0:
        est_pa += (implied_total - 4.5) * 0.08

    k_rate   = f("k_rate", 0.228);   bb_rate = f("bb_rate", 0.082)
    slg      = f("slg_proxy", 0.398); iso    = f("iso_proxy", 0.165)
    woba     = f("woba", 0.315);      barrel = f("barrel_rate", 0.070)
    sb_rate  = 0.05

    contact_rate  = 1 - k_rate
    hit_rate      = max(0.180, woba * 0.85)
    hr_per_pa     = barrel * 0.35
    xbh_per_pa    = (iso * 0.6) * contact_rate
    single_per_pa = max(0, hit_rate - hr_per_pa - xbh_per_pa)

    park_hr = PARK_HR_FACTORS.get(park_team, 1.0)
    park_tb = PARK_TB_FACTORS.get(park_team, 1.0)
    hr_per_pa     *= park_hr
    single_per_pa *= park_tb

    if not weather.get("is_dome"):
        we = weather.get("wind_effect", "neutral")
        if we == "strong_out": hr_per_pa *= 1.25
        elif we == "out":      hr_per_pa *= 1.15
        elif we == "in":       hr_per_pa *= 0.80

    pit_fip   = float(pitcher_statcast.get("fip", 4.10))
    pit_k     = float(pitcher_statcast.get("k_rate_allowed", 0.228))
    pit_adj   = 1.0 + (pit_fip - 4.0) * 0.04
    pit_k_adj = 1.0 - (pit_k - 0.228) * 1.5
    q_adj     = (pit_adj + pit_k_adj) / 2

    proj_singles = max(0, single_per_pa * est_pa * q_adj)
    proj_hr      = hr_per_pa * est_pa * q_adj
    proj_xbh     = max(0, xbh_per_pa * est_pa * q_adj)
    proj_doubles = proj_xbh * 0.65
    proj_triples = proj_xbh * 0.05
    proj_bb      = bb_rate * est_pa
    proj_sb      = sb_rate * est_pa
    rbi_rate     = 0.32 if lineup_slot <= 4 else 0.22
    run_rate     = 0.38 if lineup_slot <= 3 else (0.28 if lineup_slot <= 6 else 0.20)
    if implied_total > 0:
        rbi_rate *= implied_total / 4.5
        run_rate *= implied_total / 4.5
    proj_hits = proj_singles + proj_doubles + proj_triples + proj_hr
    proj_rbi  = proj_hits * rbi_rate + proj_hr
    proj_runs = proj_hits * run_rate + proj_hr

    pp_pts = (proj_singles * PP_SCORING["single"] + proj_doubles * PP_SCORING["double"] +
              proj_triples * PP_SCORING["triple"] + proj_hr * PP_SCORING["hr"] +
              proj_rbi * PP_SCORING["rbi"] + proj_runs * PP_SCORING["run"] +
              proj_bb * PP_SCORING["bb"] + proj_sb * PP_SCORING["sb"])

    variance = pp_pts * 0.45
    return {
        "pp_proj":    round(pp_pts, 1),
        "pp_ceiling": round(pp_pts + variance, 1),
        "pp_floor":   round(max(0, pp_pts - variance * 0.6), 1),
        "est_pa":     round(est_pa, 1),
    }

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


# ============================================================================
# DFS STACK COMMAND CENTER
# ============================================================================

def _norm_name_dfs(name: str) -> str:
    """Normalize player name for salary matching."""
    import re
    return re.sub(r"[^a-z]", "", name.lower())


def compute_game_stack_scores(plays: List[Dict]) -> List[Dict]:
    """
    Rank all games on today's slate by DFS stack value.
    Uses: game O/U + park HR factor + wind bonus + dome penalty.
    Returns list of game dicts sorted best → worst.
    """
    games_seen = {}
    for p in plays:
        gid = p.get("game_id", "")
        if not gid:
            continue
        home = p.get("park", "")  # park = home team abbr
        away = p.get("opponent", "") if p.get("team", "") == home else p.get("team", "")
        if gid not in games_seen:
            games_seen[gid] = {
                "game_id": gid,
                "home_team": home,
                "away_team": away,
                "game_total": p.get("game_total", 9.0),
                "park": home,
                "is_dome": p.get("is_dome", False),
                "wind_speed": p.get("wind_speed", 0),
                "wind_effect": p.get("wind_effect", "neutral"),
                "temperature": p.get("temperature", 70),
                "players": [],
            }
        games_seen[gid]["players"].append(p)

    results = []
    for gid, g in games_seen.items():
        game_total = g["game_total"]
        park = g["park"]
        park_hr = PARK_HR_FACTORS.get(park, 1.0)
        wind_effect = g["wind_effect"]
        is_dome = g["is_dome"]
        temp = g["temperature"]

        # Park bonus: scale 0.87–1.35 → -6 to +12
        park_bonus = round((park_hr - 1.0) * 40, 1)

        # Wind bonus
        wind_bonus = 0
        if not is_dome:
            if wind_effect in ("out_strong", "out"):
                wind_bonus = 6 if wind_effect == "out_strong" else 3
            elif wind_effect in ("in_strong", "in"):
                wind_bonus = -5 if wind_effect == "in_strong" else -2

        # Temp penalty (cold suppresses offense)
        temp_adj = 0
        if not is_dome:
            if temp < 45:
                temp_adj = -4
            elif temp < 55:
                temp_adj = -2
            elif temp > 80:
                temp_adj = 1

        stack_score = round(game_total + park_bonus + wind_bonus + temp_adj, 1)

        # Derive teams properly
        home_team = g["home_team"]
        away_team = g["away_team"]
        # Fix: identify home/away from players
        home_players = [p for p in g["players"] if p.get("park") == p.get("team")]
        away_players = [p for p in g["players"] if p.get("park") != p.get("team")]
        if not home_players:
            home_players = [p for p in g["players"] if p.get("team") == home_team]
        if not away_players:
            away_players = [p for p in g["players"] if p.get("team") != home_team]

        home_implied = home_players[0].get("implied_total", 0) if home_players else 0
        away_implied = away_players[0].get("implied_total", 0) if away_players else 0

        results.append({
            "game_id": gid,
            "home_team": home_team,
            "away_team": away_team,
            "game_total": game_total,
            "park": park,
            "park_hr": park_hr,
            "park_bonus": park_bonus,
            "wind_effect": wind_effect,
            "wind_bonus": wind_bonus,
            "temp": temp,
            "temp_adj": temp_adj,
            "is_dome": is_dome,
            "stack_score": stack_score,
            "home_implied": home_implied,
            "away_implied": away_implied,
            "all_players": g["players"],
        })

    results.sort(key=lambda x: x["stack_score"], reverse=True)
    return results


def compute_team_stack_score(team: str, game: Dict, plays: List[Dict]) -> Dict:
    """
    Score a specific team as a DFS stack target within a game.
    Components: implied total (40%), SP vulnerability (25%),
    HR potential (20%), hot streak count (15%).
    """
    team_players = [p for p in plays if p.get("team") == team
                    and p.get("game_id") == game["game_id"]
                    and p.get("lineup_slot", 10) <= 9]

    if not team_players:
        return {"team": team, "stack_score": 0, "players": [], "components": {}}

    # Sort by lineup slot for proper stack ordering
    team_players.sort(key=lambda x: x.get("lineup_slot", 9))

    # 1. Implied total (0-40 pts)
    implied = team_players[0].get("implied_total", 4.5)
    implied_score = min(40, max(0, (implied - 2.5) / (7.0 - 2.5) * 40))

    # 2. SP vulnerability — use sub_pitcher score of opposing batters (0-25 pts)
    # Higher sub_pitcher = more vulnerable opposing SP
    avg_pit_vuln = sum(p.get("sub_pitcher", 50) for p in team_players) / len(team_players)
    pit_score = min(25, max(0, (avg_pit_vuln - 30) / (80 - 30) * 25))

    # 3. HR potential — avg hr_score of top 4 hitters in lineup slots 1-5 (0-20 pts)
    top4 = sorted(team_players[:6], key=lambda x: x.get("hr_score", 0), reverse=True)[:4]
    avg_hr = sum(p.get("hr_score", 0) for p in top4) / max(1, len(top4))
    hr_score_comp = min(20, max(0, (avg_hr - 30) / (85 - 30) * 20))

    # 4. Hot streak count bonus (0-15 pts)
    streaking = sum(
        1 for p in team_players
        if p.get("sub_streak", 50) >= 65
        and p.get("lineup_slot", 10) <= 7
    )
    streak_bonus = {0: 0, 1: 5, 2: 10}.get(min(streaking, 2), 15)
    streak_score = min(15, streak_bonus)

    total = round(implied_score + pit_score + hr_score_comp + streak_score, 1)

    # Platoon advantage vs opposing SP
    sp_hand = team_players[0].get("sp_hand", "R") if team_players else "R"
    favorable_platoon = sum(
        1 for p in team_players[:6]
        if (p.get("batter_hand") == "L" and sp_hand == "R") or
           (p.get("batter_hand") == "R" and sp_hand == "L")
    )

    return {
        "team": team,
        "stack_score": total,
        "implied": implied,
        "avg_hr_score": round(avg_hr, 1),
        "streaking_count": streaking,
        "sp_name": team_players[0].get("sp_name", "TBD") if team_players else "TBD",
        "sp_hand": sp_hand,
        "favorable_platoon": favorable_platoon,
        "players": team_players,
        "components": {
            "implied_score": round(implied_score, 1),
            "pit_vuln_score": round(pit_score, 1),
            "hr_potential_score": round(hr_score_comp, 1),
            "streak_score": round(streak_score, 1),
        }
    }


def get_ranked_team_stacks(plays: List[Dict], min_players: int = 3) -> List[Dict]:
    """
    Unified team stack ranker — replaces compute_game_stack_scores for lineup building.
    Returns teams ranked by compute_team_stack_score (implied + SP vuln + HR + streaks).
    Game environment (park, wind, O/U) is INCLUDED inside compute_team_stack_score,
    so this is the single correct signal for both display and lineup construction.
    """
    # Build game dict for each game (needed by compute_team_stack_score)
    games_by_id = {}
    for p in plays:
        gid = p.get("game_id","")
        if not gid:
            continue
        if gid not in games_by_id:
            home = p.get("park","")
            away = p.get("opponent","") if p.get("team","") == home else p.get("team","")
            games_by_id[gid] = {
                "game_id": gid,
                "home_team": home,
                "away_team": away,
                "game_total": p.get("game_total",9.0),
                "park": home,
                "is_dome": p.get("is_dome",False),
                "wind_effect": p.get("wind_effect","neutral"),
                "temperature": p.get("temperature",70),
                "home_implied": 0,
                "away_implied": 0,
                "park_hr": PARK_HR_FACTORS.get(home, 1.0),
                "stack_score": 0,  # filled below from team scores
            }
        # Track implied totals per team side
        team = p.get("team","")
        impl = p.get("implied_total",0)
        g    = games_by_id[gid]
        if team == g["home_team"] and impl > g["home_implied"]:
            g["home_implied"] = impl
        elif team == g["away_team"] and impl > g["away_implied"]:
            g["away_implied"] = impl

    # Score every team
    team_scores = []
    seen_teams  = set()
    for gid, game in games_by_id.items():
        for team in [game["home_team"], game["away_team"]]:
            if not team or team in seen_teams:
                continue
            n_players = len([p for p in plays if p.get("team","") == team])
            if n_players < min_players:
                continue
            sd = compute_team_stack_score(team, game, plays)
            sd["game_id"]    = gid
            sd["game_total"] = game["game_total"]
            sd["home_team"]  = game["home_team"]
            sd["away_team"]  = game["away_team"]
            sd["opp_team"]   = game["away_team"] if team == game["home_team"] else game["home_team"]
            sd["park_hr"]    = game["park_hr"]
            sd["wind_effect"]= game["wind_effect"]
            sd["is_dome"]    = game["is_dome"]
            sd["game_label"] = f"{game['away_team']}@{game['home_team']}"
            sd["park"]       = game["park"]
            sd["n_players"]  = n_players
            team_scores.append(sd)
            seen_teams.add(team)

    team_scores.sort(key=lambda x: x["stack_score"], reverse=True)
    return team_scores


def get_sp_targets(plays: List[Dict], salary_data: Dict) -> Dict:
    """
    Build SP target board from plays data.
    Returns {"aces": [...], "values": [...]} each sorted best first.
    """
    pitchers = {}
    for p in plays:
        sp = p.get("sp_name", "TBD")
        if not sp or sp == "TBD":
            continue
        if sp not in pitchers:
            opp_team_players = [x for x in plays if x.get("team") == p.get("team")
                                 and x.get("sp_name") == sp]
            opp_k_rates = [x.get("k_rate", 0.22) for x in opp_team_players if x.get("k_rate", 0) > 0]
            opp_team_k = round(sum(opp_k_rates) / len(opp_k_rates), 3) if opp_k_rates else 0.22

            # SP's own K rate from pitcher label or _pitcher_k_rate
            sp_k = p.get("_pitcher_k_rate", 0)

            # Implied total allowed (the OPPOSING team's implied = what SP gives up)
            opp_implied = p.get("implied_total", 4.5)  # this is batter's team implied

            pitchers[sp] = {
                "name": sp,
                "hand": p.get("sp_hand", "R"),
                "team": p.get("opponent", ""),
                "opp_team": p.get("team", ""),
                "sp_k_rate": sp_k,
                "opp_team_k_rate": opp_team_k,
                "opp_implied": opp_implied,
                "park": p.get("park", ""),
                "game_id": p.get("game_id", ""),
                "is_dome": p.get("is_dome", False),
            }

    sp_list = []
    for sp_name, sp in pitchers.items():
        # Ace score: K rate (40%) + opp team K rate (25%) + low implied (25%) + home (10%)
        k_score = min(40, sp["sp_k_rate"] * 160)  # 0.25 K% → 40pts
        opp_k_score = min(25, sp["opp_team_k_rate"] * 100)  # high opp K% = good for SP
        implied_score = min(25, max(0, (5.5 - sp["opp_implied"]) / (5.5 - 2.5) * 25))
        ace_score = round(k_score + opp_k_score + implied_score, 1)

        # Salary match
        norm = _norm_name_dfs(sp_name)
        sal_match = next(
            (v for k, v in salary_data.items() if _norm_name_dfs(k) == norm), None
        )
        salary = sal_match.get("salary", 0) if sal_match else 0

        # Value score
        value_score = round(ace_score / (salary / 1000), 2) if salary > 0 else 0.0

        sp_list.append({
            **sp,
            "ace_score": ace_score,
            "salary": salary,
            "value_score": value_score,
        })

    aces = sorted(sp_list, key=lambda x: x["ace_score"], reverse=True)[:5]
    values = sorted(
        [s for s in sp_list if s["salary"] > 0],
        key=lambda x: x["value_score"], reverse=True
    )[:5]

    return {"aces": aces, "values": values, "all": sp_list}


def build_dfs_lineup(
    plays: List[Dict],
    salary_data: Dict,
    primary_team: str,
    secondary_team: str,
    sp1_name: str,
    sp2_name: str,
    platform: str = "DK",
    lineup_num: int = 0,
) -> Optional[Dict]:
    """
    Build a single GPP lineup.
    DK: 2SP + 1C + 1B + 2B + 3B + SS + 3OF + UTIL (10 hitters)
    FD: 1SP + 1C + 1B + 2B + 3B + SS + 3OF + UTIL (9 hitters)
    Stack: 5-man primary + 3-man secondary (DK 5-3) or 4-4 (FD).
    """
    if platform == "DK":
        hitter_slots = ["C","1B","2B","3B","SS","OF","OF","OF","UTIL","UTIL"]
        primary_target = 5
        secondary_target = 3
        sal_cap = 50000
        sp_slots = 2
    else:  # FD
        hitter_slots = ["C","1B","2B","3B","SS","OF","OF","OF","UTIL"]
        primary_target = 4
        secondary_target = 4
        sal_cap = 35000
        sp_slots = 1

    def get_sal(name):
        norm = _norm_name_dfs(name)
        match = next((v for k, v in salary_data.items() if _norm_name_dfs(k) == norm), None)
        return match if match else None

    def can_fill(player, slot):
        pos = player.get("batter_position", "").upper()
        if slot == "UTIL":
            return pos in ("C","1B","2B","3B","SS","OF","DH","")
        if slot == "OF":
            return pos in ("OF","LF","CF","RF","")
        return pos == slot or pos == "" or slot == "UTIL"

    # Get salary-matched players for primary and secondary stacks
    primary_pool = []
    for p in plays:
        if p.get("team") != primary_team:
            continue
        sal_info = get_sal(p["name"])
        if not sal_info:
            continue
        primary_pool.append({**p, "salary": sal_info["salary"], "position": sal_info.get("position","OF")})

    secondary_pool = []
    for p in plays:
        if p.get("team") != secondary_team:
            continue
        sal_info = get_sal(p["name"])
        if not sal_info:
            continue
        secondary_pool.append({**p, "salary": sal_info["salary"], "position": sal_info.get("position","OF")})

    if len(primary_pool) < primary_target or len(secondary_pool) < secondary_target:
        return None

    # Sort by score (GPP: use ceiling = score + hr_score blend)
    def gpp_sort(p):
        base = p.get("score", 0)
        hr_bonus = p.get("hr_score", 0) * 0.3
        # Differentiate lineup 2/3 by deprioritizing chalk
        chalk_penalty = 5 if lineup_num > 0 and p.get("lineup_slot", 5) <= 2 else 0
        return base + hr_bonus - chalk_penalty

    primary_pool.sort(key=gpp_sort, reverse=True)
    secondary_pool.sort(key=gpp_sort, reverse=True)

    # Pick stack players
    primary_stack = primary_pool[:primary_target + 2][:primary_target]
    secondary_stack = secondary_pool[:secondary_target + 2][:secondary_target]

    hitters = primary_stack + secondary_stack
    total_sal = sum(p["salary"] for p in hitters)

    # SP salaries
    sp_sal = 0
    sps_used = []
    for sp_name in ([sp1_name, sp2_name] if sp_slots == 2 else [sp1_name]):
        sal_info = get_sal(sp_name)
        if sal_info:
            sp_sal += sal_info["salary"]
            sps_used.append({"name": sp_name, "salary": sal_info["salary"]})

    remaining = sal_cap - total_sal - sp_sal
    if remaining < 0:
        return None  # Over cap

    return {
        "platform": platform,
        "primary_team": primary_team,
        "secondary_team": secondary_team,
        "hitters": hitters,
        "sps": sps_used,
        "total_salary": total_sal + sp_sal,
        "salary_remaining": remaining,
        "total_proj": round(sum(p.get("score", 0) for p in hitters), 1),
        "stack_label": f"{primary_target}-{secondary_target}",
    }



# ============================================================================
# FD COMMAND CENTER + LINEUP BUILDER — FanDuel Only
# Shared state key: "fd_slate_data" stores {salary_data, slate_games, fd_plays}
# ============================================================================

def _parse_fd_csv(file_bytes: bytes) -> Dict:
    """
    Parse a FanDuel export CSV.
    Returns dict with keys: salary_data, sp_salary_data, slate_games, raw_df
    salary_data: {nickname: {salary, position, team, game, opponent, fppg}}
    sp_salary_data: {nickname: {salary, position, team, game, opponent, fppg}}
    slate_games: sorted list of "AWAY@HOME" strings in the slate
    raw_df: full DataFrame for downstream use
    """
    import io as _io
    df = pd.read_csv(_io.StringIO(file_bytes.decode("utf-8")))

    salary_data    = {}
    sp_salary_data = {}
    slate_games    = set()

    # Column detection — FD uses: Nickname, Position, Salary, Team, Opponent, Game, FPPG
    name_col = next((c for c in df.columns if c.lower() in ("nickname","name","player name")), None)
    pos_col  = next((c for c in df.columns if c.lower() == "position"), None)
    sal_col  = next((c for c in df.columns if c.lower() == "salary"), None)
    team_col = next((c for c in df.columns if c.lower() == "team"), None)
    opp_col  = next((c for c in df.columns if c.lower() == "opponent"), None)
    game_col = next((c for c in df.columns if c.lower() == "game"), None)
    fppg_col = next((c for c in df.columns if c.lower() == "fppg"), None)
    bat_col  = next((c for c in df.columns if "batting" in c.lower() or c.lower() == "batting order"), None)
    inj_col  = next((c for c in df.columns if "injury" in c.lower() and "indicator" in c.lower()), None)
    rp_col   = next((c for c in df.columns if "roster" in c.lower()), None)

    if not (name_col and sal_col):
        raise ValueError(f"Could not find Name/Salary columns. Found: {list(df.columns)}")

    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        if not name or name == "nan":
            continue
        try:
            sal = int(str(row[sal_col]).replace("$","").replace(",","").strip())
        except Exception:
            continue

        pos_raw = str(row[pos_col]).strip() if pos_col else "OF"
        team    = str(row[team_col]).strip() if team_col else ""
        opp     = str(row[opp_col]).strip() if opp_col else ""
        game    = str(row[game_col]).strip() if game_col else ""
        fppg    = float(row[fppg_col]) if fppg_col and pd.notna(row[fppg_col]) else 0.0
        bat_ord = str(row[bat_col]).strip() if bat_col else ""
        inj_str_raw = str(row[inj_col]).strip() if inj_col and pd.notna(row[inj_col]) else ""
        injured = inj_str_raw == "DTD"  # Only DTD = active warning; IL/O = excluded below
        roster_pos = str(row[rp_col]).strip() if rp_col else pos_raw

        # FD primary position for lineup construction
        # FD uses first listed position as primary for slot filling
        primary_pos = pos_raw.split("/")[0].strip()

        entry = {
            "salary":      sal,
            "position":    primary_pos,
            "roster_pos":  roster_pos,   # full eligibility string e.g. "2B/SS/UTIL"
            "pos_raw":     pos_raw,
            "team":        team,
            "opponent":    opp,
            "game":        game,
            "fppg":        fppg,
            "bat_order":   bat_ord,
            "injured":     injured,
        }

        if game:
            slate_games.add(game)

        # Injury classification:
        # IL/O-Postponed = excluded from playable pool (game not happening or player out)
        # DTD = playable but flagged with warning
        # NA = not applicable (treat as healthy)
        inj_indicator = entry.get("injured", "")  # we'll fix below
        inj_str = str(row.get(inj_col, "") if inj_col else "").strip()
        inj_det = str(row.get(next((c for c in df.columns if "detail" in c.lower()), ""), "") if inj_col else "").strip()

        is_postponed = inj_det.lower() == "postponed" or inj_str == "O"
        is_out       = inj_str in ("IL",) and not is_postponed
        is_dtd       = inj_str == "DTD"

        entry["injured"]     = is_dtd        # ⚠️ only shown for DTD
        entry["is_postponed"]= is_postponed   # game not happening — exclude
        entry["is_out"]      = is_out         # on IL — keep in CSV, flag

        is_pitcher = primary_pos == "P" or pos_raw == "P"
        # Exclude postponed game players from playable pool entirely
        if is_postponed:
            pass  # skip — game not happening
        elif is_pitcher:
            sp_salary_data[name] = entry
        else:
            salary_data[name] = entry

    return {
        "salary_data":    salary_data,
        "sp_salary_data": sp_salary_data,
        "slate_games":    sorted(slate_games),
        "raw_df":         df,
        "total_players":  len(salary_data) + len(sp_salary_data),
    }


def _fd_name_match(model_name: str, salary_data: Dict) -> Optional[Dict]:
    """
    Match a model player name to a salary entry.
    Priority: exact → normalized exact → last name + first initial.
    Returns the salary entry dict or None.
    """
    import unicodedata as _uda
    def norm(s):
        s = str(s).lower().strip()
        s = ''.join(c for c in _uda.normalize('NFD', s) if _uda.category(c) != 'Mn')
        return s.replace('.','').replace("'",'').replace('-',' ').replace('jr','').replace('sr','').replace('  ',' ').strip()

    model_norm = norm(model_name)
    model_parts = model_norm.split()
    model_last  = model_parts[-1] if model_parts else ""
    model_first = model_parts[0][:3] if len(model_parts) > 1 else ""

    best_match = None
    for csv_name, entry in salary_data.items():
        csv_norm = norm(csv_name)
        if csv_norm == model_norm:
            return entry   # exact match — done
        if not best_match:
            csv_parts = csv_norm.split()
            csv_last  = csv_parts[-1] if csv_parts else ""
            csv_first = csv_parts[0][:3] if len(csv_parts) > 1 else ""
            if csv_last == model_last and csv_first == model_first:
                best_match = entry

    return best_match


def _slate_teams(slate_games: List[str]) -> set:
    """Extract all team abbreviations from a list of 'AWAY@HOME' game strings."""
    teams = set()
    for g in slate_games:
        parts = g.replace(" ", "").split("@")
        if len(parts) == 2:
            teams.add(parts[0].strip())
            teams.add(parts[1].strip())
    return teams


def _build_fd_plays_with_salaries(plays: List[Dict], salary_data: Dict,
                                   sp_salary_data: Dict, slate_games: List[str]) -> List[Dict]:
    """
    Merge model plays with FD salary data.
    Filters plays to only slate teams.
    Adds: fd_salary, fd_position, fd_value, fd_game, fd_fppg, fd_bat_order, fd_injured.
    Returns list sorted by fd_proj descending.
    """
    slate_team_set = _slate_teams(slate_games) if slate_games else None

    fd_plays = []
    for p in plays:
        team = p.get("team", "")

        # Filter to slate teams only when slate is loaded
        if slate_team_set and team not in slate_team_set:
            continue

        # Build minimal stat dicts for projection
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
        pit_mock = {"k_rate_allowed": 0.220, "hard_hit_allowed": 0.370, "fip": 4.10}
        try:
            pl = p.get("pitcher_label", "")
            if "K%:" in pl:
                pit_mock["k_rate_allowed"] = float(pl.split("K%:")[1].split("%")[0].strip()) / 100
            if "FIP:" in pl:
                pit_mock["fip"] = float(pl.split("FIP:")[1].strip().split()[0])
        except Exception:
            pass

        fd_p = compute_fd_projection(
            statcast=bat_mock,
            pitcher_statcast=pit_mock,
            lineup_slot=p.get("lineup_slot", 5),
            implied_total=p.get("implied_total", 0),
            batter_hand=p.get("batter_hand", "R"),
            sp_hand=p.get("sp_hand", "R"),
            park_team=p.get("park", team),
            weather=p.get("weather", {}),
        )

        # Match salary
        sal_entry = _fd_name_match(p["name"], salary_data)
        fd_salary  = sal_entry["salary"]    if sal_entry else 0
        fd_pos     = sal_entry["position"]  if sal_entry else (p.get("batter_position","") or "OF")
        fd_game    = sal_entry["game"]      if sal_entry else ""
        fd_fppg    = sal_entry["fppg"]      if sal_entry else 0.0
        fd_bat_ord = sal_entry["bat_order"] if sal_entry else ""
        fd_injured = sal_entry["injured"]   if sal_entry else False
        fd_roster  = sal_entry["roster_pos"] if sal_entry else fd_pos

        value = round(fd_p["fd_proj"] / (fd_salary / 1000), 2) if fd_salary > 0 else 0.0
        ownership = compute_ownership_projection(
            fd_proj=fd_p["fd_proj"], salary=fd_salary,
            implied_total=p.get("implied_total", 0),
            lineup_slot=p.get("lineup_slot", 5),
            barrel_rate=p.get("barrel_rate", 0.07),
            name=p["name"],
        )

        fd_plays.append({
            **p, **fd_p,
            "fd_salary":    fd_salary,
            "fd_position":  fd_pos,
            "fd_roster_pos": fd_roster,
            "fd_value":     value,
            "fd_game":      fd_game,
            "fd_fppg":      fd_fppg,
            "fd_bat_order": fd_bat_ord,
            "fd_injured":   fd_injured,
            "ownership":    ownership,
            "salary_matched": sal_entry is not None,
        })

    fd_plays.sort(key=lambda x: x["fd_proj"], reverse=True)
    return fd_plays


def _position_eligible(roster_pos: str, slot: str) -> bool:
    """Check if a player is eligible for a FD roster slot."""
    eligible = [p.strip() for p in roster_pos.upper().replace(" ","").split("/")]
    return slot.upper() in eligible


def _assign_fd_slots(hitters: List[Dict]) -> Optional[List[Dict]]:
    """
    Assign 8 hitters to FD slots: C/1B, 2B, 3B, SS, OF, OF, OF, UTIL.
    Uses roster_pos eligibility string from FD CSV.
    Returns slot-assigned list or None if cannot fill all slots.
    """
    HITTER_SLOTS = ["C/1B", "2B", "3B", "SS", "OF", "OF", "OF", "UTIL"]
    slot_assignments = [None] * 8
    remaining = list(hitters)

    def is_eligible(player, slot):
        rp = player.get("fd_roster_pos", player.get("fd_position", "OF")).upper()
        eligible = [s.strip() for s in rp.split("/")]
        if slot == "UTIL":
            return True   # UTIL accepts anyone
        if slot == "C/1B":
            return "C" in eligible or "1B" in eligible
        return slot in eligible

    # Greedy: fill most restrictive slots first (C/1B, SS are hardest to fill)
    slot_priority = [("SS", 3), ("C/1B", 0), ("2B", 1), ("3B", 2),
                     ("OF", 4), ("OF", 5), ("OF", 6), ("UTIL", 7)]

    for slot_name, slot_idx in slot_priority:
        # Among remaining players eligible for this slot, pick highest fd_proj
        candidates = [p for p in remaining if is_eligible(p, slot_name)]
        if not candidates:
            # No eligible player — try UTIL fallback
            if remaining:
                candidates = remaining
            else:
                return None
        best = max(candidates, key=lambda x: x.get("fd_proj", 0))
        slot_assignments[slot_idx] = {**best, "slot": slot_name}
        remaining.remove(best)

    if any(s is None for s in slot_assignments):
        return None
    return slot_assignments


def _singleton_score(p: Dict) -> float:
    """
    Score a player as a singleton (non-stack) based on HR upside + hot streak.
    50% HR score, 35% streak score, 15% FD projection (sanity check).
    """
    hr_sc     = p.get("hr_score", 0)
    streak_sc = p.get("sub_streak", p.get("streak_score", 50))
    fd_proj   = p.get("fd_proj", 0)
    # Normalize: hr_score and streak_score are already 0-100
    # fd_proj normalize: 20pts = 100, 10pts = 50, etc.
    fd_norm   = min(100, fd_proj * 5)
    return hr_sc * 0.50 + streak_sc * 0.35 + fd_norm * 0.15


def _build_fd_gpp_lineup(fd_plays: List[Dict], primary_team: str,
                          secondary_team: str, sp_name: str,
                          sp_salary_data: Dict, lineup_num: int = 0,
                          ace_sp_name: str = "",
                          locked_players: List[str] = None) -> Optional[Dict]:
    """
    Build a single FanDuel GPP lineup.
    FD roster: P, C/1B, 2B, 3B, SS, OF, OF, OF, UTIL (9 spots)
    Salary cap: $35,000

    SP Strategy (value-first):
      1. Start with value SP (lower salary, solid score-per-dollar)
      2. Build 4-4 stack within remaining budget
      3. Attempt to upgrade to ace SP if salary allows
      4. Use remaining salary: first try to upgrade stack players,
         then if enough left, add the best singleton (HR+streak+proj blend)
         from a third game for a 4-3-1 structure

    Differentiation per lineup_num:
      0: Best 4+4, value SP, try ace upgrade, try singleton
      1: Differentiated 4+4 (rotate 1 player each stack), ace SP if affordable
      2: Max diff (skip top player each stack), force singleton from HR/streak list
    """
    SAL_CAP         = 35000
    SAL_MIN         = 2000
    SAL_SOFT_TARGET = 34200   # target within $800 of cap — leaving $1800+ is a waste
    MIN_SINGLETON_SAL = 2000  # singleton must have at least this salary

    # ── SP SETUP ─────────────────────────────────────────────────────────
    val_entry = _fd_name_match(sp_name, sp_salary_data) if sp_name else None
    ace_entry = _fd_name_match(ace_sp_name, sp_salary_data) if ace_sp_name else None

    if not val_entry:
        return None

    val_sal   = val_entry["salary"]
    ace_sal   = ace_entry["salary"] if ace_entry else 0
    val_game  = val_entry.get("game", "")

    # Start with value SP
    active_sp_name = sp_name
    active_sp_sal  = val_sal
    active_sp_entry = val_entry

    hitter_budget = SAL_CAP - active_sp_sal
    if hitter_budget < SAL_MIN * 8:
        return None

    # ── ELIGIBILITY ───────────────────────────────────────────────────────
    def eligible(p):
        return (p.get("fd_salary", 0) > 0
                and not p.get("fd_injured", False)
                and not p.get("is_postponed", False))

    primary_pool   = [p for p in fd_plays if p.get("team") == primary_team and eligible(p)]
    secondary_pool = [p for p in fd_plays if p.get("team") == secondary_team and eligible(p)]

    if len(primary_pool) < 4 or len(secondary_pool) < 4:
        return None

    # ── GPP SCORING ───────────────────────────────────────────────────────
    def gpp_value(p, lnum):
        base = (p.get("fd_proj",0) * 0.60
                + p.get("fd_ceiling",0) * 0.25
                + p.get("hr_score",0) * 0.15)
        if lnum >= 1:
            own = p.get("ownership", 20)
            if own > 35:  base *= 0.82
            elif own > 25: base *= 0.92
        return base

    primary_pool.sort(key=lambda p: gpp_value(p, lineup_num), reverse=True)
    secondary_pool.sort(key=lambda p: gpp_value(p, lineup_num), reverse=True)

    prim_cands = primary_pool[:6]
    sec_cands  = secondary_pool[:6]

    # ── STACK SELECTION WITH DIFFERENTIATION ─────────────────────────────
    def pick_stack(pool, n, lnum):
        if lnum == 0:
            return list(pool[:n])
        elif lnum == 1:
            base = list(pool[:n])
            if len(pool) > n:
                lowest_idx = min(range(len(base)), key=lambda i: base[i].get("fd_proj",0))
                base[lowest_idx] = pool[n]
            return base
        else:
            # Lineup 3: skip top player for max differentiation
            offset = pool[1:] if len(pool) > n else pool
            return list(offset[:n])

    # Lineup 2 uses 3-man secondary to make room for singleton
    use_singleton = (lineup_num >= 1)
    secondary_n   = 3 if use_singleton else 4

    primary_stack   = pick_stack(prim_cands, 4, lineup_num)
    secondary_stack = pick_stack(sec_cands, secondary_n, lineup_num)

    # ── SALARY ACCOUNTING ────────────────────────────────────────────────
    def total_sal(hitters, sp_sal):
        return sum(p["fd_salary"] for p in hitters) + sp_sal

    hitters     = primary_stack + secondary_stack
    current_sal = total_sal(hitters, active_sp_sal)

    def sal_score(p):
        """Composite player score for swap decisions."""
        return (p.get("fd_proj",0) * 0.55 +
                p.get("fd_ceiling",0) * 0.30 +
                (p.get("fd_salary",0) / 1000.0) * 0.15)

    # ── Over cap: swap worst player for cheaper alternative ──────────────
    for _ in range(15):
        if current_sal <= SAL_CAP:
            break
        overflow = current_sal - SAL_CAP
        worst = min(hitters, key=sal_score)
        team  = worst.get("team","")
        pool  = primary_pool if team == primary_team else secondary_pool
        subs  = sorted(
            [p for p in pool
             if p not in hitters and p["fd_salary"] <= worst["fd_salary"] - overflow],
            key=sal_score, reverse=True
        )
        if not subs:
            # Try cross-team
            subs = sorted(
                [p for p in fd_plays
                 if eligible(p) and p not in hitters
                 and p["fd_salary"] <= worst["fd_salary"] - overflow
                 and p.get("team","") in {primary_team, secondary_team}],
                key=sal_score, reverse=True
            )
        if subs:
            hitters     = [subs[0] if p is worst else p for p in hitters]
            current_sal = total_sal(hitters, active_sp_sal)
        else:
            break

    if current_sal > SAL_CAP:
        return None

    # ── ACE SP UPGRADE ATTEMPT ────────────────────────────────────────────
    # Only attempt if ace is specified, costs more than value SP, and different pitcher
    upgraded_to_ace = False
    if ace_entry and ace_sal > active_sp_sal and ace_sp_name != sp_name:
        sal_after_ace = total_sal(hitters, ace_sal)
        if sal_after_ace <= SAL_CAP:
            # Ace fits — upgrade
            active_sp_name  = ace_sp_name
            active_sp_sal   = ace_sal
            active_sp_entry = ace_entry
            current_sal     = sal_after_ace
            upgraded_to_ace = True
        # else: keep value SP, use that budget for hitter upgrades

    # ── SINGLETON SELECTION (4-3-1 structure) ────────────────────────────
    # Pick the best non-stack player by singleton_score (HR+streak+proj blend)
    # Must come from a game NOT involving primary team, secondary team, or SP
    singleton       = None
    singleton_added = False

    if use_singleton:
        stack_teams  = {primary_team, secondary_team}
        sp_game_str  = active_sp_entry.get("game","")
        stack_games  = set()
        for p in hitters:
            g = p.get("fd_game","")
            if g: stack_games.add(g)

        # On small slates, "third game" restriction is too tight — relax to just
        # "not from primary or secondary team" so singletons are always available
        singleton_pool = [
            p for p in fd_plays
            if eligible(p)
            and p.get("team","") not in stack_teams
            and p not in hitters
        ]
        # Prefer third-game players when available (deeper slate differentiation)
        third_game_sings = [p for p in singleton_pool
                            if p.get("fd_game","") not in stack_games]
        if third_game_sings:
            singleton_pool = third_game_sings + [p for p in singleton_pool if p not in third_game_sings]

        if singleton_pool:
            singleton_pool.sort(key=_singleton_score, reverse=True)
            # Try to fit best singleton within budget
            budget_left = SAL_CAP - current_sal
            for cand in singleton_pool[:10]:
                if cand["fd_salary"] <= budget_left and cand["fd_salary"] >= MIN_SINGLETON_SAL:
                    singleton = cand
                    hitters.append(cand)
                    current_sal += cand["fd_salary"]
                    singleton_added = True
                    break

    # ── HITTER SALARY OPTIMIZER ─────────────────────────────────────────
    # Pass 1: Swap-upgrade — for each player, check if any better player
    # fits in their "slot budget" (budget remaining + that player's salary).
    # This is the Ohtani fix: swap a cheap player OUT to afford Ohtani.
    for _pass in range(4):
        improved = False
        for i, cur in enumerate(list(hitters)):
            slot_budget = SAL_CAP - current_sal + cur["fd_salary"]
            team = cur.get("team","")
            pool = primary_pool if team == primary_team else secondary_pool
            cands = sorted(
                [p for p in (pool + [x for x in fd_plays if eligible(x) and x.get("team","") not in {primary_team, secondary_team}])
                 if p not in hitters
                 and p["fd_salary"] <= slot_budget
                 and sal_score(p) > sal_score(cur) * 1.02],
                key=sal_score, reverse=True
            )
            if cands:
                hitters[i]  = cands[0]
                current_sal = total_sal(hitters, active_sp_sal)
                improved    = True
                break
        if not improved:
            break

    # Pass 2: Fill remaining salary — replace cheapest player with best
    # option that uses more salary (even if slightly lower proj is OK)
    for _ in range(12):
        if current_sal >= SAL_SOFT_TARGET:
            break
        budget_left = SAL_CAP - current_sal
        if budget_left < 200:
            break
        cheapest = min(hitters, key=lambda x: x.get("fd_salary",0))
        upgrades = sorted(
            [p for p in fd_plays
             if eligible(p) and p not in hitters
             and cheapest["fd_salary"] + 100 <= p.get("fd_salary",0) <= cheapest["fd_salary"] + budget_left
             and p.get("fd_proj",0) >= cheapest.get("fd_proj",0) * 0.78],
            key=lambda x: (x.get("fd_salary",0), x.get("fd_proj",0)),
            reverse=True
        )
        if upgrades:
            hitters     = [upgrades[0] if p is cheapest else p for p in hitters]
            current_sal = total_sal(hitters, active_sp_sal)
        else:
            break

    # ── SLOT ASSIGNMENT ──────────────────────────────────────────────────
    assigned = _assign_fd_slots(hitters)
    if assigned is None:
        hitter_slots = ["C/1B","2B","3B","SS","OF","OF","OF","UTIL"]
        assigned = [{**h, "slot": hitter_slots[min(i, len(hitter_slots)-1)]}
                    for i, h in enumerate(hitters[:8])]

    sp_slot = {
        "name": active_sp_name, "salary": active_sp_sal, "slot": "P",
        "team": active_sp_entry.get("team",""), "fd_proj": 0,
        "fd_ceiling": 0, "fd_floor": 0, "ownership": 0,
        "streak_label": "", "fd_position": "P",
        "sp_upgraded": upgraded_to_ace,
    }

    all_players = [sp_slot] + assigned
    final_sal   = sum(p.get("fd_salary", p.get("salary",0)) for p in all_players)

    # Tag singleton for display
    if singleton_added and singleton:
        for p in all_players:
            if p.get("name") == singleton.get("name"):
                p["is_singleton"] = True

    # Determine structure label
    if singleton_added:
        structure = f"4-{secondary_n}-1"
    else:
        structure = f"4-{secondary_n}"

    return {
        "players":          all_players,
        "sp_name":          active_sp_name,
        "sp_salary":        active_sp_sal,
        "sp_upgraded":      upgraded_to_ace,
        "primary_team":     primary_team,
        "secondary_team":   secondary_team,
        "singleton":        singleton.get("name","") if singleton else "",
        "singleton_team":   singleton.get("team","") if singleton else "",
        "total_salary":     final_sal,
        "salary_remaining": SAL_CAP - final_sal,
        "total_proj":       sum(p.get("fd_proj",0) for p in assigned),
        "total_ceiling":    sum(p.get("fd_ceiling",0) for p in assigned),
        "structure":        structure,
        "lineup_num":       lineup_num + 1,
    }


def display_fd_command_center(plays: List[Dict]):
    """
    Tab 9 — FanDuel DFS Command Center
    Single salary CSV upload feeds:
      - Slate game filter (only show teams in the FD contest)
      - Game Stack Ranker (filtered to slate)
      - Team Stack Viewer
      - SP Target Board (salaries from same CSV)
      - FD Projections table
      - Top Game Stacks
      - Value Plays
    """
    st.header("🎯 DFS Command Center")

    # Site toggle — all display logic below uses this to switch data source
    site = st.radio("Platform", ["FanDuel", "DraftKings"], horizontal=True, key="cmd_site_toggle")

    if site == "DraftKings":
        st.caption(f"DK scoring: {DK_HITTER_SCORING}")
        dk_slate = st.session_state.get("dk_slate_data")

        if not plays:
            st.info("Run the model first to generate player data.")
            return

        # ── DK CSV Upload ─────────────────────────────────────────────────
        st.subheader("📥 Load DraftKings Slate")
        col_up, col_info = st.columns([2, 3])
        with col_up:
            dk_csv_cmd = st.file_uploader(
                "Upload DraftKings CSV (from contest lobby)",
                type=["csv"], key="dk_cmd_csv_upload",
            )
        with col_info:
            st.info("DraftKings → MLB → Contest → Export to CSV → upload here.")

        if dk_csv_cmd:
            parsed = _parse_dk_csv(dk_csv_cmd)
            if "error" not in parsed:
                st.session_state.dk_slate_data = parsed
                st.success(f"✅ {parsed['n_hitters']} hitters · {parsed['n_pitchers']} pitchers · {len(parsed['slate_games'])} games")
            dk_slate = st.session_state.get("dk_slate_data")

        if not dk_slate:
            st.info("Upload DraftKings CSV above to see DK-specific stack rankings, SP board, and projections.")
            return

        dk_salary   = dk_slate["salary_data"]
        dk_sp_sal   = dk_slate["sp_salary_data"]
        dk_plays    = _build_dk_plays_with_salaries(plays, dk_salary, dk_sp_sal)

        # ── DK Game Stack Ranker ──────────────────────────────────────────
        st.markdown("---")
        st.subheader("🏟️ Game Stack Ranker")
        st.caption("Games ranked by run environment — O/U + park factor + wind. Stack the #1 or #2 game.")
        # Filter to DK slate teams only
        dk_slate_teams = set(p.get("team","") for p in dk_plays if p.get("dk_salary",0) > 0)
        dk_slate_raw   = [p for p in (st.session_state.get("plays") or plays)
                          if p.get("team","") in dk_slate_teams]
        game_stacks = compute_game_stack_scores(dk_slate_raw if dk_slate_raw else plays)
        if game_stacks:
            gs_rows = []
            for i, gs in enumerate(game_stacks[:12]):
                label    = "🥇 PRIMARY" if i == 0 else ("🥈 SECONDARY" if i == 1 else ("🥉 CONSIDER" if i == 2 else "❌ FADE"))
                home_t   = gs.get("home_team","")
                away_t   = gs.get("away_team","")
                game_str = f"{away_t}@{home_t}" if home_t and away_t else "—"
                home_imp = gs.get("home_implied", 0)
                away_imp = gs.get("away_implied", 0)
                park_hr  = gs.get("park_hr", gs.get("park_factor", 1.0))
                park_str = f"+{(park_hr-1)*100:.0f}%" if park_hr > 1.0 else f"{(park_hr-1)*100:.0f}%"
                wind_eff = gs.get("wind_effect","neutral")
                wind_str = "🏟️ Dome" if gs.get("is_dome") else {
                    "out_strong":"💨 Out (+)","out":"💨 Out","in_strong":"💨 In (-)","in":"💨 In"
                }.get(wind_eff,"→ Neutral")
                gs_rows.append({
                    "":          label,
                    "Game":      game_str,
                    "O/U":       f"{gs.get('game_total',0):.1f}",
                    "Home Impl": f"{home_imp:.1f}R",
                    "Away Impl": f"{away_imp:.1f}R",
                    "Park":      park_str,
                    "Wind":      wind_str,
                    "Score":     f"{gs.get('stack_score',0):.1f}",
                })
            st.dataframe(pd.DataFrame(gs_rows), use_container_width=True, hide_index=True)

        # ── DK SP Target Board ────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🎯 SP Target Board")
        st.caption(f"DK SP scoring: {DK_SP_SCORING}")
        sp_rows = []
        for s in sorted(dk_sp_sal.values(), key=lambda x: -x["salary"])[:10]:
            proj = _compute_dk_sp_proj(s)
            sp_rows.append({
                "SP": s["name"], "Type": s.get("position","SP"), "Team": s["team"],
                "Salary": f"${s['salary']:,}", "DK Proj": f"{proj['dk_sp_proj']:.1f}",
                "Ceiling": f"{proj['dk_sp_ceiling']:.1f}", "Proj IP": f"{proj['proj_ip']:.1f}",
                "Proj K": f"{proj['proj_k']:.0f}", "Value": f"{proj['dk_sp_value']:.2f}x",
            })
        if sp_rows:
            st.dataframe(pd.DataFrame(sp_rows), use_container_width=True, hide_index=True)

        # ── DK Projections Table ──────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 DK Projections — All Batters")
        if dk_plays:
            dk_proj_rows = []
            for p in dk_plays[:40]:
                dk_proj_rows.append({
                    "DK Proj":  f"{p['dk_proj']:.1f}",
                    "Ceiling":  f"{p['dk_ceiling']:.1f}",
                    "Player":   p["name"],
                    "Pos":      p.get("dk_position",""),
                    "Team":     p.get("dk_team",""),
                    "Salary":   f"${p['dk_salary']:,}",
                    "Value":    f"{p['dk_value']:.2f}x",
                    "Own%":     f"{p.get('dk_ownership',15):.0f}%",
                    "Slot":     f"#{p.get('lineup_slot','')}",
                    "Opp SP":   p.get("sp_name","")[:16],
                    "O1.5 Sc":  f"{p.get('score',0):.0f}",
                })
            def _dkp_proj(v):
                try:
                    f = float(v)
                    if f >= 18: return "color:#00ff88;font-weight:bold"
                    if f >= 13: return "color:#ffdd00"
                    return ""
                except: return ""
            def _dkp_val(v):
                try:
                    f = float(str(v).replace("x",""))
                    if f >= 4.0: return "color:#00ffcc;font-weight:bold"
                    if f >= 3.0: return "color:#66dd88"
                    return ""
                except: return ""
            def _dkp_own(v):
                try:
                    f = float(str(v).replace("%",""))
                    if f >= 35: return "color:#ff4444"
                    if f <= 12: return "color:#00ff88"
                    return ""
                except: return ""
            df_dkp = pd.DataFrame(dk_proj_rows)
            if not df_dkp.empty:
                st.dataframe(
                    df_dkp.style.map(_dkp_proj, subset=["DK Proj","Ceiling"]).map(_dkp_val, subset=["Value"]).map(_dkp_own, subset=["Own%"]),
                    use_container_width=True, hide_index=True
                )
            else:
                st.dataframe(df_dkp, use_container_width=True, hide_index=True)

        # ── DK Value Plays ────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("💎 DK Value Plays")
        st.caption("Min-salary plays with strong projections — mass GPP differentiators")
        dk_values = sorted(
            [p for p in dk_plays if 0 < p["dk_salary"] <= 4000 and p["dk_proj"] >= 7.0 and p.get("score",0) >= 50],
            key=lambda x: x["dk_value"], reverse=True
        )
        if dk_values:
            val_rows = []
            for p in dk_values[:10]:
                angles = []
                if p["dk_value"] >= 3.5: angles.append(f"elite {p['dk_value']:.2f}x value")
                elif p["dk_value"] >= 2.8: angles.append(f"strong {p['dk_value']:.2f}x value")
                if p.get("dk_ownership",50) <= 15: angles.append("low owned")
                if p.get("score",0) >= 65: angles.append(f"TB score {p.get('score',0):.0f}")
                val_rows.append({
                    "Player": p["name"], "Team": p.get("dk_team",""), "Pos": p.get("dk_position",""),
                    "Salary": f"${p['dk_salary']:,}", "DK Proj": f"{p['dk_proj']:.1f}",
                    "Ceiling": f"{p['dk_ceiling']:.1f}", "Value": f"{p['dk_value']:.2f}x",
                    "Own%": f"{p.get('dk_ownership',15):.0f}%", "Score": f"{p.get('score',0):.0f}",
                    "🎯 Angle": ", ".join(angles) if angles else "value floor",
                })
            st.dataframe(pd.DataFrame(val_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No DK value plays found (salary ≤ $4,000, proj ≥ 7.0, TB score ≥ 50)")

        return   # ── End DraftKings mode ──────────────────────────────────

    # ═══════════════════════════════════════════════════════════════════
    # FANDUEL MODE (original logic continues below unchanged)
    # ═══════════════════════════════════════════════════════════════════
    st.caption("FD scoring: 1B=3 | 2B=6 | 3B=9 | HR=12 | RBI=3.5 | R=3.2 | BB=3 | SB=6 | Out=-0.25")

    if not plays:
        st.info("Run the model first to generate player data.")
        return

    # ── SALARY CSV UPLOAD ─────────────────────────────────────────────────
    st.subheader("📥 Load FanDuel Slate")
    col_up, col_info = st.columns([2, 3])
    with col_up:
        fd_csv_file = st.file_uploader(
            "Upload FanDuel CSV (from contest lobby)",
            type=["csv"],
            key="fd_cmd_csv_upload",
            help="FanDuel → Lobby → MLB → Contest → Export Players List"
        )
    with col_info:
        if not fd_csv_file:
            st.info(
                "📋 **How to get the CSV:** FanDuel Lobby → MLB → click any contest "
                "→ 'Export Players List' button → upload here.\n\n"
                "This single file loads all hitter salaries, SP salaries, and defines "
                "which games are in your slate."
            )

    # Parse CSV and store in session state
    if fd_csv_file:
        try:
            parsed = _parse_fd_csv(fd_csv_file.read())
            st.session_state["fd_slate_data"] = parsed
            st.success(
                f"✅ {parsed['total_players']} players loaded — "
                f"{len(parsed['salary_data'])} hitters | "
                f"{len(parsed['sp_salary_data'])} pitchers | "
                f"{len(parsed['slate_games'])} games in slate"
            )
        except Exception as e:
            st.error(f"CSV parse error: {e}")

    slate_data = st.session_state.get("fd_slate_data", {})
    salary_data    = slate_data.get("salary_data", {})
    sp_salary_data = slate_data.get("sp_salary_data", {})
    slate_games    = slate_data.get("slate_games", [])
    has_salaries   = len(salary_data) > 0

    # ── SLATE GAME FILTER ─────────────────────────────────────────────────
    if slate_games:
        st.markdown("---")
        st.subheader("🗓️ Slate Games")
        st.caption("All games in your FanDuel contest. Uncheck to exclude games from analysis.")
        selected_games = st.multiselect(
            "Active slate games:",
            options=slate_games,
            default=slate_games,
            key="fd_slate_game_filter",
        )
        active_teams = _slate_teams(selected_games)
    else:
        selected_games = []
        active_teams = set(p.get("team","") for p in plays)

    # Filter plays to slate
    slate_plays = [p for p in plays if p.get("team","") in active_teams] if active_teams else plays

    # Build fd_plays with salaries
    fd_plays = _build_fd_plays_with_salaries(
        plays=slate_plays,
        salary_data=salary_data,
        sp_salary_data=sp_salary_data,
        slate_games=selected_games,
    )

    # Store for lineup builder tab
    st.session_state["fd_plays_current"] = fd_plays
    st.session_state["fd_sp_salary_data"] = sp_salary_data

    if has_salaries:
        matched = sum(1 for p in fd_plays if p.get("salary_matched"))
        st.caption(f"✅ {matched}/{len(fd_plays)} players salary-matched to CSV")

    st.markdown("---")

    if not has_salaries:
        st.info("📥 Upload the FanDuel CSV above to unlock stack rankings, SP board, projections, and value plays.")
        return

    # ── Compute team stack rankings ─────────────────────────────────────────
    # game_scores kept for bring-back display; team_ranks is the primary ranking signal
    game_scores = compute_game_stack_scores(slate_plays)
    team_ranks  = get_ranked_team_stacks(slate_plays, min_players=3)

        # ═══════════════════════════════════════════════════════════════════════
    # SECTION 2 — TEAM STACK VIEWER (ranked list, no game selection required)
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🔗 Stack Rankings")
    st.caption(
        "Teams ranked by composite stack score: implied runs + SP vulnerability + HR potential + hot streaks. "
        "Expand any team to see their top batters, salary, and hot/cold status."
    )

    if team_ranks:
        for rank, sd in enumerate(team_ranks[:12], 1):
            team   = sd["team"]
            score  = sd["stack_score"]
            impl   = sd["implied"]
            opp    = sd["opp_team"]
            comp   = sd["components"]
            park_hr = sd["park_hr"]
            we     = sd["wind_effect"]

            # Rank badge color
            if rank == 1:   badge_color, badge = "#00ff88", "🥇 #1 STACK"
            elif rank == 2: badge_color, badge = "#00cc66", "🥈 #2 STACK"
            elif rank == 3: badge_color, badge = "#ffcc00", "🥉 #3 STACK"
            elif rank <= 6: badge_color, badge = "#ff8800", f"#{rank}"
            else:           badge_color, badge = "#666688", f"#{rank}"

            wind_str = "🏟️ Dome" if sd["is_dome"] else {
                "out_strong":"💨 Out Strong","out":"💨 Out",
                "in_strong":"💨 In Strong","in":"💨 In"
            }.get(we,"→ Neutral")

            impl_str = f"{impl:.1f}R" if impl > 0.5 else "— R"
            hot_flag = "🔥 " if sd["streaking_count"] >= 2 else ""

            # st.expander does not render HTML — use plain text only
            # Show team name only (this is a team ranking, not a matchup listing)
            label_plain = (
                f"{hot_flag}{team} (@ {opp})  |  "
                f"{badge} · Score {score:.0f}  |  "
                f"Impl {impl_str} · O/U {sd['game_total']:.1f} · "
                f"Park {park_hr:.2f}x · {wind_str}"
            )

            with st.expander(label_plain, expanded=(rank <= 2)):
                # Color badge shown inside expander via markdown
                st.markdown(
                    f"<span style='color:{badge_color};font-weight:700;font-size:14px'>"
                    f"{badge}</span> &nbsp; Score <b>{score:.0f}</b>",
                    unsafe_allow_html=True
                )
                # Stack score breakdown in one line
                st.caption(
                    f"Implied {comp['implied_score']:.0f}/40 · "
                    f"SP Vuln {comp['pit_vuln_score']:.0f}/25 (opp: {sd['sp_name']} {sd['sp_hand']}HP) · "
                    f"HR Potential {comp['hr_potential_score']:.0f}/20 · "
                    f"Streaks {comp['streak_score']:.0f}/15"
                )

                # Player table
                players = sorted(sd["players"], key=lambda x: x.get("lineup_slot", 9))
                rows_p  = []
                for p in players[:7]:
                    sal_entry = _fd_name_match(p["name"], salary_data)
                    sal_str   = f"${sal_entry['salary']:,}" if sal_entry else "—"
                    fppg_str  = f"{sal_entry['fppg']:.1f}" if sal_entry else "—"
                    inj_flag  = " ⚠️" if (sal_entry and sal_entry.get("injured")) else ""
                    hot       = "🔥" if "Hot" in p.get("streak_label","") else (
                                "❄️" if "Cold" in p.get("streak_label","") else "—")
                    fd_p      = next((fp for fp in fd_plays if fp["name"] == p["name"]), None)
                    proj_str  = f"{fd_p['fd_proj']:.1f}" if fd_p else "—"
                    rows_p.append({
                        "#":       p.get("lineup_slot","?"),
                        "Player":  p["name"] + inj_flag,
                        "H":       p.get("batter_hand","?"),
                        "Salary":  sal_str,
                        "FD Proj": proj_str,
                        "FPPG":    fppg_str,
                        "Score":   f"{p.get('score',0):.0f}",
                        "HR Sc":   f"{p.get('hr_score',0):.0f}",
                        "Hot":     hot,
                    })
                if rows_p:
                    st.dataframe(
                        pd.DataFrame(rows_p), use_container_width=True,
                        hide_index=True, height=min(320, 55+len(rows_p)*35)
                    )

                # Bring-back note
                opp_stack = next((s for s in team_ranks if s["team"] == opp), None)
                if opp_stack:
                    opp_top = sorted(opp_stack["players"],
                                     key=lambda x: x.get("score",0), reverse=True)[:3]
                    if opp_top:
                        brings = " · ".join(
                            f"{p['name']} (${(_fd_name_match(p['name'], salary_data) or {}).get('salary',0):,})"
                            for p in opp_top
                        )
                        st.caption(f"🔄 Bring-back targets from {opp}: {brings}")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 3 — SP TARGET BOARD
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("⚾ SP Target Board")
    st.caption("FD SP scoring: W=6 | QS=4 | K=3 | IP×3 | ER×-3 | H×-0.6 | BB×-0.6")

    # Build SP projections from plays data
    def _project_fd_sp_from_plays(sp_name, sp_plays, sp_sal_entry):
        """Project FD SP fantasy points from play data."""
        # Get stats from pitcher label
        k_rate, fip, whip = 0.228, 4.10, 1.30
        pl = sp_plays[0].get("pitcher_label","") if sp_plays else ""
        try:
            if "K%:" in pl:
                k_rate = float(pl.split("K%:")[1].split("%")[0].strip()) / 100
            if "FIP:" in pl:
                fip = float(pl.split("FIP:")[1].strip().split()[0])
            if "WHIP:" in pl:
                whip = float(pl.split("WHIP:")[1].strip().split()[0])
        except Exception:
            pass

        # Also use stored _pitcher_k_rate from the batter plays (more reliable post-fix)
        k_rates = [p.get("_pitcher_k_rate", 0) for p in sp_plays if p.get("_pitcher_k_rate", 0) > 0.05]
        if k_rates:
            k_rate = sum(k_rates) / len(k_rates)

        opp_implied = sum(p.get("implied_total",0) for p in sp_plays) / max(1,len(sp_plays))
        # Guard: if implied is 0 (odds missing), use league avg 4.5
        if opp_implied < 0.5:
            opp_implied = 4.5

        # IP projection from FIP
        if fip < 3.0:   proj_ip = 6.5
        elif fip < 3.5: proj_ip = 6.2
        elif fip < 4.0: proj_ip = 5.8
        elif fip < 4.5: proj_ip = 5.4
        else:           proj_ip = 4.8

        # Adjust for opponent run environment
        proj_ip *= max(0.85, 1.0 - (opp_implied - 4.5) * 0.04)

        bf       = proj_ip * 3.3
        proj_k   = k_rate * bf
        proj_er  = max(0, (fip / 9) * proj_ip)
        proj_hbb = whip * proj_ip
        proj_h   = proj_hbb * 0.70
        proj_bb  = proj_hbb * 0.30

        win_prob = 0.50
        if fip < 3.5 and opp_implied < 4.0: win_prob = 0.65
        elif fip > 4.5 or opp_implied > 5.0: win_prob = 0.35

        qs_prob = 0.70 if proj_ip >= 5.8 and proj_er <= 3.0 else 0.35

        fd_pts = (proj_k * 3.0 + proj_ip * 3.0 + proj_er * -3.0 +
                  proj_h * -0.6 + proj_bb * -0.6 +
                  win_prob * 6.0 + qs_prob * 4.0)

        variance = fd_pts * 0.40
        ceiling  = fd_pts + variance
        floor    = max(0, fd_pts - variance * 0.6)

        # SP Score 0-100: lower FIP + lower opp implied + higher K% = better score
        # 100 = elite ace in perfect matchup, 50 = league avg, 0 = bad matchup
        fip_score     = max(0, min(100, (6.0 - fip) / (6.0 - 2.5) * 60))      # FIP: 60% weight
        k_score       = max(0, min(100, (k_rate - 0.12) / (0.35 - 0.12) * 25)) # K%: 25% weight
        matchup_score = max(0, min(100, (6.0 - opp_implied) / (6.0 - 3.0) * 15))# Opp implied: 15%
        sp_dfs_score  = round(fip_score + k_score + matchup_score)
        grade = f"{sp_dfs_score}"  # just the number — display layer adds color

        salary  = sp_sal_entry["salary"] if sp_sal_entry else 0
        value   = round(fd_pts / (salary / 1000), 2) if salary > 0 else 0.0

        return {
            "name":        sp_name,
            "hand":        sp_plays[0].get("sp_hand","R") if sp_plays else "R",
            "opp":         sp_plays[0].get("team","") if sp_plays else "",
            "park":        sp_plays[0].get("park","") if sp_plays else "",
            "k_rate":      k_rate,
            "fip":         fip,
            "opp_implied": opp_implied,
            "proj_ip":     round(proj_ip,1),
            "proj_k":      round(proj_k,1),
            "proj_er":     round(proj_er,1),
            "win_prob":    round(win_prob*100),
            "qs_prob":     round(qs_prob*100),
            "fd_sp_proj":  round(fd_pts,1),
            "fd_ceiling":  round(ceiling,1),
            "fd_floor":    round(floor,1),
            "salary":      salary,
            "value":       value,
            "grade":       grade,
        }

    # Collect SP data from plays
    pitchers_seen = {}
    for p in slate_plays:
        sp = p.get("sp_name","TBD")
        if sp and sp != "TBD":
            if sp not in pitchers_seen:
                pitchers_seen[sp] = []
            pitchers_seen[sp].append(p)

    sp_projections = []
    for sp_name, sp_plays_list in pitchers_seen.items():
        sal_entry = _fd_name_match(sp_name, sp_salary_data) if sp_salary_data else None
        proj = _project_fd_sp_from_plays(sp_name, sp_plays_list, sal_entry)
        sp_projections.append(proj)

    sp_projections.sort(key=lambda x: x["fd_sp_proj"], reverse=True)
    # Cache for Lineup Builder tab SP tier selectors
    st.session_state["fd_sp_projections"] = sp_projections

    if sp_projections:
        # Top callout
        top_sp   = sp_projections[0]
        value_sp = max((s for s in sp_projections if s["salary"] > 0),
                       key=lambda x: x["value"], default=None)
        st.markdown(
            f"⭐ **Top Target:** {top_sp['name']} ({top_sp['opp']} opp) — "
            f"**{top_sp['fd_sp_proj']:.1f}** proj | Ceil: {top_sp['fd_ceiling']:.1f} | "
            f"FIP: {top_sp['fip']:.2f} | {top_sp['proj_k']:.0f} K | {top_sp['grade']}"
        )
        if value_sp:
            st.markdown(
                f"💰 **Top Value:** {value_sp['name']} — "
                f"{value_sp['fd_sp_proj']:.1f} proj | ${value_sp['salary']:,} | "
                f"{value_sp['value']:.2f}x value"
            )

        sp_rows = []
        for s in sp_projections:
            sp_rows.append({
                "Score":    s["grade"],  # numeric 0-100 SP DFS score
                "SP":       s["name"],
                "H":        s["hand"],
                "Opp":      s["opp"],
                "Park":     s["park"],
                "FD Proj":  f"{s['fd_sp_proj']:.1f}",
                "Ceiling":  f"{s['fd_ceiling']:.1f}",
                "Floor":    f"{s['fd_floor']:.1f}",
                "FIP":      f"{s['fip']:.2f}",
                "K%":       f"{s['k_rate']*100:.0f}%",
                "Proj K":   f"{s['proj_k']:.0f}",
                "Proj IP":  f"{s['proj_ip']:.1f}",
                "Win%":     f"{s['win_prob']:.0f}%",
                "QS%":      f"{s['qs_prob']:.0f}%",
                "Opp Imp":  f"{s['opp_implied']:.1f}" if s['opp_implied'] > 0.5 else "—",
                "Salary":   f"${s['salary']:,}" if s["salary"] > 0 else "—",
                "Value":    f"{s['value']:.2f}x" if s["value"] > 0 else "—",
            })

        sp_df = pd.DataFrame(sp_rows)

        def _cg(v):
            try:
                sc = int(float(str(v)))
                if sc >= 75: return "color:#00ff88;font-weight:bold"
                if sc >= 60: return "color:#66dd88"
                if sc >= 45: return "color:#ffdd00"
                if sc >= 30: return "color:#ff8800"
                return "color:#ff4444"
            except: return ""

        def _cp(v):
            try:
                f = float(v)
                if f >= 35: return "color:#00ff88;font-weight:bold"
                if f >= 25: return "color:#ffdd00"
                if f < 15:  return "color:#ff4444"
                return ""
            except: return ""

        if sp_df.empty or "Score" not in sp_df.columns:
            st.info("SP projections unavailable — run the model first.")
        else:
            st.dataframe(
                sp_df.style.map(_cg, subset=["Score"]).map(_cp, subset=["FD Proj","Ceiling"]),
                use_container_width=True, hide_index=True
            )
    else:
        st.info("Run model first to see SP projections.")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 4 — FD PROJECTIONS TABLE
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📊 FD Projections — All Batters")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        min_proj = st.slider("Min projection", 0.0, 40.0, 5.0, 0.5, key="fd_cmd_min_proj")
    with col_f2:
        teams_avail = sorted(set(p["team"] for p in fd_plays))
        team_filter = st.multiselect("Filter team", teams_avail, default=[], key="fd_cmd_team_filter")

    filtered = [p for p in fd_plays
                if p["fd_proj"] >= min_proj
                and (not team_filter or p["team"] in team_filter)]

    if filtered:
        proj_rows = []
        for p in filtered:
            inj = " ⚠️" if p.get("fd_injured") else ""
            proj_rows.append({
                "FD Proj":  f"{p['fd_proj']:.1f}",
                "Ceiling":  f"{p['fd_ceiling']:.1f}",
                "Floor":    f"{p['fd_floor']:.1f}",
                "Player":   p["name"] + inj,
                "Pos":      p.get("fd_position","—"),
                "Team":     p["team"],
                "Sal":      f"${p['fd_salary']:,}" if p["fd_salary"] > 0 else "—",
                "Value":    f"{p['fd_value']:.1f}x" if p["fd_value"] > 0 else "—",
                "Own%":     f"{p['ownership']:.0f}%",
                "Slot":     f"#{p.get('lineup_slot','')}",
                "Opp SP":   p.get("sp_name","")[:16],
                "xSLG":     f"{p.get('xslg',0):.3f}" if p.get("xslg") else "—",
                "K%":       f"{p.get('k_rate',0)*100:.1f}%" if p.get("k_rate") else "—",
                "Park":     p.get("park",""),
                "O1.5 Sc":  f"{p.get('score',0):.0f}",
            })

        df_proj = pd.DataFrame(proj_rows)

        def _cc(v):
            try:
                f = float(str(v))
                if f >= 20: return "color:#00ff88;font-weight:bold"
                if f >= 14: return "color:#ffdd00;font-weight:bold"
                if f >= 9:  return "color:#ff8800"
                return ""
            except: return ""

        def _co(v):
            try:
                f = float(str(v).replace("%",""))
                if f <= 10: return "color:#00ff88"
                if f >= 35: return "color:#ff4444"
                return ""
            except: return ""

        def _cv(v):
            try:
                f = float(str(v).replace("x",""))
                if f >= 4.0: return "color:#00ff88;font-weight:bold"
                if f >= 3.0: return "color:#ffdd00"
                return ""
            except: return ""

        st.dataframe(
            df_proj.style.map(_cc, subset=["FD Proj","Ceiling"]).map(_co, subset=["Own%"]).map(_cv, subset=["Value"]),
            use_container_width=True, height=420, hide_index=True
        )

        csv_proj = df_proj.to_csv(index=False)
        st.download_button(
            "📥 Export FD Projections CSV", csv_proj,
            f"fd_proj_{datetime.now(EST).strftime('%Y%m%d')}.csv", "text/csv",
            key="fd_cmd_export_proj"
        )
    else:
        st.info("No players match the current filters.")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 5 — TOP GAME STACKS
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🔥 Top Game Stacks")
    st.caption("GPP: stack 4-5 batters from the same high-total game for correlated upside")

    game_groups = {}
    for p in fd_plays:
        key = f"{p.get('game_id','')}_{p.get('team','')}"
        if key not in game_groups:
            game_groups[key] = {
                "game": f"{p.get('opponent','')}@{p.get('team','')}",
                "team": p.get("team",""),
                "implied": p.get("implied_total", 0),
                "players": [],
            }
        game_groups[key]["players"].append(p)

    stack_scores = []
    for key, grp in game_groups.items():
        players = sorted(grp["players"], key=lambda x: x["fd_proj"], reverse=True)
        top4 = players[:4]
        if len(top4) < 3:
            continue
        avg_proj = sum(p["fd_proj"] for p in top4) / len(top4)
        avg_ceil = sum(p["fd_ceiling"] for p in top4) / len(top4)
        impl     = grp["implied"] if grp["implied"] > 0.5 else 4.5
        score    = avg_proj * 0.5 + avg_ceil * 0.3 + impl * 2
        stack_scores.append({
            "game": grp["game"], "team": grp["team"],
            "implied": grp["implied"], "top4": top4,
            "avg_proj": round(avg_proj,1), "avg_ceil": round(avg_ceil,1),
            "score": round(score,1),
        })

    stack_scores.sort(key=lambda x: x["score"], reverse=True)

    for i, stack in enumerate(stack_scores[:3], 1):
        impl_str = f"{stack['implied']:.1f}" if stack["implied"] > 0.5 else "—"
        with st.expander(
            f"#{i} Stack: {stack['team']} ({stack['game']}) — "
            f"Implied: {impl_str} runs | Avg Proj: {stack['avg_proj']:.1f} pts",
            expanded=(i == 1)
        ):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**4-man stack ({stack['team']}):**")
                for p in stack["top4"][:4]:
                    sal_str = f"${p['fd_salary']:,}" if p["fd_salary"] > 0 else "no sal"
                    own_str = f"{p['ownership']:.0f}% own"
                    hot     = "🔥 " if "Hot" in p.get("streak_label","") else ""
                    st.write(f"• {hot}**{p['name']}** #{p.get('lineup_slot','')} — {p['fd_proj']:.1f} proj | {sal_str} | {own_str}")
            with col_b:
                st.markdown("**Bring-back targets (opp team):**")
                opp_team = stack["top4"][0].get("opponent","") if stack["top4"] else ""
                bring_backs = sorted(
                    [p for p in fd_plays if p["team"] == opp_team and p.get("implied_total",0) > 0.5],
                    key=lambda x: x["fd_proj"], reverse=True
                )[:3]
                for p in bring_backs:
                    sal_str = f"${p['fd_salary']:,}" if p["fd_salary"] > 0 else "no sal"
                    st.write(f"• {p['name']} — {p['fd_proj']:.1f} proj | {sal_str}")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 5b — VALUE STACKS
    # 3 best value stacks: strong implied total but lower ownership/salary
    # Ideal as secondary stacks when pairing with an ACE pitcher
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("💰 Value Stacks")
    st.caption("High-value secondary stack targets — lower salary, lower ownership, solid implied. Best paired as secondary stack when rostering an ACE SP.")

    # Get team stack scores for all slate teams
    all_team_stack_data = []
    for g in game_scores:
        for team in [g.get("home_team",""), g.get("away_team","")]:
            if not team:
                continue
            team_plays = [p for p in fd_plays if p.get("team") == team and p.get("fd_salary",0) > 0]
            if len(team_plays) < 3:
                continue
            avg_proj    = sum(p.get("fd_proj",0) for p in team_plays[:4]) / min(4, len(team_plays))
            avg_sal     = sum(p.get("fd_salary",0) for p in team_plays[:4]) / min(4, len(team_plays))
            avg_own     = sum(p.get("ownership",20) for p in team_plays[:4]) / min(4, len(team_plays))
            implied     = team_plays[0].get("implied_total", 4.5) if team_plays else 4.5
            # Value score: good proj + low salary + low ownership + decent implied
            value_score = (
                min(100, avg_proj / 18.0 * 40) +       # proj quality (40 pts)
                min(100, (5000 - avg_sal) / 3000 * 30) + # salary efficiency (30 pts)
                min(100, (35 - avg_own) / 30 * 20) +    # ownership leverage (20 pts)
                min(100, implied / 6.0 * 10)             # run environment (10 pts)
            )
            all_team_stack_data.append({
                "team": team,
                "game": f"{g.get('away_team','')}@{g.get('home_team','')}",
                "implied": implied,
                "avg_proj": avg_proj,
                "avg_sal": avg_sal,
                "avg_own": avg_own,
                "value_score": value_score,
                "players": team_plays[:4],
                "game_total": g.get("game_total", 8.5),
            })

    # Sort by value score, exclude the top 2 primary stacks (those are in Top Game Stacks)
    primary_teams = set()
    for gs in sorted(game_scores, key=lambda x: -x["stack_score"])[:2]:
        primary_teams.add(gs.get("home_team",""))
        primary_teams.add(gs.get("away_team",""))

    value_stacks = sorted(
        [s for s in all_team_stack_data if s["team"] not in primary_teams],
        key=lambda x: -x["value_score"]
    )[:3]

    if value_stacks:
        vcols = st.columns(3)
        for vcol, vs in zip(vcols, value_stacks):
            with vcol:
                st.markdown(
                    f"<div style='background:#0d1a0d;border:1px solid #00cc66;"
                    f"border-radius:10px;padding:12px 14px;margin-bottom:8px'>"
                    f"<div style='color:#00cc66;font-size:0.7rem;font-weight:700'>💰 VALUE STACK</div>"
                    f"<div style='color:#00ff88;font-size:1.3rem;font-weight:800'>{vs['team']}</div>"
                    f"<div style='color:#9090a8;font-size:0.75rem'>{vs['game']} | O/U {vs['game_total']:.1f} | "
                    f"Impl {vs['implied']:.1f}R</div>"
                    f"<div style='color:#aaa;font-size:0.72rem;margin-top:6px'>"
                    f"Avg proj {vs['avg_proj']:.1f}pts · Avg sal ${vs['avg_sal']:,.0f} · "
                    f"Avg own {vs['avg_own']:.0f}%</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                for p in vs["players"]:
                    sal_str = f"${p['fd_salary']:,}" if p.get("fd_salary",0) > 0 else "no sal"
                    st.write(f"• **{p['name']}** #{p.get('lineup_slot','?')} · {p.get('fd_proj',0):.1f}pts · {sal_str}")
    else:
        st.info("Stack data available after model run + CSV upload.")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 6 — VALUE PLAYS
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("💎 Value Plays")
    st.caption("Min-salary plays with strong projections relative to cost — GPP differentiators")

    if has_salaries:
        values = sorted(
            [p for p in fd_plays
             if 0 < p["fd_salary"] <= 3200
             and p["fd_proj"] >= 8.0
             and p.get("score", 0) >= 52],
            key=lambda x: x["fd_value"], reverse=True
        )
        if values:
            val_rows = []
            for p in values[:10]:
                angles = []
                if p["fd_value"] >= 5.0:       angles.append(f"elite {p['fd_value']:.1f}x value")
                elif p["fd_value"] >= 4.0:     angles.append(f"strong {p['fd_value']:.1f}x value")
                if p.get("ownership", 50) <= 20: angles.append("low owned")
                if p.get("score", 0) >= 65:    angles.append(f"TB score {p.get('score',0):.0f}")
                if p.get("hr_score", 0) >= 55: angles.append("HR upside")
                sp_name = p.get("sp_name", "")
                if sp_name: angles.append(f"vs {sp_name[:12]}")
                val_rows.append({
                    "Player":   p["name"],
                    "Team":     p["team"],
                    "Pos":      p.get("fd_position",""),
                    "Salary":   f"${p['fd_salary']:,}",
                    "FD Proj":  f"{p['fd_proj']:.1f}",
                    "Ceiling":  f"{p['fd_ceiling']:.1f}",
                    "Value":    f"{p['fd_value']:.1f}x",
                    "Own%":     f"{p['ownership']:.0f}%",
                    "Slot":     f"#{p.get('lineup_slot','')}",
                    "Opp SP":   p.get("sp_name","")[:18],
                    "Score":    f"{p.get('score',0):.0f}",
                    "🎯 Angle": ", ".join(angles) if angles else "value floor",
                })

            def _vval(v):
                try:
                    f = float(str(v).replace("x",""))
                    if f >= 5.0: return "color:#00ff88;font-weight:bold"
                    if f >= 4.0: return "color:#ffdd00"
                    return ""
                except: return ""
            def _vown(v):
                try:
                    f = float(str(v).replace("%",""))
                    if f <= 15: return "color:#00ff88"
                    if f >= 35: return "color:#ff4444"
                    return ""
                except: return ""

            df_val = pd.DataFrame(val_rows)
            st.dataframe(
                df_val.style.map(_vval, subset=["Value"]).map(_vown, subset=["Own%"]),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No value plays found under $3,200 with proj ≥ 8.0 and TB Score ≥ 52")
    else:
        st.info("📥 Upload FanDuel CSV above to see value plays.")



# ============================================================================
# FD HAND BUILDER — Single high-stakes entry, full data visibility
# ============================================================================

def display_fd_hand_builder(plays: List[Dict]):
    """
    Tab 8 — Hand Builder (FD + DK)
    Quick-reference intel for manual single-entry lineup construction.
    Shows: top stack games, top 10 players per position, contrarian angles.
    No clicking, no locking — just clean data to inform your decisions.
    """
    st.markdown("""
    <style>
    .hb2-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #e94560; border-radius: 12px;
        padding: 20px 24px; margin-bottom: 20px;
    }
    .hb2-title { color: #e94560; font-size: 26px; font-weight: 800; }
    .hb2-sub { color: #9090a8; font-size: 13px; margin-top: 4px; }
    .pos-header {
        background: #12122a; border-left: 4px solid #e94560;
        padding: 8px 14px; border-radius: 0 8px 8px 0;
        margin: 16px 0 8px 0; font-weight: 700; font-size: 15px; color: #e0e0ff;
    }
    .stack-card {
        background: #1a1a2e; border: 1px solid #2a2a4a;
        border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;
    }
    .stack-rank { color: #e94560; font-size: 22px; font-weight: 900; }
    .stack-team { color: #00ff88; font-size: 18px; font-weight: 700; }
    .stack-note { color: #9090a8; font-size: 12px; margin-top: 4px; }
    .contrarian-card {
        background: linear-gradient(135deg,#0d2818,#1a2a0d);
        border: 1px solid #00cc66; border-radius: 10px;
        padding: 14px 18px; margin-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Auto-detect which platform has data loaded — default to whichever was most recently loaded
    has_fd = bool(st.session_state.get("fd_plays_current"))
    has_dk = bool(st.session_state.get("dk_slate_data"))

    if has_dk and not has_fd:
        auto_idx = 1   # DraftKings
    else:
        auto_idx = 0   # FanDuel (default)

    # Site toggle — auto-selects based on what's loaded, user can override
    hb_site = st.radio("Platform", ["FanDuel", "DraftKings"], horizontal=True,
                       index=auto_idx, key="hb_site_toggle")

    st.markdown(f"""
    <div class="hb2-header">
        <div class="hb2-title">🔬 {hb_site} Hand Builder — Intel Dashboard</div>
        <div class="hb2-sub">Quick-reference for single high-stakes entries.
        Top stacks · Top plays by position · Contrarian angles.
        Use this alongside your own ownership research before finalizing.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── DraftKings mode ───────────────────────────────────────────────────────
    if hb_site == "DraftKings":
        dk_slate = st.session_state.get("dk_slate_data")
        if not dk_slate:
            st.warning("⚠️ Upload DraftKings CSV in the **Command Center** tab first.")
            return

        dk_plays = _build_dk_plays_with_salaries(
            plays, dk_slate["salary_data"], dk_slate["sp_salary_data"]
        )
        if not dk_plays:
            st.info("No DK salary-matched players found.")
            return

        # Top Stacks — filter to DK slate teams only
        st.markdown("### 🔗 Top Stacks — Where to Build Your Core")
        st.caption("Ranked by implied total + park factor + wind. Stack 5 batters from your #1 game.")
        dk_slate_tms   = set(p.get("dk_team", p.get("team","")) for p in dk_plays if p.get("dk_salary",0) > 0)
        dk_slate_raws  = [p for p in (st.session_state.get("plays") or plays)
                          if p.get("team","") in dk_slate_tms]
        game_stacks = compute_game_stack_scores(dk_slate_raws if dk_slate_raws else plays)
        top3_games  = game_stacks[:3] if game_stacks else []
        if top3_games:
            gcols = st.columns(3)
            labels = ["🥇 PRIMARY", "🥈 SECONDARY", "🥉 CONSIDER"]
            colors = ["#ff8800", "#3399ff", "#9966ff"]
            for col, gs, lbl, clr in zip(gcols, top3_games, labels, colors):
                # Assign all variables BEFORE rendering
                home_t   = gs.get("home_team","")
                away_t   = gs.get("away_team","")
                home_imp = gs.get("home_implied", 0)
                away_imp = gs.get("away_implied", 0)
                best_team = home_t if home_imp >= away_imp else away_t
                opp_team  = away_t if best_team == home_t else home_t
                top_team_plays = sorted(
                    [p for p in dk_plays
                     if p.get("dk_team","") == best_team or p.get("team","") == best_team],
                    key=lambda x: x["dk_proj"], reverse=True
                )[:5]
                with col:
                    st.markdown(
                        f"<div style='color:{clr};font-size:0.7rem;font-weight:700'>{lbl}</div>"
                        f"<div style='color:#00ff88;font-size:1.4rem;font-weight:800'>{best_team}</div>"
                        f"<div style='color:#9090a8;font-size:0.75rem'>{away_t}@{home_t} | O/U {gs.get('game_total',0):.1f} | Impl {max(home_imp,away_imp):.1f}R</div>",
                        unsafe_allow_html=True
                    )
                    for p in top_team_plays:
                        hot = "🔥 " if p.get("sub_streak",0) >= 65 else ""
                        st.write(f"{hot}**{p['name']}** #{p.get('lineup_slot','?')} · ${p['dk_salary']:,} · {p['dk_proj']:.1f}pts · {p.get('dk_ownership',15):.0f}% own")

        st.markdown("---")

        # DK SP Board
        st.markdown("### ⚾ Starting Pitcher — Pick Two")
        st.caption("DK uses 2 SP slots. Score = FIP quality + K rate + matchup.")
        dk_sp_sal = dk_slate["sp_salary_data"]
        sp_rows = []
        for s in sorted(dk_sp_sal.values(), key=lambda x: -x["salary"])[:8]:
            proj = _compute_dk_sp_proj(s)
            sp_rows.append({
                "SP": s["name"], "Type": s.get("position","SP"), "Team": s["team"],
                "Salary": f"${s['salary']:,}", "DK Proj": f"{proj['dk_sp_proj']:.1f}",
                "Ceiling": f"{proj['dk_sp_ceiling']:.1f}", "Proj IP": f"{proj['proj_ip']:.1f}",
                "Proj K": f"{proj['proj_k']:.0f}", "Value": f"{proj['dk_sp_value']:.2f}x",
            })
        if sp_rows:
            st.dataframe(pd.DataFrame(sp_rows), use_container_width=True, hide_index=True)

        st.markdown("---")

        # DK Top plays by position
        st.markdown("### 🏆 Top Plays by Position")
        DK_POSITIONS = [("C", "⚡ Catcher"), ("1B", "💠 First Base"), ("2B", "♦ Second Base"),
                        ("3B", "🔷 Third Base"), ("SS", "⚡ Shortstop"), ("OF", "🌿 Outfield")]

        # Shared color functions — matches FD formatting exactly
        def _dk_cproj(v):
            try:
                f = float(str(v))
                if f >= 16: return "color:#00ff88;font-weight:bold"
                if f >= 12: return "color:#ffdd00"
                if f >= 9:  return "color:#ff8800"
                return ""
            except: return ""
        def _dk_cown(v):
            try:
                f = float(str(v).replace("%",""))
                if f >= 35: return "color:#ff4444"
                if f <= 12: return "color:#00ff88"
                return ""
            except: return ""
        def _dk_cval(v):
            try:
                f = float(str(v).replace("x",""))
                if f >= 4.0: return "color:#00ffcc;font-weight:bold"
                if f >= 3.0: return "color:#66dd88"
                return ""
            except: return ""

        for slot, header in DK_POSITIONS:
            eligible_pos = [p for p in dk_plays if _dk_pos_eligible(str(p.get("dk_position","")), slot)]
            eligible_pos = sorted(eligible_pos, key=lambda x: x["dk_proj"], reverse=True)[:8]
            if not eligible_pos:
                continue
            st.markdown(f"<div class='pos-header'>{header}</div>", unsafe_allow_html=True)
            pos_rows = []
            for p in eligible_pos:
                angles = []
                if p.get("score",0) >= 65:     angles.append(f"TB score {p.get('score',0):.0f}")
                if p.get("dk_value",0) >= 3.0:  angles.append(f"{p['dk_value']:.2f}x value")
                if p.get("dk_ownership",50) <= 15: angles.append("contrarian")
                if p.get("sub_streak",0) >= 65: angles.append("🔥 hot streak")
                pos_rows.append({
                    "Hot":    "🔥" if p.get("sub_streak",0) >= 65 else "—",
                    "Player": p["name"],
                    "Team":   p.get("dk_team", p.get("team","")),
                    "Slot":   f"#{p.get('lineup_slot','?')}",
                    "Salary": f"${p['dk_salary']:,}",
                    "Proj":   f"{p['dk_proj']:.1f}",
                    "Ceil":   f"{p['dk_ceiling']:.1f}",
                    "Value":  f"{p['dk_value']:.2f}x",
                    "Own%":   f"{p.get('dk_ownership',15):.0f}%",
                    "Opp SP": p.get("sp_name","")[:14],
                    "💡 Why": ", ".join(angles) if angles else "solid floor",
                })
            df_pos = pd.DataFrame(pos_rows)
            if not df_pos.empty:
                styled = df_pos.style.map(_dk_cproj, subset=["Proj","Ceil"]).map(_dk_cown, subset=["Own%"]).map(_dk_cval, subset=["Value"])
                st.dataframe(styled, use_container_width=True, hide_index=True)
            else:
                st.dataframe(df_pos, use_container_width=True, hide_index=True)

        return   # ── End DK Hand Builder ──────────────────────────────────

    # ═══════════════════════════════════════════════════════════════════
    # FANDUEL MODE (original logic continues below)
    # ═══════════════════════════════════════════════════════════════════
    fd_plays    = st.session_state.get("fd_plays_current", [])
    sp_sal_data = st.session_state.get("fd_sp_salary_data", {})
    sp_proj     = st.session_state.get("fd_sp_projections", [])

    if not fd_plays:
        st.warning("⚠️ Load the FanDuel CSV in the **Command Center** tab first.")
        return

    eligible = [p for p in fd_plays
                if p.get("fd_salary",0) > 0
                and not p.get("is_postponed",False)
                and not p.get("fd_injured",False)]

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 1 — TOP STACKS
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="pos-header">🔗 Top Stacks — Where to Build Your Core</div>',
                unsafe_allow_html=True)
    st.caption("Teams ranked by composite stack score: implied runs + SP vulnerability + HR potential + hot streaks.")

    # Filter to slate teams only, use team-level ranking
    raw_plays   = st.session_state.get("plays", fd_plays)
    slate_teams = set(p.get("team","") for p in fd_plays if p.get("fd_salary",0) > 0)
    slate_plays = [p for p in (raw_plays or fd_plays) if p.get("team","") in slate_teams]
    top_team_ranks = get_ranked_team_stacks(slate_plays if slate_plays else fd_plays, min_players=3)

    col_s1, col_s2, col_s3 = st.columns(3)
    for i, (col, sd) in enumerate(zip([col_s1, col_s2, col_s3], top_team_ranks[:3])):
        with col:
            rank_labels = ["🥇 PRIMARY", "🥈 SECONDARY", "🥉 CONSIDER"]
            rank_colors = ["#e94560","#ffcc00","#9090a8"]
            stack_team = sd["team"]
            vs_team    = sd.get("opp_team","")
            team_impl  = sd.get("implied", 0)
            comp       = sd.get("components", {})

            park_hr  = sd.get("park_hr", 1.0)
            wind_eff = sd.get("wind_effect","neutral")
            is_dome  = sd.get("is_dome", False)
            wind_str = "🏟️ Dome" if is_dome else {
                "out_strong":"💨 Out","out":"💨 Out","in_strong":"💨 In","in":"💨 In"
            }.get(wind_eff,"→ Neutral")

            # Top 4 players from stack team
            team_players = sorted(
                [p for p in eligible if p.get("team") == stack_team],
                key=lambda x: x.get("fd_proj",0), reverse=True
            )[:4]

            impl_str = f"{team_impl:.1f}" if team_impl > 0.5 else "—"
            st.markdown(
                f"<div style='color:{rank_colors[i]};font-weight:900;font-size:13px'>{rank_labels[i]}</div>"
                f"<div style='color:#00ff88;font-size:22px;font-weight:900'>{stack_team}</div>"
                f"<div style='color:#9090a8;font-size:12px'>vs {vs_team} | O/U {sd.get('game_total',0):.1f} | "
                f"Impl {impl_str}R | Park {park_hr:.2f}x | {wind_str}</div>",
                unsafe_allow_html=True
            )
            if team_players:
                for p in team_players:
                    hot = "🔥 " if "Hot" in p.get("streak_label","") else ""
                    sal = f"${p['fd_salary']:,}" if p["fd_salary"] > 0 else "—"
                    st.markdown(
                        f"<div style='padding:3px 0;font-size:13px'>"
                        f"{hot}<b>{p['name']}</b> #{p.get('lineup_slot','')} "
                        f"<span style='color:#9090a8'>{p.get('fd_position','')} · {sal} · "
                        f"{p.get('fd_proj',0):.1f}pts · {p.get('ownership',0):.0f}% own</span></div>",
                        unsafe_allow_html=True
                    )
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 2 — SP TARGET BOARD
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="pos-header">⚾ Starting Pitcher — Pick One</div>',
                unsafe_allow_html=True)
    st.caption("FD uses 1 SP. Score = FIP quality + K rate + matchup. Higher = better DFS play.")

    if sp_proj:
        sp_cols = st.columns(min(4, len(sp_proj[:8])))
        for i, (col, s) in enumerate(zip(sp_cols * 4, sp_proj[:8])):
            sal_e = _fd_name_match(s["name"], sp_sal_data) if sp_sal_data else None
            sal_v = sal_e["salary"] if sal_e else 0
            try:
                score_int = int(float(str(s.get("grade","0"))))
            except (ValueError, TypeError):
                score_int = 0
            sc_color = "#00ff88" if score_int >= 75 else "#ffdd00" if score_int >= 55 else "#ff8800"
            with col:
                st.markdown(
                    f"<div style='background:#1a1a2e;border:1px solid #2a2a4a;border-radius:8px;"
                    f"padding:10px;margin-bottom:8px;text-align:center'>"
                    f"<div style='color:{sc_color};font-size:24px;font-weight:900'>{score_int}</div>"
                    f"<div style='font-size:12px;font-weight:700;color:#e0e0ff'>{s['name']}</div>"
                    f"<div style='color:#9090a8;font-size:11px'>{s.get('opp','')} opp | FIP {s['fip']:.2f}</div>"
                    f"<div style='color:#9090a8;font-size:11px'>K% {s.get('k_rate',0)*100:.0f}% | "
                    f"${sal_v:,}" + (" | Impl " + str(round(s.get('opp_implied',0),1)) if s.get('opp_implied',0) > 0.5 else "") +
                    f"</div></div>",
                    unsafe_allow_html=True
                )

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 3 — TOP 10 BY POSITION
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="pos-header">🎯 Top Plays by Position</div>',
                unsafe_allow_html=True)
    st.caption("Sorted by FD projection. Color: proj (green=elite) · value (teal) · own% (red=chalk).")

    def _pos_filter(plays_list, pos):
        pos_map = {
            "C/1B":  lambda rp: any(s in rp for s in ["C","1B"]),
            "2B":    lambda rp: "2B" in rp,
            "3B":    lambda rp: "3B" in rp,
            "SS":    lambda rp: "SS" in rp,
            "OF":    lambda rp: any(s in rp for s in ["OF","LF","RF","CF"]),
            "UTIL":  lambda rp: True,
        }
        check = pos_map.get(pos, lambda rp: True)
        return [p for p in plays_list
                if check(p.get("fd_roster_pos", p.get("fd_position","OF")).upper())]

    def _build_pos_df(pos_plays, n=10):
        rows = []
        for p in sorted(pos_plays, key=lambda x: x.get("fd_proj",0), reverse=True)[:n]:
            hot    = "🔥" if "Hot" in p.get("streak_label","") else ""
            sal    = p.get("fd_salary",0)
            proj   = p.get("fd_proj",0)
            ceil   = p.get("fd_ceiling",0)
            val    = p.get("fd_value",0)
            own    = p.get("ownership",0)
            hr_sc  = p.get("hr_score",0)
            streak = p.get("sub_streak",50)
            slot   = p.get("lineup_slot","")
            sp     = p.get("sp_name","")[:14]
            hand   = p.get("batter_hand","")
            team   = p.get("team","")
            # Why this player — build a note
            notes = []
            if proj >= 18:   notes.append("elite proj")
            if val >= 4.0:   notes.append(f"{val:.1f}x value")
            if hr_sc >= 55:  notes.append(f"HR score {hr_sc:.0f}")
            if streak >= 72: notes.append("🔥 blazing hot")
            if own <= 15:    notes.append("contrarian")
            if sal <= 2500:  notes.append("min-price")
            rows.append({
                "Hot":    hot or "—",
                "Player": p["name"],
                "Team":   team,
                "Slot":   f"#{slot}" if slot else "—",
                "H":      hand,
                "Salary": f"${sal:,}" if sal else "—",
                "Proj":   f"{proj:.1f}",
                "Ceil":   f"{ceil:.1f}",
                "Value":  f"{val:.1f}x" if val else "—",
                "Own%":   f"{own:.0f}%",
                "HR Sc":  int(hr_sc),
                "Streak": int(streak),
                "Opp SP": sp,
                "💡 Why": ", ".join(notes) if notes else "solid floor",
            })
        return pd.DataFrame(rows)

    def _style_pos_df(df):
        def _cproj(v):
            try:
                f = float(v)
                if f >= 20: return "color:#00ff88;font-weight:bold"
                if f >= 15: return "color:#ffdd00"
                if f >= 10: return "color:#ff8800"
                return ""
            except: return ""
        def _cown(v):
            try:
                f = float(str(v).replace("%",""))
                if f >= 35: return "color:#ff4444"
                if f <= 12: return "color:#00ff88"
                return ""
            except: return ""
        def _cval(v):
            try:
                f = float(str(v).replace("x",""))
                if f >= 4.5: return "color:#00ffcc;font-weight:bold"
                if f >= 3.5: return "color:#66dd88"
                return ""
            except: return ""
        if df.empty or "Proj" not in df.columns:
            return df.style
        return df.style.map(_cproj, subset=["Proj","Ceil"]).map(_cown, subset=["Own%"]).map(_cval, subset=["Value"])

    POSITIONS = [
        ("C/1B",  "⚡ Catcher / First Base"),
        ("2B",    "🔷 Second Base"),
        ("3B",    "🔶 Third Base"),
        ("SS",    "💫 Shortstop"),
        ("OF",    "🌿 Outfield (3 slots — pick 3)"),
        ("UTIL",  "🔄 UTIL (any hitter)"),
    ]

    for pos_key, pos_label in POSITIONS:
        pos_plays = _pos_filter(eligible, pos_key)
        df = _build_pos_df(pos_plays, n=10)
        st.markdown(f"<div class='pos-header'>{pos_label}</div>", unsafe_allow_html=True)
        if df.empty:
            st.info(f"No salary-matched {pos_key} players in this slate.")
        else:
            st.dataframe(
                _style_pos_df(df),
                use_container_width=True, hide_index=True,
                height=min(420, 55 + len(df) * 35)
            )

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 4 — CONTRARIAN ANGLES
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="pos-header">🎲 Contrarian Angles — How to Get Different</div>',
                unsafe_allow_html=True)
    st.caption("Low-ownership players with real upside. In tournaments you win by being RIGHT when others are wrong.")

    contrarians = sorted(
        [p for p in eligible
         if p.get("ownership",100) <= 18
         and p.get("fd_proj",0) >= 10.0
         and p.get("fd_salary",0) >= 2000],
        key=lambda x: x.get("fd_proj",0) / max(1, x.get("ownership",20)),
        reverse=True
    )[:8]

    if contrarians:
        c_rows = []
        for p in contrarians:
            hr_sc  = p.get("hr_score",0)
            streak = p.get("sub_streak",50)
            proj   = p.get("fd_proj",0)
            own    = p.get("ownership",0)
            sal    = p.get("fd_salary",0)
            # Build the sell
            reasons = []
            if hr_sc >= 50: reasons.append(f"HR score {hr_sc:.0f}/100")
            if streak >= 65: reasons.append("hot streak")
            if p.get("fd_ceiling",0) >= 25: reasons.append(f"ceiling {p['fd_ceiling']:.0f}pts")
            opp_sp = p.get("sp_name","")
            sp_vuln = p.get("sub_pitcher",50)
            if sp_vuln >= 58: reasons.append(f"opp SP vulnerable ({opp_sp})")
            implied = p.get("implied_total",0)
            if implied >= 4.8: reasons.append(f"{implied:.1f}R implied")
            c_rows.append({
                "Player":   p["name"],
                "Team":     p.get("team",""),
                "Pos":      p.get("fd_position",""),
                "Salary":   f"${sal:,}",
                "Proj":     f"{proj:.1f}",
                "Ceil":     f"{p.get('fd_ceiling',0):.1f}",
                "Own%":     f"{own:.0f}%",
                "HR Sc":    int(hr_sc),
                "Opp SP":   opp_sp[:14],
                "🎯 Sell":  ", ".join(reasons) if reasons else "solid matchup, low owned",
            })

        df_c = pd.DataFrame(c_rows)
        def _cown_c(v):
            try:
                f = float(str(v).replace("%",""))
                if f <= 8:  return "color:#00ff88;font-weight:bold"
                if f <= 15: return "color:#66dd88"
                return ""
            except: return ""
        def _cproj_c(v):
            try:
                f = float(v)
                if f >= 18: return "color:#00ff88;font-weight:bold"
                if f >= 13: return "color:#ffdd00"
                return ""
            except: return ""
        if not df_c.empty and "Proj" in df_c.columns:
            st.dataframe(
                df_c.style.map(_cown_c, subset=["Own%"]).map(_cproj_c, subset=["Proj","Ceil"]),
                use_container_width=True, hide_index=True
            )
    else:
        st.info("No strong contrarian plays identified on this slate.")


def _build_all_singleton_lineup(fd_plays: List[Dict], sp_salary_data: Dict,
                                 sp_proj_list: List[Dict]) -> Optional[Dict]:
    """
    Benchmark lineup: Top 8 hitters by FD proj, paired with ace SP.
    Uses salary upgrade pass to target within $1,000 of the $35K cap.
    """
    SAL_CAP    = 35000
    SAL_TARGET = 34000   # leave at most $1,000 on the table

    eligible = sorted(
        [p for p in fd_plays
         if p.get("fd_salary",0) > 0
         and not p.get("is_postponed",False)
         and not p.get("fd_injured",False)],
        key=lambda x: x.get("fd_proj",0), reverse=True
    )
    if not eligible:
        return None

    # Use ace SP — benchmark wants highest ceiling
    sp_with_sal = [s for s in sp_proj_list if s.get("salary",0) > 0]
    if not sp_with_sal:
        return None
    ace_sp   = sorted(sp_with_sal, key=lambda x: x.get("fd_sp_proj", x.get("value",0)), reverse=True)
    sp_name  = ace_sp[0]["name"]
    sp_entry = _fd_name_match(sp_name, sp_salary_data)
    if not sp_entry:
        for s in ace_sp:
            sp_entry = _fd_name_match(s["name"], sp_salary_data)
            if sp_entry:
                sp_name = s["name"]
                break
    if not sp_entry:
        return None

    sp_salary     = sp_entry["salary"]
    hitter_budget = SAL_CAP - sp_salary

    # Greedy pick
    chosen, used_sal = [], 0
    for p in eligible:
        if len(chosen) >= 8:
            break
        if used_sal + p["fd_salary"] <= hitter_budget:
            chosen.append(p)
            used_sal += p["fd_salary"]

    if len(chosen) < 8:
        return None

    # Salary upgrade: replace cheapest with better option within budget
    for _ in range(10):
        if used_sal >= SAL_TARGET:
            break
        budget_left = hitter_budget - used_sal
        cheapest    = min(chosen, key=lambda x: x["fd_salary"])
        upgrades = sorted(
            [p for p in eligible
             if p not in chosen
             and p["fd_salary"] > cheapest["fd_salary"]
             and p["fd_salary"] - cheapest["fd_salary"] <= budget_left],
            key=lambda x: x.get("fd_proj",0), reverse=True
        )
        if upgrades:
            best  = upgrades[0]
            chosen = [best if p is cheapest else p for p in chosen]
            used_sal = sum(p["fd_salary"] for p in chosen)
        else:
            break

    assigned = _assign_fd_slots(chosen)
    if assigned is None:
        slots = ["C/1B","2B","3B","SS","OF","OF","OF","UTIL"]
        assigned = [{**h, "slot": slots[min(i,7)]} for i,h in enumerate(chosen[:8])]

    sp_slot = {"name": sp_name, "salary": sp_salary, "slot": "P",
               "team": sp_entry.get("team",""), "fd_proj": 0,
               "fd_ceiling": 0, "fd_floor": 0, "ownership": 0,
               "streak_label": "", "fd_position": "P"}

    all_players = [sp_slot] + assigned
    final_sal   = sum(p.get("fd_salary", p.get("salary",0)) for p in all_players)
    team_counts = {}
    for p in assigned:
        t = p.get("team","")
        if t: team_counts[t] = team_counts.get(t,0)+1

    return {
        "players": all_players, "sp_name": sp_name, "sp_salary": sp_salary,
        "sp_upgraded": False,
        "primary_team": max(team_counts, key=team_counts.get) if team_counts else "",
        "secondary_team": "", "singleton": "", "singleton_team": "",
        "total_salary": final_sal, "salary_remaining": SAL_CAP - final_sal,
        "total_proj": round(sum(p.get("fd_proj",0) for p in assigned), 1),
        "total_ceiling": round(sum(p.get("fd_ceiling",0) for p in assigned), 1),
        "structure": "ALL-SINGLETON", "lineup_num": 0, "is_benchmark": True,
    }

def detect_slate_shape(
    stack_teams: List[tuple],   # [(team, score), ...] sorted best→worst
    fd_plays: List[Dict],
    n_lineups: int,
    site: str = "FD",           # "FD" or "DK"
) -> Dict:
    """
    Analyze slate concentration to determine optimal lineup shape allocation.

    Returns:
        {
          "mode": "heavy_stack" | "balanced" | "spread",
          "concentration": float,   # top-2 avg / all-teams avg
          "slate_size": int,         # number of viable stacking teams
          "shapes": {shape_name: count},  # how many lineups of each shape
          "rationale": str,
        }

    FD shapes:  4-4, 4-3-1, 4-2-2, 4-1-1-1-1
    DK shapes:  5-3, 5-2-1, 5-1-1-1

    Rules:
    - Heavy stack (concentration > 1.35): 70-80% double-stack, rest singleton diversity
    - Balanced (1.15–1.35): standard 35/35/20/10 split
    - Spread (<1.15): shift toward singletons — high singleton quality may outperform
    - Small slates (≤5 viable teams): evaluate singleton proj vs stack proj, pick best
    """
    if not stack_teams:
        return {"mode": "spread", "shapes": {}, "slate_size": 0,
                "concentration": 1.0, "rationale": "No viable stack teams found"}

    slate_size  = len(stack_teams)
    scores      = [s for _, s in stack_teams]
    all_avg     = sum(scores) / len(scores) if scores else 1.0
    top2_avg    = sum(scores[:2]) / 2 if len(scores) >= 2 else scores[0]
    concentration = top2_avg / all_avg if all_avg > 0 else 1.0

    stacked_lu = n_lineups - 1  # last always singleton benchmark

    if site == "FD":
        # FD shapes: 4-4 (double-stack heavy), 4-3-1, 4-2-2, 4-1-1-1-1, singleton
        if concentration >= 1.35 or slate_size <= 4:
            # Heavy stack: 2 clear games dominate → maximize correlation
            mode = "heavy_stack"
            rationale = f"Concentration {concentration:.2f} — 2 games dominate the slate. Maximize double-stack."
            alloc_44    = round(stacked_lu * 0.45)
            alloc_431   = round(stacked_lu * 0.35)
            alloc_422   = round(stacked_lu * 0.15)
            alloc_4111  = stacked_lu - alloc_44 - alloc_431 - alloc_422
        elif concentration >= 1.15:
            # Balanced: standard mixed construction
            mode = "balanced"
            rationale = f"Concentration {concentration:.2f} — balanced slate. Mix double-stack with diversification."
            alloc_44    = round(stacked_lu * 0.35)
            alloc_431   = round(stacked_lu * 0.35)
            alloc_422   = round(stacked_lu * 0.20)
            alloc_4111  = stacked_lu - alloc_44 - alloc_431 - alloc_422
        else:
            # Spread: value is distributed — singletons can compete
            mode = "spread"
            rationale = f"Concentration {concentration:.2f} — spread slate. Shift to singleton diversity."
            alloc_44    = round(stacked_lu * 0.20)
            alloc_431   = round(stacked_lu * 0.30)
            alloc_422   = round(stacked_lu * 0.25)
            alloc_4111  = stacked_lu - alloc_44 - alloc_431 - alloc_422

        shapes = {
            "4-4":     max(0, alloc_44),
            "4-3-1":   max(0, alloc_431),
            "4-2-2":   max(0, alloc_422),
            "4-1-1-1-1": max(0, alloc_4111),
        }

    else:  # DK
        # DK shapes: 5-3 (heavy), 5-2-1, 5-1-1-1, singleton
        if concentration >= 1.35 or slate_size <= 4:
            mode = "heavy_stack"
            rationale = f"Concentration {concentration:.2f} — 2 games dominate. Maximize 5-3 correlation."
            alloc_53    = round(stacked_lu * 0.45)
            alloc_521   = round(stacked_lu * 0.35)
            alloc_5111  = stacked_lu - alloc_53 - alloc_521
        elif concentration >= 1.15:
            mode = "balanced"
            rationale = f"Concentration {concentration:.2f} — balanced. Mix 5-3 and 5-2-1."
            alloc_53    = round(stacked_lu * 0.35)
            alloc_521   = round(stacked_lu * 0.35)
            alloc_5111  = stacked_lu - alloc_53 - alloc_521
        else:
            mode = "spread"
            rationale = f"Concentration {concentration:.2f} — spread slate. 5-2-1 dominant, more singletons."
            alloc_53    = round(stacked_lu * 0.20)
            alloc_521   = round(stacked_lu * 0.40)
            alloc_5111  = stacked_lu - alloc_53 - alloc_521

        shapes = {
            "5-3":     max(0, alloc_53),
            "5-2-1":   max(0, alloc_521),
            "5-1-1-1": max(0, alloc_5111),
        }

    return {
        "mode":          mode,
        "concentration": round(concentration, 3),
        "slate_size":    slate_size,
        "shapes":        shapes,
        "rationale":     rationale,
    }


def _build_fd_portfolio(fd_plays: List[Dict], sp_salary_data: Dict,
                         n_lineups: int, max_exposure: float,
                         min_salary: int, sp_proj_list: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Build N GPP lineups with structured stack exposure + singleton diversity.

    Architecture (for 22 lineups):
      Lineups 1-12:  Stack A primary (55%) — rotate Stack B/C secondary, rotate singletons
      Lineups 13-19: Stack B primary (32%) — rotate Stack A/C secondary, rotate singletons
      Lineups 20-21: Stack C primary (9%)  — Stack A/B secondary, singletons
      Lineup 22:     ALL-SINGLETON benchmark — pure model, no correlation

    Stack D included dynamically: if game_score[3] within 1.0 of game_score[0],
    add 2 Stack D lineups by redistributing from Stack C allocation.

    Singleton diversity: once top stacks are locked, each lineup gets a unique
    singleton from the HR+streak ranked pool. No singleton appears > 30% of lineups.
    """
    SAL_CAP   = 35000
    TOTAL_LU  = n_lineups

    if not fd_plays or not sp_salary_data:
        return [], {}

    # ── SP pool (up to 4 unique SPs, rotated across portfolio) ───────────────
    sp_with_sal = [s for s in sp_proj_list if s.get("salary",0) > 0]
    sp_ace      = sorted(sp_with_sal, key=lambda x: x.get("fd_sp_proj",0), reverse=True)[:3]
    sp_value    = sorted(sp_with_sal, key=lambda x: x.get("value",0), reverse=True)[:3]

    sp_pool = []
    seen_sp = set()
    for a, v in zip(sp_ace + sp_ace, sp_value + sp_value):
        for sp in [a, v]:
            if sp["name"] not in seen_sp:
                sp_pool.append(sp["name"])
                seen_sp.add(sp["name"])
    if not sp_pool:
        return [], {"error": "No SPs with salary found"}

    # SP exposure caps: no SP in more than 45% of stacked lineups
    sp_cap_lineups = max(1, int((TOTAL_LU - 1) * 0.45))

    # ── Stack team identification — use team-level scores (NOT game scores) ─────
    # Team stack score incorporates: implied total + SP vulnerability + HR potential
    # + hot streaks. This is superior to game O/U alone.
    raw_plays   = st.session_state.get("plays", fd_plays)
    team_ranks  = get_ranked_team_stacks(raw_plays if raw_plays else fd_plays, min_players=4)

    def team_has_pool(team):
        return len([p for p in fd_plays
                    if p.get("team") == team
                    and p.get("fd_salary",0) > 0
                    and not p.get("is_postponed",False)]) >= 4

    stack_teams = [(sd["team"], sd["stack_score"]) for sd in team_ranks if team_has_pool(sd["team"])]

    if len(stack_teams) < 2:
        return [], {"error": f"Not enough teams with 4+ salary-matched players: {[t for t,_ in stack_teams]}"}

    stack_a = stack_teams[0][0]
    stack_b = stack_teams[1][0] if len(stack_teams) > 1 else stack_a
    stack_c = stack_teams[2][0] if len(stack_teams) > 2 else stack_b
    stack_d = stack_teams[3][0] if len(stack_teams) > 3 else None

    # ── Slate-aware shape allocation ─────────────────────────────────────────
    # detect_slate_shape() analyzes stack concentration to determine
    # optimal 4-4 / 4-3-1 / 4-2-2 / 4-1-1-1-1 split dynamically.
    # User preference: 4-4 and 4-3-1 dominate (70-80%) except on spread slates.
    stacked_count = TOTAL_LU - 1  # last lineup is always singleton benchmark
    slate_analysis = detect_slate_shape(stack_teams, fd_plays, TOTAL_LU, site="FD")
    shapes = slate_analysis["shapes"]

    alloc_44   = shapes.get("4-4",     0)
    alloc_431  = shapes.get("4-3-1",   0)
    alloc_422  = shapes.get("4-2-2",   0)
    alloc_4111 = shapes.get("4-1-1-1-1", 0)

    # Normalize to stacked_count (rounding may drift)
    total_alloc = alloc_44 + alloc_431 + alloc_422 + alloc_4111
    if total_alloc != stacked_count:
        alloc_431 += stacked_count - total_alloc   # absorb rounding into 4-3-1

    # Store analysis for display
    st.session_state["fd_slate_analysis"] = slate_analysis

    # ── Build schedule: (primary, secondary, shape) ───────────────────────────
    # For each shape, secondary count = how many batters from secondary team
    # 4-4: sec=4, 4-3-1: sec=3, 4-2-2: sec=2 (two mini-stacks), 4-1-1-1-1: sec=1
    schedule = []
    secondary_options = [stack_b, stack_c, stack_a, stack_b, stack_d or stack_c,
                         stack_c, stack_a, stack_b]

    def pick_secondary(primary, idx):
        opts = [t for t in secondary_options + [stack_b, stack_c, stack_d or stack_c]
                if t and t != primary]
        return opts[idx % len(opts)] if opts else stack_b

    def pick_tertiary(primary, secondary, idx):
        """For 4-2-2 and 4-1-1-1-1: pick a third team."""
        opts = [t for _, t in stack_teams if t not in (primary, secondary)]
        return opts[idx % len(opts)] if opts else secondary

    sec_idx = 0
    for i in range(alloc_44):
        schedule.append((stack_a, pick_secondary(stack_a, sec_idx), "4-4", 4))
        sec_idx += 1
    for i in range(alloc_431):
        # Alternate primary between stack_a and stack_b for variety
        primary = stack_a if i % 2 == 0 else stack_b
        secondary = pick_secondary(primary, sec_idx)
        schedule.append((primary, secondary, "4-3-1", 3))
        sec_idx += 1
    for i in range(alloc_422):
        primary = stack_a if i % 2 == 0 else stack_b
        secondary = pick_secondary(primary, sec_idx)
        schedule.append((primary, secondary, "4-2-2", 2))
        sec_idx += 1
    for i in range(alloc_4111):
        primary = stack_a
        secondary = pick_secondary(primary, sec_idx)
        schedule.append((primary, secondary, "4-1-1-1-1", 1))
        sec_idx += 1

    # ── Singleton pool (ranked by HR+streak blend) ────────────────────────────
    all_eligible = sorted(
        [p for p in fd_plays
         if p.get("fd_salary",0) > 0
         and not p.get("is_postponed",False)
         and not p.get("fd_injured",False)],
        key=_singleton_score, reverse=True
    )
    singleton_cap = max(1, int(stacked_count * 0.30))  # no singleton > 30% of lineups
    singleton_usage = {}  # name → count used

    def next_singleton(primary_team, secondary_team, used_in_this_lineup):
        """Pick highest-scored singleton not over-exposed, from a different game."""
        stack_teams_set = {primary_team, secondary_team}
        for cand in all_eligible:
            name = cand["name"]
            if name in used_in_this_lineup:
                continue
            if cand.get("team","") in stack_teams_set:
                continue
            if singleton_usage.get(name, 0) >= singleton_cap:
                continue
            return cand
        # Fallback: relax team constraint
        for cand in all_eligible:
            name = cand["name"]
            if name in used_in_this_lineup:
                continue
            if singleton_usage.get(name, 0) >= singleton_cap:
                continue
            return cand
        return None

    # ── Build stacked lineups ─────────────────────────────────────────────────
    lineups         = []
    player_exposure = {}
    sp_usage        = {}

    for sched_idx, (primary_team, secondary_team, shape, sec_count) in enumerate(schedule):
        # Pick SP (rotate, respect cap)
        sp_name = None
        for sp_cand in sp_pool:
            if sp_usage.get(sp_cand, 0) < sp_cap_lineups:
                sp_name = sp_cand
                break
        if not sp_name:
            sp_name = sp_pool[sched_idx % len(sp_pool)]

        lu = _build_fd_gpp_lineup(
            fd_plays=fd_plays,
            primary_team=primary_team,
            secondary_team=secondary_team,
            sp_name=sp_name,
            sp_salary_data=sp_salary_data,
            lineup_num=sched_idx % 6,
            ace_sp_name="",
        )

        if lu is None:
            # Retry with relaxed secondary — try all stacks
            for alt_sec in [stack_a, stack_b, stack_c, stack_d] if stack_d else [stack_a, stack_b, stack_c]:
                if alt_sec == primary_team or alt_sec is None:
                    continue
                lu = _build_fd_gpp_lineup(
                    fd_plays=fd_plays,
                    primary_team=primary_team,
                    secondary_team=alt_sec,
                    sp_name=sp_name,
                    sp_salary_data=sp_salary_data,
                    lineup_num=sched_idx % 6,
                    ace_sp_name="",
                )
                if lu is not None:
                    break

        # If still None, try rotating through all available SPs
        if lu is None:
            for alt_sp in sp_pool[:4]:
                if alt_sp == sp_name:
                    continue
                for alt_sec in [stack_b, stack_c, stack_a]:
                    if alt_sec == primary_team:
                        continue
                    lu = _build_fd_gpp_lineup(
                        fd_plays=fd_plays,
                        primary_team=primary_team,
                        secondary_team=alt_sec,
                        sp_name=alt_sp,
                        sp_salary_data=sp_salary_data,
                        lineup_num=sched_idx % 6,
                        ace_sp_name="",
                    )
                    if lu is not None:
                        break
                if lu is not None:
                    break

        if lu is None:
            continue

        # Salary floor: require near-cap spending (within $1,500 of cap)
        # Relax by $1000 for later lineups to maximise count on small slates
        if len(lineups) < alloc_44:
            effective_min = min_salary
        else:
            effective_min = min_salary - 1000
        if lu["total_salary"] < effective_min:
            continue

        # ── Exposure: soft guidance, never block lineups ─────────────────────
        # On small slates (3-6 games), hard exposure caps prevent reaching
        # the target lineup count. Let the portfolio build and report exposure —
        # the user can decide if they want to rebalance manually.
        # We still try lineup variation to diversify where possible.
        if sched_idx % 3 == 2 and sched_idx % 6 < 5:
            # Every 3rd lineup: try alternate variation for diversity
            alt_lu = _build_fd_gpp_lineup(
                fd_plays=fd_plays,
                primary_team=primary_team,
                secondary_team=secondary_team,
                sp_name=sp_name,
                sp_salary_data=sp_salary_data,
                lineup_num=(sched_idx % 6) + 1,
                ace_sp_name="",
            )
            if alt_lu and alt_lu["total_salary"] >= effective_min:
                # Only use alt if it produces different players
                orig_names = {p.get("name","") for p in lu["players"]}
                alt_names  = {p.get("name","") for p in alt_lu["players"]}
                if len(orig_names - alt_names) >= 2:   # at least 2 different players
                    lu = alt_lu

        # Track names already in this lineup (for singleton dedup)
        names_in_lu = set(p.get("name","") for p in lu["players"])

        # Pick and inject singleton (replace worst secondary player)
        sing_cand = next_singleton(primary_team, secondary_team, names_in_lu)
        if sing_cand:
            sing_name = sing_cand["name"]
            # Check if singleton already in lineup (via stack)
            if sing_name not in names_in_lu:
                # Find secondary player with lowest fd_proj to swap
                sec_players = [p for p in lu["players"]
                               if p.get("team","") == secondary_team
                               and p.get("slot","") not in ("P",)]
                if sec_players:
                    worst_sec = min(sec_players, key=lambda x: x.get("fd_proj",0))
                    # Only swap if singleton is a meaningful upgrade or different game
                    sing_team = sing_cand.get("team","")
                    if (sing_team not in {primary_team, secondary_team} and
                            _singleton_score(sing_cand) > _singleton_score(worst_sec) * 0.7):
                        lu["players"] = [
                            {**sing_cand, "slot": worst_sec.get("slot","OF"),
                             "is_singleton": True}
                            if p is worst_sec else p
                            for p in lu["players"]
                        ]
                        lu["singleton"]      = sing_name
                        lu["singleton_team"] = sing_team
                        lu["structure"]      = shape   # e.g. "4-3-1", "4-4", "4-2-2"
                        singleton_usage[sing_name] = singleton_usage.get(sing_name, 0) + 1

        # Track exposure
        for p in lu["players"]:
            n = p.get("name","")
            if n:
                player_exposure[n] = player_exposure.get(n,0) + 1
        sp_usage[sp_name] = sp_usage.get(sp_name, 0) + 1

        lu["lineup_index"] = len(lineups) + 1
        lu["sp_tier"]      = "ACE" if sp_name in [s["name"] for s in sp_ace[:1]] else "VALUE"
        lu["primary_stack"] = primary_team
        lineups.append(lu)

    # ── Lineup 22 (N): All-singleton benchmark ────────────────────────────────
    benchmark = _build_all_singleton_lineup(fd_plays, sp_salary_data, sp_proj_list)
    if benchmark:
        benchmark["lineup_index"] = len(lineups) + 1
        benchmark["sp_tier"]      = "VALUE"
        for p in benchmark["players"]:
            n = p.get("name","")
            if n: player_exposure[n] = player_exposure.get(n,0) + 1
        lineups.append(benchmark)

    # ── Exposure report ───────────────────────────────────────────────────────
    n_built = len(lineups)
    sa = st.session_state.get("fd_slate_analysis", {})
    shapes_summary = " · ".join(f"{k}:{v}" for k, v in sa.get("shapes", {}).items()) if sa else "—"
    exposure_report = {
        "_stack_plan": {
            "A": f"{stack_a} (primary)",
            "B": f"{stack_b} (secondary)",
            "C": f"{stack_c} (coverage)",
            "D": f"{stack_d} (coverage)" if stack_d else "not included",
            "benchmark": "1 all-singleton lineup",
            "stack_d_reason": (
                f"Stack D ({stack_d}) included" if stack_d else "Stack D not included"
            ),
            "shapes": shapes_summary,
            "mode": sa.get("mode", "balanced"),
        }
    }
    for name, count in sorted(player_exposure.items(), key=lambda x: -x[1]):
        pct = count / max(1, n_built) * 100
        exposure_report[name] = {"count": count, "pct": round(pct,1),
                                  "over": pct > max_exposure * 100}

    return lineups, exposure_report


def display_fd_portfolio_builder(plays: List[Dict]):
    """
    Tab 9 — FD Lineup Portfolio Builder
    Bulk GPP entry — auto-builds N lineups with systematic variation.
    Exports FanDuel bulk upload CSV.
    """
    st.markdown("""
    <style>
    .port-header {
        background: linear-gradient(135deg, #0f0f23 0%, #1a0a2e 50%, #2d0a4e 100%);
        border: 1px solid #9b59b6;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }
    .port-title { color: #9b59b6; font-size: 28px; font-weight: 800; }
    .port-sub { color: #a0a0b0; font-size: 13px; margin-top: 4px; }
    .lineup-row {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 8px;
        padding: 8px 14px;
        margin-bottom: 4px;
        font-size: 13px;
    }
    .exposure-bar { background:#222; border-radius:4px; height:6px; overflow:hidden; margin-top:2px; }
    .exposure-fill-ok   { background:#00cc66; height:6px; }
    .exposure-fill-warn { background:#ffcc00; height:6px; }
    .exposure-fill-over { background:#ff4444; height:6px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="port-header">
        <div class="port-title">🚀 FD Lineup Portfolio Builder</div>
        <div class="port-sub">Auto-builds N GPP lineups with systematic SP/stack/singleton variation — exports FanDuel bulk upload CSV.</div>
    </div>
    """, unsafe_allow_html=True)

    fd_plays    = st.session_state.get("fd_plays_current", [])
    sp_sal_data = st.session_state.get("fd_sp_salary_data", {})
    sp_proj     = st.session_state.get("fd_sp_projections", [])
    slate_data  = st.session_state.get("fd_slate_data", {})

    if not fd_plays:
        st.warning("⚠️ Load the FanDuel CSV in the **FD Command Center** tab first.")
        return

    has_salaries = any(p.get("fd_salary",0) > 0 for p in fd_plays)
    if not has_salaries:
        st.warning("No salary data found — upload the FanDuel CSV in Command Center.")
        return

    # ── CONTROLS ─────────────────────────────────────────────────────────────
    st.subheader("⚙️ Portfolio Settings")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_lineups = st.select_slider(
            "Lineups to build", options=[3,5,10,15,22,25,30],
            value=22, key="port_n_lineups"
        )
    with col2:
        max_exp = st.slider(
            "Max player exposure %", 20, 80, 60, 5,
            key="port_max_exp",
            help="No player appears in more than this % of lineups"
        )
    with col3:
        min_sal = st.slider(
            "Min salary", 32000, 35000, 34000, 500,
            key="port_min_sal",
            help="Discard lineups below this salary total"
        )
    with col4:
        st.write("")
        st.write("")
        show_exp = st.checkbox("Show exposure report", value=True, key="port_show_exp")

    # ── SP PREVIEW ────────────────────────────────────────────────────────────
    if sp_proj:
        with st.expander("📊 SP Pool Preview (auto-selected)"):
            sp_with_sal = [s for s in sp_proj if s.get("salary",0) > 0]
            sp_ace   = sorted(sp_with_sal, key=lambda x: x.get("fd_sp_proj",0), reverse=True)[:3]
            sp_value = sorted(sp_with_sal, key=lambda x: x.get("value",0), reverse=True)[:3]
            col_a, col_v = st.columns(2)
            with col_a:
                st.markdown("**🎯 Ace tier (highest proj)**")
                for s in sp_ace:
                    st.write(f"  {s['name']} — {s['fd_sp_proj']:.1f} proj | ${s['salary']:,}")
            with col_v:
                st.markdown("**💰 Value tier (best $/proj)**")
                for s in sp_value:
                    st.write(f"  {s['name']} — {s['value']:.2f}x | ${s['salary']:,}")

    st.markdown("---")

    # ── BUILD BUTTON ──────────────────────────────────────────────────────────
    build_col, status_col = st.columns([1, 3])
    with build_col:
        build_btn = st.button(
            f"🚀 Build {n_lineups} Lineups",
            type="primary",
            key="port_build_btn",
        )
    with status_col:
        matched = sum(1 for p in fd_plays if p.get("salary_matched"))
        st.caption(f"✅ {matched} salary-matched players | {len(sp_sal_data)} pitchers | "
                   f"Cap: $35,000 | Min: ${min_sal:,} | Max exposure: {max_exp}%")

    if build_btn:
        with st.spinner(f"Building {n_lineups} lineups..."):
            lineups, exp_report = _build_fd_portfolio(
                fd_plays=fd_plays,
                sp_salary_data=sp_sal_data,
                n_lineups=n_lineups,
                max_exposure=max_exp / 100,
                min_salary=min_sal,
                sp_proj_list=sp_proj,
            )

        if not lineups:
            st.error(
                f"❌ Could not build lineups. Check:\n"
                f"• At least 2 teams need 4+ salary-matched players\n"
                f"• Pitchers need salary data\n"
                f"• Error: {exp_report.get('error','unknown')}"
            )
        else:
            st.session_state["port_lineups"]    = lineups
            st.session_state["port_exp_report"] = exp_report
            st.success(f"✅ Built {len(lineups)} lineups successfully")

    # ── DISPLAY RESULTS ───────────────────────────────────────────────────────
    lineups    = st.session_state.get("port_lineups", [])
    exp_report = st.session_state.get("port_exp_report", {})

    if lineups:
        st.markdown("---")

        # ── Summary stats ─────────────────────────────────────────────────────
        avg_sal  = sum(lu["total_salary"] for lu in lineups) / len(lineups)
        avg_proj = sum(lu["total_proj"] for lu in lineups) / len(lineups)
        avg_ceil = sum(lu["total_ceiling"] for lu in lineups) / len(lineups)
        min_s    = min(lu["total_salary"] for lu in lineups)
        max_s    = max(lu["total_salary"] for lu in lineups)

        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        col_a.metric("Lineups Built",  len(lineups))
        col_b.metric("Avg Salary",     f"${avg_sal:,.0f}")
        col_c.metric("Sal Range",      f"${min_s:,}–${max_s:,}")
        col_d.metric("Avg Proj",       f"{avg_proj:.1f}")
        col_e.metric("Avg Ceiling",    f"{avg_ceil:.1f}")

        # ── Export button (top) ───────────────────────────────────────────────
        st.subheader("📥 Export")
        st.caption(
            "FanDuel bulk upload format. Download → go to FanDuel contest → "
            "click 'Import Lineups' → upload this CSV."
        )

        # Build FD bulk import CSV
        import io as _io_port
        buf = _io_port.StringIO()
        # FD format header
        # FD bulk import format: OF slots must map correctly
        # Internal slots: OF1, OF2, OF3 → CSV columns: OF, OF, OF (same name, different positions)
        buf.write("entry-id,contest-id,contest-name,P,C/1B,2B,3B,SS,OF,OF,OF,UTIL\n")
        fd_slot_order = ["P","C/1B","2B","3B","SS","OF1","OF2","OF3","UTIL"]
        for lu in lineups:
            player_by_slot = {p.get("slot",""):p for p in lu["players"]}
            row_names = []
            for slot in fd_slot_order:
                p = player_by_slot.get(slot)
                row_names.append(p["name"] if p else "")
            buf.write(f",,Propex GPP,{','.join(row_names)}\n")

        csv_str = buf.getvalue()
        st.download_button(
            "📥 Download FanDuel Bulk Import CSV",
            data=csv_str,
            file_name=f"propex_fd_lineups_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="port_dl_csv",
        )

        # ── Lineup grid ───────────────────────────────────────────────────────
        st.subheader(f"📋 All {len(lineups)} Lineups")
        st.caption("★ = primary stack  ◆ = secondary  💎 = singleton")

        # Show stack plan from exposure report
        plan = exp_report.get("_stack_plan",{})
        if plan:
            with st.expander("📊 Stack Allocation Plan"):
                # Show slate analysis
                sa = st.session_state.get("fd_slate_analysis", {})
                if sa:
                    mode_color = {"heavy_stack": "#00ff88", "balanced": "#ffdd00", "spread": "#ff8800"}.get(sa.get("mode",""), "#aaa")
                    st.markdown(
                        f"<span style='color:{mode_color};font-weight:700'>Slate Mode: {sa.get('mode','').upper().replace('_',' ')}</span>"
                        f" — Concentration: {sa.get('concentration',0):.2f} — {sa.get('slate_size',0)} viable teams",
                        unsafe_allow_html=True
                    )
                    st.caption(sa.get("rationale",""))
                    shapes_display = sa.get("shapes", {})
                    shape_cols = st.columns(len(shapes_display) + 1)
                    for col, (sh, cnt) in zip(shape_cols, shapes_display.items()):
                        col.metric(sh, f"{cnt} lineups")
                    shape_cols[-1].metric("Benchmark", "1 lineup")
                else:
                    col_pa, col_pb, col_pc, col_pd = st.columns(4)
                    col_pa.metric("Stack A", plan.get("A","—"))
                    col_pb.metric("Stack B", plan.get("B","—"))
                    col_pc.metric("Stack C", plan.get("C","—"))
                    col_pd.metric("Stack D", plan.get("D","not included"))
                    st.caption(f"Stack D decision: {plan.get('stack_d_reason','—')}")
                    st.caption(f"Benchmark: {plan.get('benchmark','—')}")

        grid_rows = []
        for lu in lineups:
            is_bench = lu.get("is_benchmark", False)
            sp_tag   = "🧪" if is_bench else ("🎯" if lu.get("sp_tier","") == "ACE" else "💰")
            sing_name = lu.get("singleton","")
            struct    = lu.get("structure","4-4")
            label     = "🧪 BENCHMARK" if is_bench else struct

            grid_rows.append({
                "#":         lu["lineup_index"],
                "SP":        f"{sp_tag} {lu['sp_name']}",
                "Type":      label,
                "Primary":   lu.get("primary_stack", lu.get("primary_team","")),
                "Secondary": lu.get("secondary_team","—") if not is_bench else "ALL-SINGLETON",
                "Singleton": sing_name or ("N/A" if is_bench else "—"),
                "Proj":      f"{lu['total_proj']:.1f}",
                "Ceiling":   f"{lu['total_ceiling']:.1f}",
                "Salary":    f"${lu['total_salary']:,}",
                "Left":      f"${lu['salary_remaining']:,}",
            })

        grid_df = pd.DataFrame(grid_rows)

        def _gproj(v):
            try:
                f = float(v)
                if f >= 120: return "color:#00ff88;font-weight:bold"
                if f >= 100: return "color:#ffdd00"
                return ""
            except: return ""

        def _gsal(v):
            try:
                f = float(str(v).replace("$","").replace(",",""))
                if f >= 34500: return "color:#00ff88"
                if f >= 33000: return "color:#ffdd00"
                return "color:#ff8800"
            except: return ""

        st.dataframe(
            grid_df.style.map(_gproj, subset=["Proj","Ceiling"]).map(_gsal, subset=["Salary"])
                         .map(lambda v: "color:#9b59b6;font-style:italic" if "BENCHMARK" in str(v) else "", subset=["Type"]),
            use_container_width=True, hide_index=True, height=min(600, 55 + len(lineups)*35)
        )

        # ── Per-lineup expanders ──────────────────────────────────────────────
        with st.expander(f"🔍 View Individual Lineup Details ({len(lineups)} lineups)"):
            for lu in lineups:
                with st.expander(
                    f"Lineup #{lu['lineup_index']} [{lu.get('structure','4-4')}] — "
                    f"Proj {lu['total_proj']:.1f} | ${lu['total_salary']:,}",
                    expanded=False
                ):
                    lu_rows = []
                    for p in lu["players"]:
                        is_sp = p.get("slot","") == "P"
                        prim  = p.get("team","") == lu["primary_team"]
                        sing  = p.get("is_singleton",False)
                        tag   = "★" if prim and not is_sp else ("◆" if not is_sp and not sing else ("💎" if sing else ""))
                        lu_rows.append({
                            "Slot":    p.get("slot",""),
                            "":        tag,
                            "Player":  p["name"],
                            "Team":    p.get("team",""),
                            "Salary":  f"${p.get('fd_salary', p.get('salary',0)):,}",
                            "FD Proj": f"{p.get('fd_proj',0):.1f}" if not is_sp else "—",
                        })
                    st.dataframe(pd.DataFrame(lu_rows), use_container_width=True, hide_index=True)
                    # FD import string
                    slot_order_ = ["P","C/1B","2B","3B","SS","OF1","OF2","OF3","UTIL"]
                    pbs = {p.get("slot",""):p for p in lu["players"]}
                    names = ",".join(pbs.get(s,{}).get("name","") for s in slot_order_)
                    st.code(names, language=None)

        # ── Exposure report ───────────────────────────────────────────────────
        if show_exp and exp_report:
            st.markdown("---")
            st.subheader("📊 Player Exposure Report")
            st.caption(f"Max allowed: {max_exp}% | Red = over limit")

            exp_rows = []
            # Skip internal metadata keys (e.g. _stack_plan) — only process player entries
            player_entries = {k: v for k, v in exp_report.items()
                              if not k.startswith("_") and isinstance(v, dict) and "pct" in v}
            for name, data in sorted(player_entries.items(), key=lambda x: -x[1]["pct"]):
                pct  = data["pct"]
                cnt  = data["count"]
                over = data["over"]
                exp_rows.append({
                    "Player":   name,
                    "Count":    cnt,
                    "Exposure": f"{pct:.0f}%",
                    "Status":   "⚠️ OVER" if over else "✅ OK",
                })

            exp_df = pd.DataFrame(exp_rows)
            def _es(v):
                return "color:#ff4444;font-weight:bold" if "OVER" in str(v) else "color:#00cc66"
            st.dataframe(
                exp_df.style.map(_es, subset=["Status"]),
                use_container_width=True, hide_index=True, height=320
            )





# ═══════════════════════════════════════════════════════════════════════════════
# DRAFTKINGS DFS PIPELINE
# Roster: SP, SP, C, 1B, 2B, 3B, SS, OF, OF, OF  ($50,000 cap)
# DK hitter scoring: 1B=3, 2B=5, 3B=8, HR=10, RBI=2, R=2, BB=2, SB=5, HBP=2
# DK SP scoring: IP*2.25 + K*2 + W=4 + ER*-2 + H*-0.6 + BB*-0.6
# No flex slot. Two pitcher slots (both labeled "P" in roster position).
# RP with salary >= 7500 treated as usable starters; true relievers excluded.
# ═══════════════════════════════════════════════════════════════════════════════

# DK scoring constants (differs from FD)
DK_HITTER_SCORING = "1B=3 | 2B=5 | 3B=8 | HR=10 | RBI=2 | R=2 | BB=2 | SB=5 | HBP=2"
DK_SP_SCORING     = "IP×2.25 | K×2 | W=+4 | ER×-2 | H×-0.6 | BB×-0.6"
DK_SALARY_CAP     = 50000
DK_MIN_SALARY     = 47000   # floor to avoid salary dumps
DK_SP_MIN_SALARY  = 7500    # RPs below this are true relievers — never use


def _parse_dk_csv(uploaded_file) -> Dict:
    """
    Parse DraftKings salary CSV export.
    Columns: Position, Name + ID, Name, ID, Roster Position, Salary, Game Info, TeamAbbrev, AvgPointsPerGame
    Returns {"salary_data": {...}, "sp_salary_data": {...}, "slate_games": [...], "raw_df": df}
    Filters: RP with salary < DK_SP_MIN_SALARY excluded as true relievers.
    Multi-position hitters (e.g. "1B/OF", "2B/SS") stored with all eligible positions.
    """
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        return {"error": str(e)}

    # Normalise column names
    col_map = {
        "Position": "position", "Name": "name", "ID": "dk_id",
        "Roster Position": "roster_pos", "Salary": "salary",
        "Game Info": "game_info", "TeamAbbrev": "team", "AvgPointsPerGame": "fppg",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["salary"]   = pd.to_numeric(df.get("salary", 0), errors="coerce").fillna(0).astype(int)
    df["fppg"]     = pd.to_numeric(df.get("fppg", 0),   errors="coerce").fillna(0)
    df["name"]     = df["name"].astype(str).str.strip()
    df["team"]     = df.get("team", pd.Series([""] * len(df))).astype(str).str.strip().str.upper()
    df["position"] = df.get("position", pd.Series([""] * len(df))).astype(str).str.strip()
    df["roster_pos"] = df.get("roster_pos", df["position"]).astype(str).str.strip()
    df["game_info"]  = df.get("game_info", pd.Series([""] * len(df))).astype(str).str.strip()

    # ── Parse slate games from game_info strings ─────────────────────────────
    slate_games = []
    seen_games  = set()
    for gi in df["game_info"].dropna().unique():
        # Format: "LAD@HOU 05/06/2026 02:10PM ET"
        parts = gi.split()
        if parts and "@" in parts[0]:
            game_key = parts[0]
            if game_key not in seen_games:
                seen_games.add(game_key)
                teams = game_key.split("@")
                if len(teams) == 2:
                    slate_games.append({"away": teams[0], "home": teams[1], "game_key": game_key, "game_info": gi})

    # ── Split pitchers vs hitters ─────────────────────────────────────────────
    # DK labels: SP = confirmed starter, RP = second-tier (Yamamoto, Strider etc)
    # Both use roster_pos = "P"
    # Rule: use if position in (SP, RP) AND salary >= DK_SP_MIN_SALARY
    pitcher_mask = df["position"].isin(["SP", "RP"]) & (df["salary"] >= DK_SP_MIN_SALARY)
    hitter_mask  = ~df["position"].isin(["SP", "RP"])

    sp_df     = df[pitcher_mask].copy()
    hitter_df = df[hitter_mask].copy()

    # ── Build SP salary dict ──────────────────────────────────────────────────
    sp_salary_data = {}
    for _, row in sp_df.iterrows():
        sp_salary_data[row["name"].lower()] = {
            "name":     row["name"],
            "salary":   int(row["salary"]),
            "fppg":     float(row["fppg"]),
            "team":     row["team"],
            "position": row["position"],   # "SP" or "RP" (DK label, both usable)
            "game_info": row.get("game_info", ""),
        }

    # ── Build hitter salary dict ──────────────────────────────────────────────
    salary_data = {}
    for _, row in hitter_df.iterrows():
        roster_pos = str(row.get("roster_pos", row["position"]))
        salary_data[row["name"].lower()] = {
            "name":       row["name"],
            "salary":     int(row["salary"]),
            "fppg":       float(row["fppg"]),
            "team":       row["team"],
            "roster_pos": roster_pos,      # may be "1B/OF", "2B/SS" etc
            "game_info":  row.get("game_info", ""),
        }

    return {
        "salary_data":    salary_data,
        "sp_salary_data": sp_salary_data,
        "slate_games":    slate_games,
        "raw_df":         df,
        "n_hitters":      len(salary_data),
        "n_pitchers":     len(sp_salary_data),
    }


def _dk_pos_eligible(roster_pos_str: str, slot: str) -> bool:
    """
    Check if a hitter's DK roster_pos is eligible for a given lineup slot.
    DK slots: C, 1B, 2B, 3B, SS, OF
    Multi-position strings: "1B/OF", "2B/SS", "3B/OF", "2B/3B", "OF/SS", etc.
    Empty/nan: treated as OF-eligible (safest fallback for unknown positions).
    """
    if not roster_pos_str or roster_pos_str in ("nan", "P", "SP", "RP"):
        # Unknown hitter position — allow OF slot as fallback
        return slot == "OF"
    eligible = [p.strip().upper() for p in str(roster_pos_str).split("/")]
    return slot.upper() in eligible


def _dk_name_match(name: str, salary_data: Dict) -> Dict:
    """Fuzzy name match for DK salary dict. Same logic as _fd_name_match."""
    if not salary_data:
        return {}
    name_lower = name.lower().strip()
    # Exact
    if name_lower in salary_data:
        return salary_data[name_lower]
    # Remove suffixes
    for suffix in [" jr.", " sr.", " ii", " iii", " iv"]:
        clean = name_lower.replace(suffix, "").strip()
        if clean in salary_data:
            return salary_data[clean]
    # Partial — first + last token match
    parts = name_lower.split()
    if len(parts) >= 2:
        first, last = parts[0], parts[-1]
        for k, v in salary_data.items():
            k_parts = k.split()
            if len(k_parts) >= 2 and k_parts[0] == first and k_parts[-1] == last:
                return v
    # Last name only fallback
    if parts:
        last = parts[-1]
        matches = [v for k, v in salary_data.items() if k.split()[-1] == last]
        if len(matches) == 1:
            return matches[0]
    return {}


def _compute_dk_hitter_proj(p: Dict, sal_entry: Dict) -> Dict:
    """
    Project DraftKings fantasy points for a hitter.
    DK scoring: 1B=3, 2B=5, 3B=8, HR=10, RBI=2, R=2, BB=2, SB=5, HBP=2
    Uses model xSLG, ISO, barrel%, implied total for projection.
    """
    xslg     = p.get("xslg", 0.380) or 0.380
    iso      = p.get("iso_proxy", 0.150) or 0.150
    implied  = p.get("implied_total", 4.5) or 4.5
    slot     = p.get("lineup_slot", 5) or 5
    k_rate   = p.get("k_rate", 0.22) or 0.22
    bb_rate  = p.get("bb_rate", 0.09) or 0.09
    barrel   = p.get("barrel_rate", 0.07) or 0.07
    hh_rate  = p.get("hard_hit_rate", 0.37) or 0.37
    fppg     = sal_entry.get("fppg", 0) or 0

    # Expected PA from lineup slot (DK games avg ~38 PA/game for full team)
    slot_pa_mult = {1: 1.15, 2: 1.10, 3: 1.05, 4: 1.02, 5: 0.98, 6: 0.95, 7: 0.90, 8: 0.85, 9: 0.80}
    pa_mult = slot_pa_mult.get(int(slot), 0.95)

    # Base expected AB ~3.8, adjust by implied (higher implied = more PA)
    base_pa  = 3.8 * pa_mult * (0.85 + implied / 30.0)
    contact_pa = base_pa * (1.0 - k_rate)

    # Hit distribution from xSLG + ISO
    # HR rate from barrel% (r=0.93); 2B/3B from ISO-HR
    hr_rate   = min(0.12, barrel * 0.55)
    xb_rate   = max(0, (iso - hr_rate * 1.4) / 2.0)  # split between 2B/3B
    h_rate    = max(0, xslg / 1.8 - hr_rate - xb_rate)  # singles residual
    bb_pa     = base_pa * bb_rate

    exp_1b  = contact_pa * h_rate
    exp_2b  = contact_pa * xb_rate * 0.85
    exp_3b  = contact_pa * xb_rate * 0.15
    exp_hr  = contact_pa * hr_rate
    exp_bb  = bb_pa
    exp_rbi = (exp_hr + exp_2b * 0.5 + exp_1b * 0.3) * (implied / 4.7)
    exp_r   = (implied / 9.0) * pa_mult
    exp_sb  = 0.04 * (hh_rate / 0.37)   # stolen base rate proxy

    proj = (
        exp_1b * 3.0 +
        exp_2b * 5.0 +
        exp_3b * 8.0 +
        exp_hr * 10.0 +
        exp_rbi * 2.0 +
        exp_r   * 2.0 +
        exp_bb  * 2.0 +
        exp_sb  * 5.0
    )

    # Blend with FPPG history (60/40) when available
    if fppg > 3.0:
        proj = proj * 0.55 + fppg * 0.45

    ceiling = proj * 1.55 + exp_hr * 3.0
    floor   = proj * 0.50
    value   = proj / (sal_entry.get("salary", 1) / 1000.0) if sal_entry.get("salary", 0) > 0 else 0.0

    return {
        "dk_proj":    round(proj, 1),
        "dk_ceiling": round(ceiling, 1),
        "dk_floor":   round(floor, 1),
        "dk_value":   round(value, 2),
    }


def _compute_dk_sp_proj(sp_data: Dict) -> Dict:
    """
    Project DK fantasy points for a starting pitcher.
    DK SP scoring: IP*2.25 + K*2 + W=4 + ER*-2 + H*-0.6 + BB*-0.6
    """
    fip      = sp_data.get("fip", 4.20) or 4.20
    k_rate   = sp_data.get("k_rate", 0.22) or 0.22
    salary   = sp_data.get("salary", 8000) or 8000
    fppg     = sp_data.get("fppg", 0) or 0
    opp_imp  = sp_data.get("opp_implied", 4.5) or 4.5

    # Projected IP: quality starters average 5.5-6.5 IP
    # FIP-based: elite (FIP<3.0)=6.5 IP, avg (4.20)=5.5 IP, bad (5.0+)=4.5 IP
    proj_ip  = max(4.0, min(7.5, 9.5 - fip * 0.77))
    proj_k   = proj_ip * (k_rate / 0.22) * 0.95  # ~0.95 K per IP at league avg K%
    proj_er  = proj_ip * (fip / 9.0)
    proj_h   = proj_ip * 0.90   # ~0.9 H/IP
    proj_bb  = proj_ip * (0.32 - k_rate * 0.20)  # BB/IP inversely related to K%
    win_prob = max(0.25, min(0.75, 0.55 - (fip - 3.80) * 0.06))

    proj = (
        proj_ip  * 2.25 +
        proj_k   * 2.0  +
        win_prob * 4.0  +
        proj_er  * -2.0 +
        proj_h   * -0.6 +
        proj_bb  * -0.6
    )
    proj = max(5.0, proj)

    ceiling = proj * 1.35 + 6.0   # extra K game ceiling
    floor   = max(0.0, proj * 0.4)
    value   = proj / (salary / 1000.0) if salary > 0 else 0.0

    # Blend with FPPG when available
    if fppg > 5.0:
        proj    = proj * 0.55 + fppg * 0.45
        ceiling = proj * 1.35 + 6.0

    return {
        "dk_sp_proj":    round(proj, 1),
        "dk_sp_ceiling": round(ceiling, 1),
        "dk_sp_floor":   round(floor, 1),
        "dk_sp_value":   round(value, 2),
        "proj_ip":       round(proj_ip, 1),
        "proj_k":        round(proj_k, 1),
        "win_prob":      round(win_prob, 2),
    }


def _build_dk_plays_with_salaries(plays: List[Dict], salary_data: Dict, sp_salary_data: Dict) -> List[Dict]:
    """
    Enrich model plays with DK salary data and DK-specific projections.
    Returns salary-matched plays only (unmatched excluded from DFS output).
    """
    dk_plays = []
    for p in plays:
        sal = _dk_name_match(p["name"], salary_data)
        if not sal:
            continue
        proj_data = _compute_dk_hitter_proj(p, sal)
        dk_entry = {
            **p,
            "dk_salary":   sal["salary"],
            "dk_position": sal.get("roster_pos", "OF"),
            "dk_proj":     proj_data["dk_proj"],
            "dk_ceiling":  proj_data["dk_ceiling"],
            "dk_floor":    proj_data["dk_floor"],
            "dk_value":    proj_data["dk_value"],
            "dk_fppg":     sal.get("fppg", 0),
            "dk_team":     sal.get("team", p["team"]),
        }
        # Estimated ownership (same formula as FD, DK field is larger so scale down)
        slot   = p.get("lineup_slot", 5)
        score  = p.get("score", 50)
        salary = sal["salary"]
        own_base = max(5, min(55,
            (score - 45) * 1.1 +
            (5.5 - slot) * 2.5 +
            ((6000 - salary) / 1000.0) * 2.0 +
            15.0
        ))
        dk_entry["dk_ownership"] = round(own_base, 1)
        dk_plays.append(dk_entry)

    dk_plays.sort(key=lambda x: x["dk_proj"], reverse=True)
    return dk_plays


def _build_dk_gpp_lineup(
    dk_plays: List[Dict],
    primary_team: str,
    secondary_team: str,
    sp1_name: str,
    sp2_name: str,
    sp_salary_data: Dict,
    lineup_num: int = 0,
) -> Dict | None:
    """
    Build one valid DK GPP lineup.
    Slots: SP, SP, C, 1B, 2B, 3B, SS, OF, OF, OF
    Cap: $50,000. Floor: DK_MIN_SALARY.
    Stack: 4 batters from primary_team, 2-3 from secondary_team, rest singletons.
    No true relievers (salary gate already applied in parser).
    """
    # ── Select pitchers ───────────────────────────────────────────────────────
    sp1_entry = _dk_name_match(sp1_name, sp_salary_data)
    sp2_entry = _dk_name_match(sp2_name, sp_salary_data)
    if not sp1_entry or not sp2_entry:
        return None
    sp1_sal = sp1_entry["salary"]
    sp2_sal = sp2_entry["salary"]

    # ── Filter hitter pool: primary (4) + secondary (2-3) + singletons ───────
    def _team_match(p, team):
        return (p.get("dk_team","") == team or p.get("team","") == team)

    primary_pool   = [p for p in dk_plays if _team_match(p, primary_team)]
    secondary_pool = [p for p in dk_plays if _team_match(p, secondary_team)]
    # Other pool: everyone not on primary or secondary (for singleton slot)
    other_pool     = [p for p in dk_plays if
                      not _team_match(p, primary_team) and not _team_match(p, secondary_team)]

    # If pools too small, supplement primary from entire dk_plays pool
    if len(primary_pool) < 3:
        primary_pool = sorted(dk_plays, key=lambda x: x["dk_proj"], reverse=True)
    if len(secondary_pool) < 2:
        secondary_pool = [p for p in dk_plays if not _team_match(p, primary_team)]

    # Sort by DK proj descending
    primary_pool.sort(key=lambda x: x["dk_proj"], reverse=True)
    secondary_pool.sort(key=lambda x: x["dk_proj"], reverse=True)
    other_pool.sort(key=lambda x: x["dk_proj"], reverse=True)

    # 8 hitter slots: C, 1B, 2B, 3B, SS, OF, OF, OF
    DK_HITTER_SLOTS = ["C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]

    def fill_slots(player_pool: List[Dict], slots: List[str], used_names: set,
                   budget: int, slot_counts: Dict[str, int]) -> tuple:
        """Greedy slot filler. Returns (filled_players, remaining_budget, updated_slot_counts)."""
        filled = []
        remaining = budget
        counts   = dict(slot_counts)
        for slot in slots:
            best = None
            best_proj = -1
            for p in player_pool:
                nm = p.get("name", "")
                if nm in used_names:
                    continue
                rp = str(p.get("dk_position", ""))
                if not _dk_pos_eligible(rp, slot):
                    continue
                if p["dk_salary"] > remaining:
                    continue
                if p["dk_proj"] > best_proj:
                    best_proj = p["dk_proj"]
                    best = p
            if best is None:
                return None, 0, {}
            filled.append({**best, "_slot": slot})
            used_names.add(best["name"])
            remaining -= best["dk_salary"]
            counts[slot] = counts.get(slot, 0) + 1
        return filled, remaining, counts

    budget = DK_SALARY_CAP - sp1_sal - sp2_sal
    if budget < 16000:
        return None

    used = set()
    slot_counts: Dict[str, int] = {}

    # ── DK optimal GPP shape: 5 primary + 2 secondary + 1 singleton ──────────
    # 5-2-1 is the dominant DK tournament shape. 5-man primary stack creates
    # correlated upside for large-field GPPs. Secondary gives a mini-stack.
    # Rotate WHICH 5 slots go to primary so portfolio lineups vary.
    primary_slot_sets = [
        ["C",  "1B", "SS", "OF", "OF"],   # 0 — C anchor
        ["1B", "2B", "SS", "OF", "OF"],   # 1 — middle infield heavy
        ["C",  "2B", "3B", "OF", "OF"],   # 2 — C + corner
        ["1B", "3B", "SS", "OF", "OF"],   # 3 — power corner
        ["C",  "1B", "2B", "OF", "OF"],   # 4 — top order skew
        ["1B", "SS", "3B", "OF", "OF"],   # 5 — balanced infield
    ]
    primary_slots = primary_slot_sets[lineup_num % len(primary_slot_sets)]
    prim_players, budget, slot_counts = fill_slots(primary_pool, primary_slots, used, budget, slot_counts)
    if prim_players is None:
        return None

    # Secondary: 2 slots — mini-stack from secondary game
    remaining_slots = [s for s in DK_HITTER_SLOTS if slot_counts.get(s, 0) == 0 or
                       (s == "OF" and slot_counts.get("OF", 0) < 3)]
    secondary_slots = remaining_slots[:2]
    sec_players, budget, slot_counts = fill_slots(secondary_pool, secondary_slots, used, budget, slot_counts)
    if sec_players is None:
        sec_players = []

    # Singleton: 1 remaining slot — prefer third team for GPP differentiation
    used_slots_count: Dict[str, int] = {}
    for p in (prim_players + sec_players):
        s = p["_slot"]
        used_slots_count[s] = used_slots_count.get(s, 0) + 1
    open_slots = []
    for s in DK_HITTER_SLOTS:
        needed = 1 if s != "OF" else 3
        placed  = used_slots_count.get(s, 0)
        open_slots.extend([s] * max(0, needed - placed))

    third_pool = [p for p in other_pool
                  if p.get("dk_team", p.get("team","")) not in (primary_team, secondary_team)]
    sing_pool  = third_pool + other_pool + primary_pool + secondary_pool
    sing_players, budget, _ = fill_slots(sing_pool, open_slots, used, budget, slot_counts)
    if sing_players is None:
        return None

    all_hitters = prim_players + sec_players + sing_players
    if len(all_hitters) != 8:
        return None

    total_salary = sp1_sal + sp2_sal + sum(p["dk_salary"] for p in all_hitters)
    if total_salary > DK_SALARY_CAP:
        return None

    # ── Salary upgrade loop — target within $2000 of $50K cap ────────────
    SAL_TARGET = DK_SALARY_CAP - 2000   # $48,000 target
    for _ in range(8):
        if total_salary >= SAL_TARGET:
            break
        budget_left = DK_SALARY_CAP - total_salary
        cheapest    = min(all_hitters, key=lambda x: x.get("dk_salary",0))
        c_sal       = cheapest.get("dk_salary", 0)
        c_slot      = cheapest.get("_slot","OF")
        # Find upgrade: same slot eligibility, higher salary within budget, better proj
        upgrades = sorted(
            [p for p in dk_plays
             if p not in all_hitters
             and _dk_pos_eligible(str(p.get("dk_position","")), c_slot)
             and p["dk_salary"] > c_sal
             and p["dk_salary"] - c_sal <= budget_left
             and p.get("dk_proj",0) > cheapest.get("dk_proj",0) * 0.80],
            key=lambda x: x.get("dk_proj",0), reverse=True
        )
        if upgrades:
            best = upgrades[0]
            all_hitters = [{**best, "_slot": c_slot} if p is cheapest else p
                           for p in all_hitters]
            total_salary = sp1_sal + sp2_sal + sum(p["dk_salary"] for p in all_hitters)
        else:
            break

    if total_salary < DK_MIN_SALARY:
        return None

    total_proj    = sum(p["dk_proj"] for p in all_hitters)
    total_ceiling = sum(p.get("dk_ceiling", p["dk_proj"] * 1.4) for p in all_hitters)

    # Add SP projections
    sp1_proj_data = _compute_dk_sp_proj({**sp1_entry, "opp_implied": 4.5})
    sp2_proj_data = _compute_dk_sp_proj({**sp2_entry, "opp_implied": 4.5})
    total_proj    += sp1_proj_data["dk_sp_proj"] + sp2_proj_data["dk_sp_proj"]
    total_ceiling += sp1_proj_data["dk_sp_ceiling"] + sp2_proj_data["dk_sp_ceiling"]

    return {
        "sp1":          sp1_entry["name"],
        "sp2":          sp2_entry["name"],
        "sp1_salary":   sp1_sal,
        "sp2_salary":   sp2_sal,
        "hitters":      all_hitters,
        "players":      all_hitters,
        "total_salary": total_salary,
        "total_proj":   round(total_proj, 1),
        "total_ceiling": round(total_ceiling, 1),
        "primary_team": primary_team,
        "secondary_team": secondary_team,
    }


def _build_dk_singleton_lineup(
    dk_plays: List[Dict],
    sp1_name: str,
    sp2_name: str,
    sp_salary_data: Dict,
) -> Dict | None:
    """Build all-singleton DK lineup — top 8 hitters by DK proj regardless of team."""
    sp1 = _dk_name_match(sp1_name, sp_salary_data)
    sp2 = _dk_name_match(sp2_name, sp_salary_data)
    if not sp1 or not sp2:
        return None

    budget = DK_SALARY_CAP - sp1["salary"] - sp2["salary"]
    DK_HITTER_SLOTS = ["C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
    used = set()
    hitters = []
    slot_count: Dict[str, int] = {}

    pool = sorted(dk_plays, key=lambda x: x["dk_proj"], reverse=True)
    for slot in DK_HITTER_SLOTS:
        for p in pool:
            if p["name"] in used:
                continue
            if not _dk_pos_eligible(str(p.get("dk_position", "")), slot):
                continue
            if p["dk_salary"] > budget:
                continue
            hitters.append({**p, "_slot": slot})
            used.add(p["name"])
            budget -= p["dk_salary"]
            slot_count[slot] = slot_count.get(slot, 0) + 1
            break

    if len(hitters) != 8:
        return None

    total_salary  = sp1["salary"] + sp2["salary"] + sum(h["dk_salary"] for h in hitters)
    total_proj    = sum(h["dk_proj"] for h in hitters)
    sp1_proj = _compute_dk_sp_proj(sp1)
    sp2_proj = _compute_dk_sp_proj(sp2)
    total_proj   += sp1_proj["dk_sp_proj"] + sp2_proj["dk_sp_proj"]
    total_ceiling = sum(h.get("dk_ceiling", h["dk_proj"] * 1.4) for h in hitters)
    total_ceiling += sp1_proj["dk_sp_ceiling"] + sp2_proj["dk_sp_ceiling"]

    return {
        "sp1": sp1["name"], "sp2": sp2["name"],
        "sp1_salary": sp1["salary"], "sp2_salary": sp2["salary"],
        "hitters": hitters, "players": hitters,
        "total_salary": total_salary,
        "total_proj": round(total_proj, 1),
        "total_ceiling": round(total_ceiling, 1),
        "primary_team": "SINGLETON",
        "secondary_team": "SINGLETON",
    }


def display_dk_portfolio_builder(plays: List[Dict]):
    """
    Tab 10 — DraftKings Portfolio Builder
    Roster: SP SP C 1B 2B 3B SS OF OF OF ($50,000 cap)
    Builds 10 or 22 lineups with structured stack exposure.
    Max 50% exposure per player or stack.
    Exports DK bulk upload CSV format.
    No true relievers (RP salary < $7,500 excluded at parse time).
    """
    st.header("⚡ DraftKings Portfolio Builder")
    st.caption(f"DK scoring: {DK_HITTER_SCORING}  |  SP: {DK_SP_SCORING}")
    st.caption("Roster: SP · SP · C · 1B · 2B · 3B · SS · OF · OF · OF  |  Cap: $50,000  |  Shapes: 5-3 / 5-2-1 / 5-1-1-1 (slate-adaptive)  |  Max exposure: 50%")

    if not plays:
        st.info("Run the model first.")
        return

    # ── SALARY UPLOAD ─────────────────────────────────────────────────────────
    # Use salary data already loaded in Command Center (same session key)
    dk_slate = st.session_state.get("dk_slate_data")

    st.subheader("📥 DraftKings Slate")
    if dk_slate:
        n_h = dk_slate.get("n_hitters", 0)
        n_p = dk_slate.get("n_pitchers", 0)
        n_g = len(dk_slate.get("slate_games", []))
        st.success(f"✅ Slate loaded from Command Center: {n_h} hitters · {n_p} pitchers · {n_g} games")
        st.caption("To reload, upload a new CSV in the Command Center tab → DraftKings mode.")
    else:
        col_up, col_info = st.columns([2, 3])
        with col_up:
            dk_csv_file = st.file_uploader(
                "Upload DraftKings CSV",
                type=["csv"], key="dk_portfolio_csv",
                help="Or upload in Command Center → DraftKings mode first."
            )
        with col_info:
            st.info("Upload DraftKings CSV here, or switch to the Command Center tab → DraftKings to load it there first.")
        if dk_csv_file:
            parsed = _parse_dk_csv(dk_csv_file)
            if "error" in parsed:
                st.error(f"CSV parse error: {parsed['error']}")
                return
            st.session_state.dk_slate_data = parsed
            dk_slate = parsed
            st.success(f"✅ Loaded: {parsed['n_hitters']} hitters · {parsed['n_pitchers']} pitchers")
        else:
            st.info("📥 Upload DraftKings CSV above, or load it in Command Center → DraftKings.")
            return

    salary_data    = dk_slate["salary_data"]
    sp_salary_data = dk_slate["sp_salary_data"]
    slate_games    = dk_slate["slate_games"]

    # Build DK plays
    dk_plays = _build_dk_plays_with_salaries(plays, salary_data, sp_salary_data)
    if not dk_plays:
        st.warning("No model plays matched DK salaries. Check that the correct slate CSV is uploaded.")
        return

    st.caption(f"✅ {len(dk_plays)} salary-matched players")

    st.markdown("---")

    # ── SP BOARD — Auto-optimized, no manual picker ───────────────────────────
    st.markdown("---")
    st.subheader("⚾ Starting Pitcher Board")
    st.caption("Portfolio auto-selects and rotates SPs. Ace + Value default pairing. SP cap: 45% per pitcher across lineup set.")

    # Filter SP pool to ONLY pitchers whose team is in today's model plays
    # This prevents pitchers from other slates or non-playing pitchers
    playing_teams = set(p.get("team","") for p in plays if p.get("team",""))
    sp_options_all = sorted(
        [s for s in sp_salary_data.values()
         if s.get("team","") in playing_teams],
        key=lambda x: -x["salary"]
    )
    if not sp_options_all:
        # Fallback: use all if team filter is too strict
        sp_options_all = sorted(sp_salary_data.values(), key=lambda x: -x["salary"])

    def _dk_sp_score(s):
        proj = _compute_dk_sp_proj(s)
        fppg_norm  = min(100, (s.get("fppg", 0) / 30.0) * 100)
        proj_norm  = min(100, (proj["dk_sp_proj"] / 25.0) * 100)
        value_norm = min(100, (proj["dk_sp_value"] / 2.5) * 100)
        return fppg_norm * 0.50 + proj_norm * 0.30 + value_norm * 0.20

    # Additional filter: only use pitchers confirmed in today's model
    playing_teams_port = set(p.get("team","") for p in plays if p.get("team",""))
    sp_options_all_filtered = [s for s in sp_options_all
                                if s.get("team","") in playing_teams_port]
    if not sp_options_all_filtered:
        sp_options_all_filtered = sp_options_all  # fallback
    sp_options_scored = sorted(sp_options_all_filtered, key=_dk_sp_score, reverse=True)
    sp_pool_for_portfolio = [s["name"] for s in sp_options_scored[:6]]

    auto_ace   = sp_options_scored[0]["name"] if sp_options_scored else ""
    value_pool = sorted([s for s in sp_options_scored if s["name"] != auto_ace],
                        key=lambda x: _compute_dk_sp_proj(x)["dk_sp_value"], reverse=True)
    auto_value = value_pool[0]["name"] if value_pool else (sp_options_scored[1]["name"] if len(sp_options_scored) > 1 else "")

    sp_rows = []
    for s in sp_options_scored[:12]:
        proj_data = _compute_dk_sp_proj(s)
        badge = "🥇 ACE" if s["name"] == auto_ace else ("💰 VALUE" if s["name"] == auto_value else "")
        sp_rows.append({
            "Auto":    badge,
            "SP":      s["name"],
            "Type":    s.get("position", "SP"),
            "Team":    s["team"],
            "Salary":  f"${s['salary']:,}",
            "FPPG":    f"{s.get('fppg',0):.1f}",
            "DK Proj": f"{proj_data['dk_sp_proj']:.1f}",
            "Ceiling": f"{proj_data['dk_sp_ceiling']:.1f}",
            "Proj K":  f"{proj_data['proj_k']:.0f}",
            "Value":   f"{proj_data['dk_sp_value']:.2f}x",
        })
    if sp_rows:
        st.dataframe(pd.DataFrame(sp_rows), use_container_width=True, hide_index=True)

    st.markdown(
        f"**Auto-selected:** 🥇 **{auto_ace}** (Ace) + 💰 **{auto_value}** (Value)  "
        f"— portfolio rotates through top {len(sp_pool_for_portfolio)} SPs with 45% cap per pitcher."
    )

    sp1_name = auto_ace
    sp2_name = auto_value


    # ── STACK CONFIGURATION — Auto-optimized ───────────────────────────────────
    st.markdown("---")
    st.subheader("🔗 Stack Configuration")
    st.caption("Auto-selected from game stack scores. Override if desired.")

    # Auto-select stacks from game scores — avoid teams opposing our SPs
    game_stacks = compute_game_stack_scores(plays)
    sp1_team = sp_salary_data.get(sp1_name.lower(), {}).get("team", "")
    sp2_team = sp_salary_data.get(sp2_name.lower(), {}).get("team", "")
    sp_opp_teams = set()  # teams OPPOSING our SPs (don't want to stack against our pitcher)
    for sp_t in [sp1_team, sp2_team]:
        if sp_t:
            for g in game_stacks:
                if g.get("home_team") == sp_t:
                    sp_opp_teams.add(g.get("away_team",""))
                elif g.get("away_team") == sp_t:
                    sp_opp_teams.add(g.get("home_team",""))

    # Build ranked team list with player count filter
    ranked_teams = []
    seen_t = set()
    for g in game_stacks:
        for t in [g.get("home_team",""), g.get("away_team","")]:
            if not t or t in seen_t:
                continue
            n_players = len([p for p in dk_plays
                             if p.get("dk_team","") == t or p.get("team","") == t])
            if n_players >= 3:
                ranked_teams.append((t, g["stack_score"], n_players, t in sp_opp_teams))
                seen_t.add(t)

    # Default: top 3 non-opposing teams
    safe_teams  = [t for t, sc, n, is_opp in ranked_teams if not is_opp]
    risky_teams = [t for t, sc, n, is_opp in ranked_teams if is_opp]
    all_sorted  = safe_teams + risky_teams  # safe first

    auto_a = all_sorted[0] if len(all_sorted) > 0 else ""
    auto_b = all_sorted[1] if len(all_sorted) > 1 else auto_a
    auto_c = all_sorted[2] if len(all_sorted) > 2 else auto_b

    team_options = [t for t, *_ in ranked_teams] if ranked_teams else                    sorted(set(p.get("team","") for p in dk_plays if p.get("team","")))

    # Show auto-selected stacks with override option
    with st.expander("🔧 Override Stack Selection (auto-selected below)", expanded=False):
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            stack_a_idx = team_options.index(auto_a) if auto_a in team_options else 0
            stack_a = st.selectbox("🥇 Stack A (Primary)", team_options, index=stack_a_idx, key="dk_stack_a")
        with col_s2:
            rem_b = [t for t in team_options if t != stack_a]
            stack_b_idx = rem_b.index(auto_b) if auto_b in rem_b else 0
            stack_b = st.selectbox("🥈 Stack B (Secondary)", rem_b, index=stack_b_idx, key="dk_stack_b")
        with col_s3:
            rem_c = [t for t in team_options if t not in (stack_a, stack_b)]
            stack_c_idx = rem_c.index(auto_c) if auto_c in rem_c else 0
            stack_c = st.selectbox("🥉 Stack C (Coverage)", rem_c if rem_c else team_options, index=min(stack_c_idx, len(rem_c or team_options)-1), key="dk_stack_c")
    # Use auto values if expander not expanded (default)
    if "dk_stack_a" not in st.session_state:
        stack_a, stack_b, stack_c = auto_a, auto_b, auto_c
    else:
        stack_a = st.session_state.get("dk_stack_a", auto_a)
        stack_b = st.session_state.get("dk_stack_b", auto_b)
        stack_c = st.session_state.get("dk_stack_c", auto_c)

    # Show auto-selection summary
    stack_info_rows = []
    for t, sc, n, is_opp in ranked_teams[:6]:
        tag = "🥇" if t == stack_a else ("🥈" if t == stack_b else ("🥉" if t == stack_c else ""))
        warn = " ⚠️ opp SP" if is_opp else ""
        stack_info_rows.append({"": tag, "Team": t, "Stack Score": f"{sc:.1f}", "Players": n, "Note": warn})
    if stack_info_rows:
        st.dataframe(pd.DataFrame(stack_info_rows), use_container_width=True, hide_index=True)

    for sp_t, sp_n in [(sp1_team, sp1_name), (sp2_team, sp2_name)]:
        if sp_t and sp_t in sp_opp_teams and sp_t in (stack_a, stack_b, stack_c):
            st.warning(f"⚠️ {sp_n} ({sp_t}) pitches against your stack team — scoring may be suppressed.")

    # ── BUILD CONTROLS ─────────────────────────────────────────────────────────
    st.markdown("---")
    col_b1, col_b2, col_b3 = st.columns([1, 1, 2])
    with col_b1:
        n_lineups_choice = st.radio("Lineups to build", ["10", "22"], horizontal=True, key="dk_n_lineups")
        n_lineups = int(n_lineups_choice)
    with col_b2:
        max_exposure = st.slider("Max player exposure", 30, 60, 50, 5, key="dk_exposure",
                                  help="Max % of lineups any single player can appear in") / 100.0
    with col_b3:
        min_salary = st.slider("Min lineup salary", 46000, 49500, DK_MIN_SALARY, 500, key="dk_min_sal")

    build_btn = st.button("⚡ Build DraftKings Lineups", type="primary", key="dk_build_btn")

    if build_btn:
        stacked_lu = n_lineups - 1  # last is singleton benchmark

        # ── Slate-aware shape allocation ──────────────────────────────────────
        # Build stack_teams list from current game scores for detect_slate_shape
        # Only rank teams from the DK salary import (not all model teams)
        dk_salary_teams = set(
            v.get("team","") for v in sp_salary_data.values()
            if v.get("team","")
        ) | set(p.get("dk_team", p.get("team","")) for p in dk_plays)
        dk_raw_plays = [p for p in (st.session_state.get("plays") or plays)
                        if p.get("team","") in dk_salary_teams]

        # Use team-level scores (NOT game scores) for DK stack ranking
        dk_team_ranks  = get_ranked_team_stacks(dk_raw_plays if dk_raw_plays else plays, min_players=3)
        dk_stack_teams = [(sd["team"], sd["stack_score"]) for sd in dk_team_ranks
                          if sd["team"] in dk_salary_teams
                          and len([p for p in dk_plays
                                   if p.get("dk_team","") == sd["team"] or p.get("team","") == sd["team"]]) >= 3]

        dk_slate_analysis = detect_slate_shape(dk_stack_teams, dk_plays, n_lineups, site="DK")
        dk_shapes = dk_slate_analysis["shapes"]
        st.session_state["dk_slate_analysis"] = dk_slate_analysis

        alloc_53   = dk_shapes.get("5-3",     0)
        alloc_521  = dk_shapes.get("5-2-1",   0)
        alloc_5111 = dk_shapes.get("5-1-1-1", 0)

        # Normalize
        total_dk_alloc = alloc_53 + alloc_521 + alloc_5111
        if total_dk_alloc != stacked_lu:
            alloc_521 += stacked_lu - total_dk_alloc

        # Show slate analysis immediately
        mode_color = {"heavy_stack": "#00ff88", "balanced": "#ffdd00", "spread": "#ff8800"}.get(
            dk_slate_analysis.get("mode",""), "#aaa")
        st.info(
            f"**Slate: {dk_slate_analysis.get('mode','').upper().replace('_',' ')}** — "
            f"Concentration {dk_slate_analysis.get('concentration',0):.2f} | "
            f"5-3: {alloc_53} · 5-2-1: {alloc_521} · 5-1-1-1: {alloc_5111} · Singleton: 1  — "
            f"{dk_slate_analysis.get('rationale','')}"
        )

        lineups      = []
        exp_tracker: Dict[str, int] = {}
        max_count    = max(1, int(n_lineups * max_exposure))

        sp_pool_all   = sorted(sp_salary_data.values(), key=lambda x: -x["salary"])
        sp_pool_names = [s["name"] for s in sp_pool_all]

        # Build DK schedule: (primary, secondary, shape, sec_target)
        # sec_target = how many batters from secondary team (3 for 5-3, 2 for 5-2-1, 1 for 5-1-1-1)
        def pick_dk_secondary(primary, idx, sec_target):
            opts = [t for t, _ in dk_stack_teams if t != primary]
            return opts[idx % len(opts)] if opts else secondary_team_default

        secondary_team_default = stack_b if stack_b and stack_b != stack_a else (stack_c or stack_a)

        dk_schedule = []
        sec_idx = 0
        for i in range(alloc_53):
            primary = stack_a if i % 2 == 0 else stack_b
            secondary = pick_dk_secondary(primary, sec_idx, 3)
            dk_schedule.append((primary, secondary, "5-3", 3))
            sec_idx += 1
        for i in range(alloc_521):
            primary = stack_a if i % 3 < 2 else stack_b   # 2/3 from stack_a, 1/3 stack_b
            secondary = pick_dk_secondary(primary, sec_idx, 2)
            dk_schedule.append((primary, secondary, "5-2-1", 2))
            sec_idx += 1
        for i in range(alloc_5111):
            primary = stack_a
            secondary = pick_dk_secondary(primary, sec_idx, 1)
            dk_schedule.append((primary, secondary, "5-1-1-1", 1))
            sec_idx += 1

        for sched_idx, (stack_team, secondary_team, shape, sec_target) in enumerate(dk_schedule):
            if len(lineups) >= n_lineups - 1:
                break
            lu = _build_dk_gpp_lineup(
                dk_plays=dk_plays,
                primary_team=stack_team,
                secondary_team=secondary_team,
                sp1_name=sp1_name,
                sp2_name=sp2_name,
                sp_salary_data=sp_salary_data,
                lineup_num=sched_idx,
            )
            # Retry with alt secondary
            if lu is None:
                for alt_sec in [t for t, _ in dk_stack_teams if t != stack_team]:
                    lu = _build_dk_gpp_lineup(
                        dk_plays=dk_plays,
                        primary_team=stack_team,
                        secondary_team=alt_sec,
                        sp1_name=sp1_name,
                        sp2_name=sp2_name,
                        sp_salary_data=sp_salary_data,
                        lineup_num=sched_idx,
                    )
                    if lu is not None:
                        break
            # Retry with alt SP pairing
            if lu is None and len(sp_pool_names) >= 3:
                for alt_sp2 in sp_pool_names[2:5]:
                    if alt_sp2 in (sp1_name, sp2_name):
                        continue
                    lu = _build_dk_gpp_lineup(
                        dk_plays=dk_plays,
                        primary_team=stack_team,
                        secondary_team=secondary_team,
                        sp1_name=sp1_name,
                        sp2_name=alt_sp2,
                        sp_salary_data=sp_salary_data,
                        lineup_num=sched_idx,
                    )
                    if lu is not None:
                        break
            if lu is None:
                continue
            effective_min = min_salary - 1000   # relax floor to maximise lineup count
            if lu["total_salary"] < effective_min:
                continue
            # Exposure: soft guidance only — never block on small slates
            # Count how many players are over the soft cap for logging
            # but never prevent lineup from being added
            _over = [p.get("name","") for p in lu["hitters"]
                     if exp_tracker.get(p.get("name",""), 0) >= max_count]
            # Only skip if ALL core players are severely over-exposed (>2x cap)
            hard_over = [p.get("name","") for p in lu["hitters"]
                         if exp_tracker.get(p.get("name",""), 0) >= max_count * 2]
            if len(hard_over) >= 5:   # 5+ players at 2x cap = truly unacceptable
                continue
            for p in lu["hitters"]:
                nm = p.get("name","")
                exp_tracker[nm] = exp_tracker.get(nm, 0) + 1
            lu["shape"] = shape
            lu["primary_team"]   = stack_team
            lu["secondary_team"] = secondary_team
            lineups.append(lu)

        # ── Singleton benchmark (last lineup) ─────────────────────────────────
        sing_lu = _build_dk_singleton_lineup(dk_plays, sp1_name, sp2_name, sp_salary_data)
        if sing_lu:
            sing_lu["_singleton"] = True
            lineups.append(sing_lu)

        st.session_state.dk_port_lineups = lineups
        st.session_state.dk_exp_tracker  = exp_tracker
        st.success(f"✅ Built {len(lineups)} lineups successfully")

    # ── DISPLAY LINEUPS ────────────────────────────────────────────────────────
    lineups = st.session_state.get("dk_port_lineups", [])
    if not lineups:
        return

    # Summary metrics
    avg_sal  = sum(lu["total_salary"] for lu in lineups) / len(lineups)
    avg_proj = sum(lu["total_proj"] for lu in lineups) / len(lineups)
    avg_ceil = sum(lu["total_ceiling"] for lu in lineups) / len(lineups)
    sal_min  = min(lu["total_salary"] for lu in lineups)
    sal_max  = max(lu["total_salary"] for lu in lineups)

    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Lineups Built",  len(lineups))
    m2.metric("Avg Salary",     f"${avg_sal:,.0f}")
    m3.metric("Sal Range",      f"${sal_min:,}–${sal_max:,}")
    m4.metric("Avg Proj",       f"{avg_proj:.1f}")
    m5.metric("Avg Ceiling",    f"{avg_ceil:.1f}")

    # ── Export ────────────────────────────────────────────────────────────────
    st.subheader("📤 Export")
    st.caption("DraftKings bulk upload format: SP,SP,C,1B,2B,3B,SS,OF,OF,OF")

    def build_dk_export_csv(lineups: List[Dict]) -> str:
        rows = []
        for lu in lineups:
            sp1_n = lu.get("sp1", "")
            sp2_n = lu.get("sp2", "")
            slot_map = {h["_slot"]: h for h in lu["hitters"]}
            def get_slot(slot):
                return slot_map.get(slot, {}).get("name", "")
            row = ",".join([
                sp1_n, sp2_n,
                get_slot("C"), get_slot("1B"), get_slot("2B"),
                get_slot("3B"), get_slot("SS"),
                *[h["name"] for h in lu["hitters"] if h["_slot"] == "OF"][:3]
            ])
            rows.append(row)
        header = "SP,SP,C,1B,2B,3B,SS,OF,OF,OF"
        return header + "\n" + "\n".join(rows)

    dk_csv_out = build_dk_export_csv(lineups)
    st.download_button(
        "📥 Download DraftKings Bulk Upload CSV",
        dk_csv_out,
        f"dk_lineups_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv",
        key="dk_export_btn"
    )

    # ── Lineup grid ───────────────────────────────────────────────────────────
    st.subheader(f"📋 All {len(lineups)} Lineups")
    st.caption("Shape: slate-adaptive (5-3 / 5-2-1 / 5-1-1-1) · Last lineup = all-singleton benchmark 🧪")

    grid_rows = []
    for i, lu in enumerate(lineups, 1):
        is_sing = lu.get("_singleton", False)
        ltype   = "🧪 ALL-SINGLETON" if is_sing else f"{lu.get('shape','5-2-1')} · {lu.get('primary_team','')}+{lu.get('secondary_team','')}"
        grid_rows.append({
            "#":         i,
            "SP1":       lu.get("sp1",""),
            "SP2":       lu.get("sp2",""),
            "Type":      ltype,
            "Primary":   lu.get("primary_team",""),
            "Secondary": lu.get("secondary_team",""),
            "Proj":      f"{lu['total_proj']:.1f}",
            "Ceiling":   f"{lu['total_ceiling']:.1f}",
            "Salary":    f"${lu['total_salary']:,}",
            "Left":      f"${DK_SALARY_CAP - lu['total_salary']:,}",
        })

    st.dataframe(pd.DataFrame(grid_rows), use_container_width=True, hide_index=True)

    # Expandable lineup detail cards
    for i, lu in enumerate(lineups, 1):
        is_sing = lu.get("_singleton", False)
        label = (f"Lineup #{i} [ALL-SINGLETON] — "
                 f"Proj {lu['total_proj']:.1f} | ${lu['total_salary']:,}") if is_sing else (
                 f"Lineup #{i} — "
                 f"{lu.get('primary_team','')}+{lu.get('secondary_team','')} — "
                 f"Proj {lu['total_proj']:.1f} | ${lu['total_salary']:,}")
        with st.expander(label, expanded=(i == 1)):
            sp_col, hit_col = st.columns([1, 2])
            with sp_col:
                st.markdown("**⚾ Pitchers**")
                st.write(f"SP1: {lu.get('sp1','')} — ${lu.get('sp1_salary',0):,}")
                st.write(f"SP2: {lu.get('sp2','')} — ${lu.get('sp2_salary',0):,}")
            with hit_col:
                st.markdown("**🏏 Hitters**")
                hit_rows = []
                for h in lu["hitters"]:
                    team = h.get("dk_team") or h.get("team","")
                    is_primary   = (team == lu.get("primary_team",""))
                    is_secondary = (team == lu.get("secondary_team",""))
                    flag = "★" if is_primary else ("♦" if is_secondary else "💎")
                    hit_rows.append({
                        "Slot":   h["_slot"],
                        "":       flag,
                        "Player": h.get("name",""),
                        "Team":   team,
                        "Salary": f"${h['dk_salary']:,}",
                        "DK Proj": f"{h.get('dk_proj',0):.1f}",
                    })
                st.dataframe(pd.DataFrame(hit_rows), use_container_width=True, hide_index=True)

    # ── Player Exposure Report ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Player Exposure Report")
    exp_tracker = st.session_state.get("dk_exp_tracker", {})
    if exp_tracker:
        max_allowed_count = max(1, int(len(lineups) * max_exposure))
        exp_rows = []
        for name, count in sorted(exp_tracker.items(), key=lambda x: -x[1]):
            pct  = round(count / len(lineups) * 100, 1)
            over = pct > max_exposure * 100
            exp_rows.append({"Player": name, "Count": count, "Exposure%": f"{pct:.1f}%", "Over": "⚠️" if over else "✅"})
        st.dataframe(pd.DataFrame(exp_rows), use_container_width=True, hide_index=True)


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



if __name__ == "__main__":
    from app import main
    main()
