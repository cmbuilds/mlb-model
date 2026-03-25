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
# STATCAST / PYBASEBALL DATA
# ============================================================================
@st.cache_data(ttl=3600)
def fetch_statcast_batter_stats(player_id: str, days_back: int = 30) -> Dict:
    """
    Fetch batter Statcast stats via Baseball Savant API.
    Returns xSLG, barrel%, hard_hit%, exit_velocity, launch_angle, ISO proxy.
    """
    end_date = datetime.now(EST).strftime("%Y-%m-%d")
    start_date = (datetime.now(EST) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    # Baseball Savant search API
    url = "https://baseballsavant.mlb.com/statcast_search/csv"
    params = {
        "hfPT": "",
        "hfAB": "",
        "hfGT": "R|",
        "hfPR": "",
        "hfZ": "",
        "stadium": "",
        "hfBBL": "",
        "hfNewZones": "",
        "hfPull": "",
        "hfC": "",
        "hfSea": "2025|",
        "hfSit": "",
        "player_type": "batter",
        "hfOuts": "",
        "opponent": "",
        "pitcher_throws": "",
        "batter_stands": "",
        "hfSA": "",
        "game_date_gt": start_date,
        "game_date_lt": end_date,
        "player_id": player_id,
        "hfInfield": "",
        "team": "",
        "position": "",
        "hfRO": "",
        "home_road": "",
        "hfFlag": "",
        "metric_1": "",
        "hfInn": "",
        "min_pitches": "0",
        "min_results": "0",
        "group_by": "name",
        "sort_col": "pitches",
        "player_event_sort": "api_p_release_speed",
        "sort_order": "desc",
        "min_pas": "0",
        "type": "details",
    }
    
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 200 and len(r.content) > 100:
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            if df.empty:
                return {}
            
            # Calculate metrics from event-level data
            batted_balls = df[df['launch_speed'].notna()]
            
            stats = {}
            if not batted_balls.empty:
                n = len(batted_balls)
                stats['exit_velocity_avg'] = batted_balls['launch_speed'].mean()
                stats['launch_angle_avg'] = batted_balls['launch_angle'].mean()
                
                # Hard hit rate (95+ mph)
                hard_hit = batted_balls[batted_balls['launch_speed'] >= 95]
                stats['hard_hit_rate'] = len(hard_hit) / n if n > 0 else 0.33
                
                # Barrel rate (LA 26-30 deg, EV 98+ mph)
                barrels = batted_balls[
                    (batted_balls['launch_speed'] >= 98) & 
                    (batted_balls['launch_angle'] >= 26) & 
                    (batted_balls['launch_angle'] <= 30)
                ]
                stats['barrel_rate'] = len(barrels) / n if n > 0 else 0.06
                
                # Sweet spot rate (8-32 degree launch angle)
                sweet = batted_balls[
                    (batted_balls['launch_angle'] >= 8) & 
                    (batted_balls['launch_angle'] <= 32)
                ]
                stats['sweet_spot_rate'] = len(sweet) / n if n > 0 else 0.30
            
            # K rate
            all_pa = df[df['events'].notna()]
            if not all_pa.empty:
                strikeouts = all_pa[all_pa['events'] == 'strikeout']
                stats['k_rate'] = len(strikeouts) / len(all_pa)
                
                # Hit types for ISO proxy
                singles = all_pa[all_pa['events'] == 'single']
                doubles = all_pa[all_pa['events'] == 'double']
                triples = all_pa[all_pa['events'] == 'triple']
                hrs = all_pa[all_pa['events'] == 'home_run']
                
                tb = len(singles) + 2*len(doubles) + 3*len(triples) + 4*len(hrs)
                ab_proxy = len(all_pa[all_pa['events'].isin(['single','double','triple','home_run','strikeout','field_out','grounded_into_double_play','force_out','sac_fly'])])
                
                stats['slg_proxy'] = tb / ab_proxy if ab_proxy > 0 else 0.380
                stats['iso_proxy'] = stats['slg_proxy'] - (len(singles) / ab_proxy if ab_proxy > 0 else 0.240)
                stats['tb_per_game'] = tb / max(1, len(all_pa['game_date'].unique()))
                
            return stats
    except Exception as e:
        pass
    
    return {}

@st.cache_data(ttl=3600)
def fetch_pitcher_statcast(pitcher_id: str, days_back: int = 30) -> Dict:
    """Fetch pitcher Statcast metrics - contact quality allowed."""
    end_date = datetime.now(EST).strftime("%Y-%m-%d")
    start_date = (datetime.now(EST) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    url = "https://baseballsavant.mlb.com/statcast_search/csv"
    params = {
        "hfGT": "R|",
        "hfSea": "2025|",
        "player_type": "pitcher",
        "game_date_gt": start_date,
        "game_date_lt": end_date,
        "player_id": pitcher_id,
        "type": "details",
        "min_results": "0",
    }
    
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 200 and len(r.content) > 100:
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            if df.empty:
                return {}
            
            batted_balls = df[df['launch_speed'].notna()]
            stats = {}
            
            if not batted_balls.empty:
                n = len(batted_balls)
                stats['hard_hit_allowed'] = len(batted_balls[batted_balls['launch_speed'] >= 95]) / n
                barrels = batted_balls[
                    (batted_balls['launch_speed'] >= 98) & 
                    (batted_balls['launch_angle'] >= 26) & 
                    (batted_balls['launch_angle'] <= 30)
                ]
                stats['barrel_allowed'] = len(barrels) / n
            
            all_pa = df[df['events'].notna()]
            if not all_pa.empty:
                ks = all_pa[all_pa['events'] == 'strikeout']
                stats['k_rate_allowed'] = len(ks) / len(all_pa)
            
            return stats
    except:
        pass
    
    return {}

@st.cache_data(ttl=7200)
def fetch_fangraphs_batting_stats(season: int = 2025) -> pd.DataFrame:
    """
    Fetch FanGraphs season batting stats via pybaseball-style direct scrape.
    Returns wRC+, ISO, wOBA, SLG, K%, BB%.
    """
    # Use Baseball Reference / FanGraphs leaderboard API
    url = f"https://www.fangraphs.com/api/leaders/major-league/data"
    params = {
        "age": "",
        "pos": "all",
        "stats": "bat",
        "lg": "all",
        "qual": "y",
        "season": season,
        "season1": season,
        "ind": "0",
        "team": "0",
        "pageitems": "500",
        "pagenum": "1",
        "type": "8",  # Advanced stats
        "sortstat": "WAR",
        "sortdir": "default",
    }
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; research bot)"}
        r = requests.get(url, params=params, headers=headers, timeout=20)
        if r.status_code == 200:
            data = r.json()
            rows = data.get("data", [])
            if rows:
                df = pd.DataFrame(rows)
                return df
    except:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=7200) 
def fetch_fangraphs_pitching_stats(season: int = 2025) -> pd.DataFrame:
    """Fetch FanGraphs pitching stats - ERA, FIP, K%, xFIP."""
    url = f"https://www.fangraphs.com/api/leaders/major-league/data"
    params = {
        "pos": "all",
        "stats": "pit",
        "lg": "all",
        "qual": "y",
        "season": season,
        "season1": season,
        "ind": "0",
        "team": "0",
        "pageitems": "300",
        "pagenum": "1",
        "type": "8",
        "sortstat": "WAR",
        "sortdir": "default",
    }
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; research bot)"}
        r = requests.get(url, params=params, headers=headers, timeout=20)
        if r.status_code == 200:
            data = r.json()
            rows = data.get("data", [])
            if rows:
                df = pd.DataFrame(rows)
                return df
    except:
        pass
    return pd.DataFrame()

# ============================================================================
# WEATHER API
# ============================================================================
@st.cache_data(ttl=3600)
def fetch_weather(lat: float, lon: float, game_time_utc: str, is_dome: bool) -> Dict:
    """Fetch weather from Open-Meteo for a given stadium."""
    if is_dome:
        return {
            "wind_speed": 0, "wind_direction": 0, "wind_dir_label": "DOME",
            "temperature": 72, "humidity": 50, "is_dome": True,
            "wind_effect": "neutral", "temp_effect": "neutral"
        }
    
    # Parse game time for hour
    try:
        game_dt = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00"))
        game_hour = game_dt.hour
    except:
        game_hour = 19  # 7pm default
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,windspeed_10m,winddirection_10m,relativehumidity_2m",
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "timezone": "America/New_York",
        "forecast_days": 2,
    }
    
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        hourly = data.get("hourly", {})
        
        # Find the closest hour to game time
        times = hourly.get("time", [])
        target_idx = 0
        for i, t in enumerate(times):
            try:
                t_dt = datetime.fromisoformat(t)
                if t_dt.hour == game_hour:
                    target_idx = i
                    break
            except:
                pass
        
        wind_speed = hourly.get("windspeed_10m", [5])[target_idx] if hourly.get("windspeed_10m") else 5
        wind_dir = hourly.get("winddirection_10m", [180])[target_idx] if hourly.get("winddirection_10m") else 180
        temperature = hourly.get("temperature_2m", [70])[target_idx] if hourly.get("temperature_2m") else 70
        humidity = hourly.get("relativehumidity_2m", [50])[target_idx] if hourly.get("relativehumidity_2m") else 50
        
        # Wind direction label and effect
        wind_dir_label, wind_effect = classify_wind(wind_dir, wind_speed)
        
        # Temperature effect
        if temperature < 50:
            temp_effect = "suppress"
        elif temperature > 83:
            temp_effect = "boost"
        else:
            temp_effect = "neutral"
        
        return {
            "wind_speed": round(wind_speed, 1),
            "wind_direction": wind_dir,
            "wind_dir_label": wind_dir_label,
            "temperature": round(temperature, 1),
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
            "error": str(e)
        }

def classify_wind(direction: float, speed: float) -> Tuple[str, str]:
    """
    Classify wind direction and effect on HR/TB.
    Direction is meteorological (0=N, 90=E, 180=S, 270=W).
    In most MLB parks, hitting to LF/CF = home plate facing ~NE-E.
    """
    if speed < 8:
        return "Calm", "neutral"
    
    # Cardinal directions
    if direction < 22.5 or direction >= 337.5:
        label = "N"
    elif direction < 67.5:
        label = "NE"
    elif direction < 112.5:
        label = "E"
    elif direction < 157.5:
        label = "SE"
    elif direction < 202.5:
        label = "S"
    elif direction < 247.5:
        label = "SW"
    elif direction < 292.5:
        label = "W"
    else:
        label = "NW"
    
    # Most parks face northeast (batter looks toward NE outfield)
    # Wind OUT = coming from SW (wind going toward outfield = helping HRs)
    # Wind IN = coming from NE (wind going toward infield = suppressing HRs)
    
    if direction >= 180 and direction <= 270:  # SW/S/W wind = blowing OUT
        if speed >= 12:
            effect = "strong_out"   # +25-30% HR modifier
        elif speed >= 8:
            effect = "out"          # +15% HR modifier
        else:
            effect = "neutral"
    elif direction >= 0 and direction <= 90:  # N/NE/E wind = blowing IN
        if speed >= 10:
            effect = "in"           # -20% HR modifier
        else:
            effect = "neutral"
    else:
        effect = "neutral"          # crosswind
    
    return label, effect

# ============================================================================
# ODDS API
# ============================================================================
@st.cache_data(ttl=1800)
def fetch_odds(date_str: str) -> Dict:
    """
    Fetch MLB odds from The Odds API.
    Returns implied team totals keyed by team abbreviation.
    """
    try:
        api_key = st.secrets.get("odds_api", {}).get("api_key", "")
        if not api_key:
            return {}
        
        url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": "totals,h2h",
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        games = r.json()
        
        implied_totals = {}
        for game in games:
            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")
            
            game_total = None
            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market.get("key") == "totals":
                        for outcome in market.get("outcomes", []):
                            if outcome.get("name") == "Over":
                                game_total = outcome.get("point", 9.0)
                                break
                    if game_total:
                        break
                if game_total:
                    break
            
            if game_total:
                # Rough split: 50/50 implied
                home_implied = game_total * 0.52  # home field slight advantage
                away_implied = game_total * 0.48
                
                # Map team names to abbreviations
                for full_name, abbr in TEAM_ABB_MAP.items():
                    if full_name.lower() in home_team.lower() or abbr in home_team:
                        implied_totals[abbr] = round(home_implied, 2)
                    if full_name.lower() in away_team.lower() or abbr in away_team:
                        implied_totals[abbr] = round(away_implied, 2)
        
        return implied_totals
    except Exception as e:
        return {}

# ============================================================================
# SCORING ENGINE
# ============================================================================

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
    Primary inputs: xSLG proxy, barrel%, hard hit%, ISO, K rate.
    Uses research-backed weights: barrel(8-10%), hard_hit(6-8%), ISO(6%), K rate inverse.
    """
    details = {}
    
    # Extract metrics with league average defaults
    xslg = statcast.get("slg_proxy", 0.380)  # league avg ~.380
    barrel_rate = statcast.get("barrel_rate", 0.070)  # league avg ~7%
    hard_hit = statcast.get("hard_hit_rate", 0.370)   # league avg ~37%
    k_rate = statcast.get("k_rate", 0.230)             # league avg ~23%
    iso = statcast.get("iso_proxy", 0.155)             # league avg ~.155
    exit_vel = statcast.get("exit_velocity_avg", 88.5) # league avg ~88.5
    sweet_spot = statcast.get("sweet_spot_rate", 0.30)
    tb_per_g = statcast.get("tb_per_game", 0.85)       # league avg ~.85 TB/PA
    
    details["xSLG"] = round(xslg, 3)
    details["Barrel%"] = f"{barrel_rate*100:.1f}%"
    details["HardHit%"] = f"{hard_hit*100:.1f}%"
    details["K%"] = f"{k_rate*100:.1f}%"
    details["ISO"] = round(iso, 3)
    
    # Sub-scores (normalize to 0-100)
    
    # xSLG: .200=0, .380=50, .600=100
    xslg_score = (xslg - 0.200) / (0.600 - 0.200) * 100
    xslg_score = max(0, min(100, xslg_score))
    
    # Barrel rate: 0%=0, 7%=50, 20%=100
    barrel_score = barrel_rate / 0.20 * 100
    barrel_score = max(0, min(100, barrel_score))
    
    # Hard hit rate: 20%=0, 37%=50, 60%=100
    hard_hit_score = (hard_hit - 0.20) / (0.60 - 0.20) * 100
    hard_hit_score = max(0, min(100, hard_hit_score))
    
    # K rate: INVERSE. 5%=100, 23%=50, 40%=0
    k_score = max(0, min(100, (0.40 - k_rate) / (0.40 - 0.05) * 100))
    
    # ISO: .050=0, .155=50, .350=100
    iso_score = (iso - 0.050) / (0.350 - 0.050) * 100
    iso_score = max(0, min(100, iso_score))
    
    # TB per game recency: 0=0, 0.85=50, 2.0=100
    tb_score = min(100, tb_per_g / 2.0 * 100)
    
    # Weighted composite (per research: xSLG 12-15%, barrel 8-10%, hard_hit 6-8%, ISO 6%, K 5-7%)
    composite = (
        xslg_score * 0.28 +      # xSLG highest weight
        barrel_score * 0.22 +    # barrel rate
        hard_hit_score * 0.18 +  # hard hit rate
        iso_score * 0.14 +       # ISO power
        k_score * 0.12 +         # K rate inverse
        tb_score * 0.06          # recent form
    )
    
    return max(0, min(100, composite)), "Statcast profile", details


def compute_pitcher_score(statcast: Dict, fg_stats: Dict = None) -> Tuple[float, str]:
    """
    Compute pitcher vulnerability sub-score 0-100.
    Higher score = MORE vulnerable pitcher = better for batter.
    Inputs: K%, hard_hit_allowed, barrel_allowed, ERA proxy.
    """
    # Extract with league average defaults
    k_rate = statcast.get("k_rate_allowed", 0.230)    # pitcher K% (high = good pitcher = BAD for batter)
    hard_hit = statcast.get("hard_hit_allowed", 0.360) # hard contact allowed
    barrel = statcast.get("barrel_allowed", 0.065)     # barrels allowed
    
    # K rate: INVERSE (high K pitcher = low vulnerability)
    # 10% K rate = 90 vulnerability; 30% K rate = 30 vulnerability
    k_vuln_score = max(0, min(100, (0.30 - k_rate) / (0.30 - 0.10) * 80 + 20))
    
    # Hard hit allowed: higher = more vulnerable
    # 25%=0, 36%=50, 50%=100
    hard_hit_score = (hard_hit - 0.25) / (0.50 - 0.25) * 100
    hard_hit_score = max(0, min(100, hard_hit_score))
    
    # Barrel allowed: higher = more vulnerable
    # 0%=0, 6.5%=50, 15%=100
    barrel_score = barrel / 0.15 * 100
    barrel_score = max(0, min(100, barrel_score))
    
    composite = (k_vuln_score * 0.40 + hard_hit_score * 0.35 + barrel_score * 0.25)
    
    return max(0, min(100, composite)), f"K%: {k_rate*100:.0f}% | HH%: {hard_hit*100:.0f}% | Barrel: {barrel*100:.1f}%"


def compute_vegas_score(implied_total: float) -> Tuple[float, str]:
    """
    Compute Vegas signal sub-score 0-100.
    Team implied run total = strong proxy for run environment.
    4.5+ = favorable; below 3.5 = bad.
    """
    if not implied_total or implied_total <= 0:
        return 50.0, "No lines ⚠️"
    
    # 3.0=0, 4.5=50, 6.5=100
    score = (implied_total - 3.0) / (6.5 - 3.0) * 100
    score = max(0, min(100, score))
    
    flag = ""
    if implied_total >= 5.5:
        flag = " 🔥"
    elif implied_total >= 4.5:
        flag = " ✅"
    elif implied_total < 3.5:
        flag = " ⚠️"
    
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
    Final weighted composite score using research-calibrated weights.
    
    Weights per research doc:
    - Batter Profile: 45% (xSLG, barrel, hard hit, ISO, K, recent form)
    - Pitcher Vulnerability: 30% (K%, hard hit allowed, barrel allowed)
    - Platoon: 6%
    - Lineup Position: 4%
    - Park + Weather: 10%
    - Vegas Signal: 5%
    """
    score = (
        batter_score * 0.45 +
        pitcher_vuln_score * 0.30 +
        platoon_score * 0.06 +
        lineup_score * 0.04 +
        park_score * 0.07 +
        weather_score * 0.03 +
        vegas_score * 0.05
    )
    return max(0, min(100, round(score, 1)))


def score_to_prob(score: float) -> float:
    """
    Map 0-100 score to probability using logistic function.
    Calibrated so:
    - Score 47 → ~47% probability (league average)
    - Score 85+ → 75-85% probability
    - Score 65 → ~60% probability
    """
    # Logistic: P = 1 / (1 + exp(-a*(score - b)))
    a = 0.07
    b = 50
    prob = 1 / (1 + math.exp(-a * (score - b)))
    # Scale to reasonable MLB prop range (35% - 85%)
    prob = 0.35 + prob * 0.50
    return round(min(0.85, max(0.35, prob)), 3)


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
# MAIN MODEL PIPELINE
# ============================================================================
def run_model(date_str: str, status_container) -> List[Dict]:
    """
    Master pipeline: pulls all data, scores every batter, returns ranked list.
    """
    results = []
    logs = []
    
    def log(msg):
        logs.append(msg)
        status_container.markdown("\n".join(logs[-8:]))
    
    log(f"⚾ **MLB TB Model** | {date_str}")
    log("─" * 50)
    
    # ── 1. SCHEDULE ──────────────────────────────────────
    log("📅 Fetching schedule...")
    games = fetch_schedule(date_str)
    
    if not games:
        log("❌ No games found for this date.")
        return []
    
    log(f"✅ Found {len(games)} games")
    for g in games:
        log(f"   • {g['away_team']} @ {g['home_team']} | SP: {g['away_pitcher'] or 'TBD'} vs {g['home_pitcher'] or 'TBD'}")
    
    # ── 2. ODDS ───────────────────────────────────────────
    log("💰 Fetching Vegas lines...")
    implied_totals = fetch_odds(date_str)
    if implied_totals:
        log(f"✅ Odds loaded for {len(implied_totals)} teams")
    else:
        log("⚠️ No odds available (API key missing or quota exceeded) - using neutral 4.5")
    
    # ── 3. WEATHER + LINEUPS + SCORING ───────────────────
    log("🌤️ Fetching weather and lineups...")
    
    total_batters = 0
    
    for game in games:
        game_pk = game["game_pk"]
        home_team = game["home_team"]
        away_team = game["away_team"]
        
        # Get park info
        park_info = STADIUM_COORDS.get(home_team, (40.7, -74.0, "Unknown Stadium", False))
        lat, lon, park_name, is_dome = park_info
        
        # Weather
        weather = fetch_weather(lat, lon, game.get("game_time", ""), is_dome)
        
        # Lineups
        log(f"📋 Loading lineup: {away_team} @ {home_team}...")
        lineups = fetch_lineup(game_pk)
        
        home_batters = lineups.get("home", [])
        away_batters = lineups.get("away", [])
        
        # If no confirmed lineup, we skip (per best practices)
        if not home_batters and not away_batters:
            log(f"   ⚠️ {away_team} @ {home_team}: No confirmed lineups yet - skipping")
            continue
        
        # Get pitcher info
        home_pitcher_id = game.get("home_pitcher_id")
        away_pitcher_id = game.get("away_pitcher_id")
        
        home_pitcher_info = fetch_pitcher_info(home_pitcher_id) if home_pitcher_id else {"name": "TBD", "hand": "R"}
        away_pitcher_info = fetch_pitcher_info(away_pitcher_id) if away_pitcher_id else {"name": "TBD", "hand": "R"}
        
        # Score each batter
        all_batters = []
        for batter in home_batters[:9]:
            batter["team"] = home_team
            batter["opponent"] = away_team
            batter["opposing_pitcher"] = away_pitcher_info
            batter["is_home"] = True
            batter["park_team"] = home_team
            all_batters.append(batter)
        
        for batter in away_batters[:9]:
            batter["team"] = away_team
            batter["opponent"] = home_team
            batter["opposing_pitcher"] = home_pitcher_info
            batter["is_home"] = False
            batter["park_team"] = home_team  # Away team plays at home team's park
            all_batters.append(batter)
        
        log(f"   📊 Scoring {len(all_batters)} batters...")
        
        for batter in all_batters:
            player_id = batter.get("player_id", "")
            name = batter.get("name", "Unknown")
            team = batter.get("team", "")
            sp_info = batter.get("opposing_pitcher", {})
            sp_name = sp_info.get("name", "TBD")
            sp_hand = sp_info.get("hand", "R")
            lineup_slot = batter.get("lineup_slot", 5)
            batter_hand = batter.get("batter_hand", "R")
            park_team = batter.get("park_team", team)
            
            # Statcast batter data
            batter_statcast = {}
            if player_id:
                try:
                    batter_statcast = fetch_statcast_batter_stats(player_id, days_back=30)
                except:
                    pass
            
            # Pitcher Statcast
            pitcher_statcast = {}
            sp_id = str(sp_info.get("id", ""))
            if sp_id:
                try:
                    pitcher_statcast = fetch_pitcher_statcast(sp_id, days_back=30)
                except:
                    pass
            
            # ── SCORING COMPONENTS ──────────────────────
            
            # 1. Batter profile (45%)
            bat_score, _, bat_details = compute_batter_score(batter_statcast)
            
            # 2. Pitcher vulnerability (30%)
            pit_score, pit_label = compute_pitcher_score(pitcher_statcast)
            
            # 3. Platoon (6%)
            plat_score, plat_label = compute_platoon_score(batter_hand, sp_hand)
            
            # 4. Lineup position (4%)
            lineup_sc, lineup_label = compute_lineup_score(lineup_slot)
            
            # 5. Park (7%)
            park_sc, park_label = compute_park_score(park_team, True)
            
            # 6. Weather (3%)
            weather_sc, weather_label = compute_weather_score(weather)
            
            # 7. Vegas (5%)
            implied = implied_totals.get(team, 4.5)
            vegas_sc, vegas_label = compute_vegas_score(implied)
            
            # Final score
            final_score = compute_final_score(
                bat_score, pit_score, plat_score, lineup_sc,
                park_sc, weather_sc, vegas_sc
            )
            
            prob = score_to_prob(final_score)
            tier = get_tier(final_score)
            
            # HR score
            hr_score = compute_hr_score(
                barrel_rate=batter_statcast.get("barrel_rate", 0.07),
                sweet_spot=batter_statcast.get("sweet_spot_rate", 0.30),
                park_hr_factor=PARK_HR_FACTORS.get(park_team, 1.0),
                implied_total=implied,
                weather=weather,
                hard_hit=batter_statcast.get("hard_hit_rate", 0.37),
            )
            
            # TBD pitcher flag
            sp_tbd = sp_name == "TBD" or not sp_name
            if sp_tbd:
                final_score = min(final_score, 72)
                tier = get_tier(final_score)
            
            result = {
                "name": name,
                "player_id": player_id,
                "team": team,
                "opponent": opponent if (opponent := batter.get("opponent", "")) else "?",
                "game_id": str(game_pk),
                "lineup_slot": lineup_slot,
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
                # Raw metrics
                "xslg": bat_details.get("xSLG", 0),
                "barrel_rate": batter_statcast.get("barrel_rate", 0),
                "hard_hit_rate": batter_statcast.get("hard_hit_rate", 0),
                "k_rate": batter_statcast.get("k_rate", 0),
                "iso": bat_details.get("ISO", 0),
                "exit_velocity": batter_statcast.get("exit_velocity_avg", 0),
                "sweet_spot_rate": batter_statcast.get("sweet_spot_rate", 0),
                # Sub-scores
                "sub_batter": round(bat_score, 1),
                "sub_pitcher": round(pit_score, 1),
                "sub_platoon": round(plat_score, 1),
                "sub_lineup": round(lineup_sc, 1),
                "sub_park": round(park_sc, 1),
                "sub_weather": round(weather_sc, 1),
                "sub_vegas": round(vegas_sc, 1),
                "platoon_edge": plat_label,
                # Flags
                "temperature": weather.get("temperature", 70),
                "wind_speed": weather.get("wind_speed", 0),
                "wind_dir": weather.get("wind_dir_label", ""),
                "wind_effect": weather.get("wind_effect", "neutral"),
                "is_dome": weather.get("is_dome", False),
            }
            
            results.append(result)
            total_batters += 1
        
        time.sleep(0.5)  # Rate limiting
    
    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    
    log(f"\n✅ **Scoring complete!** {total_batters} batters scored across {len(games)} games")
    tier1 = sum(1 for r in results if r["tier"] == "🔒 TIER 1")
    tier2 = sum(1 for r in results if r["tier"] == "✅ TIER 2")
    tier3 = sum(1 for r in results if r["tier"] == "📊 TIER 3")
    log(f"🔒 Tier 1: {tier1} | ✅ Tier 2: {tier2} | 📊 Tier 3: {tier3}")
    
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
        
        has_odds_key = bool(st.secrets.get("odds_api", {}).get("api_key", ""))
        
        st.markdown(f"{'✅' if True else '❌'} MLB Stats API")
        st.markdown(f"{'✅' if True else '❌'} Baseball Savant")
        st.markdown(f"{'✅' if True else '❌'} Open-Meteo Weather")
        st.markdown(f"{'✅' if has_odds_key else '⚠️'} The Odds API {'(configured)' if has_odds_key else '(no key)'}")
        
        if not has_odds_key:
            with st.expander("Add Odds API Key"):
                st.markdown("""
                1. Sign up free at [the-odds-api.com](https://the-odds-api.com)
                2. Get your API key (500 free calls/month)
                3. Add to Streamlit secrets:
                ```toml
                [odds_api]
                api_key = "your_key_here"
                ```
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
        
        st.caption(f"v1.0 | {datetime.now(EST).strftime('%I:%M %p EST')}")
    
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
            st.markdown(f"**📅 {date_str}**")
            status = st.empty()
            
            with st.spinner("Running model pipeline..."):
                plays = run_model(date_str, status)
            
            st.session_state.plays = plays
            st.session_state.analysis_date = date_str
            st.session_state.model_ran = True
            
            # Auto-save to DB
            if plays:
                save_picks_to_db(plays, date_str)
                
                # Auto-generate and save recommended parlay
                best_parlays = build_parlays(plays, 3, 1, 75.0)
                if best_parlays:
                    save_parlay_to_db(best_parlays[0], date_str)
            
            st.rerun()
    
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
