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
    conn.commit()()
    conn.close()

def fetch_results_for_date(date_str: str) -> Dict:
    """
    Auto-fetch actual total bases for all tiered pending picks on a given date.
    Hits MLB Stats API boxscore endpoint for each game, calculates TB per player,
    updates DB. Returns summary dict with counts.
    Singles=1, Doubles=2, Triples=3, HR=4. Walks/HBP/SB = 0.
    """
    conn = sqlite3.connect(DB_PATH)
    pending = pd.read_sql(
        "SELECT * FROM picks WHERE date=? AND result='pending' AND tier NOT LIKE '%NO PLAY%'",
        conn, params=(date_str,)
    )
    conn.close()

    if pending.empty:
        return {"status": "no_pending", "updated": 0, "skipped": 0, "details": []}

    # Get all unique game_ids for this date
    game_ids = pending["game_id"].dropna().unique().tolist()
    game_ids = [g for g in game_ids if g and str(g).strip() not in ("", "0", "nan")]

    # If no game_ids stored, try fetching from schedule
    if not game_ids:
        try:
            url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=boxscore"
            r = requests.get(url, timeout=10)
            data = r.json()
            for date_obj in data.get("dates", []):
                for game in date_obj.get("games", []):
                    game_ids.append(str(game.get("gamePk", "")))
        except Exception:
            pass

    # Build player_name -> TB map by fetching each boxscore
    player_tb_map = {}  # player_id -> tb, also player_name -> tb as fallback

    for game_id in game_ids:
        if not game_id:
            continue
        try:
            url = f"https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore"
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                continue
            box = r.json()

            for side in ["away", "home"]:
                team_data = box.get("teams", {}).get(side, {})
                players = team_data.get("players", {})
                for pid, pdata in players.items():
                    stats = pdata.get("stats", {}).get("batting", {})
                    if not stats:
                        continue
                    singles = stats.get("hits", 0) - stats.get("doubles", 0) - stats.get("triples", 0) - stats.get("homeRuns", 0)
                    tb = (
                        max(0, singles) * 1 +
                        stats.get("doubles", 0) * 2 +
                        stats.get("triples", 0) * 3 +
                        stats.get("homeRuns", 0) * 4
                    )
                    player_id = str(pdata.get("person", {}).get("id", ""))
                    player_name = pdata.get("person", {}).get("fullName", "")
                    if player_id:
                        player_tb_map[player_id] = {"tb": tb, "name": player_name, "game_id": game_id}
                    if player_name:
                        player_tb_map[player_name.lower()] = {"tb": tb, "name": player_name, "game_id": game_id}
        except Exception:
            continue

    # Now match pending picks and update DB
    updated = 0
    skipped = 0
    postponed = 0
    details = []

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    for _, row in pending.iterrows():
        pick_id = row["pick_id"]
        player_id = str(row.get("player_id", "")).strip()
        player_name = str(row.get("player_name", "")).strip()

        # Try match by player_id first, then name
        match = player_tb_map.get(player_id) or player_tb_map.get(player_name.lower())

        if match is None:
            # Check if game was postponed by seeing if game_id has any data at all
            game_id = str(row.get("game_id", "")).strip()
            if game_id and game_id not in [str(g) for g in game_ids]:
                result_label = "postponed"
                c.execute("UPDATE picks SET result=? WHERE pick_id=?", (result_label, pick_id))
                postponed += 1
                details.append({"name": player_name, "tb": None, "result": "postponed"})
            else:
                skipped += 1
                details.append({"name": player_name, "tb": None, "result": "not_found"})
            continue

        tb = match["tb"]
        result = "hit" if tb >= 2 else "miss"
        c.execute("UPDATE picks SET result=?, tb_actual=? WHERE pick_id=?", (result, tb, pick_id))
        updated += 1
        details.append({"name": player_name, "tb": tb, "result": result})

    conn.commit()
    conn.close()

    return {
        "status": "done",
        "updated": updated,
        "skipped": skipped,
        "postponed": postponed,
        "details": details
    }


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

@st.cache_data(ttl=10800)
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
        "qual": "0",   # No minimum PA — load every MLB player including bench
        "minpa": "1",  # At least 1 PA to filter out pure pitchers
        "season": season, "season1": season,
        "ind": "0", "team": "0", "pageitems": "2000", "pagenum": "1",
        "sortdir": "default",
        "sortstat": "PA",  # Sort by PA so all batters with plate appearances are included
    }
    
    frames = {}
    # Load 2025 full season (primary — large sample, reliable)
    # and 2026 season start (recency signal for hot/cold players + covers returning injured)
    for yr, label in [(season, "primary"), (season + 1, "recent")]:
        for stat_type, slabel in [("8", "adv"), ("24", "sc")]:
            key = f"{label}_{slabel}"
            try:
                # For 2026 (recent): use minpa=0 to catch returning injured players
                # who may have only 1-5 PA in opening week
                yr_params = {**common_params,
                             "season": yr, "season1": yr,
                             "type": stat_type,
                             "sortstat": "WAR" if stat_type == "8" else "xSLG"}
                if label == "recent":
                    yr_params["minpa"] = "0"  # no PA floor for 2026 — catch everyone on roster
                    yr_params["qual"]  = "0"
                r = requests.get(base_url, params=yr_params, headers=headers, timeout=20)
                if r.status_code == 200:
                    rows = r.json().get("data", [])
                    if rows:
                        frames[key] = pd.DataFrame(rows)
            except Exception:
                pass
    
    # Pybaseball fallback if all API calls failed
    if not frames:
        try:
            from pybaseball import batting_stats
            df = batting_stats(season, qual=1)
            if df is not None and not df.empty:
                frames["pybaseball"] = df
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    # Strategy: prefer 2026 data where available (recency), supplement with 2025
    # If 2026 advanced data loaded, use it as base (covers returning injured players)
    if "recent_adv" in frames and len(frames["recent_adv"]) > 100:
        # 2026 has meaningful data — use as primary, supplement with 2025 for players missing
        base_key = "recent_adv"
        fallback_key = "primary_adv"
        # Union merge: 2026 players get 2026 stats, 2025-only players still included
        if fallback_key in frames:
            try:
                for k in ["playerid", "PlayerID", "IDfg", "xMLBAMID"]:
                    if k in frames[base_key].columns and k in frames[fallback_key].columns:
                        # Outer join: keeps all players from both seasons
                        result = frames[fallback_key].copy()
                        new_cols_2026 = [c for c in frames[base_key].columns
                                         if c not in result.columns or c == k]
                        merged = result.merge(
                            frames[base_key][[k] + [c for c in new_cols_2026 if c != k]],
                            on=k, how="outer", suffixes=("_2025", "_2026")
                        )
                        result = merged
                        break
                else:
                    result = frames[base_key].copy()
            except Exception:
                result = frames.get("primary_adv", frames[base_key]).copy()
        else:
            result = frames[base_key].copy()
    else:
        # No meaningful 2026 data yet — use 2025 as primary
        base_key = "primary_adv" if "primary_adv" in frames else list(frames.keys())[0]
        result = frames[base_key].copy()

    # Merge in remaining frames (statcast columns etc)
    for key, df in frames.items():
        if df is result or key in ("primary_adv", "recent_adv"):
            continue
        try:
            for k in ["playerid", "PlayerID", "IDfg", "xMLBAMID", "Name"]:
                if k in result.columns and k in df.columns:
                    suffix = "_2026" if "recent" in key else "_sc" if "_sc" in key else "_x"
                    new_cols = [c for c in df.columns if c not in result.columns]
                    if new_cols:
                        result = result.merge(df[[k] + new_cols], on=k, how="left",
                                              suffixes=("", suffix))
                    break
        except Exception:
            pass

    return clean_fangraphs_df(result)


@st.cache_data(ttl=10800)
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
    
    # Pre-convert xMLBAMID to string int for fast matching
    if "xMLBAMID" in df.columns:
        df["xMLBAMID"] = df["xMLBAMID"].apply(
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

    # 1. xMLBAMID match — most reliable (FanGraphs xMLBAMID = MLB MLBAM player_id)
    if mlb_id and "xMLBAMID" in df.columns:
        try:
            m = df[df["xMLBAMID"].astype(str).str.split(".").str[0] == str(mlb_id)]
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
        if ev and ev > 50:
            stats["exit_velocity_avg"] = ev

        # Sweet spot% (launch angle 8-32 degrees) — key for HR model
        sweet = safe_get(row, 'Sweetspot%', 'sweet_spot_percent', default=None)
        if sweet is not None and sweet > 0:
            stats["sweet_spot_rate"] = sweet if sweet < 1 else sweet / 100

        # wRC+ — store explicitly for downstream O0.5/PP models
        if wrc and wrc > 0:
            stats["wrc_plus"] = float(wrc)

        # 2026 recent form blending — if 2026 YTD stats available, blend 70/30
        # 2026 columns have _2026 suffix after the outer join merge
        xslg_2026 = safe_get(row, 'xSLG_2026', default=None)
        if xslg_2026 and 0.100 < float(xslg_2026) < 1.000:
            # 60% 2025 season + 40% 2026 YTD (recency weighted)
            stats["slg_proxy"] = stats["slg_proxy"] * 0.60 + float(xslg_2026) * 0.40

        barrel_2026 = safe_get(row, 'Barrel%_2026', default=None)
        if barrel_2026 and float(barrel_2026) > 0:
            b26 = float(barrel_2026) if float(barrel_2026) < 1 else float(barrel_2026) / 100
            stats["barrel_rate"] = stats["barrel_rate"] * 0.65 + b26 * 0.35

        k_2026 = safe_get(row, 'K%_2026', default=None)
        if k_2026 and float(k_2026) > 0:
            k26 = float(k_2026) if float(k_2026) < 1 else float(k_2026) / 100
            stats["k_rate"] = stats["k_rate"] * 0.65 + k26 * 0.35

        stats["data_source"] = "fangraphs"

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
        "barrel_allowed":   0.065,
        "era":              4.20,
        "fip":              4.10,
        "xfip":             4.10,
        "whip":             1.30,
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

    # Weighted composite — research-calibrated sub-weights
    # Barrel% leads: r=0.93 with HR, better XBH predictor than xSLG
    # xSLG secondary: r=0.87 Y-to-Y, best single contact quality metric
    # HH% up slightly: r²=0.67 Y-to-Y, more predictive than ISO for TB
    composite = (
        barrel_score  * 0.26 +   # barrel% — strongest XBH/HR predictor (r=0.93 w/ HR)
        xslg_score    * 0.22 +   # xSLG — overall contact quality (r=0.87 Y-to-Y)
        wrc_score     * 0.18 +   # wRC+ — offensive context (r=0.78 Y-to-Y)
        hard_hit_score* 0.17 +   # hard hit% — sustainable contact (r²=0.67 Y-to-Y)
        iso_score     * 0.10 +   # ISO — raw power (r=0.71 Y-to-Y)
        k_score       * 0.07     # K% inverse — PA completion rate
    )

    return max(0, min(100, composite)), "Contact quality", details


def compute_pitcher_score(statcast: Dict, fg_stats: Dict = None) -> Tuple[float, str]:
    """
    Pitcher VULNERABILITY score 0-100.
    HIGH score = pitcher is hittable = good for batter TB.
    LOW score = elite pitcher = suppresses TB.
    Webb/Fried type = ~20-30. League avg pitcher = ~50. Mop-up arm = ~75+.
    Blends SP stats (60%) with league-avg bullpen (40%) to account for
    bullpen innings batters will face after SP exit.
    """
    def f(key, default):
        try: return float(statcast.get(key, default))
        except: return float(default)

    k_rate   = f("k_rate_allowed",   0.228)
    hard_hit = f("hard_hit_allowed", 0.340)
    barrel   = f("barrel_allowed",   0.065)
    era      = f("era",              4.20)
    fip      = f("fip",              4.10)
    whip     = f("whip",             1.30)

    # K% INVERSE — high K pitcher = low vulnerability
    k_vuln = max(0, min(100, (0.35 - k_rate) / (0.35 - 0.10) * 100))

    # Hard hit allowed
    hh_vuln = max(0, min(100, (hard_hit - 0.28) / (0.50 - 0.28) * 100))

    # Barrel% allowed
    barrel_vuln = max(0, min(100, (barrel - 0.03) / (0.14 - 0.03) * 100))

    # ERA/FIP quality
    era_use = fip if fip > 0 else era
    era_vuln = max(0, min(100, (era_use - 2.0) / (7.0 - 2.0) * 100))

    # WHIP — contacts + walks per inning (new signal)
    # 0.90=0, 1.30=50, 1.80=100
    whip_vuln = max(0, min(100, (whip - 0.90) / (1.80 - 0.90) * 100))

    # SP composite
    sp_score = (
        k_vuln      * 0.38 +
        hh_vuln     * 0.22 +
        barrel_vuln * 0.18 +
        era_vuln    * 0.14 +
        whip_vuln   * 0.08    # WHIP adds baserunner context
    )

    # Bullpen blend: avg team bullpen FIP ~4.5, K% ~23%, WHIP ~1.35
    # Batters see ~40% of PAs against bullpen (3-4 IP out of 9)
    bp_vuln = 42.0  # league avg bullpen vulnerability
    blended = sp_score * 0.60 + bp_vuln * 0.40

    label = f"K%: {k_rate*100:.0f}% | WHIP: {whip:.2f} | FIP: {era_use:.2f}"
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




def compute_final_score(
    batter_score: float,
    pitcher_vuln_score: float,
    platoon_score: float,
    lineup_score: float,
    park_score: float,
    weather_score: float,
    vegas_score: float,
    tto_bonus: float = 0.0,
) -> float:
    """
    Final weighted composite. Research-calibrated weights.
    TTO bonus added as additional signal on top of base score.
    
    Weight rationale:
    - Batter 43%: dominant signal, most Y-to-Y stability
    - Pitcher 25%: real but less predictive than batter for TB props
      (pitchers suppress scoring but XBH vs them is less predictable than Ks)
    - Platoon 7%: +56 SLG effect is large and well-documented
    - Weather 4%: wind 15+ mph = ~12% more HRs (material)
    - Park 7%: Coors/GABP vs pitcher's parks = real difference
    - TTO 5%: 3rd TTO +17-20 wOBA is well-documented
    - Vegas 5%: team total correlates r=0.61 with scoring
    - Lineup 4%: 1 extra PA from slot 1 vs 9 = real but small
    """
    raw = (
        batter_score      * 0.43 +
        pitcher_vuln_score* 0.25 +   # 28->25: pitcher matters less for TB than K props
        platoon_score     * 0.07 +   # 6->7: +56 SLG effect is large, deserves more
        lineup_score      * 0.04 +
        park_score        * 0.07 +
        weather_score     * 0.04 +   # 3->4: wind 15+ mph = 12% more HRs
        vegas_score       * 0.05 +
        tto_bonus         * 0.05    # 4->5: 3rd TTO +17-20 wOBA is well-documented
    )
    # Calibration offset: raw league-avg matchup → target ~52
    calibrated = raw + 10.0
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


def get_tier(score: float) -> str:
    """Map score to tier label."""
    if score >= 80:
        return "🔒 TIER 1"
    elif score >= 70:
        return "✅ TIER 2"
    elif score >= 60:
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
    # Reset match tracking
    st.session_state["_matched"] = 0
    st.session_state["_unmatched"] = 0
    st.session_state["_search_names"] = []

    # ── 0. BULK STATS (one call loads all players) ───────
    log("Loading 2025 season batting stats...", "run")
    batting_df = load_all_batting_stats(2025)
    pitching_df = load_all_pitching_stats(2025)
    statcast_df = pd.DataFrame()  # merged into batting_df now

    if not batting_df.empty:
        log(f"Batting stats: {len(batting_df)} players loaded", "ok")
        batting_df = prepare_lookup_df(batting_df)  # build _norm_name index ONCE
        # Store debug info
        st.session_state.batting_cols = list(batting_df.columns)
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
        log("Batting stats unavailable — all scores use league averages", "warn")
    if not pitching_df.empty:
        log(f"Pitching stats: {len(pitching_df)} pitchers loaded", "ok")
        pitching_df = prepare_lookup_df(pitching_df)  # build _norm_name index ONCE
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
                xmlbam_sample = batting_df["xMLBAMID"].head(3).tolist() if has_xmlbamid else []
                st.session_state["lookup_diag"] = {
                    "searching_for": f"{name} (MLBAM id={player_id})",
                    "batting_df_rows": n_rows,
                    "name_col_used": nc,
                    "first_3_norm_names": [str(v) for v in sample_vals],
                    "xMLBAMID_exists": has_xmlbamid,
                    "xMLBAMID_sample": [str(v) for v in xmlbam_sample],
                    "data_source": batter_statcast.get("data_source"),
                    "matched": batter_statcast.get("data_source") == "fangraphs",
                }

            # Store first 5 searched names for debug
            if "_search_names" not in st.session_state:
                st.session_state["_search_names"] = []
            if len(st.session_state["_search_names"]) < 5:
                st.session_state["_search_names"].append(f"'{name}' (id={player_id})")
            st.session_state["search_sample"] = st.session_state["_search_names"]

            # Track match rate
            if batter_statcast.get("data_source") == "fangraphs":
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

            # ── SCORE COMPONENTS ────────────────────────
            bat_score, _, bat_details = compute_batter_score(batter_statcast)
            pit_score, pit_label = compute_pitcher_score(pitcher_statcast)
            plat_score, plat_label = compute_platoon_score(batter_hand, sp_hand)
            lineup_sc, lineup_label = compute_lineup_score(lineup_slot)
            park_sc, park_label = compute_park_score(park_team, True)
            weather_sc, weather_label = compute_weather_score(weather)
            implied = implied_totals.get(team, 0)
            vegas_sc, vegas_label = compute_vegas_score(implied)
            tto_sc, tto_label = compute_tto_bonus(lineup_slot)

            final_score = compute_final_score(
                bat_score, pit_score, plat_score, lineup_sc,
                park_sc, weather_sc, vegas_sc, tto_sc
            )

            # Caps & flags
            sp_tbd = not sp_name or sp_name == "TBD"
            if sp_tbd:
                final_score = min(final_score, 72)
            if not batter.get("lineup_confirmed", True):
                final_score = min(final_score, 70)

            prob = score_to_prob(final_score)
            tier = get_tier(final_score)

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

        styled = df.style.applymap(color_tier, subset=["Tier"]).applymap(color_score, subset=["Score"])
        if "Edge" in df.columns:
            styled = styled.applymap(color_edge, subset=["Edge"])
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
            pitcher_mock = {"k_rate_allowed": 0.228, "era": 4.20, "fip": 4.10, "whip": 1.30, "hard_hit_allowed": 0.340}
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
            pitcher_mock = {"k_rate_allowed": 0.228, "era": 4.20, "fip": 4.10, "whip": 1.30, "hard_hit_allowed": 0.340}
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

    # K rate: MOST critical for O0.5 — K = 0 hits, period
    # 8%=100, 20%=60, 35%=0  (tighter range — even "good" K% matters)
    k_score = max(0, min(100, (0.35 - k_rate) / (0.35 - 0.08) * 100))

    # wRC+: 60=0, 100=50, 170=100
    wrc_score = max(0, min(100, (wrc_plus - 60) / (170 - 60) * 100))

    # BB%: 3%=0, 8%=50, 18%=100 (discipline = PA completion = more chances)
    bb_score = max(0, min(100, (bb_rate - 0.03) / (0.18 - 0.03) * 100))

    # wOBA: .250=0, .315=50, .420=100
    woba_score = max(0, min(100, (woba - 0.250) / (0.420 - 0.250) * 100))

    # Hard hit: 28%=0, 38%=50, 56%=100 (hard contact harder to field = more hits)
    hh_score = max(0, min(100, (hard_hit - 0.28) / (0.56 - 0.28) * 100))

    # xSLG: still matters but less — any hit counts
    # .200=0, .398=50, .600=100
    xslg_score = max(0, min(100, (xslg - 0.200) / (0.600 - 0.200) * 100))

    # Weights: K% dominates, then contact/discipline metrics
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

    k_rate  = f("k_rate_allowed",   0.228)
    era     = f("era",              4.20)
    fip     = f("fip",              4.10)
    whip    = f("whip",             1.30)
    hard_hit= f("hard_hit_allowed", 0.340)

    # K% inverse — most critical for O0.5
    # 10%=100, 22%=50, 35%=0
    k_vuln = max(0, min(100, (0.35 - k_rate) / (0.35 - 0.10) * 100))

    # WHIP: 0.90=0, 1.30=50, 1.80=100 (more baserunners = pitcher hittable)
    whip_vuln = max(0, min(100, (whip - 0.90) / (1.80 - 0.90) * 100))

    # ERA/FIP: 2.0=0, 4.2=50, 7.0=100
    era_use = fip if fip > 0 else era
    era_vuln = max(0, min(100, (era_use - 2.0) / (7.0 - 2.0) * 100))

    # Hard hit allowed (secondary for O0.5)
    hh_vuln = max(0, min(100, (hard_hit - 0.26) / (0.50 - 0.26) * 100))

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
    if score >= 78:
        return "🔒 SAFE+"
    elif score >= 68:
        return "✅ SAFE"
    elif score >= 58:
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
) -> Tuple[float, float, str, Dict]:
    """
    Compute O0.5 score, probability, tier, and details for a single player.
    Returns (score, prob, tier, details_dict).
    Calibrated: avg batter vs avg pitcher = ~58, elite bat vs weak pitcher = 75+
    """
    # Inject wRC+ proxy from xSLG if not present (fixes default=100 issue)
    enriched = dict(batter_statcast)
    if enriched.get("wrc_plus", 100) == 100.0:
        xslg = float(enriched.get("slg_proxy", 0.398))
        enriched["wrc_plus"] = max(40, min(220, 100 + (xslg - 0.398) / 0.005))

    bat_score, _, bat_details = compute_batter_score_hits(enriched)
    pit_score, pit_label = compute_pitcher_score_hits(pitcher_statcast)
    plat_score, plat_label = compute_platoon_score(batter_hand, sp_hand)
    lineup_sc, lineup_label = compute_lineup_score(lineup_slot)
    park_sc, _ = compute_park_score(park_team, True)
    weather_sc, _ = compute_weather_score(weather)
    vegas_sc, _ = compute_vegas_score(implied)
    tto_sc, _ = compute_tto_bonus(lineup_slot)

    # O0.5 weights: K% dominates, park matters less than O1.5
    raw = (
        bat_score  * 0.42 +
        pit_score  * 0.32 +
        plat_score * 0.05 +
        lineup_sc  * 0.05 +
        park_sc    * 0.04 +
        weather_sc * 0.02 +
        vegas_sc   * 0.05 +
        tto_sc     * 0.05
    )
    # Calibration offset: avg batter (bat=40) vs avg pitcher (pit=45) raw ~= 43
    # Target: avg matchup = ~58, Tiers start at 58/68/78
    score = max(0, min(100, round(raw + 15.0, 1)))

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
            "k_rate_allowed": 0.228, "era": 4.20, "fip": 4.10,
            "whip": 1.30, "hard_hit_allowed": 0.340,
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

        styled = df.style.applymap(color_htier, subset=["Tier"]).applymap(color_hscore, subset=["H-Score"])
        st.dataframe(styled, use_container_width=True, height=500)

        csv = df.to_csv(index=False)
        st.download_button("📥 Export O0.5 Plays", csv,
                           f"o05_picks_{datetime.now(EST).strftime('%Y%m%d')}.csv", "text/csv")

    # Top plays detail
    st.markdown("---")
    st.subheader("🏆 Top O0.5 Plays — Full Breakdown")
    top = [p for p in filtered if p["h_score"] >= 65][:5]
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
            st.download_button("📥 Export All Picks", csv, "mlb_picks_history.csv", "text/csv")
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
    pit_hh   = float(pitcher_statcast.get("hard_hit_allowed", 0.340))
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
            "k_rate_allowed":  0.228,
            "hard_hit_allowed":0.340,
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
                  .applymap(color_proj, subset=["FD Proj","Ceiling"])
                  .applymap(color_own,  subset=["Own%"])
                  .applymap(color_val,  subset=["Value"]))
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
                     .applymap(color_sp_grade, subset=["Grade"])
                     .applymap(color_sp_proj,  subset=["FD Proj","Ceiling"])
                     .applymap(color_sp_val,   subset=["Value"]))
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

    k_rate   = f("k_rate",        0.228)
    bb_rate  = f("bb_rate",       0.082)
    slg      = f("slg_proxy",     0.398)
    iso      = f("iso_proxy",     0.165)
    woba     = f("woba",          0.315)
    hard_hit = f("hard_hit_rate", 0.370)
    barrel   = f("barrel_rate",   0.070)
    # Sprint speed proxy for SB (default avg runner ~0.05 SB/game)
    # Will be refined when sprint speed data added; power hitters ~0.02, speedsters ~0.15
    sb_rate  = max(0, min(0.20, (iso - 0.100) * -0.3 + 0.05))  # inverse of ISO = more speed

    hit_rate   = max(0.180, woba * 0.85)
    hr_per_pa  = barrel * 0.35
    xbh_per_pa = iso * 0.6 * (1 - k_rate)
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

    # Pitcher quality
    pit_k   = float(pitcher_statcast.get("k_rate_allowed", 0.228))
    pit_fip = float(pitcher_statcast.get("fip", 4.10))
    quality_adj = 1.0 + (pit_fip - 4.0) * 0.04
    k_adj       = 1.0 - (pit_k - 0.228) * 1.5
    adj = (quality_adj + k_adj) / 2

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
        }
        pit_mock = {"k_rate_allowed": 0.228, "fip": 4.10}
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
                  .applymap(color_proj, subset=["PP Proj","Ceiling"])
                  .applymap(color_conf, subset=["Conf"]))
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


def display_results_tracker():
    """Display performance tracking and historical results."""
    st.header("📈 Results Tracker")

    # ── ONE-CLICK FETCH RESULTS ───────────────────────────────────────────────
    st.subheader("🔁 Fetch Results")
    st.caption("Select a date and pull actual total bases from MLB Stats API for all tiered picks.")

    col_d, col_b, col_r = st.columns([2, 1, 3])
    with col_d:
        fetch_date = st.date_input(
            "Results date",
            value=datetime.now(EST).date() - timedelta(days=1),
            key="fetch_date_input"
        )
    with col_b:
        st.write("")
        st.write("")
        fetch_clicked = st.button("⚡ Fetch Results", type="primary", use_container_width=True)

    if fetch_clicked:
        fetch_date_str = fetch_date.strftime("%Y-%m-%d")
        with col_r:
            st.write("")
            st.write("")
            with st.spinner(f"Fetching MLB box scores for {fetch_date_str}..."):
                summary = fetch_results_for_date(fetch_date_str)

        if summary["status"] == "no_pending":
            st.info(f"No pending tiered picks found for {fetch_date_str}. Run the model for that date first, or all picks are already resolved.")
        else:
            updated = summary["updated"]
            skipped = summary["skipped"]
            postponed = summary.get("postponed", 0)
            details = summary["details"]

            if updated > 0:
                hits = sum(1 for d in details if d["result"] == "hit")
                misses = sum(1 for d in details if d["result"] == "miss")
                st.success(f"✅ Updated {updated} picks — {hits} HIT / {misses} MISS")
            if postponed > 0:
                st.warning(f"⚠️ {postponed} pick(s) marked postponed — game did not occur")
            if skipped > 0:
                st.warning(f"⚠️ {skipped} pick(s) not found in box scores — player may not have appeared")

            # Show per-pick breakdown
            if details:
                rows = []
                for d in details:
                    if d["result"] == "hit":
                        icon = "✅ HIT"
                    elif d["result"] == "miss":
                        icon = "❌ MISS"
                    elif d["result"] == "postponed":
                        icon = "⏸️ PPD"
                    else:
                        icon = "❓ N/A"
                    rows.append({
                        "Player": d["name"],
                        "TB": d["tb"] if d["tb"] is not None else "—",
                        "Result": icon
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.rerun()

    st.markdown("---")

    # ── LOAD PICKS ────────────────────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()

    try:
        picks_df = pd.read_sql("SELECT * FROM picks ORDER BY date DESC, model_score DESC", conn)
        parlays_df = pd.read_sql("SELECT * FROM parlays ORDER BY date DESC", conn)
    except Exception:
        picks_df = pd.DataFrame()
        parlays_df = pd.DataFrame()
    conn.close()

    if picks_df.empty:
        st.info("📊 No data yet. Run the model and log results to start tracking.")
        return

    # ── OVERALL PERFORMANCE ───────────────────────────────────────────────────
    resolved = picks_df[picks_df["result"].isin(["hit", "miss"])]
    if not resolved.empty:
        total_hits = len(resolved[resolved["result"] == "hit"])
        total = len(resolved)
        hit_rate = total_hits / total if total > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Overall Record", f"{total_hits}-{total - total_hits}")
        with col2: st.metric("Hit Rate", f"{hit_rate*100:.1f}%")
        with col3:
            t1 = resolved[resolved["tier"] == "🔒 TIER 1"]
            r = len(t1[t1["result"]=="hit"]) / len(t1) if len(t1) > 0 else 0
            st.metric("Tier 1 Hit%", f"{r*100:.1f}%" if len(t1) > 0 else "—")
        with col4:
            t2 = resolved[resolved["tier"] == "✅ TIER 2"]
            r = len(t2[t2["result"]=="hit"]) / len(t2) if len(t2) > 0 else 0
            st.metric("Tier 2 Hit%", f"{r*100:.1f}%" if len(t2) > 0 else "—")

    st.markdown("---")

    # ── PICK LOG ──────────────────────────────────────────────────────────────
    st.subheader("📋 Pick Log")

    col1, col2 = st.columns(2)
    with col1:
        date_start = st.date_input("From", value=datetime.now(EST).date() - timedelta(days=30), key="rt_start")
    with col2:
        date_end = st.date_input("To", value=datetime.now(EST).date(), key="rt_end")

    filtered = picks_df[
        (picks_df["date"] >= str(date_start)) &
        (picks_df["date"] <= str(date_end))
    ]
    if not filtered.empty:
        display_cols = ["date","player_name","team","opponent","sp_name",
                        "lineup_slot","model_score","tier","result","tb_actual","implied_total"]
        avail = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[avail], use_container_width=True)

    st.markdown("---")

    # ── MANUAL OVERRIDE (fallback) ────────────────────────────────────────────
    with st.expander("✏️ Manual Result Entry (fallback if auto-fetch misses a player)"):
        st.caption("Use this only if the auto-fetch above didn't find a player's result.")
        pending = picks_df[picks_df["result"] == "pending"]
        if not pending.empty:
            opts = {f"{r['player_name']} ({r['team']}) - {r['date']}": r["pick_id"]
                    for _, r in pending.head(30).iterrows()}
            sel = st.selectbox("Select pick:", list(opts.keys()))
            col1, col2, col3 = st.columns(3)
            with col1:
                tb = st.number_input("Actual TB", 0, 16, 0)
            with col2:
                result = "hit" if tb >= 2 else "miss"
                st.metric("Result", result.upper())
            with col3:
                if st.button("💾 Save"):
                    update_pick_result(opts[sel], result, tb)
                    st.success(f"✅ {tb} TB = {result.upper()}")
                    st.rerun()
        else:
            st.caption("No pending picks.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if not picks_df.empty:
            st.download_button("📥 Export Picks", picks_df.to_csv(index=False),
                               "mlb_picks.csv", "text/csv")
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
            matched = st.session_state.get("_matched", 0)
            unmatched = st.session_state.get("_unmatched", 0)
            total = matched + unmatched
            if total > 0:
                rate = matched / total * 100
                color = "✅" if rate > 80 else "⚠️" if rate > 50 else "❌"
                st.markdown(f"**{color} Player match rate: {matched}/{total} ({rate:.0f}%)**")
                if rate < 80:
                    st.warning("Low match rate — scores using league averages for unmatched players")
            
            # Show raw name samples from DataFrame so we can diagnose lookup failures
            if "lookup_diag" in st.session_state:
                st.markdown("**🔬 Live lookup diagnostic (first batter):**")
                st.json(st.session_state.lookup_diag)
                st.markdown("**Raw _name values in batting DataFrame (first 10):**")
                st.code("\n".join(st.session_state.batting_df_sample))
            if "norm_name_sample" in st.session_state:
                st.markdown("**Normalized _norm_name values (first 10):**")
                st.code("\n".join(st.session_state.norm_name_sample))
            if "search_sample" in st.session_state:
                st.markdown("**Names we searched for (first 5):**")
                st.code("\n".join(st.session_state.search_sample))
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

        st.caption(f"v1.2 | {datetime.now(EST).strftime('%I:%M %p EST')}")
    
    # ── MAIN CONTENT ──────────────────────────────────────
    st.title("⚾ MLB Total Bases Analyzer V1.0")
    st.caption("Fully automated over 1.5 TB prop model | HardRock Bet | 1B=1 2B=2 3B=3 HR=4")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 O1.5 Leaderboard",
        "🎯 Tiered Breakdown",
        "💰 Parlay Builder",
        "🎯 O0.5 Any Hit",
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
        if st.session_state.plays:
            display_hr_plays(st.session_state.plays)
        else:
            st.info("Run the model first to see HR plays.")

    with tab6:
        display_dfs_tab(st.session_state.plays if st.session_state.plays else [])

    with tab7:
        display_prizepicks_tab(st.session_state.plays if st.session_state.plays else [])

    with tab8:
        display_results_tracker()


if __name__ == "__main__":
    main()
