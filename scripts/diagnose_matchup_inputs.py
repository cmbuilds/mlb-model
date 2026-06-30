"""
diagnose_matchup_inputs.py — DIAGNOSTICS ONLY, no fixes.

For 5 batters in good matchup spots today, break down every O1.5
component score and show whether it's a real per-game value or a
neutral/league-average default.

Components inspected:
  batter_score    — from compute_batter_score() with real SQLite stats
  pitcher_vuln    — from compute_pitcher_score() + bullpen; SP matched or league_avg?
  park_score      — static PARK_HR_FACTORS lookup by home team
  weather_score   — from Open-Meteo per-game data (or default if unavailable)
  vegas_score     — from Odds API implied total (or 0.0 if no API key)
  platoon_score   — batter hand vs SP hand (real or defaulted?)
  lineup_score    — real slot or default?
  matchup_score   — pitch-type matchup (real or league-avg pitch mix?)
  streak_score    — MLB API last-7 games (live call)
  bvp_score       — MLB API batter-vs-pitcher (live call)
  bullpen_vuln    — team_bullpen_scores[opp_team] (real from rebuilt db)
"""

import sys, os, json, re, unicodedata
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import sqlite3
import pandas as pd
import requests

from scoring.park    import compute_park_score, compute_platoon_score, \
                            compute_lineup_score, compute_pitch_matchup_score
from scoring.weather import compute_weather_score
from scoring.vegas   import compute_vegas_score
from scoring.streak  import compute_streak_score, compute_bvp_score, compute_tto_bonus
from scoring.batter  import compute_batter_score
from scoring.pitcher import compute_pitcher_score, compute_team_bullpen_scores
from scoring.final   import compute_final_score
from config          import TIERS_FULL

DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mlb_stats.db")

# ── helpers ───────────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

def safe_float(v, default=None):
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return default

def find_row(df, name, mlb_id=""):
    """Simplified version of the monolith's find_player_row()."""
    if df is None or df.empty:
        return None
    for id_col in ("xMLBAMID", "MLBAMID", "_mlb_id"):
        if mlb_id and id_col in df.columns:
            try:
                m = df[df[id_col].astype(str).str.split(".").str[0] == str(mlb_id)]
                if not m.empty:
                    return m.iloc[0]
            except Exception:
                pass
    norm = _norm(name)
    if "_norm_name" in df.columns:
        m = df[df["_norm_name"] == norm]
        if not m.empty:
            return m.iloc[0]
        parts = norm.split()
        last = parts[-1] if parts else ""
        first = parts[0] if len(parts) > 1 else ""
        if last:
            cands = df[df["_norm_name"].str.contains(last, na=False, regex=False)]
            if not cands.empty:
                if first:
                    refined = cands[cands["_norm_name"].str.contains(first[:3], na=False)]
                    if not refined.empty:
                        return refined.iloc[0]
                if len(cands) == 1:
                    return cands.iloc[0]
    return None

def get_sp_stats(pitcher_name, pitcher_mlb_id, pitching_df):
    """Mirrors get_pitcher_stats() from the monolith."""
    defaults = {
        "k_rate_allowed": 0.228, "bb_rate_allowed": 0.082,
        "hard_hit_allowed": 0.360, "barrel_allowed": 0.070,
        "era": 4.20, "fip": 4.20, "xfip": 4.10, "whip": 1.30,
        "pct_FF": 0.35, "pct_SI": 0.17, "pct_SL": 0.20,
        "pct_CH": 0.10, "pct_CU": 0.11, "pct_FC": 0.07,
        "data_source": "league_avg",
    }
    prov = {k: "league_avg" for k in ["k_rate_allowed","bb_rate_allowed",
            "hard_hit_allowed","barrel_allowed","era","fip","whip","swstr_pct"]}
    defaults["_provenance"] = prov

    row = find_row(pitching_df, pitcher_name, pitcher_mlb_id)
    if row is None:
        defaults["swstr_pct"] = round(defaults["k_rate_allowed"] * 0.49, 4)
        defaults["data_source"] = "league_avg — NO MATCH"
        return defaults

    def sg(*cols):
        for c in cols:
            v = safe_float(row.get(c))
            if v is not None:
                return v
        return None

    k = sg("K%"); bb = sg("BB%"); hard = sg("Hard%"); barrel = sg("Barrel%")
    era = sg("ERA"); fip = sg("FIP","xFIP"); whip = sg("WHIP")

    if k and k > 0:
        defaults["k_rate_allowed"] = k if k < 1 else k/100; prov["k_rate_allowed"] = "measured"
    if bb and bb > 0:
        defaults["bb_rate_allowed"] = bb if bb < 1 else bb/100; prov["bb_rate_allowed"] = "measured"
    if hard and hard > 0:
        defaults["hard_hit_allowed"] = hard if hard < 1 else hard/100; prov["hard_hit_allowed"] = "measured"
    if barrel and barrel > 0:
        defaults["barrel_allowed"] = barrel if barrel < 1 else barrel/100; prov["barrel_allowed"] = "measured"
    if era and 0 < era < 20:
        defaults["era"] = era; prov["era"] = "measured"
    if fip and 0 < fip < 20:
        defaults["fip"] = fip; prov["fip"] = "measured"
    if whip and 0 < whip < 5:
        defaults["whip"] = whip; prov["whip"] = "measured"

    defaults["data_source"] = "db_matched"
    k_eff = defaults["k_rate_allowed"]
    defaults["swstr_pct"] = round(k_eff * 0.49, 4)
    return defaults

def fetch_today_games():
    """Fetch today's schedule with probable pitchers from MLB Stats API."""
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId": 1, "hydrate": "probablePitcher,team,venue,weather", "date": "2026-06-30"}
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    games = []
    for date_block in r.json().get("dates", []):
        for g in date_block.get("games", []):
            home = g.get("teams",{}).get("home",{})
            away = g.get("teams",{}).get("away",{})
            home_abb = home.get("team",{}).get("abbreviation","?")
            away_abb = away.get("team",{}).get("abbreviation","?")
            home_sp  = home.get("probablePitcher",{})
            away_sp  = away.get("probablePitcher",{})
            venue    = g.get("venue",{}).get("name","?")
            games.append({
                "game_pk":  g.get("gamePk"),
                "home":     home_abb,
                "away":     away_abb,
                "venue":    venue,
                "home_sp_name": home_sp.get("fullName","TBD"),
                "home_sp_id":   str(home_sp.get("id","")),
                "away_sp_name": away_sp.get("fullName","TBD"),
                "away_sp_id":   str(away_sp.get("id","")),
            })
    return games

def batter_row_to_stats(row):
    """Map a SQLite batter_stats row to the dict compute_batter_score() expects."""
    def g(key, default=0.0):
        v = row.get(key)
        try:
            return float(v) if v is not None and str(v) not in ("","nan","None") else default
        except (TypeError, ValueError):
            return default

    barrel   = g("barrel_batted_rate", 0.070)
    hard_hit = g("hard_hit_percent",   0.370)
    k_pct    = g("K%",  0.228)
    if barrel > 1:  barrel   /= 100.0
    if hard_hit > 1: hard_hit /= 100.0
    if k_pct > 1:   k_pct    /= 100.0

    return {
        "slg_proxy":    g("xSLG", 0.398),
        "barrel_rate":  barrel,
        "hard_hit_rate": hard_hit,
        "k_rate":       k_pct,
        "iso_proxy":    g("ISO", 0.165),
        "wrc_plus":     g("wRC+", 100.0),
        "woba":         g("wOBA", 0.315),
        "ev50":         g("ev50", 0.0),
        "bat_speed":    g("bat_speed", 0.0),
        "blast_rate":   g("blast_rate", 0.0),
        "data_source":  str(row.get("data_source","league_avg")),
        "_provenance":  {},
    }

# ─── STEP 1: Load db ──────────────────────────────────────────────────────────
print("=" * 68)
print("STEP 1 — Load SQLite db")
print("=" * 68)

con = sqlite3.connect(DB)
bat_df  = pd.read_sql("SELECT * FROM batter_stats",  con)
pit_df  = pd.read_sql("SELECT * FROM pitcher_stats", con)
con.close()

# Build norm name index
bat_df["_norm_name"] = bat_df.get("_name", bat_df.get("Name","")).apply(_norm)
pit_df["_norm_name"] = pit_df.get("_name", pit_df.get("Name","")).apply(_norm)

print(f"  batter_stats:  {len(bat_df)} rows  cols: barrel_batted_rate={'barrel_batted_rate' in bat_df.columns} hard_hit_percent={'hard_hit_percent' in bat_df.columns} wRC+={'wRC+' in bat_df.columns}")
print(f"  pitcher_stats: {len(pit_df)} rows  cols: Team={'Team' in pit_df.columns} FIP={'FIP' in pit_df.columns} K%={'K%' in pit_df.columns} WHIP={'WHIP' in pit_df.columns}")
print(f"  pitcher_stats xMLBAMID col present: {'xMLBAMID' in pit_df.columns}")
print(f"  pitcher_stats _mlb_id  col present: {'_mlb_id' in pit_df.columns}")

# ─── STEP 2: Fetch today's games and SP match rate ────────────────────────────
print()
print("=" * 68)
print("STEP 2 — Today's probable SPs and their db match status")
print("=" * 68)

try:
    games = fetch_today_games()
    print(f"  Found {len(games)} games today (2026-06-30)")
except Exception as e:
    print(f"  ⚠ MLB API fetch failed: {e}")
    games = []

# Compute bullpen scores from db
bullpen_scores = compute_team_bullpen_scores(pit_df)
print(f"  Bullpen scores computed for {len(bullpen_scores)} teams  range: "
      f"{min(bullpen_scores.values()):.1f}–{max(bullpen_scores.values()):.1f}")

sp_results = []  # list of (game, sp_name, sp_id, batting_team, stats_dict)
for g in games:
    for side in ("away", "home"):
        bat_team = g["away"] if side == "away" else g["home"]
        opp_team = g["home"] if side == "away" else g["away"]
        sp_name  = g[f"{side}_sp_name"]
        sp_id    = g[f"{side}_sp_id"]
        sp_stats = get_sp_stats(sp_name, sp_id, pit_df)
        bp_vuln  = bullpen_scores.get(opp_team, 42.0)
        sp_results.append({
            "matchup":    f"{g['away']}@{g['home']}",
            "bat_team":   bat_team,
            "opp_team":   opp_team,
            "home_team":  g["home"],
            "sp_name":    sp_name,
            "sp_id":      sp_id,
            "sp_source":  sp_stats["data_source"],
            "sp_K":       sp_stats["k_rate_allowed"],
            "sp_FIP":     sp_stats["fip"],
            "sp_WHIP":    sp_stats["whip"],
            "sp_prov":    sp_stats["_provenance"],
            "sp_stats":   sp_stats,
            "bp_vuln":    bp_vuln,
            "game_pk":    g["game_pk"],
        })

# Print SP match table
print()
print(f"  {'Matchup':20s}  {'SP':22s}  {'Source':28s}  {'K%':>5s}  {'FIP':>5s}  {'WHIP':>5s}")
print(f"  {'-'*20}  {'-'*22}  {'-'*28}  {'-'*5}  {'-'*5}  {'-'*5}")
for r in sp_results:
    matched = "✓" if r["sp_source"] == "db_matched" else "✗"
    src_label = r["sp_source"] if r["sp_source"] != "league_avg — NO MATCH" else "league_avg (NO MATCH)"
    print(f"  {r['matchup']:20s}  {r['sp_name'][:22]:22s}  {matched} {src_label[:26]:26s}"
          f"  {r['sp_K']*100:5.1f}  {r['sp_FIP']:5.2f}  {r['sp_WHIP']:5.2f}")

total_sps = len(sp_results)
matched_sps = sum(1 for r in sp_results if r["sp_source"] == "db_matched")
print(f"\n  SP match rate: {matched_sps}/{total_sps} ({matched_sps/max(1,total_sps)*100:.0f}%)")
print(f"  Unmatched SPs fall through to league_avg: K%=22.8%, FIP=4.20, WHIP=1.30")

# ─── STEP 3: Odds API check ───────────────────────────────────────────────────
print()
print("=" * 68)
print("STEP 3 — Vegas implied totals (Odds API)")
print("=" * 68)

# Check if secrets.toml exists and has odds key
secrets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".streamlit", "secrets.toml")
has_secrets = os.path.exists(secrets_path)
has_odds_key = False
if has_secrets:
    try:
        txt = open(secrets_path).read()
        has_odds_key = "odds_api" in txt and "api_key" in txt and \
                       'api_key = ""' not in txt and "api_key = ''" not in txt
    except Exception:
        pass

print(f"  .streamlit/secrets.toml exists:   {has_secrets}")
print(f"  odds_api.api_key configured:       {has_odds_key}")
if not has_odds_key:
    print()
    print("  ⚠ NO ODDS API KEY CONFIGURED")
    print("  → fetch_odds() returns {}  →  implied_totals = {}")
    print("  → implied = implied_totals.get(team, 0)  →  implied = 0 for ALL teams")
    print("  → score_one_batter: vegas_sc = 0.0  (not neutral 50, not default 40 — literally ZERO)")
    print("  → This pulls every final score down by ~0.05 × (50 - 0) × weight ≈ 2.5 pts vs neutral")
    vegas_sc_example = 0.0
else:
    print("  Odds key present — attempting live fetch...")
    try:
        import toml
        secrets = toml.load(secrets_path)
        key = secrets.get("odds_api",{}).get("api_key","")
        url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
        r = requests.get(url, params={"apiKey":key,"regions":"us","markets":"totals,h2h","oddsFormat":"american"}, timeout=10)
        if r.status_code == 200:
            odds_data = r.json()
            print(f"  ✓ Live odds fetched: {len(odds_data)} games")
            vegas_sc_example = None
        else:
            print(f"  ✗ Odds API returned {r.status_code}")
            vegas_sc_example = 0.0
    except Exception as ex:
        print(f"  ✗ Odds fetch error: {ex}")
        vegas_sc_example = 0.0

# ─── STEP 4: Pick 5 batters in strong spots ───────────────────────────────────
print()
print("=" * 68)
print("STEP 4 — Component breakdown for 5 batters in strong matchup spots")
print("=" * 68)

# Identify "strong matchup" batters: high batter_score, facing matched SP with
# high SP vuln (low K, high FIP), at a hitter's park.
# Since we don't have real batting lineup slots or handedness from today's API
# without a heavier fetch, we'll use bat_df records and match to today's games.

from lib.constants import PARK_HR_FACTORS as _PHF
HITTER_PARKS = {k: v for k, v in _PHF.items() if v > 1.02}  # run-favoring

# Find strong candidates: batters with high wRC+ in the db who play for teams
# whose today's opponent has a matched, high-FIP SP, in a hitter's park.
#
# Proxy: best-scoring teams (home or away) where opponent SP is matched in db
# and SP FIP >= 4.5 and park is hitter-friendly
candidates = []
for sp_info in sp_results:
    if sp_info["sp_source"] != "db_matched":
        continue
    if sp_info["sp_FIP"] < 4.2:  # want weak SPs (high FIP = bad pitcher)
        continue
    home = sp_info["home_team"]
    park_sc, park_label = compute_park_score(home, True)
    if park_sc < 55:  # want hitter-friendly parks
        continue
    # Find top batters on the batting team (bat_team faces this SP)
    bat_team = sp_info["bat_team"]
    team_batters = bat_df[bat_df.get("Team", bat_df.columns[0] if "Team" in bat_df.columns else "Team") == bat_team] \
        if "Team" in bat_df.columns else pd.DataFrame()
    if team_batters.empty:
        # Fall back: just add some top overall batters
        team_batters = bat_df.nlargest(5, "wRC+") if "wRC+" in bat_df.columns else bat_df.head(5)
    for _, br in team_batters.nlargest(3, "wRC+").iterrows() if "wRC+" in bat_df.columns else []:
        candidates.append({
            "batter_name":  br.get("_name", "?"),
            "team":         bat_team,
            "opp_team":     sp_info["opp_team"],
            "sp_name":      sp_info["sp_name"],
            "sp_stats":     sp_info["sp_stats"],
            "bp_vuln":      sp_info["bp_vuln"],
            "park_team":    home,
            "park_score":   park_sc,
            "park_label":   park_label,
            "sp_source":    sp_info["sp_source"],
            "batter_row":   br,
            "wrc_plus":     safe_float(br.get("wRC+"), 100.0),
        })

# If we don't have enough from team-filtered batters, just take top-wRC+ batters globally
# and pair with the weakest SP matchup in a hitter park
if len(candidates) < 5 and sp_results:
    # Pick the weakest SP in a hitter park
    hitter_park_games = [r for r in sp_results if compute_park_score(r["home_team"], True)[0] > 50]
    if hitter_park_games:
        worst_sp = max(hitter_park_games, key=lambda r: r["sp_FIP"])
        top_batters = bat_df.nlargest(10, "wRC+") if "wRC+" in bat_df.columns else bat_df.head(10)
        for _, br in top_batters.iterrows():
            if len(candidates) >= 5:
                break
            if any(c["batter_name"] == br.get("_name","?") for c in candidates):
                continue
            candidates.append({
                "batter_name": br.get("_name", "?"),
                "team":        worst_sp["bat_team"],
                "opp_team":    worst_sp["opp_team"],
                "sp_name":     worst_sp["sp_name"],
                "sp_stats":    worst_sp["sp_stats"],
                "bp_vuln":     worst_sp["bp_vuln"],
                "park_team":   worst_sp["home_team"],
                "park_score":  compute_park_score(worst_sp["home_team"], True)[0],
                "park_label":  compute_park_score(worst_sp["home_team"], True)[1],
                "sp_source":   worst_sp["sp_source"],
                "batter_row":  br,
                "wrc_plus":    safe_float(br.get("wRC+"), 100.0),
            })

# Score each candidate, show full breakdown
NEUTRAL_WEATHER = {"temp_f": 72, "wind_mph": 5, "wind_dir": "In", "condition": "Clear"}

print(f"\n  Note on live inputs not available in this script:")
print(f"  • lineup_slot: defaulted to 3 (cleanup — real in the app from MLB API lineup fetch)")
print(f"  • batter_hand: defaulted to R (real in the app from roster fetch)")
print(f"  • sp_hand:     defaulted to R (real in the app from pitcher row or roster fetch)")
print(f"  • weather:     defaulted to neutral 72°F/5mph/Clear (real in the app from Open-Meteo)")
print(f"  • vegas/implied: 0.0 (real in the app ONLY if Odds API key configured — see Step 3)")
print(f"  • recent form / BvP: skipped (live MLB API calls — real in the app)")

DEFAULT_LINEUP = 3
DEFAULT_BATTER_HAND = "R"
DEFAULT_SP_HAND = "R"

for i, c in enumerate(candidates[:5]):
    br = c["batter_row"]
    b_stats = batter_row_to_stats(br)
    sp_stats = c["sp_stats"]

    bat_score, _, bat_details = compute_batter_score(b_stats)

    bp_vuln   = c["bp_vuln"]
    pit_score, pit_label = compute_pitcher_score(sp_stats, bullpen_vuln=bp_vuln)

    matchup_sc, matchup_label = compute_pitch_matchup_score(b_stats, sp_stats)
    plat_sc,   plat_label    = compute_platoon_score(DEFAULT_BATTER_HAND, DEFAULT_SP_HAND)
    lineup_sc, lineup_label  = compute_lineup_score(DEFAULT_LINEUP)
    park_sc,   park_label    = compute_park_score(c["park_team"], True)
    weather_sc, weather_label = compute_weather_score(NEUTRAL_WEATHER)
    vegas_sc_val = 0.0  # No Odds API key; real app gets 0.0 too
    tto_sc, tto_label        = compute_tto_bonus(DEFAULT_LINEUP)

    recent_form = {"tb_per_game": None, "games": 0}  # skip live call
    streak_sc, streak_label = compute_streak_score(recent_form, b_stats["slg_proxy"])

    bvp_data = {}  # skip live call
    bvp_sc, bvp_label, bvp_sig = compute_bvp_score(bvp_data, b_stats["slg_proxy"])

    final = compute_final_score(
        batter_score=bat_score, pitcher_vuln_score=pit_score,
        platoon_score=plat_sc, lineup_score=lineup_sc,
        park_score=park_sc, weather_score=weather_sc,
        vegas_score=vegas_sc_val, tto_bonus=tto_sc,
        pitch_matchup_score=matchup_sc, streak_score=streak_sc,
        bvp_score=bvp_sc, bvp_weight_boost=0.0, proxy_mode=False,
    )
    tier_label = "T1" if final >= 80 else "T2" if final >= 70 else "T3" if final >= 60 else "NP"

    # What would final be with real SP data if SP is unmatched?
    if c["sp_source"] != "db_matched":
        sp_stats_fake = {**sp_stats, "data_source": "db_matched"}  # pretend matched to show variance
        pit_score_matched, _ = compute_pitcher_score(sp_stats_fake, bullpen_vuln=bp_vuln)
    else:
        pit_score_matched = pit_score

    print()
    print(f"  ─── Batter {i+1}: {c['batter_name']} ({c['team']}) vs {c['sp_name']} ({c['opp_team']}@{c['park_team']}) ───")
    print(f"  FINAL SCORE: {final:.1f} [{tier_label}]")
    print()
    print(f"  {'Component':22s}  {'Value':>7s}  {'Source / provenance'}")
    print(f"  {'-'*22}  {'-'*7}  {'-'*42}")

    # batter
    b_prov = sp_stats.get("_provenance",{})  # note: batter prov not tracked in simple dict
    b_src = b_stats.get("data_source","?")
    barrel_s = "measured" if safe_float(br.get("barrel_batted_rate")) is not None else "league_avg"
    hh_s     = "measured" if safe_float(br.get("hard_hit_percent"))   is not None else "league_avg"
    print(f"  {'batter_score':22s}  {bat_score:7.1f}  db_matched (barrel={barrel_s}, HH={hh_s}, wRC+={safe_float(br.get('wRC+'),100):.0f})")
    print(f"    xSLG={b_stats['slg_proxy']:.3f}  Barrel%={b_stats['barrel_rate']*100:.1f}  HH%={b_stats['hard_hit_rate']*100:.1f}  K%={b_stats['k_rate']*100:.1f}  wRC+={b_stats['wrc_plus']:.0f}")

    # pitcher
    sp_prov = sp_stats["_provenance"]
    k_src  = sp_prov.get("k_rate_allowed","?")
    fip_src = sp_prov.get("fip","?")
    wh_src = sp_prov.get("whip","?")
    print(f"  {'pitcher_vuln_score':22s}  {pit_score:7.1f}  SP: {c['sp_source']}")
    print(f"    SP K%={sp_stats['k_rate_allowed']*100:.1f}({k_src})  FIP={sp_stats['fip']:.2f}({fip_src})  WHIP={sp_stats['whip']:.2f}({wh_src})")
    print(f"    bullpen_vuln={bp_vuln:.1f} (real from db, opp={c['opp_team']})")
    print(f"    [pit_score = 0.60×SP_score + 0.40×bullpen_vuln]")

    print(f"  {'park_score':22s}  {park_sc:7.1f}  REAL — static PARK_HR_FACTORS[{c['park_team']}]  label: {park_label}")
    print(f"  {'weather_score':22s}  {weather_sc:7.1f}  DEFAULTED (neutral 72°F/5mph/Clear; real in app from Open-Meteo)")
    print(f"  {'vegas_score':22s}  {vegas_sc_val:7.1f}  {'ZERO — no Odds API key → implied=0 → score=0.0' if not has_odds_key else 'REAL from Odds API'}")
    print(f"  {'platoon_score':22s}  {plat_sc:7.1f}  DEFAULTED to R-vs-R (real in app from roster/pitcher rows)")
    print(f"  {'lineup_score':22s}  {lineup_sc:7.1f}  DEFAULTED to slot 3 (real in app from MLB API lineups)")
    print(f"  {'pitch_matchup_score':22s}  {matchup_sc:7.1f}  {matchup_label}")
    print(f"  {'streak_score':22s}  {streak_sc:7.1f}  SKIPPED (live MLB API; real in app from last-7 game log)")
    print(f"  {'bvp_score':22s}  {bvp_sc:7.1f}  SKIPPED (live MLB API; real in app from vsPlayer endpoint)")
    print(f"  {'tto_bonus':22s}  {tto_sc:7.1f}  DEFAULTED to slot 3 (same as lineup_slot)")

# ─── STEP 5: Variance check — does pitcher quality actually move the score? ───
print()
print("=" * 68)
print("STEP 5 — Does SP quality actually move the final score?")
print("=" * 68)
print("  Holding batter fixed (Juan Soto wRC+=167, league-avg matchup inputs),")
print("  varying ONLY SP quality from elite to replacement-level:\n")

# Use Juan Soto from db if available, else best batter
soto_row = find_row(bat_df, "Juan Soto", "")
if soto_row is None:
    soto_row = bat_df.nlargest(1, "wRC+").iloc[0] if "wRC+" in bat_df.columns else bat_df.iloc[0]
    print(f"  (Soto not found, using {soto_row.get('_name','?')} instead)")

soto_stats = batter_row_to_stats(soto_row)
soto_bat_score, _, _ = compute_batter_score(soto_stats)

sp_scenarios = [
    ("Elite SP (Skenes-tier)",   {"k_rate_allowed":0.310,"fip":2.50,"whip":0.95,"hard_hit_allowed":0.28,"barrel_allowed":0.04,"era":2.50,"data_source":"db_matched","_provenance":{"k_rate_allowed":"measured","fip":"measured","whip":"measured"}}),
    ("Above-avg SP",             {"k_rate_allowed":0.260,"fip":3.40,"whip":1.10,"hard_hit_allowed":0.34,"barrel_allowed":0.06,"era":3.40,"data_source":"db_matched","_provenance":{"k_rate_allowed":"measured","fip":"measured","whip":"measured"}}),
    ("League-avg SP (default)",  {"k_rate_allowed":0.228,"fip":4.20,"whip":1.30,"hard_hit_allowed":0.36,"barrel_allowed":0.07,"era":4.20,"data_source":"league_avg","_provenance":{"k_rate_allowed":"league_avg","fip":"league_avg","whip":"league_avg"}}),
    ("Below-avg SP",             {"k_rate_allowed":0.190,"fip":4.70,"whip":1.45,"hard_hit_allowed":0.40,"barrel_allowed":0.09,"era":4.70,"data_source":"db_matched","_provenance":{"k_rate_allowed":"measured","fip":"measured","whip":"measured"}}),
    ("Bad SP (replacement)",     {"k_rate_allowed":0.160,"fip":5.50,"whip":1.65,"hard_hit_allowed":0.45,"barrel_allowed":0.12,"era":5.50,"data_source":"db_matched","_provenance":{"k_rate_allowed":"measured","fip":"measured","whip":"measured"}}),
]

print(f"  {'Scenario':32s}  {'pit_score':>9s}  {'final':>7s}  {'tier':>4s}  {'Δ from lg-avg':>13s}")
print(f"  {'-'*32}  {'-'*9}  {'-'*7}  {'-'*4}  {'-'*13}")

lg_final = None
for label, sp_s in sp_scenarios:
    bp = 44.5  # league-avg bullpen (what 30-team avg computes to)
    ps, _ = compute_pitcher_score(sp_s, bullpen_vuln=bp)
    fs = compute_final_score(
        batter_score=soto_bat_score, pitcher_vuln_score=ps,
        platoon_score=50.0, lineup_score=50.0, park_score=50.0,
        weather_score=50.0, vegas_score=50.0, tto_bonus=0.0,
        pitch_matchup_score=50.0, streak_score=50.0,
        bvp_score=50.0, bvp_weight_boost=0.0, proxy_mode=False,
    )
    tier_s = "T1" if fs >= 80 else "T2" if fs >= 70 else "T3" if fs >= 60 else "NP"
    if "League-avg" in label:
        lg_final = fs
    delta = f"{fs - lg_final:+.1f}" if lg_final is not None else "  —"
    print(f"  {label:32s}  {ps:9.1f}  {fs:7.1f}  {tier_s:>4s}  {delta:>13s}")

print()
print("  → If the SP score is live and varies, the final score shifts by this much.")
print("  → If the SP is unmatched (league_avg default), EVERY batter sees the same")
print("    league-avg pit_score regardless of who they're actually facing.")
