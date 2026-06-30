#!/usr/bin/env python3
"""
scripts/diag_gaps.py — Diagnostics only, no fixes.

Gap 1: Bullpen vulnerability stuck at 42.0 for every team.
  - Print actual columns of the live pitching DataFrame
  - Confirm whether "Team" column exists and is populated
  - Confirm whether "FIP" column exists
  - Trace where Team/FIP are produced or dropped through the pipeline

Gap 2: K% not measured for Kyle Stowers, Griffin Conine, Cole Carrigg.
  - Check MLB Stats API hitting: are they present, and what is their K%?
  - Check whether they're in the Savant CSV
  - Determine: genuine data gap vs merge miss
"""

import sys, io
from pathlib import Path

import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

YEAR    = 2026
TIMEOUT = 25
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
    ),
    "Accept": "text/csv,application/json,*/*",
}

MYSTERY_BATTERS = {
    "Kyle Stowers":   None,
    "Griffin Conine": None,
    "Cole Carrigg":   None,
}

def section(t):
    print(f"\n{'='*64}\n  {t}\n{'='*64}")

def _csv(url, label):
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 200 and r.content:
            df = pd.read_csv(io.StringIO(r.text))
            if not df.empty:
                print(f"  [OK]   {label}: {len(df)} rows")
                return df
        print(f"  [FAIL] {label}: HTTP {r.status_code}")
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
    return pd.DataFrame()

def _norm_id(df, preferred_col="mlbam_id"):
    """Ensure mlbam_id column exists by normalizing from any known ID col."""
    for col in ("player_id", "mlbam_id", "xMLBAMID", "IDfg"):
        if col in df.columns:
            df = df.copy()
            df[preferred_col] = (
                df[col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
            )
            return df
    return df


# ════════════════════════════════════════════════════════════
#  GAP 1: BULLPEN VULNERABILITY
# ════════════════════════════════════════════════════════════

section("GAP 1a — MLB Stats API pitching: raw columns + Team/FIP presence")

print(f"\nFetching MLB Stats API pitching (playerPool=All)...")
try:
    r = requests.get(
        f"https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=pitching&season={YEAR}"
        f"&limit=2000&offset=0&sportId=1&playerPool=All",
        timeout=TIMEOUT
    )
    splits = r.json().get("stats", [{}])[0].get("splits", [])
    print(f"  [OK]   MLB API pitching: {len(splits)} pitchers")

    rows = []
    for s in splits:
        p   = s.get("player", {})
        st_ = s.get("stat", {})
        tm  = s.get("team", {})
        ip  = float(str(st_.get("inningsPitched","0") or "0"))
        so  = int(st_.get("strikeOuts", 0) or 0)
        bb  = int(st_.get("baseOnBalls", 0) or 0)
        tbf = int(st_.get("battersFaced", 0) or 0)
        h   = int(st_.get("hits", 0) or 0)
        er  = int(st_.get("earnedRuns", 0) or 0)
        hr  = int(st_.get("homeRuns", 0) or 0)
        gs  = int(st_.get("gamesStarted", 0) or 0)
        g   = int(st_.get("gamesPlayed", 0) or 0)
        rows.append({
            "mlbam_id": str(p.get("id","")),
            "_name":    p.get("fullName",""),
            "Team":     tm.get("abbreviation",""),
            "ERA":      round(er/ip*9, 2) if ip > 0 else None,
            "WHIP":     round((h+bb)/ip, 3) if ip > 0 else None,
            "K%":       round(so/tbf, 4) if tbf > 0 else None,
            "BB%":      round(bb/tbf, 4) if tbf > 0 else None,
            "GS":       gs, "G": g, "IP": ip,
            # FIP components — FIP itself requires a constant (~3.1) not in API
            "_HR_allowed": hr,
            "_SO": so, "_BB": bb, "_IP": ip,
        })

    mlb_pit = pd.DataFrame(rows)
    mlb_pit = mlb_pit[mlb_pit["IP"] > 0]

    print(f"\n  Columns produced by MLB API pitching fetch:")
    print(f"    {list(mlb_pit.columns)}")
    print(f"\n  'Team' column present : {'Team' in mlb_pit.columns}")
    print(f"  'FIP'  column present : {'FIP' in mlb_pit.columns}")
    print(f"  'ERA'  column present : {'ERA' in mlb_pit.columns}")

    # Check Team population
    team_filled = mlb_pit["Team"].replace("", pd.NA).notna().sum()
    team_empty  = (mlb_pit["Team"] == "").sum()
    print(f"\n  Team values: {team_filled} filled / {team_empty} empty")
    print(f"  Sample teams: {sorted(mlb_pit['Team'].dropna().unique())[:15]}")

    # Spot-check: team column for a few relievers
    relievers_sample = mlb_pit[(mlb_pit["GS"] == 0) & (mlb_pit["IP"] > 5)].head(8)
    print(f"\n  Sample relievers (GS=0, IP>5):")
    print(relievers_sample[["_name","Team","K%","ERA","GS","G","IP"]].to_string(index=False))

except Exception as e:
    print(f"  [FAIL] MLB API pitching: {e}")
    mlb_pit = pd.DataFrame()


section("GAP 1b — FanGraphs type=8: does FIP actually arrive?")

print(f"\nFetching FanGraphs type=8 pitching (FIP source)...")
try:
    import random
    fg_r8 = requests.get(
        "https://www.fangraphs.com/api/leaders/major-league/data",
        params={
            "pos":"all","stats":"pit","lg":"all","qual":"0",
            "type":"8","season":YEAR,"season1":YEAR,"ind":"0",
            "team":"0","pageitems":"1000","pagenum":"1","minip":"0",
        },
        headers={
            "User-Agent": random.choice([
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
            ]),
            "Referer": "https://www.fangraphs.com/leaders/major-league",
            "Accept":  "application/json",
        },
        timeout=12
    )
    fg_data = fg_r8.json().get("data", [])
    if fg_data:
        fg8 = pd.DataFrame(fg_data)
        print(f"  [OK]   FanGraphs type=8: {len(fg8)} rows")
        fip_present = "FIP" in fg8.columns
        print(f"  'FIP' column present: {fip_present}")
        if fip_present:
            print(f"  FIP sample values (first 5): {fg8['FIP'].dropna().head(5).tolist()}")
        team_col_fg = next((c for c in ("Team","team","Tm") if c in fg8.columns), None)
        print(f"  Team column in FanGraphs type=8: {team_col_fg} (columns: {list(fg8.columns[:12])})")
        # Check ID column
        id_col = next((c for c in ("xMLBAMID","MLBAMID","IDfg","playerid") if c in fg8.columns), None)
        print(f"  ID column: {id_col}")
    else:
        print(f"  [FAIL] FanGraphs type=8: HTTP {fg_r8.status_code} or empty data")
        fg8 = pd.DataFrame()
except Exception as e:
    print(f"  [FAIL] FanGraphs type=8: {e}")
    fg8 = pd.DataFrame()


section("GAP 1c — After full merge: what does compute_team_bullpen_scores see?")

print("\nSimulating the pitching_df as load_all_pitching_stats() builds it...")

# Base: MLB API (has Team, K%, ERA, GS, G, IP — no FIP)
# Bonus: FanGraphs type=8 (has FIP if reachable)
# All of this mirrors what the monolith does at runtime

combined_pit = mlb_pit.copy()

if not fg8.empty:
    id_col = next((c for c in ("xMLBAMID","MLBAMID","IDfg","playerid") if c in fg8.columns), None)
    if id_col:
        fg8 = fg8.copy()
        fg8["mlbam_id"] = fg8[id_col].apply(
            lambda x: str(int(float(x))) if pd.notna(x) and str(x) not in ("","nan") else ""
        )
        want = [c for c in ["FIP","xFIP","SIERA","SwStr%"] if c in fg8.columns]
        if want:
            sub = fg8[["mlbam_id"] + want].drop_duplicates("mlbam_id")
            combined_pit = combined_pit.merge(sub, on="mlbam_id", how="left")
            print(f"  Merged FanGraphs type=8 cols: {want}")
            if "FIP" in combined_pit.columns:
                fip_filled = combined_pit["FIP"].notna().sum()
                print(f"  FIP filled for {fip_filled}/{len(combined_pit)} pitchers after merge")

print(f"\n  Final pitching_df columns:")
print(f"    {list(combined_pit.columns)}")
print(f"\n  'Team' present: {'Team' in combined_pit.columns}")
print(f"  'FIP'  present: {'FIP'  in combined_pit.columns}")

# Now run compute_team_bullpen_scores on it and trace the exit point
print("\n  Tracing compute_team_bullpen_scores()...")

df_check = combined_pit.copy()

# Check #1: team column detection
team_col = None
for candidate in ["Team", "team", "Tm", "tm", "TEAM"]:
    if candidate in df_check.columns:
        team_col = candidate
        break

if team_col is None:
    print(f"  ❌ EXIT POINT: no team column found → returns {{}} → every team defaults to 42.0")
    print(f"  Columns checked: ['Team', 'team', 'Tm', 'tm', 'TEAM']")
    print(f"  Actual columns:  {list(df_check.columns)}")
else:
    print(f"  ✅ team_col = '{team_col}'")
    # Check #2: GS/G for reliever filter
    has_gs = "GS" in df_check.columns
    has_g  = "G"  in df_check.columns
    print(f"  'GS' present: {has_gs} | 'G' present: {has_g}")
    if has_gs and has_g:
        df_check["_GS"] = pd.to_numeric(df_check["GS"], errors="coerce").fillna(0)
        df_check["_G"]  = pd.to_numeric(df_check["G"],  errors="coerce").fillna(0)
        relievers = df_check[
            (df_check["_GS"] == 0) |
            ((df_check["_G"] > 0) & (df_check["_GS"] / df_check["_G"].replace(0,1) < 0.30))
        ]
        print(f"  Relievers identified: {len(relievers)} / {len(df_check)} total")
    # Check #3: FIP in relievers
    fip_present = "FIP" in df_check.columns
    print(f"  'FIP' present for groupby: {fip_present}")
    if fip_present:
        fip_fill_pct = df_check["FIP"].notna().mean() * 100
        print(f"  FIP fill rate: {fip_fill_pct:.0f}%")
    # Check #4: how many unique teams would be grouped
    unique_teams = df_check[team_col].replace("", pd.NA).dropna().nunique()
    print(f"  Unique teams in '{team_col}': {unique_teams}")
    # Sample team→score preview using ERA as FIP proxy
    try:
        from mlb_tb_analyzer import compute_team_bullpen_scores
        scores = compute_team_bullpen_scores(combined_pit)
        if scores:
            print(f"\n  compute_team_bullpen_scores() → {len(scores)} teams")
            sample = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:6]
            print(f"  Top 6 most vulnerable: {sample}")
            sample2 = sorted(scores.items(), key=lambda x: x[1])[:6]
            print(f"  Top 6 least vulnerable: {sample2}")
        else:
            print(f"\n  compute_team_bullpen_scores() → empty dict (returns {{}})")
            print(f"  → every team.get() defaults to 42.0 — BUG CONFIRMED")
    except Exception as e:
        print(f"  compute_team_bullpen_scores() raised: {e}")


# ════════════════════════════════════════════════════════════
#  GAP 2: MISSING BATTER K%
# ════════════════════════════════════════════════════════════

section("GAP 2a — MLB Stats API hitting: do greyed batters appear at all?")

print(f"\nFetching MLB Stats API hitting (default, no playerPool=All)...")
try:
    r = requests.get(
        f"https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=hitting&season={YEAR}&limit=2000&offset=0&sportId=1",
        timeout=TIMEOUT
    )
    splits_hit = r.json().get("stats",[{}])[0].get("splits",[])
    print(f"  [OK] Default MLB API hitting: {len(splits_hit)} batters")

    r2 = requests.get(
        f"https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=hitting&season={YEAR}&limit=2000&offset=0&sportId=1&playerPool=All",
        timeout=TIMEOUT
    )
    splits_all = r2.json().get("stats",[{}])[0].get("splits",[])
    print(f"  [OK] playerPool=All MLB API hitting: {len(splits_all)} batters")
except Exception as e:
    print(f"  [FAIL] {e}")
    splits_hit = []; splits_all = []

# Build two lookup dicts: name → {id, K%, PA, ...}
def build_lookup(splits):
    out = {}
    for s in splits:
        p   = s.get("player",{})
        st_ = s.get("stat",{})
        pa  = int(st_.get("plateAppearances",0) or 0)
        so  = int(st_.get("strikeOuts",0) or 0)
        out[p.get("fullName","").lower()] = {
            "id":   str(p.get("id","")),
            "name": p.get("fullName",""),
            "PA":   pa,
            "SO":   so,
            "K%":   round(so/pa, 4) if pa > 0 else None,
            "SLG":  float(st_.get("slg",0) or 0),
            "AVG":  float(st_.get("avg",0) or 0),
        }
    return out

default_lookup = build_lookup(splits_hit)
all_lookup     = build_lookup(splits_all)

print(f"\n  Searching for greyed batters:")
for batter_name in MYSTERY_BATTERS:
    key = batter_name.lower()
    in_default = key in default_lookup
    in_all     = key in all_lookup

    if in_all:
        info = all_lookup[key]
        MYSTERY_BATTERS[batter_name] = info["id"]
        k_str = f"{info['K%']*100:.1f}%" if info['K%'] else "N/A"
        print(f"\n  {batter_name}:")
        print(f"    MLB ID     = {info['id']}")
        print(f"    PA         = {info['PA']}")
        print(f"    K%         = {k_str}  (SO={info['SO']})")
        print(f"    AVG/SLG    = {info['AVG']:.3f} / {info['SLG']:.3f}")
        print(f"    in DEFAULT fetch (no playerPool=All): {'YES' if in_default else 'NO ← missing here'}")
        print(f"    in ALL     fetch (playerPool=All):    YES")
        if not in_default:
            print(f"    → MERGE MISS: player exists but filtered out by PA minimum in default fetch")
    else:
        print(f"\n  {batter_name}:")
        print(f"    NOT FOUND in either MLB API fetch (may be minor league, IL, or name mismatch)")
        # Try partial match
        partials = [v for k,v in all_lookup.items() if batter_name.split()[-1].lower() in k]
        if partials:
            print(f"    Partial matches by last name: {[(p['name'],p['PA']) for p in partials[:3]]}")


section("GAP 2b — Savant batter CSV: do greyed batters appear there?")

bat_df = _csv(
    f"https://baseballsavant.mlb.com/leaderboard/statcast"
    f"?year={YEAR}&position=&team=&min=1&type=batter&csv=true",
    "Savant batter statcast CSV"
)

if not bat_df.empty:
    bat_df = _norm_id(bat_df)
    id_col = "player_id" if "player_id" in bat_df.columns else "mlbam_id"
    brl_col = "brl_percent" if "brl_percent" in bat_df.columns else None

    print(f"\n  Searching for greyed batters in Savant CSV:")
    for batter_name, mid in MYSTERY_BATTERS.items():
        # Try by name
        name_match = None
        last = batter_name.split()[-1].lower()
        name_col = next((c for c in ("last_name, first_name","name","Name") if c in bat_df.columns), None)
        if name_col:
            hits = bat_df[bat_df[name_col].astype(str).str.lower().str.contains(last, na=False)]
            if not hits.empty:
                name_match = hits.iloc[0]

        # Try by mlbam_id
        id_match = None
        if mid and id_col in bat_df.columns:
            norm = bat_df[id_col].astype(str).str.replace(r"\.0$","",regex=True)
            id_rows = bat_df[norm == str(mid)]
            if not id_rows.empty:
                id_match = id_rows.iloc[0]

        found = id_match if id_match is not None else name_match
        if found is not None:
            brl = found.get("brl_percent", found.get("barrel_batted_rate", "?"))
            pa_sv = found.get("pa", found.get("attempts", "?"))
            print(f"\n  {batter_name}: FOUND in Savant CSV")
            print(f"    PA (Savant): {pa_sv}")
            print(f"    brl_percent: {brl}")
        else:
            print(f"\n  {batter_name}: NOT FOUND in Savant CSV")
            print(f"    → Savant requires min batted ball events; may be below threshold")


section("GAP 2c — fetch_pipeline.py SQLite: are they in batter_stats?")

import sqlite3
try:
    conn = sqlite3.connect("data/mlb_stats.db")
    cols_bat = [d[0] for d in conn.execute("SELECT * FROM batter_stats LIMIT 0").description]
    print(f"\n  batter_stats columns: {cols_bat[:20]}")
    print()

    for batter_name, mid in MYSTERY_BATTERS.items():
        # Try by name
        name_part = batter_name.split()[-1]
        rows = conn.execute(
            f"SELECT * FROM batter_stats WHERE _name LIKE ? OR mlbam_id=?",
            (f"%{name_part}%", str(mid) if mid else "")
        ).fetchall()

        if rows:
            for row in rows:
                d = dict(zip(cols_bat, row))
                krate = d.get("K%") or d.get("k_rate")
                print(f"  {batter_name}: FOUND in SQLite (id={d.get('mlbam_id')}, PA={d.get('pa')})")
                print(f"    K%={krate}  prov_krate={d.get('prov_krate')}  barrel={d.get('barrel_batted_rate')} prov_barrel={d.get('prov_barrel')}")
        else:
            print(f"  {batter_name}: NOT IN SQLite batter_stats")
    conn.close()
except Exception as e:
    print(f"  [FAIL] SQLite check: {e}")

print()
print("="*64)
print("  DONE — findings only, no changes made")
print("="*64)
