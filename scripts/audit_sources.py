#!/usr/bin/env python3
"""
scripts/audit_sources.py — Data source audit (read-only, no app changes).

Tests each upstream data source independently for known players and prints
whether real values came back or the source failed/returned defaults.

Known players used as ground truth:
  Batters  : Aaron Judge (592450), Julio Rodriguez (677594)
  Pitchers : Garrett Crochet (661563), Tarik Skubal (669373)
  Game logs: Jackson Chourio (682928)
"""

import sys
import traceback
from io import StringIO

import requests
import pandas as pd

YEAR = 2026
TIMEOUT = 15
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.0 Safari/605.1.15"
    ),
    "Referer": "https://baseballsavant.mlb.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

KNOWN_BATTERS = {
    "592450": "Aaron Judge",
    "677594": "Julio Rodriguez",
}
KNOWN_PITCHERS = {
    "661563": "Garrett Crochet",
    "669373": "Tarik Skubal",
}
CHOURIO_ID = "682928"


def sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def ok(label, value):
    print(f"  ✅ {label}: {value}")


def fail(label, reason):
    print(f"  ❌ {label}: {reason}")


def warn(label, reason):
    print(f"  ⚠️  {label}: {reason}")


def fetch_csv(url, label):
    """Fetch a CSV endpoint. Returns (DataFrame or None, status_code, error_msg)."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code != 200:
            return None, r.status_code, f"HTTP {r.status_code}"
        if len(r.content) < 500:
            return None, r.status_code, f"Response too small ({len(r.content)} bytes) — likely blocked"
        df = pd.read_csv(StringIO(r.text))
        if df.empty:
            return None, r.status_code, "Empty DataFrame"
        return df, r.status_code, None
    except Exception as e:
        return None, None, str(e)


def fetch_json(url, label):
    """Fetch a JSON endpoint. Returns (parsed dict/list or None, status_code, error_msg)."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code != 200:
            return None, r.status_code, f"HTTP {r.status_code}"
        return r.json(), r.status_code, None
    except Exception as e:
        return None, None, str(e)


def find_player_in_df(df, player_id, id_cols=("player_id", "mlbam_id", "xMLBAMID", "IDfg")):
    """Try multiple ID column names to find a player."""
    for col in id_cols:
        if col in df.columns:
            mask = df[col].astype(str) == str(player_id)
            if mask.any():
                return df[mask].iloc[0], col
    # Try name-based fallback in player_name column
    for col in ("player_name", "name", "Name"):
        if col in df.columns:
            for pid, name in {**KNOWN_BATTERS, **KNOWN_PITCHERS}.items():
                if pid == str(player_id):
                    mask = df[col].str.lower().str.contains(name.split()[1].lower(), na=False)
                    if mask.any():
                        return df[mask].iloc[0], col
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1: Savant Expected Stats (xSLG, xwOBA)
# ─────────────────────────────────────────────────────────────────────────────
sep("SOURCE 1: Savant Expected Stats (xSLG, xwOBA)")
url = f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=batter&year={YEAR}&min=1&csv=true"
print(f"  URL: {url}")
df, status, err = fetch_csv(url, "savant_xstats")
if df is None:
    fail("Fetch", err)
else:
    print(f"  HTTP {status} | {len(df)} rows | columns: {list(df.columns[:8])}")
    for pid, name in KNOWN_BATTERS.items():
        row, id_col = find_player_in_df(df, pid)
        if row is not None:
            xslg = row.get("est_slg", row.get("xslg", "missing"))
            xwoba = row.get("est_woba", row.get("xwoba", "missing"))
            ok(name, f"xSLG={xslg}  xwOBA={xwoba}  (matched via {id_col})")
        else:
            fail(name, f"player_id {pid} not found in {len(df)}-row DataFrame")


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2: Savant Statcast Leaderboard — Batter (barrel%, HH%, EV)
# ─────────────────────────────────────────────────────────────────────────────
sep("SOURCE 2: Savant Statcast Leaderboard — Batter (barrel%, HH%, EV)")
url = f"https://baseballsavant.mlb.com/leaderboard/statcast?year={YEAR}&min=1&type=batter&csv=true"
print(f"  URL: {url}")
df2, status, err = fetch_csv(url, "savant_statcast_bat")
if df2 is None:
    fail("Fetch", err)
else:
    print(f"  HTTP {status} | {len(df2)} rows | columns: {list(df2.columns[:8])}")
    for pid, name in KNOWN_BATTERS.items():
        row, id_col = find_player_in_df(df2, pid)
        if row is not None:
            barrel = row.get("barrel_batted_rate", row.get("Barrel%", "missing"))
            hh = row.get("hard_hit_percent", row.get("hard_hit_rate", "missing"))
            ev = row.get("avg_exit_velocity", row.get("EV", "missing"))
            ok(name, f"barrel%={barrel}  HH%={hh}  EV={ev}  (via {id_col})")
        else:
            fail(name, f"player_id {pid} not found")


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 3: Savant Bat Tracking (bat speed, blast rate)
# ─────────────────────────────────────────────────────────────────────────────
sep("SOURCE 3: Savant Bat Tracking (bat speed, blast rate)")
url = f"https://baseballsavant.mlb.com/leaderboard/bat-tracking?year={YEAR}&minSwings=50&type=batter&csv=true"
print(f"  URL: {url}")
df3, status, err = fetch_csv(url, "savant_bat_track")
if df3 is None:
    fail("Fetch", err)
else:
    print(f"  HTTP {status} | {len(df3)} rows | columns: {list(df3.columns[:8])}")
    for pid, name in KNOWN_BATTERS.items():
        row, id_col = find_player_in_df(df3, pid)
        if row is not None:
            spd = row.get("bat_speed", "missing")
            blast = row.get("blast_rate", "missing")
            ok(name, f"bat_speed={spd}  blast_rate={blast}  (via {id_col})")
        else:
            fail(name, f"player_id {pid} not found")


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 4: FanGraphs Batting — type=8 (wRC+, BB%, K%)
# ─────────────────────────────────────────────────────────────────────────────
sep("SOURCE 4: FanGraphs Batting type=8 (wRC+, BB%, K%)")
url = (
    f"https://www.fangraphs.com/api/leaders/major-league/data"
    f"?pos=all&stats=bat&lg=all&qual=y&type=8&season={YEAR}&season1={YEAR}"
    f"&ind=0&team=0&page=1_500"
)
print(f"  URL: {url}")
data, status, err = fetch_json(url, "fangraphs_bat")
if data is None:
    fail("Fetch", err)
else:
    rows = data.get("data", data) if isinstance(data, dict) else data
    if not rows:
        fail("Parse", "No rows in response")
    else:
        df_fg = pd.DataFrame(rows)
        print(f"  HTTP {status} | {len(df_fg)} rows | columns: {list(df_fg.columns[:10])}")
        for pid, name in KNOWN_BATTERS.items():
            row, id_col = find_player_in_df(df_fg, pid, id_cols=("xMLBAMID", "MLBAMID", "player_id", "IDfg"))
            if row is not None:
                wrc = row.get("wRC+", row.get("wrc_plus", "missing"))
                kpct = row.get("K%", "missing")
                bbpct = row.get("BB%", "missing")
                ok(name, f"wRC+={wrc}  K%={kpct}  BB%={bbpct}  (via {id_col})")
            else:
                # Try name match
                for col in ("PlayerName", "player_name", "Name", "name"):
                    if col in df_fg.columns:
                        mask = df_fg[col].str.lower().str.contains(name.split()[-1].lower(), na=False)
                        if mask.any():
                            r = df_fg[mask].iloc[0]
                            wrc = r.get("wRC+", r.get("wrc_plus", "missing"))
                            ok(f"{name} (name match)", f"wRC+={wrc}")
                            break
                else:
                    fail(name, f"not found (tried ID cols + name cols: {[c for c in df_fg.columns if 'name' in c.lower() or 'id' in c.lower()][:6]})")


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 5: FanGraphs Pitching — type=8 (FIP, xFIP, K%)
# ─────────────────────────────────────────────────────────────────────────────
sep("SOURCE 5: FanGraphs Pitching type=8 (FIP, xFIP, K%)")
url = (
    f"https://www.fangraphs.com/api/leaders/major-league/data"
    f"?pos=all&stats=pit&lg=all&qual=y&type=8&season={YEAR}&season1={YEAR}"
    f"&ind=0&team=0&page=1_500"
)
print(f"  URL: {url}")
data, status, err = fetch_json(url, "fangraphs_pit")
if data is None:
    fail("Fetch", err)
else:
    rows = data.get("data", data) if isinstance(data, dict) else data
    if not rows:
        fail("Parse", "No rows in response")
    else:
        df_fgp = pd.DataFrame(rows)
        print(f"  HTTP {status} | {len(df_fgp)} rows | columns: {list(df_fgp.columns[:10])}")
        for pid, name in KNOWN_PITCHERS.items():
            row, id_col = find_player_in_df(df_fgp, pid, id_cols=("xMLBAMID", "MLBAMID", "player_id", "IDfg"))
            if row is not None:
                fip = row.get("FIP", "missing")
                xfip = row.get("xFIP", "missing")
                kpct = row.get("K%", "missing")
                ok(name, f"FIP={fip}  xFIP={xfip}  K%={kpct}  (via {id_col})")
            else:
                for col in ("PlayerName", "player_name", "Name", "name"):
                    if col in df_fgp.columns:
                        mask = df_fgp[col].str.lower().str.contains(name.split()[-1].lower(), na=False)
                        if mask.any():
                            r = df_fgp[mask].iloc[0]
                            fip = r.get("FIP", "missing")
                            ok(f"{name} (name match)", f"FIP={fip}")
                            break
                else:
                    fail(name, f"not found (ID cols: {[c for c in df_fgp.columns if 'id' in c.lower()][:6]})")


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 6: Savant Statcast Leaderboard — Pitcher (barrel%/HH% allowed)
# ─────────────────────────────────────────────────────────────────────────────
sep("SOURCE 6: Savant Statcast — Pitcher (barrel%/HH% allowed)")
url = f"https://baseballsavant.mlb.com/leaderboard/statcast?year={YEAR}&min=1&type=pitcher&csv=true"
print(f"  URL: {url}")
df6, status, err = fetch_csv(url, "savant_statcast_pit")
if df6 is None:
    fail("Fetch", err)
else:
    print(f"  HTTP {status} | {len(df6)} rows | columns: {list(df6.columns[:8])}")
    for pid, name in KNOWN_PITCHERS.items():
        row, id_col = find_player_in_df(df6, pid)
        if row is not None:
            barrel = row.get("barrel_batted_rate", row.get("Barrel%", "missing"))
            hh = row.get("hard_hit_percent", row.get("hard_hit_rate", "missing"))
            k_pct = row.get("k_percent", row.get("K%", "missing"))
            ok(name, f"barrel%_allowed={barrel}  HH%_allowed={hh}  K%={k_pct}  (via {id_col})")
        else:
            fail(name, f"player_id {pid} not found")


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 7: MLB Stats API — Pitching stats columns
# ─────────────────────────────────────────────────────────────────────────────
sep("SOURCE 7: MLB Stats API — Pitching season stats")
url = (
    f"https://statsapi.mlb.com/api/v1/stats"
    f"?stats=season&group=pitching&gameType=R&season={YEAR}"
    f"&playerPool=all&limit=500&hydrate=person"
)
print(f"  URL: {url}")
data, status, err = fetch_json(url, "mlbapi_pit")
if data is None:
    fail("Fetch", err)
else:
    splits = []
    for stat_grp in data.get("stats", []):
        splits.extend(stat_grp.get("splits", []))
    print(f"  HTTP {status} | {len(splits)} player-splits")
    if splits:
        sample = splits[0].get("stat", {})
        print(f"  Columns available in 'stat' dict: {sorted(sample.keys())}")
        # Look up known pitchers
        for pid, name in KNOWN_PITCHERS.items():
            found = None
            for s in splits:
                person_id = str(s.get("player", {}).get("id", ""))
                if person_id == pid:
                    found = s.get("stat", {})
                    break
            if found:
                cols = {k: found[k] for k in ("strikeOuts", "inningsPitched", "era", "whip", "battersFaced") if k in found}
                ok(name, str(cols))
            else:
                fail(name, f"player_id {pid} not found in {len(splits)} splits")
    else:
        fail("Splits", "No data returned")


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 8: MLB Stats API — Game logs for Jackson Chourio (per-game vs cumulative)
# ─────────────────────────────────────────────────────────────────────────────
sep("SOURCE 8: MLB Stats API — Game logs for Jackson Chourio (682928)")
url = (
    f"https://statsapi.mlb.com/api/v1/people/{CHOURIO_ID}/stats"
    f"?stats=gameLog&group=hitting&gameType=R&season={YEAR}&limit=12"
)
print(f"  URL: {url}")
data, status, err = fetch_json(url, "mlbapi_gamelogs")
if data is None:
    fail("Fetch", err)
else:
    splits = []
    for stat_grp in data.get("stats", []):
        splits.extend(stat_grp.get("splits", []))
    print(f"  HTTP {status} | {len(splits)} game splits returned")
    if splits:
        print(f"\n  Last 7 games (most recent first):")
        print(f"  {'Date':<12} {'AB':>4} {'H':>4} {'2B':>4} {'3B':>4} {'HR':>4} {'TB':>4} {'RBI':>4} {'Note'}")
        print(f"  {'-'*70}")
        for i, s in enumerate(splits[:7]):
            st = s.get("stat", {})
            date = s.get("date", "?")
            ab = st.get("atBats", "?")
            h = st.get("hits", "?")
            d = st.get("doubles", "?")
            t = st.get("triples", "?")
            hr = st.get("homeRuns", "?")
            tb = st.get("totalBases", "?")
            rbi = st.get("rbi", "?")
            # Check if TB looks per-game (< 20) or cumulative (> 30)
            try:
                tb_val = int(tb)
                note = "✅ per-game" if tb_val < 15 else "⚠️ CUMULATIVE?" if tb_val > 30 else "?"
            except Exception:
                note = ""
            print(f"  {date:<12} {str(ab):>4} {str(h):>4} {str(d):>4} {str(t):>4} {str(hr):>4} {str(tb):>4} {str(rbi):>4}  {note}")
        print(f"\n  All available stat keys: {sorted(splits[0].get('stat', {}).keys())}")
    else:
        fail("Game logs", "No splits returned — player may not have played yet this season")

print(f"\n{'='*60}")
print("  AUDIT COMPLETE")
print('='*60)
