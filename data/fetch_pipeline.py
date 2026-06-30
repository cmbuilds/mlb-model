"""
fetch_pipeline.py — MLB Model V3.0 data fetch layer
Run: python3 data/fetch_pipeline.py [--season YYYY]

Fetches all player stats, validates coverage, writes data/mlb_stats.db.
No Streamlit imports — runs standalone or as a cron job.

Sources (in priority order):
  1. Savant xStats CSV     → xSLG, xwOBA, xBA             (measured)
  2. Savant Statcast CSV   → Barrel%, HH%, EV              (measured, column-name fix applied)
  3. Savant bat tracking   → bat_speed, blast_rate          (measured)
  4. MLB Stats API hitting → SLG, K%, BB%, ISO, OBP, AVG  (measured)
  5. FanGraphs JSON        → wRC+, FIP, xFIP               (measured if alive, skip on 403)
  Pitching mirrors the same sources for SP stats.

Validation gate (1.3): halt and exit non-zero if <70% expected batters have barrel% measured.
Output (1.4): data/mlb_stats.db — tables: batter_stats, pitcher_stats, fetch_log.
"""

import argparse
import io
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from lib.constants import TEAM_ABB_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fetch_pipeline")

DB_PATH = os.path.join(os.path.dirname(__file__), "mlb_stats.db")

# Savant Statcast CSV now returns these names — map to the names the rest of the
# app expects.  This is the fix for the column-drift bug found in Phase S.4.
STATCAST_COL_MAP = {
    "brl_percent":           "barrel_batted_rate",
    "avg_hit_speed":         "avg_exit_velocity",
    "ev95percent":           "hard_hit_percent",
    "anglesweetspotpercent": "sweet_spot_percent",
    "ev50":                  "ev50",
    "whiff_percent":         "swstr_pct",    # Savant SwStr% column (Phase 4.3)
}

# Bat-tracking CSV column → pipeline name
BAT_TRACKING_COL_MAP = {
    "id":                         "mlbam_id",
    "name":                       "_name",
    "avg_bat_speed":              "bat_speed",
    "blast_per_swing":            "blast_rate",
    "squared_up_per_swing":       "squared_up_rate",
    "hard_swing_rate":            "fast_swing_rate",
    "swing_length":               "swing_length",
}

# xStats CSV column → pipeline name
XSTATS_COL_MAP = {
    "player_id": "mlbam_id",
    "est_slg":   "xSLG",
    "est_woba":  "xwOBA",
    "est_ba":    "xBA",
    "woba":      "wOBA",
    "slg":       "SLG_sv",   # Savant observed SLG (keep separate from MLB API SLG)
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
    "Accept": "text/csv,application/json,*/*",
}

VALIDATION_MIN_COVERAGE = 0.70  # halt if fewer than 70% of batters have barrel% measured

# ─────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_csv(url: str, label: str) -> pd.DataFrame:
    try:
        r = requests.get(url, headers=HEADERS, timeout=25)
        if r.status_code == 200 and r.content:
            df = pd.read_csv(io.StringIO(r.text))
            if not df.empty:
                log.info(f"  {label}: {len(df)} rows")
                return df
            log.warning(f"  {label}: empty CSV")
        else:
            log.warning(f"  {label}: HTTP {r.status_code}")
    except Exception as e:
        log.warning(f"  {label}: {e}")
    return pd.DataFrame()


def _normalize_mlbam_id(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce mlbam_id to a clean string integer in-place."""
    if "mlbam_id" not in df.columns:
        for cand in ("player_id", "id", "IDfg", "xMLBAMID", "MLBAMID"):
            if cand in df.columns:
                df = df.rename(columns={cand: "mlbam_id"})
                break
    if "mlbam_id" in df.columns:
        df["mlbam_id"] = (
            pd.to_numeric(df["mlbam_id"], errors="coerce")
            .dropna()
            .astype(int)
            .astype(str)
        )
        # Re-align after the numeric coercion drops NaN rows
        df = df[df["mlbam_id"].str.match(r"^\d+$")]
    return df


def _pct_to_decimal(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """If a % column is stored as 0–100, divide by 100."""
    for col in cols:
        if col not in df.columns:
            continue
        try:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) > 0 and float(vals.median()) > 1.5:
                df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0
        except Exception:
            pass
    return df


def _merge_on_id(base: pd.DataFrame, other: pd.DataFrame, cols: list) -> pd.DataFrame:
    if other.empty or "mlbam_id" not in other.columns or "mlbam_id" not in base.columns:
        return base
    keep = [c for c in cols if c in other.columns and c not in base.columns]
    if not keep:
        return base
    sub = other[["mlbam_id"] + keep].drop_duplicates(subset=["mlbam_id"])
    return base.merge(sub, on="mlbam_id", how="left")


# ─────────────────────────────────────────────────────────────────────────────
# Source fetchers — batters
# ─────────────────────────────────────────────────────────────────────────────

def fetch_savant_xstats(season: int) -> pd.DataFrame:
    url = (f"https://baseballsavant.mlb.com/leaderboard/expected_statistics"
           f"?type=batter&year={season}&position=&team=&min=1&csv=true")
    df = _fetch_csv(url, f"Savant xStats {season}")
    if df.empty:
        return df
    df = df.rename(columns=XSTATS_COL_MAP)
    df = _normalize_mlbam_id(df)
    # Normalize Savant name column
    if "last_name, first_name" in df.columns:
        df["_name"] = df["last_name, first_name"].apply(
            lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")])) if pd.notna(s) else ""
        )
    return df


def fetch_savant_statcast(season: int, player_type: str = "batter") -> pd.DataFrame:
    """Fetch Savant Statcast leaderboard CSV and apply the column-name fixes."""
    url = (f"https://baseballsavant.mlb.com/leaderboard/statcast"
           f"?year={season}&position=&team=&min=1&type={player_type}&csv=true")
    df = _fetch_csv(url, f"Savant Statcast ({player_type}) {season}")
    if df.empty:
        return df
    # Apply the column-drift fix (the core S.4 bug)
    df = df.rename(columns=STATCAST_COL_MAP)
    df = _normalize_mlbam_id(df)
    if "last_name, first_name" in df.columns:
        df["_name"] = df["last_name, first_name"].apply(
            lambda s: " ".join(reversed([p.strip() for p in str(s).split(",")])) if pd.notna(s) else ""
        )
    # Normalize to decimals
    df = _pct_to_decimal(df, ["barrel_batted_rate", "hard_hit_percent", "sweet_spot_percent"])
    return df


def fetch_bat_tracking(season: int) -> pd.DataFrame:
    url = (f"https://baseballsavant.mlb.com/leaderboard/bat-tracking"
           f"?year={season}&minSwings=50&type=batter&csv=true")
    df = _fetch_csv(url, f"Savant BatTracking {season}")
    if df.empty:
        return df
    df = df.rename(columns=BAT_TRACKING_COL_MAP)
    df = _normalize_mlbam_id(df)
    df = _pct_to_decimal(df, ["blast_rate", "squared_up_rate", "fast_swing_rate"])
    return df


def fetch_mlb_hitting(season: int) -> pd.DataFrame:
    url = (f"https://statsapi.mlb.com/api/v1/stats"
           f"?stats=season&group=hitting&season={season}&limit=2000&offset=0&sportId=1&playerPool=All")
    try:
        r = requests.get(url, timeout=25)
        if r.status_code != 200:
            log.warning(f"  MLB hitting API {season}: HTTP {r.status_code}")
            return pd.DataFrame()
        splits = r.json().get("stats", [{}])[0].get("splits", [])
        if not splits:
            log.warning(f"  MLB hitting API {season}: empty splits")
            return pd.DataFrame()
        rows = []
        for s in splits:
            p   = s.get("player", {})
            st_ = s.get("stat", {})
            pa  = int(st_.get("plateAppearances", 0) or 0)
            if pa == 0:
                continue
            so  = int(st_.get("strikeOuts", 0) or 0)
            bb  = int(st_.get("baseOnBalls", 0) or 0)
            hr  = int(st_.get("homeRuns", 0) or 0)
            ab  = int(st_.get("atBats", 0) or 0)
            d2  = int(st_.get("doubles", 0) or 0)
            d3  = int(st_.get("triples", 0) or 0)
            tb  = int(st_.get("totalBases", 0) or 0)
            try:
                slg = float(st_.get("slg", 0) or 0)
                avg = float(st_.get("avg", 0) or 0)
                obp = float(st_.get("obp", 0) or 0)
                bab = float(st_.get("babip", 0) or 0)
            except (ValueError, TypeError):
                slg = avg = obp = bab = 0.0
            iso = round(slg - avg, 3) if slg >= avg else 0.0
            hr_pa      = hr / pa
            xbh_pa     = (d2 + d3 + hr) / pa
            tb_pa      = tb / pa
            # Hard-hit proxy from counting stats (used only when Savant unavailable)
            hard_proxy = min(0.55, max(0.25, slg * 0.55 + (1 - so / pa) * 0.20))
            rows.append({
                "mlbam_id":    str(p.get("id", "")),
                "_name":       p.get("fullName", ""),
                "SLG":         slg,
                "AVG":         avg,
                "OBP":         obp,
                "ISO":         iso,
                "K%":          round(so / pa, 3),
                "BB%":         round(bb / pa, 3),
                "PA":          pa,
                "HR":          hr,
                "doubles":     d2,
                "triples":     d3,
                "totalBases":  tb,
                "BABIP":       bab,
                "hr_per_pa":   round(hr_pa, 4),
                "xbh_per_pa":  round(xbh_pa, 4),
                "tb_per_pa":   round(tb_pa, 4),
                "hard_proxy":  round(hard_proxy, 3),
            })
        df = pd.DataFrame(rows)
        df = _normalize_mlbam_id(df)
        log.info(f"  MLB hitting API {season}: {len(df)} rows")
        return df
    except Exception as e:
        log.warning(f"  MLB hitting API {season}: {e}")
        return pd.DataFrame()


def fetch_fangraphs_hitting(season: int) -> pd.DataFrame:
    """FanGraphs batting (wRC+, FIP context). Skip cleanly on 403 — currently blocked."""
    url = "https://www.fangraphs.com/api/leaders/major-league/data"
    params = {
        "pos": "all", "stats": "bat", "lg": "all", "qual": "0",
        "type": "8", "season": season, "season1": season,
        "ind": "0", "team": "0", "pageitems": "2000", "pagenum": "1",
        "minpa": "0", "sortdir": "default", "sortstat": "WAR",
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if r.status_code == 403:
            log.info(f"  FanGraphs hitting {season}: 403 (URL changed — skipping)")
            return pd.DataFrame()
        if r.status_code == 200:
            data = r.json().get("data", [])
            if data:
                df = pd.DataFrame(data)
                for id_col in ("xMLBAMID", "MLBAMID", "IDfg"):
                    if id_col in df.columns:
                        df["mlbam_id"] = df[id_col].apply(
                            lambda x: str(int(float(x))) if pd.notna(x) and str(x) not in ("", "nan") else ""
                        )
                        break
                if "mlbam_id" not in df.columns:
                    log.warning(f"  FanGraphs hitting {season}: no ID column found")
                    return pd.DataFrame()
                df = _normalize_mlbam_id(df)
                log.info(f"  FanGraphs hitting {season}: {len(df)} rows")
                return df
        log.warning(f"  FanGraphs hitting {season}: HTTP {r.status_code}")
    except Exception as e:
        log.warning(f"  FanGraphs hitting {season}: {e}")
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Source fetchers — pitchers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_mlb_pitching(season: int) -> pd.DataFrame:
    url = (f"https://statsapi.mlb.com/api/v1/stats"
           f"?stats=season&group=pitching&season={season}&limit=2000&offset=0&sportId=1&playerPool=All")
    try:
        r = requests.get(url, timeout=25)
        if r.status_code != 200:
            log.warning(f"  MLB pitching API {season}: HTTP {r.status_code}")
            return pd.DataFrame()
        splits = r.json().get("stats", [{}])[0].get("splits", [])
        if not splits:
            log.warning(f"  MLB pitching API {season}: empty splits")
            return pd.DataFrame()
        rows = []
        for s in splits:
            p   = s.get("player", {})
            st_ = s.get("stat", {})
            tm  = s.get("team", {})
            try:
                ip = float(str(st_.get("inningsPitched", "0") or "0"))
            except (ValueError, TypeError):
                ip = 0.0
            if ip == 0:
                continue
            so  = int(st_.get("strikeOuts", 0) or 0)
            bb  = int(st_.get("baseOnBalls", 0) or 0)
            tbf = int(st_.get("battersFaced", 0) or 0)
            h   = int(st_.get("hits", 0) or 0)
            er  = int(st_.get("earnedRuns", 0) or 0)
            gs  = int(st_.get("gamesStarted", 0) or 0)
            hr_a = int(st_.get("homeRuns", 0) or 0)
            era  = round(er / ip * 9, 2) if ip > 0 else 4.50
            whip = round((h + bb) / ip, 3) if ip > 0 else 1.35
            k_pct = round(so / tbf, 3) if tbf > 0 else 0.228
            bb_pct = round(bb / tbf, 3) if tbf > 0 else 0.082
            h9 = h / ip * 9 if ip > 0 else 9.0
            # Proxies used only when Savant unavailable
            barrel_proxy  = min(0.18, hr_a / tbf / 0.029 * 0.065) if tbf > 0 else 0.065
            hard_proxy_p  = min(0.50, max(0.25, 0.28 + (h9 - 9.0) * 0.012))
            rows.append({
                "mlbam_id":       str(p.get("id", "")),
                "_name":          p.get("fullName", ""),
                "Team":           TEAM_ABB_MAP.get(tm.get("name", ""), tm.get("abbreviation", "")),
                "ERA":            era,
                "WHIP":           whip,
                "K%":             k_pct,
                "BB%":            bb_pct,
                "GS":             gs,
                "IP":             ip,
                "HR_allowed":     hr_a,
                "H_per_9":        round(h9, 2),
                "barrel_proxy":   round(barrel_proxy, 4),
                "hard_proxy_pit": round(hard_proxy_p, 3),
            })
        df = pd.DataFrame(rows)
        df = _normalize_mlbam_id(df)
        log.info(f"  MLB pitching API {season}: {len(df)} rows")
        return df
    except Exception as e:
        log.warning(f"  MLB pitching API {season}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Provenance assignment
# ─────────────────────────────────────────────────────────────────────────────

def assign_batter_provenance(df: pd.DataFrame) -> pd.DataFrame:
    """Add prov_* columns: 'measured' | 'proxy' | 'league_avg' per key stat."""
    def prov(col, proxy_col=None):
        if col in df.columns:
            measured = pd.to_numeric(df[col], errors="coerce").notna() & (pd.to_numeric(df[col], errors="coerce") > 0)
            if proxy_col and proxy_col in df.columns:
                has_proxy = pd.to_numeric(df[proxy_col], errors="coerce").notna()
                return measured.map({True: "measured"}).fillna(
                    has_proxy.map({True: "proxy"}).fillna("league_avg")
                )
            return measured.map({True: "measured"}).fillna("league_avg")
        return pd.Series("league_avg", index=df.index)

    df["prov_xslg"]     = prov("xSLG")
    df["prov_xwoba"]    = prov("xwOBA")
    df["prov_barrel"]   = prov("barrel_batted_rate", proxy_col="hr_per_pa")
    df["prov_hh"]       = prov("hard_hit_percent",   proxy_col="hard_proxy")
    df["prov_ev"]       = prov("avg_exit_velocity")
    df["prov_bat_speed"]= prov("bat_speed")
    df["prov_blast"]    = prov("blast_rate")
    df["prov_krate"]    = prov("K%")
    df["prov_woba"]     = prov("wOBA",  proxy_col="xwOBA")
    df["prov_slg"]      = prov("SLG")
    df["prov_iso"]      = prov("ISO")
    return df


def assign_pitcher_provenance(df: pd.DataFrame) -> pd.DataFrame:
    def prov(col, proxy_col=None):
        if col in df.columns:
            measured = pd.to_numeric(df[col], errors="coerce").notna() & (pd.to_numeric(df[col], errors="coerce") > 0)
            if proxy_col and proxy_col in df.columns:
                has_proxy = pd.to_numeric(df[proxy_col], errors="coerce").notna()
                return measured.map({True: "measured"}).fillna(
                    has_proxy.map({True: "proxy"}).fillna("league_avg")
                )
            return measured.map({True: "measured"}).fillna("league_avg")
        return pd.Series("league_avg", index=df.index)

    df["prov_krate"]    = prov("K%")
    df["prov_bbrate"]   = prov("BB%")
    df["prov_era"]      = prov("ERA")
    df["prov_barrel_a"] = prov("barrel_batted_rate", proxy_col="barrel_proxy")
    df["prov_hh_a"]     = prov("hard_hit_percent",   proxy_col="hard_proxy_pit")
    df["prov_ev_a"]     = prov("avg_exit_velocity")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Build frames
# ─────────────────────────────────────────────────────────────────────────────

def build_batter_frame(cur: int, pri: int) -> pd.DataFrame:
    log.info("Fetching batter data...")
    log.info(f"  season={cur} (current), {pri} (prior)")

    xs_c  = fetch_savant_xstats(cur)
    xs_p  = fetch_savant_xstats(pri)
    sc_c  = fetch_savant_statcast(cur, "batter")
    sc_p  = fetch_savant_statcast(pri, "batter")
    bt_c  = fetch_bat_tracking(cur)
    bt_p  = fetch_bat_tracking(pri)
    mlb_c = fetch_mlb_hitting(cur)
    mlb_p = fetch_mlb_hitting(pri)
    fg_c  = fetch_fangraphs_hitting(cur)

    # Prefer current season; fall back to prior for players not yet in current
    xstats = xs_c  if not xs_c.empty  else xs_p
    sc     = sc_c  if not sc_c.empty  else sc_p
    bt     = bt_c  if not bt_c.empty  else bt_p
    mlb    = mlb_c if not mlb_c.empty else mlb_p

    # Pick the biggest non-empty frame as the base
    candidates = [(len(f), f) for f in [xstats, sc, mlb] if not f.empty and "mlbam_id" in f.columns]
    if not candidates:
        log.error("No batter data from any source — aborting.")
        return pd.DataFrame()

    result = max(candidates, key=lambda x: x[0])[1].copy()

    result = _merge_on_id(result, xstats, ["xSLG", "xwOBA", "xBA", "wOBA", "SLG_sv"])
    result = _merge_on_id(result, sc,     ["barrel_batted_rate", "hard_hit_percent",
                                            "avg_exit_velocity", "sweet_spot_percent", "ev50"])
    result = _merge_on_id(result, mlb,    ["SLG", "AVG", "OBP", "ISO", "K%", "BB%", "PA",
                                            "HR", "doubles", "triples", "totalBases", "BABIP",
                                            "hr_per_pa", "xbh_per_pa", "tb_per_pa", "hard_proxy"])
    result = _merge_on_id(result, bt,     ["bat_speed", "blast_rate", "squared_up_rate",
                                            "fast_swing_rate", "swing_length"])
    # Blend in current-season sc if base came from prior
    if not sc_c.empty and sc is not sc_c:
        result = _merge_on_id(result, sc_c, ["barrel_batted_rate", "hard_hit_percent", "avg_exit_velocity"])
    if not xs_c.empty and xstats is not xs_c:
        result = _merge_on_id(result, xs_c, ["xSLG", "xwOBA"])
    if not fg_c.empty:
        result = _merge_on_id(result, fg_c, ["wRC+", "FIP", "xFIP"])

    result = assign_batter_provenance(result)
    result["fetch_season"] = cur
    result["fetched_at"]   = datetime.now(timezone.utc).isoformat()
    return result


def fetch_savant_pitcher_swstr(season: int) -> pd.DataFrame:
    """
    Fetch per-pitcher aggregate swinging-strike rate (SwStr%) from Savant.

    Source: pitch-arsenal-stats endpoint — returns per-pitch-type rows;
    aggregated here to per-pitcher by weighted average:
        swstr_pct = sum(whiff_percent * pitches) / sum(pitches)

    Returns a DataFrame with [mlbam_id, swstr_pct].
    """
    url = (f"https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
           f"?year={season}&min_pa=25&type=pitcher&bats=all&csv=true")
    df = _fetch_csv(url, f"Savant SwStr% (pitcher) {season}")
    if df.empty:
        return pd.DataFrame()

    df = _normalize_mlbam_id(df)
    if "mlbam_id" not in df.columns:
        return pd.DataFrame()

    for col in ("whiff_percent", "pitches"):
        if col not in df.columns:
            return pd.DataFrame()

    df["whiff_percent"] = pd.to_numeric(df["whiff_percent"], errors="coerce").fillna(0.0)
    df["pitches"]       = pd.to_numeric(df["pitches"],       errors="coerce").fillna(0.0)
    df["_wh_x_p"]       = df["whiff_percent"] * df["pitches"]

    agg = (df.groupby("mlbam_id", as_index=False)
             .agg(_wh_x_p=("_wh_x_p", "sum"), _pitches=("pitches", "sum"))
             .assign(swstr_pct=lambda x: x["_wh_x_p"] / x["_pitches"].clip(lower=1))
             [["mlbam_id", "swstr_pct"]])

    # Convert percentage (e.g. 25.3) to decimal (0.253)
    if agg["swstr_pct"].max() > 1.0:
        agg["swstr_pct"] = agg["swstr_pct"] / 100.0

    log.info(f"  Savant SwStr% (pitcher) {season}: {len(agg)} pitchers aggregated")
    return agg


def build_pitcher_frame(cur: int, pri: int) -> pd.DataFrame:
    log.info("Fetching pitcher data...")
    sc_c    = fetch_savant_statcast(cur, "pitcher")
    sc_p    = fetch_savant_statcast(pri, "pitcher")
    sw_c    = fetch_savant_pitcher_swstr(cur)
    sw_p    = fetch_savant_pitcher_swstr(pri)
    mlb_c   = fetch_mlb_pitching(cur)
    mlb_p   = fetch_mlb_pitching(pri)

    sc    = sc_c  if not sc_c.empty  else sc_p
    sw    = sw_c  if not sw_c.empty  else sw_p
    mlb   = mlb_c if not mlb_c.empty else mlb_p

    candidates = [(len(f), f) for f in [sc, mlb] if not f.empty and "mlbam_id" in f.columns]
    if not candidates:
        log.error("No pitcher data from any source — aborting.")
        return pd.DataFrame()

    result = max(candidates, key=lambda x: x[0])[1].copy()
    result = _merge_on_id(result, sc,  ["barrel_batted_rate", "hard_hit_percent",
                                         "avg_exit_velocity", "sweet_spot_percent"])
    result = _merge_on_id(result, sw,  ["swstr_pct"])   # aggregate SwStr% from pitch-arsenal
    result = _merge_on_id(result, mlb, ["ERA", "WHIP", "K%", "BB%", "GS", "IP",
                                         "HR_allowed", "H_per_9", "barrel_proxy", "hard_proxy_pit"])
    if not sc_c.empty and sc is not sc_c:
        result = _merge_on_id(result, sc_c, ["barrel_batted_rate", "hard_hit_percent", "avg_exit_velocity"])

    result = assign_pitcher_provenance(result)
    result["fetch_season"] = cur
    result["fetched_at"]   = datetime.now(timezone.utc).isoformat()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Validation (1.3)
# ─────────────────────────────────────────────────────────────────────────────

def validate_batter_frame(df: pd.DataFrame) -> bool:
    if df.empty:
        log.error("VALIDATION FAILED: batter frame is empty")
        return False

    total = len(df)

    # Coverage: barrel% measured
    barrel_measured = (df.get("prov_barrel", pd.Series()) == "measured").sum()
    barrel_pct = barrel_measured / total
    log.info(f"  barrel%  measured: {barrel_measured}/{total} = {barrel_pct:.0%}")

    if barrel_pct < VALIDATION_MIN_COVERAGE:
        log.error(
            f"VALIDATION FAILED: barrel% measured in only {barrel_pct:.0%} of batters "
            f"(threshold {VALIDATION_MIN_COVERAGE:.0%}). Savant Statcast CSV may be unreachable."
        )
        return False

    # K% measured (MLB API always provides this)
    k_measured = (df.get("prov_krate", pd.Series()) == "measured").sum()
    log.info(f"  K%       measured: {k_measured}/{total} = {k_measured/total:.0%}")

    # xSLG measured
    xslg_measured = (df.get("prov_xslg", pd.Series()) == "measured").sum()
    log.info(f"  xSLG     measured: {xslg_measured}/{total} = {xslg_measured/total:.0%}")

    log.info("  VALIDATION PASSED")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# SQLite write (1.4)
# ─────────────────────────────────────────────────────────────────────────────

def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate columns (case-insensitive) — SQLite treats them as the same."""
    seen: dict[str, int] = {}
    drop = []
    for i, col in enumerate(df.columns):
        key = col.lower()
        if key in seen:
            drop.append(i)
        else:
            seen[key] = i
    if drop:
        keep = [i for i in range(len(df.columns)) if i not in drop]
        df = df.iloc[:, keep]
    return df


def write_db(batters: pd.DataFrame, pitchers: pd.DataFrame) -> None:
    batters  = _dedup_columns(batters)
    pitchers = _dedup_columns(pitchers)
    con = sqlite3.connect(DB_PATH)
    try:
        batters.to_sql("batter_stats",  con, if_exists="replace", index=False)
        pitchers.to_sql("pitcher_stats", con, if_exists="replace", index=False)

        # fetch_log row
        log_row = pd.DataFrame([{
            "fetched_at":      datetime.now(timezone.utc).isoformat(),
            "batter_rows":     len(batters),
            "pitcher_rows":    len(pitchers),
            "barrel_measured": int((batters.get("prov_barrel", pd.Series()) == "measured").sum()),
            "xslg_measured":   int((batters.get("prov_xslg",  pd.Series()) == "measured").sum()),
            "k_measured":      int((batters.get("prov_krate",  pd.Series()) == "measured").sum()),
        }])
        log_row.to_sql("fetch_log", con, if_exists="append", index=False)
        con.commit()
        log.info(f"Wrote {DB_PATH}: {len(batters)} batters, {len(pitchers)} pitchers")
    finally:
        con.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLB Model fetch pipeline")
    parser.add_argument("--season", type=int, default=datetime.now().year,
                        help="Current season year (default: current calendar year)")
    args = parser.parse_args()

    cur = args.season
    pri = cur - 1
    log.info(f"=== MLB fetch pipeline  current={cur}  prior={pri} ===")
    t0 = time.time()

    batters  = build_batter_frame(cur, pri)
    pitchers = build_pitcher_frame(cur, pri)

    log.info("Validating...")
    if not validate_batter_frame(batters):
        log.error("Pipeline halted due to validation failure.")
        sys.exit(1)

    write_db(batters, pitchers)
    log.info(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
