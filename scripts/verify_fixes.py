#!/usr/bin/env python3
"""
scripts/verify_fixes.py — Prove that the June-30 data-accuracy fixes produce
real measured values, not proxies.

Tests:
  Batters  Aaron Judge (592450), Julio Rodríguez (677594)
           → Barrel%, Hard Hit%, Avg EV — must all be measured
  Pitchers Garrett Crochet (676979), Tarik Skubal (669373)
           → barrel% allowed, hard-hit% allowed — measured after rename fix
           → SwStr% — must read real pitch-arsenal value (Skubal ~32%, not K%×0.49 ~16%)
  Chourio  recent form — TB/game must be ~1.7, not ~4.7 (oldest-games bug)
  Summary  proxy count across both pitchers via get_pitcher_stats()
"""

import sys, io
from pathlib import Path

import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

YEAR = 2026
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
    ),
    "Referer": "https://baseballsavant.mlb.com/",
    "Accept": "text/csv,application/json,*/*",
}

BATTER_IDS  = {"Aaron Judge": "592450", "Julio Rodríguez": "677594"}
PITCHER_IDS = {"Garrett Crochet": "676979", "Tarik Skubal": "669373"}
CHOURIO_ID  = "694192"

BATTER_RENAME  = {"brl_percent": "barrel_batted_rate", "avg_hit_speed": "avg_exit_velocity",
                  "ev95percent": "hard_hit_percent",   "anglesweetspotpercent": "sweet_spot_percent"}
PITCHER_RENAME = {"brl_percent": "barrel_batted_rate", "avg_hit_speed": "avg_exit_velocity",
                  "ev95percent": "hard_hit_percent"}


def _get(url, label, **kw):
    try:
        r = requests.get(url, headers=HEADERS, timeout=25, **kw)
        if r.status_code == 200 and r.content:
            df = pd.read_csv(io.StringIO(r.text))
            if not df.empty:
                print(f"  [OK]   {label}: {len(df)} rows")
                return df
        print(f"  [FAIL] {label}: HTTP {r.status_code}")
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
    return pd.DataFrame()


def _find(df, mlbam):
    for col in ("player_id", "pitcher_id", "batter_id", "mlbam_id"):
        if col not in df.columns:
            continue
        norm = df[col].astype(str).str.replace(r"\.0$", "", regex=True)
        mask = norm == str(mlbam)
        if mask.any():
            return df[mask].iloc[0]
    return None


def _v(row, *keys):
    for k in keys:
        if k in row.index and pd.notna(row[k]):
            try:
                v = float(row[k])
                if v != 0:
                    return v
            except Exception:
                pass
    return None


def fmt_pct(v, label="MISSING"):
    if v is None:
        return label
    return f"{v*100:.1f}%" if v <= 1 else f"{v:.1f}%"


def fmt_ev(v):
    return "MISSING" if v is None else f"{float(v):.1f} mph"


def section(t):
    print(f"\n{'='*62}\n  {t}\n{'='*62}")


# ─────────────────────────────────────────────────────────────
# 1. Batter Statcast
# ─────────────────────────────────────────────────────────────
section("BATTER STATCAST  barrel% / hard-hit% / Avg EV")

bat_df = _get(
    f"https://baseballsavant.mlb.com/leaderboard/statcast"
    f"?year={YEAR}&position=&team=&min=1&type=batter&csv=true",
    "Savant batter statcast CSV"
)
if not bat_df.empty:
    raw_cc = [c for c in bat_df.columns if any(k in c for k in ("brl","ev95","hit_speed","hard_hit","barrel"))]
    bat_df = bat_df.rename(columns=BATTER_RENAME)
    new_cc = [c for c in bat_df.columns if any(k in c for k in ("barrel","hard_hit","avg_exit"))]
    print(f"  raw contact cols : {raw_cc}")
    print(f"  renamed to       : {new_cc}")

print()
for name, mid in BATTER_IDS.items():
    row = _find(bat_df, mid) if not bat_df.empty else None
    if row is None:
        print(f"  {name} ({mid}): ❌ NOT FOUND")
        continue
    brl = _v(row, "barrel_batted_rate")
    hh  = _v(row, "hard_hit_percent")
    ev  = _v(row, "avg_exit_velocity")
    print(f"  {name} ({mid}):")
    print(f"    Barrel%   = {fmt_pct(brl):<8}  [{'measured' if brl is not None else 'MISSING ❌'}]")
    print(f"    Hard-Hit% = {fmt_pct(hh):<8}  [{'measured' if hh  is not None else 'MISSING ❌'}]")
    print(f"    Avg EV    = {fmt_ev(ev):<12}  [{'measured' if ev  is not None else 'MISSING ❌'}]")


# ─────────────────────────────────────────────────────────────
# 2. Pitcher Statcast (Barrel% / HH% allowed)
# ─────────────────────────────────────────────────────────────
section("PITCHER STATCAST  barrel% allowed / hard-hit% allowed")

pit_df = _get(
    f"https://baseballsavant.mlb.com/leaderboard/statcast"
    f"?year={YEAR}&position=SP-RP&team=&min=1&type=pitcher&csv=true",
    "Savant pitcher statcast CSV"
)
if not pit_df.empty:
    raw_cc = [c for c in pit_df.columns if any(k in c for k in ("brl","ev95","hit_speed","hard_hit","barrel"))]
    pit_df = pit_df.rename(columns=PITCHER_RENAME)
    new_cc = [c for c in pit_df.columns if any(k in c for k in ("barrel","hard_hit","avg_exit"))]
    print(f"  raw contact cols : {raw_cc}")
    print(f"  renamed to       : {new_cc}")
    if "barrel_batted_rate" in pit_df.columns and "Barrel%" not in pit_df.columns:
        pit_df["Barrel%"] = pit_df["barrel_batted_rate"]
    if "hard_hit_percent" in pit_df.columns and "Hard%" not in pit_df.columns:
        pit_df["Hard%"] = pit_df["hard_hit_percent"]

print()
for name, mid in PITCHER_IDS.items():
    row = _find(pit_df, mid) if not pit_df.empty else None
    if row is None:
        print(f"  {name} ({mid}): ❌ NOT FOUND in Savant pitcher CSV")
        continue
    brl = _v(row, "barrel_batted_rate", "Barrel%")
    hh  = _v(row, "hard_hit_percent", "Hard%")
    ev  = _v(row, "avg_exit_velocity")
    print(f"  {name} ({mid}):")
    print(f"    Barrel% allowed   = {fmt_pct(brl):<8}  [{'measured' if brl is not None else 'MISSING ❌'}]")
    print(f"    Hard-Hit% allowed = {fmt_pct(hh):<8}  [{'measured' if hh  is not None else 'MISSING ❌'}]")
    print(f"    Avg EV allowed    = {fmt_ev(ev):<12}  [{'measured' if ev  is not None else 'MISSING ❌'}]")


# ─────────────────────────────────────────────────────────────
# 3. SwStr% from pitch arsenal
# ─────────────────────────────────────────────────────────────
section("PITCHER SwStr%  real (pitch arsenal) vs K%×0.49 (proxy)")

arsenal_df = _get(
    f"https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
    f"?type=pitcher&pitchType=&year={YEAR}&team=&min=10&csv=true",
    "Savant pitch-arsenal-stats"
)

# MLB API for K%/ERA (playerPool=All returns all pitchers, not just qualified)
mlb_pit_df = pd.DataFrame()
try:
    r = requests.get(
        f"https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=pitching&season={YEAR}&limit=2000&offset=0&sportId=1&playerPool=All",
        timeout=25
    )
    splits = r.json().get("stats", [{}])[0].get("splits", [])
    rows = []
    for s in splits:
        p, st_ = s.get("player", {}), s.get("stat", {})
        ip  = float(str(st_.get("inningsPitched", "0") or "0"))
        so  = int(st_.get("strikeOuts", 0) or 0)
        bb  = int(st_.get("baseOnBalls", 0) or 0)
        tbf = int(st_.get("battersFaced", 0) or 0)
        h   = int(st_.get("hits", 0) or 0)
        er  = int(st_.get("earnedRuns", 0) or 0)
        rows.append({
            "mlbam_id": str(p.get("id", "")),
            "_name":    p.get("fullName", ""),
            "K%":       round(so / tbf, 4) if tbf > 0 else None,
            "BB%":      round(bb / tbf, 4) if tbf > 0 else None,
            "ERA":      round(er / ip * 9, 2) if ip > 0 else None,
            "WHIP":     round((h + bb) / ip, 3) if ip > 0 else None,
            "IP":       ip,
        })
    mlb_pit_df = pd.DataFrame(rows)
    mlb_pit_df = mlb_pit_df[mlb_pit_df["IP"] > 0]
    print(f"  [OK]   MLB Stats API pitching (playerPool=All): {len(mlb_pit_df)} pitchers")
except Exception as e:
    print(f"  [FAIL] MLB Stats API: {e}")


def weighted_swstr(arsenal_df, mlbam):
    """Weighted-average whiff% across pitch types, using pitch count as weight."""
    if arsenal_df.empty:
        return None
    row = _find(arsenal_df, mlbam)
    if row is None:
        return None
    # _find returns a single row; we need all rows for this pitcher
    for col in ("player_id", "pitcher_id", "mlbam_id"):
        if col not in arsenal_df.columns:
            continue
        norm = arsenal_df[col].astype(str).str.replace(r"\.0$", "", regex=True)
        sub = arsenal_df[norm == str(mlbam)].copy()
        if sub.empty:
            continue
        if "whiff_percent" in sub.columns and "pitches" in sub.columns:
            sub = sub.dropna(subset=["whiff_percent", "pitches"])
            total = sub["pitches"].sum()
            if total > 0:
                return (sub["whiff_percent"] * sub["pitches"]).sum() / total / 100
        break
    return None


print()
for name, mid in PITCHER_IDS.items():
    mlb_row  = _find(mlb_pit_df, mid) if not mlb_pit_df.empty else None
    k_rate   = float(mlb_row["K%"])  if (mlb_row is not None and mlb_row.get("K%") is not None) else None
    era_val  = float(mlb_row["ERA"]) if (mlb_row is not None and mlb_row.get("ERA") is not None) else None
    whip_val = float(mlb_row["WHIP"])if (mlb_row is not None and mlb_row.get("WHIP") is not None) else None

    swstr_real  = weighted_swstr(arsenal_df, mid)
    swstr_proxy = round(k_rate * 0.49, 4) if k_rate else None

    if swstr_real is not None and swstr_real > 0:
        swstr_used = swstr_real
        swstr_prov = "measured (pitch arsenal, weighted avg)"
    elif swstr_proxy:
        swstr_used = swstr_proxy
        swstr_prov = f"PROXY — K%×0.49 = {k_rate*100:.1f}%×0.49"
    else:
        swstr_used = None
        swstr_prov = "MISSING"

    print(f"  {name} ({mid}):")
    print(f"    K%            = {fmt_pct(k_rate):<8}  [{'measured' if k_rate  else 'MISSING ❌'}]")
    print(f"    ERA           = {f'{era_val:.2f}' if era_val else 'MISSING':<8}  [{'measured' if era_val else 'MISSING ❌'}]")
    print(f"    WHIP          = {f'{whip_val:.3f}' if whip_val else 'MISSING':<8}  [{'measured' if whip_val else 'MISSING ❌'}]")
    if swstr_real is not None:
        print(f"    SwStr% (real) = {swstr_real*100:.1f}%      [measured — pitch arsenal weighted avg]")
    else:
        print(f"    SwStr% (real) = NOT FOUND in arsenal")
    if swstr_proxy:
        print(f"    SwStr% proxy  = {swstr_proxy*100:.1f}%      [K%×0.49 fallback]")
    if swstr_used:
        prov_tag = "✅ measured" if swstr_prov.startswith("measured") else "⚠️  PROXY"
        print(f"    → MODEL USES  = {swstr_used*100:.1f}%      [{prov_tag}: {swstr_prov}]")
    print()


# ─────────────────────────────────────────────────────────────
# 4. Chourio recent form
# ─────────────────────────────────────────────────────────────
section("RECENT FORM  Chourio TB/game — oldest-7 bug vs fixed newest-7")

try:
    url = (f"https://statsapi.mlb.com/api/v1/people/{CHOURIO_ID}/stats"
           f"?stats=gameLog&group=hitting&gameType=R&season={YEAR}&limit=20")
    r = requests.get(url, timeout=12)
    splits = r.json().get("stats", [{}])[0].get("splits", [])
    print(f"  [OK] gameLog: {len(splits)} splits, in chronological order (oldest → newest)")

    print(f"\n  All games (oldest → newest):")
    for i, s in enumerate(splits):
        st_ = s.get("stat", {})
        d   = s.get("date", "")
        tb  = st_.get("totalBases", 0)
        h   = st_.get("hits", 0)
        hr  = st_.get("homeRuns", 0)
        mark = "  ◄ in FIXED window (newest 7)" if i >= len(splits) - 7 else \
               ("  ◄ in BUGGY window (oldest 7)" if i < 7 else "")
        print(f"    [{i:2d}] {d}  H={h}  HR={hr}  TB={tb}{mark}")

    old7   = splits[:7]
    new7   = splits[-7:]
    old_tb = sum(s.get("stat",{}).get("totalBases",0) for s in old7)
    new_tb = sum(s.get("stat",{}).get("totalBases",0) for s in new7)

    print(f"\n  BUGGY  splits[:7]  = oldest 7 games : {old_tb} TB in {len(old7)} games = {old_tb/max(len(old7),1):.2f} TB/game")
    print(f"  FIXED  splits[-7:] = newest 7 games : {new_tb} TB in {len(new7)} games = {new_tb/max(len(new7),1):.2f} TB/game")
    print(f"  ESPN reported ~13 TB / 7 days  →  fixed value {'✅ matches' if abs(new_tb - 13) <= 2 else '⚠️  differs'}")
except Exception as e:
    print(f"  [FAIL] {e}")


# ─────────────────────────────────────────────────────────────
# 5. End-to-end: get_pitcher_stats() with combined pitching_df
# ─────────────────────────────────────────────────────────────
section("END-TO-END  get_pitcher_stats() — all fields, all provenance")

try:
    from mlb_tb_analyzer import get_pitcher_stats

    # Normalize Savant pitcher CSV: player_id → mlbam_id (mirrors _normalize_savant_pit)
    pit_norm = pit_df.copy()
    if not pit_norm.empty:
        for pid_col in ("player_id", "IDfg", "mlbam_id"):
            if pid_col in pit_norm.columns:
                pit_norm["mlbam_id"] = (
                    pit_norm[pid_col].astype(str)
                    .str.replace(r"\.0$", "", regex=True)
                    .str.strip()
                )
                break
        if "barrel_batted_rate" in pit_norm.columns and "Barrel%" not in pit_norm.columns:
            pit_norm["Barrel%"] = pit_norm["barrel_batted_rate"]
        if "hard_hit_percent" in pit_norm.columns and "Hard%" not in pit_norm.columns:
            pit_norm["Hard%"] = pit_norm["hard_hit_percent"]

    # Combine: MLB API (K%, ERA, WHIP) + Savant CSV (Barrel%, Hard%) + arsenal (SwStr%)
    if not mlb_pit_df.empty and not pit_norm.empty and "mlbam_id" in pit_norm.columns:
        sc_cols = [c for c in ["mlbam_id","barrel_batted_rate","hard_hit_percent",
                               "avg_exit_velocity","Barrel%","Hard%"] if c in pit_norm.columns]
        combined = mlb_pit_df.merge(
            pit_norm[sc_cols].drop_duplicates("mlbam_id"), on="mlbam_id", how="left"
        )
    else:
        combined = mlb_pit_df.copy()

    # Inject arsenal SwStr% and add xMLBAMID alias (used by find_player_row)
    for mid in PITCHER_IDS.values():
        sw = weighted_swstr(arsenal_df, mid)
        if sw is not None:
            combined.loc[combined["mlbam_id"] == mid, "swstr_pct"] = sw
    combined["xMLBAMID"] = combined["mlbam_id"]

    print(f"  Combined pitching_df: {len(combined)} rows")
    sc_present = [c for c in ["Barrel%","Hard%","swstr_pct"] if c in combined.columns]
    print(f"  Savant cols present : {sc_present}")
    print()

    all_proxies = 0
    all_measured = 0

    KEY_FIELDS = ["k_rate_allowed", "barrel_allowed", "hard_hit_allowed", "swstr_pct", "era", "whip"]

    for name, mid in PITCHER_IDS.items():
        res  = get_pitcher_stats(name, mid, combined)
        prov = res["_provenance"]

        print(f"  {name} ({mid}):")
        for f in KEY_FIELDS:
            v    = res.get(f, 0)
            p    = prov.get(f, "?")
            if f in ("k_rate_allowed", "barrel_allowed", "hard_hit_allowed"):
                disp = fmt_pct(v)
            elif f == "swstr_pct":
                proxy_flag = res.get("swstr_pct_is_proxy", True)
                disp = f"{v*100:.1f}%  (proxy={proxy_flag})"
            elif f == "era":
                disp = f"{v:.2f}"
            else:
                disp = f"{v:.3f}"
            tag = "✅ measured" if p == "measured" else ("⚠️  proxy" if p == "proxy" else "❌ league_avg")
            if p in ("proxy", "league_avg"):
                all_proxies += 1
            else:
                all_measured += 1
            print(f"    {f:<22} = {disp:<22}  [{tag}]")
        print()

    total = all_proxies + all_measured
    pct   = all_measured / total * 100 if total else 0
    print(f"  Data completeness: {all_measured}/{total} fields measured ({pct:.0f}%)")
    print(f"  Proxies active   : {all_proxies}/{total}")
    if all_proxies <= 2:
        print(f"  ✅ Target: ≤2 proxies across both pitchers")
    else:
        print(f"  ⚠️  {all_proxies} proxies — see which fields are still falling back above")

except ImportError as e:
    print(f"  [SKIP] Cannot import mlb_tb_analyzer: {e}")

print()
print("=" * 62)
print("  DONE")
print("=" * 62)
