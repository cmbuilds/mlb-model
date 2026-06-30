"""
proxy_mode_before_after.py

1. Confirms proxy_mode evaluates False with the rebuilt db columns.
2. Loads batter_stats from SQLite (exactly as the app does via _load_from_db).
3. Computes a synthetic final score per batter using:
     - compute_batter_score()  with each player's real stats from the db
     - League-average sub-scores for everything else (pitcher=50, park=50,
       platoon=50, vegas=50, streak=50, weather=50, matchup=50, bvp=50, lineup=50)
     This isolates the effect of the proxy-mode switch on the score+tier math
     independently of today's specific matchups/lineups.
4. Shows tier counts BEFORE (proxy: +9.5 offset, thresholds 75/65/55) and
   AFTER (full: +7.0 offset, thresholds 80/70/60).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import sqlite3
import pandas as pd
from scoring.batter import compute_batter_score
from scoring.final  import compute_final_score
from config import TB_OFFSET_FULL, TB_OFFSET_PROXY, TIERS_FULL, TIERS_PROXY

DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mlb_stats.db")

# ─── 1. Proxy-mode check — verify the new condition with rebuilt db columns ──
print("=" * 60)
print("STEP 1 — proxy_mode condition with rebuilt db")
print("=" * 60)

con = sqlite3.connect(DB)
bat_df = pd.read_sql("SELECT * FROM batter_stats", con)
con.close()

bat_cols = list(bat_df.columns)
_has_full = (
    ("barrel_batted_rate" in bat_cols or "Barrel%" in bat_cols)
    and ("hard_hit_percent" in bat_cols or "Hard%" in bat_cols)
    and ("wRC+" in bat_cols)
)

# OLD (broken) check — substring on "savant+mlbapi"
_bat_src = "savant+mlbapi"  # what run_model() sets for the SQLite load path
old_proxy = ("mlbapi" in _bat_src or _bat_src in ("mlbapi_only",)
             or "disk_cache_stale" in _bat_src or not _has_full)

# NEW (fixed) check
new_proxy = not _has_full

print(f"  Rebuilt db columns include barrel_batted_rate: {'barrel_batted_rate' in bat_cols}")
print(f"  Rebuilt db columns include hard_hit_percent:   {'hard_hit_percent' in bat_cols}")
print(f"  Rebuilt db columns include wRC+:               {'wRC+' in bat_cols}")
print(f"  _has_full = {_has_full}")
print(f"  _batting_source label = {_bat_src!r}")
print()
print(f"  OLD proxy_mode (substring bug):  {old_proxy}  ← WRONG, fires on 'savant+mlbapi'")
print(f"  NEW proxy_mode (column check):   {new_proxy}  ← CORRECT")
print()
print(f"  Active with OLD: offset=+{TB_OFFSET_PROXY}, Tier1≥{TIERS_PROXY['🔒 TIER 1']}, Tier2≥{TIERS_PROXY['✅ TIER 2']}, Tier3≥{TIERS_PROXY['📊 TIER 3']}")
print(f"  Active with NEW: offset=+{TB_OFFSET_FULL}, Tier1≥{TIERS_FULL['🔒 TIER 1']}, Tier2≥{TIERS_FULL['✅ TIER 2']}, Tier3≥{TIERS_FULL['📊 TIER 3']}")

# ─── 2. Compute synthetic scores for every batter in the db ──────────────────
print()
print("=" * 60)
print("STEP 2 — synthetic final scores (real batter stats + league-avg matchup)")
print("=" * 60)

def row_to_statcast(row):
    """Map SQLite batter_stats row to the dict compute_batter_score expects."""
    def g(key, default=0.0):
        v = row.get(key)
        try: return float(v) if v is not None and str(v) not in ("", "nan", "None") else default
        except (ValueError, TypeError): return default

    xslg     = g("xSLG", 0.398)
    barrel   = g("barrel_batted_rate", 0.070)
    hard_hit = g("hard_hit_percent",   0.370)
    k_pct    = g("K%",  0.228)
    iso      = g("ISO", 0.165)
    wrc_plus = g("wRC+", 100.0)
    woba     = g("wOBA", 0.315)
    ev50     = g("ev50", 0.0)
    bat_spd  = g("bat_speed", 0.0)
    blast    = g("blast_rate", 0.0)

    # barrel_batted_rate comes from Savant as fraction (0.07 = 7%)
    # If stored >1, divide (shouldn't be the case after pipeline fix, but guard anyway)
    if barrel > 1:
        barrel /= 100.0
    if hard_hit > 1:
        hard_hit /= 100.0
    if k_pct > 1:
        k_pct /= 100.0

    return {
        "slg_proxy":    xslg,
        "barrel_rate":  barrel,
        "hard_hit_rate": hard_hit,
        "k_rate":       k_pct,
        "iso_proxy":    iso,
        "wrc_plus":     wrc_plus,
        "woba":         woba,
        "ev50":         ev50,
        "bat_speed":    bat_spd,
        "blast_rate":   blast,
    }

# League-average sub-scores (plateau for pitcher, park, platoon, etc.)
LG_AVG = 50.0

scores_before = []  # proxy mode (old, wrong)
scores_after  = []  # full mode (new, correct)

for _, row in bat_df.iterrows():
    sc_dict = row_to_statcast(row)
    bat_score, _, _ = compute_batter_score(sc_dict)

    # compute_final_score takes the weighted composite; pass batter sub-score
    # and league-average everything else, both modes
    raw_composite = compute_final_score(
        batter_score=bat_score,
        pitcher_vuln_score=LG_AVG,
        platoon_score=LG_AVG,
        lineup_score=LG_AVG,
        park_score=LG_AVG,
        weather_score=LG_AVG,
        vegas_score=LG_AVG,
        tto_bonus=0.0,
        pitch_matchup_score=LG_AVG,
        streak_score=LG_AVG,
        bvp_score=LG_AVG,
        bvp_weight_boost=0.0,
        proxy_mode=False,   # we need the raw pre-offset; get it by using offset=0 temporarily
    )
    # raw_composite = weighted_sum/1.03 + TB_OFFSET_FULL (7.0); back out the offset
    raw_pre_offset = raw_composite - TB_OFFSET_FULL

    score_proxy = round(min(100, max(0, raw_pre_offset + TB_OFFSET_PROXY)), 1)
    score_full  = round(min(100, max(0, raw_pre_offset + TB_OFFSET_FULL)),  1)
    scores_before.append(score_proxy)
    scores_after.append(score_full)

def apply_tiers(scores, tiers):
    t1 = sum(1 for s in scores if s >= tiers["🔒 TIER 1"])
    t2 = sum(1 for s in scores if tiers["✅ TIER 2"] <= s < tiers["🔒 TIER 1"])
    t3 = sum(1 for s in scores if tiers["📊 TIER 3"] <= s < tiers["✅ TIER 2"])
    np = sum(1 for s in scores if s < tiers["📊 TIER 3"])
    return t1, t2, t3, np

t1b, t2b, t3b, npb = apply_tiers(scores_before, TIERS_PROXY)
t1a, t2a, t3a, npa = apply_tiers(scores_after,  TIERS_FULL)

print(f"\n  Batters scored: {len(bat_df)}")
print(f"  Score range BEFORE (proxy):  {min(scores_before):.1f} – {max(scores_before):.1f}")
print(f"  Score range AFTER  (full):   {min(scores_after):.1f} – {max(scores_after):.1f}")

print()
print(f"  {'':20s}  {'BEFORE (proxy, wrong)':>22s}  {'AFTER (full, correct)':>22s}  {'Delta':>6s}")
print(f"  {'Tier 1':20s}  {'≥75':>22s}  {'≥80':>22s}")
print(f"  {'':20s}  {t1b:>22d}  {t1a:>22d}  {t1a-t1b:>+6d}")
print(f"  {'Tier 2':20s}  {'65–74':>22s}  {'70–79':>22s}")
print(f"  {'':20s}  {t2b:>22d}  {t2a:>22d}  {t2a-t2b:>+6d}")
print(f"  {'Tier 3':20s}  {'55–64':>22s}  {'60–69':>22s}")
print(f"  {'':20s}  {t3b:>22d}  {t3a:>22d}  {t3a-t3b:>+6d}")
print(f"  {'No Play':20s}  {'<55':>22s}  {'<60':>22s}")
print(f"  {'':20s}  {npb:>22d}  {npa:>22d}  {npa-npb:>+6d}")
print(f"  {'Total':20s}  {t1b+t2b+t3b+npb:>22d}  {t1a+t2a+t3a+npa:>22d}")

# ─── 3. Detail: which proxy-Tier-2 plays drop to Tier 3 in full mode? ────────
print()
print("=" * 60)
print("STEP 3 — Proxy Tier 2 plays that drop to Tier 3 in full mode")
print("=" * 60)

drops = []
for i, (sb, sa) in enumerate(zip(scores_before, scores_after)):
    row = bat_df.iloc[i]
    proxy_tier = ("T1" if sb >= 75 else "T2" if sb >= 65 else "T3" if sb >= 55 else "NP")
    full_tier  = ("T1" if sa >= 80 else "T2" if sa >= 70 else "T3" if sa >= 60 else "NP")
    if proxy_tier != full_tier:
        drops.append((row.get("_name","?"), sb, sa, proxy_tier, full_tier))

drops.sort(key=lambda x: x[1], reverse=True)
if drops:
    print(f"  Players whose tier changes ({len(drops)} total):")
    print(f"  {'Name':25s}  {'Proxy Score':>11s}  {'Full Score':>10s}  {'From':>5s}  {'To':>5s}")
    for name, sb, sa, pt, ft in drops[:30]:
        print(f"  {name:25s}  {sb:11.1f}  {sa:10.1f}  {pt:>5s}  {ft:>5s}")
    if len(drops) > 30:
        print(f"  ... and {len(drops)-30} more")
else:
    print("  No tier changes (all players land in the same tier both ways)")

# ─── 4. Banner check ──────────────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 4 — Banner condition after fix")
print("=" * 60)
print(f"  _has_full = {_has_full}")
print(f"  NEW _is_proxy_ui = not _has_full = {not _has_full}")
if not _has_full:
    print("  → Banner: 'Proxy Data Mode' SHOWN  ← still a problem")
else:
    print("  → Banner: 'Proxy Data Mode' HIDDEN  ✓ correct")
    print("  → No proxy warning; 'disk_cache_fresh' branch also False (no pkl cache)")
    print("  → Banner shows nothing (SQLite path: neither warning nor fresh-cache badge)")
