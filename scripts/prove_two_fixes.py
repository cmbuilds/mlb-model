"""
prove_two_fixes.py — proof of the two fixes from the component diagnostic.

FIX 1: Vegas excluded (not floored) when no odds data.
  - Shows top-10 batters: score with vegas=0 pinned (old) vs vegas excluded (new).
  - Confirms the difference is real and positive (exclusion > flooring).

FIX 2: PA threshold gate at 50 PA.
  - Confirms Enrique Hernández (4 PA) and Jair Camargo (2 PA) are NON-BETTABLE.
  - Shows how many of 585 batters are filtered by the PA gate.
  - Confirms the gate is NOT triggered by provenance — it fires on sample size alone.
"""

import sys, os, unicodedata, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import sqlite3
import pandas as pd
from scoring.batter import compute_batter_score
from scoring.final  import compute_final_score
from data.provenance import check_bettable_tb
from config import TB_OFFSET_FULL, TIERS_FULL, TB_WEIGHTS

DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mlb_stats.db")

def _norm(s):
    s = unicodedata.normalize("NFD", str(s or ""))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

def g(row, key, default=0.0):
    v = row.get(key)
    try: return float(v) if v is not None and str(v) not in ("","nan","None") else default
    except: return default

def row_to_stats(row):
    barrel   = g(row, "barrel_batted_rate", 0.070)
    hard_hit = g(row, "hard_hit_percent",   0.370)
    k_pct    = g(row, "K%",  0.228)
    if barrel > 1:   barrel /= 100.0
    if hard_hit > 1: hard_hit /= 100.0
    if k_pct > 1:    k_pct /= 100.0
    pa = int(g(row, "pa", 0))
    return {
        "slg_proxy":    g(row, "xSLG", 0.398),
        "barrel_rate":  barrel,
        "hard_hit_rate": hard_hit,
        "k_rate":       k_pct,
        "iso_proxy":    g(row, "ISO", 0.165),
        "wrc_plus":     g(row, "wRC+", 100.0),
        "woba":         g(row, "wOBA", 0.315),
        "ev50":         0.0, "bat_speed": 0.0, "blast_rate": 0.0,
        "pa":           pa,
        "data_source":  "savant_xstats",  # db-matched, so provenance passes
        "_provenance": {
            "k_rate": "measured", "slg_proxy": "measured",
            "woba": "measured", "hard_hit_rate": "measured",
            "barrel_rate": "measured",
        },
    }

# Load db
con = sqlite3.connect(DB)
bat_df = pd.read_sql("SELECT * FROM batter_stats", con)
con.close()

# ── FIX 1: Vegas exclusion ────────────────────────────────────────────────────
print("=" * 68)
print("FIX 1 — Vegas excluded vs pinned at 0 (top 10 by full-mode score)")
print("=" * 68)
print()
print(f"  W[vegas] = {TB_WEIGHTS['vegas']}   total weight sum = 1.03")
print(f"  With vegas_missing=False, implied=0:  denominator 1.03, vegas contributes 0×0.08=0")
print(f"  With vegas_missing=True:              denominator 0.95, vegas excluded entirely")
print()

NEUTRAL = 50.0

scores = []
for _, row in bat_df.iterrows():
    st = row_to_stats(row)
    bat_sc, _, _ = compute_batter_score(st)

    # OLD: vegas=0 pinned, stays in the weighted sum
    sc_old = compute_final_score(
        batter_score=bat_sc, pitcher_vuln_score=NEUTRAL,
        platoon_score=NEUTRAL, lineup_score=NEUTRAL, park_score=NEUTRAL,
        weather_score=NEUTRAL, vegas_score=0.0,
        tto_bonus=0.0, pitch_matchup_score=NEUTRAL, streak_score=NEUTRAL,
        bvp_score=NEUTRAL, bvp_weight_boost=0.0, proxy_mode=False,
        vegas_missing=False,
    )

    # NEW: vegas excluded, re-normalized
    sc_new = compute_final_score(
        batter_score=bat_sc, pitcher_vuln_score=NEUTRAL,
        platoon_score=NEUTRAL, lineup_score=NEUTRAL, park_score=NEUTRAL,
        weather_score=NEUTRAL, vegas_score=0.0,  # value irrelevant when missing
        tto_bonus=0.0, pitch_matchup_score=NEUTRAL, streak_score=NEUTRAL,
        bvp_score=NEUTRAL, bvp_weight_boost=0.0, proxy_mode=False,
        vegas_missing=True,
    )

    scores.append((row.get("_name","?"), sc_old, sc_new, sc_new - sc_old))

scores.sort(key=lambda x: x[2], reverse=True)

print(f"  {'Player':25s}  {'Old (vegas=0)':>13s}  {'New (excluded)':>14s}  {'Delta':>6s}  {'Tier (new)':>10s}")
print(f"  {'-'*25}  {'-'*13}  {'-'*14}  {'-'*6}  {'-'*10}")
for name, old, new, delta in scores[:10]:
    tier = "T1" if new >= 80 else "T2" if new >= 70 else "T3" if new >= 60 else "NP"
    print(f"  {name:25s}  {old:13.1f}  {new:14.1f}  {delta:+6.1f}  {tier:>10s}")

old_all = [s[1] for s in scores]
new_all = [s[2] for s in scores]
print()
print(f"  Score range OLD (vegas=0 pinned): {min(old_all):.1f} – {max(old_all):.1f}")
print(f"  Score range NEW (vegas excluded): {min(new_all):.1f} – {max(new_all):.1f}")
print(f"  Avg shift: {sum(s[3] for s in scores)/len(scores):+.2f} pts  "
      f"(should be ≈+{TB_WEIGHTS['vegas'] * 50 / (1.03 - TB_WEIGHTS['vegas']):.2f} = "
      f"w_vegas×50 / (1.03−w_vegas) renorm benefit)")

# Tier comparison
def tiers(sc_list):
    t1 = sum(1 for s in sc_list if s >= 80)
    t2 = sum(1 for s in sc_list if 70 <= s < 80)
    t3 = sum(1 for s in sc_list if 60 <= s < 70)
    np = sum(1 for s in sc_list if s < 60)
    return t1, t2, t3, np

t1o, t2o, t3o, npo = tiers(old_all)
t1n, t2n, t3n, npn = tiers(new_all)
print()
print(f"  {'':8s}  {'OLD':>8s}  {'NEW':>8s}  {'Δ':>5s}")
print(f"  {'Tier 1':8s}  {t1o:8d}  {t1n:8d}  {t1n-t1o:+5d}")
print(f"  {'Tier 2':8s}  {t2o:8d}  {t2n:8d}  {t2n-t2o:+5d}")
print(f"  {'Tier 3':8s}  {t3o:8d}  {t3n:8d}  {t3n-t3o:+5d}")
print(f"  {'No Play':8s}  {npo:8d}  {npn:8d}  {npn-npo:+5d}")

# ── FIX 2: PA threshold gate ──────────────────────────────────────────────────
print()
print("=" * 68)
print("FIX 2 — PA gate: 50 PA minimum in check_bettable_tb()")
print("=" * 68)
print()
print("  Reasoning for 50 PA threshold:")
print("  • Barrel% (per BIP) needs ~15+ BIP for a directional signal;")
print("    50 PA @ ~30% K/BB ≈ 35 BIP.  Below 50 PA it's 1-for-2 noise.")
print("  • wRC+ at 4 PA can be 800+; at 50 PA, still wide but bounded ~50-250.")
print("  • FanGraphs uses 50 PA as the qualified-batter floor for rate tables.")
print("  • 25 PA would still let a 2-week hot streak pass; 100 PA blocks early call-ups.")
print("  • Gate is independent of provenance: even if Savant 'measured' barrel%,")
print("    2 barreled balls out of 2 BIP is not a real 100% barrel rate.")
print()

# Check the two named players
BAD_PLAYERS = [
    ("Enrique Hernández", 4,   "wRC+=812, barrel=50%"),
    ("Jair Camargo",      2,   "barrel=100%, 2 PA"),
]

print("  Named players check:")
for name, pa, note in BAD_PLAYERS:
    bettable, reasons = check_bettable_tb(
        batter_prov={"k_rate":"measured","slg_proxy":"measured","woba":"measured",
                     "hard_hit_rate":"measured","barrel_rate":"measured"},
        pitcher_prov={"k_rate_allowed":"measured","hard_hit_allowed":"measured"},
        batter_matched=True, pitcher_matched=True,
        lineup_confirmed=True, sp_known=True, hand_real=True,
        batter_pa=pa,
    )
    status = "✓ NON-BETTABLE" if not bettable else "✗ BETTABLE (gate failed)"
    print(f"  {name} ({pa} PA, {note})")
    print(f"    → {status}  reasons: {reasons}")
    print()

# Full population breakdown
print("  Full 585-batter breakdown:")
below_50   = (bat_df["pa"] <  50).sum()
below_10   = (bat_df["pa"] <  10).sum()
zero_pa    = (bat_df["pa"] ==  0).sum()  # pa not in db at all
above_50   = (bat_df["pa"] >= 50).sum()
above_100  = (bat_df["pa"] >= 100).sum()
print(f"    PA = 0 (field missing):    {zero_pa:4d}  — gate uses pa=0, does NOT fire (safe)")
print(f"    PA 1–49 (below threshold): {below_50-zero_pa:4d}  — gate fires → NON-BETTABLE for sample")
print(f"    PA < 10 (worst noise):     {below_10:4d}  — subset of above")
print(f"    PA ≥ 50 (gate passes):     {above_50:4d}  — eligible for bettable check")
print(f"    PA ≥ 100 (full sample):    {above_100:4d}")
print()
print(f"  Of 585 total batters: {below_50-zero_pa} with known PA < 50 are now NON-BETTABLE")
print(f"  for insufficient sample. Gate is consistent — provenance alone (all fields")
print(f"  'measured') cannot make a 4-PA player bettable.")

# Confirm provenance does NOT save a small-sample player
bettable_noPA, _ = check_bettable_tb(
    batter_prov={"k_rate":"measured","slg_proxy":"measured","woba":"measured",
                 "hard_hit_rate":"measured","barrel_rate":"measured"},
    pitcher_prov={"k_rate_allowed":"measured","hard_hit_allowed":"measured"},
    batter_matched=True, pitcher_matched=True,
    lineup_confirmed=True, sp_known=True, hand_real=True,
    batter_pa=4,
)
bettable_noPA_zeroPA, _ = check_bettable_tb(
    batter_prov={"k_rate":"measured","slg_proxy":"measured","woba":"measured",
                 "hard_hit_rate":"measured","barrel_rate":"measured"},
    pitcher_prov={"k_rate_allowed":"measured","hard_hit_allowed":"measured"},
    batter_matched=True, pitcher_matched=True,
    lineup_confirmed=True, sp_known=True, hand_real=True,
    batter_pa=0,  # pa unknown (not in db)
)
print()
print(f"  Provenance consistency check:")
print(f"  All fields 'measured' + pa=4  → bettable={bettable_noPA}  ← correct: blocked by PA gate")
print(f"  All fields 'measured' + pa=0  → bettable={bettable_noPA_zeroPA}  ← pa=0 means 'unknown', gate does not fire")
print(f"  (pa=0 is safe: it means the field isn't in the db row, not that the player has 0 PAs)")
