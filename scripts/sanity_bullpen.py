"""
sanity_bullpen.py — Run the REAL compute_team_bullpen_scores() with real data.

Fetches:
  1. MLB Stats API pitching (playerPool=All) → K%, WHIP, GS, G, IP
  2. FanGraphs type=8 → FIP, xFIP (the source the monolith merges)
  3. FanGraphs type=24 → Hard% allowed (Savant)

Merges them and calls scoring.pitcher.compute_team_bullpen_scores() — the actual
function the app calls. Shows:
  - All 30 teams sorted worst→best
  - Raw K%/FIP/WHIP/Hard% inputs for worst and best teams
  - Which data columns are populated vs defaulted
  - League-average calculation to anchor the 0–100 scale
  - Effect on final blended pitcher score vs old 42.0 baseline
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import requests
import pandas as pd
from scoring.pitcher import compute_team_bullpen_scores

SEASON = 2026
FG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Accept": "application/json,*/*",
    "Referer": "https://www.fangraphs.com/leaders/major-league",
}
FG_URL = "https://www.fangraphs.com/api/leaders/major-league/data"
from lib.constants import TEAM_ABB_MAP

# ─── 1. MLB Stats API pitching ────────────────────────────────────────────────
print("Fetching MLB Stats API pitching (playerPool=All)...")
url_pit = (
    f"https://statsapi.mlb.com/api/v1/stats"
    f"?stats=season&group=pitching&season={SEASON}"
    f"&limit=2000&offset=0&sportId=1&playerPool=All"
)
r = requests.get(url_pit, timeout=30)
splits = r.json().get("stats", [{}])[0].get("splits", [])
print(f"  → {len(splits)} splits")

rows = []
for s in splits:
    p   = s.get("player", {})
    st_ = s.get("stat", {})
    tm  = s.get("team", {})
    try:
        ip = float(str(st_.get("inningsPitched", "0") or "0"))
    except (ValueError, TypeError):
        ip = 0.0
    so   = int(st_.get("strikeOuts", 0) or 0)
    bb   = int(st_.get("baseOnBalls", 0) or 0)
    tbf  = int(st_.get("battersFaced", 0) or 0)
    h    = int(st_.get("hits", 0) or 0)
    er   = int(st_.get("earnedRuns", 0) or 0)
    gs   = int(st_.get("gamesStarted", 0) or 0)
    g    = int(st_.get("gamesPlayed", 0) or 0)
    _tm_name = tm.get("name", "")
    _tm_abbr = TEAM_ABB_MAP.get(_tm_name, tm.get("abbreviation", ""))
    rows.append({
        "mlbam_id": str(p.get("id", "")),
        "_name":    p.get("fullName", ""),
        "Team":     _tm_abbr,
        "IP":       ip,
        "GS":       gs,
        "G":        g,
        "K%":       round(so / tbf, 4) if tbf > 0 else None,
        "WHIP":     round((h + bb) / ip, 3) if ip > 0 else None,
        "ERA":      round(er / ip * 9, 2) if ip > 0 else None,
    })

mlb_df = pd.DataFrame(rows)
print(f"  → {len(mlb_df)} rows in df, columns: {list(mlb_df.columns)}")

# ─── 2. FanGraphs type=8 (FIP, xFIP, WHIP) ───────────────────────────────────
print("\nFetching FanGraphs type=8 (FIP)...")
fg8_df = pd.DataFrame()
for yr in [SEASON, SEASON - 1]:
    try:
        r8 = requests.get(FG_URL, params={
            "pos": "all", "stats": "pit", "lg": "all", "qual": "0",
            "type": "8", "season": yr, "season1": yr, "ind": "0",
            "team": "0", "pageitems": "1000", "pagenum": "1", "minip": "0",
        }, headers=FG_HEADERS, timeout=15)
        if r8.status_code == 200 and r8.json().get("data"):
            fg8_df = pd.DataFrame(r8.json()["data"])
            print(f"  → FG type=8 ({yr}): {len(fg8_df)} rows, cols sample: {list(fg8_df.columns[:15])}")
            break
        else:
            print(f"  → FG type=8 ({yr}): HTTP {r8.status_code}")
    except Exception as e:
        print(f"  → FG type=8 ({yr}): {e}")

# ─── 3. FanGraphs type=24 (Hard%, Barrel% from Savant) ───────────────────────
print("\nFetching FanGraphs type=24 (Hard% allowed)...")
fg24_df = pd.DataFrame()
for yr in [SEASON, SEASON - 1]:
    try:
        r24 = requests.get(FG_URL, params={
            "pos": "all", "stats": "pit", "lg": "all", "qual": "0",
            "type": "24", "season": yr, "season1": yr, "ind": "0",
            "team": "0", "pageitems": "1000", "pagenum": "1", "minip": "0",
        }, headers=FG_HEADERS, timeout=15)
        if r24.status_code == 200 and r24.json().get("data"):
            fg24_df = pd.DataFrame(r24.json()["data"])
            print(f"  → FG type=24 ({yr}): {len(fg24_df)} rows, cols sample: {list(fg24_df.columns[:20])}")
            break
        else:
            print(f"  → FG type=24 ({yr}): HTTP {r24.status_code}")
    except Exception as e:
        print(f"  → FG type=24 ({yr}): {e}")

# ─── 4. Merge ─────────────────────────────────────────────────────────────────
print("\nMerging...")
merged = mlb_df.copy()

# FG8: find ID column and grab FIP
if not fg8_df.empty:
    for id_col in ("xMLBAMID", "MLBAMID", "IDfg", "playerid", "PlayerID"):
        if id_col in fg8_df.columns:
            fg8_sub = fg8_df.copy()
            fg8_sub["_fgid"] = fg8_sub[id_col].apply(
                lambda x: str(int(float(x))) if pd.notna(x) and str(x) not in ("", "nan") else ""
            )
            keep = [c for c in ["FIP", "xFIP", "SIERA"] if c in fg8_sub.columns]
            if keep:
                fg8_sub = fg8_sub[["_fgid"] + keep].drop_duplicates("_fgid")
                fg8_sub = fg8_sub.rename(columns={"_fgid": "mlbam_id"})
                merged = merged.merge(fg8_sub, on="mlbam_id", how="left")
                fip_hit = merged["FIP"].notna().sum() if "FIP" in merged.columns else 0
                print(f"  FG8 merged: FIP populated for {fip_hit}/{len(merged)} pitchers")
            break
    else:
        print("  FG8: no ID column found")

# FG24: find ID column and grab Hard%
if not fg24_df.empty:
    hard_cols = [c for c in fg24_df.columns if any(k in c.lower() for k in ("hard", "hh%", "ev", "barrel"))]
    print(f"  FG24 Statcast-style columns: {hard_cols}")
    for id_col in ("xMLBAMID", "MLBAMID", "IDfg", "playerid", "PlayerID"):
        if id_col in fg24_df.columns:
            fg24_sub = fg24_df.copy()
            fg24_sub["_fgid"] = fg24_sub[id_col].apply(
                lambda x: str(int(float(x))) if pd.notna(x) and str(x) not in ("", "nan") else ""
            )
            keep24 = [c for c in hard_cols if c not in merged.columns]
            if keep24:
                fg24_sub = fg24_sub[["_fgid"] + keep24].drop_duplicates("_fgid")
                fg24_sub = fg24_sub.rename(columns={"_fgid": "mlbam_id"})
                merged = merged.merge(fg24_sub, on="mlbam_id", how="left")
                print(f"  FG24 merged cols: {keep24}")
            break

# Print final column list
print(f"\nFinal pitching_df columns: {list(merged.columns)}")
print(f"FIP populated: {'FIP' in merged.columns and merged['FIP'].notna().sum()} rows")
hh_col = next((c for c in ["Hard%", "HardHit%", "HH%"] if c in merged.columns), None)
print(f"Hard% column: {hh_col!r} → {'populated' if hh_col else 'MISSING — will default to 0.340'}")

# ─── 5. Run the REAL function ─────────────────────────────────────────────────
print("\n" + "="*60)
print("compute_team_bullpen_scores() — REAL FUNCTION OUTPUT")
print("="*60)

scores = compute_team_bullpen_scores(merged)
print(f"Teams returned: {len(scores)}/30")

if not scores:
    print("ERROR: empty dict — inspect column names above")
    sys.exit(1)

sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
vals = list(scores.values())
print(f"Range: {min(vals):.1f} – {max(vals):.1f}   (league-avg baseline = 42.0)")
print(f"All-same: {len(set(round(v,1) for v in vals)) == 1}")
print()
print(f"  {'Team':5s} {'Score':>6s}")
for team, score in sorted_scores:
    marker = " ← worst" if score == max(vals) else (" ← best" if score == min(vals) else "")
    print(f"  {team:5s} {score:6.1f}{marker}")

# ─── 6. Raw inputs for best and worst ─────────────────────────────────────────
print()
print("="*60)
print("RAW INPUTS: worst bullpen vs best bullpen")
print("="*60)

worst_team = sorted_scores[0][0]
best_team  = sorted_scores[-1][0]

def get_team_reliever_inputs(df, team):
    """Filter to relievers for a team, show median K%/FIP/WHIP/Hard%."""
    t = df[df["Team"] == team].copy()
    if t.empty:
        return None

    # Classify relievers using same logic as the function
    if "GS" in t.columns and "G" in t.columns:
        t["_GS"] = pd.to_numeric(t["GS"], errors="coerce").fillna(0)
        t["_G"]  = pd.to_numeric(t["G"],  errors="coerce").fillna(0)
        t["_IP"] = pd.to_numeric(t["IP"], errors="coerce").fillna(0)
        rel = t[(t["_GS"] == 0) | ((t["_G"] > 0) & (t["_GS"] / t["_G"].replace(0,1) < 0.30))]
        if len(rel) / max(1, len(t)) > 0.70:
            t["_ipg"] = t["_IP"] / t["_G"].replace(0,1)
            rel = t[t["_ipg"] < 2.0]
    else:
        rel = t

    fip_col  = "FIP"  if "FIP"  in rel.columns else "xFIP" if "xFIP" in rel.columns else None
    hh_col_l = next((c for c in ["Hard%", "HardHit%", "HH%"] if c in rel.columns), None)

    def med(col):
        if col and col in rel.columns:
            v = pd.to_numeric(rel[col], errors="coerce").dropna()
            return v.median() if not v.empty else None
        return None

    return {
        "n_relievers": len(rel),
        "K%_med":      med("K%"),
        "FIP_med":     med(fip_col),
        "WHIP_med":    med("WHIP"),
        "Hard%_med":   med(hh_col_l),
        "FIP_col":     fip_col,
        "Hard%_col":   hh_col_l,
    }

def show_team(team, score):
    inp = get_team_reliever_inputs(merged, team)
    if inp is None:
        print(f"  {team}: no rows found")
        return
    kv   = f"{inp['K%_med']*100:.1f}%" if inp['K%_med'] is not None else "DEFAULT(22.8%)"
    fipv = f"{inp['FIP_med']:.2f}"    if inp['FIP_med']  is not None else f"DEFAULT(4.50) [{inp['FIP_col']!r} col missing]"
    whv  = f"{inp['WHIP_med']:.3f}"   if inp['WHIP_med'] is not None else "DEFAULT(1.35)"
    hhv  = f"{inp['Hard%_med']*100:.1f}%" if inp['Hard%_med'] is not None else f"DEFAULT(34.0%) [{inp['Hard%_col']!r} col missing]"
    print(f"  {team}  score={score:.1f}  n_relievers={inp['n_relievers']}")
    print(f"    K%(med)={kv}  FIP(med)={fipv}  WHIP(med)={whv}  Hard%(med)={hhv}")

print(f"\nWorst bullpen: {worst_team} ({scores[worst_team]:.1f})")
show_team(worst_team, scores[worst_team])

print(f"\nBest bullpen:  {best_team} ({scores[best_team]:.1f})")
show_team(best_team, scores[best_team])

# ─── 7. Scale sanity — league-average calculation ─────────────────────────────
print()
print("="*60)
print("SCALE ANCHOR — what does league-average input produce?")
print("="*60)

LG_K    = 0.228
LG_FIP  = 4.50
LG_WHIP = 1.35
LG_HH   = 0.340

k_vuln    = max(0.0, min(100.0, (0.35 - LG_K)   / (0.35 - 0.10) * 100))
era_vuln  = max(0.0, min(100.0, (LG_FIP - 2.0)  / (7.0  - 2.0)  * 100))
whip_vuln = max(0.0, min(100.0, (LG_WHIP - 0.90)/ (1.80  - 0.90) * 100))
hh_vuln   = max(0.0, min(100.0, (LG_HH - 0.28)  / (0.50  - 0.28) * 100))
lg_score  = k_vuln * 0.38 + hh_vuln * 0.22 + era_vuln * 0.22 + whip_vuln * 0.18

print(f"  Input: K%={LG_K*100:.1f}%  FIP={LG_FIP}  WHIP={LG_WHIP}  Hard%={LG_HH*100:.1f}%")
print(f"  k_vuln={k_vuln:.1f}  era_vuln={era_vuln:.1f}  whip_vuln={whip_vuln:.1f}  hh_vuln={hh_vuln:.1f}")
print(f"  League-avg bp_score = {lg_score:.1f}  (old hardcoded default = 42.0)")

# ─── 8. Effect on final blended pitcher score ─────────────────────────────────
print()
print("="*60)
print("DOWNSTREAM EFFECT on compute_pitcher_score() — league-avg SP")
print("="*60)

# A league-average SP in FIP-only mode (no Savant HH%/barrel%):
# k_vuln=50, era_vuln=50, whip_vuln=50 → sp_score = 50*0.50 + 50*0.35 + 50*0.15 = 50
SP_SCORE_LG = 50.0

bp_old  = 42.0
bp_best = scores[best_team]
bp_worst= scores[worst_team]
bp_lg   = lg_score

blended_old   = SP_SCORE_LG * 0.60 + bp_old   * 0.40
blended_best  = SP_SCORE_LG * 0.60 + bp_best  * 0.40
blended_worst = SP_SCORE_LG * 0.60 + bp_worst * 0.40
blended_lg    = SP_SCORE_LG * 0.60 + bp_lg    * 0.40

print(f"  For league-avg SP (sp_score=50), blended = sp*0.60 + bullpen*0.40:")
print(f"    old hardcoded 42.0 bullpen  → blended = {blended_old:.1f}")
print(f"    new league-avg {bp_lg:.1f} bullpen  → blended = {blended_lg:.1f}")
print(f"    new best bullpen ({best_team}) {bp_best:.1f} → blended = {blended_best:.1f}")
print(f"    new worst bullpen ({worst_team}) {bp_worst:.1f} → blended = {blended_worst:.1f}")
print()
drift = blended_lg - blended_old
print(f"  Systematic drift at league-avg: {drift:+.1f} pts vs old baseline")
print(f"  (Positive = scores shifted higher/more-hittable vs old all-42 assumption)")
