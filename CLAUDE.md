# Propex MLB Model — Claude Code Guide

## What this is
A local Streamlit MLB prop-betting model that scores batters/pitchers for **Over 1.5 Total Bases,
Over 0.5 hits, pitcher strikeouts, home runs, and moneyline**. Real money rides on the output, so
**reliability and data honesty beat features.** Always.

## Why we're rebuilding — read this first
The current V2.1 is a ~10k-line single file that **silently substitutes proxy / league-average values
for missing data and shows them as real, confident tiers.** That is the core defect. The Savant/FanGraphs
data was blocked because the old host (Streamlit Cloud) ran on blocklisted datacenter IPs; running locally
fixes the reachability. The job now is a disciplined modular rebuild that **never hides missing data.**

## Non-negotiable principles (do not violate)
1. **Fail loud, never silent.** Missing data must be visible, never papered over with a default that looks real.
2. **No fabrication.** Never derive Statcast values (barrel%, EV50, bat speed, blast rate, wRC+) from other
   stats and present them as measured. If a value isn't measured: omit it, or flag it explicitly as derived.
3. **Provenance per field.** Every stat carries a source tag: `measured` / `proxy` / `league_avg`. The UI shows it.
4. **Bettable gate.** A prop is bettable ONLY if its *required core* is all `measured`: player matched to real
   stats (not league-avg), real batter handedness (not defaulted R), opposing SP known with real stats, lineup
   confirmed (real slot), plus the market's core stats measured (TB/O0.5: K%, xSLG, wOBA, HH% · HR: barrel%,
   HH%, ISO · K props: SP K% + opp lineup K% · ML: both SPs' vuln stats, both lineups' wRC+, odds present).
   Otherwise it **greys out, is marked NON-BETTABLE with reasons, and is excluded from parlay/portfolio
   auto-selection.** Imperfect data = no bet.
5. **Validate before trust.** No market is "done" until the backtest shows its calibration + ROI.
   Real data ≠ a proven edge — the backtest decides that.

## How to work
- Follow `context_05_overhaul_changelog.md` (the tracker). Phase order: **S → 0 → 1 → 2 → 3 → 4.**
- Do **one tracker item at a time.** Don't batch unrelated changes.
- Use **Plan Mode** for any multi-file change (especially Phase 2 modularization): show the plan, get approval,
  then execute.
- After every change: run `python3 -m py_compile` then `pytest`. A change is not trusted until both pass.
- **Reuse the proven scoring math** from the existing file. The rebuild is about structure + honesty, not
  re-deriving weights. Do NOT re-tune weights by feel — weight calibration is Phase 3, driven by backtest data.
- When an item is genuinely done AND tested: check it off in the tracker and append a dated session-log line.

## Git discipline
- Branch per phase. Commit per completed tracker item with a clear message.
- **Never blind-accept large diffs on this project.** Surface the diff, explain what changed and why, let me review.

## Target structure (Phase 2)
```
data/      # fetch_pipeline, loaders, validation, provenance
scoring/   # batter, pitcher, platoon, park, weather, vegas, streak, bvp, k, hr — pure functions, NO streamlit
markets/   # tb_o15, hits_o05, k_props, hr, moneyline — each owns its weights/calibration
ui/        # thin Streamlit tabs — render only, call into scoring/markets
lib/       # constants (parks, teams, handedness), name matching, utils
tests/
app.py     # thin shell
```

## Data sources
- **pybaseball** (Savant / FanGraphs / BBRef) — run locally, fetch once, cache to disk, rate-limit politely.
- **MLB Stats API** — schedule, lineups, probable pitchers, game logs, vsPlayer (free, live).
- **Open-Meteo** — weather (free).
- **The Odds API** — odds; Business tier needed for real player props + historical (for the backtest).
- **ump-scorecards** — umpire K/zone (K props, O0.5).

## Do NOT
- Touch the DFS tabs (FanDuel / DraftKings / PrizePicks) until everything else is validated.
- Refactor the monolith piecemeal — extract cleanly into the structure above.
- Ship any change that makes the model display a number it cannot source.
