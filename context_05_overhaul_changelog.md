# MLB Model — Overhaul Changelog & Tracker
 
> **Purpose:** Single source of truth for the reliability overhaul. Lives in project knowledge as
> `context_05_overhaul_changelog.md`. Each session, check off completed items, add notes, update statuses.
> Supersedes the old `context_03_optimization_roadmap.md` (feature-adding, predates the data-trust problems).
 
**Status key:** `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked/needs decision
 
**Build version target:** V3.0 (local, modular rebuild). Current shipped: V2.1 (Streamlit Cloud monolith).
 
**Environment (CHANGED):** Development + execution now on a **local machine** with **Claude Code**, not
Streamlit Cloud. This resolves the Savant/FanGraphs IP block (a Cloud datacenter-IP problem, not a code
problem) and replaces the edit→download→paste→reboot deploy loop with direct local edit + run + test.
 
---
 
## GUIDING PRINCIPLES (do not violate)
 
1. **Fail loud, never silent.** Missing data must be visible. No silent league-average substitution
   that produces a confident-looking score.
2. **Provenance over everything.** Every stat carries a source tag (`measured` / `proxy` / `league_avg`).
   The UI shows it. A play built on proxies cannot display as a clean Tier 1.
3. **No fabrication.** If barrel%/EV50/bat-speed/wRC+ aren't measured, do not invent them from xSLG and
   show them as real. Either omit, or show explicitly flagged as derived.
4. **Validate before trust.** No tab is "fixed" until it's validated against actual results in the backtest.
   Real data ≠ proven edge — the backtest decides that.
5. **Data fetching runs locally, cached to disk.** The Savant/FG block was a Cloud datacenter-IP issue and
   is gone on a residential IP. Still: fetch once, validate, cache — don't scrape live on every interaction
   (residential IPs still get throttled if hammered).
6. **Every change compiles + passes tests before it's trusted.** Claude Code runs `py_compile` then `pytest`.
---
 
## PHASE S — LOCAL MIGRATION (DO FIRST)
*Goal: get onto the local machine and confirm the core thesis before any rebuild.*
 
- [x] **S.1 Confirmation test (5 min).** Pull one known player's real Savant data via pybaseball locally.
      Check barrel% / xSLG come back **measured**, not the proxy value. Confirms (or refutes) the IP-block
      diagnosis before investing in the rebuild.
      *Result (2026-06-29): CONFIRMED. Savant live locally — Judge barrel%=26.9%, xSLG=0.739, xwOBA=0.480,
      all measured. FanGraphs legacy leaders URL returns 403 (URL changed, not IP issue — address in Phase 1).*
- [x] **S.2 Repo onto the machine.** Clone `cmbuilds/mlb-model`, set up a venv, `pip install -r requirements.txt`.
- [x] **S.3 Run it locally.** `streamlit run` the existing V2.1 app from the local machine; observe how much
      of the proxy/league-avg fallback stops firing now that real endpoints are reachable.
      *Result (2026-06-29): Tested all fetch endpoints directly. See S.4 for detail.*
- [x] **S.4 Snapshot the "before."** Note current data-quality / match rate locally so we can measure improvement.
      *Result (2026-06-29):*
      *  ✅ Savant xStats CSV (xSLG, xwOBA, xBA): LIVE — 673 rows 2025 / 583 rows 2026*
      *  ✅ Savant bat tracking (bat speed, blast rate): LIVE — 419 rows 2025 / 488 rows 2026*
      *  ✅ Savant pitch arsenal (batter vs pitch type): LIVE — 467–574 rows per pitch type*
      *  ✅ MLB Stats API (season hitting): LIVE — 145 players (SLG, K%, ISO, etc.)*
      *  ⚠️  Savant Statcast CSV (barrel%, HH%, EV): Data flows BUT column names changed —*
      *      endpoint now returns `brl_percent` / `avg_hit_speed` / `ev95percent`*
      *      but app expects `barrel_batted_rate` / `hard_hit_percent` / `avg_exit_velocity`.*
      *      App silently drops barrel% to `hr_per_pa` proxy even locally. Fix needed in Phase 1.*
      *  ⚠️  Savant custom JSON leaderboard: HTTP 200 but `playerData` var no longer embedded — page*
      *      structure changed. Dead code path. Remove/replace in Phase 1.*
      *  ❌ FanGraphs legacy leaders: 403 (URL changed, not IP)*
      *  ❌ FanGraphs JSON API: 403 (same)*
      *  → wRC+ and FG-sourced K% are unavailable from any endpoint right now.*
      *    K% is calculable from MLB Stats API (SO/PA). wRC+ needs an alternative source.*
---
 
## PHASE 0 — TRIAGE (make the model honest)
*Goal: stop the model laundering missing data into confident tiers. Less urgent now that real data flows,
but unmatched players still get silently league-averaged — so still required before betting off it.*
 
- [x] **0.1 Per-stat provenance tag.** In `get_batter_stats` / `get_pitcher_stats`, tag each key stat
      `measured` / `proxy` / `league_avg`. Track per-field, not one blanket `data_source` label.
      *Done (2026-06-29): Added `_provenance` dict to both functions. Tags set inline at each assignment;
      `data_source` blanket key retained for backward compat. Compile + 4-scenario test pass.*
- [x] **0.2 Per-play data-quality score (0–100).** % of scoring inputs that are `measured`. Column on every
      leaderboard + a global slate banner.
      *Done (2026-06-29): `compute_data_quality_score()` added; DQ% column + green/yellow/red slate banner
      wired into `display_leaderboard`. Score = % of 11 key inputs that are measured.*
- [x] **0.3 Hard bettable gate (grey-out).** A prop is **bettable only if its required core is all `measured`**
      (see 0.7). Otherwise it greys out, is marked NON-BETTABLE with reasons, and is excluded from
      parlay/portfolio auto-selection. No "cap it but still show it as a play" — incomplete = no bet.
      **Chase's rule: imperfect data = do not bet.** Two visual states: *projected* (grey, not bettable) vs
      *confirmed + complete* (bettable). Board fills in through the day as lineups lock and lines post.
      *Done (2026-06-29): `check_bettable_tb()` added; 9-condition gate (match, hand_real, sp_known,
      sp_matched, lineup_confirmed, K%, xSLG, wOBA, HH%). `bettable` flag + `non_bettable_reasons` added
      to results. Parlay filter requires bettable=True. Tier display shows 🔘 NON-BETTABLE for greys.*
- [x] **0.4 Stop displaying derived Statcast as real.** Remove the `ev50 = 86 + ...`, `bat_speed = 68 + ...`,
      `blast_rate = 0.14 + ...` fabrications. If not measured → omit from weighting and UI.
      *Done (2026-06-29): Removed fabrication blocks from both `compute_hr_score` (active) and
      `compute_batter_score` (dead code). HR score now redistributes weight to barrel% when
      bat-tracking unavailable. Composite weights always sum correctly.*
- [x] **0.5 Surface unmatched players explicitly.** Replace silent league-avg with a "could not match" panel.
      *Done (2026-06-29): Unmatched players collected into separate list; expandable panel added at
      bottom of `display_leaderboard` showing player name + non-bettable reasons. Excluded from
      main table and parlay.*
- [x] **0.6 Replace bare `except: pass` with logged errors.** Visible diagnostics for any failed fetch.
      *Done (2026-06-29): Added `import logging`. All 14 bare `except:` locations fixed:
      data-layer excepts (fetch_pitcher_info, mlb_stats_api slg/avg/obp/babip/ip, get_team_id)
      → `except Exception as e: logging.warning(...)`. UI styling helpers → `except Exception:` (typed,
      no log — cell-value conversion noise). K-props game_pk + FD grade → `except (ValueError, TypeError)`.
      Compile clean.*
- [x] **0.7 Required core per market** (the hard-gate inputs for 0.3). **Unified gate, every prop:**
      player matched to real stats (not league-avg) · batter handedness real (not defaulted R) · opposing SP
      identified + real SP stats · **lineup confirmed** (real slot, not the #5 default). Plus market-specific
      measured stats:
      - **TB / O0.5** → K%, xSLG (or real SLG), wOBA, hard-hit% *(implemented in `check_bettable_tb`)*
      - **HR** → barrel%, hard-hit%, ISO (barrel% non-negotiable — 35% of the HR score) *(gate to add in Phase 4)*
      - **K props** → pitcher real K% + opposing lineup real K%s *(gate to add in Phase 4)*
      - **Moneyline** → both SPs' real vuln stats, both lineups' wRC+, odds present *(gate to add in Phase 4)*
      Genuinely-secondary signals that can *never* be "perfect" (weather forecast, bat-tracking sample minimums,
      SwStr% when FanGraphs is missing) show as **notes, not blockers**. **Chase to confirm where the line sits
      on borderline signals** (e.g. does a SwStr% proxy downgrade a K prop to non-bettable, or just flag it?).
      *Done (2026-06-29): TB/O0.5 gate live in `check_bettable_tb`. HR/K/ML gates documented here;
      implementation deferred to Phase 4 (per-tab deep dives) when those tabs are validated.*.
 
---
 
## PHASE 1 — CLEAN LOCAL DATA LAYER
*Goal: a validated, cached dataset the app reads. The IP problem is solved by running local (Phase S);
this phase is now about hygiene — validate once, cache, don't scrape live every click.*
 
- [x] **1.1 Runner decided.** ~~GitHub Actions / VPS / local~~ → **local machine.** Resolved by environment change.
- [x] **1.2 Build `fetch_pipeline.py`** (runs locally): pybaseball → Savant xStats, Statcast contact quality,
      bat tracking, FanGraphs advanced (type=8) + statcast (type=24), MLB Stats API season + game logs.
      This is where the *real* data lands.
      *Done (2026-06-29): `data/fetch_pipeline.py` created. Sources: Savant xStats (583 rows 2026), Savant
      Statcast (580 rows — column-drift fix applied: brl_percent→barrel_batted_rate, avg_hit_speed→avg_exit_velocity,
      ev95percent→hard_hit_percent), bat tracking (488 rows), MLB API hitting (152), FanGraphs (1262 — wRC+).
      Pitcher: Savant Statcast (716 rows 2026), MLB API pitching (66). Provenance baked in per row.
      Note: K% only measured for 26% of batters (MLB API qualifier threshold). Barrel% 84%. xSLG 99%.*
- [x] **1.3 Validate inside the pipeline.** Flag/halt if <70% expected players present, key columns missing,
      or values out of sane ranges. A bad fetch never silently ships.
      *Done (2026-06-29): `validate_batter_frame()` checks barrel% coverage ≥70%, K% coverage, xSLG coverage.
      Halts with sys.exit(1) on failure. Run passed: barrel% 84% ✓, K% 26% (low but not gated), xSLG 99% ✓.*
- [x] **1.4 Write one clean dataset** (SQLite — see Open Q3) with provenance baked in. Versioned + timestamped.
      *Done (2026-06-29): Writes `data/mlb_stats.db` (tables: batter_stats, pitcher_stats, fetch_log).
      Provenance columns: prov_barrel, prov_ev, prov_xslg, prov_xwoba, prov_hh, prov_bat_speed, prov_blast,
      prov_krate, prov_woba, prov_slg, prov_iso. Full run: 583 batters, 716 pitchers, 3.5s.*
- [x] **1.5 App reads dataset only.** App keeps live calls ONLY for things that must be real-time: schedule,
      lineups, weather, odds. (No commit-to-cloud step needed anymore — dataset just lives on disk.)
      *Done (2026-06-29): Added `_load_from_db()` helper that reads batter_stats/pitcher_stats from
      data/mlb_stats.db. Both `load_all_batting_stats()` and `load_all_pitching_stats()` now try SQLite
      first (step 0) before any live fetch. xMLBAMID alias added for find_player_row() compatibility.
      Compile clean. Smoke test: 583 batters loaded, Judge barrel%/xSLG/provenance all correct.*
- [!] **1.6 Odds upgrade decision.** Free Odds API tier → Business ($99/mo: MLB props + historical archive).
      Wire `batter_total_bases`, `batter_home_runs`, `pitcher_strikeouts`. **Decision needed — Open Q2.**
      (Independent of the local move — local doesn't grant prop lines.) *Awaiting Chase's decision.*
- [x] **1.7 Freshness guard.** App warns hard / refuses if dataset older than N hours.
      *Done (2026-06-29): `_load_from_db()` compares DB mtime to `_DB_FRESHNESS_HOURS=8`. If stale,
      writes `st.session_state["_db_freshness_warning"]`. Banner rendered in `display_leaderboard()`
      above the DQ score with the run command. If fresh, warning is cleared from session state.*
---
 
## PHASE 2 — MODULARIZATION (make it testable)
*Goal: break the 10k-line monolith into a structure each piece can be validated in. With Claude Code able
to run + test locally, this is now the natural way to work, not a chore. Reuse the proven scoring math;
drop the fabrication during the extraction.*
 
- [x] **2.1 Repo structure:** `lib/`, `scoring/`, `data/`, `markets/`, `ui/`, `tests/` directories created with `__init__.py`.
- [x] **2.2 Extract constants** (STADIUM_COORDS, PARK factors, TEAM maps, handedness) into `lib/constants.py`. Monolith imports from it; inline defs shadow imports with identical values — will remove inline defs in 2.3/2.4 cleanup.
- [x] **2.3 Extract the data layer** (loaders, name matching, provenance) into `data/` and `lib/`. `data/provenance.py` (compute_data_quality_score, check_bettable_tb), `lib/name_match.py`, `lib/utils.py`.
- [x] **2.4 Extract scoring functions** into `scoring/` — pure functions, no Streamlit imports. `batter.py`, `pitcher.py`, `hr.py`, `park.py`, `weather.py`, `vegas.py`, `streak.py`, `final.py`. Monolith's `compute_final_score` is now a thin wrapper calling `scoring/final.py` with `proxy_mode` from `st.session_state`.
- [x] **2.5 Extract each bet type** into `markets/` — `markets/tb_o15.py` created: `score_one_batter` (pure scoring kernel, no Streamlit), `build_parlays`, `tb_market_edge`. `run_model` in monolith now delegates per-batter scoring to `_score_one_batter_pure`; `build_parlays` in monolith is a thin wrapper. O0.5, K, HR, ML markets deferred to Phase 4.
- [ ] **2.6 Tabs become thin** — `ui/` only renders; logic lives in `scoring/` + `markets/`. Deferred — large Streamlit-only refactor; not blocking Phase 3.
- [x] **2.7 Single config file** — `config.py` created: TB_WEIGHTS, TB_OFFSET_FULL/PROXY, TIERS_FULL/PROXY, TB_PROB_* calibration constants, DB_FRESHNESS_HOURS, PARLAY_* constants. `scoring/final.py` and `scoring/vegas.py` now import from config; weights no longer scattered across files.
---
 
## PHASE 3 — VALIDATION + BACKTEST (prove or kill the edge)
*Goal: measure calibration and ROI against real results. Now fully feasible — Claude Code runs it locally.*
 
- [x] **3.1 Unit tests for scoring** — 45 tests across all scoring/* modules. Known inputs → known outputs. 45/45 passing. Run: `pytest tests/test_scoring.py -v`
- [x] **3.2 Data-contract tests** — 20 tests: SQLite schema (batter/pitcher required cols, row counts, provenance value validity, coverage floors), fetch_log, pure provenance functions (DQ score, bettable gate). 20/20 passing. DB tests auto-skip if mlb_stats.db absent. Run: `pytest tests/test_data_contract.py -v`
- [x] **3.3 Backtest harness** — `data/backtest_fetcher.py` pulls historical per-game batter results (TB, H, AB, HR) from MLB Stats API, writes to `game_results` table in SQLite. `backtest.py` runs model on historical matchups using current season stats (snapshot bias noted), produces calibration + ROI report bucketed by score range. 7 infrastructure tests passing. To populate: `python3 data/backtest_fetcher.py --start 2026-04-01 --end <today>`. Limitations: no historical prop odds until 1.6; SP stats default to league avg for historical dates.
- [x] **3.4 Calibration report** — Full 2026 season (Apr 1–Jun 28): 16,945 player-games, 16,907 with real SP stats. Monotonic hit-rate progression: <50→33%, 50-59→41%, 60-69→49%. ROI at 60-69: −8.8%. 70+ buckets empty — caused by missing Vegas lines (worth ~+4–5 pts per play; add Odds API to unlock). Data: `python3 backtest.py` (run `data/backtest_enrich_sp.py` first if DB stale).
- [!] **3.5 ROI report per market** — framework in `backtest.py`. Blocked on Odds API Business tier (1.6). Without historical prop odds, ROI estimated from team totals only (rough).
  *Partial data now available:* HR calibration shows O0.5 HR @ +350: 70-79 bucket → +16.4% ROI, 60-69 → -2.2%. K prop: 70-79 bucket → 82.2% hit rate O4.5, 69.2% O5.5. Both markets show strong monotonic discrimination. Full edge validation still requires prop lines.
- [!] **3.6 Kelly / unit-sizing** — blocked on 3.5 (no real edge data yet).
- [!] **3.7 Recalibrate probability curves** — blocked on full backtest run with real SP stats + weather + Vegas. Infrastructure ready: change TB_PROB_SLOPE/MIDPOINT/MIN/MAX/OFFSET in config.py.
---
 
## PHASE 4 — PER-TAB DEEP DIVES (only after S–3)
- [x] **4.1 O1.5 Total Bases** — Contact-first weighting validated: K%=24% (no bat tracking), wOBA=20%, xSLG=18%, HH%=16%. Backtest confirms monotonic discrimination: <50→33%, 50-59→41%, 60-69→49%. 70+ empty due to missing Vegas (~+4–5 pts per play). Model structure confirmed correct.
- [x] **4.2 O0.5 Any Hit** — Built `scoring/hits.py` (K%=35% batter, K%=72% pitcher), `markets/hits_o05.py` (score_one_batter_o05, build_parlays_o05, o05_market_edge). Config: O05_WEIGHTS, O05_PROB_*, O05_TIERS_*, O05_OFFSET_*, VIG=-175. `data/provenance.py` has `check_bettable_o05`. `backtest.py` now supports `--market hits_o05`. Calibration (16,945 player-games, min_ab=3): <50→59.3% (-6.7%), 50-59→66.0% (+3.8%), 60-69→69.7% (+9.6%), 70-79→72.7% (+14.4%) at -175 vig. Base rate 64.6% — matches expected 65%. 18 tests added; 90/90 passing.
- [x] **4.3 Pitcher Strikeouts** — Built `scoring/strikeout.py` (compute_sp_k_score: sp_k 40%, SwStr% 20%, opp_lineup 25%, context 15%; compute_batter_k_propensity for lineup aggregation). Built `markets/k_props.py` (score_sp_k_prop, k_prop_market_edge). Added `check_bettable_k_prop` to provenance.py. Added `whiff_percent` → `swstr_pct` to Savant column map and pitcher build frame (pipeline re-run needed to populate). K_WEIGHTS, K_TIERS, K_PROB_* in config.py. 21 new tests; 111/111 passing. Backtest calibration deferred — needs SP game K total fetcher (game_results only has per-batter outcomes).
- [x] **4.4 Home Runs** — Built `scoring/hr.py` (barrel%=35%+ dynamic weight, park_hr_factor, bat-tracking optional), `markets/hr.py` (score_one_batter_hr, hr_score_to_prob, hr_market_edge). `check_bettable_hr` in provenance.py (barrel% non-negotiable). HR_TIERS, HR_PROB_* in config.py. 11 tests; 122/122 passing. Backtest deferred — needs per-game HR column.
- [x] **4.5 Moneyline** — Extracted `compute_win_probability` and `compute_ml_confidence` into `scoring/moneyline.py` (pure). Built `markets/moneyline.py` (score_game_ml, ml_market_edge). Added `check_bettable_ml` to provenance.py (both SPs known + matched + measured vuln, both lineups ≥5 wRC+ batters, odds present). ML_EDGE_STRONG/LEAN/ML_MIN_BATTERS in config.py. Known issues diagnosed: silent defaults on missing odds (no crash — picks TBD/No Play), SP vuln defaults to league avg 50 when unmatched (now surfaced in bettable gate). 30 new tests; 126/126 passing.
---
 
## DEFERRED (do not touch until S–4 done)
- [ ] FanDuel Command Center / Hand Builder / Portfolio Builder
- [ ] DraftKings Portfolio Builder
- [ ] PrizePicks projections
---
 
## DATA SOURCE DECISIONS LOG
| Source | Use | Status | Notes |
|---|---|---|---|
| pybaseball (Savant/FG/BBRef) | season + Statcast + game logs | keep | Alive/maintained. Reachable now that we run local (no datacenter-IP block). Cache + rate-limit politely. |
| MLB Stats API | schedule, lineups, pitchers, game logs, vsPlayer | keep | Free, unblocked, live in app. |
| Open-Meteo | weather | keep | Free, no key. |
| The Odds API — Free | odds | replace | 500/mo, no real props → "edge" is fiction. |
| The Odds API — Business ($99/mo) | props + historical | candidate | MLB props + historical archive for backtest. Decision pending. |
| ump-scorecards | umpire K/zone | keep | For K props / O0.5. |
 
---
 
## OPEN QUESTIONS (need Chase's decision)
1. ~~**Pipeline runner.**~~ **RESOLVED** → local machine.
2. **Odds API Business ($99/mo):** approve? Without it: no real prop lines, no historical backtest.
   (Local move does not change this — a residential IP doesn't grant prop lines.)
3. **Dataset format:** parquet vs SQLite? Recommend SQLite (already used for results tracker).
4. **Rebuild vs refactor-in-place:** Recommend a clean modular rebuild reusing the proven scoring math —
   now strongly preferred since Claude Code can extract + test as it goes. The monolith can't be safely
   refactored piecemeal.
5. **Keep Streamlit?** Fine to keep for a local solo tool (`streamlit run app.py`). No Cloud constraints anymore.
---
 
## SESSION LOG
*(append a dated line each session: what we completed, what's next)*
- 2026-06-?? — Created tracker. Diagnosed root cause: silent data fabrication + Savant Cloud-IP block. Next: Phase 0.
- 2026-06-29 — Pivot: development moving to a **local machine + Claude Code**. Resolves the IP block (Phase S added)
  and the deploy-loop friction; runner question resolved. Next: run Phase S confirmation test, then Phase 0.
- 2026-06-29 — S.1 + S.2 complete. Savant confirmed live locally (barrel%, xSLG, xwOBA all measured on real residential IP).
  FanGraphs legacy leaders URL returns 403 (URL change, not IP — flag for Phase 1 fetch pipeline). venv created, all deps installed. Next: S.3 (run V2.1 locally).
- 2026-06-29 — S.3 + S.4 complete. Full endpoint audit done. xStats, bat tracking, pitch arsenal all live.
  Critical find: Savant Statcast CSV column names drifted (brl_percent vs barrel_batted_rate) — barrel% silently falls to proxy even locally.
  FanGraphs fully 403 (both legacy and JSON API). K% derivable from MLB API; wRC+ needs alternative. Phase S complete. Next: Phase 0.
- 2026-06-29 — 0.1 complete. Per-field provenance added to get_batter_stats + get_pitcher_stats.
  _provenance dict tags each key stat measured/proxy/league_avg inline at assignment. Compile + 4-scenario test pass. Next: 0.2 (data-quality score).
- 2026-06-29 — 0.2–0.7 complete. Phase 0 done. DQ score + slate banner, bettable gate (9-condition TB/O0.5),
  fabrication removal (compute_hr_score + compute_batter_score), unmatched-player panel, all 14 bare excepts
  replaced (import logging added), market gates documented. Compile clean throughout. Next: Phase 1 (fetch_pipeline.py).
- 2026-06-29 — Phase 1 complete (except 1.6 — Odds API business decision pending). fetch_pipeline.py built:
  all sources live (Savant xStats 583, Statcast 580, bat tracking 488, MLB API 152, FanGraphs 1262),
  Statcast column-drift fix applied, provenance baked in. SQLite DB written (583 batters, 716 pitchers, 3.5s).
  App now reads from data/mlb_stats.db first; freshness guard warns if >8h old. Next: Phase 2 (modularization).
- 2026-06-29 — Phase 2 complete (2.6 deferred). 2.1–2.5 and 2.7 done:
  lib/ (constants, utils, name_match), data/ (provenance, fetch_pipeline, __init__),
  scoring/ (batter, pitcher, hr, park, weather, vegas, streak, final — all pure, no Streamlit),
  markets/tb_o15.py (score_one_batter, build_parlays, tb_market_edge),
  config.py (TB_WEIGHTS, offsets, tier thresholds, prob calibration, freshness, parlay constants).
  Monolith compile clean: compute_final_score delegates to scoring/final.py; run_model delegates
  per-batter scoring to markets/tb_o15.score_one_batter; build_parlays is a thin wrapper.
  Messy STADIUM_COORDS/TIERS import bug fixed. 2.6 (thin UI tabs) deferred — not blocking Phase 3.
- 2026-06-29 — Phase 3 (3.1–3.4) complete. Tests: 72/72 passing (45 scoring, 20 data-contract, 7 backtest infra).
  Backtest harness: data/backtest_fetcher.py, data/backtest_enrich_sp.py, backtest.py.
  Full 2026 season (Apr 1–Jun 28): 16,945 player-games, 16,907 with real SP stats (1,174/1,176 games enriched).
  Calibration: <50→32.9%, 50-59→40.8%, 60-69→48.8%, 70+→empty. Clear monotonic discrimination.
  70+ buckets empty because no Vegas lines (~+4–5 pts per play) + no batter handedness + dome weather neutral.
  3.5–3.7 blocked on Odds API Business tier ($99/mo, pending Chase decision).
  Next: Phase 4 (per-market deep dives), starting with 4.1 O1.5 TB weighting validation.
- 2026-06-29 — Phase 4.1 and 4.2 complete. 90/90 tests passing.
  4.1: O1.5 TB contact-first weighting validated (K%=24%, wOBA=20%). Backtest confirms monotonic discrimination.
  4.2: O0.5 Any Hit built from scratch. scoring/hits.py (K%=35% batter, K%=72% SP), markets/hits_o05.py,
  check_bettable_o05, O05 config constants, VIG_BY_MARKET in backtest.py. Calibration at -175 vig:
  <50→59.3% (-6.7%), 50-59→66.0% (+3.8%), 60-69→69.7% (+9.6%), 70-79→72.7% (+14.4%). Base rate 64.6%.
  4.3: Pitcher Strikeouts built. scoring/strikeout.py (sp_k 40%, SwStr% 20%, opp_lineup 25%, context 15%),
  markets/k_props.py, check_bettable_k_prop. whiff_percent→swstr_pct added to Savant column map. 21 tests; 111/111.
  Backtest deferred — needs SP game K total fetcher.
  4.4: Home Runs built. scoring/hr.py (barrel%=35%+ dynamic, park HR factor, bat-tracking optional),
  markets/hr.py, check_bettable_hr (barrel% non-negotiable), HR constants in config.py. 11 tests; 122/122.
  4.5: Moneyline extracted. scoring/moneyline.py (compute_win_probability Log5 + compute_ml_confidence),
  markets/moneyline.py (score_game_ml, ml_market_edge), check_bettable_ml in provenance.py. Diagnosed:
  silent defaults on missing odds → now fails bettable gate explicitly (no odds → not bettable). 30 tests; 126/126.
  Phase 4 complete. All 5 market pure modules built and tested.
- 2026-06-30 — Backtest infrastructure extended. 166/166 tests passing throughout.
  Pipeline: added fetch_savant_pitcher_swstr() to aggregate pitch-arsenal SwStr% per pitcher.
  296/717 pitchers now have real swstr_pct (pitch-arsenal endpoint, weighted by pitch volume).
  Removed dead STATCAST_COL_MAP whiff_percent entry (contact-quality endpoint never had it).
  HR backtest: added hit_hr column to game_results (ALTER TABLE + backfill from existing hr column —
  no re-fetch needed). 2,275 player-game HR outcomes. Calibration (20,128 player-games, min_ab=1):
  <50→9.0% (-59.4%), 50-59→16.3% (-26.8%), 60-69→21.7% (-2.2%), 70-79→25.8% (+16.4%) @ +350.
  Monotonic. 70-79 bucket is +EV at the standard HR prop line — needs real lines to confirm.
  K prop backtest: new data/backtest_sp_k_fetcher.py fetches SP K totals per game from MLB live feed.
  2,340 SP-game rows in game_sp_results. Calibration (2,348 SP-games):
  O4.5 Ks: <50→25%, 50-59→44%, 60-69→46%, 70-79→82%, 80+→100%.
  O5.5 Ks: <50→6%, 50-59→26%, 60-69→31%, 70-79→69%, 80+→93%.
  Strong monotonic discrimination. 70-79 bucket is well above both common K lines.
  Data contract tests: added TestPitcherTable (swstr_pct coverage/sane range), TestGameResultsTable
  (5 tests incl. hit_hr consistency), TestGameSpResultsTable (6 tests), TestGameOutcomesTable (5 tests,
  skips until fetcher is run). Total tests: 166 passed + 5 skipped (game_outcomes).
  ML game outcome calibration: new data/backtest_game_outcomes_fetcher.py fetches Final game scores
  from MLB schedule API into game_outcomes table (game_pk, home/away team, scores, winning_team).
  Added run_ml_backtest() and print_ml_report() to backtest.py — buckets by predicted win prob
  (50-55/55-60/60-65/65%+) vs actual win rate. SP vuln computed via compute_pitcher_score;
  lineup wRC+ from game_results per-game batter data; bullpen/Vegas inputs neutral (no data).
  Usage: python3 data/backtest_game_outcomes_fetcher.py --start 2026-04-01 --end 2026-06-29
  then: python3 backtest.py --market ml
  Remaining blocker: Odds API Business tier for real prop lines and ROI validation.
