"""app.py — Thin Streamlit entry point.

All display logic lives in ui/. All scoring logic lives in scoring/ + markets/.
This file contains only main() and imports.
"""

import pytz
import streamlit as st
from datetime import datetime

from ui.hits_tab        import display_hits_tab
from ui.hot_streaks_tab import display_hot_streaks_tab
from ui.hr_tab          import display_hr_plays
from ui.k_props_tab     import display_k_props_tab
from ui.leaderboard     import display_leaderboard
from ui.moneyline_tab   import display_moneyline_tab

from mlb_tb_analyzer import (
    display_dk_portfolio_builder,
    display_fd_command_center,
    display_fd_hand_builder,
    display_fd_portfolio_builder,
    display_results_tracker,
    fetch_moneyline_odds,
    fetch_odds,
    fetch_schedule,
    fetch_team_run_differential,
    fetch_umpire_data,
    init_db,
    run_model,
    save_picks_to_db,
)

EST = pytz.timezone("US/Eastern")


def main():
    # Initialize DB
    init_db()

    # Init session state
    if "plays" not in st.session_state:
        st.session_state.plays = []
    if "analysis_date" not in st.session_state:
        st.session_state.analysis_date = None
    if "model_ran" not in st.session_state:
        st.session_state.model_ran = False

    # ── SIDEBAR ───────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        today_est = datetime.now(EST).date()
        selected_date = st.date_input("📅 Select Date", value=today_est)
        date_str = selected_date.strftime("%Y-%m-%d")

        st.markdown("---")

        st.subheader("💰 Bankroll")
        unit_size = st.number_input("Unit Size ($)", min_value=1, max_value=1000, value=25)

        st.markdown("---")

        # V1.9: type="primary" renders grey on dark Streamlit themes. Use CSS override.
        st.markdown("""<style>
        div[data-testid="stSidebar"] div.stButton:first-of-type > button {
            background-color: #00cc66 !important; color: #000 !important;
            font-weight: 700 !important; border: none !important; font-size: 1rem !important;
        }
        div[data-testid="stSidebar"] div.stButton:first-of-type > button:hover {
            background-color: #00ff88 !important;
        }
        </style>""", unsafe_allow_html=True)
        run_btn = st.button("⚾ Run Today's Model", use_container_width=True)

        if st.button("🔄 Clear Cache + Rerun", use_container_width=True):
            st.cache_data.clear()
            st.session_state.plays = []
            st.rerun()

        st.markdown("---")

        # Data source status
        st.subheader("📡 Data Sources")

        try:
            odds_key = st.secrets.get("odds_api", {}).get("api_key", "")
            has_odds_key = bool(odds_key and odds_key.strip())
        except Exception:
            has_odds_key = False

        st.markdown(f"✅ MLB Stats API *(always accessible)*")
        bat_src   = st.session_state.get("_batting_source", "")
        pit_src   = st.session_state.get("_pitching_source", "")
        bat_cols  = st.session_state.get("batting_cols", [])
        has_xstats    = "xSLG" in bat_cols or "est_slg" in bat_cols
        has_statcast  = "Barrel%" in bat_cols or "barrel_batted_rate" in bat_cols
        has_bat_track = "bat_speed" in bat_cols
        if not bat_src:
            st.markdown("⏳ Baseball Savant *(run model to check)*")
        elif has_xstats and has_statcast and has_bat_track:
            st.markdown("✅ Baseball Savant *(full: xStats + Statcast + Bat Tracking)*")
        elif has_xstats and has_statcast:
            st.markdown("🟡 Baseball Savant *(partial: xStats + Statcast, no bat tracking)*")
        elif has_xstats:
            st.markdown("⚠️ Baseball Savant *(xStats only — Statcast leaderboard blocked [502])*")
        else:
            st.markdown("❌ Baseball Savant *(blocked — using MLB API proxies)*")
        st.markdown(f"✅ Open-Meteo Weather")

        if has_odds_key:
            st.markdown(f"✅ The Odds API *(live lines)*")
        else:
            st.markdown(f"⚠️ The Odds API *(no key — scores degraded)*")
            with st.expander("🔑 Add Odds API Key (required for best scores)"):
                st.markdown("""
**Step 1:** Sign up free at [the-odds-api.com](https://the-odds-api.com)
— 500 calls/month free (~$0 for daily use all season)

**Step 2:** Go to Streamlit Cloud → your app → **⋮ Settings → Secrets**

**Step 3:** Paste this (replace with your real key):
```toml
[odds_api]
api_key = "your_key_here"
```
**Step 4:** Save → app auto-restarts

**Usage math:** 1 call per model run × ~180 game days = ~180 calls/season.
Free tier (500/mo) is more than enough.
                """)

        st.markdown("---")

        with st.expander("📖 Model Info"):
            st.markdown("""
            **Scoring: 1B=1, 2B=2, 3B=3, HR=4**
            Walks, HBP, SB = 0 TB (never counted)

            **V1.8 Weights:**
            - ⚾ Pitcher Vuln: 30% (K%, HH%, FIP, barrel, WHIP — wider scale)
            - 🏏 Batter: 28% (xSLG, barrel, HH%, wRC+, ISO, K%)
            - 🤚 Platoon: 12% (LHB vs RHP = +56 SLG documented)
            - 💰 Vegas: 8% (implied total r=0.61 with scoring)
            - 🏟️ Park: 7% (Coors/Petco effects real)
            - 📈 Streak: 5% (last 7 games form)
            - 🔄 TTO: 4% (times through order bonus)
            - 🌤️ Weather: 4% (wind direction/speed)
            - 🎯 Pitch Mix: 2%
            - 📊 BvP: 2% (career vs SP)
            - 📋 Lineup: 1%

            **V1.8 Tiers:**
            - 🔒 Tier 1 (78+): Strong play, parlay anchor
            - ✅ Tier 2 (68-77): Viable, parlay filler
            - 📊 Tier 3 (58-67): Marginal, single only
            - ❌ Below 58: Fade
            """)

        with st.expander("🔍 Debug: Data Quality Check"):
            import pandas as pd
            st.caption("Expand after running model to verify all key stats loaded")

            # ── Raw gameLog API probe ──────────────────────────────────────
            st.markdown("**🔬 Recent Form API Probe** — verify raw game log data")
            probe_name = st.text_input("Enter player name to probe (e.g. 'Brice Turang'):", key="probe_name_input")
            if st.button("🔍 Probe API", key="probe_api_btn") and probe_name and st.session_state.plays:
                probe_p = next((p for p in st.session_state.plays
                                if probe_name.lower() in p.get("name","").lower()), None)
                if probe_p:
                    pid = str(probe_p.get("player_id",""))
                    import datetime as _dt
                    yr = _dt.datetime.now().year
                    url = (f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
                           f"?stats=gameLog&group=hitting&gameType=R&season={yr}&limit=12")
                    try:
                        import requests as _rq
                        resp = _rq.get(url, timeout=8)
                        st.caption(f"Player: {probe_p['name']} | ID: {pid} | URL: {url}")
                        st.caption(f"HTTP status: {resp.status_code}")
                        if resp.status_code == 200:
                            splits = resp.json().get("stats",[{}])[0].get("splits",[])
                            st.caption(f"Splits returned: {len(splits)}")
                            rows = []
                            for i, s in enumerate(splits[:10]):
                                st_ = s.get("stat",{})
                                rows.append({
                                    "game#": i,
                                    "date": s.get("date","?"),
                                    "AB": st_.get("atBats","?"),
                                    "H": st_.get("hits","?"),
                                    "2B": st_.get("doubles","?"),
                                    "3B": st_.get("triples","?"),
                                    "HR": st_.get("homeRuns","?"),
                                    "TB(api)": st_.get("totalBases","?"),
                                    "season": s.get("season","?"),
                                })
                            if rows:
                                st.dataframe(pd.DataFrame(rows), hide_index=True)
                    except Exception as e:
                        st.error(f"Probe failed: {e}")
                else:
                    st.warning(f"Player '{probe_name}' not found in current model run")

            st.markdown("---")
            bat_src  = st.session_state.get("_batting_source", "not_run")
            pit_src  = st.session_state.get("_pitching_source", "not_run")
            src_icon = {
                # Live fetch sources
                "fangraphs_live": "✅", "fangraphs+pybaseball": "✅",
                "savant+mlbapi": "✅", "mlbapi+savant": "✅",
                "savant+mlbapi+fangraphs": "✅",
                "mlbapi": "✅", "mlbapi_only": "⚠️",
                "savant_xstats": "✅", "savant_statcast": "✅",
                "matched": "✅",
                # Disk cache
                "disk_cache": "💾", "disk_cache_fresh": "💾",
                "disk_cache_stale": "⚠️",
                # Failure states
                "failed": "❌", "not_run": "⏳", "unknown": "❓",
            }
            st.markdown(f"**Batting stats source:** {src_icon.get(bat_src,'❓')} `{bat_src}`")
            st.markdown(f"**Pitching stats source:** {src_icon.get(pit_src,'❓')} `{pit_src}`")

            # Show FanGraphs errors if any
            fg_bat_errs = st.session_state.get("_fg_batting_errors", [])
            fg_pit_errs = st.session_state.get("_fg_pitching_errors", [])
            if fg_bat_errs:
                st.error("**FanGraphs batting fetch errors:**\n" + "\n".join(fg_bat_errs))
            if fg_pit_errs:
                st.error("**FanGraphs pitching fetch errors:**\n" + "\n".join(fg_pit_errs))

            # Arsenal merge error surfacing
            _am_err = st.session_state.get("_arsenal_merge_err")
            _ba_err = st.session_state.get("_batter_arsenal_err")
            if _am_err:
                st.error(f"⚠️ Pitcher arsenal merge error (non-fatal): {_am_err}")
            if _ba_err:
                st.error(f"⚠️ Batter arsenal merge error (non-fatal): {_ba_err}")

            # Disk cache status
            import os as _os, time as _time
            _cache_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "stat_cache")
            for _cname in ("batting_stats", "pitching_stats"):
                _p = _os.path.join(_cache_dir, f"{_cname}.pkl")
                if _os.path.exists(_p):
                    _age = (_time.time() - _os.path.getmtime(_p)) / 3600
                    _age_label = f"{_age:.1f}h old"
                    _freshness = "✅ fresh" if _age < 6 else ("⚠️ stale" if _age > 120 else "💾 cached")
                    st.caption(f"💾 Disk cache `{_cname}`: {_age_label} — {_freshness}")
                else:
                    st.caption(f"⬜ No disk cache for `{_cname}` — run `seed_stat_cache.py` locally and commit `stat_cache/`")

            st.markdown("---")
            matched = st.session_state.get("_matched", 0)
            unmatched = st.session_state.get("_unmatched", 0)
            total = matched + unmatched
            if total > 0:
                rate = matched / total * 100
                color = "✅" if rate > 80 else "⚠️" if rate > 50 else "❌"
                st.markdown(f"**{color} Player match rate: {matched}/{total} ({rate:.0f}%)**")
                if rate < 80:
                    st.warning("Low match rate — scores using league averages for unmatched players")

            # Show raw name samples — use .get() everywhere to avoid AttributeError
            # before model has been run (session_state keys may not exist yet)
            lookup_diag = st.session_state.get("lookup_diag")
            batting_df_sample = st.session_state.get("batting_df_sample", [])
            norm_name_sample  = st.session_state.get("norm_name_sample", [])
            search_sample     = st.session_state.get("search_sample", [])
            batting_cols      = st.session_state.get("batting_cols", [])
            pitching_cols     = st.session_state.get("pitching_cols", [])
            sample_player     = st.session_state.get("sample_player")

            if lookup_diag:
                st.markdown("**🔬 Live lookup diagnostic (first batter):**")
                st.json(lookup_diag)
                if batting_df_sample:
                    st.markdown("**Raw _name values in batting DataFrame (first 10):**")
                    st.code("\n".join(str(x) for x in batting_df_sample))
            if norm_name_sample:
                st.markdown("**Normalized _norm_name values (first 10):**")
                st.code("\n".join(str(x) for x in norm_name_sample))
            if search_sample:
                st.markdown("**Names we searched for (first 5):**")
                st.code("\n".join(str(x) for x in search_sample))
            if batting_cols:
                critical_bat = ["xSLG", "xwOBA", "SLG", "ISO", "K%", "BB%", "OBP",
                                "wRC+", "Barrel%", "Hard%", "EV"]
                # What provides each stat (real source vs proxy)
                _source_map = {
                    "xSLG":    ("Savant xStats ✅", None),
                    "xwOBA":   ("Savant xStats ✅", None),
                    "SLG":     ("MLB Stats API ✅", None),
                    "ISO":     ("MLB Stats API ✅", None),
                    "K%":      ("MLB Stats API ✅", None),
                    "BB%":     ("MLB Stats API ✅", None),
                    "OBP":     ("MLB Stats API ✅", None),
                    "wRC+":    ("FanGraphs (blocked ❌)", "OBP proxy ~±25pts"),
                    "Barrel%": ("Savant Statcast (blocked ❌)", "HR/PA proxy ~±5-8%"),
                    "Hard%":   ("Savant Statcast (blocked ❌)", "SLG+K% proxy ~±6-10%"),
                    "EV":      ("Savant Statcast (blocked ❌)", "xSLG+ISO derived"),
                }
                st.markdown("**📊 Batting data — real vs proxy:**")
                real_count = 0
                proxy_count = 0
                for c in critical_bat:
                    src, proxy_note = _source_map.get(c, ("Unknown", None))
                    if c in batting_cols:
                        st.markdown(f"✅ **`{c}`** — {src.split(' ✅')[0]}")
                        real_count += 1
                    elif proxy_note:
                        st.markdown(f"🔄 **`{c}`** — {proxy_note} *(real source: {src.split(' (')[0]})*")
                        proxy_count += 1
                    else:
                        st.markdown(f"❌ **`{c}`** — missing, league avg used")

                completeness = real_count / len(critical_bat) * 100
                color = "🟢" if completeness >= 80 else "🟡" if completeness >= 50 else "🔴"
                st.caption(f"Total columns: {len(batting_cols)} | "
                           f"{color} Data completeness: {completeness:.0f}% real | "
                           f"{proxy_count} proxies active")
                if proxy_count > 0:
                    st.warning(f"⚠️ {proxy_count} stats are derived proxies, not real Statcast data. "
                               "Scores are adjusted with +13pt offset to compensate.")
                    with st.expander("🔧 Fix: Generate local cache seeder script"):
                        st.markdown("""
**Why this happens:** Streamlit Cloud IPs are blocked by Baseball Savant's
Statcast leaderboard endpoint. The real fix is seeding the cache from your
local machine daily.

**Steps:**
1. Download the seeder script below
2. Run it locally: `python3 seed_stat_cache.py`
3. Commit the generated `stat_cache/` folder to your GitHub repo
4. Streamlit Cloud will use the cached data automatically

The cache is valid for 6 hours — run the script before posting daily picks.
                        """)
                        seed_script = '#!/usr/bin/env python3\n# Propex stat cache seeder -- run locally daily, commit stat_cache/ to GitHub\n'
                        seed_script += '"""Fetches real Savant Statcast data from your local IP where endpoints are unblocked."""\n'
                        seed_script += """
import requests, pandas as pd, pickle, os, time

CACHE_DIR = os.path.join(os.path.dirname(__file__), "stat_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def fetch(url, label):
    print(f"Fetching {label}...")
    r = requests.get(url, timeout=30, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
        "Referer": "https://baseballsavant.mlb.com/",
    })
    if r.status_code == 200 and len(r.content) > 1000:
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        print(f"  ✅ {len(df)} rows, columns: {list(df.columns[:6])}")
        return df
    print(f"  ❌ HTTP {r.status_code}")
    return pd.DataFrame()

yr = 2026

# Fetch all Savant endpoints
xstats = fetch(f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=batter&year={yr}&min=1&csv=true", "xStats")
statcast = fetch(f"https://baseballsavant.mlb.com/leaderboard/statcast?year={yr}&min=1&type=batter&csv=true", "Statcast")
bat_track = fetch(f"https://baseballsavant.mlb.com/leaderboard/bat-tracking?year={yr}&minSwings=50&type=batter&csv=true", "Bat Tracking")

# Merge into one batting frame
result = xstats.copy() if not xstats.empty else pd.DataFrame()
if not statcast.empty and not result.empty:
    # normalize player_id
    for df in [result, statcast, bat_track]:
        if "player_id" in df.columns:
            df["mlbam_id"] = df["player_id"].astype(str)
    result = result.merge(statcast[["mlbam_id","barrel_batted_rate","hard_hit_percent","avg_exit_velocity"]].dropna(subset=["mlbam_id"]), on="mlbam_id", how="left")
    if not bat_track.empty:
        result = result.merge(bat_track[["mlbam_id","bat_speed","blast_rate"]].dropna(subset=["mlbam_id"]), on="mlbam_id", how="left")

if not result.empty:
    path = os.path.join(CACHE_DIR, "batting_stats.pkl")
    with open(path, "wb") as f:
        pickle.dump({"df": result, "ts": time.time()}, f)
    print(f"✅ Batting cache saved: {len(result)} players, {len(result.columns)} columns")
    print(f"   Columns: {list(result.columns)}")
else:
    print("❌ Could not build batting cache")
"""
                        st.download_button(
                            "⬇️ Download cache seeder script",
                            seed_script,
                            "seed_stat_cache.py",
                            "text/plain",
                            key="dl_seed_script"
                        )
            else:
                st.info("Run the model to see batting column status.")
            if pitching_cols:
                critical_pit = ["K%", "ERA", "FIP", "xFIP", "Hard%", "Barrel%",
                                "SO", "TBF", "xERA", "G", "GS", "Team", "WHIP"]
                st.markdown("**Pitching — critical columns:**")
                for c in critical_pit:
                    st.markdown(f"{'✅' if c in pitching_cols else '❌'} `{c}`")
                st.caption(f"Total pitching columns: {len(pitching_cols)}")
            else:
                st.info("Run the model to see pitching column status.")
            if sample_player:
                st.markdown("**Sample player (Judge):**")
                st.json(sample_player)

        # ── V1.3 NEW: Bullpen Quality Debug ──────────────────────────────
        with st.expander("🔬 Debug: V1.3 Bullpen Quality Scores"):
            st.caption("Per-team bullpen vulnerability (0=unhittable, 100=mop-up arms). "
                       "High score = weak bullpen = good for batters. "
                       "Was fixed at 42.0 for all teams in V1.2.")
            bp_scores = st.session_state.get("team_bullpen_scores", {})
            if bp_scores:
                bp_df = pd.DataFrame([
                    {"Team": t, "Bullpen Vuln Score": v,
                     "Quality": "🔒 Elite" if v < 35 else "✅ Good" if v < 45 else "⚠️ Average" if v < 55 else "💀 Weak"}
                    for t, v in sorted(bp_scores.items(), key=lambda x: x[1])
                ])

                def color_bp(val):
                    try:
                        v = float(val)
                        if v < 35:  return "color: #00ff88; font-weight: bold"
                        elif v < 45: return "color: #66ddff"
                        elif v < 55: return "color: #ffdd00"
                        return "color: #ff4444; font-weight: bold"
                    except: return ""

                styled_bp = bp_df.style.map(color_bp, subset=["Bullpen Vuln Score"])
                st.dataframe(styled_bp, use_container_width=True)
                st.caption(f"✅ {len(bp_scores)} teams scored | League avg baseline: 42.0")

                # Show score impact on a sample batter
                st.markdown("**Score impact example (avg batter, avg SP, score = 55):**")
                example_rows = []
                for label, bp_v in [("Best bullpen (score ~28)", 28), ("League avg (42)", 42), ("Worst bullpen (score ~62)", 62)]:
                    sp_score_ex = 45.0  # avg SP
                    blended_ex  = sp_score_ex * 0.60 + bp_v * 0.40
                    example_rows.append({"Scenario": label, "SP Score": 45, "BP Vuln": bp_v,
                                         "Blended Pit Score": round(blended_ex, 1)})
                st.table(pd.DataFrame(example_rows))
            else:
                st.warning("Bullpen scores not computed — pitching_df may be missing GS or Team columns. "
                           "All teams defaulting to league average (42.0). "
                           "Run model and check pitching critical columns above.")

        # ── V1.3 NEW: Per-player score breakdown debug ───────────────────
        with st.expander("🔬 Debug: Score Component Breakdown (top 10 plays)"):
            st.caption("Shows exactly how each sub-score contributed to the final score. "
                       "BP Vuln = the opponent's bullpen score used for that batter.")
            plays_debug = st.session_state.get("plays", [])
            if plays_debug:
                debug_rows = []
                for p in plays_debug[:10]:
                    debug_rows.append({
                        "Player":     p["name"],
                        "Team":       p["team"],
                        "Opp":        p.get("opponent", "?"),
                        "Final":      p["score"],
                        "Batter":     p.get("sub_batter", "—"),
                        "Pitcher":    p.get("sub_pitcher", "—"),
                        "Matchup":    p.get("sub_matchup", "—"),
                        "BP Vuln":    p.get("bullpen_vuln", "—"),
                        "Platoon":    p.get("sub_platoon", "—"),
                        "Park":       p.get("sub_park", "—"),
                        "Weather":    p.get("sub_weather", "—"),
                        "Vegas":      p.get("sub_vegas", "—"),
                        "Lineup":     p.get("sub_lineup", "—"),
                    })
                st.dataframe(pd.DataFrame(debug_rows), use_container_width=True)
            else:
                st.info("Run the model to see per-player score breakdowns.")

        st.caption(f"v1.6 | {datetime.now(EST).strftime('%I:%M %p EST')}")

    # ── MAIN CONTENT ──────────────────────────────────────
    st.title("⚾ MLB TB Analyzer V2.1")
    import hashlib as _hlib
    _sig = _hlib.md5(b"v21_contact_model_dk_slate_aware").hexdigest()[:6]
    st.caption(f"🔑 Build sig: {_sig} — V2.1: Contact-first TB model | DK pipeline | Slate-aware stacks | 2026 game logs")
    st.caption("Fully automated | HardRock Bet | 1B=1 2B=2 3B=3 HR=4 | FD + DK portfolios | Team stack scores")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
        "📊 O1.5 Leaderboard",
        "🎯 O0.5 Any Hit",
        "⚡ K Props",
        "🏦 Moneyline",
        "🔥 Hot Streaks",
        "💣 HR Plays",
        "🎯 Command Center",
        "🔬 Hand Builder",
        "🚀 FD Portfolio",
        "⚡ DK Portfolio",
        "📈 Results Tracker",
    ])

    # ── RUN MODEL ────────────────────────────────────────
    if run_btn:
        with tab1:
            st.markdown(f"**📅 Running model for {date_str}...**")
            status = st.container()
            plays = run_model(date_str, status)
            st.session_state.plays = plays
            st.session_state.analysis_date = date_str
            st.session_state.model_ran = True
            if plays:
                save_picks_to_db(plays, date_str)
                # ── V1.9: Pre-fetch ump data and run differential for new tabs ──
                try:
                    st.session_state["_ump_data"] = fetch_umpire_data()
                except Exception:
                    st.session_state["_ump_data"] = {}
                try:
                    st.session_state["_run_diffs"] = fetch_team_run_differential(date_str, days=7)
                except Exception:
                    st.session_state["_run_diffs"] = {}
                try:
                    st.session_state["_ml_odds"] = fetch_moneyline_odds(date_str)
                except Exception:
                    st.session_state["_ml_odds"] = {}
                st.rerun()
            else:
                st.warning(f"No games or lineups found for {date_str}.")
                st.info("Opening Day is March 27, 2026. Change the date in the sidebar.")

    # ── DISPLAY TABS ─────────────────────────────────────
    with tab1:
        if st.session_state.plays:
            date_label = st.session_state.analysis_date or date_str
            games = fetch_schedule(date_label)
            if games:
                game_labels = " • ".join([f"{g['away_team']}@{g['home_team']}" for g in games])
                st.caption(f"📅 {date_label} | {len(games)} games: {game_labels}")
            display_leaderboard(st.session_state.plays)
        else:
            st.info("👈 Click **⚾ Run Today's Model** to fetch today's plays")

    with tab2:
        if st.session_state.plays:
            display_hits_tab(st.session_state.plays)
        else:
            st.info("Run the model first to see O0.5 any-hit plays.")

    with tab3:
        # ── K Props ──────────────────────────────────────
        if st.session_state.plays:
            ump_data = st.session_state.get("_ump_data", {})
            display_k_props_tab(st.session_state.plays, ump_data)
        else:
            st.info("Run the model first to see K prop scores.")

    with tab4:
        # ── Moneyline ────────────────────────────────────
        if st.session_state.plays:
            _games_for_ml = fetch_schedule(st.session_state.analysis_date or date_str)
            _ml_odds      = st.session_state.get("_ml_odds", {})
            _run_diffs    = st.session_state.get("_run_diffs", {})
            _impl_totals  = {}
            try:
                _impl_totals = fetch_odds(st.session_state.analysis_date or date_str)
            except Exception:
                pass
            _bp_scores    = st.session_state.get("team_bullpen_scores", {})
            display_moneyline_tab(
                games=_games_for_ml,
                plays=st.session_state.plays,
                ml_odds=_ml_odds,
                run_diffs=_run_diffs,
                implied_totals=_impl_totals,
                team_bullpen_scores=_bp_scores,
            )
        else:
            st.info("Run the model first to see moneyline analysis.")

    with tab5:
        # ── Hot Streaks ───────────────────────────────────
        if st.session_state.plays:
            display_hot_streaks_tab(st.session_state.plays)
        else:
            st.info("Run the model first to see hot streak rankings.")

    with tab6:
        if st.session_state.plays:
            display_hr_plays(st.session_state.plays)
        else:
            st.info("Run the model first to see HR plays.")

    with tab7:
        display_fd_command_center(st.session_state.plays if st.session_state.plays else [])

    with tab8:
        display_fd_hand_builder(st.session_state.plays if st.session_state.plays else [])

    with tab9:
        display_fd_portfolio_builder(st.session_state.plays if st.session_state.plays else [])

    with tab10:
        display_dk_portfolio_builder(st.session_state.plays if st.session_state.plays else [])

    with tab11:
        display_results_tracker()


if __name__ == "__main__":
    main()
