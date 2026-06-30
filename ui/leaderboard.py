"""ui/leaderboard.py — O1.5 Total Bases leaderboard tab.

Thin Streamlit renderer. All scoring is pre-computed in plays dicts by run_model().
"""

import pandas as pd
import pytz
import streamlit as st
from datetime import datetime
from typing import Dict, List

EST = pytz.timezone("US/Eastern")


def display_leaderboard(plays: List[Dict]):
    """Display the full ranked leaderboard with tier color coding."""

    if not plays:
        st.info("No scored plays available. Run the model first.")
        return

    # ── Global slate DQ banner ────────────────────────────────────────────
    avg_dq      = sum(p.get("dq_score", 0) for p in plays) / max(len(plays), 1)
    n_bettable  = sum(1 for p in plays if p.get("bettable", False))
    n_total     = len(plays)
    if avg_dq >= 80:
        st.success(f"✅ **Slate data quality: {avg_dq:.0f}/100** — "
                   f"{n_bettable}/{n_total} props bettable (all required inputs measured)")
    elif avg_dq >= 50:
        st.warning(f"⚠️ **Slate data quality: {avg_dq:.0f}/100** — "
                   f"{n_bettable}/{n_total} props bettable — some inputs are proxies or missing")
    else:
        st.error(f"❌ **Slate data quality: {avg_dq:.0f}/100** — "
                 f"{n_bettable}/{n_total} props bettable — most scoring inputs are estimates. "
                 "DO NOT BET until data quality improves.")

    # Dataset freshness warning (1.7)
    _fresh_warn = st.session_state.get("_db_freshness_warning")
    if _fresh_warn:
        st.warning(f"⏰ **Stale dataset:** {_fresh_warn}")

    # Proxy mode indicator — driven by column presence, not source label.
    # "savant+mlbapi" contains "mlbapi" as a substring; string matching gives
    # false positives. Use the columns that are actually in the loaded data.
    _bat_cols_ui = st.session_state.get("batting_cols", [])
    _is_proxy_ui = not (
        ("barrel_batted_rate" in _bat_cols_ui or "Barrel%" in _bat_cols_ui)
        and ("hard_hit_percent" in _bat_cols_ui or "Hard%" in _bat_cols_ui)
        and ("wRC+" in _bat_cols_ui)
    )
    # Source label is still used for the cache-provenance banner below — a
    # separate concern from proxy detection (which uses column presence above).
    _bat_src_ui = st.session_state.get("_batting_source", "")
    if _is_proxy_ui:
        st.warning("⚠️ **Proxy Data Mode** — Savant unavailable. Using MLB Stats API derived signals. "
                   "Tier thresholds adjusted −5 pts: Tier 1 ≥75 · Tier 2 ≥65 · Tier 3 ≥55. "
                   "Scores run ~5-8 pts lower than full-Savant mode.")
    elif "disk_cache_fresh" in _bat_src_ui:
        st.info("💾 Serving from fresh disk cache — full Savant column set, normal thresholds.")

    # BvP "owns" alert — surface elite matchups prominently regardless of tier
    bvp_owns = [p for p in plays if p.get("bvp_sig") == "owns"]
    if bvp_owns:
        owns_names = " · ".join(
            f"{p['name']} ({p['team']}) {p.get('bvp_label','')}" for p in bvp_owns[:3]
        )
        st.error(f"🔥 **OWNS MATCHUP** — elite career dominance vs today's SP: {owns_names}")

    bvp_fades = [p for p in plays if p.get("bvp_sig") in ("fade", "dominated")]
    if bvp_fades:
        fade_names = " · ".join(f"{p['name']} ({p['team']})" for p in bvp_fades[:3])
        st.warning(f"⚠️ **BvP Fades** — career struggles vs today's SP: {fade_names}")

    # Summary metrics
    tier1 = [p for p in plays if p["tier"] == "🔒 TIER 1"]
    tier2 = [p for p in plays if p["tier"] == "✅ TIER 2"]
    tier3 = [p for p in plays if p["tier"] == "📊 TIER 3"]
    no_play = [p for p in plays if p["tier"] == "❌ NO PLAY"]

    # Tier summary banner
    if tier1:
        st.success(f"🔒 {len(tier1)} TIER 1 PLAYS — Parlay anchors. Don't sleep on these.")
    elif tier2:
        st.info(f"✅ {len(tier2)} TIER 2 PLAYS — Solid value, build parlays around these.")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("🔒 Tier 1", len(tier1))
    with col2: st.metric("✅ Tier 2", len(tier2))
    with col3: st.metric("📊 Tier 3", len(tier3))
    with col4: st.metric("❌ No Play", len(no_play))
    with col5: st.metric("Total Batters", len(plays))

    st.markdown("---")

    hide_nb = st.sidebar.toggle("Hide non-bettable", value=False)

    # Filters
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        tier_filter = st.multiselect("Filter by Tier",
            ["🔒 TIER 1", "✅ TIER 2", "📊 TIER 3", "❌ NO PLAY"],
            default=["🔒 TIER 1", "✅ TIER 2", "📊 TIER 3"])
    with col_f2:
        teams = sorted(list(set(p["team"] for p in plays)))
        team_filter = st.multiselect("Filter by Team", teams, default=[])
    with col_f3:
        min_score_filter = st.slider("Min Score", 0, 100, 50)
    with col_f4:
        hand_filter = st.multiselect("Batter Hand", ["L", "R", "B"], default=[])

    # Apply filters
    filtered = plays
    if tier_filter:
        filtered = [p for p in filtered if p["tier"] in tier_filter]
    if team_filter:
        filtered = [p for p in filtered if p["team"] in team_filter]
    if min_score_filter > 0:
        filtered = [p for p in filtered if p["score"] >= min_score_filter]
    if hand_filter:
        filtered = [p for p in filtered if p["batter_hand"] in hand_filter]
    if hide_nb:
        filtered = [p for p in filtered if p.get("bettable", True)]

    st.markdown(f"**Showing {len(filtered)} batters**")

    # Build display dataframe
    rows = []
    for p in filtered:
        wind_icon = ""
        if p.get("wind_effect") == "strong_out":
            wind_icon = "💨⬆️"
        elif p.get("wind_effect") == "out":
            wind_icon = "💨"
        elif p.get("wind_effect") == "in":
            wind_icon = "💨⬇️"
        elif p.get("is_dome"):
            wind_icon = "🏟️"

        tbd_flag = " ⚠️TBD" if p.get("sp_tbd") else ""

        if not p.get("bettable", True):
            top_reason = p.get("non_bettable_reasons", ["unknown"])
            tier_display = f"🔘 NON-BETTABLE ({top_reason[0]})"
        else:
            tier_display = p["tier"]

        rows.append({
            "Score": f"{p['score']:.0f}",
            "Tier": tier_display,
            "Player": p["name"],
            "Team": p["team"],
            "Vs": p["opponent"],
            "Slot": f"#{p['lineup_slot']}",
            "Hand": p["batter_hand"],
            "Opp SP": p["sp_name"][:20] + tbd_flag,
            "SP 🤚": p["sp_hand"],
            "Prob": f"{p['prob']*100:.0f}%",
            "Edge": f"{p.get('market_edge', 0):+.0f}%" if p.get('implied_total', 0) > 0 else "—",
            "xSLG": f"{p['xslg']:.3f}" if p["xslg"] else "—",
            "Barrel%": f"{p['barrel_rate']*100:.1f}%" if p["barrel_rate"] else "—",
            "HH%": f"{p['hard_hit_rate']*100:.1f}%" if p["hard_hit_rate"] else "—",
            "K%": f"{p['k_rate']*100:.1f}%" if p["k_rate"] else "—",
            "Platoon": p["platoon_label"].split("(")[0].strip(),
            "Form 🔥": p.get("streak_label", "—").replace("Form: ", ""),
            "BvP★": "🔥 OWNS" if p.get("bvp_sig") == "owns"
                    else "🟢 Edge" if p.get("bvp_sig") == "edge"
                    else "🔴 Fade" if p.get("bvp_sig") == "fade"
                    else "⚠️ Dom'd" if p.get("bvp_sig") == "dominated"
                    else "—",
            "BvP Stats": p.get("bvp_label", "—").replace("BvP: ", ""),
            "Park": p["park"],
            f"Wind{wind_icon}": p["weather_label"].split("|")[0].strip() if "|" in p["weather_label"] else p["weather_label"],
            "°F": f"{p['temperature']:.0f}°",
            "Imp.Runs": f"{p['implied_total']:.1f}" if p.get('implied_total', 0) > 0 else "—",
            "HR Score": f"{p['hr_score']:.0f}",
            "DQ": f"{p.get('dq_score', 0)}%",
        })

    if rows:
        df = pd.DataFrame(rows)

        def color_tier(val):
            if "TIER 1" in str(val) or "🔒" in str(val):
                return "color: #00ff88; font-weight: bold"
            elif "TIER 2" in str(val) or "✅" in str(val):
                return "color: #ffdd00; font-weight: bold"
            elif "TIER 3" in str(val) or "📊" in str(val):
                return "color: #ff8800; font-weight: bold"
            return "color: #888888"

        def color_score(val):
            try:
                v = float(str(val))
                if v >= 80: return "color: #00ff88; font-weight: bold"
                elif v >= 70: return "color: #ffdd00; font-weight: bold"
                elif v >= 60: return "color: #ff8800"
                return "color: #888888"
            except Exception:
                return ""

        def color_edge(val):
            try:
                v = float(str(val).replace("%", "").replace("+", ""))
                if v >= 10: return "color: #00ff88; font-weight: bold"
                elif v >= 5: return "color: #66dd88; font-weight: bold"
                elif v >= 0: return "color: #ffdd00"
                return "color: #ff4444"
            except Exception:
                return ""

        def color_form(val):
            v = str(val)
            if "🔥" in v or "Hot" in v:  return "color: #ff8800; font-weight: bold"
            if "❄️" in v or "Cold" in v: return "color: #88aaff"
            return ""

        def color_bvp(val):
            v = str(val)
            if "🔥" in v or "OWNS" in v: return "color: #ff6600; font-weight: bold; font-size: 1.05em"
            if "🟢" in v or "Edge" in v: return "color: #00ff88; font-weight: bold"
            if "🔴" in v or "Fade" in v: return "color: #ff4444; font-weight: bold"
            if "⚠️" in v or "Dom" in v:  return "color: #ffaa00; font-weight: bold"
            return ""

        def color_dq(val):
            try:
                v = int(str(val).replace("%", ""))
                if v >= 80: return "color: #00ff88; font-weight: bold"
                elif v >= 55: return "color: #ffdd00"
                return "color: #ff4444"
            except Exception:
                return ""

        def color_tier_nb(val):
            if "NON-BETTABLE" in str(val):
                return "color: #555555; font-style: italic"
            return color_tier(val)

        styled = (df.style
                  .map(color_tier_nb, subset=["Tier"])
                  .map(color_score, subset=["Score"]))
        if "Edge" in df.columns:
            styled = styled.map(color_edge, subset=["Edge"])
        if "Form 🔥" in df.columns:
            styled = styled.map(color_form, subset=["Form 🔥"])
        if "BvP Stats" in df.columns:
            styled = styled.map(color_bvp, subset=["BvP Stats"])
        if "BvP★" in df.columns:
            styled = styled.map(color_bvp, subset=["BvP★"])
        if "DQ" in df.columns:
            styled = styled.map(color_dq, subset=["DQ"])
        st.dataframe(styled, use_container_width=True, height=500)

        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Export CSV", csv,
            f"mlb_tb_picks_{datetime.now(EST).strftime('%Y%m%d')}.csv",
            "text/csv",
        )

    # ── Parlay Targets ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔥 Parlay Targets — Hot Streak + Model Convergence")
    st.caption(
        "Players who score well on BOTH the O1.5 TB model AND the hot streak signal. "
        "Highest-conviction parlay legs — model confidence backed by real recent form."
    )

    parlay_candidates = [
        p for p in (filtered if filtered else plays)
        if p.get("score", 0) >= 60
        and p.get("sub_streak", 0) >= 55
        and p.get("recent_tb_per_game") is not None
        and p.get("recent_games", 0) >= 3
        and p.get("bettable", False)
    ]
    parlay_candidates.sort(
        key=lambda x: x.get("score", 0) * 0.60 + x.get("sub_streak", 0) * 0.40,
        reverse=True,
    )
    top_parlay = parlay_candidates[:4]

    if top_parlay:
        card_cols = st.columns(min(4, len(top_parlay)))
        for col, p in zip(card_cols, top_parlay):
            model_sc  = p.get("score", 0)
            streak_sc = p.get("sub_streak", 0)
            tb_pg     = p.get("recent_tb_per_game", 0) or 0
            prob      = p.get("prob", 0.5)
            composite = model_sc * 0.60 + streak_sc * 0.40

            if composite >= 72:
                card_color = "#00ff88"; label = "🔥🔥 ELITE"
            elif composite >= 64:
                card_color = "#ff8800"; label = "🔥 STRONG"
            else:
                card_color = "#ffdd00"; label = "📈 SOLID"

            model_color  = "#00ff88" if model_sc >= 70 else "#ffdd00" if model_sc >= 60 else "#ff8800"
            streak_color = "#ff4444" if streak_sc >= 70 else "#ff8800" if streak_sc >= 60 else "#ffdd00"

            with col:
                st.markdown(
                    f"<div style='background:#1a1a2e;border:2px solid {card_color};"
                    f"border-radius:12px;padding:14px 12px;text-align:center;margin-bottom:6px'>"
                    f"<div style='font-size:0.65rem;color:{card_color};font-weight:700;"
                    f"letter-spacing:0.1em;margin-bottom:4px'>{label}</div>"
                    f"<div style='font-size:1.05rem;font-weight:800;color:#e0e0ff;margin-bottom:2px'>"
                    f"{p['name']}</div>"
                    f"<div style='font-size:0.75rem;color:#9090a8;margin-bottom:10px'>"
                    f"{p['team']} · #{p.get('lineup_slot','?')} · vs {p.get('sp_name','TBD')[:14]}</div>"
                    f"<div style='display:flex;justify-content:space-around;margin-bottom:8px'>"
                    f"<div><div style='font-size:1.4rem;font-weight:900;color:{model_color}'>{model_sc:.0f}</div>"
                    f"<div style='font-size:0.6rem;color:#888'>MODEL</div></div>"
                    f"<div style='color:#444;font-size:1.2rem;padding-top:4px'>|</div>"
                    f"<div><div style='font-size:1.4rem;font-weight:900;color:{streak_color}'>{streak_sc:.0f}</div>"
                    f"<div style='font-size:0.6rem;color:#888'>STREAK</div></div>"
                    f"</div>"
                    f"<div style='font-size:0.75rem;color:#b0b0c8'>"
                    f"{tb_pg:.2f} TB/g last 7 · {prob*100:.0f}% prob</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("")
        if len(top_parlay) >= 2:
            leg2 = top_parlay[:2]
            leg3 = top_parlay[:3]
            p2_str = " + ".join(f"{p['name']} O1.5 TB" for p in leg2)
            p3_str = " + ".join(f"{p['name']} O1.5 TB" for p in leg3)
            pcol2, pcol3 = st.columns(2)
            with pcol2:
                st.markdown(
                    f"<div style='background:#0d1a0d;border:1px solid #00cc66;"
                    f"border-radius:8px;padding:10px 12px'>"
                    f"<div style='color:#00cc66;font-size:0.7rem;font-weight:700;"
                    f"letter-spacing:0.1em;margin-bottom:4px'>2-LEG PARLAY</div>"
                    f"<div style='color:#e0e0e0;font-size:0.85rem;font-weight:600'>{p2_str}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with pcol3:
                if len(top_parlay) >= 3:
                    st.markdown(
                        f"<div style='background:#0d1a0d;border:1px solid #00cc66;"
                        f"border-radius:8px;padding:10px 12px'>"
                        f"<div style='color:#00cc66;font-size:0.7rem;font-weight:700;"
                        f"letter-spacing:0.1em;margin-bottom:4px'>3-LEG PARLAY</div>"
                        f"<div style='color:#e0e0e0;font-size:0.85rem;font-weight:600'>{p3_str}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("Need 3+ qualifying plays for a 3-leg parlay.")
    else:
        st.info(
            "No plays currently qualify (need Score ≥ 60 + Streak Score ≥ 55 + 3 recent games). "
            "Run the model once lineups post — streak data requires confirmed lineup slots."
        )

    # ── Top Plays — full breakdown ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🏆 Top Plays — Full Breakdown")

    top5 = [p for p in filtered if p["score"] >= 60][:5]
    for i, p in enumerate(top5, 1):
        with st.expander(
            f"{p['tier']} #{i}: {p['name']} ({p['team']}) — Score: {p['score']:.0f} | Prob: {p['prob']*100:.0f}%"
        ):
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.markdown("**🏏 Batter Profile**")
                st.write(f"• xSLG: {p['xslg']:.3f}" if p["xslg"] else "• xSLG: Limited data")
                st.write(f"• Barrel%: {p['barrel_rate']*100:.1f}%" if p["barrel_rate"] else "• Barrel%: —")
                st.write(f"• HardHit%: {p['hard_hit_rate']*100:.1f}%" if p["hard_hit_rate"] else "• HardHit%: —")
                st.write(f"• K%: {p['k_rate']*100:.1f}%" if p["k_rate"] else "• K%: —")
                st.write(f"• Exit Velo: {p['exit_velocity']:.1f} mph" if p["exit_velocity"] else "• Exit Velo: —")
                ev50 = p.get("ev50", 0)
                st.write(f"• EV50: {ev50:.1f} mph" if ev50 and ev50 > 50 else "• EV50: —")
                bs = p.get("bat_speed", 0)
                st.write(f"• Bat Speed: {bs:.1f} mph" if bs and bs > 30 else "• Bat Speed: —")
                streak_lbl = p.get("streak_label", "")
                if streak_lbl and streak_lbl not in ("Form: no data", "Form: no baseline"):
                    st.write(f"• {streak_lbl}")
                bvp_lbl = p.get("bvp_label", "")
                bvp_sig = p.get("bvp_sig", "no_data")
                if bvp_lbl and bvp_sig != "no_data":
                    st.write(f"• {bvp_lbl}")
                br = p.get("blast_rate", 0)
                st.write(f"• Blast%: {br*100:.1f}%" if br and br > 0 else "• Blast%: —")
                st.write(f"• Lineup: {p['lineup_label']}")
                st.write(f"• Platoon: {p['platoon_label']}")

            with col_b:
                st.markdown("**⚾ Pitcher Matchup**")
                tbd_note = " ⚠️ TBD — score capped at 72" if p["sp_tbd"] else ""
                st.write(f"• SP: {p['sp_name']}{tbd_note}")
                st.write(f"• SP Hand: {p['sp_hand']}")
                st.write(f"• {p['pitcher_label']}")
                st.markdown("**🌤️ Environment**")
                st.write(f"• {p['park_label']}")
                st.write(f"• {p['weather_label']}")
                st.write(f"• Implied Runs: {p['implied_total']:.1f}")

            with col_c:
                st.markdown("**📊 Score Breakdown**")
                sub_labels = {
                    "⚾ Pitcher Vuln (30%)": p["sub_pitcher"],
                    "🏏 Batter (28%)": p["sub_batter"],
                    "🤚 Platoon (12%)": p["sub_platoon"],
                    "💰 Vegas (8%)": p["sub_vegas"],
                    "🏟️ Park (7%)": p["sub_park"],
                    "📈 Streak (5%)": p.get("sub_streak", 50),
                    "🔄 TTO (4%)": p.get("sub_tto", 50),
                    "🌤️ Weather (4%)": p["sub_weather"],
                    "🎯 Pitch Mix (2%)": p.get("sub_matchup", 50),
                    "📊 BvP (2%)": p.get("sub_bvp", 50),
                    "📋 Lineup (1%)": p["sub_lineup"],
                }
                matchup_lbl = p.get("matchup_label", "")
                if matchup_lbl and matchup_lbl != "Pitch mix: avg splits":
                    st.caption(f"🎯 {matchup_lbl}")
                for label, val in sub_labels.items():
                    bar_color = "#00ff88" if val >= 70 else "#ffdd00" if val >= 50 else "#ff4444"
                    bar_width = int(val)
                    st.markdown(f"{label}: **{val:.0f}**")
                    st.markdown(
                        f'<div style="background:#333;border-radius:4px;height:6px;width:100%">'
                        f'<div style="background:{bar_color};width:{bar_width}%;height:6px;border-radius:4px">'
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )
                st.markdown(f"**🎯 Final Score: {p['score']:.0f} ({p['prob']*100:.0f}%)**")
                st.markdown(f"**💣 HR Score: {p['hr_score']:.0f}**")

    # ── Unmatched players ─────────────────────────────────────────────────────
    unmatched = [p for p in plays
                 if p.get("_batter_prov", {}).get("k_rate", "league_avg") == "league_avg"
                 and p.get("_batter_prov", {}).get("slg_proxy", "league_avg") == "league_avg"]
    if unmatched:
        with st.expander(
            f"⚠️ {len(unmatched)} players scored on league averages — could not match to real stats"
        ):
            st.caption(
                "These players were not found in any data source. All scores use 2025 MLB league "
                "averages. Do not bet any of these. Check name spelling or verify they're on an "
                "active roster."
            )
            for p in unmatched:
                nb_reasons = "; ".join(p.get("non_bettable_reasons", []))
                st.markdown(
                    f"- **{p['name']}** ({p['team']}) — {nb_reasons or 'no stat match found'}"
                )
