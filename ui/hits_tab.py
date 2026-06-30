"""ui/hits_tab.py — Over 0.5 Total Bases tab.

Thin Streamlit renderer. Scoring delegated to markets.hits_o05.score_one_batter_o05.
Tier labels: TIER 1 (80+) | TIER 2 (70-79) | TIER 3 (60-69) | NO PLAY (<60)
"""

import pandas as pd
import streamlit as st
from typing import Dict, List

from markets.hits_o05 import score_one_batter_o05


def display_hits_tab(plays: List[Dict]):
    """
    Over 0.5 Total Bases tab.
    Re-scores all players using contact/hit model via score_one_batter_o05.
    Tiers: TIER 1 (80+), TIER 2 (70-79), TIER 3 (60-69), NO PLAY (<60).
    """
    st.header("🎯 Over 0.5 Total Bases")
    st.caption("Any hit model — singles count. Higher base rate (~65%). Contact + K-avoidance driven.")

    if not plays:
        st.info("Run the model first.")
        return

    # Re-score every player using the Phase 4.2 hits market module
    hits_plays = []
    for p in plays:
        batter_stats = {
            "k_rate":        p.get("k_rate", 0.228),
            "slg_proxy":     p.get("xslg", 0.398),
            "woba":          p.get("woba", p.get("xslg", 0.398) * 0.78),
            "hard_hit_rate": p.get("hard_hit_rate", 0.370),
            "wrc_plus":      p.get("wrc_plus", 100.0) or 100.0,
            "_provenance": {
                "k_rate":        p.get("prov_krate", "league_avg"),
                "hard_hit_rate": p.get("prov_hh", "league_avg"),
                "slg_proxy":     p.get("prov_xslg", "league_avg"),
            },
        }
        pitcher_stats = {
            "k_rate_allowed":   p.get("_pitcher_k_rate", 0.228),
            "hard_hit_allowed": p.get("hard_hit_allowed", 0.370),
            "_provenance": {"k_rate_allowed": "measured"},
        }

        result = score_one_batter_o05(
            name=p["name"],
            player_id=str(p.get("player_id", "")),
            team=p["team"],
            opponent=p.get("opponent", ""),
            game_pk=str(p.get("game_pk", "")),
            batter_hand=p.get("batter_hand", "R"),
            hand_real=p.get("hand_real", False),
            sp_hand=p.get("sp_hand", "R"),
            sp_name=p.get("sp_name", "TBD"),
            sp_id=str(p.get("sp_id", "")),
            lineup_slot=p.get("lineup_slot", 5),
            lineup_confirmed=p.get("lineup_confirmed", True),
            batter_position="",
            park_team=p.get("park", p.get("team", "")),
            batter_stats=batter_stats,
            pitcher_stats=pitcher_stats,
            recent_form=p.get("_recent_form", {}),
            implied=p.get("implied_total", 0.0),
            prop_implied=None,
            team_bullpen_scores={},
        )

        hits_plays.append({
            **p,
            "h_score": result["score"],
            "h_prob":  result["prob"],
            "h_tier":  result["tier"],
        })

    hits_plays.sort(key=lambda x: x["h_score"], reverse=True)

    # Summary metrics
    tier1  = [p for p in hits_plays if p["h_tier"] == "🔒 TIER 1"]
    tier2  = [p for p in hits_plays if p["h_tier"] == "✅ TIER 2"]
    tier3  = [p for p in hits_plays if p["h_tier"] == "📊 TIER 3"]
    no_play = [p for p in hits_plays if p["h_tier"] == "❌ NO PLAY"]

    if tier1:
        st.success(f"🔒 {len(tier1)} TIER 1 plays — elite contact spots, parlay anchors")
    elif tier2:
        st.info(f"✅ {len(tier2)} TIER 2 plays — strong contact matchups")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("🔒 TIER 1", len(tier1))
    with col2: st.metric("✅ TIER 2", len(tier2))
    with col3: st.metric("📊 TIER 3", len(tier3))
    with col4: st.metric("❌ NO PLAY", len(no_play))

    st.caption("Tier thresholds: TIER 1 80+ | TIER 2 70-79 | TIER 3 60-69 | NO PLAY <60")
    st.markdown("---")

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        tier_filter = st.multiselect(
            "Filter tier", ["🔒 TIER 1", "✅ TIER 2", "📊 TIER 3", "❌ NO PLAY"],
            default=["🔒 TIER 1", "✅ TIER 2", "📊 TIER 3"], key="hits_tier_filter"
        )
    with col_f2:
        teams = sorted(set(p["team"] for p in hits_plays))
        team_filter = st.multiselect("Filter team", teams, default=[], key="hits_team_filter")
    with col_f3:
        min_score = st.slider("Min score", 0, 100, 60, key="hits_min_score")

    filtered = [p for p in hits_plays
                if p["h_tier"] in tier_filter
                and (not team_filter or p["team"] in team_filter)
                and p["h_score"] >= min_score]

    st.markdown(f"**Showing {len(filtered)} batters**")

    # Build table
    rows = []
    for p in filtered:
        tbd = " ⚠️" if p.get("sp_tbd") else ""
        rows.append({
            "H-Score": f"{p['h_score']:.0f}",
            "Tier": p["h_tier"],
            "Player": p["name"],
            "Team": p["team"],
            "Vs": p["opponent"],
            "Slot": f"#{p['lineup_slot']}",
            "Hand": p["batter_hand"],
            "Opp SP": p["sp_name"][:20] + tbd,
            "SP 🤚": p["sp_hand"],
            "Prob": f"{p['h_prob']*100:.0f}%",
            "K%": f"{p.get('k_rate', 0)*100:.1f}%" if p.get("k_rate") else "—",
            "HH%": f"{p.get('hard_hit_rate', 0)*100:.1f}%" if p.get("hard_hit_rate") else "—",
            "xSLG": f"{p.get('xslg', 0):.3f}" if p.get("xslg") else "—",
            "Platoon": p.get("platoon_label", "").split("(")[0].strip(),
            "Park": p.get("park", ""),
            "Imp.Runs": f"{p.get('implied_total', 0):.1f}" if p.get("implied_total", 0) > 0 else "—",
            "O1.5 Score": f"{p['score']:.0f}",  # cross-reference
        })

    if rows:
        df = pd.DataFrame(rows)

        def color_htier(val):
            s = str(val)
            if "TIER 1" in s: return "color: #00ff88; font-weight: bold"
            elif "TIER 2" in s: return "color: #66ddff; font-weight: bold"
            elif "TIER 3" in s: return "color: #ffdd00"
            return "color: #888888"

        def color_hscore(val):
            try:
                v = float(str(val))
                if v >= 80: return "color: #00ff88; font-weight: bold"
                elif v >= 70: return "color: #66ddff; font-weight: bold"
                elif v >= 60: return "color: #ffdd00"
                return "color: #888888"
            except: return ""

        styled = df.style.map(color_htier, subset=["Tier"]).map(color_hscore, subset=["H-Score"])
        st.dataframe(styled, use_container_width=True, height=500)

        csv = df.to_csv(index=False)
        st.download_button("📥 Export O0.5 Plays", csv,
                           f"o05_picks_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", "text/csv")

    # Top plays detail
    st.markdown("---")
    st.subheader("🏆 Top O0.5 Plays — Full Breakdown")
    top = [p for p in filtered if p["h_score"] >= 60][:5]
    for i, p in enumerate(top, 1):
        with st.expander(f"{p['h_tier']} #{i}: {p['name']} ({p['team']}) — H-Score: {p['h_score']:.0f} | Prob: {p['h_prob']*100:.0f}% | O1.5 Score: {p['score']:.0f}"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Contact Profile (O0.5 drivers)**")
                st.write(f"• K%: {p.get('k_rate', 0)*100:.1f}% ← most critical" if p.get("k_rate") else "• K%: —")
                st.write(f"• Hard Hit%: {p.get('hard_hit_rate', 0)*100:.1f}%" if p.get("hard_hit_rate") else "• HH%: —")
                st.write(f"• xSLG: {p.get('xslg', 0):.3f}" if p.get("xslg") else "• xSLG: —")
                st.write(f"• Platoon: {p.get('platoon_label', '—')}")
                st.write(f"• Lineup: {p.get('lineup_label', '—')}")
            with col_b:
                st.markdown("**Pitcher + Environment**")
                st.write(f"• SP: {p['sp_name']} ({p['sp_hand']}HP)")
                st.write(f"• {p.get('pitcher_label', '—')}")
                st.write(f"• Park: {p.get('park_label', p.get('park', '—'))}")
                st.write(f"• Implied runs: {p.get('implied_total', 0):.1f}" if p.get("implied_total", 0) > 0 else "• Implied: —")
                st.markdown(f"**O0.5 Score: {p['h_score']:.0f} ({p['h_prob']*100:.0f}%)**")
                st.markdown(f"*vs O1.5 Score: {p['score']:.0f} ({p['prob']*100:.0f}%)*")

    # Mixed parlay note
    st.markdown("---")
    st.info("💡 **Mixed Parlay tip:** Combine TIER 1 plays from this tab with Tier 1/2 plays from the O1.5 Leaderboard in the Parlay Builder. High-contact bats + power matchups = diversified legs.")
