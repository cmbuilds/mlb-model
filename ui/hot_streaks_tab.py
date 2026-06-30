"""ui/hot_streaks_tab.py — Hot Streaks tab.

Thin Streamlit renderer. All streak data is pre-computed in plays dicts by run_model().
"""

import io as _io
import pandas as pd
import streamlit as st
from typing import Dict, List


def display_hot_streaks_tab(plays: List[Dict]):
    """
    Tab: 🔥 Hot Streaks
    Ranks the top 10 hottest batters by recent form score and TB/game.
    """
    st.header("🔥 Hot Streaks — Top Batters in Form")
    st.caption(
        "Top batters ranked by recent performance (last 7 games). "
        "TB/g = total bases per game over last 7. "
        "🔥 Hot = recent TB/g is 20%+ above season baseline. "
        "Cold or no-data batters are excluded."
    )

    if not plays:
        st.info("Run the model first to see streak data.")
        return

    streak_plays = []
    for p in plays:
        streak_label = p.get("streak_label", "")
        sub_streak   = p.get("sub_streak", 50.0)
        tb_pg        = p.get("recent_tb_per_game")
        recent_games = p.get("recent_games", 0)

        if tb_pg is None or recent_games < 3:
            continue
        if sub_streak < 40:
            continue

        season_slg = p.get("xslg", 0.398) or 0.398
        season_exp = season_slg * 3.7
        pct_above  = ((tb_pg / season_exp) - 1.0) * 100 if season_exp > 0 else 0.0

        streak_plays.append({
            "name":          p["name"],
            "team":          p["team"],
            "opponent":      p.get("opponent", ""),
            "sp_name":       p.get("sp_name", "TBD"),
            "sp_hand":       p.get("sp_hand", "R"),
            "batter_hand":   p.get("batter_hand", "R"),
            "lineup_slot":   p.get("lineup_slot", 9),
            "sub_streak":    sub_streak,
            "tb_per_game":   tb_pg,
            "recent_games":  recent_games,
            "season_exp":    round(season_exp, 2),
            "pct_above":     round(pct_above, 1),
            "streak_label":  streak_label,
            "hr_last_7":     p.get("_hr_last7", 0),
            "h_last_7":      p.get("_h_last7", 0),
            "ab_last_7":     p.get("_ab_last7", 0),
            "model_score":   p.get("score", 0),
            "tier":          p.get("tier", "❌ NO PLAY"),
            "barrel_rate":   p.get("barrel_rate", 0),
            "hard_hit_rate": p.get("hard_hit_rate", 0),
            "bvp_sig":       p.get("bvp_sig", ""),
            "bvp_label":     p.get("bvp_label", ""),
            "hr_score":      p.get("hr_score", 0),
            "prob":          p.get("prob", 0),
        })

    def streak_composite(p):
        streak_sc = p["sub_streak"]
        tb_norm   = min(100, max(0, (p["tb_per_game"] - 0.5) / (3.5 - 0.5) * 100))
        return streak_sc * 0.50 + tb_norm * 0.50

    streak_plays.sort(key=streak_composite, reverse=True)

    if not streak_plays:
        st.info("No recent form data available — lineups may not be confirmed yet, "
                "or no batters have 3+ recent games loaded.")
        return

    top10 = streak_plays[:10]
    cold_count = len([p for p in plays
                      if p.get("sub_streak", 50) < 40 and p.get("recent_tb_per_game") is not None])

    blazing = [p for p in streak_plays if p["sub_streak"] >= 70]
    hot     = [p for p in streak_plays if 60 <= p["sub_streak"] < 70]
    warm    = [p for p in streak_plays if 50 <= p["sub_streak"] < 60]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🔥🔥 Blazing (70+)", len(blazing))
    m2.metric("🔥 Hot (60-69)",     len(hot))
    m3.metric("📈 Warm (50-59)",    len(warm))
    m4.metric("❄️ Cold (faded)",    cold_count)

    st.markdown("---")
    st.subheader(f"🏆 Top {len(top10)} Batters in Form")

    for rank, p in enumerate(top10, 1):
        score     = p["sub_streak"]
        tb_pg     = p["tb_per_game"]
        pct_above = p["pct_above"]

        if score >= 70:
            border = "#ff4444"; flame = "🔥🔥"; heat_label = "BLAZING"
        elif score >= 60:
            border = "#ff8800"; flame = "🔥"; heat_label = "HOT"
        else:
            border = "#ffdd00"; flame = "📈"; heat_label = "WARM"

        model_sc = p["model_score"]
        if model_sc >= 80:
            tier_color = "#00ff88"
        elif model_sc >= 70:
            tier_color = "#ffdd00"
        elif model_sc >= 60:
            tier_color = "#ff8800"
        else:
            tier_color = "#666666"

        with st.container():
            col_rank, col_player, col_streak, col_context, col_badges = st.columns([0.4, 2.2, 2.6, 2.6, 1.6])

            with col_rank:
                st.markdown(
                    f"<div style='text-align:center;margin-top:8px'>"
                    f"<div style='font-size:1.6rem;font-weight:900;color:{border}'>{rank}</div>"
                    f"<div style='font-size:0.65rem;color:{border};font-weight:bold'>{heat_label}</div>"
                    f"</div>",
                    unsafe_allow_html=True)

            with col_player:
                st.markdown(f"**{p['name']}** &nbsp;`{p['team']}`")
                vs_str = f"vs {p['opponent']}" if p["opponent"] else ""
                sp_str = (f"· {p['sp_name'][:20]} ({p['sp_hand']}HP)"
                          if p["sp_name"] not in ("TBD", "") else "· SP TBD")
                st.caption(f"#{p['lineup_slot']} · {p['batter_hand']}HB {vs_str} {sp_str}")
                bvp = p.get("bvp_sig", "")
                if bvp == "owns":
                    st.caption("🔥 OWNS this pitcher career")
                elif bvp == "edge":
                    st.caption("🟢 Edge vs this pitcher")
                elif bvp == "dominated":
                    st.caption("⚠️ Dominated by this pitcher")

            with col_streak:
                bl = p["season_exp"]
                bar_pct = min(100, int(tb_pg / max(bl, 0.1) * 60))
                pct_str = (f"+{pct_above:.0f}% above baseline"
                           if pct_above >= 0 else f"{pct_above:.0f}% below")
                st.markdown(f"{flame} **{tb_pg:.2f} TB/g** last {p['recent_games']}g")
                st.markdown(
                    f"<div style='font-size:0.75rem;color:#aaa;margin-bottom:2px'>"
                    f"Baseline: {bl:.2f} · {pct_str}</div>"
                    f"<div style='background:#333;border-radius:4px;height:6px;width:100%'>"
                    f"<div style='background:{border};width:{bar_pct}%;height:6px;border-radius:4px'>"
                    f"</div></div>",
                    unsafe_allow_html=True)
                st.markdown(" ")
                h7 = p["h_last_7"]; ab7 = p["ab_last_7"]; hr7 = p["hr_last_7"]
                avg7 = f"{h7/ab7:.3f}" if ab7 > 0 else "—"
                hr_str = f" · {hr7}HR" if hr7 > 0 else ""
                st.caption(f"Last {p['recent_games']}g line: {h7}/{ab7} ({avg7}){hr_str}")

            with col_context:
                barrel = p["barrel_rate"] * 100
                hh = p["hard_hit_rate"] * 100
                st.caption(f"**Barrel%:** {barrel:.1f}%  |  **HH%:** {hh:.1f}%")
                st.caption(f"**HR Score:** {p['hr_score']:.0f}/100  |  **O1.5 Prob:** {p['prob']*100:.0f}%")
                lbl = (p["streak_label"].replace("Form: ", "")
                       .replace("🔥 Hot", "").replace("❄️ Cold", "").strip())
                if lbl:
                    st.caption(lbl)

            with col_badges:
                st.markdown(
                    f"<div style='text-align:center;background:#1a1a2e;"
                    f"border:2px solid {tier_color};border-radius:10px;padding:6px 2px;margin-bottom:4px'>"
                    f"<div style='font-size:1.2rem;font-weight:900;color:{tier_color}'>{model_sc:.0f}</div>"
                    f"<div style='font-size:0.6rem;color:#aaa'>MODEL</div></div>",
                    unsafe_allow_html=True)
                st.markdown(
                    f"<div style='text-align:center;background:#1a1a2e;"
                    f"border:2px solid {border};border-radius:10px;padding:6px 2px'>"
                    f"<div style='font-size:1.2rem;font-weight:900;color:{border}'>{score:.0f}</div>"
                    f"<div style='font-size:0.6rem;color:#aaa'>STREAK</div></div>",
                    unsafe_allow_html=True)

        st.markdown("<hr style='margin:6px 0;border-color:#2a2a2a'>", unsafe_allow_html=True)

    if len(streak_plays) > 10:
        st.markdown("---")
        with st.expander(f"📋 Full Streak Rankings ({len(streak_plays)} batters with recent data)"):
            full_rows = []
            for i, p in enumerate(streak_plays, 1):
                h7 = p["h_last_7"]; ab7 = p["ab_last_7"]; hr7 = p["hr_last_7"]
                platoon_str = (
                    "LHB" if p["batter_hand"] == "L"
                    else "Switch" if p["batter_hand"] == "S"
                    else "RHB"
                ) + " vs " + p["sp_hand"] + "HP"
                full_rows.append({
                    "Rank":         i,
                    "Player":       p["name"],
                    "Hand":         p["batter_hand"],
                    "Team":         p["team"],
                    "Streak Score": p["sub_streak"],
                    "TB/g (last7)": f"{p['tb_per_game']:.2f}",
                    "vs Baseline":  f"{p['pct_above']:+.0f}%",
                    "Hitting Line": f"{h7}/{ab7}" + (f" {hr7}HR" if hr7 else ""),
                    "Games":        p["recent_games"],
                    "Model Score":  p["model_score"],
                    "Tier":         p["tier"],
                    "Platoon":      platoon_str,
                    "vs":           p["opponent"],
                    "Opp SP":       p["sp_name"][:18],
                })
            df_str = pd.DataFrame(full_rows)

            def color_streak_sc(val):
                try:
                    v = float(val)
                    if v >= 70: return "color:#ff4444;font-weight:bold"
                    if v >= 60: return "color:#ff8800;font-weight:bold"
                    if v >= 50: return "color:#ffdd00"
                    return "color:#888888"
                except Exception:
                    return ""

            styled_str = df_str.style.map(color_streak_sc, subset=["Streak Score"])
            st.dataframe(styled_str, use_container_width=True)

            _csv_buf = _io.StringIO()
            df_str.to_csv(_csv_buf, index=False)
            st.download_button(
                label="📥 Export Hot Streaks CSV",
                data=_csv_buf.getvalue(),
                file_name=f"hot_streaks_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="streak_export_btn",
            )

    st.markdown("---")
    st.info(
        "💡 **How to use Hot Streaks:** Cross-reference these batters with the O1.5 Leaderboard. "
        "A hot batter (70+ streak score) with a Tier 2+ model score and a weak opposing pitcher "
        "is your highest-confidence same-game parlay anchor."
    )
