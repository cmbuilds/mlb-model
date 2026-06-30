"""ui/hr_tab.py — Home Run Plays tab.

Thin Streamlit renderer. All HR scoring is pre-computed in plays dicts by run_model().
"""

import pandas as pd
import streamlit as st
from typing import Dict, List

from lib.constants import PARK_HR_FACTORS, STADIUM_COORDS


def display_hr_plays(plays: List[Dict]):
    """Display top HR upside plays."""

    st.header("💣 Home Run Plays")
    st.caption("Top 10 daily HR candidates. Powered by barrel rate, hard hit%, exit velocity, ISO, park factor, wind, and implied total.")

    hr_sorted = sorted(plays, key=lambda x: x["hr_score"], reverse=True)[:10]

    rows = []
    for p in hr_sorted:
        wind_label = p.get("weather_label", "").split("|")[0].strip()
        park_name = STADIUM_COORDS.get(p["park"], (0, 0, p["park"], False))[2]
        sweet = p.get("sweet_spot_rate", 0)

        rows.append({
            "HR Score": f"{p['hr_score']:.0f}",
            "Player": p["name"],
            "Team": p["team"],
            "Opp SP": p["sp_name"][:20],
            "Barrel%": f"{p['barrel_rate']*100:.1f}%" if p["barrel_rate"] else "—",
            "HH%": f"{p['hard_hit_rate']*100:.1f}%" if p.get("hard_hit_rate") else "—",
            "EV": f"{p.get('exit_velocity', 0):.1f}" if p.get("exit_velocity", 0) > 0 else "—",
            "ISO": f"{p.get('iso', 0):.3f}" if p.get("iso", 0) > 0 else "—",
            "Sweet Spot%": f"{sweet*100:.1f}%" if sweet and sweet != 0.305 else "—",
            "Park HR Factor": f"{PARK_HR_FACTORS.get(p['park'], 1.0):.2f}x",
            "Park": park_name[:20],
            "Wind": p.get("wind_dir", "") + f" {p.get('wind_speed', 0):.0f}mph",
            "Wind Effect": "🔥 Out" if p.get("wind_effect") == "strong_out" else "💨 Out" if p.get("wind_effect") == "out" else "❄️ In" if p.get("wind_effect") == "in" else "🏟️" if p.get("is_dome") else "—",
            "Imp. Runs": f"{p['implied_total']:.1f}",
            "TB Score": f"{p['score']:.0f}",
        })

    if rows:
        df = pd.DataFrame(rows)

        def color_hr(val):
            try:
                v = float(val)
                if v >= 75: return "color: #ff4444; font-weight: bold"
                elif v >= 60: return "color: #ff8800; font-weight: bold"
                return ""
            except Exception:
                return ""

        styled = df.style.map(color_hr, subset=["HR Score"])
        st.dataframe(styled, use_container_width=True)

    # Top 3 HR plays detailed
    st.markdown("---")
    st.subheader("🔥 Top 3 HR Plays — Detail")

    for i, p in enumerate(hr_sorted[:3], 1):
        with st.expander(f"#{i}: {p['name']} ({p['team']}) — HR Score: {p['hr_score']:.0f}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Barrel%:** {p['barrel_rate']*100:.1f}%" if p["barrel_rate"] else "**Barrel%:** —")
                st.write(f"**Hard Hit%:** {p['hard_hit_rate']*100:.1f}%" if p.get("hard_hit_rate") else "**Hard Hit%:** —")
                st.write(f"**Exit Velocity:** {p.get('exit_velocity', 0):.1f} mph" if p.get("exit_velocity", 0) > 0 else "**Exit Velocity:** —")
                st.write(f"**ISO:** {p.get('iso', 0):.3f}" if p.get("iso", 0) > 0 else "**ISO:** —")
                st.write(f"**Park:** {STADIUM_COORDS.get(p['park'], (0,0,p['park'],False))[2]}")
                st.write(f"**Park HR Factor:** {PARK_HR_FACTORS.get(p['park'], 1.0):.2f}x")
            with col2:
                st.write(f"**Wind:** {p.get('wind_dir', '')} @ {p.get('wind_speed', 0):.0f}mph ({p.get('wind_effect', 'neutral')})")
                st.write(f"**Temp:** {p['temperature']:.0f}°F")
                st.write(f"**Implied Runs:** {p['implied_total']:.1f}")
                st.write(f"**Total Bases Score:** {p['score']:.0f}")

            # SGP opportunity check
            if p["score"] >= 60:
                same_game = [op for op in hr_sorted if op["game_id"] == p["game_id"] and op["name"] != p["name"]]
                if same_game:
                    st.success(f"⭐ SGP Opportunity: {p['name']} HR + {same_game[0]['name']} O1.5 TB in same game!")
