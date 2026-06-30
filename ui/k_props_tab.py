"""ui/k_props_tab.py — K Props tab.

Thin Streamlit renderer. Scoring delegated to scoring.strikeout.compute_sp_k_score.
"""

import pandas as pd
import streamlit as st
from typing import Dict, List

from scoring.strikeout import compute_sp_k_score


def display_k_props_tab(plays: List[Dict], ump_data: Dict):
    """
    Tab: ⚡ K Props — PITCHER strikeout prop model (e.g. Crochet O7.5 Ks).
    Ranks today's starting pitchers by their projected strikeout upside.
    Score 0-100: HIGH = elite K pitcher in a favorable matchup = bet the OVER on their K prop.

    Inputs:
      - SP K% + SwStr% (primary strikeout predictors)
      - Opposing lineup K% average (how much does this lineup swing and miss)
      - Umpire zone tendency (k_rate_added)
      - Game environment (low implied total = pitcher's duel = more Ks)
      - Innings opportunity (SP projected to go deep vs short hook)
    """
    st.header("⚡ K Props — Pitcher Strikeout Model")
    st.caption(
        "Ranks today's starters by projected strikeout upside. "
        "**HIGH score = bet the OVER on that pitcher's K prop.** "
        "e.g. Garrett Crochet O7.5 Ks, Paul Skenes O6.5 Ks. "
        "Score 80+ = elite K spot. Inputs: SP K%/SwStr%, opposing lineup K%, umpire zone, game total."
    )

    if not plays:
        st.info("Run the model first to see pitcher K prop scores.")
        return

    pitching_df = st.session_state.get("_pitching_df_global", pd.DataFrame())

    # ── Build one row per unique starting pitcher ─────────────────────────────
    # Collect games from plays
    seen_pitchers = {}  # sp_name -> aggregated data
    for p in plays:
        sp_name = p.get("sp_name", "TBD")
        if not sp_name or sp_name == "TBD":
            continue
        team = p.get("opponent", "")  # pitcher's team is the opponent of the batter
        game_id = p.get("game_id", "")

        # Batter K% for this opponent batter (contributes to lineup K avg)
        batter_k = p.get("k_rate", 0.228) or 0.228

        if sp_name not in seen_pitchers:
            seen_pitchers[sp_name] = {
                "sp_name":   sp_name,
                "sp_team":   team,
                "sp_hand":   p.get("sp_hand", "R"),
                "game_id":   game_id,
                "opp_team":  p.get("team", ""),
                "_pitcher_k_rate":  p.get("_pitcher_k_rate", 0.228),
                "_pitcher_swstr":   p.get("_pitcher_swstr", 0.0),
                "implied_total":    p.get("implied_total", 4.5),
                "batter_k_list":    [batter_k],
                "n_batters":        1,
            }
        else:
            seen_pitchers[sp_name]["batter_k_list"].append(batter_k)
            seen_pitchers[sp_name]["n_batters"] += 1

    if not seen_pitchers:
        st.warning("No starting pitcher data found. Run the model with confirmed lineups.")
        return

    # ── Score each pitcher ────────────────────────────────────────────────────
    pitcher_rows = []
    for sp_name, d in seen_pitchers.items():
        pit_k     = d["_pitcher_k_rate"] or 0.228
        pit_swstr = d["_pitcher_swstr"]  or 0.0
        opp_k_avg = sum(d["batter_k_list"]) / len(d["batter_k_list"]) if d["batter_k_list"] else 0.228
        implied   = d["implied_total"] or 4.5
        n_bat     = d["n_batters"]

        # Umpire adjustment
        game_pk_int = None
        try: game_pk_int = int(d["game_id"])
        except (ValueError, TypeError): pass
        ump_entry = ump_data.get(game_pk_int, {})
        ump_k_adj = float(ump_entry.get("k_rate_added", 0.0))
        ump_name  = ump_entry.get("ump_name", "—")

        # ── Score via pure module ─────────────────────────────────────────────
        pit_stats_tmp = {"k_rate_allowed": pit_k, "swstr_pct": pit_swstr}
        final, _k_label, details = compute_sp_k_score(
            pitcher_stats=pit_stats_tmp,
            opp_lineup_k_avg=opp_k_avg,
            implied_total=implied,
            ump_k_adj=ump_k_adj,
        )

        # Tier
        if final >= 80:
            tier = "⚡ Elite K Spot"
        elif final >= 70:
            tier = "🔥 Strong K"
        elif final >= 60:
            tier = "📊 Lean K"
        else:
            tier = "➖ Skip"

        # Projected K range (rough: elite K SP ~8-9 Ks/9, avg ~6.5)
        # Use K% * ~27 batters faced (avg ~6IP)
        proj_bf = 21 if implied <= 4.0 else 18  # fewer BF in high-scoring games
        proj_ks = round(pit_k * proj_bf, 1)

        pitcher_rows.append({
            "final":           final,
            "tier":            tier,
            "sp_name":         sp_name,
            "sp_team":         d["sp_team"],
            "opp_team":        d["opp_team"],
            "sp_hand":         d["sp_hand"],
            "pit_k":           pit_k,
            "pit_swstr":       pit_swstr,
            "swstr_is_proxy":  not details["swstr_real"],
            "opp_k_avg":       opp_k_avg,
            "implied":         implied,
            "ump_name":        ump_name,
            "ump_k_adj":       ump_k_adj * 100,
            "n_batters":       n_bat,
            "sp_k_score":      details["sub_sp_k"],
            "swstr_score":     details["sub_swstr"],
            "opp_k_score":     details["sub_opp_lineup"],
            "ctx_score":       details["sub_context"],
            "proj_ks":         proj_ks,
        })

    pitcher_rows.sort(key=lambda x: x["final"], reverse=True)

    # ── Summary metrics ──────────────────────────────────────────────────────
    elite = [r for r in pitcher_rows if r["final"] >= 80]
    strong= [r for r in pitcher_rows if 70 <= r["final"] < 80]
    lean  = [r for r in pitcher_rows if 60 <= r["final"] < 70]
    skip  = [r for r in pitcher_rows if r["final"] < 60]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("⚡ Elite (80+)",  len(elite))
    m2.metric("🔥 Strong (70-79)", len(strong))
    m3.metric("📊 Lean (60-69)", len(lean))
    m4.metric("➖ Skip",         len(skip))
    m5.metric("SPs Today",       len(pitcher_rows))

    # ── Ranked pitcher cards ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Pitcher K Prop Rankings")
    st.caption("Ranked from best K spot to worst. Higher score = bet the OVER on that pitcher's K line.")

    for rank, r in enumerate(pitcher_rows, 1):
        sc = r["final"]
        if sc >= 80:
            border = "#ff4444"; flame = "⚡⚡"
        elif sc >= 70:
            border = "#ff8800"; flame = "🔥"
        elif sc >= 60:
            border = "#ffdd00"; flame = "📊"
        else:
            border = "#444444"; flame = "➖"

        with st.container():
            c_rank, c_pit, c_stats, c_badge = st.columns([0.4, 2.8, 4.2, 1.4])

            with c_rank:
                st.markdown(
                    f"<div style='text-align:center;margin-top:10px'>"
                    f"<div style='font-size:1.6rem;font-weight:900;color:{border}'>{rank}</div>"
                    f"<div style='font-size:0.6rem;color:{border}'>{flame}</div></div>",
                    unsafe_allow_html=True)

            with c_pit:
                st.markdown(f"**{r['sp_name']}** &nbsp;`{r['sp_team']}`")
                st.caption(f"vs {r['opp_team']} lineup · {r['sp_hand']}HP")
                tier_col = "#ff4444" if "Elite" in r["tier"] else "#ff8800" if "Strong" in r["tier"] else "#ffdd00" if "Lean" in r["tier"] else "#666"
                st.markdown(f"<span style='background:#111;border:1px solid {tier_col};"
                            f"color:{tier_col};border-radius:5px;padding:1px 8px;"
                            f"font-size:0.75rem'>{r['tier']}</span>", unsafe_allow_html=True)
                proj_str = f"Proj Ks: ~{r['proj_ks']:.0f}"
                st.caption(proj_str)

            with c_stats:
                pk_pct  = f"{r['pit_k']*100:.1f}%"
                sw_pct  = f"{r['pit_swstr']*100:.1f}%{'~' if r.get('swstr_is_proxy') else ''}"
                ok_pct  = f"{r['opp_k_avg']*100:.1f}%"
                impl    = f"{r['implied']:.1f}" if r["implied"] > 0 else "N/A"
                ump_s   = f"{r['ump_name']} ({r['ump_k_adj']:+.1f}pp)" if r["ump_name"] != "—" else "—"

                col_a, col_b = st.columns(2)
                with col_a:
                    st.caption(f"**SP K%:** {pk_pct}")
                    st.caption(f"**SP SwStr%:** {sw_pct}")
                    st.caption(f"**Umpire:** {ump_s}")
                with col_b:
                    st.caption(f"**Opp Lineup K%:** {ok_pct}")
                    st.caption(f"**Game Total:** {impl}")
                    st.caption(f"**BFs scored:** {r['n_batters']}/9")

            with c_badge:
                st.markdown(
                    f"<div style='text-align:center;background:#1a1a2e;"
                    f"border:2px solid {border};border-radius:10px;padding:10px 4px;margin-top:4px'>"
                    f"<div style='font-size:1.6rem;font-weight:900;color:{border}'>{sc:.0f}</div>"
                    f"<div style='font-size:0.65rem;color:#aaa'>K SCORE</div>"
                    f"<div style='font-size:0.7rem;color:{border};margin-top:2px'>"
                    f"~{r['proj_ks']:.0f} Ks</div></div>",
                    unsafe_allow_html=True)

        st.markdown("<hr style='margin:6px 0;border-color:#2a2a2a'>", unsafe_allow_html=True)

    # ── Full table ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Full Pitcher K Table")

    table_rows = []
    for r in pitcher_rows:
        table_rows.append({
            "K Score":        r["final"],
            "Tier":           r["tier"],
            "Pitcher":        r["sp_name"],
            "Team":           r["sp_team"],
            "Hand":           r["sp_hand"],
            "vs Lineup":      r["opp_team"],
            "SP K%":          f"{r['pit_k']*100:.1f}%",
            "SP SwStr%":      f"{r['pit_swstr']*100:.1f}%{'~' if r.get('swstr_is_proxy') else ''}",
            "Opp Lineup K%":  f"{r['opp_k_avg']*100:.1f}%",
            "Game Total":     f"{r['implied']:.1f}" if r["implied"] > 0 else "N/A",
            "Proj Ks":        f"~{r['proj_ks']:.0f}",
        })

    df_k = pd.DataFrame(table_rows)

    def color_k_score(val):
        try:
            v = float(val)
            if v >= 80: return "color:#ff4444;font-weight:bold"
            if v >= 70: return "color:#ff8800;font-weight:bold"
            if v >= 60: return "color:#ffdd00"
            return "color:#888888"
        except: return ""

    def color_k_tier(val):
        s = str(val)
        if "Elite" in s: return "color:#ff4444;font-weight:bold"
        if "Strong" in s: return "color:#ff8800;font-weight:bold"
        if "Lean" in s:  return "color:#ffdd00"
        return "color:#888888"

    styled_k = df_k.style.map(color_k_score, subset=["K Score"]).map(color_k_tier, subset=["Tier"])
    st.dataframe(styled_k, use_container_width=True)

    csv_k = df_k.to_csv(index=False)
    st.download_button("📥 Export K Props CSV", csv_k, "k_props.csv", "text/csv", key="dl_kprops")

    st.markdown("---")
    st.caption(
        "💡 **How to use:** Find pitchers with K Score 70+. "
        "Cross with their actual posted K prop line (e.g. Crochet -115 O7.5 Ks on HardRock). "
        "Proj Ks is a rough estimate based on K% × ~18-21 projected batters faced. "
        "Elite K spots (80+) = high confidence OVER plays."
    )
