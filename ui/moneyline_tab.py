"""ui/moneyline_tab.py — Moneyline win-probability tab.

Thin Streamlit renderer. No picks, no edge, no market comparison.
Shows what the data says about who wins each game and by how much.
The user compares these probabilities to live odds themselves.

Gate: TBD/unconfirmed starter → UNKNOWN card, no projection.
Sort: most lopsided games first (largest |win% - 50%|).
"""

import pandas as pd
import streamlit as st
from typing import Dict, List

from markets.moneyline import score_game_ml
from scoring.moneyline import compute_team_offense_score


def display_moneyline_tab(
    games: List[Dict],
    plays: List[Dict],
    team_bullpen_scores: Dict,
):
    """
    Render the Moneyline tab.

    games: from fetch_schedule() — each has home_team, away_team,
           home_pitcher/away_pitcher, home_pitcher_id/away_pitcher_id.
    plays: model plays from run_model() — used for lineup wRC+ aggregation.
    team_bullpen_scores: {team_abbr: vuln_0_to_100} from compute_team_bullpen_scores().
    """
    from mlb_tb_analyzer import get_pitcher_stats

    st.header("🏦 Moneyline — Win Probability")
    st.caption(
        "Model win probability based on SP quality, lineup offense, and bullpen. "
        "No market comparison. Compare these numbers to live odds yourself."
    )

    if not games:
        st.info("Run the model first to see win probability analysis.")
        return
    if not plays:
        st.info("Run the model first to populate lineup data.")
        return

    pitching_df = st.session_state.get("_pitching_df_global", pd.DataFrame())
    LEAGUE_AVG_BP = 42.0

    # ── Score all games ───────────────────────────────────────────────────────
    scored = []
    for game in games:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        home_sp = (game.get("home_pitcher") or "").strip() or "TBD"
        away_sp = (game.get("away_pitcher") or "").strip() or "TBD"
        home_sp_id = str(game.get("home_pitcher_id") or "")
        away_sp_id = str(game.get("away_pitcher_id") or "")

        home_stats = get_pitcher_stats(home_sp, home_sp_id, pitching_df)
        away_stats = get_pitcher_stats(away_sp, away_sp_id, pitching_df)

        home_bp = team_bullpen_scores.get(home, LEAGUE_AVG_BP)
        away_bp = team_bullpen_scores.get(away, LEAGUE_AVG_BP)

        home_wrc, home_n = compute_team_offense_score(plays, home)
        away_wrc, away_n = compute_team_offense_score(plays, away)

        home_sp_matched = home_stats.get("data_source", "league_avg") != "league_avg"
        away_sp_matched = away_stats.get("data_source", "league_avg") != "league_avg"

        result = score_game_ml(
            home_team=home, away_team=away,
            home_sp_name=home_sp, away_sp_name=away_sp,
            home_sp_stats=home_stats, away_sp_stats=away_stats,
            home_wrc_plus=home_wrc, away_wrc_plus=away_wrc,
            home_n_batters=home_n, away_n_batters=away_n,
            home_bp_vuln=home_bp, away_bp_vuln=away_bp,
            home_bp_is_estimate=(home_bp == LEAGUE_AVG_BP),
            away_bp_is_estimate=(away_bp == LEAGUE_AVG_BP),
            home_sp_matched=home_sp_matched,
            away_sp_matched=away_sp_matched,
            home_sp_prov=home_stats.get("_provenance", {}),
            away_sp_prov=away_stats.get("_provenance", {}),
            park_key=game.get("home_team", ""),
            game_time=game.get("game_time", ""),
        )
        scored.append(result)

    # Sort: blocked games last; within unblocked, most lopsided first
    def _sort_key(r):
        if r["blocked"] or r["home_win_prob"] is None:
            return -1.0
        return abs(r["home_win_prob"] - 0.50)

    scored.sort(key=_sort_key, reverse=True)

    # ── Summary row ───────────────────────────────────────────────────────────
    unblocked = [r for r in scored if not r["blocked"]]
    blocked   = [r for r in scored if r["blocked"]]
    strong_lean = [r for r in unblocked
                   if r["home_win_prob"] is not None and
                   abs(r["home_win_prob"] - 0.50) >= 0.10]

    m1, m2, m3 = st.columns(3)
    m1.metric("Games with projection", len(unblocked))
    m2.metric("Clear leans (≥60% or ≤40%)", len(strong_lean))
    m3.metric("Blocked (TBD SP)", len(blocked))

    st.markdown("---")

    # ── Distribution table (user-requested spread check) ─────────────────────
    with st.expander("📊 Win-% Distribution — all games sorted", expanded=True):
        tbl_rows = []
        for r in scored:
            if r["blocked"]:
                tbl_rows.append({
                    "Matchup":   f"{r['away_team']} @ {r['home_team']}",
                    "Away SP":   r["away_sp_name"],
                    "Home SP":   r["home_sp_name"],
                    "Away Win%": "—",
                    "Home Win%": "—",
                    "SP Edge":   "—",
                    "Off Edge":  "—",
                    "BP Edge":   "—",
                    "Status":    "⚠️ TBD SP",
                })
            else:
                d = r["drivers"]
                hwp = r["home_win_prob"]
                awp = r["away_win_prob"]
                favor = r["home_team"] if hwp >= 0.50 else r["away_team"]
                spread = abs(hwp - 0.50) * 100
                tbl_rows.append({
                    "Matchup":   f"{r['away_team']} @ {r['home_team']}",
                    "Away SP":   r["away_sp_name"][:20] if r["away_sp_name"] else "—",
                    "Home SP":   r["home_sp_name"][:20] if r["home_sp_name"] else "—",
                    "Away Win%": f"{awp*100:.1f}%",
                    "Home Win%": f"{hwp*100:.1f}%",
                    "SP Edge":   f"{d.get('sp_edge_pts',0)*100:+.1f}%",
                    "Off Edge":  f"{d.get('off_edge_pts',0)*100:+.1f}%",
                    "BP Edge":   f"{d.get('bp_edge_pts',0)*100:+.1f}%",
                    "Status":    f"← {favor} {spread:.0f}pt lean",
                })

        if tbl_rows:
            def _color_wp(val):
                try:
                    v = float(str(val).replace("%", ""))
                    if v >= 65: return "color: #00ff88; font-weight: bold"
                    if v >= 58: return "color: #66ddff"
                    if v <= 35: return "color: #ff4444; font-weight: bold"
                    if v <= 42: return "color: #ffaa44"
                except Exception:
                    pass
                return ""

            df_tbl = pd.DataFrame(tbl_rows)
            s = df_tbl.style.map(_color_wp, subset=["Away Win%", "Home Win%"])
            st.dataframe(s, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🎯 Game Cards")

    # ── Per-game cards ────────────────────────────────────────────────────────
    for r in scored:
        home = r["home_team"]; away = r["away_team"]
        matchup = f"{away} @ {home}"

        if r["blocked"]:
            # TBD SP card
            with st.container():
                st.markdown(
                    f"<div style='border:1px solid #444;border-radius:8px;"
                    f"padding:12px 16px;margin-bottom:8px;background:#111'>"
                    f"<b>{matchup}</b> &nbsp; "
                    f"<span style='color:#ffaa00'>⚠️ SP not confirmed — no win probability</span><br>"
                    f"<span style='color:#666;font-size:0.8rem'>"
                    f"{' · '.join(r['block_reasons'])}"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )
            continue

        hwp = r["home_win_prob"]
        awp = r["away_win_prob"]
        d = r["drivers"]

        # Color: green for high confidence, yellow for mild lean, grey near 50%
        home_pct = hwp * 100
        away_pct = awp * 100
        spread = abs(hwp - 0.50)

        if spread >= 0.15:
            border = "#00ff88"
        elif spread >= 0.08:
            border = "#ffdd00"
        else:
            border = "#555555"

        def _pct_color(p):
            if p >= 65: return "#00ff88"
            if p >= 58: return "#66ddff"
            if p <= 35: return "#ff4444"
            if p <= 42: return "#ffaa44"
            return "#cccccc"

        with st.container():
            h_col, sp_col, d_col = st.columns([2, 3, 4])

            with h_col:
                # Win% display
                aw_c = _pct_color(away_pct)
                hw_c = _pct_color(home_pct)
                favor_str = f"← {home}" if hwp >= 0.50 else f"← {away}"
                st.markdown(
                    f"<div style='border-left:3px solid {border};"
                    f"padding:8px 12px;margin-bottom:4px'>"
                    f"<div style='font-size:0.85rem;color:#aaa'>{matchup}</div>"
                    f"<div style='margin-top:6px'>"
                    f"<span style='color:{aw_c};font-size:1.4rem;font-weight:900'>"
                    f"{away_pct:.1f}%</span>"
                    f"<span style='color:#666;font-size:0.9rem'> {away} </span>"
                    f"</div>"
                    f"<div style='margin-top:2px'>"
                    f"<span style='color:{hw_c};font-size:1.4rem;font-weight:900'>"
                    f"{home_pct:.1f}%</span>"
                    f"<span style='color:#666;font-size:0.9rem'> {home} (home) </span>"
                    f"</div>"
                    f"<div style='color:{border};font-size:0.75rem;margin-top:4px'>"
                    f"{favor_str} · {spread*100:.0f}pt lean</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with sp_col:
                home_sp = r["home_sp_name"] or "TBD"
                away_sp = r["away_sp_name"] or "TBD"
                hv = r.get("home_sp_vuln"); av = r.get("away_sp_vuln")
                hv_s = f"vuln {hv:.0f}" if hv is not None else "—"
                av_s = f"vuln {av:.0f}" if av is not None else "—"

                def _sp_qual(v):
                    if v is None: return "#aaa"
                    if v < 30: return "#00ff88"
                    if v < 45: return "#66ddff"
                    if v < 58: return "#ffdd00"
                    return "#ff4444"

                st.markdown(
                    f"<div style='padding:8px 0;font-size:0.8rem'>"
                    f"<div style='color:#aaa;margin-bottom:4px'>Starting Pitchers</div>"
                    f"<div><span style='color:{_sp_qual(av)}'>{away_sp[:22]}</span>"
                    f" <span style='color:#555'>({av_s})</span> {away}</div>"
                    f"<div><span style='color:{_sp_qual(hv)}'>{home_sp[:22]}</span>"
                    f" <span style='color:#555'>({hv_s})</span> {home}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Data flags
                flags = r.get("data_flags", [])
                if flags:
                    for fl in flags:
                        st.caption(f"⚠️ {fl}")
                if not r["bettable"] and r["non_bettable_reasons"]:
                    st.caption(f"Data incomplete: {' · '.join(r['non_bettable_reasons'][:2])}")

            with d_col:
                # Driver breakdown table
                sp_pts  = d.get("sp_edge_pts", 0) * 100
                off_pts = d.get("off_edge_pts", 0) * 100
                bp_pts  = d.get("bp_edge_pts", 0) * 100
                hfa_pts = d.get("hfa", 0.535) * 100

                def _sgn(v): return f"{v:+.1f}%" if v != 0 else "—"
                def _ec(v): return "#00ff88" if v > 0.5 else "#ff4444" if v < -0.5 else "#888"

                st.markdown(
                    f"<div style='font-size:0.78rem;padding:6px 0'>"
                    f"<table style='width:100%;border-collapse:collapse'>"
                    f"<tr><td style='color:#888;padding:1px 4px'>Home field</td>"
                    f"<td style='color:#aaa;padding:1px 4px'>+{hfa_pts:.1f}%</td>"
                    f"<td style='color:#555;font-size:0.7rem'>baseline</td></tr>"
                    f"<tr><td style='color:#888;padding:1px 4px'>SP quality</td>"
                    f"<td style='color:{_ec(sp_pts)};padding:1px 4px'>{_sgn(sp_pts)}</td>"
                    f"<td style='color:#555;font-size:0.7rem'>"
                    f"vuln Δ {d.get('away_sp_vuln',50):.0f}→{d.get('home_sp_vuln',50):.0f}</td></tr>"
                    f"<tr><td style='color:#888;padding:1px 4px'>Offense</td>"
                    f"<td style='color:{_ec(off_pts)};padding:1px 4px'>{_sgn(off_pts)}</td>"
                    f"<td style='color:#555;font-size:0.7rem'>"
                    f"wRC+ {d.get('away_wrc',100):.0f}→{d.get('home_wrc',100):.0f}</td></tr>"
                    f"<tr><td style='color:#888;padding:1px 4px'>Bullpen</td>"
                    f"<td style='color:{_ec(bp_pts)};padding:1px 4px'>{_sgn(bp_pts)}</td>"
                    f"<td style='color:#555;font-size:0.7rem'>"
                    f"vuln {d.get('away_bp_vuln',42):.0f}→{d.get('home_bp_vuln',42):.0f}"
                    f"{'[est]' if r.get('home_bp_estimate') or r.get('away_bp_estimate') else ''}"
                    f"</td></tr>"
                    f"<tr><td style='color:#aaa;padding:3px 4px;border-top:1px solid #333'>"
                    f"Model</td>"
                    f"<td style='color:{_ec(d.get('raw',0.535)*100-53.5)};padding:3px 4px;"
                    f"border-top:1px solid #333;font-weight:bold'>"
                    f"{away} {awp*100:.1f}% / {home} {hwp*100:.1f}%</td>"
                    f"<td style='color:#555;font-size:0.7rem;border-top:1px solid #333'>"
                    f"{'⚠️ clamped' if d.get('clamped') else ''}</td></tr>"
                    f"</table></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<hr style='margin:4px 0;border-color:#1e1e1e'>",
                    unsafe_allow_html=True)
