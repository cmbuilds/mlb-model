"""ui/moneyline_tab.py — Moneyline tab.

Thin Streamlit renderer. Win probability delegated to scoring.moneyline.compute_win_probability.
"""

import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple

from lib.utils import _norm
from scoring.moneyline import compute_win_probability
from scoring.pitcher import compute_pitcher_score


def compute_team_offense_score(plays: List[Dict], team: str) -> Tuple[float, int]:
    """Aggregate wRC+ for a team's confirmed lineup. Returns (avg_wrc_plus, n_batters)."""
    tp = [p for p in plays if p.get("team", "") == team]
    vals = [p.get("wrc_plus", 100.0) for p in tp if p.get("wrc_plus", 100.0) > 0]
    return (round(sum(vals) / len(vals), 1), len(vals)) if vals else (100.0, 0)


def display_moneyline_tab(games: List[Dict], plays: List[Dict],
                          ml_odds: Dict, run_diffs: Dict,
                          implied_totals: Dict,
                          team_bullpen_scores: Dict):
    """
    Tab: 🏦 Moneyline
    Games ranked by ML Confidence Score (0-100).
    Confidence = edge magnitude + SP quality + lineup data + odds availability.
    Strong Edge >7% | Lean 4-7% | No Play <4%.
    """
    # Deferred to avoid circular import at module load time
    from mlb_tb_analyzer import get_pitcher_stats

    st.header("🏦 Moneyline — Win Probability Model")
    st.caption(
        "Log5-style win probability vs Vegas implied odds (vig-removed). "
        "**Edge = model% minus market%.** Strong Edge >7% · Lean 4-7% · No Play <4%. "
        "**Confidence score** ranks games by conviction: edge magnitude + SP quality + "
        "lineup data completeness + odds availability."
    )

    if not games:
        st.info("Run the model first to see moneyline analysis.")
        return

    pitching_df = st.session_state.get("_pitching_df_global", pd.DataFrame())

    # ── Build game rows ──────────────────────────────────────────────────────
    game_rows = []
    for game in games:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        home_sp = game.get("home_pitcher") or "TBD"
        away_sp = game.get("away_pitcher") or "TBD"
        home_sp_id = str(game.get("home_pitcher_id") or "")
        away_sp_id = str(game.get("away_pitcher_id") or "")

        hsr = get_pitcher_stats(home_sp, home_sp_id, pitching_df)
        asr = get_pitcher_stats(away_sp, away_sp_id, pitching_df)

        hbp = team_bullpen_scores.get(home, 42.0)
        abp = team_bullpen_scores.get(away, 42.0)

        hv, _ = compute_pitcher_score(hsr, bullpen_vuln=hbp)
        av, _ = compute_pitcher_score(asr, bullpen_vuln=abp)
        hsr["_sp_vuln"] = hv; asr["_sp_vuln"] = av

        hwrc, hn = compute_team_offense_score(plays, home)
        awrc, an = compute_team_offense_score(plays, away)

        hrd = run_diffs.get(home, 0.0)
        ard = run_diffs.get(away, 0.0)

        hir = implied_totals.get(home, 0.0)
        air = implied_totals.get(away, 0.0)

        hwp, lbl = compute_win_probability(hsr, asr, hwrc, awrc, hbp, abp, hrd, ard, hir, air)
        awp = round(1.0 - hwp, 4)

        # Match ML odds
        ml_key = None
        for k in ml_odds:
            parts = k.split("|")
            if len(parts) == 2:
                if (_norm(parts[0]) in _norm(game.get("away_team_name", away)) or
                    _norm(away) in _norm(parts[0])):
                    ml_key = k; break
                if (_norm(parts[1]) in _norm(game.get("home_team_name", home)) or
                    _norm(home) in _norm(parts[1])):
                    ml_key = k; break

        if ml_key and ml_key in ml_odds:
            mkt = ml_odds[ml_key]
            hmkt = mkt["home_implied"]; amkt = mkt["away_implied"]
            ho_raw = mkt["home_odds"]; ao_raw = mkt["away_odds"]
            hos = f"{ho_raw:+.0f}"; aos = f"{ao_raw:+.0f}"
            has_odds = True
        else:
            hmkt = amkt = None; ho_raw = ao_raw = None
            hos = aos = "N/A"; has_odds = False

        hedge = round((hwp - hmkt) * 100, 1) if hmkt is not None else None
        aedge = round((awp - amkt) * 100, 1) if amkt is not None else None

        # ── Confidence score ─────────────────────────────────────────────────
        def _conf(side_edge, sp_name, n_bat, sp_vuln, has_o):
            if side_edge is None or side_edge < 0:
                return 0.0
            e_pts  = min(50.0, side_edge / 7.0 * 50.0)
            sp_pts = max(0.0, min(20.0, (60.0 - sp_vuln) / 60.0 * 20.0))
            l_pts  = min(15.0, n_bat / 9.0 * 15.0)
            o_pts  = 10.0 if has_o else 0.0
            s_pts  = 5.0 if sp_name not in ("TBD", "", None) else 0.0
            return round(min(100.0, e_pts + sp_pts + l_pts + o_pts + s_pts), 1)

        hconf = _conf(hedge, home_sp, hn, hsr.get("_sp_vuln", 50), has_odds)
        aconf = _conf(aedge, away_sp, an, asr.get("_sp_vuln", 50), has_odds)

        # Best pick
        if hedge is not None and aedge is not None:
            if hedge >= 4 and hedge >= aedge:
                pick = home; pedge = hedge; pconf = hconf
                pwp = hwp; pods = hos; pmkt = hmkt
                psp = home_sp; osp = away_sp; pop_vuln = hv
            elif aedge >= 4 and aedge > hedge:
                pick = away; pedge = aedge; pconf = aconf
                pwp = awp; pods = aos; pmkt = amkt
                psp = away_sp; osp = home_sp; pop_vuln = av
            else:
                pick = None; pedge = max(hedge or 0, aedge or 0)
                pconf = 0; pwp = None; pods = "—"; pmkt = None
                psp = "—"; osp = "—"; pop_vuln = 50
        else:
            pick = None; pedge = 0; pconf = 0
            pwp = None; pods = "—"; pmkt = None
            psp = "—"; osp = "—"; pop_vuln = 50

        if pick and pedge >= 7:
            ptier = "🔥 Strong Edge"
        elif pick and pedge >= 4:
            ptier = "📊 Lean"
        else:
            ptier = "➖ No Play"

        game_rows.append({
            "matchup": f"{away} @ {home}", "away": away, "home": home,
            "home_sp": home_sp, "away_sp": away_sp,
            "hwp": hwp, "awp": awp, "hmkt": hmkt, "amkt": amkt,
            "hedge": hedge, "aedge": aedge,
            "hos": hos, "aos": aos, "ho_raw": ho_raw, "ao_raw": ao_raw,
            "hconf": hconf, "aconf": aconf,
            "pick": pick, "pedge": pedge, "pconf": pconf,
            "ptier": ptier, "pwp": pwp, "pods": pods, "pmkt": pmkt,
            "psp": psp, "osp": osp,
            "hn": hn, "an": an, "hrd": hrd, "ard": ard,
            "hwrc": hwrc, "awrc": awrc,
            "hv": hv, "av": av, "hbp": hbp, "abp": abp,
            "has_odds": has_odds, "lbl": lbl,
        })

    if not game_rows:
        st.warning("No games to display.")
        return

    game_rows.sort(key=lambda x: (x["pconf"], x["pedge"] or 0), reverse=True)

    # ── Summary metrics ──────────────────────────────────────────────────────
    strong = [r for r in game_rows if r["ptier"] == "🔥 Strong Edge"]
    lean   = [r for r in game_rows if r["ptier"] == "📊 Lean"]
    nop    = [r for r in game_rows if r["ptier"] == "➖ No Play"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Games",              len(game_rows))
    m2.metric("🔥 Strong Edge",    len(strong))
    m3.metric("📊 Lean",           len(lean))
    m4.metric("➖ No Play",        len(nop))

    if not any(r["has_odds"] for r in game_rows):
        st.warning("⚠️ No Odds API key — edge = N/A. Add key in sidebar for full model.")

    # ── Ranked pick cards ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🏆 ML Picks — Ranked by Confidence")
    st.caption("Games ranked by Confidence Score (0-100). "
               "Confidence = edge magnitude (50%) + SP quality (20%) + "
               "lineup data (15%) + odds loaded (10%) + SP known (5%).")

    for rank, r in enumerate(game_rows, 1):
        tier = r["ptier"]; conf = r["pconf"]
        if "Strong" in tier:
            border = "#00ff88"; rank_col = "#00ff88"
        elif "Lean" in tier:
            border = "#ffdd00"; rank_col = "#ffdd00"
        else:
            border = "#444444"; rank_col = "#666666"

        with st.container():
            c_rank, c_pick, c_probs, c_conf = st.columns([0.5, 3.0, 4.0, 1.5])

            with c_rank:
                st.markdown(
                    f"<div style='text-align:center;margin-top:8px'>"
                    f"<div style='font-size:1.6rem;font-weight:900;color:{rank_col}'>{rank}</div>"
                    f"<div style='font-size:0.6rem;color:#666'>RANK</div></div>",
                    unsafe_allow_html=True)

            with c_pick:
                st.markdown(f"**{r['matchup']}**")
                if r["pick"]:
                    e_str = f"+{r['pedge']:.1f}%" if r["pedge"] > 0 else f"{r['pedge']:.1f}%"
                    odds_str = r["pods"] if r["pods"] != "—" else "N/A (no odds loaded)"
                    st.markdown(
                        f"<span style='color:{border};font-weight:bold;font-size:1.05rem'>"
                        f"✅ {r['pick']} {odds_str}</span> &nbsp;"
                        f"<span style='color:{border}'>Edge: {e_str}</span>",
                        unsafe_allow_html=True)
                    st.caption(f"SP: {r['psp'][:22]}  ·  Opp SP: {r['osp'][:22]}")
                else:
                    st.markdown("<span style='color:#666'>➖ No Play — edge below threshold</span>",
                                unsafe_allow_html=True)
                    st.caption(f"Home SP: {r['home_sp']}  ·  Away SP: {r['away_sp']}")
                tier_c = "#00ff88" if "Strong" in tier else "#ffdd00" if "Lean" in tier else "#666"
                st.markdown(f"<span style='background:#111;border:1px solid {tier_c};"
                            f"color:{tier_c};border-radius:5px;padding:1px 8px;"
                            f"font-size:0.75rem'>{tier}</span>", unsafe_allow_html=True)

            with c_probs:
                h = r["home"]; a = r["away"]
                hm_s = f"{r['hmkt']*100:.1f}%" if r["hmkt"] else "N/A"
                am_s = f"{r['amkt']*100:.1f}%" if r["amkt"] else "N/A"
                he_s = f"{r['hedge']:+.1f}%" if r["hedge"] is not None else "N/A"
                ae_s = f"{r['aedge']:+.1f}%" if r["aedge"] is not None else "N/A"
                st.caption(
                    f"**{h}**: Model {r['hwp']*100:.1f}% · Mkt {hm_s} · Edge {he_s} · "
                    f"SP vuln {r['hv']:.0f} · wRC+ {r['hwrc']:.0f} · {r['hn']}bat")
                st.caption(
                    f"**{a}**: Model {r['awp']*100:.1f}% · Mkt {am_s} · Edge {ae_s} · "
                    f"SP vuln {r['av']:.0f} · wRC+ {r['awrc']:.0f} · {r['an']}bat")
                st.caption(
                    f"7d RunDiff: {h} {r['hrd']:+.1f} / {a} {r['ard']:+.1f} · "
                    f"Bullpen vuln: {h} {r['hbp']:.0f} / {a} {r['abp']:.0f}")

            with c_conf:
                cc = "#00ff88" if conf >= 60 else "#ffdd00" if conf >= 35 else "#666"
                st.markdown(
                    f"<div style='text-align:center;background:#1a1a2e;"
                    f"border:2px solid {cc};border-radius:10px;padding:10px 4px'>"
                    f"<div style='font-size:1.5rem;font-weight:900;color:{cc}'>{conf:.0f}</div>"
                    f"<div style='font-size:0.6rem;color:#aaa'>CONF</div></div>",
                    unsafe_allow_html=True)

        st.markdown("<hr style='margin:6px 0;border-color:#2a2a2a'>", unsafe_allow_html=True)

    # ── Full table ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Full Games Table")

    tbl = []
    for r in game_rows:
        tbl.append({
            "Rank":     game_rows.index(r) + 1,
            "Matchup":  r["matchup"],
            "Pick":     f"{r['pick']} ({r['ptier']})" if r["pick"] else "➖ No Play",
            "Conf":     r["pconf"],
            "Away Mdl": f"{r['awp']*100:.1f}%",
            "Home Mdl": f"{r['hwp']*100:.1f}%",
            "Away Mkt": f"{r['amkt']*100:.1f}%" if r["amkt"] else "N/A",
            "Home Mkt": f"{r['hmkt']*100:.1f}%" if r["hmkt"] else "N/A",
            "Away Odds":r["aos"], "Home Odds": r["hos"],
            "Away Edge":f"{r['aedge']:+.1f}%" if r["aedge"] is not None else "N/A",
            "Home Edge":f"{r['hedge']:+.1f}%" if r["hedge"] is not None else "N/A",
            "Away SP":  r["away_sp"][:18], "Home SP": r["home_sp"][:18],
        })

    df_ml = pd.DataFrame(tbl)

    def _cp(val):
        s = str(val)
        if "Strong" in s: return "color:#00ff88;font-weight:bold"
        if "Lean" in s:   return "color:#ffdd00"
        return "color:#888888"

    def _ce(val):
        try:
            v = float(str(val).replace("%","").replace("+",""))
            if v >= 7:  return "color:#00ff88;font-weight:bold"
            if v >= 4:  return "color:#ffdd00"
            if v <= -4: return "color:#ff4444"
        except Exception: pass
        return ""

    def _cc(val):
        try:
            v = float(val)
            if v >= 60: return "color:#00ff88;font-weight:bold"
            if v >= 35: return "color:#ffdd00"
        except Exception: pass
        return "color:#888888"

    ecols = [c for c in df_ml.columns if "Edge" in c]
    s = df_ml.style.map(_cp, subset=["Pick"]).map(_cc, subset=["Conf"])
    for ec in ecols:
        s = s.map(_ce, subset=[ec])
    st.dataframe(s, use_container_width=True)
    csv = df_ml.to_csv(index=False)
    st.download_button("📥 Export Moneyline CSV", csv, "moneyline.csv", "text/csv", key="dl_ml")

    if not any(r["has_odds"] for r in game_rows):
        st.warning("⚠️ Add Odds API key in sidebar for edge calculation.")
