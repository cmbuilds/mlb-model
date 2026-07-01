"""
dfs/ui/dfs_tabs.py — platform-agnostic DFS board + FD builder UI.

Thin Streamlit renderer — no scoring math here.
Three logical sections:
  1. Consensus Value Board (both sites)
  2. Stack Command Center
  3. FD Lineup Builder (auto-build from CONFIDENT pool)
  DK board is rendered alongside FD board — DK auto-build is D1.

Provenance display rules (non-negotiable):
  - FLAGGED rows are greyed with reason shown.
  - Ownership always shows "(modeled)" suffix until D2 external source.
  - Source count shown per row ("1 src" tonight).
  - Data freshness warning if plays are stale.
"""

from __future__ import annotations

import io
import os
import tempfile
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import pytz
import streamlit as st

from dfs.contracts import ConsensusRow, ConfidenceState, Provenance
from dfs.consensus import build_consensus_board, compute_stack_scores

EST = pytz.timezone("US/Eastern")

# Freshness threshold: warn if model data is older than this
STALE_HOURS = 2


# ─── main entry point ─────────────────────────────────────────────────────────
def display_dfs_tabs(plays: List[Dict]):
    """
    Render the DFS section. Called from app.py with the current plays list.
    plays must be the output of run_model() (list of per-player score dicts).
    """
    if not plays:
        st.info("No model output yet. Run the model first, then return here.")
        return

    st.title("🎯 DFS Command Center")

    # ── Freshness guard ───────────────────────────────────────────────────────
    _freshness_warning(plays)

    # ── Salary CSV upload (both sites) ────────────────────────────────────────
    st.markdown("### 💰 Load Salaries")
    sal_col1, sal_col2 = st.columns(2)
    with sal_col1:
        fd_file = st.file_uploader(
            "FanDuel salary CSV", type=["csv"], key="fd_salary_csv",
            help="Download from FanDuel lineup tool → Export CSV"
        )
    with sal_col2:
        dk_file = st.file_uploader(
            "DraftKings salary CSV", type=["csv"], key="dk_salary_csv",
            help="Download from DraftKings lineup tool → Export CSV"
        )

    fd_salaries = _load_salary_csv(fd_file, site="fd")
    dk_salaries = _load_salary_csv(dk_file, site="dk")

    # ── Build consensus boards ────────────────────────────────────────────────
    try:
        fd_board = build_consensus_board(plays, site="fd")
    except Exception as e:
        st.error(f"FD consensus failed: {e}")
        fd_board = []

    try:
        dk_board = build_consensus_board(plays, site="dk")
    except Exception as e:
        st.error(f"DK consensus failed: {e}")
        dk_board = []

    # Merge salaries + augment with pitchers from salary CSV
    if fd_salaries and fd_board:
        from dfs.sources.salaries import merge_salaries_into_board, pitchers_from_salary_csv
        fd_board, fd_matched = merge_salaries_into_board(fd_board, fd_salaries)
        if fd_matched < len(fd_board) // 2:
            st.warning(f"⚠️ FD salary match rate low: {fd_matched}/{len(fd_board)} players matched by name")
        # Add pitchers from salary CSV — model produces batter-only plays
        fd_pitchers = pitchers_from_salary_csv(fd_salaries, site="fd")
        if fd_pitchers:
            fd_board = fd_board + fd_pitchers
            n_conf_p = sum(1 for p in fd_pitchers if p.state.value == "CONFIDENT")
            st.caption(f"📋 {len(fd_pitchers)} pitcher(s) loaded from FD salary CSV ({n_conf_p} CONFIDENT via site FPPG)")

    if dk_salaries and dk_board:
        from dfs.sources.salaries import merge_salaries_into_board, pitchers_from_salary_csv
        dk_board, dk_matched = merge_salaries_into_board(dk_board, dk_salaries)
        if dk_matched < len(dk_board) // 2:
            st.warning(f"⚠️ DK salary match rate low: {dk_matched}/{len(dk_board)} players matched by name")
        dk_pitchers = pitchers_from_salary_csv(dk_salaries, site="dk")
        if dk_pitchers:
            dk_board = dk_board + dk_pitchers
            n_conf_p = sum(1 for p in dk_pitchers if p.state.value == "CONFIDENT")
            st.caption(f"📋 {len(dk_pitchers)} pitcher(s) loaded from DK salary CSV ({n_conf_p} CONFIDENT via site FPPG)")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_fd, tab_dk, tab_stacks, tab_build = st.tabs([
        "🟣 FanDuel Board",
        "🟢 DraftKings Board",
        "🔥 Stack Ranks",
        "🏗️ FD Lineup Builder",
    ])

    with tab_fd:
        _render_board(fd_board, site="fd", has_salaries=bool(fd_salaries))

    with tab_dk:
        _render_dk_board(dk_board, has_salaries=bool(dk_salaries))

    with tab_stacks:
        _render_stacks(fd_board or dk_board, plays)

    with tab_build:
        _render_fd_builder(fd_board, plays)


# ─── Value Board ──────────────────────────────────────────────────────────────
def _render_board(board: List[ConsensusRow], site: str, has_salaries: bool):
    site_label = "FanDuel" if site == "fd" else "DraftKings"
    cap = 35000 if site == "fd" else 50000
    currency_label = f"${cap//1000}K cap"

    if not board:
        st.info(f"No {site_label} projections yet. Run the model first.")
        return

    n_conf  = sum(1 for r in board if r.state == ConfidenceState.CONFIDENT)
    n_flag  = sum(1 for r in board if r.state == ConfidenceState.FLAGGED)
    n_excl  = sum(1 for r in board if r.state == ConfidenceState.EXCLUDED)
    n_total = len(board)

    _state_banner(n_conf, n_flag, n_excl, n_total, has_salaries)

    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        show_flagged = st.checkbox("Show FLAGGED players", value=True, key=f"{site}_show_flagged")
    with f2:
        pos_filter = st.multiselect(
            "Position", sorted(set(r.position for r in board)),
            default=[], key=f"{site}_pos_filter"
        )
    with f3:
        min_pts = st.slider("Min projected pts", 0.0, 50.0, 8.0, 0.5, key=f"{site}_min_pts")

    filtered = board
    if not show_flagged:
        filtered = [r for r in filtered if r.state != ConfidenceState.FLAGGED]
    if pos_filter:
        filtered = [r for r in filtered if r.position in pos_filter]
    filtered = [r for r in filtered if r.consensus_pts >= min_pts]

    # Build table
    rows = []
    for r in filtered:
        state_icon = {"CONFIDENT": "✅", "FLAGGED": "⚠️", "EXCLUDED": "❌"}.get(r.state.value, "?")
        value_str  = f"{r.consensus_value:.2f}" if r.salary >= 1000 else "—"
        own_str    = f"{r.own_pct:.1f}% (modeled)"
        src_str    = f"{r.source_count} src"
        flag_str   = f" ← {r.flagged_reason}" if r.flagged_reason else ""

        rows.append({
            "State":    f"{state_icon} {r.state.value}{flag_str}",
            "Player":   r.name,
            "Pos":      r.position,
            "Team":     r.team,
            "Vs":       r.opponent,
            "Salary":   f"${r.salary:,}" if r.salary > 0 else "—",
            "Proj Pts": f"{r.consensus_pts:.1f}",
            "Ceiling":  f"{r.model_ceiling:.1f}",
            "Floor":    f"{r.model_floor:.1f}",
            "Value":    value_str,
            "Own%":     own_str,
            "Slot":     f"#{r.lineup_slot}" if r.lineup_slot else "?",
            "SP":       r.sp_name[:18],
            "TB Score": f"{r.score:.0f}",
            "DQ":       f"{r.dq_score}%",
            "Sources":  src_str,
        })

    if rows:
        df = pd.DataFrame(rows)

        def _state_color(val):
            v = str(val)
            if "CONFIDENT" in v or "✅" in v: return "color: #00ff88; font-weight: bold"
            if "FLAGGED"   in v or "⚠️" in v: return "color: #888888; font-style: italic"
            if "EXCLUDED"  in v or "❌" in v: return "color: #555555"
            return ""

        def _pts_color(val):
            try:
                v = float(str(val))
                if v >= 30:  return "color: #00ff88; font-weight: bold"
                if v >= 22:  return "color: #ffdd00; font-weight: bold"
                if v >= 15:  return "color: #ff8800"
                return "color: #888888"
            except Exception: return ""

        def _val_color(val):
            try:
                v = float(str(val))
                if v >= 4.0: return "color: #00ff88; font-weight: bold"
                if v >= 3.0: return "color: #ffdd00"
                if v >= 2.0: return "color: #ff8800"
                return "color: #888888"
            except Exception: return ""

        styled = (df.style
                  .map(_state_color, subset=["State"])
                  .map(_pts_color,   subset=["Proj Pts", "Ceiling"])
                  .map(_val_color,   subset=["Value"]))
        st.dataframe(styled, use_container_width=True, height=520)

        # CSV export
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button(
            f"📥 Export {site_label} Board CSV", csv_bytes,
            f"dfs_{site}_board_{datetime.now(EST).strftime('%Y%m%d_%H%M')}.csv",
            "text/csv", key=f"{site}_export_csv",
        )
    else:
        st.info("No players match the current filters.")


def _render_dk_board(board: List[ConsensusRow], has_salaries: bool):
    """DK board — same as FD board but with board-only banner."""
    st.info(
        "🟢 **DraftKings — Board Only Tonight.** "
        "Use this board to manually build your Battery ($40K) and 10×$18 entries. "
        "DK auto-optimizer activates in D1 after validation."
    )
    _render_board(board, site="dk", has_salaries=has_salaries)


# ─── Stack Center ─────────────────────────────────────────────────────────────
def _render_stacks(board: List[ConsensusRow], plays: List[Dict]):
    st.markdown("### 🔥 Stack Ranks — by consensus ceiling")
    st.caption(
        "Team stack score = sum of top-4 hitters' model ceiling + implied-total bonus. "
        "Ownership is modeled — use as a leverage signal, not field truth."
    )

    if not board:
        st.info("No consensus data yet.")
        return

    stack_scores = compute_stack_scores(board)

    rows = []
    for team, score in list(stack_scores.items())[:15]:
        team_rows = sorted(
            [r for r in board if r.team == team and r.state != ConfidenceState.EXCLUDED],
            key=lambda r: r.consensus_pts, reverse=True
        )
        top4 = team_rows[:4]
        implied = next((r.implied_total for r in team_rows if r.implied_total > 0), 0.0)
        park    = next((r.park for r in team_rows), "")
        conf_pct = round(sum(1 for r in top4 if r.state == ConfidenceState.CONFIDENT) / max(len(top4), 1) * 100)
        own_avg  = round(sum(r.own_pct for r in top4) / max(len(top4), 1), 1)

        rows.append({
            "Team":        team,
            "Stack Score": f"{score:.1f}",
            "Imp. Total":  f"{implied:.1f}" if implied else "—",
            "Park":        park,
            "Top Hitters": " · ".join(f"{r.name} ({r.consensus_pts:.1f})" for r in top4),
            "CONF%":       f"{conf_pct}%",
            "Own% (mdl)":  f"{own_avg:.1f}%",
        })

    if rows:
        df = pd.DataFrame(rows)

        def _stack_color(val):
            try:
                v = float(str(val))
                if v >= 80: return "color: #00ff88; font-weight: bold"
                if v >= 60: return "color: #ffdd00"
                return "color: #ff8800"
            except Exception: return ""

        styled = df.style.map(_stack_color, subset=["Stack Score"])
        st.dataframe(styled, use_container_width=True, height=380)
    else:
        st.info("No stack data.")


# ─── FD Lineup Builder ────────────────────────────────────────────────────────
def _render_fd_builder(board: List[ConsensusRow], plays: List[Dict]):
    st.markdown("### 🏗️ FanDuel Lineup Builder")
    st.caption(
        "Auto-builds from the CONFIDENT pool only. FLAGGED players appear in the "
        "board above but are excluded here — manually add them via lineup tool if desired."
    )

    if not board:
        st.info("No FD consensus board yet. Upload FD salary CSV and run the model.")
        return

    n_conf = sum(1 for r in board if r.state == ConfidenceState.CONFIDENT and r.salary >= 1000)
    st.metric("CONFIDENT players with salary", n_conf, help="Minimum 10 needed to build")

    if n_conf < 10:
        st.warning(
            f"⚠️ Only {n_conf} CONFIDENT players have salaries loaded. "
            "Upload the FD salary CSV to enable auto-build."
        )
        return

    # Contest type
    contest_type = st.radio(
        "Contest type",
        ["Single entry (balanced, $40K Battery equiv.)", "Multi-entry GPP (Deuces Wild, $300K)"],
        key="fd_contest_type",
    )

    # Optional manual overrides
    with st.expander("⚙️ Manual overrides"):
        all_names = sorted(r.name for r in board if r.state == ConfidenceState.CONFIDENT and r.salary >= 1000)
        locked_names  = st.multiselect("Lock players", all_names, default=[], key="fd_locked")
        stack_teams   = sorted(set(r.team for r in board if r.state == ConfidenceState.CONFIDENT))
        stack_team    = st.selectbox("Force stack team", ["(auto)"] + stack_teams, key="fd_stack_team")
        stack_team    = None if stack_team == "(auto)" else stack_team

    col_build, _ = st.columns([1, 3])
    with col_build:
        build_btn = st.button("⚡ Build Lineups", type="primary", key="fd_build_btn")

    if build_btn:
        from dfs.optimize import (
            build_fd_lineups, export_fd_csv,
            CONTEST_SINGLE_ENTRY, CONTEST_MULTI_ENTRY_GPP,
        )
        contest = (CONTEST_MULTI_ENTRY_GPP
                   if "Multi" in contest_type else CONTEST_SINGLE_ENTRY)

        with st.spinner(f"Building {contest.n_lineups} lineup(s)…"):
            try:
                lineups = build_fd_lineups(
                    board=board,
                    contest=contest,
                    locked_names=locked_names or None,
                    stack_team=stack_team,
                )
            except Exception as e:
                st.error(f"❌ Build failed: {e}")
                return

        st.success(f"✅ {len(lineups)} lineup(s) built")

        for i, lu in enumerate(lineups, 1):
            with st.expander(f"Lineup {i} — {lu['total_proj']:.1f} pts projected (ceil: {lu.get('total_ceiling', '?')})"):
                lu_df = pd.DataFrame(lu["players"])
                st.dataframe(lu_df, use_container_width=True)
                st.caption(f"Salary used: ${lu['total_salary']:,} / $35,000")

        # Bulk export
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            export_fd_csv(lineups, tmp.name)
            with open(tmp.name, "rb") as f:
                csv_bytes = f.read()
        os.unlink(tmp.name)

        st.download_button(
            "📥 Download FD bulk-import CSV", csv_bytes,
            f"fd_lineups_{datetime.now(EST).strftime('%Y%m%d_%H%M')}.csv",
            "text/csv", key="fd_download_csv",
        )


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _state_banner(n_conf, n_flag, n_excl, n_total, has_salaries):
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("✅ CONFIDENT", n_conf)
    with c2: st.metric("⚠️ FLAGGED", n_flag)
    with c3: st.metric("❌ EXCLUDED", n_excl)
    with c4: st.metric("Total Batters", n_total)

    if not has_salaries:
        st.info(
            "💡 Upload a salary CSV above to see value (pts/salary) and enable auto-build. "
            "Board is usable without it — download and manually build."
        )

    if n_conf == 0:
        st.error(
            "❌ No CONFIDENT players. Every player is on proxy/unconfirmed data — "
            "DO NOT auto-build. Check lineup confirmation status and re-run the model."
        )
    elif n_conf < n_total * 0.5:
        st.warning(
            f"⚠️ {n_flag} players FLAGGED (proxy data or unconfirmed lineup). "
            "Only the ✅ CONFIDENT pool is eligible for auto-build."
        )


def _load_salary_csv(uploaded_file, site: str) -> Optional[List[Dict]]:
    if uploaded_file is None:
        return None
    content = uploaded_file.read().decode("utf-8", errors="replace")
    try:
        if site == "fd":
            from dfs.sources.salaries import parse_fd_salary_csv
            return parse_fd_salary_csv(content)
        else:
            from dfs.sources.salaries import parse_dk_salary_csv
            return parse_dk_salary_csv(content)
    except Exception as e:
        st.error(f"❌ Failed to parse {site.upper()} salary CSV: {e}")
        return None


def _freshness_warning(plays: List[Dict]):
    """Warn if model data looks stale (no game time available to check, use session state)."""
    import streamlit as st
    run_ts = st.session_state.get("_model_run_ts")
    if run_ts is None:
        st.warning(
            "⏰ Model run timestamp unknown. Re-run the model to ensure fresh projections "
            "before lock. Stale data = stale projections."
        )
        return
    try:
        now_utc  = datetime.now(timezone.utc)
        run_dt   = datetime.fromisoformat(run_ts).replace(tzinfo=timezone.utc)
        age_hrs  = (now_utc - run_dt).total_seconds() / 3600
        if age_hrs > STALE_HOURS:
            st.warning(
                f"⏰ **Model data is {age_hrs:.1f}h old** — rerun before lock "
                f"(>{STALE_HOURS}h threshold). Lineups and SPs may have changed."
            )
    except Exception:
        pass
