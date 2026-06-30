"""
backtest.py — Calibration and ROI report for the O1.5 TB model.

Prerequisites:
  1. Run fetch_pipeline.py to populate data/mlb_stats.db with season stats.
  2. Run data/backtest_fetcher.py to populate the game_results table with
     actual per-game batter outcomes.

Usage:
    python3 backtest.py [--season 2026] [--market tb_o15] [--min-ab 3]

What this does:
  - Loads batter season stats from SQLite
  - Loads actual game results from game_results table
  - Scores each historical batter-game using markets/tb_o15.score_one_batter
    (using current season stats as a proxy for day-of stats — aware this
    introduces selection bias; the backtest still measures calibration)
  - Produces calibration report: actual hit rate per score bucket
  - Produces ROI report at standard −115 vig (when prop odds present)

Limitations tracked in context_05_overhaul_changelog.md:
  - Using current season stats for all historical dates (snapshot bias)
  - No historical prop odds until Odds API Business tier
"""

import argparse
import logging
import sqlite3
import sys
from collections import defaultdict
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH = "data/mlb_stats.db"
# Market-specific breakeven probabilities at standard prop lines
VIG_BY_MARKET = {
    "tb_o15":   0.535,   # -115 American odds
    "hits_o05": 0.636,   # -175 American odds (standard O0.5 line)
    "hr":       0.222,   # +350 American odds (representative qualifying HR line)
}


def _load_batter_stats(conn: sqlite3.Connection, season: int) -> Dict[int, Dict]:
    """Load season batter stats keyed by player_id (mlbam_id)."""
    rows = conn.execute(
        "SELECT * FROM batter_stats WHERE fetch_season = ?", (season,)
    ).fetchall()
    if not rows:
        log.warning(f"No batter stats for season {season} in DB — run fetch_pipeline.py first")
        return {}

    cols = [d[0] for d in conn.execute("SELECT * FROM batter_stats LIMIT 0").description]
    result: Dict[int, Dict] = {}
    for row in rows:
        d = dict(zip(cols, row))
        pid = d.get("mlbam_id")
        if pid:
            result[int(pid)] = d
    log.info(f"Loaded {len(result)} batter stat rows (season {season})")
    return result


def _load_game_results(conn: sqlite3.Connection, min_ab: int = 3) -> List[Dict]:
    """Load historical game results (only qualifying plate appearances)."""
    rows = conn.execute(
        "SELECT * FROM game_results WHERE ab >= ?", (min_ab,)
    ).fetchall()
    if not rows:
        log.warning("No game_results found — run data/backtest_fetcher.py first")
        return []
    cols = [d[0] for d in conn.execute("SELECT * FROM game_results LIMIT 0").description]
    result = [dict(zip(cols, r)) for r in rows]
    log.info(f"Loaded {len(result)} game-batter rows (min_ab={min_ab})")
    return result


def _stat_row_to_scoring_input(db_row: Dict) -> Dict:
    """Convert a batter_stats DB row to the format expected by compute_batter_score."""
    def f(key, default):
        v = db_row.get(key)
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    return {
        "slg_proxy":         f("xSLG",              0.398),
        "barrel_rate":       f("barrel_batted_rate", 0.070),
        "hard_hit_rate":     f("hard_hit_percent",   0.370),
        "k_rate":            f("K%",                 0.228),
        "iso_proxy":         f("ISO",                0.165),
        "wrc_plus":          f("wRC+",               100.0),
        "woba":              f("xwOBA",              0.315),
        "ev50":              f("ev50",               0.0),
        "bat_speed":         f("bat_speed",          0.0),
        "blast_rate":        f("blast_rate",         0.0),
        "sweet_spot_rate":   f("sweet_spot_percent", 0.30),
        "exit_velocity_avg": f("avg_exit_velocity",  88.5),
        "data_source":     "sqlite_batter_stats",
        "_provenance": {
            "k_rate":        db_row.get("prov_krate",    "league_avg"),
            "slg_proxy":     db_row.get("prov_slg",      "league_avg"),
            "xslg":          db_row.get("prov_xslg",     "league_avg"),
            "xwoba":         db_row.get("prov_xwoba",    "league_avg"),
            "woba":          db_row.get("prov_woba",     "league_avg"),
            "hard_hit_rate": db_row.get("prov_hh",       "league_avg"),
            "barrel_rate":   db_row.get("prov_barrel",   "league_avg"),
            "iso_proxy":     db_row.get("prov_iso",      "league_avg"),
        },
    }


def _load_game_pitchers(conn: sqlite3.Connection) -> Dict[int, Dict]:
    """Load game_pitchers table as dict keyed by game_pk."""
    try:
        rows = conn.execute("SELECT * FROM game_pitchers").fetchall()
        cols = [d[0] for d in conn.execute("SELECT * FROM game_pitchers LIMIT 0").description]
        return {int(r[cols.index("game_pk")]): dict(zip(cols, r)) for r in rows}
    except Exception:
        return {}


def _load_pitcher_stats(conn: sqlite3.Connection, season: int) -> Dict[int, Dict]:
    """Load pitcher_stats DB rows keyed by mlbam_id."""
    try:
        rows = conn.execute(
            "SELECT * FROM pitcher_stats WHERE fetch_season=?", (season,)
        ).fetchall()
        if not rows:
            rows = conn.execute("SELECT * FROM pitcher_stats").fetchall()
        cols = [d[0] for d in conn.execute("SELECT * FROM pitcher_stats LIMIT 0").description]
        result: Dict[int, Dict] = {}
        for r in rows:
            d = dict(zip(cols, r))
            pid = d.get("mlbam_id")
            if pid:
                result[int(pid)] = d
        return result
    except Exception:
        return {}


def _pitcher_row_to_input(db_row: Dict) -> Dict:
    """Convert pitcher_stats DB row to the format expected by compute_pitcher_score."""
    def f(key, default):
        v = db_row.get(key)
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    k_pct = f("K%", 0.228)
    if k_pct > 1.0:
        k_pct /= 100.0

    return {
        "k_rate_allowed":   k_pct,
        "hard_hit_allowed": f("Hard%", 0.370) if f("Hard%", 0.370) <= 1.0 else f("Hard%", 0.370) / 100.0,
        "barrel_allowed":   f("Barrel%", 0.070) if f("Barrel%", 0.070) <= 1.0 else f("Barrel%", 0.070) / 100.0,
        "era":              f("ERA",  4.20),
        "fip":              f("FIP",  4.20),
        "whip":             f("WHIP", 1.30),
        "data_source":      "sqlite_pitcher_stats",
        "_provenance": {
            "k_rate_allowed":   "measured" if db_row.get("K%") is not None else "league_avg",
            "hard_hit_allowed": "measured" if db_row.get("Hard%") is not None else "league_avg",
        },
    }


_LEAGUE_AVG_PITCHER = {
    "k_rate_allowed": 0.228, "hard_hit_allowed": 0.370,
    "barrel_allowed": 0.070, "era": 4.20, "fip": 4.20, "whip": 1.30,
    "data_source": "league_avg",
    "_provenance": {"k_rate_allowed": "league_avg", "hard_hit_allowed": "league_avg"},
}


def _score_one(market: str, result: Dict, batter_input: Dict, pitcher_input: Dict,
               sp_id_used: str, proxy_mode: bool) -> Optional[Dict]:
    """Score a single player-game for the given market. Returns None on failure."""
    common = dict(
        name=str(result.get("player_name", "")),
        player_id=str(result.get("player_id", "")),
        team=str(result.get("team", "")),
        opponent=str(result.get("opponent", "")),
        game_pk=str(result.get("game_pk", 0)),
        batter_hand="R",
        hand_real=False,
        sp_hand="R",
        sp_name="",
        sp_id=sp_id_used,
        lineup_slot=int(result.get("lineup_slot", 5) or 5),
        lineup_confirmed=True,
        batter_position="",
        park_team=str(result.get("team", "")),
        batter_stats=batter_input,
        pitcher_stats=pitcher_input,
        proxy_mode=proxy_mode,
    )
    try:
        if market == "tb_o15":
            from markets.tb_o15 import score_one_batter
            return score_one_batter(
                **common,
                recent_form={},
                bvp_data={},
                weather={"is_dome": True},
                implied=0.0,
                prop_implied=None,
                team_bullpen_scores={},
            )
        elif market == "hits_o05":
            from markets.hits_o05 import score_one_batter_o05
            return score_one_batter_o05(
                **common,
                recent_form={},
                implied=0.0,
                prop_implied=None,
                team_bullpen_scores={},
            )
        elif market == "hr":
            from markets.hr import score_one_batter_hr
            return score_one_batter_hr(
                **common,
                weather={"is_dome": True},
                implied=0.0,
                prop_implied=None,
                team_bullpen_scores={},
            )
    except Exception as e:
        log.debug(f"Scoring failed ({market}): {e}")
    return None


def run_backtest(
    season: int = 2026,
    market: str = "tb_o15",
    min_ab: int = 3,
) -> Dict:
    conn = sqlite3.connect(DB_PATH)
    batter_stats   = _load_batter_stats(conn, season)
    game_results   = _load_game_results(conn, min_ab=min_ab)
    game_pitchers  = _load_game_pitchers(conn)
    pitcher_stats  = _load_pitcher_stats(conn, season)
    conn.close()

    if not batter_stats or not game_results:
        return {}

    # Outcome column per market
    outcome_col = {"tb_o15": "hit_o15", "hits_o05": "hit_o05", "hr": "hit_hr"}.get(market, "hit_o15")

    sp_matched = 0
    sp_games_available = len(game_pitchers)
    log.info(f"SP data available for {sp_games_available}/{len(set(r['game_pk'] for r in game_results))} games")

    buckets = {"<50": [], "50-59": [], "60-69": [], "70-79": [], "80+": []}

    scored = 0
    unmatched = 0
    proxy_count = 0

    for result in game_results:
        pid = result.get("player_id")
        if not pid:
            continue

        stat_row = batter_stats.get(int(pid))
        if not stat_row:
            unmatched += 1
            continue

        batter_input = _stat_row_to_scoring_input(stat_row)

        game_pk = int(result.get("game_pk", 0))
        gp = game_pitchers.get(game_pk)
        pitcher_input = _LEAGUE_AVG_PITCHER
        sp_id_used = ""
        if gp:
            batter_team = str(result.get("team", "")).lower()
            home_team = str(gp.get("home_team", "")).lower()
            if batter_team in home_team or home_team in batter_team:
                opp_sp_id = gp.get("away_sp_id")
            else:
                opp_sp_id = gp.get("home_sp_id")
            if opp_sp_id and int(opp_sp_id) in pitcher_stats:
                pitcher_row = pitcher_stats[int(opp_sp_id)]
                pitcher_input = _pitcher_row_to_input(pitcher_row)
                sp_id_used = str(opp_sp_id)
                sp_matched += 1

        proxy_mode = batter_input["_provenance"].get("k_rate") != "measured"
        if proxy_mode:
            proxy_count += 1

        scored_result = _score_one(market, result, batter_input, pitcher_input,
                                   sp_id_used, proxy_mode)
        if scored_result is None:
            continue

        score = scored_result["score"]
        actual_hit = result[outcome_col]

        if score >= 80:
            buckets["80+"].append(actual_hit)
        elif score >= 70:
            buckets["70-79"].append(actual_hit)
        elif score >= 60:
            buckets["60-69"].append(actual_hit)
        elif score >= 50:
            buckets["50-59"].append(actual_hit)
        else:
            buckets["<50"].append(actual_hit)

        scored += 1

    log.info(f"\nScored {scored} player-games | Unmatched batters: {unmatched} | "
             f"SP matched: {sp_matched} | Proxy-mode: {proxy_count}")
    return {
        "market": market,
        "vig_factor": VIG_BY_MARKET.get(market, 0.535),
        "scored": scored,
        "unmatched": unmatched,
        "sp_matched": sp_matched,
        "proxy_count": proxy_count,
        "buckets": buckets,
    }


def print_calibration_report(results: Dict) -> None:
    if not results:
        print("No results to report.")
        return

    market = results.get("market", "tb_o15")
    vig_factor = results.get("vig_factor", 0.535)
    market_label = {"tb_o15": "O1.5 TB", "hits_o05": "O0.5 HITS", "hr": "O0.5 HR"}.get(market, market.upper())
    vig_label    = {0.535: "-115", 0.636: "-175", 0.222: "+350"}.get(vig_factor, f"{vig_factor:.3f}")
    print("\n" + "=" * 60)
    print(f"  {market_label} MODEL — CALIBRATION REPORT")
    print("=" * 60)
    print(f"  Total player-games scored: {results['scored']}")
    print(f"  Unmatched to season stats: {results['unmatched']}")
    print(f"  SP matched (real stats):   {results.get('sp_matched', 0)}")
    print(f"  Proxy mode (est K%):       {results['proxy_count']}")
    print()
    print(f"  {'Score Bucket':<12} {'Plays':>6} {'Hit Rate':>10} {f'ROI @ {vig_label}':>12}")
    print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*12}")

    total_plays = 0
    total_wins = 0

    for bucket, hits in results["buckets"].items():
        n = len(hits)
        if n == 0:
            print(f"  {bucket:<12} {'—':>6}")
            continue
        win_rate = sum(hits) / n
        roi = win_rate / vig_factor - 1.0
        total_plays += n
        total_wins += sum(hits)
        roi_str = f"+{roi*100:.1f}%" if roi >= 0 else f"{roi*100:.1f}%"
        print(f"  {bucket:<12} {n:>6} {win_rate*100:>9.1f}% {roi_str:>12}")

    if total_plays > 0:
        overall = total_wins / total_plays
        overall_roi = overall / vig_factor - 1.0
        roi_str = f"+{overall_roi*100:.1f}%" if overall_roi >= 0 else f"{overall_roi*100:.1f}%"
        print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*12}")
        print(f"  {'OVERALL':<12} {total_plays:>6} {overall*100:>9.1f}% {roi_str:>12}")

    print()
    sp_matched = results.get("sp_matched", 0)
    sp_note = (f"{sp_matched:,} player-games used real SP stats (run backtest_enrich_sp.py to add more)"
               if sp_matched else "SP stats are league-average for all games (run backtest_enrich_sp.py)")
    print("  ⚠️  Caveats:")
    print(f"  - {sp_note}")
    print("  - Uses current season stats for ALL historical dates (snapshot bias)")
    print("  - Batter handedness and weather defaulted to neutral")
    print("  - ROI assumes no historical prop odds (Odds API Business needed for real vig)")
    print("=" * 60)


def run_sp_k_backtest(season: int = 2026) -> Dict:
    """
    SP-level K prop calibration.

    Loads game_sp_results (actual SP K totals) and scores each SP using
    compute_sp_k_score with the opposing lineup's average K propensity derived
    from batter_stats. Buckets scores against two outcome thresholds:
      hit_k5: SP threw ≥5 Ks (over 4.5 line)
      hit_k6: SP threw ≥6 Ks (over 5.5 line — the most common main K prop)
    """
    conn = sqlite3.connect(DB_PATH)
    pitcher_stats  = _load_pitcher_stats(conn, season)
    batter_stats   = _load_batter_stats(conn, season)

    try:
        sp_rows = conn.execute("SELECT * FROM game_sp_results").fetchall()
        sp_cols = [d[0] for d in conn.execute("SELECT * FROM game_sp_results LIMIT 0").description]
        sp_games = [dict(zip(sp_cols, r)) for r in sp_rows]
    except Exception:
        log.warning("game_sp_results table not found — run data/backtest_sp_k_fetcher.py first")
        conn.close()
        return {}

    # Load game_results to build opposing lineups per (game_pk, team)
    try:
        gr_rows = conn.execute(
            "SELECT game_pk, player_id, team, lineup_slot FROM game_results"
        ).fetchall()
    except Exception:
        gr_rows = []
    conn.close()

    # Index: game_pk → {team → [(player_id, lineup_slot)]}
    from collections import defaultdict
    game_lineups: Dict = defaultdict(lambda: defaultdict(list))
    for game_pk, pid, team, slot in gr_rows:
        game_lineups[game_pk][str(team)].append((int(pid), int(slot or 5)))

    from scoring.strikeout import compute_sp_k_score, compute_batter_k_propensity

    _NEUTRAL_PITCHER = {
        "k_rate_allowed": 0.228, "hard_hit_allowed": 0.370,
        "whip": 1.30, "era": 4.20, "swstr_pct": 0.0,
        "_provenance": {"k_rate_allowed": "league_avg"},
    }

    buckets_k5 = {"<50": [], "50-59": [], "60-69": [], "70-79": [], "80+": []}
    buckets_k6 = {"<50": [], "50-59": [], "60-69": [], "70-79": [], "80+": []}

    scored = unmatched_sp = 0

    for row in sp_games:
        sp_id  = row.get("sp_id")
        sp_team = str(row.get("team", ""))
        game_pk = int(row.get("game_pk", 0))
        sp_k    = int(row.get("sp_k", 0))
        hit_k5  = 1 if sp_k >= 5 else 0
        hit_k6  = 1 if sp_k >= 6 else 0

        if not sp_id or int(sp_id) not in pitcher_stats:
            unmatched_sp += 1
            continue

        sp_row   = pitcher_stats[int(sp_id)]
        sp_input = _pitcher_row_to_input(sp_row)

        # Opposing lineup: batters from the other teams in this game
        all_teams_in_game = game_lineups.get(game_pk, {})
        opp_batters = [
            (pid, slot)
            for t, lineup in all_teams_in_game.items()
            for pid, slot in lineup
            if t != sp_team
        ]

        k_propensities = []
        for pid, slot in opp_batters:
            bat_row = batter_stats.get(pid)
            if not bat_row:
                continue
            bat_input = _stat_row_to_scoring_input(bat_row)
            k_prop, _, _ = compute_batter_k_propensity(bat_input, sp_input, lineup_slot=slot)
            k_propensities.append(k_prop)

        opp_lineup_k_avg = sum(k_propensities) / len(k_propensities) if k_propensities else 50.0
        n_batters = len(k_propensities)

        k_score, _, _ = compute_sp_k_score(
            pitcher_stats=sp_input,
            opp_lineup_k_avg=opp_lineup_k_avg,
            implied_total=4.5,   # neutral game total (no Vegas lines)
            ump_k_adj=0.0,
        )

        def _bucket(score):
            if score >= 80: return "80+"
            if score >= 70: return "70-79"
            if score >= 60: return "60-69"
            if score >= 50: return "50-59"
            return "<50"

        b = _bucket(k_score)
        buckets_k5[b].append(hit_k5)
        buckets_k6[b].append(hit_k6)
        scored += 1

    log.info(f"\nK prop backtest: {scored} SP-games scored | Unmatched SPs: {unmatched_sp}")
    return {
        "market": "k_prop",
        "scored": scored,
        "unmatched_sp": unmatched_sp,
        "buckets_k5": buckets_k5,
        "buckets_k6": buckets_k6,
    }


def print_sp_k_report(results: Dict) -> None:
    if not results:
        print("No K prop results to report.")
        return

    print("\n" + "=" * 65)
    print("  PITCHER K PROP MODEL — CALIBRATION REPORT")
    print("=" * 65)
    print(f"  SP-games scored:  {results['scored']}")
    print(f"  Unmatched SPs:    {results['unmatched_sp']}")
    print()

    for threshold, key, label in ((5, "buckets_k5", "O4.5 Ks"), (6, "buckets_k6", "O5.5 Ks")):
        buckets = results[key]
        print(f"  Outcome: SP threw ≥{threshold} Ks ({label})")
        print(f"  {'Score Bucket':<12} {'Games':>6} {'Hit Rate':>10}")
        print(f"  {'-'*12} {'-'*6} {'-'*10}")
        for bucket, hits in buckets.items():
            n = len(hits)
            if n == 0:
                print(f"  {bucket:<12} {'—':>6}")
                continue
            rate = sum(hits) / n
            print(f"  {bucket:<12} {n:>6} {rate*100:>9.1f}%")
        print()

    print("  ⚠️  Caveats:")
    print("  - Uses current season SP stats for historical dates (snapshot bias)")
    print("  - Game totals defaulted to 4.5 (no Vegas lines)")
    print("  - Umpire K adjustment defaulted to 0")
    print("=" * 65)


def _pitcher_to_vuln(pitcher_input: Dict) -> float:
    """Compute SP vulnerability (0–100) using the pure scoring function."""
    try:
        from scoring.pitcher import compute_pitcher_score
        score, _, _ = compute_pitcher_score(pitcher_input)
        return float(score)
    except Exception:
        return 50.0


def run_ml_backtest(season: int = 2026) -> Dict:
    """
    ML game outcome calibration.

    For each completed game in game_outcomes, predicts the winner using
    compute_win_probability with:
      - Real SP k/hard-hit stats → computed _sp_vuln via compute_pitcher_score
      - Per-game offensive wRC+ average from batter_stats + game_results lineups
      - Bullpen vulnerability defaulted to 50 (neutral — no per-game BP data)
      - No Vegas implied runs (Odds API not subscribed)

    Buckets by predicted probability for the modeled pick side;
    reports actual win rate per bucket.
    """
    from scoring.moneyline import compute_win_probability

    conn = sqlite3.connect(DB_PATH)
    pitcher_stats  = _load_pitcher_stats(conn, season)
    batter_stats   = _load_batter_stats(conn, season)
    game_pitchers  = _load_game_pitchers(conn)

    try:
        go_rows = conn.execute("SELECT * FROM game_outcomes").fetchall()
        go_cols = [d[0] for d in conn.execute("SELECT * FROM game_outcomes LIMIT 0").description]
        game_outcomes_list = [dict(zip(go_cols, r)) for r in go_rows]
    except Exception:
        log.warning("game_outcomes table not found — run data/backtest_game_outcomes_fetcher.py first")
        conn.close()
        return {}

    # Build per-game per-team wRC+ lists from game_results
    try:
        gr_rows = conn.execute(
            "SELECT game_pk, player_id, team FROM game_results"
        ).fetchall()
    except Exception:
        gr_rows = []
    conn.close()

    game_team_wrc: Dict = defaultdict(lambda: defaultdict(list))
    for game_pk, pid, team in gr_rows:
        if not pid or not team:
            continue
        stat_row = batter_stats.get(int(pid))
        if stat_row:
            wrc = stat_row.get("wRC+")
            if wrc is not None:
                try:
                    game_team_wrc[game_pk][str(team)].append(float(wrc))
                except (TypeError, ValueError):
                    pass

    # Cache vuln scores to avoid recomputing for the same SP repeatedly
    _vuln_cache: Dict[int, float] = {}

    def _sp_vuln(sp_id: int) -> float:
        if sp_id not in _vuln_cache:
            row = pitcher_stats.get(sp_id)
            inp = _pitcher_row_to_input(row) if row else _LEAGUE_AVG_PITCHER
            _vuln_cache[sp_id] = _pitcher_to_vuln(inp)
        return _vuln_cache[sp_id]

    buckets = {
        "50-55%": [],
        "55-60%": [],
        "60-65%": [],
        "65%+":   [],
    }

    scored = unmatched_sp = tied = 0

    for game in game_outcomes_list:
        game_pk      = int(game.get("game_pk", 0))
        winning_team = str(game.get("winning_team", ""))

        if winning_team == "tie":
            tied += 1
            continue

        gp = game_pitchers.get(game_pk)

        home_sp_stats = _LEAGUE_AVG_PITCHER
        away_sp_stats = _LEAGUE_AVG_PITCHER
        home_sp_vuln  = 50.0
        away_sp_vuln  = 50.0

        if gp:
            home_sp_id = gp.get("home_sp_id")
            away_sp_id = gp.get("away_sp_id")
            if home_sp_id and int(home_sp_id) in pitcher_stats:
                home_sp_stats = _pitcher_row_to_input(pitcher_stats[int(home_sp_id)])
                home_sp_vuln  = _sp_vuln(int(home_sp_id))
            else:
                unmatched_sp += 1
            if away_sp_id and int(away_sp_id) in pitcher_stats:
                away_sp_stats = _pitcher_row_to_input(pitcher_stats[int(away_sp_id)])
                away_sp_vuln  = _sp_vuln(int(away_sp_id))
            else:
                unmatched_sp += 1
        else:
            unmatched_sp += 2

        home_sp_stats = {**home_sp_stats, "_sp_vuln": home_sp_vuln}
        away_sp_stats = {**away_sp_stats, "_sp_vuln": away_sp_vuln}

        # Split batters by home vs away using game_pitchers home_team for the key
        game_teams = game_team_wrc.get(game_pk, {})
        gp_home_str = str(gp.get("home_team", "")).lower() if gp else ""

        home_wrc_vals: list = []
        away_wrc_vals: list = []
        for team_str, wrc_list in game_teams.items():
            team_lower = team_str.lower()
            if gp_home_str and (team_lower in gp_home_str or gp_home_str in team_lower):
                home_wrc_vals.extend(wrc_list)
            else:
                away_wrc_vals.extend(wrc_list)

        home_off_wrc = sum(home_wrc_vals) / len(home_wrc_vals) if home_wrc_vals else 100.0
        away_off_wrc = sum(away_wrc_vals) / len(away_wrc_vals) if away_wrc_vals else 100.0

        home_wp, _ = compute_win_probability(
            home_sp_stats=home_sp_stats,
            away_sp_stats=away_sp_stats,
            home_off_wrc=home_off_wrc,
            away_off_wrc=away_off_wrc,
            home_bp_vuln=50.0,
            away_bp_vuln=50.0,
            home_run_diff=0.0,
            away_run_diff=0.0,
            home_implied_runs=0.0,
            away_implied_runs=0.0,
        )

        # Pick the side the model favors and check if it won
        if home_wp >= 0.5:
            pick_side = "home"
            pick_prob = home_wp
        else:
            pick_side = "away"
            pick_prob = 1.0 - home_wp

        actual_correct = 1 if winning_team == pick_side else 0

        if pick_prob >= 0.65:
            buckets["65%+"].append(actual_correct)
        elif pick_prob >= 0.60:
            buckets["60-65%"].append(actual_correct)
        elif pick_prob >= 0.55:
            buckets["55-60%"].append(actual_correct)
        else:
            buckets["50-55%"].append(actual_correct)

        scored += 1

    log.info(f"\nML backtest: {scored} games scored | Unmatched SPs: {unmatched_sp} | Ties: {tied}")
    return {
        "market": "ml",
        "scored": scored,
        "unmatched_sp": unmatched_sp,
        "tied": tied,
        "buckets": buckets,
    }


def print_ml_report(results: Dict) -> None:
    if not results:
        print("No ML backtest results to report.")
        return

    print("\n" + "=" * 65)
    print("  MONEYLINE MODEL — CALIBRATION REPORT")
    print("=" * 65)
    print(f"  Games scored:     {results['scored']}")
    print(f"  Ties excluded:    {results['tied']}")
    print(f"  Unmatched SPs:    {results['unmatched_sp']}")
    print()
    print("  Predicted win prob for modeled pick side vs actual win rate")
    print(f"  {'Prob Bucket':<12} {'Games':>6} {'Actual Win%':>12}  {'vs Expected':>12}")
    print(f"  {'-'*12} {'-'*6} {'-'*12}  {'-'*12}")

    BUCKET_MIDPOINTS = {"50-55%": 0.525, "55-60%": 0.575, "60-65%": 0.625, "65%+": 0.675}
    total_games = 0
    total_wins  = 0

    for bucket, hits in results["buckets"].items():
        n = len(hits)
        if n == 0:
            print(f"  {bucket:<12} {'—':>6}")
            continue
        win_rate = sum(hits) / n
        expected = BUCKET_MIDPOINTS.get(bucket, 0.55)
        diff = win_rate - expected
        diff_str = f"+{diff*100:.1f}%" if diff >= 0 else f"{diff*100:.1f}%"
        total_games += n
        total_wins  += sum(hits)
        print(f"  {bucket:<12} {n:>6} {win_rate*100:>11.1f}%  {diff_str:>12}")

    if total_games > 0:
        overall = total_wins / total_games
        print(f"  {'-'*12} {'-'*6} {'-'*12}  {'-'*12}")
        print(f"  {'OVERALL':<12} {total_games:>6} {overall*100:>11.1f}%")

    print()
    print("  ⚠️  Caveats:")
    print("  - SP vuln from current season stats, not day-of (snapshot bias)")
    print("  - wRC+ lineup avg from season-level stats, not game-day roster")
    print("  - Bullpen vuln fixed at 50.0 (neutral) — no per-game BP data")
    print("  - No Vegas implied runs blend (Odds API Business needed)")
    print("  - Home/away wRC+ split via team-name substring match — may misalign")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="MLB prop model backtest")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--market", default="tb_o15",
                        choices=["tb_o15", "hits_o05", "hr", "k_prop", "ml"])
    parser.add_argument("--min-ab", type=int, default=3,
                        help="Minimum at-bats to include a player-game (batter markets)")
    args = parser.parse_args()

    if args.market == "k_prop":
        results = run_sp_k_backtest(season=args.season)
        print_sp_k_report(results)
    elif args.market == "ml":
        results = run_ml_backtest(season=args.season)
        print_ml_report(results)
    else:
        results = run_backtest(season=args.season, market=args.market, min_ab=args.min_ab)
        print_calibration_report(results)


if __name__ == "__main__":
    main()
