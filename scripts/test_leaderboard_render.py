"""
test_leaderboard_render.py

End-to-end render test for display_leaderboard() without a browser.

Mocks the Streamlit surface (session_state + all UI calls), injects realistic
play dicts built from the real SQLite db, and verifies:

  1. No NameError — leaderboard runs to completion
  2. Table rows are built (not 0 rows)
  3. Proxy banner: NOT shown when batting_cols has full-Savant columns
  4. disk_cache_fresh banner: SHOWN when _batting_source = "disk_cache_fresh"
  5. Scenario 3: proxy banner IS shown when batting_cols lacks Savant columns

Runs three scenarios back-to-back.
"""

import sys, os, types
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import sqlite3
import pandas as pd

# ── Play dict builder ─────────────────────────────────────────────────────────
DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mlb_stats.db")

def _safe(v, default=0.0):
    try: return float(v) if v is not None else default
    except: return default

def make_play(row, score=66.0, bettable=True, pa=150):
    name = str(row.get("_name", "Test Player"))
    return {
        "name":               name,
        "player_id":          str(row.get("mlbam_id", "000000")),
        "team":               "HOU",
        "opponent":           "COL",
        "game_id":            "999999",
        "lineup_slot":        3,
        "lineup_confirmed":   True,
        "batter_hand":        "R",
        "batter_position":    "OF",
        "sp_name":            "Tanner Gordon",
        "sp_hand":            "R",
        "sp_tbd":             False,
        "score":              score,
        "prob":               0.62,
        "tier":               "📊 TIER 3" if score < 70 else "✅ TIER 2",
        "park":               "COL",
        "park_label":         "COL (1.22x TB | 1.35x HR)",
        "weather_label":      "75°F | 8mph Out",
        "wind_effect":        "out",
        "is_dome":            False,
        "implied_total":      0.0,
        "market_edge":        0.0,
        "edge_label":         "no lines",
        "tto_label":          "3rd TTO territory",
        "platoon_label":      "Even (R vs R)",
        "lineup_label":       "Cleanup (slot 3)",
        "pitcher_label":      "db_matched",
        "hr_score":           42.0,
        "xslg":               _safe(row.get("xSLG"), 0.398),
        "barrel_rate":        _safe(row.get("barrel_batted_rate"), 0.07),
        "hard_hit_rate":      _safe(row.get("hard_hit_percent"), 0.37),
        "k_rate":             _safe(row.get("K%"), 0.228),
        "bb_rate":            0.082,
        "wrc_plus":           _safe(row.get("wRC+"), 100.0),
        "iso":                _safe(row.get("ISO"), 0.165),
        "exit_velocity":      88.5,
        "sweet_spot_rate":    0.30,
        "sub_batter":         52.0,
        "sub_pitcher":        57.0,
        "sub_matchup":        50.0,
        "matchup_label":      "neutral pitch mix",
        "sub_streak":         50.0,
        "streak_label":       "No recent data",
        "recent_tb_per_game": None,
        "recent_games":       0,
        "_hr_last7":          0,
        "game_total":         9.2,
        "_h_last7":           0,
        "_ab_last7":          0,
        "sub_bvp":            50.0,
        "bvp_label":          "No BvP data",
        "bvp_sig":            "neutral",
        "bvp_ab":             0,
        "bvp_slg":            None,
        "bvp_hr":             0,
        "bvp_xbh":            0,
        "sp_id":              "999",
        "sub_platoon":        50.0,
        "sub_lineup":         84.0,
        "sub_park":           100.0,
        "sub_weather":        50.0,
        "sub_vegas":          0.0,
        "vegas_missing":      True,
        "batter_pa":          pa,
        "bullpen_vuln":       44.5,
        "platoon_edge":       "Even (R vs R)",
        "bat_speed":          0.0,
        "blast_rate":         0.0,
        "ev50":               0.0,
        "sprint_speed":       27.0,
        "dq_score":           72,
        "bettable":           bettable,
        "non_bettable_reasons": [] if bettable else [f"insufficient sample ({pa} PA < 50 minimum)"],
        "_batter_prov":       {"k_rate":"measured","slg_proxy":"measured","woba":"measured",
                               "hard_hit_rate":"measured","barrel_rate":"measured"},
        "_pitcher_prov":      {"k_rate_allowed":"measured","hard_hit_allowed":"measured"},
        "temperature":        75,
        "wind_speed":         8,
        "wind_dir":           "Out",
    }

# Load real batters from db
con = sqlite3.connect(DB)
bat_df = pd.read_sql('SELECT * FROM batter_stats WHERE pa >= 100 ORDER BY "wRC+" DESC LIMIT 5', con)
low_pa = pd.read_sql("SELECT * FROM batter_stats WHERE pa < 10 LIMIT 1", con)
con.close()

scores = [68.9, 67.7, 65.9, 65.6, 63.0]
plays_full = [make_play(row, score=scores[i]) for i, (_, row) in enumerate(bat_df.iterrows())]
if not low_pa.empty:
    plays_full.append(make_play(low_pa.iloc[0], score=61.0, bettable=False, pa=int(low_pa.iloc[0]["pa"])))

print(f"Built {len(plays_full)} play dicts ({len(plays_full)-1} bettable, 1 NON-BETTABLE)")
for p in plays_full:
    print(f"  {p['name']:30s} score={p['score']} bettable={p['bettable']}")
print()

# ── Streamlit mock surface ────────────────────────────────────────────────────
class _ColCtx:
    """Fake column context manager — mimics 'with col1: ...'"""
    def __enter__(self): return self
    def __exit__(self, *a): return False

def _make_columns(n):
    if isinstance(n, (list, tuple)):
        return [_ColCtx() for _ in n]
    return [_ColCtx() for _ in range(n)]

class _SidebarMock:
    """Mock for st.sidebar — returns safe defaults for widget calls."""
    def __getattr__(self, name):
        def _noop(*a, **kw):
            if name in ("toggle", "checkbox"): return kw.get("value", False)
            if name == "multiselect": return kw.get("default", [])
            if name == "slider":
                if len(a) >= 4: return a[3]
                return kw.get("value", 0)
            return None
        return _noop

class _MockST:
    """Catch-all Streamlit mock that records banner calls and no-ops everything else."""
    def __init__(self, session_state, banners):
        self.session_state = session_state
        self._banners = banners
        self.sidebar = _SidebarMock()

    def warning(self, msg, **kw): self._banners.append(("warning", str(msg)[:100]))
    def success(self, msg, **kw): self._banners.append(("success", str(msg)[:100]))
    def info(self,    msg, **kw): self._banners.append(("info",    str(msg)[:100]))
    def error(self,   msg, **kw): self._banners.append(("error",   str(msg)[:100]))
    def dataframe(self, df, **kw):
        # df may be a Styler; get underlying shape via .data if present
        n = len(df.data) if hasattr(df, "data") else len(df)
        self._banners.append(("dataframe", f"{n} rows"))

    def columns(self, n): return _make_columns(n)

    # Everything else is a no-op that returns a safe default
    def __getattr__(self, name):
        def _noop(*a, **kw):
            if name == "multiselect": return kw.get("default", [])
            if name == "slider":
                if len(a) >= 4: return a[3]
                return kw.get("value", 0)
            if name in ("expander", "container", "form"):
                return _ColCtx()
            return None
        return _noop

def run_scenario(label, batting_source, batting_cols):
    print("=" * 68)
    print(f"SCENARIO: {label}")
    print("=" * 68)

    banners = []
    session = {
        "_batting_source":       batting_source,
        "batting_cols":          batting_cols,
        "_db_freshness_warning": None,
    }
    mock_st = _MockST(session_state=session, banners=banners)

    import ui.leaderboard as lb
    original_st = lb.st
    lb.st = mock_st

    # Patch pd.DataFrame to count rows when leaderboard builds its table
    rows_built = []
    original_pd = lb.pd

    class _PatchPD:
        """Minimal pandas proxy that intercepts DataFrame() to count rows."""
        def __getattr__(self, n): return getattr(original_pd, n)
        def DataFrame(self, rows=None, **kw):
            if rows is not None:
                try: rows_built.append(len(rows))
                except: pass
            return original_pd.DataFrame(rows or [], **kw)

    lb.pd = _PatchPD()

    try:
        lb.display_leaderboard(plays_full)
        status = "✓ NO ERROR"
    except NameError as e:
        status = f"✗ NameError: {e}"
    except Exception as e:
        import traceback
        status = f"✗ {type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        lb.st = original_st
        lb.pd = original_pd

    print(f"  Status:       {status}")
    print(f"  Rows built:   {rows_built[0] if rows_built else 'N/A (no rows?)'}")
    print()
    print("  Banners fired:")
    for kind, msg in banners:
        icon = {"success":"✅","warning":"⚠️","info":"ℹ️","error":"❌","dataframe":"📊"}.get(kind,"·")
        print(f"    {icon} [{kind:9s}] {msg}")

    proxy_banner = any("Proxy Data Mode" in m for _, m in banners)
    cache_banner = any("disk cache" in m.lower() for _, m in banners)
    has_rows     = bool(rows_built and rows_built[0] > 0)

    print()
    print("  Check results:")
    print(f"    No NameError:       {'PASS' if 'NameError' not in status else 'FAIL'}")
    print(f"    Table has rows:     {'PASS' if has_rows else 'FAIL'} ({rows_built[0] if rows_built else 0})")
    print(f"    Proxy banner:       {proxy_banner}")
    print(f"    Cache banner:       {cache_banner}")
    return proxy_banner, cache_banner, has_rows, status

# ── Three scenarios ───────────────────────────────────────────────────────────

FULL_COLS = ["barrel_batted_rate", "hard_hit_percent", "wRC+", "xSLG", "K%", "wOBA", "ISO"]
PROXY_COLS = ["K%", "BB%", "SLG", "AVG", "OBP", "ISO"]   # no barrel/HH/wRC+

p1, c1, r1, s1 = run_scenario(
    "SQLite / savant+mlbapi — expect: NO proxy, NO cache",
    batting_source="savant+mlbapi",
    batting_cols=FULL_COLS,
)
print()

p2, c2, r2, s2 = run_scenario(
    "disk_cache_fresh — expect: NO proxy, YES cache",
    batting_source="disk_cache_fresh",
    batting_cols=FULL_COLS,
)
print()

p3, c3, r3, s3 = run_scenario(
    "mlbapi_only (true proxy mode) — expect: YES proxy, NO cache",
    batting_source="mlbapi_only",
    batting_cols=PROXY_COLS,
)
print()

# ── Final verdict ─────────────────────────────────────────────────────────────
print("=" * 68)
print("FINAL VERDICT")
print("=" * 68)

checks = {
    "S1 no NameError":       "NameError" not in s1,
    "S2 no NameError":       "NameError" not in s2,
    "S3 no NameError":       "NameError" not in s3,
    "S1 rows built":         r1,
    "S2 rows built":         r2,
    "S3 rows built":         r3,
    "S1 no proxy banner":    not p1,
    "S1 no cache banner":    not c1,
    "S2 no proxy banner":    not p2,
    "S2 cache banner fires": c2,
    "S3 proxy banner fires": p3,
    "S3 no cache banner":    not c3,
}
for desc, result in checks.items():
    print(f"  {'PASS' if result else 'FAIL'}  {desc}")

all_pass = all(checks.values())
print()
print(f"  {'ALL CHECKS PASS — ready to push' if all_pass else 'ONE OR MORE CHECKS FAILED — DO NOT PUSH'}")
