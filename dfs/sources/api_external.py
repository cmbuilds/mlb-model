"""
dfs/sources/api_external.py

Pluggable adapter for external DFS data sources (projections / ownership).

Design goal: the consensus core (dfs/consensus.py) depends ONLY on the
ProjectionSource / OwnershipSource protocols below — never on a concrete vendor.
Swapping BluecollarDFS -> FTN -> SportsDataIO is a config change, not a code change.

Non-negotiables inherited from CLAUDE.md (do not relax):
  * Fail loud, never silent. A bad/empty fetch raises or logs — it never
    defaults to league-average or a fabricated number.
  * Provenance per row. Every row is tagged with its source + kind.
  * No fabrication. A source returns what it measured/modeled; the adapter
    never invents a value to fill a gap.

Orthogonality principle (drives consensus weighting):
  An external projection MODEL correlates with our model and the market, so it
  adds little information (weight 0.5). Market-implied (sharp money) and field
  ownership are the orthogonal, high-value signals (weight 1.0).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger("dfs.sources")


# ─────────────────────────────────────────────────────────────────────────────
# Enums / provenance
# ─────────────────────────────────────────────────────────────────────────────
class SourceKind(str, Enum):
    MODEL = "model"                    # our own projection (the edge)
    MARKET = "market"                  # implied from sportsbook props (sharp, orthogonal)
    EXTERNAL_MODEL = "external_model"  # another vendor's projection model (correlated)
    OWNERSHIP = "ownership"            # field ownership projection


class Provenance(str, Enum):
    MEASURED = "measured"
    PROXY = "proxy"
    LEAGUE_AVG = "league_avg"
    MARKET = "market"
    MODEL_EXTERNAL = "model_external"
    MODELED = "modeled"                # our modeled ownership fallback — NEVER shown as field own.


# Consensus weights by kind. Market is the orthogonal truth; a correlated
# external model is discounted. Tune in one place; consensus.py imports this.
ORTHOGONALITY_WEIGHT: Dict[SourceKind, float] = {
    SourceKind.MODEL: 1.0,
    SourceKind.MARKET: 1.0,
    SourceKind.EXTERNAL_MODEL: 0.5,
    SourceKind.OWNERSHIP: 1.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Row contracts (mirror dfs/contracts.py; defined here so the adapter compiles
# standalone — import from contracts once that module lands).
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ProjectionRow:
    player: str
    team: str
    position: str
    site: str                 # "dk" | "fd"
    proj_pts: float
    source: str               # vendor id, e.g. "bluecollardfs"
    kind: SourceKind
    provenance: Provenance
    salary: Optional[int] = None
    updated: Optional[str] = None     # ISO ts of the vendor's last refresh


@dataclass(frozen=True)
class OwnershipRow:
    player: str
    team: str
    site: str
    slate: str
    own_pct: float
    source: str
    provenance: Provenance
    updated: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Protocols the consensus core depends on
# ─────────────────────────────────────────────────────────────────────────────
@runtime_checkable
class ProjectionSource(Protocol):
    id: str
    kind: SourceKind
    def fetch_projections(self, *, site: str, date: str) -> List[ProjectionRow]: ...


@runtime_checkable
class OwnershipSource(Protocol):
    id: str
    def fetch_ownership(self, *, site: str, date: str, slate: str) -> List[OwnershipRow]: ...


class SourceError(RuntimeError):
    """Raised on any failed/empty fetch. Fail loud — never silently degrade."""


# ─────────────────────────────────────────────────────────────────────────────
# Base: shared fail-loud HTTP
# ─────────────────────────────────────────────────────────────────────────────
class _HttpSource(ABC):
    id: str = "base"
    kind: SourceKind = SourceKind.EXTERNAL_MODEL

    def __init__(self, cfg: Dict, http_get: Optional[Callable] = None):
        # http_get injected for testability; defaults to requests.get.
        self._cfg = cfg or {}
        if http_get is not None:
            self._get = http_get
        else:
            import requests  # local import keeps module import cheap
            self._get = requests.get

    def _json(self, url: str, headers: Optional[Dict] = None, params: Optional[Dict] = None):
        try:
            r = self._get(url, headers=headers or {}, params=params or {}, timeout=20)
        except Exception as e:  # network/timeout — loud, logged, no silent default
            logger.error("[%s] request failed: %s", self.id, e)
            raise SourceError(f"{self.id}: request failed ({e})") from e
        if getattr(r, "status_code", 200) != 200:
            logger.error("[%s] HTTP %s for %s", self.id, r.status_code, url)
            raise SourceError(f"{self.id}: HTTP {r.status_code}")
        try:
            return r.json()
        except Exception as e:
            raise SourceError(f"{self.id}: bad JSON ({e})") from e

    @abstractmethod
    def fetch_projections(self, *, site: str, date: str) -> List[ProjectionRow]: ...


# ─────────────────────────────────────────────────────────────────────────────
# Concrete: BluecollarDFS  (cheap, MLB DK/FD JSON proj + salaries + value)
#   GET https://bluecollardfs.com/api/{sport}_{site}
#   Header: Authorization: ApiKey <key>   (200 req/day; cache >=5 min)
#   NOTE: projections + salaries only — NO ownership. Enforced below.
# ─────────────────────────────────────────────────────────────────────────────
class BluecollarDFSProjections(_HttpSource):
    id = "bluecollardfs"
    kind = SourceKind.EXTERNAL_MODEL  # correlated → consensus weight 0.5

    BASE = "https://bluecollardfs.com/api"

    def fetch_projections(self, *, site: str, date: str) -> List[ProjectionRow]:
        key = self._cfg.get("api_key")
        if not key:
            raise SourceError("bluecollardfs: missing api_key in config")
        url = f"{self.BASE}/mlb_{site}"
        data = self._json(url, headers={"Authorization": f"ApiKey {key}"})
        slates = (data or {}).get("slates") or []
        rows: List[ProjectionRow] = []
        for slate in slates:
            for p in slate.get("players", []):
                if p.get("projection") is None:
                    continue
                rows.append(ProjectionRow(
                    player=p["name"], team=p.get("team", ""),
                    position=p.get("position", ""), site=site,
                    proj_pts=float(p["projection"]),
                    salary=p.get("salary"),
                    source=self.id, kind=self.kind,
                    provenance=Provenance.MODEL_EXTERNAL,
                    updated=data.get("updated"),
                ))
        if not rows:
            raise SourceError("bluecollardfs: returned no projections (fail loud)")
        return rows


# ─────────────────────────────────────────────────────────────────────────────
# Concrete: Odds-implied  (market projection — the orthogonal signal)
# ─────────────────────────────────────────────────────────────────────────────
class OddsImpliedProjections:
    id = "odds_market"
    kind = SourceKind.MARKET  # orthogonal → weight 1.0

    def __init__(self, prop_fetcher: Callable[[str], Dict], to_points: Callable[[Dict], float]):
        self._fetch = prop_fetcher
        self._to_points = to_points

    def fetch_projections(self, *, site: str, date: str) -> List[ProjectionRow]:
        props = self._fetch(date) or {}
        if not props:
            raise SourceError("odds_market: no prop lines (Business tier required for full markets)")
        rows: List[ProjectionRow] = []
        for name, line in props.items():
            pts = self._to_points(line)
            if pts is None:
                continue
            rows.append(ProjectionRow(
                player=name, team=line.get("team", ""),
                position=line.get("position", ""), site=site,
                proj_pts=float(pts), salary=None,
                source=self.id, kind=self.kind, provenance=Provenance.MARKET,
            ))
        if not rows:
            raise SourceError("odds_market: no convertible props")
        return rows


# ─────────────────────────────────────────────────────────────────────────────
# Scaffolds: FTN and SportsDataIO — fail loud until configured
# ─────────────────────────────────────────────────────────────────────────────
class FTNSource(_HttpSource):
    id = "ftn"
    kind = SourceKind.EXTERNAL_MODEL

    def fetch_projections(self, *, site: str, date: str) -> List[ProjectionRow]:
        raise SourceError("ftn: not configured (D-Q3 not selected)")

    def fetch_ownership(self, *, site: str, date: str, slate: str) -> List[OwnershipRow]:
        raise SourceError("ftn: ownership not configured (D-Q3 not selected)")


class SportsDataIOSource(_HttpSource):
    id = "sportsdataio"
    kind = SourceKind.EXTERNAL_MODEL

    def fetch_projections(self, *, site: str, date: str) -> List[ProjectionRow]:
        raise SourceError("sportsdataio: not configured (commercial tier; D-Q3 not selected)")


# ─────────────────────────────────────────────────────────────────────────────
# Registry / factory
# ─────────────────────────────────────────────────────────────────────────────
_PROJECTION_REGISTRY: Dict[str, type] = {
    BluecollarDFSProjections.id: BluecollarDFSProjections,
    FTNSource.id: FTNSource,
    SportsDataIOSource.id: SportsDataIOSource,
}


def get_enabled_projection_sources(cfg: Dict) -> List[ProjectionSource]:
    """
    cfg example:
      {"sources": {"bluecollardfs": {"enabled": True, "api_key": "..."}}}
    Returns instantiated sources for every enabled vendor.
    """
    out: List[ProjectionSource] = []
    for vid, vcfg in (cfg.get("sources") or {}).items():
        if not vcfg.get("enabled"):
            continue
        klass = _PROJECTION_REGISTRY.get(vid)
        if klass is None:
            raise SourceError(f"unknown source '{vid}' in config")
        out.append(klass(vcfg))
    return out


__all__ = [
    "SourceKind", "Provenance", "ORTHOGONALITY_WEIGHT",
    "ProjectionRow", "OwnershipRow",
    "ProjectionSource", "OwnershipSource", "SourceError",
    "BluecollarDFSProjections", "OddsImpliedProjections",
    "FTNSource", "SportsDataIOSource",
    "get_enabled_projection_sources",
]
