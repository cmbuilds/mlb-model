"""dfs/contracts.py — shared dataclasses and enums for the DFS module."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

# Re-export from api_external so the rest of dfs/ imports from one place.
from dfs.sources.api_external import (
    SourceKind, Provenance, ORTHOGONALITY_WEIGHT,
    ProjectionRow, OwnershipRow, SourceError,
)


class ConfidenceState(str, Enum):
    CONFIDENT  = "CONFIDENT"   # model core measured + lineup confirmed
    FLAGGED    = "FLAGGED"     # proxy data or missing source
    EXCLUDED   = "EXCLUDED"    # lineup unconfirmed / player unmatched


@dataclass
class ConsensusRow:
    """One player row in the consensus board — the output of consensus.py."""
    name: str
    team: str
    opponent: str
    position: str             # "C","1B","2B","3B","SS","OF","SP","RP"
    site: str                 # "fd" | "dk"
    salary: int               # 0 if not yet loaded
    lineup_slot: int          # 0 = unknown

    # Projections
    consensus_pts: float
    consensus_value: float    # pts / (salary/1000)
    model_pts: float
    model_ceiling: float
    model_floor: float

    # State
    state: ConfidenceState
    source_count: int         # how many sources contributed to consensus_pts
    sources_used: List[str]   # e.g. ["model"]
    flagged_reason: str       # empty string if CONFIDENT

    # Ownership (always modeled until D2 external source)
    own_pct: float            # projected %
    own_provenance: Provenance  # Provenance.MODELED tonight

    # Divergence
    divergence: float         # spread across sources (0 if single-source)
    divergence_flag: bool

    # Extras (from plays dict)
    bettable: bool
    batter_hand: str
    sp_name: str
    sp_hand: str
    park: str
    implied_total: float
    game_id: str
    score: float              # TB model 0-100 score (confidence modifier)
    hr_score: float
    dq_score: int

    # Stack group (filled by consensus.py after grouping)
    stack_score: float = 0.0


__all__ = [
    "SourceKind", "Provenance", "ORTHOGONALITY_WEIGHT",
    "ProjectionRow", "OwnershipRow", "SourceError",
    "ConfidenceState", "ConsensusRow",
]
