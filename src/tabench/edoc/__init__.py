"""EDOC-1 substrate: the external-dynamic-engine observational certificate.

Design: docs/design/adr-036-external-dynamic-observational-certificate.md.

The shared substrate for the dynamic-external engine rows (``sumo-duaiterate``
first, then ``matsim`` / ``dtalite-simulation``): the versioned canonicalization
module (:mod:`~tabench.edoc.canon`), the frozen ``EdocScenario`` family, the
occupancy-aware frozen-field builder + per-first-edge origin-wait profiles
(:mod:`~tabench.edoc.field`), the certifier-owned label-correcting time-dependent
shortest path (:mod:`~tabench.edoc.tdsp`), the pinned-engine replay-harness
types (:mod:`~tabench.edoc.replay`), and the P8 macrorep + bootstrap-CI harness
for the stochastic track (:mod:`~tabench.edoc.macrorep`, adr-036 R5). The
certificate itself (gates G0-G4 + the ``RG_D1`` frozen-field best-response gap)
lives in :mod:`tabench.metrics.edoc_gaps`.
"""

from __future__ import annotations

from .canon import (
    CANON_VERSION,
    canonicalize_matsim,
    canonicalize_sumo,
    hash_matsim_artifacts,
    hash_sumo_artifacts,
    is_hashed_artifact,
    is_hashed_matsim_artifact,
)
from .field import (
    FrozenField,
    OriginWaitProfile,
    build_field_from_records,
    build_origin_waits,
)
from .macrorep import MacrorepResult, certify_macroreps, per_seed_scenarios
from .replay import EmittedBundle, ReplayAgent, ReplayResult, assert_engine_pin
from .scenario import EdocScenario
from .tdsp import evaluate_route, td_shortest_path

__all__ = [
    "CANON_VERSION",
    "EdocScenario",
    "EmittedBundle",
    "FrozenField",
    "MacrorepResult",
    "OriginWaitProfile",
    "ReplayAgent",
    "ReplayResult",
    "assert_engine_pin",
    "build_field_from_records",
    "build_origin_waits",
    "canonicalize_matsim",
    "canonicalize_sumo",
    "certify_macroreps",
    "evaluate_route",
    "hash_matsim_artifacts",
    "hash_sumo_artifacts",
    "is_hashed_artifact",
    "is_hashed_matsim_artifact",
    "per_seed_scenarios",
    "td_shortest_path",
]
