"""DynamicScenario: frozen, content-hashed dynamic-network-loading instances.

Design: docs/design/adr-010-dnl-core.md. The scenario reuses the static
:class:`~tabench.core.scenario.Network` ONLY for topology and zone
conventions (``init_node``/``term_node``/``n_nodes``/``n_zones``/
``first_thru_node``); the static BPR fields (capacity, free_flow_time, b,
power, toll, ...) are IGNORED by DNL and EXCLUDED from the dnl content hash —
the kinematic-wave physics live in :class:`~tabench.dnl.fd.LinkDynamics`.
``core/scenario.py`` is never edited (only the public ``Network`` class is
imported, read-only), so no static content hash can move.

Boundary conventions (validated here, enforced by the loader): every zone
node (ids ``1..n_zones``) is a pure boundary in DNL — origins inject from a
vertical queue, destinations absorb — so zone centroids never carry through
traffic (the first_thru_node semantics of the TNTP convention). Turning
fractions therefore cover EXACTLY the interior (non-zone) nodes that
topologically need a split: >= 1 incoming and >= 2 outgoing links.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..core.scenario import Network
from .demand import DynamicDemand, TurningFractions
from .fd import LinkDynamics
from .grid import TimeGrid, assert_wave_resolved

__all__ = ["DynamicScenario"]


def _as_f64(x: np.ndarray) -> np.ndarray:
    # Local re-implementation (NOT imported from core.scenario): the only
    # coupling to the frozen static machinery is the public Network class.
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))


@dataclass(frozen=True)
class DynamicScenario:
    """Frozen, content-hashed dynamic loading instance (P2 for DNL).

    ``family`` names the scenario lineage used by the fairness gate (P7),
    like the static :class:`~tabench.core.scenario.Scenario`; it defaults to
    ``name`` and is provenance, never hashed content.

    ``turns`` must cover exactly the interior nodes with >= 1 incoming and
    >= 2 outgoing links (missing or extra entries raise); single-out nodes
    carry an implicit column of ones and zone nodes are boundaries (no
    through traffic, so no turn split can apply there). ``turns=None`` is the
    canonical spelling for "no turn data" (an empty :class:`TurningFractions`
    is rejected at its own construction, so ``None`` and "empty" cannot hash
    identically).
    """

    name: str
    network: Network  # existing class, imported read-only, NEVER modified
    dynamics: LinkDynamics
    demand: DynamicDemand
    grid: TimeGrid
    turns: TurningFractions | None = None
    family: str = ""  # P7 lineage semantics, like static Scenario

    def __post_init__(self) -> None:
        net = self.network
        if self.demand.n_zones != net.n_zones:
            raise ValueError(
                f"DynamicScenario '{self.name}': demand has {self.demand.n_zones} "
                f"zones, network declares {net.n_zones}"
            )
        if self.dynamics.n_links != net.n_links:
            raise ValueError(
                f"DynamicScenario '{self.name}': dynamics covers "
                f"{self.dynamics.n_links} links, network has {net.n_links}"
            )
        if not self.family:
            object.__setattr__(self, "family", self.name)

        in_deg = np.bincount(net.term_node, minlength=net.n_nodes + 1)
        out_deg = np.bincount(net.init_node, minlength=net.n_nodes + 1)

        # Every zone with positive production needs an outgoing link, every
        # zone with positive attraction an incoming one (else its vehicles
        # could never enter/arrive and conservation would be vacuous).
        by_origin = self.demand.rates.sum(axis=(0, 2))
        by_destination = self.demand.rates.sum(axis=(0, 1))
        for z in range(1, net.n_zones + 1):
            if by_origin[z - 1] > 0 and out_deg[z] == 0:
                raise ValueError(
                    f"DynamicScenario '{self.name}': zone {z} has positive "
                    "production but no outgoing link"
                )
            if by_destination[z - 1] > 0 and in_deg[z] == 0:
                raise ValueError(
                    f"DynamicScenario '{self.name}': zone {z} has positive "
                    "attraction but no incoming link"
                )

        # Turn coverage: EXACTLY the interior nodes that need a split.
        needed = {
            n
            for n in range(net.n_zones + 1, net.n_nodes + 1)
            if in_deg[n] >= 1 and out_deg[n] >= 2
        }
        given: dict[int, np.ndarray] = {}
        if self.turns is not None:
            given = dict(self.turns.frac)
        for node_id in given:
            if 1 <= node_id <= net.n_zones:
                raise ValueError(
                    f"DynamicScenario '{self.name}': turning fractions given for "
                    f"zone node {node_id} — zone centroids are boundaries in DNL "
                    "and carry no through traffic (first_thru_node semantics)"
                )
        missing = needed - set(given)
        if missing:
            raise ValueError(
                f"DynamicScenario '{self.name}': missing turning fractions for "
                f"diverge node(s) {sorted(missing)} (interior, >= 1 in and >= 2 "
                "out links)"
            )
        extra = set(given) - needed
        if extra:
            raise ValueError(
                f"DynamicScenario '{self.name}': extra turning-fraction entries "
                f"for node(s) {sorted(extra)} that need no split (single-out "
                "nodes carry an implicit column of ones)"
            )
        for node_id, matrix in given.items():
            expected = (int(in_deg[node_id]), int(out_deg[node_id]))
            if matrix.shape != expected:
                raise ValueError(
                    f"DynamicScenario '{self.name}': turning fractions at node "
                    f"{node_id} have shape {matrix.shape}, topology requires "
                    f"{expected} (ascending-sorted in/out link lists)"
                )

        assert_wave_resolved(
            self.grid,
            self.dynamics.length,
            self.dynamics.free_speed,
            self.dynamics.wave_speed,
        )

    def content_hash(self) -> str:
        """SHA-256 over the canonical serialization of all scored content.

        Domain-separated from the static hash (``"tabench-dnl-scenario-v1;"``
        prefix), covering topology, KW physics, demand, grid, and turns —
        and nothing else: the static BPR fields are not DNL content, so a
        BPR edit never moves a dnl hash (and vice versa the static golden
        hashes are safe by construction: no hashed static class is touched).
        ``init_node``/``term_node`` pass through the same float64 byte path
        (int64 -> float64 is exact for node ids); +inf hashes as its unique
        IEEE-754 pattern; NaN is rejected by validation before hashing.
        """
        h = hashlib.sha256()
        h.update(b"tabench-dnl-scenario-v1;")  # domain separation
        net = self.network
        h.update(f"nodes={net.n_nodes};zones={net.n_zones};ftn={net.first_thru_node};".encode())
        for label, arr in (
            ("init", net.init_node),
            ("term", net.term_node),
            ("len", self.dynamics.length),
            ("vf", self.dynamics.free_speed),
            ("w", self.dynamics.wave_speed),
            ("kj", self.dynamics.jam_density),
            ("qmax", self.dynamics.capacity),
            ("dbp", self.demand.breakpoints),
            ("drate", self.demand.rates),
        ):
            h.update(label.encode())
            h.update(_as_f64(arr).tobytes())
        h.update(f"dt={float(self.grid.dt)!r};K={self.grid.n_steps};".encode())
        if self.turns is not None:
            for node_id, m in self.turns.frac:
                h.update(f"turn{node_id}".encode())
                h.update(_as_f64(m).tobytes())
        return h.hexdigest()
