"""Scenario objects: frozen, content-hashed, declarative problem instances.

Design principle P2 (docs/ARCHITECTURE.md): a scenario is data, never code.
Node ids follow the TNTP convention: 1-based, zones are nodes ``1..n_zones``,
and nodes with id below ``first_thru_node`` are zone centroids that cannot
carry through traffic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Network", "Demand", "ReferenceSolution", "Scenario"]


def _as_f64(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float64)


@dataclass(frozen=True)
class Network:
    """Directed road network with BPR-family link performance functions.

    The generalized link cost (in time units) is

        t_a(v) = fft_a * (1 + b_a * (v / cap_a)^power_a)
                 + toll_weight * toll_a + distance_weight * length_a

    where the toll and distance weights are per-network metadata (they are
    NOT stored in ``.tntp`` files; see docs/ARCHITECTURE.md P9).

    ``units`` records per-network unit conventions (e.g. Sioux Falls free-flow
    times are 0.01 hours) as documentation; costs are always computed in the
    network's native units.
    """

    name: str
    n_nodes: int
    n_zones: int
    first_thru_node: int
    init_node: np.ndarray  # (n_links,) int64, 1-based tail node ids
    term_node: np.ndarray  # (n_links,) int64, 1-based head node ids
    capacity: np.ndarray
    length: np.ndarray
    free_flow_time: np.ndarray
    b: np.ndarray  # BPR "alpha"
    power: np.ndarray  # BPR "beta"
    toll: np.ndarray
    link_type: np.ndarray
    toll_weight: float = 0.0
    distance_weight: float = 0.0
    units: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        n = self.n_links
        for name in ("term_node", "capacity", "length", "free_flow_time", "b", "power", "toll"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"Network '{self.name}': column {name!r} has wrong length")
        pairs = set(zip(self.init_node.tolist(), self.term_node.tolist(), strict=True))
        if len(pairs) != n:
            raise ValueError(
                f"Network '{self.name}' contains parallel (duplicate) links; "
                "TABenchmark v0 requires unique (init_node, term_node) pairs "
                "(see the known-defects registry for affected TNTP networks)."
            )
        if np.any(self.free_flow_time <= 0):
            raise ValueError(
                f"Network '{self.name}': free_flow_time must be strictly positive "
                "(zero-cost links break shortest-path sparsity)."
            )
        if np.any(self.capacity <= 0):
            raise ValueError(f"Network '{self.name}': capacity must be strictly positive")
        if np.any((self.init_node < 1) | (self.init_node > self.n_nodes)) or np.any(
            (self.term_node < 1) | (self.term_node > self.n_nodes)
        ):
            raise ValueError(f"Network '{self.name}': node ids out of range 1..n_nodes")

    @property
    def n_links(self) -> int:
        return len(self.init_node)

    @property
    def fixed_cost(self) -> np.ndarray:
        """Flow-independent generalized-cost component per link."""
        return self.toll_weight * self.toll + self.distance_weight * self.length

    def link_cost(self, link_flows: np.ndarray) -> np.ndarray:
        """Generalized link travel cost t_a(v_a) at the given flows."""
        v = np.maximum(np.asarray(link_flows, dtype=np.float64), 0.0)
        ratio = v / self.capacity
        return self.free_flow_time * (1.0 + self.b * ratio**self.power) + self.fixed_cost

    def link_cost_integral(self, link_flows: np.ndarray) -> np.ndarray:
        """Per-link Beckmann integral: integral of t_a(s) ds from 0 to v_a."""
        v = np.maximum(np.asarray(link_flows, dtype=np.float64), 0.0)
        ratio = v / self.capacity
        variable = self.free_flow_time * (v + self.b * v * ratio**self.power / (self.power + 1.0))
        return variable + self.fixed_cost * v


@dataclass(frozen=True)
class Demand:
    """Origin-destination demand. ``matrix[i, j]`` is flow from zone i+1 to j+1."""

    matrix: np.ndarray  # (n_zones, n_zones) float64

    def __post_init__(self) -> None:
        m = self.matrix
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("Demand matrix must be square (n_zones, n_zones)")
        if np.any(m < 0):
            raise ValueError("Demand must be nonnegative")

    @property
    def n_zones(self) -> int:
        return self.matrix.shape[0]

    @property
    def total(self) -> float:
        return float(self.matrix.sum())


@dataclass(frozen=True)
class ReferenceSolution:
    """Best-known solution used as a regression oracle (with provenance)."""

    link_flows: np.ndarray
    source: str
    note: str = ""


@dataclass(frozen=True)
class Scenario:
    """A frozen benchmark instance: network + demand (+ optional oracle).

    ``family`` names the scenario lineage used by the fairness gate (P7):
    a learned model whose ``trained_on`` includes this family is refused
    evaluation on this scenario.
    """

    name: str
    network: Network
    demand: Demand
    reference: ReferenceSolution | None = None
    family: str = field(default="")

    def __post_init__(self) -> None:
        if self.demand.n_zones != self.network.n_zones:
            raise ValueError(
                f"Scenario '{self.name}': demand has {self.demand.n_zones} zones, "
                f"network declares {self.network.n_zones}"
            )
        if not self.family:
            object.__setattr__(self, "family", self.name)

    def content_hash(self) -> str:
        """SHA-256 over the canonical serialization of all scored content (P2)."""
        h = hashlib.sha256()
        net = self.network
        h.update(
            f"nodes={net.n_nodes};zones={net.n_zones};ftn={net.first_thru_node};"
            f"tw={net.toll_weight!r};dw={net.distance_weight!r};".encode()
        )
        for label, arr in (
            ("init", net.init_node),
            ("term", net.term_node),
            ("cap", net.capacity),
            ("len", net.length),
            ("fft", net.free_flow_time),
            ("b", net.b),
            ("pow", net.power),
            ("toll", net.toll),
            ("od", self.demand.matrix),
        ):
            h.update(label.encode())
            h.update(_as_f64(arr).tobytes())
        return h.hexdigest()
