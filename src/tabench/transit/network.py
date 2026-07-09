"""Transit network primitives for the Spiess & Florian (1989) optimal-strategy
assignment (docs/design/adr-014-transit-strategy.md).

A transit network is a directed **multigraph** whose arcs each carry an
in-vehicle / traversal time ``c_a >= 0`` and a service frequency
``f_a in (0, inf]``. A finite frequency is a boardable line (the passenger waits
an expected ``1/f_a`` before the first vehicle arrives); ``f_a = inf`` is a
deterministic arc (walk, transfer, or in-vehicle continuation — no wait).
Parallel arcs (several lines serving the same ordered stop pair) are ALLOWED and
are exactly the "common lines" the model is about — which is why this is a fresh
container and NOT the road :class:`~tabench.core.scenario.Network` (whose
``__post_init__`` forbids parallel links and whose costs are flow-dependent BPR;
transit here is uncongested / frequency-based). ``core/scenario.py`` is never
touched, so no static content hash can move.

Node ids are 0-based ``0..n_nodes-1`` (the transit graph has no zone-centroid
through-traffic convention; every node may be an origin, destination, or
interchange).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

__all__ = [
    "TransitNetwork",
    "TransitDemand",
    "TransitScenario",
    "TransitStrategy",
    "TransitReference",
]


def _as_f64(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))


def _as_i64(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.int64))


@dataclass(frozen=True)
class TransitNetwork:
    """Directed transit multigraph: arc ``a`` runs ``tail[a] -> head[a]`` with
    in-vehicle time ``time[a] >= 0`` and frequency ``freq[a] in (0, inf]``
    (``inf`` = deterministic, no wait)."""

    n_nodes: int
    tail: np.ndarray  # (n_arcs,) int64, 0-based
    head: np.ndarray  # (n_arcs,)
    time: np.ndarray  # (n_arcs,) float64, in-vehicle/traversal time >= 0
    freq: np.ndarray  # (n_arcs,) float64, frequency in (0, inf]

    def __post_init__(self) -> None:
        tail = _as_i64(self.tail)
        head = _as_i64(self.head)
        time = _as_f64(self.time)
        freq = _as_f64(self.freq)
        object.__setattr__(self, "tail", tail)
        object.__setattr__(self, "head", head)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "freq", freq)
        n = tail.size
        if not (head.size == n == time.size == freq.size):
            raise ValueError("TransitNetwork tail/head/time/freq must be equal length")
        if self.n_nodes < 1:
            raise ValueError("TransitNetwork needs >= 1 node")
        if n:
            ends = np.concatenate([tail, head])
            if ends.min() < 0 or ends.max() >= self.n_nodes:
                raise ValueError("TransitNetwork arc endpoints out of range 0..n_nodes-1")
        if np.any(time < 0) or not np.all(np.isfinite(time)):
            raise ValueError("TransitNetwork arc times must be finite and >= 0")
        # Frequency: strictly positive; +inf is the deterministic sentinel (allowed),
        # NaN / <=0 are not.
        if np.any(np.isnan(freq)) or np.any(freq <= 0):
            raise ValueError("TransitNetwork frequencies must be > 0 (inf = deterministic)")

    @property
    def n_arcs(self) -> int:
        return self.tail.size


@dataclass(frozen=True)
class TransitDemand:
    """Trip volumes as ``(origin, destination, volume)`` triples (0-based node ids)."""

    origins: np.ndarray  # (n_pairs,) int64
    destinations: np.ndarray  # (n_pairs,)
    volumes: np.ndarray  # (n_pairs,) float64 >= 0

    def __post_init__(self) -> None:
        o_raw = np.asarray(self.origins)
        d_raw = np.asarray(self.destinations)
        o = _as_i64(self.origins)
        d = _as_i64(self.destinations)
        v = _as_f64(self.volumes)
        object.__setattr__(self, "origins", o)
        object.__setattr__(self, "destinations", d)
        object.__setattr__(self, "volumes", v)
        if not (o.size == d.size == v.size):
            raise ValueError("TransitDemand origins/destinations/volumes must be equal length")
        if np.any(v < 0) or not np.all(np.isfinite(v)):
            raise ValueError("TransitDemand volumes must be finite and >= 0")
        if o.size:
            # Node ids must be exact nonnegative integers: a fractional id would be
            # silently truncated by the int64 cast, and a negative id would wrap to
            # the wrong node under numpy indexing (both -> a bogus but well-formed
            # solve). The upper bound is checked in TransitScenario (needs n_nodes).
            if not (np.array_equal(o, o_raw) and np.array_equal(d, d_raw)):
                raise ValueError("TransitDemand origins/destinations must be integer node ids")
            if o.min() < 0 or d.min() < 0:
                raise ValueError("TransitDemand node ids must be >= 0")
            if np.any(o == d):
                raise ValueError("TransitDemand: origin == destination (intrazonal) not allowed")

    @property
    def n_pairs(self) -> int:
        return self.origins.size

    @property
    def total(self) -> float:
        return float(self.volumes.sum())


@dataclass(frozen=True)
class TransitStrategy:
    """One emitted optimal-strategy solution (the transit analogue of FlowState).

    ``arc_volumes`` are the per-arc passenger volumes; ``labels`` are the
    per-destination expected-cost-to-destination vectors ``u`` (one ``(n_nodes,)``
    array per distinct demand destination), carried as an immutable tuple so the
    harness can recompute the optimality residual from them (P1). ``pair_costs``
    is the optimal expected cost from each demand pair's origin, aligned with the
    scenario's demand pairs. ``dest_arc_volumes`` is the PER-DESTINATION arc-volume
    decomposition ``(destination, v^d)`` — needed to certify the wait term, which
    is per-(node, destination): a node shared by two destinations can carry
    differently-weighted flow for each, so the harness cannot recover the correct
    waits from the summed ``arc_volumes`` alone. It defaults to empty; the
    certifier then falls back to treating the aggregate as a single destination
    (valid only when the scenario has one destination).
    """

    arc_volumes: np.ndarray  # (n_arcs,) summed over destinations
    labels: tuple[tuple[int, np.ndarray], ...]  # (destination, u[0..n_nodes-1]) per dest
    pair_costs: np.ndarray  # (n_pairs,)
    dest_arc_volumes: tuple[tuple[int, np.ndarray], ...] = ()  # (destination, v^d) per dest


@dataclass(frozen=True)
class TransitReference:
    """Best-known / analytic solution for regression (with provenance)."""

    expected_total_cost: float  # sum_pairs volume * u[origin]
    source: str
    note: str = ""


@dataclass(frozen=True)
class TransitScenario:
    """Frozen, content-hashed transit-assignment instance (P2 for transit)."""

    name: str
    network: TransitNetwork
    demand: TransitDemand
    family: str = ""
    reference: TransitReference | None = None

    def __post_init__(self) -> None:
        net = self.network
        dem = self.demand
        if dem.n_pairs:
            if dem.origins.max() >= net.n_nodes or dem.destinations.max() >= net.n_nodes:
                raise ValueError(
                    f"TransitScenario '{self.name}': demand node id out of range "
                    f"0..{net.n_nodes - 1}"
                )
        if not self.family:
            object.__setattr__(self, "family", self.name)

    def content_hash(self) -> str:
        """SHA-256 over the canonical serialization, domain-separated from the
        static / DNL hashes (``"tabench-transit-scenario-v1;"`` prefix). ``+inf``
        frequencies hash as their unique IEEE-754 pattern."""
        h = hashlib.sha256()
        h.update(b"tabench-transit-scenario-v1;")
        net = self.network
        h.update(f"nodes={net.n_nodes};".encode())
        for label, arr in (
            ("tail", net.tail),
            ("head", net.head),
            ("time", net.time),
            ("freq", net.freq),
            ("dorig", self.demand.origins),
            ("ddest", self.demand.destinations),
            ("dvol", self.demand.volumes),
        ):
            h.update(label.encode())
            h.update(_as_f64(arr).tobytes())
        return h.hexdigest()
