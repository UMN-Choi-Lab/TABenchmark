"""SODTAScenario: the Merchant & Nemhauser (1978) exit-function SO-DTA instance.

The first network DTA model (docs/design/adr-020-merchant-nemhauser.md): a
discrete-time, SINGLE-DESTINATION, system-optimal program routing time-varying
node demands ``d_j(t)`` through links whose per-period outflow is governed by
exit functions ``g_a(x_a)`` of the start-of-period occupancy. This is a
network-wide *optimization* paradigm — not the repo's route-choice equilibria
nor the dnl/ loading kernels — so it lives in its own parallel module (like
``transit/`` and ``bottleneck/``), touching no road/DNL code.

Exit functions are piecewise-linear concave, stored as the pointwise minimum of
affine pieces ``g_a(x) = min_p (slope_p * x + intercept_p)`` — the classical
piecewise-linearization under which the Carey (1987) relaxation of the program
is an LP (see ``solve.py``). Every link's BINDING intercept-0 piece must have
slope exactly 1 (``g(x) = x`` near empty), which enforces the standing M-N
assumptions ``g(0) = 0`` and ``g(x) <= x`` (nothing exits an empty link; outflow
never exceeds occupancy) AND clearability under the benchmark's terminal
convention — a slope < 1 decays occupancy geometrically and can never reach
``x(T) = 0`` for any horizon (links with longer free-flow times are modeled as
chains, the coordinated discretization of Carey & Watling 2012). Slopes >= 0
keep ``g`` nondecreasing, and a min of affine pieces is concave by construction.
Only the nondecreasing branch is representable — as in M-N. All arrays are
frozen read-only after validation so the content hash cannot silently desync.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

__all__ = ["SODTAScenario"]


def _as_f64(x) -> np.ndarray:
    # always copy: scenario arrays are frozen read-only, the caller's must not be
    return np.array(x, dtype=np.float64, order="C")


def _as_i64(x) -> np.ndarray:
    return np.array(x, dtype=np.int64, order="C")


@dataclass(frozen=True)
class SODTAScenario:
    """Frozen, content-hashed Merchant-Nemhauser instance (P2).

    Nodes are ``0..n_nodes-1``; link ``a`` runs ``link_tail[a] -> link_head[a]``.
    ``exit_pieces[a]`` is a tuple of ``(slope, intercept)`` affine pieces whose
    pointwise minimum is ``g_a``. ``demand[t, j]`` is the exogenous demand
    (vehicles, all bound for ``destination``) generated at node ``j`` during
    period ``t`` — its first dimension sets the horizon ``T``. ``cost_weights``
    are the per-link linear cost rates ``w_a > 0`` in the objective
    ``sum_t sum_a w_a * x_a(t)`` (all-ones = total system travel time in
    vehicle-periods). ``family`` is P7 lineage (provenance only, unhashed).
    """

    name: str
    n_nodes: int
    destination: int
    link_tail: np.ndarray  # (n_links,) int64
    link_head: np.ndarray  # (n_links,) int64
    exit_pieces: tuple[tuple[tuple[float, float], ...], ...]  # per link: ((slope, icpt), ...)
    demand: np.ndarray  # (T, n_nodes) float64, >= 0, zero at destination
    cost_weights: np.ndarray | None = None  # (n_links,) float64 > 0; default all-ones
    family: str = field(default="")

    def __post_init__(self) -> None:
        tail = _as_i64(self.link_tail)
        head = _as_i64(self.link_head)
        dem = _as_f64(self.demand)
        w = np.ones(tail.size) if self.cost_weights is None else _as_f64(self.cost_weights)
        pieces = tuple(
            tuple((float(s), float(c)) for s, c in link_pieces)
            for link_pieces in self.exit_pieces
        )
        object.__setattr__(self, "link_tail", tail)
        object.__setattr__(self, "link_head", head)
        object.__setattr__(self, "demand", dem)
        object.__setattr__(self, "cost_weights", w)
        object.__setattr__(self, "exit_pieces", pieces)
        if not self.family:
            object.__setattr__(self, "family", self.name)

        name = self.name
        if self.n_nodes < 2:
            raise ValueError(f"SODTAScenario '{name}': need >= 2 nodes")
        if not (0 <= self.destination < self.n_nodes):
            raise ValueError(f"SODTAScenario '{name}': destination out of range")
        n_links = tail.size
        if head.size != n_links or w.size != n_links or len(pieces) != n_links:
            raise ValueError(
                f"SODTAScenario '{name}': link_tail/link_head/exit_pieces/cost_weights "
                "must have equal length"
            )
        if n_links == 0:
            raise ValueError(f"SODTAScenario '{name}': need >= 1 link")
        ends = np.concatenate([tail, head])
        if ends.min() < 0 or ends.max() >= self.n_nodes:
            raise ValueError(f"SODTAScenario '{name}': link endpoints out of range")
        if np.any(tail == head):
            raise ValueError(f"SODTAScenario '{name}': self-loop links not allowed")
        if np.any(tail == self.destination):
            raise ValueError(
                f"SODTAScenario '{name}': links out of the destination are not allowed "
                "(the destination is absorbing)"
            )
        if not np.all(np.isfinite(w)) or np.any(w <= 0.0):
            raise ValueError(
                f"SODTAScenario '{name}': cost_weights must be finite and > 0 "
                "(zero-cost links would let flow strand at no cost)"
            )
        for a, link_pieces in enumerate(pieces):
            if not link_pieces:
                raise ValueError(f"SODTAScenario '{name}': link {a} has no exit pieces")
            for s, c in link_pieces:
                if not (np.isfinite(s) and np.isfinite(c)) or s < 0.0 or c < 0.0:
                    raise ValueError(
                        f"SODTAScenario '{name}': link {a} exit pieces need finite "
                        "slope >= 0 and intercept >= 0 (nondecreasing branch only)"
                    )
            zero_slopes = [s for s, c in link_pieces if c == 0.0]
            if not zero_slopes or min(zero_slopes) != 1.0:
                raise ValueError(
                    f"SODTAScenario '{name}': link {a} needs its binding intercept-0 "
                    "exit piece to have slope exactly 1 (g(x) = x near empty: enforces "
                    "g(0) = 0, g(x) <= x, AND one-period uncongested clearance — a "
                    "slope < 1 decays geometrically and can never satisfy terminal "
                    "clearance x(T) = 0 for ANY horizon; model longer free-flow times "
                    "as chains of links, the coordinated discretization)"
                )
        if dem.ndim != 2 or dem.shape[1] != self.n_nodes:
            raise ValueError(f"SODTAScenario '{name}': demand must be (T, n_nodes)")
        if dem.shape[0] < 1:
            raise ValueError(f"SODTAScenario '{name}': need horizon T >= 1")
        if not np.all(np.isfinite(dem)) or np.any(dem < 0.0):
            raise ValueError(f"SODTAScenario '{name}': demand must be finite and >= 0")
        if np.any(dem[:, self.destination] != 0.0):
            raise ValueError(f"SODTAScenario '{name}': demand at the destination must be 0")
        if dem.sum() <= 0.0:
            raise ValueError(f"SODTAScenario '{name}': total demand must be > 0")
        # Every node generating demand must reach the destination (BFS on links),
        # else the program is trivially infeasible under terminal clearance.
        reaches = np.zeros(self.n_nodes, dtype=bool)
        reaches[self.destination] = True
        frontier = [self.destination]
        into: dict[int, list[int]] = {}
        for a in range(n_links):
            into.setdefault(int(head[a]), []).append(int(tail[a]))
        while frontier:
            j = frontier.pop()
            for i in into.get(j, ()):
                if not reaches[i]:
                    reaches[i] = True
                    frontier.append(i)
        bad = np.nonzero(dem.sum(axis=0) > 0.0)[0]
        bad = bad[~reaches[bad]]
        if bad.size:
            raise ValueError(
                f"SODTAScenario '{name}': demand nodes {bad.tolist()} cannot reach "
                "the destination"
            )
        # Freeze the arrays: the content hash and any evaluator's cached reference
        # optimum are only trustworthy if the instance truly cannot move.
        for arr in (tail, head, dem, w):
            arr.flags.writeable = False

    @property
    def n_links(self) -> int:
        return self.link_tail.size

    @property
    def n_periods(self) -> int:
        return self.demand.shape[0]

    def exit_flow(self, occupancies: np.ndarray) -> np.ndarray:
        """``g_a`` applied columnwise: ``out[..., a] = min_p (s_p * x[..., a] + c_p)``.

        ``occupancies``' last axis indexes links; any leading shape is allowed.
        """
        x = _as_f64(occupancies)
        if x.shape[-1] != self.n_links:
            raise ValueError("occupancies last axis must index the links")
        out = np.empty_like(x)
        for a, link_pieces in enumerate(self.exit_pieces):
            vals = [s * x[..., a] + c for s, c in link_pieces]
            out[..., a] = vals[0] if len(vals) == 1 else np.minimum.reduce(vals)
        return out

    def content_hash(self) -> str:
        """SHA-256 over the canonical serialization, domain-separated from every
        other scenario space (``"tabench-dta-scenario-v1;"`` prefix)."""
        h = hashlib.sha256()
        h.update(b"tabench-dta-scenario-v1;")
        h.update(f"nodes={self.n_nodes};dest={self.destination};".encode())
        for label, arr in (
            ("tail", self.link_tail),
            ("head", self.link_head),
            ("w", self.cost_weights),
            ("demand", self.demand),
        ):
            h.update(label.encode())
            h.update(_as_f64(arr).tobytes())
        for a, link_pieces in enumerate(self.exit_pieces):
            h.update(f"g{a}=".encode())
            for s, c in link_pieces:
                h.update(f"({s!r},{c!r})".encode())
            h.update(b";")
        return h.hexdigest()
