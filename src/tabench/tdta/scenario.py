"""TDTAScenario: frozen, content-hashed time-dependent traffic-assignment instances.

Design: docs/design/adr-031-peeta-mahmassani.md. A ``TDTAScenario`` composes the
DNL primitives read-only — :class:`~tabench.core.scenario.Network` (topology),
:class:`~tabench.dnl.fd.LinkDynamics`, :class:`~tabench.dnl.demand.DynamicDemand`,
:class:`~tabench.dnl.grid.TimeGrid` — and ADDS the one thing dynamic network
loading cannot express (ADR-010's exogenous time-invariant ``TurningFractions``
is not route choice): an enumerated per-OD **path set** whose per-departure-
interval split is the model's decision variable (Peeta & Mahmassani 1995's fixed
departure times, route choice only). It also pins the loading ``kernel``
(``"ctm"`` or ``"ltm"``) the certificate is defined against.

Hash domain ``"tabench-tdta-scenario-v1;"`` — domain-separated from every other
scenario hash, so no existing dnl/dta/static hash can move.

The v1 topology restriction (the decidability guarantee): the union of declared
path links has **no effective interior diverge** — at every interior node, each
incoming link is followed by exactly one outgoing link across all paths (one-hot
turn rows), so paths branch only at their origins and may only merge downstream.
Per-commodity experienced times are then EXACTLY decidable from aggregate link
curves (the origin branch is model-controlled and observed; a merge preserves
attribution because each incoming link's outflow is observed separately, and FIFO
does the rest), so ADR-010's C8 multi-in undecidability is designed out, not
tolerated. General interior diverges (per-commodity emission + time-varying
turns, ADR-010 R7/R8) are the named v2 with a domain-string bump.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from ..core.scenario import Network
from ..dnl.ctm import CTMLink
from ..dnl.demand import DynamicDemand
from ..dnl.fd import LinkDynamics
from ..dnl.grid import TimeGrid, assert_wave_resolved
from ..dnl.link import LinkModelFactory
from ..dnl.ltm import LTMLink

__all__ = ["TDPath", "TDTAScenario"]

_KERNELS = {"ctm": CTMLink, "ltm": LTMLink}

# float64 equilibrium-conditioning floor on total demand (Dossier B entry 9):
# below this a route-swap gap cannot be resolved. Matches the default certifier
# tolerance so a scenario the certifier could only censor never constructs.
_MIN_DEMAND = 1e-6


@dataclass(frozen=True)
class TDPath:
    """One declared route: an OD pair plus its ordered link chain.

    ``origin``/``destination`` are 1-based zone ids; ``links`` is a tuple of
    0-based link indices forming a connected chain from ``origin`` to
    ``destination``. Scenarios are data (P2): the path set is enumerated
    content, hashed exactly like vi-due's route arrays.
    """

    origin: int
    destination: int
    links: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", int(self.origin))
        object.__setattr__(self, "destination", int(self.destination))
        object.__setattr__(self, "links", tuple(int(a) for a in self.links))
        if len(self.links) < 1:
            raise ValueError("TDPath must contain at least one link")

    @property
    def od(self) -> tuple[int, int]:
        return (self.origin, self.destination)

    @property
    def first_link(self) -> int:
        return self.links[0]


@dataclass(frozen=True)
class TDTAScenario:
    """Frozen, content-hashed time-dependent route-choice instance (P2)."""

    name: str
    network: Network  # existing class, imported read-only, NEVER modified
    dynamics: LinkDynamics
    demand: DynamicDemand
    grid: TimeGrid
    paths: tuple[TDPath, ...]
    kernel: str = "ctm"
    family: str = ""

    def __post_init__(self) -> None:
        net = self.network
        if self.demand.n_zones != net.n_zones:
            raise ValueError(
                f"TDTAScenario '{self.name}': demand has {self.demand.n_zones} zones, "
                f"network declares {net.n_zones}"
            )
        if self.dynamics.n_links != net.n_links:
            raise ValueError(
                f"TDTAScenario '{self.name}': dynamics covers {self.dynamics.n_links} "
                f"links, network has {net.n_links}"
            )
        if self.kernel not in _KERNELS:
            raise ValueError(
                f"TDTAScenario '{self.name}': kernel must be one of {sorted(_KERNELS)}, "
                f"got {self.kernel!r}"
            )
        object.__setattr__(self, "paths", tuple(self.paths))
        if len(self.paths) < 1:
            raise ValueError(f"TDTAScenario '{self.name}': needs >= 1 path")
        if not self.family:
            object.__setattr__(self, "family", self.name)

        # Demand must not extend past the grid horizon (review MINOR): the
        # artifact has no columns for the post-horizon tail, so a conforming
        # emission can never carry it and every honest plan would silently
        # censor. Raise at construction (the ADR-020/021 eager-config discipline).
        if float(self.demand.breakpoints[-1]) > self.grid.horizon * (1.0 + 1e-9):
            raise ValueError(
                f"TDTAScenario '{self.name}': demand extends to t="
                f"{float(self.demand.breakpoints[-1])!r} beyond the grid horizon "
                f"{self.grid.horizon!r} — the emission has no columns for the tail"
            )
        # Degenerate-demand conditioning gate (review MINOR / Dossier B entry 9):
        # a total demand at/below the float64 conditioning floor cannot resolve an
        # equilibrium; raise at construction rather than only censoring at scoring.
        if self.demand.total() <= _MIN_DEMAND:
            raise ValueError(
                f"TDTAScenario '{self.name}': total demand {self.demand.total()!r} is "
                f"at/below the conditioning floor {_MIN_DEMAND!r}; the equilibrium is "
                "unresolvable at float64"
            )

        assert_wave_resolved(
            self.grid, self.dynamics.length, self.dynamics.free_speed, self.dynamics.wave_speed
        )
        self._validate_paths()
        self._validate_kernel_alignment()

    # ---------------------------------------------------------------- validation

    def _validate_paths(self) -> None:
        net = self.network
        n_zones = net.n_zones
        seen: set[tuple[int, tuple[int, ...]]] = set()
        first_links: dict[int, int] = {}
        # interior node -> {in_link -> out_link} used across all paths (one-hot)
        node_in_out: dict[int, dict[int, int]] = {}
        for pi, p in enumerate(self.paths):
            if not (1 <= p.origin <= n_zones) or not (1 <= p.destination <= n_zones):
                raise ValueError(
                    f"TDTAScenario '{self.name}': path {pi} OD ({p.origin}, "
                    f"{p.destination}) is not a zone pair (1..{n_zones})"
                )
            if p.origin == p.destination:
                raise ValueError(
                    f"TDTAScenario '{self.name}': path {pi} is intrazonal ({p.origin})"
                )
            key = (p.origin, p.links)
            if key in seen:
                raise ValueError(
                    f"TDTAScenario '{self.name}': duplicate path {p.links} for origin {p.origin}"
                )
            seen.add(key)
            for a in p.links:
                if not (0 <= a < net.n_links):
                    raise ValueError(
                        f"TDTAScenario '{self.name}': path {pi} references link {a} "
                        f"outside 0..{net.n_links - 1}"
                    )
            if int(net.init_node[p.links[0]]) != p.origin:
                raise ValueError(
                    f"TDTAScenario '{self.name}': path {pi} first link {p.links[0]} does "
                    f"not start at origin {p.origin}"
                )
            if int(net.term_node[p.links[-1]]) != p.destination:
                raise ValueError(
                    f"TDTAScenario '{self.name}': path {pi} last link {p.links[-1]} does "
                    f"not end at destination {p.destination}"
                )
            for a, b in zip(p.links, p.links[1:], strict=False):
                node = int(net.term_node[a])
                if int(net.init_node[b]) != node:
                    raise ValueError(
                        f"TDTAScenario '{self.name}': path {pi} is disconnected between "
                        f"links {a} and {b}"
                    )
                if node <= n_zones:
                    raise ValueError(
                        f"TDTAScenario '{self.name}': path {pi} routes THROUGH zone node "
                        f"{node} (zones are boundaries; through traffic is forbidden)"
                    )
                prior = node_in_out.setdefault(node, {}).get(a)
                if prior is not None and prior != b:
                    raise ValueError(
                        f"TDTAScenario '{self.name}': interior diverge at node {node} "
                        f"(in-link {a} feeds both out-links {prior} and {b}); the v1 "
                        "certificate requires one-hot turn rows (paths branch only at "
                        "origins). General interior diverges are the v2 extension."
                    )
                node_in_out[node][a] = b
            fl = p.first_link
            if fl in first_links and first_links[fl] != pi:
                raise ValueError(
                    f"TDTAScenario '{self.name}': first link {fl} is shared by paths "
                    f"{first_links[fl]} and {pi} — per-path origin injection needs a "
                    "private first link per path"
                )
            first_links[fl] = pi

        # OD coverage: exactly the positive-demand ODs carry declared paths.
        demand_ods = {
            (int(i) + 1, int(j) + 1)
            for i, j in zip(*np.nonzero(self.demand.rates.sum(axis=0)), strict=False)
        }
        path_ods = {p.od for p in self.paths}
        if demand_ods != path_ods:
            raise ValueError(
                f"TDTAScenario '{self.name}': declared path ODs {sorted(path_ods)} do "
                f"not match positive-demand ODs {sorted(demand_ods)} (every served OD "
                "needs >= 1 path and no path may serve a zero-demand OD)"
            )

    def _validate_kernel_alignment(self) -> None:
        # Build one link model per USED link at construction: this raises exactly
        # the kernel's own construction errors (CTM cell alignment / w<=vf, LTM
        # finite jam) up front, so a scenario that a loader could not run cannot
        # be constructed.
        factory = self.link_factory
        grid = self.grid
        for a in self.used_links():
            factory(self.dynamics.fd(a), float(self.dynamics.length[a]), grid)

    # ---------------------------------------------------------------- accessors

    @property
    def link_factory(self) -> LinkModelFactory:
        return _KERNELS[self.kernel]

    @property
    def n_paths(self) -> int:
        return len(self.paths)

    def used_links(self) -> list[int]:
        """Ascending list of link indices that appear in at least one path."""
        used: set[int] = set()
        for p in self.paths:
            used.update(p.links)
        return sorted(used)

    def first_link_of(self) -> np.ndarray:
        """``(n_paths,)`` int array of each path's private first link."""
        return np.array([p.first_link for p in self.paths], dtype=np.int64)

    def paths_by_od(self) -> dict[tuple[int, int], list[int]]:
        """Map each OD pair to the list of declared path indices serving it."""
        out: dict[tuple[int, int], list[int]] = {}
        for pi, p in enumerate(self.paths):
            out.setdefault(p.od, []).append(pi)
        return out

    def destinations(self) -> list[int]:
        """Ascending list of distinct destination zones over all paths."""
        return sorted({p.destination for p in self.paths})

    @property
    def single_destination(self) -> bool:
        return len(self.destinations()) == 1

    def declared_paths_omitting_shortest(self, tol: float = 1e-9) -> list[tuple[int, int]]:
        """ODs whose DECLARED path set omits a free-flow shortest path (NON-gating,
        review MINOR). The certificate scores relative to the DECLARED path
        universe (§adr-031): an all-on-A plan on a network that also contains an
        idle byte-identical route B certifies gap 0 if B is not declared, and Z*
        itself contracts to the used links. Restricted choice sets are legitimate
        scenario design, so this is a diagnostic (the builtin anchors assert it
        returns empty), NOT a constructor gate.

        Returns the list of OD pairs where the minimum free-flow time over the
        OD's declared paths strictly exceeds the network's free-flow shortest-path
        time (a faster physical route exists but is not declared)."""
        net = self.network
        ff = self.dynamics.length / self.dynamics.free_speed  # per-link free-flow time
        # adjacency: node -> list of (head_node, ff_weight)
        adj: dict[int, list[tuple[int, float]]] = {}
        for a in range(net.n_links):
            adj.setdefault(int(net.init_node[a]), []).append((int(net.term_node[a]), float(ff[a])))
        bad: list[tuple[int, int]] = []
        for od, plist in self.paths_by_od().items():
            net_min = self._dijkstra_ff(adj, od[0], od[1], net.n_nodes)
            decl_min = min(sum(float(ff[a]) for a in self.paths[p].links) for p in plist)
            if math.isfinite(net_min) and decl_min > net_min + tol:
                bad.append(od)
        return bad

    @staticmethod
    def _dijkstra_ff(
        adj: dict[int, list[tuple[int, float]]], src: int, dst: int, n_nodes: int
    ) -> float:
        """Free-flow shortest-path time src -> dst (``math.inf`` if unreachable)."""
        import heapq

        dist = {src: 0.0}
        pq: list[tuple[float, int]] = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                return d
            if d > dist.get(u, math.inf):
                continue
            for v, w in adj.get(u, ()):
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist.get(dst, math.inf)

    def node_turns(self) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Per interior node used by the paths: ``(in_links, out_links, turns)``
        with ``in_links``/``out_links`` ascending and ``turns`` the one-hot
        row-stochastic matrix implied by the (interior-diverge-free) path set.

        A merge (several in-links, one used out-link) gets a column of ones; a
        genuine used diverge is impossible by the v1 restriction, so every row is
        one-hot. Feeds the per-path loader's :class:`~tabench.dnl.node.TampereNode`.
        """
        net = self.network
        # collect, per interior node, the used in->out adjacencies
        adj: dict[int, dict[int, int]] = {}
        for p in self.paths:
            for a, b in zip(p.links, p.links[1:], strict=False):
                node = int(net.term_node[a])
                adj.setdefault(node, {})[a] = b
        result: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for node, mapping in adj.items():
            ins = np.array(sorted(np.flatnonzero(net.term_node == node)), dtype=np.int64)
            outs = np.array(sorted(np.flatnonzero(net.init_node == node)), dtype=np.int64)
            turns = np.zeros((ins.size, outs.size), dtype=np.float64)
            for i, a in enumerate(ins):
                out_link = mapping.get(int(a))
                if out_link is None:
                    # an in-link not used by any path: send it nowhere it matters;
                    # give it a deterministic column so the matrix is row-stochastic.
                    turns[i, 0] = 1.0
                else:
                    turns[i, int(np.flatnonzero(outs == out_link)[0])] = 1.0
            result[node] = (ins, outs, turns)
        return result

    # ---------------------------------------------------------------- SO cell LP

    def cell_count(self, a: int) -> int:
        """CTM cell count of link ``a`` at CFL = 1 (``L / (vf*dt)``)."""
        dx = float(self.dynamics.free_speed[a]) * self.grid.dt
        n = float(self.dynamics.length[a]) / dx
        n_cells = int(round(n))
        if n_cells < 1 or abs(n - n_cells) > 1e-9 * max(1.0, n):
            raise ValueError(
                f"TDTAScenario '{self.name}': link {a} length {self.dynamics.length[a]!r} "
                f"is not CTM cell-aligned at dt={self.grid.dt!r} (vf*dt={dx!r})"
            )
        return n_cells

    def derive_cell_scenario(self):
        """Derive the corresponding :class:`~tabench.dta.cells.CellSODTAScenario`
        for the TD-SO bound (ADR-021 cross-model truth).

        Each cell of each used ``CTMLink`` becomes an LP cell (``Q = q_max*dt``,
        ``N = kappa*vf*dt``, ``delta = w/vf``); each origin an inf-storage source
        cell carrying its total demand as initial occupancy (the first-interval
        burst convention that matches the ADR-021 corridor pin exactly); the
        single destination a sink. The LP optimum lower-bounds every strict-CTM
        loading, so ``so_bound_gap = (TSTT - Z*) / Z* >= -tol`` always.

        Requires ``kernel == "ctm"``, a single destination, and a first-interval
        demand burst (so the origin queue is representable as source initial
        occupancy). Raises ``ValueError`` otherwise — the SO bound is then simply
        not reported (``so_bound_gap = NaN``), never faked.
        """
        from ..dta.cells import CellSODTAScenario  # local: dta must stay dnl-free

        if self.kernel != "ctm":
            raise ValueError("SO cell LP is defined for the CTM kernel only")
        dests = self.destinations()
        if len(dests) != 1:
            raise ValueError("SO cell LP requires a single destination")
        if self.demand.breakpoints.shape[0] != 2:
            raise ValueError(
                "SO cell LP requires a first-interval demand burst (one demand "
                "period) so the origin queue maps to source initial occupancy"
            )
        # ...and that single period must fit inside ONE grid step (review MAJOR):
        # a demand SPREAD over several steps (one period but breakpoints[1] > dt)
        # is loaded gradually and avoids the queue the burst-as-initial-occupancy
        # LP charges, so its Z* would be a spurious positive bound. Reject it ->
        # so_bound_gap is NaN (never faked), the UE gap still scores.
        if float(self.demand.breakpoints[1]) > self.grid.dt * (1.0 + 1e-9):
            raise ValueError(
                "SO cell LP requires the demand period to fit in one grid step "
                f"(breakpoints[1]={float(self.demand.breakpoints[1])!r} > dt="
                f"{self.grid.dt!r}); a spread demand has no burst initial occupancy"
            )
        dyn = self.dynamics
        dt = self.grid.dt
        used = self.used_links()

        # cell index layout: sources (one per producing origin), then link cells
        # in used-link order, then the single sink.
        origins = sorted({p.origin for p in self.paths})
        source_of = {o: i for i, o in enumerate(origins)}
        n_sources = len(origins)
        first_cell: dict[int, int] = {}
        last_cell: dict[int, int] = {}
        cap: list[float] = [math.inf] * n_sources
        sto: list[float] = [math.inf] * n_sources
        dlt: list[float] = [1.0] * n_sources
        idx = n_sources
        for a in used:
            n_cells = self.cell_count(a)
            first_cell[a] = idx
            last_cell[a] = idx + n_cells - 1
            dx = float(dyn.free_speed[a]) * dt
            for _ in range(n_cells):
                cap.append(float(dyn.capacity[a]) * dt)
                sto.append(float(dyn.jam_density[a]) * dx)
                dlt.append(float(dyn.wave_speed[a]) / float(dyn.free_speed[a]))
            idx += n_cells
        sink = idx
        cap.append(math.inf)
        sto.append(math.inf)
        dlt.append(1.0)
        n_cells_total = sink + 1

        conn_tail: list[int] = []
        conn_head: list[int] = []

        def add(t: int, h: int) -> None:
            conn_tail.append(t)
            conn_head.append(h)

        for a in used:
            n_cells = self.cell_count(a)
            for i in range(n_cells - 1):
                add(first_cell[a] + i, first_cell[a] + i + 1)
        # origin -> first cell of each of its first links; interior adjacencies;
        # link -> sink for links into the destination.
        seen_source_edges: set[tuple[int, int]] = set()
        for p in self.paths:
            src = source_of[p.origin]
            edge = (src, first_cell[p.first_link])
            if edge not in seen_source_edges:
                add(*edge)
                seen_source_edges.add(edge)
        adj_seen: set[tuple[int, int]] = set()
        for p in self.paths:
            for a, b in zip(p.links, p.links[1:], strict=False):
                edge = (last_cell[a], first_cell[b])
                if edge not in adj_seen:
                    add(*edge)
                    adj_seen.add(edge)
        sink_seen: set[int] = set()
        for p in self.paths:
            a = p.links[-1]
            if last_cell[a] not in sink_seen:
                add(last_cell[a], sink)
                sink_seen.add(last_cell[a])

        # total demand per origin -> source initial occupancy (burst)
        totals = self.demand.cumulative(np.array([self.demand.breakpoints[-1]]))[0]
        x0 = np.zeros(n_cells_total)
        for o in origins:
            x0[source_of[o]] = float(totals[o - 1].sum())
        demand = np.zeros((self.grid.n_steps, n_cells_total))

        return CellSODTAScenario(
            name=f"{self.name}-cells",
            n_cells=n_cells_total,
            sink=sink,
            conn_tail=np.array(conn_tail, dtype=np.int64),
            conn_head=np.array(conn_head, dtype=np.int64),
            capacity=np.array(cap),
            storage=np.array(sto),
            delta=np.array(dlt),
            demand=demand,
            initial_occupancy=x0,
            family=self.family,
        )

    # ---------------------------------------------------------------- hash

    def content_hash(self) -> str:
        """SHA-256 over the canonical serialization of all scored content,
        domain-separated (``"tabench-tdta-scenario-v1;"`` prefix): topology, KW
        physics, demand, grid, kernel, and the enumerated path set — nothing
        else, so no existing scenario hash can move. Every array is LENGTH-FRAMED
        (the Newell/newell-3det lesson, applied defense-in-depth while the tdta
        hashes are still unpublished): a bare byte concatenation is only
        non-ambiguous because the array lengths are mutually pinned today, which a
        future refactor could break — the explicit ``:size;`` frame removes the
        latent boundary-migration collision."""
        h = hashlib.sha256()
        h.update(b"tabench-tdta-scenario-v1;")
        net = self.network
        h.update(
            f"nodes={net.n_nodes};zones={net.n_zones};ftn={net.first_thru_node};".encode()
        )
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
            framed = np.ascontiguousarray(arr, dtype=np.float64)
            h.update(f"{label}:{framed.size};".encode())
            h.update(framed.tobytes())
        h.update(f"dt={float(self.grid.dt)!r};K={self.grid.n_steps};kernel={self.kernel};".encode())
        for p in self.paths:
            h.update(f"p{p.origin}>{p.destination}:{','.join(map(str, p.links))};".encode())
        return h.hexdigest()
