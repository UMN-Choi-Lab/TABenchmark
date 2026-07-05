"""Shared bush machinery for origin-based UE solvers (algb, tapas).

Both Dial's Algorithm B (``algb``) and TAPAS (``tapas``) confine each origin's
traffic to an acyclic sub-network (a "bush") on the centroid-splitting
:class:`PathEngine` expanded graph, and both need the same four primitives:
a Kahn topological sort with a hard cyclicity guard, min/max DAG labels, a
free-flow AON bush initialisation, and the drop/add bush-improvement step. This
module is the single home for those primitives so the two solvers cannot drift.

The methods here are a *pure relocation* of the versions first written for
``algb`` -- the bodies are byte-identical, only moved onto a mixin -- so they
change no float operation ordering and cannot perturb ``algb``'s (BLAS-sensitive)
tail-convergence behaviour. ``AlgorithmBModel`` keeps its own ``_shift_pass``
(the Newton flow-shift that IS on that sensitive path) and only inherits the
graph-pure primitives below. ``_BushState`` is re-exported from ``algb`` so the
existing ``from tabench.models.algb import _BushState`` import keeps working.

``walk_to_divergence`` factors out the "walk two label-chains back to their
common divergence node" logic that ``algb._shift_pass`` performs inline; TAPAS
reuses it as its PAS (paired-alternative-segment) constructor. ``algb`` keeps
its own inline copy so its hot path stays untouched.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import dijkstra

from ..core.scenario import Scenario
from ._paths import PathEngine

__all__ = ["_BushState", "_BushMachinery", "walk_to_divergence"]


class _BushState:
    """Per-origin bush: link-membership mask, origin link flows, topo order."""

    __slots__ = ("in_bush", "x", "topo", "topo_pos", "reachable")

    def __init__(self, n_links: int, n_exp: int) -> None:
        self.in_bush = np.zeros(n_links, dtype=bool)
        self.x = np.zeros(n_links)
        self.topo: np.ndarray | None = None
        self.topo_pos = np.full(n_exp, -1, dtype=np.int64)
        self.reachable: np.ndarray | None = None


def walk_to_divergence(
    topo_pos: np.ndarray,
    minp: np.ndarray,
    maxp: np.ndarray,
    tails: np.ndarray,
    j: int,
    kmin: int,
    kmax: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Walk the min- and max-label chains from ``j`` back to a common node.

    ``kmin``/``kmax`` are the two distinct in-links to merge node ``j`` (the
    min-label / SP-tree link and the max-label / longest-used link). Advancing
    whichever chain is topologically later guarantees the two chains provably
    meet at the divergence node (TAsK ``performFlowMove`` / ``algb._shift_pass``).
    Returns ``(seg_min, seg_max)`` link-index arrays for the two segments of the
    resulting PAS, or ``None`` if a chain is exhausted before they meet.
    """
    seg_min: list[int] = []
    seg_max: list[int] = []
    i_min = i_max = int(j)
    first = True
    while i_min != i_max or first:
        if first:
            seg_min.append(int(kmin))
            seg_max.append(int(kmax))
            i_min = int(tails[kmin])
            i_max = int(tails[kmax])
            first = False
            continue
        if topo_pos[i_max] > topo_pos[i_min]:
            k = maxp[i_max]
            if k < 0:
                return None  # max chain exhausted (no used approach)
            seg_max.append(int(k))
            i_max = int(tails[k])
        else:
            k = minp[i_min]
            if k < 0:
                return None
            seg_min.append(int(k))
            i_min = int(tails[k])
    if i_min != i_max:
        return None
    return np.asarray(seg_min, dtype=np.int64), np.asarray(seg_max, dtype=np.int64)


class _BushMachinery:
    """Mixin of the graph-pure bush primitives shared by algb and tapas.

    Expects the consuming model's ``_setup`` to have populated the graph state
    (see :meth:`_setup_bush_graph`) plus ``self._drop_tol`` before any scan.
    """

    def _setup_bush_graph(self, scenario: Scenario) -> None:
        """Populate the expanded-graph state every bush primitive reads.

        Sets ``_network``, ``_n_links``, ``_engine``, ``_tails``, ``_heads``,
        ``_n_exp``, ``_dest_index``, ``_in_links``, ``_od`` and ``_origins`` --
        everything except the model-specific factor scalars.
        """
        network = scenario.network
        self._network = network
        self._n_links = network.n_links
        engine = PathEngine(network)  # reuse the centroid-splitting expansion
        self._engine = engine
        self._tails = engine._tails
        self._heads = engine._heads
        self._n_exp = engine._n_expanded
        self._dest_index = engine._dest_index
        in_links: list[list[int]] = [[] for _ in range(self._n_exp)]
        for k, h in enumerate(self._heads):
            in_links[h].append(k)
        self._in_links = [np.asarray(lst, dtype=np.int64) for lst in in_links]
        self._od = scenario.demand.matrix
        self._origins = np.nonzero(self._od.sum(axis=1) > 0)[0]

    def _kahn(self, bush: _BushState) -> None:
        """Topological order over bush links restricted to reachable nodes.

        Raises RuntimeError on a cycle -- the acyclicity invariant guard.
        """
        indeg = np.zeros(self._n_exp, dtype=np.int64)
        out_links: dict[int, list[int]] = {}
        for k in np.nonzero(bush.in_bush)[0]:
            indeg[self._heads[k]] += 1
            out_links.setdefault(int(self._tails[k]), []).append(int(k))
        order: list[int] = []
        stack = [int(n) for n in np.nonzero(bush.reachable)[0] if indeg[n] == 0]
        while stack:
            n = stack.pop()
            order.append(n)
            for k in out_links.get(n, ()):
                h = int(self._heads[k])
                indeg[h] -= 1
                if indeg[h] == 0:
                    stack.append(h)
        if len(order) != int(bush.reachable.sum()):
            raise RuntimeError("bush is cyclic: topological sort failed")
        bush.topo = np.asarray(order, dtype=np.int64)
        bush.topo_pos[:] = -1
        bush.topo_pos[bush.topo] = np.arange(len(order))

    def _initial_bushes(self, v: np.ndarray) -> tuple[list[_BushState], np.ndarray, int]:
        """Free-flow shortest-path trees plus AON load. Costs one sp_call."""
        network = self._network
        engine = self._engine
        costs = network.link_cost(np.zeros(self._n_links))
        dist, pred = dijkstra(
            engine._graph(costs),
            directed=True,
            indices=self._origins,
            return_predecessors=True,
        )
        lookup = engine._link_lookup
        bushes: list[_BushState] = []
        for row, o in enumerate(self._origins):
            bush = _BushState(self._n_links, self._n_exp)
            bush.reachable = np.isfinite(dist[row])
            origin_idx = int(o)
            for j in np.nonzero(bush.reachable)[0]:
                p = pred[row, j]
                if p >= 0:
                    bush.in_bush[lookup[(int(p), int(j))]] = True
            # AON load of the demand row down the predecessor tree.
            node_volume = np.zeros(self._n_exp)
            for d in np.nonzero(self._od[o] > 0)[0]:
                if d == o:
                    continue
                di = self._dest_index(d + 1)
                if not np.isfinite(dist[row, di]):
                    raise RuntimeError(f"zone {d + 1} unreachable from zone {o + 1}")
                node_volume[di] += self._od[o, d]
            active = np.nonzero(np.isfinite(dist[row]))[0]
            for j in active[np.argsort(-dist[row, active], kind="stable")]:
                vol = node_volume[j]
                if vol <= 0.0 or int(j) == origin_idx:
                    continue
                p = int(pred[row, j])
                bush.x[lookup[(p, int(j))]] += vol
                node_volume[p] += vol
            self._kahn(bush)
            v += bush.x
            bushes.append(bush)
        return bushes, v, 1

    def _scan(
        self, bush: _BushState, origin_idx: int, t: np.ndarray, rule: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Min (L) and max (U) labels plus predecessor links, in topo order.

        ``rule="used"`` restricts the max labels to links carrying positive
        origin flow (Dial 2006 / TAP-B LONGEST_USED_PATH; shifts need a donor).
        ``rule="used_or_sp"`` also admits any link achieving the exact min
        label at its head -- i.e. every link the drop rule can retain (TAP-B
        LONGEST_USED_OR_SP) -- keeping U finite and the acyclicity potential
        valid for bush updates.
        """
        neg = -np.inf
        L = np.full(self._n_exp, np.inf)
        U = np.full(self._n_exp, neg)
        minp = np.full(self._n_exp, -1, dtype=np.int64)
        maxp = np.full(self._n_exp, -1, dtype=np.int64)
        L[origin_idx] = 0.0
        U[origin_idx] = 0.0
        eps = self._drop_tol
        for j in bush.topo:
            if j == origin_idx:
                continue
            best_l = np.inf
            best_lk = -1
            for k in self._in_links[j]:
                if not bush.in_bush[k]:
                    continue
                cand = L[self._tails[k]] + t[k]
                if cand < best_l:
                    best_l = cand
                    best_lk = k
            L[j] = best_l
            minp[j] = best_lk
            best_u = neg
            best_uk = -1
            for k in self._in_links[j]:
                if not bush.in_bush[k]:
                    continue
                if bush.x[k] > eps:
                    eligible = True
                elif rule == "used_or_sp":
                    # A link the drop rule keeps (it attains the exact min
                    # label at j) must be U-eligible too, or such a kept
                    # non-argmin tie link could break the strict-U-increase
                    # acyclicity invariant the add criterion relies on.
                    eligible = L[self._tails[k]] + t[k] == best_l
                else:
                    eligible = False
                if not eligible:
                    continue
                ut = U[self._tails[k]]
                if ut == neg:
                    continue
                cand = ut + t[k]
                if cand > best_u:
                    best_u = cand
                    best_uk = k
            U[j] = best_u
            maxp[j] = best_uk
        return L, U, minp, maxp

    def _update_bush(self, bush: _BushState, origin_idx: int, t: np.ndarray) -> int:
        """Drop unused links, add shortcuts, re-toposort. Returns links added.

        Strict criterion (Nie 2010 via TAsK worthAdding / TAP-B updateBushB):
        add (i, j) iff L_i + t_ij < L_j AND U_i < U_j. Fallback when nothing
        qualifies: the max-label (modified-shortcut) criterion U_i + t_ij < U_j
        (provably acyclic; Boyles sec. 6.4.3). Shortest-path-tree links are
        never dropped (float-exact min-label equality, as in TAP-B).
        """
        L, U, minp, maxp = self._scan(bush, origin_idx, t, "used_or_sp")
        eps = self._drop_tol
        keep = bush.in_bush.copy()
        for k in np.nonzero(bush.in_bush)[0]:
            if bush.x[k] > eps:
                continue
            if L[self._tails[k]] + t[k] == L[self._heads[k]]:
                bush.x[k] = 0.0  # keep the SP tree; zero the sub-eps residue
                continue
            keep[k] = False
            bush.x[k] = 0.0  # discard sub-eps residue (<= drop_tol per link)
        added: list[int] = []
        for k in range(self._n_links):
            if keep[k]:
                continue
            i, j = int(self._tails[k]), int(self._heads[k])
            if not (bush.reachable[i] and bush.reachable[j]):
                continue
            if U[i] == -np.inf:  # no path to extend (TAP-B guard)
                continue
            if L[i] + t[k] < L[j] and U[i] < U[j]:
                added.append(k)
        if not added:
            for k in range(self._n_links):
                if keep[k]:
                    continue
                i, j = int(self._tails[k]), int(self._heads[k])
                if not (bush.reachable[i] and bush.reachable[j]):
                    continue
                if U[i] == -np.inf:
                    continue
                if U[i] + t[k] < U[j]:
                    added.append(k)
        if added:
            keep[np.asarray(added, dtype=np.int64)] = True
        if not np.array_equal(keep, bush.in_bush):
            bush.in_bush = keep
            self._kahn(bush)  # re-sort after EVERY topology change
        return len(added)
