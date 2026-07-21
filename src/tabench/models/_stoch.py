"""Dial's STOCH logit network loading — the pinned SUE loading map.

This is the deterministic map ``L(costs, demand, theta)`` that defines the
benchmark's logit-SUE task (see docs/design/adr-001): Dial's (1971)
double-pass algorithm with the origin-based efficient-link criterion
``r(i) < r(j)`` exactly as presented in Sheffi (1985, section 11.2), computed
in the log domain so node weights cannot overflow. "Stochastic" refers to
traveler perception, not algorithmic randomness: no RNG is involved.

Both the ``sue-msa`` reference solver and the harness Evaluator call this
map — the SUE certificate is the fixed-point residual ``||v - L(t(v))||_1``
per traveler, so model and scorer must share one pinned implementation.

Centroid restrictions (``first_thru_node``) are honored exactly by running
on the PathEngine expanded graph, whose Dijkstra labels already make paths
through centroids impossible.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import dijkstra

from ..core.scenario import Demand, Network
from ._numerics import logsumexp
from ._paths import PathEngine

__all__ = ["StochEngine"]


class StochEngine:
    """Reusable per-network engine for Dial-STOCH logit loading."""

    def __init__(self, network: Network) -> None:
        self.network = network
        self._paths = PathEngine(network)  # expanded-graph arrays + validation
        self._tails = self._paths._tails
        self._heads = self._paths._heads
        n = self._paths._n_expanded
        # CSC-style incoming-link index: links sorted by head node.
        order = np.argsort(self._heads, kind="stable")
        self._in_links = order
        heads_sorted = self._heads[order]
        self._in_start = np.searchsorted(heads_sorted, np.arange(n))
        self._in_end = np.searchsorted(heads_sorted, np.arange(n) + 1)
        # Expanded index -> original 1-based node id (covers shadow heads).
        self._node_id = np.arange(1, n + 1, dtype=np.int64)
        for node0, head in enumerate(self._paths._head_index):
            self._node_id[head] = node0 + 1

    def _incoming(self, node: int) -> np.ndarray:
        return self._in_links[self._in_start[node] : self._in_end[node]]

    def load(self, costs: np.ndarray, demand: Demand, theta: float) -> np.ndarray:
        """Logit shares over Dial-efficient paths at fixed ``costs``.

        Returns link flows routing all of ``demand``. Pure function of
        ``(costs, demand, theta)``. Complexity is one Dijkstra sweep per
        origin plus O(links) forward/backward passes (node loop in Python —
        fine at v0.x scenario sizes).
        """
        if not np.isfinite(theta) or theta <= 0:
            raise ValueError(f"theta must be finite and > 0, got {theta!r}")
        costs = np.asarray(costs, dtype=np.float64)
        graph = self._paths._graph(costs)  # validates costs > 0 and finite
        od = demand.matrix
        origins = np.nonzero(od.sum(axis=1) > 0)[0]  # 0-based zone indices
        flows = np.zeros(self.network.n_links, dtype=np.float64)
        if origins.size == 0:
            return flows

        dist = dijkstra(graph, directed=True, indices=origins)
        tails, heads = self._tails, self._heads

        for row, o in enumerate(origins):
            r = dist[row]
            # Efficient links: strictly increasing origin labels (no tie
            # tolerance — the strict rule is part of the task definition).
            finite = np.isfinite(r[tails]) & np.isfinite(r[heads])
            efficient = finite & (r[tails] < r[heads])
            x = np.full(self.network.n_links, -np.inf)
            x[efficient] = theta * (
                r[heads[efficient]] - r[tails[efficient]] - costs[efficient]
            )
            # x <= 0 up to float rounding (shortest-path optimality), so
            # likelihoods exp(x) never overflow; exactly 0 on tree links.

            reachable = np.nonzero(np.isfinite(r))[0]
            order = reachable[np.argsort(r[reachable], kind="stable")]
            origin_index = int(o)  # origin's tail role keeps its original index

            # Forward pass ascending r: b(j) = log W(j), Sheffi's weight
            # recursion in the log domain (logsumexp with max shift).
            b = np.full(r.size, -np.inf)
            b[origin_index] = 0.0
            for j in order:
                if j == origin_index:
                    continue
                terms = x[self._incoming(j)] + b[tails[self._incoming(j)]]
                b[j] = logsumexp(terms)  # stays -inf if no finite incoming term

            # Backward pass descending r: split node volume over incoming
            # efficient links with logit fractions phi = exp(x + b_tail - b_node).
            volume = np.zeros(r.size)
            for d in np.nonzero(od[o] > 0)[0]:
                if d == o:
                    continue  # intrazonal demand never enters the network
                di = self._paths._dest_index(d + 1)
                if not np.isfinite(r[di]):
                    raise RuntimeError(
                        f"Zone {d + 1} unreachable from zone {o + 1} at current costs"
                    )
                volume[di] += od[o, d]

            for j in order[::-1]:
                vol = volume[j]
                if vol <= 0.0 or j == origin_index:
                    continue
                if not np.isfinite(b[j]):
                    raise RuntimeError(
                        f"No efficient path reaches loaded node {self._node_id[j]} "
                        f"from zone {o + 1} (label ties can sever efficient paths "
                        "when costs saturate float64 resolution)"
                    )
                links = self._incoming(j)
                w = x[links] + b[tails[links]] - b[j]
                mask = np.isfinite(w)
                links, w = links[mask], w[mask]
                phi = np.exp(w)  # each w <= 0 by definition of b(j)
                phi /= phi.sum()  # renormalize: conserve demand to float precision
                contribution = vol * phi
                flows[links] += contribution
                np.add.at(volume, tails[links], contribution)

        return flows
