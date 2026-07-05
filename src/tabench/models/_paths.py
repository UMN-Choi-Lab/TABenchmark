"""Shortest-path / all-or-nothing engine honoring TNTP centroid semantics.

Nodes with id below ``first_thru_node`` are zone centroids that may start or
end trips but cannot carry through traffic. This is enforced exactly by node
splitting: each restricted centroid ``c`` keeps its original index for the
*tail* (origin) role and receives a shadow index for the *head* (destination)
role, with no arc connecting the two — so no path can traverse a centroid.

All-or-nothing loading uses the predecessor tree from Dijkstra with a
descending-distance sweep, which is O(nodes log nodes + links) per origin and
scales to large networks.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from ..core.scenario import Demand, Network

__all__ = ["PathEngine"]


class PathEngine:
    """Reusable per-network engine for shortest paths and AON assignment."""

    def __init__(self, network: Network) -> None:
        self.network = network
        n = network.n_nodes
        ftn = network.first_thru_node

        # 1-based centroid ids that cannot carry through traffic.
        restricted = np.arange(1, ftn) if ftn > 1 else np.array([], dtype=np.int64)
        self._head_index = np.arange(1, n + 1) - 1  # default: head index = node - 1
        self._n_expanded = n
        if restricted.size:
            shadow = {int(c): n + k for k, c in enumerate(restricted)}
            for c, s in shadow.items():
                self._head_index[c - 1] = s
            self._n_expanded = n + restricted.size

        self._tails = network.init_node - 1
        self._heads = self._head_index[network.term_node - 1]
        self._link_lookup = {
            (int(u), int(v)): k
            for k, (u, v) in enumerate(zip(self._tails, self._heads, strict=True))
        }

    def _dest_index(self, zone: int) -> int:
        """Expanded-graph index at which distances to ``zone`` are read."""
        return int(self._head_index[zone - 1])

    def _graph(self, costs: np.ndarray) -> csr_matrix:
        if np.any(costs <= 0) or not np.all(np.isfinite(costs)):
            raise ValueError("Link costs must be strictly positive and finite")
        n = self._n_expanded
        return csr_matrix((costs, (self._tails, self._heads)), shape=(n, n))

    def shortest_paths(
        self, costs: np.ndarray, demand: Demand
    ) -> tuple[dict[tuple[int, int], np.ndarray], float]:
        """Explicit shortest path per positive-demand OD pair at fixed costs.

        Returns ``(paths, sptt)``: ``paths`` maps 0-based zone pairs
        ``(origin, destination)`` to the path's link indices in traversal
        order (backtracked through the node-split expanded graph, so paths
        never traverse restricted centroids); ``sptt`` is as in
        :meth:`all_or_nothing`. One batched Dijkstra over all origins.
        """
        graph = self._graph(np.asarray(costs, dtype=np.float64))
        od = demand.matrix
        origins = np.nonzero(od.sum(axis=1) > 0)[0]
        paths: dict[tuple[int, int], np.ndarray] = {}
        sptt = 0.0
        if origins.size == 0:
            return paths, sptt

        dist, pred = dijkstra(
            graph, directed=True, indices=origins, return_predecessors=True
        )
        for row, o in enumerate(origins):
            origin_index = int(o)  # origin's tail role keeps its original index
            for d in np.nonzero(od[o] > 0)[0]:
                if d == o:
                    continue  # intrazonal demand never enters the network
                di = self._dest_index(d + 1)
                if not np.isfinite(dist[row, di]):
                    raise RuntimeError(
                        f"Zone {d + 1} unreachable from zone {o + 1} at current costs"
                    )
                sptt += od[o, d] * dist[row, di]
                links = []
                j = int(di)
                while j != origin_index:
                    p = int(pred[row, j])
                    links.append(self._link_lookup[(p, j)])
                    j = p
                paths[(int(o), int(d))] = np.asarray(links[::-1], dtype=np.int64)
        return paths, float(sptt)

    def od_cost_matrix(self, costs: np.ndarray, demand: Demand) -> np.ndarray:
        """Shortest-path cost for every positive-demand OD pair, as a dense
        ``(n_zones, n_zones)`` matrix (0 where ``demand`` is 0).

        One batched Dijkstra over all origins. Used by the elastic-demand
        certificate to evaluate ``D_rs(u_rs)`` from emitted link flows (the
        harness never trusts a model's self-reported OD costs — P1).
        """
        graph = self._graph(np.asarray(costs, dtype=np.float64))
        od = demand.matrix
        nz = self.network.n_zones
        out = np.zeros((nz, nz), dtype=np.float64)
        origins = np.nonzero(od.sum(axis=1) > 0)[0]
        if origins.size == 0:
            return out
        dist = dijkstra(graph, directed=True, indices=origins, return_predecessors=False)
        for row, o in enumerate(origins):
            for d in np.nonzero(od[o] > 0)[0]:
                if d == o:
                    continue
                di = self._dest_index(int(d) + 1)
                if not np.isfinite(dist[row, di]):
                    raise RuntimeError(
                        f"Zone {int(d) + 1} unreachable from zone {int(o) + 1} at current costs"
                    )
                out[o, d] = dist[row, di]
        return out

    def all_or_nothing(
        self, costs: np.ndarray, demand: Demand
    ) -> tuple[np.ndarray, float]:
        """Assign all demand to current shortest paths.

        Returns ``(link_flows, sptt)`` where ``sptt`` is the total travel cost
        if every trip used its shortest path at the given (fixed) costs.
        """
        graph = self._graph(np.asarray(costs, dtype=np.float64))
        od = demand.matrix
        origins = np.nonzero(od.sum(axis=1) > 0)[0]  # 0-based zone indices
        flows = np.zeros(self.network.n_links, dtype=np.float64)
        sptt = 0.0

        if origins.size == 0:
            return flows, sptt

        dist, pred = dijkstra(
            graph, directed=True, indices=origins, return_predecessors=True
        )

        for row, o in enumerate(origins):
            node_volume = np.zeros(self._n_expanded, dtype=np.float64)
            for d in np.nonzero(od[o] > 0)[0]:
                if d == o:
                    continue  # intrazonal demand never enters the network
                di = self._dest_index(d + 1)
                if not np.isfinite(dist[row, di]):
                    raise RuntimeError(
                        f"Zone {d + 1} unreachable from zone {o + 1} at current costs"
                    )
                node_volume[di] += od[o, d]
                sptt += od[o, d] * dist[row, di]

            # Sweep nodes farthest-first, pushing volume up the predecessor tree.
            reached = np.nonzero(node_volume > 0)[0]
            if reached.size == 0:
                continue
            active = np.nonzero(np.isfinite(dist[row]))[0]
            order = active[np.argsort(-dist[row, active], kind="stable")]
            origin_index = int(o)  # origin's tail role keeps its original index
            for j in order:
                vol = node_volume[j]
                if vol <= 0.0 or j == origin_index:
                    continue
                p = pred[row, j]
                if p < 0:
                    raise RuntimeError(f"No predecessor for reached node {j}")
                flows[self._link_lookup[(int(p), int(j))]] += vol
                node_volume[p] += vol

        return flows, float(sptt)
