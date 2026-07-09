"""Spiess & Florian (1989) optimal-strategy transit assignment solver.

Two passes per destination (docs/design/adr-014-transit-strategy.md):

* **label-setting** — process arcs in nondecreasing onward cost ``u[head] + time``
  (a Dijkstra-like sweep). Arc ``a = (i, j)`` is *attractive* iff its onward cost
  is strictly below node ``i``'s current expected cost ``u_i``; adding it updates

      u_i <- (f_i u_i + f_a (u_j + c_a)) / (f_i + f_a),   f_i <- f_i + f_a,

  a frequency-weighted combination. The expected wait ``1/F_i`` enters as the
  ``+1`` seeded on the FIRST attractive line at each node (``u_i <- 1/f_a +
  (u_j + c_a)``) — omitting the seed silently drops the wait, so it is explicit.
  A deterministic arc (``f_a = inf``, zero wait) that beats the current cost
  dominates: it becomes the sole attractive arc and closes the node.

* **loading** — process nodes farthest-from-destination first (decreasing ``u``),
  splitting each node's volume over its attractive arcs by frequency share
  ``v_a = (f_a / f_i) V_i`` (all of it on the deterministic arc when the node is
  deterministic). Every predecessor has strictly larger ``u`` (an attractive arc
  has ``u_i > u_j + c_a >= u_j``), so inflow is complete before a node is split.

The whole thing is a convex LP, so the greedy strategy is globally optimal
(Spiess & Florian 1989). Uncongested / frequency-based: costs are flow-independent
and there is no equilibration loop.
"""

from __future__ import annotations

import numpy as np

from .network import TransitScenario, TransitStrategy

__all__ = ["optimal_strategy", "OptimalStrategyModel"]


def _solve_to_dest(
    net, dest: int
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Optimal-strategy labels to one destination.

    Returns ``(u, f_node, node_arcs)``: expected cost-to-destination ``u``, the
    attractive combined frequency per node ``f_node`` (``inf`` if closed by a
    deterministic arc), and the ordered attractive-arc indices per node.
    """
    n, m = net.n_nodes, net.n_arcs
    tail, head, time, freq = net.tail, net.head, net.time, net.freq
    u = np.full(n, np.inf)
    u[dest] = 0.0
    f_node = np.zeros(n)
    node_arcs: list[list[int]] = [[] for _ in range(n)]
    processed = np.zeros(m, dtype=bool)

    while True:
        # Global minimum onward-cost unprocessed arc with a finalized head label.
        best = -1
        best_key = np.inf
        for a in range(m):
            if processed[a] or tail[a] == dest:
                continue
            j = head[a]
            if not np.isfinite(u[j]):
                continue
            key = u[j] + time[a]
            if key < best_key:  # ties: lower arc index wins (deterministic, P8)
                best_key = key
                best = a
        if best < 0:
            break
        a = best
        processed[a] = True
        i = int(tail[a])
        key = float(best_key)
        if not (key < u[i]):  # not strictly attractive; later (larger-key) arcs at i fail too
            continue
        fa = float(freq[a])
        if not np.isfinite(fa):
            # Deterministic zero-wait arc beats the current cost: it dominates,
            # becomes the sole attractive arc, and closes the node.
            u[i] = key
            f_node[i] = np.inf
            node_arcs[i] = [a]
        elif not np.isfinite(f_node[i]):
            # Node already closed by a deterministic arc; a larger-key arc can
            # never be attractive here (key >= u_i), so this is unreachable.
            continue
        else:
            if f_node[i] == 0.0:
                u[i] = 1.0 / fa + key  # seed the expected wait on the first line
            else:
                u[i] = (f_node[i] * u[i] + fa * key) / (f_node[i] + fa)
            f_node[i] += fa
            node_arcs[i].append(a)
    return u, f_node, node_arcs


def _load(
    net, dest: int, node_arcs, f_node: np.ndarray, u: np.ndarray, origin_vol
) -> np.ndarray:
    """Frequency-share loading of one destination's demand onto the strategy."""
    n, m = net.n_nodes, net.n_arcs
    head, freq = net.head, net.freq
    volume = np.zeros(n)
    for origin, vol in origin_vol:
        volume[origin] += vol
    arc_volumes = np.zeros(m)
    # Farthest-first: every predecessor of i has strictly larger u, so all inflow
    # to i has arrived before i is split.
    order = np.argsort(-u, kind="stable")
    for i in order:
        i = int(i)
        if i == dest or not np.isfinite(u[i]) or volume[i] <= 0.0:
            continue
        arcs = node_arcs[i]
        if not arcs:
            continue
        if not np.isfinite(f_node[i]):
            a = arcs[0]  # deterministic: all flow on the closing arc
            arc_volumes[a] += volume[i]
            volume[int(head[a])] += volume[i]
        else:
            for a in arcs:
                v = float(freq[a]) / f_node[i] * volume[i]
                arc_volumes[a] += v
                volume[int(head[a])] += v
    return arc_volumes


def optimal_strategy(scenario: TransitScenario) -> TransitStrategy:
    """Assign the scenario's demand by optimal strategies (Spiess & Florian 1989).

    Solves one label-setting + loading pass per distinct demand destination and
    accumulates the arc volumes. Deterministic (no equilibration).
    """
    net = scenario.network
    dem = scenario.demand
    arc_volumes = np.zeros(net.n_arcs)
    labels: list[tuple[int, np.ndarray]] = []
    dest_arc_volumes: list[tuple[int, np.ndarray]] = []
    pair_costs = np.zeros(dem.n_pairs)

    for dest in np.unique(dem.destinations):
        dest = int(dest)
        u, f_node, node_arcs = _solve_to_dest(net, dest)
        labels.append((dest, u.copy()))
        mask = dem.destinations == dest
        origin_vol = list(
            zip(dem.origins[mask].tolist(), dem.volumes[mask].tolist(), strict=True)
        )
        v_dest = _load(net, dest, node_arcs, f_node, u, origin_vol)
        dest_arc_volumes.append((dest, v_dest))
        arc_volumes += v_dest
        pair_costs[mask] = u[dem.origins[mask]]

    return TransitStrategy(
        arc_volumes=arc_volumes,
        labels=tuple(labels),
        pair_costs=pair_costs,
        dest_arc_volumes=tuple(dest_arc_volumes),
    )


class OptimalStrategyModel:
    """Spiess & Florian (1989) optimal-strategy transit assignment.

    A standalone transit model (its scenario is a :class:`TransitScenario`, not
    the road :class:`~tabench.core.scenario.Scenario`, so it is not in the road
    ``MODEL_REGISTRY`` — the same parallel-module pattern as the DNL core).
    """

    name = "transit-strategy"

    def solve(self, scenario: TransitScenario) -> TransitStrategy:
        return optimal_strategy(scenario)
