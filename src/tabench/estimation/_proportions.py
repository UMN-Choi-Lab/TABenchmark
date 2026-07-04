"""MSA-averaged per-OD assignment proportions (ADR-002, Decision 3).

Every classical T2 estimator scores an OD vector ``g`` against link counts
through an **assignment-proportion matrix** ``P`` with ``P[a, k]`` = the
fraction of pair-``k`` demand traversing link ``a``. We extract ``P`` by running
the inner assignment as a method of successive averages over all-or-nothing
trees, accumulating ``P = (1/K) sum_k P_k`` where ``P_k`` is the sparse 0/1
per-pair shortest-path incidence at iteration-``k`` costs.

Two properties matter and both are by construction:

* ``v = P @ g`` holds to machine precision — the returned ``msa_flows`` are
  literally ``P @ g``, and the MSA flow trajectory is the same running mean of
  the same trees, so the modeled flow an estimator differentiates is exactly
  the flow the certificate would see for that ``g`` at these proportions.
* Equilibrium route ties are averaged through the trajectory rather than
  tie-broken to one arbitrary tree (at UE every used route ties by definition;
  a single tree makes the misfit gradient fragile — on Braess at prior D=4 it
  flips sign with the tie-break).

Congested coupling is the standard outer fixed point (assign current ``g`` ->
extract ``P`` -> estimate -> repeat; Cascetta & Postorino 2001); this helper is
one inner extraction, charged ``k_inner`` shortest-path calls.
"""

from __future__ import annotations

import numpy as np

from ..core.scenario import Demand, Network
from ..models._paths import PathEngine

__all__ = ["active_pairs", "proportion_matrix", "od_from_pairs"]


def od_from_pairs(
    prior_matrix: np.ndarray, pairs: list[tuple[int, int]], g: np.ndarray
) -> np.ndarray:
    """Scatter a per-pair demand vector into a full OD matrix.

    Off-diagonal cells outside ``pairs`` stay zero (support is fixed by the
    prior); the diagonal (intrazonal demand, which never enters the network) is
    carried over from ``prior_matrix`` unchanged.
    """
    od = np.zeros_like(prior_matrix, dtype=np.float64)
    np.fill_diagonal(od, np.diag(prior_matrix))
    for (i, j), value in zip(pairs, np.asarray(g, dtype=np.float64), strict=True):
        od[i, j] = value
    return od


def active_pairs(demand_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Ordered 0-based ``(origin, destination)`` pairs with positive off-diagonal demand."""
    m = np.asarray(demand_matrix, dtype=np.float64)
    rows, cols = np.nonzero(m > 0)
    return [(int(i), int(j)) for i, j in zip(rows, cols, strict=True) if i != j]


def proportion_matrix(
    network: Network,
    demand: Demand,
    k_inner: int,
    pairs: list[tuple[int, int]] | None = None,
    engine: PathEngine | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    """Extract ``(P, pairs, msa_flows)`` at the equilibrium of ``demand``.

    ``P`` has shape ``(n_links, len(pairs))``; ``msa_flows = P @ g`` where ``g``
    is the per-pair demand vector, so the identity ``v = P g`` is exact.
    ``k_inner >= 1`` is the number of MSA/AON sweeps (one shortest-path call
    each). Pairs default to the positive off-diagonal support of ``demand``.
    """
    if k_inner < 1:
        raise ValueError("k_inner must be >= 1")
    engine = engine or PathEngine(network)
    pairs = pairs if pairs is not None else active_pairs(demand.matrix)
    n_links = network.n_links
    n_pairs = len(pairs)
    g = np.array([demand.matrix[i, j] for (i, j) in pairs], dtype=np.float64)
    col_of = {pair: k for k, pair in enumerate(pairs)}

    p_sum = np.zeros((n_links, n_pairs), dtype=np.float64)
    if n_pairs == 0:
        return p_sum, pairs, np.zeros(n_links)

    v = np.zeros(n_links, dtype=np.float64)
    for k in range(1, int(k_inner) + 1):
        costs = network.link_cost(v)
        paths, _ = engine.shortest_paths(costs, demand)
        p_k = np.zeros((n_links, n_pairs), dtype=np.float64)
        for pair, links in paths.items():
            p_k[links, col_of[pair]] = 1.0
        p_sum += p_k
        y_k = p_k @ g
        v = ((k - 1) * v + y_k) / k

    p = p_sum / float(k_inner)
    msa_flows = p @ g  # exact v = P g by construction
    return p, pairs, msa_flows
