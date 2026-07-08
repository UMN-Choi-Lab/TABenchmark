"""Davis & Nihan (1993) large-population link-count covariance (route-level).

The stationary spatial covariance of the Davis-Nihan Gaussian limit (Prop 3,
*Operations Research* 41(1):169-178) is the finite-population multinomial
route-choice covariance aggregated to links. In the paper's notation
(``COV[x | s]`` on p. 171),

    Q_stat = sum_j (g_j^2 / N_j) * Delta_j^T (diag(p_j) - p_j p_j^T) Delta_j,

where at the network equilibrium of demand ``g`` each OD pair ``j`` uses a route
set with probabilities ``p_j``, ``Delta_j`` is that set's route-link incidence,
and ``N_j`` is the finite traveler population for pair ``j`` (each traveler
carrying weight ``g_j / N_j`` so the pair still totals ``g_j``). We extract the
route set and ``p_j`` from the MSA/all-or-nothing trees at equilibrium costs, so
the link marginal ``Delta_j^T p_j`` equals the proportion column ``P[:, j]`` and
the **mean loading is exactly ``P g = x_UE``** -- the deterministic UE flow the
T2 certifier pins. As ``N_j`` grows the covariance vanishes as ``1 / N`` (Prop 2
SLLN). It is *exact* on link-disjoint route sets (the two-route anchor, where
each route is one multinomial cell) and structurally faithful (non-diagonal,
demand-conserving) in general.

This helper depends only on ``core`` and ``models`` (never ``estimation``), so
``observe`` can import it without the ``estimation -> observe`` cycle.
"""

from __future__ import annotations

import numpy as np

from ..core.scenario import Demand, Network
from ..models._paths import PathEngine

__all__ = ["active_od_pairs", "dn_spatial_covariance", "psd_factor"]


def active_od_pairs(demand_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Ordered 0-based ``(origin, destination)`` pairs with positive off-diagonal demand.

    A local copy of ``estimation._proportions.active_pairs`` so ``observe`` does
    not import ``estimation`` (which would be circular).
    """
    m = np.asarray(demand_matrix, dtype=np.float64)
    rows, cols = np.nonzero(m > 0)
    return [(int(i), int(j)) for i, j in zip(rows, cols, strict=True) if i != j]


def psd_factor(cov: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """A factor ``L`` with ``L @ L.T == cov`` for a symmetric PSD (possibly
    singular) ``cov``, via its eigendecomposition (Cholesky needs strict PD; the
    DN covariance is rank-deficient by flow conservation). Negative eigenvalues
    from round-off are clipped to zero.
    """
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    w, v = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)
    if w.max(initial=0.0) > 0.0:
        w[w < rtol * w.max()] = 0.0
    return v * np.sqrt(w)


def dn_spatial_covariance(
    network: Network,
    demand: Demand,
    g_pairs: np.ndarray,
    n_travelers: np.ndarray,
    k_inner: int,
    pairs: list[tuple[int, int]] | None = None,
    engine: PathEngine | None = None,
) -> np.ndarray:
    """Route-level Davis-Nihan multinomial link-count covariance at equilibrium.

    Runs the same MSA/all-or-nothing loop that ``proportion_matrix`` uses, but
    records each OD pair's *route* frequencies (not just the link marginal), then
    aggregates the per-pair multinomial covariance ``diag(p_j) - p_j p_j^T`` to
    link space and scales each pair by ``g_j^2 / N_j``. Returns the
    ``(n_links, n_links)`` stationary spatial covariance ``Q_stat``; the mean
    loading it fluctuates around is ``P g = x_UE`` by construction.
    """
    if k_inner < 1:
        raise ValueError("k_inner must be >= 1")
    engine = engine or PathEngine(network)
    pairs = pairs if pairs is not None else active_od_pairs(demand.matrix)
    if set(pairs) != set(active_od_pairs(demand.matrix)):
        # The equilibrium `v` below is assigned from `pairs` only; a strict subset
        # of the active OD pairs would understate congestion and silently corrupt
        # Q. Fail loud rather than return a wrong covariance (the shipped caller
        # always passes the full active set).
        raise ValueError(
            "dn_spatial_covariance requires `pairs` to be exactly the active "
            "OD-pair set of `demand`, not a subset (equilibrium congestion is "
            "assigned from these pairs)."
        )
    n_links = network.n_links
    n_pairs = len(pairs)
    q_cov = np.zeros((n_links, n_links), dtype=np.float64)
    if n_pairs == 0:
        return q_cov

    g = np.asarray(g_pairs, dtype=np.float64)
    n_trav = np.asarray(n_travelers, dtype=np.float64)
    col_of = {pair: k for k, pair in enumerate(pairs)}
    route_counts: list[dict[tuple[int, ...], int]] = [{} for _ in range(n_pairs)]

    v = np.zeros(n_links, dtype=np.float64)
    for k in range(1, int(k_inner) + 1):
        costs = network.link_cost(v)
        paths, _ = engine.shortest_paths(costs, demand)
        p_k = np.zeros((n_links, n_pairs), dtype=np.float64)
        for pair, links in paths.items():
            if pair not in col_of:
                continue
            c = col_of[pair]
            idx = np.asarray(links, dtype=np.int64).ravel()
            p_k[idx, c] = 1.0
            route = tuple(sorted(int(x) for x in idx))
            route_counts[c][route] = route_counts[c].get(route, 0) + 1
        y_k = p_k @ g
        v = ((k - 1) * v + y_k) / k

    for c, table in enumerate(route_counts):
        if not table:
            continue
        routes = list(table.keys())
        freqs = np.array([table[r] for r in routes], dtype=np.float64)
        p_j = freqs / freqs.sum()  # route probabilities; Delta_j^T p_j == P[:, c]
        delta = np.zeros((len(routes), n_links), dtype=np.float64)
        for r_idx, route in enumerate(routes):
            delta[r_idx, list(route)] = 1.0
        a_j = np.diag(p_j) - np.outer(p_j, p_j)  # multinomial route covariance
        contrib = delta.T @ a_j @ delta  # aggregate to link space
        scale = (g[c] ** 2) / max(float(n_trav[c]), 1.0)
        q_cov += scale * contrib
    return q_cov
