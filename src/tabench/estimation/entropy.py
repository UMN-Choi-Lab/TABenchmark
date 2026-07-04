"""vzw-entropy: the Van Zuylen & Willumsen (1980) most-likely trip matrix.

The maximum-entropy matrix reproducing the counts has the multiplicative form
``T_ij = t_ij * prod_a X_a^{p^a_ij}`` (prior ``t`` = task prior). We realize it
by cyclic multiplicative count-balancing: for each observed link ``a`` with
modeled flow ``v_a = sum_ij p^a_ij T_ij > 0``, scale every OD cell contributing
to ``a`` so that ``v_a`` matches the count ``hat c_a`` exactly. Counts enter as
the per-period mean.

The exponent ``p^a_ij`` makes each pass a damped step (``p <= 1``). For a link
crossed by a single OD pair with proportion ``p`` the update scales the estimate
by ``(hat c / (p T))^p``, converging *geometrically* to the exact ``T = hat c /
p`` — one pass on the two-route link-0 sensor (``p = 0.625``, noiseless count
``2.5``) gives ``T = 4^0.625``, not the fixed point in a single touch. For two
mutually inconsistent sensors on one pair the pass no longer oscillates: it
converges to a compromise whose observed-count residual stays above tolerance,
so ``counts_consistent`` is keyed on that converged residual and reported
``False``. Zero prior cells stay zero.

Deterministic. Outer congested coupling re-extracts proportions at the current
estimate (Cascetta & Postorino 2001) and keeps the best-observed-RMSE outer
iterate, so noisy counts that push the fixed point into a bypass-saturated
regime cannot make it return worse than its start; the balancing itself is free,
so the cost is ``outer_iters * k_inner`` shortest-path calls.
"""

from __future__ import annotations

import numpy as np

from ..core.budget import Budget, BudgetCoords
from ..core.factors import FactorSpec
from ..core.rng import RngBundle
from ..core.scenario import Demand
from ..models._paths import PathEngine
from ._proportions import active_pairs, od_from_pairs, proportion_matrix
from .base import (
    EstimationTask,
    ODEstimator,
    ODResultBundle,
    ODTrace,
    _estimation_capabilities,
    register_estimator,
)

__all__ = ["vzw_balance", "VZWEntropyEstimator"]


def _obs_rmse(p_obs: np.ndarray, g: np.ndarray, counts: np.ndarray) -> float:
    v = p_obs @ g
    return float(np.sqrt(np.mean((v - counts) ** 2)))


def vzw_balance(
    prior: np.ndarray,
    p_obs: np.ndarray,
    counts: np.ndarray,
    n_passes: int,
    tol: float = 1e-9,
) -> tuple[np.ndarray, list[np.ndarray], bool]:
    """Cyclic multiplicative count-balancing of ``prior`` toward ``counts``.

    ``p_obs`` is the ``(n_obs, n_pairs)`` proportion block on the observed
    sensors; ``counts`` the per-period-mean count per sensor. Returns
    ``(g, trajectory, counts_consistent)`` where ``g`` is the converged iterate
    after ``n_passes`` damped balancing passes and ``counts_consistent`` flags
    whether its observed-count residual has settled below ``tol`` (mutually
    inconsistent counts converge to a compromise that stays above tolerance).
    ``trajectory`` records ``g`` after every single-link update.
    """
    g = np.array(prior, dtype=np.float64, copy=True)
    counts = np.asarray(counts, dtype=np.float64)
    n_obs = len(counts)
    trajectory = [g.copy()]
    for _ in range(int(n_passes)):
        for a in range(n_obs):
            v_a = float(p_obs[a] @ g)
            if v_a <= 0.0:  # sensor no active pair can reach: skip (Decision 3)
                continue
            mask = p_obs[a] > 0.0
            g[mask] = g[mask] * (counts[a] / v_a) ** p_obs[a][mask]
            trajectory.append(g.copy())
    residual = _obs_rmse(p_obs, g, counts) if n_obs else 0.0
    scale = max(1.0, float(counts.mean())) if n_obs else 1.0
    counts_consistent = residual <= tol * scale
    return g, trajectory, counts_consistent


@register_estimator
class VZWEntropyEstimator(ODEstimator):
    """Maximum-entropy OD estimation by cyclic count-balancing (Van Zuylen &
    Willumsen 1980)."""

    name = "vzw-entropy"
    capabilities = _estimation_capabilities(deterministic=True)
    factors = {
        "k_inner": FactorSpec(
            default=60, kind="int", bounds=(1, 5000),
            doc="Inner MSA/AON sweeps per proportion extraction (Decision 3).",
        ),
        "outer_iters": FactorSpec(
            default=25, kind="int", bounds=(1, 5000),
            doc="Outer assign<->balance fixed-point iterations.",
        ),
        "balance_passes": FactorSpec(
            default=40, kind="int", bounds=(1, 100000),
            doc="Cyclic balancing passes over the sensor set per outer iteration.",
        ),
        "tol": FactorSpec(
            default=1e-9, kind="float", bounds=(0.0, 1.0),
            doc="Observed-RMSE tolerance below which counts are called consistent.",
        ),
    }

    def estimate(
        self, task: EstimationTask, budget: Budget, rng: RngBundle, trace: ODTrace
    ) -> ODResultBundle:
        network = task.network
        engine = PathEngine(network)
        prior_matrix = task.prior.matrix
        pairs = active_pairs(prior_matrix)
        sensors = np.asarray(task.dataset.payload["sensor_links"], dtype=np.int64)
        counts_mean = np.asarray(task.dataset.payload["counts"], dtype=np.float64).mean(axis=0)

        k_inner = self.factor_values["k_inner"]
        outer_iters = self.factor_values["outer_iters"]
        balance_passes = self.factor_values["balance_passes"]
        tol = self.factor_values["tol"]

        g_pr = np.array([prior_matrix[i, j] for (i, j) in pairs], dtype=np.float64)
        g = g_pr.copy()
        best_g, best_resid, best_consistent = g_pr.copy(), np.inf, True
        sp_calls = 0
        consistent = True
        resid = np.inf
        coords = BudgetCoords(iterations=0, sp_calls=0, wall_ms=0.0)
        stride = max(1, int(outer_iters) // 15)
        for it in range(1, int(outer_iters) + 1):
            demand_g = Demand(matrix=od_from_pairs(prior_matrix, pairs, g))
            p, _, _ = proportion_matrix(network, demand_g, k_inner, pairs=pairs, engine=engine)
            sp_calls += int(k_inner)
            p_obs = p[sensors]
            if it == 1:
                # Seed the best iterate from the prior at its own proportions, so
                # the outer fixed point never returns worse than its start.
                best_resid = _obs_rmse(p_obs, g_pr, counts_mean)
            g, _, consistent = vzw_balance(g, p_obs, counts_mean, balance_passes, tol=tol)
            resid = _obs_rmse(p_obs, g, counts_mean)
            coords = BudgetCoords(iterations=it, sp_calls=sp_calls, wall_ms=0.0)
            done = budget.exhausted(coords)
            if it % stride == 0 or it == int(outer_iters) or done:
                # Sparse emission: each checkpoint is a full pinned certificate
                # (ADR-002 Decision 2). Always keep the final iterate.
                trace.record(
                    od_from_pairs(prior_matrix, pairs, g),
                    coords,
                    obs_count_rmse=resid,
                    counts_consistent=float(consistent),
                )
            if resid < best_resid:
                best_resid, best_g, best_consistent = resid, g.copy(), consistent
            if done:
                break
        # Re-record the best self-obs-RMSE iterate as the final artifact: the
        # outer assign<->balance fixed point can walk into a bypass-saturated
        # regime under noisy counts, so keep the best-observed iterate (ADR-002
        # Decision 3.1 safeguard, mirroring gls/spiess). len==0 keeps the trace
        # non-empty when every stride point fell between records.
        if len(trace) == 0 or best_resid < resid:
            coords = BudgetCoords(
                iterations=coords.iterations + 1, sp_calls=sp_calls, wall_ms=0.0
            )
            trace.record(
                od_from_pairs(prior_matrix, pairs, best_g),
                coords,
                obs_count_rmse=best_resid,
                counts_consistent=float(best_consistent),
            )

        return ODResultBundle(
            estimator_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
