"""gls: Cascetta's (1984) generalized-least-squares OD estimator.

Balances a quadratic deviation from the prior against a quadratic count misfit:

    ghat = argmin_{g>=0} (g - g_pr)' W^-1 (g - g_pr) + (cbar - P g)' V^-1 (cbar - P g)

with ``W = diag((cv_prior * g_pr)^2 + eps)`` (prior quality from the task card)
and ``V = diag(max(cbar_a, 1) / n_periods)`` (the Poisson variance of a
period-mean count). We solve it as a nonnegative bounded least-squares problem on
the whitened stacked system ``[W^-1/2 ; V^-1/2 P] g ~ [W^-1/2 g_pr ; V^-1/2 cbar]``
(``scipy.optimize.lsq_linear``, bounds ``[0, inf)``) — restoring the
nonnegativity Cascetta notes but drops for his closed form.

Because ``W^-1`` is strictly positive definite, the problem is strictly convex
and has a unique minimizer for *any* sensor set: GLS stays well-posed even when
the counts alone do not identify the demand, which is why it is the default T2
baseline. Deterministic; cost ``outer_iters * k_inner`` shortest-path calls.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

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

__all__ = ["gls_solve", "GLSEstimator"]


def gls_solve(
    p_obs: np.ndarray,
    counts: np.ndarray,
    prior: np.ndarray,
    w_var: np.ndarray,
    v_var: np.ndarray,
) -> np.ndarray:
    """Nonnegative GLS estimate on the whitened stacked system.

    ``w_var`` / ``v_var`` are the prior and count variances (the diagonals of
    ``W`` and ``V``). Returns the minimizing ``g >= 0``. For a single pair and a
    single sensor with ``W = V = 1`` this is the closed form
    ``(g_pr + p*cbar) / (1 + p^2)`` (Decision-6 anchor 4).
    """
    prior = np.asarray(prior, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    w_root = 1.0 / np.sqrt(w_var)
    v_root = 1.0 / np.sqrt(v_var)
    n_pairs = prior.size
    top = np.diag(w_root)
    bottom = v_root[:, None] * np.asarray(p_obs, dtype=np.float64)
    a = np.vstack([top, bottom])
    b = np.concatenate([w_root * prior, v_root * counts])
    result = lsq_linear(a, b, bounds=(0.0, np.inf), method="bvls")
    return np.asarray(result.x, dtype=np.float64).reshape(n_pairs)


@register_estimator
class GLSEstimator(ODEstimator):
    """Generalized-least-squares OD estimation (Cascetta 1984)."""

    name = "gls"
    capabilities = _estimation_capabilities(deterministic=True)
    factors = {
        "k_inner": FactorSpec(
            default=60, kind="int", bounds=(1, 5000),
            doc="Inner MSA/AON sweeps per proportion extraction (Decision 3).",
        ),
        "outer_iters": FactorSpec(
            default=15, kind="int", bounds=(1, 5000),
            doc="Outer assign<->estimate fixed-point iterations.",
        ),
        "cv_prior": FactorSpec(
            default=0.3, kind="float", bounds=(1e-6, 100.0),
            doc="Assumed prior coefficient of variation (sets W); matches the card cv.",
        ),
        "prior_var_floor": FactorSpec(
            default=1e-6, kind="float", bounds=(0.0, 1e12),
            doc="eps added to the prior variance so W^-1 stays finite for tiny cells.",
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
        counts = np.asarray(task.dataset.payload["counts"], dtype=np.float64)
        n_periods = counts.shape[0]
        counts_mean = counts.mean(axis=0)

        k_inner = self.factor_values["k_inner"]
        outer_iters = self.factor_values["outer_iters"]
        cv_prior = self.factor_values["cv_prior"]
        floor = self.factor_values["prior_var_floor"]

        g_pr = np.array([prior_matrix[i, j] for (i, j) in pairs], dtype=np.float64)
        w_var = (cv_prior * g_pr) ** 2 + floor
        v_var = np.maximum(counts_mean, 1.0) / max(n_periods, 1)

        g = g_pr.copy()
        best_g = g_pr.copy()
        best_resid = np.inf
        sp_calls = 0
        for it in range(1, int(outer_iters) + 1):
            demand_g = Demand(matrix=od_from_pairs(prior_matrix, pairs, g))
            p, _, _ = proportion_matrix(network, demand_g, k_inner, pairs=pairs, engine=engine)
            sp_calls += int(k_inner)
            p_obs = p[sensors]
            if it == 1:
                # Seed the best iterate from the prior, measured at its own
                # proportions, so the estimate can never certify worse than its
                # own starting point (ADR-002 Decision 3 outer-loop safeguard).
                best_resid = float(np.sqrt(np.mean((p_obs @ g_pr - counts_mean) ** 2)))
            g = gls_solve(p_obs, counts_mean, g_pr, w_var, v_var)
            coords = BudgetCoords(iterations=it, sp_calls=sp_calls, wall_ms=0.0)
            od = od_from_pairs(prior_matrix, pairs, g)
            resid = float(np.sqrt(np.mean((p_obs @ g - counts_mean) ** 2)))
            trace.record(od, coords, obs_count_rmse=resid)
            if resid < best_resid:
                best_resid, best_g = resid, g.copy()
            if budget.exhausted(coords):
                break
        # Re-record the best self-obs-RMSE iterate as the final artifact so the
        # outer fixed point cannot return a strictly dominated last iterate
        # (mirrors vzw/spsa's safeguards; ADR-002 Decision 3, item 2).
        if best_resid < resid:
            coords = BudgetCoords(
                iterations=coords.iterations + 1, sp_calls=sp_calls, wall_ms=0.0
            )
            trace.record(
                od_from_pairs(prior_matrix, pairs, best_g), coords, obs_count_rmse=best_resid
            )

        return ODResultBundle(
            estimator_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
