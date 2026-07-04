"""spiess: Spiess's (1990, CRT-693) gradient OD-adjustment.

Bilevel descent on the count misfit ``Z(g) = 1/2 sum_a (v_a(g) - hat c_a)^2``
under locally constant proportions. From the sparse ``P`` the gradient is
``dZ/dg_ij = sum_a p^a_ij (v_a - hat c_a)`` (misfit accumulated along used
paths, one pass), and the update is multiplicative,
``g_ij <- g_ij (1 - lambda grad_ij)`` — which preserves nonnegativity and never
creates trips for zero-prior pairs (Spiess's stated design feature: the prior's
*structure* is the regularizer).

The step is the linearized optimum ``lambda* = sum_a w_a (v_a - hat c_a) /
sum_a w_a^2`` with ``w_a = sum_ij p^a_ij g_ij grad_ij``, capped at
``1 / max_ij grad_ij`` for feasibility. Because ``Z`` is nonconvex through the
equilibrium map, the Armijo safeguard is applied **retrospectively**: after each
outer re-assignment we compare ``Z`` under the fresh proportions with the last
accepted iterate's; if it rose, the step overshot the equilibrium map, so we
revert to that iterate and halve a persistent damping factor (up to
``max_halvings`` times) before re-stepping — no extra assignment, one inner
assignment (``k_inner`` shortest-path calls) per outer iteration. A best
self-obs-RMSE iterate is kept so the descent never returns something it measures
as worse than its own starting point (ADR-002 Decision 3). Deterministic.
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

__all__ = ["spiess_step", "SpiessEstimator"]


def spiess_step(
    p_obs: np.ndarray, g: np.ndarray, counts: np.ndarray, damp: float = 1.0
) -> np.ndarray:
    """One damped Spiess descent step at fixed proportions ``p_obs``.

    Returns ``max(g * (1 - damp * lambda* * grad), 0)`` with the linearized
    optimum ``lambda*`` capped at ``1 / max grad`` for feasibility. If no descent
    direction exists (degenerate ``w``) or the optimal step is nonpositive, ``g``
    is returned unchanged. The Armijo safeguard is applied *retrospectively* in
    the outer loop (``SpiessEstimator``), where ``Z`` is re-assessed under a
    fresh assignment — the frozen-proportion ``Z`` this step minimizes is convex
    in ``lambda`` and would never trigger a halving (ADR-002 Decision 3.3).
    """
    g = np.asarray(g, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    v = p_obs @ g
    misfit = v - counts
    grad = p_obs.T @ misfit
    w = p_obs @ (g * grad)
    denom = float(w @ w)
    if denom <= 0.0:
        return g.copy()
    lam = float(w @ misfit) / denom
    grad_max = float(grad.max()) if grad.size else 0.0
    if grad_max > 0.0:
        lam = min(lam, 1.0 / grad_max)
    if lam <= 0.0:
        return g.copy()
    return np.maximum(g * (1.0 - float(damp) * lam * grad), 0.0)


@register_estimator
class SpiessEstimator(ODEstimator):
    """Gradient OD adjustment on the count misfit (Spiess 1990)."""

    name = "spiess"
    capabilities = _estimation_capabilities(deterministic=True)
    factors = {
        "k_inner": FactorSpec(
            default=60, kind="int", bounds=(1, 5000),
            doc="Inner MSA/AON sweeps per proportion extraction (Decision 3).",
        ),
        "outer_iters": FactorSpec(
            default=40, kind="int", bounds=(1, 5000),
            doc="Outer descent (assign<->step) iterations.",
        ),
        "max_halvings": FactorSpec(
            default=8, kind="int", bounds=(0, 64),
            doc="Armijo halvings of the linearized step before accepting it.",
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
        max_halvings = self.factor_values["max_halvings"]

        g_pr = np.array([prior_matrix[i, j] for (i, j) in pairs], dtype=np.float64)
        g = g_pr.copy()
        best_g, best_resid = g_pr.copy(), np.inf
        # Retrospective-Armijo state: the last accepted iterate, the proportions
        # extracted there, its re-assigned misfit Z, and a persistent damping.
        prev_g, prev_p_obs, prev_z = g_pr.copy(), None, np.inf
        damp = 1.0
        halvings = 0
        sp_calls = 0
        coords = BudgetCoords(iterations=0, sp_calls=0, wall_ms=0.0)
        resid = np.inf
        stride = max(1, int(outer_iters) // 15)
        for it in range(1, int(outer_iters) + 1):
            demand_g = Demand(matrix=od_from_pairs(prior_matrix, pairs, g))
            p, _, _ = proportion_matrix(network, demand_g, k_inner, pairs=pairs, engine=engine)
            sp_calls += int(k_inner)
            p_obs = p[sensors]
            misfit = p_obs @ g - counts_mean
            z_cur = 0.5 * float(misfit @ misfit)  # Z under a fresh re-assignment
            resid = float(np.sqrt(np.mean(misfit**2)))
            if it > 1 and z_cur > prev_z and halvings < max_halvings:
                # The last step raised Z under re-assignment (Z is nonconvex
                # through the equilibrium map): revert to the last accepted
                # iterate, halve the persistent damping, and re-step from it
                # using its already-extracted proportions (no new assignment).
                halvings += 1
                damp *= 0.5
                g = spiess_step(prev_p_obs, prev_g, counts_mean, damp=damp)
                continue
            coords = BudgetCoords(iterations=it, sp_calls=sp_calls, wall_ms=0.0)
            if it == 1:
                best_resid = resid  # seed the best iterate from the prior itself
            if resid < best_resid:
                best_resid, best_g = resid, g.copy()
            done = budget.exhausted(coords)
            if it % stride == 0 or it == int(outer_iters) or done:
                # Sparse emission: each checkpoint is a full pinned certificate
                # (ADR-002 Decision 2). Always keep the final iterate.
                trace.record(od_from_pairs(prior_matrix, pairs, g), coords, obs_count_rmse=resid)
            if done:
                break
            prev_g, prev_p_obs, prev_z = g.copy(), p_obs, z_cur
            g = spiess_step(p_obs, g, counts_mean, damp=damp)
        # Re-record the best self-obs-RMSE iterate as the final artifact so the
        # outer descent cannot return a strictly dominated last iterate
        # (mirrors vzw/spsa's safeguards; ADR-002 Decision 3, item 3). The
        # len==0 guard keeps the trace non-empty if every stride point was a
        # reverted step.
        if len(trace) == 0 or best_resid < resid:
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
