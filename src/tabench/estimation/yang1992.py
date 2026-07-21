"""od-congested: Yang, Sasaki, Iida & Asakura (1992) bilevel ODME on congested networks.

Yang et al. confront the fact that on a CONGESTED network the assignment
proportions ``P`` depend on the very demand being estimated -- observed link
counts sit on *equilibrium* flows, not on a fixed-proportion loading. They cast
OD estimation as a BILEVEL program with a single scalar trade-off ``theta``:

    min_{g>=0}  theta * ||g - g_pr||^2  +  (1 - theta) * sum_{a in obs} (x_a(g) - cbar_a)^2
    s.t.        x(g) = UE(g)                                         [lower level]

The upper level balances staying near the prior/target ``g_pr`` against matching
the observed counts ``cbar``; the lower level pins ``x`` to user equilibrium.
``theta`` in ``(0, 1)`` is Yang's ``gamma1 / (gamma1 + gamma2)``: ``theta -> 1``
trusts the prior (and recovers it in the limit), ``theta -> 0`` fits the counts
(the prior then only fixes the support and nonnegativity). This is the
distinctive object -- a single deterministic trade-off, DIFFERENT from ``gls``'s
statistical covariance weighting (``W`` from prior CV, ``V`` from Poisson count
variance) and from ``spiess``'s count-misfit-only descent; the three coincide
only when ``W`` and ``V`` are proportional to the identity.

Solution -- the congested outer fixed point (Cascetta & Postorino 2001, the same
loop ``gls``/``spiess``/``vzw-entropy`` run). On each outer iteration we assign
the current estimate ``g`` to extract the equilibrium proportions ``P(g)`` (MSA
over all-or-nothing trees, :func:`~tabench.estimation._proportions.proportion_matrix`),
freeze them, and solve the now strictly-convex upper-level QP for ``g``; then we
re-assign and repeat until the demand and the congestion it induces are mutually
consistent. Freezing ``P`` makes the QP the Yang analog of the ``gls`` whitened
nonnegative least squares -- the scalar-``theta`` weighting in place of ``W``,
``V`` (see :func:`yang_solve`). In the UNCONGESTED limit ``P`` does not move with
``g``, so a single outer pass already lands the fixed point (an analytic anchor:
for one pair and one sensor with proportion ``p`` the estimate is the closed form
``g* = (theta*g_pr + (1-theta)*p*cbar) / (theta + (1-theta)*p^2)``).

Certificate (P1). Nothing model-specific: the harness recomputes the count-fit
and OD-fit from the emitted OD matrix through a pinned reference assignment
(:mod:`tabench.metrics.estimation`), so ``self_report`` here is provenance only,
exactly as for the shipped estimators. A best self-obs-RMSE iterate is kept so
the outer fixed point never returns a strictly dominated last iterate (ADR-002
Decision 3). Deterministic; cost ``outer_iters * k_inner`` shortest-path calls.

Scope. This ships Yang's OBJECTIVE (the theta-weighted bilevel program) solved by
the iterative assign-then-estimate scheme. Yang et al. also derive a
sensitivity-analysis-based (SAB) variant that linearizes the equilibrium flows
through their demand derivatives ``dv/dq``; that is NOT implemented here -- the
solution method is the shared frozen-proportion outer loop (faithful to the
bilevel program's iterative form, but not the SAB derivative method).

Sourcing. Yang, Sasaki, Iida & Asakura (1992, *Transportation Research Part B*
26(6):417-434, ``yang1992estimation``) is the canonical bilevel ODME-on-congested-
networks paper; the bilevel objective, the UE lower level and the underdetermination/
seed-regularizer structure are taken from the open Boyles et al. TNA sensitivity
chapter, and the congested outer fixed point is Cascetta & Postorino (2001), both
already shipped in ``_proportions.py``. No numbers from the primary are reproduced.
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

__all__ = ["yang_solve", "Yang1992Estimator"]


def yang_solve(
    p_obs: np.ndarray, counts: np.ndarray, prior: np.ndarray, theta: float
) -> np.ndarray:
    """Nonnegative minimizer of Yang's theta-weighted upper-level QP at fixed ``P``.

    ``argmin_{g>=0} theta * ||g - g_pr||^2 + (1 - theta) * ||P_obs g - cbar||^2``,
    solved as a whitened stacked nonnegative least squares with weight
    ``sqrt(theta)`` on the prior block and ``sqrt(1 - theta)`` on the count block
    (``scipy.optimize.lsq_linear``, bounded VLS). This is the scalar-``theta``
    case of the ``gls`` whitened system: the single deterministic trade-off in
    place of the statistical covariances ``W``, ``V``. Strictly convex for any
    ``theta`` in ``(0, 1)`` (the prior block alone has full column rank), so the
    minimizer is unique for any sensor set. For a single pair and single sensor
    with proportion ``p`` this is the closed form
    ``(theta*g_pr + (1-theta)*p*cbar) / (theta + (1-theta)*p^2)``.
    """
    prior = np.asarray(prior, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    p_obs = np.asarray(p_obs, dtype=np.float64)
    n_pairs = prior.size
    w_prior = float(np.sqrt(theta))
    w_count = float(np.sqrt(1.0 - theta))
    top = w_prior * np.eye(n_pairs)
    bottom = w_count * p_obs
    a = np.vstack([top, bottom])
    b = np.concatenate([w_prior * prior, w_count * counts])
    result = lsq_linear(a, b, bounds=(0.0, np.inf), method="bvls")
    return np.asarray(result.x, dtype=np.float64).reshape(n_pairs)


@register_estimator
class Yang1992Estimator(ODEstimator):
    """Bilevel OD estimation on congested networks (Yang et al. 1992)."""

    name = "od-congested"
    capabilities = _estimation_capabilities(deterministic=True)
    factors = {
        "theta": FactorSpec(
            default=0.5,
            kind="float",
            bounds=(1e-6, 1.0 - 1e-6),
            doc="Bilevel trade-off theta in (0,1): weight on staying near the prior "
            "vs matching the counts (Yang's gamma1/(gamma1+gamma2)). theta->1 trusts "
            "the prior and recovers it in the limit; theta->0 fits the counts, the "
            "prior only fixing the support and nonnegativity.",
        ),
        "k_inner": FactorSpec(
            default=60,
            kind="int",
            bounds=(1, 5000),
            doc="Inner MSA/AON sweeps per equilibrium proportion extraction (Decision 3).",
        ),
        "outer_iters": FactorSpec(
            default=15,
            kind="int",
            bounds=(1, 5000),
            doc="Outer assign<->estimate bilevel fixed-point iterations.",
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
        counts_mean = np.asarray(
            task.dataset.payload["counts"], dtype=np.float64
        ).mean(axis=0)

        theta = self.factor_values["theta"]
        k_inner = self.factor_values["k_inner"]
        outer_iters = self.factor_values["outer_iters"]

        g_pr = np.array([prior_matrix[i, j] for (i, j) in pairs], dtype=np.float64)
        if g_pr.size == 0:
            # No positive off-diagonal prior support: nothing to estimate. Emit the
            # prior unchanged (graceful empty-support handling, like spiess) rather
            # than passing a zero-column system to the least-squares solve. Empty
            # support predicts all-zero counts, so the honest self-obs residual is
            # RMS(counts_mean), matching the dynamic-family guards (never 0.0).
            coords = BudgetCoords(iterations=1, sp_calls=0, wall_ms=0.0)
            resid = float(np.sqrt(np.mean(counts_mean ** 2)))
            trace.record(prior_matrix, coords, obs_count_rmse=resid)
            return ODResultBundle(
                estimator_name=self.name,
                final=trace.final,
                trace=trace,
                factors=dict(self.factor_values),
                seed_info=rng.describe(),
            )
        g = g_pr.copy()
        best_g = g_pr.copy()
        best_resid = np.inf
        sp_calls = 0
        resid = np.inf
        coords = BudgetCoords(iterations=0, sp_calls=0, wall_ms=0.0)
        stride = max(1, int(outer_iters) // 15)
        for it in range(1, int(outer_iters) + 1):
            # Lower level: assign the current estimate to get the CONGESTED
            # proportions P(g) at (near-)equilibrium, then freeze them.
            demand_g = Demand(matrix=od_from_pairs(prior_matrix, pairs, g))
            p, _, _ = proportion_matrix(
                network, demand_g, k_inner, pairs=pairs, engine=engine
            )
            sp_calls += int(k_inner)
            p_obs = p[sensors]
            if it == 1:
                # Seed the best iterate from the prior at its own proportions so
                # the estimate can never certify worse than its starting point
                # (ADR-002 Decision 3 outer-loop safeguard).
                best_resid = float(
                    np.sqrt(np.mean((p_obs @ g_pr - counts_mean) ** 2))
                )
            # Upper level: the theta-weighted convex QP at frozen proportions.
            g = yang_solve(p_obs, counts_mean, g_pr, theta)
            coords = BudgetCoords(iterations=it, sp_calls=sp_calls, wall_ms=0.0)
            resid = float(np.sqrt(np.mean((p_obs @ g - counts_mean) ** 2)))
            if resid < best_resid:
                best_resid, best_g = resid, g.copy()
            done = budget.exhausted(coords)
            # Sparse emission (ADR-002 Decision 2): every checkpoint is a full
            # pinned certificate, so emit ~15 spaced points plus always the final
            # iterate, never one per iteration (mirrors spiess).
            if it % stride == 0 or it == int(outer_iters) or done:
                trace.record(
                    od_from_pairs(prior_matrix, pairs, g), coords, obs_count_rmse=resid
                )
            if done:
                break
        # Re-record the best self-obs-RMSE iterate as the final artifact so the
        # outer fixed point cannot return a strictly dominated last iterate
        # (mirrors gls/spiess/vzw; ADR-002 Decision 3, item 2).
        if best_resid < resid:
            coords = BudgetCoords(
                iterations=coords.iterations + 1, sp_calls=sp_calls, wall_ms=0.0
            )
            trace.record(
                od_from_pairs(prior_matrix, pairs, best_g),
                coords,
                obs_count_rmse=best_resid,
            )

        return ODResultBundle(
            estimator_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
