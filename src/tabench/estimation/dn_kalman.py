"""od-kalman: Davis & Nihan (1993) linear-Gaussian OD estimation from a time
series of link counts.

Davis & Nihan (1993, *Operations Research* 41(1):169-178) prove that as the
traveler population grows, a general stochastic assignment process converges to a
deterministic mean (the SUE/UE loading, Prop 2) plus a stationary linear-Gaussian
fluctuation -- a VAR(1)/VARMA(1,1) process whose covariance is their eq. (8)
(Prop 3). Their stated application (Section 4, "Implications and Implementation")
is exactly this estimator: *"the VAR(v) process admits a state-space description,
which can then be combined with the discrete-time Kalman filter to produce an
algorithm for computing the likelihood function of a time-series of observed link
volume counts. This, in turn, permits estimation of model parameters, such as the
underlying travel demands, via prediction-error minimization."*

For a **static** OD demand observed through a stationary Gaussian process, the
prediction-error / Kalman estimate is the generalized least squares estimate
whitened by the covariance of the count *time-mean* (the steady-state Kalman gain
for a constant state reduces to this batch BLUE). Two things make that covariance
non-classical and make ``od-kalman`` distinct from ``gls`` / ``od-congested``,
which both collapse the series to a mean vector and discard everything else:

1. **Cross-link (spatial) covariance.** The DN innovation covariance
   ``Q = sum_j (g_j^2/N_j) Delta_j^T (diag p_j - p_j p_j^T) Delta_j`` is
   *non-diagonal* -- links sharing a route are correlated. ``od-kalman`` whitens
   by the full sample covariance of the count series, not ``gls``'s diagonal
   Poisson ``V``.
2. **Temporal (autocorrelation) correction.** Day-to-day persistence ``rho``
   makes consecutive counts dependent, so ``T`` correlated observations carry the
   information of only ``T / tau`` independent ones, where ``tau`` is the
   integrated autocorrelation time. ``od-kalman`` inflates the count-mean
   covariance by ``tau`` (an effective-sample-size correction); a naive mean
   (``tau = 1``) over-trusts correlated counts. For an AR(1) the *asymptotic*
   (large-``T``) integrated autocorrelation is ``tau = (1 + rho) / (1 - rho)`` --
   the reduction of Davis & Nihan's VARMA(1,1) autocorrelation to its pure-AR(1)
   form; ``sigma_mean = sigma_stat * tau / T`` uses that asymptotic constant, so
   at small ``T`` it *conservatively* overstates the count-mean variance (biasing
   slightly toward the prior), the standard effective-sample-size approximation.

Solution -- the same congested outer fixed point as ``gls`` / ``od-congested``
(Cascetta & Postorino 2001): assign the current estimate to freeze the
equilibrium proportions ``P``, solve the DN-whitened GLS for ``g``, re-assign and
repeat. A best-self-obs-RMSE iterate is kept so the outer loop never returns a
strictly dominated last iterate (ADR-002 Decision 3). Certificate (P1): the
harness recomputes the count-fit and OD-fit from the emitted OD matrix through a
pinned reference assignment, so ``self_report`` here is provenance only.

Scope. This ships the DN linear-Gaussian estimator for the *stationary VAR(1)*
count process emitted by :class:`~tabench.observe.levels.DayToDayCounts` (Prop 3
centered on the UE loading). The temporal structure is fit as an AR(1) integrated
autocorrelation; the full VARMA(1,1) MA term (which arises when the latent
anticipation ``g(t)`` is eliminated, p. 175) is *not* separately identified -- the
VAR(1) generator makes the pure-AR(1) form correct (no MA term to cancel), so
``rho_hat`` estimates the AR(1) persistence directly. The self-report and outer
loop mirror ``gls`` piece for piece.

Sourcing. Davis & Nihan (1993, ``davis1993large``) is the primary; the paper was
read (JSTOR stable/171951, Operations Research 41(1)). The eq. (8) covariance and
the Kalman/prediction-error program are Section 4; the SUE mean is Prop 2. No
number from the paper is reproduced -- the anchors are hand-derived closed forms.
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

__all__ = ["dn_gls_solve", "ar1_tau", "DavisNihanKalmanEstimator"]


def ar1_tau(counts: np.ndarray) -> tuple[float, float]:
    """AR(1) integrated-autocorrelation time of a count series (``T`` x ``S``).

    Estimates the pooled lag-1 autocorrelation as the trace ratio
    ``sum_{t,s} z_{t,s} z_{t+1,s} / sum_{t,s} z_{t,s}^2`` of the demeaned series
    (``z = counts - counts.mean(0)``) and returns ``(tau, rho_hat)`` with
    ``tau = (1 + rho_hat) / (1 - rho_hat)`` -- the variance-inflation factor of
    the time-mean of an AR(1). Summing each sensor's OWN autocovariance (rather
    than pooling sensors into a scalar mean) is robust to cross-link
    anti-correlation: for a VAR(1) with ``Phi = rho I`` and innovation covariance
    ``Q`` the numerator is ``rho * tr(Q)`` and the denominator ``tr(Q)``, so
    ``rho_hat -> rho`` even when the monitored links are on competing routes whose
    fluctuations cancel in the mean (a mean-pooled estimate collapses to
    ``tau ~ 1`` there, silently discarding the temporal correction). ``rho_hat``
    is clipped to ``[0, 0.98]`` (negative / anti-persistent -> no inflation,
    ``tau = 1``; the ceiling keeps ``tau`` finite); ``tau = 1`` for an IID series.
    """
    z = np.asarray(counts, dtype=np.float64)
    z = z - z.mean(axis=0, keepdims=True)
    if z.shape[0] < 3:
        return 1.0, 0.0
    denom = float(np.sum(z * z))
    if denom <= 0.0:
        return 1.0, 0.0
    rho = float(np.sum(z[:-1] * z[1:]) / denom)
    rho = min(max(rho, 0.0), 0.98)
    return (1.0 + rho) / (1.0 - rho), rho


def _inv_sqrt_psd(cov: np.ndarray, rtol: float = 1e-10, abs_floor: float = 1e-12) -> np.ndarray:
    """Symmetric inverse square root of a PSD matrix via eigendecomposition.

    Eigenvalues below ``max(rtol * max_eigenvalue, abs_floor)`` are floored (the
    DN count covariance is near-singular by flow conservation) so the whitening
    stays finite. The **absolute** floor is load-bearing: without it an exactly
    singular covariance (e.g. a caller who overrides ``cov_ridge=0`` on a
    single-route sub-network, where ``Q`` is rank-deficient) drives the relative
    floor to a denormal and the whitening to ``~1e155`` -- a finite-but-corrupted
    solve that the ``isfinite`` certificate would not catch. With ``abs_floor`` a
    zero-variance direction gets a large-but-finite weight (a noise-free
    observation, fit tightly) and no arithmetic overflows. For a ``1 x 1``
    covariance above the floor this is exactly ``1 / sqrt(cov)``.
    """
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    w, v = np.linalg.eigh(cov)
    floor = max(rtol * w.max(initial=0.0), abs_floor)
    w = np.where(w > floor, w, floor)
    return (v / np.sqrt(w)) @ v.T


def dn_gls_solve(
    p_obs: np.ndarray,
    counts_mean: np.ndarray,
    prior: np.ndarray,
    sigma_mean: np.ndarray,
    w_var: np.ndarray,
) -> np.ndarray:
    """Nonnegative GLS whitened by the FULL Davis-Nihan count-mean covariance.

    ``argmin_{g>=0} (P g - cbar)^T Sigma_mean^{-1} (P g - cbar)
                    + (g - g_pr)^T W^{-1} (g - g_pr)``,

    solved as a bounded least squares on the whitened stacked system
    ``[W^-1/2 ; Sigma_mean^-1/2 P] g ~ [W^-1/2 g_pr ; Sigma_mean^-1/2 cbar]``.
    ``sigma_mean`` is the ``(S, S)`` covariance of the count *time-mean*
    (``Sigma_stat * tau / T``); its off-diagonals (cross-link DN structure) and
    the ``tau`` inflation are what distinguish this from ``gls`` (diagonal ``V``,
    ``tau = 1``). Because ``W^-1`` is strictly positive definite the problem is
    strictly convex for any sensor set. For a single pair and single sensor with
    scalar ``Sigma_mean = s^2`` and ``W = w^2`` this is the closed form
    ``(g_pr / w^2 + p * cbar / s^2) / (1 / w^2 + p^2 / s^2)``.
    """
    prior = np.asarray(prior, dtype=np.float64)
    counts_mean = np.asarray(counts_mean, dtype=np.float64)
    p_obs = np.atleast_2d(np.asarray(p_obs, dtype=np.float64))
    n_pairs = prior.size
    w_root = 1.0 / np.sqrt(w_var)
    s_inv_root = _inv_sqrt_psd(np.atleast_2d(sigma_mean))
    top = np.diag(w_root)
    bottom = s_inv_root @ p_obs
    a = np.vstack([top, bottom])
    b = np.concatenate([w_root * prior, s_inv_root @ counts_mean])
    result = lsq_linear(a, b, bounds=(0.0, np.inf), method="bvls")
    return np.asarray(result.x, dtype=np.float64).reshape(n_pairs)


@register_estimator
class DavisNihanKalmanEstimator(ODEstimator):
    """Davis & Nihan (1993) linear-Gaussian OD estimation from a count series."""

    name = "od-kalman"
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
        "cov_ridge": FactorSpec(
            default=1e-3, kind="float", bounds=(0.0, 1e12),
            doc="Ridge added to the sample count covariance so the DN whitening is "
            "well-posed when periods are few (T < S+2) or a sensor is near-constant.",
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
        n_periods, n_sensors = counts.shape
        counts_mean = counts.mean(axis=0)

        k_inner = self.factor_values["k_inner"]
        outer_iters = self.factor_values["outer_iters"]
        cv_prior = self.factor_values["cv_prior"]
        floor = self.factor_values["prior_var_floor"]
        ridge = self.factor_values["cov_ridge"]

        # Temporal (AR) + spatial (cross-link) DN covariance of the count time-mean.
        tau, _rho = ar1_tau(counts)
        if n_periods >= n_sensors + 2:
            sigma_stat = np.atleast_2d(np.cov(counts, rowvar=False))
        else:
            var = counts.var(axis=0, ddof=1) if n_periods > 1 else np.ones(n_sensors)
            sigma_stat = np.diag(np.maximum(var, 0.0))
        sigma_stat = sigma_stat + ridge * np.eye(n_sensors)
        sigma_mean = sigma_stat * (tau / max(n_periods, 1))

        g_pr = np.array([prior_matrix[i, j] for (i, j) in pairs], dtype=np.float64)
        if g_pr.size == 0:
            # No positive off-diagonal prior support: emit the prior unchanged
            # (graceful empty-support handling, mirrors gls/od-congested).
            coords = BudgetCoords(iterations=1, sp_calls=0, wall_ms=0.0)
            trace.record(prior_matrix, coords, obs_count_rmse=0.0)
            return ODResultBundle(
                estimator_name=self.name,
                final=trace.final,
                trace=trace,
                factors=dict(self.factor_values),
                seed_info=rng.describe(),
            )
        w_var = (cv_prior * g_pr) ** 2 + floor

        g = g_pr.copy()
        best_g = g_pr.copy()
        best_resid = np.inf
        sp_calls = 0
        resid = np.inf
        coords = BudgetCoords(iterations=0, sp_calls=0, wall_ms=0.0)
        stride = max(1, int(outer_iters) // 15)
        for it in range(1, int(outer_iters) + 1):
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
                best_resid = float(np.sqrt(np.mean((p_obs @ g_pr - counts_mean) ** 2)))
            g = dn_gls_solve(p_obs, counts_mean, g_pr, sigma_mean, w_var)
            coords = BudgetCoords(iterations=it, sp_calls=sp_calls, wall_ms=0.0)
            resid = float(np.sqrt(np.mean((p_obs @ g - counts_mean) ** 2)))
            if resid < best_resid:
                best_resid, best_g = resid, g.copy()
            done = budget.exhausted(coords)
            # Sparse emission (ADR-002 Decision 2): every checkpoint triggers a
            # full pinned bfw re-assignment, so emit ~15 spaced points plus always
            # the final iterate, never one per iteration (mirrors od-congested).
            if it % stride == 0 or it == int(outer_iters) or done:
                trace.record(
                    od_from_pairs(prior_matrix, pairs, g), coords, obs_count_rmse=resid
                )
            if done:
                break
        # Re-record the best self-obs-RMSE iterate as the final artifact so the
        # outer fixed point cannot return a strictly dominated last iterate
        # (mirrors gls/spiess/od-congested; ADR-002 Decision 3, item 2).
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
