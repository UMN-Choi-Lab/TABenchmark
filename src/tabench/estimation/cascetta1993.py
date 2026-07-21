"""od-dynamic: Cascetta, Inaudi & Marquis (1993) within-day dynamic OD estimation.

Two estimators recover a time-slice OD profile ``d_h`` (``h = 1..H``) from
time-sliced link counts ``c_{t,a}`` (``t = 1..T``) through the exogenous lagged
assignment map (:mod:`._dynamic_map`) — the paper's defining contribution:

* **simultaneous** (``od-dynamic-sim``): one whitened nonnegative GLS over the
  full block-lower-banded stacked system (all slices jointly), the statistically
  efficient estimator;
* **sequential** (``od-dynamic-seq``): slice by slice, each slice estimated from
  its earliest observed interval with the earlier slices frozen and subtracted
  (no covariance propagation), the online-capable but provably less efficient
  estimator.

Both are single-shot GLS solves reusing the ``gls`` whitened-stacked NNLS pattern
(``scipy.optimize.lsq_linear``, ``bounds=(0, inf)``); there is no inner assignment
and no outer fixed point, because the map is exogenous and demand-independent
(congestion feedback is ``cascetta2001fixed``, out of scope), so ``sp_calls = 0``
for both — the T2 sp-call budget is inert on this track (documented, not a fake
charge). The certificate (P1) recomputes the per-interval count-fit and the
descriptive OD-fit from the emitted profile through a harness-regenerated map
(:mod:`tabench.metrics.estimation_dynamic`); ``self_report`` is provenance only.

The GLS core acts on a **general** block map (``blocks[t] = {h: block}``), so it
covers a time-varying stacked map (the A2b pure-math pin) as well as the
benchmark's time-invariant tensor. Sourcing and anchors: docs/design/adr-023.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from ..core.budget import Budget, BudgetCoords
from ..core.factors import FactorSpec
from ..core.rng import RngBundle
from ._dynamic_map import stacked_tensor_map, tensor_blocks
from ._proportions import od_from_pairs
from .base import ODResultBundle, ODTrace
from .dynamic_base import (
    DynamicEstimationTask,
    DynamicODEstimator,
    _dynamic_estimation_capabilities,
    register_dynamic_estimator,
)

__all__ = [
    "dynamic_gls_simultaneous",
    "dynamic_gls_sequential",
    "SimultaneousDynamicGLSEstimator",
    "SequentialDynamicGLSEstimator",
]


def _nnls(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    result = lsq_linear(a, b, bounds=(0.0, np.inf), method="bvls")
    return np.asarray(result.x, dtype=np.float64).reshape(n)


def dynamic_gls_simultaneous(
    a_stacked: np.ndarray,
    counts: np.ndarray,
    prior: np.ndarray,
    w_prior: np.ndarray,
    v_count: np.ndarray,
) -> np.ndarray:
    """One whitened nonnegative GLS on the full stacked system (all slices jointly).

    ``argmin_{x>=0} (x - z)^T W^-1 (x - z) + (A x - c)^T V^-1 (A x - c)`` on the
    stacked unknown ``x`` (``H*P``), solved as bounded least squares on the
    whitened system ``[W^-1/2 ; V^-1/2 A] x ~ [W^-1/2 z ; V^-1/2 c]``. ``a_stacked``
    is ``(T*S, H*P)``; ``counts`` / ``v_count`` are ``(T*S,)`` (per interval-sensor
    count and its variance); ``prior`` / ``w_prior`` are ``(H*P,)`` (per slice-pair
    prior and its variance). Because ``W^-1`` is positive definite the problem is
    strictly convex for any sensor set. Returns the flat ``(H*P,)`` estimate.
    """
    prior = np.asarray(prior, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    w_root = 1.0 / np.sqrt(np.asarray(w_prior, dtype=np.float64))
    v_root = 1.0 / np.sqrt(np.asarray(v_count, dtype=np.float64))
    top = np.diag(w_root)
    bottom = v_root[:, None] * np.asarray(a_stacked, dtype=np.float64)
    a = np.vstack([top, bottom])
    b = np.concatenate([w_root * prior, v_root * counts])
    return _nnls(a, b, prior.size)


def dynamic_gls_sequential(
    blocks: list[dict[int, np.ndarray]],
    counts_mean: np.ndarray,
    prior_profile: np.ndarray,
    w_prior: np.ndarray,
    v_count: np.ndarray,
    n_intervals: int,
) -> np.ndarray:
    """Slice-by-slice GLS: each slice from its earliest observed interval, frozen.

    For ``h = 0..H-1`` in order, find the earliest interval ``t_h >= h`` with a
    nonzero block for slice ``h`` (the slice's first crossing of a monitored link);
    subtract the frozen contributions of already-estimated *earlier* slices from
    ``c_{t_h}``; solve the slice-local nonnegative GLS for ``d_h`` against that
    block. The two information losses that make the sequential estimator provably
    less efficient than the simultaneous one: (i) the slice's own LATER crossings
    (intervals ``> t_h``) are discarded, and (ii) earlier-slice estimates are
    frozen with no covariance propagation. (Later-slice contamination *at* ``t_h``
    cannot occur on a time-invariant tensor — a later slice's earliest nonzero lag
    is the same, so its first crossing is strictly after ``t_h``; on a general
    time-varying block map any such contribution is treated as zero, a documented
    bias of the plug-in scheme.) A slice never observed within the horizon keeps
    its prior. ``blocks[t] = {h: (S,P)}``;
    ``counts_mean`` / ``v_count`` are ``(T, S)``; ``prior_profile`` / ``w_prior``
    are ``(H, P)``. Returns the ``(H, P)`` estimate.
    """
    prior_profile = np.asarray(prior_profile, dtype=np.float64)
    counts_mean = np.asarray(counts_mean, dtype=np.float64)
    dhat = prior_profile.copy()
    n_slices = prior_profile.shape[0]
    for h in range(n_slices):
        t_h = None
        for t in range(h, int(n_intervals)):
            block = blocks[t].get(h)
            if block is not None and np.any(block != 0.0):
                t_h = t
                break
        if t_h is None:
            continue  # never observed within horizon -> keep prior
        resid = counts_mean[t_h].copy()
        for hp, block in blocks[t_h].items():
            if hp < h:  # earlier slice, already estimated and frozen
                resid = resid - block @ dhat[hp]
        diag = np.atleast_2d(blocks[t_h][h])
        w_root = 1.0 / np.sqrt(w_prior[h])
        v_root = 1.0 / np.sqrt(v_count[t_h])
        a = np.vstack([np.diag(w_root), v_root[:, None] * diag])
        b = np.concatenate([w_root * prior_profile[h], v_root * resid])
        dhat[h] = _nnls(a, b, prior_profile.shape[1])
    return dhat


def _profile_pairs(prior_profile: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Extract the ``(H, P)`` per-slice per-pair demand from an ``(H, Z, Z)`` profile."""
    return np.array(
        [[prior_profile[h, i, j] for (i, j) in pairs] for h in range(prior_profile.shape[0])],
        dtype=np.float64,
    ).reshape(prior_profile.shape[0], len(pairs))


def _scatter_profile(
    prior_profile: np.ndarray, pairs: list[tuple[int, int]], d_pairs: np.ndarray
) -> np.ndarray:
    """Scatter an ``(H, P)`` estimate back into an ``(H, Z, Z)`` profile.

    Each slice's diagonal (intrazonal demand, never on the network) is carried
    from the prior slice; off-support cells stay zero (``od_from_pairs`` per slice).
    """
    return np.stack(
        [od_from_pairs(prior_profile[h], pairs, d_pairs[h]) for h in range(d_pairs.shape[0])]
    )


def _prepare(task: DynamicEstimationTask, cv_prior: float, floor: float):
    """Shared task unpacking: per-pair prior, counts mean, and the GLS variances."""
    payload = task.dataset.payload
    pairs = [(int(i), int(j)) for i, j in payload["pairs"]]
    m_obs = np.asarray(payload["lag_tensor"], dtype=np.float64)  # (L+1, S, P)
    counts = np.asarray(payload["counts"], dtype=np.float64)  # (n_days, T, S)
    n_days = counts.shape[0]
    counts_mean = counts.mean(axis=0)  # (T, S)
    n_intervals = counts_mean.shape[0]
    n_slices = int(task.dataset.meta["n_slices"])
    prior_pairs = _profile_pairs(task.prior_profile, pairs)  # (H, P)
    w_prior = (cv_prior * prior_pairs) ** 2 + floor  # (H, P)
    v_count = np.maximum(counts_mean, 1.0) / max(n_days, 1)  # (T, S)
    return pairs, m_obs, counts_mean, w_prior, v_count, n_slices, n_intervals


@register_dynamic_estimator
class SimultaneousDynamicGLSEstimator(DynamicODEstimator):
    """Simultaneous dynamic GLS (Cascetta et al. 1993): all slices jointly."""

    name = "od-dynamic-sim"
    capabilities = _dynamic_estimation_capabilities(deterministic=True)
    factors = {
        "cv_prior": FactorSpec(
            default=0.3, kind="float", bounds=(1e-6, 100.0),
            doc="Assumed prior coefficient of variation (sets W); matches the card cv.",
        ),
        "prior_var_floor": FactorSpec(
            default=1e-6, kind="float", bounds=(1e-12, 1e12),
            doc="eps added to the prior variance so W^-1 stays finite for tiny "
            "cells; strictly positive — floor 0 with a zero prior cell makes "
            "the whitened row infinite and hangs lsq_linear (adr-023 review).",
        ),
    }

    def estimate(
        self,
        task: DynamicEstimationTask,
        budget: Budget,
        rng: RngBundle,
        trace: ODTrace,
    ) -> ODResultBundle:
        cv_prior = self.factor_values["cv_prior"]
        floor = self.factor_values["prior_var_floor"]
        pairs, m_obs, counts_mean, w_prior, v_count, n_slices, n_intervals = _prepare(
            task, cv_prior, floor
        )
        n_pairs = len(pairs)
        if n_pairs == 0:
            coords = BudgetCoords(iterations=1, sp_calls=0, wall_ms=0.0)
            # Same residual formula as the normal path, evaluated at the emitted
            # (prior) profile: with no active pairs the map predicts all-zero
            # counts, so obs_count_rmse is the RMS of the period-mean counts.
            # Emitting it (not an empty self_report) keeps the key present --
            # parity with the normal record below.
            resid = self._self_obs_rmse(
                m_obs, _profile_pairs(task.prior_profile, pairs), counts_mean, n_intervals
            )
            trace.record(task.prior_profile, coords, obs_count_rmse=resid)
            return self._bundle(trace, rng)
        a_stacked = stacked_tensor_map(m_obs, n_slices, n_intervals)
        x = dynamic_gls_simultaneous(
            a_stacked,
            counts_mean.reshape(-1),
            _profile_pairs(task.prior_profile, pairs).reshape(-1),
            w_prior.reshape(-1),
            v_count.reshape(-1),
        ).reshape(n_slices, n_pairs)
        profile = _scatter_profile(task.prior_profile, pairs, x)
        coords = BudgetCoords(iterations=1, sp_calls=0, wall_ms=0.0)
        resid = self._self_obs_rmse(m_obs, x, counts_mean, n_intervals)
        trace.record(profile, coords, obs_count_rmse=resid)
        return self._bundle(trace, rng)

    @staticmethod
    def _self_obs_rmse(m_obs, x, counts_mean, n_intervals) -> float:
        from ._dynamic_map import predict_interval_counts

        pred = predict_interval_counts(m_obs, x, n_intervals)
        return float(np.sqrt(np.mean((pred - counts_mean) ** 2)))

    def _bundle(self, trace: ODTrace, rng: RngBundle) -> ODResultBundle:
        return ODResultBundle(
            estimator_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )


@register_dynamic_estimator
class SequentialDynamicGLSEstimator(DynamicODEstimator):
    """Sequential dynamic GLS (Cascetta et al. 1993): slice by slice, frozen carryover."""

    name = "od-dynamic-seq"
    capabilities = _dynamic_estimation_capabilities(deterministic=True)
    factors = {
        "cv_prior": FactorSpec(
            default=0.3, kind="float", bounds=(1e-6, 100.0),
            doc="Assumed prior coefficient of variation (sets W); matches the card cv.",
        ),
        "prior_var_floor": FactorSpec(
            default=1e-6, kind="float", bounds=(1e-12, 1e12),
            doc="eps added to the prior variance so W^-1 stays finite for tiny "
            "cells; strictly positive — floor 0 with a zero prior cell makes "
            "the whitened row infinite and hangs lsq_linear (adr-023 review).",
        ),
    }

    def estimate(
        self,
        task: DynamicEstimationTask,
        budget: Budget,
        rng: RngBundle,
        trace: ODTrace,
    ) -> ODResultBundle:
        cv_prior = self.factor_values["cv_prior"]
        floor = self.factor_values["prior_var_floor"]
        pairs, m_obs, counts_mean, w_prior, v_count, n_slices, n_intervals = _prepare(
            task, cv_prior, floor
        )
        n_pairs = len(pairs)
        if n_pairs == 0:
            coords = BudgetCoords(iterations=1, sp_calls=0, wall_ms=0.0)
            # Same residual formula as the normal path, evaluated at the emitted
            # (prior) profile: no active pairs -> the map predicts all-zero counts,
            # so obs_count_rmse is the RMS of the period-mean counts. Recording it
            # keeps the key present (parity with the normal record below), instead
            # of an empty self_report.
            from ._dynamic_map import predict_interval_counts

            pred = predict_interval_counts(
                m_obs, _profile_pairs(task.prior_profile, pairs), n_intervals
            )
            resid = float(np.sqrt(np.mean((pred - counts_mean) ** 2)))
            trace.record(task.prior_profile, coords, obs_count_rmse=resid)
            return self._bundle(trace, rng)
        blocks = tensor_blocks(m_obs, n_slices, n_intervals)
        x = dynamic_gls_sequential(
            blocks,
            counts_mean,
            _profile_pairs(task.prior_profile, pairs),
            w_prior,
            v_count,
            n_intervals,
        )
        profile = _scatter_profile(task.prior_profile, pairs, x)
        from ._dynamic_map import predict_interval_counts

        pred = predict_interval_counts(m_obs, x, n_intervals)
        resid = float(np.sqrt(np.mean((pred - counts_mean) ** 2)))
        coords = BudgetCoords(iterations=int(n_intervals), sp_calls=0, wall_ms=0.0)
        trace.record(profile, coords, obs_count_rmse=resid)
        return self._bundle(trace, rng)

    def _bundle(self, trace: ODTrace, rng: RngBundle) -> ODResultBundle:
        return ODResultBundle(
            estimator_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
