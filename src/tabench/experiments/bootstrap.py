"""Nonparametric bootstrap confidence intervals for the stochastic track (P5).

Macroreplications produce independent certified statistics (e.g. the final
certified probit residual across ``M`` solver trajectories, adr-003 Decision
4). This module aggregates them into a percentile confidence interval of the
mean, resampling on the reserved ``SOURCE_BOOTSTRAP`` stream so the interval is
byte-reproducible from ``root_seed`` alone (P8).

Percentile, never parametric (P5): the macrorep spread is small and skewed, so
a normal-theory interval would misstate coverage. ``run_experiment`` already
lists ``SOURCE_BOOTSTRAP`` in its manifest but never drew from it — this closes
that gap.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..core.rng import SOURCE_BOOTSTRAP, RngBundle

__all__ = ["BootstrapCI", "bootstrap_ci", "bootstrap_curve_band"]


class BootstrapCI(NamedTuple):
    """A percentile bootstrap confidence interval of the mean."""

    point: float  # mean of the observed values
    lo: float  # lower percentile of the resampled means
    hi: float  # upper percentile of the resampled means
    level: float  # nominal coverage, e.g. 0.95


def bootstrap_ci(
    values: np.ndarray,
    root_seed: int,
    b: int = 10000,
    level: float = 0.95,
) -> BootstrapCI:
    """Percentile bootstrap CI of the mean of ``values``.

    Draws ``b`` resamples of size ``len(values)`` with replacement, indexing on
    ``RngBundle(root_seed, macrorep=0).generator(SOURCE_BOOTSTRAP)``, and takes
    the central ``level`` percentiles of the resampled means. Deterministic in
    ``(values, root_seed, b, level)``.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty 1-D array")
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    point = float(values.mean())
    n = values.size
    gen = RngBundle(root_seed, macrorep=0).generator(SOURCE_BOOTSTRAP)
    idx = gen.integers(0, n, size=(b, n))
    means = values[idx].mean(axis=1)
    alpha = 1.0 - level
    lo, hi = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return BootstrapCI(point=point, lo=float(lo), hi=float(hi), level=level)


def bootstrap_curve_band(
    curve_matrix: np.ndarray,
    root_seed: int,
    b: int = 10000,
    level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Functional percentile bootstrap band of a mean curve over macroreplications.

    ``curve_matrix`` is ``(M, K)``: ``M`` macrorep curves each evaluated on a
    shared ``K``-point mesh. Draws ``b`` resamples of the ``M`` curves with
    replacement on the reserved ``SOURCE_BOOTSTRAP`` stream, forms each
    resample's pointwise mean curve, and returns the central ``level`` percentiles
    of those mean curves at each mesh point as ``(lo, hi)`` arrays.

    This is the curve-valued sibling of :func:`bootstrap_ci`: one resampling
    level (macroreps only), the same stream discipline, byte-reproducible from
    ``root_seed`` alone (P8). Identical macrorep curves give a zero-width band.
    A resample mean that touches a censored ``+inf`` stays ``+inf`` and the
    percentile is inf-honest (never ``NaN``, unlike ``np.percentile``'s
    ``inf - inf`` interpolation). Requires ``M >= 2`` (a single curve has no
    sampling spread; mirrors ``bootstrap_ci``'s ``values.size > 1`` gate). Used by
    ``experiments.profiles`` for progress-curve confidence bands (P5).
    """
    matrix = np.asarray(curve_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("curve_matrix must be an (M, K) array with M >= 2 macrorep curves")
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    m = matrix.shape[0]
    gen = RngBundle(root_seed, macrorep=0).generator(SOURCE_BOOTSTRAP)
    idx = gen.integers(0, m, size=(b, m))
    means = matrix[idx].mean(axis=1)  # (b, K)
    alpha = 1.0 - level
    lo = _inf_aware_percentile(means, 100.0 * alpha / 2.0)
    hi = _inf_aware_percentile(means, 100.0 * (1.0 - alpha / 2.0))
    return lo, hi


def _inf_aware_percentile(samples: np.ndarray, q: float) -> np.ndarray:
    """Column-wise linear-interpolation percentile that is honest at ``+inf``.

    ``np.percentile`` interpolates ``inf - inf`` to ``NaN`` when the bracketing
    order statistics are both infinite; here an infinite bracket yields that
    infinity (the honest censored value), never ``NaN``.
    """
    ordered = np.sort(samples, axis=0)
    n = ordered.shape[0]
    pos = (q / 100.0) * (n - 1)
    lo_i = int(np.floor(pos))
    hi_i = int(np.ceil(pos))
    frac = pos - lo_i
    low = ordered[lo_i]
    high = ordered[hi_i]
    # Mask both brackets to finite BEFORE subtracting so no inf - inf ever runs
    # (np.where would still evaluate the discarded branch); then resolve the
    # infinite-bracket cells explicitly to the reached infinity — never a NaN or a
    # swallowed RuntimeWarning (adr-032 M3).
    both_finite = np.isfinite(low) & np.isfinite(high)
    diff = np.where(both_finite, high, 0.0) - np.where(both_finite, low, 0.0)
    out = low + frac * diff
    out = np.where(np.isinf(high) & (frac > 0.0), high, out)
    out = np.where(np.isinf(low), low, out)
    out = np.where(low == high, low, out)
    return out
