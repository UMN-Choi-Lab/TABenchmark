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

__all__ = ["BootstrapCI", "bootstrap_ci"]


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
