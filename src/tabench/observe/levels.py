"""Observation processes: seeded projections of ground truth (P3).

Ground truth (equilibrium link flows; later, realized route flows) is stored
once; each data level derives a dataset from it deterministically given an
RNG stream. Per Hazelton (2015, AOAS805):

* per-period counts are distributed, never day-averages — the dependence
  pattern across periods carries information the means do not;
* identifiability from counts requires distinct nonzero columns of the
  route-link incidence matrix restricted to monitored links, so the check is
  provided here and sensor configurations should report it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.scenario import Scenario

__all__ = [
    "Dataset",
    "DataLevel",
    "FullOD",
    "LinkCounts",
    "DayToDayCounts",
    "DynamicLinkCounts",
    "StalePriorOD",
    "random_sensor_mask",
    "distinct_nonzero_columns",
]


@dataclass(frozen=True)
class Dataset:
    """One observed dataset with generation metadata."""

    kind: str
    payload: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


class DataLevel(ABC):
    """An observation process applied to scenario ground truth."""

    name: str = "abstract"

    @abstractmethod
    def observe(
        self, scenario: Scenario, link_flows: np.ndarray, rng: np.random.Generator
    ) -> Dataset:
        """Project ground truth into an observed dataset."""


class FullOD(DataLevel):
    """Complete origin-destination table: the classical full-information level."""

    name = "full_od"

    def observe(
        self, scenario: Scenario, link_flows: np.ndarray, rng: np.random.Generator
    ) -> Dataset:
        return Dataset(
            kind=self.name,
            payload={"od_matrix": scenario.demand.matrix.copy()},
            meta={"n_zones": scenario.demand.n_zones},
        )


class LinkCounts(DataLevel):
    """Per-period counts on a monitored subset of links.

    ``noise='poisson'`` draws each period's count as Poisson with mean equal
    to the ground-truth link flow (independent across periods);
    ``noise='none'`` repeats exact flows.
    """

    name = "link_counts"

    def __init__(
        self, sensor_links: np.ndarray, n_periods: int = 1, noise: str = "poisson"
    ) -> None:
        if noise not in ("poisson", "none"):
            raise ValueError(f"Unknown noise model {noise!r}")
        if n_periods < 1:
            raise ValueError("n_periods must be >= 1")
        self.sensor_links = np.asarray(sensor_links, dtype=np.int64)
        self.n_periods = int(n_periods)
        self.noise = noise

    def observe(
        self, scenario: Scenario, link_flows: np.ndarray, rng: np.random.Generator
    ) -> Dataset:
        truth = np.asarray(link_flows, dtype=np.float64)[self.sensor_links]
        if self.noise == "poisson":
            counts = rng.poisson(lam=np.tile(truth, (self.n_periods, 1))).astype(np.float64)
        else:
            counts = np.tile(truth, (self.n_periods, 1))
        return Dataset(
            kind=self.name,
            payload={"counts": counts, "sensor_links": self.sensor_links.copy()},
            meta={
                "n_periods": self.n_periods,
                "noise": self.noise,
                "coverage": len(self.sensor_links) / scenario.network.n_links,
            },
        )


class DayToDayCounts(DataLevel):
    """Davis & Nihan (1993) large-population day-to-day link-count series (ADR-012).

    A benchmark realization of the Davis-Nihan Gaussian limit (Prop 3): the
    monitored link counts are a stationary vector-autoregressive VAR(1) process

        x(t) = x_UE + e(t),   e(t) = rho * e(t-1) + a(t),   a(t) ~ N(0, (1-rho^2) Q),

    so ``e(t)`` has stationary covariance ``Q`` -- the route-level DN multinomial
    covariance (:func:`~tabench.observe._dn_process.dn_spatial_covariance`) -- and
    the series is centered on ``x_UE = P g`` (the deterministic UE loading the T2
    certifier pins, so an SUE certifier is not required). ``rho`` in ``[0, 1)`` is
    the day-to-day persistence dial standing in for Davis-Nihan's cost-adjustment
    memory; ``rho = 0`` gives IID (only cross-link-correlated) counts. The finite
    population is ``N_j = max(1, round(population_scale * g_j))`` per OD pair, so
    the fluctuation vanishes as ``1 / population_scale`` (Prop 2 SLLN) and the
    process reduces to ``noise='none'`` as ``population_scale -> inf``.

    The temporal (AR) and cross-link (off-diagonal ``Q``) structure is exactly
    what ``od-kalman`` exploits and the classical (``mean``-collapsing) T2
    estimators discard; counts are the large-``N`` Gaussian object, so individual
    periods may be non-integer or (rarely, at low flow) negative -- faithful to
    the limit, and the certifier scores the period *mean*, which is ``x_UE``.

    The observed and held-out sensor sets are drawn as *independent* realizations
    of this process (separate RNG substreams, as for ``LinkCounts``); both are
    centered on ``x_UE``, so the mean-based held-out metric stays consistent. The
    cross-link correlation this level carries is exploited *within* the observed
    series, not across the obs/held-out split.
    """

    name = "day_to_day_counts"

    def __init__(
        self,
        sensor_links: np.ndarray,
        n_periods: int = 30,
        population_scale: float = 50.0,
        rho: float = 0.5,
        k_inner: int = 80,
    ) -> None:
        if n_periods < 1:
            raise ValueError("n_periods must be >= 1")
        if not np.isfinite(population_scale) or population_scale <= 0:
            raise ValueError("population_scale must be finite and > 0")
        if not (0.0 <= rho < 1.0):
            raise ValueError("rho (day-to-day persistence) must be in [0, 1)")
        if k_inner < 1:
            raise ValueError("k_inner must be >= 1")
        self.sensor_links = np.asarray(sensor_links, dtype=np.int64)
        self.n_periods = int(n_periods)
        self.population_scale = float(population_scale)
        self.rho = float(rho)
        self.k_inner = int(k_inner)

    def observe(
        self, scenario: Scenario, link_flows: np.ndarray, rng: np.random.Generator
    ) -> Dataset:
        from ._dn_process import active_od_pairs, dn_spatial_covariance, psd_factor

        x_ue = np.asarray(link_flows, dtype=np.float64)
        n_links = x_ue.size
        demand = scenario.demand
        pairs = active_od_pairs(demand.matrix)
        g = np.array([demand.matrix[i, j] for (i, j) in pairs], dtype=np.float64)
        n_trav = np.maximum(1, np.rint(self.population_scale * g)).astype(np.int64)
        q_cov = dn_spatial_covariance(
            scenario.network, demand, g, n_trav, self.k_inner, pairs=pairs
        )
        factor = psd_factor(q_cov)  # (n_links, n_links); factor @ factor.T == Q
        rank = factor.shape[1]
        inn_scale = float(np.sqrt(max(1.0 - self.rho * self.rho, 0.0)))

        counts_full = np.empty((self.n_periods, n_links), dtype=np.float64)
        e = factor @ rng.standard_normal(rank)  # e(0) ~ N(0, Q): start stationary
        for t in range(self.n_periods):
            counts_full[t] = x_ue + e
            a = inn_scale * (factor @ rng.standard_normal(rank))
            e = self.rho * e + a
        counts = counts_full[:, self.sensor_links]

        return Dataset(
            kind=self.name,
            payload={"counts": counts, "sensor_links": self.sensor_links.copy()},
            meta={
                "n_periods": self.n_periods,
                "noise": "day_to_day",
                "population_scale": self.population_scale,
                "rho": self.rho,
                "coverage": len(self.sensor_links) / scenario.network.n_links,
            },
        )


class DynamicLinkCounts(DataLevel):
    """Within-day time-sliced link counts on a monitored subset (ADR-023).

    Cascetta, Inaudi & Marquis (1993) observe counts *per interval* ``t = 1..T``
    on monitored links, generated from a time-slice demand profile through the
    exogenous lagged assignment map. This level takes the already-computed
    **expected interval crossing counts** ``(T, n_links)`` (``sum_l M[l] @
    d*_{t-l}`` on the full network, from :func:`~tabench.estimation._dynamic_map.
    predict_interval_counts`) and draws the observed series:

    * ``noise='poisson'`` draws each ``(day, interval, sensor)`` count as Poisson
      with mean equal to the expected crossing count (independent across days —
      day-to-day sampling of the same within-day profile);
    * ``noise='none'`` repeats the exact expected counts ``n_days`` times.

    The estimand is the ``(H, Z, Z)`` profile, so unlike ``LinkCounts`` the count
    axis ``t`` is *never* collapsed to a mean by the certifier — averaging over
    ``t`` is exactly the information-destroying move the dynamic anchors punish.
    Payload: ``counts`` ``(n_days, T, S)`` and the ``sensor_links``.
    """

    name = "dynamic_link_counts"

    def __init__(
        self, sensor_links: np.ndarray, n_days: int = 1, noise: str = "poisson"
    ) -> None:
        if noise not in ("poisson", "none"):
            raise ValueError(f"Unknown noise model {noise!r}")
        if n_days < 1:
            raise ValueError("n_days must be >= 1")
        self.sensor_links = np.asarray(sensor_links, dtype=np.int64)
        self.n_days = int(n_days)
        self.noise = noise

    def observe(
        self, scenario: Scenario, link_flows: np.ndarray, rng: np.random.Generator
    ) -> Dataset:
        """``link_flows`` here is the expected interval-crossing counts ``(T, n_links)``."""
        expected = np.asarray(link_flows, dtype=np.float64)
        if expected.ndim != 2:
            raise ValueError("DynamicLinkCounts expects (T, n_links) expected crossing counts")
        truth = expected[:, self.sensor_links]  # (T, S)
        if self.noise == "poisson":
            counts = rng.poisson(
                lam=np.tile(truth, (self.n_days, 1, 1))
            ).astype(np.float64)
        else:
            counts = np.tile(truth, (self.n_days, 1, 1))
        return Dataset(
            kind=self.name,
            payload={"counts": counts, "sensor_links": self.sensor_links.copy()},
            meta={
                "n_days": self.n_days,
                "n_intervals": int(expected.shape[0]),
                "noise": self.noise,
                "coverage": len(self.sensor_links) / scenario.network.n_links,
            },
        )


class StalePriorOD(DataLevel):
    """A degraded target/seed OD matrix: truth times multiplicative Gamma noise.

    Each positive off-diagonal truth cell is scaled by an i.i.d. draw from a
    Gamma with mean 1 and coefficient of variation ``cv`` (``shape = 1/cv**2``,
    ``scale = cv**2``); ``cv = 0`` returns the truth unchanged. Zero cells stay
    zero — a survey knows which OD pairs exist even when it misstates their
    size, so the truth's *support* leaks (a limitation stated on every T2 card;
    a uniform-support variant is a one-line dial). Intrazonal (diagonal) demand
    is passed through unchanged: it never enters the network and is never
    estimated.

    Drawn on the reserved ``SOURCE_PRIOR`` stream so the prior is independent
    of the observed counts (``SOURCE_OBSERVATION``).
    """

    name = "stale_prior_od"

    def __init__(self, cv: float = 0.3) -> None:
        if not np.isfinite(cv) or cv < 0:
            raise ValueError(f"cv must be finite and >= 0, got {cv!r}")
        self.cv = float(cv)

    def observe(
        self, scenario: Scenario, link_flows: np.ndarray, rng: np.random.Generator
    ) -> Dataset:
        truth = scenario.demand.matrix
        prior = truth.copy()
        if self.cv > 0:
            off = ~np.eye(truth.shape[0], dtype=bool)
            positive = off & (truth > 0)
            shape = 1.0 / self.cv**2
            scale = self.cv**2
            noise = rng.gamma(shape=shape, scale=scale, size=int(positive.sum()))
            prior[positive] = truth[positive] * noise
        return Dataset(
            kind=self.name,
            payload={"prior_od": prior},
            meta={
                "n_zones": scenario.demand.n_zones,
                "cv": self.cv,
                "support": "truth (zero cells stay zero; support leak stated on card)",
            },
        )


def random_sensor_mask(
    n_links: int, coverage: float, rng: np.random.Generator
) -> np.ndarray:
    """Deterministically sample a sensor subset at the given coverage fraction."""
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    n_sensors = max(1, round(coverage * n_links))
    return np.sort(rng.choice(n_links, size=n_sensors, replace=False))


def distinct_nonzero_columns(incidence: np.ndarray) -> bool:
    """Hazelton's identifiability condition on a (monitored links x routes) matrix.

    Returns True iff every column is nonzero and no two columns are equal —
    the condition under which mean route-flow parameters are identifiable
    from repeated link-count observations (Hazelton 2015, Prop. 1).
    """
    a = np.asarray(incidence)
    if a.ndim != 2:
        raise ValueError("incidence must be a 2-D (links x routes) array")
    if np.any(~a.any(axis=0)):
        return False
    unique = np.unique(a, axis=1)
    return unique.shape[1] == a.shape[1]
