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
