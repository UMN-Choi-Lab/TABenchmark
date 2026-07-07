"""Time-varying demand for dynamic network loading (dnl-core, adr-010).

Layer-1 component: pure numpy, deliberately imports neither ``grid.py`` nor
``fd.py`` — piecewise-constant OD departure rates need no time grid (the
cumulative demand curve is closed-form and grid-independent), and turning
fractions are demand-side routing data, not link physics.

Zone conventions follow the static side (core/scenario.py, P2/P9): zones are
1-based nodes ``1..n_zones``; ``rates[p, i, j]`` refers to the OD pair from
zone ``i+1`` to zone ``j+1``. Intrazonal demand never enters the network, so
the diagonal must be zero (same rule as the static :class:`~tabench.core.scenario.Demand`).

Coverage of :class:`TurningFractions` against actual network topology is
checked at ``DynamicScenario`` construction — this module stays Network-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DynamicDemand",
    "TurningFractions",
]


def _as_f64(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))


@dataclass(frozen=True)
class DynamicDemand:
    """Piecewise-constant OD departure rates on [0, T_demand].

    ``rates[p, i, j]`` = departure rate (veh/tu) from zone i+1 to zone j+1
    during ``[breakpoints[p], breakpoints[p+1])``. Cumulative demand
    ``D_ij(t)`` is the piecewise-linear integral — exact, grid-independent,
    hashable. The rate is zero before ``breakpoints[0] = 0`` and after
    ``breakpoints[-1]``, so the cumulative is 0 for t <= 0 and constant
    (= per-OD totals) beyond the last breakpoint.
    """

    breakpoints: np.ndarray  # (P+1,) float64, strictly increasing, breakpoints[0] = 0.0
    rates: np.ndarray  # (P, n_zones, n_zones) float64, >= 0, finite, zero diagonal

    def __post_init__(self) -> None:
        bp = _as_f64(self.breakpoints)
        rates = _as_f64(self.rates)
        object.__setattr__(self, "breakpoints", bp)
        object.__setattr__(self, "rates", rates)
        if bp.ndim != 1 or bp.shape[0] < 2:
            raise ValueError(
                "DynamicDemand breakpoints must be a 1-D array of P+1 >= 2 period edges"
            )
        if not np.isfinite(bp).all():
            raise ValueError("DynamicDemand breakpoints must be finite")
        if bp[0] != 0.0:
            raise ValueError(
                f"DynamicDemand breakpoints must start at 0.0 exactly, got {bp[0]!r} "
                "(demand shares the simulation clock; t0 = 0 always)"
            )
        if np.any(np.diff(bp) <= 0):
            raise ValueError("DynamicDemand breakpoints must be strictly increasing")
        if rates.ndim != 3 or rates.shape[1] != rates.shape[2]:
            raise ValueError("DynamicDemand rates must have shape (P, n_zones, n_zones)")
        if rates.shape[0] != bp.shape[0] - 1:
            raise ValueError(
                f"DynamicDemand rates has {rates.shape[0]} periods but breakpoints "
                f"define {bp.shape[0] - 1}"
            )
        if rates.shape[1] < 1:
            raise ValueError("DynamicDemand needs at least one zone")
        if not np.isfinite(rates).all():
            raise ValueError("DynamicDemand rates must be finite")
        if np.any(rates < 0):
            raise ValueError("DynamicDemand rates must be nonnegative")
        if np.any(np.diagonal(rates, axis1=1, axis2=2) != 0):
            raise ValueError(
                "DynamicDemand rates must have a zero diagonal "
                "(intrazonal demand never enters the network; same rule as static)"
            )

    @property
    def n_zones(self) -> int:
        return self.rates.shape[1]

    def cumulative(self, t: np.ndarray) -> np.ndarray:
        """D_ij(t): (len(t), n_zones, n_zones) piecewise-linear cumulative demand.

        Exact closed form (searchsorted + linear remainder), no quadrature.
        0 for t <= 0; constant after ``breakpoints[-1]`` (rate 0 beyond the
        last period). +/-inf query times are allowed (clipped); NaN raises.
        """
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        if t.ndim != 1:
            raise ValueError("DynamicDemand.cumulative expects a scalar or 1-D array of times")
        if np.isnan(t).any():
            raise ValueError("DynamicDemand.cumulative got NaN query times")
        bp = self.breakpoints
        n_periods = self.rates.shape[0]
        # cumulative demand at the period edges: (P+1, n_zones, n_zones)
        cum_edges = np.zeros((n_periods + 1, self.n_zones, self.n_zones))
        np.cumsum(np.diff(bp)[:, None, None] * self.rates, axis=0, out=cum_edges[1:])
        tc = np.clip(t, bp[0], bp[-1])  # rate is 0 outside [0, breakpoints[-1]]
        idx = np.searchsorted(bp, tc, side="right") - 1
        idx = np.clip(idx, 0, n_periods - 1)
        return cum_edges[idx] + (tc - bp[idx])[:, None, None] * self.rates[idx]

    def total(self) -> float:
        """sum_ij D_ij(inf) = sum_p (bp[p+1] - bp[p]) * rates[p].sum() — vehicle scale V."""
        return float(np.sum(np.diff(self.breakpoints)[:, None, None] * self.rates))


@dataclass(frozen=True)
class TurningFractions:
    """Exogenous, time-invariant turn splits (demand-side routing data).

    For each node with >= 1 incoming and >= 2 outgoing links: an
    ``(n_in, n_out)`` row-stochastic matrix aligned with the node's
    ASCENDING-sorted incoming/outgoing link-index lists. Entries are sorted by
    node id ascending (hash-canonical: one byte representation per content).
    Single-out nodes need no entry (implicit column of ones); an empty tuple
    is rejected — use ``turns=None`` on the scenario instead (otherwise None
    and ``()`` would hash identically, breaking canonicality).

    Time-varying turning fractions are v2 (leading time axis + domain-string
    bump; see the adr-010 risk register).
    """

    frac: tuple[tuple[int, np.ndarray], ...]  # ((node_id, matrix), ...) node_id ascending

    def __post_init__(self) -> None:
        entries = tuple((int(node_id), _as_f64(m)) for node_id, m in self.frac)
        object.__setattr__(self, "frac", entries)
        if len(entries) == 0:
            raise ValueError(
                "TurningFractions must contain at least one node entry "
                "(use turns=None on the scenario for no turn data)"
            )
        node_ids = [node_id for node_id, _ in entries]
        if any(b <= a for a, b in zip(node_ids, node_ids[1:], strict=False)):
            raise ValueError(
                "TurningFractions node ids must be strictly increasing "
                f"(hash-canonical ordering), got {node_ids}"
            )
        for node_id, m in entries:
            if m.ndim != 2 or m.shape[0] < 1 or m.shape[1] < 1:
                raise ValueError(
                    f"TurningFractions node {node_id}: matrix must be 2-D (n_in, n_out) "
                    "with at least one row and one column"
                )
            if not np.isfinite(m).all():
                raise ValueError(f"TurningFractions node {node_id}: matrix must be finite")
            if np.any(m < 0):
                raise ValueError(f"TurningFractions node {node_id}: matrix must be nonnegative")
            if np.any(np.abs(m.sum(axis=1) - 1.0) > 1e-12):
                raise ValueError(
                    f"TurningFractions node {node_id}: rows must sum to 1 within 1e-12"
                )
