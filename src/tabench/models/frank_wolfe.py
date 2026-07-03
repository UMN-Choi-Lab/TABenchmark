"""Frank-Wolfe (convex combinations) for deterministic user equilibrium.

The link-based workhorse of four decades of practice (Frank & Wolfe 1956;
LeBlanc, Morlok & Pierskalla 1975): all-or-nothing subproblem + exact line
search on the Beckmann objective along the search direction.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import brentq

from ..core.budget import Budget, BudgetCoords
from ..core.capabilities import Capabilities
from ..core.factors import FactorSpec
from ..core.results import ResultBundle, Trace
from ..core.rng import RngBundle
from ..core.scenario import Network, Scenario
from ._paths import PathEngine
from .base import TrafficAssignmentModel, register_model

__all__ = ["FrankWolfeModel"]


def _line_search(network: Network, v: np.ndarray, d: np.ndarray, xtol: float) -> float:
    """Exact step: root of g(a) = t(v + a d) . d on [0, 1].

    g is nondecreasing (link costs are nondecreasing in flow and d is fixed),
    g(0) = SPTT - TSTT <= 0; if g(1) <= 0 the full step is optimal.
    """

    def g(alpha: float) -> float:
        return float(network.link_cost(v + alpha * d) @ d)

    g0 = g(0.0)
    if g0 >= 0.0:
        return 0.0
    if g(1.0) <= 0.0:
        return 1.0
    return float(brentq(g, 0.0, 1.0, xtol=xtol))


@register_model
class FrankWolfeModel(TrafficAssignmentModel):
    """Link-based Frank-Wolfe with exact line search."""

    name = "fw"
    capabilities = Capabilities(
        paradigm="static_ue",
        deterministic=True,
        provides_gap=True,
        seedable=True,
    )
    factors = {
        "line_search_xtol": FactorSpec(
            default=1e-12,
            kind="float",
            bounds=(1e-16, 1e-3),
            doc="Absolute tolerance of the Brent line search on the step size.",
        ),
    }

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        start = time.perf_counter()
        network = scenario.network
        engine = PathEngine(network)
        xtol = self.factor_values["line_search_xtol"]
        sp_calls = 0

        v, _ = engine.all_or_nothing(
            network.link_cost(np.zeros(network.n_links)), scenario.demand
        )
        sp_calls += 1

        k = 0
        while True:
            k += 1
            costs = network.link_cost(v)
            y, sptt = engine.all_or_nothing(costs, scenario.demand)
            sp_calls += 1
            tstt = float(v @ costs)
            gap = (tstt - sptt) / tstt if tstt > 0 else 0.0
            objective = float(network.link_cost_integral(v).sum())

            coords = BudgetCoords(
                iterations=k,
                sp_calls=sp_calls,
                wall_ms=1000.0 * (time.perf_counter() - start),
            )
            trace.record(
                v, coords, relative_gap=gap, tstt=tstt, sptt=sptt, beckmann=objective
            )

            if budget.exhausted(coords):
                break
            d = y - v
            alpha = _line_search(network, v, d, xtol)
            if alpha <= 0.0:
                break  # first-order optimal at current point
            v = v + alpha * d

        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
