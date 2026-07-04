"""Method of successive averages for deterministic user equilibrium.

The classical predetermined-step-size scheme: v_{k+1} = v_k + (y_k - v_k)/k,
with y_k the all-or-nothing assignment at costs t(v_k). Converges for the
convex UE program (though slowly); included as the canonical simple baseline.
"""

from __future__ import annotations

import time

import numpy as np

from ..core.budget import Budget, BudgetCoords
from ..core.capabilities import Capabilities
from ..core.results import ResultBundle, Trace
from ..core.rng import RngBundle
from ..core.scenario import Scenario
from ._paths import PathEngine
from .base import TrafficAssignmentModel, register_model

__all__ = ["MSAModel"]


@register_model
class MSAModel(TrafficAssignmentModel):
    """Method of successive averages with 1/k step sizes."""

    name = "msa"
    capabilities = Capabilities(
        paradigm="static_ue",
        deterministic=True,
        provides_gap=True,
        seedable=True,
    )

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        start = time.perf_counter()
        network = scenario.network
        engine = PathEngine(network)
        sp_calls = 0

        # Iteration 0: all-or-nothing at free-flow costs.
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

            coords = BudgetCoords(
                iterations=k,
                sp_calls=sp_calls,
                wall_ms=1000.0 * (time.perf_counter() - start),
            )
            trace.record(v, coords, relative_gap=gap, tstt=tstt, sptt=sptt)

            if budget.exhausted(coords) or budget.target_met(gap):
                break
            v = v + (y - v) / k

        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
