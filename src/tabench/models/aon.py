"""All-or-nothing assignment: the capacity-blind pre-equilibrium baseline.

Historically the standard practice before Beckmann's formulation was solvable
at scale; kept as a genuine benchmark model so the harness demonstrably
certifies non-equilibrium methods honestly (large external gap, no exclusion).
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

__all__ = ["AllOrNothingModel"]


@register_model
class AllOrNothingModel(TrafficAssignmentModel):
    """Assign every trip to its free-flow shortest path."""

    name = "aon"
    capabilities = Capabilities(
        paradigm="heuristic",
        deterministic=True,
        provides_gap=False,
        seedable=True,
    )

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        start = time.perf_counter()
        engine = PathEngine(scenario.network)
        free_flow_costs = scenario.network.link_cost(np.zeros(scenario.network.n_links))
        flows, _ = engine.all_or_nothing(free_flow_costs, scenario.demand)
        coords = BudgetCoords(
            iterations=1, sp_calls=1, wall_ms=1000.0 * (time.perf_counter() - start)
        )
        trace.record(flows, coords)
        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
