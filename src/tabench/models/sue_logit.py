"""Logit stochastic user equilibrium via MSA with Dial's STOCH loading.

The reference SUE solver: method of successive averages (Powell & Sheffi
1982 predetermined step sizes) around the pinned Dial loading map (Dial
1971; Sheffi 1985 ch. 11-12). Fisk's (1980) equivalent convex program is the
theory anchor, but its entropy term is a function of *path* flows, which
emitted link flows do not determine — so no objective is reported and the
certificate is the SUE fixed-point residual ``||v - L(t(v), theta)||_1 /
total demand`` instead (docs/design/adr-001).

``theta`` is task data read from ``scenario.sue_theta`` — never a model
factor, so no solver can tune the problem definition (P7). Dial's loading is
closed-form and deterministic ("stochastic" refers to traveler perception),
so this model runs on the deterministic track.
"""

from __future__ import annotations

import time

import numpy as np

from ..core.budget import Budget, BudgetCoords
from ..core.capabilities import Capabilities
from ..core.results import ResultBundle, Trace
from ..core.rng import RngBundle
from ..core.scenario import Scenario
from ._stoch import StochEngine
from .base import TrafficAssignmentModel, register_model

__all__ = ["DialSUEModel"]


@register_model
class DialSUEModel(TrafficAssignmentModel):
    """MSA around Dial-STOCH loading: v_{k+1} = v_k + (L(t(v_k)) - v_k)/k."""

    name = "sue-msa"
    capabilities = Capabilities(
        paradigm="sue",
        deterministic=True,
        provides_gap=True,
        seedable=True,
    )

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        theta = scenario.sue_theta
        if theta is None:
            raise ValueError(
                "sue-msa requires an SUE scenario (scenario.sue_theta is None); "
                "theta is task data, not a model factor"
            )
        start = time.perf_counter()
        network = scenario.network
        engine = StochEngine(network)
        total = scenario.demand.total
        loads = 0

        # Iteration 0: stochastic loading at free-flow costs.
        v = engine.load(
            network.link_cost(np.zeros(network.n_links)), scenario.demand, theta
        )
        loads += 1

        k = 0
        while True:
            k += 1
            costs = network.link_cost(v)
            y = engine.load(costs, scenario.demand, theta)
            loads += 1
            residual = float(np.abs(y - v).sum() / total) if total > 0 else 0.0

            coords = BudgetCoords(
                iterations=k,
                sp_calls=loads,  # one Dial load ~ one AON sweep (same unit)
                wall_ms=1000.0 * (time.perf_counter() - start),
            )
            trace.record(v, coords, sue_fixed_point_residual=residual)

            # The convergence target applies to this model's self-monitored
            # convergence measure: the SUE fixed-point residual (ADR-001).
            if budget.exhausted(coords) or budget.target_met(residual):
                break
            v = v + (y - v) / k

        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
