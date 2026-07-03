"""Black-box wrapper for arbitrary Python callables (P4).

Wraps anything that maps a scenario to link flows — a trained GNN, a
surrogate, a heuristic — into the standard model contract. The harness then
certifies its output exactly as it does for white-box solvers (P1) and
enforces the training-lineage fairness gate (P7).
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from ...core.budget import Budget, BudgetCoords
from ...core.capabilities import Capabilities
from ...core.results import ResultBundle, Trace
from ...core.rng import RngBundle
from ...core.scenario import Scenario
from ..base import TrafficAssignmentModel

__all__ = ["CallableModel"]

FlowFn = Callable[[Scenario, np.random.Generator], np.ndarray]


class CallableModel(TrafficAssignmentModel):
    """Adapter turning ``fn(scenario, rng) -> link_flows`` into a benchmark model.

    ``capabilities`` are instance-level here (unlike library solvers, which
    declare them on the class), because they describe the wrapped artifact:
    pass ``trained_on`` lineage for learned models so the fairness gate can
    act on it.
    """

    name = "callable"

    def __init__(
        self,
        fn: FlowFn,
        name: str = "callable",
        paradigm: str = "learned",
        deterministic: bool = False,
        seedable: bool = True,
        trained_on: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self._fn = fn
        self.name = name
        self.capabilities = Capabilities(
            paradigm=paradigm,
            deterministic=deterministic,
            provides_gap=False,
            seedable=seedable,
            trained_on=trained_on,
        )

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        start = time.perf_counter()
        generator = rng.generator(source=0)
        flows = np.asarray(self._fn(scenario, generator), dtype=np.float64)
        if flows.shape != (scenario.network.n_links,):
            raise ValueError(
                f"Wrapped callable returned shape {flows.shape}, expected "
                f"({scenario.network.n_links},)"
            )
        coords = BudgetCoords(
            iterations=1, sp_calls=0, wall_ms=1000.0 * (time.perf_counter() - start)
        )
        trace.record(flows, coords)
        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors={},
            seed_info=rng.describe(),
        )
