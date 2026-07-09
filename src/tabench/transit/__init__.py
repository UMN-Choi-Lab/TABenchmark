"""Transit assignment: Spiess & Florian (1989) optimal strategies.

A parallel module (like ``tabench.dnl``): its own frozen, content-hashed
:class:`TransitScenario` (a directed multigraph with per-arc frequencies, not the
road BPR :class:`~tabench.core.scenario.Network`), its own solver, and its own
certifier (``tabench.metrics.TransitEvaluator``). See
docs/design/adr-014-transit-strategy.md.
"""

from .builtin import (
    common_lines_expected_cost,
    common_lines_scenario,
    common_lines_unattractive_scenario,
)
from .network import (
    TransitDemand,
    TransitNetwork,
    TransitReference,
    TransitScenario,
    TransitStrategy,
)
from .strategy import OptimalStrategyModel, optimal_strategy

__all__ = [
    "TransitNetwork",
    "TransitDemand",
    "TransitScenario",
    "TransitStrategy",
    "TransitReference",
    "optimal_strategy",
    "OptimalStrategyModel",
    "common_lines_scenario",
    "common_lines_unattractive_scenario",
    "common_lines_expected_cost",
]
