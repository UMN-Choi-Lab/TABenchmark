"""Time-dependent SO/UE traffic assignment (Peeta & Mahmassani 1995, adr-031).

A parallel module (like ``dnl/``, ``dta/``, ``bottleneck/``): the first iterative
simulation-based time-dependent route-choice equilibrium in the benchmark —
fixed-departure TD-UE and its TD-SO twin, MSA over an enumerated per-OD path set,
the repo's own CTM/LTM loading as the constraint evaluator, and an experienced-
time route-swap certificate plus an lp-so-dta bound.
"""

from .artifact import TDPathFlows
from .builtin import (
    pm_corridor_scenario,
    pm_diamond_scenario,
    pm_merge_scenario,
    pm_wedge_scenario,
)
from .loader import PathLoader
from .scenario import TDPath, TDTAScenario
from .solve import solve_td_so, solve_td_ue

__all__ = [
    "TDPath",
    "TDTAScenario",
    "TDPathFlows",
    "PathLoader",
    "solve_td_ue",
    "solve_td_so",
    "pm_corridor_scenario",
    "pm_diamond_scenario",
    "pm_wedge_scenario",
    "pm_merge_scenario",
]
