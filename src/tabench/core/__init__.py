"""Core abstractions: scenarios, capabilities, budgets, traces, randomness."""

from .budget import Budget, BudgetCoords
from .capabilities import Capabilities, ContaminationError, assert_fair_evaluation
from .factors import FactorSpec, resolve_factors
from .results import FlowState, ResultBundle, Trace
from .rng import (
    SOURCE_BOOTSTRAP,
    SOURCE_EVALUATION,
    SOURCE_OBSERVATION,
    RngBundle,
)
from .scenario import Demand, Network, ReferenceSolution, Scenario

__all__ = [
    "Budget",
    "BudgetCoords",
    "Capabilities",
    "ContaminationError",
    "assert_fair_evaluation",
    "FactorSpec",
    "resolve_factors",
    "FlowState",
    "ResultBundle",
    "Trace",
    "RngBundle",
    "SOURCE_OBSERVATION",
    "SOURCE_EVALUATION",
    "SOURCE_BOOTSTRAP",
    "Demand",
    "Network",
    "ReferenceSolution",
    "Scenario",
]
