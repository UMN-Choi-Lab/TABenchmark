"""TABenchmark: a shared benchmark for 50 years of traffic assignment models.

See docs/ARCHITECTURE.md for the design and docs/REFERENCES.md for the
verified reference canon this benchmark implements.
"""

from .core import (
    Budget,
    BudgetCoords,
    Capabilities,
    ContaminationError,
    Demand,
    FactorSpec,
    FlowState,
    Network,
    ReferenceSolution,
    ResultBundle,
    RngBundle,
    Scenario,
    Trace,
)
from .data import braess_scenario, load_scenario, two_route_scenario
from .experiments import run_experiment
from .metrics import Evaluator, nrmse, rmse
from .models import (
    MODEL_REGISTRY,
    AllOrNothingModel,
    BiconjugateFrankWolfeModel,
    CallableModel,
    ConjugateFrankWolfeModel,
    DialSUEModel,
    FrankWolfeModel,
    MSAModel,
    TrafficAssignmentModel,
    register_model,
)

__version__ = "0.2.0"

__all__ = [
    "Budget",
    "BudgetCoords",
    "Capabilities",
    "ContaminationError",
    "Demand",
    "FactorSpec",
    "FlowState",
    "Network",
    "ReferenceSolution",
    "ResultBundle",
    "RngBundle",
    "Scenario",
    "Trace",
    "braess_scenario",
    "load_scenario",
    "two_route_scenario",
    "run_experiment",
    "Evaluator",
    "nrmse",
    "rmse",
    "MODEL_REGISTRY",
    "AllOrNothingModel",
    "BiconjugateFrankWolfeModel",
    "CallableModel",
    "ConjugateFrankWolfeModel",
    "DialSUEModel",
    "FrankWolfeModel",
    "MSAModel",
    "TrafficAssignmentModel",
    "register_model",
    "__version__",
]
