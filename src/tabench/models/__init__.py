"""Benchmark models: white-box solvers and black-box adapters."""

from .adapters import CallableModel
from .aon import AllOrNothingModel
from .base import MODEL_REGISTRY, TrafficAssignmentModel, register_model
from .frank_wolfe import BiconjugateFrankWolfeModel, ConjugateFrankWolfeModel, FrankWolfeModel
from .gradient_projection import GradientProjectionModel
from .msa import MSAModel
from .so import SystemOptimumModel, marginal_network
from .sue_logit import DialSUEModel

__all__ = [
    "CallableModel",
    "AllOrNothingModel",
    "MODEL_REGISTRY",
    "TrafficAssignmentModel",
    "register_model",
    "BiconjugateFrankWolfeModel",
    "ConjugateFrankWolfeModel",
    "DialSUEModel",
    "FrankWolfeModel",
    "GradientProjectionModel",
    "MSAModel",
    "SystemOptimumModel",
    "marginal_network",
]
