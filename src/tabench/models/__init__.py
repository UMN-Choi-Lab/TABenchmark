"""Benchmark models: white-box solvers and black-box adapters."""

from .adapters import CallableModel
from .aon import AllOrNothingModel
from .base import MODEL_REGISTRY, TrafficAssignmentModel, register_model
from .frank_wolfe import BiconjugateFrankWolfeModel, ConjugateFrankWolfeModel, FrankWolfeModel
from .msa import MSAModel
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
    "MSAModel",
]
