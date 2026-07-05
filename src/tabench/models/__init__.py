"""Benchmark models: white-box solvers and black-box adapters."""

from .adapters import CallableModel
from .algb import AlgorithmBModel
from .aon import AllOrNothingModel
from .base import MODEL_REGISTRY, TrafficAssignmentModel, register_model
from .dtd_swap import RouteSwapDTDModel
from .elastic import ElasticDemandFWModel
from .evans import EvansCombinedModel
from .frank_wolfe import BiconjugateFrankWolfeModel, ConjugateFrankWolfeModel, FrankWolfeModel
from .gradient_projection import GradientProjectionModel
from .learned import LearnedSurrogateModel
from .msa import MSAModel
from .oba import OriginBasedModel
from .so import SystemOptimumModel, marginal_network
from .sue_logit import DialSUEModel
from .sue_probit import SueProbitMsaModel
from .tapas import TapasModel

__all__ = [
    "AlgorithmBModel",
    "CallableModel",
    "AllOrNothingModel",
    "MODEL_REGISTRY",
    "TrafficAssignmentModel",
    "register_model",
    "BiconjugateFrankWolfeModel",
    "ConjugateFrankWolfeModel",
    "DialSUEModel",
    "RouteSwapDTDModel",
    "ElasticDemandFWModel",
    "EvansCombinedModel",
    "SueProbitMsaModel",
    "FrankWolfeModel",
    "GradientProjectionModel",
    "LearnedSurrogateModel",
    "MSAModel",
    "OriginBasedModel",
    "SystemOptimumModel",
    "TapasModel",
    "marginal_network",
]
