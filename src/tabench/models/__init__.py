"""Benchmark models: white-box solvers and black-box adapters."""

from .adapters import CallableModel
from .algb import AlgorithmBModel
from .aon import AllOrNothingModel
from .base import MODEL_REGISTRY, TrafficAssignmentModel, register_model
from .br_ue import BoundedlyRationalUEModel
from .dtd_friesz import FrieszDTDModel
from .dtd_horowitz import CostSmoothingSUEModel
from .dtd_link import LinkBasedDTDModel
from .dtd_stochastic import CascettaStochasticProcessModel
from .dtd_swap import RouteSwapDTDModel
from .dtd_swap_sue import RouteSwapSUEModel
from .dtd_unifying import UnifyingDTDModel
from .elastic import ElasticDemandFWModel
from .evans import EvansCombinedModel
from .frank_wolfe import BiconjugateFrankWolfeModel, ConjugateFrankWolfeModel, FrankWolfeModel
from .gradient_projection import GradientProjectionModel
from .learned import LearnedSurrogateModel
from .msa import MSAModel
from .oba import OriginBasedModel
from .sc_tap import SideConstrainedModel
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
    "CostSmoothingSUEModel",
    "DialSUEModel",
    "BoundedlyRationalUEModel",
    "CascettaStochasticProcessModel",
    "FrieszDTDModel",
    "LinkBasedDTDModel",
    "RouteSwapDTDModel",
    "RouteSwapSUEModel",
    "UnifyingDTDModel",
    "ElasticDemandFWModel",
    "EvansCombinedModel",
    "SueProbitMsaModel",
    "FrankWolfeModel",
    "GradientProjectionModel",
    "LearnedSurrogateModel",
    "MSAModel",
    "OriginBasedModel",
    "SideConstrainedModel",
    "SystemOptimumModel",
    "TapasModel",
    "marginal_network",
]
