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

# Torch models are an OPTIONAL extra (``pip install tabench[torch]``): the
# numpy/scipy core must import without torch. Guard the import and swallow ONLY a
# missing-torch failure (``exc.name == 'torch'``) — any other ImportError is a
# real bug in the module and must propagate. When torch is absent the model is
# simply not registered, so ``MODEL_REGISTRY``/``tabench list`` lack it and the
# register_model invariant (every registered model is instantiable) is preserved.
try:
    from .implicit_ue import ImplicitUENNModel  # noqa: F401  (registered + conditional __all__)

    _HAS_TORCH = True
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by the torch-free CI legs
    if exc.name != "torch":
        raise
    _HAS_TORCH = False
from .multiclass import MulticlassModel
from .oba import OriginBasedModel
from .sc_tap import SideConstrainedModel
from .so import SystemOptimumModel, marginal_network
from .sue_logit import DialSUEModel
from .sue_probit import SueProbitMsaModel
from .tapas import TapasModel
from .vi_asym import AsymmetricVIModel

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
    "MulticlassModel",
    "OriginBasedModel",
    "SideConstrainedModel",
    "SystemOptimumModel",
    "TapasModel",
    "AsymmetricVIModel",
    "marginal_network",
]

# Append the torch model to the public API only when its optional dependency is
# present, so ``from tabench.models import *`` on a core install never fails.
if _HAS_TORCH:
    __all__.append("ImplicitUENNModel")
