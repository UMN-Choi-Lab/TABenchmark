"""Benchmark models: white-box solvers and black-box adapters."""

from .adapters import CallableModel

# The MATSim EDOC adapter re-exports UNCONDITIONALLY (Java-only engine, no
# optional python import to guard — adr-039); availability is a runtime probe.
from .adapters.matsim_edoc import MatsimAdapter

# The SUMO marouter adapter is an OPTIONAL extra (``pip install tabench[sumo]``);
# it is registered (and importable here) only when the ``eclipse-sumo`` wheel is
# present. Importing ``.adapters`` above already ran its guarded registration, so
# this block only re-exports the class when available (mirrors the torch models).
try:
    from .adapters.sumo_duaiterate import SumoDuaIterateAdapter  # noqa: F401
    from .adapters.sumo_marouter import SumoMarouterModel  # noqa: F401

    _HAS_SUMO = True
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by the sumo-free legs
    if exc.name != "sumo":
        raise
    _HAS_SUMO = False

# The DTALite static-assignment adapter is an OPTIONAL extra
# (``pip install tabench[dtalite]``); registered (and importable here) only when the
# ``DTALite`` wheel is present. Importing ``.adapters`` above already ran its guarded
# registration, so this block only re-exports the class when available (the sumo
# precedent). Swallow ONLY a missing-``DTALite`` failure (exact case: ``DTALite``).
try:
    from .adapters.dtalite_tap import DTALiteTapModel  # noqa: F401

    _HAS_DTALITE = True
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by the dtalite-free legs
    if exc.name != "DTALite":
        raise
    _HAS_DTALITE = False
from .algb import AlgorithmBModel
from .aon import AllOrNothingModel
from .base import MODEL_REGISTRY, TrafficAssignmentModel, register_model
from .br_ue import BoundedlyRationalUEModel
from .dtd_cumlog import CumLogDTDModel
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
    from .het_gnn import HetGNNModel  # noqa: F401  (registered + conditional __all__)
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
    "MatsimAdapter",
    "AllOrNothingModel",
    "MODEL_REGISTRY",
    "TrafficAssignmentModel",
    "register_model",
    "BiconjugateFrankWolfeModel",
    "ConjugateFrankWolfeModel",
    "CostSmoothingSUEModel",
    "CumLogDTDModel",
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
    __all__.append("HetGNNModel")

# Append the SUMO adapter to the public API only when its optional dependency is
# present, so ``from tabench.models import *`` on a core install never fails.
if _HAS_SUMO:
    __all__.append("SumoMarouterModel")
    __all__.append("SumoDuaIterateAdapter")

# Likewise the DTALite adapter, behind the optional ``[dtalite]`` extra.
if _HAS_DTALITE:
    __all__.append("DTALiteTapModel")
