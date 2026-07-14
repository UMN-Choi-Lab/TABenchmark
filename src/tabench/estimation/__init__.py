"""T2 estimation: recover OD demand from link counts (ADR-002).

A demand-free contract (``ODEstimator`` over ``EstimationTask``) parallel to the
T1 model contract, the shipped classical estimators, and the shared
MSA-averaged proportion extraction they score against. The harness certifies
every emitted OD matrix through a pinned reference assignment
(``tabench.metrics.estimation``).
"""

from ._dynamic_map import (
    MAP_RECIPE,
    lagged_assignment_tensor,
    predict_interval_counts,
    stacked_tensor_map,
    tensor_blocks,
)
from ._proportions import active_pairs, od_from_pairs, proportion_matrix
from .base import (
    ESTIMATOR_REGISTRY,
    CallableEstimator,
    EstimationTask,
    ODEstimator,
    ODResultBundle,
    ODState,
    ODTrace,
    PriorBaseline,
    register_estimator,
)
from .cascetta1993 import (
    SequentialDynamicGLSEstimator,
    SimultaneousDynamicGLSEstimator,
    dynamic_gls_sequential,
    dynamic_gls_simultaneous,
)
from .dn_kalman import DavisNihanKalmanEstimator, ar1_tau, dn_gls_solve
from .dynamic_base import (
    DYNAMIC_ESTIMATOR_REGISTRY,
    DynamicEstimationTask,
    DynamicODEstimator,
    DynamicPriorBaseline,
    register_dynamic_estimator,
)
from .entropy import VZWEntropyEstimator, vzw_balance
from .gls import GLSEstimator, gls_solve
from .spiess import SpiessEstimator, spiess_step
from .spsa import SPSAEstimator
from .yang1992 import Yang1992Estimator, yang_solve

__all__ = [
    "ESTIMATOR_REGISTRY",
    "CallableEstimator",
    "EstimationTask",
    "ODEstimator",
    "ODResultBundle",
    "ODState",
    "ODTrace",
    "PriorBaseline",
    "register_estimator",
    "active_pairs",
    "od_from_pairs",
    "proportion_matrix",
    "VZWEntropyEstimator",
    "vzw_balance",
    "GLSEstimator",
    "gls_solve",
    "SpiessEstimator",
    "spiess_step",
    "SPSAEstimator",
    "Yang1992Estimator",
    "yang_solve",
    "DavisNihanKalmanEstimator",
    "ar1_tau",
    "dn_gls_solve",
    "DYNAMIC_ESTIMATOR_REGISTRY",
    "DynamicEstimationTask",
    "DynamicODEstimator",
    "DynamicPriorBaseline",
    "register_dynamic_estimator",
    "SimultaneousDynamicGLSEstimator",
    "SequentialDynamicGLSEstimator",
    "dynamic_gls_simultaneous",
    "dynamic_gls_sequential",
    "lagged_assignment_tensor",
    "predict_interval_counts",
    "stacked_tensor_map",
    "tensor_blocks",
    "MAP_RECIPE",
]
