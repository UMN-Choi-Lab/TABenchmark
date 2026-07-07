"""T2 estimation: recover OD demand from link counts (ADR-002).

A demand-free contract (``ODEstimator`` over ``EstimationTask``) parallel to the
T1 model contract, the shipped classical estimators, and the shared
MSA-averaged proportion extraction they score against. The harness certifies
every emitted OD matrix through a pinned reference assignment
(``tabench.metrics.estimation``).
"""

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
]
