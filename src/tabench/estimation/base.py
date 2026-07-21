"""The estimation contract: one abstract method over a demand-free task (P4, P7).

T2 estimators recover an OD matrix from a prior and observed link counts. The
contract mirrors :class:`~tabench.models.base.TrafficAssignmentModel` piece for
piece (same ``Capabilities``, ``FactorSpec`` resolution, registry pattern, and a
``CallableEstimator`` adapter), but over an :class:`EstimationTask` that, *by
construction*, contains no true demand — fairness is structural, never a
convention the estimator is asked to honor (ADR-002, Decision 1).

``ODState``/``ODTrace``/``ODResultBundle`` are the OD-matrix analogues of
``FlowState``/``Trace``/``ResultBundle``; the harness certifies emitted OD
matrices through a pinned reference assignment (``metrics.estimation``), so
``self_report`` entries here are provenance only, exactly as for T1.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from ..core.budget import Budget
from ..core.capabilities import Capabilities
from ..core.factors import FactorSpec, resolve_factors
from ..core.rng import RngBundle
from ..core.scenario import Demand, Network
from ..observe.levels import Dataset

__all__ = [
    "EstimationTask",
    "ODState",
    "ODTrace",
    "ODResultBundle",
    "ODEstimator",
    "ESTIMATOR_REGISTRY",
    "register_estimator",
    "CallableEstimator",
    "PriorBaseline",
]


@dataclass(frozen=True)
class EstimationTask:
    """Everything a T2 estimator may see. Contains NO true demand, by design.

    ``network`` carries its declared cost functions, so white-box estimators
    may run their own inner assignments (their inner solver and gap are
    declared factors, paid from their own budget); ``prior`` is the stale
    seed/target matrix; ``dataset`` is the ``LinkCounts`` payload; and
    ``identifiability`` is the public per-task report (Decision 4). The true
    ``scenario.demand`` is simply absent from this object — there is nothing to
    peek at.
    """

    name: str
    network: Network
    prior: Demand
    dataset: Dataset
    identifiability: Mapping[str, Any]
    scenario_hash: str
    certificate: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 0
    heldout_digest: str = ""

    def __post_init__(self) -> None:
        # A 0-sensor task carries no observational signal: every count-fitting
        # estimator's obs_count_rmse is an undefined mean-of-empty (NaN) and the
        # certificate cannot rank it, so the instance is meaningless for every
        # estimator. No default or random config builds one -- the runner's random
        # sensor draw floors at one sensor (runner._draw_sensors) -- but the public
        # runner DOES accept an explicit empty sensor list
        # (run_estimation_experiment(..., estimation={'sensors': {'kind': 'explicit',
        # 'links': []}})), which used to emit NaN-censored rows; reject it here so it
        # fails fast with a clear error instead.
        sensors = self.dataset.payload.get("sensor_links")
        if sensors is not None and len(sensors) == 0:
            raise ValueError(
                "EstimationTask requires at least one sensor (link count); a "
                "0-sensor task has no observed counts to estimate from."
            )

    def content_hash(self) -> str:
        """SHA-256 instance pin over scenario hash + prior bytes + sensor links +
        observed counts + dataset dials + certificate pin + held-out digest + seed.

        The sensor-link and count bytes make this a true *instance* pin: two
        explicit sensor placements of equal coverage, or two macroreps of a
        ``cv=0`` task, no longer collide. ``heldout_digest`` (a SHA-256 the runner
        computes over the sorted held-out links + held-out n_periods) folds the
        held-out design in *without* leaking held-out sensor identities to the
        estimator (P7).
        """
        h = hashlib.sha256()
        h.update(self.scenario_hash.encode())
        h.update(np.ascontiguousarray(self.prior.matrix, dtype=np.float64).tobytes())
        h.update(
            np.ascontiguousarray(
                self.dataset.payload["sensor_links"], dtype=np.int64
            ).tobytes()
        )
        h.update(
            np.ascontiguousarray(self.dataset.payload["counts"], dtype=np.float64).tobytes()
        )
        dials = self.dataset.meta
        for key in sorted(dials):
            h.update(f"{key}={dials[key]!r};".encode())
        for key in sorted(self.certificate):
            h.update(f"cert:{key}={self.certificate[key]!r};".encode())
        h.update(f"heldout_digest={self.heldout_digest};".encode())
        h.update(f"seed={int(self.seed)};".encode())
        return h.hexdigest()


@dataclass(frozen=True)
class ODState:
    """One emitted OD estimate, mirroring ``FlowState``."""

    od_matrix: np.ndarray
    coords: Any  # BudgetCoords
    self_report: dict[str, float] = field(default_factory=dict)


class ODTrace:
    """Ordered stream of OD checkpoints emitted during one ``estimate`` call."""

    def __init__(self) -> None:
        self.checkpoints: list[ODState] = []

    def record(self, od_matrix: np.ndarray, coords: Any, **self_report: float) -> None:
        """Record a checkpoint. The OD matrix is copied defensively."""
        self.checkpoints.append(
            ODState(
                od_matrix=np.array(od_matrix, dtype=np.float64, copy=True),
                coords=coords,
                self_report=dict(self_report),
            )
        )

    @property
    def final(self) -> ODState:
        if not self.checkpoints:
            raise RuntimeError("ODTrace is empty: the estimator recorded no checkpoints")
        return self.checkpoints[-1]

    def __len__(self) -> int:
        return len(self.checkpoints)

    def __iter__(self):
        return iter(self.checkpoints)


@dataclass
class ODResultBundle:
    """Everything one ``estimate`` call produced, with provenance."""

    estimator_name: str
    final: ODState
    trace: ODTrace
    factors: dict[str, Any] = field(default_factory=dict)
    seed_info: dict[str, Any] = field(default_factory=dict)


class ODEstimator(ABC):
    """Base class every T2 estimator or wrapper implements."""

    name: ClassVar[str] = "unnamed"
    capabilities: ClassVar[Capabilities]
    factors: ClassVar[dict[str, FactorSpec]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "factors" not in cls.__dict__:
            cls.factors = dict(cls.factors)

    def __init__(self, **factor_overrides: Any) -> None:
        self.factor_values = resolve_factors(self.factors, factor_overrides)

    @abstractmethod
    def estimate(
        self,
        task: EstimationTask,
        budget: Budget,
        rng: RngBundle,
        trace: ODTrace,
    ) -> ODResultBundle:
        """Estimate an OD matrix, emitting checkpoints to ``trace``.

        Implementations must respect ``budget`` (expressed in
        assignment-equivalent ``sp_calls``) and record at least one checkpoint.
        Certification runs a full pinned assignment per checkpoint, so emit
        O(10-20) checkpoints -- sparsely spaced (e.g. every ``max(1, iters //
        15)`` iterations) plus always the final iterate -- not one per iteration
        (ADR-002 Decision 2). Self-reported metrics go into checkpoint
        ``self_report`` entries; they are provenance, never scores.
        """


ESTIMATOR_REGISTRY: dict[str, type[ODEstimator]] = {}


def register_estimator(cls: type[ODEstimator]) -> type[ODEstimator]:
    """Class decorator adding an estimator to the name registry.

    Mirrors ``register_model``: only self-contained estimators belong here (the
    CLI instantiates them with no arguments), so a registered class must declare
    class-level ``name`` and ``capabilities``. Adapter-style estimators with
    per-instance capabilities (``CallableEstimator``) are passed to the runner
    directly and must not be registered.
    """
    if "name" not in cls.__dict__ or cls.name == "unnamed":
        raise TypeError(f"{cls.__qualname__} must declare a class-level `name`")
    if "capabilities" not in cls.__dict__:
        raise TypeError(
            f"{cls.__qualname__} must declare class-level `capabilities`; "
            "adapter-style estimators with per-instance capabilities should be "
            "passed to the runner directly instead of being registered."
        )
    key = cls.name
    if key in ESTIMATOR_REGISTRY:
        raise ValueError(f"Estimator name {key!r} already registered")
    ESTIMATOR_REGISTRY[key] = cls
    return cls


def _estimation_capabilities(
    paradigm: str = "estimation",
    deterministic: bool = True,
    seedable: bool = True,
    trained_on: tuple[str, ...] = (),
) -> Capabilities:
    """A T2 ``Capabilities`` with the estimation input/output signature."""
    return Capabilities(
        paradigm=paradigm,
        deterministic=deterministic,
        provides_gap=False,
        seedable=seedable,
        inputs_required=frozenset({"link_counts", "prior_od"}),
        outputs=frozenset({"od_estimate"}),
        trained_on=trained_on,
    )


ODFn = Callable[[EstimationTask, np.random.Generator], np.ndarray]


class CallableEstimator(ODEstimator):
    """Adapter turning ``fn(task, rng) -> od_matrix`` into a benchmark estimator.

    ``capabilities`` are instance-level (they describe the wrapped artifact):
    pass ``trained_on`` lineage for learned inverse models so the fairness gate
    can act on it, exactly as ``CallableModel`` does for T1.
    """

    name = "callable_estimator"

    def __init__(
        self,
        fn: ODFn,
        name: str = "callable_estimator",
        paradigm: str = "learned",
        deterministic: bool = False,
        seedable: bool = True,
        trained_on: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self._fn = fn
        self.name = name
        self.capabilities = _estimation_capabilities(
            paradigm=paradigm,
            deterministic=deterministic,
            seedable=seedable,
            trained_on=trained_on,
        )

    def estimate(
        self, task: EstimationTask, budget: Budget, rng: RngBundle, trace: ODTrace
    ) -> ODResultBundle:
        from ..core.budget import BudgetCoords

        generator = rng.generator(source=0)
        od = np.asarray(self._fn(task, generator), dtype=np.float64)
        coords = BudgetCoords(iterations=1, sp_calls=0, wall_ms=0.0)
        trace.record(od, coords)
        return ODResultBundle(
            estimator_name=self.name,
            final=trace.final,
            trace=trace,
            factors={},
            seed_info=rng.describe(),
        )


class PriorBaseline(ODEstimator):
    """Do-nothing anchor: emit the prior unchanged (BO4Mob Improvement%).

    Every T2 leaderboard needs the baseline every improvement is measured
    against; a prior that is already the truth is unbeatable, one that is far
    off certifies a terrible (honest) count-fit.
    """

    name = "prior"
    capabilities = _estimation_capabilities(deterministic=True)

    def estimate(
        self, task: EstimationTask, budget: Budget, rng: RngBundle, trace: ODTrace
    ) -> ODResultBundle:
        from ..core.budget import BudgetCoords

        coords = BudgetCoords(iterations=1, sp_calls=0, wall_ms=0.0)
        trace.record(task.prior.matrix, coords)
        return ODResultBundle(
            estimator_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )


register_estimator(PriorBaseline)
