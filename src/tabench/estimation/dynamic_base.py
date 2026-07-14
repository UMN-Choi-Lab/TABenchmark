"""The within-day dynamic estimation contract (ADR-023, parallel to ADR-002).

Cascetta, Inaudi & Marquis (1993) estimate a *sequence* of time-slice OD matrices
``d_h`` (departures in slice ``h = 1..H``) from time-sliced link counts
``c_{t,a}`` (interval ``t = 1..T``), linked by an exogenous lagged assignment map
(``_dynamic_map``). This is a genuinely different estimand from static T2: the
time axis is the *signal* (the ``(H, Z, Z)`` profile), not replication (``gls``
collapses it) nor day-to-day noise around a static OD (``od-kalman``). So the
track gets its own task type, ABC, and registry — the honest gate that keeps the
CLI from running a dynamic estimator on a static task (the sibling-registry
rationale ADR-002 Decision 1 used against a shared ``models/estimators/``).

``ODTrace`` / ``ODState`` / ``ODResultBundle`` from :mod:`.base` are reused
verbatim: they copy any ndarray, so an ``(H, Z, Z)`` profile checkpoint records
exactly as a static ``(Z, Z)`` one, and the dynamic certifier reads it back.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from ..core.budget import Budget, BudgetCoords
from ..core.capabilities import Capabilities
from ..core.factors import FactorSpec, resolve_factors
from ..core.rng import RngBundle
from ..core.scenario import Network
from ..observe.levels import Dataset
from .base import ODResultBundle, ODTrace

__all__ = [
    "DynamicEstimationTask",
    "DynamicODEstimator",
    "DYNAMIC_ESTIMATOR_REGISTRY",
    "register_dynamic_estimator",
    "DynamicPriorBaseline",
    "_dynamic_estimation_capabilities",
]


@dataclass(frozen=True)
class DynamicEstimationTask:
    """Everything a dynamic estimator may see. Contains NO true profile, by design.

    ``prior_profile`` is the stale ``(H, Z, Z)`` seed (one prior OD matrix per
    departure slice). ``dataset`` is the :class:`DynamicLinkCounts` payload:
    ``counts`` ``(n_days, T, S)``, the ``sensor_links``, the sensor-restricted
    exogenous ``lag_tensor`` ``(L+1, S, P)``, and the active ``pairs`` (the profile
    column order). The map is exogenous and hashed; the certifier regenerates the
    *full-network* map from the same recipe and never trusts this payload's tensor.
    """

    name: str
    network: Network
    prior_profile: np.ndarray  # (H, Z, Z)
    dataset: Dataset
    identifiability: Mapping[str, Any]
    scenario_hash: str
    certificate: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 0
    heldout_digest: str = ""

    def content_hash(self) -> str:
        """SHA-256 instance pin (domain-prefixed ``tabench-t2d-task-v2;``).

        Covers the scenario hash, the slicing dials (``H``, ``T``, slice length,
        map recipe id) from ``dataset.meta``, the prior-profile bytes, the active
        ``pairs`` (the estimand's cell layout — two tasks whose estimands live in
        different OD cells must never hash equal, even hand-built ones; adr-023
        review), the sensor links, the observed-count bytes, the exogenous
        lag-tensor bytes, the certificate pin, the held-out digest, and the seed.
        The lag-tensor and per-slice count bytes make two macroreps of a
        ``cv=0`` / ``noise='none'`` task, or two slicings of one scenario,
        distinct instances.
        """
        h = hashlib.sha256()
        h.update(b"tabench-t2d-task-v2;")
        h.update(self.scenario_hash.encode())
        h.update(np.ascontiguousarray(self.prior_profile, dtype=np.float64).tobytes())
        payload = self.dataset.payload
        h.update(np.ascontiguousarray(payload["pairs"], dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(payload["sensor_links"], dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(payload["counts"], dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(payload["lag_tensor"], dtype=np.float64).tobytes())
        dials = self.dataset.meta
        for key in sorted(dials):
            h.update(f"{key}={dials[key]!r};".encode())
        for key in sorted(self.certificate):
            h.update(f"cert:{key}={self.certificate[key]!r};".encode())
        h.update(f"heldout_digest={self.heldout_digest};".encode())
        h.update(f"seed={int(self.seed)};".encode())
        return h.hexdigest()


def _dynamic_estimation_capabilities(
    deterministic: bool = True, seedable: bool = True, trained_on: tuple[str, ...] = ()
) -> Capabilities:
    """A dynamic-T2 ``Capabilities`` with the within-day estimation signature."""
    return Capabilities(
        paradigm="estimation",
        deterministic=deterministic,
        provides_gap=False,
        seedable=seedable,
        inputs_required=frozenset({"dynamic_link_counts", "prior_od_profile"}),
        outputs=frozenset({"od_profile_estimate"}),
        trained_on=trained_on,
    )


class DynamicODEstimator(ABC):
    """Base class every within-day dynamic OD estimator implements.

    Mirrors :class:`~tabench.estimation.base.ODEstimator` piece for piece over a
    :class:`DynamicEstimationTask`; the emitted artifact is a full ``(H, Z, Z)``
    profile at every checkpoint, recorded through the shared ``ODTrace``.
    """

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
        task: DynamicEstimationTask,
        budget: Budget,
        rng: RngBundle,
        trace: ODTrace,
    ) -> ODResultBundle:
        """Estimate the ``(H, Z, Z)`` profile, recording it to ``trace``.

        The exogenous map is given, so there is no inner assignment: both shipped
        estimators are single-shot solves with ``sp_calls = 0``. Emit the full
        profile at every checkpoint so certification is always well-defined.
        """


DYNAMIC_ESTIMATOR_REGISTRY: dict[str, type[DynamicODEstimator]] = {}


def register_dynamic_estimator(cls: type[DynamicODEstimator]) -> type[DynamicODEstimator]:
    """Class decorator adding a dynamic estimator to its own name registry.

    Deliberately separate from ``ESTIMATOR_REGISTRY``: a static estimator would
    mis-read the ``(H, Z, Z)`` task and a dynamic one would mis-read a static
    ``(Z, Z)`` task, so the registries are the type gate (ADR-002 Decision 1
    rationale, ADR-023).
    """
    if "name" not in cls.__dict__ or cls.name == "unnamed":
        raise TypeError(f"{cls.__qualname__} must declare a class-level `name`")
    if "capabilities" not in cls.__dict__:
        raise TypeError(f"{cls.__qualname__} must declare class-level `capabilities`")
    key = cls.name
    if key in DYNAMIC_ESTIMATOR_REGISTRY:
        raise ValueError(f"Dynamic estimator name {key!r} already registered")
    DYNAMIC_ESTIMATOR_REGISTRY[key] = cls
    return cls


class DynamicPriorBaseline(DynamicODEstimator):
    """Do-nothing anchor: emit the prior profile unchanged (BO4Mob Improvement%).

    Every dynamic leaderboard needs the baseline every improvement is measured
    against; a prior profile that is already the truth is unbeatable, one that is
    far off certifies a terrible (honest) per-interval count-fit.
    """

    name = "prior-profile"
    capabilities = _dynamic_estimation_capabilities(deterministic=True)

    def estimate(
        self,
        task: DynamicEstimationTask,
        budget: Budget,
        rng: RngBundle,
        trace: ODTrace,
    ) -> ODResultBundle:
        coords = BudgetCoords(iterations=1, sp_calls=0, wall_ms=0.0)
        trace.record(task.prior_profile, coords)
        return ODResultBundle(
            estimator_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )


register_dynamic_estimator(DynamicPriorBaseline)
