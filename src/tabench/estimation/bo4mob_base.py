"""The BO4Mob T2 OD-estimation contract (adr-041, a THIRD T2 sibling family).

BO4Mob (Ryu, Kwon, Choi, Deshwal, Kang & Osorio 2025, arXiv:2510.18824) poses a
San Jose freeway network as a high-dimensional black-box OD-estimation problem:
choose a continuous OD vector over a fixed set of active ``(fromTaz, toTaz)``
pairs to fit real Caltrans PeMS link counts under a mesoscopic SUMO run. This is
a genuinely different estimand from static T2 (:class:`EstimationTask`) and
within-day dynamic T2 (:class:`DynamicEstimationTask`): there is **no** declared
BPR network, **no** true OD, and **no** ``bfw``-certifiable assignment — truth is
the real sensor panel, scored by re-running the pinned ``eclipse-sumo`` engine
(:mod:`tabench.metrics.estimation_bo4mob`). So, exactly as adr-023 did for the
dynamic track, the family gets its own task type, ABC, and registry — the honest
type gate that keeps the CLI from running a static/dynamic estimator on a BO4Mob
task, or a BO4Mob estimator on a ``(Z, Z)`` one (ADR-002 Decision 1 rationale).

``ODTrace`` / ``ODState`` / ``ODResultBundle`` from :mod:`.base` are reused
verbatim: they copy any ndarray, so a 1-D OD-vector checkpoint records exactly as
a static ``(Z, Z)`` one (the dynamic sibling's own docstring establishes this
precedent for a non-``(Z, Z)`` estimand). This module imports **no** ``sumo`` —
the do-nothing prior baseline registers unconditionally; the engine lives only in
the certifier, behind the adr-027/029 subprocess discipline (adr-041).

**Dual-benchmark honesty (adr-034, carried here).** BO4Mob is the lab's OWN
benchmark; these instances are hosted as scenarios/tasks/certificates, never as
validation of TABench methods, and this family does NOT reproduce BO4Mob's own
published SPSA/BO leaderboard rankings.
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
from ..observe.levels import Dataset
from .base import ODResultBundle, ODTrace

__all__ = [
    "Bo4MobEstimationTask",
    "Bo4MobODEstimator",
    "BO4MOB_ESTIMATOR_REGISTRY",
    "register_bo4mob_estimator",
    "Bo4MobPriorBaseline",
    "_bo4mob_estimation_capabilities",
]


@dataclass(frozen=True)
class Bo4MobEstimationTask:
    """Everything a BO4Mob T2 estimator may see. Contains NO held-out data, by design.

    ``pairs`` is the fixed, ordered active ``(fromTaz, toTaz)`` OD-cell layout
    (from the instance's ``od.xml`` template); the estimand is a 1-D OD vector
    over it. ``prior_vector`` is the stale seed (BO4Mob's ``od_for_single_run``
    example OD). ``dataset`` is the TRAIN-only sensor payload: the anchor
    ``(date, hour)`` PeMS counts the estimator may fit — ``payload['link_ids']``
    and ``payload['counts']`` (P7: **only** the TRAIN anchor; the held-out panel
    lives solely inside the certifier and is folded in through ``heldout_digest``,
    never constructed into this object). ``identifiability`` is a light,
    provenance-only diagnostic (sensor-vs-active-pair coverage — BO4Mob has no
    declared assignment for Hazelton's rank test, so it never gates; adr-041).
    ``engine`` pins the exact ``eclipse-sumo`` version the certificate re-runs.
    """

    name: str
    instance_key: str
    pairs: tuple[tuple[str, str], ...]
    prior_vector: np.ndarray
    dataset: Dataset
    identifiability: Mapping[str, Any]
    engine: Mapping[str, Any]
    certificate: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 0
    heldout_digest: str = ""

    def content_hash(self) -> str:
        """SHA-256 instance pin (domain-prefixed ``tabench-t2-bo4mob-task-v1;``).

        Covers the instance key, the active pair LAYOUT (two tasks whose estimands
        live in different OD cells must never hash equal — the adr-023 lesson
        carried to BO4Mob's string-keyed pairs), the prior-vector bytes, the TRAIN
        link ids + count bytes, the TRAIN metadata dials, the exact engine pin, the
        certificate (engine flags + sim/OD window), the held-out digest (the ONLY
        held-out design that enters task identity, P7), and the seed.
        """
        h = hashlib.sha256()
        h.update(b"tabench-t2-bo4mob-task-v1;")
        h.update(f"instance={self.instance_key};".encode())
        h.update(("pairs=" + "|".join(f"{a},{b}" for a, b in self.pairs) + ";").encode())
        h.update(b"prior;")
        h.update(np.ascontiguousarray(self.prior_vector, dtype=np.float64).tobytes())
        payload = self.dataset.payload
        h.update(("train_links=" + "|".join(payload["link_ids"]) + ";").encode())
        h.update(b"train_counts;")
        h.update(np.ascontiguousarray(payload["counts"], dtype=np.float64).tobytes())
        meta = self.dataset.meta
        for key in sorted(meta):
            h.update(f"meta:{key}={meta[key]!r};".encode())
        for key in sorted(self.engine):
            h.update(f"engine:{key}={self.engine[key]!r};".encode())
        for key in sorted(self.certificate):
            h.update(f"cert:{key}={self.certificate[key]!r};".encode())
        h.update(f"heldout_digest={self.heldout_digest};".encode())
        h.update(f"seed={int(self.seed)};".encode())
        return h.hexdigest()


def _bo4mob_estimation_capabilities(
    deterministic: bool = True, seedable: bool = True, trained_on: tuple[str, ...] = ()
) -> Capabilities:
    """A BO4Mob-T2 ``Capabilities`` with the count-fit OD-estimation signature."""
    return Capabilities(
        paradigm="estimation",
        deterministic=deterministic,
        provides_gap=False,
        seedable=seedable,
        inputs_required=frozenset({"link_counts", "prior_od"}),
        outputs=frozenset({"od_estimate"}),
        trained_on=trained_on,
    )


class Bo4MobODEstimator(ABC):
    """Base class every BO4Mob OD estimator implements.

    Mirrors :class:`~tabench.estimation.base.ODEstimator` piece for piece over a
    :class:`Bo4MobEstimationTask`; the emitted artifact is a 1-D OD vector over
    ``task.pairs`` at every checkpoint, recorded through the shared ``ODTrace``.
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
        task: Bo4MobEstimationTask,
        budget: Budget,
        rng: RngBundle,
        trace: ODTrace,
    ) -> ODResultBundle:
        """Estimate the OD vector over ``task.pairs``, recording it to ``trace``.

        A black-box estimator (a future SPSA over the pipeline) runs the engine as
        its own inner oracle, paid from its own budget; the do-nothing prior
        baseline is a single-shot emit. Record at least one checkpoint (the final
        emitted vector) so certification is always well-defined.
        """


BO4MOB_ESTIMATOR_REGISTRY: dict[str, type[Bo4MobODEstimator]] = {}


def register_bo4mob_estimator(cls: type[Bo4MobODEstimator]) -> type[Bo4MobODEstimator]:
    """Class decorator adding a BO4Mob estimator to its own name registry.

    Deliberately separate from ``ESTIMATOR_REGISTRY`` / ``DYNAMIC_ESTIMATOR_REGISTRY``:
    a static or dynamic estimator would mis-read a BO4Mob task (1-D vector over
    string pairs, no ``(Z, Z)`` matrix) and vice versa, so the registries are the
    type gate (ADR-002 Decision 1 rationale, ADR-023, adr-041).
    """
    if "name" not in cls.__dict__ or cls.name == "unnamed":
        raise TypeError(f"{cls.__qualname__} must declare a class-level `name`")
    if "capabilities" not in cls.__dict__:
        raise TypeError(f"{cls.__qualname__} must declare class-level `capabilities`")
    key = cls.name
    if key in BO4MOB_ESTIMATOR_REGISTRY:
        raise ValueError(f"BO4Mob estimator name {key!r} already registered")
    BO4MOB_ESTIMATOR_REGISTRY[key] = cls
    return cls


class Bo4MobPriorBaseline(Bo4MobODEstimator):
    """Do-nothing anchor: emit the prior OD vector unchanged (BO4Mob Improvement%).

    Every BO4Mob leaderboard needs the baseline every improvement is measured
    against; a prior already fitting the held-out days is unbeatable, one far off
    certifies a terrible (honest) held-out count-fit. Registers **unconditionally**
    and imports **no** ``sumo`` — the engine is the certifier's job, not the
    estimator's (unlike the simulator-in-the-loop spsa-sumo guard; adr-041).
    """

    name = "bo4mob-prior"
    capabilities = _bo4mob_estimation_capabilities(deterministic=True)

    def estimate(
        self,
        task: Bo4MobEstimationTask,
        budget: Budget,
        rng: RngBundle,
        trace: ODTrace,
    ) -> ODResultBundle:
        coords = BudgetCoords(iterations=1, sp_calls=0, wall_ms=0.0)
        trace.record(task.prior_vector, coords)
        return ODResultBundle(
            estimator_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )


register_bo4mob_estimator(Bo4MobPriorBaseline)
