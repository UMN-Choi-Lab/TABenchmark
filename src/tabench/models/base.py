"""The model contract: one abstract method plus a capabilities declaration (P4).

A white-box model is one whose internals match the scenario's declared cost
functions (``Network.link_cost``); the harness certifies *any* model's output
through those functions regardless (P1), so white-box status affects what the
model may exploit, not how it is scored.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ..core.budget import Budget
from ..core.capabilities import Capabilities
from ..core.factors import FactorSpec, resolve_factors
from ..core.results import ResultBundle, Trace
from ..core.rng import RngBundle
from ..core.scenario import Scenario

__all__ = ["TrafficAssignmentModel", "MODEL_REGISTRY", "register_model"]


class TrafficAssignmentModel(ABC):
    """Base class every benchmark model or wrapper implements."""

    name: ClassVar[str] = "unnamed"
    capabilities: ClassVar[Capabilities]
    factors: ClassVar[dict[str, FactorSpec]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Give every subclass its own factors dict so mutating one class's
        # declaration can never leak into other models' resolved factors.
        if "factors" not in cls.__dict__:
            cls.factors = dict(cls.factors)

    def __init__(self, **factor_overrides: Any) -> None:
        self.factor_values = resolve_factors(self.factors, factor_overrides)

    @abstractmethod
    def solve(
        self,
        scenario: Scenario,
        budget: Budget,
        rng: RngBundle,
        trace: Trace,
    ) -> ResultBundle:
        """Run the model, emitting checkpoints to ``trace``.

        Implementations must respect ``budget`` and record at least one
        checkpoint. Self-reported metrics go into checkpoint ``self_report``
        entries; they are provenance, never scores.
        """


MODEL_REGISTRY: dict[str, type[TrafficAssignmentModel]] = {}


def register_model(cls: type[TrafficAssignmentModel]) -> type[TrafficAssignmentModel]:
    """Class decorator adding a model to the name registry (BO4Mob pattern).

    Only self-contained models belong here: the registry is what the CLI
    instantiates with no arguments, so a registered class must declare its
    ``name`` and ``capabilities`` at class level. Adapter-style models with
    per-instance capabilities (e.g. ``CallableModel``) are used by passing
    instances directly to ``run_experiment`` and must not be registered.
    """
    if "name" not in cls.__dict__ or cls.name == "unnamed":
        raise TypeError(f"{cls.__qualname__} must declare a class-level `name`")
    if "capabilities" not in cls.__dict__:
        raise TypeError(
            f"{cls.__qualname__} must declare class-level `capabilities`; "
            "adapter-style models with per-instance capabilities should be "
            "passed to run_experiment directly instead of being registered."
        )
    key = cls.name
    if key in MODEL_REGISTRY:
        raise ValueError(f"Model name {key!r} already registered")
    MODEL_REGISTRY[key] = cls
    return cls
