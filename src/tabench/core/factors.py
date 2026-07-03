"""Typed hyperparameter ("factor") specifications, SimOpt-style (P4).

Every model declares its tunable factors with defaults so experiment
manifests can log the complete configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["FactorSpec", "resolve_factors"]


@dataclass(frozen=True)
class FactorSpec:
    """Specification of one model hyperparameter."""

    default: Any
    kind: str = "float"  # float | int | bool | str
    bounds: tuple[float, float] | None = None
    doc: str = ""


def resolve_factors(
    specs: dict[str, FactorSpec], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Merge user overrides into declared factor defaults, validating names/bounds."""
    unknown = set(overrides) - set(specs)
    if unknown:
        raise ValueError(f"Unknown factors {sorted(unknown)}; declared: {sorted(specs)}")
    values: dict[str, Any] = {}
    for name, spec in specs.items():
        value = overrides.get(name, spec.default)
        if spec.bounds is not None:
            lo, hi = spec.bounds
            if not (lo <= value <= hi):
                raise ValueError(f"Factor {name}={value!r} outside bounds [{lo}, {hi}]")
        values[name] = value
    return values
