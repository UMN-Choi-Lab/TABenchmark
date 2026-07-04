"""Budgets count work in hardware-free coordinates (P6).

Wall-clock is always recorded but never used as the primary ranking axis.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Budget", "BudgetCoords"]


@dataclass(frozen=True)
class BudgetCoords:
    """Where a checkpoint sits in budget space. All coordinates recorded (P6)."""

    iterations: int = 0
    sp_calls: int = 0
    wall_ms: float = 0.0


@dataclass(frozen=True)
class Budget:
    """Resource limits for one solve. ``None`` means unconstrained on that axis.

    ``target_relative_gap`` is an optional convergence early-stop, not a
    resource axis: solvers that monitor their own convergence measure MAY
    stop once it is reached (the certified metrics are still recomputed by
    the harness). For UE solvers the measure is the relative gap; for the
    SUE solver it is the fixed-point residual (ADR-001). Boyce,
    Ralevic-Dekic & Bar-Gera (2004) recommend RG <= 1e-4 for stable link
    flows; a resource axis is still required so that non-converging models
    (and single-shot models, which ignore the target) terminate.
    """

    iterations: int | None = None
    sp_calls: int | None = None
    wall_seconds: float | None = None
    target_relative_gap: float | None = None

    def __post_init__(self) -> None:
        if self.iterations is None and self.sp_calls is None and self.wall_seconds is None:
            raise ValueError(
                "Budget must constrain at least one resource axis "
                "(target_relative_gap alone cannot bound a non-converging model)"
            )

    def exhausted(self, coords: BudgetCoords) -> bool:
        if self.iterations is not None and coords.iterations >= self.iterations:
            return True
        if self.sp_calls is not None and coords.sp_calls >= self.sp_calls:
            return True
        if self.wall_seconds is not None and coords.wall_ms >= 1000.0 * self.wall_seconds:
            return True
        return False

    def target_met(self, relative_gap: float | None) -> bool:
        """True when a self-monitored relative gap satisfies the target."""
        return (
            self.target_relative_gap is not None
            and relative_gap is not None
            and relative_gap <= self.target_relative_gap
        )
