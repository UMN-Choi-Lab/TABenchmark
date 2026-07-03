"""Checkpoint traces and result bundles.

Models emit :class:`FlowState` checkpoints through a :class:`Trace`; the
harness-side evaluator recomputes every scored metric from the emitted link
flows (P1). ``self_report`` entries are provenance only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .budget import BudgetCoords

__all__ = ["FlowState", "Trace", "ResultBundle"]


@dataclass(frozen=True)
class FlowState:
    """One emitted solution state."""

    link_flows: np.ndarray
    coords: BudgetCoords
    self_report: dict[str, float] = field(default_factory=dict)


class Trace:
    """Ordered stream of checkpoints emitted during one solve."""

    def __init__(self) -> None:
        self.checkpoints: list[FlowState] = []

    def record(
        self,
        link_flows: np.ndarray,
        coords: BudgetCoords,
        **self_report: float,
    ) -> None:
        """Record a checkpoint. Link flows are copied defensively."""
        self.checkpoints.append(
            FlowState(
                link_flows=np.array(link_flows, dtype=np.float64, copy=True),
                coords=coords,
                self_report=dict(self_report),
            )
        )

    @property
    def final(self) -> FlowState:
        if not self.checkpoints:
            raise RuntimeError("Trace is empty: the model recorded no checkpoints")
        return self.checkpoints[-1]

    def __len__(self) -> int:
        return len(self.checkpoints)

    def __iter__(self):
        return iter(self.checkpoints)


@dataclass
class ResultBundle:
    """Everything one solve produced, with provenance."""

    model_name: str
    final: FlowState
    trace: Trace
    factors: dict[str, Any] = field(default_factory=dict)
    seed_info: dict[str, Any] = field(default_factory=dict)
