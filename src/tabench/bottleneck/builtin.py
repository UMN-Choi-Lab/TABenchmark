"""Registered Vickrey bottleneck anchor instances."""

from __future__ import annotations

from .scenario import BottleneckScenario

__all__ = ["vickrey_worked_scenario"]


def vickrey_worked_scenario() -> BottleneckScenario:
    """The worked instance (N=6000, s=3000/h, alpha=1, beta=0.5, gamma=2, t*=9):
    C* = 0.8, window 7.4-9.4, peak at 8.2, max queue 2400, UE total 4800, SO 2400
    (PoA = 2). All hand-derived and machine-verified in ``test_bottleneck.py``."""
    return BottleneckScenario(
        name="vickrey-worked",
        n_travelers=6000.0,
        capacity=3000.0,
        alpha=1.0,
        beta=0.5,
        gamma=2.0,
        t_star=9.0,
    )
