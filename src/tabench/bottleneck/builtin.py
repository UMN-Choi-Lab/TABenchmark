"""Registered Vickrey bottleneck anchor instances."""

from __future__ import annotations

from .due import DUEScenario
from .scenario import BottleneckScenario

__all__ = ["vickrey_worked_scenario", "friesz_two_route_scenario"]


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


def friesz_two_route_scenario() -> DUEScenario:
    """The SRDC-DUE worked instance (adr-022): N=6000, routes (f=0.2, s=3000)
    and (f=0.7, s=1500), alpha=1, beta=0.5, gamma=2, t*=9. Common cost C=0.9,
    split (5250, 750), queue costs (0.7, 0.2), windows [7.4, 9.15]/[7.9, 8.4],
    total cost 5400; both routes used iff N > alpha*s1*(f2-f1)/delta = 3750."""
    return DUEScenario(
        name="friesz-two-route",
        n_travelers=6000.0,
        alpha=1.0,
        beta=0.5,
        gamma=2.0,
        t_star=9.0,
        route_free_flow=[0.2, 0.7],
        route_capacity=[3000.0, 1500.0],
    )
