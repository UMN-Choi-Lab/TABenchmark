"""Registered Newell three-detector anchor instances (adr-024).

Five hand-derivable anchors: A1 free-flow clean, A2 asymmetric interior shock
(adr-016 anchor (c) numbers), A3 aligned symmetric bottleneck (the LTM==CTM
cross-model truth anchor, adr-016 anchor (b) move), A4 the seeded noisy
discrimination card, A5 the masked-upstream observability edge.
"""

from __future__ import annotations

import numpy as np

from ..dnl import TimeGrid
from .scenario import ThreeDetectorScenario

__all__ = [
    "newell_free_flow_scenario",
    "newell_spillback_scenario",
    "newell_symmetric_scenario",
    "newell_noisy_scenario",
    "newell_masked_upstream_scenario",
]

_X_INTERIOR = np.array([1.0, 2.0, 3.0])


def newell_free_flow_scenario() -> ThreeDetectorScenario:
    """A1: free-flow, no active bottleneck (meter above inflow). The congested
    branch never binds, so ``N(x,t) = N_up(t - x/vf)`` exactly at every interior
    x — an oracle/validity row (noise='none', never ranked)."""
    return ThreeDetectorScenario(
        name="newell-free-flow",
        vf=1.0, w=1.0, kappa=4.0, length=4.0, meter_cap=2.0,
        inflow_breakpoints=np.array([0.0, 8.0]), inflow_rates=np.array([1.0]),
        grid=TimeGrid(dt=1.0, n_steps=14), x_query=_X_INTERIOR,
        noise="none", family="newell-three-detector",
    )


def newell_spillback_scenario() -> ThreeDetectorScenario:
    """A2: asymmetric spillback with an interior Rankine-Hugoniot shock (adr-016
    anchor (c): vf=2, w=1, kappa=3, cap=2, L=4, inflow 1.0 into a 0.5 meter). The
    shock is born at (x=4, t=2) and travels upstream at speed -0.25, so the
    min-switch passes x=2 at t=10 and reaches x=0 at t=18; post-shock density is
    kappa - q_B/w = 2.5. Clean oracle row."""
    return ThreeDetectorScenario(
        name="newell-spillback",
        vf=2.0, w=1.0, kappa=3.0, length=4.0, capacity=2.0, meter_cap=0.5,
        inflow_breakpoints=np.array([0.0, 24.0]), inflow_rates=np.array([1.0]),
        grid=TimeGrid(dt=1.0, n_steps=24), x_query=_X_INTERIOR,
        noise="none", family="newell-three-detector",
    )


def newell_symmetric_scenario() -> ThreeDetectorScenario:
    """A3: aligned symmetric bottleneck (adr-016 anchor (b): vf=w=1, kappa=4,
    cap=2, L=4, inflow 1.5 into a 0.5 meter). The truth-generating LTM boundary
    curves reproduce CTM byte-for-byte on this grid, doubly validating the truth
    generator (the lp-so-dta corridor==CTMLink move). Clean oracle row."""
    return ThreeDetectorScenario(
        name="newell-symmetric",
        vf=1.0, w=1.0, kappa=4.0, length=4.0, capacity=2.0, meter_cap=0.5,
        inflow_breakpoints=np.array([0.0, 12.0]), inflow_rates=np.array([1.5]),
        grid=TimeGrid(dt=1.0, n_steps=12), x_query=_X_INTERIOR,
        noise="none", family="newell-three-detector",
    )


def newell_noisy_scenario() -> ThreeDetectorScenario:
    """A4: the seeded noisy discrimination card — A3 physics under a Gaussian
    cumulative-reading level. The isotonic estimator strictly beats the naive
    running-max baseline in interior RMSE by a pinned deterministic margin; this
    IS the ranked task (rankable=1)."""
    return ThreeDetectorScenario(
        name="newell-noisy",
        vf=1.0, w=1.0, kappa=4.0, length=4.0, capacity=2.0, meter_cap=0.5,
        inflow_breakpoints=np.array([0.0, 12.0]), inflow_rates=np.array([1.5]),
        grid=TimeGrid(dt=0.25, n_steps=48), x_query=_X_INTERIOR,
        noise="gaussian", read_sigma=1.2, n_days=1, seed=20260714,
        family="newell-three-detector",
    )


def newell_masked_upstream_scenario() -> ThreeDetectorScenario:
    """A5: observability edge — the upstream detector is missing during a
    full-spillback window, so the congested branch N_dn(t-(L-x)/w)+kappa*(L-x)
    alone must pin the interior wherever it is active (a Newell-specific
    identifiability statement). Spillback physics, Gaussian level."""
    return ThreeDetectorScenario(
        name="newell-masked-upstream",
        vf=2.0, w=1.0, kappa=3.0, length=4.0, capacity=2.0, meter_cap=0.5,
        inflow_breakpoints=np.array([0.0, 24.0]), inflow_rates=np.array([1.0]),
        grid=TimeGrid(dt=1.0, n_steps=24), x_query=_X_INTERIOR,
        noise="none", up_windows=((11.0, 17.0),), family="newell-three-detector",
    )
