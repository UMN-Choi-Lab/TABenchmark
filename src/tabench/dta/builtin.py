"""Registered Merchant-Nemhauser anchor instances.

Both are hand-solved in ``docs/design/adr-020-merchant-nemhauser.md`` and
machine-verified (aggregate lower-bound arguments + LP + exact duality) in
``tests/test_dta_mn.py``.
"""

from __future__ import annotations

import numpy as np

from .scenario import SODTAScenario

__all__ = ["mn_parallel_scenario", "mn_metering_scenario"]


def mn_parallel_scenario() -> SODTAScenario:
    """Parallel-route capacity metering (anchor A): 6 vehicles at node 0 in
    period 0 choose between a fast capacitated route (link 0->2, ``g = min(x, 2)``,
    1 period free-flow) and a slow uncapacitated one (0->1->2, ``g = x`` each,
    2 periods). SO optimum = 10 vehicle-periods, achieved by any split sending
    2..4 vehicles down the fast route; every optimum exits the fast link at its
    capacity bound in period 1 (``E(1) = 2``)."""
    demand = np.zeros((5, 3))
    demand[0, 0] = 6.0
    return SODTAScenario(
        name="mn-parallel",
        n_nodes=3,
        destination=2,
        link_tail=[0, 0, 1],
        link_head=[2, 1, 2],
        exit_pieces=(((1.0, 0.0), (0.0, 2.0)), ((1.0, 0.0),), ((1.0, 0.0),)),
        demand=demand,
    )


def mn_metering_scenario() -> SODTAScenario:
    """Series bottleneck where holding back is STRICTLY optimal (anchor B):
    4 vehicles at node 0 traverse 0 -A-> 1 -B-> 2 with ``g_A = min(x, 2)``
    (weight 1) and ``g_B = min(x, 1)`` (weight 2 — the downstream street is
    twice as costly per vehicle-period). Relaxed SO optimum = 18, and EVERY
    optimum meters link A at rate 1 while ``g_A(x_A(1)) = 2`` (strict slack in
    the exit bound); the naive M-N equality form ``e = g(x)`` is decision-free
    here and costs 22."""
    demand = np.zeros((7, 3))
    demand[0, 0] = 4.0
    return SODTAScenario(
        name="mn-metering",
        n_nodes=3,
        destination=2,
        link_tail=[0, 1],
        link_head=[1, 2],
        exit_pieces=(((1.0, 0.0), (0.0, 2.0)), ((1.0, 0.0), (0.0, 1.0))),
        demand=demand,
        cost_weights=[1.0, 2.0],
    )
