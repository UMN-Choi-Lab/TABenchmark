"""Registered analytical-DTA anchor instances.

The Merchant-Nemhauser pair is hand-solved in
``docs/design/adr-020-merchant-nemhauser.md`` and machine-verified in
``tests/test_dta_mn.py``; the Ziliaskopoulos cell anchors are hand-solved in
``docs/design/adr-021-lp-so-dta.md`` and machine-verified in
``tests/test_dta_zil.py``.
"""

from __future__ import annotations

import numpy as np

from .cells import CellSODTAScenario
from .scenario import SODTAScenario

__all__ = [
    "mn_parallel_scenario",
    "mn_metering_scenario",
    "zil_diverge_spillback_scenario",
    "zil_corridor_scenario",
]


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


def zil_diverge_spillback_scenario() -> CellSODTAScenario:
    """The Ziliaskopoulos LP anchor (adr-021): 6 vehicles queued at source R
    face a diverge at A between a short route through the tiny bottleneck cell
    B (``Q=1``, ``N=1`` — one vehicle of storage) and a longer route C -> D
    (``Q=2`` each). SO optimum = 26 veh-intervals via the spillback pair lemma
    ``y_BS(s) + y_BS(s+1) <= 1``; cell B is jam-full at ``t=2`` in EVERY
    optimum (the storage row is tight with dual price -1), and relaxing
    ``N_B`` to 2 drops the optimum to 25 — a finite-storage effect the
    Merchant-Nemhauser exit-function model cannot represent."""
    demand = np.zeros((8, 6))
    x0 = np.zeros(6)
    x0[0] = 6.0
    inf = np.inf
    return CellSODTAScenario(
        name="zil-diverge-spillback",
        n_cells=6,  # R=0 source, A=1 diverge, B=2 bottleneck, C=3, D=4, S=5 sink
        sink=5,
        conn_tail=[0, 1, 1, 3, 2, 4],
        conn_head=[1, 2, 3, 4, 5, 5],
        capacity=[inf, 10.0, 1.0, 2.0, 2.0, inf],
        storage=[inf, 20.0, 1.0, 10.0, 10.0, inf],
        delta=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        demand=demand,
        initial_occupancy=x0,
    )


def zil_corridor_scenario() -> CellSODTAScenario:
    """A control-free corridor R -> A -> B(Q=1, N=2) -> S with 6 vehicles: the
    LP optimum (33 veh-intervals) equals the strict CTM loading exactly, which
    the tests cross-check against the repo's own ``CTMLink``/``NetworkLoader``
    (the B cell is the triangular FD ``vf=w=1, kappa=2, capacity=1``)."""
    demand = np.zeros((10, 4))
    x0 = np.zeros(4)
    x0[0] = 6.0
    inf = np.inf
    return CellSODTAScenario(
        name="zil-corridor",
        n_cells=4,  # R=0 source, A=1, B=2 bottleneck, S=3 sink
        sink=3,
        conn_tail=[0, 1, 2],
        conn_head=[1, 2, 3],
        capacity=[inf, 10.0, 1.0, inf],
        storage=[inf, 20.0, 2.0, inf],
        delta=[1.0, 1.0, 1.0, 1.0],
        demand=demand,
        initial_occupancy=x0,
    )
