"""Built-in TD-UE / TD-SO anchor instances (Peeta & Mahmassani 1995, adr-031).

Hand-solvable, machine-verified in ``tests/test_tdta.py``. The paper's own 50-node
DYNASMART numerics are engine-bound and irreproducible (adr-031 sourcing note), so
every anchor is derived from scratch on the repo's own DNL — a symmetric two-route
diamond (exact TD-UE), an SO != UE wedge (the paper's headline made executable),
the ADR-021 corridor as a single-path cross-model pin, and a merge for attribution.
"""

from __future__ import annotations

import numpy as np

from ..core.scenario import Network
from ..dnl.demand import DynamicDemand
from ..dnl.fd import LinkDynamics
from ..dnl.grid import TimeGrid
from .scenario import TDPath, TDTAScenario

__all__ = [
    "pm_corridor_scenario",
    "pm_diamond_scenario",
    "pm_wedge_scenario",
    "pm_merge_scenario",
]


def _network(name: str, n_nodes: int, n_zones: int, init: list[int], term: list[int]) -> Network:
    """Static topology carrier; DNL physics live in ``LinkDynamics`` (P2)."""
    init_arr = np.asarray(init, dtype=np.int64)
    term_arr = np.asarray(term, dtype=np.int64)
    n_links = init_arr.size
    return Network(
        name=name,
        n_nodes=n_nodes,
        n_zones=n_zones,
        first_thru_node=1,
        init_node=init_arr,
        term_node=term_arr,
        capacity=np.ones(n_links),
        length=np.zeros(n_links),
        free_flow_time=np.ones(n_links),
        b=np.zeros(n_links),
        power=np.ones(n_links),
        toll=np.zeros(n_links),
        link_type=np.ones(n_links, dtype=np.int64),
    )


def pm_corridor_scenario() -> TDTAScenario:
    """Single-path cross-model pin (anchor C): the ADR-021 ``zil_corridor`` as a
    TDTAScenario. Origin 1 -> link 0 (fast) -> node 3 -> link 1 (Q=1, N=2
    bottleneck) -> destination 2, 6 vehicles in the first interval. With one path
    the split is forced, so ``tdue_gap = 0`` trivially and the harness loading's
    TSTT must equal the LP optimum 33 exactly through the NEW code path."""
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 6.0
    return TDTAScenario(
        name="pm-corridor",
        network=_network("pm-corridor", n_nodes=3, n_zones=2, init=[1, 3], term=[3, 2]),
        dynamics=LinkDynamics(
            length=np.array([1.0, 1.0]),
            free_speed=np.array([1.0, 1.0]),
            wave_speed=np.array([1.0, 1.0]),
            jam_density=np.array([20.0, 2.0]),
            capacity=np.array([10.0, 1.0]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 1.0]), rates=rates),
        grid=TimeGrid(dt=1.0, n_steps=10),
        paths=(TDPath(1, 2, (0, 1)),),
        kernel="ctm",
    )


def pm_diamond_scenario(kernel: str = "ctm") -> TDTAScenario:
    """Symmetric two-route diamond (anchor A): origin 1 -> {route via node 3,
    route via node 4} -> destination 2, the two routes byte-identical. By symmetry
    the exact TD-UE is the 50/50 split every interval (equal experienced times,
    ``tdue_gap = 0``); an all-on-one split queues one bottleneck while the twin
    sits idle, a hand-computable positive gap. Each route is a fast link then a
    Q=1 bottleneck link; 4 vehicles in the first interval.

    ``kernel`` picks the loading operator (``"ctm"`` or ``"ltm"``) — the two agree
    to machine precision on this aligned instance (the ADR-016 ltm==ctm pin)."""
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 4.0
    # links: 0 (1->3) 1 (3->2) route A; 2 (1->4) 3 (4->2) route B
    return TDTAScenario(
        name=f"pm-diamond-{kernel}",
        network=_network(
            "pm-diamond", n_nodes=4, n_zones=2, init=[1, 3, 1, 4], term=[3, 2, 4, 2]
        ),
        dynamics=LinkDynamics(
            length=np.array([1.0, 1.0, 1.0, 1.0]),
            free_speed=np.array([1.0, 1.0, 1.0, 1.0]),
            wave_speed=np.array([1.0, 1.0, 1.0, 1.0]),
            jam_density=np.array([20.0, 20.0, 20.0, 20.0]),
            capacity=np.array([10.0, 1.0, 10.0, 1.0]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 1.0]), rates=rates),
        grid=TimeGrid(dt=1.0, n_steps=16),
        paths=(TDPath(1, 2, (0, 1)), TDPath(1, 2, (2, 3))),
        kernel=kernel,
    )


def pm_wedge_scenario() -> TDTAScenario:
    """SO != UE wedge (anchor B), the paper's headline made executable. Origin 1
    to destination 2 by a FAST but capacitated route (link 0 -> node 3 -> link 1,
    Q=1 bottleneck, free-flow time 2) or a SLOWER uncongested route (link 2 ->
    node 4 -> link 3, free-flow time 3, ample capacity). 6 vehicles depart in the
    first interval. Selfish UE overloads the fast route until its queue delay
    matches the slow route's extra free-flow time; the system optimum diverts
    more traffic to the slow route earlier, cutting total queueing — so SO TSTT is
    STRICTLY below UE TSTT (verified against the lp-so-dta bound)."""
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 6.0
    # links: 0 (1->3) fast feeder, 1 (3->2) Q=1 bottleneck; 2 (1->4) + 3 (4->2)
    # slow route (link 3 length 2 -> free-flow time 3 total), ample capacity.
    return TDTAScenario(
        name="pm-wedge",
        network=_network(
            "pm-wedge", n_nodes=4, n_zones=2, init=[1, 3, 1, 4], term=[3, 2, 4, 2]
        ),
        dynamics=LinkDynamics(
            length=np.array([1.0, 1.0, 1.0, 2.0]),
            free_speed=np.array([1.0, 1.0, 1.0, 1.0]),
            wave_speed=np.array([1.0, 1.0, 1.0, 1.0]),
            jam_density=np.array([20.0, 20.0, 20.0, 20.0]),
            capacity=np.array([10.0, 1.0, 10.0, 10.0]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 1.0]), rates=rates),
        grid=TimeGrid(dt=1.0, n_steps=16),
        paths=(TDPath(1, 2, (0, 1)), TDPath(1, 2, (2, 3))),
        kernel="ctm",
    )


def pm_merge_scenario() -> TDTAScenario:
    """Merge attribution anchor (anchor D): two origins feed a shared bottleneck.
    Origin 1 has two routes into node 5 (a two-route diverge at the origin);
    origin 2 has one approach into node 5; node 5 -> link (Q=1 bottleneck) ->
    destination 3. Per-commodity experienced times stay exactly decidable at the
    merge (each incoming link's outflow is observed separately + FIFO), the point
    of the interior-diverge-free restriction. Not closed-form — pinned against an
    independent per-vehicle event simulation and the C0-C8 loader oracle."""
    # zones 1, 2 (origins), 3 (destination); interior nodes 4, 5, 6
    # links: 0 (1->4) 1 (4->5) origin-1 route A; 2 (1->6) 3 (6->5) origin-1 route B;
    #        4 (2->5) origin-2 approach; 5 (5->3) shared bottleneck
    rates = np.zeros((1, 3, 3))
    rates[0, 0, 2] = 3.0  # origin 1 -> dest 3
    rates[0, 1, 2] = 2.0  # origin 2 -> dest 3
    return TDTAScenario(
        name="pm-merge",
        network=_network(
            "pm-merge",
            n_nodes=6,
            n_zones=3,
            init=[1, 4, 1, 6, 2, 5],
            term=[4, 5, 6, 5, 5, 3],
        ),
        dynamics=LinkDynamics(
            length=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            free_speed=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            wave_speed=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            jam_density=np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0]),
            capacity=np.array([10.0, 10.0, 10.0, 10.0, 10.0, 1.0]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 1.0]), rates=rates),
        grid=TimeGrid(dt=1.0, n_steps=20),
        paths=(
            TDPath(1, 3, (0, 1, 5)),
            TDPath(1, 3, (2, 3, 5)),
            TDPath(2, 3, (4, 5)),
        ),
        kernel="ctm",
    )
