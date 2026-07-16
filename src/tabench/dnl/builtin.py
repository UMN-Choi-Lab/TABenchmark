"""Built-in analytic DNL scenarios used by the DNL core tests."""

from __future__ import annotations

import math

import numpy as np

from ..core.scenario import Network
from .demand import DynamicDemand
from .fd import LinkDynamics
from .grid import TimeGrid
from .scenario import DynamicScenario

__all__ = [
    "single_link_dynamic_scenario",
    "bottleneck_dynamic_scenario",
    "triangular_bottleneck_dynamic_scenario",
    "greenshields_bottleneck_dynamic_scenario",
]


def _network(name: str, n_nodes: int, init: list[int], term: list[int]) -> Network:
    """Static topology carrier; DNL physics live in ``LinkDynamics``."""
    init_arr = np.asarray(init, dtype=np.int64)
    term_arr = np.asarray(term, dtype=np.int64)
    n_links = init_arr.size
    return Network(
        name=name,
        n_nodes=n_nodes,
        n_zones=2,
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


def _corridor_network(name: str) -> Network:
    """Origin zone 1 -> interior node 3 -> dest zone 2 (two links, a series
    node): the shape ``triangular_bottleneck_dynamic_scenario`` and
    ``greenshields_bottleneck_dynamic_scenario`` share, mirroring the private
    ``_corridor_network`` test helper in ``tests/test_dnl_ctm.py`` /
    ``tests/test_dnl_ltm.py``."""
    return _network(name, n_nodes=3, init=[1, 3], term=[3, 2])


def single_link_dynamic_scenario() -> DynamicScenario:
    """Anchor 1: one point-queue link, free-flow translation, no metering."""
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 0.5
    return DynamicScenario(
        name="dnl-single-link",
        network=_network("dnl-single-link", n_nodes=2, init=[1], term=[2]),
        dynamics=LinkDynamics(
            length=np.array([1.0]),
            free_speed=np.array([1.0]),
            wave_speed=np.array([math.inf]),
            jam_density=np.array([math.inf]),
            capacity=np.array([1.0]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 10.0]), rates=rates),
        grid=TimeGrid(dt=0.5, n_steps=24),
    )


def bottleneck_dynamic_scenario() -> DynamicScenario:
    """Anchor 2: two point-queue links with a capacity bottleneck at the sink link."""
    rates = np.zeros((2, 2, 2))
    rates[0, 0, 1] = 1.5
    rates[1, 0, 1] = 0.5
    return DynamicScenario(
        name="dnl-bottleneck",
        network=_network("dnl-bottleneck", n_nodes=3, init=[1, 3], term=[3, 2]),
        dynamics=LinkDynamics(
            length=np.array([1.0, 1.0]),
            free_speed=np.array([2.0, 1.0]),
            wave_speed=np.array([math.inf, math.inf]),
            jam_density=np.array([math.inf, math.inf]),
            capacity=np.array([4.0, 1.0]),
        ),
        demand=DynamicDemand(
            breakpoints=np.array([0.0, 4.0, 12.0]),
            rates=rates,
        ),
        grid=TimeGrid(dt=0.5, n_steps=32),
    )


def triangular_bottleneck_dynamic_scenario() -> DynamicScenario:
    """Anchor 3: finite-jam triangular-FD corridor with a capacity bottleneck —
    the anchor ``CTMLink``/``LTMLink`` run on (anchors 1-2 above are point-queue,
    ``kappa = inf``, and both link models reject that). Symmetric ``vf = w = 1``,
    ``kappa = 4`` (capacity 2) feeding a 0.5-capacity sink link at arrival rate
    1.5: a Rankine-Hugoniot backward shock builds from ``t = L/vf = 4``, byte-
    identical to ``tests/test_dnl_ctm.py``'s / ``tests/test_dnl_ltm.py``'s
    ``_bottleneck_scenario`` (RH shock speed -0.5, storage ``k_B*L = 14`` at the
    ``n_steps = 12`` horizon) — reused here as the shared, importable instance."""
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 1.5
    return DynamicScenario(
        name="dnl-triangular-bottleneck",
        network=_corridor_network("dnl-triangular-bottleneck"),
        dynamics=LinkDynamics(
            length=np.array([4.0, 1.0]),
            free_speed=np.array([1.0, 1.0]),
            wave_speed=np.array([1.0, 1.0]),
            jam_density=np.array([4.0, 4.0]),
            capacity=np.array([2.0, 0.5]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 12.0]), rates=rates),
        grid=TimeGrid(dt=1.0, n_steps=12),
    )


def greenshields_bottleneck_dynamic_scenario() -> DynamicScenario:
    """Anchor 4: Greenshields-consistent corridor (``wave_speed = free_speed``
    and ``capacity = vf*kappa/4`` on every link — the constraint
    :class:`~tabench.dnl.godunov.GodunovLink` gates on) with a capacity
    bottleneck: the anchor the smooth parabolic FD runs on, distinct in shape
    from ``triangular_bottleneck_dynamic_scenario`` (``kappa = 8`` upstream vs
    ``4``) so the two anchors are not the same instance under a different link
    model. Upstream ``vf = 1``, ``kappa = 8`` -> capacity 2; bottleneck ``vf =
    1``, ``kappa = 2`` -> capacity 0.5; arrival rate 1.5 (uncongested inflow,
    above the bottleneck capacity) builds a queue on the parabolic branch."""
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 1.5
    return DynamicScenario(
        name="dnl-greenshields-bottleneck",
        network=_corridor_network("dnl-greenshields-bottleneck"),
        dynamics=LinkDynamics(
            length=np.array([4.0, 1.0]),
            free_speed=np.array([1.0, 1.0]),
            wave_speed=np.array([1.0, 1.0]),
            jam_density=np.array([8.0, 2.0]),
            capacity=np.array([2.0, 0.5]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 20.0]), rates=rates),
        grid=TimeGrid(dt=1.0, n_steps=20),
    )
