"""Built-in analytic DNL scenarios used by the DNL core tests."""

from __future__ import annotations

import math

import numpy as np

from ..core.scenario import Network
from .demand import DynamicDemand
from .fd import LinkDynamics
from .grid import TimeGrid
from .scenario import DynamicScenario

__all__ = ["single_link_dynamic_scenario", "bottleneck_dynamic_scenario"]


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
