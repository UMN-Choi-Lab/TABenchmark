"""NetworkLoader integration tests for DNL built-in scenarios."""

import math

import numpy as np
import pytest

import tabench
from tabench.core.scenario import Network
from tabench.dnl import (
    DynamicDemand,
    DynamicScenario,
    LinkDynamics,
    NetworkLoader,
    TimeGrid,
    TurningFractions,
    bottleneck_dynamic_scenario,
    single_link_dynamic_scenario,
)
from tabench.dnl._reference import PointQueueLink
from tabench.metrics import DNLEvaluator

GOLDEN_SINGLE_LINK_HASH = "93b258aa6ae6c35264006b3969bb940e90a3ad71158fdd4bcf8e8f6e1ad6a2d7"
GOLDEN_BOTTLENECK_HASH = "ecdea09f1c569e0e775f294b0950ab3dbea4e2982c81d2685ba9a7ade463266e"


def _network(name: str, n_nodes: int, n_zones: int, init: list[int], term: list[int]) -> Network:
    init_node = np.asarray(init, dtype=np.int64)
    term_node = np.asarray(term, dtype=np.int64)
    n_links = init_node.size
    return Network(
        name=name,
        n_nodes=n_nodes,
        n_zones=n_zones,
        first_thru_node=1,
        init_node=init_node,
        term_node=term_node,
        capacity=np.ones(n_links),
        length=np.zeros(n_links),
        free_flow_time=np.ones(n_links),
        b=np.zeros(n_links),
        power=np.ones(n_links),
        toll=np.zeros(n_links),
        link_type=np.ones(n_links, dtype=np.int64),
    )


def _assert_zero_residuals(metrics: dict[str, float]) -> None:
    for key in (
        "conservation_residual",
        "capacity_residual",
        "storage_residual",
        "causality_residual",
        "fifo_residual",
        "demand_coupling_residual",
        "kw_backward_residual",
        "kw_backward_residual_rel",
    ):
        assert metrics[key] == pytest.approx(0.0)


def test_public_dnl_exports_are_available() -> None:
    assert tabench.NetworkLoader is NetworkLoader
    assert tabench.DNLEvaluator is DNLEvaluator
    assert callable(tabench.single_link_dynamic_scenario)
    assert callable(tabench.bottleneck_dynamic_scenario)


def test_builtin_scenario_hashes_are_pinned() -> None:
    assert single_link_dynamic_scenario().content_hash() == GOLDEN_SINGLE_LINK_HASH
    assert bottleneck_dynamic_scenario().content_hash() == GOLDEN_BOTTLENECK_HASH


def test_single_link_loader_matches_free_flow_translation_and_certifies() -> None:
    scenario = single_link_dynamic_scenario()
    output = NetworkLoader(scenario, PointQueueLink).run()

    edges = np.arange(scenario.grid.n_steps + 1, dtype=np.float64)
    expected_in = np.minimum(0.25 * edges, 5.0)[None, :]
    expected_out = np.zeros_like(expected_in)
    expected_out[0, 2:] = expected_in[0, :-2]
    expected_release = np.vstack([expected_in[0], np.zeros(expected_in.shape[1])])

    np.testing.assert_allclose(output.n_in, expected_in)
    np.testing.assert_allclose(output.n_out, expected_out)
    np.testing.assert_allclose(output.origin_release, expected_release)

    metrics = DNLEvaluator(scenario).evaluate(output)
    assert metrics["dnl_feasible"] == 1.0
    assert metrics["dnl_cleared"] == 1.0
    assert metrics["tstt"] == pytest.approx(5.0)
    assert metrics["total_delay"] == pytest.approx(0.0)
    assert metrics["unserved_demand"] == pytest.approx(0.0)
    assert metrics["vehicles_completed"] == pytest.approx(5.0)
    _assert_zero_residuals(metrics)


def test_bottleneck_loader_matches_capacity_limited_corridor_and_certifies() -> None:
    scenario = bottleneck_dynamic_scenario()
    output = NetworkLoader(scenario, PointQueueLink).run()

    demand = np.r_[
        0.75 * np.arange(9, dtype=np.float64),
        6.0 + 0.25 * np.arange(1, 17, dtype=np.float64),
        np.full(8, 10.0),
    ]
    transfer = np.r_[
        np.zeros(2),
        np.linspace(0.5, 8.0, 16),
        np.linspace(8.25, 10.0, 8),
        np.full(7, 10.0),
    ]
    sink_out = np.r_[
        np.zeros(4),
        np.linspace(0.5, 8.0, 16),
        np.linspace(8.25, 10.0, 8),
        np.full(5, 10.0),
    ]
    expected_n_in = np.vstack([demand, transfer])
    expected_n_out = np.vstack([transfer, sink_out])
    expected_release = np.vstack([demand, np.zeros_like(demand)])

    assert demand.shape == (scenario.grid.n_steps + 1,)
    np.testing.assert_allclose(output.n_in, expected_n_in)
    np.testing.assert_allclose(output.n_out, expected_n_out)
    np.testing.assert_allclose(output.origin_release, expected_release)

    metrics = DNLEvaluator(scenario).evaluate(output)
    assert metrics["dnl_feasible"] == 1.0
    assert metrics["dnl_cleared"] == 1.0
    assert metrics["tstt"] == pytest.approx(23.0)
    assert metrics["total_delay"] == pytest.approx(8.0)
    assert metrics["unserved_demand"] == pytest.approx(0.0)
    assert metrics["vehicles_completed"] == pytest.approx(10.0)
    _assert_zero_residuals(metrics)


def test_network_loader_is_deterministic() -> None:
    scenario = bottleneck_dynamic_scenario()
    first = NetworkLoader(scenario, PointQueueLink).run()
    second = NetworkLoader(scenario, PointQueueLink).run()

    assert first.scenario_hash == second.scenario_hash
    np.testing.assert_array_equal(first.n_in, second.n_in)
    np.testing.assert_array_equal(first.n_out, second.n_out)
    np.testing.assert_array_equal(first.origin_release, second.origin_release)


def test_network_loader_refuses_unsupplied_diverge_node_model() -> None:
    rates = np.zeros((1, 3, 3))
    rates[0, 0, 1] = 0.6
    rates[0, 0, 2] = 0.4
    scenario = DynamicScenario(
        name="loader-diverge",
        network=_network("loader-diverge", n_nodes=4, n_zones=3, init=[1, 4, 4], term=[4, 2, 3]),
        dynamics=LinkDynamics(
            length=np.ones(3),
            free_speed=np.ones(3),
            wave_speed=np.full(3, math.inf),
            jam_density=np.full(3, math.inf),
            capacity=np.ones(3),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 1.0]), rates=rates),
        grid=TimeGrid(dt=0.5, n_steps=4),
        turns=TurningFractions(frac=((4, np.array([[0.6, 0.4]])),)),
    )

    with pytest.raises(ValueError, match="explicit NodeModel"):
        NetworkLoader(scenario, PointQueueLink)
