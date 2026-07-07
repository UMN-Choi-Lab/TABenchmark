"""DNLEvaluator certificate tests for DNL outputs."""

import math

import numpy as np
import pytest

from tabench.core.scenario import Network
from tabench.dnl import DNLOutput, DynamicDemand, DynamicScenario, LinkDynamics, TimeGrid
from tabench.metrics import DNLEvaluator


def _network(name: str) -> Network:
    return Network(
        name=name,
        n_nodes=2,
        n_zones=2,
        first_thru_node=1,
        init_node=np.array([1], dtype=np.int64),
        term_node=np.array([2], dtype=np.int64),
        capacity=np.ones(1),
        length=np.zeros(1),
        free_flow_time=np.ones(1),
        b=np.zeros(1),
        power=np.ones(1),
        toll=np.zeros(1),
        link_type=np.ones(1, dtype=np.int64),
    )


def _point_queue_scenario() -> DynamicScenario:
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 0.5
    return DynamicScenario(
        name="certificate-single-link",
        network=_network("certificate-single-link"),
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


def _finite_storage_scenario() -> DynamicScenario:
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 0.5
    return DynamicScenario(
        name="certificate-finite-storage",
        network=_network("certificate-finite-storage"),
        dynamics=LinkDynamics(
            length=np.array([1.0]),
            free_speed=np.array([1.0]),
            wave_speed=np.array([1.0]),
            jam_density=np.array([1.0]),
            capacity=np.array([0.5]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 4.0]), rates=rates),
        grid=TimeGrid(dt=0.5, n_steps=8),
    )


def _valid_point_queue_output(scenario: DynamicScenario) -> DNLOutput:
    edges = np.arange(scenario.grid.n_steps + 1, dtype=np.float64)
    cumulative = np.minimum(0.25 * edges, 5.0)
    n_in = cumulative[None, :]
    n_out = np.zeros_like(n_in)
    n_out[0, 2:] = cumulative[:-2]
    release = np.vstack([cumulative, np.zeros_like(cumulative)])
    return DNLOutput(
        scenario_hash=scenario.content_hash(),
        grid=scenario.grid,
        n_in=n_in,
        n_out=n_out,
        origin_release=release,
    )


def _with_arrays(
    output: DNLOutput,
    *,
    scenario_hash: str | None = None,
    n_in: np.ndarray | None = None,
    n_out: np.ndarray | None = None,
    origin_release: np.ndarray | None = None,
) -> DNLOutput:
    return DNLOutput(
        scenario_hash=output.scenario_hash if scenario_hash is None else scenario_hash,
        grid=output.grid,
        n_in=output.n_in.copy() if n_in is None else n_in,
        n_out=output.n_out.copy() if n_out is None else n_out,
        origin_release=output.origin_release.copy() if origin_release is None else origin_release,
    )


def _assert_censored(metrics: dict[str, float]) -> None:
    assert metrics["dnl_feasible"] == 0.0
    assert metrics["dnl_cleared"] == 0.0
    assert np.isnan(metrics["tstt"])


def test_valid_hand_built_output_certifies_and_scores() -> None:
    scenario = _point_queue_scenario()
    output = _valid_point_queue_output(scenario)

    metrics = DNLEvaluator(scenario).evaluate(output)

    assert metrics["dnl_feasible"] == 1.0
    assert metrics["dnl_cleared"] == 1.0
    assert metrics["tstt"] == pytest.approx(5.0)
    assert metrics["total_delay"] == pytest.approx(0.0)
    assert metrics["unserved_demand"] == pytest.approx(0.0)
    assert metrics["vehicles_completed"] == pytest.approx(5.0)
    assert metrics["vehicles_in_network"] == pytest.approx(0.0)


def test_wrong_scenario_hash_is_censored_not_raised() -> None:
    scenario = _point_queue_scenario()
    output = _valid_point_queue_output(scenario)
    wrong_hash = _with_arrays(output, scenario_hash="wrong-scenario")

    metrics = DNLEvaluator(scenario).evaluate(wrong_hash)

    _assert_censored(metrics)


def test_wrong_output_shapes_raise_as_wrapper_errors() -> None:
    scenario = _point_queue_scenario()
    output = _valid_point_queue_output(scenario)
    evaluator = DNLEvaluator(scenario)
    edges = scenario.grid.n_steps + 1

    bad_links = DNLOutput(
        scenario_hash=scenario.content_hash(),
        grid=scenario.grid,
        n_in=np.zeros((2, edges)),
        n_out=np.zeros((2, edges)),
        origin_release=output.origin_release,
    )
    with pytest.raises(ValueError, match="n_in/n_out shape"):
        evaluator.evaluate(bad_links)

    bad_release = DNLOutput(
        scenario_hash=scenario.content_hash(),
        grid=scenario.grid,
        n_in=output.n_in,
        n_out=output.n_out,
        origin_release=np.zeros((1, edges)),
    )
    with pytest.raises(ValueError, match="origin_release shape"):
        evaluator.evaluate(bad_release)


def test_nonfinite_output_is_censored() -> None:
    scenario = _point_queue_scenario()
    output = _valid_point_queue_output(scenario)
    n_in = output.n_in.copy()
    n_in[0, 1] = np.nan

    metrics = DNLEvaluator(scenario).evaluate(_with_arrays(output, n_in=n_in))

    _assert_censored(metrics)


def test_nonmonotone_counts_are_censored() -> None:
    scenario = _point_queue_scenario()
    output = _valid_point_queue_output(scenario)
    n_out = output.n_out.copy()
    n_out[0, 5] = n_out[0, 4] - 0.1

    metrics = DNLEvaluator(scenario).evaluate(_with_arrays(output, n_out=n_out))

    _assert_censored(metrics)


def test_capacity_violation_is_censored_with_diagnostic_residual() -> None:
    scenario = _point_queue_scenario()
    output = _valid_point_queue_output(scenario)
    n_in = output.n_in.copy()
    release = output.origin_release.copy()
    n_in[0, 1:] += 0.5
    release[0, 1:] += 0.5

    metrics = DNLEvaluator(scenario).evaluate(
        _with_arrays(output, n_in=n_in, origin_release=release)
    )

    _assert_censored(metrics)
    assert metrics["capacity_residual"] > 0.0


def test_free_flow_causality_violation_is_censored() -> None:
    scenario = _point_queue_scenario()
    output = _valid_point_queue_output(scenario)
    n_out = output.n_out.copy()
    n_out[0, 1:3] = 0.1

    metrics = DNLEvaluator(scenario).evaluate(_with_arrays(output, n_out=n_out))

    _assert_censored(metrics)
    assert metrics["causality_residual"] > 0.0


def test_release_above_demand_is_censored() -> None:
    scenario = _point_queue_scenario()
    output = _valid_point_queue_output(scenario)
    release = output.origin_release.copy()
    release[0, 1] += 0.1

    metrics = DNLEvaluator(scenario).evaluate(_with_arrays(output, origin_release=release))

    _assert_censored(metrics)
    assert metrics["demand_coupling_residual"] > 0.0


def test_finite_storage_bound_violation_is_censored() -> None:
    scenario = _finite_storage_scenario()
    cumulative = 0.25 * np.arange(scenario.grid.n_steps + 1, dtype=np.float64)
    output = DNLOutput(
        scenario_hash=scenario.content_hash(),
        grid=scenario.grid,
        n_in=cumulative[None, :],
        n_out=np.zeros((1, cumulative.size)),
        origin_release=np.vstack([cumulative, np.zeros_like(cumulative)]),
    )

    metrics = DNLEvaluator(scenario).evaluate(output)

    _assert_censored(metrics)
    assert metrics["storage_residual"] > 0.0


def test_evaluator_does_not_mutate_output_arrays() -> None:
    scenario = _point_queue_scenario()
    output = _valid_point_queue_output(scenario)
    before = (output.n_in.copy(), output.n_out.copy(), output.origin_release.copy())

    metrics = DNLEvaluator(scenario).evaluate(output)

    assert metrics["dnl_feasible"] == 1.0
    np.testing.assert_array_equal(output.n_in, before[0])
    np.testing.assert_array_equal(output.n_out, before[1])
    np.testing.assert_array_equal(output.origin_release, before[2])
