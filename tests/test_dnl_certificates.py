"""DNLEvaluator certificate tests for DNL outputs."""

import math

import numpy as np
import pytest

from tabench.core.scenario import Network
from tabench.dnl import DNLOutput, DynamicDemand, DynamicScenario, LinkDynamics, TimeGrid
from tabench.metrics import DNLEvaluator
from tabench.metrics.dnl_gaps import _earliest_times


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


# ---------------------------------------------------------------------------
# Mutation-killing tests (adversarial review, 2026-07-07): the certificate
# battery was green under four surviving mutations because every existing
# fixture violates far above tolerance / is single-origin / is strictly
# increasing. These pin the exact behaviours those mutations flip, so a future
# refactor cannot silently weaken a P1 certificate and stay green.
# ---------------------------------------------------------------------------


def _two_origin_scenario() -> DynamicScenario:
    """Two origin zones (1, 2) feeding one destination zone (3) on direct
    links 1->3, 2->3. Exercises the per-origin coupling loop of C1, which a
    single-origin fixture cannot (global conservation subsumes it there)."""
    net = Network(
        name="cert-two-origin",
        n_nodes=3,
        n_zones=3,
        first_thru_node=1,
        init_node=np.array([1, 2], dtype=np.int64),
        term_node=np.array([3, 3], dtype=np.int64),
        capacity=np.ones(2),
        length=np.zeros(2),
        free_flow_time=np.ones(2),
        b=np.zeros(2),
        power=np.ones(2),
        toll=np.zeros(2),
        link_type=np.ones(2, dtype=np.int64),
    )
    rates = np.zeros((1, 3, 3))
    rates[0, 0, 2] = 0.5  # zone 1 -> zone 3
    rates[0, 1, 2] = 0.5  # zone 2 -> zone 3
    return DynamicScenario(
        name="cert-two-origin",
        network=net,
        dynamics=LinkDynamics(
            length=np.array([1.0, 1.0]),
            free_speed=np.array([1.0, 1.0]),
            wave_speed=np.array([math.inf, math.inf]),
            jam_density=np.array([math.inf, math.inf]),
            capacity=np.array([1.0, 1.0]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 8.0]), rates=rates),
        grid=TimeGrid(dt=0.5, n_steps=20),
    )


def test_tolerance_magnitude_is_pinned() -> None:
    """C7 violation of ~1e-8 veh (well above the default eps_count = 1e-9 * V
    but far below a 1e-7 default): censored by the default evaluator. Kills the
    tol 1e-9 -> 1e-7 default mutation, which no coarse fixture constrains."""
    scenario = _point_queue_scenario()
    output = _valid_point_queue_output(scenario)
    bump = 1e-8
    # Bump inflow AND release together so the ONLY violated certificate is C7
    # (demand coupling): C1 origin coupling stays exact, capacity/causality/
    # storage/global conservation are all preserved by the matched shift.
    n_in = output.n_in.copy()
    release = output.origin_release.copy()
    n_in[0, 10:] += bump
    release[0, 10:] += bump
    perturbed = _with_arrays(output, n_in=n_in, origin_release=release)

    strict = DNLEvaluator(scenario).evaluate(perturbed)  # default tol = 1e-9
    assert strict["dnl_feasible"] == 0.0
    assert strict["demand_coupling_residual"] == pytest.approx(bump, rel=1e-3)

    loose = DNLEvaluator(scenario, tol=1e-7).evaluate(perturbed)
    assert loose["dnl_feasible"] == 1.0  # the boundary the default must hold


def test_capacity_certificate_checks_the_inflow_side() -> None:
    """An inflow burst above capacity with a compliant outflow: the reported
    capacity_residual must reflect the INFLOW flux. Kills the mutation that
    replaces flux = max(d_in, d_out) with flux = d_out (the shipped fixture is
    an outflow burst, so d_out alone passed it)."""
    scenario = _point_queue_scenario()
    output = _valid_point_queue_output(scenario)
    n_in = output.n_in.copy()
    # One inflow step of 0.6 veh (> F = q_max * dt = 0.5); outflow stays smooth
    # at 0.25/step, so a d_out-only capacity check would see no violation.
    n_in[0, 1:] += 0.35  # d_in at step 1 becomes 0.25 + 0.35 = 0.60 > 0.50
    metrics = DNLEvaluator(scenario).evaluate(_with_arrays(output, n_in=n_in))

    assert metrics["capacity_residual"] == pytest.approx(0.1, abs=1e-6)


def test_conservation_checks_per_origin_coupling() -> None:
    """A globally balanced but per-origin mismatched output (link inflows 0.5x
    and 1.5x the two origins' equal releases): the origin-coupling loop of C1
    must flag it. Kills deletion of that loop, which the global vehicle
    identity leaves green on any single-origin fixture."""
    scenario = _two_origin_scenario()
    edges = np.arange(scenario.grid.n_steps + 1, dtype=np.float64)
    cum = np.minimum(0.25 * edges, 4.0)  # each origin releases 4 veh
    # Global totals are balanced (0.5 + 1.5 = 2 x cum) but each origin's link
    # inflow no longer matches its own release.
    n_in = np.vstack([0.5 * cum, 1.5 * cum])
    n_out = np.zeros_like(n_in)
    n_out[:, 2:] = n_in[:, :-2]  # free-flow shift by tau_ff = 2 steps
    release = np.vstack([cum, cum, np.zeros_like(cum)])
    out = DNLOutput(
        scenario_hash=scenario.content_hash(),
        grid=scenario.grid,
        n_in=n_in,
        n_out=n_out,
        origin_release=release,
    )
    metrics = DNLEvaluator(scenario).evaluate(out)

    assert metrics["dnl_feasible"] == 0.0
    # Per-step coupling gap |0.125 - 0.25| = 0.125 on each origin; the global
    # identity (mutant's only remaining check) is exactly balanced -> 0.0.
    assert metrics["conservation_residual"] == pytest.approx(0.125)


def test_earliest_times_uses_earliest_edge_on_a_plateau() -> None:
    """The C6 curve inversion must take the EARLIEST time a level is reached on
    a plateau (searchsorted side='left'). Kills the side='left' -> 'right'
    mutation, invisible on strictly increasing fixtures."""
    curve = np.array([0.0, 1.0, 1.0, 1.0, 2.0])  # level 1 held over t in [1, 3]
    times = _earliest_times(curve, np.array([1.0]), dt=1.0)
    assert times[0] == pytest.approx(1.0)  # earliest, not the plateau's end (3.0)
