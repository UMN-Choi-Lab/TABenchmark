"""DNLEvaluator certificate tests for DNL outputs."""

import math

import numpy as np
import pytest

from tabench.core.scenario import Network
from tabench.dnl import (
    DNLOutput,
    DynamicDemand,
    DynamicScenario,
    LinkDynamics,
    TimeGrid,
    TurningFractions,
)
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


# ---------------------------------------------------------------------------
# C6 off CFL = 1 (the joint hardening, gated DNL review 2026-07-09): the
# certificate is now C4's envelope evaluated at level-inversion times over the
# UNION of both curves' edge levels, so it neither false-censors a correct
# unaligned emission nor false-accepts a sub-step violation. tau_ff/dt = 2.5
# (unaligned) throughout.
# ---------------------------------------------------------------------------


def _unaligned_scenario() -> DynamicScenario:
    """Single point-queue link, tau_ff = L/vf = 1.0 but dt = 0.4 (tau_ff/dt =
    2.5, CFL < 1 yet wave-resolved: dt <= L/vf). Off the cell-aligned point."""
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 0.5
    return DynamicScenario(
        name="cert-unaligned",
        network=_network("cert-unaligned"),
        dynamics=LinkDynamics(
            length=np.array([1.0]),
            free_speed=np.array([1.0]),
            wave_speed=np.array([math.inf]),
            jam_density=np.array([math.inf]),
            capacity=np.array([1.0]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 8.0]), rates=rates),
        grid=TimeGrid(dt=0.4, n_steps=25),
    )


def _unaligned_output(scenario: DynamicScenario, lag: float) -> DNLOutput:
    """Exact free-flow translation ``n_out(t) = n_in(t - lag)`` sampled at grid
    edges. ``lag = tau_ff = 1.0`` is a correct emission; ``lag < 1.0`` exits
    faster than free flow (a genuine C6/C4 violation)."""
    edges = scenario.grid.edges
    nin = 0.5 * np.minimum(edges, 8.0)
    nout = 0.5 * np.minimum(np.maximum(edges - lag, 0.0), 8.0)
    release = np.vstack([nin, np.zeros_like(nin)])
    return DNLOutput(
        scenario_hash=scenario.content_hash(),
        grid=scenario.grid,
        n_in=nin[None, :],
        n_out=nout[None, :],
        origin_release=release,
    )


def test_c6_unaligned_correct_emission_not_false_censored() -> None:
    """A correct free-flow emission on an unaligned grid certifies: the joint
    fix relaxes the entry side by C4's own one-step grid step, so the O(dt)
    time-quantization the old level-difference form charged no longer false-
    censors. The travel time is exactly tau_ff = 1.0 everywhere."""
    scenario = _unaligned_scenario()
    metrics = DNLEvaluator(scenario).evaluate(_unaligned_output(scenario, lag=1.0))
    assert metrics["dnl_feasible"] == 1.0
    assert metrics["fifo_residual"] <= 1e-9  # count residual, within tolerance


def test_c6_unaligned_free_flow_violation_is_censored_with_count_residual() -> None:
    """An emission that exits after only 0.4 tu (< tau_ff = 1.0) on the unaligned
    grid is censored, and fifo_residual is a positive COUNT (matching C4/C5),
    not a time deficit."""
    scenario = _unaligned_scenario()
    metrics = DNLEvaluator(scenario).evaluate(_unaligned_output(scenario, lag=0.4))
    assert metrics["dnl_feasible"] == 0.0
    assert metrics["fifo_residual"] > 0.1  # vehicles that beat free flow (a count)


def test_c6_samples_exit_curve_plateau_levels() -> None:
    """C6 draws levels from the UNION of both curves, so an exit-curve plateau
    level (1.5) that an n_in one-step jump skips over is still sampled and its
    real free-flow violation censored (level 1.5 exits at t = 0.8 having entered
    at t ~ 1.0, travel ~ 0). NB C4 independently catches this same fixture at the
    grid edge where n_out jumps to 1.5 — union sampling is self-containment /
    belt-and-suspenders, NOT a C4 blind-spot (the adversarial DNL review's fuzz
    found no violation C6 catches that C4 does not; off-CFL completeness is not
    promised by either, matching C4/C5 scope)."""
    scenario = _unaligned_scenario()  # dt = 0.4, tau_ff = 1.0
    K = scenario.grid.n_steps
    # n_in jumps 1.0 -> 2.0 across step [0.8, 1.2], so 1.5 is crossed mid-step at
    # t = 1.0 and is NOT an n_in edge value; free flow => level 1.5 must exit
    # >= 1.0 + 1.0 = 2.0.
    nin = np.zeros(K + 1)
    nin[2] = 1.0  # t = 0.8
    nin[3:] = 2.0  # t >= 1.2 (jump over 1.5)
    # n_out reaches the 1.5 plateau at t = 0.8 (far too early) then 2.0 later.
    nout = np.zeros(K + 1)
    nout[2:5] = 1.5  # t in [0.8, 1.6]: exit-curve edge value 1.5 (in the union)
    nout[5:] = 2.0
    nout = np.minimum.accumulate(nout[::-1])[::-1]  # ensure monotone non-decreasing
    nout = np.maximum.accumulate(nout)
    release = np.vstack([nin, np.zeros_like(nin)])
    out = DNLOutput(
        scenario_hash=scenario.content_hash(),
        grid=scenario.grid,
        n_in=nin[None, :],
        n_out=nout[None, :],
        origin_release=release,
    )
    metrics = DNLEvaluator(scenario).evaluate(out)
    assert metrics["dnl_feasible"] == 0.0
    assert metrics["fifo_residual"] > 0.0


# ---------------------------------------------------------------------------
# C8 turning-fraction fidelity (gated DNL review 2026-07-09): scenario.turns is
# now read and a mandated split is censored at EVERY diverge -- d_in[out_j] ==
# sum_i frac[i,j]*d_out[in_i] is decidable from aggregate counts at multi-in nodes
# too (each incoming link's outflow is observed separately). Fully sufficient at a
# 1-in diverge; necessary-not-sufficient at multi-in (cross-row-cancelling splits
# are unobservable from per-link aggregate counts).
# ---------------------------------------------------------------------------


def _diverge_scenario(split: tuple[float, float] = (0.6, 0.4)) -> DynamicScenario:
    """Origin zone 1 -> interior diverge node 4 -> dest zones 2, 3. Node 4 has
    one incoming link (0: 1->4) and two outgoing (1: 4->2, 2: 4->3), the 1-in
    diverge C8 gates. Demand attracts both dests so the scenario validates."""
    net = Network(
        name="cert-diverge",
        n_nodes=4,
        n_zones=3,
        first_thru_node=4,
        init_node=np.array([1, 4, 4], dtype=np.int64),
        term_node=np.array([4, 2, 3], dtype=np.int64),
        capacity=np.ones(3),
        length=np.zeros(3),
        free_flow_time=np.ones(3),
        b=np.zeros(3),
        power=np.ones(3),
        toll=np.zeros(3),
        link_type=np.ones(3, dtype=np.int64),
    )
    rates = np.zeros((1, 3, 3))
    rates[0, 0, 1] = 0.6  # zone 1 -> zone 2
    rates[0, 0, 2] = 0.4  # zone 1 -> zone 3
    turns = TurningFractions(frac=((4, np.array([list(split)])),))
    return DynamicScenario(
        name="cert-diverge",
        network=net,
        dynamics=LinkDynamics(
            length=np.ones(3),
            free_speed=np.ones(3),
            wave_speed=np.array([math.inf, math.inf, math.inf]),
            jam_density=np.array([math.inf, math.inf, math.inf]),
            capacity=np.ones(3),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 8.0]), rates=rates),
        grid=TimeGrid(dt=0.5, n_steps=24),
        turns=turns,
    )


def _diverge_output(scenario: DynamicScenario, emit: tuple[float, float]) -> DNLOutput:
    """Free-flow (2-step lag) emission that splits the diverge outflow by
    ``emit``; conservation holds for any split summing to 1, so only C8 can
    distinguish ``emit`` from the scenario's mandated turn fractions."""
    edges = scenario.grid.edges
    d1 = np.minimum(edges, 8.0)  # zone-1 cumulative demand, total rate 1.0
    nout0 = np.minimum(np.maximum(edges - 1.0, 0.0), 8.0)  # link 0 free-flow exit
    nin1, nin2 = emit[0] * nout0, emit[1] * nout0
    nout1 = emit[0] * np.minimum(np.maximum(edges - 2.0, 0.0), 8.0)
    nout2 = emit[1] * np.minimum(np.maximum(edges - 2.0, 0.0), 8.0)
    n_in = np.vstack([d1, nin1, nin2])
    n_out = np.vstack([nout0, nout1, nout2])
    release = np.vstack([d1, np.zeros_like(d1), np.zeros_like(d1)])
    return DNLOutput(
        scenario_hash=scenario.content_hash(),
        grid=scenario.grid,
        n_in=n_in,
        n_out=n_out,
        origin_release=release,
    )


def test_c8_correct_diverge_split_certifies() -> None:
    """An emission whose realized split matches the mandated turn fractions
    (0.6, 0.4) certifies with zero turn residual."""
    scenario = _diverge_scenario(split=(0.6, 0.4))
    metrics = DNLEvaluator(scenario).evaluate(_diverge_output(scenario, emit=(0.6, 0.4)))
    assert metrics["dnl_feasible"] == 1.0
    assert metrics["turn_residual"] == pytest.approx(0.0, abs=1e-9)


def test_c8_wrong_diverge_split_is_censored() -> None:
    """An emission splitting 0.5/0.5 while the scenario mandates 0.6/0.4 is
    censored by C8 ALONE: conservation (0.5 + 0.5 = 1.0), capacity, causality,
    storage all still pass, isolating the turning-fraction gate."""
    scenario = _diverge_scenario(split=(0.6, 0.4))
    metrics = DNLEvaluator(scenario).evaluate(_diverge_output(scenario, emit=(0.5, 0.5)))
    assert metrics["dnl_feasible"] == 0.0
    assert metrics["turn_residual"] > 0.0
    assert metrics["conservation_residual"] <= 1e-9  # C1 is NOT what caught it


def _merge_diverge_scenario() -> DynamicScenario:
    """Two origins (zones 1, 2) merge at interior node 5 and diverge to two
    dests (zones 3, 4): node 5 has 2 incoming and 2 outgoing links, so its turn
    matrix is (2, 2). The aggregate column split d_in[out_j] == sum_i
    frac[i,j]*d_out[in_i] IS decidable from per-link counts (each incoming link's
    outflow is observed), so C8 gates it (necessary; not sufficient for per-row
    cross-cancelling splits)."""
    net = Network(
        name="cert-merge-diverge",
        n_nodes=5,
        n_zones=4,
        first_thru_node=5,
        init_node=np.array([1, 2, 5, 5], dtype=np.int64),
        term_node=np.array([5, 5, 3, 4], dtype=np.int64),
        capacity=np.ones(4),
        length=np.zeros(4),
        free_flow_time=np.ones(4),
        b=np.zeros(4),
        power=np.ones(4),
        toll=np.zeros(4),
        link_type=np.ones(4, dtype=np.int64),
    )
    rates = np.zeros((1, 4, 4))
    rates[0, 0, 2] = 0.25  # z1 -> z3
    rates[0, 0, 3] = 0.25  # z1 -> z4
    rates[0, 1, 2] = 0.25  # z2 -> z3
    rates[0, 1, 3] = 0.25  # z2 -> z4
    # Non-uniform per-incoming splits: link 0 -> (0.7, 0.3), link 1 -> (0.2, 0.8).
    turns = TurningFractions(frac=((5, np.array([[0.7, 0.3], [0.2, 0.8]])),))
    return DynamicScenario(
        name="cert-merge-diverge",
        network=net,
        dynamics=LinkDynamics(
            length=np.ones(4),
            free_speed=np.ones(4),
            wave_speed=np.full(4, math.inf),
            jam_density=np.full(4, math.inf),
            capacity=np.ones(4),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 8.0]), rates=rates),
        grid=TimeGrid(dt=0.5, n_steps=28),
        turns=turns,
    )


def _merge_diverge_output(scenario: DynamicScenario, out2_share: np.ndarray | None) -> DNLOutput:
    """Merge-diverge emission where both origins inject at rate 0.5, merge at
    node 5, and the merged flow splits into out-link 5->3 (index 2) by the
    per-step ``out2_share`` fraction (remainder to 5->4). ``out2_share=None`` uses
    the turn-mandated share sum_i frac[i, 0] * d_out[in_i] (a correct emission)."""
    edges = scenario.grid.edges
    din = np.minimum(0.5 * edges, 4.0)  # each origin's release & merge-link inflow
    nout_in = np.minimum(np.maximum(0.5 * (edges - 1.0), 0.0), 4.0)  # merge-link exits
    d_merge = np.diff(2.0 * nout_in)  # total per-step flow arriving at node 5
    if out2_share is None:
        # correct: frac[0,0]*d_out[in0] + frac[1,0]*d_out[in1] with equal inflows
        share2 = (0.7 + 0.2) / 2.0  # = 0.45 of the merged flow to out-link index 2
    else:
        share2 = out2_share
    cin2 = np.concatenate([[0.0], np.cumsum(share2 * d_merge)])
    cin3 = 2.0 * nout_in - cin2  # remainder to out-link index 3 (conserves node 5)
    lag = np.minimum(np.maximum(0.5 * (edges - 2.0), 0.0), 4.0)  # free-flow exit shape
    n_in = np.vstack([din, din, cin2, cin3])
    n_out = np.vstack([nout_in, nout_in, (cin2[-1] / 4.0) * lag, (cin3[-1] / 4.0) * lag])
    release = np.vstack([din, din, np.zeros_like(din), np.zeros_like(din)])
    return DNLOutput(
        scenario_hash=scenario.content_hash(),
        grid=scenario.grid,
        n_in=n_in,
        n_out=n_out,
        origin_release=release,
    )


def test_c8_multi_in_correct_split_certifies() -> None:
    """A multi-in (merge+diverge) emission whose realized split obeys
    d_in[out_j] == sum_i frac[i,j] * d_out[in_i] certifies. With equal inflows
    and turns [[0.7,0.3],[0.2,0.8]] the mandated shares are 0.45/0.55."""
    scenario = _merge_diverge_scenario()
    metrics = DNLEvaluator(scenario).evaluate(_merge_diverge_output(scenario, out2_share=None))
    assert metrics["dnl_feasible"] == 1.0
    assert metrics["turn_residual"] == pytest.approx(0.0, abs=1e-9)


def test_c8_multi_in_node_is_gated() -> None:
    """A 2-in diverge IS gated: each incoming link's outflow d_out[in_i] is
    observed separately, so d_in[out_j] == sum_i frac[i,j] * d_out[in_i] is fully
    decidable from aggregate counts. An emission routing ALL merged flow to one
    outgoing link (grossly violating the 0.45/0.55 mandate) is now censored by C8
    alone (regression for the adversarial-review CRITICAL: the old 1-in-only
    abstention silently certified this)."""
    scenario = _merge_diverge_scenario()
    all_to_out2 = np.ones(scenario.grid.n_steps)  # 100% to out-link index 2 every step
    out = _merge_diverge_output(scenario, out2_share=all_to_out2)
    metrics = DNLEvaluator(scenario).evaluate(out)
    assert metrics["dnl_feasible"] == 0.0
    assert metrics["turn_residual"] > 0.0
    assert metrics["conservation_residual"] <= 1e-9  # C1 is NOT what caught it
