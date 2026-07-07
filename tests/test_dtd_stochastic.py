"""Tests for Cascetta's (1989) stochastic-process day-to-day model (dtd-stochastic).

The benchmark's first genuinely stochastic day-to-day model: each day a finite
population of travelers (``N_od = max(1, round(scale * d_od))`` per OD pair)
draws routes by multinomial sampling from the Dial-STOCH logit fractions at the
perceived costs ``p``, and the memory smooths toward the EXPERIENCED costs of
the realized flow ``p <- (1 - w) p + w t(v)`` (dtd-horowitz's filter, driven by
the sampled realization -- so ``{p}`` is a Markov chain, not a deterministic
map). Daily flows never converge; the emitted flow is the burnt-in time
average, which converges (ergodic theorem) to the stationary mean ~ logit SUE
with a finite-population bias that vanishes as the population grows (Davis &
Nihan 1993). Validated as an SUE model -- on the two-route anchor the time
average certifies against the analytic binary-logit split via the EXISTING
ADR-001 Dial-STOCH residual (P1, same mechanism as sue-msa/dtd-horowitz) --
PLUS its distinctive signature: the daily flows keep a persistent
O(1/sqrt(N)) deviation from the mean even while the time-average residual is
small, the deviation shrinks monotonically with population_scale (the
Davis-Nihan large-population / dtd-horowitz limit), and the certified residual
floors at O(bias + SE) rather than solver precision -- honestly witnessing that
the stationary MEAN is only approximately SUE at finite population, so anchor
tolerances are 0.05, not 1e-5.
"""

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from tabench import (
    Budget,
    CascettaStochasticProcessModel,
    Demand,
    DialSUEModel,
    Evaluator,
    Network,
    RngBundle,
    Scenario,
    Trace,
    braess_scenario,
    two_route_scenario,
)
from tabench.metrics.gaps import node_balance_residual
from tabench.models import dtd_stochastic
from tabench.models._stoch import StochEngine
from tabench.models.dtd_stochastic import _sampled_dial_load

# Golden content hash of the Braess scenario, unchanged: this model adds no
# scenario field, so every existing content hash must stay byte-identical.
BRAESS_GOLDEN_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"


def _fixed_point_route_a(theta: float, demand: float = 4.0) -> float:
    """Root of the binary-logit fixed point ``f_A = D / (1 + exp(theta (c_A -
    c_B)))`` on the two-route anchor -- the target the stationary MEAN of the
    day-to-day chain approximates. Recomputed here, never a trusted digit."""

    def residual(f_a: float) -> float:
        c_a = 2.0 + f_a
        c_b = 1.5 + 2.0 * (demand - f_a)
        return f_a - demand / (1.0 + math.exp(theta * (c_a - c_b)))

    return brentq(residual, 0.0, demand, xtol=1e-12)


@pytest.fixture(scope="module")
def scenario():
    return two_route_scenario()  # demand 4, theta 0.5, logit


@pytest.fixture(scope="module")
def anchor_trace(scenario):
    """The anchor run: population_scale 25 (N = 100 travelers), w = 0.3,
    burn_in 500, 3000 days, RngBundle(0) -- shared across tests (the trace is
    read-only)."""
    trace = Trace()
    CascettaStochasticProcessModel(
        smoothing_weight=0.3, population_scale=25.0, burn_in_days=500
    ).solve(scenario, Budget(iterations=3000), RngBundle(0), trace)
    return trace


def _solve(sc, model=None, macrorep=0, **budget_kwargs):
    trace = Trace()
    (model or CascettaStochasticProcessModel()).solve(
        sc, Budget(**budget_kwargs), RngBundle(0, macrorep=macrorep), trace
    )
    return trace


def _braess_sue(theta: float = 0.1) -> Scenario:
    """A multi-path logit-SUE task on the Braess network (three OD paths) built
    from the shipped Braess network -- exercises the sampler on a real >2-route
    network, not just the analytic anchor."""
    b = braess_scenario()
    return Scenario(
        name="braess-sue",
        network=b.network,
        demand=b.demand,
        sue_family="logit",
        sue_theta=theta,
    )


# ------------------------------------------------------------- convergence
def test_time_average_converges_to_logit_sue(anchor_trace, scenario):
    """The burnt-in time average settles within 0.05 of the analytic binary-logit
    SUE split and certifies to residual < 0.05 -- NOT to solver precision: at
    N = 100 travelers the stationary mean carries a finite-population bias and
    the time average a sampling SE, so the certified residual floors at
    O(bias + SE) ~ 0.01 (the honest finite-population caveat)."""
    f_a = _fixed_point_route_a(theta=0.5)
    final = anchor_trace.final
    expected = np.array([f_a, f_a, 4.0 - f_a, 4.0 - f_a])
    np.testing.assert_allclose(final.link_flows, expected, atol=0.05)
    metrics = Evaluator(scenario).evaluate(final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["node_balance_residual"] <= 1e-6 * scenario.demand.total
    assert metrics["sue_fixed_point_residual"] < 0.05
    # The stationary window opened after the burn-in: 3000 - 500 days.
    assert final.self_report["window_days"] == 2500.0
    # The logit SUE is NOT the deterministic UE: the UE gap stays strictly
    # positive as a descriptive column (like sue-msa/dtd-horowitz).
    assert metrics["relative_gap"] > 0.01


def test_cross_solver_agrees_with_sue_msa(anchor_trace, scenario):
    """dtd-stochastic's stationary time average and sue-msa's deterministic
    fixed point agree on the anchor to the finite-population tolerance -- the
    chain's mean approximates the SAME pinned Dial-STOCH fixed point."""
    msa = _solve(scenario, DialSUEModel(), iterations=500, target_relative_gap=1e-9)
    np.testing.assert_allclose(
        anchor_trace.final.link_flows, msa.final.link_flows, atol=0.05
    )


def test_converges_on_multipath_braess_sue():
    """On a genuinely multi-path network (three Braess routes) the sampled
    day-to-day process stays demand-feasible at EVERY day (each traveler is
    routed; the emitted average is a convex combination of feasible loads) and
    the time average certifies to a loose tolerance."""
    sc = _braess_sue(theta=0.1)
    trace = _solve(
        sc,
        CascettaStochasticProcessModel(
            smoothing_weight=0.3, population_scale=25.0, burn_in_days=200
        ),
        iterations=800,
    )
    worst = max(node_balance_residual(sc, s.link_flows) for s in trace)
    assert worst <= 1e-6 * sc.demand.total
    metrics = Evaluator(sc).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["sue_fixed_point_residual"] < 0.05


# ------------------------------------- stationary distribution, not fixed point
def test_daily_variability_persists(anchor_trace):
    """The distinctive property (the difference from dtd-horowitz): the DAILY
    flows never converge -- the tail daily deviation ||v_n - vbar||_1 / D stays
    bounded away from 0 at finite population -- even while the TIME-AVERAGE
    residual is small. Equilibrium is a stationary distribution whose spread is
    the model's object, not a fixed point."""
    tail = [s.self_report["daily_flow_deviation"] for s in list(anchor_trace)[-100:]]
    assert float(np.mean(tail)) > 0.05
    assert anchor_trace.final.self_report["sue_fixed_point_residual"] < 0.05


def test_variability_shrinks_with_population_scale(scenario):
    """Davis-Nihan large-population limit: the persistent daily deviation is
    O(1/sqrt(N)), so it shrinks monotonically in population_scale (scale 1 >>
    scale 25 >> scale 1e4); the deterministic-in-the-mean dtd-horowitz dynamics
    are the scale -> infinity limit."""

    def tail_deviation(scale: float) -> float:
        trace = _solve(
            scenario,
            CascettaStochasticProcessModel(
                smoothing_weight=0.3, population_scale=scale, burn_in_days=100
            ),
            iterations=400,
        )
        return float(
            np.mean([s.self_report["daily_flow_deviation"] for s in list(trace)[-100:]])
        )

    dev_small, dev_mid, dev_large = map(tail_deviation, (1.0, 25.0, 1e4))
    assert dev_small > dev_mid > dev_large
    assert dev_large < 0.05 < dev_small


def test_seeded_reproducibility_and_macrorep_independence(scenario):
    """P8: same (root_seed, macrorep) replays byte-identically (per-day Philox
    streams, no global seed); a new macrorep is an independent trajectory
    (different daily realizations) with a consistent stationary mean."""

    def run(macrorep: int) -> Trace:
        return _solve(
            scenario,
            CascettaStochasticProcessModel(
                smoothing_weight=0.3, population_scale=25.0, burn_in_days=100
            ),
            macrorep=macrorep,
            iterations=400,
        )

    first, replay, other = run(0), run(0), run(1)
    for a, b in zip(first, replay, strict=True):
        np.testing.assert_array_equal(a.link_flows, b.link_flows)
    assert any(
        not np.array_equal(a.link_flows, c.link_flows)
        for a, c in zip(first, other, strict=True)
    )
    # Different trajectories, same stationary mean (to sampling tolerance).
    np.testing.assert_allclose(
        first.final.link_flows, other.final.link_flows, atol=0.05
    )


# ------------------------------------------------------- sampler unbiasedness
def test_sampled_load_is_unbiased(scenario):
    """The load-bearing new code: the multinomial backward pass is an unbiased
    sampler of the pinned Dial-STOCH load -- multinomial means telescope node by
    node to the deterministic recursion, so the mean of many sampled loads at
    FIXED costs matches StochEngine.load at those costs to CLT tolerance, and
    every single draw routes all demand."""
    engine = StochEngine(scenario.network)
    costs = scenario.network.link_cost(np.zeros(scenario.network.n_links))
    expected = engine.load(costs, scenario.demand, 0.5)
    n_samples = 2000
    acc = np.zeros(scenario.network.n_links)
    for i in range(n_samples):
        gen = RngBundle(0).generator(source=0, replication=i)
        sample = _sampled_dial_load(engine, costs, scenario.demand, 0.5, gen, 25.0)
        if i < 5:  # every draw is demand-feasible, not just the mean
            assert node_balance_residual(scenario, sample) <= 1e-9
        acc += sample
    # Per-link CLT SE ~ 0.0044 at N = 100 travelers and 2000 draws; 0.02 is
    # ~4.5 sigma.
    np.testing.assert_allclose(acc / n_samples, expected, atol=0.02)


# ------------------------------------------------------------- honesty (P1)
def test_self_report_matches_harness_certificate(scenario):
    """P1 honesty: the model's self-reported residual equals the one the harness
    recomputes -- both call the SAME pinned StochEngine.load on the emitted time
    average (the model loads a second time each day precisely to certify), so
    they agree to float precision at every checkpoint."""
    trace = _solve(
        scenario,
        CascettaStochasticProcessModel(population_scale=25.0),
        iterations=60,
    )
    evaluator = Evaluator(scenario)
    for state in list(trace)[::10]:
        certified = evaluator.evaluate(state.link_flows)["sue_fixed_point_residual"]
        assert certified == pytest.approx(
            state.self_report["sue_fixed_point_residual"], rel=1e-9, abs=1e-15
        )


# ------------------------------------------------------------- stability mirror
def test_unstable_at_memoryless_weight(scenario):
    """Stability mirror of dtd-horowitz (empirical, flagged): at w = 1.0
    (memoryless 'use yesterday's cost') the MEAN map is unstable on the anchor
    (above the same w* ~ 0.81 threshold dtd-horowitz derives), the chain
    oscillates instead of mixing around the SUE, and the time average fails to
    certify small."""
    trace = _solve(
        scenario,
        CascettaStochasticProcessModel(
            smoothing_weight=1.0, population_scale=25.0, burn_in_days=100
        ),
        iterations=600,
    )
    assert trace.final.self_report["sue_fixed_point_residual"] > 0.1


# --------------------------------------------------------------------- guards
def test_requires_sue_scenario():
    """A deterministic (non-SUE) scenario has no theta: refuse it (theta is task
    data, not a model factor)."""
    with pytest.raises(ValueError, match="sue_theta|SUE scenario"):
        CascettaStochasticProcessModel().solve(
            braess_scenario(), Budget(iterations=5), RngBundle(0), Trace()
        )


def test_rejects_probit_scenario():
    """The logit stochastic-process model must refuse a probit-SUE task and
    point at the probit solver."""
    probit = two_route_scenario(sue_theta=0.1, sue_family="probit")
    with pytest.raises(ValueError, match="probit"):
        CascettaStochasticProcessModel().solve(
            probit, Budget(iterations=5), RngBundle(0), Trace()
        )


# ------------------------------------------------------------------- mechanics
def test_paradigm_and_registry():
    from tabench.models import MODEL_REGISTRY

    assert "dtd-stochastic" in MODEL_REGISTRY
    caps = MODEL_REGISTRY["dtd-stochastic"]().capabilities
    assert caps.paradigm == "day_to_day"
    # Genuinely stochastic: routed onto the macrorep track, unlike every other
    # day-to-day model.
    assert caps.deterministic is False
    assert caps.seedable is True


def test_bookkeeping_and_conservation(scenario):
    trace = _solve(scenario, iterations=10)
    assert len(trace) == 10
    # Two Dial-unit loads per day -- one Dijkstra-sweep sampled load to realize
    # the daily flow, one deterministic load at the average's costs to CERTIFY
    # -- both counted in sp_calls, with no separate day-0 load, so at k days
    # the count is exactly 2k (mirrors dtd-horowitz).
    assert trace.final.coords.sp_calls == 20
    v = trace.final.link_flows
    assert np.all(v >= 0)
    metrics = Evaluator(scenario).evaluate(v)
    # Every traveler is routed every day and the emitted average is a convex
    # combination of daily loads, so checkpoints balance to the float-noise
    # floor.
    assert metrics["feasible"] == 1.0
    assert metrics["node_balance_residual"] <= 1e-6 * scenario.demand.total
    for key in ("sue_fixed_point_residual", "daily_flow_deviation", "window_days"):
        assert key in trace.final.self_report
    # 10 days is inside the default 200-day burn-in: the emission is the
    # pre-stationary running mean, flagged by window_days == 0.
    assert trace.final.self_report["window_days"] == 0.0


def test_braess_content_hash_preserved():
    """This model adds no scenario field: the golden Braess content hash must be
    byte-identical."""
    assert braess_scenario().content_hash() == BRAESS_GOLDEN_HASH


# ------------------------------------------- adversarial-review regressions
def _high_curvature_sue(theta: float = 0.5) -> Scenario:
    """The congested all-BPR-power-4 multi-OD net (shared with the dtd-swap /
    dtd-link / dtd-friesz suites) as a logit-SUE task: the benchmark's congested
    day-to-day instance class, on which the mean map's stability threshold falls
    below the smoothing_weight default and the pinned certificate map amplifies
    flow error. Hardcoded for determinism."""
    init = np.array([1, 2, 2, 2, 3, 3, 4], dtype=np.int64)
    term = np.array([2, 1, 3, 4, 2, 4, 2], dtype=np.int64)
    cap = np.array([3.8995, 3.3823, 3.9535, 2.5282, 1.5649, 3.4335, 3.9491])
    fft = np.array([4.0593, 1.4442, 2.3014, 1.9029, 4.0468, 1.7647, 1.1414])
    m = len(init)
    net = Network(
        name="highcurv-dtd", n_nodes=4, n_zones=4, first_thru_node=1,
        init_node=init, term_node=term, capacity=cap, length=np.zeros(m),
        free_flow_time=fft, b=np.full(m, 0.15), power=np.full(m, 4.0),
        toll=np.zeros(m), link_type=np.ones(m, dtype=np.int64),
    )
    od = np.zeros((4, 4))
    for (i, j), val in {
        (0, 1): 5.9561, (0, 2): 6.3265, (1, 0): 5.3462, (1, 2): 7.3173,
        (1, 3): 3.538, (2, 1): 8.2523, (2, 3): 8.6405,
    }.items():
        od[i, j] = val
    return Scenario(
        "highcurv-sue", net, Demand(od), sue_family="logit", sue_theta=theta
    )


@pytest.fixture(scope="module")
def highcurv_fixed_point():
    """The pinned Dial-STOCH fixed point of the congested instance (sue-msa run
    to 2000 iterations, certified residual ~9e-4): the reference flow the
    regressions measure L1 flow error against."""
    sc = _high_curvature_sue()
    trace = _solve(sc, DialSUEModel(), iterations=2000)
    return sc, trace.final.link_flows


def test_congested_stability_threshold_is_task_dependent(highcurv_fixed_point):
    """REGRESSION (adversarial review, Major 1): on the congested BPR-power-4
    instance the mean map's stability threshold is far below the 0.3
    smoothing_weight default -- at defaults the chain orbits far from the SUE
    (L1 flow error > 0.1 per unit demand vs the pinned Dial fixed point), while
    at w = 0.01 (below the documented ~0.03 task threshold) the burnt-in time
    average lands within 0.02. The pin is on FLOW error -- the quantity w
    controls -- because the certified residual on this instance floors high
    regardless (the Major-2 amplification). The FactorSpec doc must keep
    declaring the threshold task-dependent instead of quoting only the anchor's
    ~0.81."""
    doc = CascettaStochasticProcessModel.factors["smoothing_weight"].doc
    assert "TASK-DEPENDENT" in doc and "0.03" in doc
    sc, v_star = highcurv_fixed_point
    total = sc.demand.total
    far = _solve(sc, CascettaStochasticProcessModel(), iterations=1200)
    assert np.abs(far.final.link_flows - v_star).sum() / total > 0.1
    near = _solve(
        sc,
        CascettaStochasticProcessModel(
            smoothing_weight=0.01, population_scale=25.0, burn_in_days=500
        ),
        iterations=2000,
    )
    assert np.abs(near.final.link_flows - v_star).sum() / total < 0.02


def test_certificate_amplifies_flow_error_on_congested_instance(highcurv_fixed_point):
    """REGRESSION (adversarial review, Major 2): the anchor's certified-floor
    calibration does not transfer -- on the congested instance the pinned Dial
    certificate map amplifies an L1-normalized flow perturbation of the fixed
    point by >> 1 (measured ~53x along the demand-feasible free-flow-blend
    direction at 1e-3), so the scored column saturates at O(0.1-1) for any
    finite-population time average even when the flow error is ~1e-3. The
    certificate stays SOUND -- at the (numerically converged) fixed point it
    reads ~1e-3, no false accept -- and the module docstring must keep
    documenting the amplification."""
    assert "amplif" in dtd_stochastic.__doc__
    sc, v_star = highcurv_fixed_point
    total = sc.demand.total
    evaluator = Evaluator(sc)
    assert evaluator.evaluate(v_star)["sue_fixed_point_residual"] < 2e-3
    # A demand-feasible perturbation direction: blend toward the free-flow Dial
    # load (both flows route the same demand, so the blend is never censored
    # and the FULL P1 scoring path certifies it).
    engine = StochEngine(sc.network)
    free_flow = engine.load(
        sc.network.link_cost(np.zeros(sc.network.n_links)), sc.demand, sc.sue_theta
    )
    eps = 1e-3
    blend = eps * total / np.abs(free_flow - v_star).sum()
    v = (1.0 - blend) * v_star + blend * free_flow
    metrics = evaluator.evaluate(v)
    assert metrics["feasible"] == 1.0
    flow_err = float(np.abs(v - v_star).sum() / total)  # == eps by construction
    assert metrics["sue_fixed_point_residual"] > 10.0 * flow_err


def test_burn_in_handoff_has_no_reset_discontinuity(scenario):
    """REGRESSION (adversarial review, Major 3): the emission hands off from the
    day-1 running mean to the stationary average only once the stationary window
    is burn_in days long. The replaced hard reset at day burn_in + 1 emitted a
    single day's multinomial sample: certified residual 0.53 at day 201 vs 0.014
    at day 200 (~38x collapse, seed 0), and Budget(iterations=201..396) all
    scored worse than iterations=200 -- a budget-quality inversion exactly where
    typical budgets land. Now the certified residual never collapses across the
    boundary (or across the handoff at 2 x burn_in) and the provenance column
    reports the window that actually backs the emission."""
    trace = _solve(
        scenario,
        CascettaStochasticProcessModel(
            smoothing_weight=0.3, population_scale=25.0, burn_in_days=200
        ),
        iterations=420,
    )
    resid = [s.self_report["sue_fixed_point_residual"] for s in trace]
    # Day 201 (the old 38x spike) stays in family with day 200 (observed 1.19x)
    assert resid[200] < 3.0 * resid[199]
    # ... and no day in [195, 420] -- spanning the burn-in boundary AND the
    # handoff at day 400 -- collapses (old max 0.53, new max ~0.025).
    assert max(resid[194:]) < 0.1
    # Provenance: window_days flags the pre-handoff running mean with 0, then
    # reports the stationary window backing the emission.
    window = [s.self_report["window_days"] for s in trace]
    assert window[398] == 0.0  # day 399: still the day-1 running mean
    assert window[399] == 200.0  # day 400: the stationary average takes over
    assert window[419] == 220.0
