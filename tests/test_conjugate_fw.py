"""Tests for conjugate-direction Frank-Wolfe (CFW/BFW) and the convergence
protocol, locking in the invariants of Mitradjieva & Lindberg (2013):
exact conjugacy identities when coefficients are unclamped, monotone
Beckmann descent under exact line search, a common UE limit, and the
published FW > CFW > BFW iteration ordering on Sioux Falls.
"""

import math

import numpy as np
import pytest
from conftest import load_or_skip

from tabench import (
    BiconjugateFrankWolfeModel,
    Budget,
    ConjugateFrankWolfeModel,
    Demand,
    Evaluator,
    FrankWolfeModel,
    Network,
    RngBundle,
    Scenario,
    Trace,
    braess_scenario,
    run_experiment,
)

REF_FLOWS = np.array([4.0, 2.0, 2.0, 2.0, 4.0])


@pytest.fixture(scope="module")
def braess():
    return braess_scenario()


@pytest.fixture(scope="module")
def siouxfalls():
    return load_or_skip("siouxfalls")


def _solve(model_cls, scenario, **budget_kwargs):
    trace = Trace()
    model_cls().solve(scenario, Budget(**budget_kwargs), RngBundle(0), trace)
    return trace


# ---------------------------------------------------------------- correctness


@pytest.mark.parametrize("model_cls", [ConjugateFrankWolfeModel, BiconjugateFrankWolfeModel])
def test_analytic_braess_equilibrium(braess, model_cls):
    trace = _solve(model_cls, braess, iterations=25)
    metrics = Evaluator(braess).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert abs(metrics["relative_gap"]) < 1e-10
    np.testing.assert_allclose(trace.final.link_flows, REF_FLOWS, atol=1e-5)


@pytest.mark.parametrize(
    "model_cls", [FrankWolfeModel, ConjugateFrankWolfeModel, BiconjugateFrankWolfeModel]
)
def test_monotone_beckmann_descent(siouxfalls, model_cls):
    trace = _solve(model_cls, siouxfalls, iterations=60)
    objectives = [s.self_report["beckmann"] for s in trace]
    pairs = zip(objectives, objectives[1:], strict=False)
    assert all(b2 <= b1 + 1e-10 * abs(b1) for b1, b2 in pairs)


def test_all_variants_reach_the_same_equilibrium(siouxfalls):
    """At self-gap 1e-5 every variant's objective sits just above the optimum.

    The Beckmann objective can never be below the optimum (recomputed from
    the best-known flows), and at relative gap g the excess is O(g).
    """
    evaluator = Evaluator(siouxfalls)
    oracle = evaluator.evaluate(siouxfalls.reference.link_flows)["beckmann_objective"]
    finals = {}
    for cls in (FrankWolfeModel, ConjugateFrankWolfeModel, BiconjugateFrankWolfeModel):
        trace = _solve(cls, siouxfalls, iterations=3000, target_relative_gap=1e-5)
        finals[cls.name] = trace.final.link_flows
        objective = evaluator.evaluate(trace.final.link_flows)["beckmann_objective"]
        assert oracle - 1e-9 * oracle <= objective <= oracle * (1.0 + 5e-5)
    # Sioux Falls link flows are unique (strictly increasing costs).
    scale = float(siouxfalls.network.capacity.max())
    assert np.abs(finals["cfw"] - finals["fw"]).max() < 1e-2 * scale
    assert np.abs(finals["bfw"] - finals["fw"]).max() < 1e-2 * scale


def test_published_iteration_ordering(siouxfalls):
    """Iterations to self-gap 1e-4: BFW < CFW << FW (published 124/357/1869)."""
    iters = {}
    for cls in (FrankWolfeModel, ConjugateFrankWolfeModel, BiconjugateFrankWolfeModel):
        trace = _solve(cls, siouxfalls, iterations=3000, target_relative_gap=1e-4)
        iters[cls.name] = trace.final.coords.iterations
    assert iters["bfw"] <= iters["cfw"] < iters["fw"]
    assert iters["cfw"] < 0.5 * iters["fw"]
    assert iters["bfw"] < 300
    assert iters["cfw"] < 800


# ------------------------------------------------------- conjugacy identities


class _ProbedCFW(ConjugateFrankWolfeModel):
    """Capture (v, s_prev, y) at every conjugate step for identity checks."""

    name = "cfw-probe"

    def __init__(self, **overrides):
        super().__init__(**overrides)
        self.probes = []

    def _search_point(self, network, v, y, state):
        s_prev = state.get("s_prev")
        s = super()._search_point(network, v, y, state)
        if s_prev is not None:
            self.probes.append((network, v.copy(), s_prev.copy(), y.copy(), s.copy()))
        return s


def test_cfw_conjugacy_identity(siouxfalls):
    """When a is unclamped, d_bar^T H (s - v) = 0 exactly (paper eq 5)."""
    model = _ProbedCFW()
    trace = Trace()
    model.solve(siouxfalls, Budget(iterations=40), RngBundle(0), trace)
    checked = 0
    delta = ConjugateFrankWolfeModel._DELTA
    for network, v, s_prev, y, s in model.probes:
        h = network.link_cost_derivative(v)
        d_bar = s_prev - v
        d_fw = y - v
        numer = float(d_bar @ (h * d_fw))
        denom = float(d_bar @ (h * (d_fw - d_bar)))
        if abs(denom) <= 1e-30:
            continue
        a = numer / denom
        if not (0.0 < a < 1.0 - delta):
            continue  # clamped: identity not expected to hold
        d = s - v
        scale = math.sqrt(float(d_bar @ (h * d_bar))) * math.sqrt(float(d @ (h * d)))
        assert abs(float(d_bar @ (h * d))) <= 1e-9 * max(scale, 1e-30)
        checked += 1
    assert checked >= 5, "too few unclamped conjugate steps exercised"


class _ProbedBFW(BiconjugateFrankWolfeModel):
    """Capture full BFW steps (needs s_prev, s_prev2, tau) for identity checks."""

    name = "bfw-probe"

    def __init__(self, **overrides):
        super().__init__(**overrides)
        self.probes = []

    def _search_point(self, network, v, y, state):
        full = state.get("s_prev") is not None and state.get("s_prev2") is not None
        if full:
            snapshot = (
                v.copy(),
                state["s_prev"].copy(),
                state["s_prev2"].copy(),
                state["alpha_prev"],
                y.copy(),
            )
        s = super()._search_point(network, v, y, state)
        if full:
            self.probes.append((network, *snapshot, s.copy()))
        return s


def test_bfw_conjugacy_identities(siouxfalls):
    """Unclamped mu, nu satisfy the two linear identities of paper eq 9/App A."""
    model = _ProbedBFW()
    trace = Trace()
    model.solve(siouxfalls, Budget(iterations=40), RngBundle(0), trace)
    checked = 0
    for network, v, s_prev, s_prev2, tau, y, s in model.probes:
        h = network.link_cost_derivative(v)
        d_fw = y - v
        d_bar = s_prev - v
        d_bbar = tau * s_prev + (1.0 - tau) * s_prev2 - v
        mu_num = float(d_bbar @ (h * d_fw))
        mu_den = float(d_bbar @ (h * (s_prev2 - s_prev)))
        nu_num = float(d_bar @ (h * d_fw))
        nu_den = float(d_bar @ (h * d_bar))
        if abs(mu_den) <= 1e-30 or abs(nu_den) <= 1e-30:
            continue
        mu = -mu_num / mu_den
        nu = -nu_num / nu_den + mu * tau / (1.0 - tau)
        if mu <= 0.0 or nu <= 0.0:
            continue  # clamped: identities not expected to hold
        beta0 = 1.0 / (1.0 + mu + nu)
        beta1 = nu * beta0
        beta2 = mu * beta0
        np.testing.assert_allclose(
            beta0 * y + beta1 * s_prev + beta2 * s_prev2, s, rtol=0, atol=1e-12 * np.abs(s).max()
        )
        r1 = beta0 * mu_num + beta2 * mu_den
        r2 = beta0 * nu_num + (beta1 - beta2 * tau / (1.0 - tau)) * nu_den
        assert abs(r1) <= 1e-9 * max(abs(beta0 * mu_num), abs(beta2 * mu_den), 1e-30)
        assert abs(r2) <= 1e-9 * max(abs(beta0 * nu_num), abs(nu_den), 1e-30)
        checked += 1
    assert checked >= 3, "too few full BFW steps exercised"


# ------------------------------------------------------------------ machinery


def test_search_point_stays_feasible_convex(braess):
    """Search points are convex combinations of AON assignments: flows stay >= 0."""
    for cls in (ConjugateFrankWolfeModel, BiconjugateFrankWolfeModel):
        trace = _solve(cls, braess, iterations=25)
        for state in trace:
            assert np.all(state.link_flows >= -1e-12)


def test_link_cost_derivative_edge_cases():
    net = braess_scenario().network  # power = 1 everywhere
    h = net.link_cost_derivative(np.zeros(net.n_links))
    # p = 1: constant slope fft*b/cap, defined at v = 0 (numpy 0**0 = 1).
    np.testing.assert_allclose(h, net.free_flow_time * net.b / net.capacity)

    from dataclasses import replace

    # p > 1 at v = 0 -> 0; p = 0 -> exactly 0 for any v; 0 < p < 1 at v = 0 -> 0.
    for power, at_zero in ((4.0, 0.0), (0.0, 0.0), (0.5, 0.0)):
        variant = replace(net, power=np.full(net.n_links, power))
        h0 = variant.link_cost_derivative(np.zeros(net.n_links))
        assert np.all(np.isfinite(h0))
        np.testing.assert_allclose(h0, at_zero)
    # 0 < p < 1 at v > 0 is finite and positive.
    variant = replace(net, power=np.full(net.n_links, 0.5))
    hp = variant.link_cost_derivative(np.ones(net.n_links))
    assert np.all(np.isfinite(hp)) and np.all(hp > 0)


def test_target_gap_early_stop_and_manifest(braess, tmp_path):
    result = run_experiment(
        braess,
        [FrankWolfeModel()],
        Budget(iterations=500, target_relative_gap=1e-6),
        out_dir=tmp_path,
    )
    last = result.rows[-1]
    assert last["iterations"] < 500
    assert last["self_relative_gap"] <= 1e-6
    assert result.manifest["budget"]["target_relative_gap"] == 1e-6


def test_budget_requires_a_resource_axis():
    with pytest.raises(ValueError, match="resource axis"):
        Budget(target_relative_gap=1e-4)


# ------------------------------------------- degenerate-direction restart (B4)


def _degenerate_conjugate_scenario() -> Scenario:
    """An 8-link 4-node power-4 BPR network on which BFW's conjugate search
    direction DEGENERATES (the exact line search returns alpha<=0 while the search
    point s != the AON point y) before convergence, so the plain-FW restart is
    load-bearing. Hardcoded from a random-network search hit: with the restart BFW
    reaches the unique UE (gap ~1e-16 by iteration 12); without it, BFW breaks at
    iteration 5 with relative gap ~0.30 (a false first-order-optimal stop)."""
    init = np.array([1, 2, 1, 4, 2, 3, 3, 4], dtype=np.int64)
    term = np.array([2, 1, 4, 1, 3, 2, 4, 3], dtype=np.int64)
    cap = np.array(
        [0.460019, 1.036316, 1.114387, 0.571556, 1.548781, 0.493242, 0.965088, 1.178458]
    )
    fft = np.array(
        [4.875652, 6.281187, 7.64054, 9.606405, 3.55781, 6.836925, 7.265944, 3.634487]
    )
    m = len(init)
    net = Network(
        name="fw-restart", n_nodes=4, n_zones=3, first_thru_node=1,
        init_node=init, term_node=term, capacity=cap, length=np.zeros(m),
        free_flow_time=fft, b=np.full(m, 0.15), power=np.full(m, 4.0),
        toll=np.zeros(m), link_type=np.ones(m, dtype=np.int64),
    )
    od = np.zeros((3, 3))
    for (i, j), val in {
        (0, 1): 39.07111, (0, 2): 15.98951, (1, 2): 21.495838,
        (2, 0): 6.06211, (2, 1): 18.098534,
    }.items():
        od[i, j] = val
    return Scenario("fw-restart", net, Demand(od))


def test_bfw_degenerate_direction_restart_reaches_ue():
    """BFW's degenerate-conjugate-direction restart is load-bearing here: the
    conjugate direction degenerates mid-run (line search alpha<=0 with s != y), and
    ONLY the restart -- clear the conjugacy state, retake a plain FW step -- lets BFW
    push past it to the unique UE (link flows are unique on a strictly-increasing-cost
    BPR network). Existing tests only ``continue`` past this branch on nice networks;
    this reaches it and asserts its effect. MUTANT KILL: disabling the restart branch
    makes BFW break at iteration 5 with relative gap ~0.30, so the convergence
    assertion fails."""
    sc = _degenerate_conjugate_scenario()
    trace = Trace()
    BiconjugateFrankWolfeModel().solve(sc, Budget(iterations=60), RngBundle(0), trace)
    metrics = Evaluator(sc).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-6  # restart-off stalls at ~0.30
    # Deterministic (per capabilities): two runs are byte-identical.
    again = Trace()
    BiconjugateFrankWolfeModel().solve(sc, Budget(iterations=60), RngBundle(0), again)
    np.testing.assert_array_equal(trace.final.link_flows, again.final.link_flows)
