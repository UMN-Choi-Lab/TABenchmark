"""Tests for Friesz et al.'s (1994) route-based day-to-day disequilibrium (dtd-friesz).

Unlike ``dtd-swap`` (Smith 1984, proportional route swaps) and ``dtd-link`` (He,
Guo & Liu 2010, link-space projection), dtd-friesz evolves per-OD route flows by
the projected dynamical system ``h-dot = P_K(h, -c(h))`` -- the projection of the
NEGATIVE route-cost vector onto the demand-feasible set -- discretized by the
Bertsekas & Gafni (1982) step ``h_{k+1} = P_K(h_k - a c(h_k))``. Because
``partial Z / partial h_p = c_p`` exactly (Z = Beckmann), this is projected
gradient descent on Beckmann in ROUTE space; its rest point is the same Wardrop
UE, so it is validated as a UE model (converges to the analytic Braess UE and
toward the Sioux Falls best-known objective) PLUS its distinctive PDS signature:

* MONOTONE Beckmann (Lyapunov) descent to the UE value -- the projected-gradient
  descent guarantee; the Armijo backtracking on the Beckmann objective is what
  delivers it, doubling as the step-overshoot regression.
* INVARIANCE -- the EXACT per-OD Euclidean simplex projection conserves the OD
  demand, so the emitted link flows stay in the OD-feasible set at EVERY recorded
  day (node-balance ~ 0), the hallmark of a *projected* dynamical system.
* It reaches the IDENTICAL certified UE that route-swap dtd-swap and link-based
  dtd-link reach from the IDENTICAL all-or-nothing start, isolating the
  route-space projected-gradient paradigm from Smith's swap and He-Guo-Liu's link
  projection.
* The route-flow excess-cost disequilibrium G(h) = sum h_p (c_p - u_w) collapses
  to ~0 (reported as provenance, never scored).
"""

import numpy as np
import pytest
from conftest import load_or_skip

from tabench import (
    Budget,
    Demand,
    Evaluator,
    FrieszDTDModel,
    LinkBasedDTDModel,
    Network,
    RngBundle,
    RouteSwapDTDModel,
    Scenario,
    Trace,
    braess_scenario,
    two_route_scenario,
)
from tabench.models.dtd_friesz import _project_simplex

REF_FLOWS = np.array([4.0, 2.0, 2.0, 2.0, 4.0])
SIOUXFALLS_TNTP_OBJECTIVE = 42.31335287107440
SIOUXFALLS_UNIT_FACTOR = 1e5


@pytest.fixture(scope="module")
def braess():
    return braess_scenario()


@pytest.fixture(scope="module")
def siouxfalls():
    return load_or_skip("siouxfalls")


def _solve(scenario, model=None, **budget_kwargs):
    trace = Trace()
    (model or FrieszDTDModel()).solve(
        scenario, Budget(**budget_kwargs), RngBundle(0), trace
    )
    return trace


# ------------------------------------------------------------- convergence
def test_converges_to_braess_ue(braess):
    """The route-space projected-gradient dynamics settle on the exact Wardrop UE
    (route flows are non-unique, but the link flows are)."""
    trace = _solve(braess, iterations=800, target_relative_gap=1e-8)
    metrics = Evaluator(braess).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-7
    np.testing.assert_allclose(trace.final.link_flows, REF_FLOWS, atol=1e-4)


def test_converges_to_two_route_ue():
    """The two-route analytic anchor (c_A = 2 + f_A, c_B = 1.5 + 2 f_B, demand 4)
    has the deterministic UE f_A = 2.5, f_B = 1.5 (common cost 4.5); the projected
    gradient descent reaches it exactly (link flows [2.5, 2.5, 1.5, 1.5])."""
    scenario = two_route_scenario(sue_theta=None)  # plain fixed-demand UE
    trace = _solve(scenario, iterations=800, target_relative_gap=1e-9)
    metrics = Evaluator(scenario).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-7
    np.testing.assert_allclose(
        trace.final.link_flows, np.array([2.5, 2.5, 1.5, 1.5]), atol=1e-4
    )


def test_scales_to_siouxfalls(siouxfalls):
    """On a real network the gap keeps shrinking and the Beckmann objective
    approaches the published optimum (a single Lipschitz-normalized step_size
    generalizes from the O(10) Braess costs to the O(1e-2) Sioux Falls costs)."""
    trace = _solve(siouxfalls, iterations=500, target_relative_gap=1e-4)
    gaps = [s.self_report["relative_gap"] for s in trace]
    metrics = Evaluator(siouxfalls).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert gaps[-1] < 5e-3  # steadily converging toward UE
    assert gaps[-1] < gaps[0]
    obj = metrics["beckmann_objective"] / SIOUXFALLS_UNIT_FACTOR
    assert obj == pytest.approx(SIOUXFALLS_TNTP_OBJECTIVE, rel=3e-3)


# --------------------------------------------- the exact simplex projection
def test_projection_is_exact_and_feasible():
    """The Bertsekas-Gafni step is an EXACT Euclidean projection onto the scaled
    simplex {x >= 0 : sum x = q}: it conserves the total and never goes negative,
    and it is idempotent on already-feasible points (a nonexpansive projection)."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        m = int(rng.integers(1, 8))
        y = rng.normal(size=m) * 5.0
        q = float(rng.uniform(0.5, 10.0))
        x = _project_simplex(y, q)
        assert x.shape == (m,)
        assert np.all(x >= -1e-15)
        assert x.sum() == pytest.approx(q, abs=1e-12)
        # A feasible point projects to itself.
        np.testing.assert_allclose(_project_simplex(x, q), x, atol=1e-12)


def test_projected_gradient_direction_handcheck():
    """HAND-CHECK of the projected-gradient DIRECTION (the analytic dynamics anchor).
    Two-route OD (c_A = 2 + f_A, c_B = 1.5 + 2 f_B, demand 4, UE f_A = 2.5). From
    the equal split f_A = f_B = 2 the frozen costs are c_A = 4, c_B = 5.5; the
    INTERIOR simplex projection subtracts the per-OD mean gradient, so the first
    projected step y_p = h_p - a c_p moves flow toward the cheaper route by
    Df_A = -a (c_A - cbar) = +a (c_B - c_A)/2 = +0.75 a (Df_B = -0.75 a) --
    monotonically toward f_A = 2.5. A hand-checkable sign+magnitude on the step."""
    for a in (0.01, 0.05, 0.1):
        y = np.array([2.0 - a * 4.0, 2.0 - a * 5.5])  # h - a c, h = [2, 2]
        x = _project_simplex(y, 4.0)
        np.testing.assert_allclose(x, np.array([2.0 + 0.75 * a, 2.0 - 0.75 * a]), atol=1e-12)
        assert x[0] > 2.0 > x[1]  # flow moves toward the cheaper route A


# ------------------------------------------------------- Lyapunov / stability
def test_beckmann_is_a_monotone_lyapunov_function(braess):
    """The distinctive day-to-day property: along the projected-gradient dynamics
    the Beckmann objective decreases monotonically to the UE value (the descent
    guarantee of projected gradient on the convex Beckmann program). The Armijo
    backtracking on Z is what enforces it."""
    trace = _solve(braess, iterations=400)
    beckmann = [s.self_report["beckmann"] for s in trace]
    assert all(beckmann[i] >= beckmann[i + 1] - 1e-9 for i in range(len(beckmann) - 1))
    # ... and it lands on the same Beckmann value a UE solver reaches.
    assert beckmann[-1] == pytest.approx(386.0, abs=1e-2)


def test_beckmann_monotone_on_siouxfalls(siouxfalls):
    """Monotone Lyapunov descent holds on a congested multi-OD network too (the
    Lipschitz-normalized step keeps the projected step well-scaled as flows load
    up)."""
    trace = _solve(siouxfalls, iterations=300)
    beckmann = [s.self_report["beckmann"] for s in trace]
    assert all(
        beckmann[i] >= beckmann[i + 1] - 1e-6 * abs(beckmann[i])
        for i in range(len(beckmann) - 1)
    )


# --------------------------------------------- distinctive: projected dynamics
def test_link_flows_stay_in_omega_every_day(braess, siouxfalls):
    """INVARIANCE (the projected-dynamical-system hallmark): the EXACT per-OD
    Euclidean simplex projection conserves the OD demand, so the emitted link
    flows conserve the exact demand at every intersection on EVERY recorded day --
    not only at convergence (node-balance ~ 0). dtd-swap conserves structurally
    via within-OD swaps; dtd-friesz conserves by projecting onto the demand
    simplex."""
    for scenario in (braess, siouxfalls):
        trace = _solve(scenario, iterations=50)
        evaluator = Evaluator(scenario)
        tol = 1e-6 * scenario.demand.total
        for state in trace:
            assert evaluator.evaluate(state.link_flows)["node_balance_residual"] <= tol


def test_all_three_dtd_paradigms_reach_same_ue(braess):
    """CONTRAST (the paradigm difference): the route-space projected gradient
    (dtd-friesz), the route swap (dtd-swap), and the link-space projection
    (dtd-link) start from the IDENTICAL free-flow all-or-nothing state and reach
    the IDENTICAL certified UE link flows -- the unique Braess equilibrium --
    isolating the difference to the adjustment mechanism, not the fixed point."""
    friesz = _solve(braess, iterations=800, target_relative_gap=1e-9).final.link_flows
    swap = _solve(
        braess, RouteSwapDTDModel(), iterations=800, target_relative_gap=1e-9
    ).final.link_flows
    link = _solve(
        braess, LinkBasedDTDModel(), iterations=800, target_relative_gap=1e-9
    ).final.link_flows
    np.testing.assert_allclose(friesz, REF_FLOWS, atol=1e-4)
    np.testing.assert_allclose(friesz, swap, atol=1e-4)
    np.testing.assert_allclose(friesz, link, atol=1e-4)


def test_excess_cost_disequilibrium_collapses(braess):
    """The route-flow excess-cost disequilibrium G(h) = sum h_p (c_p - u_w) (the
    tatonnement/VI measure, TSTT - SPTT on the working set) is positive in
    disequilibrium and collapses toward zero as the dynamics reach UE (reported as
    provenance, never scored)."""
    trace = _solve(braess, iterations=800, target_relative_gap=1e-9)
    excess = [s.self_report["excess_cost"] for s in trace]
    assert excess[-1] < 1e-4
    # Positive in disequilibrium (once column generation opens >1 route) and shrinks.
    assert max(excess) > 1.0
    assert excess[-1] < max(excess)


def _high_curvature_scenario():
    """A congested all-BPR-power-4 multi-OD network (shared with the dtd-swap /
    dtd-link suites) on which an aggressive projected step overshoots and Beckmann
    rises; the Armijo backtracking must restore monotone descent. Hardcoded for
    determinism."""
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
    return Scenario("highcurv-dtd", net, Demand(od))


def test_armijo_backtracking_restores_monotone_descent():
    """REGRESSION for the step-overshoot fix. The local Lipschitz cost step
    ignores the aggregate BPR curvature along the projected ray, so an AGGRESSIVE
    step (step_size=4) overshoots the Beckmann minimum on high-curvature (power-4)
    congested links and the Beckmann Lyapunov function RISES mid-run. Disabling
    backtracking (max_backtracks=0) reproduces the rise; the default Armijo
    backtracking on Z restores strict monotone descent and still converges."""
    sc = _high_curvature_scenario()

    def max_rel_rise(model):
        trace = Trace()
        model.solve(sc, Budget(iterations=60), RngBundle(0), trace)
        beck = [s.self_report["beckmann"] for s in trace]
        return max(
            (beck[i + 1] - beck[i]) / max(abs(beck[i]), 1.0) for i in range(len(beck) - 1)
        )

    # Aggressive raw step (no backtracking) overshoots: Beckmann rises materially.
    assert max_rel_rise(FrieszDTDModel(step_size=4.0, max_backtracks=0)) > 1e-3
    # Default (Armijo backtracking) is strictly monotone non-increasing.
    assert max_rel_rise(FrieszDTDModel(step_size=4.0)) < 1e-9
    # ... and it still converges to the UE.
    trace = Trace()
    FrieszDTDModel(step_size=4.0).solve(
        sc, Budget(iterations=1500, target_relative_gap=1e-4), RngBundle(0), trace
    )
    assert Evaluator(sc).evaluate(trace.final.link_flows)["relative_gap"] < 1e-3


# -------------------------------------------------------------- mechanics
def test_paradigm_and_registry():
    from tabench.models import MODEL_REGISTRY

    assert "dtd-friesz" in MODEL_REGISTRY
    assert MODEL_REGISTRY["dtd-friesz"]().capabilities.paradigm == "day_to_day"


def test_bookkeeping_and_conservation(braess):
    trace = _solve(braess, iterations=10)
    assert len(trace) == 10
    # One Dijkstra at init + one per recorded day.
    assert trace.final.coords.sp_calls == 10 + 1
    v = trace.final.link_flows
    assert np.all(v >= 0)
    metrics = Evaluator(braess).evaluate(v)
    # The exact simplex projection conserves OD demand: node balance at the floor.
    assert metrics["node_balance_residual"] <= 1e-6 * braess.demand.total
    for key in ("relative_gap", "tstt", "sptt", "beckmann", "excess_cost"):
        assert key in trace.final.self_report


# --------------------------------------------------------------- prune_tol (B4)
def test_prune_tol_changes_retained_flows(siouxfalls):
    """B4 (on): prune_tol is load-bearing -- it decides which zero-flow non-shortest
    routes are dropped from the working set, and dropping different routes changes the
    retained set and thus the (non-converged) link-flow trajectory. On Sioux Falls a
    loosened prune_tol=1e-6 yields link flows NOT byte-identical to the tight default
    1e-14 (measured max|diff| ~5), while staying feasible. The adjustment_rate<1 blend
    leaves small-positive prunable route residuals (the full projected step would zero
    them exactly, so any tol prunes identically). MUTANT KILL: hardcoding prune_tol
    (ignoring the factor) collapses the two runs to byte-identical (max|diff| 0)."""
    def flows(prune_tol):
        trace = Trace()
        FrieszDTDModel(prune_tol=prune_tol, adjustment_rate=0.8).solve(
            siouxfalls, Budget(iterations=40), RngBundle(0), trace
        )
        return trace.final.link_flows

    tight, loose = flows(1e-14), flows(1e-6)
    assert not np.array_equal(tight, loose)
    assert np.abs(tight - loose).max() > 1e-3
    assert Evaluator(siouxfalls).evaluate(loose)["feasible"] == 1.0


def test_prune_tol_default_is_pinned(siouxfalls):
    """B4 (off): prune_tol defaults to 1e-14, so a default run is byte-identical to an
    explicit prune_tol=1e-14 run -- the off-pin (the existing Braess/two-route
    convergence anchors, which never set prune_tol, pin the default flows)."""
    def flows(model):
        trace = Trace()
        model.solve(siouxfalls, Budget(iterations=40), RngBundle(0), trace)
        return trace.final.link_flows

    np.testing.assert_array_equal(
        flows(FrieszDTDModel(adjustment_rate=0.8)),
        flows(FrieszDTDModel(adjustment_rate=0.8, prune_tol=1e-14)),
    )
