"""Tests for Smith's (1984) route-swap day-to-day dynamics (dtd-swap).

Unlike the UE *solvers*, dtd-swap models the disequilibrium adjustment process:
travelers swap from costlier to cheaper routes each day. Its fixed point is the
Wardrop UE, so it is validated as a UE model (converges to the analytic Braess UE
and toward the Sioux Falls best-known objective) PLUS its distinctive dynamical
signature: the Beckmann objective is a Lyapunov function that decreases
monotonically to the UE value (the identity Zdot = -a*V), and Smith's flow-
weighted disequilibrium V(h) vanishes at equilibrium. The monotone-decrease tests
are also the regression that pins the Smith & Wisten step bound -- without it the
raw swap overshoots into a 2-day limit cycle and never converges.
"""

import numpy as np
import pytest
from conftest import load_or_skip

from tabench import (
    Budget,
    Demand,
    Evaluator,
    Network,
    RngBundle,
    RouteSwapDTDModel,
    Scenario,
    Trace,
    braess_scenario,
)

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
    (model or RouteSwapDTDModel()).solve(
        scenario, Budget(**budget_kwargs), RngBundle(0), trace
    )
    return trace


# ------------------------------------------------------------- convergence
def test_converges_to_braess_ue(braess):
    """The route-swap dynamics settle on the exact Wardrop UE (route flows are
    non-unique, but the link flows are)."""
    trace = _solve(braess, iterations=800, target_relative_gap=1e-8)
    metrics = Evaluator(braess).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-7
    np.testing.assert_allclose(trace.final.link_flows, REF_FLOWS, atol=1e-4)


def test_scales_to_siouxfalls(siouxfalls):
    """On a real network the gap keeps shrinking and the Beckmann objective
    approaches the published optimum (day-to-day dynamics converge slowly, so this
    demonstrates scaling, not a tight terminal gap)."""
    trace = _solve(siouxfalls, iterations=500, target_relative_gap=1e-4)
    gaps = [s.self_report["relative_gap"] for s in trace]
    metrics = Evaluator(siouxfalls).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert gaps[-1] < 5e-3  # steadily converging toward UE
    assert gaps[-1] < gaps[0]
    obj = metrics["beckmann_objective"] / SIOUXFALLS_UNIT_FACTOR
    assert obj == pytest.approx(SIOUXFALLS_TNTP_OBJECTIVE, rel=3e-3)


# ------------------------------------------------------- Lyapunov / stability
def test_beckmann_is_a_monotone_lyapunov_function(braess):
    """The distinctive day-to-day property AND the step-bound regression: along
    Smith's swap the Beckmann objective decreases monotonically to the UE value
    (Zdot = -a*V <= 0). Without the Smith & Wisten 1/(B M) step cap the raw swap
    limit-cycles and this fails."""
    trace = _solve(braess, iterations=400)
    beckmann = [s.self_report["beckmann"] for s in trace]
    assert all(beckmann[i] >= beckmann[i + 1] - 1e-9 for i in range(len(beckmann) - 1))
    # ... and it lands on the same Beckmann value a UE solver reaches.
    assert beckmann[-1] == pytest.approx(386.0, abs=1e-2)


def test_beckmann_monotone_on_siouxfalls(siouxfalls):
    """Monotone Lyapunov descent holds on a congested multi-OD network too (the
    step bound uses the largest route cost, so it adapts as flows load up)."""
    trace = _solve(siouxfalls, iterations=300)
    beckmann = [s.self_report["beckmann"] for s in trace]
    assert all(
        beckmann[i] >= beckmann[i + 1] - 1e-6 * abs(beckmann[i])
        for i in range(len(beckmann) - 1)
    )


def _high_curvature_scenario():
    """A congested all-BPR-power-4 multi-OD network (from an adversarial fuzz) on
    which the raw B*M step overshoots and Beckmann rises mid-run; the Armijo
    backtracking must restore monotone descent. Hardcoded for determinism."""
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
    """REGRESSION for the curvature-overshoot fix. The Smith & Wisten B*M step
    bounds the cost LEVEL but not its DERIVATIVE, so on high-curvature (power-4)
    congested links the raw step overshoots and the Beckmann Lyapunov function
    RISES mid-run (an adversarial review found this at the default step_safety).
    Disabling backtracking (max_backtracks=0) reproduces the rise; the default
    Armijo backtracking on Z restores strict monotone descent and still
    converges."""
    sc = _high_curvature_scenario()

    def max_rel_rise(model):
        trace = Trace()
        model.solve(sc, Budget(iterations=60), RngBundle(0), trace)
        beck = [s.self_report["beckmann"] for s in trace]
        return max(
            (beck[i + 1] - beck[i]) / max(abs(beck[i]), 1.0) for i in range(len(beck) - 1)
        )

    # Raw B*M step (no backtracking) overshoots: Beckmann rises materially.
    assert max_rel_rise(RouteSwapDTDModel(max_backtracks=0)) > 1e-3
    # Default (Armijo backtracking) is strictly monotone non-increasing.
    assert max_rel_rise(RouteSwapDTDModel()) < 1e-9
    # ... and it still converges to the UE.
    trace = Trace()
    RouteSwapDTDModel().solve(
        sc, Budget(iterations=1500, target_relative_gap=1e-4), RngBundle(0), trace
    )
    assert Evaluator(sc).evaluate(trace.final.link_flows)["relative_gap"] < 1e-3


def test_smith_disequilibrium_vanishes_at_equilibrium(braess):
    """Smith's flow-weighted disequilibrium V(h) = sum h_p ([c_p - c_k]+)^2 is a
    Lyapunov measure that is zero iff Wardrop UE; it collapses toward zero as the
    dynamics converge (reported as provenance, never scored)."""
    trace = _solve(braess, iterations=800, target_relative_gap=1e-9)
    diseq = [s.self_report["smith_disequilibrium"] for s in trace]
    assert diseq[-1] < 1e-6
    # It is strictly positive in disequilibrium (once >1 route exists) and shrinks.
    assert max(diseq) > 1.0
    assert diseq[-1] < max(diseq)


# -------------------------------------------------------------- mechanics
def test_paradigm_and_registry():
    from tabench.models import MODEL_REGISTRY

    assert "dtd-swap" in MODEL_REGISTRY
    assert MODEL_REGISTRY["dtd-swap"]().capabilities.paradigm == "day_to_day"


def test_bookkeeping_and_conservation(braess):
    trace = _solve(braess, iterations=10)
    assert len(trace) == 10
    # One Dijkstra at init + one per recorded day.
    assert trace.final.coords.sp_calls == 10 + 1
    v = trace.final.link_flows
    assert np.all(v >= 0)
    metrics = Evaluator(braess).evaluate(v)
    # Demand is conserved structurally (swaps stay within an OD), so link flows
    # rebuilt from route flows balance to the float-noise floor at every day.
    assert metrics["node_balance_residual"] <= 1e-6 * braess.demand.total
    for key in ("relative_gap", "beckmann", "smith_disequilibrium"):
        assert key in trace.final.self_report


# --------------------------------------------------------------- prune_tol (B4)
def test_prune_tol_changes_retained_flows(siouxfalls):
    """B4 (on): prune_tol is load-bearing -- it decides which small-flow non-shortest
    routes are dropped from the working set, and dropping different routes changes the
    retained set and thus the (non-converged) link-flow trajectory.

    swap_rate is deliberately SMALL (0.05): at the default 1.0 the adaptively-capped
    step can zero a costlier route EXACTLY in one day, so no flow ever occupies the
    (1e-14, 1e-6] prune window and both tolerances prune the same set -- which is
    platform-dependent (byte-identical runs on the CI 3.12 numpy, different locally).
    A small swap_rate makes the decay of costlier routes multiplicative, so a decaying
    flow must transit the window's eight decades over many days -- a property a 1-ulp
    arithmetic difference cannot erase -- and the loosened prune_tol=1e-6 provably
    drops routes the tight default 1e-14 retains. MUTANT KILL: hardcoding prune_tol
    (ignoring the factor) collapses the two runs to byte-identical."""
    def flows(prune_tol):
        trace = Trace()
        RouteSwapDTDModel(prune_tol=prune_tol, swap_rate=0.05).solve(
            siouxfalls, Budget(iterations=150), RngBundle(0), trace
        )
        return trace.final.link_flows

    tight, loose = flows(1e-14), flows(1e-6)
    assert not np.array_equal(tight, loose)
    assert Evaluator(siouxfalls).evaluate(loose)["feasible"] == 1.0


def test_prune_tol_default_is_pinned(siouxfalls):
    """B4 (off): prune_tol defaults to 1e-14, so a default run is byte-identical to an
    explicit prune_tol=1e-14 run -- the off-pin (the existing Braess convergence
    anchors, which never set prune_tol, pin the default flows)."""
    def flows(model):
        trace = Trace()
        model.solve(siouxfalls, Budget(iterations=150), RngBundle(0), trace)
        return trace.final.link_flows

    np.testing.assert_array_equal(
        flows(RouteSwapDTDModel()), flows(RouteSwapDTDModel(prune_tol=1e-14))
    )
