"""Tests for He, Guo & Liu's (2010) link-based day-to-day dynamics (dtd-link).

Unlike ``dtd-swap`` (Smith 1984), whose state is per-OD *route* flows, dtd-link
adjusts the aggregate *link*-flow vector directly on the feasible polytope Omega:
each day the emitted link flows move toward the frozen-cost proximal target
x*(v) = Proj_Omega(v - a t(v)). Its rest point is the same Wardrop UE, so it is
validated as a UE model (converges to the analytic Braess UE and toward the
Sioux Falls best-known objective) PLUS its distinctive link-based signature:

* INVARIANCE -- the emitted link flows never leave the OD-feasible set Omega
  (node-balance residual ~ 0 at EVERY recorded day), the He et al. invariance
  principle; and it reaches the identical certified UE that the *route-swap*
  dtd-swap reaches from the identical all-or-nothing start, isolating the
  link-target-vs-route-swap paradigm difference.
* MONOTONE Beckmann (Lyapunov) descent -- the RBAP signature; the Armijo
  backtracking on the Beckmann objective is what guarantees it, doubling as the
  step-overshoot regression (an aggressive step raises Beckmann without it).
"""

import numpy as np
import pytest
from conftest import load_or_skip

from tabench import (
    Budget,
    Demand,
    Evaluator,
    LinkBasedDTDModel,
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
    (model or LinkBasedDTDModel()).solve(
        scenario, Budget(**budget_kwargs), RngBundle(0), trace
    )
    return trace


# ------------------------------------------------------------- convergence
def test_converges_to_braess_ue(braess):
    """The link-flow dynamics settle on the exact Wardrop UE (route flows are
    non-unique, but the link flows are)."""
    trace = _solve(braess, iterations=800, target_relative_gap=1e-8)
    metrics = Evaluator(braess).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-7
    np.testing.assert_allclose(trace.final.link_flows, REF_FLOWS, atol=1e-4)


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


# ------------------------------------------------------- Lyapunov / stability
def test_beckmann_is_a_monotone_lyapunov_function(braess):
    """The distinctive day-to-day property: along the link-based dynamics the
    Beckmann objective decreases monotonically to the UE value (He et al.'s
    Lyapunov argument). The Armijo backtracking on Z is what guarantees it."""
    trace = _solve(braess, iterations=400)
    beckmann = [s.self_report["beckmann"] for s in trace]
    assert all(beckmann[i] >= beckmann[i + 1] - 1e-9 for i in range(len(beckmann) - 1))
    # ... and it lands on the same Beckmann value a UE solver reaches.
    assert beckmann[-1] == pytest.approx(386.0, abs=1e-2)


def test_beckmann_monotone_on_siouxfalls(siouxfalls):
    """Monotone Lyapunov descent holds on a congested multi-OD network too (the
    Lipschitz-normalized step keeps the projected target well-scaled as flows
    load up)."""
    trace = _solve(siouxfalls, iterations=300)
    beckmann = [s.self_report["beckmann"] for s in trace]
    assert all(
        beckmann[i] >= beckmann[i + 1] - 1e-6 * abs(beckmann[i])
        for i in range(len(beckmann) - 1)
    )


# --------------------------------------------- distinctive: link-based operation
def test_link_flows_stay_in_omega_every_day(braess, siouxfalls):
    """INVARIANCE (He et al. 2010): because the state is the LINK-flow vector
    moving between feasible points of the OD polytope Omega, the emitted flows
    conserve the exact OD demand at every intersection on EVERY recorded day --
    not only at convergence. This is the link-based signature the certificate
    exposes (dtd-swap conserves structurally via route flows; dtd-link conserves
    by staying inside the link-flow polytope)."""
    for scenario in (braess, siouxfalls):
        trace = _solve(scenario, iterations=50)
        evaluator = Evaluator(scenario)
        tol = 1e-6 * scenario.demand.total
        for state in trace:
            assert evaluator.evaluate(state.link_flows)["node_balance_residual"] <= tol


def test_link_target_and_route_swap_reach_same_ue(braess):
    """CONTRAST (the paradigm difference): the link-target projection (dtd-link)
    and the route-swap dynamics (dtd-swap) start from the IDENTICAL free-flow
    all-or-nothing state and reach the IDENTICAL certified UE link flows -- the
    unique Braess equilibrium -- isolating the difference to the adjustment
    mechanism (link-space projection vs per-OD route swaps), not the fixed
    point."""
    link = _solve(braess, iterations=800, target_relative_gap=1e-9).final.link_flows
    swap = _solve(
        braess, RouteSwapDTDModel(), iterations=800, target_relative_gap=1e-9
    ).final.link_flows
    np.testing.assert_allclose(link, REF_FLOWS, atol=1e-4)
    np.testing.assert_allclose(swap, REF_FLOWS, atol=1e-4)
    np.testing.assert_allclose(link, swap, atol=1e-4)


def _high_curvature_scenario():
    """A congested all-BPR-power-4 multi-OD network (shared with the dtd-swap
    suite) on which an aggressive projected step overshoots and Beckmann rises;
    the Armijo backtracking must restore monotone descent. Hardcoded for
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
    """REGRESSION for the step-overshoot fix. The unit proximal metric ignores the
    true BPR cost curvature, so an AGGRESSIVE step (step_size=4) overshoots the
    Beckmann minimum along the projected ray on high-curvature (power-4) congested
    links and the Beckmann Lyapunov function RISES mid-run. Disabling backtracking
    (max_backtracks=0) reproduces the rise; the default Armijo backtracking on Z
    restores strict monotone descent and still converges."""
    sc = _high_curvature_scenario()

    def max_rel_rise(model):
        trace = Trace()
        model.solve(sc, Budget(iterations=60), RngBundle(0), trace)
        beck = [s.self_report["beckmann"] for s in trace]
        return max(
            (beck[i + 1] - beck[i]) / max(abs(beck[i]), 1.0) for i in range(len(beck) - 1)
        )

    # Aggressive raw step (no backtracking) overshoots: Beckmann rises materially.
    assert max_rel_rise(LinkBasedDTDModel(step_size=4.0, max_backtracks=0)) > 1e-3
    # Default (Armijo backtracking) is strictly monotone non-increasing.
    assert max_rel_rise(LinkBasedDTDModel(step_size=4.0)) < 1e-9
    # ... and it still converges to the UE.
    trace = Trace()
    LinkBasedDTDModel(step_size=4.0).solve(
        sc, Budget(iterations=1500, target_relative_gap=1e-4), RngBundle(0), trace
    )
    assert Evaluator(sc).evaluate(trace.final.link_flows)["relative_gap"] < 1e-3


# -------------------------------------------------------------- mechanics
def test_paradigm_and_registry():
    from tabench.models import MODEL_REGISTRY

    assert "dtd-link" in MODEL_REGISTRY
    assert MODEL_REGISTRY["dtd-link"]().capabilities.paradigm == "day_to_day"


def test_bookkeeping_and_conservation(braess):
    trace = _solve(braess, iterations=10)
    assert len(trace) == 10
    # One Dijkstra at init + one per recorded day.
    assert trace.final.coords.sp_calls == 10 + 1
    v = trace.final.link_flows
    assert np.all(v >= 0)
    metrics = Evaluator(braess).evaluate(v)
    # Link flows stay in the OD-feasible polytope: node balance at the noise floor.
    assert metrics["node_balance_residual"] <= 1e-6 * braess.demand.total
    for key in ("relative_gap", "tstt", "sptt", "beckmann"):
        assert key in trace.final.self_report
