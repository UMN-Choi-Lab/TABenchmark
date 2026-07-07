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


def _overlapping_paths_stall_scenario():
    """A congested power-4 *trellis*: L=4 transitions between layers of W=3 nodes,
    every node in a layer feeding every node in the next; layer-0 -> layer-4 ODs
    each have many heavily-OVERLAPPING multi-hop paths. This is the instance that
    exposes the inner-solve staleness bug: within one proximal sweep several paths
    shift onto the same basic path, so a per-OD-cached basic cost goes stale and
    later shifts overshoot the 1-D minimizer, flipping the aggregate target into a
    Beckmann ascent -- the outer Armijo then collapses the day step and the
    certified gap stalls ~1e-2, far above UE. Hardcoded for determinism (topology
    from the layered pattern; caps/ffts/demands pinned)."""
    W, L = 3, 4
    init, term = [], []
    for layer in range(L):
        for a in range(W):
            for b in range(W):
                init.append(layer * W + a + 1)
                term.append((layer + 1) * W + b + 1)
    init = np.array(init, dtype=np.int64)
    term = np.array(term, dtype=np.int64)
    m = len(init)
    cap = np.array([
        1.0118, 1.4505, 0.6442, 1.4486, 0.8118, 0.9233, 1.3277, 0.9092, 1.0496,
        0.5276, 1.2535, 1.0381, 0.8297, 1.2884, 0.8032, 0.9535, 0.634, 0.9031,
        0.7035, 0.7623, 1.2504, 0.7804, 0.9852, 1.4807, 1.4617, 1.2248, 1.0412,
        0.7769, 0.6607, 1.4699, 1.0161, 0.6159, 1.1235, 1.2767, 1.113, 1.4173,
    ])
    fft = np.array([
        1.0792, 2.0572, 1.9187, 1.1247, 2.2827, 2.7053, 2.1859, 1.5202, 2.6798,
        2.019, 2.0218, 2.5061, 1.2958, 2.6393, 2.3666, 2.5742, 1.3832, 2.6047,
        1.3826, 1.1631, 2.7105, 2.7226, 2.7531, 1.9438, 1.5481, 1.0142, 2.2914,
        2.4398, 2.6711, 1.5638, 1.4304, 2.2787, 2.6101, 2.9273, 1.301, 1.9644,
    ])
    net = Network(
        name="trellis-dtd", n_nodes=(L + 1) * W, n_zones=(L + 1) * W,
        first_thru_node=1, init_node=init, term_node=term, capacity=cap,
        length=np.zeros(m), free_flow_time=fft, b=np.full(m, 0.15),
        power=np.full(m, 4.0), toll=np.zeros(m),
        link_type=np.ones(m, dtype=np.int64),
    )
    od = np.zeros(((L + 1) * W, (L + 1) * W))
    for (i, j), val in {
        (0, 12): 9.8946, (0, 13): 7.629, (0, 14): 8.4296, (1, 12): 5.7176,
        (1, 13): 8.8326, (1, 14): 10.0116, (2, 12): 9.5688, (2, 13): 9.8505,
        (2, 14): 8.7697,
    }.items():
        od[i, j] = val
    return Scenario("trellis-dtd", net, Demand(od))


def test_default_config_reaches_ue_on_overlapping_high_curvature():
    """REGRESSION for the stale inner-solve fix (adversarial-review MAJOR). The
    per-day proximal target x*(v) is solved by per-OD pairwise flow shifts toward
    the cheapest working path; the proximal path costs MUST be recomputed from the
    live target before EACH shift (true Gauss-Seidel). Caching them once per sweep
    let later shifts onto the same basic path overshoot the 1-D minimizer, so on
    congested high-curvature (power-4) instances with many overlapping paths the
    aggregate target became a Beckmann ascent, the Armijo collapsed the day step,
    and the DEFAULT-config certified gap stalled ~1e-2 -- never reaching UE even
    with far more iterations. With the fix the default config converges deep below
    UE. Certified purely from the emitted link flows (P1)."""
    sc = _overlapping_paths_stall_scenario()
    evaluator = Evaluator(sc)
    trace = _solve(sc, iterations=800, target_relative_gap=1e-8)
    gaps = [s.self_report["relative_gap"] for s in trace]
    metrics = evaluator.evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    # Old (stale-cost) inner solve plateaued ~9e-3 here; the fix reaches deep UE.
    assert min(gaps) < 1e-6
    assert metrics["relative_gap"] < 1e-6
    # Conservation stays at the noise floor throughout (link flows never leave Omega).
    tol = 1e-6 * sc.demand.total
    for state in trace:
        assert evaluator.evaluate(state.link_flows)["node_balance_residual"] <= tol


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
