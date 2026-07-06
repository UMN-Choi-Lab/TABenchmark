"""Tests for Smith & Watling (2016) route-swap SUE dynamics (dtd-swap-sue).

The SUE sibling of ``dtd-swap``: the SAME proportional route-swap day-to-day
process, but swaps are driven by the FISK-GENERALIZED cost ``C_k = c_k +
(1/theta) ln h_k`` instead of the raw travel time, so the rest point is the
logit stochastic user equilibrium (Fisk 1980), not deterministic Wardrop UE. It
is validated as an SUE model -- it converges to the analytic logit-SUE fixed
point of the two-route anchor and self-reports the SAME Dial-STOCH certificate
the harness recomputes (P1) -- PLUS its distinctive dynamical signature: Fisk's
convex objective is a Lyapunov function that decreases monotonically to the SUE
minimum (``Fdot = -a V``), and the generalized-cost disequilibrium ``V(h)``
vanishes at equilibrium (Smith & Watling 2016). On the anchor the swap rest
point coincides with the ``sue-msa`` / Dial-STOCH fixed point (cross-solver
check); as ``theta -> infinity`` it collapses to the deterministic Wardrop UE.
"""

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from tabench import (
    Budget,
    Demand,
    DialSUEModel,
    Evaluator,
    Network,
    RngBundle,
    RouteSwapSUEModel,
    Scenario,
    Trace,
    braess_scenario,
    two_route_scenario,
)
from tabench.models._paths import PathEngine
from tabench.models._stoch import StochEngine

# Golden content hash of the Braess scenario, unchanged: this model adds no
# scenario field, so every existing content hash must stay byte-identical.
BRAESS_GOLDEN_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"


def _fixed_point_route_a(theta: float, demand: float = 4.0) -> float:
    """Root of the binary-logit fixed point ``f_A = D / (1 + exp(theta (c_A -
    c_B)))`` on the two-route anchor -- the same scalar equation the swap rest
    point equalizes (``C_A = C_B``). Recomputed here, never a trusted digit."""

    def residual(f_a: float) -> float:
        c_a = 2.0 + f_a
        c_b = 1.5 + 2.0 * (demand - f_a)
        return f_a - demand / (1.0 + math.exp(theta * (c_a - c_b)))

    return brentq(residual, 0.0, demand, xtol=1e-12)


@pytest.fixture(scope="module")
def scenario():
    return two_route_scenario()  # demand 4, theta 0.5, logit


def _solve(sc, model=None, **budget_kwargs):
    trace = Trace()
    (model or RouteSwapSUEModel()).solve(
        sc, Budget(**budget_kwargs), RngBundle(0), trace
    )
    return trace


def _braess_sue(theta: float = 0.1) -> Scenario:
    """A multi-path logit-SUE task on the Braess network (three OD paths, so the
    working set genuinely grows) built from the shipped Braess network."""
    b = braess_scenario()
    return Scenario(
        name="braess-sue",
        network=b.network,
        demand=b.demand,
        sue_family="logit",
        sue_theta=theta,
    )


def _three_disjoint_sue() -> Scenario:
    """Three LINK-DISJOINT 2-link routes 1->mid_k->2 on one OD, tuned so route 0
    (links 0,1) is Dial-efficient with a substantial logit share yet is NEVER the
    strict shortest path: routes 1,2 are cheaper at every flow, so one-shortest-
    path-per-day column generation never generates route 0 and never loads it,
    while Dial does (adversarial-review MAJOR). Route 0's second leg is nearly
    flat (large capacity); routes 1,2 congest (BPR power 4) up to -- but not past
    -- route 0's cost, so route 0 carries ~5% of demand at the logit SUE."""
    # Link order per route k: (1->mid_k, mid_k->2)
    init = np.array([1, 3, 1, 4, 1, 5], dtype=np.int64)
    term = np.array([3, 2, 4, 2, 5, 2], dtype=np.int64)
    fft = np.array([1.0, 2.2, 1.0, 1.0, 1.0, 1.0])
    b = np.array([0.0, 0.15, 0.0, 0.15, 0.0, 0.15])
    cap = np.array([1.0, 20.0, 1.0, 4.0, 1.0, 4.0])
    power = np.array([1.0, 4.0, 1.0, 4.0, 1.0, 4.0])
    net = Network(
        name="three-disjoint-sue",
        n_nodes=5,
        n_zones=2,
        first_thru_node=1,
        init_node=init,
        term_node=term,
        capacity=cap,
        length=np.zeros(6),
        free_flow_time=fft,
        b=b,
        power=power,
        toll=np.zeros(6),
        link_type=np.ones(6, dtype=np.int64),
    )
    od = np.zeros((2, 2))
    od[0, 1] = 7.0
    return Scenario(
        name="three-disjoint-sue",
        network=net,
        demand=Demand(od),
        sue_family="logit",
        sue_theta=2.0,
    )


def _grid_sue(n: int = 4, theta: float = 0.5, demand: float = 5.0) -> Scenario:
    """A single-OD ``n x n`` grid SUE task with OVERLAPPING routes: corner-to-
    opposite-corner demand over a bidirectional grid, so most Dial-efficient
    paths are never a shortest path (they are enumerated only by the full
    efficient-set column generation). Deterministic free-flow times."""
    def node(r: int, c: int) -> int:
        return r * n + c + 1

    init: list[int] = []
    term: list[int] = []
    for r in range(n):
        for c in range(n):
            if c + 1 < n:
                init += [node(r, c), node(r, c + 1)]
                term += [node(r, c + 1), node(r, c)]
            if r + 1 < n:
                init += [node(r, c), node(r + 1, c)]
                term += [node(r + 1, c), node(r, c)]
    nl = len(init)
    fft = 1.0 + np.random.default_rng(0).uniform(0.0, 1.0, nl)  # deterministic
    net = Network(
        name="grid-sue",
        n_nodes=n * n,
        n_zones=n * n,
        first_thru_node=1,
        init_node=np.array(init, dtype=np.int64),
        term_node=np.array(term, dtype=np.int64),
        capacity=np.full(nl, 2.0),
        length=np.zeros(nl),
        free_flow_time=fft,
        b=np.full(nl, 0.15),
        power=np.full(nl, 4.0),
        toll=np.zeros(nl),
        link_type=np.ones(nl, dtype=np.int64),
    )
    od = np.zeros((n * n, n * n))
    od[0, n * n - 1] = demand
    return Scenario(
        name="grid-sue",
        network=net,
        demand=Demand(od),
        sue_family="logit",
        sue_theta=theta,
    )


# ------------------------------------------------------------- convergence
def test_converges_to_logit_sue_fixed_point(scenario):
    """The generalized-cost route-swap settles on the analytic binary-logit SUE
    (route flows are non-unique, but the link flows are), and self-reports the
    certified residual that -> 0 there."""
    f_a = _fixed_point_route_a(theta=0.5)
    trace = _solve(scenario, iterations=500, target_relative_gap=1e-8)
    expected = np.array([f_a, f_a, 4.0 - f_a, 4.0 - f_a])
    np.testing.assert_allclose(trace.final.link_flows, expected, atol=1e-3)
    metrics = Evaluator(scenario).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["sue_fixed_point_residual"] < 1e-5
    # The logit SUE is NOT the deterministic UE: the UE gap stays strictly
    # positive as a descriptive column (like sue-msa).
    assert metrics["relative_gap"] > 0.01


def test_converges_on_multipath_braess_sue():
    """On a genuinely multi-path network (three Braess routes, so column
    generation grows the working set) the swap still drives the certified SUE
    residual toward zero and stays demand-feasible at every day."""
    sc = _braess_sue(theta=0.1)
    trace = _solve(sc, iterations=1000, target_relative_gap=1e-8)
    metrics = Evaluator(sc).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["node_balance_residual"] <= 1e-6 * sc.demand.total
    assert metrics["sue_fixed_point_residual"] < 1e-5


# --------------------------------------------- full-efficient-set column gen (P1)
def test_efficient_paths_enumerates_full_dial_support():
    """The mechanism behind the fix: ``PathEngine.efficient_paths`` returns the
    WHOLE Dial-efficient route set, not one shortest path, and the multinomial
    logit over that set reproduces ``StochEngine.load`` (Dial) to float precision
    -- the identity that lets the path-flow swap certify the Dial residual. On an
    overlapping grid the efficient set has many routes though only one is the
    shortest path, so one-shortest-path-per-day column generation cannot supply
    the SUE support (adversarial-review MAJOR)."""
    sc = _grid_sue()
    net = sc.network
    engine = PathEngine(net)
    stoch = StochEngine(net)
    key = (0, net.n_nodes - 1)
    costs = net.link_cost(np.random.default_rng(7).uniform(0.0, 3.0, net.n_links))

    efficient = engine.efficient_paths(costs, sc.demand)[key]
    shortest, _ = engine.shortest_paths(costs, sc.demand)
    assert len(efficient) > 1  # the full efficient set, not the single SP
    # ... yet single-shortest-path column generation would supply exactly one.
    assert not isinstance(shortest[key], list)

    route_costs = np.array([costs[p].sum() for p in efficient])
    weights = np.exp(-sc.sue_theta * (route_costs - route_costs.min()))
    shares = weights / weights.sum()
    logit = np.zeros(net.n_links)
    for links, share in zip(efficient, sc.demand.total * shares, strict=True):
        logit[links] += share
    np.testing.assert_allclose(logit, stoch.load(costs, sc.demand, sc.sue_theta), atol=1e-10)


def test_disjoint_efficient_route_never_shortest_is_generated_and_loaded():
    """Fix for the K>=3 link-disjoint defect: an efficient route that is never
    the strict shortest path (route 0) was never column-generated and never
    loaded, stranding the certified residual at O(0.2); enumerating the full
    efficient set now loads it, so the residual drives to ~0 and the link flows
    match ``sue-msa`` (both fixed points of the same Dial map). Directly refutes
    docstring claim (i)."""
    sc = _three_disjoint_sue()
    evaluator = Evaluator(sc)
    swap = _solve(sc, iterations=3000, target_relative_gap=1e-8)
    msa = _solve(sc, DialSUEModel(), iterations=1500, target_relative_gap=1e-9)

    metrics = evaluator.evaluate(swap.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["sue_fixed_point_residual"] < 1e-5
    # Route 0 (link 0) carries its Dial logit share -- it is no longer stranded
    # at zero the way one-shortest-path column generation left it.
    assert swap.final.link_flows[0] > 0.1
    np.testing.assert_allclose(
        swap.final.link_flows, msa.final.link_flows, atol=1e-3
    )


def test_overlapping_grid_reaches_dial_fixed_point():
    """Fix for the overlapping-network defect: on a single-OD grid whose
    efficient routes overlap and are mostly never shortest paths, the full
    efficient-set column generation drives the certified residual to ~0 and the
    link flows to the ``sue-msa`` / Dial fixed point -- NOT a plateau. Directly
    refutes docstring claim (ii) (the false 'analogous to dtd-swap's UE gap not
    reaching a tight value')."""
    sc = _grid_sue()
    evaluator = Evaluator(sc)
    swap = _solve(sc, iterations=3000, target_relative_gap=1e-8)
    msa = _solve(sc, DialSUEModel(), iterations=1500, target_relative_gap=1e-9)

    metrics = evaluator.evaluate(swap.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["node_balance_residual"] <= 1e-6 * sc.demand.total
    assert metrics["sue_fixed_point_residual"] < 1e-5
    np.testing.assert_allclose(
        swap.final.link_flows, msa.final.link_flows, atol=1e-3
    )


# ------------------------------------------------------- Lyapunov / stability
def test_fisk_is_a_monotone_lyapunov_function(scenario):
    """The distinctive day-to-day property: along the C-swap Fisk's SUE convex
    objective F = Beckmann(v) + (1/theta) sum h(ln h - 1) decreases monotonically
    to the logit-SUE minimum (Fdot = -a V <= 0). The terminal value is recomputed
    from the analytic fixed point, never trusted as a digit."""
    trace = _solve(scenario, iterations=400)
    fisk = [s.self_report["fisk_objective"] for s in trace]
    assert all(
        fisk[i] >= fisk[i + 1] - 1e-9 * max(abs(fisk[i]), 1.0)
        for i in range(len(fisk) - 1)
    )
    # ... and it lands on the analytic Fisk minimum at the logit-SUE fixed point.
    f_a = _fixed_point_route_a(theta=0.5)
    f_b = 4.0 - f_a
    v_star = np.array([f_a, f_a, f_b, f_b])
    beckmann = float(scenario.network.link_cost_integral(v_star).sum())
    entropy = (f_a * (math.log(f_a) - 1.0) + f_b * (math.log(f_b) - 1.0)) / 0.5
    assert fisk[-1] == pytest.approx(beckmann + entropy, abs=1e-6)


def test_sue_disequilibrium_vanishes_at_equilibrium(scenario):
    """The generalized-cost disequilibrium V(h) = sum h_p ([C_p - C_k]+)^2 is a
    Lyapunov measure that is zero iff logit SUE; it is strictly positive off
    equilibrium (once >1 route exists) and collapses to zero as the dynamics
    converge (reported as provenance, never scored)."""
    trace = _solve(scenario, iterations=500, target_relative_gap=1e-9)
    diseq = [s.self_report["sue_disequilibrium"] for s in trace]
    assert diseq[-1] < 1e-6
    assert max(diseq) > 1.0
    assert diseq[-1] < max(diseq)


# ------------------------------------------------------------- honesty (P1)
def test_self_report_matches_harness_certificate(scenario):
    """P1 honesty: the model's self-reported residual equals the one the harness
    recomputes -- both call the SAME pinned StochEngine.load, so they agree to
    float precision at every checkpoint."""
    trace = _solve(scenario, iterations=50)
    evaluator = Evaluator(scenario)
    for state in list(trace)[::10]:
        certified = evaluator.evaluate(state.link_flows)["sue_fixed_point_residual"]
        assert certified == pytest.approx(
            state.self_report["sue_fixed_point_residual"], rel=1e-9, abs=1e-15
        )


# ------------------------------------------------------------- theta / limits
def test_theta_large_approaches_ue():
    """As theta grows the logit-SUE rest point -> the deterministic Wardrop UE
    (f_A = 2.5 at D = 4) and the certified UE relative gap -> 0 -- the swap's
    entropy term (1/theta) ln h vanishes, recovering dtd-swap's dynamics."""
    f_a = [_fixed_point_route_a(theta) for theta in (0.5, 2.0, 5.0, 50.0)]
    assert all(abs(b - 2.5) < abs(a - 2.5) for a, b in zip(f_a, f_a[1:], strict=False))

    stiff = two_route_scenario(sue_theta=50.0)
    trace = _solve(stiff, iterations=3000, target_relative_gap=1e-8)
    assert abs(trace.final.link_flows[0] - 2.5) < 0.01
    assert Evaluator(stiff).evaluate(trace.final.link_flows)["relative_gap"] < 1e-3


# ---------------------------------------------------------------- cross-solver
def test_cross_solver_agrees_with_sue_msa(scenario):
    """sue-swap (day-to-day dynamics) and sue-msa (the MSA solver) reach the SAME
    certified logit-SUE link flows on the anchor -- both fixed points of the same
    Dial-STOCH map."""
    swap = _solve(scenario, iterations=500, target_relative_gap=1e-9)
    msa = _solve(scenario, DialSUEModel(), iterations=500, target_relative_gap=1e-9)
    np.testing.assert_allclose(
        swap.final.link_flows, msa.final.link_flows, atol=1e-4
    )


# --------------------------------------------------------------------- guards
def test_requires_sue_scenario():
    """A deterministic (non-SUE) scenario has no theta: refuse it (theta is task
    data, not a model factor)."""
    with pytest.raises(ValueError, match="sue_theta|SUE scenario"):
        RouteSwapSUEModel().solve(
            braess_scenario(), Budget(iterations=5), RngBundle(0), Trace()
        )


def test_rejects_probit_scenario():
    """The logit route-swap must refuse a probit-SUE task and point at the probit
    solver."""
    probit = two_route_scenario(sue_theta=0.1, sue_family="probit")
    with pytest.raises(ValueError, match="probit"):
        RouteSwapSUEModel().solve(probit, Budget(iterations=5), RngBundle(0), Trace())


# ------------------------------------------------------------------- mechanics
def test_paradigm_and_registry():
    from tabench.models import MODEL_REGISTRY

    assert "dtd-swap-sue" in MODEL_REGISTRY
    assert MODEL_REGISTRY["dtd-swap-sue"]().capabilities.paradigm == "day_to_day"


def test_bookkeeping_and_conservation(scenario):
    trace = _solve(scenario, iterations=10)
    assert len(trace) == 10
    # Per day: one Dijkstra (column generation) + one Dial-STOCH load (residual),
    # both counted in sp_calls, plus the day-0 free-flow Dijkstra -- but the day-k
    # column-generation Dijkstra runs AFTER the day-k checkpoint, so at k days the
    # count is 1 (day 0) + k (Dial loads) + (k-1) (column Dijkstras) = 2k.
    assert trace.final.coords.sp_calls == 20
    v = trace.final.link_flows
    assert np.all(v >= 0)
    metrics = Evaluator(scenario).evaluate(v)
    # Renormalization conserves each OD's demand exactly, so link flows rebuilt
    # from route flows balance to the float-noise floor at every day.
    assert metrics["node_balance_residual"] <= 1e-6 * scenario.demand.total
    for key in ("sue_fixed_point_residual", "fisk_objective", "sue_disequilibrium"):
        assert key in trace.final.self_report


def test_braess_content_hash_preserved():
    """This model adds no scenario field: the golden Braess content hash must be
    byte-identical."""
    assert braess_scenario().content_hash() == BRAESS_GOLDEN_HASH
