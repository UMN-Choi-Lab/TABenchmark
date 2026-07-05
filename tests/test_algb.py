"""Tests for Dial's Algorithm B bush-based user equilibrium (algb).

Invariants mirror the gradient-projection suite plus Algorithm B's own
guarantees: analytic Braess UE; deep certified convergence on Sioux Falls in
far fewer iterations than gradient projection; the zero-derivative full-shift
fallback (certified in one iteration); restricted-centroid handling on Anaheim
through the bush's expanded-graph structure; monotone Beckmann descent (a
measured property, not a theoretical guarantee); the design's strict per-round
sp_calls accounting with its separate ``bush_scan_rounds`` self-report; and
acyclicity under float-exact shortest-path ties (the used-or-SP eligibility
rule the auditor's correction pins).
"""

import dataclasses

import numpy as np
import pytest
from conftest import load_or_skip

from tabench import (
    AlgorithmBModel,
    Budget,
    Demand,
    Evaluator,
    GradientProjectionModel,
    Network,
    RngBundle,
    Scenario,
    Trace,
    braess_scenario,
)
from tabench.models.algb import _BushState

REF_FLOWS = np.array([4.0, 2.0, 2.0, 2.0, 4.0])


@pytest.fixture(scope="module")
def braess():
    return braess_scenario()


@pytest.fixture(scope="module")
def siouxfalls():
    return load_or_skip("siouxfalls")


def _solve(scenario, **budget_kwargs):
    trace = Trace()
    AlgorithmBModel().solve(scenario, Budget(**budget_kwargs), RngBundle(0), trace)
    return trace


def _grid_scenario(rows=3, cols=3, demand=10.0):
    """Symmetric bidirectional grid whose corner-to-corner demand ties exactly.

    Uniform links give multiple bitwise-equal shortest paths between opposite
    corners, so bush construction and updates must retain non-argmin tie links
    without breaking acyclicity — the case the used-or-SP eligibility rule
    guards.
    """

    def nid(r, c):
        return r * cols + c + 1

    init: list[int] = []
    term: list[int] = []
    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((0, 1), (1, 0)):
                rr, cc = r + dr, c + dc
                if rr < rows and cc < cols:
                    init += [nid(r, c), nid(rr, cc)]
                    term += [nid(rr, cc), nid(r, c)]
    init_node = np.asarray(init, dtype=np.int64)
    term_node = np.asarray(term, dtype=np.int64)
    m = len(init_node)
    n = rows * cols
    network = Network(
        name="grid",
        n_nodes=n,
        n_zones=n,
        first_thru_node=1,
        init_node=init_node,
        term_node=term_node,
        capacity=np.ones(m),
        length=np.ones(m),
        free_flow_time=np.ones(m),
        b=np.full(m, 0.15),
        power=np.full(m, 4.0),
        toll=np.zeros(m),
        link_type=np.ones(m, dtype=np.int64),
    )
    od = np.zeros((n, n))
    corners = ((nid(0, 0), nid(rows - 1, cols - 1)), (nid(0, cols - 1), nid(rows - 1, 0)))
    for a, z in corners:
        od[a - 1, z - 1] = demand
        od[z - 1, a - 1] = demand
    return Scenario(name="grid", network=network, demand=Demand(od))


def test_analytic_braess_equilibrium(braess):
    trace = _solve(braess, iterations=10, target_relative_gap=1e-14)
    metrics = Evaluator(braess).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert abs(metrics["relative_gap"]) < 1e-10
    np.testing.assert_allclose(trace.final.link_flows, REF_FLOWS, atol=1e-5)


def test_deep_convergence_on_siouxfalls(siouxfalls):
    """Certified gap < 1e-10 within 25 iterations (measured 18)."""
    trace = _solve(siouxfalls, iterations=25, target_relative_gap=1e-10)
    metrics = Evaluator(siouxfalls).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-10
    assert trace.final.coords.iterations < 25  # converged before the cap
    assert np.abs(trace.final.link_flows - siouxfalls.reference.link_flows).max() < 1e-3


def test_far_fewer_iterations_than_gp(siouxfalls):
    """Algorithm B reaches a certified 1e-8 gap in ~4x fewer iterations."""
    algb = _solve(siouxfalls, iterations=3000, target_relative_gap=1e-8)
    gp = Trace()
    GradientProjectionModel().solve(
        siouxfalls, Budget(iterations=3000, target_relative_gap=1e-8), RngBundle(0), gp
    )
    algb_iters = algb.final.coords.iterations
    gp_iters = gp.final.coords.iterations
    assert algb_iters < 25
    assert algb_iters < 0.3 * gp_iters


def test_monotone_beckmann_descent(siouxfalls):
    # Measured monotone on Sioux Falls; Algorithm B does not line-search the
    # aggregate Beckmann objective, so this is an empirical regression, not a
    # theoretical guarantee (design note in the module docstring).
    trace = _solve(siouxfalls, iterations=28)
    objectives = [s.self_report["beckmann"] for s in trace]
    pairs = zip(objectives, objectives[1:], strict=False)
    assert all(b2 <= b1 + 1e-10 * abs(b1) for b1, b2 in pairs)


def test_zero_derivative_certifies_in_one_iteration(braess):
    """All-constant costs: the re-probe full-shift lands on AON = UE at once."""
    net = braess.network
    constant = dataclasses.replace(
        net,
        b=np.zeros(net.n_links),
        free_flow_time=np.array([10.0, 50.0, 10.0, 50.0, 10.0]),
    )
    scenario = dataclasses.replace(braess, network=constant, reference=None)
    trace = _solve(scenario, iterations=1)
    metrics = Evaluator(scenario).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert abs(metrics["relative_gap"]) < 1e-12


def test_bookkeeping_invariants(siouxfalls):
    """Nonnegative flows and demand conservation at the emitted checkpoints."""
    trace = _solve(siouxfalls, iterations=20)
    v = trace.final.link_flows
    metrics = Evaluator(siouxfalls).evaluate(v)
    assert metrics["feasible"] == 1.0
    assert np.all(v >= 0)
    # Link flows are rebuilt exactly from the sum of bush flows before every
    # checkpoint, so the balance residual stays at the float-noise floor.
    assert metrics["node_balance_residual"] <= 1e-6 * siouxfalls.demand.total


def test_budget_accounting(braess):
    trace = _solve(braess, iterations=7)
    assert len(trace) == 7
    assert trace.final.coords.iterations == 7
    first = trace.checkpoints[0]
    # Strict per-round accounting: init Dijkstra + one bush-update round + ten
    # shift rounds + the honest-gap AON = 13 sp_calls at the first iteration.
    assert first.coords.sp_calls == 13
    assert first.self_report["bush_scan_rounds"] == 11.0


def test_restricted_centroids_honored():
    """Anaheim (first_thru_node=39): bushes never route through centroids."""
    scenario = load_or_skip("anaheim")
    trace = _solve(scenario, iterations=25, target_relative_gap=1e-10)
    metrics = Evaluator(scenario).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-8


def test_acyclicity_under_exact_shortest_path_ties():
    """Float-exact SP ties must not break the bush acyclicity invariant.

    A clean certified solve on this tie-heavy symmetric grid is an end-to-end
    smoke check (the hard Kahn assert raises on any cyclic bush). It does not,
    on its own, exercise the used-or-SP fix — the state that distinguishes the
    rules is pinned white-box below.
    """
    scenario = _grid_scenario()
    trace = _solve(scenario, iterations=40, target_relative_gap=1e-12)
    metrics = Evaluator(scenario).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-8


def _argmin_only_max_labels(model, bush, origin_idx, t):
    """Pre-audit max labels: among zero-flow in-links, only the single argmin
    predecessor is U-eligible (the bug the used-or-SP rule fixed)."""
    neg = -np.inf
    n = model._n_exp
    labels_l = np.full(n, np.inf)
    labels_u = np.full(n, neg)
    labels_l[origin_idx] = 0.0
    labels_u[origin_idx] = 0.0
    eps = model._drop_tol
    for j in bush.topo:
        if j == origin_idx:
            continue
        best_l, best_lk = np.inf, -1
        for k in model._in_links[j]:
            if not bush.in_bush[k]:
                continue
            cand = labels_l[model._tails[k]] + t[k]
            if cand < best_l:
                best_l, best_lk = cand, k
        labels_l[j] = best_l
        best_u = neg
        for k in model._in_links[j]:
            if not bush.in_bush[k]:
                continue
            if not (bush.x[k] > eps or k == best_lk):
                continue
            ut = labels_u[model._tails[k]]
            if ut == neg:
                continue
            best_u = max(best_u, ut + t[k])
        labels_u[j] = best_u
    return labels_u


def test_used_or_sp_eligibility_pins_potential_invariant():
    """White-box regression for the auditor's acyclicity correction.

    The used-or-SP rule makes every link the drop rule can retain (positive
    flow OR an exact-min-label tie) U-eligible, so U strictly increases along
    each retained link — the potential the bush-add criterion relies on to stay
    acyclic. A full solve does not reach the distinguishing state (the grid
    smoke passes even under the pre-audit rule), so this pins it directly on a
    4-node witness: node 3 is reached both by the flow-carrying 2->3 and the
    zero-flow 4->3, tied at cost 3 from origin 1. The shipped rule gives
    U[node3]=5 >= U[node4]+t(4->3)=5; the argmin-only rule drops the tie link,
    giving U[node3]=3 < 5 — a violated potential the Kahn assert later trips on.
    """
    init = np.array([2, 1, 2, 4, 1], dtype=np.int64)  # 2->3, 1->2, 2->4, 4->3, 1->4
    term = np.array([3, 2, 4, 3, 4], dtype=np.int64)
    net = Network(
        name="tie-witness", n_nodes=4, n_zones=2, first_thru_node=1,
        init_node=init, term_node=term, capacity=np.ones(5), length=np.zeros(5),
        free_flow_time=np.array([1.0, 2.0, 1.0, 2.0, 1.0]),
        b=np.zeros(5), power=np.ones(5), toll=np.zeros(5),
        link_type=np.ones(5, dtype=np.int64),
        units=(("time", "abstract"), ("flow", "abstract")),
    )
    model = AlgorithmBModel()
    model._setup(
        Scenario(
            name="tie", network=net, demand=Demand(matrix=np.zeros((2, 2))),
            family="test",
        )
    )
    bush = _BushState(model._n_links, model._n_exp)
    bush.in_bush = np.ones(model._n_links, dtype=bool)
    bush.reachable = np.ones(model._n_exp, dtype=bool)
    bush.x = np.array([1.0, 0.0, 1.0, 0.0, 1.0])  # zero flow on 1->2 and 4->3
    model._kahn(bush)
    t = net.free_flow_time.copy()

    def violations(labels):
        return [
            int(k)
            for k in range(model._n_links)
            if labels[model._tails[k]] > -np.inf
            and labels[model._heads[k]] < labels[model._tails[k]] + t[k] - 1e-12
        ]

    _, u_fix, _, _ = model._scan(bush, 0, t, "used_or_sp")
    # Shipped rule keeps the potential intact...
    assert violations(u_fix) == []
    # ...and it is load-bearing: the pre-audit argmin-only labels violate it on
    # this exact state, so the test is not vacuous.
    assert violations(_argmin_only_max_labels(model, bush, 0, t)) != []
