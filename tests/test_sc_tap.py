"""Tests for side-constrained UE (sc-tap, Larsson & Patriksson 1995).

SC-TAP is UE under hard link-capacity constraints v_a <= u_a, solved by an
augmented-Lagrangian wrapper around Frank-Wolfe. It reduces EXACTLY to plain UE
when no capacity binds, and where a capacity binds the recovered multiplier
beta_a is the queueing delay / toll that equalizes augmented route costs. The
scored certificate is capacity feasibility (exact, link-visible); the raw-cost
gap is positive at a correct SC equilibrium and is not the acceptance criterion.
"""

import dataclasses

import numpy as np
import pytest

from tabench import (
    Budget,
    Demand,
    Evaluator,
    FrankWolfeModel,
    Network,
    RngBundle,
    Scenario,
    SideConstrainedModel,
    Trace,
    braess_scenario,
    sc_two_route_scenario,
)

GOLDEN_BRAESS_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"


def _solve(scenario, iterations=300):
    trace = Trace()
    SideConstrainedModel().solve(scenario, Budget(iterations=iterations), RngBundle(0), trace)
    return trace


# ---------------------------------------------------------------- limits
def test_no_binding_reduces_exactly_to_ue():
    """When every capacity is slack, SC-TAP is literally the unconstrained
    Beckmann program, so its link flows equal the shipped FW solver's (f_A=5.5)."""
    sc = sc_two_route_scenario(demand=10.0, cap=1e6)  # nothing binds
    v = _solve(sc).final.link_flows
    ue = Trace()
    FrankWolfeModel().solve(
        dataclasses.replace(sc, side_capacities=None),
        Budget(iterations=300, target_relative_gap=1e-12),
        RngBundle(0),
        ue,
    )
    np.testing.assert_allclose(v, [5.5, 5.5, 4.5, 4.5], atol=1e-3)
    np.testing.assert_allclose(v, ue.final.link_flows, atol=1e-3)


# ------------------------------------------------------------- binding anchor
def test_binding_capacity_anchor():
    """Capacity 4 on link 3->2 binds: hand-checked flows (4,4,6,6) and queueing
    multiplier beta = 1 + D - 2*cap = 3."""
    sc = sc_two_route_scenario(demand=10.0, cap=4.0)
    trace = _solve(sc)
    v = trace.final.link_flows
    metrics = Evaluator(sc).evaluate(v)
    np.testing.assert_allclose(v, [4.0, 4.0, 6.0, 6.0], atol=1e-4)
    assert trace.final.self_report["max_multiplier"] == pytest.approx(3.0, abs=1e-3)
    assert metrics["feasible"] == 1.0
    assert metrics["sc_capacity_feasible"] == 1.0
    assert metrics["max_capacity_violation"] <= 1e-6


def test_monotone_capacity_sweep():
    """Tightening the capacity pushes flow off the binding link (f_A down) and
    raises the multiplier (beta up) -- the closed form f_A=cap, beta=1+D-2cap."""
    fa, betas = [], []
    for cap in (5.0, 4.0, 3.0, 2.0):
        trace = _solve(sc_two_route_scenario(10.0, cap))
        fa.append(trace.final.link_flows[0])
        betas.append(trace.final.self_report["max_multiplier"])
    assert all(fa[i] >= fa[i + 1] - 1e-6 for i in range(len(fa) - 1))  # f_A non-increasing
    assert all(betas[i] <= betas[i + 1] + 1e-6 for i in range(len(betas) - 1))  # beta up
    # closed form beta = 1 + D - 2*cap at each cap
    for cap, beta in zip((5.0, 4.0, 3.0, 2.0), betas, strict=True):
        assert beta == pytest.approx(max(0.0, 1.0 + 10.0 - 2.0 * cap), abs=1e-2)


# --------------------------------------------------------- certificate teeth
def test_certificate_censors_capacity_violation():
    """A flow that exceeds a link capacity is not SC-feasible, even if it routes
    the demand -- the scored capacity check catches it."""
    sc = sc_two_route_scenario(demand=10.0, cap=4.0)
    ev = Evaluator(sc)
    # The plain-UE split (f_A=5.5) violates the cap-4 constraint on link 3->2.
    over = np.array([5.5, 5.5, 4.5, 4.5])
    m = ev.evaluate(over)
    assert m["sc_capacity_feasible"] == 0.0
    assert m["max_capacity_violation"] == pytest.approx(1.5, abs=1e-9)  # 5.5 - 4
    # The SC equilibrium flow is feasible.
    assert ev.evaluate(np.array([4.0, 4.0, 6.0, 6.0]))["sc_capacity_feasible"] == 1.0


def test_capacity_tolerance_is_per_link_not_demand_scaled():
    """REGRESSION (adversarial-review MAJOR 1): the capacity check is relative to
    each link's OWN capacity, not scaled by total demand. On a high-demand network
    a small absolute overload must still be censored -- a demand-scaled tolerance
    (feasibility_tol * total_demand) would silently pass it. Here total demand is
    1e6, link 3->2 is capped at 5e5, and an overload of 0.8 (rel 1.6e-6 > tol 1e-6)
    must be rejected -- yet 0.8 <= 1e-6 * 1e6 = 1.0 would have passed the old rule."""
    sc = sc_two_route_scenario(demand=1e6, cap=5e5)
    # routes the full 1e6; link 3->2 (index 1) is over its 5e5 cap by 0.8
    over = np.array([5e5 + 0.8, 5e5 + 0.8, 5e5 - 0.8, 5e5 - 0.8])
    m = Evaluator(sc).evaluate(over)
    assert m["feasible"] == 1.0  # routes the demand, conserves flow
    assert m["max_capacity_violation"] == pytest.approx(0.8, abs=1e-6)
    assert m["sc_capacity_feasible"] == 0.0  # per-link relative catches it


# ---------------------------------------------------------------- scenario
def test_side_capacities_hashed_only_when_set_and_golden_preserved():
    assert braess_scenario().content_hash() == GOLDEN_BRAESS_HASH
    sc = sc_two_route_scenario(10.0, 4.0)
    other = dataclasses.replace(sc, side_capacities=sc.side_capacities * 0.9)
    none = dataclasses.replace(sc, side_capacities=None)
    assert len({sc.content_hash(), other.content_hash(), none.content_hash()}) == 3


def test_side_capacities_validated_and_mutually_exclusive():
    sc = sc_two_route_scenario(10.0, 4.0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        dataclasses.replace(sc, br_epsilon=1.0)
    with pytest.raises(ValueError, match="must be finite and > 0"):
        dataclasses.replace(sc, side_capacities=np.array([1.0, 0.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="shape"):
        dataclasses.replace(sc, side_capacities=np.array([1.0, 2.0]))


def test_infeasible_capacity_is_reported_not_crashed():
    """A capacity below a link's FORCED flow (the link is the ONLY path for some OD
    demand, so no routing can avoid it) makes the instance infeasible -- SC-TAP has
    no solution. The augmented Lagrangian must NOT diverge/crash: the model caps its
    penalties and stops, and the certificate reports the constraint violated
    (sc_capacity_feasible=0). Here 1->3->2 is the only path for demand 10, capped at
    4 (forced flow 10 > 4). A fuzz confirmed sc-tap converges to feasibility on 100%
    of FEASIBLE instances and only reports infeasibility on genuinely-infeasible
    ones (min-cut-classified), never crashing on a solvable instance."""
    init = np.array([1, 3], dtype=np.int64)
    term = np.array([3, 2], dtype=np.int64)
    net = Network(
        name="series", n_nodes=3, n_zones=2, first_thru_node=1,
        init_node=init, term_node=term, capacity=np.ones(2), length=np.zeros(2),
        free_flow_time=np.array([1.0, 1.0]), b=np.full(2, 0.15), power=np.full(2, 4.0),
        toll=np.zeros(2), link_type=np.ones(2, dtype=np.int64),
    )
    od = np.zeros((2, 2))
    od[0, 1] = 10.0
    sc = Scenario("inf", net, Demand(od), side_capacities=np.array([1e6, 4.0]))
    trace = _solve(sc, iterations=100)  # must not crash / diverge
    metrics = Evaluator(sc).evaluate(trace.final.link_flows)
    assert metrics["sc_capacity_feasible"] == 0.0  # infeasible instance reported
    assert metrics["max_capacity_violation"] > 0.0
    assert np.all(np.isfinite(trace.final.link_flows))


# ---------------------------------------------------------------- mechanics
def test_registry_paradigm_and_requires_scenario():
    from tabench.models import MODEL_REGISTRY

    assert "sc-tap" in MODEL_REGISTRY
    assert MODEL_REGISTRY["sc-tap"]().capabilities.paradigm == "static_sc_ue"
    with pytest.raises(ValueError, match="side_capacities"):
        SideConstrainedModel().solve(
            braess_scenario(), Budget(iterations=1), RngBundle(0), Trace()
        )
