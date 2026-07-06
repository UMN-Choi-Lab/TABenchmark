"""Tests for boundedly-rational user equilibrium (br-ue, Mahmassani & Chang 1987).

BR-UE is a NEW equilibrium concept, not a UE solver: used routes need only lie
within an absolute indifference band ``epsilon`` of the OD minimum, so the
equilibrium is a SET and the emitted flow sits at the BAND EDGE (used-route excess
~ epsilon), NOT at the Wardrop point (excess ~ 0). The tests pin: the analytic
two-route band edge and the distinctness-from-UE gate; the epsilon->0 (Wardrop) and
epsilon->infinity (any feasible) limits; and the certificate's honest
AEC<=epsilon necessary-not-sufficient character (a concentration counterexample).
"""

import dataclasses

import numpy as np
import pytest

from tabench import (
    BoundedlyRationalUEModel,
    Budget,
    Demand,
    ElasticDemand,
    Evaluator,
    FrankWolfeModel,
    Network,
    RngBundle,
    Scenario,
    Trace,
    br_two_route_scenario,
    braess_scenario,
)

GOLDEN_BRAESS_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"


def _solve(scenario, iterations=3000):
    trace = Trace()
    BoundedlyRationalUEModel().solve(scenario, Budget(iterations=iterations), RngBundle(0), trace)
    return trace


# --------------------------------------------------------- the band edge
def test_lands_at_band_edge_not_the_ue():
    """From the free-flow-AON start the band-relaxed swap stops at the band EDGE:
    f_A = f_A* + epsilon/2 = 6 at D=10, epsilon=1 (link flows (6,6,4,4)), with a
    used-route excess of ~epsilon -- NOT the Wardrop point (excess ~0). This is the
    distinctness gate: a UE solver stopped early would give excess ~0."""
    sc = br_two_route_scenario(demand=10.0, epsilon=1.0)
    trace = _solve(sc)
    v = trace.final.link_flows
    metrics = Evaluator(sc).evaluate(v)
    np.testing.assert_allclose(v, [6.0, 6.0, 4.0, 4.0], atol=1e-2)
    assert metrics["feasible"] == 1.0
    assert metrics["br_acceptable"] == 1.0
    assert metrics["average_excess_cost"] <= 1.0  # AEC <= epsilon (necessary)
    # DISTINCTNESS: the used-route excess is at the band edge (~epsilon), far from
    # the UE's ~0 -- so this is a genuine BR-UE, not a renamed early-stopped UE.
    assert trace.final.self_report["band_excess"] == pytest.approx(1.0, abs=1e-2)
    ue = Trace()
    FrankWolfeModel().solve(sc, Budget(iterations=200, target_relative_gap=1e-10), RngBundle(0), ue)
    assert abs(v[0] - ue.final.link_flows[0]) > 0.4  # 6.0 vs the UE's 5.5


def test_band_edge_scales_with_epsilon():
    """Larger epsilon -> wider band -> emitted flow farther from UE (edge f_A =
    5.5 + epsilon/2), a monotone relationship."""
    f1 = _solve(br_two_route_scenario(10.0, 1.0)).final.link_flows[0]
    f3 = _solve(br_two_route_scenario(10.0, 3.0)).final.link_flows[0]
    assert f1 == pytest.approx(6.0, abs=1e-2)  # 5.5 + 0.5
    assert f3 == pytest.approx(7.0, abs=1e-2)  # 5.5 + 1.5
    assert f3 > f1  # wider band -> farther from the UE split 5.5


# ------------------------------------------------------------------ limits
def test_epsilon_to_zero_recovers_wardrop_ue():
    """As epsilon -> 0 the band collapses and BR-UE -> the exact Wardrop UE (the
    shipped FW solver's split f_A* = 5.5)."""
    v = _solve(br_two_route_scenario(10.0, 1e-3)).final.link_flows
    ue = Trace()
    FrankWolfeModel().solve(
        br_two_route_scenario(10.0, 1e-3), Budget(iterations=300, target_relative_gap=1e-12),
        RngBundle(0), ue,
    )
    assert v[0] == pytest.approx(5.5, abs=1e-2)
    np.testing.assert_allclose(v, ue.final.link_flows, atol=1e-2)


def test_large_epsilon_admits_the_aon_start():
    """As epsilon -> infinity every feasible flow is acceptable, so the band
    triggers no swap and the pinned free-flow all-or-nothing start is emitted
    unchanged (all demand on the free-flow-cheapest route A)."""
    sc = br_two_route_scenario(10.0, 100.0)
    trace = _solve(sc, iterations=50)
    v = trace.final.link_flows
    assert v[0] == pytest.approx(10.0, abs=1e-9)  # AON start, no swaps
    assert trace.final.coords.iterations == 1  # rest at day 1
    assert Evaluator(sc).evaluate(v)["br_acceptable"] == 1.0


def test_satisfies_band_on_congested_multi_od_network():
    """REGRESSION for the engine choice. A proportional route swap drains a
    low-flow out-of-band route only at a rate ~ its vanishing flow, so on this
    congested all-power-4 multi-OD network (from a fuzz) it left a used route ~15
    above the band even after thousands of days. The band-thresholded gradient-
    projection Newton shift brings every used route into the band (band_excess <=
    epsilon) in ~100 iterations, so the emitted flow is a genuine epsilon-BRUE."""
    init = np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=np.int64)
    term = np.array([2, 3, 5, 4, 5, 1, 5, 1, 3, 3, 4], dtype=np.int64)
    cap = np.array([2.0379, 2.5332, 3.6736, 3.3267, 1.9544, 3.7727, 2.4127, 3.0813,
                    1.3216, 1.3136, 1.6057])
    fft = np.array([4.5378, 3.7192, 4.3969, 3.5777, 2.6262, 3.0663, 3.3738, 4.4485,
                    2.7527, 4.569, 3.4549])
    m = len(init)
    net = Network(
        name="congested-br", n_nodes=5, n_zones=5, first_thru_node=1,
        init_node=init, term_node=term, capacity=cap, length=np.zeros(m),
        free_flow_time=fft, b=np.full(m, 0.15), power=np.full(m, 4.0),
        toll=np.zeros(m), link_type=np.ones(m, dtype=np.int64),
    )
    od = np.zeros((5, 5))
    for (i, j), val in {
        (0, 4): 4.6141, (1, 0): 2.5035, (1, 2): 5.5097, (2, 1): 6.8684, (3, 1): 2.8589,
        (3, 2): 5.019, (3, 4): 2.0996, (4, 1): 4.2529,
    }.items():
        od[i, j] = val
    sc = Scenario("congested-br", net, Demand(od), br_epsilon=2.0)
    trace = _solve(sc, iterations=500)
    metrics = Evaluator(sc).evaluate(trace.final.link_flows)
    # Every used route is within the band (the true per-OD disaggregate check).
    assert trace.final.self_report["band_excess"] <= 2.0 + 1e-6
    assert metrics["feasible"] == 1.0
    assert metrics["br_acceptable"] == 1.0


# --------------------------------------------------- certificate honesty
def test_certificate_is_necessary_not_sufficient():
    """HONEST LIMITATION (adr-008): br_acceptable = (AEC <= epsilon) is NECESSARY
    but not sufficient. Because AEC is the demand-weighted MEAN excess, a flow can
    concentrate a little traffic on a route far outside the band and still average
    under it. Link flows cannot exclude this (route flows are never emitted). We
    pin it transparently: three constant-cost disjoint routes, two cheap (cost 10)
    carrying 99% and one at cost 60 (excess 50*epsilon) carrying 1% -> AEC = 0.5 <=
    epsilon=1, so br_acceptable = 1.0 even though a used route is 50x outside the
    band."""
    # Routes 1-3-2, 1-4-2 (cost 10 each), 1-5-2 (cost 60); constant costs (b=0).
    init = np.array([1, 3, 1, 4, 1, 5], dtype=np.int64)
    term = np.array([3, 2, 4, 2, 5, 2], dtype=np.int64)
    fft = np.array([5.0, 5.0, 5.0, 5.0, 30.0, 30.0])  # route costs 10, 10, 60
    net = Network(
        name="conc", n_nodes=5, n_zones=2, first_thru_node=1,
        init_node=init, term_node=term, capacity=np.ones(6), length=np.zeros(6),
        free_flow_time=fft, b=np.zeros(6), power=np.ones(6),
        toll=np.zeros(6), link_type=np.ones(6, dtype=np.int64),
    )
    od = np.zeros((2, 2))
    od[0, 1] = 10.0
    sc = Scenario("conc", net, Demand(od), br_epsilon=1.0)
    # 99% on the two cheap routes, 1% on the cost-60 route (excess 50).
    v = np.array([4.95, 4.95, 4.95, 4.95, 0.1, 0.1])
    metrics = Evaluator(sc).evaluate(v)
    assert metrics["feasible"] == 1.0
    assert metrics["average_excess_cost"] == pytest.approx(0.5, abs=1e-9)  # <= 1
    assert metrics["br_acceptable"] == 1.0  # FALSE accept -- the documented gap
    # A grossly out-of-band flow (all on the cost-60 route) IS rejected: AEC = 50.
    bad = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 10.0])
    assert Evaluator(sc).evaluate(bad)["br_acceptable"] == 0.0


# ---------------------------------------------------------------- scenario
def test_br_epsilon_hashed_only_when_set_and_golden_preserved():
    assert braess_scenario().content_hash() == GOLDEN_BRAESS_HASH
    sc = br_two_route_scenario(10.0, 1.0)
    other = dataclasses.replace(sc, br_epsilon=2.0)
    none = dataclasses.replace(sc, br_epsilon=None)
    assert len({sc.content_hash(), other.content_hash(), none.content_hash()}) == 3


def test_br_epsilon_mutually_exclusive_and_validated():
    sc = br_two_route_scenario(10.0, 1.0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        dataclasses.replace(sc, elastic_demand=ElasticDemand("linear", 10.0))
    with pytest.raises(ValueError, match="mutually exclusive"):
        dataclasses.replace(sc, sue_theta=0.5)
    with pytest.raises(ValueError, match="br_epsilon must be finite and > 0"):
        dataclasses.replace(sc, br_epsilon=0.0)


# ---------------------------------------------------------------- mechanics
def test_registry_paradigm_and_requires_scenario():
    from tabench.models import MODEL_REGISTRY

    assert "br-ue" in MODEL_REGISTRY
    assert MODEL_REGISTRY["br-ue"]().capabilities.paradigm == "static_br_ue"
    with pytest.raises(ValueError, match="br_epsilon"):
        BoundedlyRationalUEModel().solve(
            braess_scenario(), Budget(iterations=1), RngBundle(0), Trace()
        )
