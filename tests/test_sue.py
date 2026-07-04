"""Tests for the logit-SUE task: Dial-STOCH loading, MSA-SUE, and the
fixed-point certificate (docs/design/adr-001).

The two-route scenario reduces Dial's loading to a binary logit, so the SUE
fixed point is the root of a scalar equation solved here with brentq — the
tests never trust pre-computed digits.
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
    Scenario,
    Trace,
    braess_scenario,
    two_route_scenario,
)
from tabench.models._stoch import StochEngine

# Golden content hash of the Braess scenario BEFORE sue_theta existed: the
# conditional hash append must leave every existing hash unchanged.
BRAESS_GOLDEN_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"


def _fixed_point_route_a(theta: float, demand: float = 4.0) -> float:
    """Root of f_A = D / (1 + exp(theta (c_A(f_A) - c_B(D - f_A))))."""

    def residual(f_a: float) -> float:
        c_a = 2.0 + f_a
        c_b = 1.5 + 2.0 * (demand - f_a)
        return f_a - demand / (1.0 + math.exp(theta * (c_a - c_b)))

    return brentq(residual, 0.0, demand, xtol=1e-12)


@pytest.fixture(scope="module")
def scenario():
    return two_route_scenario()  # demand 4, theta 0.5


# ------------------------------------------------------------------- loading


def test_dial_load_matches_binary_logit_on_fixed_times(scenario):
    """At free-flow costs the load is the exact binary logit split."""
    net = scenario.network
    engine = StochEngine(net)
    t0 = net.link_cost(np.zeros(net.n_links))
    flows = engine.load(t0, scenario.demand, theta=0.5)
    # c_A = 2, c_B = 1.5 at free flow: share_A = 1/(1 + exp(0.5 * 0.5)).
    share_a = 1.0 / (1.0 + math.exp(0.5 * (2.0 - 1.5)))
    expected = 4.0 * np.array([share_a, share_a, 1.0 - share_a, 1.0 - share_a])
    np.testing.assert_allclose(flows, expected, atol=1e-9)


def test_dial_load_braess_three_path_logit():
    """On Braess at free-flow costs all three paths are efficient: 3-way logit."""
    scenario = braess_scenario()
    net = scenario.network
    engine = StochEngine(net)
    t0 = net.link_cost(np.zeros(net.n_links))  # (1e-6, 50, 10, 50, 1e-6)
    theta = 0.1
    flows = engine.load(t0, scenario.demand, theta)
    # Path costs: 1-3-2 and 1-4-2 cost 50+1e-6, bypass 1-3-4-2 costs 10+2e-6.
    costs = np.array([t0[0] + t0[3], t0[1] + t0[4], t0[0] + t0[2] + t0[4]])
    shares = np.exp(-theta * costs)
    shares /= shares.sum()
    d = scenario.demand.total
    expected = d * np.array(
        [
            shares[0] + shares[2],  # 1->3
            shares[1],  # 1->4
            shares[2],  # 3->4
            shares[0],  # 3->2
            shares[1] + shares[2],  # 4->2
        ]
    )
    np.testing.assert_allclose(flows, expected, atol=1e-6)


def test_loader_honors_first_thru_node():
    """A cheap path through a restricted centroid must carry no flow."""

    def build(first_thru_node: int) -> Scenario:
        eps = 1e-9
        network = Network(
            name="ftn-probe",
            n_nodes=3,
            n_zones=3,
            first_thru_node=first_thru_node,
            init_node=np.array([1, 1, 3], dtype=np.int64),
            term_node=np.array([2, 3, 2], dtype=np.int64),
            capacity=np.ones(3),
            length=np.zeros(3),
            free_flow_time=np.array([10.0, 1.0, 1.0]),
            b=np.full(3, eps),  # effectively constant costs
            power=np.ones(3),
            toll=np.zeros(3),
            link_type=np.ones(3, dtype=np.int64),
        )
        od = np.zeros((3, 3))
        od[0, 1] = 1.0
        return Scenario(name="ftn-probe", network=network, demand=Demand(matrix=od))

    theta = 1.0
    restricted = build(first_thru_node=4)  # all zones are restricted centroids
    net = restricted.network
    flows = StochEngine(net).load(
        net.link_cost(np.zeros(3)), restricted.demand, theta
    )
    np.testing.assert_allclose(flows, [1.0, 0.0, 0.0], atol=1e-9)

    open_net = build(first_thru_node=1)  # zones may carry through traffic
    flows = StochEngine(open_net.network).load(
        open_net.network.link_cost(np.zeros(3)), open_net.demand, theta
    )
    direct_share = 1.0 / (1.0 + math.exp(theta * (10.0 - 2.0)))
    np.testing.assert_allclose(
        flows, [direct_share, 1.0 - direct_share, 1.0 - direct_share], atol=1e-6
    )


# ------------------------------------------------------------ solver + certificate


def test_msa_converges_to_scalar_fixed_point(scenario):
    f_a = _fixed_point_route_a(theta=0.5)
    trace = Trace()
    DialSUEModel().solve(scenario, Budget(iterations=500), RngBundle(0), trace)
    expected = np.array([f_a, f_a, 4.0 - f_a, 4.0 - f_a])
    np.testing.assert_allclose(trace.final.link_flows, expected, atol=1e-3)
    assert trace.final.self_report["sue_fixed_point_residual"] < 1e-5


def test_certificate_zero_at_analytic_fixed_point(scenario):
    f_a = _fixed_point_route_a(theta=0.5)
    metrics = Evaluator(scenario).evaluate(np.array([f_a, f_a, 4.0 - f_a, 4.0 - f_a]))
    assert metrics["feasible"] == 1.0
    assert metrics["sue_fixed_point_residual"] < 1e-8
    # UE metrics stay as strictly positive descriptive columns at SUE.
    assert metrics["relative_gap"] > 0.01


def test_certificate_positive_for_perturbed_flows(scenario):
    f_a = _fixed_point_route_a(theta=0.5) - 0.5  # shift 0.5 units A -> B
    metrics = Evaluator(scenario).evaluate(np.array([f_a, f_a, 4.0 - f_a, 4.0 - f_a]))
    assert metrics["feasible"] == 1.0  # still conserves demand
    assert metrics["sue_fixed_point_residual"] > 0.1


def test_theta_large_approaches_ue():
    gaps = [abs(_fixed_point_route_a(theta) - 2.5) for theta in (0.5, 2.0, 5.0, 50.0)]
    assert all(g2 < g1 for g1, g2 in zip(gaps, gaps[1:], strict=False))
    assert gaps[-1] < 0.005

    stiff = two_route_scenario(sue_theta=50.0)
    trace = Trace()
    DialSUEModel().solve(stiff, Budget(iterations=500), RngBundle(0), trace)
    assert abs(trace.final.link_flows[0] - 2.5) < 0.01
    assert Evaluator(stiff).evaluate(trace.final.link_flows)["relative_gap"] < 1e-3


def test_zero_flows_still_censored_on_sue_scenarios(scenario):
    metrics = Evaluator(scenario).evaluate(np.zeros(4))
    assert metrics["feasible"] == 0.0
    assert math.isnan(metrics["sue_fixed_point_residual"])


def test_self_report_matches_harness_certificate(scenario):
    """Honesty check (P1): the solver's residual equals the recomputed one."""
    trace = Trace()
    DialSUEModel().solve(scenario, Budget(iterations=50), RngBundle(0), trace)
    evaluator = Evaluator(scenario)
    for state in list(trace)[::10]:
        certified = evaluator.evaluate(state.link_flows)["sue_fixed_point_residual"]
        assert certified == pytest.approx(
            state.self_report["sue_fixed_point_residual"], rel=1e-9, abs=1e-15
        )


def test_sue_model_requires_theta():
    ue_scenario = braess_scenario()
    with pytest.raises(ValueError, match="sue_theta"):
        DialSUEModel().solve(ue_scenario, Budget(iterations=5), RngBundle(0), Trace())


# ------------------------------------------------------------------- hashing


def test_existing_content_hashes_preserved():
    assert braess_scenario().content_hash() == BRAESS_GOLDEN_HASH


def test_theta_is_content_hashed():
    hashes = {
        two_route_scenario(sue_theta=None).content_hash(),
        two_route_scenario(sue_theta=0.5).content_hash(),
        two_route_scenario(sue_theta=50.0).content_hash(),
    }
    assert len(hashes) == 3


def test_invalid_theta_rejected():
    with pytest.raises(ValueError, match="sue_theta"):
        two_route_scenario(sue_theta=-1.0)
    with pytest.raises(ValueError, match="sue_theta"):
        two_route_scenario(sue_theta=float("nan"))
