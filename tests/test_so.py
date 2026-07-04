"""Tests for system-optimum assignment, price of anarchy, and first-best tolls.

Analytic anchor: Braess with demand 6. UE = (4,2,2,2,4), TSTT = 552.
SO puts nothing on the bypass — marginal costs (20v, 50+2v, 10+2v) make the
bypass route strictly worse at the symmetric split — so SO = (3,3,0,3,3),
TSTT = 498, and PoA = 552/498 ~ 1.1084 (< 4/3, consistent with the
Roughgarden-Tardos bound for affine latencies).
"""

import dataclasses
import math

import numpy as np
import pytest
from conftest import load_or_skip

from tabench import (
    BiconjugateFrankWolfeModel,
    Budget,
    Evaluator,
    RngBundle,
    SystemOptimumModel,
    Trace,
    braess_scenario,
    marginal_cost_tolls,
    marginal_costs,
    price_of_anarchy,
    tolled_network,
)
from tabench.models.so import marginal_network

SO_FLOWS = np.array([3.0, 3.0, 0.0, 3.0, 3.0])
UE_FLOWS = np.array([4.0, 2.0, 2.0, 2.0, 4.0])


@pytest.fixture(scope="module")
def braess():
    return braess_scenario()


def _solve(model, scenario, **budget_kwargs):
    trace = Trace()
    model.solve(scenario, Budget(**budget_kwargs), RngBundle(0), trace)
    return trace


def test_marginal_cost_identity(braess):
    """t + v t' computed directly must equal the transformed network's cost."""
    net = braess.network
    transformed = marginal_network(net)
    for v in (np.zeros(5), UE_FLOWS, SO_FLOWS, np.array([1.5, 0.1, 6.0, 0.0, 2.2])):
        np.testing.assert_allclose(
            marginal_costs(net, v), transformed.link_cost(v), rtol=1e-12
        )


def test_braess_system_optimum(braess):
    trace = _solve(SystemOptimumModel(), braess, iterations=100, target_relative_gap=1e-10)
    flows = trace.final.link_flows
    np.testing.assert_allclose(flows, SO_FLOWS, atol=1e-4)
    # The marginal network's Beckmann IS the true SO objective, self-reported
    # under its truthful name: tstt.
    tstt = float(flows @ braess.network.link_cost(flows))
    assert tstt == pytest.approx(498.0, abs=1e-3)
    assert trace.final.self_report["tstt"] == pytest.approx(tstt, rel=1e-9)
    assert trace.final.self_report["so_relative_gap"] <= 1e-10


def test_so_flows_are_feasible_but_not_ue(braess):
    """SO flows pass the audit; their certified UE gap is strictly positive."""
    metrics = Evaluator(braess).evaluate(SO_FLOWS)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] > 0.01


def test_braess_price_of_anarchy(braess):
    poa = price_of_anarchy(braess, UE_FLOWS, SO_FLOWS)
    # rel 1e-6: the zero-intercept links carry a tiny fft = 1e-6 by construction
    assert poa == pytest.approx(552.0 / 498.0, rel=1e-6)
    assert 1.0 < poa <= 4.0 / 3.0  # affine-latency bound (Roughgarden-Tardos)


def test_first_best_tolls_decentralize_the_optimum(braess):
    """UE on the tolled network reproduces the SO flows (Yang-Huang)."""
    net = braess.network
    tolls = marginal_cost_tolls(net, SO_FLOWS)
    np.testing.assert_allclose(tolls, [30.0, 3.0, 0.0, 3.0, 30.0], atol=1e-4)
    tolled = tolled_network(net, tolls)
    scenario = dataclasses.replace(braess, network=tolled, reference=None)
    trace = _solve(
        BiconjugateFrankWolfeModel(), scenario, iterations=200, target_relative_gap=1e-10
    )
    np.testing.assert_allclose(trace.final.link_flows, SO_FLOWS, atol=1e-3)


def test_tolled_network_guards_and_composition():
    net = braess_scenario().network
    with pytest.raises(ValueError, match="shape"):
        tolled_network(net, np.ones(3))
    with pytest.raises(ValueError, match="nonnegative"):
        tolled_network(net, -np.ones(net.n_links))
    # Tolling composes: pre-existing toll contributions fold in exactly.
    once = tolled_network(net, np.ones(net.n_links))
    twice = tolled_network(once, 2.0 * np.ones(net.n_links))
    np.testing.assert_allclose(twice.fixed_cost, net.fixed_cost + 3.0)


def test_certified_so_metrics_and_honesty(braess, tmp_path):
    """run_experiment auto-enables SO columns; self-report matches harness."""
    from tabench import run_experiment

    result = run_experiment(
        braess,
        [SystemOptimumModel(), BiconjugateFrankWolfeModel()],
        Budget(iterations=60, target_relative_gap=1e-10),
        out_dir=tmp_path,
    )
    by_model = {}
    for row in result.rows:
        by_model[row["model"]] = row
    so_row, ue_row = by_model["so-bfw"], by_model["bfw"]
    # Honesty: harness-certified SO gap equals the solver's self-report.
    assert so_row["so_relative_gap"] == pytest.approx(
        so_row["self_so_relative_gap"], abs=1e-12
    )
    assert so_row["so_relative_gap"] < 1e-9
    # Cross gaps are strictly positive: UE flows are not SO and vice versa.
    assert ue_row["so_relative_gap"] > 0.05
    assert so_row["relative_gap"] > 0.1
    # SO achieves a strictly lower total system travel time than UE.
    assert so_row["tstt"] < ue_row["tstt"]


def test_two_route_system_optimum_analytic():
    """SO of the two-route net: f_A = 31/12, f_B = 17/12; PoA = 864/863."""
    from tabench import two_route_scenario

    scenario = two_route_scenario(sue_theta=None)
    trace = _solve(SystemOptimumModel(), scenario, iterations=100, target_relative_gap=1e-12)
    f_a, f_b = 31.0 / 12.0, 17.0 / 12.0
    np.testing.assert_allclose(trace.final.link_flows, [f_a, f_a, f_b, f_b], atol=1e-6)
    ue = np.array([2.5, 2.5, 1.5, 1.5])  # theta -> infinity limit at demand 4
    poa = price_of_anarchy(scenario, ue, trace.final.link_flows)
    assert poa == pytest.approx(864.0 / 863.0, rel=1e-6)


def test_censored_flows_get_nan_so_columns(braess):
    """Audit failures censor the SO columns too — no gameable negative gaps."""
    evaluator = Evaluator(braess, so_metrics=True)
    metrics = evaluator.evaluate(0.9 * UE_FLOWS)
    assert metrics["feasible"] == 0.0
    assert math.isnan(metrics["so_relative_gap"])
    assert math.isnan(metrics["tstt_mc"])
    # Exact SO flows: feasible, and the certified SO gap is never
    # meaningfully negative.
    at_so = evaluator.evaluate(SO_FLOWS)
    assert at_so["feasible"] == 1.0
    assert at_so["so_relative_gap"] >= -1e-9


def test_so_rows_not_scored_against_ue_oracle(braess, tmp_path):
    """The UE best-known oracle must not fill flow_rmse for SO-goal models."""
    from tabench import run_experiment

    result = run_experiment(
        braess, [SystemOptimumModel()], Budget(iterations=10), out_dir=tmp_path
    )
    assert all(row["flow_rmse_vs_reference"] == "" for row in result.rows)


def test_so_target_gap_early_stop():
    """Budget.target_relative_gap applies to the self-monitored SO gap."""
    scenario = load_or_skip("siouxfalls")
    trace = _solve(SystemOptimumModel(), scenario, iterations=300, target_relative_gap=1e-4)
    assert trace.final.coords.iterations < 300
    assert trace.final.self_report["so_relative_gap"] <= 1e-4


def test_siouxfalls_so_below_ue():
    scenario = load_or_skip("siouxfalls")
    so_trace = _solve(SystemOptimumModel(), scenario, iterations=300, target_relative_gap=1e-6)
    ue_trace = _solve(
        BiconjugateFrankWolfeModel(), scenario, iterations=300, target_relative_gap=1e-6
    )
    net = scenario.network
    so, ue = so_trace.final.link_flows, ue_trace.final.link_flows
    assert Evaluator(scenario).evaluate(so)["feasible"] == 1.0
    tstt_so = float(so @ net.link_cost(so))
    tstt_ue = float(ue @ net.link_cost(ue))
    assert tstt_so < tstt_ue
    poa = price_of_anarchy(scenario, ue, so)
    assert 1.0 < poa < 1.5  # BPR power 4: modest PoA expected at this congestion
