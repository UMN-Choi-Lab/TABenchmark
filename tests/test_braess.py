"""Analytic anchor tests: the Braess network has a hand-checkable UE.

If the harness cannot reproduce (4, 2, 2, 2, 4) with route time 92, nothing
downstream is trustworthy.
"""

import numpy as np
import pytest

from tabench import (
    AllOrNothingModel,
    Budget,
    CallableModel,
    ContaminationError,
    Evaluator,
    FrankWolfeModel,
    MSAModel,
    RngBundle,
    Trace,
    braess_scenario,
    run_experiment,
)

REF_FLOWS = np.array([4.0, 2.0, 2.0, 2.0, 4.0])
REF_ROUTE_TIME = 92.0


@pytest.fixture(scope="module")
def scenario():
    return braess_scenario()


def _solve(model, scenario, iterations):
    trace = Trace()
    bundle = model.solve(scenario, Budget(iterations=iterations), RngBundle(0), trace)
    return bundle


def test_analytic_flows_have_zero_gap(scenario):
    metrics = Evaluator(scenario).evaluate(REF_FLOWS)
    assert metrics["relative_gap"] < 1e-6
    assert metrics["feasible"] == 1.0
    assert metrics["node_balance_residual"] == pytest.approx(0.0, abs=1e-12)


def test_frank_wolfe_reaches_analytic_ue(scenario):
    bundle = _solve(FrankWolfeModel(), scenario, iterations=500)
    flows = bundle.final.link_flows
    assert np.allclose(flows, REF_FLOWS, atol=2e-2)
    metrics = Evaluator(scenario).evaluate(flows)
    assert metrics["relative_gap"] < 1e-5
    # At UE every used route costs 92, so TSTT / demand == route time.
    assert metrics["tstt"] / scenario.demand.total == pytest.approx(REF_ROUTE_TIME, abs=0.1)
    # off=no-op for the multiclass engine's per-class field: a single-class model
    # leaves class_link_flows at the core/results.py default (None). Only the
    # multiclass solver (adr-013) ever populates it.
    assert bundle.final.class_link_flows is None


def test_msa_pinned_flow_anchor(scenario):
    """Regression anchor: MSA on Braess (D=6) at 200 iterations converges TOWARD
    the analytic UE [4,2,2,2,4] but not all the way (1/k averaging is slow). The
    measured max deviation from UE at 200 iters is ~0.0201 (flows
    [4.01005, 1.98995, 1.98995, 2.02010, 3.97990]); pin to UE at atol=0.045
    (~2x the measured deviation) so a convergence-speed regression is caught."""
    bundle = _solve(MSAModel(), scenario, iterations=200)
    np.testing.assert_allclose(bundle.final.link_flows, REF_FLOWS, atol=0.045)


def test_aon_pinned_free_flow_route(scenario):
    """Regression anchor: AON is closed-form on Braess. Free-flow route costs are
    1->3->2 = eps+50, 1->4->2 = 50+eps, and the bypass 1->3->4->2 = eps+10+eps ~ 10
    (the zero-intercept links carry a tiny fft=1e-6). The bypass is strictly
    cheapest, so all 6 units load onto it: links 1->3 (idx 0), 3->4 (idx 2), 4->2
    (idx 4) carry 6, the two flat links carry 0 -> exact flow [6,0,6,0,6]."""
    bundle = _solve(AllOrNothingModel(), scenario, iterations=1)
    np.testing.assert_allclose(bundle.final.link_flows, [6.0, 0.0, 6.0, 0.0, 6.0], atol=1e-9)


def test_frank_wolfe_self_report_matches_harness(scenario):
    """Honesty check (P1): white-box self-reported gap == certified gap."""
    bundle = _solve(FrankWolfeModel(), scenario, iterations=50)
    state = bundle.final
    certified = Evaluator(scenario).evaluate(state.link_flows)["relative_gap"]
    assert state.self_report["relative_gap"] == pytest.approx(certified, rel=1e-9, abs=1e-12)


def test_msa_converges_slowly_but_surely(scenario):
    bundle = _solve(MSAModel(), scenario, iterations=200)
    metrics = Evaluator(scenario).evaluate(bundle.final.link_flows)
    assert metrics["relative_gap"] < 1e-2


def test_aon_certified_with_large_gap(scenario):
    """Non-equilibrium models are scored honestly, not excluded (P5)."""
    bundle = _solve(AllOrNothingModel(), scenario, iterations=1)
    metrics = Evaluator(scenario).evaluate(bundle.final.link_flows)
    assert metrics["relative_gap"] > 0.1
    assert metrics["feasible"] == 1.0


def test_fairness_gate_blocks_contaminated_model(scenario):
    leaky = CallableModel(
        fn=lambda s, g: REF_FLOWS,
        name="memorizer",
        trained_on=("builtin-braess",),
    )
    with pytest.raises(ContaminationError):
        run_experiment(scenario, [leaky], Budget(iterations=1))


def test_blackbox_callable_gets_certified(scenario):
    """A black-box wrapper receives the same external certification (P1)."""
    oracle_cheat = CallableModel(fn=lambda s, g: REF_FLOWS, name="oracle", trained_on=())
    result = run_experiment(scenario, [oracle_cheat], Budget(iterations=1))
    row = result.rows[-1]
    assert row["relative_gap"] < 1e-6
    assert row["model"] == "oracle"
