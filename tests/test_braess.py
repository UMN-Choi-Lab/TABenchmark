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
