"""Tests for the first learned (black-box) model — ``learned-surrogate``.

The point is not that the tiny per-link regressor is a good solver (it is not);
it is that a LEARNED model is subject to the *same* certification as Frank-Wolfe.
The harness recomputes the equilibrium gap and the demand-feasibility audit from
the emitted link flows (P1), so high link-flow correlation with the equilibrium
(what the ML-TA literature reports) does not by itself earn certification. These
tests pin: the wrapper contract (runs, deterministic, mixes with classical
solvers in one grid), the ``trained_on`` fairness gate (refused on its synthetic
training family), and the accuracy-vs-certification contrast on a real network.
"""

import numpy as np
import pytest
from conftest import load_or_skip

from tabench import (
    BiconjugateFrankWolfeModel,
    Budget,
    Evaluator,
    LearnedSurrogateModel,
    RngBundle,
    Trace,
    braess_scenario,
    run_experiment,
)
from tabench.core.capabilities import ContaminationError
from tabench.models.base import MODEL_REGISTRY
from tabench.models.learned import TRAINING_FAMILY, _random_network_scenario


def _flows(scenario, model, iterations=1):
    trace = Trace()
    model.solve(scenario, Budget(iterations=iterations), RngBundle(0), trace)
    return trace.final.link_flows


# ----------------------------------------------------------- wrapper contract
def test_registered_as_learned_paradigm():
    assert "learned-surrogate" in MODEL_REGISTRY
    caps = LearnedSurrogateModel.capabilities
    assert caps.paradigm == "learned"
    assert caps.deterministic is True
    # Declares the training family AND the training instances' content hashes.
    assert TRAINING_FAMILY in caps.trained_on
    assert len(caps.trained_on) > 1


def test_runs_and_is_certified_like_any_model():
    """A learned model emits link flows and is scored through the same
    Evaluator; it must not crash even when its flows are censored."""
    scenario = braess_scenario()
    v = _flows(scenario, LearnedSurrogateModel())
    assert v.shape == (scenario.network.n_links,)
    assert np.all(v >= 0.0)
    metrics = Evaluator(scenario).evaluate(v)  # never raises on approximate flows
    assert metrics["feasible"] in (0.0, 1.0)


def test_deterministic():
    a = _flows(braess_scenario(), LearnedSurrogateModel())
    b = _flows(braess_scenario(), LearnedSurrogateModel())
    np.testing.assert_array_equal(a, b)


def test_pinned_link_flows_anchor():
    """Regression anchor: on the fixed Braess scenario (seedless, deterministic per
    the model's capabilities) the surrogate emits fixed link flows. Two runs are
    byte-identical, and the emitted final flows are pinned to the measured values
    [4.32019858, 0.01432931, 4.32019858, 0.01432931, 4.32019858] at atol=1e-5
    (tight but safe against BLAS-level drift in the offline ridge fit). Pins the
    end-to-end train->predict pipeline against silent surrogate regressions."""
    a = _flows(braess_scenario(), LearnedSurrogateModel())
    b = _flows(braess_scenario(), LearnedSurrogateModel())
    np.testing.assert_array_equal(a, b)  # deterministic (per capabilities)
    measured = [4.320198576963898, 0.014329312516191, 4.320198576963898,
                0.014329312516191, 4.320198576963898]
    np.testing.assert_allclose(a, measured, atol=1e-5)


def test_mixes_with_classical_solvers_in_one_grid():
    scenario = braess_scenario()
    result = run_experiment(
        scenario,
        [BiconjugateFrankWolfeModel(), LearnedSurrogateModel()],
        Budget(iterations=100, target_relative_gap=1e-8),
        seed=0,
    )
    last = {row["model"]: row for row in result.rows}
    assert "bfw" in last and "learned-surrogate" in last
    assert last["bfw"]["feasible"] == 1.0  # classical solver certifies
    # The learned row exists and is scored (censored or not) — the run survived
    # a black box regardless of its output quality.
    assert "feasible" in last["learned-surrogate"]


# ----------------------------------------------------------- fairness gate
def test_fairness_gate_blocks_training_family():
    """The surrogate declares trained_on=('synthetic-net',); evaluating it on a
    synthetic-net scenario is refused (train/test contamination)."""
    train_scenario = _random_network_scenario(1, 8, 3, 4)
    assert train_scenario.family == TRAINING_FAMILY
    with pytest.raises(ContaminationError):
        run_experiment(
            train_scenario,
            [LearnedSurrogateModel()],
            Budget(iterations=1),
            seed=0,
        )


def test_evaluated_on_disjoint_tntp_is_allowed():
    """TNTP scenarios are a different family, so the gate permits evaluation."""
    scenario = load_or_skip("siouxfalls")
    assert scenario.family != TRAINING_FAMILY
    # Does not raise; produces a scored (here censored) row.
    result = run_experiment(
        scenario, [LearnedSurrogateModel()], Budget(iterations=1), seed=0
    )
    assert result.rows[-1]["model"] == "learned-surrogate"


# --------------------------------------------- accuracy vs. certification
def test_high_link_flow_accuracy_does_not_imply_certification():
    """The headline (Anaheim): the surrogate correlates strongly with the
    best-known equilibrium flows — the metric the ML-TA literature reports —
    yet the harness's demand-feasibility audit still censors it. Link-flow
    accuracy is not a certificate."""
    scenario = load_or_skip("anaheim")
    assert scenario.reference is not None
    v = _flows(scenario, LearnedSurrogateModel())
    oracle = scenario.reference.link_flows
    corr = float(np.corrcoef(v, oracle)[0, 1])
    metrics = Evaluator(scenario).evaluate(v)
    assert corr > 0.85  # "looks great" by link-flow standards
    assert metrics["feasible"] == 0.0  # ... but not demand-feasible (P1)


# --------------------------------------------------- training data sanity
def test_training_networks_are_valid_and_connected():
    """Every synthetic training network is a valid, strongly-connected BPR
    instance (so all OD pairs are reachable and the training solves succeed)."""
    from tabench.models.learned import _TRAINING_SPECS

    for spec in _TRAINING_SPECS:
        sc = _random_network_scenario(*spec)
        assert sc.family == TRAINING_FAMILY
        assert sc.demand.total > 0
        # Bi-conjugate FW converges -> reachable OD pairs, valid network.
        trace = Trace()
        BiconjugateFrankWolfeModel().solve(
            sc, Budget(iterations=50, target_relative_gap=1e-6), RngBundle(0), trace
        )
        assert Evaluator(sc).evaluate(trace.final.link_flows)["feasible"] == 1.0
