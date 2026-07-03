"""Regression tests for the certification gates found in adversarial review.

These lock in the fixes for: unrouted/under-scaled demand certifying as
feasible (the audit was blind to OD totals), invalid black-box output
crashing whole experiments, silent result overwrites on duplicate model
names, and shared mutable class-level factor declarations.
"""

import math

import numpy as np
import pytest

from tabench import (
    Budget,
    CallableModel,
    Capabilities,
    Evaluator,
    FrankWolfeModel,
    braess_scenario,
    run_experiment,
)
from tabench.models.base import TrafficAssignmentModel, register_model

REF_FLOWS = np.array([4.0, 2.0, 2.0, 2.0, 4.0])


@pytest.fixture(scope="module")
def scenario():
    return braess_scenario()


def test_zero_flows_are_censored_not_perfect(scenario):
    """An all-zero 'model' must not top a gap-sorted leaderboard."""
    metrics = Evaluator(scenario).evaluate(np.zeros(5))
    assert metrics["feasible"] == 0.0
    assert math.isnan(metrics["relative_gap"])
    assert math.isnan(metrics["average_excess_cost"])


def test_underscaled_flows_fail_demand_audit(scenario):
    """0.9x equilibrium flows conserve at intersections but not at zones."""
    metrics = Evaluator(scenario).evaluate(0.9 * REF_FLOWS)
    assert metrics["feasible"] == 0.0
    assert math.isnan(metrics["relative_gap"])


def test_negative_flow_censored_without_crashing_the_run(scenario, tmp_path):
    """A surrogate emitting one negative flow is censored; other models' results survive."""
    bad = CallableModel(
        fn=lambda s, g: np.array([4.0, 2.0, 2.0, 2.0, -0.5]),
        name="bad-surrogate",
    )
    result = run_experiment(
        scenario,
        [FrankWolfeModel(), bad],
        Budget(iterations=20),
        out_dir=tmp_path,
    )
    by_model = {}
    for row in result.rows:
        by_model[row["model"]] = row
    assert by_model["fw"]["feasible"] == 1.0
    assert by_model["bad-surrogate"]["feasible"] == 0.0
    assert math.isnan(by_model["bad-surrogate"]["relative_gap"])
    assert list(tmp_path.glob("*.csv")), "CSV must be written despite the bad model"


def test_nonfinite_flows_censored(scenario):
    metrics = Evaluator(scenario).evaluate(np.array([1.0, np.nan, 1.0, 1.0, 1.0]))
    assert metrics["feasible"] == 0.0


def test_tiny_negative_noise_is_clipped_not_censored(scenario):
    # Floating-point dust from a solver is clipped, not treated as a violation.
    metrics = Evaluator(scenario).evaluate(REF_FLOWS + np.array([-1e-12, 0, 0, 0, 0]))
    assert metrics["feasible"] == 1.0


def test_duplicate_model_names_rejected(scenario):
    with pytest.raises(ValueError, match="Duplicate model names"):
        run_experiment(
            scenario,
            [FrankWolfeModel(), FrankWolfeModel(line_search_xtol=1e-6)],
            Budget(iterations=2),
        )


def test_sweep_with_distinct_names_keeps_all_results(scenario):
    a = FrankWolfeModel(line_search_xtol=1e-4)
    a.name = "fw-loose"
    b = FrankWolfeModel(line_search_xtol=1e-12)
    b.name = "fw-tight"
    result = run_experiment(scenario, [a, b], Budget(iterations=5))
    assert set(result.manifest["models"]) == {"fw-loose", "fw-tight"}
    assert result.manifest["models"]["fw-loose"]["factors"]["line_search_xtol"] == 1e-4
    assert result.manifest["models"]["fw-tight"]["factors"]["line_search_xtol"] == 1e-12
    assert {k[0] for k in result.bundles} == {"fw-loose", "fw-tight"}


def test_register_model_rejects_adapter_style_classes():
    with pytest.raises(TypeError, match="capabilities"):

        @register_model
        class NoCaps(TrafficAssignmentModel):
            name = "nocaps"

            def solve(self, scenario, budget, rng, trace):  # pragma: no cover
                raise NotImplementedError


def test_factor_declarations_do_not_leak_across_classes():
    class ModelA(TrafficAssignmentModel):
        name = "leak-a"
        capabilities = Capabilities(
            paradigm="heuristic", deterministic=True, provides_gap=False, seedable=True
        )

        def solve(self, scenario, budget, rng, trace):  # pragma: no cover
            raise NotImplementedError

    class ModelB(ModelA):
        name = "leak-b"

    from tabench.core.factors import FactorSpec

    ModelB.factors["ghost"] = FactorSpec(default=1.0)
    assert "ghost" not in ModelA.factors
    assert "ghost" not in TrafficAssignmentModel.factors


def test_manifest_records_rng_and_runs(scenario):
    result = run_experiment(scenario, [FrankWolfeModel()], Budget(iterations=3), seed=11)
    assert result.manifest["rng"]["root_seed"] == 11
    assert result.manifest["runs"][0]["seed_info"] == {"root_seed": 11, "macrorep": 0}
    assert "git_commit" in result.manifest["environment"]
