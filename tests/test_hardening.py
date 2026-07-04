"""Regression tests for the v0.x adversarial-review fixes.

Locks in: BPR parameter validation, warning-free bounded Hessian, SUE
certification censoring instead of crashing on label saturation, the SUE
solver honoring the convergence target, collision-proof output stems,
theta in the manifest, and clean CLI error paths.
"""

import json
import math
import warnings

import numpy as np
import pytest

from tabench import (
    Budget,
    Demand,
    DialSUEModel,
    Evaluator,
    FrankWolfeModel,
    Network,
    RngBundle,
    Scenario,
    Trace,
    braess_scenario,
    run_experiment,
    two_route_scenario,
)
from tabench.cli import main


def _network(**overrides) -> Network:
    base = dict(
        name="probe",
        n_nodes=2,
        n_zones=2,
        first_thru_node=1,
        init_node=np.array([1], dtype=np.int64),
        term_node=np.array([2], dtype=np.int64),
        capacity=np.ones(1),
        length=np.zeros(1),
        free_flow_time=np.ones(1),
        b=np.full(1, 0.15),
        power=np.full(1, 4.0),
        toll=np.zeros(1),
        link_type=np.ones(1, dtype=np.int64),
    )
    base.update(overrides)
    return Network(**base)


def test_negative_b_rejected_at_construction():
    with pytest.raises(ValueError, match="nonnegative"):
        _network(b=np.array([-0.5]))


def test_negative_power_rejected_at_construction():
    with pytest.raises(ValueError, match="power"):
        _network(power=np.array([-1.0]))


def test_link_cost_derivative_finite_and_silent_for_fractional_power():
    """0 < p < 1 at subnormal flows must clamp, not overflow with a warning."""
    net = _network(power=np.array([0.01]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        h = net.link_cost_derivative(np.array([5e-324]))
    assert np.all(np.isfinite(h))


def test_sue_certification_censors_on_label_saturation():
    """Audit-passing flows whose costs saturate Dijkstra labels must censor
    the SUE residual, never crash the scoring loop."""
    network = Network(
        name="saturated",
        n_nodes=4,
        n_zones=2,
        first_thru_node=1,
        init_node=np.array([1, 3, 4], dtype=np.int64),
        term_node=np.array([3, 4, 2], dtype=np.int64),
        capacity=np.array([1e-4, 1.0, 1.0]),
        length=np.zeros(3),
        free_flow_time=np.array([1.0, 1e-6, 1.0]),
        b=np.array([0.15, 0.0, 0.0]),
        power=np.array([4.0, 1.0, 1.0]),
        toll=np.zeros(3),
        link_type=np.ones(3, dtype=np.int64),
    )
    od = np.zeros((2, 2))
    od[0, 1] = 4.0
    scenario = Scenario(
        name="saturated", network=network, demand=Demand(matrix=od), sue_theta=0.5
    )
    v = np.array([4.0, 4.0, 4.0])  # the unique feasible routing
    metrics = Evaluator(scenario).evaluate(v)  # must not raise
    assert metrics["feasible"] == 1.0  # the audit passes; only SUE is censored
    assert math.isnan(metrics["sue_fixed_point_residual"])


def test_sue_msa_honors_convergence_target():
    scenario = two_route_scenario()
    trace = Trace()
    DialSUEModel().solve(
        scenario, Budget(iterations=500, target_relative_gap=1e-3), RngBundle(0), trace
    )
    assert trace.final.coords.iterations < 500
    assert trace.final.self_report["sue_fixed_point_residual"] <= 1e-3


def test_output_stems_encode_target_gap_and_scenario_instance(tmp_path):
    """Runs differing only in --target-gap or theta must not share filenames."""
    braess = braess_scenario()
    run_experiment(braess, [FrankWolfeModel()], Budget(iterations=50), out_dir=tmp_path)
    run_experiment(
        braess,
        [FrankWolfeModel()],
        Budget(iterations=50, target_relative_gap=1e-3),
        out_dir=tmp_path,
    )
    run_experiment(
        two_route_scenario(sue_theta=0.5), [DialSUEModel()], Budget(iterations=5),
        out_dir=tmp_path,
    )
    run_experiment(
        two_route_scenario(sue_theta=2.0), [DialSUEModel()], Budget(iterations=5),
        out_dir=tmp_path,
    )
    assert len(list(tmp_path.glob("*.csv"))) == 4
    assert len(list(tmp_path.glob("*.manifest.json"))) == 4


def test_manifest_pins_sue_theta(tmp_path):
    result = run_experiment(
        two_route_scenario(sue_theta=2.0), [DialSUEModel()], Budget(iterations=5),
        out_dir=tmp_path,
    )
    assert result.manifest["scenario_sue_theta"] == 2.0
    manifest = json.loads(next(tmp_path.glob("*.manifest.json")).read_text())
    assert manifest["scenario_sue_theta"] == 2.0


def test_cli_sue_card_theta_override_drops_stale_reference(tmp_path):
    card = tmp_path / "card.yaml"
    card.write_text("scenario: tworoute\nsue:\n  theta: 2.0\nbudgets:\n  iterations: 20\n")
    out = tmp_path / "out"
    assert main(["run", "--config", str(card), "--models", "sue-msa", "--out", str(out)]) == 0
    csv_text = next(out.glob("*.csv")).read_text()
    header, first_row = csv_text.splitlines()[:2]
    rmse_index = header.split(",").index("flow_rmse_vs_reference")
    assert first_row.split(",")[rmse_index] == ""  # stale theta=0.5 oracle dropped


def test_cli_invalid_yaml_card_exits_cleanly(tmp_path, capsys):
    card = tmp_path / "bad.yaml"
    card.write_text("scenario: [unclosed\n")
    assert main(["run", "--config", str(card)]) == 2
    assert "invalid scenario card YAML" in capsys.readouterr().err


def test_cli_null_budgets_section_defaults(tmp_path):
    card = tmp_path / "nullbudgets.yaml"
    card.write_text("scenario: braess\nbudgets:\n")
    assert main(["run", "--config", str(card), "--models", "fw"]) == 0


def test_cli_sue_summary_prints_ranking_metric(capsys):
    assert main(["run", "--scenario", "tworoute", "--models", "sue-msa,fw",
                 "--iterations", "20"]) == 0
    out = capsys.readouterr().out
    assert "SUE residual" in out
    assert "ranked by the certified SUE residual" in out
