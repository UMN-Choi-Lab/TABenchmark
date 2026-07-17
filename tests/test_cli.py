"""CLI smoke tests (offline scenarios only)."""

import json
import os
from pathlib import Path

import pytest

from tabench.cli import main


def test_list_runs():
    assert main(["list"]) == 0


def test_run_sue_card_without_theta_exit2(tmp_path: Path):
    """A sue block declaring family/r_cert but no theta is rejected (exit 2):
    theta defines (and hashes) the SUE instance, so silently defaulting it
    would run the wrong benchmark and blame the scenario."""
    card = tmp_path / "bad-probit.yaml"
    card.write_text(
        "scenario: tworoute\ntasks: [t1_sue]\nsue: {family: probit, r_cert: 64}\n"
    )
    assert main(["run", "--config", str(card), "--models", "sue-probit-msa"]) == 2


def test_run_probit_card_dispatch(tmp_path: Path, capsys):
    """A probit sue card runs end to end (exit 0), prints the ADR-003 tie rule,
    and the manifest records the pinned r_cert and a macrorep bootstrap CI."""
    card = tmp_path / "tworoute-probit.yaml"
    card.write_text(
        "scenario: tworoute\ntasks: [t1_sue]\n"
        "sue: {theta: 0.1, family: probit, r_cert: 64}\n"
    )
    out = tmp_path / "results"
    code = main(
        ["run", "--config", str(card), "--models", "sue-probit-msa",
         "--iterations", "8", "--macroreps", "3", "--out", str(out)]
    )
    assert code == 0
    assert "sue_residual_floor" in capsys.readouterr().out
    manifest = json.loads(next(out.glob("*.manifest.json")).read_text())
    assert manifest["certificate_r_cert"] == 64
    assert manifest["scenario_sue_family"] == "probit"
    assert manifest["bootstrap"]["sue-probit-msa"]["n_macroreps"] == 3


def _write_t2_card(tmp_path: Path) -> Path:
    card = tmp_path / "braess-t2.yaml"
    card.write_text(
        "scenario: braess\n"
        "tasks: [t2_estimation]\n"
        "estimation:\n"
        "  sensors: {kind: explicit, links: [1, 2, 3]}\n"
        "  heldout: {kind: explicit, links: [0, 4]}\n"
        "  n_periods: 1\n"
        "  noise: none\n"
        "  prior: {kind: stale, cv: 0.0}\n"
        "budgets: {sp_calls: 300}\n"
    )
    return card


def test_run_t2_card_dispatch(tmp_path: Path, capsys):
    """A t2_estimation card runs end to end via cli.main (exit 0), prints the
    identifiability line, and writes a CSV + T2 manifest. No --models, so this
    also exercises the T2 default-estimator resolution (the sentinel)."""
    card = _write_t2_card(tmp_path)
    out = tmp_path / "results"
    code = main(["run", "--config", str(card), "--out", str(out)])
    assert code == 0
    captured = capsys.readouterr()
    assert "identifiability:" in captured.out
    assert len(list(out.glob("*.csv"))) == 1
    manifest = json.loads(next(out.glob("*.manifest.json")).read_text())
    assert manifest["task"] == "t2_estimation"
    assert "identifiability" in manifest


def test_run_t2_unknown_estimator_exit2(tmp_path: Path):
    """An unknown estimator name on a T2 card exits 2."""
    card = _write_t2_card(tmp_path)
    assert main(["run", "--config", str(card), "--models", "nonsense"]) == 2


def test_run_t2_explicit_t1_model_exit2(tmp_path: Path):
    """An explicit T1 model name (aon) on a T2 card is taken literally and
    errors cleanly (exit 2), rather than being silently replaced by the default
    estimator set (the --models sentinel fix, ADR-002)."""
    card = _write_t2_card(tmp_path)
    assert main(["run", "--config", str(card), "--models", "aon"]) == 2


def test_run_braess_writes_outputs(tmp_path: Path):
    out = tmp_path / "results"
    code = main(
        [
            "run",
            "--scenario",
            "braess",
            "--models",
            "aon,msa,fw",
            "--iterations",
            "50",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    csv_files = list(out.glob("*.csv"))
    manifests = list(out.glob("*.manifest.json"))
    assert len(csv_files) == 1
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["scenario"] == "braess"
    assert set(manifest["models"]) == {"aon", "msa", "fw"}
    assert len(manifest["scenario_hash"]) == 64


# --- BO4Mob T2 estimation card (adr-041) ---------------------------------------
# The card carries `bo4mob_instance`, NOT `scenario` — a bo4mob key is never a
# load_scenario Scenario (adr-034), so the CLI must dispatch it BEFORE the
# scenario-card shape check and before any load_scenario call.


def _write_bo4mob_card(tmp_path: Path, instance: str = "1ramp") -> Path:
    card = tmp_path / "bo4mob.yaml"
    card.write_text(
        "tasks: [t2_bo4mob_estimation]\n"
        f'bo4mob_instance: "{instance}"\n'
        "budgets: {sp_calls: 1}\n"
    )
    return card


def test_run_bo4mob_missing_instance_exit2(tmp_path: Path):
    """A bo4mob card without `bo4mob_instance` exits 2 (before any sumo/runner)."""
    card = tmp_path / "bad.yaml"
    card.write_text("tasks: [t2_bo4mob_estimation]\nbudgets: {sp_calls: 1}\n")
    assert main(["run", "--config", str(card)]) == 2


def test_run_bo4mob_unknown_estimator_exit2(tmp_path: Path):
    """An unknown bo4mob estimator name exits 2 (before the runner is reached)."""
    card = _write_bo4mob_card(tmp_path)
    assert main(["run", "--config", str(card), "--models", "nonsense"]) == 2


def test_run_bo4mob_hpc_instance_exit2(tmp_path: Path):
    """5fullRegion is HPC-only: the card dispatches, the runner refuses to fetch it
    (Bo4MobHpcOnlyError), and the CLI exits 2 (never auto-runs the 74 MB instance)."""
    pytest.importorskip("sumo")
    card = _write_bo4mob_card(tmp_path, instance="5fullRegion")
    assert main(["run", "--config", str(card)]) == 2


def test_run_bo4mob_card_dispatch(tmp_path: Path, capsys):
    """A t2_bo4mob_estimation card runs end to end via cli.main (exit 0), prints the
    dual-benchmark disclosure + the heldout_nrmse ranking, and writes a CSV + manifest
    whose task/ranking_metric keys are the D2 ones. No `scenario` key — so this also
    confirms dispatch happens BEFORE load_scenario (a bo4mob key would KeyError there)."""
    pytest.importorskip("sumo")
    if not os.environ.get("TABENCH_REQUIRE_DATA"):
        try:
            from tabench.data.bo4mob import BO4MOB_REGISTRY, fetch_bo4mob, fetch_bo4mob_heldout

            fetch_bo4mob(BO4MOB_REGISTRY["1ramp"])
            fetch_bo4mob_heldout("1ramp")
        except Exception as exc:  # offline dev box
            pytest.skip(f"bo4mob 1ramp data unavailable: {exc}")
    card = _write_bo4mob_card(tmp_path)
    out = tmp_path / "results"
    code = main(["run", "--config", str(card), "--out", str(out)])
    assert code == 0
    captured = capsys.readouterr().out
    assert "heldout_nrmse" in captured  # ranked by the D2 column
    assert "does NOT reproduce BO4Mob's own SPSA/BO leaderboard" in captured
    manifest = json.loads(next(out.glob("*.manifest.json")).read_text())
    assert manifest["task"] == "t2_bo4mob_estimation"
    assert manifest["instance"] == "1ramp"
    assert manifest["heldout"]["ranking_metric"] == "heldout_nrmse"
