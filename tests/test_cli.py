"""CLI smoke tests (offline scenarios only)."""

import json
from pathlib import Path

from tabench.cli import main


def test_list_runs():
    assert main(["list"]) == 0


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
