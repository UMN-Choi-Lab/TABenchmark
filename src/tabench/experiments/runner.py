"""Experiment runner: (scenario x model) grid with certified scoring and manifests.

For each pair the model runs under its budget, then the harness evaluator
recomputes metrics at every emitted checkpoint (P1). Results are written as
CSV (one row per checkpoint) plus a ``manifest.json`` recording the complete
provenance (P7): scenario hashes, seeds, package versions, environment.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy

import tabench

from ..core.budget import Budget
from ..core.capabilities import assert_fair_evaluation
from ..core.results import ResultBundle, Trace
from ..core.rng import SOURCE_BOOTSTRAP, SOURCE_EVALUATION, SOURCE_OBSERVATION, RngBundle
from ..core.scenario import Scenario
from ..metrics.flows import rmse
from ..metrics.gaps import Evaluator
from ..models.base import TrafficAssignmentModel

__all__ = ["ExperimentResult", "run_experiment"]

_CSV_FIELDS = [
    "scenario",
    "scenario_hash",
    "model",
    "macrorep",
    "iterations",
    "sp_calls",
    "wall_ms",
    "tstt",
    "sptt",
    "relative_gap",
    "average_excess_cost",
    "beckmann_objective",
    "node_balance_residual",
    "feasible",
    "sue_fixed_point_residual",
    "flow_rmse_vs_reference",
    "self_relative_gap",
    "self_sue_residual",
]


@dataclass
class ExperimentResult:
    """In-memory result of one grid run."""

    rows: list[dict[str, Any]]
    bundles: dict[tuple[str, str], ResultBundle]
    manifest: dict[str, Any] = field(default_factory=dict)


def _git_commit() -> str:
    """Best-effort git commit of the working tree (pip installs: unavailable)."""
    try:
        root = Path(__file__).resolve()
        out = subprocess.run(
            ["git", "-C", str(root.parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "unavailable"
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"


def _score_bundle(
    scenario: Scenario, bundle: ResultBundle, macrorep: int
) -> list[dict[str, Any]]:
    evaluator = Evaluator(scenario)
    scenario_hash = scenario.content_hash()
    reference = scenario.reference
    rows = []
    for state in bundle.trace:
        metrics = evaluator.evaluate(state.link_flows)
        row: dict[str, Any] = {
            "scenario": scenario.name,
            "scenario_hash": scenario_hash[:16],
            "model": bundle.model_name,
            "macrorep": macrorep,
            "iterations": state.coords.iterations,
            "sp_calls": state.coords.sp_calls,
            "wall_ms": round(state.coords.wall_ms, 3),
            **{k: metrics[k] for k in metrics},
            "flow_rmse_vs_reference": (
                rmse(state.link_flows, reference.link_flows) if reference is not None else ""
            ),
            "self_relative_gap": state.self_report.get("relative_gap", ""),
            "self_sue_residual": state.self_report.get("sue_fixed_point_residual", ""),
        }
        rows.append(row)
    return rows


def run_experiment(
    scenario: Scenario,
    models: list[TrafficAssignmentModel],
    budget: Budget,
    seed: int = 0,
    macroreps: int = 1,
    out_dir: str | Path | None = None,
) -> ExperimentResult:
    """Run every model on the scenario and certify all checkpoints.

    Deterministic models are run with ``macroreps=1`` regardless of the
    argument (deterministic track, P5).
    """
    names = [model.name for model in models]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(
            f"Duplicate model names {sorted(duplicates)}: results and manifests are "
            "keyed by name, so every instance needs a distinct one — e.g. "
            "`model.name = 'fw-xtol1e-4'` when sweeping factors."
        )

    rows: list[dict[str, Any]] = []
    bundles: dict[tuple[str, str], ResultBundle] = {}

    for model in models:
        assert_fair_evaluation(model.capabilities, scenario)
        reps = 1 if model.capabilities.deterministic else macroreps
        for m in range(reps):
            rng = RngBundle(root_seed=seed, macrorep=m)
            trace = Trace()
            bundle = model.solve(scenario, budget, rng, trace)
            bundles[(model.name, f"m{m}")] = bundle
            rows.extend(_score_bundle(scenario, bundle, m))

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario.name,
        "scenario_hash": scenario.content_hash(),
        "scenario_family": scenario.family,
        "scenario_sue_theta": scenario.sue_theta,
        "models": {
            model.name: {
                "capabilities": {
                    "paradigm": model.capabilities.paradigm,
                    "deterministic": model.capabilities.deterministic,
                    "provides_gap": model.capabilities.provides_gap,
                    "seedable": model.capabilities.seedable,
                    "trained_on": list(model.capabilities.trained_on),
                },
                "factors": model.factor_values,
            }
            for model in models
        },
        "budget": {
            "iterations": budget.iterations,
            "sp_calls": budget.sp_calls,
            "wall_seconds": budget.wall_seconds,
            "target_relative_gap": budget.target_relative_gap,
        },
        "seed": seed,
        "macroreps": macroreps,
        "rng": {
            "root_seed": seed,
            "schema": (
                "numpy SeedSequence spawn_key=(macrorep, source, replication) "
                "with Philox (see tabench.core.rng, P8)"
            ),
            "reserved_sources": {
                "observation": SOURCE_OBSERVATION,
                "evaluation": SOURCE_EVALUATION,
                "bootstrap": SOURCE_BOOTSTRAP,
            },
        },
        "runs": [
            {"model": key[0], "macrorep": key[1], "seed_info": bundle.seed_info}
            for key, bundle in bundles.items()
        ],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "tabench": tabench.__version__,
            "git_commit": _git_commit(),
        },
        "notes": (
            "wall_ms is recorded, never a ranking axis (P6); per-machine "
            "wall-clock calibration is planned for the stochastic track."
        ),
    }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Encode the experiment factors in the file name (BO4Mob convention)
        # so distinct runs can never silently overwrite each other. The short
        # content hash covers every instance-defining scenario field (e.g.
        # sue_theta), present and future.
        budget_part = "-".join(
            f"{axis}{value}"
            for axis, value in (
                ("it", budget.iterations),
                ("sp", budget.sp_calls),
                ("ws", budget.wall_seconds),
                ("gap", budget.target_relative_gap),
            )
            if value is not None
        )
        stem = (
            f"{scenario.name}-{scenario.content_hash()[:8]}_"
            f"{'-'.join(names)}_{budget_part}_seed-{seed}"
        )
        with open(out / f"{stem}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        with open(out / f"{stem}.manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    return ExperimentResult(rows=rows, bundles=bundles, manifest=manifest)
