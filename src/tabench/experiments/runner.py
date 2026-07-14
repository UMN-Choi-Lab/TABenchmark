"""Experiment runner: (scenario x model) grid with certified scoring and manifests.

For each pair the model runs under its budget, then the harness evaluator
recomputes metrics at every emitted checkpoint (P1). Results are written as
CSV (one row per checkpoint) plus a ``manifest.json`` recording the complete
provenance (P7): scenario hashes, seeds, package versions, environment.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
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
from ..core.rng import (
    SOURCE_BOOTSTRAP,
    SOURCE_EVALUATION,
    SOURCE_OBSERVATION,
    SOURCE_PRIOR,
    RngBundle,
)
from ..core.scenario import Demand, Scenario
from ..estimation._dynamic_map import (
    MAP_RECIPE,
    lagged_assignment_tensor,
    predict_interval_counts,
    stacked_tensor_map,
)
from ..estimation._proportions import active_pairs, proportion_matrix
from ..estimation.base import EstimationTask, ODEstimator, ODTrace
from ..estimation.dynamic_base import DynamicEstimationTask, DynamicODEstimator
from ..metrics.estimation import CERTIFICATE_DEFAULTS, ODCertifier
from ..metrics.estimation_dynamic import DynamicODCertifier
from ..metrics.flows import rmse
from ..metrics.gaps import Evaluator
from ..models.base import TrafficAssignmentModel
from ..models.frank_wolfe import BiconjugateFrankWolfeModel
from ..observe.levels import (
    DayToDayCounts,
    DynamicLinkCounts,
    LinkCounts,
    StalePriorOD,
    distinct_nonzero_columns,
)
from .bootstrap import bootstrap_ci

__all__ = [
    "ExperimentResult",
    "run_experiment",
    "run_estimation_experiment",
    "run_dynamic_estimation_experiment",
    "identifiability_report",
    "dynamic_identifiability_report",
]

# Sensor *placement* draws deterministically from this reserved substream so it
# stays independent of the per-macrorep count draws (replication 0..macroreps).
_SENSOR_PLACEMENT_REPLICATION = 1_000_000

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
    "sue_residual_se",
    "sue_residual_floor",
    "so_relative_gap",
    "so_average_excess_cost",
    "tstt_mc",
    "sptt_mc",
    "realized_demand",
    "flow_rmse_vs_reference",
    "self_relative_gap",
    "self_sue_residual",
    "self_so_relative_gap",
    "self_realized_demand",
    "proportionality_residual",
    "pas_proportionality_max",
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
    scenario: Scenario, bundle: ResultBundle, macrorep: int, evaluator: Evaluator
) -> list[dict[str, Any]]:
    scenario_hash = scenario.content_hash()
    reference = scenario.reference
    rows = []
    for state in bundle.trace:
        # Per-class flows (multiclass only; None elsewhere) let the harness
        # recompute the per-class VI residual from a first-class object (adr-013).
        metrics = evaluator.evaluate(state.link_flows, state.class_link_flows)
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
            "self_so_relative_gap": state.self_report.get("so_relative_gap", ""),
            # Elastic (variable) demand (ADR-005): `realized_demand` is scored
            # (recomputed as Sum_rs D_rs(u_rs) from the flows); the self_ column
            # is the model's own realized-demand report, provenance only.
            "self_realized_demand": state.self_report.get("realized_demand", ""),
            # TAPAS route-flow proportionality diagnostics (ADR-004): provenance
            # only, never scored -- a route-flow property invisible to link flows.
            "proportionality_residual": state.self_report.get("proportionality_residual", ""),
            "pas_proportionality_max": state.self_report.get("pas_proportionality_max", ""),
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
    so_metrics: bool | None = None,
    r_cert: int = 2000,
) -> ExperimentResult:
    """Run every model on the scenario and certify all checkpoints.

    Deterministic models are run with ``macroreps=1`` regardless of the
    argument (deterministic track, P5). Certified system-optimum columns
    (``so_metrics``) are enabled automatically when the grid contains a
    ``static_so`` model, uniformly for every model in the run; pass
    ``so_metrics=True``/``False`` to override.

    ``r_cert`` is the probit-SUE certificate's pinned Monte Carlo sample count
    (adr-003); it is ignored for non-probit scenarios and never hashed into the
    instance (certificate protocol, not instance data).
    """
    names = [model.name for model in models]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(
            f"Duplicate model names {sorted(duplicates)}: results and manifests are "
            "keyed by name, so every instance needs a distinct one — e.g. "
            "`model.name = 'fw-xtol1e-4'` when sweeping factors."
        )

    if so_metrics is None:
        so_metrics = any(m.capabilities.paradigm == "static_so" for m in models)
    # The certificate pins its evaluation stream on root_seed = the run seed, so
    # every macrorep and model is certified against one common sampled map.
    evaluator = Evaluator(scenario, so_metrics=so_metrics, root_seed=seed, r_cert=r_cert)

    rows: list[dict[str, Any]] = []
    bundles: dict[tuple[str, str], ResultBundle] = {}

    for model in models:
        assert_fair_evaluation(model.capabilities, scenario)
        # A UE best-known oracle does not apply to SO-goal models: their
        # flow_rmse_vs_reference stays blank rather than misleading.
        score_scenario = (
            dataclasses.replace(scenario, reference=None)
            if model.capabilities.paradigm == "static_so"
            else scenario
        )
        reps = 1 if model.capabilities.deterministic else macroreps
        for m in range(reps):
            rng = RngBundle(root_seed=seed, macrorep=m)
            trace = Trace()
            bundle = model.solve(scenario, budget, rng, trace)
            bundles[(model.name, f"m{m}")] = bundle
            rows.extend(_score_bundle(score_scenario, bundle, m, evaluator))

    # Stochastic track (P5/P8): aggregate the final certified metric across
    # macroreps into a percentile bootstrap CI (adr-003 Decision 4). Only
    # non-deterministic models run more than once carry one; deterministic
    # models have a single trajectory and no sampling spread.
    bootstrap_block: dict[str, Any] = {}
    if macroreps > 1:
        metric_key = (
            "sue_fixed_point_residual"
            if scenario.sue_family == "probit"
            else "relative_gap"
        )
        for model in models:
            if model.capabilities.deterministic:
                continue
            finals: dict[int, float] = {}
            for row in rows:
                if row["model"] == model.name:
                    finals[row["macrorep"]] = row.get(metric_key, float("nan"))
            values = np.array([finals[m] for m in sorted(finals)], dtype=float)
            if values.size > 1 and np.isfinite(values).all():
                ci = bootstrap_ci(values, root_seed=seed)
                bootstrap_block[model.name] = {
                    "metric": metric_key,
                    "point": ci.point,
                    "lo": ci.lo,
                    "hi": ci.hi,
                    "level": ci.level,
                    "n_macroreps": int(values.size),
                }

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario.name,
        "scenario_hash": scenario.content_hash(),
        "scenario_family": scenario.family,
        "scenario_sue_theta": scenario.sue_theta,
        "scenario_sue_family": scenario.sue_family,
        # Certificate protocol, recorded not hashed (adr-003): the pinned MC
        # sample count only matters for probit scenarios.
        "certificate_r_cert": r_cert if scenario.sue_family == "probit" else None,
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
        # Percentile bootstrap CIs of the final certified metric across
        # macroreps (empty unless macroreps > 1 with a stochastic model).
        "bootstrap": bootstrap_block,
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


# --------------------------------------------------------------------------- T2

_EST_CSV_FIELDS = [
    "scenario",
    "scenario_hash",
    "task_hash",
    "estimator",
    "macrorep",
    "iterations",
    "sp_calls",
    "wall_ms",
    "od_feasible",
    "obs_count_rmse",
    "obs_mean_count_rmse",
    "oracle_obs_count_rmse",
    "heldout_count_rmse",
    "oracle_heldout_count_rmse",
    "heldout_flow_rmse",
    "od_rmse",
    "od_nrmse",
    "total_demand_error",
    "od_identifiable",
    "certificate_gap",
    "certificate_converged",
    "self_obs_count_rmse",
]


def _draw_sensors(
    n_links: int, spec: dict[str, Any], rng: np.random.Generator, exclude: Any = None
) -> np.ndarray:
    """Draw a sensor set, either explicit links or a random coverage fraction.

    ``exclude`` links (the observed set, when drawing the held-out set) are
    removed from the candidate pool so the two sets stay disjoint (P3).
    """
    if spec.get("kind", "random") == "explicit":
        return np.sort(np.asarray(spec["links"], dtype=np.int64))
    coverage = float(spec.get("coverage", 0.3))
    excluded = set() if exclude is None else {int(x) for x in np.asarray(exclude).tolist()}
    available = np.array([i for i in range(n_links) if i not in excluded], dtype=np.int64)
    if available.size == 0:
        return np.array([], dtype=np.int64)
    n = min(max(1, round(coverage * n_links)), available.size)
    return np.sort(rng.choice(available, size=n, replace=False))


def identifiability_report(
    network: Any, truth_demand: Demand, obs_sensors: np.ndarray, k_inner: int = 40
) -> dict[str, Any]:
    """Per-(sensor set, task) identifiability report (ADR-002 Decision 4).

    Builds the truth-side proportion matrix ``P*`` (harness-only; the estimator
    never sees it) and reports Hazelton's Prop. 1 column condition, the
    mean-count ``linear_identifiable`` rank test (dense only for
    ``n_zones <= 100``), and the unseen/confounded-pair diagnostics.
    """
    obs_sensors = np.asarray(obs_sensors, dtype=np.int64)
    pairs = active_pairs(truth_demand.matrix)
    n_active = len(pairs)
    report: dict[str, Any] = {
        "n_active_pairs": n_active,
        "n_obs_sensors": int(obs_sensors.size),
    }
    if n_active == 0:
        report.update(
            hazelton_condition=False,
            n_unseen_pairs=0,
            n_confounded_pairs=0,
            linear_identifiable=False,
            rank_not_computed=False,
        )
        return report
    p_star, _, _ = proportion_matrix(network, truth_demand, k_inner, pairs=pairs)
    sub = p_star[obs_sensors]
    report["hazelton_condition"] = (
        bool(distinct_nonzero_columns(sub)) if obs_sensors.size else False
    )
    zero_cols = ~sub.any(axis=0)
    report["n_unseen_pairs"] = int(zero_cols.sum())
    nz = sub[:, ~zero_cols]
    if nz.shape[1]:
        _, inverse, counts = np.unique(
            nz, axis=1, return_inverse=True, return_counts=True
        )
        class_size = counts[inverse.ravel()]
        report["n_confounded_pairs"] = int((class_size > 1).sum())
    else:
        report["n_confounded_pairs"] = 0
    if truth_demand.n_zones <= 100:
        rank = int(np.linalg.matrix_rank(sub)) if obs_sensors.size else 0
        report["rank"] = rank
        report["linear_identifiable"] = bool(rank == n_active)
        report["rank_not_computed"] = False
    else:
        report["linear_identifiable"] = False
        report["rank_not_computed"] = True
    return report


def _score_estimator(
    estimator: ODEstimator,
    task: EstimationTask,
    budget: Budget,
    rng: RngBundle,
    certifier: ODCertifier,
    macrorep: int,
    scenario: Scenario,
    rows: list[dict[str, Any]],
    bundles: dict[tuple[str, str], Any],
) -> None:
    trace = ODTrace()
    bundle = estimator.estimate(task, budget, rng, trace)
    bundles[(estimator.name, f"m{macrorep}")] = bundle
    scenario_hash = scenario.content_hash()[:16]
    task_hash = task.content_hash()[:16]
    for state in trace:
        metrics = certifier.certify(state.od_matrix)
        row: dict[str, Any] = {
            "scenario": scenario.name,
            "scenario_hash": scenario_hash,
            "task_hash": task_hash,
            "estimator": estimator.name,
            "macrorep": macrorep,
            "iterations": state.coords.iterations,
            "sp_calls": state.coords.sp_calls,
            "wall_ms": round(state.coords.wall_ms, 3),
            **metrics,
            "self_obs_count_rmse": state.self_report.get("obs_count_rmse", ""),
        }
        rows.append(row)


def run_estimation_experiment(
    scenario: Scenario,
    estimators: list[ODEstimator],
    budget: Budget,
    seed: int = 0,
    macroreps: int = 1,
    out_dir: str | Path | None = None,
    estimation: dict[str, Any] | None = None,
) -> ExperimentResult:
    """Run every estimator on the scenario's T2 task and certify all checkpoints.

    The dataset is stochastic (counts and/or a Gamma-perturbed prior), so
    ``reps = macroreps`` whenever ``noise != 'none'`` or the prior ``cv > 0``,
    even for deterministic estimators; a deterministic estimator on a fully
    deterministic task collapses to one rep (ADR-002 Decision 5). Every emitted
    OD checkpoint is certified through the pinned reference assignment; nothing
    is thinned.
    """
    estimation = dict(estimation or {})
    network = scenario.network
    n_links = network.n_links

    if scenario.sue_theta is not None:
        # T2 certifies against the pinned deterministic-UE assignment only; SUE-
        # pinned certificates are deferred (ADR-002 Decision 2). Enforce the
        # semantic in the runner, not just the CLI, so the public API is safe.
        raise ValueError(
            f"scenario {scenario.name!r} is an SUE instance (sue_theta="
            f"{scenario.sue_theta}); T2 estimation certifies against the pinned "
            "deterministic-UE assignment only (ADR-002 defers SUE-pinned "
            "certificates). Pass the UE instance, e.g. "
            "dataclasses.replace(scenario, sue_theta=None, reference=None)."
        )

    names = [e.name for e in estimators]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"Duplicate estimator names {sorted(duplicates)}")

    certificate = dict(CERTIFICATE_DEFAULTS)
    certificate.update(estimation.get("certificate") or {})
    if str(certificate["assignment"]) != "bfw":
        # The pin's model component is hashed and recorded, so it must be
        # enforced, not decorative (ADR-002 Decision 2). Only bfw ships in v1.
        raise ValueError(
            f"unsupported certificate assignment {certificate['assignment']!r}: "
            "only 'bfw' is a supported certificate pin this sprint (SUE-pinned "
            "certificates are deferred, ADR-002)"
        )

    pin_solver = BiconjugateFrankWolfeModel(
        line_search_xtol=float(certificate["line_search_xtol"])
    )
    pin_budget = Budget(
        iterations=int(certificate["max_iterations"]),
        target_relative_gap=float(certificate["target_relative_gap"]),
    )
    truth_trace = Trace()
    pin_solver.solve(scenario, pin_budget, RngBundle(seed), truth_trace)
    oracle_flows = truth_trace.final.link_flows

    place_rng = RngBundle(root_seed=seed, macrorep=0).generator(
        SOURCE_OBSERVATION, replication=_SENSOR_PLACEMENT_REPLICATION
    )
    obs_sensors = _draw_sensors(
        n_links, estimation.get("sensors") or {"kind": "random", "coverage": 0.3}, place_rng
    )
    ho_cfg = estimation.get("heldout") or {"kind": "random", "coverage": 0.1}
    heldout_sensors = _draw_sensors(n_links, ho_cfg, place_rng, exclude=obs_sensors)

    overlap = np.intersect1d(obs_sensors, heldout_sensors)
    if overlap.size:
        # heldout_count_rmse is the ranking column and must be out of sample;
        # disjointness is enforced by the harness, never by convention (P7). The
        # exclude= path only covers random draws, so validate explicit sets here.
        raise ValueError(
            "held-out sensors must be disjoint from observed sensors "
            "(ADR-002 heldout_count_rmse contract, P7); overlapping links: "
            f"{overlap.tolist()}"
        )

    n_periods = int(estimation.get("n_periods", 1))
    ho_periods = int(ho_cfg.get("n_periods", n_periods))
    # Digest of the held-out design folded into the task hash without exposing
    # the held-out sensor identities to the estimator (ADR-002 Decision 1).
    ho_hash = hashlib.sha256()
    ho_hash.update(np.ascontiguousarray(np.sort(heldout_sensors), dtype=np.int64).tobytes())
    ho_hash.update(f"ho_periods={ho_periods};".encode())
    heldout_digest = ho_hash.hexdigest()
    noise = estimation.get("noise", "poisson")
    # Davis-Nihan day-to-day dials (ADR-012); ignored for other noise modes.
    dn_population_scale = float(estimation.get("population_scale", 50.0))
    dn_rho = float(estimation.get("rho", 0.5))
    prior_cfg = estimation.get("prior") or {"kind": "stale", "cv": 0.3}
    prior_cv = float(prior_cfg.get("cv", 0.3))
    id_k_inner = int(estimation.get("identifiability_k_inner", 40))

    ident = identifiability_report(network, scenario.demand, obs_sensors, k_inner=id_k_inner)

    stochastic = (noise != "none") or (prior_cv > 0.0)
    n_data = macroreps if stochastic else 1

    data_reps: list[tuple[EstimationTask, ODCertifier]] = []
    for dr in range(n_data):
        rb = RngBundle(root_seed=seed, macrorep=dr)
        prior_ds = StalePriorOD(cv=prior_cv).observe(
            scenario, oracle_flows, rb.generator(SOURCE_PRIOR)
        )
        prior = Demand(matrix=prior_ds.payload["prior_od"])
        if noise == "day_to_day":
            # Davis-Nihan large-population VAR(1) count series (ADR-012), centered
            # on the UE loading so the pinned-UE certifier still applies.
            obs_ds = DayToDayCounts(
                obs_sensors, n_periods, dn_population_scale, dn_rho, id_k_inner
            ).observe(scenario, oracle_flows, rb.generator(SOURCE_OBSERVATION))
            ho_ds = DayToDayCounts(
                heldout_sensors, ho_periods, dn_population_scale, dn_rho, id_k_inner
            ).observe(scenario, oracle_flows, rb.generator(SOURCE_EVALUATION))
        else:
            obs_ds = LinkCounts(obs_sensors, n_periods, noise).observe(
                scenario, oracle_flows, rb.generator(SOURCE_OBSERVATION)
            )
            ho_ds = LinkCounts(heldout_sensors, ho_periods, noise).observe(
                scenario, oracle_flows, rb.generator(SOURCE_EVALUATION)
            )
        task = EstimationTask(
            name=scenario.name,
            network=network,
            prior=prior,
            dataset=obs_ds,
            identifiability=ident,
            scenario_hash=scenario.content_hash(),
            certificate=certificate,
            seed=seed,
            heldout_digest=heldout_digest,
        )
        certifier = ODCertifier(
            scenario,
            obs_sensors,
            heldout_sensors,
            obs_ds.payload["counts"],
            ho_ds.payload["counts"],
            oracle_flows,
            ident,
            certificate,
        )
        data_reps.append((task, certifier))

    rows: list[dict[str, Any]] = []
    bundles: dict[tuple[str, str], Any] = {}
    for est in estimators:
        assert_fair_evaluation(est.capabilities, scenario)
        if stochastic:
            for dr in range(n_data):
                task, certifier = data_reps[dr]
                _score_estimator(
                    est, task, budget, RngBundle(seed, macrorep=dr),
                    certifier, dr, scenario, rows, bundles,
                )
        else:
            task, certifier = data_reps[0]
            reps = 1 if est.capabilities.deterministic else macroreps
            for m in range(reps):
                _score_estimator(
                    est, task, budget, RngBundle(seed, macrorep=m),
                    certifier, m, scenario, rows, bundles,
                )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": "t2_estimation",
        "scenario": scenario.name,
        "scenario_hash": scenario.content_hash(),
        "scenario_family": scenario.family,
        "estimation": {
            "data_level": estimation.get("data_level", "link_counts"),
            "sensors": {"links": obs_sensors.tolist(), "n": int(obs_sensors.size)},
            "heldout": {"links": heldout_sensors.tolist(), "n": int(heldout_sensors.size)},
            "n_periods": n_periods,
            "heldout_n_periods": ho_periods,
            "noise": noise,
            **(
                {"population_scale": dn_population_scale, "rho": dn_rho}
                if noise == "day_to_day"
                else {}
            ),
            "prior": {"kind": prior_cfg.get("kind", "stale"), "cv": prior_cv},
            "stochastic": stochastic,
        },
        "certificate": certificate,
        "identifiability": ident,
        "estimators": {
            e.name: {
                "capabilities": {
                    "paradigm": e.capabilities.paradigm,
                    "deterministic": e.capabilities.deterministic,
                    "seedable": e.capabilities.seedable,
                    "inputs_required": sorted(e.capabilities.inputs_required),
                    "outputs": sorted(e.capabilities.outputs),
                    "trained_on": list(e.capabilities.trained_on),
                },
                "factors": e.factor_values,
            }
            for e in estimators
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
                "prior": SOURCE_PRIOR,
            },
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "tabench": tabench.__version__,
            "git_commit": _git_commit(),
        },
        "notes": (
            "T2 estimation: each emitted OD matrix is certified through the pinned "
            "reference assignment; count-fit and OD-fit are a pair, ranked by "
            "heldout_count_rmse (ADR-002)."
        ),
    }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        budget_part = "-".join(
            f"{axis}{value}"
            for axis, value in (
                ("it", budget.iterations),
                ("sp", budget.sp_calls),
                ("ws", budget.wall_seconds),
            )
            if value is not None
        )
        # task_hash[:8] pins the estimation block (sensors, held-out digest,
        # dataset dials, certificate) so card variants sharing a scenario/budget/
        # seed can never silently overwrite each other's CSV (ADR-002 Decision 1).
        task_hash8 = data_reps[0][0].content_hash()[:8]
        stem = (
            f"{scenario.name}-{scenario.content_hash()[:8]}_t2-{task_hash8}_"
            f"{'-'.join(names)}_{budget_part}_seed-{seed}"
        )
        with open(out / f"{stem}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_EST_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        with open(out / f"{stem}.manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    return ExperimentResult(rows=rows, bundles=bundles, manifest=manifest)


# ----------------------------------------------------------------- T2 (dynamic)

_DYN_CSV_FIELDS = [
    "scenario",
    "scenario_hash",
    "task_hash",
    "estimator",
    "macrorep",
    "iterations",
    "sp_calls",
    "wall_ms",
    "od_feasible",
    "obs_count_rmse",
    "obs_mean_count_rmse",
    "oracle_obs_count_rmse",
    "heldout_count_rmse",
    "oracle_heldout_count_rmse",
    "heldout_flow_rmse",
    "od_rmse",
    "od_nrmse",
    "total_demand_error",
    "profile_rmse",
    "od_identifiable",
    "map_recipe",
    "n_slices",
    "n_intervals",
    "self_obs_count_rmse",
]

# Max stacked-map columns for which the dense SVD rank test is run (dynamic
# cards are small; big instances would carry hazelton_condition + a flag).
_DYN_RANK_CAP = 4000
# Practical identifiability floor on the stacked map's smallest singular value:
# below it, OD shifts exist whose count signature is smaller than any realistic
# noise, so od_identifiable must not be asserted (adr-023 review).
_DYN_SIGMA_TOL = 1e-6


def dynamic_identifiability_report(
    full_map: np.ndarray,
    obs_sensors: np.ndarray,
    pairs: list[tuple[int, int]],
    n_slices: int,
    n_intervals: int,
) -> dict[str, Any]:
    """Identifiability of the stacked within-day obs map (ADR-023).

    Builds the block-lower-banded stacked map ``A`` (``(T*S, H*P)``) restricted to
    the observed sensors and reports its numerical SVD rank **and smallest
    singular value** (the map is generated by the same linear recipe the counts
    are, so this is structural, but the rank itself is a float computation at a
    documented tolerance — never "exact"), Hazelton's distinct-nonzero-column
    condition, the number of horizon-**truncated** slices (all-zero column blocks —
    the genuinely new edge), and the number of **confounded** columns (duplicate
    column classes, including cross-slice temporal confounds).
    ``linear_identifiable`` (full column rank AND ``sigma_min > 1e-6``) gates
    ``od_identifiable`` in the certificate.
    """
    obs_sensors = np.asarray(obs_sensors, dtype=np.int64)
    n_pairs = len(pairs)
    n_cols = int(n_slices) * n_pairs
    report: dict[str, Any] = {
        "n_active_pairs": n_pairs,
        "n_slices": int(n_slices),
        "n_intervals": int(n_intervals),
        "n_obs_sensors": int(obs_sensors.size),
        "n_columns": n_cols,
    }
    obs_map = full_map[:, obs_sensors, :]
    a = stacked_tensor_map(obs_map, n_slices, n_intervals)
    if n_cols == 0 or a.shape[0] == 0:
        report.update(
            hazelton_condition=False,
            n_truncated_slices=int(n_slices),
            n_confounded_columns=0,
            rank=0,
            linear_identifiable=False,
            rank_not_computed=False,
        )
        return report
    truncated = sum(
        1
        for h in range(int(n_slices))
        if not np.any(a[:, h * n_pairs:(h + 1) * n_pairs])
    )
    report["n_truncated_slices"] = int(truncated)
    nonzero = a[:, a.any(axis=0)]
    if nonzero.shape[1]:
        _, inverse, counts = np.unique(
            nonzero, axis=1, return_inverse=True, return_counts=True
        )
        class_size = counts[inverse.ravel()]
        report["n_confounded_columns"] = int((class_size > 1).sum())
    else:
        report["n_confounded_columns"] = 0
    report["hazelton_condition"] = bool(distinct_nonzero_columns(a))
    if n_cols <= _DYN_RANK_CAP:
        # numerical SVD rank, NOT symbolically exact (adr-023 review): the gate
        # reports sigma_min explicitly and requires it above a documented
        # practical tolerance — full nominal rank with sigma_min ~ 0 means OD
        # shifts exist that are count-invisible below any realistic noise, and
        # a flaky default matrix_rank tolerance must not decide identifiability
        svals = np.linalg.svd(a, compute_uv=False)
        smax = float(svals[0]) if svals.size else 0.0
        rank = int((svals > max(a.shape) * np.finfo(np.float64).eps * max(smax, 1.0)).sum())
        sigma_min = float(svals[-1]) if svals.size == n_cols else 0.0
        report["rank"] = rank
        report["sigma_min"] = sigma_min
        report["linear_identifiable"] = bool(rank == n_cols and sigma_min > _DYN_SIGMA_TOL)
        report["rank_not_computed"] = False
    else:
        report["sigma_min"] = float("nan")
        report["linear_identifiable"] = False
        report["rank_not_computed"] = True
    return report


def _default_profile(n_slices: int, spec: Any) -> np.ndarray:
    """Per-slice demand multipliers ``rho_h`` (the within-day profile dial).

    An explicit list is taken literally; otherwise a gentle non-flat ramp so the
    slices genuinely differ (a flat profile would make the time axis degenerate).
    """
    if isinstance(spec, (list, tuple)):
        rho = np.asarray(spec, dtype=np.float64)
        if rho.shape != (n_slices,):
            raise ValueError(
                f"estimation.profile must have length n_slices={n_slices}, got {rho.shape}"
            )
        if np.any(rho < 0) or not np.all(np.isfinite(rho)):
            raise ValueError("estimation.profile multipliers must be finite and >= 0")
        return rho
    if n_slices == 1:
        return np.ones(1, dtype=np.float64)
    return np.linspace(0.6, 1.4, n_slices, dtype=np.float64)


def _score_dynamic_estimator(
    estimator: DynamicODEstimator,
    task: DynamicEstimationTask,
    budget: Budget,
    rng: RngBundle,
    certifier: DynamicODCertifier,
    macrorep: int,
    scenario: Scenario,
    map_recipe: str,
    n_slices: int,
    n_intervals: int,
    rows: list[dict[str, Any]],
    bundles: dict[tuple[str, str], Any],
) -> None:
    trace = ODTrace()
    bundle = estimator.estimate(task, budget, rng, trace)
    bundles[(estimator.name, f"m{macrorep}")] = bundle
    scenario_hash = scenario.content_hash()[:16]
    task_hash = task.content_hash()[:16]
    for state in trace:
        metrics = certifier.certify(state.od_matrix)
        row: dict[str, Any] = {
            "scenario": scenario.name,
            "scenario_hash": scenario_hash,
            "task_hash": task_hash,
            "estimator": estimator.name,
            "macrorep": macrorep,
            "iterations": state.coords.iterations,
            "sp_calls": state.coords.sp_calls,
            "wall_ms": round(state.coords.wall_ms, 3),
            **metrics,
            "map_recipe": map_recipe,
            "n_slices": n_slices,
            "n_intervals": n_intervals,
            "self_obs_count_rmse": state.self_report.get("obs_count_rmse", ""),
        }
        rows.append(row)


def run_dynamic_estimation_experiment(
    scenario: Scenario,
    estimators: list[DynamicODEstimator],
    budget: Budget,
    seed: int = 0,
    macroreps: int = 1,
    out_dir: str | Path | None = None,
    estimation: dict[str, Any] | None = None,
) -> ExperimentResult:
    """Run every within-day dynamic estimator on the scenario's T2d task (ADR-023).

    Parallel to :func:`run_estimation_experiment`: same reserved RNG streams,
    sensor-placement substream, obs/held-out disjointness guard, held-out digest,
    and stochastic-implies-reps semantics. The estimand is the ``(H, Z, Z)``
    profile ``d*_h = rho_h * scenario.demand.matrix`` from a hashed per-slice
    profile dial; counts flow through the frozen exogenous lag tensor (no
    assignment), so certification is exact linear algebra.
    """
    estimation = dict(estimation or {})
    network = scenario.network

    if scenario.sue_theta is not None:
        raise ValueError(
            f"scenario {scenario.name!r} is an SUE instance (sue_theta="
            f"{scenario.sue_theta}); within-day dynamic T2 estimation uses the "
            "exogenous free-flow lag map on the deterministic instance (ADR-023). "
            "Pass the UE instance, e.g. "
            "dataclasses.replace(scenario, sue_theta=None, reference=None)."
        )
    names = [e.name for e in estimators]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"Duplicate estimator names {sorted(duplicates)}")

    pairs = active_pairs(scenario.demand.matrix)
    n_slices = int(estimation.get("n_slices", 2))
    if n_slices < 1:
        raise ValueError(f"estimation.n_slices must be >= 1, got {n_slices}")
    slice_length = float(estimation.get("slice_length", 1.0))
    map_recipe = str(estimation.get("map", MAP_RECIPE))
    if map_recipe != MAP_RECIPE:
        raise ValueError(
            f"estimation.map={map_recipe!r} is not a known recipe; the only "
            f"shipped exogenous map is {MAP_RECIPE!r} (adr-023)"
        )
    n_lags_cfg = estimation.get("n_lags")

    full_map = lagged_assignment_tensor(
        network, pairs, slice_length, None if n_lags_cfg is None else int(n_lags_cfg)
    )
    n_lags = full_map.shape[0] - 1
    n_intervals = int(estimation.get("n_intervals", n_slices + n_lags))

    rho = _default_profile(n_slices, estimation.get("profile"))
    truth_profile = rho[:, None, None] * scenario.demand.matrix[None, :, :]
    truth_pairs = np.array(
        [[truth_profile[h, i, j] for (i, j) in pairs] for h in range(n_slices)],
        dtype=np.float64,
    ).reshape(n_slices, len(pairs))
    expected = predict_interval_counts(full_map, truth_pairs, n_intervals)  # (T, n_links)

    n_links = network.n_links
    place_rng = RngBundle(root_seed=seed, macrorep=0).generator(
        SOURCE_OBSERVATION, replication=_SENSOR_PLACEMENT_REPLICATION
    )
    obs_sensors = _draw_sensors(
        n_links, estimation.get("sensors") or {"kind": "random", "coverage": 0.3}, place_rng
    )
    ho_cfg = estimation.get("heldout") or {"kind": "random", "coverage": 0.1}
    heldout_sensors = _draw_sensors(n_links, ho_cfg, place_rng, exclude=obs_sensors)
    overlap = np.intersect1d(obs_sensors, heldout_sensors)
    if overlap.size:
        raise ValueError(
            "held-out sensors must be disjoint from observed sensors "
            f"(ADR-023 heldout_count_rmse contract, P7); overlapping links: {overlap.tolist()}"
        )
    if heldout_sensors.size == 0:
        raise ValueError(
            "held-out sensor set is empty — heldout_count_rmse is THE ranking "
            "column (ADR-023), so an all-NaN held-out design must be explicit, "
            "not silent; add heldout links or widen coverage"
        )

    n_days = int(estimation.get("n_days", estimation.get("n_periods", 1)))
    noise = estimation.get("noise", "poisson")
    prior_cfg = estimation.get("prior") or {"kind": "stale", "cv": 0.3}
    prior_cv = float(prior_cfg.get("cv", 0.3))

    ho_hash = hashlib.sha256()
    ho_hash.update(np.ascontiguousarray(np.sort(heldout_sensors), dtype=np.int64).tobytes())
    ho_hash.update(f"ho_intervals={n_intervals};ho_days={n_days};".encode())
    heldout_digest = ho_hash.hexdigest()

    ident = dynamic_identifiability_report(
        full_map, obs_sensors, pairs, n_slices, n_intervals
    )
    obs_map = full_map[:, obs_sensors, :]
    certificate = {"assignment": "exact-linear-map", "map_recipe": map_recipe}

    stochastic = (noise != "none") or (prior_cv > 0.0)
    n_data = macroreps if stochastic else 1

    data_reps: list[tuple[DynamicEstimationTask, DynamicODCertifier]] = []
    for dr in range(n_data):
        rb = RngBundle(root_seed=seed, macrorep=dr)
        prior_profile = _stale_profile(
            truth_profile, prior_cv, rb.generator(SOURCE_PRIOR)
        )
        obs_ds = DynamicLinkCounts(obs_sensors, n_days, noise).observe(
            scenario, expected, rb.generator(SOURCE_OBSERVATION)
        )
        ho_ds = DynamicLinkCounts(heldout_sensors, n_days, noise).observe(
            scenario, expected, rb.generator(SOURCE_EVALUATION)
        )
        dataset = dataclasses.replace(
            obs_ds,
            payload={
                **obs_ds.payload,
                "lag_tensor": obs_map,
                "pairs": np.asarray(pairs, dtype=np.int64),
            },
            meta={
                **obs_ds.meta,
                "n_slices": n_slices,
                "slice_length": slice_length,
                "n_lags": n_lags,
                "map_recipe": map_recipe,
            },
        )
        task = DynamicEstimationTask(
            name=scenario.name,
            network=network,
            prior_profile=prior_profile,
            dataset=dataset,
            identifiability=ident,
            scenario_hash=scenario.content_hash(),
            certificate=certificate,
            seed=seed,
            heldout_digest=heldout_digest,
        )
        certifier = DynamicODCertifier(
            scenario,
            obs_sensors,
            heldout_sensors,
            obs_ds.payload["counts"],
            ho_ds.payload["counts"],
            truth_profile,
            slice_length,
            n_lags,
            n_intervals,
            ident,
        )
        data_reps.append((task, certifier))

    rows: list[dict[str, Any]] = []
    bundles: dict[tuple[str, str], Any] = {}
    for est in estimators:
        assert_fair_evaluation(est.capabilities, scenario)
        if stochastic:
            for dr in range(n_data):
                task, certifier = data_reps[dr]
                _score_dynamic_estimator(
                    est, task, budget, RngBundle(seed, macrorep=dr), certifier, dr,
                    scenario, map_recipe, n_slices, n_intervals, rows, bundles,
                )
        else:
            task, certifier = data_reps[0]
            reps = 1 if est.capabilities.deterministic else macroreps
            for m in range(reps):
                _score_dynamic_estimator(
                    est, task, budget, RngBundle(seed, macrorep=m), certifier, m,
                    scenario, map_recipe, n_slices, n_intervals, rows, bundles,
                )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": "t2_dynamic_estimation",
        "scenario": scenario.name,
        "scenario_hash": scenario.content_hash(),
        "scenario_family": scenario.family,
        "estimation": {
            "data_level": "dynamic_link_counts",
            "sensors": {"links": obs_sensors.tolist(), "n": int(obs_sensors.size)},
            "heldout": {"links": heldout_sensors.tolist(), "n": int(heldout_sensors.size)},
            "n_slices": n_slices,
            "n_intervals": n_intervals,
            "slice_length": slice_length,
            "n_lags": n_lags,
            "map": map_recipe,
            "profile": rho.tolist(),
            "n_days": n_days,
            "noise": noise,
            "prior": {"kind": prior_cfg.get("kind", "stale"), "cv": prior_cv},
            "stochastic": stochastic,
        },
        "identifiability": ident,
        "estimators": {
            e.name: {
                "capabilities": {
                    "paradigm": e.capabilities.paradigm,
                    "deterministic": e.capabilities.deterministic,
                    "seedable": e.capabilities.seedable,
                    "inputs_required": sorted(e.capabilities.inputs_required),
                    "outputs": sorted(e.capabilities.outputs),
                    "trained_on": list(e.capabilities.trained_on),
                },
                "factors": e.factor_values,
            }
            for e in estimators
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
                "prior": SOURCE_PRIOR,
            },
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "tabench": tabench.__version__,
            "git_commit": _git_commit(),
        },
        "notes": (
            "T2 within-day dynamic estimation (Cascetta et al. 1993): the estimand is "
            "the (H, Z, Z) time-slice profile; counts flow through a frozen exogenous "
            "lag map, so the certificate is exact linear algebra (no pinned assignment). "
            "Ranked by heldout_count_rmse; OD/profile columns are descriptive (ADR-023)."
        ),
    }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        budget_part = "-".join(
            f"{axis}{value}"
            for axis, value in (
                ("it", budget.iterations),
                ("sp", budget.sp_calls),
                ("ws", budget.wall_seconds),
            )
            if value is not None
        )
        task_hash8 = data_reps[0][0].content_hash()[:8]
        stem = (
            f"{scenario.name}-{scenario.content_hash()[:8]}_t2d-{task_hash8}_"
            f"{'-'.join(names)}_{budget_part}_seed-{seed}"
        )
        with open(out / f"{stem}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_DYN_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        with open(out / f"{stem}.manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    return ExperimentResult(rows=rows, bundles=bundles, manifest=manifest)


def _stale_profile(
    truth_profile: np.ndarray, cv: float, rng: np.random.Generator
) -> np.ndarray:
    """Per-slice stale prior: each positive off-diagonal cell scaled by Gamma noise.

    Mirrors :class:`~tabench.observe.levels.StalePriorOD` (mean 1, coefficient of
    variation ``cv``; ``cv = 0`` returns truth) but applied independently per
    ``(slice, pair)`` on the reserved ``SOURCE_PRIOR`` stream, so the prior profile
    is independent of the observed counts. Zero cells and diagonals stay put.
    """
    prior = np.asarray(truth_profile, dtype=np.float64).copy()
    if cv <= 0:
        return prior
    n_zones = prior.shape[1]
    off = ~np.eye(n_zones, dtype=bool)
    shape = 1.0 / cv**2
    scale = cv**2
    for h in range(prior.shape[0]):
        positive = off & (truth_profile[h] > 0)
        draws = rng.gamma(shape=shape, scale=scale, size=int(positive.sum()))
        prior[h][positive] = truth_profile[h][positive] * draws
    return prior
