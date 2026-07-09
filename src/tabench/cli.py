"""Command-line interface: ``tabench list | fetch | run``."""

from __future__ import annotations

import argparse
import dataclasses
import sys
import urllib.error
from pathlib import Path

import yaml

from .core.budget import Budget
from .core.scenario import ElasticDemand
from .data import REGISTRY, ChecksumError, citation, fetch, load_scenario
from .estimation.base import ESTIMATOR_REGISTRY
from .experiments.runner import run_estimation_experiment, run_experiment
from .models.base import MODEL_REGISTRY

_DEFAULT_ESTIMATORS = "prior,gls,vzw-entropy,spiess,spsa"


def _cmd_list(_: argparse.Namespace) -> int:
    print("Scenarios:")
    print("  braess          (built-in, analytic UE oracle)")
    print("  tworoute        (built-in, analytic logit-SUE oracle)")
    print("  elastic-tworoute(built-in, analytic elastic-demand UE oracle)")
    print("  evans           (built-in, analytic combined distribution+assignment oracle)")
    print("  br-tworoute     (built-in, analytic boundedly-rational band oracle)")
    print("  sc-tworoute     (built-in, analytic side-constrained capacity oracle)")
    print("  vi-tworoute     (built-in, analytic asymmetric-VI UE oracle)")
    print("  multiclass      (built-in, analytic multiclass-user equilibrium oracle)")
    for key, spec in sorted(REGISTRY.items()):
        print(f"  {key:<14}({spec.repo_dir}, download-on-demand)")
    print("\nModels:")
    for name in sorted(MODEL_REGISTRY):
        cls = MODEL_REGISTRY[name]
        caps = cls.capabilities
        print(f"  {name:<18}{caps.paradigm}, deterministic={caps.deterministic}")
    print("\nEstimators (T2):")
    for name in sorted(ESTIMATOR_REGISTRY):
        cls = ESTIMATOR_REGISTRY[name]
        caps = cls.capabilities
        print(f"  {name:<12}{caps.paradigm}, deterministic={caps.deterministic}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    spec = REGISTRY.get(args.scenario)
    if spec is None:
        print(f"Unknown downloadable scenario {args.scenario!r}", file=sys.stderr)
        return 2
    paths = fetch(spec, force=args.force)
    for role, path in sorted(paths.items()):
        print(f"{role:<6} {path}")
    print(f"\nCite: {citation(spec)}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    scenario_key = args.scenario
    iterations = args.iterations
    card: dict = {}
    if args.config:
        card = yaml.safe_load(Path(args.config).read_text())
        if not isinstance(card, dict) or "scenario" not in card:
            print(f"{args.config}: not a scenario card (missing 'scenario')", file=sys.stderr)
            return 2
        if scenario_key is None:
            scenario_key = card["scenario"]
        if iterations is None:
            iterations = int((card.get("budgets") or {}).get("iterations", 100))
    scenario_key = scenario_key or "braess"
    iterations = iterations or 100

    scenario = load_scenario(scenario_key)
    if "t2_estimation" in (card.get("tasks") or []):
        return _run_estimation(args, card, scenario)
    sue = card.get("sue")
    r_cert = 2000
    if isinstance(sue, dict) and "theta" not in sue and (set(sue) & {"family", "r_cert"}):
        # theta defines the SUE instance (it is content-hashed); declaring a
        # family or r_cert without it would silently run the wrong (default)
        # instance instead of the one the card asked for.
        keys = sorted(set(sue) & {"family", "r_cert"})
        print(
            f"{args.config}: sue block declares {keys} but no 'theta'; "
            "theta is required to define the SUE instance.",
            file=sys.stderr,
        )
        return 2
    if isinstance(sue, dict) and "theta" in sue:
        theta = float(sue["theta"])
        family = str(sue.get("family", scenario.sue_family))
        r_cert = int(sue.get("r_cert", r_cert))
        if theta != scenario.sue_theta or family != scenario.sue_family:
            # A different theta or family is a different benchmark instance (the
            # hash changes) — and any built-in reference oracle was certified for
            # the old one, so drop it rather than mis-score RMSE against it.
            scenario = dataclasses.replace(
                scenario, sue_theta=theta, sue_family=family, reference=None
            )
    elastic = card.get("elastic")
    if isinstance(elastic, dict) and (set(elastic) & {"form", "param"}):
        # The demand law defines the elastic instance (it is content-hashed), so
        # honor the card's form/param rather than silently running the loader's
        # default (mirrors the sue handling above).
        if not {"form", "param"} <= set(elastic):
            print(
                f"{args.config}: elastic block needs both 'form' and 'param' "
                "to define the demand law.",
                file=sys.stderr,
            )
            return 2
        law = ElasticDemand(form=str(elastic["form"]), param=float(elastic["param"]))
        if scenario.elastic_demand != law:
            scenario = dataclasses.replace(scenario, elastic_demand=law, reference=None)
    # Elastic scenarios need the elastic solver by default: fixed-demand models
    # route the reference demand, not D(u(v)), and are censored on them.
    if scenario.combined_demand is not None:
        default_models = "evans"
    elif scenario.elastic_demand is not None:
        default_models = "fw-elastic"
    elif scenario.br_epsilon is not None:
        default_models = "br-ue"
    elif scenario.side_capacities is not None:
        default_models = "sc-tap"
    elif scenario.multiclass is not None:
        # Multiclass needs the per-class solver: single-class models emit no
        # per-class flows and are censored on a multiclass task (adr-013).
        default_models = "multiclass"
    elif scenario.link_interaction is not None:
        # Non-separable VI: fixed-demand separable solvers ignore the interaction.
        default_models = "vi-asym"
    else:
        default_models = "aon,msa,fw"
    models = []
    for name in (args.models or default_models).split(","):
        name = name.strip()
        if name not in MODEL_REGISTRY:
            print(f"Unknown model {name!r}; see `tabench list`", file=sys.stderr)
            return 2
        models.append(MODEL_REGISTRY[name]())
    budget = Budget(iterations=iterations, target_relative_gap=args.target_gap)
    result = run_experiment(
        scenario, models, budget, seed=args.seed, macroreps=args.macroreps,
        out_dir=args.out, r_cert=r_cert,
    )
    last_by_model: dict[str, dict] = {}
    for row in result.rows:
        last_by_model[row["model"]] = row
    print(f"Scenario {scenario.name} (hash {scenario.content_hash()[:16]}), "
          f"budget {iterations} iterations, seed {args.seed}\n")
    is_sue = scenario.sue_theta is not None
    is_elastic = scenario.elastic_demand is not None
    has_so = any("so_relative_gap" in row for row in last_by_model.values())
    header = f"{'model':<10}{'iters':>6}{'rel. gap':>14}{'AEC':>14}{'Beckmann obj.':>18}"
    if is_sue:
        header += f"{'SUE residual':>16}"
    if has_so:
        header += f"{'SO rel. gap':>14}"
    if is_elastic:
        header += f"{'realized dem.':>16}"
    print(header)
    print("-" * len(header))
    for name, row in sorted(last_by_model.items()):
        line = (
            f"{name:<10}{row['iterations']:>6}{row['relative_gap']:>14.3e}"
            f"{row['average_excess_cost']:>14.3e}{row['beckmann_objective']:>18.6e}"
        )
        if is_sue:
            line += f"{row['sue_fixed_point_residual']:>16.3e}"
        if has_so:
            line += f"{row['so_relative_gap']:>14.3e}"
        if is_elastic:
            line += f"{float(row['realized_demand']):>16.3e}"
        print(line)
    if is_sue:
        print(
            "\nSUE task: ranked by the certified SUE residual; the UE columns "
            "are descriptive (docs/design/adr-001)."
        )
        if scenario.sue_family == "probit":
            print(
                f"Probit-SUE (adr-003, R_cert={r_cert}): the residual is a "
                "Monte Carlo estimate on a pinned evaluation stream. Differences "
                "below max(sue_residual_floor, 2*sue_residual_se) are ties."
            )
    if has_so:
        print(
            "\nSO goal: static_so models are ranked by the certified SO relative "
            "gap; the UE columns are descriptive."
        )
    if is_elastic:
        print(
            "\nElastic-demand task (adr-005): ranked by the certified relative_gap; "
            "realized_demand = Sum D(u) is the induced travel. fw-elastic solves it; "
            "fixed-demand models are censored (they route the reference demand, not D(u))."
        )
    if args.out:
        print(f"\nWrote CSV + manifest to {args.out}/")
    return 0


def _run_estimation(args: argparse.Namespace, card: dict, scenario) -> int:
    """Dispatch a ``t2_estimation`` card to the OD-estimation runner."""
    if scenario.sue_theta is not None:
        # T2 is a UE task this sprint (SUE-pinned certificates are deferred,
        # ADR-002): score against the deterministic-UE instance.
        scenario = dataclasses.replace(scenario, sue_theta=None, reference=None)
    estimation = card.get("estimation") or {}
    budgets = card.get("budgets") or {}
    sp_calls = int(budgets.get("sp_calls", 2000))
    macroreps = int(card.get("macroreps", estimation.get("macroreps", 1)))
    # ``--models`` defaults to None so a T2 card falls back to the estimator
    # default, while an *explicit* list is always taken literally (so e.g. an
    # explicit T1 name like ``aon`` on a T2 card errors cleanly, exit 2).
    models_arg = args.models or _DEFAULT_ESTIMATORS
    estimators = []
    for name in models_arg.split(","):
        name = name.strip()
        if name not in ESTIMATOR_REGISTRY:
            print(f"Unknown estimator {name!r}; see `tabench list`", file=sys.stderr)
            return 2
        estimators.append(ESTIMATOR_REGISTRY[name]())
    budget = Budget(sp_calls=sp_calls)
    result = run_estimation_experiment(
        scenario, estimators, budget, seed=args.seed, macroreps=macroreps,
        out_dir=args.out, estimation=estimation,
    )
    ident = result.manifest["identifiability"]
    print(
        f"Scenario {scenario.name} (hash {scenario.content_hash()[:16]}), "
        f"T2 estimation, budget {sp_calls} sp_calls, seed {args.seed}\n"
    )
    print(
        f"identifiability: n_active_pairs={ident['n_active_pairs']} "
        f"hazelton_condition={ident.get('hazelton_condition')} "
        f"linear_identifiable={ident['linear_identifiable']} "
        f"(sensors={result.manifest['estimation']['sensors']['n']}, "
        f"heldout={result.manifest['estimation']['heldout']['n']})\n"
    )
    last: dict[str, dict] = {}
    for row in result.rows:
        last[row["estimator"]] = row
    header = (
        f"{'estimator':<13}{'sp':>7}{'obs_rmse':>12}{'heldout_rmse':>14}"
        f"{'od_rmse':>12}{'od_ident':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, row in sorted(last.items(), key=lambda kv: _rank_key(kv[1])):
        print(
            f"{name:<13}{row['sp_calls']:>7}{row['obs_count_rmse']:>12.4e}"
            f"{row['heldout_count_rmse']:>14.4e}{row['od_rmse']:>12.4e}"
            f"{int(row['od_identifiable']):>10}"
        )
    print(
        "\nT2 task: ranked by heldout_count_rmse (out-of-sample count fit); "
        "OD columns are descriptive"
        + ("" if ident["linear_identifiable"] else " (od_identifiable=0 here)")
        + " (ADR-002)."
    )
    if args.out:
        print(f"\nWrote CSV + manifest to {args.out}/")
    return 0


def _rank_key(row: dict) -> float:
    value = row.get("heldout_count_rmse")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return value if value == value else float("inf")  # NaN sorts last


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tabench",
        description="TABenchmark: a shared benchmark for 50 years of traffic assignment models",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List available scenarios and models")

    p_fetch = sub.add_parser("fetch", help="Download and verify a scenario's data")
    p_fetch.add_argument("scenario")
    p_fetch.add_argument("--force", action="store_true", help="Re-download even if cached")

    p_run = sub.add_parser("run", help="Run models on a scenario with certified scoring")
    p_run.add_argument("--scenario", default=None, help="Scenario key (default: braess)")
    p_run.add_argument(
        "--config", default=None, help="Scenario card YAML (e.g. scenarios/1siouxfalls.yaml)"
    )
    p_run.add_argument(
        "--models",
        default=None,
        help="Comma-separated model names (default: aon,msa,fw for T1; "
        "prior,gls,vzw-entropy,spiess,spsa for T2)",
    )
    p_run.add_argument("--iterations", type=int, default=None)
    p_run.add_argument(
        "--target-gap",
        type=float,
        default=None,
        dest="target_gap",
        help="Convergence early-stop on the self-monitored relative gap "
        "(Boyce et al. 2004 recommend 1e-4); iterations still bound the run",
    )
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument(
        "--macroreps",
        type=int,
        default=1,
        help="Independent macroreplications for non-deterministic models "
        "(stochastic track, P5); deterministic models always run once. With "
        ">1 the manifest carries a bootstrap CI of the certified metric.",
    )
    p_run.add_argument("--out", default=None, help="Directory for CSV + manifest output")

    args = parser.parse_args(argv)
    handlers = {"list": _cmd_list, "fetch": _cmd_fetch, "run": _cmd_run}
    try:
        return handlers[args.command](args)
    except ChecksumError as exc:
        print(f"data integrity error: {exc}", file=sys.stderr)
        return 2
    except urllib.error.URLError as exc:
        print(f"download failed (network unavailable?): {exc}", file=sys.stderr)
        return 2
    except yaml.YAMLError as exc:
        print(f"error: invalid scenario card YAML: {exc}", file=sys.stderr)
        return 2
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
