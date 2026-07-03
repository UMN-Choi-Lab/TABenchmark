"""Command-line interface: ``tabench list | fetch | run``."""

from __future__ import annotations

import argparse
import sys
import urllib.error
from pathlib import Path

import yaml

from .core.budget import Budget
from .data import REGISTRY, ChecksumError, citation, fetch, load_scenario
from .experiments.runner import run_experiment
from .models.base import MODEL_REGISTRY


def _cmd_list(_: argparse.Namespace) -> int:
    print("Scenarios:")
    print("  braess        (built-in, analytic UE oracle)")
    for key, spec in sorted(REGISTRY.items()):
        print(f"  {key:<14}({spec.repo_dir}, download-on-demand)")
    print("\nModels:")
    for name in sorted(MODEL_REGISTRY):
        cls = MODEL_REGISTRY[name]
        caps = cls.capabilities
        print(f"  {name:<10}{caps.paradigm}, deterministic={caps.deterministic}")
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
    if args.config:
        card = yaml.safe_load(Path(args.config).read_text())
        if not isinstance(card, dict) or "scenario" not in card:
            print(f"{args.config}: not a scenario card (missing 'scenario')", file=sys.stderr)
            return 2
        if scenario_key is None:
            scenario_key = card["scenario"]
        if iterations is None:
            iterations = int(card.get("budgets", {}).get("iterations", 100))
    scenario_key = scenario_key or "braess"
    iterations = iterations or 100

    scenario = load_scenario(scenario_key)
    models = []
    for name in args.models.split(","):
        name = name.strip()
        if name not in MODEL_REGISTRY:
            print(f"Unknown model {name!r}; see `tabench list`", file=sys.stderr)
            return 2
        models.append(MODEL_REGISTRY[name]())
    budget = Budget(iterations=iterations)
    result = run_experiment(
        scenario, models, budget, seed=args.seed, out_dir=args.out
    )
    last_by_model: dict[str, dict] = {}
    for row in result.rows:
        last_by_model[row["model"]] = row
    print(f"Scenario {scenario.name} (hash {scenario.content_hash()[:16]}), "
          f"budget {iterations} iterations, seed {args.seed}\n")
    header = f"{'model':<10}{'iters':>6}{'rel. gap':>14}{'AEC':>14}{'Beckmann obj.':>18}"
    print(header)
    print("-" * len(header))
    for name, row in sorted(last_by_model.items()):
        print(
            f"{name:<10}{row['iterations']:>6}{row['relative_gap']:>14.3e}"
            f"{row['average_excess_cost']:>14.3e}{row['beckmann_objective']:>18.6e}"
        )
    if args.out:
        print(f"\nWrote CSV + manifest to {args.out}/")
    return 0


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
    p_run.add_argument("--models", default="aon,msa,fw", help="Comma-separated model names")
    p_run.add_argument("--iterations", type=int, default=None)
    p_run.add_argument("--seed", type=int, default=0)
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
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
