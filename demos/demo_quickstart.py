"""TABenchmark quickstart: certified comparison of five model vintages.

Runs on the built-in Braess scenario (no download needed):

* ``aon``  — all-or-nothing, the capacity-blind pre-1956 practice
* ``msa``  — method of successive averages
* ``fw``   — Frank-Wolfe (LeBlanc et al. 1975)
* ``cfw``/``bfw`` — conjugate direction FW (Mitradjieva & Lindberg 2013)
* a black-box callable wrapper, certified by the same external gap (P1)

Usage:
    python demos/demo_quickstart.py                       # certified table only
    python demos/demo_quickstart.py --viz                 # + OD / link-flow PNGs
    python demos/demo_quickstart.py --viz --viz-out DIR   # PNGs into DIR

The ``--viz`` option needs the optional viz extra: ``pip install 'tabench[viz]'``.
"""

import argparse
import sys

import numpy as np

from tabench import (
    AllOrNothingModel,
    BiconjugateFrankWolfeModel,
    Budget,
    CallableModel,
    ConjugateFrankWolfeModel,
    FrankWolfeModel,
    MSAModel,
    braess_scenario,
    run_experiment,
)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="TABenchmark quickstart: certified comparison of five model vintages."
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="render OD + per-model link-flow figures against a bfw ground truth",
    )
    parser.add_argument(
        "--viz-out",
        default=None,
        metavar="DIR",
        help="directory for the --viz PNGs (default: ./quickstart_viz)",
    )
    args = parser.parse_args(argv)

    # --viz-out without --viz is a no-op: warn on stderr (stdout stays byte-identical).
    if args.viz_out is not None and not args.viz:
        print("warning: --viz-out is ignored without --viz", file=sys.stderr)

    scenario = braess_scenario()

    def naive_surrogate(s, rng):
        """A stand-in 'learned' model: uniform flow spreading plus noise."""
        base = s.demand.total / 2.0
        return np.abs(base + rng.normal(0.0, 0.5, s.network.n_links))

    models = [
        AllOrNothingModel(),
        MSAModel(),
        FrankWolfeModel(),
        ConjugateFrankWolfeModel(),
        BiconjugateFrankWolfeModel(),
        CallableModel(fn=naive_surrogate, name="toy-surrogate", seedable=True),
    ]

    result = run_experiment(scenario, models, Budget(iterations=200), seed=0)

    print(f"Scenario: {scenario.name} (hash {scenario.content_hash()[:16]})")
    print(f"{'model':<16}{'certified rel. gap':>20}{'feasible':>10}")
    print("-" * 46)
    final_rows: dict[str, dict] = {}
    for row in result.rows:
        final_rows[row["model"]] = row
    for name, row in final_rows.items():
        print(f"{name:<16}{row['relative_gap']:>20.3e}{row['feasible']:>10.0f}")

    print(
        "\nThe harness certifies every model's gap externally. The black box is"
        "\nscored by the identical certificate as Frank-Wolfe -- and because its"
        "\nflows fail the demand-aware feasibility audit, its gap is censored"
        "\n(nan) rather than scored: garbage can neither crash the experiment"
        "\nnor top the leaderboard."
    )

    if args.viz:
        _render_viz(scenario, models, result, args.viz_out)


def _render_viz(scenario, models, result, out_dir) -> None:
    """Render the OD, per-model link-flow, and model-vs-GT figures as PNGs.

    A ground-truth reference is solved IN-RUN (a generous-budget bfw) and labelled
    by its OWN certified relative gap from the same run_experiment machinery — never
    a pasted analytic number (P1). Agg is forced before pyplot loads (this is a
    script, not a notebook), and a missing matplotlib fails loudly with the hint.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: never require a display
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        raise SystemExit(
            "--viz needs matplotlib; install it with `pip install 'tabench[viz]'`"
        ) from exc

    from pathlib import Path

    from tabench import viz

    out = Path(out_dir or "./quickstart_viz")
    out.mkdir(parents=True, exist_ok=True)

    # Ground truth: a high-budget bfw certified by the identical machinery.
    gt_model = BiconjugateFrankWolfeModel()
    gt_result = run_experiment(
        scenario, [gt_model], Budget(iterations=1000, target_relative_gap=1e-12), seed=0
    )
    gt_flows = gt_result.bundles[(gt_model.name, "m0")].final.link_flows
    gt_gap = gt_result.rows[-1]["relative_gap"]
    gt_label = f"{gt_model.name} (rel. gap {gt_gap:.1e})"

    # Final certified link flows per model, straight from the run's bundles.
    model_flows = {
        model.name: result.bundles[(model.name, "m0")].final.link_flows for model in models
    }

    od_png = out / "01_od_demand.png"
    viz.plot_od_demand(scenario.demand).savefig(od_png, dpi=140, bbox_inches="tight")

    flows_png = out / "02_link_flows.png"  # braess has 5 links; all are annotated
    viz.compare_models(scenario, model_flows, reference=(gt_label, gt_flows)).savefig(
        flows_png, dpi=140, bbox_inches="tight"
    )

    scatter_png = out / "03_model_vs_gt.png"
    viz.plot_flow_scatter((gt_label, gt_flows), model_flows).savefig(
        scatter_png, dpi=140, bbox_inches="tight"
    )

    print(f"\nSaved: {od_png}")
    print(f"Saved: {flows_png}")
    print(f"Saved: {scatter_png}")
    print(
        "03_model_vs_gt.png shows the CONVERGED solvers (fw, cfw, bfw, and near-converged\n"
        "msa) clustering on the y = x line, while the toy surrogate AND the capacity-blind\n"
        "aon baseline sit off it -- aon farthest of all, exactly as its ~1.9e-1 certified\n"
        "gap says. Off-diagonal is not censorship: aon is feasible and honestly scored, its\n"
        "distance from the diagonal IS its gap. The P1 story, made visual."
    )


if __name__ == "__main__":
    main()
