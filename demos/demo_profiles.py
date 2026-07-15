"""TABenchmark progress curves and solvability profiles (docs/design/adr-032).

Post-hoc SimOpt-style diagnostics over an already-certified grid run on the
built-in Braess scenario (no download needed):

* a cdf-solvability profile (fraction of the run solved to a target gap, vs
  normalized budget fraction),
* a Moré-Wild data profile (fraction solved vs work in all-or-nothing passes),
* the certified ``profiles.json`` artifact (curves + protocol + provenance).

Everything is a pure function of the certified rows: no solver, certifier, or
runner is touched. ASCII tables always print; PNGs are written only if
matplotlib is importable, so the numpy/scipy core stays dependency-free.

Usage: python demos/demo_profiles.py
"""

from __future__ import annotations

from pathlib import Path

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
from tabench.experiments.profiles import (
    Run,
    cdf_solvability,
    data_profile,
    progress_curves,
    run_provenance,
    solve_times,
    write_profiles,
)

ALPHA = 1e-4  # certified-gap solve target (Boyce et al. 2004 convergence target)
TAU = 1e-3  # Moré-Wild convergence-test level


def _ascii_profile(title: str, profile: dict, samples: list[float]) -> None:
    print(f"\n{title}")
    print(f"{'model':<14}" + "".join(f"{s:>10.2f}" for s in samples))
    print("-" * (14 + 10 * len(samples)))
    for model, curve in profile.items():
        print(f"{model:<14}" + "".join(f"{curve.lookup(s):>10.3f}" for s in samples))


def main() -> None:
    scenario = braess_scenario()

    def naive_surrogate(s, rng):
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
    run = Run.from_result(result)

    # n_origins with positive demand = one all-or-nothing pass, the Moré-Wild
    # work unit; Braess has a single origin (a scenario property, not a certified
    # row column, so it is supplied here -- adr-032 D6).
    demand = scenario.demand
    n_origins = int((demand.matrix.sum(axis=1) > 0).sum())

    curves = progress_curves(run, axis="sp_calls")
    times = {model: t for (model, _mr), t in solve_times(curves, ALPHA).items()}
    print(f"Scenario: {scenario.name} (hash {scenario.content_hash()[:16]})")
    print(f"\nα-solve times (sp_calls to certified gap < {ALPHA:g}; inf = never/censored):")
    for model in sorted(times, key=lambda m: times[m]):
        t = times[model]
        print(f"  {model:<14}{'inf' if t == float('inf') else int(t):>6}")

    cdf = cdf_solvability(run, ALPHA, axis="sp_calls")
    _ascii_profile(
        "cdf-solvability profile (fraction solved vs normalized budget fraction):",
        cdf,
        [0.1, 0.25, 0.5, 1.0],
    )

    data = data_profile(run, tau=TAU, axis="sp_calls", work_unit=float(n_origins))
    _ascii_profile(
        f"Moré-Wild data profile (fraction solved vs κ = sp_calls / {n_origins} AON-pass):",
        data,
        [1.0, 5.0, 10.0, 25.0],
    )

    # results/ is the repo's gitignored experiment-output location.
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    artifact = out_dir / "demo_profiles.json"
    protocol = {
        "metric": "relative_gap",
        "axis": "sp_calls",
        "alpha": ALPHA,
        "tau": TAU,
        "crossing": "strict-<",
        "censoring": "in-denominator",
        "aon_work_unit": n_origins,
    }
    write_profiles(
        artifact,
        {"cdf_solvability": cdf, "data_profile": data},
        protocol,
        run_provenance(run),
    )
    print(f"\nCertified artifact written: {artifact}")
    print(
        "\nThe black box discloses sp_calls=0 and fails the feasibility audit, so it is\n"
        "censored (+inf) and stays in every cdf denominator -- garbage never tops the\n"
        "profile, and MSA's early sub-α crossing is honestly a first crossing, not\n"
        "sustained convergence (the certified rows carry the whole story)."
    )

    _maybe_plot(out_dir, cdf, data, n_origins)


def _maybe_plot(out_dir: Path, cdf: dict, data: dict, n_origins: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed -- skipping PNGs; ASCII tables above are the artifact)")
        return

    for name, profile, xlabel in (
        ("cdf_solvability", cdf, "normalized budget fraction"),
        ("data_profile", data, f"κ (sp_calls / {n_origins} AON-pass)"),
    ):
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for model, curve in profile.items():
            ax.step(curve.x, curve.y, where="post", label=model)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("fraction solved")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7)
        fig.tight_layout()
        png = out_dir / f"demo_{name}.png"
        fig.savefig(png, dpi=120)
        plt.close(fig)
        print(f"Plot written: {png}")


if __name__ == "__main__":
    main()
