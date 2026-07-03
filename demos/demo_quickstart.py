"""TABenchmark quickstart: certified comparison of three model vintages.

Runs on the built-in Braess scenario (no download needed):

* ``aon``  — all-or-nothing, the capacity-blind pre-1956 practice
* ``msa``  — method of successive averages
* ``fw``   — Frank-Wolfe (LeBlanc et al. 1975)
* a black-box callable wrapper, certified by the same external gap (P1)

Usage: python demos/demo_quickstart.py
"""

import numpy as np

from tabench import (
    AllOrNothingModel,
    Budget,
    CallableModel,
    FrankWolfeModel,
    MSAModel,
    braess_scenario,
    run_experiment,
)


def main() -> None:
    scenario = braess_scenario()

    def naive_surrogate(s, rng):
        """A stand-in 'learned' model: uniform flow spreading plus noise."""
        base = s.demand.total / 2.0
        return np.abs(base + rng.normal(0.0, 0.5, s.network.n_links))

    models = [
        AllOrNothingModel(),
        MSAModel(),
        FrankWolfeModel(),
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


if __name__ == "__main__":
    main()
