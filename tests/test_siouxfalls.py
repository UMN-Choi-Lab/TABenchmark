"""Regression tests against the published Sioux Falls best-known UE solution.

The oracle objective is computed *from the best-known flows* with this
package's own Beckmann implementation, so the test is unit-convention-free:
it validates parser, cost functions, gap machinery, and Frank-Wolfe together.

Requires network access on first run (checksummed download-on-demand, P9);
skipped automatically when the data cannot be fetched.
"""

import numpy as np
import pytest
from conftest import load_or_skip

from tabench import Budget, Evaluator, FrankWolfeModel, RngBundle, Trace


@pytest.fixture(scope="module")
def scenario():
    # Offline -> skip; checksum mismatch or TABENCH_REQUIRE_DATA=1 -> fail.
    return load_or_skip("siouxfalls")


def test_parsed_dimensions(scenario):
    net = scenario.network
    assert net.n_zones == 24
    assert net.n_nodes == 24
    assert net.n_links == 76
    assert net.first_thru_node == 1
    assert scenario.demand.total == pytest.approx(360600.0)


def test_best_known_flows_certify_as_equilibrium(scenario):
    """The published solution (AEC ~3.9e-15 in native units) must get a tiny gap."""
    assert scenario.reference is not None
    metrics = Evaluator(scenario).evaluate(scenario.reference.link_flows)
    assert metrics["relative_gap"] < 1e-8
    assert metrics["feasible"] == 1.0


def test_frank_wolfe_regresses_to_best_known_objective(scenario):
    oracle_objective = Evaluator(scenario).evaluate(scenario.reference.link_flows)[
        "beckmann_objective"
    ]
    trace = Trace()
    FrankWolfeModel().solve(scenario, Budget(iterations=300), RngBundle(0), trace)
    metrics = Evaluator(scenario).evaluate(trace.final.link_flows)
    assert metrics["beckmann_objective"] == pytest.approx(oracle_objective, rel=1e-3)
    assert metrics["relative_gap"] < 1e-3
    # Objective must be monotone nonincreasing under exact line search.
    objectives = [s.self_report["beckmann"] for s in trace]
    pairs = zip(objectives, objectives[1:], strict=False)
    assert all(b2 <= b1 + 1e-9 * abs(b1) for b1, b2 in pairs)


def test_reference_flows_alignment(scenario):
    ref = scenario.reference.link_flows
    assert ref.shape == (76,)
    assert np.all(ref >= 0)
    assert np.all(np.isfinite(ref))
