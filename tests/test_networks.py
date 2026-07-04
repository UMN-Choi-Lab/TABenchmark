"""Regression tests for the downloadable network registry beyond Sioux Falls.

Each network's best-known flow file must certify as an equilibrium under
tabench's own pure-BPR Evaluator (the audit that admitted these networks
verified the published per-link costs reproduce bit-for-bit). Downloads are
checksummed and commit-pinned (P9); tests skip when offline.
"""

import numpy as np
import pytest
from conftest import load_or_skip

from tabench import Evaluator

DIMENSIONS = {
    # key: (zones, nodes, links, first_thru_node)
    "anaheim": (38, 416, 914, 39),
    "barcelona": (110, 1020, 2522, 111),
    "winnipeg": (147, 1052, 2836, 148),
}

# Published optimal Beckmann objectives (TransportationNetworks READMEs),
# reproduced by tabench's implementation from the best-known flows.
PUBLISHED_OBJECTIVE = {
    "barcelona": 1265654.92203176,
    "winnipeg": 827911.494629963,
}


@pytest.fixture(scope="module", params=sorted(DIMENSIONS))
def scenario(request):
    return load_or_skip(request.param)


def test_parsed_dimensions(scenario):
    zones, nodes, links, ftn = DIMENSIONS[scenario.name]
    net = scenario.network
    assert net.n_zones == zones
    assert net.n_nodes == nodes
    assert net.n_links == links
    assert net.first_thru_node == ftn


def test_best_known_flows_certify_as_equilibrium(scenario):
    """Certified gap ~ machine precision; may be a hair negative, never large."""
    assert scenario.reference is not None
    metrics = Evaluator(scenario).evaluate(scenario.reference.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-8


def test_reference_matches_published_objective(scenario):
    objective = Evaluator(scenario).evaluate(scenario.reference.link_flows)[
        "beckmann_objective"
    ]
    published = PUBLISHED_OBJECTIVE.get(scenario.name)
    if published is not None:
        assert objective == pytest.approx(published, rel=1e-12)


def test_reference_flows_are_clean(scenario):
    ref = scenario.reference.link_flows
    assert ref.shape == (scenario.network.n_links,)
    assert np.all(np.isfinite(ref))
    assert np.all(ref >= 0)
