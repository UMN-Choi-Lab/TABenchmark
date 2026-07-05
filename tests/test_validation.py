"""Numerical validation against independent oracles (docs/VALIDATION.md).

These are the cross-cutting checks that tie the solvers to numbers computed
*outside* the harness: the published best-known Beckmann objective, the exact
Braess UE/SO/price-of-anarchy, cross-family link-flow agreement (uniqueness of
UE link flows), and the one externally reproducible convergence-ranking result
(Mitradjieva & Lindberg 2013). Per-solver oracle regressions live in the
individual test files; this file pins the shared, provenance-bearing numbers.
"""

import numpy as np
import pytest
from conftest import load_or_skip

from tabench import (
    AlgorithmBModel,
    BiconjugateFrankWolfeModel,
    Budget,
    ConjugateFrankWolfeModel,
    Evaluator,
    FrankWolfeModel,
    GradientProjectionModel,
    RngBundle,
    SystemOptimumModel,
    TapasModel,
    Trace,
    braess_scenario,
)

#: TransportationNetworks best-known optimal Beckmann objective for Sioux Falls,
#: in TNTP native units. The repo carries free-flow times in 0.01 h, a fixed 1e5
#: unit factor vs. this published value.
SIOUXFALLS_TNTP_OBJECTIVE = 42.31335287107440
SIOUXFALLS_UNIT_FACTOR = 1e5


def _solve(scenario, model, **budget):
    trace = Trace()
    model.solve(scenario, Budget(**budget), RngBundle(0), trace)
    return trace


# ------------------------------------------------- Braess UE / SO / PoA (exact)
def test_braess_ue_so_and_price_of_anarchy():
    """The exact analytic anchor. UE routes 2 units on each of three routes
    (route cost 92 -> TSTT 552); SO leaves the bypass empty, 3 units on each
    outer route (route cost 83 -> TSTT 498); price of anarchy 552/498."""
    braess = braess_scenario()
    net = braess.network

    ue = Evaluator(braess).evaluate(np.array([4.0, 2.0, 2.0, 2.0, 4.0]))
    assert ue["feasible"] == 1.0
    assert ue["relative_gap"] < 1e-8
    # 6 travelers x route cost 92; the 10v links use fft=1e-6 (Network needs
    # fft>0), an induced offset far below this tolerance.
    assert ue["tstt"] == pytest.approx(552.0, abs=1e-3)

    so = _solve(braess, SystemOptimumModel(), iterations=300, target_relative_gap=1e-12)
    so_flows = so.final.link_flows
    np.testing.assert_allclose(so_flows, [3.0, 3.0, 0.0, 3.0, 3.0], atol=1e-4)  # bypass unused
    tstt_so = float(so_flows @ net.link_cost(so_flows))
    assert tstt_so == pytest.approx(498.0, abs=1e-3)  # 6 travelers x route cost 83
    assert 552.0 / tstt_so == pytest.approx(1.1084, abs=1e-3)  # price of anarchy


# ----------------------------------------- published optimal objective (Sioux Falls)
def test_siouxfalls_matches_published_beckmann_optimum():
    """The best-known flows reproduce the Transportation Networks published
    optimal Beckmann objective to full precision (up to the repo's 1e5 unit
    factor) — an external validation of both the flows and the objective."""
    scenario = load_or_skip("siouxfalls")
    obj = Evaluator(scenario).evaluate(scenario.reference.link_flows)["beckmann_objective"]
    assert obj / SIOUXFALLS_UNIT_FACTOR == pytest.approx(SIOUXFALLS_TNTP_OBJECTIVE, rel=1e-9)


# ------------------------------------------------- cross-solver agreement
def test_cross_family_solvers_converge_to_same_link_flows():
    """UE link flows are unique, so path-based (gp), bush-based (algb), and
    PAS-based (tapas) solvers must converge to the same vector and to the
    best-known flows — the strongest oracle-free consistency check."""
    scenario = load_or_skip("siouxfalls")
    ref = scenario.reference.link_flows
    flows = {}
    for name, model in (
        ("gp", GradientProjectionModel()),
        ("algb", AlgorithmBModel()),
        ("tapas", TapasModel()),
    ):
        v = _solve(scenario, model, iterations=300, target_relative_gap=1e-9).final.link_flows
        assert np.abs(v - ref).max() < 5e-2  # each converges to best-known
        flows[name] = v
    names = list(flows)
    max_pairwise = max(
        np.abs(flows[a] - flows[b]).max() for a in names for b in names
    )
    assert max_pairwise < 5e-2  # ... and therefore to each other (measured ~3e-3)


# --------------------------------- Mitradjieva & Lindberg 2013 ranking (the one paper pin)
def test_conjugate_directions_dramatically_accelerate_frank_wolfe():
    """The one externally reproducible convergence result: conjugate-direction
    FW variants reach a target gap in far fewer iterations than plain FW
    (Mitradjieva & Lindberg 2013, Table 3, "~10x"). Only the *ranking* /
    order-of-magnitude is pinned — absolute counts are BLAS-sensitive and the
    paper's gap definition differs from the AEC relative gap."""
    scenario = load_or_skip("siouxfalls")
    iters = {}
    for name, model in (
        ("fw", FrankWolfeModel()),
        ("cfw", ConjugateFrankWolfeModel()),
        ("bfw", BiconjugateFrankWolfeModel()),
    ):
        trace = _solve(scenario, model, iterations=4000, target_relative_gap=1e-4)
        iters[name] = trace.final.coords.iterations
    # Plain FW needs many times the iterations of either conjugate variant.
    assert iters["fw"] > 2 * iters["cfw"]
    assert iters["fw"] > 2 * iters["bfw"]
