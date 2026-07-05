"""Tests for Bar-Gera's (2002) Origin-Based Assignment (oba).

OBA is a bush-based exact UE solver, distinct from Algorithm B: it rebalances
approach *proportions* toward the least-mean-cost approach using mean-cost (M)
and derivative (D) labels, rather than shifting flow between the longest/shortest
paths. Being an exact solver it converges to the unique UE link flows, so it is
validated the same way as the other bush/path solvers: the analytic Braess UE,
cross-family link-flow agreement with algb, the published Sioux Falls Beckmann
optimum, a monotonically shrinking certified gap, and exact demand conservation
at every checkpoint (the regression that pins the dead-node activation fix).
"""

import numpy as np
import pytest
from conftest import load_or_skip

from tabench import (
    AlgorithmBModel,
    Budget,
    Demand,
    Evaluator,
    Network,
    OriginBasedModel,
    RngBundle,
    Scenario,
    Trace,
    braess_scenario,
)

REF_FLOWS = np.array([4.0, 2.0, 2.0, 2.0, 4.0])
# Published Sioux Falls optimal Beckmann objective (TransportationNetworks), in
# the repo's native units (the stored fft are 0.01 h, hence the 1e5 factor); the
# same anchor test_validation and test_siouxfalls pin.
SIOUXFALLS_TNTP_OBJECTIVE = 42.31335287107440
SIOUXFALLS_UNIT_FACTOR = 1e5


@pytest.fixture(scope="module")
def braess():
    return braess_scenario()


@pytest.fixture(scope="module")
def siouxfalls():
    return load_or_skip("siouxfalls")


def _solve(scenario, model=None, **budget_kwargs):
    trace = Trace()
    (model or OriginBasedModel()).solve(
        scenario, Budget(**budget_kwargs), RngBundle(0), trace
    )
    return trace


# ------------------------------------------------------------------ Braess
def test_analytic_braess_equilibrium(braess):
    trace = _solve(braess, iterations=30, target_relative_gap=1e-13)
    metrics = Evaluator(braess).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert abs(metrics["relative_gap"]) < 1e-10
    np.testing.assert_allclose(trace.final.link_flows, REF_FLOWS, atol=1e-5)


# ------------------------------------------------------------- Sioux Falls
def test_deep_convergence_on_siouxfalls(siouxfalls):
    """OBA drives the certified gap deep (it was the first method to reach
    near-machine-precision gaps where Frank-Wolfe stalls). The exact tail rate is
    BLAS-sensitive, so this pins the robust property: certified below 1e-8 within
    a modest budget, flows matching the best-known solution."""
    trace = _solve(siouxfalls, iterations=60, target_relative_gap=1e-10)
    metrics = Evaluator(siouxfalls).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-8
    assert np.abs(trace.final.link_flows - siouxfalls.reference.link_flows).max() < 1e-2


def test_cross_solver_agreement_with_algb(siouxfalls):
    """OBA and Algorithm B are different algorithms that must converge to the
    SAME unique UE link flows — the core correctness check for OBA (best-known
    flows are unique for UE). Deep-converge both and compare link flows."""
    v_oba = _solve(siouxfalls, iterations=80, target_relative_gap=1e-10).final.link_flows
    algb = Trace()
    AlgorithmBModel().solve(
        siouxfalls, Budget(iterations=80, target_relative_gap=1e-10), RngBundle(0), algb
    )
    v_algb = algb.final.link_flows
    rel = np.abs(v_oba - v_algb) / np.maximum(np.abs(v_algb), 1.0)
    assert rel.max() < 1e-5  # two solver families, one equilibrium


def test_matches_published_beckmann_optimum(siouxfalls):
    """The converged OBA flows reproduce the published Sioux Falls optimal
    Beckmann objective to high precision (external validation of flows AND
    objective, unit-free up to the network's 1e5 factor)."""
    v = _solve(siouxfalls, iterations=80, target_relative_gap=1e-11).final.link_flows
    obj = Evaluator(siouxfalls).evaluate(v)["beckmann_objective"] / SIOUXFALLS_UNIT_FACTOR
    assert obj == pytest.approx(SIOUXFALLS_TNTP_OBJECTIVE, rel=1e-6)


def test_monotone_gap_decrease(siouxfalls):
    """OBA's signature: the certified relative gap shrinks (near-)monotonically
    toward machine precision, rather than tailing off like Frank-Wolfe."""
    trace = _solve(siouxfalls, iterations=40)
    gaps = [s.self_report["relative_gap"] for s in trace]
    # Allow only float-noise-scale non-monotonicity on the deep tail.
    assert all(gaps[i] >= gaps[i + 1] - 1e-12 for i in range(len(gaps) - 1))
    assert gaps[-1] < gaps[0]


# ------------------------------------------------------- conservation teeth
def test_flows_conserve_demand_on_siouxfalls(siouxfalls):
    """REGRESSION for the dead-node activation fix: when a proportion shift routes
    flow onto a link out of a node that carried zero flow at the pass start, the
    alpha->flow rebuild must pull matching inflow into that node (uniform
    dead-node proportions), or it emits a phantom source. Without the fix Sioux
    Falls' node-balance residual blows up to ~1200 by the first iteration; with
    it, flows conserve demand to the float-noise floor at every checkpoint."""
    for iters in (1, 3, 20):
        v = _solve(siouxfalls, iterations=iters).final.link_flows
        metrics = Evaluator(siouxfalls).evaluate(v)
        assert np.all(v >= 0)
        assert metrics["node_balance_residual"] <= 1e-6 * siouxfalls.demand.total


def _seed60_high_curvature_scenario():
    """A 6-node, all-power-4, congested multi-origin network (from an adversarial
    fuzz) on which the RAW Newton step limit-cycles. Hardcoded so the regression
    is deterministic."""
    init = np.array([1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6], dtype=np.int64)
    term = np.array([2, 3, 4, 1, 5, 6, 2, 4, 1, 2, 3, 5, 6, 1, 2, 3, 4, 4, 5], dtype=np.int64)
    cap = np.array([1.2234, 1.2038, 1.5758, 1.6027, 1.6185, 3.0149, 1.2294, 2.503, 1.1163,
                    2.4804, 3.9527, 2.1138, 3.7177, 2.4418, 3.7431, 3.9999, 2.8382, 2.5788, 1.3576])
    fft = np.array([1.4091, 3.183, 2.3146, 4.1823, 3.4566, 1.0628, 1.3345, 1.3548, 2.1421,
                    1.7182, 2.7441, 1.4667, 2.7193, 3.1451, 2.4086, 3.5294, 2.675, 4.0104, 4.7781])
    m = len(init)
    net = Network(
        name="seed60-highcurv", n_nodes=6, n_zones=6, first_thru_node=1,
        init_node=init, term_node=term, capacity=cap, length=np.zeros(m),
        free_flow_time=fft, b=np.full(m, 0.15), power=np.full(m, 4.0),
        toll=np.zeros(m), link_type=np.ones(m, dtype=np.int64),
    )
    od = np.zeros((6, 6))
    for (i, j), val in {
        (0, 1): 7.1346, (0, 2): 8.123, (0, 3): 6.1997, (0, 4): 4.1655, (1, 0): 7.3985,
        (1, 2): 5.1896, (2, 1): 5.1951, (3, 2): 3.5809, (3, 4): 8.8344, (4, 0): 7.7107,
        (4, 1): 5.0294, (4, 2): 3.9438, (5, 0): 3.3682, (5, 2): 3.9455, (5, 3): 3.176,
    }.items():
        od[i, j] = val
    return Scenario("seed60-highcurv", net, Demand(od))


def test_step_damping_prevents_high_curvature_limit_cycle():
    """REGRESSION for the Newton-overshoot fix. OBA's D label is only an
    approximate second derivative, so on high-curvature (BPR power > 1) objectives
    the RAW step (step_scale=1) overshoots and limit-cycles: on this instance it
    stalls at a ~30% certified gap forever. The damped default (step_scale=0.5)
    keeps the exact direction with a stable magnitude and drives it to machine
    precision, agreeing with Algorithm B on the unique UE flows. An adversarial
    fuzz found the raw step stalls on 25/357 such nets (worst 73%); damped, 0."""
    sc = _seed60_high_curvature_scenario()
    # Raw step limit-cycles: big gap even with a large iteration budget.
    raw = _solve(sc, OriginBasedModel(step_scale=1.0), iterations=800, target_relative_gap=1e-12)
    assert Evaluator(sc).evaluate(raw.final.link_flows)["relative_gap"] > 1e-2
    # Damped default converges to the UE and matches Algorithm B.
    v_oba = _solve(sc, iterations=800, target_relative_gap=1e-11).final.link_flows
    m_oba = Evaluator(sc).evaluate(v_oba)
    assert m_oba["feasible"] == 1.0
    assert m_oba["relative_gap"] < 1e-8
    algb = Trace()
    AlgorithmBModel().solve(
        sc, Budget(iterations=800, target_relative_gap=1e-11), RngBundle(0), algb
    )
    rel = np.abs(v_oba - algb.final.link_flows) / np.maximum(np.abs(algb.final.link_flows), 1.0)
    assert rel.max() < 1e-5


def test_restricted_centroids_honored():
    """Anaheim (first_thru_node=39): both built-in anchors have ftn=1, so this is
    the only test exercising OBA's shadow-node reconstruction path (the term/
    x_node build over the centroid-split expanded graph). A clean certified solve
    means no bush ever routes through a centroid and the rebuild conserves on the
    expanded graph."""
    scenario = load_or_skip("anaheim")
    trace = _solve(scenario, iterations=60, target_relative_gap=1e-9)
    metrics = Evaluator(scenario).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-7


# -------------------------------------------------------------- mechanics
def test_registry_and_paradigm():
    from tabench.models import MODEL_REGISTRY

    assert "oba" in MODEL_REGISTRY
    assert MODEL_REGISTRY["oba"]().capabilities.paradigm == "static_ue"


def test_budget_and_bookkeeping(braess):
    trace = _solve(braess, iterations=6)
    assert len(trace) == 6
    assert trace.final.coords.iterations == 6
    for key in ("relative_gap", "tstt", "sptt", "beckmann", "bush_scan_rounds"):
        assert key in trace.final.self_report
    # sp_calls are strictly increasing (one honest-gap AON + scan rounds per iter).
    calls = [s.coords.sp_calls for s in trace]
    assert all(calls[i] < calls[i + 1] for i in range(len(calls) - 1))
