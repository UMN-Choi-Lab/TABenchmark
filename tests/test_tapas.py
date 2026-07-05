"""Tests for TAPAS (Bar-Gera 2010), paired-alternative-segment UE.

Covers the shared UE guarantees (analytic Braess; certified deep convergence on
Sioux Falls; link-flow agreement with Algorithm B, since UE link flows are
unique; restricted centroids on Anaheim; the zero-derivative full-shift
fallback; conservation and budget accounting) plus what makes TAPAS TAPAS: the
proportionality adjustment (Boyles TNA v1.0 eq. 6.100). The proportionality
tests pin three things -- that the shift drives the *route* flows proportional
(a 5-order-of-magnitude residual drop the pure-UE run never achieves), that it
holds *link* flows fixed while doing so (its defining invariant), and that the
residual diagnostic actually discriminates per-origin proportionality on
link-flow-identical inputs (a naive aggregate check would not).
"""

import dataclasses

import numpy as np
import pytest
from conftest import load_or_skip

from tabench import (
    AlgorithmBModel,
    Budget,
    Evaluator,
    RngBundle,
    TapasModel,
    Trace,
    braess_scenario,
)
from tabench.models._bush import _BushState

REF_FLOWS = np.array([4.0, 2.0, 2.0, 2.0, 4.0])


@pytest.fixture(scope="module")
def braess():
    return braess_scenario()


@pytest.fixture(scope="module")
def siouxfalls():
    return load_or_skip("siouxfalls")


def _solve(scenario, model=None, **budget_kwargs):
    trace = Trace()
    (model or TapasModel()).solve(scenario, Budget(**budget_kwargs), RngBundle(0), trace)
    return trace


# ---------------------------------------------------------------- UE anchors
def test_analytic_braess_equilibrium(braess):
    trace = _solve(braess, iterations=30, target_relative_gap=1e-14)
    metrics = Evaluator(braess).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert abs(metrics["relative_gap"]) < 1e-10
    np.testing.assert_allclose(trace.final.link_flows, REF_FLOWS, atol=1e-4)


def test_deep_convergence_on_siouxfalls(siouxfalls):
    """Certified convergence on Sioux Falls (1e-8 reached by ~22 iterations).

    The tail rate is BLAS-sensitive like algb's, so this pins the robust
    cross-platform property: certified below 1e-6 within 30 iterations, flows
    matching the best-known solution.
    """
    trace = _solve(siouxfalls, iterations=30, target_relative_gap=1e-10)
    metrics = Evaluator(siouxfalls).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-6
    assert np.abs(trace.final.link_flows - siouxfalls.reference.link_flows).max() < 1e-2


def test_link_flows_match_algb_on_siouxfalls(siouxfalls):
    """UE link flows are unique for strictly increasing separable BPR, so TAPAS
    and Algorithm B must converge to the same v despite different (non-unique)
    route/segment decompositions. Cross-checks aggregate link flows only.
    """
    tapas = _solve(siouxfalls, iterations=40, target_relative_gap=1e-8)
    algb = _solve(siouxfalls, AlgorithmBModel(), iterations=40, target_relative_gap=1e-8)
    assert np.abs(tapas.final.link_flows - algb.final.link_flows).max() < 1e-1


# ----------------------------------------------------------- proportionality
def test_proportionality_shift_drives_route_flows_proportional(siouxfalls):
    """The headline: with the proportionality adjustment on (prop_rounds>0) the
    route flows become proportional (residual collapses), while pure UE
    (prop_rounds=0) leaves them badly non-proportional at the SAME link flows.
    Measured ~1.4e-2 vs ~7e-8 at 20 iterations; asserted with wide margins.
    """
    pure = _solve(siouxfalls, TapasModel(prop_rounds=0), iterations=20)
    prop = _solve(siouxfalls, TapasModel(prop_rounds=5), iterations=20)
    r_pure = pure.final.self_report["proportionality_residual"]
    r_prop = prop.final.self_report["proportionality_residual"]
    assert r_pure > 1e-3  # pure UE is far from proportional
    assert r_prop < 1e-4  # the adjustment restores proportionality
    assert r_pure > 50 * r_prop  # ...by orders of magnitude


def test_proportionality_residual_discriminates():
    """The residual must distinguish per-origin proportionality on inputs with
    IDENTICAL link flows. One isolated PAS (segA=link0, segB=link1) shared by two
    origins; both decompositions give link flows v=[6,6]. Proportional (each
    origin 50/50) scores ~0; non-proportional (origin A all on segA, B all on
    segB) scores a clear positive residual. A naive check reading only aggregate
    link flows would score both zero and fail this.
    """
    model = TapasModel()
    pool = [{"segA": np.array([0]), "segB": np.array([1]), "origins": [0, 1]}]

    def bushes_from(x0, x1):
        b0, b1 = _BushState(2, 3), _BushState(2, 3)
        b0.x = np.array(x0, dtype=float)
        b1.x = np.array(x1, dtype=float)
        return [b0, b1]

    # Proportional: both origins split 50/50 -> aggregate 50/50, no deviation.
    prop = bushes_from([3.0, 3.0], [3.0, 3.0])
    r_prop, m_prop = model._proportionality_residual(pool, prop, demand_total=12.0)
    assert r_prop < 1e-12
    assert m_prop < 1e-12

    # Non-proportional: A entirely on segA, B entirely on segB. Link flows are
    # STILL v=[6,6] -- only the per-origin split differs.
    nonp = bushes_from([6.0, 0.0], [0.0, 6.0])
    v_agg = nonp[0].x + nonp[1].x
    np.testing.assert_allclose(v_agg, [6.0, 6.0])  # identical aggregate link flows
    r_nonp, m_nonp = model._proportionality_residual(pool, nonp, demand_total=12.0)
    assert r_nonp == pytest.approx(0.5)  # (|6-3| + |0-3|) / 12
    assert m_nonp == pytest.approx(0.5)  # each origin's ratio is 0.5 off the mean

    # Unequal weights pin the *flow-weighted* aggregate pi = sum(a)/sum(w)
    # (eq. 6.94), which an unweighted mean(a/w) would get wrong. Origin A: 10 on
    # segA; origin B: 2 on segB -> pi = 10/12 = 5/6 (weighted), not 1/2. The
    # equal-weight cases above cannot tell the two formulas apart; this can.
    uneq = bushes_from([10.0, 0.0], [0.0, 2.0])
    r_uneq, m_uneq = model._proportionality_residual(pool, uneq, demand_total=12.0)
    assert r_uneq == pytest.approx(5 / 18)  # 2*(10 - 10*5/6)/12; mean(a/w) gives 0.5
    assert m_uneq == pytest.approx(5 / 6)  # origin B: |0 - 5/6|; mean(a/w) gives 0.5


def test_proportionality_adjust_isolated_is_exact_and_preserves_link_flows():
    """On an isolated PAS the eq. 6.100 heuristic is exact in one step, and it
    must not move link flows (sum over origins is conserved)."""
    model = TapasModel()
    pas = {"segA": np.array([0]), "segB": np.array([1]), "origins": [0, 1]}
    b0, b1 = _BushState(2, 3), _BushState(2, 3)
    b0.x = np.array([6.0, 0.0])
    b1.x = np.array([0.0, 6.0])
    v_before = b0.x + b1.x
    model._proportionality_adjust(pas, [b0, b1])
    v_after = b0.x + b1.x
    np.testing.assert_allclose(v_after, v_before, atol=1e-12)  # link flows fixed
    np.testing.assert_allclose(b0.x, [3.0, 3.0], atol=1e-12)  # now proportional
    np.testing.assert_allclose(b1.x, [3.0, 3.0], atol=1e-12)
    r, _ = model._proportionality_residual([pas], [b0, b1], demand_total=12.0)
    assert r < 1e-12


def test_proportionality_adjust_preserves_link_flows_on_siouxfalls(siouxfalls):
    """The invariant on a real multi-origin pool: a full proportionality pass
    leaves aggregate link flows unchanged to float precision, even as it moves
    per-origin flow (the residual drops)."""
    model = TapasModel()
    model._setup(siouxfalls)
    network = model._network
    v = np.zeros(model._n_links)
    bushes, v, _ = model._initial_bushes(v)
    t = network.link_cost(v)
    dt = network.link_cost_derivative(v)
    for _ in range(4):  # build a rich, disturbed multi-origin pool
        for b, o in zip(bushes, model._origins, strict=True):
            model._update_bush(b, int(o), t)
        for _ in range(4):
            for pas in model._identify_pas(bushes, t):
                model._cost_equilibrate(pas, bushes, v, t, dt)
        v = sum((b.x for b in bushes), np.zeros(model._n_links))
        t = network.link_cost(v)
        dt = network.link_cost_derivative(v)
    pool = model._identify_pas(bushes, t)
    assert any(len(p["origins"]) >= 2 for p in pool)  # a genuine multi-origin pool
    v_before = sum((b.x for b in bushes), np.zeros(model._n_links))
    r_before, _ = model._proportionality_residual(pool, bushes, siouxfalls.demand.total)
    for pas in pool:
        model._proportionality_adjust(pas, bushes)
    v_after = sum((b.x for b in bushes), np.zeros(model._n_links))
    r_after, _ = model._proportionality_residual(pool, bushes, siouxfalls.demand.total)
    assert np.abs(v_before - v_after).max() < 1e-9  # link flows conserved
    assert r_after < r_before  # ...while proportionality improves
    assert all((b.x >= -1e-12).all() for b in bushes)  # feasibility kept


# --------------------------------------------------------- shared invariants
def test_zero_derivative_certifies_in_one_iteration(braess):
    """All-constant costs: the re-probe full-shift lands on AON = UE at once."""
    net = braess.network
    constant = dataclasses.replace(
        net,
        b=np.zeros(net.n_links),
        free_flow_time=np.array([10.0, 50.0, 10.0, 50.0, 10.0]),
    )
    scenario = dataclasses.replace(braess, network=constant, reference=None)
    trace = _solve(scenario, iterations=1)
    metrics = Evaluator(scenario).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert abs(metrics["relative_gap"]) < 1e-12


def test_restricted_centroids_honored():
    """Anaheim (first_thru_node=39): bushes never route through centroids."""
    scenario = load_or_skip("anaheim")
    trace = _solve(scenario, iterations=25, target_relative_gap=1e-10)
    metrics = Evaluator(scenario).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-8


def test_bookkeeping_invariants(siouxfalls):
    """Nonnegative flows and demand conservation at the emitted checkpoints."""
    trace = _solve(siouxfalls, iterations=20)
    v = trace.final.link_flows
    metrics = Evaluator(siouxfalls).evaluate(v)
    assert metrics["feasible"] == 1.0
    assert np.all(v >= 0)
    assert metrics["node_balance_residual"] <= 1e-6 * siouxfalls.demand.total


def test_budget_accounting(braess):
    trace = _solve(braess, iterations=7)
    assert len(trace) == 7
    assert trace.final.coords.iterations == 7
    first = trace.checkpoints[0]
    # sp_calls at iteration 1 = 1 (init Dijkstra) + bush_scan_rounds + 1 (honest
    # gap AON); the exact round count is convergence-dependent, so pin the
    # accounting identity rather than a magic number. Proportionality
    # adjustments do no shortest-path work and are excluded (recorded separately).
    assert first.coords.sp_calls == first.self_report["bush_scan_rounds"] + 2
    for key in (
        "proportionality_residual",
        "pas_proportionality_max",
        "pas_pool_size",
        "pas_cost_shift_rounds",
        "pas_prop_rounds",
    ):
        assert key in first.self_report
