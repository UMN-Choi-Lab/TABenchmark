"""Tests for elastic (variable) demand UE — ``fw-elastic`` + the adr-005 certificate.

The demand between an OD pair is a decreasing function of its equilibrium cost,
``d = D(u)``. We solve it as a fixed-demand UE on the Gartner excess-demand
augmented network (Sheffi 1985 ch. 6; Boyles TNA sec. 9.1) and certify it
purely from the emitted real link flows: the harness recomputes the
demand-consistent demand ``d* = D(u(v))`` and scores route equilibrium
(relative_gap) and demand consistency (node_balance vs d*) against it.

The anchor is a two-route network with linear demand whose elastic UE is exact
and rational (recomputed here with brentq, not trusted): ``u=5, f_A=3, f_B=2``,
realized demand 5, link flows ``(3,3,2,2)``.
"""

import dataclasses

import numpy as np
import pytest
from conftest import load_or_skip
from scipy.optimize import brentq

from tabench import (
    Budget,
    Demand,
    ElasticDemand,
    ElasticDemandFWModel,
    Evaluator,
    FrankWolfeModel,
    Network,
    RngBundle,
    Scenario,
    Trace,
    braess_scenario,
    elastic_two_route_scenario,
)
from tabench.models._paths import PathEngine

# The pinned golden Braess hash must survive adding the elastic_demand field.
GOLDEN_BRAESS_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"


@pytest.fixture(scope="module")
def elastic():
    return elastic_two_route_scenario()


def _solve(scenario, model=None, **budget_kwargs):
    trace = Trace()
    (model or ElasticDemandFWModel()).solve(
        scenario, Budget(**budget_kwargs), RngBundle(0), trace
    )
    return trace


def _analytic_two_route(d0=10.0, u0=10.0):
    """Recompute the two-route elastic UE independently (no trusted digits).

    Both routes used: c_A = 2 + f_A, c_B = 3 + f_B, common cost u, so
    f_A = u - 2, f_B = u - 3, and demand consistency f_A + f_B = D(u).
    """
    def g(u):
        return (u - 2.0) + (u - 3.0) - d0 * max(0.0, 1.0 - u / u0)

    u = brentq(g, 3.0, u0)
    f_a, f_b = u - 2.0, u - 3.0
    return u, f_a, f_b


# ----------------------------------------------------------------- UE anchor
def test_analytic_elastic_equilibrium(elastic):
    u, f_a, f_b = _analytic_two_route()
    trace = _solve(elastic, iterations=300, target_relative_gap=1e-13)
    v = trace.final.link_flows
    metrics = Evaluator(elastic).evaluate(v)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-9
    assert metrics["node_balance_residual"] < 1e-8
    np.testing.assert_allclose(v, [f_a, f_a, f_b, f_b], atol=1e-6)
    # Certified realized demand equals D(u) = f_A + f_B, recomputed above.
    assert metrics["realized_demand"] == pytest.approx(f_a + f_b, abs=1e-6)


def test_certificate_recomputes_demand_from_flows(elastic):
    """The harness never trusts the model: it recomputes d* = D(u(v)) from the
    emitted flows and the content-hashed demand law. At the oracle flows the
    certified realized demand equals D evaluated at the independently computed
    equilibrium cost."""
    u, f_a, f_b = _analytic_two_route()
    v = np.array([f_a, f_a, f_b, f_b])  # exact oracle, not the solver output
    metrics = Evaluator(elastic).evaluate(v)
    assert metrics["realized_demand"] == pytest.approx(10.0 * (1.0 - u / 10.0), abs=1e-9)
    assert metrics["relative_gap"] < 1e-9
    assert metrics["feasible"] == 1.0


# ------------------------------------------------------------- demand forms
def test_exponential_form_converges(elastic):
    sc = dataclasses.replace(
        elastic, elastic_demand=ElasticDemand("exponential", param=0.2), reference=None
    )
    metrics = Evaluator(sc).evaluate(
        _solve(sc, iterations=400, target_relative_gap=1e-11).final.link_flows
    )
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-6
    assert metrics["node_balance_residual"] < 1e-5
    assert 0.0 < metrics["realized_demand"] < 10.0  # strictly elastic, never zero


def test_inverse_demand_roundtrips_and_excess_cost_monotone():
    for form, param in (("linear", 10.0), ("exponential", 0.3)):
        law = ElasticDemand(form, param)
        d0 = np.array([8.0])
        d = np.array([3.0])
        # D(D^{-1}(d)) == d
        u = law.inverse_demand(d0, d)
        np.testing.assert_allclose(law.realized_demand(d0, u), d, atol=1e-9)
        # excess-arc cost W(e) = D^{-1}(d0 - e) is increasing in e and W(0)=0
        e = np.linspace(0.0, float(d0[0]) * 0.99, 40)
        w = law.excess_arc_cost(np.full_like(e, d0[0]), e)
        assert w[0] == pytest.approx(0.0, abs=1e-9)
        assert np.all(np.diff(w) > 0)


# --------------------------------------------------------- certificate teeth
def test_fixed_demand_solver_is_censored(elastic):
    """A fixed-demand solver routes all of d0, not the demand-consistent
    d* = D(u(v)); its zone divergences therefore disagree with d*, so the
    demand-consistency gate censors it. Being at the elastic equilibrium IS the
    feasibility requirement — an off-equilibrium routing is not a valid elastic
    solution (there is no fixed demand for it to be feasible-but-suboptimal
    against)."""
    trace = _solve(elastic, FrankWolfeModel(), iterations=100, target_relative_gap=1e-12)
    metrics = Evaluator(elastic).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 0.0
    assert metrics["node_balance_residual"] > 1e-2  # routes d0, not D(u(v))


def test_phantom_circulation_is_censored():
    """SOUNDNESS regression: a flow that routes ZERO OD demand but spins around
    a through-node cycle conserves flow everywhere, yet must NOT certify as an
    elastic UE. The demand-consistency gate (node_balance vs d*) catches it:
    the cycle produces no flow out of the origin zone, but d*=D(u)>0 demands it.
    """
    # Zones 1,2 (OD 1->2, d0=10, u0=10); through nodes 3,4,5; direct 1->2 link
    # (fft 5) plus a directed cycle 3->4->5->3.
    init = np.array([1, 3, 4, 5], dtype=np.int64)
    term = np.array([2, 4, 5, 3], dtype=np.int64)
    net = Network(
        name="phantom", n_nodes=5, n_zones=2, first_thru_node=3,
        init_node=init, term_node=term, capacity=np.ones(4), length=np.zeros(4),
        free_flow_time=np.array([5.0, 1.0, 1.0, 1.0]), b=np.zeros(4), power=np.ones(4),
        toll=np.zeros(4), link_type=np.ones(4, dtype=np.int64),
    )
    od = np.zeros((2, 2))
    od[0, 1] = 10.0
    sc = Scenario("phantom", net, Demand(od), elastic_demand=ElasticDemand("linear", 10.0))
    ev = Evaluator(sc)
    for cycle_flow in (0.0, 2.43, 5.0):  # any tuned magnitude — none may pass
        metrics = ev.evaluate(np.array([0.0, cycle_flow, cycle_flow, cycle_flow]))
        assert metrics["feasible"] == 0.0
        assert np.isnan(metrics["relative_gap"])


def test_intrazonal_demand_excluded_from_realized(elastic):
    """Intrazonal (diagonal) reference demand never enters the network and must
    not be counted in the certified realized demand."""
    sc = dataclasses.replace(
        elastic, demand=Demand(np.array([[7.0, 10.0], [0.0, 0.0]])), reference=None
    )
    trace = _solve(sc, iterations=200, target_relative_gap=1e-12)
    metrics = Evaluator(sc).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["realized_demand"] == pytest.approx(5.0, abs=1e-4)  # not 12


def test_zero_flow_is_censored(elastic):
    """Zero flow routes no demand but D(free-flow cost) > 0 here, so the
    negative-excess guard censors it (it cannot game a perfect gap)."""
    metrics = Evaluator(elastic).evaluate(np.zeros(4))
    assert metrics["feasible"] == 0.0
    assert np.isnan(metrics["relative_gap"])


def test_full_suppression_is_feasible():
    """If every route costs more than u0, demand is fully suppressed: the
    unique elastic UE is zero flow / zero realized demand, and it certifies."""
    # Min route cost is 2 (c_A at f=0); u0 = 1 < 2 suppresses all demand.
    sc = dataclasses.replace(
        elastic_two_route_scenario(),
        elastic_demand=ElasticDemand("linear", param=1.0),
        reference=None,
    )
    metrics = Evaluator(sc).evaluate(np.zeros(4))
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["realized_demand"] == pytest.approx(0.0, abs=1e-9)
    # ... and the solver finds it.
    v = _solve(sc, iterations=50, target_relative_gap=1e-10).final.link_flows
    np.testing.assert_allclose(v, 0.0, atol=1e-9)


def test_high_u0_recovers_fixed_demand():
    """As u0 -> infinity the demand becomes inelastic: realized -> d0."""
    sc = dataclasses.replace(
        elastic_two_route_scenario(),
        elastic_demand=ElasticDemand("linear", param=1e6),
        reference=None,
    )
    metrics = Evaluator(sc).evaluate(
        _solve(sc, iterations=200, target_relative_gap=1e-10).final.link_flows
    )
    assert metrics["feasible"] == 1.0
    assert metrics["realized_demand"] == pytest.approx(10.0, rel=1e-3)


# ------------------------------------------------------------------ hashing
def test_golden_braess_hash_preserved():
    assert braess_scenario().content_hash() == GOLDEN_BRAESS_HASH


def test_elastic_demand_hashed_only_when_set(elastic):
    """Adding a decay law changes the hash; a fixed-demand scenario with the
    same network/demand hashes as if the field never existed; and form/param
    are distinguished."""
    fixed = dataclasses.replace(elastic, elastic_demand=None)
    exp = dataclasses.replace(elastic, elastic_demand=ElasticDemand("exponential", 0.2))
    other_param = dataclasses.replace(elastic, elastic_demand=ElasticDemand("linear", 20.0))
    hashes = {
        elastic.content_hash(),
        fixed.content_hash(),
        exp.content_hash(),
        other_param.content_hash(),
    }
    assert len(hashes) == 4  # all distinct


def test_elastic_and_sue_mutually_exclusive():
    sc = elastic_two_route_scenario()
    with pytest.raises(ValueError, match="mutually exclusive"):
        dataclasses.replace(sc, sue_theta=0.5)


def test_elastic_demand_validation():
    with pytest.raises(ValueError, match="form"):
        ElasticDemand("quadratic", 1.0)
    with pytest.raises(ValueError, match="param"):
        ElasticDemand("linear", 0.0)


# --------------------------------------------------------------- path engine
def test_od_cost_matrix_matches_route_cost(elastic):
    """The certificate's OD-cost skim returns the true shortest-path cost."""
    engine = PathEngine(elastic.network)
    v = np.array([3.0, 3.0, 2.0, 2.0])
    costs = elastic.network.link_cost(v)
    kappa = engine.od_cost_matrix(costs, elastic.demand)
    # Route A cost 2 + f_A = 5; route B cost 3 + f_B = 5; SP = min = 5.
    assert kappa[0, 1] == pytest.approx(5.0, abs=1e-9)
    assert kappa.shape == (2, 2)
    assert kappa[1, 0] == 0.0  # no demand -> not computed


# --------------------------------------------------------------------- scale
def test_convergence_on_siouxfalls():
    """Scale demonstration. fw-elastic runs FW on an augmented network dominated
    by ~O(zones^2) excess-demand arcs, so its tail is slower than fixed-demand
    FW (adr-005): both the self-reported real-route gap and the certified
    demand-consistency residual shrink monotonically toward the elastic UE, but
    Sioux Falls needs many iterations to clear the strict feasibility gate — so
    this asserts convergence, not certification within a small budget."""
    base = load_or_skip("siouxfalls")
    sc = dataclasses.replace(
        base,
        elastic_demand=ElasticDemand("linear", param=69.0),
        reference=None,
        name="siouxfalls-elastic",
    )
    coarse = _solve(sc, iterations=200)
    fine = _solve(sc, iterations=1000)
    m_coarse = Evaluator(sc).evaluate(coarse.final.link_flows)
    m_fine = Evaluator(sc).evaluate(fine.final.link_flows)
    # node_balance_residual (demand consistency vs d*) is reported even when a
    # checkpoint is censored, and it shrinks with more work.
    assert m_fine["node_balance_residual"] < m_coarse["node_balance_residual"]
    # ... as does the self-reported real-route gap.
    assert fine.final.self_report["relative_gap"] < coarse.final.self_report["relative_gap"]
    assert fine.final.self_report["relative_gap"] < 1e-3
    # Elasticity is genuinely active (some demand priced out) at both points.
    for tr in (coarse, fine):
        realized = tr.final.self_report["realized_demand"]
        assert 0.0 < realized < base.demand.total
    # Well-converged demand consistency even if the strict 1e-6 gate isn't met.
    assert m_fine["node_balance_residual"] < 1e-4 * base.demand.total


# ----------------------------------------------------------- bookkeeping
def test_bookkeeping_and_budget(elastic):
    trace = _solve(elastic, iterations=8)
    assert len(trace) == 8
    assert trace.final.coords.iterations == 8
    # sp_calls = 1 (init AON) + one shortest-path sweep per recorded iteration.
    assert trace.final.coords.sp_calls == 8 + 1
    v = trace.final.link_flows
    assert np.all(v >= 0)
    for key in ("relative_gap", "realized_demand", "unmet_demand", "beckmann"):
        assert key in trace.final.self_report
    # unmet + realized = total reference demand at every checkpoint.
    for state in trace:
        assert state.self_report["realized_demand"] + state.self_report[
            "unmet_demand"
        ] == pytest.approx(elastic.demand.total, abs=1e-6)
