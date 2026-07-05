"""Tests for Evans (1976) combined trip distribution + assignment — the ``evans``
model + the adr-007 certificate.

Only the trip-end margins (productions ``O_i``, attractions ``D_j``) and the
gravity dispersion ``beta`` are fixed; the OD matrix is endogenous, distributed
by a doubly-constrained gravity model at the equilibrium costs. We solve it with
Evans' partial-linearization Frank-Wolfe and certify it purely from the emitted
link flows: the harness recomputes ``d* = gravity(u(v))`` and scores route
equilibrium (relative_gap) and demand consistency (node_balance vs d*).

The anchor is a symmetric bipartite network whose doubly-constrained gravity
collapses to a binary logit split — a scalar fixed point recomputed here with
brentq (no trusted digits): ``d_13 = d_24 = p``, ``d_14 = d_23 = q = 10 - p``.
"""

import dataclasses

import numpy as np
import pytest
from scipy.optimize import brentq

from tabench import (
    Budget,
    CombinedDemand,
    Demand,
    ElasticDemand,
    Evaluator,
    EvansCombinedModel,
    FrankWolfeModel,
    Network,
    RngBundle,
    Scenario,
    Trace,
    braess_scenario,
    evans_symmetric_scenario,
    load_scenario,
    run_experiment,
)
from tabench.models import MODEL_REGISTRY

# The pinned golden Braess hash must survive adding the combined_demand field.
GOLDEN_BRAESS_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"

TRIPS = 10.0
BETA = 0.5


@pytest.fixture(scope="module")
def evans():
    return evans_symmetric_scenario()


def _solve(scenario, model=None, **budget_kwargs):
    trace = Trace()
    (model or EvansCombinedModel()).solve(
        scenario, Budget(**budget_kwargs), RngBundle(0), trace
    )
    return trace


def _analytic_split(trips=TRIPS, beta=BETA):
    """Independently recompute the symmetric logit split (no trusted digits).

    By symmetry ``d_13 = d_24 = p`` on the cheaper "near" links and
    ``d_14 = d_23 = q = trips - p`` on the "far" links; the balancing factors
    cancel, leaving ``p = trips / (1 + exp(beta (c_near(p) - c_far(q))))``.
    """

    def g(p):
        c_near = 1.0 + 0.1 * p
        c_far = 3.0 + 0.1 * (trips - p)
        return p - trips / (1.0 + np.exp(beta * (c_near - c_far)))

    p = brentq(g, 0.0, trips)
    return p, trips - p  # (near-link flow, far-link flow)


# ------------------------------------------------------------------- UE anchor
def test_analytic_evans_equilibrium(evans):
    p, q = _analytic_split()
    trace = _solve(evans, iterations=200, target_relative_gap=1e-13)
    v = trace.final.link_flows
    metrics = Evaluator(evans).evaluate(v)
    assert metrics["feasible"] == 1.0
    assert abs(metrics["relative_gap"]) < 1e-9
    assert metrics["node_balance_residual"] < 1e-8
    # Link order 1->3 (near), 1->4 (far), 2->3 (far), 2->4 (near).
    np.testing.assert_allclose(v, [p, q, q, p], atol=1e-6)
    # Certified realized demand equals the total trips 2 * TRIPS (margins fixed).
    assert metrics["realized_demand"] == pytest.approx(2.0 * TRIPS, abs=1e-6)
    # beta genuinely bites: the equilibrium is NOT the beta-free 50/50 split.
    assert abs(p - TRIPS / 2.0) > 0.5


def test_certificate_recomputes_gravity_from_flows(evans):
    """The harness never trusts the model: it recomputes d* = gravity(u(v)) from
    the emitted flows and the content-hashed margins/beta. At the oracle flows
    (not the solver output) the combined equilibrium certifies exactly."""
    p, q = _analytic_split()
    v = np.array([p, q, q, p])  # exact oracle, computed above
    metrics = Evaluator(evans).evaluate(v)
    assert metrics["feasible"] == 1.0
    assert abs(metrics["relative_gap"]) < 1e-9
    assert metrics["realized_demand"] == pytest.approx(2.0 * TRIPS, abs=1e-9)


def test_self_report_matches_certificate(evans):
    """The solver's self-reported relative_gap is defined to equal the harness's
    scored combined gap (adr-007), so at the emitted flows they agree to
    floating precision — the honesty diff is ~0."""
    trace = _solve(evans, iterations=200, target_relative_gap=1e-13)
    v = trace.final.link_flows
    certified = Evaluator(evans).evaluate(v)["relative_gap"]
    assert trace.final.self_report["relative_gap"] == pytest.approx(certified, abs=1e-12)


# ----------------------------------------------------------- gravity / Furness
def test_furness_preserves_margins():
    """The doubly-constrained gravity reproduces both margins exactly (rows) /
    to tolerance (columns), zeros the intrazonal diagonal, and stays on the
    interzonal support."""
    o = np.array([5.0, 3.0, 2.0])
    d = np.array([4.0, 4.0, 2.0])
    cd = CombinedDemand(productions=o, attractions=d, beta=0.3)
    u = np.array([[0.0, 1.0, 2.0], [1.5, 0.0, 3.0], [2.0, 2.5, 0.0]])
    g = cd.gravity(u)
    np.testing.assert_allclose(g.sum(axis=1), o, atol=1e-9)
    np.testing.assert_allclose(g.sum(axis=0), d, atol=1e-8)
    assert np.all(np.diag(g) == 0.0)  # no intrazonal trips
    assert np.all(g >= 0.0)


def test_gravity_matches_hand_computed_logit_row():
    """One production-only row (a single origin, two destinations, unit
    attractiveness) must reduce to the plain multinomial logit."""
    o = np.array([6.0, 0.0])
    d = np.array([0.0, 6.0])
    # Only pair (0,1) is on support -> it must carry all 6 trips.
    cd = CombinedDemand(productions=o, attractions=d, beta=0.7)
    g = cd.gravity(np.array([[0.0, 4.2], [0.0, 0.0]]))
    assert g[0, 1] == pytest.approx(6.0, abs=1e-12)
    assert g.sum() == pytest.approx(6.0, abs=1e-12)


def test_beta_concentrates_demand_on_cheaper_pairs():
    """A larger dispersion beta pushes more demand onto the cheaper "near"
    links — the whole distribution<->assignment pipeline, at two betas."""
    v_lo = _solve(
        evans_symmetric_scenario(beta=0.1), iterations=200, target_relative_gap=1e-11
    ).final.link_flows
    v_hi = _solve(
        evans_symmetric_scenario(beta=2.0), iterations=200, target_relative_gap=1e-11
    ).final.link_flows
    # Near-link flow (index 0) is higher at the higher beta.
    assert v_hi[0] > v_lo[0]
    assert v_lo[0] == pytest.approx(TRIPS / 2.0, abs=0.7)  # near-uniform at small beta


# --------------------------------------------------------- certificate teeth
def test_fixed_demand_solver_is_censored(evans):
    """A fixed-demand solver routes the free-flow-gravity reference, not the
    congested gravity d* = gravity(u(v)); on THIS anchor d* shifts weight onto
    the pricier links, so SPTT(d*) > TSTT and the negative-excess guard censors
    it. Note (see test_aggregate_multicommodity_limitation): unlike elastic
    demand, node_balance stays 0 because the gravity always preserves the O/D
    margins, so the excess guard is the only distributional teeth — and it has a
    documented cost-degenerate hole. This anchor is deliberately non-degenerate,
    so the guard bites here."""
    trace = _solve(evans, FrankWolfeModel(), iterations=100, target_relative_gap=1e-12)
    metrics = Evaluator(evans).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 0.0
    assert np.isnan(metrics["relative_gap"])


def test_anchor_admits_only_the_true_equilibrium():
    """On the (degeneracy-free) anchor, the whole 1-parameter family of
    margin-feasible flows v(s) = (s, T-s, T-s, s) certifies with a zero gap ONLY
    at the true equilibrium split; the cost-equalizing extreme s = T is censored
    (negative excess), so the naive "dump on the cheapest links" flow does not
    game the certificate here."""
    ev = Evaluator(evans_symmetric_scenario())
    p, _ = _analytic_split()
    # The extreme "all trips on the near links" flow is censored (costs 2 vs 3
    # never equalize on this anchor, so d* differs and SPTT(d*) > TSTT).
    m_extreme = ev.evaluate(np.array([TRIPS, 0.0, 0.0, TRIPS]))
    assert m_extreme["feasible"] == 0.0
    # A wrong interior split has a strictly positive certified gap.
    s = p - 2.0
    m_wrong = ev.evaluate(np.array([s, TRIPS - s, TRIPS - s, s]))
    assert m_wrong["feasible"] == 0.0 or m_wrong["relative_gap"] > 1e-6


def test_aggregate_multicommodity_limitation():
    """HONEST LIMITATION (adr-007): the P1 certificate recomputes d* = gravity(u)
    and audits route equilibrium + AGGREGATE node balance. Because the gravity
    always reproduces the fixed O/D margins, node_balance carries no information
    about the OD *distribution*; the only distributional teeth is the
    negative-excess guard, and on a COST-DEGENERATE instance (all OD costs
    identical for every feasible flow) it collapses. This is the same
    aggregate-vs-per-OD limitation the whole harness documents (necessary, not
    sufficient for multi-commodity feasibility) — it is NOT specific to Evans:
    the fixed-demand certificate admits the identical false positive on the same
    network. We pin it transparently rather than hide it. In practice, scenarios
    with a known reference expose such flows through flow_rmse_vs_reference.
    """
    # A degenerate network: constant (flow-independent) link costs, all equal, so
    # every margin-feasible flow induces the SAME uniform OD-cost skim.
    init = np.array([1, 1, 2, 2], dtype=np.int64)
    term = np.array([3, 4, 3, 4], dtype=np.int64)
    net = Network(
        name="degenerate-uniform", n_nodes=4, n_zones=4, first_thru_node=1,
        init_node=init, term_node=term, capacity=np.ones(4), length=np.zeros(4),
        free_flow_time=np.full(4, 2.0), b=np.zeros(4), power=np.ones(4),  # constant cost 2
        toll=np.zeros(4), link_type=np.ones(4, dtype=np.int64),
    )
    o = np.array([10.0, 10.0, 0.0, 0.0])
    d = np.array([0.0, 0.0, 10.0, 10.0])
    cd = CombinedDemand(o, d, beta=0.5)
    dref = cd.gravity(np.full((4, 4), 2.0))  # = (5,5,5,5), a valid reference
    sc = Scenario("degenerate", net, Demand(dref), combined_demand=cd)
    ev = Evaluator(sc)
    # The gravity answer is (5,5,5,5). A WRONG distribution with the right
    # margins certifies as a perfect equilibrium — the known limitation.
    m = ev.evaluate(np.array([10.0, 0.0, 0.0, 10.0]))
    assert m["feasible"] == 1.0
    assert m["relative_gap"] == pytest.approx(0.0, abs=1e-9)
    # ... and the fixed-demand certificate has the IDENTICAL hole on this network
    # (proving the limitation is inherited harness-wide, not introduced by Evans).
    fixed_d = np.zeros((4, 4))
    fixed_d[0, 2] = fixed_d[0, 3] = fixed_d[1, 2] = fixed_d[1, 3] = 5.0
    mf = Evaluator(Scenario("degenerate-fixed", net, Demand(fixed_d))).evaluate(
        np.array([10.0, 0.0, 0.0, 10.0])
    )
    assert mf["feasible"] == 1.0
    assert mf["relative_gap"] == pytest.approx(0.0, abs=1e-9)


def test_phantom_circulation_is_censored():
    """SOUNDNESS regression: a flow routing ZERO OD demand but circulating a
    through-node cycle conserves flow everywhere, yet must not certify as a
    combined equilibrium. The demand-consistency gate (node_balance vs d*)
    catches it: the cycle sends nothing out of the producing zone, but the
    gravity demands O_1 = 10 trips out of it."""
    # Zone 1 produces, zone 2 attracts; through nodes 3,4,5; direct 1->2 link
    # (fft 5) plus a directed cycle 3->4->5->3.
    init = np.array([1, 3, 4, 5], dtype=np.int64)
    term = np.array([2, 4, 5, 3], dtype=np.int64)
    net = Network(
        name="phantom-combined", n_nodes=5, n_zones=2, first_thru_node=3,
        init_node=init, term_node=term, capacity=np.ones(4), length=np.zeros(4),
        free_flow_time=np.array([5.0, 1.0, 1.0, 1.0]), b=np.zeros(4), power=np.ones(4),
        toll=np.zeros(4), link_type=np.ones(4, dtype=np.int64),
    )
    cd = CombinedDemand(np.array([10.0, 0.0]), np.array([0.0, 10.0]), beta=0.5)
    dref = np.zeros((2, 2))
    dref[0, 1] = 10.0
    sc = Scenario("phantom-combined", net, Demand(dref), combined_demand=cd)
    ev = Evaluator(sc)
    for cycle_flow in (0.0, 2.43, 5.0):  # any tuned magnitude — none may pass
        metrics = ev.evaluate(np.array([0.0, cycle_flow, cycle_flow, cycle_flow]))
        assert metrics["feasible"] == 0.0
        assert np.isnan(metrics["relative_gap"])
        assert metrics["node_balance_residual"] == pytest.approx(10.0, abs=1e-9)


def test_zero_flow_is_censored(evans):
    """Zero flow routes no demand, but the gravity demands 2*TRIPS > 0 trips, so
    the demand-consistency / negative-excess guards censor it."""
    metrics = Evaluator(evans).evaluate(np.zeros(4))
    assert metrics["feasible"] == 0.0
    assert np.isnan(metrics["relative_gap"])


# ------------------------------------------------------------------ hashing
def test_golden_braess_hash_preserved():
    assert braess_scenario().content_hash() == GOLDEN_BRAESS_HASH


def test_combined_demand_hashed_only_when_set(evans):
    """Adding trip-end margins + beta changes the hash; a scenario with the same
    network/reference but no combined task hashes as if the field never existed;
    and beta is distinguished."""
    cd = evans.combined_demand
    fixed = dataclasses.replace(evans, combined_demand=None)
    other_beta = dataclasses.replace(
        evans, combined_demand=CombinedDemand(cd.productions, cd.attractions, 0.9)
    )
    hashes = {evans.content_hash(), fixed.content_hash(), other_beta.content_hash()}
    assert len(hashes) == 3  # all distinct


# --------------------------------------------------------------- validation
def test_combined_mutually_exclusive_with_sue_and_elastic(evans):
    with pytest.raises(ValueError, match="mutually exclusive"):
        dataclasses.replace(evans, sue_theta=0.5)
    with pytest.raises(ValueError, match="mutually exclusive"):
        dataclasses.replace(evans, elastic_demand=ElasticDemand("linear", 10.0))


def test_combined_demand_validation():
    with pytest.raises(ValueError, match="beta"):
        CombinedDemand(np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.0)
    with pytest.raises(ValueError, match="infeasible"):
        CombinedDemand(np.array([2.0, 0.0]), np.array([0.0, 3.0]), 0.5)  # 2 != 3
    with pytest.raises(ValueError, match="nonnegative"):
        CombinedDemand(np.array([-1.0, 2.0]), np.array([0.0, 1.0]), 0.5)


def test_model_requires_combined_scenario():
    with pytest.raises(ValueError, match="combined_demand"):
        _solve(braess_scenario(), EvansCombinedModel(), iterations=1)


# -------------------------------------------------------------- integration
def test_registry_and_load_scenario():
    assert "evans" in MODEL_REGISTRY
    sc = load_scenario("evans")
    assert sc.combined_demand is not None
    assert sc.network.n_zones == 4


def test_run_experiment_end_to_end(evans):
    """The harness auto-detects the combined task from the scenario field and
    certifies the emitted flows — no runner changes needed."""
    result = run_experiment(
        evans, [EvansCombinedModel()], Budget(iterations=100, target_relative_gap=1e-11)
    )
    rows = [r for r in result.rows if r["model"] == "evans"]
    final = rows[-1]
    assert final["feasible"] == 1.0
    assert abs(final["relative_gap"]) < 1e-8
    # The reference equilibrium flows are hit (rmse ~ 0).
    assert float(final["flow_rmse_vs_reference"]) < 1e-6


def test_bookkeeping_and_budget(evans):
    trace = _solve(evans, iterations=5)
    # One shortest-path sweep at init + one per recorded iteration.
    assert trace.final.coords.sp_calls == trace.final.coords.iterations + 1
    assert trace.final.coords.iterations <= 5
    v = trace.final.link_flows
    assert np.all(v >= 0)
    for key in (
        "relative_gap",
        "route_relative_gap",
        "distribution_gap",
        "realized_demand",
        "beckmann",
    ):
        assert key in trace.final.self_report
