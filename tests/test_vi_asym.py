"""Tests for the asymmetric variational-inequality UE model (vi-asym, adr-011).

Dafermos (1980) / Smith (1979): non-separable link costs ``t(v) = t_BPR(v) + C v``
with ``C`` possibly asymmetric, so there is NO Beckmann potential and the
equilibrium is defined by a variational inequality. The scored quantity is the VI
residual -- the ordinary normalized relative gap evaluated at the asymmetric cost,
which the harness recomputes (a VI gap needs no potential). All anchor numbers are
recomputed in-test as closed forms (house style: no trusted digits).
"""

import numpy as np
import pytest

from tabench import (
    AsymmetricVIModel,
    Budget,
    Demand,
    Evaluator,
    RngBundle,
    Scenario,
    Trace,
    braess_scenario,
    vi_two_route_scenario,
)
from tabench.models.frank_wolfe import BiconjugateFrankWolfeModel

BRAESS_GOLDEN_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"


def _f_a_star(demand: float, c13: float, c31: float) -> float:
    """Closed-form asymmetric-VI split (recomputed, never a trusted digit)."""
    return (1.0 + (1.0 - c13) * demand) / (2.0 - c13 - c31)


def _solve(scenario, model=None, **budget_kwargs):
    trace = Trace()
    (model or AsymmetricVIModel()).solve(
        scenario, Budget(**budget_kwargs), RngBundle(0), trace
    )
    return trace


def test_golden_braess_hash_preserved():
    """The new optional link_interaction field is hashed only when set, so every
    scenario without it -- including the golden Braess anchor -- hashes identically."""
    assert braess_scenario().content_hash() == BRAESS_GOLDEN_HASH


def test_analytic_anchor():
    """On the 2-route asymmetric-interaction anchor the model reaches the
    hand-derived VI equilibrium and the certified VI residual drives to ~0."""
    demand, c13, c31 = 10.0, 0.5, 0.2
    sc = vi_two_route_scenario(demand, c13, c31)
    f_a = _f_a_star(demand, c13, c31)  # 6/1.3 = 4.615384...
    trace = _solve(sc, iterations=300, target_relative_gap=1e-12)
    expected = np.array([f_a, f_a, demand - f_a, demand - f_a])
    np.testing.assert_allclose(trace.final.link_flows, expected, atol=1e-4)
    metrics = Evaluator(sc).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-6


def test_no_interaction_reduces_to_plain_ue():
    """With a zero interaction (C = 0) the diagonalization outer loop is a no-op and
    vi-asym reproduces ordinary Frank-Wolfe UE (f_A = (D+1)/2 = 5.5), matching both
    the shipped solver and the closed form."""
    demand = 10.0
    sc = vi_two_route_scenario(demand, c13=0.0, c31=0.0)  # zero interaction
    trace = _solve(sc, iterations=300, target_relative_gap=1e-12)
    assert trace.final.link_flows[0] == pytest.approx(0.5 * (demand + 1.0), abs=1e-4)
    # Same network, no interaction field at all -> plain Beckmann UE.
    plain = Scenario(name="plain", network=sc.network, demand=sc.demand)
    fw = _solve(plain, BiconjugateFrankWolfeModel(), iterations=3000, target_relative_gap=1e-12)
    np.testing.assert_allclose(trace.final.link_flows, fw.final.link_flows, atol=1e-3)


def test_asymmetry_matters_flow_differs_from_potential_solvers():
    """The genuinely asymmetric VI equilibrium is DISTINCT from what any potential
    (Beckmann/FW) solver reaches: it differs both from the plain-UE flow (C ignored)
    AND from the symmetrized-interaction Beckmann flow (C -> (C+C^T)/2). This is the
    'VI is strictly more general than an optimization' content no shipped solver has."""
    demand, c13, c31 = 10.0, 0.5, 0.2
    sc = vi_two_route_scenario(demand, c13, c31)
    trace = _solve(sc, iterations=300, target_relative_gap=1e-12)
    f_a = trace.final.link_flows[0]
    plain_ue = 0.5 * (demand + 1.0)  # 5.5, C ignored
    c_bar = 0.5 * (c13 + c31)
    symmetrized = _f_a_star(demand, c_bar, c_bar)  # (1+(1-0.35)*10)/1.3 = 5.769...
    assert abs(f_a - plain_ue) > 0.5  # 4.615 vs 5.5
    assert abs(f_a - symmetrized) > 0.5  # 4.615 vs 5.769 -> asymmetry is load-bearing
    assert f_a == pytest.approx(_f_a_star(demand, c13, c31), abs=1e-4)


def test_vi_residual_is_harness_recomputed_and_self_report_matches():
    """P1: the certified VI residual is recomputed by the harness at the asymmetric
    cost t(v)+Cv and equals the model self-report to float precision (the model
    computes the same gap the same way); beckmann_objective is NaN (no potential)."""
    sc = vi_two_route_scenario()
    trace = _solve(sc, iterations=300, target_relative_gap=1e-12)
    metrics = Evaluator(sc).evaluate(trace.final.link_flows)
    assert metrics["relative_gap"] == pytest.approx(
        trace.final.self_report["relative_gap"], abs=1e-12
    )
    assert np.isnan(metrics["beckmann_objective"])  # no Beckmann potential exists


def test_certificate_censors_negative_augmented_cost():
    """A black box may emit any flow; if the interaction drives an augmented cost
    non-positive the certificate censors it (feasible=0, NaN gap) instead of
    crashing the shortest-path engine."""
    # Strong asymmetric coupling with a large NEGATIVE off-diagonal: at a lopsided
    # flow the augmented cost on the coupled link goes non-positive.
    demand = 10.0
    sc = vi_two_route_scenario(demand, c13=0.5, c31=0.2)
    net = sc.network
    c = np.zeros((net.n_links, net.n_links))
    c[1, 3] = -5.0  # link 1 cost = v_1 - 5 v_3 -> negative for small v_1, large v_3
    bad = Scenario(name="bad-vi", network=net, demand=sc.demand, link_interaction=c)
    # A feasible-by-conservation flow that makes t_1 = v_1 - 5 v_3 negative.
    flow = np.array([1.0, 1.0, 9.0, 9.0])  # t_1 = 1 - 5*9 = -44 < 0
    metrics = Evaluator(bad).evaluate(flow)
    assert metrics["feasible"] == 0.0
    assert np.isnan(metrics["relative_gap"])


def test_requires_link_interaction():
    """vi-asym is the non-separable-cost model; it refuses a plain scenario."""
    plain = Scenario(
        name="plain", network=vi_two_route_scenario().network, demand=Demand(np.zeros((2, 2)))
    )
    with pytest.raises(ValueError, match="link_interaction"):
        AsymmetricVIModel().solve(plain, Budget(iterations=5), RngBundle(0), Trace())


def test_link_interaction_mutually_exclusive_and_shape_checked():
    """The scenario field is validated: mutually exclusive with the other optional
    fields, and shape (n_links, n_links)."""
    net = vi_two_route_scenario().network
    good = np.zeros((net.n_links, net.n_links))
    with pytest.raises(ValueError, match="mutually exclusive"):
        Scenario(
            name="x", network=net, demand=Demand(np.zeros((2, 2))),
            link_interaction=good, sue_theta=0.5,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        Scenario(
            name="x", network=net, demand=Demand(np.zeros((2, 2))),
            link_interaction=good, side_capacities=np.ones(net.n_links),
        )
    with pytest.raises(ValueError, match="shape"):
        Scenario(
            name="x", network=net, demand=Demand(np.zeros((2, 2))),
            link_interaction=np.zeros((3, 3)),
        )


def test_link_interaction_is_hashed():
    """Adversarial-review Major: two instances differing ONLY in the interaction
    operator C are different benchmark instances and must NOT collide in
    content_hash (deleting the conditional hash block would silently corrupt the
    P2 content-addressed identity and the fairness gate). Same C -> same hash."""
    net = vi_two_route_scenario().network
    dem = Demand(np.zeros((2, 2)))
    c1 = np.zeros((net.n_links, net.n_links))
    c1[1, 3], c1[3, 1] = 0.5, 0.2
    c2 = np.zeros((net.n_links, net.n_links))
    c2[1, 3], c2[3, 1] = 0.7, 0.2
    h1 = Scenario(name="a", network=net, demand=dem, link_interaction=c1).content_hash()
    h2 = Scenario(name="a", network=net, demand=dem, link_interaction=c2).content_hash()
    h1b = Scenario(name="a", network=net, demand=dem, link_interaction=c1.copy()).content_hash()
    assert h1 != h2  # distinct interactions -> distinct instances
    assert h1 == h1b  # same interaction -> same hash (deterministic)


def test_competitive_interaction_is_censored_not_falsely_accepted():
    """Adversarial-review Major (honest convergence scoping): on a strictly-monotone
    but COMPETITIVE/skew interaction (negative off-diagonal multiplying the
    route-concentrated free-flow start) the diagonalization cannot proceed (an
    augmented cost goes non-positive) and emits a flow the certificate CENSORS
    (feasible=0, NaN residual) -- never a false accept. This pins the corrected
    claim that strict monotonicity guarantees the VI *solution*, not diagonalization
    *convergence*."""
    net = vi_two_route_scenario().network
    c = np.zeros((net.n_links, net.n_links))
    c[1, 3] = 0.3
    c[3, 1] = -0.5  # negative: augmented cost on the coupled leg goes < 0 at the start
    sc = Scenario(name="competitive", network=net, demand=vi_two_route_scenario().demand,
                  link_interaction=c)
    trace = _solve(sc, iterations=300, target_relative_gap=1e-12)
    metrics = Evaluator(sc).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 0.0
    assert np.isnan(metrics["relative_gap"])  # honestly censored, not certified


def test_relaxation_damping_still_converges():
    """The outer-relaxation (damping) path is exercised: with relaxation=0.5 the
    diagonalization still reaches the analytic anchor equilibrium."""
    sc = vi_two_route_scenario(10.0, 0.5, 0.2)
    f_a = _f_a_star(10.0, 0.5, 0.2)
    trace = _solve(sc, AsymmetricVIModel(relaxation=0.5), iterations=400, target_relative_gap=1e-12)
    assert trace.final.link_flows[0] == pytest.approx(f_a, abs=1e-3)
    assert Evaluator(sc).evaluate(trace.final.link_flows)["relative_gap"] < 1e-6


def test_beckmann_nan_on_infeasible_vi_flow():
    """beckmann_objective is NaN for a VI task on EVERY path (no potential exists),
    including a flow that fails the feasibility audit -- not just the feasible path."""
    sc = vi_two_route_scenario()
    infeasible = np.array([5.0, 5.0, 3.0, 3.0])  # route A carries 5, route B 3: not conserving
    metrics = Evaluator(sc).evaluate(infeasible)
    assert metrics["feasible"] == 0.0
    assert np.isnan(metrics["beckmann_objective"])


def test_monotone_sweep_matches_closed_form():
    """Raising the (asymmetric) coupling c13 shifts f_A* per the closed form; the
    solver tracks it across a sweep (a family of hand-checkable oracles)."""
    demand, c31 = 10.0, 0.2
    prev = None
    for c13 in (0.1, 0.3, 0.5, 0.7):
        sc = vi_two_route_scenario(demand, c13, c31)
        trace = _solve(sc, iterations=300, target_relative_gap=1e-12)
        f_a = trace.final.link_flows[0]
        assert f_a == pytest.approx(_f_a_star(demand, c13, c31), abs=1e-3)
        if prev is not None:
            assert f_a < prev  # larger c13 -> more spillover onto A -> less A flow
        prev = f_a


def test_budget_target_relative_gap_stops_early():
    """T1 (on): the caller-facing ``Budget.target_relative_gap`` is now a real stop
    channel (parity with the sibling solvers via ``budget.target_met``). A LOOSE
    caller target (1e-2, far looser than the 1e-10 target_gap factor) stops the
    outer loop at strictly fewer sweeps than an otherwise-identical run that leaves
    the caller target unset, and the last self-reported VI gap satisfies it."""
    sc = vi_two_route_scenario(10.0, 0.5, 0.2)
    default = _solve(sc, iterations=300)  # no caller target -> factor (1e-10) gates
    loose = _solve(sc, iterations=300, target_relative_gap=1e-2)
    assert len(loose) < len(default)  # measured 4 vs 22 sweeps
    assert loose.final.self_report["relative_gap"] <= 1e-2


def test_budget_target_none_is_byte_identical_off_noop():
    """T1 (off): with a caller target of None -- or any value at least as tight as
    the model's own 1e-10 target_gap factor -- the new budget.target_met channel is
    inert (the factor fires first), so the run is byte-identical to the prior
    factor-only behavior. The existing pinned anchors (test_analytic_anchor,
    test_no_interaction_reduces_to_plain_ue) run with target_relative_gap=1e-12 and
    still pass, which is the off-pin; here we add an explicit trace-length + flow
    equality between a None-target run and a tight (1e-12) run."""
    sc = vi_two_route_scenario(10.0, 0.5, 0.2)
    none = _solve(sc, iterations=300)
    tight = _solve(sc, iterations=300, target_relative_gap=1e-12)
    assert len(none) == len(tight)
    np.testing.assert_array_equal(none.final.link_flows, tight.final.link_flows)


def test_line_search_xtol_default_is_no_op():
    """T2 (off): line_search_xtol defaults to 1e-13 -- the value the brentq inner
    line search was previously hardcoded to -- so a default run is byte-identical to
    an explicit xtol=1e-13 run. The existing pinned anchors (which never set the
    factor) are the off-pin for the emitted flows."""
    assert AsymmetricVIModel().factor_values["line_search_xtol"] == 1e-13
    sc = vi_two_route_scenario(10.0, 0.5, 0.2)
    default = _solve(sc, iterations=300, target_relative_gap=1e-12)
    explicit = _solve(
        sc, AsymmetricVIModel(line_search_xtol=1e-13), iterations=300, target_relative_gap=1e-12
    )
    np.testing.assert_array_equal(default.final.link_flows, explicit.final.link_flows)


def test_line_search_xtol_loose_changes_flows():
    """T2 (on): line_search_xtol threads into the brentq inner line search. On the
    asymmetric-VI anchor a loose xtol=1e-3 yields link_flows NOT byte-identical to
    the default (tight 1e-13) run, while still certifying a feasible, finite VI
    residual (measured max|diff| ~1.2e-3). This KILLS the "factor declared but not
    threaded from brentq" mutant: the existing off-pin compares a default run to an
    explicit-same-value run, which stays byte-identical even if the factor is
    un-threaded -- only a loose-value on-test can see the threading."""
    sc = vi_two_route_scenario(10.0, 0.5, 0.2)
    default = _solve(sc, iterations=300, target_relative_gap=1e-12)
    loose = _solve(
        sc, AsymmetricVIModel(line_search_xtol=1e-3), iterations=300, target_relative_gap=1e-12
    )
    assert not np.array_equal(default.final.link_flows, loose.final.link_flows)
    # Sanity: the loose run still certifies a feasible, finite VI-residual equilibrium.
    met = Evaluator(sc).evaluate(loose.final.link_flows)
    assert met["feasible"] == 1.0
    assert np.isfinite(met["relative_gap"])


def test_registry_and_paradigm():
    from tabench.models import MODEL_REGISTRY

    assert "vi-asym" in MODEL_REGISTRY
    caps = MODEL_REGISTRY["vi-asym"]().capabilities
    assert caps.paradigm == "static_ue_vi"
    assert caps.provides_gap is True
