"""Tests for Li, Wang & Nie's (2024) cumulative-logit day-to-day dynamics (dtd-cumlog).

The first boundedly-rational logit day-to-day model whose limit is EXACT
deterministic Wardrop UE at a finite exploitation parameter ``r``. Travelers
carry a per-OD route-VALUATION vector ``s`` over column-generated working sets,
choose by the logit map ``p = softmax(-r s)``, and ACCUMULATE experienced route
costs ``s <- s + eta_t c(p)`` (Eq. 6) -- the one-line change from the classical
successive-average scheme ``s <- (1-eta_t) s + eta_t c(p)`` (Eq. 4), whose limit
is the logit SUE. It is validated as a UE model (converges to the analytic
Braess/two-route UE and toward the Sioux Falls Beckmann optimum) PLUS its
distinctive theory:

* THE ACCUMULATION-VS-AVERAGING DISTINCTNESS GATE (Remark 3): on ONE two-route
  instance, with identical machinery and the same ``r``, the ``accumulate=True``
  update drives the UE relative gap to ~0 (f_A -> 2.5) while ``accumulate=False``
  rests at the analytic binary-logit SUE (f_A -> 2.3739 at r=1, gap > 0.01) --
  the paper's central contrast as an executable fact. This two-route gate is the
  UNIQUE accumulate-vs-average discriminator here: on Braess the route costs ~92
  saturate the r=1 logit so its SUE and UE coincide to < 1e-6, and the Braess
  convergence test alone would not kill an averaging mutant;
* r-INDEPENDENCE of the harmonic schedule (Theorem 1(i)): eta_t = 1/(t+1)
  converges for ANY r (r in {1, 10, 40} all reach the Braess UE);
* the constant-schedule STEP-SCALE HEURISTIC: Theorem 1(ii)'s eta0 < 1/(2 r L)
  is SUFFICIENT (its L is the demand-scaled route-cost Lipschitz constant); the
  reported ``eta_heuristic_scale`` 1/(2 r max_a t'_a) is a flow-INDEPENDENT house
  reference, NOT that bound in EITHER direction -- pinned by two executed
  counterexamples (a constant step above it that converges, one below it that
  diverges), a genuinely too-large step's divergence preserved (not damped),
  mirroring dtd-horowitz's instability test;
* s0-INDEPENDENCE / global stability (Sec. 6.4): different finite s0 reach the
  SAME unique WE link flow (Braess, tight) through DIFFERENT route strategies
  (Sioux Falls entropy spread -- this row's extension of the paper's 3N4L Fig. 10
  entropy histogram to Sioux Falls, whose Sec. 6.4 half reports Fig. 11 used-route
  counts);
* the VALUATION-DIVERGENCE SIGNATURE (Sec. 6.4 Fig. 11 route shedding): used-route
  valuations stabilize to finite differences (resolving Harsanyi) while
  dropped-route valuations diverge -- provenance columns, never scored.
"""

import math

import numpy as np
import pytest
from conftest import load_or_skip
from scipy.optimize import brentq

from tabench import (
    Budget,
    CumLogDTDModel,
    Demand,
    Evaluator,
    Network,
    RngBundle,
    Scenario,
    Trace,
    braess_scenario,
    elastic_two_route_scenario,
    two_route_scenario,
)
from tabench.data.builtin import br_two_route_scenario, evans_symmetric_scenario

# Golden content hash of the Braess scenario, unchanged: this model adds no
# scenario field, so every existing content hash must stay byte-identical.
BRAESS_GOLDEN_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"
BRAESS_UE_FLOWS = np.array([4.0, 2.0, 2.0, 2.0, 4.0])
SIOUXFALLS_TNTP_OBJECTIVE = 42.31335287107440
SIOUXFALLS_UNIT_FACTOR = 1e5


def _fixed_point_route_a(r: float, demand: float = 4.0) -> float:
    """Root of the binary-logit fixed point ``f_A = D / (1 + exp(r (c_A - c_B)))``
    on the two-route anchor -- the SUE split the AVERAGING variant rests at (its
    dispersion is exactly the exploitation parameter r). Recomputed via brentq,
    never a trusted digit."""

    def residual(f_a: float) -> float:
        c_a = 2.0 + f_a
        c_b = 1.5 + 2.0 * (demand - f_a)
        return f_a - demand / (1.0 + math.exp(r * (c_a - c_b)))

    return brentq(residual, 0.0, demand, xtol=1e-12)


def _three_parallel_scenario() -> Scenario:
    """The paper's three-parallel-link illustrative example (Sec. 4.3) as a
    node-split, parallel-link-free instance (TABenchmark forbids parallel links).

    Three disjoint routes O(1) -> D(2) via distinct middle nodes 3, 4, 5. The
    (flow-independent, ~0) first legs carry an identical tiny cost, so the WE
    split is set entirely by the second-leg costs ``u_1 = x``, ``u_2 = x + 1``,
    ``u_3 = x + 2.25`` (paper's u_1/u_2/u_3). With demand 3 the unique WE puts
    ``x = (2, 1, 0)``: routes 1 and 2 tie at cost 2 and route 3 (cost 2.25) is
    unused -- exactly the paper's ``p* = (2/3, 1/3, 0)``. Route 3 is never the
    shortest path, so column generation never adds it, and the model correctly
    excludes it (its would-be valuation is the +inf the theory prescribes)."""
    eps = 1e-6
    init = np.array([1, 1, 1, 3, 4, 5], dtype=np.int64)
    term = np.array([3, 4, 5, 2, 2, 2], dtype=np.int64)
    # First legs 1->3/1->4/1->5: constant 1e-4 (identical, so they cancel in the
    # WE split). Second legs via power=1 BPR: u = intercept + slope * x.
    fft = np.array([1e-4, 1e-4, 1e-4, eps, 1.0, 2.25])
    b = np.array([0.0, 0.0, 0.0, 1.0 / eps, 1.0, 1.0 / 2.25])
    net = Network(
        name="three-parallel", n_nodes=5, n_zones=2, first_thru_node=3,
        init_node=init, term_node=term, capacity=np.ones(6), length=np.zeros(6),
        free_flow_time=fft, b=b, power=np.ones(6), toll=np.zeros(6),
        link_type=np.ones(6, dtype=np.int64),
    )
    od = np.zeros((2, 2))
    od[0, 1] = 3.0
    return Scenario("three-parallel", net, Demand(od))


@pytest.fixture(scope="module")
def braess():
    return braess_scenario()


@pytest.fixture(scope="module")
def siouxfalls():
    return load_or_skip("siouxfalls")


def _solve(scenario, model=None, seed=0, **budget_kwargs):
    trace = Trace()
    (model or CumLogDTDModel()).solve(
        scenario, Budget(**budget_kwargs), RngBundle(seed), trace
    )
    return trace


# ------------------------------------------------------------- convergence
def test_converges_to_braess_ue(braess):
    """The cumulative-logit dynamics settle on the exact Wardrop UE of the
    Braess network -- the emitted softmax load reaches (4, 2, 2, 2, 4) and the
    certified relative gap drives to the target (route flows are unique here, so
    the strategy is the unique 1/3-1/3-1/3 split)."""
    trace = _solve(braess, iterations=5000, target_relative_gap=1e-7)
    metrics = Evaluator(braess).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-6
    np.testing.assert_allclose(trace.final.link_flows, BRAESS_UE_FLOWS, atol=1e-3)


def test_converges_to_two_route_ue():
    """On the two-route anchor (as a deterministic UE task, sue_theta=None) the
    softmax load reaches the hand-checkable Wardrop UE -- 2 + f_A = 1.5 + 2(4 -
    f_A) => f_A = 2.5, common route cost 4.5 -- with the certified gap driven to
    the target. The choice is a finite-r logit, yet the LIMIT is exact UE (not
    the SUE a finite-r logit would give under averaging)."""
    ue = two_route_scenario(sue_theta=None)
    trace = _solve(ue, CumLogDTDModel(r=10.0), iterations=5000, target_relative_gap=1e-8)
    np.testing.assert_allclose(
        trace.final.link_flows, np.array([2.5, 2.5, 1.5, 1.5]), atol=1e-3
    )
    metrics = Evaluator(ue).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-6


def test_three_parallel_link_example():
    """The paper's Sec. 4.3 three-parallel-link example: with u = x / x+1 / x+2.25
    and demand 3, the unique WE is x = (2, 1, 0) (routes 1, 2 tie, route 3 unused).
    This is the first dtd row whose primary supplies reproducible open numerics."""
    sc = _three_parallel_scenario()
    trace = _solve(sc, CumLogDTDModel(r=10.0), iterations=6000, target_relative_gap=1e-8)
    metrics = Evaluator(sc).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-6
    # Link order [1->3, 1->4, 1->5, 3->2, 4->2, 5->2]: routes carry (2, 1, 0).
    np.testing.assert_allclose(
        trace.final.link_flows, np.array([2.0, 1.0, 0.0, 2.0, 1.0, 0.0]), atol=1e-3
    )


def test_scales_to_siouxfalls(siouxfalls):
    """On a real network the certified gap keeps shrinking and the Beckmann
    objective approaches the published optimum. CumLog uses no higher-order
    information (only route costs), so it converges slowly -- this demonstrates
    scaling, not a tight terminal gap. The constant eta = 1 step follows the
    paper's constant-step experiments (Sec. 4.3 illustration / Sec. 6.1 condition
    (ii) sweep); we use r = 0.25, a smaller exploitation parameter that
    de-saturates faster under column generation than the paper's r = 2.5 Sioux
    Falls runs (Sec. 6.4)."""
    trace = _solve(
        siouxfalls,
        CumLogDTDModel(r=0.25, eta_schedule="constant", eta0=1.0),
        iterations=800, target_relative_gap=1e-4,
    )
    gaps = [s.self_report["relative_gap"] for s in trace]
    metrics = Evaluator(siouxfalls).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert gaps[-1] < gaps[0]  # steadily converging toward UE
    obj = metrics["beckmann_objective"] / SIOUXFALLS_UNIT_FACTOR
    assert obj == pytest.approx(SIOUXFALLS_TNTP_OBJECTIVE, rel=5e-2)


# ------------------------------------- the accumulation-vs-averaging gate
def test_accumulation_reaches_ue_averaging_reaches_sue():
    """THE HEADLINE DISTINCTNESS GATE (Remark 3), executable on ONE instance with
    identical machinery and the same r = 1: the ACCUMULATION rule (Eq. 6,
    accumulate=True) drives the UE relative gap to ~0 (f_A -> 2.5, exact Wardrop
    UE), while the AVERAGING rule (Eq. 4, accumulate=False) rests at the analytic
    binary-logit SUE (f_A -> 2.3739 at r = 1, recomputed via brentq) whose UE gap
    stays strictly positive. The ONLY difference is the one-line update rule; the
    categorically different limit (WE vs SUE) is the paper's central claim."""
    anchor = two_route_scenario(sue_theta=None)  # UE relative-gap certificate

    accumulate = _solve(
        anchor, CumLogDTDModel(r=1.0, accumulate=True),
        iterations=5000, target_relative_gap=1e-7,
    )
    m_acc = Evaluator(anchor).evaluate(accumulate.final.link_flows)
    assert m_acc["relative_gap"] < 1e-6  # exact deterministic Wardrop UE
    assert accumulate.final.link_flows[0] == pytest.approx(2.5, abs=1e-3)

    averaging = _solve(anchor, CumLogDTDModel(r=1.0, accumulate=False), iterations=4000)
    m_avg = Evaluator(anchor).evaluate(averaging.final.link_flows)
    # The averaging variant rests at the logit SUE (dispersion r = 1), NOT UE:
    # its link flows match the analytic binary-logit split and its UE gap is > 0.
    f_a_sue = _fixed_point_route_a(r=1.0)
    # The averaging run lands ~1e-14 from the brentq SUE root, so a tight tolerance
    # gate-backs the "matched to six digits" claim (and would catch an averaging
    # mutant that reached a nearby-but-wrong rest point).
    assert averaging.final.link_flows[0] == pytest.approx(f_a_sue, abs=5e-7)
    assert m_avg["relative_gap"] > 0.01
    # ... and the two limits are decisively different route splits.
    assert abs(accumulate.final.link_flows[0] - averaging.final.link_flows[0]) > 0.1


# ------------------------------------------------- Theorem 1 conditions
def test_harmonic_converges_for_any_r(braess):
    """Theorem 1(i): under the harmonic schedule eta_t = 1/(t+1), CumLog converges
    to WE for ANY exploitation parameter r (paper Sec. 6.1). r in {1, 10, 40} all
    reach the Braess UE -- the robustness the successive-average model lacks (its
    WE coupling is a knife-edge, Sec. 6.2)."""
    for r in (1.0, 10.0, 40.0):
        trace = _solve(braess, CumLogDTDModel(r=r), iterations=6000, target_relative_gap=1e-7)
        metrics = Evaluator(braess).evaluate(trace.final.link_flows)
        assert metrics["feasible"] == 1.0, r
        assert metrics["relative_gap"] < 1e-5, r


def test_constant_schedule_converges_and_can_diverge(braess):
    """The constant schedule genuinely converges (small step) or diverges (large
    step): on demand-6 Braess, eta0 = 0.02 drives the certified gap to machine
    precision while eta0 = 0.3 never settles -- the divergence is preserved, never
    damped (cf. dtd-horowitz)."""
    below = _solve(
        braess, CumLogDTDModel(r=1.0, eta_schedule="constant", eta0=0.02),
        iterations=2000,
    )
    assert Evaluator(braess).evaluate(below.final.link_flows)["relative_gap"] < 1e-6

    above = _solve(
        braess, CumLogDTDModel(r=1.0, eta_schedule="constant", eta0=0.3),
        iterations=2000,
    )
    gaps = [s.self_report["relative_gap"] for s in above]
    assert gaps[-1] > 0.01  # never settles
    assert max(gaps[-50:]) > 0.01


def test_eta_heuristic_scale_is_not_a_bound(braess):
    """HONESTY PIN: the reported eta_heuristic_scale = 1/(2 r max_a t'_a) is the
    house step-scale reference, NOT the Theorem 1(ii) bound (whose L is the
    demand-scaled route-cost Lipschitz constant). Because Braess costs are linear
    the heuristic is FLOW-INDEPENDENT (0.05 at r=1 on both demand levels), so it is
    a bound in NEITHER direction: (i) on demand-6 a constant step ABOVE it
    (eta0=0.08) still converges to machine precision; (ii) on demand-60 a constant
    step BELOW it (eta0=0.02) diverges -- true stability is flow-dependent."""
    braess60 = braess_scenario(demand=60.0)

    # (i) above the heuristic on demand-6 -> still converges to machine precision.
    above_heur = _solve(
        braess, CumLogDTDModel(r=1.0, eta_schedule="constant", eta0=0.08),
        iterations=4000,
    )
    assert above_heur.final.self_report["eta_heuristic_scale"] == pytest.approx(0.05, abs=1e-9)
    assert above_heur.final.self_report["relative_gap"] < 1e-9  # measured ~6e-16

    # (ii) below the heuristic on demand-60 -> diverges (flow-dependent stability).
    below_heur = _solve(
        braess60, CumLogDTDModel(r=1.0, eta_schedule="constant", eta0=0.02),
        iterations=4000,
    )
    assert below_heur.final.self_report["eta_heuristic_scale"] == pytest.approx(0.05, abs=1e-9)
    gaps = [s.self_report["relative_gap"] for s in below_heur]
    assert min(gaps[-50:]) > 0.1  # measured tail ~0.93; the heuristic mis-predicts


# -------------------------------------------------------- s0-independence
def test_s0_independence_braess_link_flows(braess):
    """Global stability (Sec. 6.4): CumLog reaches the SAME unique WE link flow
    from different finite s0. Two runs with different random initial valuations
    (init_valuation_scale > 0, distinct seeds) both converge to the exact Braess
    UE link flow -- the certificate is insensitive to the initial valuations."""
    a = _solve(
        braess, CumLogDTDModel(r=1.0, init_valuation_scale=3.0), seed=1,
        iterations=3000, target_relative_gap=1e-8,
    )
    b = _solve(
        braess, CumLogDTDModel(r=1.0, init_valuation_scale=3.0), seed=2,
        iterations=3000, target_relative_gap=1e-8,
    )
    np.testing.assert_allclose(a.final.link_flows, BRAESS_UE_FLOWS, atol=1e-3)
    np.testing.assert_allclose(a.final.link_flows, b.final.link_flows, atol=1e-3)


def test_s0_independence_strategy_diversity(siouxfalls):
    """Different finite s0 reach the WE through DIFFERENT route strategies (Sec.
    6.4): on Sioux Falls two runs from different random initial valuations produce
    materially different strategy entropies while both stay demand-feasible -- the
    strategy polyhedron the unique WE link flow admits. The paper's entropy
    histogram (Fig. 10, spread 10.77-12.84) is the 3N4L experiment; the paper's
    Sioux Falls half of Sec. 6.4 reports used-route COUNTS (Fig. 11), so applying
    the entropy diversity to Sioux Falls is this row's own extension."""
    a = _solve(
        siouxfalls,
        CumLogDTDModel(r=0.25, eta_schedule="constant", eta0=1.0, init_valuation_scale=5.0),
        seed=1, iterations=400,
    )
    b = _solve(
        siouxfalls,
        CumLogDTDModel(r=0.25, eta_schedule="constant", eta0=1.0, init_valuation_scale=5.0),
        seed=2, iterations=400,
    )
    ent_a = a.final.self_report["strategy_entropy"]
    ent_b = b.final.self_report["strategy_entropy"]
    assert abs(ent_a - ent_b) > 0.5  # genuinely different route strategies
    assert Evaluator(siouxfalls).evaluate(a.final.link_flows)["feasible"] == 1.0
    assert Evaluator(siouxfalls).evaluate(b.final.link_flows)["feasible"] == 1.0


# ------------------------------------------- valuation-divergence signature
def test_valuation_divergence_signature(siouxfalls):
    """The distinctive state signature (Figs. 2, 11): used-route valuations
    stabilize to a finite spread (resolving Harsanyi -- equal-cost routes carry
    unequal, finitely-differing valuations), while dropped-route valuations
    diverge (their max grows far beyond the used spread). Both are provenance,
    never scored."""
    trace = _solve(
        siouxfalls,
        CumLogDTDModel(r=0.25, eta_schedule="constant", eta0=1.0),
        iterations=400,
    )
    used = [s.self_report["used_valuation_spread"] for s in trace]
    vmax = [s.self_report["valuation_max"] for s in trace]
    # Used-route spread stabilizes to a finite constant.
    mid = len(used) // 2
    assert used[-1] == pytest.approx(used[mid], rel=0.2)
    assert used[-1] < 1e3
    # Dropped-route valuations diverge far beyond the used spread and keep growing.
    assert vmax[-1] > 100.0 * used[-1]
    assert vmax[-1] > vmax[mid]


# ------------------------------------------------------------- honesty (P1)
def test_self_report_matches_harness_certificate(braess):
    """P1 honesty: the model's self-reported relative gap equals the one the
    harness recomputes from the emitted link flows -- both are
    (TSTT - SPTT)/TSTT with SPTT from the same all-or-nothing map -- so they
    agree to float precision at every checkpoint."""
    trace = _solve(braess, iterations=50)
    evaluator = Evaluator(braess)
    for state in list(trace)[::10]:
        certified = evaluator.evaluate(state.link_flows)["relative_gap"]
        assert certified == pytest.approx(
            state.self_report["relative_gap"], rel=1e-9, abs=1e-15
        )


# --------------------------------------------------------------------- guards
def test_refuses_sue_scenario():
    """The limit is Wardrop UE regardless of r, so an SUE task is refused -- and
    critically, scenario.sue_theta must NEVER be mapped to the model factor r."""
    with pytest.raises(ValueError, match="SUE|sue_theta|theta"):
        CumLogDTDModel().solve(
            two_route_scenario(sue_theta=0.5), Budget(iterations=5), RngBundle(0), Trace()
        )


def test_refuses_elastic_combined_br_scenarios():
    """dtd-cumlog is a fixed-demand point-set UE model: it refuses elastic,
    combined, and BR-UE tasks (its bounded rationality is process-level, not the
    concept-level indifference band of br-ue / adr-008)."""
    with pytest.raises(ValueError, match="elastic"):
        CumLogDTDModel().solve(
            elastic_two_route_scenario(), Budget(iterations=5), RngBundle(0), Trace()
        )
    with pytest.raises(ValueError, match="combined"):
        CumLogDTDModel().solve(
            evans_symmetric_scenario(), Budget(iterations=5), RngBundle(0), Trace()
        )
    with pytest.raises(ValueError, match="BR-UE|br-ue|band"):
        CumLogDTDModel().solve(
            br_two_route_scenario(), Budget(iterations=5), RngBundle(0), Trace()
        )


# ------------------------------------------------------------------- mechanics
def test_paradigm_and_registry():
    from tabench.models import MODEL_REGISTRY

    assert "dtd-cumlog" in MODEL_REGISTRY
    assert MODEL_REGISTRY["dtd-cumlog"]().capabilities.paradigm == "day_to_day"


def test_bookkeeping_and_conservation(braess):
    trace = _solve(braess, iterations=10)
    assert len(trace) == 10
    # One Dijkstra at init + one per recorded day (supplies both the new column
    # and SPTT); the softmax load and cumulative update cost no shortest paths.
    assert trace.final.coords.sp_calls == 10 + 1
    v = trace.final.link_flows
    assert np.all(v >= 0)
    metrics = Evaluator(braess).evaluate(v)
    # The softmax load routes the full OD demand every day, so link flows balance
    # to the float-noise floor at every checkpoint.
    assert metrics["feasible"] == 1.0
    assert metrics["node_balance_residual"] <= 1e-6 * braess.demand.total
    for key in (
        "relative_gap", "beckmann", "used_valuation_spread", "valuation_max",
        "active_routes", "strategy_entropy", "eta_heuristic_scale",
    ):
        assert key in trace.final.self_report


def test_braess_content_hash_preserved():
    """This model adds no scenario field: the golden Braess content hash must be
    byte-identical."""
    assert braess_scenario().content_hash() == BRAESS_GOLDEN_HASH


# ----------------------------------------------------------- _ACTIVE_TOL (B4)
def test_active_tol_excludes_shed_routes(siouxfalls):
    """B4: the _ACTIVE_TOL branch (a working-set route counts as active iff its logit
    choice probability p >= 1e-6) is load-bearing on the active_routes and
    used_valuation_spread self-reports -- an existing test only checks the KEY exists.
    On congested Sioux Falls routes get shed as flows settle: their valuations diverge
    and their logit shares underflow below the tolerance, so active_routes rises
    (column generation) then FALLS well below its peak (routes go inactive), while
    used_valuation_spread stays BOUNDED because the diverging shed valuations are
    excluded from the used set. MUTANT KILL: replacing `p >= _ACTIVE_TOL` with
    `p >= 0.0` counts every working-set route as active, so active_routes becomes
    monotone non-decreasing (final == peak, no shedding) and used_valuation_spread
    blows up to the diverging valuation_max (measured: real used_spread ~53 vs mutant
    ~8.6e4)."""
    trace = _solve(
        siouxfalls,
        CumLogDTDModel(r=0.25, eta_schedule="constant", eta0=1.0),
        iterations=400,
    )
    active = [s.self_report["active_routes"] for s in trace]
    used_spread = [s.self_report["used_valuation_spread"] for s in trace]
    vmax = [s.self_report["valuation_max"] for s in trace]
    # Routes are genuinely shed: active_routes drops well below its peak. Under the
    # mutant active_routes = working-set size, which only grows (never drops).
    assert max(active) - active[-1] > 20
    # The shed routes' diverging valuations are EXCLUDED from the used spread, which
    # stays bounded even as the dropped-route valuations diverge (mutant: blows up).
    assert max(used_spread) < 1e3
    assert vmax[-1] > 100.0 * used_spread[-1]
