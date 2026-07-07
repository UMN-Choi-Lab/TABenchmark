"""Tests for Cantarella & Cascetta's (1995) unifying day-to-day process (dtd-unifying).

The unifying-theory node of the day-to-day family: one two-equation process --
an exponential cost-learning filter ``p <- (1 - w) p + w t(v)`` plus a choice
update in which a fraction ``alpha_n`` of travelers reconsiders at the forecast
costs ``v <- v + alpha_n (ChoiceLoad(p) - v)`` -- whose choice map is gated per
scenario: all-or-nothing best response on deterministic scenarios (fixed point
= Wardrop UE, annealed ``alpha/n`` step) and the pinned Dial-STOCH logit load
on SUE scenarios (fixed point = logit SUE, constant ``alpha``). It is validated
on BOTH limits of the existing two-route anchor (UE ``f_A = 2.5`` at
``sue_theta=None``; the brentq binary-logit fixed point at the default
``theta=0.5``), by exact-reduction regressions (stochastic ``alpha=1`` IS
``dtd-horowitz``; deterministic ``w=1, alpha=1`` IS ``msa`` -- one-day index
offsets from init bookkeeping), PLUS its distinctive dynamical signature: the
JOINT ``(alpha, w)`` flip-stability boundary ``(2-w)(2-alpha) = alpha w |phi'|``
-- no damping is added, so ``(1, 1)`` settles into a period-2 limit cycle while
EITHER form of inertia (cost memory ``w`` small OR choice inertia ``alpha``
small) restores convergence, C&C's headline result. At ``alpha = 1`` the
boundary reduces to ``dtd-horowitz``'s documented ``w* ~ 0.81`` on the anchor
(bracketed at w = 0.7 vs 0.9), an independent consistency check.
"""

import math

import numpy as np
import pytest
from conftest import load_or_skip
from scipy.optimize import brentq

from tabench import (
    Budget,
    CostSmoothingSUEModel,
    DialSUEModel,
    Evaluator,
    MSAModel,
    RngBundle,
    Trace,
    UnifyingDTDModel,
    braess_scenario,
    two_route_scenario,
)

# Golden content hash of the Braess scenario, unchanged: this model adds no
# scenario field, so every existing content hash must stay byte-identical.
BRAESS_GOLDEN_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"
BRAESS_UE_FLOWS = np.array([4.0, 2.0, 2.0, 2.0, 4.0])
SIOUXFALLS_TNTP_OBJECTIVE = 42.31335287107440
SIOUXFALLS_UNIT_FACTOR = 1e5


def _fixed_point_route_a(theta: float, demand: float = 4.0) -> float:
    """Root of the binary-logit fixed point ``f_A = D / (1 + exp(theta (c_A -
    c_B)))`` on the two-route anchor -- the stochastic branch's rest point
    (``p = t(v)`` => the SUE split). Recomputed here, never a trusted digit."""

    def residual(f_a: float) -> float:
        c_a = 2.0 + f_a
        c_b = 1.5 + 2.0 * (demand - f_a)
        return f_a - demand / (1.0 + math.exp(theta * (c_a - c_b)))

    return brentq(residual, 0.0, demand, xtol=1e-12)


@pytest.fixture(scope="module")
def sue_anchor():
    return two_route_scenario()  # demand 4, theta 0.5, logit -> stochastic branch


@pytest.fixture(scope="module")
def ue_anchor():
    return two_route_scenario(sue_theta=None)  # same network, deterministic branch


@pytest.fixture(scope="module")
def braess():
    return braess_scenario()


@pytest.fixture(scope="module")
def siouxfalls():
    return load_or_skip("siouxfalls")


def _solve(sc, model=None, **budget_kwargs):
    trace = Trace()
    (model or UnifyingDTDModel()).solve(sc, Budget(**budget_kwargs), RngBundle(0), trace)
    return trace


# --------------------------------------------------- the two (S)UE limits
def test_deterministic_limit_recovers_two_route_ue(ue_anchor):
    """MODE GATE, deterministic branch (sue_theta=None): the annealed process
    settles on the hand-checkable Wardrop UE of the anchor -- 2 + f_A = 1.5 +
    2 (4 - f_A) => f_A = 2.5, common route cost 4.5 -- with the harness-certified
    relative gap driven to the target."""
    trace = _solve(ue_anchor, iterations=2000, target_relative_gap=1e-8)
    np.testing.assert_allclose(
        trace.final.link_flows, np.array([2.5, 2.5, 1.5, 1.5]), atol=1e-3
    )
    gaps = [s.self_report["relative_gap"] for s in trace]
    assert gaps[-1] < gaps[0]
    metrics = Evaluator(ue_anchor).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-6


def test_stochastic_limit_recovers_logit_sue(sue_anchor):
    """MODE GATE, stochastic branch (sue_theta=0.5): the constant-alpha process
    settles on the analytic binary-logit SUE (recomputed via brentq) and the
    certified ADR-001 residual drives to ~0. The logit SUE is NOT the
    deterministic UE: the UE gap stays strictly positive (descriptive, like
    sue-msa)."""
    f_a = _fixed_point_route_a(theta=0.5)
    trace = _solve(sue_anchor, iterations=500, target_relative_gap=1e-8)
    expected = np.array([f_a, f_a, 4.0 - f_a, 4.0 - f_a])
    np.testing.assert_allclose(trace.final.link_flows, expected, atol=1e-4)
    metrics = Evaluator(sue_anchor).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["sue_fixed_point_residual"] < 1e-6
    assert metrics["relative_gap"] > 0.01


def test_deterministic_braess_ue_regression(braess):
    """UE regression on the classic Braess network: the deterministic branch
    reaches the exact analytic UE link flows (4, 2, 2, 2, 4) at the MSA rate."""
    trace = _solve(braess, iterations=3000, target_relative_gap=1e-8)
    np.testing.assert_allclose(trace.final.link_flows, BRAESS_UE_FLOWS, atol=1e-3)
    metrics = Evaluator(braess).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-4


def test_cross_solver_agrees_with_sue_msa(sue_anchor):
    """dtd-unifying's stochastic branch and sue-msa reach the SAME certified
    logit-SUE link flows on the anchor -- both fixed points of the same pinned
    Dial-STOCH map."""
    unifying = _solve(sue_anchor, iterations=500, target_relative_gap=1e-9)
    msa = _solve(sue_anchor, DialSUEModel(), iterations=500, target_relative_gap=1e-9)
    np.testing.assert_allclose(
        unifying.final.link_flows, msa.final.link_flows, atol=1e-4
    )


# ------------------------------------------------------- exact reductions
def test_reduces_to_dtd_horowitz_at_alpha_one(sue_anchor):
    """EXACT REDUCTION: with alpha = 1 (everyone reconsiders daily) the
    stochastic branch is dtd-horowitz verbatim at the same w -- the emitted flow
    trajectories agree to float precision, offset by one day (this model's init
    emits its day-0 load only through day 1's update)."""
    unifying = _solve(
        sue_anchor,
        UnifyingDTDModel(memory_weight=0.5, reconsideration_rate=1.0),
        iterations=30,
    )
    horowitz = _solve(sue_anchor, CostSmoothingSUEModel(smoothing_weight=0.5), iterations=31)
    for k in range(30):
        np.testing.assert_allclose(
            unifying.checkpoints[k].link_flows,
            horowitz.checkpoints[k + 1].link_flows,
            rtol=1e-12,
            atol=1e-12,
        )


def test_reduces_to_msa_at_unit_memory_and_alpha(braess):
    """EXACT REDUCTION: with w = 1 and alpha = 1 the deterministic branch's
    annealed step alpha/n IS msa's 1/k iterate sequence exactly, including the
    AON-at-free-flow init (recorded-flow index offset by one)."""
    unifying = _solve(
        braess,
        UnifyingDTDModel(memory_weight=1.0, reconsideration_rate=1.0),
        iterations=40,
    )
    msa = _solve(braess, MSAModel(), iterations=41)
    for k in range(40):
        np.testing.assert_allclose(
            unifying.checkpoints[k].link_flows,
            msa.checkpoints[k + 1].link_flows,
            rtol=1e-12,
            atol=1e-12,
        )


# ------------------------------------------- joint (alpha, w) flip stability
def test_limit_cycles_without_inertia(sue_anchor):
    """The distinctive dynamical property: at (alpha, w) = (1, 1) -- no choice
    inertia, no cost memory -- the flip condition (2-w)(2-alpha) = 1 < alpha w
    |phi'| ~ 1.47 is violated, so the process settles into a period-2 limit
    cycle: the certified residual stays O(1) and the emitted route-A flow
    oscillates over a wide band. No damping is added, so the divergence is
    preserved (as in dtd-horowitz)."""
    trace = _solve(
        sue_anchor,
        UnifyingDTDModel(memory_weight=1.0, reconsideration_rate=1.0),
        iterations=400,
    )
    res = [s.self_report["sue_fixed_point_residual"] for s in trace]
    fa = np.array([s.link_flows[0] for s in trace])
    assert res[-1] > 1.0
    assert max(res[-50:]) > 1.0
    assert fa[-50:].max() - fa[-50:].min() > 1.0


def test_either_form_of_inertia_restores_stability(sue_anchor):
    """C&C's headline, checkable by hand from (2-w)(2-alpha) > alpha w |phi'|
    (|phi'| ~ 1.47 on the anchor): EITHER cost memory OR choice inertia
    stabilizes the same process that limit-cycles at (1, 1) -- the (0.5, 0.5)
    default (margin 2.25 vs 0.37), choice inertia alone (alpha=0.3, w=1: 1.7 vs
    0.44), and cost memory alone (alpha=1, w=0.3: 1.7 vs 0.44)."""
    for alpha, w in ((0.5, 0.5), (0.3, 1.0), (1.0, 0.3)):
        trace = _solve(
            sue_anchor,
            UnifyingDTDModel(memory_weight=w, reconsideration_rate=alpha),
            iterations=500,
        )
        assert trace.final.self_report["sue_fixed_point_residual"] < 1e-8


def test_flip_boundary_brackets_horowitz_threshold(sue_anchor):
    """At alpha = 1 the joint flip boundary reduces to dtd-horowitz's documented
    threshold w* = 2/(1 - phi') ~ 0.81 on the anchor -- an independent
    consistency check between the two shipped models: w = 0.7 converges, w = 0.9
    limit-cycles."""
    below = _solve(
        sue_anchor,
        UnifyingDTDModel(memory_weight=0.7, reconsideration_rate=1.0),
        iterations=500,
    )
    above = _solve(
        sue_anchor,
        UnifyingDTDModel(memory_weight=0.9, reconsideration_rate=1.0),
        iterations=500,
    )
    assert below.final.self_report["sue_fixed_point_residual"] < 1e-8
    assert above.final.self_report["sue_fixed_point_residual"] > 1.0


def test_perceived_cost_gap_vanishes_at_equilibrium(sue_anchor):
    """The perceived-cost gap ||p - t(v)||_1 (forecast minus experience) is a
    provenance measure that is zero iff the learning filter is at rest; it is
    O(1) off equilibrium (day 0 forecasts free-flow costs) and collapses as the
    stable dynamics converge (reported as provenance, never scored)."""
    trace = _solve(sue_anchor, iterations=400)
    gap = [s.self_report["perceived_cost_gap"] for s in trace]
    assert gap[-1] < 1e-6
    assert max(gap) > 1.0


# ------------------------------------------------------------- honesty (P1)
def test_self_report_matches_certificate_in_both_modes(ue_anchor, sue_anchor):
    """P1 honesty in BOTH gated modes: the model self-monitors the SAME quantity
    the harness recomputes with the SAME engines (AON relative gap / pinned
    Dial-STOCH residual), so self-report == certificate to float precision at
    every checkpoint."""
    det = _solve(ue_anchor, iterations=50)
    evaluator = Evaluator(ue_anchor)
    for state in list(det)[::10]:
        certified = evaluator.evaluate(state.link_flows)["relative_gap"]
        assert certified == pytest.approx(
            state.self_report["relative_gap"], rel=1e-9, abs=1e-15
        )
    sue = _solve(sue_anchor, iterations=50)
    evaluator = Evaluator(sue_anchor)
    for state in list(sue)[::10]:
        certified = evaluator.evaluate(state.link_flows)["sue_fixed_point_residual"]
        assert certified == pytest.approx(
            state.self_report["sue_fixed_point_residual"], rel=1e-9, abs=1e-15
        )


# --------------------------------------------------------------- scaling
def test_scales_to_siouxfalls_deterministic(siouxfalls):
    """On a real fixed-demand network the deterministic branch keeps shrinking
    the certified gap and the Beckmann objective approaches the published
    optimum (annealed MSA-rate dynamics converge slowly, so this demonstrates
    scaling, not a tight terminal gap)."""
    trace = _solve(siouxfalls, iterations=300, target_relative_gap=1e-6)
    gaps = [s.self_report["relative_gap"] for s in trace]
    metrics = Evaluator(siouxfalls).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert gaps[-1] < gaps[0]
    assert gaps[-1] < 5e-2
    obj = metrics["beckmann_objective"] / SIOUXFALLS_UNIT_FACTOR
    assert obj == pytest.approx(SIOUXFALLS_TNTP_OBJECTIVE, rel=5e-2)


# --------------------------------------------------------------------- guards
def test_rejects_probit_scenario():
    """The stochastic branch is the logit process: refuse a probit-SUE task and
    point at the probit solver (exactly sue-msa's guard)."""
    probit = two_route_scenario(sue_theta=0.1, sue_family="probit")
    with pytest.raises(ValueError, match="probit"):
        UnifyingDTDModel().solve(probit, Budget(iterations=5), RngBundle(0), Trace())


# ------------------------------------------------------------------- mechanics
def test_paradigm_and_registry():
    from tabench.models import MODEL_REGISTRY

    assert "dtd-unifying" in MODEL_REGISTRY
    assert MODEL_REGISTRY["dtd-unifying"]().capabilities.paradigm == "day_to_day"


def test_bookkeeping_and_conservation_both_modes(ue_anchor, sue_anchor):
    """One init load plus two loads per day (choice + certify) => sp_calls =
    2k + 1 in both modes; the emitted flow is a convex combination of
    full-demand loads, so node balance sits at the float-noise floor at EVERY
    checkpoint; each mode records its certificate plus the provenance gap."""
    for sc, keys in (
        (ue_anchor, ("relative_gap", "tstt", "sptt", "beckmann", "perceived_cost_gap")),
        (sue_anchor, ("sue_fixed_point_residual", "perceived_cost_gap")),
    ):
        trace = _solve(sc, iterations=10)
        assert len(trace) == 10
        assert trace.final.coords.sp_calls == 2 * 10 + 1
        evaluator = Evaluator(sc)
        for state in trace:
            assert np.all(state.link_flows >= 0)
            metrics = evaluator.evaluate(state.link_flows)
            assert metrics["feasible"] == 1.0
            assert metrics["node_balance_residual"] <= 1e-6 * sc.demand.total
            for key in keys:
                assert key in state.self_report


def test_braess_content_hash_preserved():
    """This model adds no scenario field: the golden Braess content hash must be
    byte-identical."""
    assert braess_scenario().content_hash() == BRAESS_GOLDEN_HASH
