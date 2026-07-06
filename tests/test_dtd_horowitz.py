"""Tests for Horowitz's (1984) cost-smoothing day-to-day SUE dynamics (dtd-horowitz).

The perceived-cost-state sibling of ``sue-msa`` and ``dtd-swap-sue``: travelers
carry an exponentially-smoothed **perceived link-cost vector** ``p``, logit-load
at those perceived costs, and update ``p`` toward the experienced costs
``p <- (1 - w) p + w t(v)``. The rest point is the logit stochastic user
equilibrium (the Dial-STOCH fixed point, Fisk/Daganzo-Sheffi), NOT deterministic
Wardrop UE. It is validated as an SUE model -- on the two-route anchor it
converges to the analytic binary-logit split and self-reports the SAME
Dial-STOCH certificate the harness recomputes (P1), matching ``sue-msa`` and
``dtd-swap-sue`` -- PLUS its distinctive dynamical signature: unlike the
always-convergent MSA and the always-stabilized route-swap models, the
constant-weight smoothing map is a nonlinear dynamical system whose SUE fixed
point is a stable attractor ONLY below a task-dependent stability threshold
``w* ~ 0.81`` on the anchor. Below it the certified residual drives to ~0; above
it (up to ``w = 1``, Horowitz's naive current-cost model) the process settles
into a period-2 limit cycle and the residual stays O(1) -- the instability the
model exists to exhibit, so NO damping/backtracking is applied.
"""

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from tabench import (
    Budget,
    CostSmoothingSUEModel,
    DialSUEModel,
    Evaluator,
    RngBundle,
    Scenario,
    Trace,
    braess_scenario,
    two_route_scenario,
)

# Golden content hash of the Braess scenario, unchanged: this model adds no
# scenario field, so every existing content hash must stay byte-identical.
BRAESS_GOLDEN_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"


def _fixed_point_route_a(theta: float, demand: float = 4.0) -> float:
    """Root of the binary-logit fixed point ``f_A = D / (1 + exp(theta (c_A -
    c_B)))`` on the two-route anchor -- the same scalar equation this model's
    perceived-cost rest point reaches (``p = t(v)`` => the SUE split). Recomputed
    here, never a trusted digit."""

    def residual(f_a: float) -> float:
        c_a = 2.0 + f_a
        c_b = 1.5 + 2.0 * (demand - f_a)
        return f_a - demand / (1.0 + math.exp(theta * (c_a - c_b)))

    return brentq(residual, 0.0, demand, xtol=1e-12)


@pytest.fixture(scope="module")
def scenario():
    return two_route_scenario()  # demand 4, theta 0.5, logit


def _solve(sc, model=None, **budget_kwargs):
    trace = Trace()
    (model or CostSmoothingSUEModel()).solve(
        sc, Budget(**budget_kwargs), RngBundle(0), trace
    )
    return trace


def _braess_sue(theta: float = 0.1) -> Scenario:
    """A multi-path logit-SUE task on the Braess network (three OD paths) built
    from the shipped Braess network -- exercises the loader on a real >2-route
    network, not just the analytic anchor."""
    b = braess_scenario()
    return Scenario(
        name="braess-sue",
        network=b.network,
        demand=b.demand,
        sue_family="logit",
        sue_theta=theta,
    )


# ------------------------------------------------------------- convergence
def test_converges_to_logit_sue_fixed_point(scenario):
    """Below the stability threshold (w = 0.3) the cost-smoothing dynamics settle
    on the analytic binary-logit SUE (route flows are non-unique, but the link
    flows are), and self-report the certified residual that -> 0 there."""
    f_a = _fixed_point_route_a(theta=0.5)
    trace = _solve(
        scenario, CostSmoothingSUEModel(smoothing_weight=0.3),
        iterations=500, target_relative_gap=1e-8,
    )
    expected = np.array([f_a, f_a, 4.0 - f_a, 4.0 - f_a])
    np.testing.assert_allclose(trace.final.link_flows, expected, atol=1e-3)
    metrics = Evaluator(scenario).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["sue_fixed_point_residual"] < 1e-5
    # The logit SUE is NOT the deterministic UE: the UE gap stays strictly
    # positive as a descriptive column (like sue-msa).
    assert metrics["relative_gap"] > 0.01


def test_converges_on_multipath_braess_sue():
    """On a genuinely multi-path network (three Braess routes) the cost-smoothing
    map still drives the certified SUE residual toward zero at a small weight and
    stays demand-feasible at every day."""
    sc = _braess_sue(theta=0.1)
    trace = _solve(
        sc, CostSmoothingSUEModel(smoothing_weight=0.3),
        iterations=1000, target_relative_gap=1e-8,
    )
    metrics = Evaluator(sc).evaluate(trace.final.link_flows)
    assert metrics["feasible"] == 1.0
    assert metrics["node_balance_residual"] <= 1e-6 * sc.demand.total
    assert metrics["sue_fixed_point_residual"] < 1e-5


def test_cross_solver_agrees_with_sue_msa(scenario):
    """dtd-horowitz (day-to-day cost smoothing) and sue-msa (the MSA solver) reach
    the SAME certified logit-SUE link flows on the anchor -- both fixed points of
    the same pinned Dial-STOCH map."""
    horowitz = _solve(
        scenario, CostSmoothingSUEModel(smoothing_weight=0.3),
        iterations=500, target_relative_gap=1e-9,
    )
    msa = _solve(scenario, DialSUEModel(), iterations=500, target_relative_gap=1e-9)
    np.testing.assert_allclose(
        horowitz.final.link_flows, msa.final.link_flows, atol=1e-4
    )


# ------------------------------------------------- stability vs. oscillation
def test_oscillates_above_stability_threshold(scenario):
    """The distinctive dynamical property: with w = 1.0 (Horowitz's naive
    current-cost model, above the anchor threshold w* ~ 0.81) the constant-weight
    smoothing map does NOT converge -- it settles into a period-2 limit cycle, so
    the certified residual stays O(1) and the emitted route-A flow oscillates over
    a wide band. No damping is added, so the divergence is preserved."""
    trace = _solve(
        scenario, CostSmoothingSUEModel(smoothing_weight=1.0), iterations=400
    )
    res = [s.self_report["sue_fixed_point_residual"] for s in trace]
    fa = np.array([s.link_flows[0] for s in trace])
    tail_res = res[-50:]
    tail_fa = fa[-50:]
    # The residual never converges: it stays large through the last 50 days.
    assert res[-1] > 1.0
    assert max(tail_res) > 1.0
    # ... and the emitted flow is a genuine limit cycle, not a fixed point.
    assert tail_fa.max() - tail_fa.min() > 1.0


def test_terminal_residual_monotone_in_smoothing_weight(scenario):
    """Crossover check: the terminal SUE residual is small for smoothing weights
    below the anchor stability threshold (w in {0.2, 0.5, 0.8}) and large above it
    (w in {0.85, 1.0}) -- the forward-Euler stability limit w* = 2/(1 - phi') ~
    0.81 of the underlying cost-learning ODE."""
    def terminal_residual(w: float) -> float:
        trace = _solve(scenario, CostSmoothingSUEModel(smoothing_weight=w), iterations=500)
        return trace.final.self_report["sue_fixed_point_residual"]

    for w in (0.2, 0.5, 0.8):
        assert terminal_residual(w) < 1e-3
    for w in (0.85, 1.0):
        assert terminal_residual(w) > 1.0


def test_perceived_cost_gap_vanishes_at_equilibrium(scenario):
    """The perceived-cost gap ||p - t(v)||_1 (perception minus experience) is a
    provenance measure that is zero iff the perceived-cost state is at rest; it is
    O(1) off equilibrium (day 0 starts at free-flow costs) and collapses to zero
    as the stable dynamics converge (reported as provenance, never scored)."""
    trace = _solve(scenario, CostSmoothingSUEModel(smoothing_weight=0.3), iterations=300)
    gap = [s.self_report["perceived_cost_gap"] for s in trace]
    assert gap[-1] < 1e-6
    assert max(gap) > 1.0
    assert gap[-1] < max(gap)


# ------------------------------------------------------------- honesty (P1)
def test_self_report_matches_harness_certificate(scenario):
    """P1 honesty: the model's self-reported residual equals the one the harness
    recomputes -- both call the SAME pinned StochEngine.load (the model loads a
    second time each day at the experienced costs precisely to certify), so they
    agree to float precision at every checkpoint."""
    trace = _solve(scenario, CostSmoothingSUEModel(smoothing_weight=0.3), iterations=50)
    evaluator = Evaluator(scenario)
    for state in list(trace)[::10]:
        certified = evaluator.evaluate(state.link_flows)["sue_fixed_point_residual"]
        assert certified == pytest.approx(
            state.self_report["sue_fixed_point_residual"], rel=1e-9, abs=1e-15
        )


# --------------------------------------------------------------------- guards
def test_requires_sue_scenario():
    """A deterministic (non-SUE) scenario has no theta: refuse it (theta is task
    data, not a model factor)."""
    with pytest.raises(ValueError, match="sue_theta|SUE scenario"):
        CostSmoothingSUEModel().solve(
            braess_scenario(), Budget(iterations=5), RngBundle(0), Trace()
        )


def test_rejects_probit_scenario():
    """The logit cost-smoothing model must refuse a probit-SUE task and point at
    the probit solver."""
    probit = two_route_scenario(sue_theta=0.1, sue_family="probit")
    with pytest.raises(ValueError, match="probit"):
        CostSmoothingSUEModel().solve(
            probit, Budget(iterations=5), RngBundle(0), Trace()
        )


# ------------------------------------------------------------------- mechanics
def test_paradigm_and_registry():
    from tabench.models import MODEL_REGISTRY

    assert "dtd-horowitz" in MODEL_REGISTRY
    assert MODEL_REGISTRY["dtd-horowitz"]().capabilities.paradigm == "day_to_day"


def test_bookkeeping_and_conservation(scenario):
    trace = _solve(scenario, iterations=10)
    assert len(trace) == 10
    # Two Dial-STOCH loads per day -- one at the perceived costs to EMIT the
    # physical flow, one at the experienced costs to CERTIFY the residual -- both
    # counted in sp_calls, with no separate day-0 load, so at k days the count is
    # exactly 2k.
    assert trace.final.coords.sp_calls == 20
    v = trace.final.link_flows
    assert np.all(v >= 0)
    metrics = Evaluator(scenario).evaluate(v)
    # Dial routes all demand every day, so the emitted link flows balance to the
    # float-noise floor at every checkpoint.
    assert metrics["feasible"] == 1.0
    assert metrics["node_balance_residual"] <= 1e-6 * scenario.demand.total
    for key in ("sue_fixed_point_residual", "perceived_cost_gap"):
        assert key in trace.final.self_report


def test_braess_content_hash_preserved():
    """This model adds no scenario field: the golden Braess content hash must be
    byte-identical."""
    assert braess_scenario().content_hash() == BRAESS_GOLDEN_HASH
