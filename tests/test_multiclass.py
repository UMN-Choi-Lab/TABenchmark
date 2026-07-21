"""Multiclass-user traffic assignment (Dafermos 1972; adr-013).

Every scored quantity is recomputed by the harness from the emitted per-class
flows; the analytic anchors are hand-derived closed forms (symmetric =
integrable, asymmetric = genuine VI), recomputed here, never trusted digits.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import tabench as tb
from tabench.core.budget import Budget
from tabench.core.results import Trace
from tabench.core.rng import RngBundle
from tabench.core.scenario import Demand, MulticlassDemand, Scenario
from tabench.data.builtin import (
    _MC_INTERACTION_ASYMMETRIC,
    _MC_INTERACTION_SYMMETRIC,
    multiclass_two_route_scenario,
)
from tabench.metrics.gaps import Evaluator

BRAESS_GOLDEN_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"


def _solve(scenario: Scenario, iterations: int = 600) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run the multiclass model; return (per-class flows, aggregate, final self_report)."""
    model = tb.MulticlassModel()
    trace = Trace()
    budget = Budget(iterations=iterations, target_relative_gap=1e-13)
    model.solve(scenario, budget, RngBundle(0), trace)
    return trace.final.class_link_flows, trace.final.link_flows, trace.final.self_report


# --------------------------------------------------------------------------- P8

def test_golden_braess_hash_preserved() -> None:
    """The new optional multiclass field must not perturb any existing hash."""
    assert tb.braess_scenario().content_hash() == BRAESS_GOLDEN_HASH


# --------------------------------------------------- MulticlassDemand validation

def test_multiclass_demand_requires_two_classes() -> None:
    with pytest.raises(ValueError, match=">= 2 classes"):
        MulticlassDemand(matrices=np.zeros((1, 2, 2)), interaction=np.zeros((1, 1)))


def test_multiclass_demand_interaction_shape() -> None:
    with pytest.raises(ValueError, match="interaction must have shape"):
        MulticlassDemand(matrices=np.zeros((2, 2, 2)), interaction=np.zeros((3, 3)))


def test_multiclass_demand_rejects_negative_and_nonfinite() -> None:
    bad = np.zeros((2, 2, 2))
    bad[0, 0, 1] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        MulticlassDemand(matrices=bad, interaction=np.eye(2))
    with pytest.raises(ValueError, match="interaction must be finite"):
        MulticlassDemand(matrices=np.zeros((2, 2, 2)), interaction=np.full((2, 2), np.inf))


def test_multiclass_demand_helpers() -> None:
    m = np.zeros((2, 2, 2))
    m[0, 0, 1] = 4.0
    m[1, 0, 1] = 2.0
    sym = MulticlassDemand(matrices=m, interaction=_MC_INTERACTION_SYMMETRIC)
    assert sym.n_classes == 2
    assert sym.n_zones == 2
    assert sym.total == pytest.approx(6.0)
    np.testing.assert_allclose(sym.total_matrix, [[0.0, 6.0], [0.0, 0.0]])
    assert sym.symmetric() is True
    asym = MulticlassDemand(matrices=m, interaction=_MC_INTERACTION_ASYMMETRIC)
    assert asym.symmetric() is False


# --------------------------------------------------------- Scenario validation

def test_scenario_demand_must_equal_class_sum() -> None:
    m = np.zeros((2, 2, 2))
    m[0, 0, 1] = 4.0
    m[1, 0, 1] = 2.0
    mc = MulticlassDemand(matrices=m, interaction=_MC_INTERACTION_SYMMETRIC)
    net = multiclass_two_route_scenario().network
    with pytest.raises(ValueError, match="must equal the multiclass class sum"):
        Scenario(name="bad", network=net, demand=Demand(np.zeros((2, 2))), multiclass=mc)


def test_scenario_multiclass_mutually_exclusive() -> None:
    sc = multiclass_two_route_scenario()
    with pytest.raises(ValueError, match="mutually exclusive"):
        dataclasses.replace(sc, link_interaction=np.zeros((4, 4)))


# ---------------------------------------------------------------- content hash

def test_content_hash_distinguishes_interaction() -> None:
    """Symmetric and asymmetric interactions are different benchmark instances."""
    sym = multiclass_two_route_scenario()
    asym = multiclass_two_route_scenario(interaction=_MC_INTERACTION_ASYMMETRIC)
    assert sym.content_hash() != asym.content_hash()
    # Deterministic: rebuilding the same instance hashes identically.
    assert sym.content_hash() == multiclass_two_route_scenario().content_hash()


def test_content_hash_distinguishes_class_demand() -> None:
    a = multiclass_two_route_scenario(g_cars=4.0, g_trucks=2.0)
    b = multiclass_two_route_scenario(g_cars=3.0, g_trucks=3.0)  # same total, diff split
    assert a.demand.total == pytest.approx(b.demand.total)
    assert a.content_hash() != b.content_hash()


# ------------------------------------------------------- analytic anchor recovery

def test_symmetric_anchor_recovery() -> None:
    """Integrable case: cars (2.5,1.5), trucks (1.5,0.5) per route, aggregate (4,2)."""
    sc = multiclass_two_route_scenario()
    v, agg, _ = _solve(sc)
    np.testing.assert_allclose(v[0], [2.5, 2.5, 1.5, 1.5], atol=1e-5)
    np.testing.assert_allclose(v[1], [1.5, 1.5, 0.5, 0.5], atol=1e-5)
    np.testing.assert_allclose(agg, [4.0, 4.0, 2.0, 2.0], atol=1e-5)
    # Aggregate emitted == class sum.
    np.testing.assert_allclose(agg, v.sum(axis=0), atol=1e-12)
    # Matches the builder's closed-form reference.
    np.testing.assert_allclose(agg, sc.reference.link_flows, atol=1e-5)


def test_asymmetric_anchor_recovery_and_class_distinctness() -> None:
    """Genuine VI: cars (2,2), trucks (1.75,0.25) per route — classes route
    differently (the multiclass signature)."""
    sc = multiclass_two_route_scenario(interaction=_MC_INTERACTION_ASYMMETRIC)
    v, agg, _ = _solve(sc)
    np.testing.assert_allclose(v[0], [2.0, 2.0, 2.0, 2.0], atol=1e-5)
    np.testing.assert_allclose(v[1], [1.75, 1.75, 0.25, 0.25], atol=1e-5)
    np.testing.assert_allclose(agg, [3.75, 3.75, 2.25, 2.25], atol=1e-5)
    # Cars split 50/50 but trucks 87.5/12.5 — genuinely distinct per-class routing.
    assert not np.allclose(v[0] / 4.0, v[1] / 2.0, atol=1e-2)


# ------------------------------------------------------------- P1 certificate

def test_certificate_zero_gap_at_equilibrium() -> None:
    for interaction in (_MC_INTERACTION_SYMMETRIC, _MC_INTERACTION_ASYMMETRIC):
        sc = multiclass_two_route_scenario(interaction=interaction)
        v, agg, self_report = _solve(sc)
        met = Evaluator(sc).evaluate(agg, v)
        assert met["feasible"] == 1.0
        assert met["relative_gap"] < 1e-8
        assert np.isnan(met["beckmann_objective"])  # no potential for coupled cost
        # Harness recomputation agrees with the model self-report (honesty check).
        assert met["relative_gap"] == pytest.approx(self_report["relative_gap"], abs=1e-9)


def test_certificate_censors_missing_class_flows() -> None:
    """An aggregate flow (no per-class breakdown) cannot certify multiclass."""
    sc = multiclass_two_route_scenario()
    _, agg, _ = _solve(sc)
    met = Evaluator(sc).evaluate(agg, None)
    assert met["feasible"] == 0.0
    assert np.isnan(met["relative_gap"])


def test_certificate_censors_negative_class_flows() -> None:
    sc = multiclass_two_route_scenario()
    v, agg, _ = _solve(sc)
    bad = v.copy()
    bad[0, 0] = -5.0
    met = Evaluator(sc).evaluate(bad.sum(axis=0), bad)
    assert met["feasible"] == 0.0


def test_certificate_censors_class_conservation_violation() -> None:
    """A flow that routes the wrong per-class demand is censored even if the
    aggregate conserves (per-class audit, adr-013)."""
    sc = multiclass_two_route_scenario()
    v, _, _ = _solve(sc)
    # Swap the two classes' flows: aggregate is unchanged, but now "cars" carries
    # the trucks' demand (4 != 2) and vice versa — per-class conservation fails.
    swapped = v[::-1].copy()
    met = Evaluator(sc).evaluate(swapped.sum(axis=0), swapped)
    assert met["feasible"] == 0.0


def test_certificate_wrong_class_shape_raises() -> None:
    sc = multiclass_two_route_scenario()
    _, agg, _ = _solve(sc)
    with pytest.raises(ValueError, match="class_link_flows shape"):
        Evaluator(sc).evaluate(agg, np.zeros((3, 4)))


# --------------------------------------------- Budget target + line-search xtol

def _run(scenario: Scenario, iterations: int, target: float | None = None, model=None) -> Trace:
    """Flexible solve: optional caller target and factor-overridden model."""
    trace = Trace()
    kw: dict = {"iterations": iterations}
    if target is not None:
        kw["target_relative_gap"] = target
    (model or tb.MulticlassModel()).solve(scenario, Budget(**kw), RngBundle(0), trace)
    return trace


def test_budget_target_relative_gap_stops_early() -> None:
    """T1 (on): the caller-facing Budget.target_relative_gap is now a real stop
    channel (parity with the sibling solvers via budget.target_met). A LOOSE caller
    target (1e-2, far looser than the 1e-10 target_gap factor) stops the outer
    Gauss-Seidel sweep at strictly fewer sweeps than an otherwise-identical run
    that leaves the caller target unset, and the last class-summed VI gap meets it."""
    sc = multiclass_two_route_scenario(interaction=_MC_INTERACTION_ASYMMETRIC)
    default = _run(sc, iterations=600)  # no caller target -> factor (1e-10) gates
    loose = _run(sc, iterations=600, target=1e-2)
    assert len(loose) < len(default)  # measured 3 vs 20 sweeps
    assert loose.final.self_report["relative_gap"] <= 1e-2


def test_budget_target_none_is_byte_identical_off_noop() -> None:
    """T1 (off): with a caller target of None -- or any value at least as tight as
    the model's own 1e-10 target_gap factor -- the new budget.target_met channel is
    inert (the factor fires first, still under its np.isfinite guard), so the run is
    byte-identical to the prior factor-only behavior. The existing pinned anchors
    (test_symmetric_anchor_recovery etc., which solve with target_relative_gap=1e-13)
    still pass -- the off-pin; here an explicit trace-length + per-class flow equality
    between a None-target run and a tight (1e-13) run."""
    sc = multiclass_two_route_scenario(interaction=_MC_INTERACTION_ASYMMETRIC)
    none = _run(sc, iterations=600)
    tight = _run(sc, iterations=600, target=1e-13)
    assert len(none) == len(tight)
    np.testing.assert_array_equal(none.final.class_link_flows, tight.final.class_link_flows)


def test_line_search_xtol_default_is_no_op() -> None:
    """T2 (off): line_search_xtol defaults to 1e-13 -- the value the brentq inner
    line search was previously hardcoded to -- so a default run is byte-identical to
    an explicit xtol=1e-13 run. The existing pinned anchors (which never set the
    factor) are the off-pin for the emitted per-class flows."""
    assert tb.MulticlassModel().factor_values["line_search_xtol"] == 1e-13
    sc = multiclass_two_route_scenario(interaction=_MC_INTERACTION_ASYMMETRIC)
    default = _run(sc, iterations=600, target=1e-13)
    explicit = _run(
        sc, iterations=600, target=1e-13, model=tb.MulticlassModel(line_search_xtol=1e-13)
    )
    np.testing.assert_array_equal(
        default.final.class_link_flows, explicit.final.class_link_flows
    )


def test_line_search_xtol_loose_changes_flows() -> None:
    """T2 (on): line_search_xtol threads into the per-class brentq inner line search.
    On the asymmetric multiclass anchor a loose xtol=1e-3 yields per-class flows NOT
    byte-identical to the default (tight 1e-13) run, while still certifying a
    feasible, finite-gap multiclass equilibrium (measured max|diff| ~3.2e-4). This
    KILLS the "factor declared but not threaded from brentq" mutant: the existing
    off-pin compares a default run to an explicit-same-value run, which stays
    byte-identical even if the factor is un-threaded -- only a loose-value on-test
    can see the threading."""
    sc = multiclass_two_route_scenario(interaction=_MC_INTERACTION_ASYMMETRIC)
    default = _run(sc, iterations=600, target=1e-13)
    loose = _run(sc, iterations=600, target=1e-13, model=tb.MulticlassModel(line_search_xtol=1e-3))
    assert not np.array_equal(
        default.final.class_link_flows, loose.final.class_link_flows
    )
    # Sanity: the loose run still certifies a feasible, finite-gap equilibrium.
    met = Evaluator(sc).evaluate(loose.final.link_flows, loose.final.class_link_flows)
    assert met["feasible"] == 1.0
    assert np.isfinite(met["relative_gap"])


# ------------------------------------------------------------- integration

def test_registry_and_paradigm() -> None:
    from tabench.models.base import MODEL_REGISTRY

    assert "multiclass" in MODEL_REGISTRY
    caps = MODEL_REGISTRY["multiclass"].capabilities
    assert caps.paradigm == "static_ue_multiclass"
    assert caps.deterministic is True


def test_load_scenario_key() -> None:
    sc = tb.load_scenario("multiclass")
    assert sc.multiclass is not None
    assert sc.multiclass.n_classes == 2


def test_determinism() -> None:
    """P8: two solves of the same instance are byte-identical."""
    sc = multiclass_two_route_scenario(interaction=_MC_INTERACTION_ASYMMETRIC)
    v1, _, _ = _solve(sc)
    v2, _, _ = _solve(sc)
    np.testing.assert_array_equal(v1, v2)


def test_end_to_end_run_experiment() -> None:
    from tabench.experiments.runner import run_experiment

    sc = multiclass_two_route_scenario()
    res = run_experiment(sc, [tb.MulticlassModel()], Budget(iterations=400))
    final = res.rows[-1]
    assert final["feasible"] == 1.0
    assert final["relative_gap"] < 1e-7
    assert final["model"] == "multiclass"


def test_missing_multiclass_raises() -> None:
    """The model refuses a scenario without multiclass demand."""
    model = tb.MulticlassModel()
    with pytest.raises(ValueError, match="requires a scenario with multiclass"):
        model.solve(tb.braess_scenario(), Budget(iterations=1), RngBundle(0), Trace())
