"""Newell (1993) three-detector interior reconstruction — anchors + certifier.

The benchmark's first traffic-state-estimation unit (adr-024): given noisy /
partial boundary detector curves, reconstruct the interior cumulative field,
scored against the harness-regenerated closed-form Newell min. Every anchor number
is recomputed in-test (house style: no trusted digits). Formulation cross-verified
from Boyles TNA §9.4.4 eq. 9.46, Rey-Jin-Ritchie 2019 eqs. 8-10, Daganzo
VWP-2006-2 eq. 17 (all open, read); primaries paywalled (attributed unread).
"""

import numpy as np
import pytest

from tabench.dnl import CTMLink, LTMLink, NetworkLoader, TimeGrid
from tabench.metrics import ThreeDetectorEvaluator
from tabench.newell import (
    ThreeDetectorField,
    newell_free_flow_scenario,
    newell_masked_upstream_scenario,
    newell_min,
    newell_min_isotonic,
    newell_noisy_scenario,
    newell_spillback_scenario,
    newell_symmetric_scenario,
    observe_detectors,
    problem_from_scenario,
    reconstruct_field,
)
from tabench.newell.scenario import ThreeDetectorScenario, _interp_curve

# ---------------------------------------------------------------------------
# A1 — free-flow exactness
# ---------------------------------------------------------------------------


def test_a1_free_flow_reconstruction_is_the_translated_upstream_curve():
    sc = newell_free_flow_scenario()
    times, n_up, n_dn = sc.truth_boundary_curves()
    M = sc.reference_field()
    # meter (2.0) is above inflow (1.0), so no bottleneck: N(x,t) = N_up(t - x/vf)
    for i, x in enumerate(sc.x_query):
        expected = _interp_curve(n_up, times - x / sc.vf, sc.grid.dt)
        np.testing.assert_allclose(M[i], expected, atol=1e-12)
    # the exact-min estimator on the clean curves reproduces M to machine precision
    ev = ThreeDetectorEvaluator(sc)
    metrics = ev.evaluate(newell_min_isotonic(problem_from_scenario(sc)))
    assert metrics["feasible"] == 1.0
    assert metrics["interior_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["rankable"] == 0.0  # clean = oracle/validity row, never ranked


# ---------------------------------------------------------------------------
# A2 — asymmetric spillback, interior Rankine-Hugoniot shock
# ---------------------------------------------------------------------------


def test_a2_min_switch_at_hand_computed_shock_crossings():
    """vf=2, w=1, kappa=3, L=4, inflow 1.0 into a 0.5 meter. Free state q_A=1,
    k_A=0.5; congested q_B=0.5, k_B=kappa-q_B/w=2.5; RH shock speed
    s=(q_A-q_B)/(k_A-k_B)=(0.5)/(-2)=-0.25 born at (x=4, t=L/vf=2). The shock
    passes interior x at t = 2 + (4-x)/0.25."""
    sc = newell_spillback_scenario()
    times, n_up, n_dn = sc.truth_boundary_curves()
    M = sc.reference_field()
    dt = sc.grid.dt
    for i, x in enumerate(sc.x_query):
        t_switch = 2.0 + (4.0 - x) / 0.25  # x=1->14, x=2->10, x=3->6
        free = _interp_curve(n_up, times - x / sc.vf, dt)
        cong = _interp_curve(n_dn, times - (sc.length - x) / sc.w, dt) + sc.kappa * (
            sc.length - x
        )
        # before the crossing the free branch binds; strictly after, the congested
        k = int(t_switch)
        assert M[i, k] == pytest.approx(free[k], abs=1e-9)
        assert M[i, k + 1] == pytest.approx(cong[k + 1], abs=1e-9)
        assert cong[k + 1] < free[k + 1] - 1e-9

    # x=2: switch value N=9 at t=10, post-shock N(2,t) = 0.5*t + 4 (slope q_B=0.5)
    i2 = int(np.where(sc.x_query == 2.0)[0][0])
    assert M[i2, 10] == pytest.approx(9.0, abs=1e-9)
    for t in (12, 16, 20):
        assert M[i2, t] == pytest.approx(0.5 * t + 4.0, abs=1e-9)


def test_a2_reconstructed_congested_density_is_k_B():
    """Post-shock the interior is uniformly congested at k_B = kappa - q_B/w = 2.5;
    recover it by the central x-difference k = -(N(3,t)-N(1,t))/(3-1)."""
    sc = newell_spillback_scenario()
    M = sc.reference_field()
    i1 = int(np.where(sc.x_query == 1.0)[0][0])
    i3 = int(np.where(sc.x_query == 3.0)[0][0])
    t = 20  # fully congested at x=1,2,3 (shock passed x=1 at t=14)
    k_density = -(M[i3, t] - M[i1, t]) / (3.0 - 1.0)
    assert k_density == pytest.approx(2.5, abs=1e-9)


def test_a2_wrong_sign_convention_moves_the_shock_crossing():
    """The '+(L-x)/w' variant (task-prompt sign, w>0) is non-causal and moves the
    x=2 min-switch from t=10 to t=14 — machine-detectable (the dossier sign guard)."""
    sc = newell_spillback_scenario()
    times, n_up, n_dn = sc.truth_boundary_curves()
    dt, x = sc.grid.dt, 2.0
    free = _interp_curve(n_up, times - x / sc.vf, dt)
    wrong = _interp_curve(n_dn, times + (sc.length - x) / sc.w, dt) + sc.kappa * (sc.length - x)
    wrong_field = np.minimum(free, wrong)
    i2 = int(np.where(sc.x_query == 2.0)[0][0])
    # the correct minus-sign field has already left the free branch by t=11
    assert sc.reference_field()[i2, 11] < free[11] - 1e-9
    # the wrong plus-sign (non-causal) field is STILL on the free branch at t=11 and
    # only switches near t=15: n_up(t-1)=n_dn(t+2)+6 -> t-1=0.5t+6 -> t=14 (equal), 15 (bound)
    assert wrong_field[11] == pytest.approx(free[11], abs=1e-9)
    assert wrong_field[15] < free[15] - 1e-9


# ---------------------------------------------------------------------------
# A3 — aligned symmetric bottleneck: cross-model truth LTM == CTM
# ---------------------------------------------------------------------------


def test_a3_truth_generator_ltm_equals_ctm_byte_for_byte():
    """The truth-generating LTM boundary curves reproduce CTM byte-for-byte on the
    aligned symmetric bottleneck (adr-016 anchor b), doubly validating the truth
    generator (the lp-so-dta corridor==CTMLink move)."""
    sc = newell_symmetric_scenario()
    truth = sc._truth_scenario()
    ltm = NetworkLoader(truth, LTMLink).run()
    ctm = NetworkLoader(truth, CTMLink).run()
    np.testing.assert_allclose(ltm.n_in, ctm.n_in, atol=1e-9)
    np.testing.assert_allclose(ltm.n_out, ctm.n_out, atol=1e-9)
    # and the oracle reconstruction is exact on this clean instance
    ev = ThreeDetectorEvaluator(sc)
    assert ev.evaluate(newell_min(problem_from_scenario(sc)))["interior_rmse"] == pytest.approx(
        0.0, abs=1e-12
    )


# ---------------------------------------------------------------------------
# A4 — the noisy discrimination anchor (acceptance test for the whole unit)
# ---------------------------------------------------------------------------


def test_a4_isotonic_strictly_beats_naive_by_a_pinned_margin():
    """On the seeded Gaussian-reading level, the isotonic-then-min estimator
    strictly dominates the naive running-max baseline in interior RMSE by a pinned
    deterministic margin. This is the acceptance test: the ranked task genuinely
    separates methods and is not a formula evaluation."""
    sc = newell_noisy_scenario()
    ev = ThreeDetectorEvaluator(sc)
    prob = problem_from_scenario(sc)
    naive = ev.evaluate(newell_min(prob))
    iso = ev.evaluate(newell_min_isotonic(prob))
    assert naive["feasible"] == 1.0 and iso["feasible"] == 1.0
    assert naive["rankable"] == 1.0 and iso["rankable"] == 1.0
    # pinned values (seed 20260714); regenerated deterministically
    assert naive["interior_rmse"] == pytest.approx(1.0927785, abs=1e-6)
    assert iso["interior_rmse"] == pytest.approx(0.4483981, abs=1e-6)
    assert naive["interior_rmse"] - iso["interior_rmse"] > 0.6


def test_a4_seeded_observation_is_deterministic():
    sc = newell_noisy_scenario()
    a = observe_detectors(sc)
    b = observe_detectors(sc)
    np.testing.assert_array_equal(a.up, b.up)
    np.testing.assert_array_equal(a.dn, b.dn)


def test_poisson_level_is_monotone_and_reconstructs_feasibly():
    """Per-interval Poisson counts (the LinkCounts precedent) give a monotone
    cumulative — a faithful low-count level whose reconstruction is feasible."""
    sc = ThreeDetectorScenario(
        name="pois", vf=1.0, w=1.0, kappa=4.0, length=4.0, capacity=2.0, meter_cap=0.5,
        inflow_breakpoints=np.array([0.0, 12.0]), inflow_rates=np.array([1.5]),
        grid=TimeGrid(dt=1.0, n_steps=12), x_query=np.array([1.0, 2.0, 3.0]),
        noise="poisson", n_days=5, seed=3,
    )
    obs = observe_detectors(sc)
    assert np.all(np.diff(obs.up, axis=1) >= 0.0)  # Poisson increments are nonnegative
    assert np.all(np.diff(obs.dn, axis=1) >= 0.0)
    m = ThreeDetectorEvaluator(sc).evaluate(newell_min_isotonic(problem_from_scenario(sc)))
    assert m["feasible"] == 1.0 and m["rankable"] == 1.0


def test_drift_dial_biases_the_record_but_preserves_the_known_zero_start():
    sc = ThreeDetectorScenario(
        name="drift", vf=1.0, w=1.0, kappa=4.0, length=4.0, capacity=2.0, meter_cap=0.5,
        inflow_breakpoints=np.array([0.0, 12.0]), inflow_rates=np.array([1.5]),
        grid=TimeGrid(dt=1.0, n_steps=12), x_query=np.array([1.0, 2.0, 3.0]),
        noise="gaussian", read_sigma=0.3, drift=0.05, n_days=3, seed=3,
    )
    obs = observe_detectors(sc)
    assert obs.up[:, 0].tolist() == [0.0, 0.0, 0.0]  # empty start is known, never drifted
    m = ThreeDetectorEvaluator(sc).evaluate(newell_min_isotonic(problem_from_scenario(sc)))
    assert m["feasible"] == 1.0


# ---------------------------------------------------------------------------
# A5 — observability edge: missing upstream window
# ---------------------------------------------------------------------------


def test_a5_congested_branch_pins_interior_exactly_where_active():
    """With the upstream detector masked over [11, 17], the congested branch alone
    pins the interior EXACTLY wherever the congested branch is active; the only
    error is where the interior is still free-flow but upstream is unobserved."""
    sc = newell_masked_upstream_scenario()
    ev = ThreeDetectorEvaluator(sc)
    M = sc.reference_field()
    field = newell_min_isotonic(problem_from_scenario(sc)).field
    err = field - M
    cong = ev._cong_active
    # exact wherever the congested branch is active
    assert np.abs(err[cong]).max() == pytest.approx(0.0, abs=1e-9)
    # the ONLY discrepancy: x=1 at t in {12,13} (shock reaches x=1 only at t=14),
    # where the dropped free branch was the truth and cong overestimates by exactly 1.0
    i1 = int(np.where(sc.x_query == 1.0)[0][0])
    nz = np.where(np.abs(err[i1]) > 1e-9)[0]
    np.testing.assert_array_equal(nz, np.array([12, 13]))
    # cong(1,12) - free(1,12) = [n_dn(9)+kappa*3] - n_up(11.5) = (3.5+9) - 11.5 = 1.0
    assert err[i1, 12] == pytest.approx(1.0, abs=1e-9)
    assert ev.evaluate(newell_min_isotonic(problem_from_scenario(sc)))["feasible"] == 1.0


# ---------------------------------------------------------------------------
# Tier-B pairwise Newell envelopes (the C5 analog)
# ---------------------------------------------------------------------------


def test_reference_field_passes_both_pairwise_envelopes_exactly():
    for scf in (newell_spillback_scenario, newell_symmetric_scenario):
        sc = scf()
        ev = ThreeDetectorEvaluator(sc)
        M = sc.reference_field()
        f = ThreeDetectorField(sc.content_hash(), sc.x_query, sc.grid.edges, M)
        m = ev.evaluate(f)
        assert m["envelope_forward_residual"] == pytest.approx(0.0, abs=1e-9)
        assert m["envelope_backward_residual"] == pytest.approx(0.0, abs=1e-9)
        assert m["envelope_exact"] == 1.0


def test_noisy_baseline_violates_backward_envelope_but_is_not_censored():
    """A legitimate noisy reconstruction violates the backward storage envelope at
    noise scale (exactly as CTM legitimately violates C5), so it is reported —
    NOT censored. Gating it would false-censor the honest baseline."""
    sc = newell_noisy_scenario()
    ev = ThreeDetectorEvaluator(sc)
    m = ev.evaluate(newell_min(problem_from_scenario(sc)))
    assert m["feasible"] == 1.0  # Tier B, non-gating
    assert m["envelope_backward_residual"] > 1e-3  # a real, reported violation
    assert m["envelope_exact"] == 0.0


# ---------------------------------------------------------------------------
# Censoring (malformed emissions) — house conventions
# ---------------------------------------------------------------------------


def _clean_field(sc):
    return ThreeDetectorField(sc.content_hash(), sc.x_query, sc.grid.edges, sc.reference_field())


def test_censor_teleport_breaks_zero_start():
    sc = newell_symmetric_scenario()
    ev = ThreeDetectorEvaluator(sc)
    bad = sc.reference_field().copy()
    bad[1, :] += 5.0  # a teleport: nonzero start + storage violation
    f = ThreeDetectorField(sc.content_hash(), sc.x_query, sc.grid.edges, bad)
    assert ev.evaluate(f)["feasible"] == 0.0


def test_censor_non_monotone_emission():
    sc = newell_symmetric_scenario()
    ev = ThreeDetectorEvaluator(sc)
    bad = sc.reference_field().copy()
    bad[0, 5] += 10.0  # a spike, then a drop -> non-monotone
    bad[0, 6] = bad[0, 4]
    f = ThreeDetectorField(sc.content_hash(), sc.x_query, sc.grid.edges, bad)
    assert ev.evaluate(f)["feasible"] == 0.0


def test_censor_eps_creep_retraction_via_aggregate_budget():
    """Many tiny per-step dips, each far below any physical scale, sum over the
    horizon into a material retraction the aggregate total-retraction budget
    catches (the adr-022 lesson — never a per-step gate)."""
    sc = newell_symmetric_scenario()
    ev = ThreeDetectorEvaluator(sc)
    m, k1 = sc.x_query.shape[0], sc.grid.n_steps + 1
    # a nearly-flat plateau so each tiny dip is a genuine reversal (retraction)
    field = np.zeros((m, k1))
    field[:, 1:] = 2.0
    field[0, 2:12:2] -= 1e-4  # 5 dips of 1e-4; total retraction 5e-4 >> eps_count ~ 1.8e-8
    f = ThreeDetectorField(sc.content_hash(), sc.x_query, sc.grid.edges, field)
    assert ev.evaluate(f)["feasible"] == 0.0
    # a single sub-eps dip stays feasible (honest float noise below the budget)
    field2 = np.zeros((m, k1))
    field2[:, 1:] = 2.0
    field2[0, 5] -= 1e-13
    f2 = ThreeDetectorField(sc.content_hash(), sc.x_query, sc.grid.edges, field2)
    assert ev.evaluate(f2)["feasible"] == 1.0


def test_censor_negative_but_not_honest_zero():
    sc = newell_symmetric_scenario()
    ev = ThreeDetectorEvaluator(sc)
    # a materially negative count is censored
    bad = sc.reference_field().copy()
    bad[1, 6] = -1.0
    f = ThreeDetectorField(sc.content_hash(), sc.x_query, sc.grid.edges, bad)
    assert ev.evaluate(f)["feasible"] == 0.0
    # an all-zero field is a legitimate (terrible) estimate — NOT censored
    zero = np.zeros_like(sc.reference_field())
    fz = ThreeDetectorField(sc.content_hash(), sc.x_query, sc.grid.edges, zero)
    mz = ev.evaluate(fz)
    assert mz["feasible"] == 1.0
    assert mz["interior_rmse"] > 0.0


def test_censor_nonfinite_and_wrong_hash_raise_on_wrong_shape():
    sc = newell_symmetric_scenario()
    ev = ThreeDetectorEvaluator(sc)
    nan = sc.reference_field().copy()
    nan[0, 3] = np.inf
    assert ev.evaluate(
        ThreeDetectorField(sc.content_hash(), sc.x_query, sc.grid.edges, nan)
    )["feasible"] == 0.0
    # wrong scenario hash censors (not raises)
    assert ev.evaluate(
        ThreeDetectorField("deadbeef", sc.x_query, sc.grid.edges, sc.reference_field())
    )["feasible"] == 0.0
    # wrong shape raises (programming error in the wrapper)
    wrong = np.zeros((sc.x_query.shape[0], sc.grid.n_steps + 2))
    with pytest.raises(ValueError, match="shape"):
        ev.evaluate(
            ThreeDetectorField(
                sc.content_hash(), sc.x_query, np.arange(sc.grid.n_steps + 2, dtype=float), wrong
            )
        )


# ---------------------------------------------------------------------------
# Triviality trap, truth-leakage, hashing, golden Braess
# ---------------------------------------------------------------------------


def test_clean_level_is_an_oracle_row_never_ranked():
    """On every clean anchor the min formula scores 0 by construction, so the level
    is reported (feasible) but flagged rankable=0 — the formula-evaluation trap."""
    for scf in (newell_free_flow_scenario, newell_spillback_scenario, newell_symmetric_scenario):
        sc = scf()
        m = ThreeDetectorEvaluator(sc).evaluate(newell_min(problem_from_scenario(sc)))
        assert m["feasible"] == 1.0
        assert m["rankable"] == 0.0
        assert m["interior_rmse"] == pytest.approx(0.0, abs=1e-12)


def test_truth_recipe_never_reaches_the_model_visible_payload():
    """The estimator-facing problem and observation carry no truth recipe (demand,
    metering, reference) — a submission cannot regenerate ground truth (adr-023)."""
    sc = newell_noisy_scenario()
    prob = problem_from_scenario(sc)
    for hidden in ("meter_cap", "inflow_rates", "inflow_breakpoints", "reference_field"):
        assert not hasattr(prob, hidden)
    obs = prob.observation
    leak = {"meter_cap", "inflow_rates", "inflow_breakpoints", "demand"}
    assert leak.isdisjoint(obs.meta.keys())
    # the observation exposes only degraded detector curves + grid + windows
    assert set(vars(obs)) <= {
        "scenario_hash", "times", "up", "dn", "up_windows", "dn_windows", "meta"
    }


def test_hash_determinism_sensitivity_and_readonly_arrays():
    sc = newell_noisy_scenario()
    assert sc.content_hash() == sc.content_hash()
    # a change in any scored field moves the hash
    base = sc.content_hash()
    assert ThreeDetectorScenario(**{**vars_of(sc), "read_sigma": 1.3}).content_hash() != base
    assert ThreeDetectorScenario(**{**vars_of(sc), "seed": sc.seed + 1}).content_hash() != base
    assert ThreeDetectorScenario(**{**vars_of(sc), "meter_cap": 0.6}).content_hash() != base
    # emitted arrays are read-only copies
    with pytest.raises(ValueError):
        sc.x_query[0] = 99.0


def vars_of(sc):
    return {
        "name": sc.name, "vf": sc.vf, "w": sc.w, "kappa": sc.kappa, "length": sc.length,
        "meter_cap": sc.meter_cap, "inflow_breakpoints": sc.inflow_breakpoints,
        "inflow_rates": sc.inflow_rates, "grid": sc.grid, "x_query": sc.x_query,
        "capacity": sc.capacity, "noise": sc.noise, "read_sigma": sc.read_sigma,
        "drift": sc.drift, "up_windows": sc.up_windows, "dn_windows": sc.dn_windows,
        "n_days": sc.n_days, "seed": sc.seed,
    }


def test_conditioning_gate_rejects_ill_scaled_storage_term():
    with pytest.raises(ValueError, match="ill-conditioned"):
        ThreeDetectorScenario(
            name="bad", vf=1.0, w=1.0, kappa=1e12, length=4.0, meter_cap=0.5,
            inflow_breakpoints=np.array([0.0, 4.0]), inflow_rates=np.array([1.0]),
            grid=TimeGrid(dt=1.0, n_steps=8), x_query=np.array([1.0, 2.0, 3.0]),
        )


def test_reconstruct_field_matches_scalar_interp_curve():
    """The vectorized reconstruction reads the same audited PWL interpolation as
    the scalar dnl.interp_curve (shared machinery, no private copy)."""
    from tabench.dnl import interp_curve

    sc = newell_spillback_scenario()
    times, n_up, n_dn = sc.truth_boundary_curves()
    M = reconstruct_field(
        sc.vf, sc.w, sc.kappa, sc.length, times, n_up, n_dn, sc.x_query, sc.grid.dt
    )
    for i, x in enumerate(sc.x_query):
        for k, t in enumerate(times):
            free = interp_curve(n_up, t - x / sc.vf, sc.grid.dt)
            cong = interp_curve(n_dn, t - (sc.length - x) / sc.w, sc.grid.dt) + sc.kappa * (
                sc.length - x
            )
            assert M[i, k] == pytest.approx(min(free, cong), abs=1e-12)


# ---------------------------------------------------------------------------
# adr-024 review regressions
# ---------------------------------------------------------------------------


def test_masked_window_reconstruction_stays_monotone_and_feasible():
    """Review MAJOR (both lenses): dropping a masked branch to +inf made the raw
    min DIP when the masked branch returned (C3 false-censor: 125/514 fuzzed
    windowed scenarios censored BOTH honest baselines, even on clean data). The
    suffix-min repair keeps the reconstruction the tightest nondecreasing upper
    bound, so honest baselines stay feasible."""
    sc = ThreeDetectorScenario(
        name="clean-mask", vf=1.0, w=1.0, kappa=4.0, length=4.0, meter_cap=2.0,
        inflow_breakpoints=np.array([0.0, 8.0]), inflow_rates=np.array([1.0]),
        grid=TimeGrid(dt=1.0, n_steps=14), x_query=np.array([1.0, 2.0, 3.0]),
        noise="none", up_windows=((5.0, 7.0),),
    )
    ev = ThreeDetectorEvaluator(sc)
    prob = problem_from_scenario(sc)
    for est in (newell_min, newell_min_isotonic):
        m = ev.evaluate(est(prob))
        assert m["feasible"] == 1.0
        assert m["retraction_residual"] == pytest.approx(0.0, abs=1e-9)


def test_double_masked_cells_are_bridged_not_inf():
    """Review MAJOR: time-disjoint windows can still doubly mask an interior cell
    AFTER the branch shifts (vf=2, w=1: both branches of x=2 read t-1), which
    yielded min(inf, inf) = inf and a C0 whole-field false-censor. Doubly-masked
    cells are now bridged from the next observed bound."""
    sc = ThreeDetectorScenario(
        name="disjoint", vf=2.0, w=1.0, kappa=3.0, length=4.0, capacity=2.0,
        meter_cap=0.5, inflow_breakpoints=np.array([0.0, 24.0]),
        inflow_rates=np.array([1.0]), grid=TimeGrid(dt=1.0, n_steps=24),
        x_query=np.array([1.0, 2.0, 3.0]), noise="none",
        up_windows=((8.0, 12.0),), dn_windows=((4.0, 8.0),),
    )
    f = newell_min(problem_from_scenario(sc))
    assert np.isfinite(f.field).all()
    metrics = ThreeDetectorEvaluator(sc).evaluate(f)
    assert metrics["feasible"] == 1.0


def test_degenerate_noise_combinations_are_rejected():
    """Review MAJOR: 'gaussian, sigma=0' was RANKED while the min formula scores
    exactly 0 (the triviality trap); 'none' with drift degraded the data while
    staying unranked. Both are constructor errors now."""
    base = dict(
        name="deg", vf=1.0, w=1.0, kappa=4.0, length=4.0, meter_cap=2.0,
        inflow_breakpoints=np.array([0.0, 8.0]), inflow_rates=np.array([1.0]),
        grid=TimeGrid(dt=1.0, n_steps=14), x_query=np.array([2.0]),
    )
    with pytest.raises(ValueError, match="read_sigma"):
        ThreeDetectorScenario(**base, noise="gaussian", read_sigma=0.0)
    with pytest.raises(ValueError, match="drift"):
        ThreeDetectorScenario(**base, noise="none", drift=0.1)


def test_infinite_scales_are_rejected():
    """Review MINOR: total_demand = inf made eps_count infinite and silently
    neutered EVERY censor gate (and enabled a hash byte-migration collision);
    drift accepted inf/nan. Both rejected at construction now."""
    base = dict(
        name="inf", vf=1.0, w=1.0, kappa=4.0, length=4.0, meter_cap=2.0,
        grid=TimeGrid(dt=1.0, n_steps=14), x_query=np.array([2.0]), noise="none",
    )
    with pytest.raises(ValueError, match="finite"):
        ThreeDetectorScenario(
            **base, inflow_breakpoints=np.array([0.0, 1e300]),
            inflow_rates=np.array([5e180]),
        )
    with pytest.raises(ValueError, match="drift"):
        ThreeDetectorScenario(
            **base, inflow_breakpoints=np.array([0.0, 8.0]),
            inflow_rates=np.array([1.0]), drift=float("nan"),
        )


def test_meter_above_apex_is_clipped_not_crashed():
    """Review MINOR: meter_cap above the geometric apex crashed three layers deep
    in LinkDynamics although a never-binding meter is a sanctioned concept (A1).
    It is clipped at the apex now — physics-identical, distinct hash."""
    base = dict(
        vf=1.0, w=1.0, kappa=4.0, length=4.0,
        inflow_breakpoints=np.array([0.0, 8.0]), inflow_rates=np.array([1.0]),
        grid=TimeGrid(dt=1.0, n_steps=14), x_query=np.array([1.0, 2.0, 3.0]),
        noise="none",
    )
    apex = 1.0 * 1.0 * 4.0 / 2.0  # = 2.0
    high = ThreeDetectorScenario(name="m3", meter_cap=3.0, **base)
    at_apex = ThreeDetectorScenario(name="m2", meter_cap=apex, **base)
    _, up_h, dn_h = high.truth_boundary_curves()
    _, up_a, dn_a = at_apex.truth_boundary_curves()
    np.testing.assert_allclose(up_h, up_a, atol=1e-12)
    np.testing.assert_allclose(dn_h, dn_a, atol=1e-12)
    assert high.content_hash() != at_apex.content_hash()


def test_poisson_lam_bound_is_rejected():
    """Review MINOR: a constructor-accepted extreme card crashed numpy's Poisson
    sampler ('lam value too large'); per-step means above 1e15 are rejected."""
    with pytest.raises(ValueError, match="Poisson"):
        ThreeDetectorScenario(
            name="lam", vf=1.0, w=1.0, kappa=4e16, length=4.0, meter_cap=2e16,
            inflow_breakpoints=np.array([0.0, 8.0]), inflow_rates=np.array([1e16]),
            grid=TimeGrid(dt=1.0, n_steps=14), x_query=np.array([2.0]),
            noise="poisson",
        )


def test_sustained_sub_eps_dip_is_tolerated_by_max_drop_gate():
    """Review MINOR: the C3 budget summed dips over cells, duration-amplifying a
    sustained sub-eps dip into a censor. The gate is now the max total drop from
    the high-water mark (the adr-022 convention): a long dip of sub-eps DEPTH
    passes; a single drop beyond eps censors."""
    sc = newell_free_flow_scenario()
    ev = ThreeDetectorEvaluator(sc)
    ref = sc.reference_field()
    eps = ev._eps_count
    dip = ref.copy()
    dip[:, 4:] -= 0.5 * eps  # sustained sub-eps-depth dip over many cells
    dip = np.maximum(dip, 0.0)
    field = ThreeDetectorField(sc.content_hash(), sc.x_query, sc.grid.edges, dip)
    assert ev.evaluate(field)["feasible"] == 1.0
    deep = ref.copy()
    deep[:, 5] = deep[:, 4] - 3.0 * eps  # one drop beyond the budget
    field2 = ThreeDetectorField(sc.content_hash(), sc.x_query, sc.grid.edges, deep)
    assert ev.evaluate(field2)["feasible"] == 0.0


def test_does_not_move_the_golden_braess_hash():
    from tabench.data.builtin import braess_scenario

    assert (
        braess_scenario().content_hash()
        == "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"
    )
