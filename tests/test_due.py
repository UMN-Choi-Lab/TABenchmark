"""Friesz et al. (1993) SRDC dynamic user equilibrium — anchors + P1 (adr-022).

The two-route anchor is hand-solved (each used route runs its own Vickrey
equilibrium at the common cost level C = (delta*N + alpha*sum s_r f_r)/sum s_r)
and machine-verified here: C = 0.9, split (5250, 750), total 5400, both-used
threshold N > 3750. The single-route f = 0 reduction IS the shipped vickrey
model — pinned by a cross-certifier check. The route axis's new failure mode
(all-on-one-route equalizes its own costs while the idle route is cheaper) is
caught by the marginal-insertion reference scan.
"""

import numpy as np
import pytest

from tabench.bottleneck import (
    BottleneckSchedule,
    DUEProfile,
    DUEScenario,
    due_closed_form,
    friesz_two_route_scenario,
    vickrey_worked_scenario,
)
from tabench.bottleneck.solve import ue_closed_form
from tabench.metrics import BottleneckEvaluator, DUEEvaluator


def _scenario(**overrides) -> DUEScenario:
    base = dict(
        name="due",
        n_travelers=6000.0,
        alpha=1.0,
        beta=0.5,
        gamma=2.0,
        t_star=9.0,
        route_free_flow=[0.2, 0.7],
        route_capacity=[3000.0, 1500.0],
    )
    base.update(overrides)
    return DUEScenario(**base)


# ---------------------------------------------------------------- anchors


def test_two_route_closed_form_structure() -> None:
    sc = friesz_two_route_scenario()
    c, used, n_r = sc.equilibrium_structure()
    assert c == pytest.approx(0.9, abs=1e-12)
    assert list(used) == [True, True]
    np.testing.assert_allclose(n_r, [5250.0, 750.0], atol=1e-9)


def test_two_route_due_certifies_zero_gap() -> None:
    sc = friesz_two_route_scenario()
    metrics = DUEEvaluator(sc).certify(due_closed_form(sc))
    assert metrics["feasible"] == 1.0
    assert metrics["due_gap"] == pytest.approx(0.0, abs=1e-6)
    # round-2 MINOR pin: boundary travelers are sampled at eps/n_r - eps, so
    # an equilibrium's gap cannot go meaningfully negative
    assert metrics["due_gap"] >= -1e-9
    assert metrics["total_cost"] == pytest.approx(5400.0, rel=1e-4)  # C * N
    assert metrics["expected_cost"] == pytest.approx(0.9, rel=1e-4)
    # route 1 queue peaks at s1 * Cq1 / alpha = 3000 * 0.7 = 2100
    assert metrics["max_queue"] == pytest.approx(2100.0, rel=1e-3)


def test_both_used_threshold() -> None:
    # Both routes used iff N > alpha*s1*(f2 - f1)/delta = 3750; at N = 3000
    # only route 1 runs (C = 0.6 < alpha*f2 = 0.7) and it still certifies.
    sc = _scenario(n_travelers=3000.0)
    c, used, n_r = sc.equilibrium_structure()
    assert c == pytest.approx(0.6, abs=1e-12)
    assert list(used) == [True, False]
    np.testing.assert_allclose(n_r, [3000.0, 0.0], atol=1e-9)
    metrics = DUEEvaluator(sc).certify(due_closed_form(sc))
    assert metrics["feasible"] == 1.0
    assert metrics["due_gap"] == pytest.approx(0.0, abs=1e-6)


def test_equal_free_flow_splits_by_capacity() -> None:
    # f = (0, 0): C = delta*N/(s1+s2) and the split is capacity-proportional.
    sc = _scenario(route_free_flow=[0.0, 0.0])
    c, _, n_r = sc.equilibrium_structure()
    assert c == pytest.approx(0.4 * 6000.0 / 4500.0, abs=1e-12)
    np.testing.assert_allclose(n_r, [4000.0, 2000.0], atol=1e-9)
    metrics = DUEEvaluator(sc).certify(due_closed_form(sc))
    assert metrics["feasible"] == 1.0
    assert metrics["due_gap"] == pytest.approx(0.0, abs=1e-6)


def test_single_route_reduction_is_the_vickrey_model() -> None:
    # One route with f = 0 IS the shipped vickrey model: same C*, and the DUE
    # profile certifies under the repo's own BottleneckEvaluator.
    sc = _scenario(route_free_flow=[0.0], route_capacity=[3000.0])
    vick = vickrey_worked_scenario()
    assert sc.equilibrium_cost() == pytest.approx(vick.equilibrium_cost, abs=1e-12)
    profile = due_closed_form(sc)
    metrics = DUEEvaluator(sc).certify(profile)
    assert metrics["feasible"] == 1.0
    assert metrics["due_gap"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["total_cost"] == pytest.approx(4800.0, rel=1e-4)
    # cross-certifier: identical emitted curve, certified by adr-019 machinery
    schedule = BottleneckSchedule(
        vick.content_hash(), profile.times, profile.cumulative[0]
    )
    cross = BottleneckEvaluator(vick).certify(schedule)
    assert cross["feasible"] == 1.0
    assert cross["equilibrium_gap"] == pytest.approx(0.0, abs=1e-6)
    # and the curves match the repo closed form pointwise
    repo = ue_closed_form(vick)
    mine = np.interp(repo.times, profile.times, profile.cumulative[0])
    np.testing.assert_allclose(mine, repo.cumulative, atol=1e-6)


def test_all_on_one_route_is_not_an_equilibrium() -> None:
    # The route axis's OWN failure mode: all 6000 on route 1 in its
    # single-route Vickrey equilibrium equalizes used costs at 1.0, but the
    # idle route 2's marginal insertion costs alpha*f2 = 0.7 — the reference
    # scan must catch it: due_gap = (1.0 - 0.7)/0.9 = 1/3.
    sc = friesz_two_route_scenario()
    donor = due_closed_form(
        _scenario(route_free_flow=[0.2], route_capacity=[3000.0])
    )
    profile = DUEProfile(
        scenario_hash=sc.content_hash(),
        times=donor.times,
        cumulative=np.vstack([donor.cumulative[0], np.zeros(donor.times.size)]),
    )
    metrics = DUEEvaluator(sc).certify(profile)
    assert metrics["feasible"] == 1.0
    assert metrics["due_gap"] == pytest.approx(1.0 / 3.0, abs=1e-3)


def test_burst_dump_is_not_a_false_equilibrium() -> None:
    # adr-019 regression transplanted: dump all mass at one instant per route —
    # per-traveler level inversion must see the intra-burst queue costs.
    sc = friesz_two_route_scenario()
    times = np.linspace(7.0, 10.0, 601)
    cum = np.zeros((2, times.size))
    cum[0] = np.where(times >= 8.5, 5250.0, 0.0)
    cum[1] = np.where(times >= 8.5, 750.0, 0.0)
    metrics = DUEEvaluator(sc).certify(
        DUEProfile(scenario_hash=sc.content_hash(), times=times, cumulative=cum)
    )
    assert metrics["feasible"] == 1.0
    assert metrics["due_gap"] > 0.5


def test_metered_so_style_profile_scores_positive_gap() -> None:
    # Uniform metering at capacity (the SO idea) is NOT a departure-time
    # equilibrium here either.
    sc = friesz_two_route_scenario()
    times = np.linspace(7.0, 10.0, 1201)
    cum = np.zeros((2, times.size))
    cum[0] = np.clip(3000.0 * (times - 7.35), 0.0, 5250.0)
    cum[1] = np.clip(1500.0 * (times - 8.05), 0.0, 750.0)
    metrics = DUEEvaluator(sc).certify(
        DUEProfile(scenario_hash=sc.content_hash(), times=times, cumulative=cum)
    )
    assert metrics["feasible"] == 1.0
    assert metrics["due_gap"] > 0.1


# ------------------------------------------------- adr-022 review regressions


def test_due_gap_is_invariant_to_the_emitted_grid_horizon() -> None:
    # Review MAJOR: the old 65-point interpolated extension bent the served
    # curve's clearing kink, so the SAME departure plan scored up to 25.6x
    # differently depending on where its emitted grid happened to end. The
    # exact served-curve construction makes the score a pure function of the
    # piecewise-linear plan: a residual-queue horizon and a long flat tail
    # (whose coarse interior segment contains the clearing kink) must agree.
    sc = friesz_two_route_scenario()
    times = np.linspace(7.0, 9.0, 401)
    cum = np.zeros((2, times.size))
    cum[0] = np.clip(5250.0 * (times - 8.4) / 0.2, 0.0, 5250.0)
    cum[1] = np.clip(750.0 * (times - 8.4) / 0.2, 0.0, 750.0)
    short = DUEProfile(scenario_hash=sc.content_hash(), times=times, cumulative=cum)
    long_t = np.concatenate([times, [12.0, 20.0]])
    long_c = np.concatenate([cum, cum[:, -1:], cum[:, -1:]], axis=1)
    padded = DUEProfile(scenario_hash=sc.content_hash(), times=long_t, cumulative=long_c)
    ev = DUEEvaluator(sc)
    m_short, m_long = ev.certify(short), ev.certify(padded)
    assert m_short["feasible"] == 1.0 and m_long["feasible"] == 1.0
    assert m_short["due_gap"] > 0.5  # a burst is far from equilibrium
    assert m_long["due_gap"] == pytest.approx(m_short["due_gap"], rel=1e-9)
    assert m_long["total_cost"] == pytest.approx(m_short["total_cost"], rel=1e-9)
    assert m_long["max_queue"] == pytest.approx(m_short["max_queue"], rel=1e-9)


def test_greedy_survives_queue_term_underflow() -> None:
    # Review MINOR: with alpha*f_1 so large that the queue term delta*N/s_1
    # rounds away, c_1 <= alpha*f_1 in float64 and the old greedy broke at
    # k = 1 leaving C = inf. The cheapest route is ALWAYS used.
    sc = _scenario(n_travelers=1e-3, route_free_flow=[1e9], route_capacity=[1e6])
    c, used, n_r = sc.equilibrium_structure()
    assert np.isfinite(c)
    assert c == pytest.approx(1e9, rel=1e-12)
    assert list(used) == [True]
    assert n_r.sum() == pytest.approx(1e-3, rel=1e-12)


def test_split_is_cancellation_free_at_large_free_flow_times() -> None:
    # Review MINOR: N_r = s*(C - alpha*f)/delta cancels catastrophically when
    # alpha*f >> delta*N. The difference form + renormalization keep the split
    # exact under a common 1e10 shift of both free-flow times (the anchor's
    # split depends only on f_2 - f_1, so it must remain (5250, 750)).
    sc = _scenario(route_free_flow=[1e10 + 0.2, 1e10 + 0.7])
    _, used, n_r = sc.equilibrium_structure()
    assert list(used) == [True, True]
    # float64 stores 1e10+0.2 only to ~2e-6, so the *instance* itself is
    # perturbed at the ~0.02-traveler level — but the split must still sum
    # to N exactly (the old form missed by more than the conservation eps)
    np.testing.assert_allclose(n_r, [5250.0, 750.0], atol=0.05)
    assert n_r.sum() == pytest.approx(6000.0, abs=1e-9)
    metrics = DUEEvaluator(sc).certify(due_closed_form(sc))
    assert metrics["feasible"] == 1.0  # the conservation gate must not fire
    assert metrics["due_gap"] == pytest.approx(0.0, abs=1e-6)


def test_degenerate_parameters_are_rejected() -> None:
    # Review MINOR: documented domain bounds — beta ~= alpha degenerates the
    # early-departure rate s*alpha/(alpha-beta); gamma/beta > 1e9 degenerates
    # the schedule-delay ratio.
    with pytest.raises(ValueError, match="alpha - beta"):
        _scenario(beta=1.0 - 1e-12)
    with pytest.raises(ValueError, match="gamma"):
        _scenario(beta=0.5, gamma=0.5 * 2e9)


def test_creeping_retraction_is_censored() -> None:
    # Review NOTE: a per-STEP nondecreasing gate lets many sub-eps retractions
    # accumulate into a real rollback (the DTA eps-accumulation family). The
    # running-max gate censors the TOTAL drop from the high-water mark.
    sc = friesz_two_route_scenario()
    profile = due_closed_form(sc)
    t, cum = profile.times, profile.cumulative.copy()
    eps = 1e-6 * 6000.0
    j = int(np.searchsorted(t, 9.16))  # past both departure windows
    assert t.size - j > 32, "closed-form grid must reach past the windows"
    for i in range(1, 31):  # each step drops 0.9*eps; total 27*eps
        cum[0, j + i] = cum[0, j] - 0.9 * eps * i
    # recover to the original level afterwards: conservation still holds
    metrics = DUEEvaluator(sc).certify(
        DUEProfile(scenario_hash=sc.content_hash(), times=t, cumulative=cum)
    )
    assert abs(cum[:, -1].sum() - 6000.0) < 1e-9  # NOT the conservation gate
    assert metrics["feasible"] == 0.0


def test_hidden_queue_drain_dip_is_found_exactly() -> None:
    # Round-2 review CRITICAL (both lenses independently): route 1 runs a full
    # Vickrey isocost schedule at cost 2, route 2 the SAME schedule truncated
    # at t=8.35 — every used traveler pays exactly 2 and every profile kink
    # and t*-f_r candidate reads exactly 2, but waiting for route 2's residual
    # queue to drain and departing at 8.75 exits at 9.45 for 1.6. A flat tail
    # at t=1e6 diluted the old hull-spanning sweep to ~250 spacing, so the
    # certifier reported due_gap = 0.0 for a strict non-equilibrium. The
    # queue-vanishing kink (a zero of A(t) - S(t + f_r)) is now an enumerated
    # candidate, so the dip is found EXACTLY at any hull.
    sc = _scenario(
        n_travelers=14325.0,
        route_free_flow=[0.7, 0.7],
        route_capacity=[3000.0, 1500.0],
    )
    t = np.array([5.0, 5.7, 7.0, 8.35, 8.95, 9.5])
    cum = np.vstack(
        [
            [0.0, 0.0, 7800.0, 9150.0, 9750.0, 9750.0],
            [0.0, 0.0, 3900.0, 4575.0, 4575.0, 4575.0],
        ]
    )
    ev = DUEEvaluator(sc)
    m_short = ev.certify(DUEProfile(sc.content_hash(), t, cum))
    t_long = np.concatenate([t, [1e6]])
    cum_long = np.concatenate([cum, cum[:, -1:]], axis=1)
    m_long = ev.certify(DUEProfile(sc.content_hash(), t_long, cum_long))
    expect = (2.0 - 1.6) / sc.equilibrium_cost()  # = 0.4/(8880/4500) ~ 0.2027
    assert m_short["feasible"] == 1.0 and m_long["feasible"] == 1.0
    assert m_short["due_gap"] == pytest.approx(expect, rel=1e-9)
    assert m_long["due_gap"] == pytest.approx(expect, rel=1e-9)


def test_min_ref_is_immune_to_grid_hull_stretching() -> None:
    # Round-2 soundness CRITICAL repro shape: a single 3500/h ramp on f=0,
    # s=3000 has true gap ~0.765; ONE far flat pad point stretched the old
    # sweep window until it certified -0.0001 (~equilibrium). The sweep is now
    # pinned to the analytic window and the kink enumeration is exact.
    sc = _scenario(route_free_flow=[0.0], route_capacity=[3000.0])
    times = np.linspace(51.0 / 7.0, 9.0, 401)
    cum = np.clip(3500.0 * (times - 51.0 / 7.0), 0.0, 6000.0)[None, :]
    ev = DUEEvaluator(sc)
    m0 = ev.certify(DUEProfile(sc.content_hash(), times, cum))
    t_pad = np.concatenate([times, [1e5]])
    c_pad = np.concatenate([cum, cum[:, -1:]], axis=1)
    m1 = ev.certify(DUEProfile(sc.content_hash(), t_pad, c_pad))
    assert m0["due_gap"] > 0.7
    assert m1["due_gap"] == pytest.approx(m0["due_gap"], rel=1e-9)


def test_sub_tolerance_population_is_censored_not_scored() -> None:
    # Round-2 review MAJOR: with N <= tol every route was skipped, max_used
    # stayed -inf, and arbitrary conserving garbage (all mass 500 h late)
    # certified feasible at due_gap = -inf.
    sc = _scenario(n_travelers=1e-7)
    profile = due_closed_form(sc)
    late = DUEProfile(sc.content_hash(), profile.times + 500.0, profile.cumulative)
    for p in (profile, late):
        metrics = DUEEvaluator(sc).certify(p)
        assert metrics["feasible"] == 0.0
        assert np.isnan(metrics["due_gap"])


def test_ill_conditioned_time_scale_is_rejected() -> None:
    # Round-2 review MAJOR: gamma=2.73e8 (inside the gamma/beta bound) with
    # C ~ 5e-6 at t* ~ 9296 — (beta+gamma)*ulp(t*) swamps C, so NO float64
    # profile can score near 0 (the honest closed form certified gap 1.0).
    # The conditioning gate rejects the instance at construction.
    with pytest.raises(ValueError, match="conditioned"):
        _scenario(
            n_travelers=0.02,
            alpha=2.53,
            beta=1.13,
            gamma=2.73e8,
            t_star=9295.84,
            route_free_flow=[0.0, 0.0],
        )


# ---------------------------------------------------------------- certification


def test_wrong_hash_is_censored() -> None:
    sc = friesz_two_route_scenario()
    profile = due_closed_form(sc)
    forged = DUEProfile(
        scenario_hash="not-this-scenario",
        times=profile.times,
        cumulative=profile.cumulative,
    )
    metrics = DUEEvaluator(sc).certify(forged)
    assert metrics["feasible"] == 0.0
    assert np.isnan(metrics["due_gap"])


def test_non_conserving_profile_is_censored() -> None:
    sc = friesz_two_route_scenario()
    profile = due_closed_form(sc)
    cum = profile.cumulative.copy()
    cum[1] *= 0.5  # deliver only half of route 2's volume
    metrics = DUEEvaluator(sc).certify(
        DUEProfile(
            scenario_hash=sc.content_hash(), times=profile.times, cumulative=cum
        )
    )
    assert metrics["feasible"] == 0.0


def test_decreasing_cumulative_is_censored() -> None:
    sc = friesz_two_route_scenario()
    profile = due_closed_form(sc)
    cum = profile.cumulative.copy()
    mid = cum.shape[1] // 2
    cum[0, mid] = cum[0, mid - 1] - 5.0
    metrics = DUEEvaluator(sc).certify(
        DUEProfile(
            scenario_hash=sc.content_hash(), times=profile.times, cumulative=cum
        )
    )
    assert metrics["feasible"] == 0.0


def test_route_count_mismatch_raises() -> None:
    sc = friesz_two_route_scenario()
    single = _scenario(route_free_flow=[0.0], route_capacity=[3000.0])
    with pytest.raises(ValueError, match="routes"):
        DUEEvaluator(sc).certify(due_closed_form(single))


# ---------------------------------------------------------------- scenario


def test_scenario_validation() -> None:
    with pytest.raises(ValueError, match="beta"):
        _scenario(beta=1.5)
    with pytest.raises(ValueError, match="gamma"):
        _scenario(gamma=-1.0)
    with pytest.raises(ValueError, match="n_travelers"):
        _scenario(n_travelers=0.0)
    with pytest.raises(ValueError, match="capacities"):
        _scenario(route_capacity=[3000.0, 0.0])
    with pytest.raises(ValueError, match="free-flow"):
        _scenario(route_free_flow=[0.2, -0.1])
    with pytest.raises(ValueError, match="equal-length"):
        _scenario(route_free_flow=[0.2])


def test_content_hash_separates_instances() -> None:
    a = friesz_two_route_scenario()
    assert a.content_hash() == friesz_two_route_scenario().content_hash()
    assert a.content_hash() != _scenario(n_travelers=6001.0).content_hash()
    assert a.content_hash() != _scenario(route_capacity=[3000.0, 1501.0]).content_hash()
    # and the DUE hash domain is separated from the single-bottleneck domain
    assert a.content_hash() != vickrey_worked_scenario().content_hash()


def test_scenario_arrays_are_read_only() -> None:
    sc = friesz_two_route_scenario()
    with pytest.raises(ValueError, match="read-only"):
        sc.route_capacity[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        sc.route_free_flow[1] = 0.0


def test_does_not_move_the_golden_braess_hash() -> None:
    from tabench.data.builtin import braess_scenario

    assert (
        braess_scenario().content_hash()
        == "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"
    )
