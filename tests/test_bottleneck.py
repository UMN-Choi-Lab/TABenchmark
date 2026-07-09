"""Vickrey (1969) single-bottleneck departure-time equilibrium — anchors + P1.

Closed-form UE/SO hand-derived and machine-verified (adr-019). The certifier
recomputes the queue + generalized costs from the EMITTED departure curve, so the
equilibrium gap is 0 for the UE and positive for any non-equilibrium schedule.
"""

import numpy as np
import pytest

from tabench.bottleneck import (
    BottleneckScenario,
    BottleneckSchedule,
    so_closed_form,
    ue_closed_form,
    vickrey_worked_scenario,
)
from tabench.metrics import BottleneckEvaluator


def test_analytic_equilibrium_cost() -> None:
    sc = vickrey_worked_scenario()  # N=6000, s=3000, a=1, b=0.5, g=2
    # C* = beta*gamma/(beta+gamma) * N/s = (0.5*2/2.5)*2 = 0.8
    assert sc.equilibrium_cost == pytest.approx(0.8)


def test_ue_certifies_zero_gap_and_recomputed_totals() -> None:
    sc = vickrey_worked_scenario()
    metrics = BottleneckEvaluator(sc).certify(ue_closed_form(sc))
    assert metrics["feasible"] == 1.0
    assert metrics["equilibrium_gap"] == pytest.approx(0.0, abs=1e-6)  # every time equal cost
    assert metrics["total_cost"] == pytest.approx(4800.0, rel=1e-4)  # C* * N
    assert metrics["expected_cost"] == pytest.approx(0.8, rel=1e-4)
    assert metrics["max_queue"] == pytest.approx(2400.0, rel=1e-4)  # s*C*/alpha
    # total queueing delay = C*N/(2 alpha) = 2400 person-hours
    assert metrics["total_travel_delay"] == pytest.approx(2400.0, rel=1e-3)


def test_ue_window_and_rates() -> None:
    sc = vickrey_worked_scenario()
    p = ue_closed_form(sc).provenance
    assert p["t1"] == pytest.approx(7.4)  # t* - C*/beta = 9 - 0.8/0.5
    assert p["t2"] == pytest.approx(9.4)  # t* + C*/gamma = 9 + 0.8/2
    assert p["t_n"] == pytest.approx(8.2)  # t* - C*/alpha = 9 - 0.8
    assert p["r_early"] == pytest.approx(6000.0)  # s*a/(a-b) = 3000/0.5
    assert p["r_late"] == pytest.approx(1000.0)  # s*a/(a+g) = 3000/3


def test_so_is_feasible_but_not_an_equilibrium() -> None:
    sc = vickrey_worked_scenario()
    metrics = BottleneckEvaluator(sc).certify(so_closed_form(sc))
    assert metrics["feasible"] == 1.0
    assert metrics["max_queue"] == pytest.approx(0.0, abs=1e-6)  # metered at capacity, no queue
    assert metrics["total_cost"] == pytest.approx(2400.0, rel=1e-4)  # half the UE
    # SO schedules everyone at capacity, so early/on-time/late travelers face very
    # different costs -> it is NOT a departure-time equilibrium.
    assert metrics["equilibrium_gap"] > 0.5


def test_price_of_anarchy_is_two_for_any_penalties() -> None:
    """UE total / SO total = 2 exactly, independent of beta, gamma (a general
    bottleneck result, not just the symmetric case)."""
    rng = np.random.default_rng(19690101)
    for _ in range(50):
        alpha = 1.0
        beta = float(rng.uniform(0.05, 0.95))  # 0 < beta < alpha
        gamma = float(rng.uniform(0.1, 5.0))
        sc = BottleneckScenario(
            name="poa", n_travelers=float(rng.uniform(1000, 9000)),
            capacity=float(rng.uniform(1000, 5000)), alpha=alpha, beta=beta, gamma=gamma,
            t_star=float(rng.uniform(6.0, 12.0)),
        )
        ev = BottleneckEvaluator(sc)
        ue = ev.certify(ue_closed_form(sc))
        so = ev.certify(so_closed_form(sc))
        assert ue["total_cost"] / so["total_cost"] == pytest.approx(2.0, rel=1e-3)


def test_perturbed_schedule_has_positive_gap() -> None:
    """Shifting some departures off the equilibrium curve breaks the equal-cost
    condition -> a positive certified gap (the certifier is not fooled by a
    near-UE schedule)."""
    sc = vickrey_worked_scenario()
    ue = ue_closed_form(sc)
    cum = ue.cumulative.copy()
    # move a slug of departures earlier (bump the mid cumulative up, renormalize)
    mid = cum.shape[0] // 2
    cum[mid : mid + 20] = np.minimum(cum[mid : mid + 20] + 300.0, sc.n_travelers)
    cum = np.maximum.accumulate(cum)  # keep monotone
    cum[-1] = sc.n_travelers
    bad = BottleneckSchedule(scenario_hash=sc.content_hash(), times=ue.times, cumulative=cum)
    metrics = BottleneckEvaluator(sc).certify(bad)
    assert metrics["feasible"] == 1.0
    assert metrics["equilibrium_gap"] > 1e-3


def test_burst_dump_is_not_a_false_equilibrium() -> None:
    """A schedule that dumps travelers in a few large bursts at the equilibrium
    window boundaries must NOT certify as an equilibrium: mid-burst travelers pay
    costs a start-of-step sample would miss. Regression for the adversarial-review
    CRITICAL false-accept (the certifier now scores per traveler by inverting both
    the arrival and the bottleneck-served curves)."""
    sc = vickrey_worked_scenario()
    ev = BottleneckEvaluator(sc)
    cstar = sc.equilibrium_cost
    t1, t2 = sc.t_star - cstar / sc.beta, sc.t_star + cstar / sc.gamma
    burst = BottleneckSchedule(
        scenario_hash=sc.content_hash(),
        times=np.array([t1, t1 + 1e-6, t2, t2 + 1e-6]),
        cumulative=np.array([0.0, sc.n_travelers / 2, sc.n_travelers / 2, sc.n_travelers]),
    )
    mb = ev.certify(burst)
    assert mb["feasible"] == 1.0
    assert mb["equilibrium_gap"] > 1.0  # grossly non-equilibrium
    assert mb["total_cost"] > 4800.0  # worse than the true UE optimum
    # dump everyone at a single instant -> also a large gap
    dump = BottleneckSchedule(
        scenario_hash=sc.content_hash(),
        times=np.array([8.9, 9.1]),
        cumulative=np.array([0.0, sc.n_travelers]),
    )
    assert ev.certify(dump)["equilibrium_gap"] > 1.0


def test_fine_grid_and_small_n_not_false_censored() -> None:
    """The UE at a very fine grid, and a tiny-N scenario, must still certify —
    regression for the fixed-epsilon 'used' mask that censored them ('no
    departures')."""
    sc = vickrey_worked_scenario()
    fine = BottleneckEvaluator(sc).certify(ue_closed_form(sc, n_steps=200000))
    assert fine["feasible"] == 1.0
    assert fine["equilibrium_gap"] == pytest.approx(0.0, abs=1e-6)
    small = BottleneckScenario(
        name="tiny", n_travelers=0.001, capacity=3000.0, alpha=1.0, beta=0.5, gamma=2.0, t_star=9.0
    )
    assert BottleneckEvaluator(small).certify(so_closed_form(small))["feasible"] == 1.0


def test_certifier_censors_nonconserving_schedule() -> None:
    sc = vickrey_worked_scenario()
    ue = ue_closed_form(sc)
    short = BottleneckSchedule(
        scenario_hash=sc.content_hash(), times=ue.times, cumulative=ue.cumulative * 0.5
    )
    m = BottleneckEvaluator(sc).certify(short)
    assert m["feasible"] == 0.0
    assert np.isnan(m["equilibrium_gap"])


def test_certifier_censors_nonmonotone_and_wrong_hash() -> None:
    sc = vickrey_worked_scenario()
    ue = ue_closed_form(sc)
    nonmono = ue.cumulative.copy()
    nonmono[ue.cumulative.shape[0] // 2] = nonmono[ue.cumulative.shape[0] // 2 - 1] - 100.0
    m1 = BottleneckEvaluator(sc).certify(
        BottleneckSchedule(scenario_hash=sc.content_hash(), times=ue.times, cumulative=nonmono)
    )
    assert m1["feasible"] == 0.0
    m2 = BottleneckEvaluator(sc).certify(
        BottleneckSchedule(scenario_hash="wrong", times=ue.times, cumulative=ue.cumulative)
    )
    assert m2["feasible"] == 0.0


def test_scenario_validation() -> None:
    base = dict(
        name="v", n_travelers=6000.0, capacity=3000.0, alpha=1.0, beta=0.5, gamma=2.0, t_star=9.0
    )
    BottleneckScenario(**base)  # ok
    with pytest.raises(ValueError, match="beta < alpha"):
        BottleneckScenario(**{**base, "beta": 1.5})  # beta > alpha
    with pytest.raises(ValueError, match="gamma"):
        BottleneckScenario(**{**base, "gamma": 0.0})
    with pytest.raises(ValueError, match="n_travelers"):
        BottleneckScenario(**{**base, "n_travelers": 0.0})
    with pytest.raises(ValueError, match="capacity"):
        BottleneckScenario(**{**base, "capacity": -1.0})


def test_content_hash_stable_and_param_sensitive() -> None:
    sc = vickrey_worked_scenario()
    assert sc.content_hash() == sc.content_hash()  # stable
    # provenance (name/family) is unhashed; a scored-parameter change moves it.
    renamed = BottleneckScenario(
        name="renamed", n_travelers=6000.0, capacity=3000.0, alpha=1.0, beta=0.5, gamma=2.0,
        t_star=9.0,
    )
    assert sc.content_hash() == renamed.content_hash()
    bumped = BottleneckScenario(
        name="v2", n_travelers=6001.0, capacity=3000.0, alpha=1.0, beta=0.5, gamma=2.0, t_star=9.0
    )
    assert sc.content_hash() != bumped.content_hash()


def test_does_not_move_the_golden_braess_hash() -> None:
    from tabench.data.builtin import braess_scenario

    assert (
        braess_scenario().content_hash()
        == "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"
    )
