"""Peeta & Mahmassani (1995) time-dependent SO/UE assignment — anchors + P1 (adr-031).

The paper's 50-node DYNASMART numerics are engine-bound and irreproducible (adr-031),
so every anchor is derived from scratch on the repo's own CTM/LTM loading:

* a single-path corridor whose harness-loaded TSTT equals the lp-so-dta LP optimum
  33 EXACTLY through the new per-path loader (the ADR-021 cross-model pin);
* a symmetric two-route diamond whose exact TD-UE (50/50) certifies ``tdue_gap = 0``,
  its all-on-one control scoring the hand gap 0.75;
* an SO != UE wedge where the system optimum's TSTT is STRICTLY below the user
  equilibrium's (the paper's headline), the SO split attaining the LP bound;
* a merge exercising per-commodity attribution under the interior-diverge-free rule.

The certifier — never the MSA solver's claim — is the arbiter (the vi-due lesson).
"""

import numpy as np
import pytest

from tabench.core.scenario import Network
from tabench.dnl import DynamicDemand, DynamicScenario, LinkDynamics, TimeGrid
from tabench.dnl.output import DNLOutput
from tabench.dta import solve_cell_so_dta, zil_corridor_scenario
from tabench.metrics import DNLEvaluator, TDTAEvaluator
from tabench.tdta import (
    PathLoader,
    TDPath,
    TDPathFlows,
    TDTAScenario,
    pm_corridor_scenario,
    pm_diamond_scenario,
    pm_merge_scenario,
    pm_wedge_scenario,
    solve_td_so,
    solve_td_ue,
)


def _net(name, n_nodes, n_zones, init, term):
    ia = np.asarray(init, dtype=np.int64)
    ta = np.asarray(term, dtype=np.int64)
    n = ia.size
    return Network(
        name=name,
        n_nodes=n_nodes,
        n_zones=n_zones,
        first_thru_node=1,
        init_node=ia,
        term_node=ta,
        capacity=np.ones(n),
        length=np.zeros(n),
        free_flow_time=np.ones(n),
        b=np.zeros(n),
        power=np.ones(n),
        toll=np.zeros(n),
        link_type=np.ones(n, dtype=np.int64),
    )


def _burst(sc: TDTAScenario, split: dict[int, float]) -> TDPathFlows:
    """A first-interval-burst emission: ``split[p]`` vehicles on path ``p`` in step 0."""
    dep = np.zeros((sc.n_paths, sc.grid.n_steps))
    for p, v in split.items():
        dep[p, 0] = v
    return TDPathFlows(sc.content_hash(), dep)


# ------------------------------------------------------------------ anchor C: corridor


def test_corridor_single_path_pins_lp_optimum_33() -> None:
    sc = pm_corridor_scenario()
    ev = TDTAEvaluator(sc)
    # a single path forces the split, so tdue_gap is trivially 0 and the loading
    # TSTT must equal the LP optimum (33) exactly through the NEW code path.
    m = ev.certify(_burst(sc, {0: 6.0}))
    assert m["feasible"] == 1.0
    assert m["tdue_gap"] == pytest.approx(0.0, abs=1e-9)
    assert m["tstt"] == pytest.approx(33.0, abs=1e-9)
    assert m["z_star"] == pytest.approx(33.0, abs=1e-9)
    assert m["so_bound_gap"] == pytest.approx(0.0, abs=1e-9)


def test_corridor_derived_cell_scenario_equals_zil_corridor() -> None:
    # The CTM-cell instance derived from the corridor is byte-for-byte the ADR-021
    # zil_corridor scenario, so its LP optimum is the same 33 veh-intervals.
    sc = pm_corridor_scenario()
    cell = sc.derive_cell_scenario()
    assert cell.content_hash() == zil_corridor_scenario().content_hash()
    assert solve_cell_so_dta(cell).provenance["objective"] == pytest.approx(33.0, abs=1e-8)


# ------------------------------------------------------------------ anchor A: diamond


def test_diamond_symmetric_ue_is_exact_zero() -> None:
    # By symmetry the exact TD-UE is the 50/50 split (equal experienced times).
    sc = pm_diamond_scenario("ctm")
    m = TDTAEvaluator(sc).certify(_burst(sc, {0: 2.0, 1: 2.0}))
    assert m["feasible"] == 1.0
    assert m["tdue_gap"] == pytest.approx(0.0, abs=1e-9)
    assert m["tdue_gap_max"] == pytest.approx(0.0, abs=1e-9)
    assert m["tstt"] == pytest.approx(14.0, abs=1e-9)


def test_diamond_all_on_one_route_scores_hand_gap() -> None:
    # Negative control + the F5 hidden-cheap-path family: routing all 4 vehicles
    # onto route A queues its bottleneck while the identical route B sits idle,
    # so the reference minimum (which scans EVERY declared path) exposes the swap.
    sc = pm_diamond_scenario("ctm")
    m = TDTAEvaluator(sc).certify(_burst(sc, {0: 4.0}))
    assert m["feasible"] == 1.0
    assert m["tdue_gap"] == pytest.approx(0.75, abs=1e-6)
    assert m["tdue_gap_max"] == pytest.approx(1.5, abs=1e-3)
    assert m["so_bound_gap"] == pytest.approx(2.0 / 7.0, abs=1e-6)  # (18-14)/14


def test_diamond_ltm_equals_ctm() -> None:
    # Cross-model pin (ADR-016 ltm==ctm): the two kernels agree on the aligned
    # diamond to machine precision. They are two hashed scenario variants.
    dep = {0: 2.0, 1: 2.0}
    ctm, ltm = pm_diamond_scenario("ctm"), pm_diamond_scenario("ltm")
    m_ctm = TDTAEvaluator(ctm).certify(_burst(ctm, dep))
    m_ltm = TDTAEvaluator(ltm).certify(_burst(ltm, dep))
    for key in ("tdue_gap", "tstt", "total_experienced_time", "max_experienced_time"):
        assert m_ctm[key] == pytest.approx(m_ltm[key], abs=1e-9)
    assert pm_diamond_scenario("ctm").content_hash() != pm_diamond_scenario("ltm").content_hash()


# ------------------------------------------------------------------ anchor B: SO != UE wedge


def test_wedge_so_split_attains_lp_bound() -> None:
    # An explicit SO split (1 vehicle on the fast route, 5 on the slow) whose
    # harness loading achieves the LP optimum 23 exactly — attainability, and the
    # SO twin's certificate.
    sc = pm_wedge_scenario()
    ev = TDTAEvaluator(sc)
    assert ev._z_star_time == pytest.approx(23.0, abs=1e-9)
    m = ev.certify(_burst(sc, {0: 1.0, 1: 5.0}))
    assert m["feasible"] == 1.0
    assert m["tstt"] == pytest.approx(23.0, abs=1e-9)
    assert m["so_bound_gap"] == pytest.approx(0.0, abs=1e-9)
    assert m["tdue_gap"] > 0.05  # the SO plan is NOT a user equilibrium


def test_wedge_so_strictly_beats_ue() -> None:
    # The paper's headline made executable: the system optimum's TSTT is strictly
    # below the user equilibrium's on the same loading.
    sc = pm_wedge_scenario()
    ev = TDTAEvaluator(sc)
    so = ev.certify(_burst(sc, {0: 1.0, 1: 5.0}))  # SO-optimal split
    ue = ev.certify(_burst(sc, {0: 3.0, 1: 3.0}))  # a user-equilibrium split
    assert so["tstt"] == pytest.approx(23.0, abs=1e-9)
    assert ue["tstt"] == pytest.approx(24.0, abs=1e-9)
    assert so["tstt"] < ue["tstt"] - 0.5
    # each split is positive under the OTHER's metric — the two equilibria differ
    assert ue["so_bound_gap"] == pytest.approx(1.0 / 23.0, abs=1e-6)
    assert so["tdue_gap"] > 0.05


def test_wedge_all_fast_is_worst() -> None:
    # Selfish myopia (all 6 on the fast route) is the worst plan: TSTT 33.
    sc = pm_wedge_scenario()
    m = TDTAEvaluator(sc).certify(_burst(sc, {0: 6.0}))
    assert m["tstt"] == pytest.approx(33.0, abs=1e-9)
    assert m["so_bound_gap"] == pytest.approx(10.0 / 23.0, abs=1e-6)


# ------------------------------------------------------------------ anchor D: merge


def test_merge_attribution_certifies_clean() -> None:
    # Two origins feed a shared bottleneck; per-commodity experienced times stay
    # decidable at the merge (each in-link's outflow observed + FIFO). A balanced
    # split certifies cleanly and attains the SO bound.
    sc = pm_merge_scenario()
    m = TDTAEvaluator(sc).certify(_burst(sc, {0: 1.5, 1: 1.5, 2: 2.0}))
    assert m["feasible"] == 1.0
    assert m["so_bound_gap"] == pytest.approx(0.0, abs=1e-9)
    assert m["tstt"] == pytest.approx(m["z_star"], abs=1e-9)


@pytest.mark.parametrize(
    "make,split,pad",
    [
        (pm_corridor_scenario, {0: 6.0}, 4),
        (pm_merge_scenario, {0: 1.5, 1: 1.5, 2: 2.0}, 14),
        (pm_wedge_scenario, {0: 3.0, 1: 3.0}, 15),
    ],
)
def test_loader_aggregate_passes_dnl_c0_c8(make, split, pad) -> None:
    # The per-path loader's AGGREGATE emission is a valid DNL output — a free,
    # powerful oracle: it passes the shipped dnl_gaps C0-C8 certificate.
    sc = make()
    dep = np.zeros((sc.n_paths, sc.grid.n_steps))
    for p, v in split.items():
        dep[p, 0] = v
    out = PathLoader(sc, dep, extra_steps=pad).run()
    dsc = DynamicScenario(
        name=sc.name + "-dnl",
        network=sc.network,
        dynamics=sc.dynamics,
        demand=sc.demand,
        grid=out.grid,
        turns=None,
    )
    o2 = DNLOutput(
        scenario_hash=dsc.content_hash(),
        grid=out.grid,
        n_in=out.n_in,
        n_out=out.n_out,
        origin_release=out.origin_release,
    )
    assert DNLEvaluator(dsc).evaluate(o2)["dnl_feasible"] == 1.0


# ------------------------------------------------------------------ MSA solvers


def test_ue_msa_converges_on_diamond() -> None:
    # The paper's experienced-time MSA reaches the exact symmetric UE; the
    # certified best iterate has gap 0 and the certifier (not the solver) arbitrates.
    sc = pm_diamond_scenario("ctm")
    flows = solve_td_ue(sc, iters=25)
    assert flows.provenance["best_gap"] == pytest.approx(0.0, abs=1e-6)
    traj = np.array(flows.provenance["trajectory"])
    # the even (pre-averaging) iterates form a monotone-decreasing envelope
    envelope = traj[0::2]
    assert np.all(np.diff(envelope) <= 1e-9)
    assert TDTAEvaluator(sc).certify(flows)["tdue_gap"] == pytest.approx(0.0, abs=1e-6)


def test_so_msa_attains_wedge_bound() -> None:
    # The marginal-cost MSA (local link marginals, the 3-point quadratic fit)
    # drives the wedge to a split that attains the LP bound.
    sc = pm_wedge_scenario()
    flows = solve_td_so(sc, iters=30)
    m = TDTAEvaluator(sc).certify(flows)
    assert m["so_bound_gap"] == pytest.approx(0.0, abs=1e-6)
    assert m["tstt"] == pytest.approx(23.0, abs=1e-6)


# ------------------------------------------------------------------ certification / censor


def test_wrong_hash_is_censored() -> None:
    sc = pm_diamond_scenario("ctm")
    dep = np.zeros((sc.n_paths, sc.grid.n_steps))
    dep[0, 0] = 2.0
    dep[1, 0] = 2.0
    m = TDTAEvaluator(sc).certify(TDPathFlows("not-this-scenario", dep))
    assert m["feasible"] == 0.0
    assert np.isnan(m["tdue_gap"])


def test_nan_departures_are_censored() -> None:
    sc = pm_diamond_scenario("ctm")
    dep = np.zeros((sc.n_paths, sc.grid.n_steps))
    dep[0, 0] = np.nan
    assert TDTAEvaluator(sc).certify(TDPathFlows(sc.content_hash(), dep))["feasible"] == 0.0


def test_negative_departures_are_censored() -> None:
    sc = pm_diamond_scenario("ctm")
    dep = np.zeros((sc.n_paths, sc.grid.n_steps))
    dep[0, 0] = 2.5
    dep[0, 1] = -0.5  # a real retraction of the cumulative curve
    dep[1, 0] = 2.0
    assert TDTAEvaluator(sc).certify(TDPathFlows(sc.content_hash(), dep))["feasible"] == 0.0


def test_departure_time_shift_is_censored() -> None:
    # F2/F4: fixed departure times give the model zero timing freedom. Moving
    # route B's mass from step 0 to step 1 (same total) violates the per-edge
    # demand-match gate and is censored — no within/across-interval gaming.
    sc = pm_diamond_scenario("ctm")
    dep = np.zeros((sc.n_paths, sc.grid.n_steps))
    dep[0, 0] = 2.0
    dep[1, 1] = 2.0  # should be in step 0
    assert TDTAEvaluator(sc).certify(TDPathFlows(sc.content_hash(), dep))["feasible"] == 0.0


def test_retraction_accumulation_is_censored() -> None:
    # The DTA eps-accumulation family: many sub-eps negative departures that each
    # pass a per-cell tolerance but sum past the aggregate mass budget.
    sc = pm_diamond_scenario("ctm")
    dep = np.zeros((sc.n_paths, sc.grid.n_steps))
    dep[0, 0] = 2.0 + 1e-6 * 12
    dep[0, 1:13] = -1e-6
    dep[1, 0] = 2.0
    assert TDTAEvaluator(sc).certify(TDPathFlows(sc.content_hash(), dep))["feasible"] == 0.0


def _short_multidest() -> TDTAScenario:
    # A short-horizon, TWO-destination instance (so no SO cell LP is derived and
    # construction never eager-clears): each OD's bottleneck needs the extension
    # to drain within the horizon.
    rates = np.zeros((1, 3, 3))
    rates[0, 0, 1] = 3.0  # 1 -> 2
    rates[0, 0, 2] = 3.0  # 1 -> 3
    return TDTAScenario(
        name="pm-short-md",
        network=_net("smd", 5, 3, [1, 4, 1, 5], [4, 2, 5, 3]),
        dynamics=LinkDynamics(
            length=np.ones(4),
            free_speed=np.ones(4),
            wave_speed=np.ones(4),
            jam_density=np.full(4, 20.0),
            capacity=np.array([10.0, 1.0, 10.0, 1.0]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 1.0]), rates=rates),
        grid=TimeGrid(1.0, 3),
        paths=(TDPath(1, 2, (0, 1)), TDPath(1, 3, (2, 3))),
        kernel="ctm",
    )


def test_stranded_flow_is_censored() -> None:
    # F3: assigned flow that does not clear the extended horizon is censored (the
    # clearing gate), exercised by forcing a zero extension pad — the honest
    # (generously padded) evaluator scores the same plan, so extension, not the
    # censor, is the default.
    sc = _short_multidest()
    dep = np.zeros((sc.n_paths, sc.grid.n_steps))
    dep[0, 0] = 3.0
    dep[1, 0] = 3.0
    flows = TDPathFlows(sc.content_hash(), dep)
    assert TDTAEvaluator(sc).certify(flows)["feasible"] == 1.0  # real pad clears

    class ZeroPad(TDTAEvaluator):
        def _clearing_pad(self) -> int:
            return 0

    assert ZeroPad(sc).certify(flows)["feasible"] == 0.0


def test_evaluator_rejects_unclearable_so_horizon() -> None:
    # The ADR-020/021 discipline: once a single-destination SO cell LP is
    # derivable, Z* is resolved eagerly and an unclearable horizon is a
    # construction-time error, never a scoring-time crash.
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 6.0
    short = TDTAScenario(
        name="pm-corr-short",
        network=_net("cs", 3, 2, [1, 3], [3, 2]),
        dynamics=LinkDynamics(
            length=np.ones(2),
            free_speed=np.ones(2),
            wave_speed=np.ones(2),
            jam_density=np.array([20.0, 2.0]),
            capacity=np.array([10.0, 1.0]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 1.0]), rates=rates),
        grid=TimeGrid(1.0, 3),  # 6 vehicles through Q=1 cannot clear in 3 steps
        paths=(TDPath(1, 2, (0, 1)),),
        kernel="ctm",
    )
    with pytest.raises(ValueError, match="clear"):
        TDTAEvaluator(short)


def test_shape_mismatch_raises() -> None:
    sc = pm_diamond_scenario("ctm")
    with pytest.raises(ValueError, match="shape mismatch"):
        TDTAEvaluator(sc).certify(TDPathFlows(sc.content_hash(), np.zeros((sc.n_paths, 99))))


def test_multi_destination_reports_no_so_bound() -> None:
    # A multi-destination (UE-only) instance: the SO cell LP is undefined, so
    # so_bound_gap is NaN — never faked — while the UE gap is still scored.
    rates = np.zeros((1, 3, 3))
    rates[0, 0, 1] = 2.0  # 1 -> 2
    rates[0, 0, 2] = 2.0  # 1 -> 3
    sc = TDTAScenario(
        name="pm-multidest",
        network=_net("md", 5, 3, [1, 4, 1, 5], [4, 2, 5, 3]),
        dynamics=LinkDynamics(
            length=np.ones(4),
            free_speed=np.ones(4),
            wave_speed=np.ones(4),
            jam_density=np.full(4, 20.0),
            capacity=np.array([10.0, 1.0, 10.0, 1.0]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 1.0]), rates=rates),
        grid=TimeGrid(1.0, 12),
        paths=(TDPath(1, 2, (0, 1)), TDPath(1, 3, (2, 3))),
        kernel="ctm",
    )
    assert not sc.single_destination
    m = TDTAEvaluator(sc).certify(_burst(sc, {0: 2.0, 1: 2.0}))
    assert m["feasible"] == 1.0
    assert np.isnan(m["so_bound_gap"])
    assert m["tdue_gap"] == pytest.approx(0.0, abs=1e-9)  # each OD has one path


# ------------------------------------------------------------------ scenario validation


def test_scenario_rejects_interior_diverge() -> None:
    # Two paths sharing link 0 then splitting at interior node 4 is an interior
    # diverge — forbidden by the v1 decidability restriction (in-link 0 would feed
    # two out-links). Both destinations are valid zones (2 and 3).
    rates = np.zeros((1, 3, 3))
    rates[0, 0, 1] = 1.0  # 1 -> 2
    rates[0, 0, 2] = 1.0  # 1 -> 3
    with pytest.raises(ValueError, match="interior diverge"):
        TDTAScenario(
            name="bad-diverge",
            network=_net("bd", 4, 3, [1, 4, 4], [4, 2, 3]),  # 0:1->4, 1:4->2, 2:4->3
            dynamics=LinkDynamics(
                length=np.ones(3),
                free_speed=np.ones(3),
                wave_speed=np.ones(3),
                jam_density=np.full(3, 20.0),
                capacity=np.full(3, 5.0),
            ),
            demand=DynamicDemand(breakpoints=np.array([0.0, 1.0]), rates=rates),
            grid=TimeGrid(1.0, 8),
            paths=(TDPath(1, 2, (0, 1)), TDPath(1, 3, (0, 2))),  # share link 0, split at node 4
        )


def test_scenario_rejects_od_without_path() -> None:
    # A positive-demand OD (1 -> 3) with no declared path fails the coverage check.
    rates = np.zeros((1, 3, 3))
    rates[0, 0, 1] = 3.0  # 1 -> 2 (has a path)
    rates[0, 0, 2] = 2.0  # 1 -> 3 (no path declared)
    with pytest.raises(ValueError, match="do not match"):
        TDTAScenario(
            name="no-path",
            network=_net("np", 4, 3, [1, 4], [4, 2]),
            dynamics=LinkDynamics(
                length=np.ones(2),
                free_speed=np.ones(2),
                wave_speed=np.ones(2),
                jam_density=np.full(2, 20.0),
                capacity=np.full(2, 5.0),
            ),
            demand=DynamicDemand(breakpoints=np.array([0.0, 1.0]), rates=rates),
            grid=TimeGrid(1.0, 8),
            paths=(TDPath(1, 2, (0, 1)),),  # covers 1->2 only
        )


def test_content_hash_separates_instances() -> None:
    a = pm_diamond_scenario("ctm")
    assert a.content_hash() == pm_diamond_scenario("ctm").content_hash()
    assert a.content_hash() != pm_diamond_scenario("ltm").content_hash()
    assert a.content_hash() != pm_wedge_scenario().content_hash()
    assert pm_corridor_scenario().content_hash() != pm_merge_scenario().content_hash()


def test_does_not_move_the_golden_braess_hash() -> None:
    from tabench.data.builtin import braess_scenario

    assert (
        braess_scenario().content_hash()
        == "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"
    )


# ---------------------------------------------------------- three-lens review regressions


def _spread_single_path(bp, rates_periods, n_steps, cap=(10.0, 10.0), jam=(20.0, 20.0)):
    rates = np.zeros((len(rates_periods), 2, 2))
    for i, r in enumerate(rates_periods):
        rates[i, 0, 1] = r
    return TDTAScenario(
        name="spread",
        network=_net("sp", 3, 2, [1, 3], [3, 2]),
        dynamics=LinkDynamics(
            length=np.ones(2),
            free_speed=np.ones(2),
            wave_speed=np.ones(2),
            jam_density=np.array(jam),
            capacity=np.array(cap),
        ),
        demand=DynamicDemand(breakpoints=np.array(bp), rates=rates),
        grid=TimeGrid(1.0, n_steps),
        paths=(TDPath(1, 2, (0, 1)),),
        kernel="ctm",
    )


def test_over_emission_cannot_forge_so_undercut() -> None:
    # MAJOR 1(a): a sub-budget over-emission in the last step used to drag the
    # V-based TSTT below Z* (certified so_bound_gap < -tol at feasible=1). The
    # availability-based TSTT is >= 0 by construction, so the cheat now scores a
    # POSITIVE gap; the undercut branch is also a hard censor.
    for make, split in ((pm_corridor_scenario, {0: 6.0}), (pm_wedge_scenario, {0: 1.0, 1: 5.0})):
        sc = make()
        ev = TDTAEvaluator(sc)
        dep = np.zeros((sc.n_paths, sc.grid.n_steps))
        for p, v in split.items():
            dep[p, 0] = v
        dep[-1, -1] += 5.9e-6  # over-emit d < budget in the last step
        m = ev.certify(TDPathFlows(sc.content_hash(), dep))
        assert m["feasible"] == 1.0
        assert m["so_bound_gap"] >= -ev.tol  # never undercuts the provable LP bound


def test_spread_demand_reports_no_so_bound() -> None:
    # MAJOR 1(b): a single demand period SPREAD over several steps is loaded
    # gradually and avoids the queue the burst-as-initial-occupancy LP charges,
    # so its Z* would be a spurious positive bound. derive_cell_scenario rejects
    # it (-> so_bound_gap = NaN, never faked); the UE gap still scores.
    sc = _spread_single_path([0.0, 5.0], [1.0], 12)  # rate 1 over [0,5): 5 veh, no congestion
    with pytest.raises(ValueError, match="one grid step"):
        sc.derive_cell_scenario()
    ev = TDTAEvaluator(sc)
    assert ev._z_star_time is None
    dep = np.zeros((1, sc.grid.n_steps))
    dep[0] = np.diff(sc.demand.cumulative(sc.grid.edges)[:, 0, 1])
    m = ev.certify(TDPathFlows(sc.content_hash(), dep))
    assert m["feasible"] == 1.0
    assert np.isnan(m["so_bound_gap"])
    assert m["tdue_gap"] == pytest.approx(0.0, abs=1e-9)  # single path, forced


def test_tstt_availability_convention_not_pre_departure() -> None:
    # MAJOR 1(c): TSTT must not charge pre-departure waiting. A 2-period demand
    # (3 veh in [0,1), 3 veh in [4,5)) on an uncongested single path: the true
    # experienced total is 6 veh * free-flow 2 = 12; the old V-based TSTT charged
    # the late batch from t=0 (30). total_experienced_time is the true 12; the
    # availability-based TSTT no longer charges pre-departure (well below 30).
    sc = _spread_single_path([0.0, 1.0, 4.0, 5.0], [3.0, 0.0, 3.0], 12)
    ev = TDTAEvaluator(sc)  # single period? no -> two periods, so no SO bound
    dep = np.zeros((1, sc.grid.n_steps))
    dep[0] = np.diff(sc.demand.cumulative(sc.grid.edges)[:, 0, 1])
    m = ev.certify(TDPathFlows(sc.content_hash(), dep))
    assert m["feasible"] == 1.0
    assert m["total_experienced_time"] == pytest.approx(12.0, abs=1e-6)  # the true experienced
    assert m["tstt"] < 20.0  # the spurious pre-departure charge (30) is gone


def test_clearing_pad_uses_interior_bottleneck() -> None:
    # MAJOR 2: the drain estimate must read the slowest USED-link capacity, not
    # only the sink links. An interior Q=0.2 bottleneck behind a wide (Q=10) sink
    # link needs ~100 clearing steps; the sink-only pad gave ~11 and false-censored
    # the only conforming emission.
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 20.0
    sc = TDTAScenario(
        name="pm-interior-bneck",
        network=_net("ib", 4, 2, [1, 3, 4], [3, 4, 2]),  # 0:1->3(Q10) 1:3->4(Q0.2) 2:4->2(Q10)
        dynamics=LinkDynamics(
            length=np.ones(3),
            free_speed=np.ones(3),
            wave_speed=np.ones(3),
            jam_density=np.full(3, 25.0),
            capacity=np.array([10.0, 0.2, 10.0]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 1.0]), rates=rates),
        grid=TimeGrid(1.0, 10),
        paths=(TDPath(1, 2, (0, 1, 2)),),
        kernel="ltm",
    )
    dep = np.zeros((1, sc.grid.n_steps))
    dep[0, 0] = 20.0
    assert TDTAEvaluator(sc).certify(TDPathFlows(sc.content_hash(), dep))["feasible"] == 1.0


@pytest.mark.parametrize("huge", [4.0, 5.0e6])
def test_per_od_gate_scale(huge) -> None:
    # MAJOR 3: gate tolerances must be PER-OD, not scaled by the largest OD. A
    # tiny OD (1 veh) beside a huge OD must not be able to shift its whole vehicle
    # or retract a real vehicle just because the global eps ~ tol*V_huge is large.
    rates = np.zeros((1, 3, 3))
    rates[0, 0, 1] = huge
    rates[0, 0, 2] = 1.0
    sc = TDTAScenario(
        name=f"hetero-{huge}",
        network=_net("het", 5, 3, [1, 4, 1, 5], [4, 2, 5, 3]),
        dynamics=LinkDynamics(
            length=np.ones(4),
            free_speed=np.ones(4),
            wave_speed=np.ones(4),
            jam_density=np.array([2e7, 2e7, 20.0, 20.0]),
            capacity=np.array([1e7, 1e7, 10.0, 10.0]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 1.0]), rates=rates),
        grid=TimeGrid(1.0, 8),
        paths=(TDPath(1, 2, (0, 1)), TDPath(1, 3, (2, 3))),
        kernel="ctm",
    )
    ev = TDTAEvaluator(sc)
    honest = np.zeros((2, 8))
    honest[0, 0] = huge
    honest[1, 0] = 1.0
    assert ev.certify(TDPathFlows(sc.content_hash(), honest))["feasible"] == 1.0
    shift = honest.copy()
    shift[1, 0] = 0.0
    shift[1, 3] = 1.0  # tiny OD's vehicle moved 3 steps late
    assert ev.certify(TDPathFlows(sc.content_hash(), shift))["feasible"] == 0.0
    retract = honest.copy()
    retract[1, 0] = 1.9
    retract[1, 1] = -0.9  # a real 0.9-vehicle cumulative retraction
    assert ev.certify(TDPathFlows(sc.content_hash(), retract))["feasible"] == 0.0


def test_max_form_matches_dense_reference() -> None:
    # MAJOR 4: tdue_gap_max was systematically UNDER-reported (model-flattering)
    # because the composed-cost peak sits at a departure-time kink between coarse
    # samples. On a non-grid-aligned LTM merge with a pulsed interfering OD, the
    # shipped max-form must match a dense reference to quadrature noise AND be
    # conservative (>= the reference, never flattering).
    from tabench.metrics.tdta_gaps import _earliest_time

    sc = TDTAScenario(
        name="pm-sweep",
        network=_net("sp", 5, 3, [1, 4, 3, 1, 5], [4, 2, 4, 5, 2]),
        dynamics=LinkDynamics(
            length=np.array([1.5, 1.0, 1.0, 2.0, 1.5]),
            free_speed=np.ones(5),
            wave_speed=np.ones(5),
            jam_density=np.full(5, 20.0),
            capacity=np.array([0.6, 1.0, 10.0, 10.0, 10.0]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 8.0, 10.0]), rates=_sweep_rates()),
        grid=TimeGrid(1.0, 24),
        paths=(TDPath(1, 2, (0, 1)), TDPath(1, 2, (3, 4)), TDPath(3, 2, (2, 1))),
        kernel="ltm",
    )
    ev = TDTAEvaluator(sc)
    a = np.diff(sc.demand.cumulative(sc.grid.edges)[:, 0, 1])
    b = np.diff(sc.demand.cumulative(sc.grid.edges)[:, 2, 1])
    rng = np.random.default_rng(3)
    frac = rng.uniform(0.2, 0.8, sc.grid.n_steps)
    dep = np.zeros((3, sc.grid.n_steps))
    dep[0] = a * frac
    dep[1] = a * (1.0 - frac)
    dep[2] = b
    m = ev.certify(TDPathFlows(sc.content_hash(), dep))
    assert m["feasible"] == 1.0
    # dense reference: the max is over ACTUAL travelers, i.e. over the count
    # LEVEL (a departure-TIME sweep would spuriously score hypothetical travelers
    # inside departure plateaus). A fine per-path level sweep of the SAME
    # composition gives the true peak; normalize by the same tc_min/V the shipped
    # metric uses (backed out of the certified tdue_gap: tc_min = tc_used/(1+gap)).
    out = PathLoader(sc, dep, extra_steps=ev._clearing_pad()).run()
    n_in, n_out, dt = out.n_in, out.n_out, out.grid.dt
    k = sc.grid.n_steps
    pc = np.zeros((sc.n_paths, out.grid.n_steps + 1))
    np.cumsum(dep, axis=1, out=pc[:, 1 : k + 1])
    pc[:, k + 1 :] = pc[:, k : k + 1]
    od_paths = {od: [(pj, sc.paths[pj].links) for pj in pl] for od, pl in sc.paths_by_od().items()}
    ref_max = -np.inf
    for pi, p in enumerate(sc.paths):
        n_p = float(pc[pi, -1])
        if n_p <= ev.tol:
            continue
        eps_lv = ev.tol * max(1.0, n_p)
        for lv in np.linspace(eps_lv, n_p - eps_lv, 30000):
            t = _earliest_time(pc[pi], float(lv), dt)
            u = ev._marginal_time((pi, p.links), t, n_in, n_out, pc, dt)
            mm = min(ev._marginal_time(c, t, n_in, n_out, pc, dt) for c in od_paths[p.od])
            ref_max = max(ref_max, u - mm)
    avg_min = m["total_experienced_time"] / ((1.0 + m["tdue_gap"]) * ev._V)  # = tc_min / V
    ref_gap_max = ref_max / avg_min
    assert m["tdue_gap_max"] >= ref_gap_max - 1e-3  # conservative, never model-flattering
    assert abs(m["tdue_gap_max"] - ref_gap_max) < 3e-3  # collapsed to quadrature noise


def _sweep_rates():
    rates = np.zeros((2, 3, 3))
    rates[0, 0, 1] = 1.0
    rates[0, 2, 1] = 0.25
    rates[1, 2, 1] = 2.5
    return rates


def test_demand_beyond_horizon_raises() -> None:
    # MINOR: demand extending past the grid horizon has no columns to be emitted;
    # raise at construction (eager-config discipline) rather than silently
    # censoring every honest plan.
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 0.3  # 0.3 over [0, 20] -> 6 veh, but grid ends at t=16
    with pytest.raises(ValueError, match="beyond the grid horizon"):
        TDTAScenario(
            name="pm-tail",
            network=_net("tail", 3, 2, [1, 3], [3, 2]),
            dynamics=LinkDynamics(
                length=np.ones(2),
                free_speed=np.ones(2),
                wave_speed=np.ones(2),
                jam_density=np.full(2, 20.0),
                capacity=np.full(2, 10.0),
            ),
            demand=DynamicDemand(breakpoints=np.array([0.0, 20.0]), rates=rates),
            grid=TimeGrid(1.0, 16),
            paths=(TDPath(1, 2, (0, 1)),),
        )


def test_degenerate_demand_raises() -> None:
    # MINOR (Dossier B entry 9): total demand at/below the conditioning floor
    # cannot resolve an equilibrium; raise at construction.
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 5e-7
    with pytest.raises(ValueError, match="conditioning floor"):
        _spread_single_path([0.0, 1.0], [5e-7], 8)


def test_declared_paths_omitting_shortest_helper() -> None:
    # MINOR: the (non-gating) completeness helper flags an OD whose declared set
    # omits a strictly faster free-flow path, and passes the anchors.
    for make in (pm_corridor_scenario, pm_diamond_scenario, pm_wedge_scenario, pm_merge_scenario):
        assert make().declared_paths_omitting_shortest() == []
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 4.0
    omit = TDTAScenario(
        name="omit",
        network=_net("om", 3, 2, [1, 3, 1], [3, 2, 2]),  # link 2 is a direct 1->2 shortcut
        dynamics=LinkDynamics(
            length=np.ones(3),
            free_speed=np.ones(3),
            wave_speed=np.ones(3),
            jam_density=np.full(3, 20.0),
            capacity=np.array([10.0, 1.0, 10.0]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 1.0]), rates=rates),
        grid=TimeGrid(1.0, 16),
        paths=(TDPath(1, 2, (0, 1)),),  # ff 2; the undeclared direct link 2 has ff 1
        kernel="ctm",
    )
    assert omit.declared_paths_omitting_shortest() == [(1, 2)]


def test_wedge_ue_label_is_convention_dependent() -> None:
    # MINOR disclosure: with a first-interval burst the per-interval MSA fixed
    # point (the 3/3 split) is NOT the per-traveler certificate minimizer — the
    # wedge has an intrinsic positive per-traveler gap floor at dt=1, so the "UE"
    # split scores a small POSITIVE tdue_gap and a strictly better split exists.
    # The SO<UE headline survives regardless (that is what the wedge pins).
    sc = pm_wedge_scenario()
    ev = TDTAEvaluator(sc)
    m33 = ev.certify(_burst(sc, {0: 3.0, 1: 3.0}))
    assert m33["tdue_gap"] == pytest.approx(1.0 / 11.0, abs=1e-3)  # positive, convention-dependent
    # a nearby split certifies a strictly lower per-traveler gap (the minimizer
    # is not the per-interval fixed point)
    best = min(
        ev.certify(_burst(sc, {0: a, 1: 6.0 - a}))["tdue_gap"]
        for a in (2.6, 2.8, 3.0, 3.2)
    )
    assert best < m33["tdue_gap"] - 1e-3
