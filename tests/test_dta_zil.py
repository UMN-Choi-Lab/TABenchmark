"""Ziliaskopoulos (2000) LP SO-DTA on CTM cells — anchors + P1 (adr-021).

The diverge/spillback anchor is hand-solved (aggregate earliest-arrival bounds
plus the ``N_B = 1`` pair lemma ``y_BS(s) + y_BS(s+1) <= 1``) and
machine-verified here: SO = 26, cell B jam-full in every optimum, and the
finite-storage row worth exactly +1 veh-interval (``N_B = 2`` gives 25). The
corridor anchor cross-checks the LP against the repo's own strict CTM loading
(``CTMLink`` + ``NetworkLoader``): both give 33 exactly.
"""

import numpy as np
import pytest

from tabench.dta import (
    CellSODTAScenario,
    CellTrajectory,
    solve_cell_so_dta,
    zil_corridor_scenario,
    zil_diverge_spillback_scenario,
)
from tabench.metrics import CellSODTAEvaluator

INF = np.inf


def _variant(sc: CellSODTAScenario, **overrides) -> CellSODTAScenario:
    base = dict(
        name="variant",
        n_cells=sc.n_cells,
        sink=sc.sink,
        conn_tail=sc.conn_tail,
        conn_head=sc.conn_head,
        capacity=sc.capacity,
        storage=sc.storage,
        delta=sc.delta,
        demand=sc.demand,
        initial_occupancy=sc.initial_occupancy,
    )
    base.update(overrides)
    return CellSODTAScenario(**base)


def _traj_from_flows(sc: CellSODTAScenario, y: np.ndarray) -> CellTrajectory:
    """Roll conservation forward from the initial condition under given flows —
    the emitted occupancies are conservation-consistent by construction."""
    x = np.zeros((sc.n_periods + 1, sc.n_cells))
    x[0] = sc.initial_occupancy
    for t in range(sc.n_periods):
        x[t + 1] = x[t] + sc.demand[t]
        for c in range(sc.n_conns):
            x[t + 1, sc.conn_tail[c]] -= y[t, c]
            x[t + 1, sc.conn_head[c]] += y[t, c]
    return CellTrajectory(scenario_hash=sc.content_hash(), occupancies=x, flows=y)


def _strict_ctm_rollout(sc: CellSODTAScenario, prefer: list[int]) -> CellTrajectory:
    """Forward strict-CTM dynamics: each cell sends ``min(x, Q)``, allocated to
    connectors in ``prefer`` order, each capped by the head's remaining
    ``min(Q, delta (N - x))`` receiving room (start-of-interval state)."""
    n_t, n_e = sc.n_periods, sc.n_conns
    y = np.zeros((n_t, n_e))
    x_t = sc.initial_occupancy.astype(float).copy()
    for t in range(n_t):
        sending = np.minimum(x_t, sc.capacity)
        room = np.minimum(
            sc.capacity,
            np.where(np.isfinite(sc.storage), sc.delta * (sc.storage - x_t), INF),
        )
        for c in sorted(range(n_e), key=prefer.index):
            i, j = int(sc.conn_tail[c]), int(sc.conn_head[c])
            flow = max(0.0, min(sending[i], room[j]))
            y[t, c] = flow
            sending[i] -= flow
            room[j] -= flow
        for c in range(n_e):
            x_t = x_t.copy()
            x_t[sc.conn_tail[c]] -= y[t, c]
            x_t[sc.conn_head[c]] += y[t, c]
        x_t = x_t + sc.demand[t]
    return _traj_from_flows(sc, y)


# ---------------------------------------------------------------- anchors


def test_diverge_spillback_anchor_optimum_is_26() -> None:
    sc = zil_diverge_spillback_scenario()
    traj = solve_cell_so_dta(sc)
    metrics = CellSODTAEvaluator(sc).certify(traj)
    assert metrics["feasible"] == 1.0
    assert metrics["total_cost"] == pytest.approx(26.0, abs=1e-8)
    assert metrics["so_optimality_gap"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["dual_gap"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["dual_infeasibility"] == pytest.approx(0.0, abs=1e-9)


def test_spillback_binds_in_the_optimum() -> None:
    # The pair lemma forces y_BS(2) = 1, hence n_B(2) = N_B = 1: the tiny cell
    # is jam-full and its storage row blocks inflow that Q_B alone would allow.
    sc = zil_diverge_spillback_scenario()
    traj = solve_cell_so_dta(sc)
    assert traj.occupancies[2, 2] == pytest.approx(1.0, abs=1e-8)  # n_B(2) = N_B


def test_storage_bound_is_worth_exactly_one_veh_interval() -> None:
    # Relaxing N_B from 1 to 2 (or to effectively infinite) drops the optimum
    # to 25 — the finite-storage/spillback effect the M-N exit-function model
    # cannot represent.
    sc = zil_diverge_spillback_scenario()
    for n_b in (2.0, 1000.0):
        storage = sc.storage.copy()
        storage[2] = n_b
        traj = solve_cell_so_dta(_variant(sc, storage=storage))
        assert traj.provenance["objective"] == pytest.approx(25.0, abs=1e-8)


def test_strict_ctm_rollout_attains_the_lp_optimum() -> None:
    # The LP optimum is CTM-realizable on this anchor: strict Godunov dynamics
    # with a B-first diverge preference reproduce J* = 26 with zero holding.
    sc = zil_diverge_spillback_scenario()
    traj = _strict_ctm_rollout(sc, prefer=[0, 1, 3, 4, 5, 2])  # A->B before A->C
    metrics = CellSODTAEvaluator(sc).certify(traj)
    assert metrics["feasible"] == 1.0
    assert metrics["total_cost"] == pytest.approx(26.0, abs=1e-8)
    assert metrics["so_optimality_gap"] == pytest.approx(0.0, abs=1e-9)


def test_all_long_route_plan_scores_its_exact_gap() -> None:
    # Never using the bottleneck B: everything through C -> D at Q = 2 per
    # interval costs 30 (hand-checked): gap = (30 - 26)/26.
    sc = zil_diverge_spillback_scenario()
    y = np.zeros((sc.n_periods, sc.n_conns))
    # connectors: 0 R>A, 1 A>B, 2 A>C, 3 C>D, 4 B>S, 5 D>S
    y[0, 0] = 6.0
    y[[1, 2, 3], 2] = 2.0  # A->C at t = 1, 2, 3
    y[[2, 3, 4], 3] = 2.0  # C->D at t = 2, 3, 4
    y[[3, 4, 5], 5] = 2.0  # D->S at t = 3, 4, 5
    metrics = CellSODTAEvaluator(sc).certify(_traj_from_flows(sc, y))
    assert metrics["feasible"] == 1.0
    assert metrics["total_cost"] == pytest.approx(30.0, abs=1e-8)
    assert metrics["so_optimality_gap"] == pytest.approx(4.0 / 26.0, abs=1e-9)
    assert np.isnan(metrics["dual_gap"])  # no certificate emitted


def test_holding_on_the_optimal_face_certifies_with_diagnostic() -> None:
    # Hold one vehicle at the source for one interval (y_RA(0) = 5, strictly
    # below all four bounds, queue left behind) and run the hand-optimal plan
    # otherwise: still J = 26 — LP holding lives on the optimal face; the
    # certifier reports it as the Tier-B holding_max diagnostic, not an error.
    sc = zil_diverge_spillback_scenario()
    y = np.zeros((sc.n_periods, sc.n_conns))
    y[0, 0] = 5.0
    y[1, 0] = 1.0
    y[[1, 3], 1] = 1.0  # A->B at t = 1, 3
    y[[1, 2], 2] = 2.0  # A->C
    y[[2, 3], 3] = 2.0  # C->D
    y[[2, 4], 4] = 1.0  # B->S
    y[[3, 4], 5] = 2.0  # D->S
    metrics = CellSODTAEvaluator(sc).certify(_traj_from_flows(sc, y))
    assert metrics["feasible"] == 1.0
    assert metrics["total_cost"] == pytest.approx(26.0, abs=1e-8)
    assert metrics["so_optimality_gap"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["holding_max"] >= 1.0 - 1e-8  # one vehicle held at the source


def test_corridor_lp_equals_repo_ctm_loading() -> None:
    # Cross-model consistency: the control-free corridor's LP optimum equals
    # the TSTT of the repo's own CTMLink/NetworkLoader loading, exactly.
    from tabench.core.scenario import Network
    from tabench.dnl import (
        CTMLink,
        DynamicDemand,
        DynamicScenario,
        LinkDynamics,
        NetworkLoader,
        TimeGrid,
    )

    sc = zil_corridor_scenario()
    traj = solve_cell_so_dta(sc)
    assert traj.provenance["objective"] == pytest.approx(33.0, abs=1e-8)
    assert CellSODTAEvaluator(sc).certify(traj)["feasible"] == 1.0

    n_links = 2
    network = Network(
        name="zil-xcheck-corridor",
        n_nodes=3,
        n_zones=2,
        first_thru_node=1,
        init_node=np.array([1, 3]),
        term_node=np.array([3, 2]),
        capacity=np.ones(n_links),
        length=np.zeros(n_links),
        free_flow_time=np.ones(n_links),
        b=np.zeros(n_links),
        power=np.ones(n_links),
        toll=np.zeros(n_links),
        link_type=np.ones(n_links, dtype=np.int64),
    )
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 6.0
    scn = DynamicScenario(
        name="zil-xcheck-corridor",
        network=network,
        dynamics=LinkDynamics(
            length=np.ones(n_links),
            free_speed=np.ones(n_links),
            wave_speed=np.ones(n_links),
            jam_density=np.array([20.0, 2.0]),
            capacity=np.array([10.0, 1.0]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 1.0]), rates=rates),
        grid=TimeGrid(dt=1.0, n_steps=10),
    )
    out = NetworkLoader(scn, CTMLink).run()
    into_dest = np.flatnonzero(scn.network.term_node == 2)
    arrivals = out.n_out[into_dest].sum(axis=0)
    tstt = float(scn.grid.dt * np.sum(6.0 - arrivals[:-1]))
    assert tstt == pytest.approx(33.0, abs=1e-9)


# ---------------------------------------------------------------- certification


def test_wrong_hash_is_censored() -> None:
    sc = zil_diverge_spillback_scenario()
    traj = solve_cell_so_dta(sc)
    forged = CellTrajectory(
        scenario_hash="not-this-scenario",
        occupancies=traj.occupancies,
        flows=traj.flows,
    )
    metrics = CellSODTAEvaluator(sc).certify(forged)
    assert metrics["feasible"] == 0.0
    assert np.isnan(metrics["so_optimality_gap"])


def test_teleport_is_censored() -> None:
    sc = zil_diverge_spillback_scenario()
    traj = solve_cell_so_dta(sc)
    occ = traj.occupancies.copy()
    occ[3, 1] += 1.0
    metrics = CellSODTAEvaluator(sc).certify(
        CellTrajectory(scenario_hash=sc.content_hash(), occupancies=occ, flows=traj.flows)
    )
    assert metrics["feasible"] == 0.0


def test_spillback_violation_is_censored() -> None:
    # Corridor: keep feeding cell B (Q=1, N=2) one vehicle per interval even
    # after it is jam-full — inflow respects Q but exceeds delta*(N - x).
    sc = zil_corridor_scenario()
    y = np.zeros((sc.n_periods, sc.n_conns))  # connectors: 0 R>A, 1 A>B, 2 B>S
    y[0, 0] = 6.0
    y[1:7, 1] = 1.0  # A->B at t = 1..6; B is full from t = 3 on
    y[3:9, 2] = 1.0  # B->S at t = 3..8
    metrics = CellSODTAEvaluator(sc).certify(_traj_from_flows(sc, y))
    assert metrics["feasible"] == 0.0


def test_stranded_flow_is_censored() -> None:
    sc = zil_diverge_spillback_scenario()
    y = np.zeros((sc.n_periods, sc.n_conns))
    traj = _traj_from_flows(sc, y)  # nobody moves: 6 vehicles stranded at R
    assert CellSODTAEvaluator(sc).certify(traj)["feasible"] == 0.0


def test_negative_flow_is_censored() -> None:
    sc = zil_diverge_spillback_scenario()
    traj = solve_cell_so_dta(sc)
    y = traj.flows.copy()
    y[5, 0] = -0.5
    metrics = CellSODTAEvaluator(sc).certify(
        CellTrajectory(
            scenario_hash=sc.content_hash(), occupancies=traj.occupancies, flows=y
        )
    )
    assert metrics["feasible"] == 0.0


def test_shadow_shift_cannot_undercut_the_optimum() -> None:
    # The adr-020 review exploit transplanted: shifting non-sink occupancies
    # down by 0.99*eps must not certify below Z*.
    sc = zil_diverge_spillback_scenario()
    traj = solve_cell_so_dta(sc)
    delta = 0.99 * 1e-6 * 6.0
    shift = np.zeros_like(traj.occupancies)
    shift[:, np.arange(sc.n_cells) != sc.sink] = delta
    shifted = CellTrajectory(
        scenario_hash=sc.content_hash(),
        occupancies=traj.occupancies - shift,
        flows=traj.flows,
    )
    metrics = CellSODTAEvaluator(sc).certify(shifted)
    assert metrics["feasible"] == 0.0 or metrics["so_optimality_gap"] >= -1e-6


def test_forged_dual_certificate_is_reported_not_believed() -> None:
    sc = zil_diverge_spillback_scenario()
    traj = solve_cell_so_dta(sc)
    assert traj.duals is not None
    forged = CellTrajectory(
        scenario_hash=sc.content_hash(),
        occupancies=traj.occupancies,
        flows=traj.flows,
        duals={"eq": traj.duals["eq"] * 3.0, "ub": traj.duals["ub"] * 3.0},
    )
    metrics = CellSODTAEvaluator(sc).certify(forged)
    assert metrics["feasible"] == 1.0
    bad = max(abs(metrics["dual_gap"]), metrics["dual_infeasibility"])
    assert bad > 1e-6


def test_evaluator_construction_rejects_unclearable_horizon() -> None:
    sc = zil_corridor_scenario()
    short = _variant(sc, demand=sc.demand[:2])  # 6 veh cannot clear in 2 steps
    with pytest.raises(ValueError, match="horizon"):
        CellSODTAEvaluator(short)


def test_shape_mismatch_raises() -> None:
    sc = zil_diverge_spillback_scenario()
    traj = solve_cell_so_dta(sc)
    with pytest.raises(ValueError, match="shape mismatch"):
        CellSODTAEvaluator(zil_corridor_scenario()).certify(traj)


# ---------------------------------------------------------------- scenario


def test_scenario_validation() -> None:
    good = dict(
        name="v",
        n_cells=3,
        sink=2,
        conn_tail=[0, 1],
        conn_head=[1, 2],
        capacity=[INF, 1.0, INF],
        storage=[INF, 2.0, INF],
        delta=[1.0, 1.0, 1.0],
        demand=np.array([[1.0, 0.0, 0.0]] + [[0.0, 0.0, 0.0]] * 4),
    )
    CellSODTAScenario(**good)
    with pytest.raises(ValueError, match="sink out of range"):
        CellSODTAScenario(**{**good, "sink": 7})
    with pytest.raises(ValueError, match="absorbing"):
        CellSODTAScenario(**{**good, "conn_tail": [0, 2], "conn_head": [1, 1]})
    with pytest.raises(ValueError, match="delta"):
        CellSODTAScenario(**{**good, "delta": [1.0, 1.5, 1.0]})
    with pytest.raises(ValueError, match="duplicate"):
        CellSODTAScenario(**{**good, "conn_tail": [0, 0], "conn_head": [1, 1]})
    with pytest.raises(ValueError, match="no incoming"):
        CellSODTAScenario(
            **{**good, "demand": np.array([[1.0, 1.0, 0.0]] + [[0.0] * 3] * 4)}
        )
    with pytest.raises(ValueError, match="exceeds storage"):
        CellSODTAScenario(**{**good, "initial_occupancy": [0.0, 5.0, 0.0]})
    with pytest.raises(ValueError, match="sink takes no demand"):
        CellSODTAScenario(
            **{**good, "demand": np.array([[1.0, 0.0, 1.0]] + [[0.0] * 3] * 4)}
        )
    with pytest.raises(ValueError, match="cannot reach"):
        CellSODTAScenario(
            **{
                **good,
                "n_cells": 4,
                "capacity": [INF, 1.0, INF, 1.0],
                "storage": [INF, 2.0, INF, INF],
                "delta": [1.0] * 4,
                "demand": np.array([[1.0, 0.0, 0.0, 1.0]] + [[0.0] * 4] * 4),
            }
        )


# ------------------------------------------- adversarial-review regressions


def test_initial_condition_teleport_is_censored() -> None:
    # Review CRITICAL: the initial gate had per-cell eps (scaled by a large x0)
    # but no aggregate budget — the ONE unbudgeted door. A cheater deletes the
    # trickle sources' vehicles at t=0 (each |claim - x0| <= eps), conjures
    # replacements at ghost cells beside the sink (delivery nets out,
    # conservation sees the CLAIMED x[0]), and buries the savings as holding at
    # the big source to dodge the undercut censor. The aggregate init budget
    # must censor it.
    big = 1.0e6
    n_t = 12
    inf = np.inf
    # cells: 0 = big source, 1-3 trickle sources (Q=0.2), 4-6 ghost cells, 7 sink
    x0 = np.zeros(8)
    x0[0] = big
    x0[1:4] = 1.0
    sc = CellSODTAScenario(
        name="init-teleport",
        n_cells=8,
        sink=7,
        conn_tail=[0, 1, 2, 3, 4, 5, 6],
        conn_head=[7, 7, 7, 7, 7, 7, 7],
        capacity=[inf, 0.2, 0.2, 0.2, inf, inf, inf, inf],
        storage=[inf] * 8,
        delta=[1.0] * 8,
        demand=np.zeros((n_t, 8)),
        initial_occupancy=x0,
    )
    evaluator = CellSODTAEvaluator(sc)
    x = np.zeros((n_t + 1, 8))
    y = np.zeros((n_t, 7))
    x[0, 0] = big
    x[0, 4:7] = 1.0  # trickles claimed 0, ghosts claimed 1: each |diff| = 1 <= eps
    burn = 3.0 * (1.0 + 0.8 + 0.6 + 0.4 + 0.2)  # the honest trickle drain cost
    y[0, 0] = big - burn
    y[1, 0] = burn  # hold `burn` at the big source for one interval
    x[1, 0] = burn
    y[0, 4:7] = 1.0  # ship the conjured ghosts immediately
    x[1, 7] = (big - burn) + 3.0
    x[2:, 7] = big + 3.0
    traj = CellTrajectory(scenario_hash=sc.content_hash(), occupancies=x, flows=y)
    assert evaluator.certify(traj)["feasible"] == 0.0
    # and the honest solver output still certifies clean on this scenario
    honest = evaluator.certify(solve_cell_so_dta(sc))
    assert honest["feasible"] == 1.0
    assert abs(honest["so_optimality_gap"]) <= 1e-9


def test_demand_into_finite_storage_source_is_rejected() -> None:
    # Review MAJOR: demand bypasses the receiving-space rows, so a pulse above
    # a finite N overfilled the source inside the LP while the certifier
    # censored every mass-conserving trajectory — including the solver's own
    # optimum. Such scenarios are now a construction-time error.
    demand = np.zeros((12, 2))
    demand[0, 0] = 10.0
    with pytest.raises(ValueError, match="infinite storage"):
        CellSODTAScenario(
            name="finite-source",
            n_cells=2,
            sink=1,
            conn_tail=[0],
            conn_head=[1],
            capacity=[1.0, INF],
            storage=[5.0, INF],
            delta=[1.0, 1.0],
            demand=demand,
        )


def test_dual_bound_is_immune_to_sign_noise_on_large_rows() -> None:
    # Review MINOR: a +1e-9 sign violation on a large-b spillback row used to
    # shift the "certified" dual bound while dual_infeasibility read ~1e-9.
    # y_ub is now clipped at 0 before the bound is computed: the bound stays
    # put (conservative) and the raw violation is still reported.
    sc = zil_diverge_spillback_scenario()
    traj = solve_cell_so_dta(sc)
    assert traj.duals is not None
    y_ub = traj.duals["ub"].copy()
    y_ub += 1e-9  # push every row (incl. zero-dual large-b rows) above 0
    noisy = CellTrajectory(
        scenario_hash=sc.content_hash(),
        occupancies=traj.occupancies,
        flows=traj.flows,
        duals={"eq": traj.duals["eq"], "ub": y_ub},
    )
    metrics = CellSODTAEvaluator(sc).certify(noisy)
    assert metrics["feasible"] == 1.0
    assert metrics["dual_gap"] == pytest.approx(0.0, abs=1e-7)  # bound unmoved
    assert 0.0 < metrics["dual_infeasibility"] <= 1e-8  # violation reported


def test_content_hash_separates_instances() -> None:
    a = zil_diverge_spillback_scenario()
    b = zil_corridor_scenario()
    assert a.content_hash() != b.content_hash()
    assert a.content_hash() == zil_diverge_spillback_scenario().content_hash()
    storage = a.storage.copy()
    storage[2] = 2.0
    assert _variant(a, storage=storage).content_hash() != a.content_hash()


def test_scenario_arrays_are_read_only() -> None:
    sc = zil_diverge_spillback_scenario()
    with pytest.raises(ValueError, match="read-only"):
        sc.storage[2] = 5.0
    with pytest.raises(ValueError, match="read-only"):
        sc.initial_occupancy[0] = 1.0


def test_does_not_move_the_golden_braess_hash() -> None:
    from tabench.data.builtin import braess_scenario

    assert (
        braess_scenario().content_hash()
        == "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"
    )
