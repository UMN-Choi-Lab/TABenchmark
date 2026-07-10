"""Merchant & Nemhauser (1978) exit-function SO-DTA — anchors + P1 (adr-020).

Both anchors are hand-derived (aggregate earliest-arrival lower bounds, see the
ADR) and machine-verified here: the parallel-route instance (SO = 10, capacity
metering) and the series instance where holding back is STRICTLY optimal
(relaxed SO = 18 vs 22 for the naive M-N equality form ``e = g(x)``). The
certifier recomputes feasibility, cost, the harness-resolved LP optimality gap,
and the pure-arithmetic dual certificate from the EMITTED trajectory alone.
"""

import numpy as np
import pytest

from tabench.dta import (
    DTATrajectory,
    SODTAScenario,
    mn_metering_scenario,
    mn_parallel_scenario,
    solve_so_dta,
)
from tabench.metrics import SODTAEvaluator


def _forward_equality_trajectory(
    scenario: SODTAScenario, inflow_plan: np.ndarray
) -> DTATrajectory:
    """Roll the naive M-N EQUALITY dynamics ``e = g(x)`` forward under a given
    first-link inflow plan, handing every exit to the unique downstream link
    (series/parallel test networks only)."""
    n_t, n_l = scenario.n_periods, scenario.n_links
    x = np.zeros((n_t + 1, n_l))
    u = np.zeros((n_t, n_l))
    e = np.zeros((n_t, n_l))
    for t in range(n_t):
        e[t] = scenario.exit_flow(x[t])
        u[t] = inflow_plan[t]
        for a in range(n_l):  # same-period hand-off to the downstream link
            j = int(scenario.link_head[a])
            if j != scenario.destination:
                nxt = np.nonzero(scenario.link_tail == j)[0]
                u[t, nxt[0]] += e[t, a]
        x[t + 1] = x[t] + u[t] - e[t]
    return DTATrajectory(
        scenario_hash=scenario.content_hash(), inflows=u, exits=e, occupancies=x
    )


# ---------------------------------------------------------------- anchor A


def test_parallel_anchor_optimum_is_10() -> None:
    sc = mn_parallel_scenario()
    traj = solve_so_dta(sc)
    metrics = SODTAEvaluator(sc).certify(traj)
    assert metrics["feasible"] == 1.0
    assert metrics["total_cost"] == pytest.approx(10.0, abs=1e-8)
    assert metrics["so_optimality_gap"] == pytest.approx(0.0, abs=1e-9)


def test_parallel_anchor_meters_the_fast_link_at_capacity() -> None:
    # The lower-bound argument forces E(1) = 2: any optimum exits the fast link
    # at its capacity bound (g = min(x, 2), tight) during period 1.
    traj = solve_so_dta(mn_parallel_scenario())
    assert traj.exits[1, 0] == pytest.approx(2.0, abs=1e-8)
    assert traj.occupancies[1, 0] >= 2.0 - 1e-8


def test_parallel_anchor_dual_certificate_verifies() -> None:
    sc = mn_parallel_scenario()
    metrics = SODTAEvaluator(sc).certify(solve_so_dta(sc))
    assert metrics["dual_gap"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["dual_infeasibility"] == pytest.approx(0.0, abs=1e-9)


def test_parallel_suboptimal_split_scores_its_exact_gap() -> None:
    # Send all 6 vehicles down the fast link: e = g meters 2/period, cost
    # 6 + 4 + 2 = 12, a certified feasible plan with gap (12 - 10)/10 = 0.2.
    sc = mn_parallel_scenario()
    plan = np.zeros((sc.n_periods, sc.n_links))
    plan[0, 0] = 6.0
    traj = _forward_equality_trajectory(sc, plan)
    metrics = SODTAEvaluator(sc).certify(traj)
    assert metrics["feasible"] == 1.0
    assert metrics["total_cost"] == pytest.approx(12.0, abs=1e-9)
    assert metrics["so_optimality_gap"] == pytest.approx(0.2, abs=1e-9)
    assert np.isnan(metrics["dual_gap"])  # no certificate emitted


# ---------------------------------------------------------------- anchor B


def test_metering_anchor_optimum_is_18_with_strict_holding_back() -> None:
    sc = mn_metering_scenario()
    traj = solve_so_dta(sc)
    metrics = SODTAEvaluator(sc).certify(traj)
    assert metrics["feasible"] == 1.0
    assert metrics["total_cost"] == pytest.approx(18.0, abs=1e-8)
    assert metrics["so_optimality_gap"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["dual_gap"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["dual_infeasibility"] == pytest.approx(0.0, abs=1e-9)
    # EVERY optimum meters link A at rate 1 while g_A(x_A(1)) = min(4, 2) = 2:
    # strict slack in the exit bound — the Carey-relaxation physics.
    assert traj.exits[1, 0] == pytest.approx(1.0, abs=1e-8)
    g_a1 = sc.exit_flow(traj.occupancies[1])[0]
    assert g_a1 == pytest.approx(2.0, abs=1e-8)
    assert metrics["exit_slack_max"] >= 1.0 - 1e-8


def test_metering_anchor_naive_equality_form_costs_22() -> None:
    # The decision-free equality dynamics flush A at 2/period and queue the
    # doubly-priced link B: cost 22 = 18 + 4 wasted vehicle-periods. It is a
    # FEASIBLE relaxed plan (e = g attains the bound) with gap (22 - 18)/18.
    sc = mn_metering_scenario()
    plan = np.zeros((sc.n_periods, sc.n_links))
    plan[0, 0] = 4.0
    traj = _forward_equality_trajectory(sc, plan)
    metrics = SODTAEvaluator(sc).certify(traj)
    assert metrics["feasible"] == 1.0
    assert metrics["total_cost"] == pytest.approx(22.0, abs=1e-9)
    assert metrics["so_optimality_gap"] == pytest.approx(4.0 / 18.0, abs=1e-9)


# ---------------------------------------------------------------- certification


def test_wrong_hash_is_censored() -> None:
    sc = mn_parallel_scenario()
    traj = solve_so_dta(sc)
    forged = DTATrajectory(
        scenario_hash="not-this-scenario",
        inflows=traj.inflows,
        exits=traj.exits,
        occupancies=traj.occupancies,
    )
    metrics = SODTAEvaluator(sc).certify(forged)
    assert metrics["feasible"] == 0.0
    assert np.isnan(metrics["so_optimality_gap"])


def _tampered(traj: DTATrajectory, sc: SODTAScenario, **arrays) -> DTATrajectory:
    parts = {
        "inflows": traj.inflows.copy(),
        "exits": traj.exits.copy(),
        "occupancies": traj.occupancies.copy(),
    }
    parts.update(arrays)
    return DTATrajectory(scenario_hash=sc.content_hash(), **parts)


def test_conservation_violation_is_censored() -> None:
    sc = mn_parallel_scenario()
    traj = solve_so_dta(sc)
    occ = traj.occupancies.copy()
    occ[2, 0] += 1.0  # teleport a vehicle
    assert SODTAEvaluator(sc).certify(_tampered(traj, sc, occupancies=occ))["feasible"] == 0.0


def test_node_imbalance_is_censored() -> None:
    sc = mn_parallel_scenario()
    traj = solve_so_dta(sc)
    u = traj.inflows.copy()
    e = traj.exits.copy()
    occ = traj.occupancies.copy()
    # move one vehicle's inflow from the fast to the slow first link WITHOUT
    # demand changing: link conservation can be patched, node balance cannot
    u[0, 0] -= 1.0
    occ[1:, 0] -= 1.0
    occ[1:, 0] = np.maximum(occ[1:, 0], 0.0)
    e_new = np.minimum(e, sc.exit_flow(occ[:-1]))
    metrics = SODTAEvaluator(sc).certify(
        _tampered(traj, sc, inflows=u, exits=e_new, occupancies=occ)
    )
    assert metrics["feasible"] == 0.0


def test_exit_bound_violation_is_censored() -> None:
    # Exit 3 vehicles from the fast link in period 1 when g = min(4, 2) = 2.
    sc = mn_parallel_scenario()
    traj = solve_so_dta(sc)
    e = traj.exits.copy()
    occ = traj.occupancies.copy()
    e[1, 0] += 1.0
    occ[2:, 0] -= 1.0  # keep link conservation consistent
    metrics = SODTAEvaluator(sc).certify(_tampered(traj, sc, exits=e, occupancies=occ))
    assert metrics["feasible"] == 0.0


def test_stranded_flow_is_censored() -> None:
    # Park the demand on the slow route's first link and never exit it.
    sc = mn_parallel_scenario()
    n_t, n_l = sc.n_periods, sc.n_links
    u = np.zeros((n_t, n_l))
    e = np.zeros((n_t, n_l))
    x = np.zeros((n_t + 1, n_l))
    u[0, 1] = 6.0
    x[1:, 1] = 6.0
    traj = DTATrajectory(
        scenario_hash=sc.content_hash(), inflows=u, exits=e, occupancies=x
    )
    assert SODTAEvaluator(sc).certify(traj)["feasible"] == 0.0


def test_negative_flow_is_censored() -> None:
    sc = mn_parallel_scenario()
    traj = solve_so_dta(sc)
    u = traj.inflows.copy()
    u[3, 0] = -0.5
    assert SODTAEvaluator(sc).certify(_tampered(traj, sc, inflows=u))["feasible"] == 0.0


def test_forged_dual_certificate_is_reported_not_believed() -> None:
    sc = mn_parallel_scenario()
    traj = solve_so_dta(sc)
    assert traj.duals is not None
    forged = DTATrajectory(
        scenario_hash=sc.content_hash(),
        inflows=traj.inflows,
        exits=traj.exits,
        occupancies=traj.occupancies,
        duals={"eq": traj.duals["eq"] * 3.0, "ub": traj.duals["ub"] * 3.0},
    )
    metrics = SODTAEvaluator(sc).certify(forged)
    assert metrics["feasible"] == 1.0  # primal untouched
    assert metrics["so_optimality_gap"] == pytest.approx(0.0, abs=1e-9)
    # the scaled certificate breaks dual feasibility and/or the zero gap
    bad = max(abs(metrics["dual_gap"]), metrics["dual_infeasibility"])
    assert bad > 1e-6


def test_shape_mismatch_raises() -> None:
    sc = mn_parallel_scenario()
    traj = solve_so_dta(sc)
    other = mn_metering_scenario()
    with pytest.raises(ValueError, match="shape mismatch"):
        SODTAEvaluator(other).certify(traj)


# ---------------------------------------------------------------- scenario


def test_scenario_validation() -> None:
    good = dict(
        name="v",
        n_nodes=3,
        destination=2,
        link_tail=[0, 1],
        link_head=[1, 2],
        exit_pieces=(((1.0, 0.0),), ((1.0, 0.0),)),
        demand=np.array([[1.0, 0.0, 0.0]]),
    )
    SODTAScenario(**good)
    with pytest.raises(ValueError, match="destination out of range"):
        SODTAScenario(**{**good, "destination": 5})
    with pytest.raises(ValueError, match="self-loop"):
        SODTAScenario(**{**good, "link_head": [0, 2]})
    with pytest.raises(ValueError, match="absorbing"):
        SODTAScenario(**{**good, "link_tail": [0, 2], "link_head": [2, 1]})
    with pytest.raises(ValueError, match="slope exactly 1"):
        SODTAScenario(**{**good, "exit_pieces": (((1.0, 0.5),), ((1.0, 0.0),))})
    with pytest.raises(ValueError, match="nondecreasing"):
        SODTAScenario(**{**good, "exit_pieces": (((-1.0, 0.0),), ((1.0, 0.0),))})
    with pytest.raises(ValueError, match="cost_weights"):
        SODTAScenario(**{**good, "cost_weights": [1.0, 0.0]})
    with pytest.raises(ValueError, match="demand at the destination"):
        SODTAScenario(**{**good, "demand": np.array([[1.0, 0.0, 1.0]])})
    with pytest.raises(ValueError, match="cannot reach"):
        SODTAScenario(
            **{**good, "n_nodes": 4, "demand": np.array([[1.0, 0.0, 0.0, 2.0]])}
        )


def test_exit_flow_evaluates_piecewise_min() -> None:
    sc = mn_parallel_scenario()  # link 0: min(x, 2); links 1, 2: x
    g = sc.exit_flow(np.array([[0.0, 0.0, 0.0], [1.0, 3.0, 0.5], [5.0, 1.0, 2.0]]))
    np.testing.assert_allclose(g, [[0.0, 0.0, 0.0], [1.0, 3.0, 0.5], [2.0, 1.0, 2.0]])


def test_content_hash_separates_instances() -> None:
    a, b = mn_parallel_scenario(), mn_metering_scenario()
    assert a.content_hash() != b.content_hash()
    assert a.content_hash() == mn_parallel_scenario().content_hash()
    tweaked_demand = a.demand.copy()
    tweaked_demand[0, 0] += 1.0
    tweaked = SODTAScenario(
        name="tweaked",
        n_nodes=a.n_nodes,
        destination=a.destination,
        link_tail=a.link_tail,
        link_head=a.link_head,
        exit_pieces=a.exit_pieces,
        demand=tweaked_demand,
    )
    assert tweaked.content_hash() != a.content_hash()


def test_too_short_horizon_raises() -> None:
    sc = mn_parallel_scenario()
    short = SODTAScenario(
        name="short",
        n_nodes=sc.n_nodes,
        destination=sc.destination,
        link_tail=sc.link_tail,
        link_head=sc.link_head,
        exit_pieces=sc.exit_pieces,
        demand=sc.demand[:1],  # 6 vehicles, 1 period: nothing can reach the sink
    )
    with pytest.raises(ValueError, match="horizon"):
        solve_so_dta(short)


# ------------------------------------------- adversarial-review regressions


def test_aggregate_leak_teleport_is_censored() -> None:
    # Review CRITICAL: per-cell tolerances scaled by TOTAL demand let ~eps-sized
    # residuals accumulate over T periods into a material teleport that
    # certified feasible=1 with a NEGATIVE gap (cost below the harness's own
    # Z*, hundreds of vehicles vanished). The aggregate mass budget + the
    # weak-duality undercut censor must kill the whole class.
    big = 1.0e6
    cap = 1.0e4
    n_t = 105
    demand = np.zeros((n_t, 2))
    demand[0, 0] = big
    sc = SODTAScenario(
        name="leak",
        n_nodes=2,
        destination=1,
        link_tail=[0],
        link_head=[1],
        exit_pieces=(((1.0, 0.0), (0.0, cap)),),
        demand=demand,
    )
    leak = 0.99 * 1e-6 * big  # 0.99x the OLD (total-demand-scaled) tolerance
    x = np.zeros((n_t + 1, 1))
    u = np.zeros((n_t, 1))
    e = np.zeros((n_t, 1))
    u[0, 0] = big - leak
    for t in range(n_t):
        g = sc.exit_flow(x[t])[0]
        e[t, 0] = min(g + leak, x[t, 0] + u[t, 0])
        x[t + 1, 0] = max(x[t, 0] + u[t, 0] - e[t, 0] - leak, 0.0)
    traj = DTATrajectory(
        scenario_hash=sc.content_hash(), inflows=u, exits=e, occupancies=x
    )
    metrics = SODTAEvaluator(sc).certify(traj)
    assert metrics["feasible"] == 0.0
    # and the honest optimum still certifies clean on the same instance
    honest = SODTAEvaluator(sc).certify(solve_so_dta(sc))
    assert honest["feasible"] == 1.0
    assert abs(honest["so_optimality_gap"]) <= 1e-9


def test_shadow_shift_cannot_undercut_the_optimum() -> None:
    # Review MAJOR: shifting an honest optimum's occupancies down by 0.99*eps
    # used to keep feasible=1 while scoring a negative gap (negative-occupancy
    # cost credit). Now: censored (aggregate exit-bound excess) or, at worst,
    # within the tolerance floor — never materially below the optimum.
    sc = mn_parallel_scenario()
    opt = solve_so_dta(sc)
    delta = 0.99 * 1e-6 * float(sc.demand.sum())
    shifted = DTATrajectory(
        scenario_hash=sc.content_hash(),
        inflows=opt.inflows,
        exits=opt.exits,
        occupancies=opt.occupancies - delta,
    )
    metrics = SODTAEvaluator(sc).certify(shifted)
    assert metrics["feasible"] == 0.0 or metrics["so_optimality_gap"] >= -1e-6


def test_evaluator_construction_rejects_unclearable_horizon() -> None:
    # Review MAJOR: certify() used to RAISE on a gate-passing trajectory when
    # the canonical LP was unsolvable (contract: only wrong shapes raise). The
    # reference optimum now resolves eagerly, so a too-short horizon fails at
    # harness configuration time, never at scoring time.
    sc = mn_parallel_scenario()
    short = SODTAScenario(
        name="short",
        n_nodes=sc.n_nodes,
        destination=sc.destination,
        link_tail=sc.link_tail,
        link_head=sc.link_head,
        exit_pieces=sc.exit_pieces,
        demand=sc.demand[:1],
    )
    with pytest.raises(ValueError, match="horizon"):
        SODTAEvaluator(short)


def test_geometric_decay_exit_functions_are_rejected() -> None:
    # Review MINOR: a link whose binding intercept-0 piece has slope < 1 decays
    # geometrically and can NEVER satisfy terminal clearance, for any horizon —
    # the validator must reject it up front instead of blaming the horizon.
    demand = np.zeros((30, 2))
    demand[0, 0] = 1.0
    with pytest.raises(ValueError, match="slope exactly 1"):
        SODTAScenario(
            name="geo",
            n_nodes=2,
            destination=1,
            link_tail=[0],
            link_head=[1],
            exit_pieces=(((0.5, 0.0),),),
            demand=demand,
        )


def test_scenario_arrays_are_read_only() -> None:
    # Review MINOR: the "frozen" scenario's arrays were mutable in place,
    # desyncing the content hash from an evaluator's cached reference optimum.
    sc = mn_metering_scenario()
    with pytest.raises(ValueError, match="read-only"):
        sc.cost_weights[1] = 0.5
    with pytest.raises(ValueError, match="read-only"):
        sc.demand[0, 0] = 99.0
    with pytest.raises(ValueError, match="read-only"):
        sc.link_head[0] = 0


def test_does_not_move_the_golden_braess_hash() -> None:
    from tabench.data.builtin import braess_scenario

    assert (
        braess_scenario().content_hash()
        == "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"
    )
