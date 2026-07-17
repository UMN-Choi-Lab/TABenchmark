"""Tests for the ``dtalite-simulation`` EDOC row (adr-036/adr-040) — DTALite
``simulation()`` on the DETERMINISTIC track.

Two halves, one file (the matsim split, adr-039 — NOT test_dtalite.py's
file-level importorskip): the adapter module imports WITHOUT the wheel (it
never imports DTALite in-host), so the engine-free half runs on every core
matrix leg, and the engine-gated half un-skips wherever the wheel is installed
(``importlib.util.find_spec`` never imports — the adr-029 banner rule).

Every measured anchor asserted here was derived with the SHIPPED estimator on
the installed DTALite==0.8.1 wheel (2026-07-17); see docs/design/adr-040.
"""

from __future__ import annotations

import dataclasses
import glob
import importlib.util
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pytest

import tabench.models.adapters.dtalite_simulation as ds
from tabench.edoc.replay import EmittedBundle, PlanReplayFailure, ReplayAgent, ReplayResult
from tabench.metrics.edoc_gaps import EdocEvaluator
from tabench.models.adapters.dtalite_simulation import (
    DTALiteSimulationAdapter,
    build_dtalite_corridor_scenario,
    certify_emitted,
    installed_engine_version,
    make_replay_runner,
    negative_control_separation,
    pinned_simulation_replay,
    reference_scenario,
    shared_bottleneck_scenario,
)

BRAESS_GOLDEN_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"
_WALL = 600.0
_ENGINE_FREE_VERSION = "0.8.1"

_requires_engine = pytest.mark.skipif(
    importlib.util.find_spec("DTALite") is None,
    reason="pip install tabench[dtalite] to run the engine-gated half",
)


def _scenario(**over):
    over.setdefault("engine_version", _ENGINE_FREE_VERSION)
    return build_dtalite_corridor_scenario("dtalite-sim-test", **over)


# ==========================================================================
# engine-free: the positional trajectory parser (R10's fourth necessity)
# ==========================================================================
_TRAJ_HEADER_LINE = (
    "agent_id,departure_time,departure_time_hhmmss,loaded_status,o_zone_id,"
    "d_zone_id,distance,travel_time,current_link_seq_no,link_ids,"
    "arrival_times,departure_times,geometry"
)
# The four measured 0.8.1 row forms (probe-verified, adr-040): completed /
# period-end-truncated / head-block-stuck (all-filler) / completed-route-B.
# Zones D=1 O=2; sorted link file order a1=1 b1=2 a2=3 b2=4 (nodes MA=3 MB=4).
_TRAJ_ROWS = [
    # v0 (engine 1): completed A; entries 0/330 s, exits 300/630 s
    '1,420,07:00:00,0,2,1,0.012,1,1;3,07:00:00;07:05:30,07:05:00;07:10:30,"LS (0, 1)"',
    # v1 (engine 2): truncated in flight — real first link then 07:00:00 filler
    '2,420,07:00:00,0,2,1,0.012,0,1;3,07:00:00;07:00:00,07:05:00;07:00:00,"LS (0, 1)"',
    # v2 (engine 3): head-block/pre-period stuck — all-filler chains
    '3,420.1,07:00:06,0,2,1,0.012,0,1;3,07:00:00;07:00:00,07:00:00;07:00:00,"LS (0, 1)"',
    # v3 (engine 4): completed B; entries 6/426, exits 426/846
    '4,420.1,07:00:06,0,2,1,0.015,1,2;4,07:00:06;07:07:06,07:07:06;07:14:06,"LS (0, 1)"',
]


def _write_traj(tmp_path, rows=None, header=_TRAJ_HEADER_LINE):
    p = tmp_path / "trajectory.csv"
    p.write_text("\n".join([header, *(rows if rows is not None else _TRAJ_ROWS)]) + "\n")
    return str(p)


@pytest.fixture()
def small_sc():
    return _scenario(n_agents=4)


def _plans_for(sc):
    routes = {"v0": ("a1", "a2"), "v1": ("a1", "a2"), "v2": ("a1", "a2"), "v3": ("b1", "b2")}
    dep = dict(zip(sc.agent_ids, (float(d) for d in sc.agent_depart), strict=True))
    return {aid: (routes[aid], dep[aid]) for aid in sc.agent_ids}


def test_parser_census_on_the_four_measured_row_forms(tmp_path, small_sc):
    parsed = ds._parse_trajectory(_write_traj(tmp_path), small_sc)
    assert parsed["v0"]["complete"] and parsed["v3"]["complete"]
    assert not parsed["v1"]["complete"]  # truncated: cur=0 != n-1
    assert not parsed["v2"]["complete"]  # stuck: all-filler, cur=0
    assert parsed["v0"]["entries"] == [0.0, 330.0] and parsed["v0"]["exit"] == 630.0
    assert parsed["v0"]["sched"] == pytest.approx(0.0)  # 420 min echo - _T0
    assert parsed["v3"]["route"] == ("b1", "b2")


def test_parser_records_upstream_charge_and_incomplete_markers(tmp_path, small_sc):
    parsed = ds._parse_trajectory(_write_traj(tmp_path), small_sc)
    agents, field, flows = ds._records_from_trajectory(parsed, _plans_for(small_sc), small_sc)
    # completed A agent: door-to-door 630 s from the scheduled 0; the transfer
    # queue (330-300=30 s) is charged to the UPSTREAM edge a1 (span 330 s)
    assert agents["v0"].experienced_time == pytest.approx(630.0)
    assert agents["v0"].depart_delay == pytest.approx(0.0)
    assert field["a1"][0][0] == pytest.approx(330.0)  # entry-to-next-entry
    assert field["a2"][55][0] == pytest.approx(300.0)  # entry 330 -> k=55
    # incomplete agents carry the census marker (G3 censors downstream)
    assert agents["v1"].experienced_time < 0.0 and agents["v2"].experienced_time < 0.0
    # and contribute NO field samples (non-observations, pair D1): only v0
    # loads a1 (v1/v2 are excluded despite their real-looking first cells)
    assert sum(len(v) for v in field.get("a1", {}).values() if v) == 2
    assert list(field["a1"]) == [0]
    # flows: v0 enters a1 in k=0 and leaves (into a2) at k=55
    assert flows["a1"][0][0] == 1.0 and flows["a1"][55][1] == 1.0


def test_parser_filler_is_chain_keyed_not_time_keyed(tmp_path, small_sc):
    """D1: a LEGITIMATE 07:00:00 first entry (an agent scheduled at t=0) must
    NOT be read as filler — the census is chain-consistency-keyed (measured:
    caseB agent 1 completes with a real 07:00:00 entry)."""
    parsed = ds._parse_trajectory(_write_traj(tmp_path), small_sc)
    assert parsed["v0"]["complete"] and parsed["v0"]["entries"][0] == 0.0


def test_parser_header_and_row_shape_drift_raise(tmp_path, small_sc):
    bad_header = _TRAJ_HEADER_LINE.replace("current_link_seq_no", "cur_seq")
    with pytest.raises(RuntimeError, match="header drifted"):
        ds._parse_trajectory(_write_traj(tmp_path, header=bad_header), small_sc)
    # a 13-field row (travel_time suddenly written) must RAISE, not re-align
    row13 = _TRAJ_ROWS[0].replace("0.012,1,1;3", "0.012,300,1,1;3")
    with pytest.raises(RuntimeError, match="fields"):
        ds._parse_trajectory(_write_traj(tmp_path, rows=[row13]), small_sc)
    with pytest.raises(RuntimeError, match="fields"):
        ds._parse_trajectory(_write_traj(tmp_path, rows=["1,420,07:00:00,0,2,1"]), small_sc)
    # an engine agent id outside the written range is garbage output
    with pytest.raises(RuntimeError, match="outside the written id range"):
        ds._parse_trajectory(
            _write_traj(tmp_path, rows=[_TRAJ_ROWS[0].replace("1,420", "99,420", 1)]),
            small_sc,
        )


def test_parser_never_reads_loaded_status(tmp_path, small_sc):
    """D2: the dead ``loaded_status`` column is NEVER read — flipping it
    changes no parse output (it still moves the G1 hash, correctly, as
    content — pinned in the canon tests)."""
    a = ds._parse_trajectory(_write_traj(tmp_path), small_sc)
    flipped = [r.replace(",0,2,1,0.01", ",1,2,1,0.01") for r in _TRAJ_ROWS]
    b = ds._parse_trajectory(_write_traj(tmp_path, rows=flipped), small_sc)
    assert a == b


# ==========================================================================
# engine-free: the vehicle.csv writer (D5) + the GMNS writers
# ==========================================================================
def test_vehicle_writer_sorts_by_departure_then_agent(tmp_path, small_sc):
    """D5 (measured: an UNSORTED vehicle.csv silently filler-corrupts the
    later-departing agent at rc=0): the certifier writes the file itself,
    sorted ascending by (departure, agent) — no model-controlled byte order
    reaches the engine."""
    plans = _plans_for(small_sc)
    adversarial = dict(reversed(list(plans.items())))  # hostile dict order
    ds._write_vehicles(small_sc, adversarial, str(tmp_path))
    lines = (tmp_path / "vehicle.csv").read_text().splitlines()
    assert lines[0] == ",".join(ds._VEHICLE_HEADER)
    ids = [int(line.split(",")[0]) for line in lines[1:]]
    deps = [float(line.split(",")[1]) for line in lines[1:]]
    assert deps == sorted(deps)
    assert ids == [1, 2, 3, 4]  # (dep, agent) ascending
    # the engine-clock offset: scenario 0 s -> 420.0 engine minutes
    assert lines[1].split(",")[1] == "420.0" and lines[3].split(",")[1] == "420.1"
    # routes are 1-based sorted-file-order link indices
    assert lines[1].split(",")[9] == "1;3" and lines[4].split(",")[9] == "2;4"


def test_vehicle_writer_refuses_unknown_agent_edge_and_empty_route(tmp_path, small_sc):
    with pytest.raises(ValueError, match="unknown agent"):
        ds._write_vehicles(small_sc, {"ghost": (("a1", "a2"), 0.0)}, str(tmp_path))
    with pytest.raises(ValueError, match="empty route"):
        ds._write_vehicles(small_sc, {"v0": ((), 0.0)}, str(tmp_path))
    with pytest.raises(ValueError, match="not a"):
        ds._write_vehicles(small_sc, {"v0": (("zz",), 0.0)}, str(tmp_path))


def test_gmns_writer_links_sorted_lanes_one_and_fftt_consistent(tmp_path, small_sc):
    """The adr-029 CRITICAL (links grouped by (from,to)), the lanes^2 trap
    (engine lanes=1, capacity = edge_lanes * 600), and the ruling-6 probe
    consequence: vdf_fftt (minutes) IS the read column, and length at 60 mph
    equals it so the geometry-derived fftt agrees by construction."""
    ds._write_gmns_sim(small_sc, str(tmp_path))
    lines = (tmp_path / "link.csv").read_text().splitlines()
    assert lines[1] == "2,3,1,1,600.0,60.0,5.0,5.0,0.15,4.0,1,0"  # a1
    assert lines[2] == "2,4,2,1,1200.0,60.0,7.0,7.0,0.15,4.0,1,0"  # b1
    assert lines[3] == "3,1,3,1,600.0,60.0,5.0,5.0,0.15,4.0,1,0"  # a2
    assert lines[4] == "4,1,4,1,1200.0,60.0,7.0,7.0,0.15,4.0,1,0"  # b2
    pairs = [tuple(map(int, line.split(",")[:2])) for line in lines[1:]]
    assert pairs == sorted(pairs)
    # settings: the 7 -> 13 engine period derived from the hashed horizon
    srow = (tmp_path / "settings.csv").read_text().splitlines()[1]
    assert srow == "20,1,7,13,1,0,1,1,0,0,0"
    # zones 1..Z with zone_id == node_id; through nodes zone 0 (adr-029)
    nrows = (tmp_path / "node.csv").read_text().splitlines()[1:]
    assert [r.split(",")[1] for r in nrows] == ["1", "2", "0", "0"]


def test_engine_period_must_be_whole_hours_and_inside_one_day():
    _scenario(n_intervals=1800)  # 1800*6 = 10800 s = 3 h — whole, accepted
    with pytest.raises(ValueError, match="whole number of hours"):
        _scenario(n_intervals=1801)  # 10806 s — not a whole number of hours
    with pytest.raises(ValueError, match="> 24"):
        _scenario(n_intervals=3600 * 3)  # 7 + 18 h = hour 25


def test_parallel_edges_are_refused():
    sc = _scenario()
    twin = dataclasses.replace(
        sc,
        edge_ids=(*sc.edge_ids, "a1x"),
        edge_tail=(*sc.edge_tail, "O"),
        edge_head=(*sc.edge_head, "MA"),
        edge_fftt=np.append(sc.edge_fftt, 300.0),
        edge_lanes=np.append(sc.edge_lanes, 1),
    )
    with pytest.raises(ValueError, match="parallel edges"):
        ds._edge_order(twin)


# ==========================================================================
# engine-free: R9 integerization + route_assignment schema
# ==========================================================================
def test_r9_integerization_largest_remainder_bound_and_interleave(small_sc):
    sc = _scenario(n_agents=1000)
    od = {("O", "D"): [(("a1", "a2"), 781.25), (("b1", "b2"), 218.75)]}
    plans = ds._integerize_route_volumes(sc, od)
    counts = {}
    for route, _dep in plans.values():
        counts[route] = counts.get(route, 0) + 1
    assert counts == {("a1", "a2"): 781, ("b1", "b2"): 219}
    # the |count - share*N| <= 1 bound (the disclosed R9 mapping floor)
    assert abs(counts[("a1", "a2")] - 781.25) <= 1.0
    # per-OD exactness: every agent got exactly one route (G2 bijection feed)
    assert set(plans) == set(sc.agent_ids)
    # departures ride through verbatim from the trip table
    assert plans["v0"][1] == 0.0 and plans["v2"][1] == 6.0
    # determinism
    assert ds._integerize_route_volumes(sc, od) == plans
    # fractional-position interleave: the minority route appears throughout
    # the agent order, not as a trailing block
    heads = [plans[f"v{i}"][0][0] for i in range(1000)]
    first_b = heads.index("b1")
    last_b = 999 - heads[::-1].index("b1")
    assert first_b < 10 and last_b > 990
    # 3.25/0.75 over 4 agents: floors [3, 0], largest remainder -> [3, 1]
    sc4 = _scenario(n_agents=4)
    plans4 = ds._integerize_route_volumes(
        sc4, {("O", "D"): [(("a1", "a2"), 3.25), (("b1", "b2"), 0.75)]}
    )
    c4 = {}
    for route, _dep in plans4.values():
        c4[route] = c4.get(route, 0) + 1
    assert c4 == {("a1", "a2"): 3, ("b1", "b2"): 1}


def test_route_assignment_parse_matches_verified_header(tmp_path, small_sc):
    header = (
        "mode,route_id,o_zone_id,d_zone_id,unique_route_id,prob,node_ids,"
        "link_ids,distance_mile,total_distance_km,total_free_flow_travel_time,"
        "total_travel_time,route_key,seed_od_volume,target_od_volume,"
        "final_est_od_volume,volume,"  # 0.8.1 writes the trailing comma
    )
    rows = [
        "auto,0,2,1,1,0.78125,2;3;1,1;3,0.006,0.01,10,10.7,6_4,1000,0,1000,781.25,",
        "auto,0,2,1,2,0.21875,2;4;1,2;4,0.008,0.014,14,14.0,6_4,1000,0,1000,218.75,",
    ]
    p = tmp_path / "route_assignment.csv"
    p.write_text("\n".join([header, *rows]) + "\n")
    routes = ds._parse_route_assignment(str(p), small_sc)
    assert routes == {
        ("O", "D"): [(("a1", "a2"), 781.25), (("b1", "b2"), 218.75)]
    }
    p.write_text(header.replace("volume,", "vol,") + "\n" + rows[0])
    with pytest.raises(RuntimeError, match="header drifted"):
        ds._parse_route_assignment(str(p), small_sc)


# ==========================================================================
# engine-free: the MSA loop on a STUBBED engine (blend arithmetic + picks)
# ==========================================================================
def _stub_replay_result(sc, plans, congested_edge="a1", cost=900.0):
    """A deterministic fake replay: the congested edge carries ``cost`` in
    every interval; everything else free-flows; every agent completes with
    its plan's field cost."""
    from tabench.edoc.field import build_field_from_records

    records = {
        congested_edge: {
            k: (cost, 1.0) for k in range(int(sc.n_intervals))
        }
    }
    field = build_field_from_records(
        records, sc.fftt_of(), sc.dt, sc.n_intervals, sc.field_semantics
    )
    agents = {}
    for aid, (route, dep) in plans.items():
        tau = float(dep)
        for e in route:
            tau += field.traversal_time(e, tau)
        agents[aid] = ReplayAgent(
            agent_id=aid, departure=float(dep), arrival=tau, route=tuple(route),
            experienced_time=tau - float(dep), depart_delay=0.0,
        )
    return ReplayResult(
        canon_hash="stub", agents=agents, field_records=records,
        flows={}, n_intervals=int(sc.n_intervals),
    )


def test_msa_blend_moves_hash_picked_fraction_and_reaches_fixed_point(monkeypatch):
    sc = _scenario(n_agents=100)
    monkeypatch.setattr(ds, "installed_engine_version", lambda: sc.engine_version)
    monkeypatch.setattr(ds, "_run_assignment_for_routes",
                        lambda _sc, _dl: {("O", "D"): [(("a1", "a2"), 100.0)]})
    monkeypatch.setattr(
        ds, "pinned_simulation_replay",
        lambda scenario, plans, *, deadline, workdir=None: _stub_replay_result(scenario, plans),
    )
    em = DTALiteSimulationAdapter(iterations=1).emit(sc)
    moved = [aid for aid, (route, _d) in em.plans.items() if route == ("b1", "b2")]
    # k=0: everyone improves (a1 at 900 s vs B at 840 s), phi = 1/2 -> 50 move
    assert len(moved) == 50
    # the picks are the hash-derived ranking — reproducible from the instance
    # hash alone (ruling 3: no RNG-library dependence)
    improvers = list(sc.agent_ids)
    assert set(moved) == ds._msa_pick(sc.content_hash(), 0, improvers, 50)
    # a different instance hash picks a different set
    assert set(moved) != ds._msa_pick("other-hash", 0, improvers, 50)
    # fixed point: with the congestion gone (free-flow stub field on every
    # edge), nobody improves and the plans pass through unchanged
    monkeypatch.setattr(
        ds, "pinned_simulation_replay",
        lambda scenario, plans, *, deadline, workdir=None: _stub_replay_result(
            scenario, plans, cost=300.0
        ),
    )
    em2 = DTALiteSimulationAdapter(iterations=3).emit(sc)
    assert all(route == ("a1", "a2") for route, _d in em2.plans.values())


def test_msa_in_loop_boost_census_raises(monkeypatch):
    """Layer (a) of the pair-12/D3 gate: an MSA iterate whose replay census
    crosses the boost onset RAISES ValueError (the model's own field would be
    boost-contaminated) — never censors, never proceeds."""
    sc = _scenario(n_agents=100, n_intervals=600)  # 1 h horizon: onset < 0
    monkeypatch.setattr(ds, "installed_engine_version", lambda: sc.engine_version)
    monkeypatch.setattr(ds, "_run_assignment_for_routes",
                        lambda _sc, _dl: {("O", "D"): [(("a1", "a2"), 100.0)]})
    monkeypatch.setattr(
        ds, "pinned_simulation_replay",
        lambda scenario, plans, *, deadline, workdir=None: _stub_replay_result(scenario, plans),
    )
    with pytest.raises(ValueError, match="boost"):
        DTALiteSimulationAdapter(iterations=1).emit(sc)
    # iterations=0 runs NO iterate replay: the final X is left to the
    # certify-time CENSOR arm (layer b), so emit succeeds
    em = DTALiteSimulationAdapter(iterations=0).emit(sc)
    censored = ds._boost_censor(sc, {"feasible": 1.0, "rg_d1": 0.5}, em.experienced)
    assert censored["feasible"] == 0.0 and censored["boost_crossing_n"] > 0
    assert np.isnan(censored["rg_d1"])


def test_boost_onset_arithmetic():
    sc = _scenario()  # 6 h horizon
    assert ds._boost_onset_s(sc) == pytest.approx(21600.0 - 720 * 6.0)  # 17280
    a = ReplayAgent("x", 0.0, 17280.0, ("a1", "a2"), 17280.0, 0.0)
    b = ReplayAgent("y", 0.0, 17279.9, ("a1", "a2"), 17279.9, 0.0)
    dead = ReplayAgent("z", 0.0, 17281.0, ("a1", "a2"), -1.0, 0.0)  # incomplete
    assert ds._boost_crossings(sc, {"x": a, "y": b, "z": dead}) == ["x"]


# ==========================================================================
# engine-free: family gates + semantic config + vetting + typing
# ==========================================================================
def test_family_shape_and_gates():
    sc = _scenario()
    assert sc.seed_list == ()  # the deterministic track: no macroreps
    assert sc.departure_quantum == 6.0 and sc.dt == 6.0
    assert sc.walk_bound == 2
    assert float(sc.agent_depart.max()) == 2994.0  # 2-per-6-s over 50 min
    with pytest.raises(ValueError, match="at most 2 departures"):
        _scenario(agents_per_slot=10, slot_step=30.0)
    with pytest.raises(ValueError, match="multiple of"):
        _scenario(n_agents=999)
    sh = shared_bottleneck_scenario(engine_version=_ENGINE_FREE_VERSION)
    assert sh.family == "dtalite-shared-bottleneck"


def test_lull_drop_gate_refuses_gaps_but_allows_continuous_streams():
    """The adr-040 lull-drop gate: the engine's t>=600 early exit silently
    DROPS departures that follow an all-completed instant. A departure after
    a lull is refused eagerly; a continuous stream past the 60-min mark is
    fine (someone is always provably in flight)."""
    # continuous 6-s singleton stream to 5994 s: constructs (the pass case)
    sc = _scenario(n_agents=1000, agents_per_slot=1, slot_step=6.0)
    assert float(sc.agent_depart.max()) == 5994.0
    # a first departure at/after the exit-check threshold is refused
    base = _scenario(n_agents=4)
    late = dataclasses.replace(
        base, agent_depart=np.array([3600.0, 3606.0, 3612.0, 3618.0])
    )
    with pytest.raises(ValueError, match="lull"):
        ds._assert_no_lull_drop(late)
    # a mid-profile gap long enough for everyone to clear is refused
    gap = dataclasses.replace(
        base, agent_depart=np.array([0.0, 6.0, 12.0, 4200.0])
    )
    with pytest.raises(ValueError, match="lull"):
        ds._assert_no_lull_drop(gap)


def test_semantic_config_carries_every_pinned_constant(monkeypatch):
    """S3 F2 from birth: every outcome-bearing writer/runner constant rides in
    the hashed semantic_config (presence), and a drift in ANY of them moves
    the derived string — and with it the instance hash (mutation loop)."""
    cfg = ds._semantic_config()
    assert "sim=import DTALite; DTALite.simulation()" in cfg  # N6 mode constant
    assert "assign=import DTALite; DTALite.assignment()" in cfg
    assert "t0=25200" in cfg
    assert "step=6" in cfg
    assert "boostWindow=720" in cfg
    assert "capPerLane=600" in cfg
    assert "freeSpeed=60" in cfg
    assert "vdf=0.15:4" in cfg
    assert "assignIters=20" in cfg and "assignPeriodH=1" in cfg
    assert "omp=1" in cfg  # the OMP=1 CORRECTNESS pin (G0)
    assert "lcg=time-step-keyed;seedable=false;macroreps=none" in cfg
    assert _scenario().semantic_config == cfg
    for name, mutated in (
        ("_SIM_CMD", "import DTALite; DTALite.simulation(x=1)"),
        ("_ASSIGN_CMD", "import DTALite; DTALite.accessibility()"),
        ("_T0", 21600.0),
        ("_SIM_STEP_S", 12.0),
        ("_BOOST_WINDOW_INTERVALS", 1200),
        ("_FREE_SPEED_MPH", 30.0),
        ("_CAP_PER_LANE_VPH", 900.0),
        ("_VDF_ALPHA", 0.3),
        ("_VDF_BETA", 2.0),
        ("_ASSIGN_ITERATIONS", 40),
        ("_ASSIGN_PERIOD_HOURS", 2),
        ("_DETERMINISTIC_DISCLOSURE", "lcg=other"),
    ):
        with monkeypatch.context() as mp:
            mp.setattr(ds, name, mutated)
            assert ds._semantic_config() != cfg, f"{name} drift did not move semantic_config"


def test_certify_emitted_vetting_is_topology_keyed_engine_free():
    """F10/F3 from birth: an unvetted topology is refused before any engine
    work, and a relabeled family STRING cannot borrow another topology's
    vetting — the key is the topology digest."""
    diamond = _scenario()
    shared = shared_bottleneck_scenario(engine_version=_ENGINE_FREE_VERSION)
    bundle = EmittedBundle(
        plans={}, experienced={}, engine_version=diamond.engine_version, seed=42
    )
    with pytest.raises(RuntimeError, match="separation-vetted"):
        certify_emitted(shared, bundle)
    diamond_digest = ds._topology_digest(diamond)
    ds._SEPARATION_VETTED_TOPOLOGIES.add(diamond_digest)
    try:
        relabeled = dataclasses.replace(shared, family=diamond.family)
        with pytest.raises(RuntimeError, match="separation-vetted"):
            certify_emitted(relabeled, bundle)  # still refused: wrong topology
    finally:
        ds._SEPARATION_VETTED_TOPOLOGIES.discard(diamond_digest)


def test_replay_timeout_typing_scenario_deadline_censors_caller_clip_raises(
    monkeypatch,
):
    """S3 F1 from birth: a mid-replay timeout is the censor signal ONLY when
    the SCENARIO-declared replay_deadline_s was binding; a tighter CALLER
    wall is certifier-side budget exhaustion and RAISES RuntimeError.
    Engine-free: the 'engine' is a stub sleeper command."""
    sc = _scenario(n_agents=4)
    monkeypatch.setattr(ds, "installed_engine_version", lambda: sc.engine_version)
    monkeypatch.setattr(ds, "_SIM_CMD", "import time; time.sleep(30)")
    plans = _plans_for(sc)

    d, clipped = ds._intersect_replay_deadline(sc, None)
    assert not clipped and d - time.perf_counter() == pytest.approx(30.0, abs=1.0)
    _d, clipped = ds._intersect_replay_deadline(sc, time.perf_counter() + 3600.0)
    assert not clipped  # a LOOSER caller wall never clips
    _d, clipped = ds._intersect_replay_deadline(sc, time.perf_counter() + 0.5)
    assert clipped

    with pytest.raises(RuntimeError, match="wall deadline") as ei:
        pinned_simulation_replay(sc, plans, deadline=time.perf_counter() + 0.5)
    assert not isinstance(ei.value, PlanReplayFailure)

    tight = dataclasses.replace(sc, replay_deadline_s=0.5)
    with pytest.raises(PlanReplayFailure, match="wall deadline"):
        pinned_simulation_replay(tight, plans, deadline=None)


def test_wall_kill_reaps_process_group(tmp_path):
    """A wall kill reaps the WHOLE process group (the S2 killpg discipline):
    a grandchild spawned by the engine child must be gone after the kill."""
    marker = tmp_path / "gc.pid"
    code = (
        "import subprocess, time, os\n"
        f"p = subprocess.Popen(['sleep', '30'])\n"
        f"open({str(marker)!r}, 'w').write(str(p.pid))\n"
        "time.sleep(30)\n"
    )
    with pytest.raises(RuntimeError, match="wall deadline"):
        ds._run(code, cwd=str(tmp_path), deadline=time.perf_counter() + 0.8, what="killpin")
    time.sleep(0.5)
    gc = int(marker.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(gc, 0)  # reaped with the group


def test_installed_engine_version_names_the_extra_when_absent(monkeypatch):
    import importlib.metadata as md

    def _absent(_name):
        raise md.PackageNotFoundError("DTALite")

    monkeypatch.setattr(md, "version", _absent)
    with pytest.raises(RuntimeError, match=r"tabench\[dtalite\]"):
        installed_engine_version()


def test_per_seed_scenarios_refuses_the_deterministic_track():
    from tabench.edoc.macrorep import per_seed_scenarios

    with pytest.raises(ValueError, match="seed_list"):
        per_seed_scenarios(_scenario())


def test_adapter_class_attrs_and_import_posture():
    assert DTALiteSimulationAdapter.name == "dtalite-simulation"
    assert DTALiteSimulationAdapter.track == "edoc-deterministic"
    assert DTALiteSimulationAdapter.seedable is False
    # NOT in MODEL_REGISTRY (EDOC producers never are — adr-037/039/040)
    from tabench.models import MODEL_REGISTRY

    assert "dtalite-simulation" not in MODEL_REGISTRY


def test_import_is_stdout_silent():
    """The banner rule holds because DTALite is never imported in-host: a
    bare import of the adapter module prints NOTHING."""
    proc = subprocess.run(
        [sys.executable, "-c", "import tabench.models.adapters.dtalite_simulation"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0 and proc.stdout == ""


def test_does_not_move_the_golden_braess_hash():
    from tabench.data.builtin import braess_scenario

    assert braess_scenario().content_hash() == BRAESS_GOLDEN_HASH


# ==========================================================================
# engine-gated: the installed 0.8.1 wheel end to end
# ==========================================================================
@pytest.fixture(scope="module")
def vetted_reference():
    """Run the family separation gate ONCE (it also separation-vets the
    topology for certify_emitted below). ~2 s: 2 emits + 4 certifier replays."""
    sc = reference_scenario()
    anchors = negative_control_separation(sc, wall_seconds=_WALL)
    return sc, anchors


@pytest.fixture(scope="module")
def converged_emission(vetted_reference):
    sc, _ = vetted_reference
    return sc, DTALiteSimulationAdapter().emit(sc, wall_seconds=_WALL)


@_requires_engine
def test_installed_version_matches_the_reference_pin():
    assert reference_scenario().engine_version == installed_engine_version()


@_requires_engine
def test_negative_control_separates_and_vets(vetted_reference):
    """The R4 anchors, re-derived with the SHIPPED estimator (adr-040):
    step-0 FW split RG_D1 0.372244 vs MSA-converged 0.050309 -> 7.40x on the
    floor-displayed basis (declared factor 5.0). Bounds are loose
    version-robust ceilings (the .so can shift under the >=0.8 floor)."""
    sc, anchors = vetted_reference
    assert anchors["separation"] >= sc.separation_factor
    assert 0.30 < anchors["control_rg_d1"] < 0.45  # measured 0.372244
    assert 0.02 < anchors["converged_rg_d1"] < 0.10  # measured 0.050309
    assert ds._topology_digest(sc) in ds._SEPARATION_VETTED_TOPOLOGIES


@_requires_engine
def test_reference_emit_certify_feasible_and_ranked(vetted_reference, converged_emission):
    sc, _ = vetted_reference
    _, emitted = converged_emission
    m = certify_emitted(sc, emitted, wall_seconds=_WALL)
    assert m["feasible"] == 1.0
    assert m["sub_floor"] == 0.0 and m["rg_d1"] > m["floor_gap"]  # RANKED
    assert m["delta"] <= sc.floor_seconds  # measured ~1.75 s vs floor 10
    assert m["max_backlog"] <= sc.backlog_bound
    assert m["br_coverage"] == 1.0
    assert m["r3_max_s"] <= sc.r3_tolerance_s  # the harness self-cross-check ran


@_requires_engine
def test_g1_replay_bit_deterministic(converged_emission):
    """The G1 determinism double PLUS the raw-byte pin: twin replays in
    different dirs emit byte-identical trajectory.csv (the engine is
    byte-deterministic at the pinned OMP=1 on linux-x86_64 — adr-040)."""
    sc, emitted = converged_emission
    w1 = tempfile.mkdtemp(prefix=f"tabench-edoc-dtalite-{os.getpid()}-g1a-")
    w2 = tempfile.mkdtemp(prefix=f"tabench-edoc-dtalite-{os.getpid()}-g1b-")
    try:
        r1 = pinned_simulation_replay(sc, emitted.plans, deadline=None, workdir=w1)
        r2 = pinned_simulation_replay(sc, emitted.plans, deadline=None, workdir=w2)
        assert r1.canon_hash == r2.canon_hash
        b1 = open(os.path.join(w1, "trajectory.csv"), "rb").read()
        b2 = open(os.path.join(w2, "trajectory.csv"), "rb").read()
        assert b1 == b2
    finally:
        import shutil

        shutil.rmtree(w1, ignore_errors=True)
        shutil.rmtree(w2, ignore_errors=True)


@_requires_engine
def test_hostile_parent_omp_is_overridden(converged_emission, monkeypatch):
    """D4: the child env ALWAYS pins OMP_NUM_THREADS=1 (a CORRECTNESS pin —
    measured at OMP=4: divergent trajectories + a SIGSEGV on a congested
    net); a hostile parent env cannot move the canonical hash."""
    sc, emitted = converged_emission
    r1 = pinned_simulation_replay(sc, emitted.plans, deadline=None)
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    r2 = pinned_simulation_replay(sc, emitted.plans, deadline=None)
    assert r1.canon_hash == r2.canon_hash


@_requires_engine
def test_r9_split_tripwire_and_dead_vehicle_csv(vetted_reference):
    """Ruling 7: the CI anchor tripwire pins route_assignment.csv's FW split
    VALUES on the reference instance (781.25/218.75 — the adr-029
    link_performance precedent), and the dead-code pin: assignment() emits a
    HEADER-ONLY vehicle.csv (TAPLite.cpp:2238) while route_assignment.csv
    carries the rows — the R9 construction NEVER reads vehicle.csv."""
    sc, _ = vetted_reference
    routes = ds._run_assignment_for_routes(sc, None)
    vols = sorted(v for _r, v in routes[("O", "D")])
    assert vols == [218.75, 781.25]
    # live dead-vehicle pin, in a kept workdir
    wd = tempfile.mkdtemp(prefix=f"tabench-edoc-dtalite-{os.getpid()}-r9pin-")
    try:
        ds._write_gmns_sim(sc, wd, period_hours=(7, 7 + ds._ASSIGN_PERIOD_HOURS))
        ds._run(ds._ASSIGN_CMD, cwd=wd, deadline=None, what="r9 pin")
        veh = open(os.path.join(wd, "vehicle.csv")).read().splitlines()
        assert len(veh) == 1  # header only: the emission is dead code
        ra = open(os.path.join(wd, "route_assignment.csv")).read().splitlines()
        assert len(ra) >= 3  # header + the two FW routes
    finally:
        import shutil

        shutil.rmtree(wd, ignore_errors=True)


@_requires_engine
def test_headblock_hazard_is_the_fast_census_variant(vetted_reference):
    """Ruling 1 (measured on the SHIPPED family): a pre-period departure
    head-blocks every later same-first-link agent at FAST rc=0 with
    all-filler rows — the census-censor variant (the pilot's infinite-loop
    variant did not reproduce here; the R6 replay deadline still ships as its
    defense). The runner returns promptly and the incomplete markers carry
    the censor signal."""
    sc, _ = vetted_reference
    plans = ds._integerize_route_volumes(sc, ds._run_assignment_for_routes(sc, None))
    doctored = dict(plans)
    doctored["v0"] = (("a1", "a2"), -600.0)  # engine minute 410, pre-period
    t0 = time.perf_counter()
    rep = pinned_simulation_replay(sc, doctored, deadline=None)
    wall = time.perf_counter() - t0
    assert wall < sc.replay_deadline_s  # no hang: the fast variant
    incomplete = {aid for aid, a in rep.agents.items() if a.experienced_time < 0.0}
    a_agents = {aid for aid, (r, _d) in doctored.items() if r[0] == "a1"}
    assert incomplete == a_agents  # the whole same-first-link cohort corrupts
    assert len(incomplete) == 781


@_requires_engine
def test_d5_unsorted_vehicle_csv_silently_corrupts_sorted_fixes(vetted_reference):
    """D5 (the NEW measured gate, probe sim4/sim5): an UNSORTED vehicle.csv
    silently filler-corrupts the later-departing agent at rc=0; the same
    agents sorted ascending both complete. The shipped writer always sorts —
    this regression drives the raw engine to prove why."""
    sc, _ = vetted_reference
    header = ",".join(ds._VEHICLE_HEADER)
    later_first = [
        "1,420.5,,auto,0,2,1,0,,1;3,0.002,0.003,0,0,k,1",  # departs LATER, listed first
        "2,420.05,,auto,0,2,1,0,,1;3,0.002,0.003,0,0,k,1",
    ]
    results = {}
    for label, rows in (
        ("unsorted", later_first),
        ("sorted", list(reversed(later_first))),
    ):
        wd = tempfile.mkdtemp(prefix=f"tabench-edoc-dtalite-{os.getpid()}-d5-")
        try:
            ds._write_gmns_sim(sc, wd)
            with open(os.path.join(wd, "vehicle.csv"), "w") as fh:
                fh.write("\n".join([header, *rows]) + "\n")
            ds._run(ds._SIM_CMD, cwd=wd, deadline=None, what=f"d5 {label}")
            import csv as _csv

            with open(os.path.join(wd, "trajectory.csv")) as fh:
                data = {r[0]: r for r in list(_csv.reader(fh))[1:]}
            results[label] = {aid: int(row[7]) for aid, row in data.items()}
        finally:
            import shutil

            shutil.rmtree(wd, ignore_errors=True)
    assert results["unsorted"]["1"] == 0  # the later-departing agent corrupts
    assert results["unsorted"]["2"] == 1
    assert results["sorted"] == {"1": 1, "2": 1}  # sorted: both complete


@_requires_engine
def test_boost_window_censor_live(vetted_reference):
    """Layers (a)-(c) of the pair-12/D3 gate on the live engine: a 1 h-horizon
    variant of the SAME (vetted) topology puts every exit inside the boost
    window (onset = 3600 - 4320 < 0), so certify_emitted CENSORS the feasible
    emission (layer b, zero extra engine calls), the MSA in-loop census
    RAISES (layer a), and the separation gate refuses via the censored-anchor
    path (layer c)."""
    sc, _ = vetted_reference
    boost_sc = build_dtalite_corridor_scenario(
        "dtalite-boost-variant", n_intervals=600
    )
    assert ds._topology_digest(boost_sc) == ds._topology_digest(sc)  # same topology
    em = DTALiteSimulationAdapter(iterations=0).emit(boost_sc, wall_seconds=_WALL)
    m = certify_emitted(boost_sc, em, wall_seconds=_WALL)
    assert m["feasible"] == 0.0 and m["boost_crossing_n"] == 1000.0
    with pytest.raises(ValueError, match="boost"):
        DTALiteSimulationAdapter(iterations=1).emit(boost_sc, wall_seconds=_WALL)
    with pytest.raises(ValueError):
        negative_control_separation(boost_sc, wall_seconds=_WALL)


@_requires_engine
def test_lull_drop_live_and_gated(vetted_reference):
    """The adr-040 lull-drop hazard, executed: a departure after an
    all-completed lull is silently DROPPED at rc=0 (the census marks it
    incomplete — the certify-time backstop), and the family builder refuses
    the profile eagerly (the construction-time gate)."""
    sc, _ = vetted_reference
    base = build_dtalite_corridor_scenario("dtalite-lull-variant", n_agents=10)
    dep = np.array([6.0 * i for i in range(9)] + [4200.0])
    lull = dataclasses.replace(base, agent_depart=dep)
    plans = {aid: (("a1", "a2"), float(d)) for aid, d in zip(lull.agent_ids, dep, strict=True)}
    rep = pinned_simulation_replay(lull, plans, deadline=None)
    assert rep.agents["v9"].experienced_time < 0.0  # dropped -> census marker
    assert sum(1 for a in rep.agents.values() if a.experienced_time >= 0.0) == 9
    with pytest.raises(ValueError, match="lull"):
        ds._assert_no_lull_drop(lull)


@_requires_engine
def test_doctored_x_censors(vetted_reference, converged_emission):
    """Forgery pair 2 (self-report substitution): a doctored experienced
    record diverges from the pinned replay and censors."""
    sc, _ = vetted_reference
    _, emitted = converged_emission
    bad = dict(emitted.experienced)
    aid = sorted(bad)[0]
    a = bad[aid]
    bad[aid] = ReplayAgent(
        agent_id=a.agent_id, departure=a.departure, arrival=a.arrival,
        route=a.route, experienced_time=a.experienced_time - 30.0,
        depart_delay=a.depart_delay,
    )
    doctored = EmittedBundle(
        plans=emitted.plans, experienced=bad,
        engine_version=emitted.engine_version, seed=emitted.seed,
    )
    m = EdocEvaluator(sc, make_replay_runner(deadline=None)).certify(doctored)
    assert m["feasible"] == 0.0 and np.isnan(m["rg_d1"])


@_requires_engine
def test_g0_pin_mismatch_raises(vetted_reference):
    sc, _ = vetted_reference
    bad = dataclasses.replace(sc, engine_version="9.9.9")
    with pytest.raises(ValueError, match="G0 engine pin"):
        DTALiteSimulationAdapter(iterations=0).emit(bad, wall_seconds=_WALL)


@_requires_engine
def test_shared_transfer_bottleneck_is_refused():
    """Pair 12: the non-separating shared-transfer topology (its queue
    cancels between the anchors on the floor-DISPLAYED basis) is REFUSED by
    the separation gate — a construction error, never a certified row."""
    sc = shared_bottleneck_scenario()
    with pytest.raises(ValueError, match="negative-control separation"):
        negative_control_separation(sc, wall_seconds=_WALL)
    assert ds._topology_digest(sc) not in ds._SEPARATION_VETTED_TOPOLOGIES


@_requires_engine
def test_temp_dir_hygiene(vetted_reference):
    """emit + a certifier replay leave NO working tree behind. The glob is
    scoped to THIS process's pid-prefixed dirs (S3 F5 from birth), so a
    concurrent engine session on the same box cannot cross-flake it."""
    sc, _ = vetted_reference
    pat = tempfile.gettempdir() + f"/tabench-edoc-dtalite-{os.getpid()}-*"
    before = set(glob.glob(pat))
    adapter = DTALiteSimulationAdapter(iterations=0)
    emitted = adapter.emit(sc, wall_seconds=_WALL)
    pinned_simulation_replay(sc, emitted.plans, deadline=None)
    assert adapter.last_workdir is None
    assert set(glob.glob(pat)) - before == set()
