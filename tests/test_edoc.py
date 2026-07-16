"""EDOC-1 substrate core tests (engine-free; synthetic replay fixtures).

Design: docs/design/adr-036-external-dynamic-observational-certificate.md.

Covers the canonicalization module + hash surface (R10), the frozen EdocScenario
+ its MECHANICAL hash-coverage (adr-024 lesson: every field participates), the
occupancy-aware frozen field + non-FIFO / poisoned-alternative regression pins
(MAJOR-1/MAJOR-3), and the G0-G4 certifier + RG_D1 scorer (censor-vs-raise). The
real SUMO engine is exercised in the sumo-gated adapter test (adr-037); here the
replay runner is injected as a deterministic fixture, so these run in the core
matrix with no extra.
"""

from __future__ import annotations

import dataclasses
import gzip

import numpy as np
import pytest

from tabench.edoc import canon
from tabench.edoc.field import FrozenField, build_field_from_records, build_origin_waits
from tabench.edoc.replay import (
    EmittedBundle,
    PlanReplayFailure,
    ReplayAgent,
    ReplayResult,
    assert_engine_pin,
)
from tabench.edoc.scenario import EdocScenario
from tabench.edoc.tdsp import evaluate_route, td_shortest_path
from tabench.metrics.edoc_gaps import EdocEvaluator

GOLDEN_BRAESS_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"


# --------------------------------------------------------------------------
# fixtures: a two-parallel-route instance (A congested, B free) + a matching
# emitted bundle and a deterministic fake replay runner.
# --------------------------------------------------------------------------
def _scenario(**over) -> EdocScenario:
    kw = dict(
        name="edoc_parallel",
        edge_ids=("A", "B"),
        edge_tail=("O", "O"),
        edge_head=("D", "D"),
        edge_fftt=np.array([100.0, 100.0]),
        edge_lanes=np.array([1, 2]),  # A is the 1-lane bottleneck, B free-flowing
        agent_ids=("v0", "v1", "v2", "v3"),
        agent_origin=("O", "O", "O", "O"),
        agent_dest=("D", "D", "D", "D"),
        agent_depart=np.array([0.0, 1.2, 2.4, 3.6]),
        engine="eclipse-sumo",
        engine_version="1.27.1",
        seed=42,
        semantic_config="duaIterate;l=20;aggregation=300",
        dt=300.0,
        n_intervals=1,
        departure_quantum=1.2,
        backlog_bound=60.0,
        separation_factor=5.0,
        floor_seconds=10.0,
        replay_deadline_s=30.0,
        walk_bound=2,
    )
    kw.update(over)
    return EdocScenario(**kw)


def _agent(aid, route, dep, tt, dd=0.0) -> ReplayAgent:
    return ReplayAgent(aid, dep, dep + dd + tt, route, tt, dd)


def _bundle_and_runner():
    plans = {
        "v0": (("A",), 0.0),
        "v1": (("A",), 1.2),
        "v2": (("B",), 2.4),
        "v3": (("B",), 3.6),
    }
    X = {
        "v0": _agent("v0", ("A",), 0.0, 150.0),
        "v1": _agent("v1", ("A",), 1.2, 150.0),
        "v2": _agent("v2", ("B",), 2.4, 100.0),
        "v3": _agent("v3", ("B",), 3.6, 100.0),
    }
    result = ReplayResult(
        canon_hash="fixed",
        agents=X,
        field_records={"A": {0: (150.0, 5.0)}},  # A loaded; B never loaded -> fftt
        flows={"A": {0: (2, 2)}, "B": {0: (2, 2)}},
        n_intervals=1,
    )

    def runner(_scenario, _plans):
        return result

    emitted = EmittedBundle(plans=plans, experienced=X, engine_version="1.27.1", seed=42)
    return emitted, runner, X, plans


# --------------------------------------------------------------------------
# canonicalization (R10)
# --------------------------------------------------------------------------
def test_canon_strips_generated_on_preserves_tripinfo_duration():
    ti = (
        b'<?xml version="1.0"?>\n<!-- generated on 2026-07-16T13:44:06 by Eclipse SUMO -->\n'
        b'<tripinfos>\n<tripinfo id="v0" duration="199.00"/>\n</tripinfos>\n'
    )
    out = canon.canonicalize_sumo("tripinfo_019.xml", ti)
    assert b"generated on" not in out
    assert b'duration="199.00"' in out  # tripinfo duration is REAL cost, preserved


def test_canon_strips_summary_duration_only():
    summ = (
        b'<!-- generated on 2026-07-16T13:44:06 by Eclipse SUMO -->\n'
        b'<summary>\n<step time="0.00" duration="0"/>\n</summary>\n'
    )
    out = canon.canonicalize_sumo("summary_019.xml", summ)
    assert b"generated on" not in out
    assert b"duration=" not in out  # summary duration is wall-clock, stripped


def test_canon_decompresses_gz():
    raw = b"<edges><edge id='A' traveltime='5'/></edges>"
    gz = gzip.compress(raw)
    assert canon.canonicalize_sumo("dump_300.0.xml.gz", gz) == raw


def test_canon_hash_surface_excludes_logs_and_is_timestamp_stable():
    # two "runs": identical sim-state, different generated-on timestamps + log text.
    def run(ts, log):
        return {
            "tripinfo_019.xml": f'<!-- generated on {ts} -->\n<tripinfo duration="9"/>'.encode(),
            "dump_300.0.xml.gz": gzip.compress(b"<edge traveltime='5'/>"),
            "iteration_019.sumo.log": log.encode(),  # excluded from the surface
            "driver.out": (log + "!").encode(),  # excluded
        }

    a = canon.hash_sumo_artifacts(run("2026-01-01T00:00:00", "wall 5ms"))
    b = canon.hash_sumo_artifacts(run("2099-12-31T23:59:59", "wall 9999ms"))
    assert a == b  # sim-state identical -> hash identical despite timestamp/log churn
    assert canon.is_hashed_artifact("tripinfo_019.xml")
    assert not canon.is_hashed_artifact("iteration_019.sumo.log")
    assert not canon.is_hashed_artifact("driver.out")


def test_canon_version_bump_moves_downstream_hashes(monkeypatch):
    arts = {"tripinfo.xml": b"<tripinfo duration='9'/>"}
    h1 = canon.hash_sumo_artifacts(arts)
    monkeypatch.setattr(canon, "_CANON_DOMAIN", b"tabench-edoc-canon-v2;")
    h2 = canon.hash_sumo_artifacts(arts)
    assert h1 != h2  # a version bump mints new instance hashes (disclosed, R10)


# --------------------------------------------------------------------------
# EdocScenario hash coverage (adr-024 mechanical lesson) + gates + golden hash
# --------------------------------------------------------------------------
def test_hash_coverage_every_field_participates():
    """Mechanical: mutate EACH field on a copy and assert content_hash moves —
    an uncovered dial is a byte-migration collision surface (adr-024)."""
    base = _scenario()
    base_h = base.content_hash()
    for f in dataclasses.fields(EdocScenario):
        val = getattr(base, f.name)
        if isinstance(val, np.ndarray):
            mutated = np.array(val, dtype=np.float64)
            mutated[0] += 1.0
        elif isinstance(val, tuple):
            mutated = ("ZZ_" + str(val[0]), *val[1:])
        elif isinstance(val, bool):
            mutated = not val
        elif isinstance(val, int):
            mutated = val + 1
        elif isinstance(val, float):
            mutated = val + 1.0
        elif isinstance(val, str):
            mutated = val + "_X"
        else:  # pragma: no cover - guard against a new field type
            raise AssertionError(f"unhandled field type for {f.name}: {type(val)}")
        clone = _scenario()
        object.__setattr__(clone, f.name, mutated)  # bypass gates: testing hash reads
        assert clone.content_hash() != base_h, f"field {f.name!r} not in content_hash"


def test_scenario_hash_stable_and_seed_sensitive():
    assert _scenario().content_hash() == _scenario().content_hash()
    assert _scenario().content_hash() != _scenario(seed=43).content_hash()


def test_scenario_frozen_and_readonly_arrays():
    s = _scenario()
    assert not s.edge_fftt.flags.writeable
    assert not s.agent_depart.flags.writeable
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.seed = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "over",
    [
        {"agent_depart": np.array([0.0, 1.25, 2.4, 3.6])},  # off the 1.2s grid
        {"separation_factor": 0.5},  # < 1
        {"field_semantics": "monotonized"},  # named future family, not shipped
        {"agent_origin": ("O", "O", "O", "X")},  # OD off the network
        {"edge_fftt": np.array([0.0, 100.0])},  # non-positive fftt
        {"edge_lanes": np.array([1, 0])},  # zero lanes (< 1, invalid numLanes)
        {"edge_lanes": np.array([1, 100_000])},  # explodes netconvert -> refuse at construction
        # F5: departure-window gate [0, dt*n_intervals) (adr-036 forgery pair 1/12).
        # horizon here = dt*n_intervals = 300*1 = 300 s; all on the 1.2s grid so the
        # window gate (not the grid gate) is what fires.
        {"agent_depart": np.array([0.0, 1.2, 2.4, 600.0])},  # on-grid, beyond horizon
        {"agent_depart": np.array([-1.2, 1.2, 2.4, 3.6])},  # on-grid, negative departure
        {"agent_depart": np.array([0.0, 1.2, 2.4, 300.0])},  # == horizon (half-open [0, 300))
    ],
)
def test_scenario_construction_gates_raise_valueerror(over):
    with pytest.raises(ValueError):
        _scenario(**over)


def test_departure_window_gate_accepts_within_horizon():
    """F5 boundary: a departure at horizon-epsilon is inside [0, dt*n_intervals)."""
    s = _scenario(agent_depart=np.array([0.0, 1.2, 2.4, 297.6]))  # < 300 s horizon
    assert s.n_agents == 4  # constructs (the gate is half-open, not off-by-one strict)


def test_does_not_move_the_golden_braess_hash():
    from tabench.data.builtin import braess_scenario

    assert braess_scenario().content_hash() == GOLDEN_BRAESS_HASH


# --------------------------------------------------------------------------
# frozen field + TD-SP regressions (MAJOR-1 / MAJOR-3)
# --------------------------------------------------------------------------
def test_tdsp_non_fifo_soundness():
    """A boundary-crossing walk (arrive M LATER, catch a cheaper interval) beats
    the label-setting earliest-arrival answer — enumeration finds it (305 not 405)."""
    out_edges = {"O": ["a", "a2"], "M": ["b"]}
    edge_head = {"a": "M", "a2": "M", "b": "D"}
    field = FrozenField(
        dt=300.0, n_intervals=2, fftt={"a": 5.0, "a2": 300.0, "b": 5.0},
        traveltime={"b": {0: 400.0, 1: 5.0}}, occupancy={"b": {0: 9.0, 1: 9.0}},
    )
    ow = build_origin_waits([], 300.0, 2)
    c_br = td_shortest_path(out_edges, edge_head, field, ow, "O", "D", 0.0, walk_bound=4)
    assert c_br == pytest.approx(305.0)  # not the unsound 405.0
    # invariant: the driven route is in the universe, so c_br <= c_drv
    c_drv_fast = evaluate_route(field, ow, ("a", "b"), 0.0)
    c_drv_slow = evaluate_route(field, ow, ("a2", "b"), 0.0)
    assert c_br <= c_drv_fast + 1e-9 and c_br <= c_drv_slow + 1e-9


def test_field_occupancy_aware_poison_resistance():
    """A burst congesting intervals 0 and 2 must NOT paint the empty interior
    interval 1 (zero occupancy -> free flow); a real standing queue carries forward."""
    poison = FrozenField(
        dt=100.0, n_intervals=3, fftt={"x": 10.0},
        traveltime={"x": {0: 100.0, 2: 100.0}}, occupancy={"x": {0: 5.0, 2: 5.0}},
    )
    assert poison.traversal_time("x", 150.0) == pytest.approx(10.0)  # gap -> free flow
    standing = FrozenField(
        dt=100.0, n_intervals=3, fftt={"x": 10.0},
        traveltime={"x": {0: 100.0, 2: 100.0}},
        occupancy={"x": {0: 5.0, 1: 5.0, 2: 5.0}},  # standing queue in the gap
    )
    assert standing.traversal_time("x", 150.0) == pytest.approx(100.0)  # carry-forward


def test_field_never_loaded_is_free_flow():
    f = build_field_from_records({}, {"z": 12.0}, dt=100.0, n_intervals=3)
    assert f.traversal_time("z", 250.0) == pytest.approx(12.0)


# --------------------------------------------------------------------------
# certifier: happy path + censor + raise (G0-G4, RG_D1)
# --------------------------------------------------------------------------
def test_certifier_happy_path_rg_d1_arithmetic():
    emitted, runner, _X, _plans = _bundle_and_runner()
    m = EdocEvaluator(_scenario(), runner).certify(emitted)
    assert m["feasible"] == 1.0
    # RG_D1 = ((150-100)+(150-100)+0+0)/(150+150+100+100) = 100/500 = 0.2
    assert m["rg_d1"] == pytest.approx(0.2)
    assert m["n_improvers"] == 2.0
    assert m["floor_gap"] == pytest.approx(10.0 / (500.0 / 4))  # 0.08
    assert m["sub_floor"] == 0.0  # 0.2 >= 0.08 -> ranked
    assert m["br_coverage"] == pytest.approx(0.5)  # A loaded, B not


def test_certifier_sub_floor_classification():
    emitted, runner, _X, _plans = _bundle_and_runner()
    # a large floor pushes floor_gap above rg_d1 -> the row is sub-floor, not ranked
    m = EdocEvaluator(_scenario(floor_seconds=40.0), runner).certify(emitted)
    assert m["floor_gap"] == pytest.approx(40.0 / 125.0)  # 0.32 > 0.2
    assert m["sub_floor"] == 1.0


def test_certifier_censors_doctored_x():
    emitted, runner, X, plans = _bundle_and_runner()
    bad = dict(X)
    bad["v0"] = _agent("v0", ("A",), 0.0, 90.0)  # claim 90s where replay says 150s
    m = EdocEvaluator(_scenario(), runner).certify(
        EmittedBundle(plans, bad, "1.27.1", 42)
    )
    assert m["feasible"] == 0.0 and np.isnan(m["rg_d1"])


def test_certifier_censors_demand_mismatch():
    emitted, runner, X, plans = _bundle_and_runner()
    dropped = {k: v for k, v in plans.items() if k != "v3"}
    m = EdocEvaluator(_scenario(), runner).certify(
        EmittedBundle(dropped, X, "1.27.1", 42)
    )
    assert m["feasible"] == 0.0


def test_certifier_raises_on_g0_pin_mismatch():
    emitted, runner, X, plans = _bundle_and_runner()
    with pytest.raises(ValueError):
        EdocEvaluator(_scenario(), runner).certify(
            EmittedBundle(plans, X, "1.26.0", 42)  # wrong engine version
        )


def test_certifier_raises_on_nondeterministic_replay():
    emitted, _runner, X, _plans = _bundle_and_runner()
    calls = {"n": 0}

    def flaky(_s, _p):
        calls["n"] += 1
        return ReplayResult(
            canon_hash=f"h{calls['n']}", agents=X,  # different hash each call
            field_records={"A": {0: (150.0, 5.0)}}, flows={}, n_intervals=1,
        )

    with pytest.raises(RuntimeError):
        EdocEvaluator(_scenario(), flaky).certify(emitted)


def test_assert_engine_pin_matches_and_mismatches():
    assert_engine_pin("1.27.1", "1.27.1")  # match: no raise
    with pytest.raises(ValueError, match="G0 engine pin"):
        assert_engine_pin("1.26.0", "1.27.1")  # installed != pinned -> RAISE


def test_certifier_raises_when_runner_reports_version_mismatch():
    """The runner contract (G0 split): a ReplayRunner that finds the installed
    engine != the instance pin RAISES ValueError, which propagates out of certify
    (a config error, never laundered into feasible=0)."""
    emitted, _runner, _X, _plans = _bundle_and_runner()
    sc = _scenario()

    def version_mismatch_runner(scenario, _plans):
        # what the real SUMO runner does before replaying: read the installed
        # engine version and assert it against the instance pin.
        assert_engine_pin("1.26.0", scenario.engine_version)
        raise AssertionError("unreachable: the pin check must raise first")

    with pytest.raises(ValueError, match="G0 engine pin"):
        EdocEvaluator(sc, version_mismatch_runner).certify(emitted)


def test_certifier_agent_symmetric_convention():
    """The agent-symmetric convention cancels the wait in the numerator but pads
    the denominator; here departDelay=0 so rg_d1 matches the profile default."""
    emitted, runner, _X, _plans = _bundle_and_runner()
    m = EdocEvaluator(_scenario(origin_wait_convention="agent_symmetric"), runner).certify(
        emitted
    )
    assert m["feasible"] == 1.0
    assert m["rg_d1"] == pytest.approx(0.2)


def test_certifier_censors_only_plan_replay_failure_not_infra():
    """F1 (crash-vs-censor, adr-036 R6): certify converts the runner's
    PlanReplayFailure (the emitted plan crashed/hung the engine) into a CENSOR, but
    a certifier-side infrastructure fault (a net-compile RuntimeError, a missing
    binary OSError) PROPAGATES — it is never laundered into feasible=0."""
    emitted, _runner, _X, _plans = _bundle_and_runner()
    sc = _scenario()

    def plan_hang_runner(_s, _p):  # the ONE censor case (R6 first arm)
        raise PlanReplayFailure("emitted plan hung the engine")

    m = EdocEvaluator(sc, plan_hang_runner).certify(emitted)
    assert m["feasible"] == 0.0 and np.isnan(m["rg_d1"])

    def infra_runtime_runner(_s, _p):  # a net-compile / deadline / read-back fault
        raise RuntimeError("netconvert infra crash on the hashed scenario")

    with pytest.raises(RuntimeError):
        EdocEvaluator(sc, infra_runtime_runner).certify(emitted)

    def missing_binary_runner(_s, _p):  # a bare OSError (engine binary gone)
        raise OSError("[Errno 2] engine binary not found")

    with pytest.raises(OSError):
        EdocEvaluator(sc, missing_binary_runner).certify(emitted)


def test_g2_half_quantum_departure_shift_censors():
    """F7: a departure shifted within +/-0.5*departure_quantum used to be accepted
    (demand-quantum-proportional slack = de-peaking timing freedom); the tightened
    exact-within-tol gate now CENSORS it (adr-036 forgery pair 10: zero freedom)."""
    emitted, runner, X, plans = _bundle_and_runner()
    sc = _scenario()  # departure_quantum = 1.2 -> the old gate allowed +/-0.6 s
    shifted = dict(plans)
    shifted["v0"] = (("A",), 0.0 + 0.5)  # trip table says exactly 0.0; 0.5 < 0.5*q
    m = EdocEvaluator(sc, runner).certify(EmittedBundle(shifted, X, "1.27.1", 42))
    assert m["feasible"] == 0.0  # was 1.0 under the half-quantum tolerance


def test_tdsp_docstring_states_certify_time_guard_not_construction_refusal():
    """F9c: the TD-SP module used to claim a construction-time walk-count refusal
    that does not exist; the amended docstring must state the certify-time
    infrastructure-guard truth (and must NOT claim construction refuses)."""
    from tabench.edoc import tdsp

    doc = tdsp.__doc__ + (td_shortest_path.__doc__ or "")
    assert "construction refuses" not in doc
    assert "certif" in doc.lower()  # names the certify-time guard
