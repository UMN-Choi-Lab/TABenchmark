"""Tests for the ``matsim`` EDOC row (adr-036/adr-039) — the first agent-based,
first stochastic-track external engine.

Two halves, one file (the row's whole surface in one place):

* **Engine-free** (top; run on every core leg): the writers, parsers, gates and
  probes of ``_matsim_io``/``matsim_edoc`` — no JVM, no jar. The engine-free
  EDOC substrate additions (MATSim canonicalizer, ``seed_list``, macroreps)
  live with the substrate in ``tests/test_edoc.py``.
* **Engine-gated** (bottom; ``@_requires_engine``): drive the REAL toolchain
  end to end on the pinned reference family. Gated per-test (not per-module)
  because the engine-free half must run on core legs; the matsim CI job sets
  ``TABENCH_MATSIM_HOME``/``TABENCH_JAVA_HOME`` so the whole file runs there.

Every measured anchor asserted here was derived with the SHIPPED estimator on
Temurin-21.0.11+10 + matsim-2025.0 (2026-07-17); see docs/design/adr-039.
"""

from __future__ import annotations

import dataclasses
import glob
import gzip
import os
import signal
import tempfile
import time

import numpy as np
import pytest

import tabench.models.adapters.matsim_edoc as me
from tabench.edoc.replay import PlanReplayFailure
from tabench.metrics.edoc_gaps import EdocEvaluator
from tabench.models.adapters._matsim_io import (
    matsim_available,
    parse_events,
    parse_output_plans,
)
from tabench.models.adapters.matsim_edoc import (
    MatsimAdapter,
    build_matsim_diamond_scenario,
    certify_emitted,
    certify_row,
    installed_engine_version,
    make_replay_runner,
    matsim_reference_scenario,
    matsim_shared_bottleneck_scenario,
    negative_control_separation,
    pinned_matsim_replay,
)

BRAESS_GOLDEN_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"
_WALL = 600.0
_ENGINE_FREE_VERSION = "matsim-2025.0;jar-md5=deadbeef;jdk-major=21"

_requires_engine = pytest.mark.skipif(
    not matsim_available(),
    reason="set TABENCH_MATSIM_HOME + TABENCH_JAVA_HOME to run the engine-gated half",
)


def _scenario(**over):
    over.setdefault("engine_version", _ENGINE_FREE_VERSION)
    return build_matsim_diamond_scenario("matsim-test", **over)


# ==========================================================================
# engine-free: parsers
# ==========================================================================
_EVENTS = b"""<?xml version="1.0" encoding="utf-8"?>
<events version="1.0">
\t<event time="10.0" type="departure" person="p1" link="home" legMode="car" />
\t<event time="12.0" type="vehicle enters traffic" person="p1" link="home" vehicle="p1" />
\t<event time="13.0" type="left link" link="home" vehicle="p1" />
\t<event time="13.0" type="entered link" link="a1" vehicle="p1" />
\t<event time="15.0" type="departure" person="p2" link="home" legMode="car" />
\t<event time="15.0" type="vehicle enters traffic" person="p2" link="home" vehicle="p2" />
\t<event time="16.0" type="left link" link="home" vehicle="p2" />
\t<event time="16.0" type="entered link" link="a1" vehicle="p2" />
\t<event time="20.0" type="departure" person="p3" link="home" legMode="car" />
\t<event time="20.0" type="vehicle enters traffic" person="p3" link="home" vehicle="p3" />
\t<event time="30.0" type="entered link" link="__ret0" vehicle="x9" />
\t<event time="40.0" type="left link" link="__ret0" vehicle="x9" />
\t<event time="103.0" type="left link" link="a1" vehicle="p1" />
\t<event time="103.0" type="entered link" link="work" vehicle="p1" />
\t<event time="113.0" type="vehicle leaves traffic" person="p1" link="work" vehicle="p1" />
\t<event time="113.0" type="arrival" person="p1" link="work" legMode="car" />
\t<event time="126.0" type="left link" link="a1" vehicle="p2" />
\t<event time="126.0" type="entered link" link="work" vehicle="p2" />
\t<event time="136.0" type="vehicle leaves traffic" person="p2" link="work" vehicle="p2" />
\t<event time="136.0" type="arrival" person="p2" link="work" legMode="car" />
</events>
"""


@pytest.fixture()
def parsed_events(tmp_path):
    p = tmp_path / "output_events.xml.gz"
    p.write_bytes(gzip.compress(_EVENTS))
    return parse_events(
        str(p), dt=60.0, n_intervals=4, edge_ids={"home", "a1", "work"}, t0=0.0
    )


def test_parse_events_agents_route_wait_and_door_to_door(parsed_events):
    agents, _field, _flows = parsed_events
    p1 = agents["p1"]
    assert p1.departure == 10.0 and p1.arrival == 113.0
    assert p1.depart_delay == pytest.approx(2.0)  # enters-traffic minus departure
    assert p1.experienced_time == pytest.approx(103.0)  # door-to-door, wait INCLUDED
    assert p1.route == ("home", "a1", "work")  # departure link + entered links
    # p3 never arrived (stuck at qsim end): absent -> the G3 census censors it
    assert "p3" not in agents and set(agents) == {"p1", "p2"}


def test_parse_events_field_interval_means_and_occupancy(parsed_events):
    _agents, field, _flows = parsed_events
    # a1 entry-interval-0 samples: p1 (13->103)=90, p2 (16->126)=110 -> mean 100
    tt, occ = field["a1"][0]
    assert tt == pytest.approx(100.0)
    assert occ == pytest.approx(2.0)
    # work entered at 103 (k=1) and 126 (k=2): full arrival-link traversal 10 s
    assert field["work"][1][0] == pytest.approx(10.0)
    assert field["work"][2][0] == pytest.approx(10.0)
    # engine-side plumbing (__ret0) is NEVER on the field surface
    assert "__ret0" not in field


def test_parse_events_flows_count_enters_traffic_and_arrivals(parsed_events):
    """Ruling 11 (the measured adr-036 correction): flows = entered-link +
    vehicle-enters-traffic in, left-link + vehicle-leaves-traffic out — the
    departure link would otherwise read zero inflow and the arrival link zero
    outflow. output_links.csv is NEVER a flow source (arrival-link undercount
    measured to zero) — this parser reads ONLY events."""
    _agents, _field, flows = parsed_events
    assert flows["home"][0] == (3.0, 2.0)  # 3 insertions (p3 too), 2 leaves
    assert flows["a1"][0] == (2.0, 0.0)
    assert flows["a1"][1] == (0.0, 1.0)  # p1 leaves a1 at 103
    assert flows["work"][1] == (1.0, 1.0)  # p1 in + out (arrival counts as leaving)
    assert "__ret0" not in flows


def test_parse_events_t0_offset_shifts_times_onto_the_scenario_axis(tmp_path):
    p = tmp_path / "ev.xml"
    shifted = _EVENTS
    for t in (b"136.0", b"126.0", b"113.0", b"103.0", b"40.0", b"30.0", b"20.0",
              b"16.0", b"15.0", b"13.0", b"12.0", b"10.0"):
        shifted = shifted.replace(
            b'time="' + t + b'"', b'time="' + str(float(t) + 3600.0).encode() + b'"'
        )
    p.write_bytes(shifted)
    agents, _f, _fl = parse_events(
        str(p), dt=60.0, n_intervals=4, edge_ids={"home", "a1", "work"}, t0=3600.0
    )
    assert agents["p1"].departure == 10.0 and agents["p1"].arrival == 113.0


def test_parse_output_plans_selected_route_and_depart(tmp_path):
    xml = b"""<?xml version="1.0" encoding="utf-8"?>
<population>
<person id="v0">
\t<plan score="1.0" selected="no">
\t\t<activity type="home" link="home" end_time="01:00:02"/>
\t\t<leg mode="car"><route type="links" start_link="home"
\t\t\tend_link="work">home b1 b2 work</route></leg>
\t\t<activity type="work" link="work"/>
\t</plan>
\t<plan score="2.0" selected="yes">
\t\t<activity type="home" link="home" end_time="01:00:02"/>
\t\t<leg mode="car"><route type="links" start_link="home"
\t\t\tend_link="work">home a1 a2 work</route></leg>
\t\t<activity type="work" link="work"/>
\t</plan>
</person>
</population>
"""
    p = tmp_path / "output_plans.xml.gz"
    p.write_bytes(gzip.compress(xml))
    plans = parse_output_plans(str(p), t0=3600.0)
    assert plans == {"v0": (("home", "a1", "a2", "work"), 2.0)}  # SELECTED plan only


def test_parsers_wrap_failures_as_runtime_error(tmp_path):
    """F9a: unparseable artifacts raise the contract RuntimeError, never a raw
    ParseError/BadGzipFile — the crash-vs-censor typing holds."""
    bad = tmp_path / "output_events.xml.gz"
    bad.write_bytes(b"\x1f\x8bnot really gzip")
    with pytest.raises(RuntimeError, match="unparseable"):
        parse_events(str(bad), dt=60.0, n_intervals=4, edge_ids=set(), t0=0.0)
    badplans = tmp_path / "output_plans.xml.gz"
    badplans.write_bytes(b"<population><person")
    with pytest.raises(RuntimeError, match="unparseable"):
        parse_output_plans(str(badplans))


# ==========================================================================
# engine-free: addressing probe (F8) + G0 pins
# ==========================================================================
def test_matsim_available_probe_is_env_addressed_and_side_effect_free(tmp_path, monkeypatch):
    monkeypatch.delenv("TABENCH_MATSIM_HOME", raising=False)
    monkeypatch.delenv("TABENCH_JAVA_HOME", raising=False)
    monkeypatch.delenv("JAVA_HOME", raising=False)
    assert matsim_available() is False  # unaddressed: no PATH fallback, ever

    (tmp_path / "matsim-2025.0").mkdir()
    (tmp_path / "matsim-2025.0" / "matsim-2025.0.jar").write_bytes(b"jar")
    jdk = tmp_path / "jdk" / "bin"
    jdk.mkdir(parents=True)
    java = jdk / "java"
    java.write_text("#!/bin/sh\nexit 0\n")
    java.chmod(0o755)
    monkeypatch.setenv("TABENCH_MATSIM_HOME", str(tmp_path))
    monkeypatch.setenv("TABENCH_JAVA_HOME", str(tmp_path / "jdk"))
    assert matsim_available() is True  # release-layout jar + executable java


def test_installed_engine_version_raises_unaddressed(monkeypatch):
    monkeypatch.delenv("TABENCH_MATSIM_HOME", raising=False)
    with pytest.raises(RuntimeError, match="unaddressed"):
        installed_engine_version()


def test_g0_full_jdk_pin_raises_on_drift(monkeypatch):
    """Ruling 8: the family-declared FULL JDK build (inside the hashed
    semantic_config) is enforced with a G0 RAISE against `java -version`."""
    monkeypatch.setattr(me, "java_version_string", lambda *a: 'Temurin-21.0.99+1 (build)')
    with pytest.raises(ValueError, match="full-JDK pin"):
        me._assert_jdk_pin()
    monkeypatch.setattr(
        me, "java_version_string", lambda *a: 'OpenJDK Temurin-21.0.11+10 (build 21.0.11+10-LTS)'
    )
    me._assert_jdk_pin()  # the pinned build passes


def test_semantic_config_carries_every_pinned_constant(monkeypatch):
    """Ruling 17 support: every certifier-side outcome-bearing family constant
    rides in the hashed semantic_config (the scenario hash-coverage test then
    covers them mechanically via the string field). A constant that falls out
    of the derivation would decouple from the instance hash — pinned here."""
    cfg = me._semantic_config()
    assert "threads=1" in cfg  # the corrected-R10 determinism pin
    assert "jdk=Temurin-21.0.11+10" in cfg  # the G0 full-JDK pin (ruling 8)
    assert "capPerLane=600" in cfg  # the engine-side capacity dial
    assert "strategies=ChangeExpBeta:0.7,ReRoute:0.3" in cfg  # route/selection ONLY
    assert "replayIt=0" in cfg  # the ONE replay-iteration constant (pair N6)
    assert "removeStuck=false" in cfg  # gridlock censors, never vanishes
    assert f"t0={me._MATSIM_T0:g}" in cfg
    # F2 (S3 review, executed: a marginal-utility drift changed 13/100 routes
    # with the hash unmoved): the certifier-written SCORING + ROUTER constants
    # are outcome-bearing and must ride in the derivation too.
    assert "router=SpeedyALT" in cfg
    assert "acts=home:12:00:00,work:08:00:00" in cfg
    assert "modeUtl=car:-6,pt:-6,walk:-6" in cfg
    sc = _scenario()
    assert sc.semantic_config == cfg  # the family builder uses the derived string
    # mutation half (F2): a drift in ANY pinned module constant moves the
    # derived string — and semantic_config is a hashed scenario field, so the
    # instance hash moves with it.
    for name, mutated in (
        ("_ROUTING_ALGORITHM", "Dijkstra"),
        ("_ACTIVITY_TYPICAL_DURATIONS", (("home", "12:00:00"), ("work", "01:00:00"))),
        ("_MODE_MARGINAL_UTILITY_HR", (("car", -6000.0), ("pt", -6.0), ("walk", -6.0))),
        ("_PINNED_FULL_JDK", "Temurin-21.0.12+7"),
        ("_MATSIM_CAP_PER_LANE_VPH", 900.0),
        ("_REPLAY_ITERATION", 1),
        ("_REPLANNING_STRATEGIES", (("ChangeExpBeta", 0.5), ("ReRoute", 0.5))),
    ):
        with monkeypatch.context() as mp:
            mp.setattr(me, name, mutated)
            assert me._semantic_config() != cfg, f"{name} drift did not move semantic_config"


# ==========================================================================
# engine-free: writers + construction gates
# ==========================================================================
def test_write_network_arithmetic_and_connectivity_plumbing(tmp_path):
    sc = _scenario()
    path = me._write_network(sc, str(tmp_path))
    text = open(path).read()
    # length = fftt * canon_speed; capacity = lanes * 600; permlanes = lanes
    assert f'<link id="a2" from="N1" to="D" length="{90.0 * 13.89:.6f}" ' in text
    assert 'capacity="600.000000"' in text  # the 1-lane bottleneck drop
    assert '<link id="home"' in text and 'capacity="2400.000000"' in text
    # exactly one deterministic return link D2 -> O0 (strong connectivity)
    assert text.count('<link id="__ret') == 1
    assert '<link id="__ret0" from="D2" to="O0"' in text


def test_readback_raises_on_capacity_drift(tmp_path):
    sc = _scenario()
    me._write_network(sc, str(tmp_path))
    doctored = open(tmp_path / "network.xml").read().replace(
        'capacity="600.000000"', 'capacity="900.000000"'
    )
    out = tmp_path / "output_network.xml.gz"
    out.write_bytes(gzip.compress(doctored.encode()))
    with pytest.raises(RuntimeError, match="read-back"):
        me._readback_network(str(out), sc)


def test_replay_config_derives_first_equals_last_from_one_constant(tmp_path):
    """Pair N6 (writer half): the replay config's firstIteration and
    lastIteration both come from _REPLAY_ITERATION — no code path can smuggle
    a replanning 'replay' without editing the hashed constant."""
    sc = _scenario()
    cfg = me._write_config(
        sc, str(tmp_path), first_it=me._REPLAY_ITERATION, last_it=me._REPLAY_ITERATION
    )
    text = open(cfg).read()
    assert f'<param name="firstIteration" value="{me._REPLAY_ITERATION}"/>' in text
    assert f'<param name="lastIteration" value="{me._REPLAY_ITERATION}"/>' in text
    assert '<param name="numberOfThreads" value="1"/>' in text
    assert text.count('numberOfThreads" value="1"') == 3  # global + qsim + eventsManager
    assert '<param name="removeStuckVehicles" value="false"/>' in text
    assert f'<param name="randomSeed" value="{sc.seed}"/>' in text


def test_config_writer_refuses_time_and_mode_mutating_strategies(tmp_path, monkeypatch):
    """G2 construction rule: a departure-time- or mode-mutating strategy in the
    pinned set is an EAGER config error (its faithful final iterate would break
    the exact-departure bijection)."""
    monkeypatch.setattr(
        me, "_REPLANNING_STRATEGIES", (("TimeAllocationMutator", 0.1), ("ReRoute", 0.9))
    )
    with pytest.raises(ValueError, match="mutates departure time"):
        me._write_config(_scenario(), str(tmp_path), first_it=0, last_it=5)
    monkeypatch.setattr(
        me, "_REPLANNING_STRATEGIES", (("SubtourModeChoice", 0.1), ("ReRoute", 0.9))
    )
    with pytest.raises(ValueError, match="mutates departure time"):
        me._write_config(_scenario(), str(tmp_path), first_it=0, last_it=5)


def test_plans_writer_gates(tmp_path):
    sc = _scenario()
    with pytest.raises(ValueError, match="empty route"):
        me._write_plans({"v0": ((), 0.0)}, sc, str(tmp_path))
    with pytest.raises(ValueError, match="integer-second"):
        me._write_plans({"v0": (("home", "a1", "a2", "work"), 0.5)}, sc, str(tmp_path))
    # ambiguous terminal links are a config error (the engine would get a
    # choice the instance does not hash)
    amb = _scenario()
    amb = dataclasses.replace(
        amb,
        edge_ids=(*amb.edge_ids, "home2"),
        edge_tail=(*amb.edge_tail, "O0"),
        edge_head=(*amb.edge_head, "O"),
        edge_fftt=np.append(amb.edge_fftt, 1.0),
        edge_lanes=np.append(amb.edge_lanes, 1),
    )
    with pytest.raises(ValueError, match="exactly one outgoing"):
        me._write_plans(None, amb, str(tmp_path))


def test_diamond_family_shape_and_gates():
    sc = _scenario()
    assert sc.seed_list == (42, 7, 123, 2024, 31337)  # exactly 5 (R5 floor == R7 bound)
    assert sc.departure_quantum == 1.0  # the engine-grid quantum (ruling 13)
    assert sc.walk_bound == 4  # driven routes are 4-edge walks -> in the universe
    assert sc.dt == 20.0 and sc.n_intervals == 90
    # off-grid departures refused by the pure-data gate (no engine needed)
    bad = np.array(sc.agent_depart)
    bad[0] += 0.37
    with pytest.raises(ValueError, match="grid"):
        dataclasses.replace(sc, agent_depart=bad)
    # the shared-edge refusal variant constructs (its refusal is engine-run)
    sh = matsim_shared_bottleneck_scenario(engine_version=_ENGINE_FREE_VERSION)
    assert sh.family == "matsim-shared-bottleneck"


def test_certify_emitted_vetting_is_topology_keyed_engine_free():
    """F10 fires BEFORE any engine work, so it pins engine-free — and the
    vetting key is the TOPOLOGY digest, not the family string (the S3 review's
    F3, executed: a never-vetted self-certifying shared-edge topology relabeled
    ``family='matsim-diamond'`` sailed past the name-keyed gate)."""
    from tabench.edoc.replay import EmittedBundle

    diamond = _scenario()
    shared = matsim_shared_bottleneck_scenario(engine_version=_ENGINE_FREE_VERSION)
    bundle = EmittedBundle(
        plans={}, experienced={}, engine_version=diamond.engine_version, seed=42
    )
    # an unvetted topology is refused outright
    with pytest.raises(RuntimeError, match="separation-vetted"):
        certify_emitted(shared, bundle)
    # simulate a legitimate diamond vetting (what negative_control_separation
    # does on success), then try to BORROW it via the family label (F3):
    diamond_digest = me._topology_digest(diamond)
    me._SEPARATION_VETTED_TOPOLOGIES.add(diamond_digest)
    try:
        relabeled = dataclasses.replace(shared, family=diamond.family)
        with pytest.raises(RuntimeError, match="separation-vetted"):
            certify_emitted(relabeled, bundle)  # still refused: wrong topology
        # while the genuinely vetted topology passes the gate (and then fails
        # later on the empty bundle's shapes — G2 censor, proving gate passage)
        m = certify_emitted(diamond, bundle)
        assert m["feasible"] == 0.0
    finally:
        me._SEPARATION_VETTED_TOPOLOGIES.discard(diamond_digest)


def test_wall_kill_reaps_process_group():
    """F2: a wall-deadline kill reaps the WHOLE process group — an orphaned JVM
    would idle at multi-hundred-MB RSS. Scaled-down: a shell parent spawns a
    grandchild sleeper; after the deadline fires the grandchild must be gone."""
    d = tempfile.mkdtemp(prefix="tabench-edoc-matsim-killpin-")
    marker = os.path.join(d, "gc.pid")
    cmd = ["/bin/sh", "-c", f"sleep 30 & echo $! > {marker}; sleep 30"]
    try:
        with pytest.raises(RuntimeError, match="wall deadline"):
            me._run(cmd, cwd=d, deadline=time.perf_counter() + 0.6, what="killpin")
        time.sleep(0.5)
        gc = int(open(marker).read().strip())
        with pytest.raises(ProcessLookupError):
            os.kill(gc, 0)  # reaped with the group -> no such process
    finally:
        try:
            gc = int(open(marker).read().strip())
            os.kill(gc, signal.SIGKILL)
        except (OSError, ValueError):
            pass
        import shutil

        shutil.rmtree(d, ignore_errors=True)


def test_replay_timeout_typing_scenario_deadline_censors_caller_clip_raises(
    tmp_path, monkeypatch
):
    """F1 (S3 review, MAJOR — executed repro: a ``now+1.5 s`` caller wall
    killed the replay JVM mid-startup and was laundered into ``feasible=0``):
    a mid-replay timeout censors (PlanReplayFailure) ONLY when the
    SCENARIO-declared ``replay_deadline_s`` was the binding wall; a tighter
    CALLER wall is a certifier-side budget exhaustion and RAISES RuntimeError.
    Engine-free: the 'engine' is a stub sleeper script."""
    fake_java = tmp_path / "jdk" / "bin" / "java"
    fake_java.parent.mkdir(parents=True)
    fake_java.write_text("#!/bin/sh\nsleep 30\n")
    fake_java.chmod(0o755)
    fake_jar = tmp_path / "matsim-2025.0.jar"
    fake_jar.write_bytes(b"stub")
    sc = _scenario()  # replay_deadline_s = 60 (the hashed family constant)
    monkeypatch.setattr(me, "java_binary", lambda: str(fake_java))
    monkeypatch.setattr(me, "matsim_jar", lambda: str(fake_jar))
    monkeypatch.setattr(me, "installed_engine_version", lambda: sc.engine_version)
    monkeypatch.setattr(me, "java_version_string", lambda *a: me._PINNED_FULL_JDK)
    plans = {"v0": (("home", "a1", "a2", "work"), 0.0)}

    # the flag itself (unit): scenario deadline binding vs caller-clipped
    d, clipped = me._intersect_replay_deadline(sc, None)
    assert not clipped and d - time.perf_counter() == pytest.approx(60.0, abs=1.0)
    _d, clipped = me._intersect_replay_deadline(sc, time.perf_counter() + 3600.0)
    assert not clipped  # a LOOSER caller wall never clips
    _d, clipped = me._intersect_replay_deadline(sc, time.perf_counter() + 0.5)
    assert clipped

    # caller-clipped kill -> infra RAISE, never the censor signal
    with pytest.raises(RuntimeError, match="wall deadline") as ei:
        pinned_matsim_replay(sc, plans, deadline=time.perf_counter() + 0.5)
    assert not isinstance(ei.value, PlanReplayFailure)

    # scenario-declared deadline expiry -> the R6 censor signal
    tight = dataclasses.replace(sc, replay_deadline_s=0.5)
    with pytest.raises(PlanReplayFailure, match="wall deadline"):
        pinned_matsim_replay(tight, plans, deadline=None)


# ==========================================================================
# engine-gated: the pinned toolchain end to end
# ==========================================================================
@pytest.fixture(scope="module")
def vetted_reference():
    """Run the family separation gate ONCE (it also separation-vets the family
    for certify_emitted/certify_row below). ~110 s: 2 states x 5 seeds."""
    sc = matsim_reference_scenario()
    anchors = negative_control_separation(sc, wall_seconds=_WALL)
    return sc, anchors


@pytest.fixture(scope="module")
def row_result(vetted_reference):
    """The row's score object: the full 5-seed macrorep certification (~75 s)."""
    sc, _anchors = vetted_reference
    return sc, certify_row(sc, iterations=10, wall_seconds=_WALL)


@pytest.fixture(scope="module")
def converged_emission(vetted_reference):
    sc, _ = vetted_reference
    return sc, MatsimAdapter(iterations=10).emit(sc, wall_seconds=_WALL)


@_requires_engine
def test_installed_version_matches_the_reference_pin():
    assert matsim_reference_scenario().engine_version == installed_engine_version()
    assert "matsim-2025.0;jar-md5=" in installed_engine_version()


@_requires_engine
def test_negative_control_separates_and_vets(vetted_reference):
    """The attributable negative control on the STOCHASTIC track: displayed
    mean-vs-mean over the pinned seed list (adr-039 ruling 4). Measured
    5.33x >= the declared 5.0 (byte-reproducible on the pinned toolchain)."""
    sc, anchors = vetted_reference
    assert anchors["separation"] >= sc.separation_factor
    assert anchors["control_rg_d1"] > anchors["converged_rg_d1"]
    # F10: vetting is recorded under the TOPOLOGY digest (adr-039 F3)
    assert me._topology_digest(sc) in me._SEPARATION_VETTED_TOPOLOGIES


@_requires_engine
def test_certify_row_is_feasible_ranked_and_ci_brackets_mean(row_result):
    sc, row = row_result
    m = row.metrics
    assert m["feasible"] == 1.0 and m["n_seeds"] == 5.0
    assert 0.0 < m["rg_d1_mean"] < 0.2  # a real, small converged stochastic gap
    assert m["rg_d1_ci_lo"] <= m["rg_d1_mean"] <= m["rg_d1_ci_hi"]
    assert m["rg_d1_ci_lo"] < m["rg_d1_ci_hi"]  # macrorep variance is real
    assert m["sub_floor"] == 0.0 and m["rg_d1_mean"] > m["floor_gap"]  # ranked
    assert m["delta_max"] <= sc.floor_seconds
    assert m["max_backlog_max"] <= sc.backlog_bound
    assert set(row.per_seed) == set(sc.seed_list)
    rgs = [row.per_seed[s]["rg_d1"] for s in sc.seed_list]
    assert len({round(v, 6) for v in rgs}) > 1  # seeds genuinely differ
    assert all(row.per_seed[s]["r3_max_s"] <= sc.r3_tolerance_s for s in sc.seed_list)


@_requires_engine
def test_g1_replay_deterministic_and_raw_byte_stable(converged_emission):
    """The G1 determinism double on the canonical hash — PLUS the stricter
    engine-gated BONUS check (never the gate): at the pinned threads=1 the raw
    decompressed event stream is byte-identical across independent replays
    (the corrected R10 record, adr-039)."""
    sc, emitted = converged_emission
    w1 = tempfile.mkdtemp(prefix="tabench-edoc-matsim-g1a-")
    w2 = tempfile.mkdtemp(prefix="tabench-edoc-matsim-g1b-")
    try:
        r1 = pinned_matsim_replay(sc, emitted.plans, deadline=None, workdir=w1)
        r2 = pinned_matsim_replay(sc, emitted.plans, deadline=None, workdir=w2)
        assert r1.canon_hash == r2.canon_hash  # the gate object
        ev1 = gzip.decompress(open(os.path.join(w1, "out", "output_events.xml.gz"), "rb").read())
        ev2 = gzip.decompress(open(os.path.join(w2, "out", "output_events.xml.gz"), "rb").read())
        assert ev1 == ev2  # raw-byte identity at threads=1 (bonus, not the gate)
    finally:
        import shutil

        shutil.rmtree(w1, ignore_errors=True)
        shutil.rmtree(w2, ignore_errors=True)


@_requires_engine
def test_replay_is_seed_independent_the_n2_record(converged_emission):
    """RECORD (adr-039, pair N2 / ruling 5): the zero-replanning replay of the
    SAME plans under a DIFFERENT pinned seed is canonically identical on the
    shipped family — the replay map is seed-INDEPENDENT, so cross-macrorep
    artifact reuse collapses to legal pair-11 optimization (a plan set robust
    under every seed's identical replay map scores its own honest mean; the
    per-seed EMISSIONS still differ because the co-evolution is seed-
    dependent, pinned by distinct per-seed rg in the row test). If this
    assertion ever flips, pair 5's per-seed replay defense has become
    discriminative — update the adr-039 record, not just this test."""
    sc, emitted = converged_emission
    r42 = pinned_matsim_replay(sc, emitted.plans, deadline=None)
    r7 = pinned_matsim_replay(
        dataclasses.replace(sc, seed=7), emitted.plans, deadline=None
    )
    # NOTE: canon hashes differ (output_config echoes the seed); the seed-
    # independence claim is about the SIMULATION STATE — events + agents.
    assert set(r42.agents) == set(r7.agents)
    for aid, a42 in r42.agents.items():
        a7 = r7.agents[aid]
        assert (a42.departure, a42.arrival, a42.route, a42.experienced_time) == (
            a7.departure, a7.arrival, a7.route, a7.experienced_time
        )


@_requires_engine
def test_doctored_x_censors(converged_emission):
    """Forgery pair 2 (self-report substitution): a doctored experienced record
    diverges from the pinned replay and censors."""
    from tabench.edoc.replay import EmittedBundle, ReplayAgent

    sc, emitted = converged_emission
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
def test_replay_config_forgery_diverges(vetted_reference):
    """Pair N6 (engine half): a patched 'replay' with lastIteration !=
    firstIteration lets replanning fire, and its output DIVERGES from the
    honest zero-replanning replay — it can never silently pass as G1's object.
    Probed on the AON control state, where ReRoute has everything to improve."""
    sc, _ = vetted_reference
    control = MatsimAdapter(iterations=0).emit(sc, wall_seconds=_WALL)
    honest = pinned_matsim_replay(sc, control.plans, deadline=None)
    wd = tempfile.mkdtemp(prefix="tabench-edoc-matsim-n6-")
    try:
        me._write_network(sc, wd)
        me._write_plans(control.plans, sc, wd)
        cfg = me._write_config(sc, wd, first_it=0, last_it=1)  # the smuggled replan
        me._run_java(
            ["org.matsim.core.controler.Controler", cfg],
            cwd=wd, deadline=None, what="n6 probe",
        )
        forged = parse_output_plans(os.path.join(wd, "out", "output_plans.xml.gz"),
                                    t0=me._MATSIM_T0)
        # the replanned iterate changed plans relative to the honest replay input
        assert forged != control.plans
        forged_replay = pinned_matsim_replay(sc, forged, deadline=None)
        assert forged_replay.canon_hash != honest.canon_hash
    finally:
        import shutil

        shutil.rmtree(wd, ignore_errors=True)


@_requires_engine
def test_shared_edge_bottleneck_is_refused():
    """The self-certifying shared-edge topology (RG_D1 = 0 on BOTH anchors,
    measured) must be REFUSED by the displayed-value separation gate — a
    construction error, never a certified row. Downscaled demand + converged
    iterations keep the refusal probe cheap; the refusal is topological, not
    scale-dependent (the full-size record is in adr-039)."""
    sc = matsim_shared_bottleneck_scenario(n_agents=60)
    with pytest.raises(ValueError, match="negative-control separation"):
        negative_control_separation(sc, wall_seconds=_WALL, converged_iterations=2)
    assert me._topology_digest(sc) not in me._SEPARATION_VETTED_TOPOLOGIES


@_requires_engine
def test_engine_pin_raises_on_version_mismatch():
    bad = dataclasses.replace(
        matsim_reference_scenario(), engine_version="matsim-9.9;jar-md5=0;jdk-major=99"
    )
    with pytest.raises(ValueError, match="engine"):
        MatsimAdapter(iterations=0).emit(bad, wall_seconds=_WALL)


@_requires_engine
def test_replay_deadline_s_is_enforced(converged_emission):
    """F3: the hashed replay_deadline_s bounds every certifier replay; a
    PRE-exhausted budget is an infra RAISE, never laundered into a censor
    (the timeout typing itself is pinned engine-free by the F1 stub test)."""
    sc, emitted = converged_emission
    deadline, clipped = me._intersect_replay_deadline(sc, None)
    assert not clipped  # no caller wall: the scenario deadline is binding
    assert deadline - time.perf_counter() == pytest.approx(
        sc.replay_deadline_s, abs=1.0
    )  # live, not None

    tight = dataclasses.replace(sc, replay_deadline_s=1e-7)
    with pytest.raises(RuntimeError) as ei:
        pinned_matsim_replay(tight, emitted.plans, deadline=None)
    assert not isinstance(ei.value, PlanReplayFailure)  # pre-exhaustion = infra


@_requires_engine
def test_unaddressed_toolchain_is_infra_raise_not_censor(converged_emission, monkeypatch):
    """F1/F8: certifying with the toolchain unaddressed RAISES (config/infra),
    never feasible=0."""
    sc, emitted = converged_emission
    monkeypatch.delenv("TABENCH_MATSIM_HOME", raising=False)
    with pytest.raises(RuntimeError) as ei:
        EdocEvaluator(sc, make_replay_runner(deadline=None)).certify(emitted)
    assert not isinstance(ei.value, PlanReplayFailure)


@_requires_engine
def test_temp_dir_hygiene(vetted_reference):
    """emit + a certifier replay leave NO working tree behind (snapshot-diff,
    the S2 F9b pattern). The glob is scoped to THIS process's pid-prefixed
    dirs (S3 review F5): the adapters mkdtemp under
    ``tabench-edoc-matsim-<pid>-*``, so a concurrent engine session on the
    same box (whose live workdirs flaked the old box-global glob, observed)
    can never appear in this snapshot. The check assumes nothing about other
    processes — only that THIS process cleans up after itself."""
    sc, _ = vetted_reference
    pat = tempfile.gettempdir() + f"/tabench-edoc-matsim-{os.getpid()}-*"
    before = set(glob.glob(pat))
    adapter = MatsimAdapter(iterations=0)
    emitted = adapter.emit(sc, wall_seconds=_WALL)
    pinned_matsim_replay(sc, emitted.plans, deadline=None)
    assert adapter.last_workdir is None
    assert set(glob.glob(pat)) - before == set()


@_requires_engine
def test_does_not_move_the_golden_braess_hash():
    from tabench.data.builtin import braess_scenario

    assert braess_scenario().content_hash() == BRAESS_GOLDEN_HASH
