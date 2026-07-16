"""Sumo-gated tests for the first EDOC-1 row: ``sumo-duaiterate`` (adr-036/adr-037).

``eclipse-sumo`` is an OPTIONAL extra; this whole file is skipped on a core install
(``pytest.importorskip('sumo')``) and runs only on the sumo CI leg. The EDOC
SUBSTRATE is exercised engine-free in ``tests/test_edoc.py`` (synthetic replay
fixtures); here we drive the REAL engine end to end on the pinned reference
instance and pin the row's load-bearing properties (VERSION-ROBUST — ranges and
inequalities, never exact split decimals, since the meso dynamics could shift
between wheel releases):

* emit -> certify is feasible and RG_D1 is a small, sane converged gap;
* the G1 replay is bit-deterministic across fresh net compiles (hazard #13);
* the negative control SEPARATES (AON control >> converged) by the declared factor,
  and a shared-edge bottleneck is REFUSED (a construction error, not a row);
* the duarouter R3 cross-check agrees with the substrate field arithmetic within
  the declared tolerance;
* the runner's G0 engine-pin RAISES on a version mismatch;
* temp-dir hygiene; and the golden Braess hash stays byte-identical.

See docs/design/adr-037-sumo-duaiterate.md.
"""

from __future__ import annotations

import dataclasses
import glob
import os
import signal
import tempfile
import time

import numpy as np
import pytest

pytest.importorskip("sumo")

import tabench.models.adapters.sumo_duaiterate as sd  # noqa: E402
from tabench.edoc.replay import PlanReplayFailure  # noqa: E402
from tabench.metrics.edoc_gaps import EdocEvaluator  # noqa: E402
from tabench.models.adapters._sumo_io import sumo_env  # noqa: E402
from tabench.models.adapters.sumo_duaiterate import (  # noqa: E402
    SumoDuaIterateAdapter,
    build_diamond_scenario,
    certify_emitted,
    compile_net,
    duarouter_recost_crosscheck,
    installed_engine_version,
    make_replay_runner,
    negative_control_separation,
    reference_scenario,
    shared_bottleneck_scenario,
)

BRAESS_GOLDEN_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"
_WALL = 300.0


@pytest.fixture(scope="module")
def converged():
    """One converged emit + certify on the reference instance, shared by the cheap
    assertions (keeps the sumo leg fast). Returns (scenario, emitted, metrics)."""
    sc = reference_scenario()
    emitted = SumoDuaIterateAdapter(iterations=18).emit(sc, wall_seconds=_WALL)
    runner = make_replay_runner(deadline=None)
    metrics = EdocEvaluator(sc, runner).certify(emitted)
    return sc, emitted, metrics


def test_installed_version_matches_the_reference_pin():
    assert reference_scenario().engine_version == installed_engine_version()


def test_emit_certify_reference_is_feasible(converged):
    sc, emitted, m = converged
    assert m["feasible"] == 1.0
    assert set(emitted.plans) == set(sc.agent_ids)  # two-sided demand bijection (G2/G3)
    assert 0.0 < m["rg_d1"] < 0.1  # a real, small converged dynamic gap
    assert m["delta"] <= sc.floor_seconds  # field represents experienced cost (resolution floor)
    assert m["max_backlog"] <= sc.backlog_bound
    assert m["br_coverage"] >= 0.9  # the driven routes are on loaded edges (Tier-B coverage)


def test_g1_replay_bit_deterministic(converged):
    """Hazard #13: the pinned replay's canonical artifact hash is byte-stable across
    two independent runs (each recompiles the net) — the G1 determinism double."""
    sc, emitted, _m = converged
    runner = make_replay_runner(deadline=None)
    r1 = runner(sc, emitted.plans)
    r2 = runner(sc, emitted.plans)
    assert r1.canon_hash == r2.canon_hash


def test_r3_duarouter_crosscheck_agrees(converged):
    sc, emitted, _m = converged
    runner = make_replay_runner(deadline=None)
    replay = runner(sc, emitted.plans)
    r3 = duarouter_recost_crosscheck(
        sc, emitted.plans, replay, deadline=None, tolerance_s=sc.r3_tolerance_s
    )
    assert r3["r3_max_s"] <= sc.r3_tolerance_s  # no RAISE: field arithmetic == engine router


def test_negative_control_separates():
    """AON control RG_D1 >> converged RG_D1 by the declared factor — the attributable
    negative control adr-030 said the dynamic track lacked."""
    sc = reference_scenario()
    anchors = negative_control_separation(sc, wall_seconds=_WALL, converged_iterations=15)
    assert anchors["separation"] >= sc.separation_factor
    assert anchors["control_rg_d1"] > anchors["converged_rg_d1"]
    # F10: a passing separation gate marks the FAMILY separation-vetted, which
    # certify_emitted then asserts before certifying.
    assert sc.family in sd._SEPARATION_VETTED_FAMILIES


def test_shared_edge_bottleneck_is_refused():
    """A drop on a SHARED edge does not separate control from converged -> the
    construction gate REFUSES it (ValueError), never a certified row."""
    sc = shared_bottleneck_scenario()
    with pytest.raises(ValueError, match="separation"):
        negative_control_separation(sc, wall_seconds=_WALL, converged_iterations=15)


def test_runner_engine_pin_raises_on_version_mismatch():
    """G0 split: emit reads the installed engine version and RAISES if it != the
    instance pin (a config error, never a censor)."""
    bad = dataclasses.replace(reference_scenario(), engine_version="0.0.0-nonexistent")
    with pytest.raises(ValueError, match="engine"):
        SumoDuaIterateAdapter(iterations=2).emit(bad, wall_seconds=_WALL)


def test_temp_dir_hygiene(converged):
    """emit + a certifier replay leave NO working tree behind. F9b: snapshot-diff
    every ``tabench-edoc-*`` dir before/after (the old ``tabench-edoc-[0-9]*`` glob
    saw neither the ``replay-``/``r3-`` prefixes nor ~84% of emit-dir suffixes)."""
    pat = tempfile.gettempdir() + "/tabench-edoc-*"
    before = set(glob.glob(pat))
    sc = reference_scenario()
    adapter = SumoDuaIterateAdapter(iterations=2)
    adapter.emit(sc, wall_seconds=_WALL)
    make_replay_runner(deadline=None)(sc, {a: (("a1", "a2"), float(d))
                                           for a, d in zip(sc.agent_ids, sc.agent_depart,
                                                           strict=True)})
    assert adapter.last_workdir is None
    assert set(glob.glob(pat)) - before == set()  # no new emit/replay/r3 dir survived


def test_does_not_move_the_golden_braess_hash():
    from tabench.data.builtin import braess_scenario

    assert braess_scenario().content_hash() == BRAESS_GOLDEN_HASH


def test_off_grid_departure_is_a_construction_error():
    """The pure-data EdocScenario gate still fires under the row (no engine needed):
    a departure off the declared quantum grid RAISES at construction."""
    sc = reference_scenario()
    bad_depart = np.array(sc.agent_depart, dtype=np.float64)
    bad_depart[0] += 0.37  # off the 2.0s grid
    with pytest.raises(ValueError, match="grid"):
        dataclasses.replace(sc, agent_depart=bad_depart)


# --------------------------------------------------------------------------
# fix-batch regression pins (S2 adversarial review; adr-036 R3/R6 + adr-037)
# --------------------------------------------------------------------------
def test_wall_kill_reaps_process_group():
    """F2: a wall-deadline kill reaps the WHOLE process group, not just the direct
    child — a SUMO tool spawns sumo/duarouter grandchildren that otherwise orphan
    and keep burning CPU. Scaled-down: a shell parent spawns a grandchild sleeper;
    after the deadline fires the grandchild must be gone (no orphan)."""
    d = tempfile.mkdtemp(prefix="tabench-edoc-killpin-")
    marker = os.path.join(d, "gc.pid")
    cmd = ["/bin/sh", "-c", f"sleep 30 & echo $! > {marker}; sleep 30"]
    try:
        with pytest.raises(RuntimeError, match="wall deadline"):
            sd._run(cmd, cwd=d, deadline=time.perf_counter() + 0.6, what="killpin")
        time.sleep(0.5)
        gc = int(open(marker).read().strip())
        with pytest.raises(ProcessLookupError):
            os.kill(gc, 0)  # reaped with the group -> no such process
    finally:
        try:
            gc = int(open(marker).read().strip())
            os.kill(gc, signal.SIGKILL)  # belt-and-suspenders cleanup on failure
        except (OSError, ValueError):
            pass
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def test_replay_deadline_s_is_enforced(converged):
    """F3: the hashed ``replay_deadline_s`` actually bounds a certifier replay. A
    1 ms declared deadline makes certify RAISE certifier-side infra (NOT a censor,
    per F1); and under a ``None`` caller wall the deadline is derived from the
    scenario field (was unbounded)."""
    sc, emitted, _m = converged
    derived = sd._intersect_replay_deadline(sc, None) - time.perf_counter()
    assert derived == pytest.approx(sc.replay_deadline_s, abs=1.0)  # live, not None

    tight = dataclasses.replace(sc, replay_deadline_s=0.001)
    runner = make_replay_runner(deadline=None)
    with pytest.raises((RuntimeError, OSError)) as ei:
        EdocEvaluator(tight, runner).certify(emitted)
    assert not isinstance(ei.value, PlanReplayFailure)  # infra RAISE, not the censor signal


def test_missing_binary_is_infra_raise_not_censor(converged, monkeypatch):
    """F1 (engine side): a missing engine binary during the certifier's replay is a
    certifier-side infrastructure fault that PROPAGATES, never laundered to
    feasible=0 (adr-036 R6 second arm)."""
    sc, emitted, _m = converged
    monkeypatch.setattr(sd, "sumo_binary", lambda name: f"/nonexistent/bin/{name}")
    with pytest.raises(RuntimeError) as ei:
        EdocEvaluator(sc, make_replay_runner(deadline=None)).certify(emitted)
    assert not isinstance(ei.value, PlanReplayFailure)


def test_certify_emitted_wires_r3_and_requires_separation_vetting(converged, monkeypatch):
    """F4 + F10: the ROW certify path runs the mandatory R3 duarouter cross-check
    and refuses an un-separation-vetted family."""
    sc, emitted, _m = converged

    # F10: an un-vetted family is refused loudly (before any engine work).
    unvetted = dataclasses.replace(sc, family="edoc-unvetted-pin")
    with pytest.raises(RuntimeError, match="separation-vetted"):
        certify_emitted(unvetted, emitted, wall_seconds=_WALL)

    sd._SEPARATION_VETTED_FAMILIES.add(sc.family)  # (negative_control_separation does this)

    # F4: certify_emitted invokes duarouter (poison it -> the cross-check must fire).
    real = sd.sumo_binary

    def poison(name):
        if name == "duarouter":
            raise AssertionError("duarouter invoked -> R3 fired")
        return real(name)

    monkeypatch.setattr(sd, "sumo_binary", poison)
    with pytest.raises(AssertionError, match="R3 fired"):
        certify_emitted(sc, emitted, wall_seconds=_WALL)
    monkeypatch.undo()

    # F4: a feasible certify emits R3 provenance and agrees within tolerance.
    m = certify_emitted(sc, emitted, wall_seconds=_WALL)
    assert m["feasible"] == 1.0
    assert m["r3_max_s"] <= sc.r3_tolerance_s
    assert "r3_mean_s" in m

    # F4: a forced R3 disagreement (tolerance 0) RAISES infra (never a censor).
    tight = dataclasses.replace(sc, r3_tolerance_s=0.0)
    sd._SEPARATION_VETTED_FAMILIES.add(tight.family)
    with pytest.raises(RuntimeError, match="R3 cross-check FAILED"):
        certify_emitted(tight, emitted, wall_seconds=_WALL)


def test_compile_refuses_subclamp_edge_and_readback_is_relative():
    """F6: an edge whose declared length ``fftt*canon_speed_mps`` sits below
    netconvert's 0.1 m min-length clamp is REFUSED at compile (the clamp can no
    longer silently corrupt free-flow time); and the read-back RAISES on a length
    drift beyond the RELATIVE tolerance (an absolute 0.5 m tolerance could never
    catch the sub-0.1 m clamp it named)."""
    sub = build_diamond_scenario("edoc-subclamp", n_agents=4, n_intervals=8,
                                 fftt_short=0.005, fftt_long=0.006)  # ~0.07 m edges
    wd = tempfile.mkdtemp(prefix="tabench-edoc-f6-")
    try:
        with pytest.raises(RuntimeError, match="min-length clamp"):
            compile_net(sub, wd, None)
    finally:
        import shutil
        shutil.rmtree(wd, ignore_errors=True)

    # synthetic read-back mismatch: a compiled length drifted > 1e-3 relative RAISES.
    wd = tempfile.mkdtemp(prefix="tabench-edoc-f6b-")
    net = os.path.join(wd, "net.net.xml")
    with open(net, "w") as fh:
        fh.write('<net><edge id="a1"><lane id="a1_0" length="100.5"/></edge></net>')
    try:
        with pytest.raises(RuntimeError, match="rewrote edge"):
            sd._readback_net(net, {"a1": 1}, {"a1": 100.0})  # declared 100 vs compiled 100.5
    finally:
        import shutil
        shutil.rmtree(wd, ignore_errors=True)


def test_sumo_env_pins_binaries_over_poisoned_ambient(monkeypatch):
    """F8: ``duaIterate.py`` locates binaries via ``sumolib.checkBinary``, which
    consults ``*_BINARY`` env vars BEFORE ``SUMO_HOME``; a poisoned ambient
    ``SUMO_BINARY`` would bypass the wheel-only rule. ``sumo_env`` pins those keys
    to the wheel's absolute binaries."""
    import sumo
    for var in ("SUMO_BINARY", "DUAROUTER_BINARY", "NETCONVERT_BINARY"):
        monkeypatch.setenv(var, f"/opt/sumo-1.12/bin/{var.split('_')[0].lower()}")
    env = sumo_env()
    wheel_bin = os.path.join(sumo.SUMO_HOME, "bin")
    for var in ("SUMO_BINARY", "DUAROUTER_BINARY", "NETCONVERT_BINARY"):
        assert env[var].startswith(wheel_bin)


def test_read_backs_wrap_parse_errors_as_runtime_error():
    """F9a: an unparseable artifact read-back RAISES the contract's RuntimeError
    (infra), not a raw xml ParseError — so the crash-vs-censor typing holds."""
    import xml.etree.ElementTree as ET

    d = tempfile.mkdtemp(prefix="tabench-edoc-f9a-")
    try:
        dump = os.path.join(d, "dump.xml.gz")
        with open(dump, "wb") as fh:
            fh.write(b"<<<not xml, not gzip>>>")
        with pytest.raises(RuntimeError, match="unparseable"):
            sd._parse_dump(dump, 300.0)

        routes = os.path.join(d, "r.rou.xml.gz")
        with open(routes, "wb") as fh:
            fh.write(b"garbage-not-gzip")
        with pytest.raises(RuntimeError, match="unparseable"):
            sd._read_plans(routes)

        ti = os.path.join(d, "tripinfo.xml")
        with open(ti, "w") as fh:
            fh.write("<tripinfos><bad")
        with pytest.raises(RuntimeError, match="unparseable"):
            sd._parse_tripinfo(ti)

        assert ET.ParseError is not RuntimeError  # the raw type must NOT escape
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
