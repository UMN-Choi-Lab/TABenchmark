"""SUMO ``duaIterate`` dynamic user-assignment as the first EDOC-1 row (adr-036/adr-037).

Unlike ``sumo-marouter`` (a STATIC macroscopic SUE whose gap is certified against
the scenario's DECLARED BPR — adr-027), ``duaIterate`` is a **dynamic mesoscopic**
assignment: it iterates ``duarouter`` best-response over a ``sumo`` meso dynamic
network load, and there is *no declared cost law* to certify against. The engine
IS the instance (adr-036): the certifier re-derives every scored number by
re-running the pinned engine on the model's emitted plans (G1), never trusting a
self-reported gap. This adapter is therefore an **EDOC producer**, not a
``TrafficAssignmentModel``: it emits the ADR's artifact contract — plans ``P``,
the door-to-door experienced record ``X``, and provenance — for
:class:`tabench.metrics.edoc_gaps.EdocEvaluator` to certify.

**The instance -> SUMO compile is deterministic and fully hashed.** The abstract
:class:`~tabench.edoc.scenario.EdocScenario` graph compiles to a SUMO net with edge
``length = fftt * canon_speed_mps`` (so free-flow time is ``fftt`` exactly),
``numLanes = edge_lanes`` (the engine-side capacity dial — meso flow capacity is a
function of lanes + freespeed + the pinned meso config), and ``speed =
canon_speed_mps``. Node coordinates are a deterministic layout and carry **no**
cost (explicit lengths override geometry; meso junction control is off), so the
whole net is a function of already-hashed fields.

**X is produced by THIS adapter's own pinned replay, never scraped from
duaIterate's internal tripinfo.** Measured on 1.27.1: a bare pinned ``sumo`` meso
replay is bit-deterministic (twin runs byte-identical), but does NOT reproduce
duaIterate's last-iteration tripinfo to the second (duaIterate's internal ``sumo``
call uses different options). So the adapter emits ``P`` = the duaIterate final
routes and derives ``X`` + the experienced-cost field from its OWN pinned replay of
``P``; the certifier's G1 re-runs the *identical* pinned replay and reproduces ``X``
by construction. The replay engine is the matched object (the A2 analogue, G1).

**adr-027 subprocess discipline (measured hazards, never laundered):**

* binaries and the ``tools/assign/duaIterate.py`` driver are addressed ONLY through
  ``sumo.SUMO_HOME`` (the wheel), never the stale ambient ``/opt/sumo-1.12``;
* ``stdin=DEVNULL``; a SINGLE wall deadline threads netconvert + every duaIterate
  iteration + the replay(s) + the duarouter R3 cross-check;
* a subprocess return code is NEVER trusted — every step re-reads the artifact it
  was supposed to write and RAISES ``RuntimeError`` (infra failure, never a
  ``feasible=0`` launder) if it is missing/unparseable, and a wall-budget kill is a
  ``RuntimeError``, not a censor;
* meso **teleport is disabled** (``--time-to-teleport -1``): a gridlocked instance
  runs to the declared horizon and shows as backlog / incomplete (censored by G3),
  rather than silently teleporting demand away (the ADR "head-block loss" hazard);
* tempdir hygiene: every run is a ``mkdtemp`` tree, ``rmtree`` in ``finally``.

``eclipse-sumo`` is an optional extra (``pip install tabench[sumo]``); this module
imports ``sumo`` and is guarded in ``models/__init__.py`` so the numpy/scipy core
stays dependency-free. Design: docs/design/adr-036/adr-037.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable

import numpy as np
import sumo

from ...edoc.field import build_origin_waits
from ...edoc.replay import (
    EmittedBundle,
    ReplayAgent,
    ReplayResult,
    assert_engine_pin,
)
from ...edoc.scenario import EdocScenario
from ...edoc.tdsp import evaluate_route
from ._subprocess import intersect_replay_deadline as _intersect_replay_deadline
from ._subprocess import remaining, run_disciplined
from ._sumo_io import sumo_binary, sumo_env

# netconvert silently clamps any edge shorter than this to it (the _sumo_io
# hazard). The row REFUSES at compile any edge whose declared length
# ``fftt * canon_speed_mps`` sits below the clamp times a small margin, so the
# clamp can never fire; the compile read-back (relative tolerance, the _sumo_io
# precedent) is the backstop for any other geometry rewrite.
_MIN_EDGE_LENGTH_M = 0.1
_LENGTH_TARGET_FACTOR = 1.05
_READBACK_LENGTH_RTOL = 1e-3  # _sumo_io._READBACK_LENGTH_RTOL precedent

# Topologies whose negative control has been separation-vetted by
# :func:`negative_control_separation` (adr-036 R-control / F10). The row's
# certification path (:func:`certify_emitted`) REFUSES to certify an instance
# whose TOPOLOGY digest is not in this set, so a non-separating topology cannot
# be silently certified — the procedural gate made structural. Keyed on the
# topology digest, NOT the family string (the S3 review's F3, an inherited S2
# defect: a never-vetted topology relabeled with a vetted family name passed a
# name-keyed gate) — runtime state only, no instance hash involved.
_SEPARATION_VETTED_TOPOLOGIES: set[str] = set()


def _topology_digest(scenario: EdocScenario) -> str:
    """The F10 vetting key (S3 review F3): what the separation gate vets is a
    TOPOLOGY — edge structure, lane (capacity) pattern, free-flow times and OD
    endpoints — so certification keys on this digest, never on the forgeable
    ``family`` STRING."""
    h = hashlib.sha256()
    h.update(b"tabench-edoc-vetting-v1;")
    for label, seq in (
        ("eid", scenario.edge_ids),
        ("etail", scenario.edge_tail),
        ("ehead", scenario.edge_head),
        ("aorg", scenario.agent_origin),
        ("adst", scenario.agent_dest),
    ):
        joined = "\x1f".join(seq).encode()
        h.update(f"{label}:{len(joined)};".encode())
        h.update(joined)
    lanes = np.ascontiguousarray(scenario.edge_lanes, dtype=np.int64)
    h.update(f"lanes:{lanes.size};".encode())
    h.update(lanes.tobytes())
    fftt = np.ascontiguousarray(scenario.edge_fftt, dtype=np.float64)
    h.update(f"fftt:{fftt.size};".encode())
    h.update(fftt.tobytes())
    return h.hexdigest()

# The certifier's zero-wait profile: R3 compares the substrate's DRIVEN field cost
# (no origin wait) to duarouter's driven re-cost (also no origin wait) — like-for-like.
_ZERO_WAIT = build_origin_waits([], dt=1.0, n_intervals=1)

__all__ = [
    "ENGINE",
    "SumoDuaIterateAdapter",
    "build_diamond_scenario",
    "certify_emitted",
    "compile_net",
    "duarouter_recost_crosscheck",
    "installed_engine_version",
    "make_replay_runner",
    "negative_control_separation",
    "pinned_meso_replay",
    "reference_scenario",
    "run_duaiterate",
    "shared_bottleneck_scenario",
]

ENGINE = "eclipse-sumo"

# The pinned meso option set (recorded in the instance's semantic_config, so it is
# hashed). Teleport OFF so gridlock censors (G3) instead of laundering demand away;
# single routing thread for byte-determinism (the marouter precedent). Kept minimal
# and explicit so a SUMO default drift is caught by the G0 version pin, not silently
# absorbed.
_MESO_OPTS: tuple[str, ...] = (
    "--mesosim",
    "--time-to-teleport", "-1",
    "--no-step-log", "true",
    "--xml-validation", "never",
)


def installed_engine_version() -> str:
    """The ``eclipse-sumo`` wheel version actually installed on this box (G0 read).

    ``importlib.metadata`` reads the wheel that ``import sumo`` resolves to — the
    same wheel whose ``SUMO_HOME`` addresses the binaries — so the version the
    certifier pins against is the version that will run."""
    return importlib.metadata.version(ENGINE)


def _duaiterate_tool() -> str:
    """Absolute path to the ``duaIterate.py`` driver inside the wheel (the
    wheel-home-only rule applies to ``tools/`` exactly as to ``bin/``, adr-027)."""
    return os.path.join(sumo.SUMO_HOME, "tools", "assign", "duaIterate.py")


# The subprocess discipline (wall deadline / process-group kill / crash-vs-censor
# typing) lives in ``_subprocess.py`` — the S2 shape, shared with the matsim and
# dtalite EDOC rows so a timeout/kill fix lands once. This row binds only the two
# engine-specific dials: the SUMO wheel env (``sumo_env()``) and the row label.
# ``_remaining`` / ``_intersect_replay_deadline`` are imported; ``_run`` is a thin
# wrapper. adr-027 subprocess hazards are documented in the module docstring above.
_LABEL = "sumo-duaiterate"


def _remaining(deadline: float | None) -> float | None:
    """This row's :func:`~tabench.models.adapters._subprocess.remaining` binding
    (RAISES if a compile/iterate phase already ate the whole wall budget)."""
    return remaining(deadline, label=_LABEL)


def _run(
    cmd: list[str], *, cwd: str, deadline: float | None, what: str,
    censor_on_fail: bool = False, censor_on_timeout: bool | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a SUMO subprocess under the shared EDOC discipline
    (:func:`~tabench.models.adapters._subprocess.run_disciplined`) with the wheel
    ``SUMO_HOME`` env (``sumo_env()``): the ONE meso-replay step passes
    ``censor_on_fail=True`` (an unexecutable plan is a
    :class:`~tabench.edoc.replay.PlanReplayFailure` censor) with the S3 F1
    ``censor_on_timeout`` caller-clip/scenario-deadline split preserved."""
    return run_disciplined(
        cmd, cwd=cwd, deadline=deadline, what=what, env=sumo_env(), label=_LABEL,
        censor_on_fail=censor_on_fail, censor_on_timeout=censor_on_timeout,
    )


# --------------------------------------------------------------------------
# instance -> SUMO net (deterministic, fully hashed)
# --------------------------------------------------------------------------
def _node_coords(scenario: EdocScenario) -> dict[str, tuple[float, float]]:
    """A deterministic 2-D node layout. Coordinates carry NO cost (explicit edge
    lengths override geometry, meso junction control is off); the grid only keeps
    netconvert reproducible and edges non-degenerate."""
    nodes = sorted(set(scenario.edge_tail) | set(scenario.edge_head))
    w = max(1, math.ceil(math.sqrt(len(nodes))))
    return {n: (300.0 * (i // w), 300.0 * (i % w)) for i, n in enumerate(nodes)}


def compile_net(scenario: EdocScenario, workdir: str, deadline: float | None) -> str:
    """Compile the abstract instance to a SUMO ``net.net.xml`` via ``netconvert``.

    Edge ``length = fftt * canon_speed`` (free-flow time == fftt), ``numLanes =
    edge_lanes``, ``speed = canon_speed``. Any edge whose declared length would sit
    below ``netconvert``'s 0.1 m min-length clamp (times a small margin) is REFUSED
    here at compile — an infrastructure RAISE, so the clamp can never silently
    corrupt free-flow time; the compile **read-back** (relative tolerance, the
    _sumo_io precedent) is the backstop that RAISES if any ``numLanes``/``length``
    was otherwise rewritten."""
    coords = _node_coords(scenario)
    lengths = scenario.length_of()
    lanes = scenario.lanes_of()
    v = float(scenario.canon_speed_mps)

    # Refuse below the netconvert min-length clamp * margin (F6): with the clamp
    # unreachable, a surviving read-back mismatch can only be some OTHER rewrite.
    floor_m = _MIN_EDGE_LENGTH_M * _LENGTH_TARGET_FACTOR
    for eid, length in lengths.items():
        if length < floor_m:
            raise RuntimeError(
                f"compile refuses edge {eid!r}: declared length {length:.5g} m "
                f"(fftt * canon_speed_mps) is below netconvert's {_MIN_EDGE_LENGTH_M} m "
                f"min-length clamp * {_LENGTH_TARGET_FACTOR} = {floor_m:.5g} m — the clamp "
                "would silently corrupt free-flow time; raise fftt or canon_speed_mps"
            )

    nod = ["<nodes>"]
    for n, (x, y) in coords.items():
        nod.append(f'  <node id="{n}" x="{x:.3f}" y="{y:.3f}"/>')
    nod.append("</nodes>")

    edg = ["<edges>"]
    for eid, tail, head in zip(
        scenario.edge_ids, scenario.edge_tail, scenario.edge_head, strict=True
    ):
        # a per-edge perpendicular bow keyed by id keeps parallel same-endpoint
        # edges geometrically distinct; the explicit length overrides its effect.
        edg.append(
            f'  <edge id="{eid}" from="{tail}" to="{head}" numLanes="{lanes[eid]}" '
            f'speed="{v:.6f}" length="{lengths[eid]:.6f}"/>'
        )
    edg.append("</edges>")

    nod_path = os.path.join(workdir, "nodes.nod.xml")
    edg_path = os.path.join(workdir, "edges.edg.xml")
    net_path = os.path.join(workdir, "net.net.xml")
    with open(nod_path, "w") as fh:
        fh.write("\n".join(nod))
    with open(edg_path, "w") as fh:
        fh.write("\n".join(edg))

    _run(
        [
            sumo_binary("netconvert"),
            "--node-files", nod_path,
            "--edge-files", edg_path,
            "--no-turnarounds", "true",
            "--offset.disable-normalization", "true",
            "--precision", "6",
            "--output-file", net_path,
        ],
        cwd=workdir, deadline=deadline, what="netconvert",
    )
    if not os.path.exists(net_path):  # rc is never trusted
        raise RuntimeError("netconvert reported success but wrote no net.net.xml")

    _readback_net(net_path, lanes, lengths)
    return net_path


def _readback_net(
    net_path: str, lanes: dict[str, int], lengths: dict[str, float]
) -> None:
    """Compile read-back (rc never trusted): re-parse ``net.net.xml`` and verify
    every scenario edge's ``numLanes`` and ``length`` survived netconvert. A dropped
    edge, a lane-count change, or a length drift beyond the RELATIVE tolerance
    (``_READBACK_LENGTH_RTOL``, the _sumo_io precedent — an absolute tolerance can
    never catch the sub-0.1 m clamp it names) RAISES ``RuntimeError`` (infra). An
    unparseable net RAISES the same contract type, not a raw ``ParseError`` (F9a)."""
    try:
        root = ET.parse(net_path).getroot()
    except ET.ParseError as exc:
        raise RuntimeError(f"compile read-back: net.net.xml unparseable ({exc})") from exc
    seen: dict[str, tuple[int, float]] = {}
    for e in root.findall("edge"):
        eid = e.get("id")
        if eid is None or eid.startswith(":") or eid not in lanes:
            continue  # internal/connector edges are not scenario edges
        lane_els = e.findall("lane")
        length = float(lane_els[0].get("length")) if lane_els else -1.0
        seen[eid] = (len(lane_els), length)
    for eid, want_lanes in lanes.items():
        if eid not in seen:
            raise RuntimeError(f"netconvert dropped scenario edge {eid!r} from the net")
        n_lane, length = seen[eid]
        if n_lane != want_lanes:
            raise RuntimeError(
                f"netconvert changed edge {eid!r} lanes {want_lanes} -> {n_lane}"
            )
        if abs(length - lengths[eid]) > _READBACK_LENGTH_RTOL * max(lengths[eid], 1e-9):
            raise RuntimeError(
                f"netconvert rewrote edge {eid!r} length {lengths[eid]:.6f} -> {length:.6f} "
                f"(> {_READBACK_LENGTH_RTOL} relative; the compiled free-flow time would "
                "differ from the hashed fftt)"
            )


def _write_trips(scenario: EdocScenario, workdir: str) -> str:
    """Per-agent junction-to-junction trips (route choice happens over scenario
    edges, so the emitted routes are pure scenario-edge walks — no src/sink edges,
    which would break G2 route validity). Sorted by (depart, id) — sumo requires a
    depart-sorted route file."""
    rows = sorted(
        zip(
            scenario.agent_ids, scenario.agent_origin, scenario.agent_dest,
            scenario.agent_depart, strict=True,
        ),
        key=lambda r: (float(r[3]), r[0]),
    )
    lines = ["<routes>"]
    for aid, o, d, dep in rows:
        lines.append(
            f'  <trip id="{aid}" depart="{float(dep):.2f}" '
            f'fromJunction="{o}" toJunction="{d}"/>'
        )
    lines.append("</routes>")
    path = os.path.join(workdir, "trips.trips.xml")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path


# --------------------------------------------------------------------------
# duaIterate assignment (produce the plans P)
# --------------------------------------------------------------------------
def run_duaiterate(
    scenario: EdocScenario,
    net_path: str,
    trips_path: str,
    workdir: str,
    deadline: float | None,
    *,
    iterations: int,
) -> str:
    """Run ``duaIterate.py`` for ``iterations`` steps and return the final
    iteration's chosen-routes file (the plans ``P``: per-vehicle route + depart).

    The engine's own convergence print is provenance ONLY (adr-036) — never gated;
    the assignment quality is measured by the certifier's RG_D1, not trusted here."""
    seed = int(scenario.seed)
    cmd = [
        sys.executable, _duaiterate_tool(),
        "-n", net_path,
        "-t", trips_path,
        "-l", str(int(iterations)),
        "-m",  # mesosim
        "--aggregation", repr(float(scenario.dt)),
        "--max-convergence-deviation", "1e-9",
        "duarouter--seed", str(seed),
        "duarouter--junction-taz", "true",
        "sumo--seed", str(seed),
    ]
    _run(cmd, cwd=workdir, deadline=deadline, what="duaIterate")
    iters = sorted(d for d in os.listdir(workdir) if d.isdigit() and len(d) == 3)
    if not iters:
        raise RuntimeError("duaIterate reported success but produced no iteration directory")
    last = os.path.join(workdir, iters[-1])
    routes = [f for f in os.listdir(last) if f.endswith(".rou.xml.gz") and ".rou.alt." not in f]
    if not routes:
        raise RuntimeError(f"duaIterate final iteration {iters[-1]} has no chosen-routes file")
    return os.path.join(last, routes[0])


# --------------------------------------------------------------------------
# the pinned meso replay (produces X + the field; the certifier's G1 runner)
# --------------------------------------------------------------------------
def _write_routes(plans: dict[str, tuple[tuple[str, ...], float]], workdir: str) -> str:
    """Write emitted plans ``{aid: (route, depart)}`` as a depart-sorted SUMO
    ``.rou.xml`` with fixed per-vehicle routes (zero replanning at replay)."""
    rows = sorted(((float(dep), aid, route) for aid, (route, dep) in plans.items()))
    lines = ["<routes>"]
    for dep, aid, route in rows:
        edges = " ".join(route)
        lines.append(
            f'  <vehicle id="{aid}" depart="{dep:.2f}"><route edges="{edges}"/></vehicle>'
        )
    lines.append("</routes>")
    path = os.path.join(workdir, "plans.rou.xml")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path


def _parse_tripinfo(path: str) -> dict[str, ReplayAgent]:
    """Parse a ``tripinfo`` file into per-agent replayed records. Experienced time
    is door-to-door: ``duration + departDelay`` (G3); ``departure`` is the scheduled
    depart (``tripinfo depart - departDelay``), so ``arrival - departure ==
    experienced_time``. Routes are filled in by the caller from the plans."""
    agents: dict[str, ReplayAgent] = {}
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise RuntimeError(f"tripinfo at {path} is unparseable ({exc})") from exc
    for ti in root.findall("tripinfo"):
        aid = ti.get("id")
        dd = float(ti.get("departDelay", "0"))
        actual_depart = float(ti.get("depart"))
        duration = float(ti.get("duration"))
        arrival = float(ti.get("arrival"))
        scheduled = actual_depart - dd
        agents[aid] = ReplayAgent(
            agent_id=aid,
            departure=scheduled,
            arrival=arrival,
            route=(),  # filled from the plans (the driven edge sequence)
            experienced_time=duration + dd,
            depart_delay=dd,
        )
    return agents


def _parse_dump(path: str, dt: float) -> tuple[
    dict[str, dict[int, tuple[float, float]]], dict[str, dict[int, tuple[float, float]]], int
]:
    """Parse a meso ``edgeData`` dump (possibly ``.gz``) into the certifier's
    model-blind field + flow records. ``field_records[edge][k] = (traveltime,
    occupancy)`` for loaded edge-intervals; ``flows[edge][k] = (entered+departed,
    left+arrived)`` so conservation holds at origins (insertions ENTER the edge) and
    sinks (arrivals LEAVE the network) — else G4 would false-censor."""
    with open(path, "rb") as fh:
        data = fh.read()
    try:
        if len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B:
            data = gzip.decompress(data)
        root = ET.fromstring(data)
    except (ET.ParseError, gzip.BadGzipFile, OSError) as exc:
        raise RuntimeError(f"meso dump at {path} is unparseable ({exc})") from exc
    field_records: dict[str, dict[int, tuple[float, float]]] = {}
    flows: dict[str, dict[int, tuple[float, float]]] = {}
    n_intervals = 0
    for iv in root.findall("interval"):
        k = int(round(float(iv.get("begin")) / dt))
        n_intervals = max(n_intervals, k + 1)
        for e in iv.findall("edge"):
            eid = e.get("id")
            tt = e.get("traveltime")
            occ = float(e.get("occupancy", "0"))
            if tt is not None:
                field_records.setdefault(eid, {})[k] = (float(tt), occ)
            entered = float(e.get("entered", "0")) + float(e.get("departed", "0"))
            left = float(e.get("left", "0")) + float(e.get("arrived", "0"))
            if entered or left:
                flows.setdefault(eid, {})[k] = (entered, left)
    return field_records, flows, n_intervals


def pinned_meso_replay(
    scenario: EdocScenario,
    plans: dict[str, tuple[tuple[str, ...], float]],
    *,
    deadline: float | None,
    net_path: str | None = None,
    workdir: str | None = None,
) -> ReplayResult:
    """One pinned ``sumo`` meso replay of ``plans`` with zero replanning (the G1
    matched object). Asserts the installed engine == the instance pin BEFORE
    running (the runner contract), compiles the net deterministically from the
    scenario if not supplied, runs the fixed-route replay, and parses tripinfo +
    dump into a :class:`ReplayResult` whose ``canon_hash`` is the G1 determinism
    object (over the canonicalized sim-state artifacts, R10).

    The scenario-declared hashed ``replay_deadline_s`` ALWAYS bounds this call
    (F3): if the caller passes no wall the deadline is derived from it, and a
    tighter caller wall wins (intersection). So the hashed hard-deadline constant
    is what actually stops a head-blocking replay, not an ad-hoc caller argument."""
    assert_engine_pin(installed_engine_version(), scenario.engine_version)
    deadline, clipped_by_caller = _intersect_replay_deadline(scenario, deadline)

    own_tmp = workdir is None
    # pid-scoped prefix (S3 review F5): hygiene snapshots diff only their own
    # process's dirs, so concurrent sessions on one box cannot cross-flake.
    workdir = workdir or tempfile.mkdtemp(prefix=f"tabench-edoc-{os.getpid()}-replay-")
    try:
        if net_path is None:
            net_path = compile_net(scenario, workdir, deadline)
        routes_path = _write_routes(plans, workdir)
        add_path = os.path.join(workdir, "edgedata.add.xml")
        dump_path = os.path.join(workdir, "dump.xml.gz")
        with open(add_path, "w") as fh:
            fh.write(
                f'<additional>\n  <edgeData id="dump" freq="{float(scenario.dt)!r}" '
                f'file="dump.xml.gz" excludeEmpty="true" minSamples="1"/>\n</additional>\n'
            )
        tripinfo_path = os.path.join(workdir, "tripinfo.xml")
        end = float(scenario.n_intervals) * float(scenario.dt)
        _run(
            [
                sumo_binary("sumo"),
                "--net-file", net_path,
                "--route-files", routes_path,
                "--additional-files", add_path,
                "--tripinfo-output", tripinfo_path,
                "--seed", str(int(scenario.seed)),
                "--begin", "0",
                "--end", repr(end),
                *_MESO_OPTS,
            ],
            cwd=workdir, deadline=deadline, what="sumo meso replay",
            # an engine CRASH here is always the plan's fault (R6 censor); a
            # TIMEOUT censors only when the SCENARIO deadline was binding — a
            # caller-clipped wall is a certifier budget fault, infra RAISE (F1).
            censor_on_fail=True,
            censor_on_timeout=not clipped_by_caller,
        )
        if not os.path.exists(tripinfo_path):
            raise RuntimeError("sumo meso replay reported success but wrote no tripinfo")
        if not os.path.exists(dump_path):
            raise RuntimeError("sumo meso replay reported success but wrote no dump")

        agents = _parse_tripinfo(tripinfo_path)
        routes = {aid: tuple(route) for aid, (route, _dep) in plans.items()}
        agents = {
            aid: ReplayAgent(
                agent_id=a.agent_id, departure=a.departure, arrival=a.arrival,
                route=routes.get(aid, ()), experienced_time=a.experienced_time,
                depart_delay=a.depart_delay,
            )
            for aid, a in agents.items()
        }
        field_records, flows, n_from_dump = _parse_dump(dump_path, float(scenario.dt))
        with open(tripinfo_path, "rb") as fh:
            tripinfo_bytes = fh.read()
        with open(dump_path, "rb") as fh:
            dump_bytes = fh.read()
        canon_hash = _hash_replay(tripinfo_bytes, dump_bytes)
        return ReplayResult(
            canon_hash=canon_hash,
            agents=agents,
            field_records=field_records,
            flows=flows,
            n_intervals=max(n_from_dump, int(scenario.n_intervals)),
        )
    finally:
        if own_tmp:
            shutil.rmtree(workdir, ignore_errors=True)


def _hash_replay(tripinfo_bytes: bytes, dump_bytes: bytes) -> str:
    """G1 determinism object: the canonicalized sim-state hash over the replay's
    experienced record + field (R10 surface — engine/driver logs are excluded)."""
    from ...edoc.canon import hash_sumo_artifacts

    return hash_sumo_artifacts({"tripinfo.xml": tripinfo_bytes, "dump.xml.gz": dump_bytes})


def make_replay_runner(*, deadline: float | None) -> Callable[..., ReplayResult]:
    """A :data:`~tabench.edoc.replay.ReplayRunner` bound to the single wall deadline,
    for injection into :class:`~tabench.metrics.edoc_gaps.EdocEvaluator`. Each call
    compiles the net fresh from the scenario (a deterministic function of hashed
    fields), so the determinism double and the emit-time replay all agree."""
    def runner(
        scenario: EdocScenario, plans: dict[str, tuple[tuple[str, ...], float]]
    ) -> ReplayResult:
        return pinned_meso_replay(scenario, plans, deadline=deadline)

    return runner


# --------------------------------------------------------------------------
# R3: duarouter cross-check (adr-036 R3; deliverable 4)
# --------------------------------------------------------------------------
def duarouter_recost_crosscheck(
    scenario: EdocScenario,
    plans: dict[str, tuple[tuple[str, ...], float]],
    replay: ReplayResult,
    *,
    deadline: float | None,
    tolerance_s: float,
) -> dict[str, float]:
    """R3 cross-check: re-cost the driven plans with the pinned ``duarouter`` on the
    replay's frozen dump, and compare to the substrate TD-SP's *driven* field cost
    on the SAME field (a like-for-like check of the harness field arithmetic against
    the engine's own weight reader — no routing, no origin wait on either side).

    A disagreement beyond ``tolerance_s`` (a hashed instance constant) is a harness
    correctness failure and RAISES — the field the certifier scores on must be the
    field the engine itself reads. Returns the measured mean/max discrepancy."""
    from ...edoc.field import build_field_from_records

    # the clip flag is irrelevant here: every R3 step is infra-typed anyway.
    deadline, _clipped = _intersect_replay_deadline(scenario, deadline)
    workdir = tempfile.mkdtemp(prefix=f"tabench-edoc-{os.getpid()}-r3-")
    try:
        net_path = compile_net(scenario, workdir, deadline)
        # re-emit the driven plans + the frozen dump for duarouter to re-cost.
        routes_path = _write_routes(plans, workdir)
        dump_path = os.path.join(workdir, "dump.xml.gz")
        _rewrite_dump(replay, dump_path, float(scenario.dt))
        recost_path = os.path.join(workdir, "recost.rou.xml")
        _run(
            [
                sumo_binary("duarouter"),
                "--net-file", net_path,
                "--route-files", routes_path,
                "--weight-files", dump_path,
                "--weight-attribute", "traveltime",
                "--keep-all-routes", "true",
                "--skip-new-routes", "true",
                "--write-costs", "true",
                "--output-file", recost_path,
                "--seed", str(int(scenario.seed)),
                "--routing-threads", "1",
                "--junction-taz", "true",
                "--xml-validation", "never",
            ],
            cwd=workdir, deadline=deadline, what="duarouter R3 re-cost",
        )
        if not os.path.exists(recost_path):
            raise RuntimeError("duarouter R3 reported success but wrote no re-cost output")
        engine_cost = _parse_route_costs(recost_path)

        field = build_field_from_records(
            replay.field_records, scenario.fftt_of(), scenario.dt, scenario.n_intervals,
            scenario.field_semantics,
        )
        trip = {
            aid: float(dep)
            for aid, _o, _d, dep in zip(
                scenario.agent_ids, scenario.agent_origin, scenario.agent_dest,
                scenario.agent_depart, strict=True,
            )
        }
        diffs = []
        for aid, (route, _dep) in plans.items():
            if aid not in engine_cost:
                raise RuntimeError(f"duarouter R3 did not re-cost agent {aid!r}")
            # substrate DRIVEN cost on the same field, no origin wait (like-for-like).
            mine = evaluate_route(field, _ZERO_WAIT, tuple(route), trip[aid])
            diffs.append(abs(engine_cost[aid] - mine))
        mean_d = sum(diffs) / len(diffs) if diffs else 0.0
        max_d = max(diffs, default=0.0)
        if max_d > tolerance_s:
            raise RuntimeError(
                f"R3 cross-check FAILED: duarouter re-cost vs substrate field cost differ by "
                f"max {max_d:.3f}s (mean {mean_d:.3f}s) > tolerance {tolerance_s}s — the harness "
                "field arithmetic disagrees with the engine's own weight reader"
            )
        return {"r3_mean_s": mean_d, "r3_max_s": max_d, "r3_tolerance_s": float(tolerance_s)}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _rewrite_dump(replay: ReplayResult, path: str, dt: float) -> None:
    """Serialize the parsed replay field back to a minimal meso ``meandata`` dump so
    ``duarouter`` re-costs on exactly the certifier's field (not a second engine
    run). Only ``traveltime`` is needed for the re-cost weight."""
    ks = sorted({k for per in replay.field_records.values() for k in per})
    lines = ['<meandata>']
    for k in ks:
        lines.append(f'  <interval begin="{k * dt:.2f}" end="{(k + 1) * dt:.2f}" id="dump">')
        for edge, per in replay.field_records.items():
            if k in per:
                tt, _occ = per[k]
                lines.append(f'    <edge id="{edge}" traveltime="{tt:.6f}"/>')
        lines.append("  </interval>")
    lines.append("</meandata>")
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(lines))


def _parse_route_costs(path: str) -> dict[str, float]:
    """Map vehicle id -> the (single) re-costed route cost from a duarouter
    ``--write-costs`` output."""
    out: dict[str, float] = {}
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise RuntimeError(f"duarouter R3 re-cost output at {path} is unparseable ({exc})") from exc
    for veh in root.findall("vehicle"):
        r = veh.find("route")
        if r is not None and r.get("cost") is not None:
            out[veh.get("id")] = float(r.get("cost"))
    return out


# --------------------------------------------------------------------------
# orchestrator: emit the EDOC artifact bundle
# --------------------------------------------------------------------------
class SumoDuaIterateAdapter:
    """Emit the EDOC-1 artifact contract for one :class:`EdocScenario` by running
    ``duaIterate`` (plans ``P``) then this adapter's pinned replay (experienced
    record ``X`` + provenance). The certifier consumes the bundle and re-runs the
    identical replay for G1. ``iterations`` is a factor (duaIterate outer steps)."""

    name = "sumo-duaiterate"

    def __init__(self, *, iterations: int = 20, keep_files: bool = False) -> None:
        self.iterations = int(iterations)
        self.keep_files = bool(keep_files)
        self.last_workdir: str | None = None

    def emit(self, scenario: EdocScenario, *, wall_seconds: float | None = None) -> EmittedBundle:
        """Compile -> duaIterate -> pinned replay, all under one wall deadline.
        Returns the emitted bundle (``P``, ``X``, provenance). RAISES on any engine/
        infra failure (adr-027); never returns a partial or self-reported result."""
        if scenario.engine != ENGINE:
            raise ValueError(
                f"sumo-duaiterate only runs eclipse-sumo instances; scenario {scenario.name!r} "
                f"pins engine {scenario.engine!r}"
            )
        assert_engine_pin(installed_engine_version(), scenario.engine_version)
        deadline = time.perf_counter() + wall_seconds if wall_seconds is not None else None
        keep = self.keep_files
        workdir = tempfile.mkdtemp(
            prefix=f"tabench-edoc-{os.getpid()}-keep-" if keep
            else f"tabench-edoc-{os.getpid()}-"
        )
        self.last_workdir = workdir
        try:
            net_path = compile_net(scenario, workdir, deadline)
            trips_path = _write_trips(scenario, workdir)
            routes_gz = run_duaiterate(
                scenario, net_path, trips_path, workdir, deadline, iterations=self.iterations
            )
            plans = _read_plans(routes_gz)
            # X + the field from THIS adapter's own pinned replay of P (reusing the
            # already-compiled net; the certifier re-runs the identical replay).
            replay = pinned_meso_replay(
                scenario, plans, deadline=deadline, net_path=net_path, workdir=workdir,
            )
            experienced = replay.agents
            return EmittedBundle(
                plans=plans,
                experienced=experienced,
                engine_version=scenario.engine_version,
                seed=int(scenario.seed),
            )
        finally:
            if not keep:
                shutil.rmtree(workdir, ignore_errors=True)
                self.last_workdir = None


def _semantic_config(dt: float) -> str:
    """The instance's semantic engine config string, DERIVED from the actually
    pinned meso options so a drift in :data:`_MESO_OPTS` moves the instance hash
    (it cannot silently change dynamics under a frozen hash)."""
    return "duaIterate;agg={:g};{}".format(dt, ";".join(_MESO_OPTS))


def build_diamond_scenario(
    name: str,
    *,
    seed: int = 42,
    n_agents: int = 720,
    depart_quantum: float = 2.0,
    n_intervals: int = 16,
    dt: float = 300.0,
    fftt_short: float = 70.0,
    fftt_long: float = 75.0,
    bottleneck_lanes: int = 1,
    free_lanes: int = 2,
    separation_factor: float = 5.0,
    floor_seconds: float = 15.0,
    r3_tolerance_s: float = 15.0,
    backlog_bound: float = 600.0,
    replay_deadline_s: float = 240.0,
    walk_bound: int = 4,
) -> EdocScenario:
    """The sumo-duaiterate row's DIAMOND family: O ->a1-> N1 ->a2-> D (route A) and
    O ->b1-> N2 ->b2-> D (route B). The route-distinguishing capacity drop is on
    ``a2`` (``bottleneck_lanes`` < ``free_lanes``, adr-036 line 557); route A is the
    UNIQUELY free-flow-shorter path (``fftt_short < fftt_long``) so the AON control
    piles onto the bottleneck and separates from the balanced converged assignment.
    Agents depart on the ``depart_quantum`` grid over the demand window. All fields
    are hashed (the instance hash covers lanes + speed-derived geometry + the pinned
    meso config); construction gates are the pure-data ``EdocScenario`` gates."""
    depart = np.array([i * depart_quantum for i in range(n_agents)], dtype=np.float64)
    return EdocScenario(
        name=name,
        edge_ids=("a1", "a2", "b1", "b2"),
        edge_tail=("O", "N1", "O", "N2"),
        edge_head=("N1", "D", "N2", "D"),
        edge_fftt=np.array([fftt_short, fftt_short, fftt_long, fftt_long]),
        edge_lanes=np.array([free_lanes, bottleneck_lanes, free_lanes, free_lanes]),
        agent_ids=tuple(f"v{i}" for i in range(n_agents)),
        agent_origin=("O",) * n_agents,
        agent_dest=("D",) * n_agents,
        agent_depart=depart,
        engine=ENGINE,
        engine_version=installed_engine_version(),
        seed=int(seed),
        semantic_config=_semantic_config(dt),
        dt=dt,
        n_intervals=n_intervals,
        departure_quantum=depart_quantum,
        backlog_bound=backlog_bound,
        separation_factor=separation_factor,
        floor_seconds=floor_seconds,
        replay_deadline_s=replay_deadline_s,
        r3_tolerance_s=r3_tolerance_s,
        walk_bound=walk_bound,
        family="sumo-duaiterate-diamond",
    )


def shared_bottleneck_scenario(*, seed: int = 42, n_agents: int = 720) -> EdocScenario:
    """A DELIBERATELY non-separating topology for the refusal demonstration: the
    1-lane drop sits on the SHARED upstream edge ``sh`` (O -> J) that BOTH routes
    traverse before the J -> D split, so the downstream choice cannot relieve the
    bottleneck. The AON control and the converged assignment congest ``sh``
    identically → the gap does not separate → :func:`negative_control_separation`
    REFUSES it (a config error, never a certified row; adr-036 line 557-558)."""
    depart = np.array([i * 2.0 for i in range(n_agents)], dtype=np.float64)
    return EdocScenario(
        name="sumo-duaiterate-shared-bottleneck",
        edge_ids=("sh", "p1", "p2", "q1", "q2"),
        edge_tail=("O", "J", "M1", "J", "M2"),
        edge_head=("J", "M1", "D", "M2", "D"),
        edge_fftt=np.array([70.0, 40.0, 40.0, 42.0, 42.0]),
        edge_lanes=np.array([1, 2, 2, 2, 2]),  # the drop is on the SHARED edge sh
        agent_ids=tuple(f"v{i}" for i in range(n_agents)),
        agent_origin=("O",) * n_agents,
        agent_dest=("D",) * n_agents,
        agent_depart=depart,
        engine=ENGINE,
        engine_version=installed_engine_version(),
        seed=int(seed),
        semantic_config=_semantic_config(300.0),
        dt=300.0,
        n_intervals=16,
        departure_quantum=2.0,
        backlog_bound=600.0,
        separation_factor=5.0,
        floor_seconds=15.0,
        replay_deadline_s=240.0,
        r3_tolerance_s=15.0,
        walk_bound=4,
        family="sumo-duaiterate-diamond",
    )


def reference_scenario() -> EdocScenario:
    """The pinned sumo-duaiterate reference instance (adr-037). Constants MEASURED
    with the SHIPPED estimator on eclipse-sumo 1.27.1 (2026-07-16): AON control
    RG_D1 ~0.139 vs converged ~0.021 → **6.5x** separation (declared factor 5.0);
    field-vs-experienced resolution delta ~8.8 s → floor 15 s; duarouter R3 max
    discrepancy ~5 s → tolerance 15 s; cross-seed converged RG_D1 spread ~0.006 <
    the ~0.09 resolution floor_gap → **deterministic single-seed track** (seed 42,
    disclosed)."""
    return build_diamond_scenario("sumo-duaiterate-ref")


def negative_control_separation(
    scenario: EdocScenario,
    *,
    wall_seconds: float | None,
    control_iterations: int = 1,
    converged_iterations: int = 20,
) -> dict[str, float]:
    """The negative-control separation gate (adr-036 R-control; deliverable 3/6).

    Runs the SAME instance twice: an AON control model (``duaIterate -l 1`` — the
    free-flow all-or-nothing step) and a converged model, certifies both, and
    requires ``RG_D1(control) / RG_D1(converged) >= scenario.separation_factor``. A
    non-separating instance (e.g. a shared-edge bottleneck) is a CONSTRUCTION error
    and RAISES ``ValueError`` — it is never a certified row. A censored control or
    converged run also RAISES (the anchor must be a real, feasible measurement).

    On success this records the scenario's FAMILY as separation-vetted (F10), which
    :func:`certify_emitted` asserts before certifying — so a non-separating family
    cannot be silently certified (the procedural R-control coupling made
    structural)."""
    from ...metrics.edoc_gaps import EdocEvaluator

    deadline = time.perf_counter() + wall_seconds if wall_seconds is not None else None

    def _emit_certify(iters: int) -> dict[str, float]:
        adapter = SumoDuaIterateAdapter(iterations=iters)
        emitted = adapter.emit(scenario, wall_seconds=_remaining(deadline))
        runner = make_replay_runner(deadline=deadline)
        return EdocEvaluator(scenario, runner).certify(emitted)

    control = _emit_certify(control_iterations)
    converged = _emit_certify(converged_iterations)
    if control["feasible"] != 1.0 or converged["feasible"] != 1.0:
        raise ValueError(
            f"negative-control separation: an anchor censored (control feasible "
            f"{control['feasible']}, converged feasible {converged['feasible']}) — "
            "the separation anchors must be real feasible measurements"
        )
    c_rg = float(converged["rg_d1"])
    a_rg = float(control["rg_d1"])
    separation = a_rg / c_rg if c_rg > 0 else float("inf")
    if separation < scenario.separation_factor:
        raise ValueError(
            f"negative-control separation FAILED for {scenario.name!r}: AON control "
            f"RG_D1 {a_rg:.5f} / converged RG_D1 {c_rg:.5f} = {separation:.2f}x < the "
            f"declared {scenario.separation_factor}x (a non-separating topology — e.g. a "
            "shared-edge bottleneck — is a construction error, never a certified row)"
        )
    # F10: mark the TOPOLOGY vetted (digest-keyed, not the family string — F3).
    _SEPARATION_VETTED_TOPOLOGIES.add(_topology_digest(scenario))
    return {
        "control_rg_d1": a_rg,
        "converged_rg_d1": c_rg,
        "separation": separation,
        "separation_factor": float(scenario.separation_factor),
    }


def certify_emitted(
    scenario: EdocScenario,
    emitted: EmittedBundle,
    *,
    wall_seconds: float | None = None,
) -> dict[str, float]:
    """The ROW's full certification path (adr-036/adr-037): the G0–G4 + ``RG_D1``
    substrate certificate, the **mandatory** R3 ``duarouter`` cross-check, and the
    negative-control separation-vetting assertion — all under ONE wall deadline.

    Why a row wrapper and not just :class:`EdocEvaluator`: the substrate certifier
    is engine-agnostic, so it cannot itself run the pinned ``duarouter`` (R3) nor
    know a family's engine-run separation status (R-control). This function couples
    both to the certificate for this row:

    * **R3 (F4):** on a feasible run it re-costs the driven plans with the pinned
      ``duarouter`` on the frozen dump and RAISES if the engine's own weight reader
      disagrees with the substrate field beyond ``r3_tolerance_s`` (infra RAISE,
      never a censor — adr-036 R3); the returned metrics carry the ``r3_*`` evidence
      that the cross-check ran.
    * **Separation-vetting (F10):** REFUSES up front unless the scenario's TOPOLOGY
      was separation-vetted by :func:`negative_control_separation` (digest-keyed —
      a relabeled family string cannot borrow another topology's vetting, the S3
      review's F3), so an un-vetted (possibly non-separating) instance cannot be
      silently certified."""
    from ...metrics.edoc_gaps import EdocEvaluator

    if _topology_digest(scenario) not in _SEPARATION_VETTED_TOPOLOGIES:
        raise RuntimeError(
            f"certify_emitted refuses {scenario.name!r} (family {scenario.family!r}): "
            "its TOPOLOGY has not been separation-vetted — run "
            "negative_control_separation on this topology first (adr-036 R-control / "
            "F10; vetting is keyed on the topology digest, not the family string)"
        )
    deadline = time.perf_counter() + wall_seconds if wall_seconds is not None else None
    runner = make_replay_runner(deadline=deadline)
    metrics = EdocEvaluator(scenario, runner).certify(emitted)
    if metrics.get("feasible") == 1.0:
        replay = runner(scenario, emitted.plans)
        r3 = duarouter_recost_crosscheck(
            scenario, emitted.plans, replay,
            deadline=deadline, tolerance_s=scenario.r3_tolerance_s,
        )
        metrics["r3_mean_s"] = r3["r3_mean_s"]
        metrics["r3_max_s"] = r3["r3_max_s"]
    return metrics


def _read_plans(routes_gz: str) -> dict[str, tuple[tuple[str, ...], float]]:
    """Read duaIterate's final chosen-routes ``.rou.xml.gz`` into emitted plans
    ``{aid: (route_edges, depart)}``."""
    with open(routes_gz, "rb") as fh:
        data = fh.read()
    try:
        if len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B:
            data = gzip.decompress(data)
        root = ET.fromstring(data)
    except (ET.ParseError, gzip.BadGzipFile, OSError) as exc:
        raise RuntimeError(f"duaIterate final routes at {routes_gz} unparseable ({exc})") from exc
    plans: dict[str, tuple[tuple[str, ...], float]] = {}
    for veh in root.findall("vehicle"):
        aid = veh.get("id")
        dep = float(veh.get("depart"))
        r = veh.find("route")
        edges = tuple((r.get("edges") if r is not None else "").split())
        plans[aid] = (edges, dep)
    return plans
