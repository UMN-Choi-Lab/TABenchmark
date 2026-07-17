"""DTALite ``simulation()`` mesoscopic queue-DNL as the third EDOC-1 row
(adr-036 / adr-040) — the DETERMINISTIC track.

Unlike ``dtalite-tap`` (adr-029: the wheel's OTHER entry point, a static
Frank-Wolfe on an exactly-mapped BPR, certified against the DECLARED cost law),
``simulation()`` is the Zhou & Taylor (2014) mesoscopic queue simulator: a
deterministic sub-second map from a plan artifact (``vehicle.csv``) to an
experienced-time artifact (``trajectory.csv``) with *no declared cost law* to
certify against. The engine IS the instance (adr-036): the certifier re-derives
every scored number by re-running the pinned engine on the model's emitted plans
(G1). This module is therefore an **EDOC producer**, not a
``TrafficAssignmentModel`` — never in ``MODEL_REGISTRY`` (the adr-037/039
posture) — and deliberately does NOT extend ``dtalite_tap.py`` (different track,
different compile map, different outputs, different failure typing).

**Deterministic track (adr-036 seed semantics).** The engine's only RNG is an
LCG re-seeded ``101 + time_step`` at every simulation step (TAPLite.cpp
5171-5174/2066-2068), so the engine consumes NO seed: ``seedable=False``, no
macroreps, ``seed_list=()`` — ``scenario.seed`` stays hashed but is
engine-INERT (disclosed here, in the family builder, in ``semantic_config``
via the disclosure string, and in adr-040). ``per_seed_scenarios`` refuses the
empty list, so macrorep misuse is structurally impossible.

**The measured engine physics the writers/parsers pin (all EXECUTED on 0.8.1,
adr-040):**

* ``simulation()`` reads free-flow time from ``vdf_fftt`` (minutes) — a
  one-column perturbation probe moved traversal with ``vdf_fftt`` and never
  with ``length``/``free_speed``; the writer keeps them CONSISTENT anyway
  (``length = vdf_fftt`` miles at 60 mph) so either read yields the hashed fftt.
* Congestion lives in the LINK-TO-LINK TRANSFER: per-link traversal is always
  exactly fftt, and the queue (the ~600 veh/h admission law,
  ``entrance_queue.size() < capacity/3600``, TAPLite.cpp:5301/4979) appears as
  the gap between exiting link ``i`` and entering link ``i+1``. Origin
  insertion is NOT admission-gated (measured: 976 veh/h inserted onto a
  600-cap link with zero origin wait), so ``depart_delay`` is ~0 on this
  engine and the entrance-choice channel is inert — the parser charges each
  edge its ENTRY-TO-NEXT-ENTRY span (the SUMO-meso upstream-storage
  convention), which decomposes the door-to-door time exactly and is what
  keeps the R2 field faithful to experienced costs. The admission law is the
  engine's cost law, documented inside the instance — no A2 cost-match exists
  (adr-030/036), G1 replay fidelity plays A2's role.
* ``rc=0`` silent failures are the engine's signature: pre-period departures
  drop silently AND head-block every later same-first-link agent (measured on
  the shipped family net: fast rc=0 with all-filler rows — the census-censor
  variant; the pilot's infinite-loop variant is also defended by the R6 replay
  deadline); an UNSORTED ``vehicle.csv`` silently filler-corrupts the
  later-departing agent (forgery pair D5) — the certifier writes the file
  itself, sorted by ``(departure_time, agent)``, so no model-controlled byte
  order ever reaches the engine; period-end truncation leaves in-flight agents
  with ``07:00:00`` filler chains at rc=0. Success is DEFINED by the parse +
  census, never the exit code (adr-029 doctrine).
* ``OMP_NUM_THREADS=1`` on the child is a CORRECTNESS pin, not hygiene
  (adr-036 G0): raw ``#pragma omp parallel for`` over shared ``std::deque`` —
  measured at OMP=4 on a congested 8-link net: 6 divergent trajectories + one
  SIGSEGV in 10 runs; default OMP on a 192-core box is ~10^4x slower. The
  child env override beats a hostile parent (adr-029 measured).
* The x10 discharge boost fires in the last 720 six-second intervals of ANY
  horizon (the ``2*60*6`` units bug, TAPLite.cpp:5116-5127): every
  constructor-side run RAISES on a census crossing ``horizon - 720*6`` and a
  certified emission whose replay crosses is CENSORED (pair 12/D3).
* The engine's ``t>=600`` early exit (TAPLite.cpp:5387) silently DROPS valid
  in-window departures that follow an all-completed lull — the family
  constructor refuses departure profiles that allow one
  (:func:`_assert_no_lull_drop`) and the G3 completion census is the
  certify-time backstop (adr-040, a hazard adr-036 did not name).

**Subprocess discipline** is the S2 shape verbatim with the S3 fixes from
birth: ``stdin=DEVNULL`` (``ExitMessage`` = ``getchar()``), one subprocess per
engine call (a second call in-process doubles state), own process group +
``killpg(SIGKILL)``, the hashed ``replay_deadline_s`` always bounding the
replay with the F1 caller-clip/scenario-deadline censor typing, pid-scoped
temp prefixes (F5), links written sorted by ``(from_node_id, to_node_id)``
(the adr-029 CRITICAL — same GMNS reader), rc never trusted, engine version
via ``importlib.metadata`` (never an import: the wheel prints a banner and
ctypes-loads the engine ``.so`` into the host).

This module imports WITHOUT the wheel (stdlib + numpy only — a named deviation
from ``dtalite_tap``'s module-scope guard, mirroring ``matsim_edoc``'s
rationale): nothing here imports ``DTALite`` in-host, so the engine-free test
half runs on the core matrix legs and engine absence surfaces as the runtime
G0 version read. Design: docs/design/adr-036 + docs/design/adr-040.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import importlib.metadata
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable

import numpy as np

from ...edoc.canon import hash_dtalite_artifacts
from ...edoc.field import build_field_from_records, build_origin_waits
from ...edoc.replay import (
    EmittedBundle,
    PlanReplayFailure,
    ReplayAgent,
    ReplayResult,
    assert_engine_pin,
)
from ...edoc.scenario import EdocScenario
from ...edoc.tdsp import evaluate_route

__all__ = [
    "ENGINE",
    "DTALiteSimulationAdapter",
    "build_dtalite_corridor_scenario",
    "certify_emitted",
    "installed_engine_version",
    "make_replay_runner",
    "negative_control_separation",
    "pinned_simulation_replay",
    "reference_scenario",
    "shared_bottleneck_scenario",
]

ENGINE = "DTALite"

# --------------------------------------------------------------------------
# pinned engine constants (every outcome-bearing one rides in the hashed
# semantic_config below — the S3 F2 discipline from birth)
# --------------------------------------------------------------------------
# ONE command constant per engine mode (the N6 one-constant discipline: there
# is no settings flag to forge — the mode IS the subprocess command).
_SIM_CMD = "import DTALite; DTALite.simulation()"
_ASSIGN_CMD = "import DTALite; DTALite.assignment()"
# Engine-clock offset: scenario time t maps to engine time t + _T0 (the writer
# adds it, the parsers subtract it — the matsim t0 precedent). 7 h: the demand
# period starts at 07:00, dodging the engine's midnight-adjacent time handling.
_T0 = 25200.0
# The engine dynamics grid (TAPLite.cpp:4950 number_of_seconds_per_interval=6).
_SIM_STEP_S = 6.0
# The x10 discharge-boost window: the last 720 six-second intervals of ANY
# horizon (the 2*60*6 units bug, TAPLite.cpp:5116-5127) = the last 72 min.
_BOOST_WINDOW_INTERVALS = 720
# GMNS geometry constants: free_speed 60 mph makes length (miles) numerically
# equal vdf_fftt (minutes), so the two fftt read paths agree by construction
# (the ruling-6 probe: simulation() reads vdf_fftt; the writer keeps both).
_FREE_SPEED_MPH = 60.0
# The engine-side capacity dial: capacity = edge_lanes * this, written with
# engine lanes=1 always (the adr-029 lanes^2 trap applies to the R9
# assignment() step, which shares the GMNS reader).
_CAP_PER_LANE_VPH = 600.0
# R9 assignment-step BPR shape (adapter plumbing: shapes P0, and is echoed in
# the replay's link.csv — hashed for safety although simulation() reads only
# vdf_fftt/capacity of these).
_VDF_ALPHA = 0.15
_VDF_BETA = 4.0
_ASSIGN_ITERATIONS = 20
# The R9 FW step runs on its own 1 h demand period (the adr-029 identity
# condition: the VDF divides volume by the period hours, so 1 h makes I = V
# and v/c the textbook ratio — the pilot's genuinely-split 781.25/218.75
# state; the 6 h SIMULATION horizon would divide I by 6, starve the FW of
# congestion, and collapse the split to AON — measured).
_ASSIGN_PERIOD_HOURS = 1
# The deterministic-track disclosure (adr-036 seed semantics), hashed.
_DETERMINISTIC_DISCLOSURE = "lcg=time-step-keyed;seedable=false;macroreps=none"

# The 13-name trajectory header at 0.8.1. Data rows carry 12 fields — the
# ``travel_time`` column is NEVER written (measured), so positional field 7 is
# ``current_link_seq_no``. Any header/row-shape drift RAISES: format drift
# bumps the canon version (R10), never gets silently re-aligned.
_TRAJ_HEADER = (
    "agent_id", "departure_time", "departure_time_hhmmss", "loaded_status",
    "o_zone_id", "d_zone_id", "distance", "travel_time",
    "current_link_seq_no", "link_ids", "arrival_times", "departure_times",
    "geometry",
)
_TRAJ_ROW_FIELDS = 12
# vehicle.csv header (the emitted-and-read 16-column shape the pilot pinned).
_VEHICLE_HEADER = (
    "agent_id", "departure_time", "departure_time_hhmmss", "mode", "route_id",
    "o_zone_id", "d_zone_id", "unique_route_id", "node_ids", "link_ids",
    "total_distance_mile", "total_distance_km", "total_free_flow_travel_time",
    "total_travel_time", "route_key", "route_volume",
)

# Topologies separation-vetted by :func:`negative_control_separation` (F10),
# keyed on the TOPOLOGY digest, never the forgeable family string (S3 F3 from
# birth) — runtime state only, no instance hash involved.
_SEPARATION_VETTED_TOPOLOGIES: set[str] = set()


def installed_engine_version() -> str:
    """The ``DTALite`` wheel version installed on this box (the G0 read).

    Read from package metadata, NEVER by importing the package (``import
    DTALite`` prints a banner and ctypes-loads the OpenMP engine ``.so`` into
    the host — adr-029). Absence is a clean infra RAISE naming the extra."""
    try:
        return importlib.metadata.version(ENGINE)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "DTALite is not installed — `pip install tabench[dtalite]` "
            "(the dtalite-simulation row needs the engine wheel)"
        ) from exc


def _semantic_config() -> str:
    """The instance's semantic engine-config string, DERIVED from every pinned
    writer/runner constant so a drift in ANY of them moves the instance hash
    (the adr-037 ``_MESO_OPTS`` pattern; S3 F2 from birth)."""
    return (
        f"dtalite-simulation;sim={_SIM_CMD};assign={_ASSIGN_CMD};"
        f"t0={_T0:g};step={_SIM_STEP_S:g};boostWindow={_BOOST_WINDOW_INTERVALS};"
        f"freeSpeed={_FREE_SPEED_MPH:g};capPerLane={_CAP_PER_LANE_VPH:g};"
        f"vdf={_VDF_ALPHA:g}:{_VDF_BETA:g};assignIters={_ASSIGN_ITERATIONS};"
        f"assignPeriodH={_ASSIGN_PERIOD_HOURS};omp=1;{_DETERMINISTIC_DISCLOSURE}"
    )


# --------------------------------------------------------------------------
# wall-deadline plumbing (the S2 discipline verbatim, S3 F1 from birth)
# --------------------------------------------------------------------------
def _remaining(deadline: float | None) -> float | None:
    """Seconds left on the single wall deadline, or ``None`` if unbudgeted.
    RAISES if the deadline already passed (a prior phase ate the budget)."""
    if deadline is None:
        return None
    left = deadline - time.perf_counter()
    if left <= 0.0:
        raise RuntimeError("dtalite-simulation wall deadline exhausted before the next step")
    return left


def _reap_group(proc: subprocess.Popen) -> None:
    """SIGKILL the subprocess's whole process GROUP, then reap it.
    ``start_new_session=True`` puts the python child (and the ctypes-loaded
    engine inside it) in its own group so one ``killpg`` takes it down; a
    mid-run kill leaves a torn/empty trajectory.csv, which is exactly why a
    timeout-killed replay's artifacts are never parsed (R6)."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()
    try:
        proc.wait(timeout=10)
    except (subprocess.TimeoutExpired, OSError):
        pass


def _intersect_replay_deadline(
    scenario: EdocScenario, deadline: float | None
) -> tuple[float, bool]:
    """The certifier's hard replay deadline: the scenario-declared hashed
    ``replay_deadline_s`` measured from now, intersected with any tighter
    caller wall — the hashed constant ALWAYS bounds a certifier replay, so a
    head-blocking plan cannot hang the certifier unboundedly (adr-036 R6; the
    pilot measured an infinite-loop head-block variant, and this deadline is
    its defense even though the shipped family reproduces the fast
    filler-corruption variant instead — both defenses ship, adr-040).

    Returns ``(deadline, clipped_by_caller)`` — the S3 F1 typing from birth:
    only an expiry of the SCENARIO-declared deadline is the model's fault
    (censor); a caller wall clipping below it is certifier-side budget
    exhaustion and must RAISE as infrastructure."""
    scen_deadline = time.perf_counter() + float(scenario.replay_deadline_s)
    if deadline is None or deadline >= scen_deadline:
        return scen_deadline, False
    return deadline, True


def _run(
    code: str, *, cwd: str, deadline: float | None, what: str,
    censor_on_fail: bool = False, censor_on_timeout: bool | None = None,
) -> None:
    """Run ONE engine entry (``python -c "import DTALite; ..."``) in a
    throwaway subprocess under the adr-029 discipline: ``stdin=DEVNULL``
    (``ExitMessage`` = ``getchar()`` + ``exit()``), CWD-confined outputs, its
    OWN process group, and the child env ALWAYS pinning ``OMP_NUM_THREADS=1``
    — a G0 CORRECTNESS requirement (measured: OMP=4 diverges and SIGSEGVs on a
    congested net; the child override beats a hostile parent).

    Crash-vs-censor (adr-036 R6, S3 F1 split): by default a timeout / OS error
    / nonzero rc is an infrastructure ``RuntimeError``. The ONE step that
    replays MODEL-emitted plans passes ``censor_on_fail=True`` (a genuine
    engine crash raises :class:`PlanReplayFailure`) and ``censor_on_timeout``
    = whether the SCENARIO deadline was binding. rc is never trusted beyond
    this function — every caller re-reads its artifact."""
    if censor_on_timeout is None:
        censor_on_timeout = censor_on_fail
    timeout = _remaining(deadline)  # pre-exhaustion -> plain RuntimeError (infra)
    cmd = [sys.executable, "-c", code]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        raise RuntimeError(f"{what}: could not execute ({exc})\n  cmd: {' '.join(cmd)}") from exc
    try:
        _out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _reap_group(proc)
        msg = f"{what}: killed by the wall deadline\n  cmd: {' '.join(cmd)}"
        if censor_on_timeout:
            raise PlanReplayFailure(msg) from exc
        raise RuntimeError(msg) from exc
    if proc.returncode != 0:
        msg = (
            f"{what}: exit {proc.returncode}\n  cmd: {' '.join(cmd)}\n"
            f"  stderr tail: {err[-800:]}"
        )
        if censor_on_fail:
            raise PlanReplayFailure(msg)
        raise RuntimeError(msg)


# --------------------------------------------------------------------------
# instance -> GMNS (deterministic, a function of hashed fields only; the
# certifier writes EVERY engine input — pair N6 closed structurally)
# --------------------------------------------------------------------------
def _node_ids(scenario: EdocScenario) -> tuple[dict[str, int], int]:
    """Deterministic node-name -> engine-int map: zone nodes (agent origins and
    destinations) get ids ``1..Z`` (the engine's ``zone_id == node_id`` rule,
    adr-029), remaining nodes ``Z+1..N``; both sorted by name."""
    zones = sorted(set(scenario.agent_origin) | set(scenario.agent_dest))
    others = sorted((set(scenario.edge_tail) | set(scenario.edge_head)) - set(zones))
    mapping = {n: i + 1 for i, n in enumerate([*zones, *others])}
    return mapping, len(zones)


def _edge_order(scenario: EdocScenario) -> tuple[dict[str, int], list[str]]:
    """Edge id -> 1-based ``link.csv`` file index with links SORTED by
    ``(from_node_id, to_node_id)`` — the adr-029 CRITICAL: the engine builds
    adjacency from contiguous from-node ranges, and vehicle routes are 1-based
    FILE-ORDER link indices. Parallel edges (same node pair) would collide in
    that adjacency, so they are refused (a config error)."""
    nid, _ = _node_ids(scenario)
    keyed = sorted(
        zip(scenario.edge_ids, scenario.edge_tail, scenario.edge_head, strict=True),
        key=lambda e: (nid[e[1]], nid[e[2]]),
    )
    pairs = [(nid[t], nid[h]) for _e, t, h in keyed]
    if len(set(pairs)) != len(pairs):
        raise ValueError(
            f"EdocScenario {scenario.name!r}: parallel edges (same node pair) cannot "
            "be compiled — the engine keys its adjacency and read-back on (from, to)"
        )
    order = [e for e, _t, _h in keyed]
    return {e: i + 1 for i, e in enumerate(order)}, order


def _engine_period_hours(scenario: EdocScenario) -> tuple[int, int]:
    """The simulation demand period ``[start, end)`` in whole engine hours:
    start = _T0, end = _T0 + the hashed field horizon (``dt * n_intervals``),
    which must be a whole number of hours (a config error otherwise — the
    engine settings only take integer hours)."""
    horizon = float(scenario.dt) * int(scenario.n_intervals)
    start_h = int(_T0 // 3600.0)
    if abs(horizon / 3600.0 - round(horizon / 3600.0)) > 1e-9:
        raise ValueError(
            f"EdocScenario {scenario.name!r}: the field horizon dt*n_intervals = "
            f"{horizon} s must be a whole number of hours (engine settings.csv "
            "takes integer demand-period hours)"
        )
    end_h = start_h + int(round(horizon / 3600.0))
    if end_h > 24:
        raise ValueError(
            f"EdocScenario {scenario.name!r}: engine period would end at hour "
            f"{end_h} > 24 (the engine clock is HH:MM:SS within one day)"
        )
    return start_h, end_h


def _write_gmns_sim(
    scenario: EdocScenario, workdir: str, *, period_hours: tuple[int, int] | None = None
) -> None:
    """Write node/link/demand/settings from hashed fields only. ``vdf_fftt``
    is the scenario fftt in MINUTES (the column simulation() reads — the
    ruling-6 probe); ``length = vdf_fftt`` miles at ``free_speed = 60`` mph so
    the geometry-derived fftt agrees by construction; engine ``lanes = 1``
    always with ``capacity = edge_lanes * _CAP_PER_LANE_VPH`` (the adr-029
    lanes^2 trap, live for the R9 assignment step); links sorted by
    ``(from_node_id, to_node_id)`` (the adr-029 CRITICAL)."""
    nid, n_zones = _node_ids(scenario)
    _index_of, order = _edge_order(scenario)
    fftt = scenario.fftt_of()
    lanes = scenario.lanes_of()
    tail = dict(zip(scenario.edge_ids, scenario.edge_tail, strict=True))
    head = dict(zip(scenario.edge_ids, scenario.edge_head, strict=True))

    with open(os.path.join(workdir, "node.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["node_id", "zone_id", "x_coord", "y_coord"])
        for _name, i in sorted(nid.items(), key=lambda kv: kv[1]):
            w.writerow([i, i if i <= n_zones else 0, float(i), 0.0])

    with open(os.path.join(workdir, "link.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["from_node_id", "to_node_id", "link_id", "lanes", "capacity",
                    "free_speed", "length", "vdf_fftt", "vdf_alpha", "vdf_beta",
                    "vdf_plf", "toll"])
        for i, e in enumerate(order):
            fftt_min = fftt[e] / 60.0
            w.writerow([
                nid[tail[e]], nid[head[e]], i + 1, 1,
                repr(lanes[e] * _CAP_PER_LANE_VPH), repr(_FREE_SPEED_MPH),
                repr(fftt_min), repr(fftt_min), repr(_VDF_ALPHA),
                repr(_VDF_BETA), 1, 0,
            ])

    od_counts: dict[tuple[str, str], int] = {}
    for o, d in zip(scenario.agent_origin, scenario.agent_dest, strict=True):
        od_counts[(o, d)] = od_counts.get((o, d), 0) + 1
    with open(os.path.join(workdir, "demand.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["o_zone_id", "d_zone_id", "volume"])
        for (o, d), n in sorted(od_counts.items(), key=lambda kv: (nid[kv[0][0]], nid[kv[0][1]])):
            w.writerow([nid[o], nid[d], repr(float(n))])

    start_h, end_h = period_hours if period_hours is not None else _engine_period_hours(scenario)
    with open(os.path.join(workdir, "settings.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["number_of_iterations", "number_of_processors",
                    "demand_period_starting_hours", "demand_period_ending_hours",
                    "first_through_node_id", "base_demand_mode", "route_output",
                    "vehicle_output", "log_file", "odme_mode", "odme_vmt"])
        w.writerow([_ASSIGN_ITERATIONS, 1, start_h, end_h, 1, 0, 1, 1, 0, 0, 0])


def _write_vehicles(
    scenario: EdocScenario,
    plans: dict[str, tuple[tuple[str, ...], float]],
    workdir: str,
) -> None:
    """Write ``vehicle.csv`` from the parsed plans dict, rows SORTED ASCENDING
    by ``(departure_time, agent)`` — the D5 mandatory gate (measured: an
    unsorted file silently filler-corrupts the later-departing agent at rc=0).
    Because the certifier regenerates this file from hashed demand + the plans
    dict, no model-controlled byte order ever reaches the engine. Departure
    times are engine MINUTES (``scenario seconds / 60 + _T0``); routes are the
    1-based sorted-file-order link indices."""
    nid, _ = _node_ids(scenario)
    index_of, _order = _edge_order(scenario)
    trip = {
        aid: (o, d)
        for aid, o, d in zip(scenario.agent_ids, scenario.agent_origin,
                             scenario.agent_dest, strict=True)
    }
    engine_id = {aid: i + 1 for i, aid in enumerate(scenario.agent_ids)}
    rows = []
    for aid, (route, dep) in plans.items():
        if aid not in trip:
            raise ValueError(f"plans carry unknown agent {aid!r} (not in the trip table)")
        if not route:
            raise ValueError(f"agent {aid!r}: empty route cannot be written")
        try:
            link_ids = ";".join(str(index_of[e]) for e in route)
        except KeyError as exc:
            raise ValueError(f"agent {aid!r}: route edge {exc.args[0]!r} is not a "
                             "scenario edge") from exc
        o, d = trip[aid]
        rows.append((float(dep) / 60.0 + _T0 / 60.0, engine_id[aid], link_ids,
                     nid[o], nid[d]))
    rows.sort(key=lambda r: (r[0], r[1]))  # D5: ascending (departure, agent)
    with open(os.path.join(workdir, "vehicle.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(_VEHICLE_HEADER))
        for dep_min, eid, link_ids, o, d in rows:
            w.writerow([eid, repr(dep_min), "", "auto", 0, o, d, 0, "",
                        link_ids, 0.002, 0.003, 0, 0, "k", 1])


# --------------------------------------------------------------------------
# trajectory.csv positional parser (R10's fourth founding necessity, realized
# in the PARSER — the hash canonicalizer is the identity; the artifact is
# byte-deterministic)
# --------------------------------------------------------------------------
def _hms_to_s(text: str) -> float:
    h, m, s = text.split(":")
    return int(h) * 3600.0 + int(m) * 60.0 + int(s)


def _parse_trajectory(
    path: str, scenario: EdocScenario
) -> dict[str, dict]:
    """Positionally parse ``trajectory.csv``: the header must be EXACTLY the 13
    known names and every data row EXACTLY 12 fields (the emitted row drops
    ``travel_time``, so positional field 7 is ``current_link_seq_no``) — any
    shape drift RAISES ``RuntimeError`` (upstream format drift bumps the canon
    version, R10; it is never silently re-aligned). ``loaded_status`` is NEVER
    read (measured dead: 0 for completed, truncated and never-loaded alike —
    pair D2).

    Returns per-agent dicts with the scheduled departure (from the
    verbatim-echoed ``departure_time``), the driven edge route, and — for
    CENSUSED-COMPLETE agents only — the entry chain and final exit on the
    scenario clock. Completion census (pair 1/D1): ``current_link_seq_no ==
    len(links) - 1`` AND a fully-parsed monotone chain; everything else
    (pre-period drop, head-block filler, period-end truncation) reads as
    incomplete — a NON-OBSERVATION excluded from every field computation, and
    a G3 censor downstream, never a parse error."""
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except (StopIteration, csv.Error) as exc:
            raise RuntimeError(f"trajectory.csv at {path} is empty/unreadable") from exc
        if tuple(header) != _TRAJ_HEADER:
            raise RuntimeError(
                "trajectory.csv header drifted from the pinned 13-column 0.8.1 "
                f"shape (canon version bump required, R10): got {header!r}"
            )
        inv_edge = {i: e for e, i in _edge_order(scenario)[0].items()}
        aid_of = {i + 1: aid for i, aid in enumerate(scenario.agent_ids)}
        out: dict[str, dict] = {}
        for row in reader:
            if not row:
                continue
            if len(row) != _TRAJ_ROW_FIELDS:
                raise RuntimeError(
                    f"trajectory.csv row has {len(row)} fields, expected exactly "
                    f"{_TRAJ_ROW_FIELDS} (the travel_time-less 0.8.1 row shape; "
                    "drift bumps the canon version, R10)"
                )
            # positional row layout (travel_time absent): 0 agent_id,
            # 1 departure_time, 2 hhmmss, 3 loaded_status (DEAD, never read),
            # 4 o_zone, 5 d_zone, 6 distance, 7 current_link_seq_no,
            # 8 link_ids, 9 arrival_times, 10 departure_times, 11 geometry.
            try:
                engine_aid = int(row[0])
                sched = float(row[1]) * 60.0 - _T0  # verbatim-echoed minutes
                cur = int(row[7])
                links = tuple(inv_edge[int(x)] for x in row[8].split(";") if x)
            except (ValueError, KeyError) as exc:
                raise RuntimeError(
                    f"trajectory.csv row unparseable ({exc}): {row[:9]!r}"
                ) from exc
            if engine_aid not in aid_of:
                raise RuntimeError(
                    f"trajectory.csv row names engine agent {engine_aid} outside the "
                    "written id range (garbage output)"
                )
            aid = aid_of[engine_aid]
            entries: list[float] = []
            exits: list[float] = []
            parsed = True
            for cell, dest in ((row[9], entries), (row[10], exits)):
                for tok in cell.split(";"):
                    if not tok or tok == "NA":
                        parsed = False
                        break
                    try:
                        dest.append(_hms_to_s(tok) - _T0)
                    except ValueError:
                        parsed = False
                        break
            n = len(links)
            complete = (
                parsed
                and cur == n - 1
                and len(entries) == n
                and len(exits) == n
                and all(b >= a for a, b in zip(entries, entries[1:], strict=False))
                and all(x >= e for e, x in zip(entries, exits, strict=True))
                # the transfer instant: entering link i+1 never precedes leaving
                # link i beyond the engine's 1 s display floor (measured residual)
                and all(entries[i + 1] >= exits[i] - 1.0 for i in range(n - 1))
            )
            out[aid] = {
                "sched": sched,
                "route": links,
                "complete": complete,
                "entries": entries if complete else [],
                "exit": exits[-1] if complete else None,
            }
        return out


def _records_from_trajectory(
    parsed: dict[str, dict],
    plans: dict[str, tuple[tuple[str, ...], float]],
    scenario: EdocScenario,
) -> tuple[
    dict[str, ReplayAgent],
    dict[str, dict[int, tuple[float, float]]],
    dict[str, dict[int, tuple[float, float]]],
]:
    """Build the ReplayResult material from a parsed trajectory.

    Per-edge cost sample = the ENTRY-TO-NEXT-ENTRY span (the last edge closes at
    its own exit): the transfer queue — where this engine's congestion lives —
    is charged to the UPSTREAM edge (the SUMO-meso upstream-storage convention),
    so the door-to-door time decomposes exactly into the per-edge spans and the
    R2 field stays faithful to experienced costs. Occupancy witness = the count
    of agents present on the edge during the interval (exact from the spans —
    the matsim event-span precedent, pair 8). Incomplete agents (drop /
    head-block / truncation) carry a negative-experienced marker so the G3
    census censors them; they contribute NO field samples (non-observations,
    pair D1)."""
    dt = float(scenario.dt)
    n_int = int(scenario.n_intervals)
    agents: dict[str, ReplayAgent] = {}
    samples: dict[str, dict[int, list[float]]] = {}
    occupancy: dict[str, dict[int, float]] = {}
    flows: dict[str, dict[int, list[float]]] = {}

    def _k(t: float) -> int:
        if t <= 0.0:
            return 0
        k = int(t // dt)
        return n_int - 1 if k >= n_int else k

    for aid, (route, dep) in plans.items():
        rec = parsed.get(aid)
        if rec is None or not rec["complete"]:
            sched = rec["sched"] if rec is not None else float(dep)
            agents[aid] = ReplayAgent(
                agent_id=aid, departure=sched, arrival=sched - 1.0,
                route=tuple(rec["route"]) if rec is not None else tuple(route),
                experienced_time=-1.0, depart_delay=0.0,
            )
            continue
        entries = rec["entries"]
        exit_last = rec["exit"]
        sched = rec["sched"]
        r = rec["route"]
        agents[aid] = ReplayAgent(
            agent_id=aid, departure=sched, arrival=exit_last, route=r,
            experienced_time=exit_last - sched, depart_delay=entries[0] - sched,
        )
        spans = [
            (r[i], entries[i], entries[i + 1] if i + 1 < len(r) else exit_last)
            for i in range(len(r))
        ]
        for edge, t_in, t_out in spans:
            k_in = _k(t_in)
            samples.setdefault(edge, {}).setdefault(k_in, []).append(t_out - t_in)
            f = flows.setdefault(edge, {})
            f.setdefault(k_in, [0.0, 0.0])[0] += 1.0
            f.setdefault(_k(t_out), [0.0, 0.0])[1] += 1.0
            occ = occupancy.setdefault(edge, {})
            for k in range(k_in, _k(t_out) + 1):
                occ[k] = occ.get(k, 0.0) + 1.0

    field_records = {
        edge: {
            k: (sum(v) / len(v), occupancy.get(edge, {}).get(k, 0.0))
            for k, v in per_k.items()
        }
        for edge, per_k in samples.items()
    }
    flow_records = {
        edge: {k: (io[0], io[1]) for k, io in per_k.items()}
        for edge, per_k in flows.items()
    }
    return agents, field_records, flow_records


# --------------------------------------------------------------------------
# the pinned replay (produces X + the field; the certifier's G1 runner)
# --------------------------------------------------------------------------
def pinned_simulation_replay(
    scenario: EdocScenario,
    plans: dict[str, tuple[tuple[str, ...], float]],
    *,
    deadline: float | None,
    workdir: str | None = None,
) -> ReplayResult:
    """One pinned ``simulation()`` replay of ``plans`` (the G1 matched object —
    with no engine router, ``simulation()`` never routes and ``assignment()``
    never simulates, so this replay is also the certifier-owned BR's frozen
    field source; the substrate TD-SP is normative-only on this row, disclosed
    in adr-040). Asserts the installed engine against the instance pin BEFORE
    running, writes EVERY input itself (N6), runs the one censor-typed
    subprocess, re-reads the artifact (rc never trusted), and hashes the raw
    ``trajectory.csv`` bytes under the R10 canonicalizer (identity — the
    artifact is byte-deterministic at the pinned OMP=1)."""
    assert_engine_pin(installed_engine_version(), scenario.engine_version)
    deadline, clipped_by_caller = _intersect_replay_deadline(scenario, deadline)

    own_tmp = workdir is None
    # pid-scoped prefix (S3 F5 from birth): hygiene snapshots diff only their
    # own process's dirs, so concurrent sessions cannot cross-flake.
    workdir = workdir or tempfile.mkdtemp(
        prefix=f"tabench-edoc-dtalite-{os.getpid()}-replay-"
    )
    try:
        _write_gmns_sim(scenario, workdir)
        _write_vehicles(scenario, plans, workdir)
        _run(
            _SIM_CMD, cwd=workdir, deadline=deadline, what="dtalite simulation replay",
            # an engine CRASH here is always the plan's fault (R6 censor); a
            # TIMEOUT censors only when the SCENARIO deadline was binding — a
            # caller-clipped wall is a certifier budget fault, infra RAISE (F1).
            censor_on_fail=True,
            censor_on_timeout=not clipped_by_caller,
        )
        traj_path = os.path.join(workdir, "trajectory.csv")
        if not os.path.exists(traj_path):  # rc is never trusted
            raise RuntimeError("simulation() reported success but wrote no trajectory.csv")
        with open(traj_path, "rb") as fh:
            traj_bytes = fh.read()
        parsed = _parse_trajectory(traj_path, scenario)
        agents, field_records, flow_records = _records_from_trajectory(
            parsed, plans, scenario
        )
        return ReplayResult(
            canon_hash=hash_dtalite_artifacts({"trajectory.csv": traj_bytes}),
            agents=agents,
            field_records=field_records,
            flows=flow_records,
            n_intervals=int(scenario.n_intervals),
        )
    finally:
        if own_tmp:
            shutil.rmtree(workdir, ignore_errors=True)


def make_replay_runner(*, deadline: float | None) -> Callable[..., ReplayResult]:
    """A :data:`~tabench.edoc.replay.ReplayRunner` bound to the single wall
    deadline, for injection into :class:`~tabench.metrics.edoc_gaps.EdocEvaluator`.
    Each call rebuilds every input fresh from the hashed scenario, so the
    emit-time replay and both certifier replays are the identical map."""
    def runner(
        scenario: EdocScenario, plans: dict[str, tuple[tuple[str, ...], float]]
    ) -> ReplayResult:
        return pinned_simulation_replay(scenario, plans, deadline=deadline)

    return runner


# --------------------------------------------------------------------------
# the boost census (pair 12 / D3 — three correctly-typed layers)
# --------------------------------------------------------------------------
def _boost_onset_s(scenario: EdocScenario) -> float:
    """Scenario-clock onset of the engine's x10 discharge-boost window: the
    last ``720`` six-second intervals of the horizon (the 2*60*6 units bug —
    the last 72 min of ANY horizon; a bare >= 2 h period never excludes it)."""
    return float(scenario.dt) * int(scenario.n_intervals) - _BOOST_WINDOW_INTERVALS * _SIM_STEP_S


def _boost_crossings(scenario: EdocScenario, agents: dict[str, ReplayAgent]) -> list[str]:
    """Agents whose experienced record is x10-discharge-contaminated.

    When the boost onset is > 0 (a partial window), only COMPLETED agents whose
    last real exit lands at/after the onset are contaminated. When the onset is
    <= 0 the ENTIRE horizon lies inside the boost window (a horizon shorter than
    the 720-interval window, e.g. a 1 h period): the instance is boost-degenerate
    — no uncontaminated record can exist for ANY agent, completed or truncated —
    so the census is the whole emission (``n_agents``), a topology-stable count
    (adr-040 pair 12: a fully-covered horizon poisons every agent)."""
    onset = _boost_onset_s(scenario)
    if onset <= 0.0:
        return sorted(agents)
    return sorted(
        aid for aid, a in agents.items()
        if a.experienced_time >= 0.0 and a.arrival >= onset
    )


def _boost_censor(
    scenario: EdocScenario,
    metrics: dict[str, float],
    experienced: dict[str, ReplayAgent],
) -> dict[str, float]:
    """The CERTIFY-TIME boost-census arm (pair 12): computed from the emitted
    ``X`` AFTER G1 has proven it equal to the replay tuple-for-tuple, so the
    census on ``X`` IS the census on the replay with zero extra engine calls.
    A crossing returns the censored metrics dict (feasible=0, scored NaN) with
    a boost-window diagnostic — a CENSOR, never a construction RAISE."""
    crossings = _boost_crossings(scenario, experienced)
    if not crossings:
        return metrics
    censored = dict.fromkeys(metrics, float("nan"))
    censored["feasible"] = 0.0
    censored["boost_crossing_n"] = float(len(crossings))
    censored["boost_onset_s"] = _boost_onset_s(scenario)
    return censored


# --------------------------------------------------------------------------
# R9: plans from route_assignment.csv (assignment()'s own vehicle.csv emission
# is dead code — TAPLite.cpp:2238 route_volume = 0, measured header-only)
# --------------------------------------------------------------------------
def _parse_route_assignment(
    path: str, scenario: EdocScenario
) -> dict[tuple[str, str], list[tuple[tuple[str, ...], float]]]:
    """Parse ``route_assignment.csv`` into per-OD ``(route_edges, volume)``
    lists. Columns are located by NAME in the emitted header (drift RAISES;
    the 0.8.1 header carries a trailing comma, which name-lookup tolerates);
    the ``volume`` column is the R9 integerization basis."""
    nid, _ = _node_ids(scenario)
    name_of = {i: n for n, i in nid.items()}
    inv_edge = {i: e for e, i in _edge_order(scenario)[0].items()}
    routes: dict[tuple[str, str], list[tuple[tuple[str, ...], float]]] = {}
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except (StopIteration, csv.Error) as exc:
            raise RuntimeError("route_assignment.csv is empty/unreadable") from exc
        try:
            cols = {name: header.index(name)
                    for name in ("o_zone_id", "d_zone_id", "link_ids", "volume")}
        except ValueError as exc:
            raise RuntimeError(
                f"route_assignment.csv header drifted (missing column: {exc}); "
                f"got {header!r}"
            ) from exc
        for row in reader:
            if not row or len(row) <= max(cols.values()):
                continue
            try:
                o = name_of[int(row[cols["o_zone_id"]])]
                d = name_of[int(row[cols["d_zone_id"]])]
                edges = tuple(
                    inv_edge[int(x)] for x in row[cols["link_ids"]].split(";") if x
                )
                vol = float(row[cols["volume"]])
            except (ValueError, KeyError) as exc:
                raise RuntimeError(
                    f"route_assignment.csv row unparseable ({exc}): {row!r}"
                ) from exc
            routes.setdefault((o, d), []).append((edges, vol))
    if not routes:
        raise RuntimeError("route_assignment.csv carries no route rows")
    return routes


def _run_assignment_for_routes(
    scenario: EdocScenario, deadline: float | None
) -> dict[tuple[str, str], list[tuple[tuple[str, ...], float]]]:
    """Run ``assignment()`` (adapter plumbing, NOT a model row — R9) and parse
    ``route_assignment.csv``. The FW step runs on its own
    ``_ASSIGN_PERIOD_HOURS`` demand period (the pilot's genuinely-split
    781.25/218.75 state); ``assignment()``'s vehicle.csv is NEVER read (dead
    code — TAPLite.cpp:2238, measured header-only)."""
    workdir = tempfile.mkdtemp(prefix=f"tabench-edoc-dtalite-{os.getpid()}-assign-")
    try:
        start_h = int(_T0 // 3600.0)
        _write_gmns_sim(
            scenario, workdir, period_hours=(start_h, start_h + _ASSIGN_PERIOD_HOURS)
        )
        _run(_ASSIGN_CMD, cwd=workdir, deadline=deadline, what="dtalite R9 assignment")
        ra_path = os.path.join(workdir, "route_assignment.csv")
        if not os.path.exists(ra_path):  # rc is never trusted
            raise RuntimeError("assignment() reported success but wrote no route_assignment.csv")
        return _parse_route_assignment(ra_path, scenario)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _integerize_route_volumes(
    scenario: EdocScenario,
    od_routes: dict[tuple[str, str], list[tuple[tuple[str, ...], float]]],
) -> dict[str, tuple[tuple[str, ...], float]]:
    """The R9 integerization rule (pinned in adr-040): per OD, largest-remainder
    round the FW ``volume`` shares onto that OD's agent COUNT (per-route
    ``|count - share*N_od| <= 1`` — the disclosed mapping floor), tie-break by
    engine route order; then assign routes to the OD's agents (in agent-id
    order) by fractional-position interleaving, so the split is proportional
    at every point of the departure profile rather than block-ordered."""
    by_od: dict[tuple[str, str], list[str]] = {}
    dep_of: dict[str, float] = {}
    for aid, o, d, dep in zip(scenario.agent_ids, scenario.agent_origin,
                              scenario.agent_dest, scenario.agent_depart, strict=True):
        by_od.setdefault((o, d), []).append(aid)
        dep_of[aid] = float(dep)
    plans: dict[str, tuple[tuple[str, ...], float]] = {}
    for od, agents in by_od.items():
        if od not in od_routes:
            raise RuntimeError(f"assignment() emitted no route for OD {od[0]}->{od[1]}")
        routes = od_routes[od]
        total = sum(v for _r, v in routes)
        if total <= 0.0:
            raise RuntimeError(f"assignment() emitted nonpositive volume for OD {od}")
        n = len(agents)
        targets = [v / total * n for _r, v in routes]
        counts = [int(t) for t in targets]
        # largest remainder, tie-break by route file order (stable sort)
        for i in sorted(range(len(routes)), key=lambda j: -(targets[j] - counts[j]))[
            : n - sum(counts)
        ]:
            counts[i] += 1
        # fractional-position interleave: route r's j-th copy at (j+0.5)/n_r
        slots: list[tuple[float, int]] = []
        for r_idx, c in enumerate(counts):
            for j in range(c):
                heapq.heappush(slots, ((j + 0.5) / c, r_idx))
        for aid in agents:
            _pos, r_idx = heapq.heappop(slots)
            plans[aid] = (routes[r_idx][0], dep_of[aid])
    return plans


# --------------------------------------------------------------------------
# the model-side BR (mirrors the substrate TD-SP composition exactly, but
# returns the argmin ROUTE — pair 11: emitting the certifier's BR is legal,
# and the MSA blend is its principled smoothed use)
# --------------------------------------------------------------------------
def _td_best_route(
    out_edges, edge_head, field, origin_waits, origin, dest, depart_time,
    walk_bound, walk_count_bound,
):
    """Best (cost, route) over every walk of <= walk_bound edges, with the
    substrate's exact composition (waiting-not-allowed; origin wait on the
    first edge only; label-correcting via explicit enumeration). Deterministic:
    DFS order is a pure function of the scenario adjacency; a strictly cheaper
    walk replaces the incumbent, ties keep the first found."""
    best_cost = float("inf")
    best_route: tuple[str, ...] | None = None
    walks = 0
    stack: list[tuple[str, float, tuple[str, ...]]] = []
    for e in out_edges.get(origin, ()):
        w = origin_waits.wait(e, depart_time)
        entry = depart_time + w
        arr = entry + field.traversal_time(e, entry)
        stack.append((edge_head[e], arr, (e,)))
    while stack:
        node, t, route = stack.pop()
        walks += 1
        if walks > walk_count_bound:
            raise RuntimeError(
                f"BR walk count exceeded {walk_count_bound} on OD {origin}->{dest} "
                f"at walk_bound={walk_bound} (infrastructure guard, R6)"
            )
        if node == dest:
            cost = t - depart_time
            if cost < best_cost:
                best_cost, best_route = cost, route
            continue
        if len(route) >= walk_bound:
            continue
        for e in out_edges.get(node, ()):
            arr = t + field.traversal_time(e, t)
            stack.append((edge_head[e], arr, (*route, e)))
    return best_cost, best_route


def _msa_pick(instance_hash: str, k: int, improvers: list[str], n_move: int) -> set[str]:
    """The hash-derived deterministic switching picks (adr-040 ruling 3): rank
    improvers by ``sha256(instance_hash;msa;k;aid)`` and take the ``n_move``
    smallest digests — reproducible from the instance hash alone, with NO
    dependence on any RNG library's stream stability."""
    ranked = sorted(
        improvers,
        key=lambda aid: hashlib.sha256(
            f"{instance_hash};msa;{k};{aid}".encode()
        ).hexdigest(),
    )
    return set(ranked[:n_move])


# --------------------------------------------------------------------------
# the adapter: R9 step-0 plans + the row-local MSA 1/(k+2) loop
# --------------------------------------------------------------------------
class DTALiteSimulationAdapter:
    """Emit the EDOC-1 artifact contract for one :class:`EdocScenario`:
    ``P0`` from the R9 ``route_assignment.csv`` construction, then
    ``iterations`` MSA ``1/(k+2)`` best-response blends — each iterate replays
    the current plans through the pinned engine, builds the substrate frozen
    field + origin-wait profiles, finds each agent's BR route with the
    certifier's own composition, and moves a hash-picked ``1/(k+2)`` fraction
    of the improvers onto their BR routes (departures never move — G2). The
    loop is ROW-LOCAL (adr-036 names no substrate MSA deliverable; hoisting
    waits for a second router-less engine). ``iterations=0`` emits the R9
    step-0 FW split as-is — the row's negative-control state. The measured
    one-step overshoot (the pilot's 0.394 -> 0.513) is WHY the model is the
    smoothed blend, not the one-step BR (R12: realized-BR is Tier-B, never
    scored).

    ``X`` is defined by this adapter's own pinned replay of the final ``P``
    (the S2/S3 artifact contract); the certifier re-runs the identical replay
    for G1. Deterministic track: ``seedable=False`` — the engine consumes no
    seed (LCG re-seeded per time step), disclosed."""

    name = "dtalite-simulation"
    track = "edoc-deterministic"
    seedable = False

    def __init__(self, *, iterations: int = 16, keep_files: bool = False) -> None:
        self.iterations = int(iterations)
        self.keep_files = bool(keep_files)
        self.last_workdir: str | None = None

    def emit(self, scenario: EdocScenario, *, wall_seconds: float | None = None) -> EmittedBundle:
        """R9 assignment -> MSA loop -> pinned replay for ``X``, all under one
        wall deadline. RAISES on any engine/infra failure and on an in-loop
        boost-census crossing (pair 12's every-constructor-side-run clause:
        the model's own field would be boost-contaminated); never returns a
        partial or self-reported result."""
        if scenario.engine != ENGINE:
            raise ValueError(
                f"dtalite-simulation only runs DTALite instances; scenario "
                f"{scenario.name!r} pins engine {scenario.engine!r}"
            )
        assert_engine_pin(installed_engine_version(), scenario.engine_version)
        deadline = time.perf_counter() + wall_seconds if wall_seconds is not None else None
        instance_hash = scenario.content_hash()
        out_edges = scenario.out_edges()
        edge_head = scenario.head_of()
        fftt = scenario.fftt_of()
        trip = {
            aid: (o, d, float(dep))
            for aid, o, d, dep in zip(scenario.agent_ids, scenario.agent_origin,
                                      scenario.agent_dest, scenario.agent_depart,
                                      strict=True)
        }

        plans = _integerize_route_volumes(
            scenario, _run_assignment_for_routes(scenario, deadline)
        )
        keep = self.keep_files
        workdir = tempfile.mkdtemp(
            prefix=f"tabench-edoc-dtalite-{os.getpid()}-keep-" if keep
            else f"tabench-edoc-dtalite-{os.getpid()}-emit-"
        )
        self.last_workdir = workdir
        try:
            for k in range(self.iterations):
                it_dir = os.path.join(workdir, f"it{k:02d}")
                os.mkdir(it_dir)
                replay = pinned_simulation_replay(
                    scenario, plans, deadline=deadline, workdir=it_dir
                )
                crossings = _boost_crossings(scenario, replay.agents)
                if crossings:
                    raise ValueError(
                        f"MSA iterate {k}: {len(crossings)} agent(s) exit inside the "
                        f"engine's x10 discharge-boost window (onset "
                        f"{_boost_onset_s(scenario):.0f} s) — the model's own field "
                        "would be boost-contaminated; extend the horizon or lower "
                        "the demand (adr-036 pair 12, constructor-side runs RAISE)"
                    )
                field = build_field_from_records(
                    replay.field_records, fftt, scenario.dt, scenario.n_intervals,
                    scenario.field_semantics,
                )
                ow = build_origin_waits(
                    [(a.first_edge, a.departure, a.depart_delay)
                     for a in replay.agents.values() if a.experienced_time >= 0.0],
                    scenario.dt, scenario.n_intervals,
                )
                improvers_by_od: dict[tuple[str, str], list[str]] = {}
                br_route: dict[str, tuple[str, ...]] = {}
                for aid in scenario.agent_ids:
                    o, d, dep = trip[aid]
                    route, _dep = plans[aid]
                    c_drv = evaluate_route(field, ow, tuple(route), dep)
                    c_br, r_br = _td_best_route(
                        out_edges, edge_head, field, ow, o, d, dep,
                        scenario.walk_bound, scenario.walk_count_bound,
                    )
                    if r_br is not None and tuple(r_br) != tuple(route) and c_drv - c_br > 1e-9:
                        improvers_by_od.setdefault((o, d), []).append(aid)
                        br_route[aid] = tuple(r_br)
                frac = 1.0 / (k + 2)
                for _od, improvers in sorted(improvers_by_od.items()):
                    n_move = int(frac * len(improvers) + 0.5)
                    for aid in _msa_pick(instance_hash, k, improvers, n_move):
                        plans[aid] = (br_route[aid], plans[aid][1])
        finally:
            if not keep:
                shutil.rmtree(workdir, ignore_errors=True)
                self.last_workdir = None

        # X is DEFINED by this adapter's own pinned replay of the final P; a
        # boost crossing in X itself is left to the certify-time CENSOR arm.
        replay = pinned_simulation_replay(scenario, plans, deadline=deadline)
        return EmittedBundle(
            plans=plans,
            experienced=replay.agents,
            engine_version=scenario.engine_version,
            seed=int(scenario.seed),
        )


# --------------------------------------------------------------------------
# the scenario family (adr-040; R4 constants measured with the SHIPPED
# estimator on the installed 0.8.1 wheel)
# --------------------------------------------------------------------------
def _topology_digest(scenario: EdocScenario) -> str:
    """The F10 vetting key (S3 F3 from birth): what the separation gate vets is
    a TOPOLOGY — edge structure, lane (capacity) pattern, free-flow times and
    OD endpoints — so certification keys on this digest, never the forgeable
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


def _assert_no_lull_drop(scenario: EdocScenario) -> None:
    """The lull-drop construction gate (adr-040; a hazard adr-036 did not
    name): ``simulate()`` breaks out at the first interval ``t >= 600`` (60
    min into the period) where every loaded agent has completed
    (TAPLite.cpp:5387), silently DROPPING any still-pending departure at rc=0.
    Refuse any departure profile that ALLOWS such an all-completed instant
    before a later departure, bounding earliest completions by the free-flow
    shortest-path time (the fastest the engine can possibly finish anyone).
    The G3 completion census remains the certify-time backstop."""
    # free-flow shortest-path lower bound per distinct OD (tiny nets: Dijkstra)
    out = scenario.out_edges()
    head = scenario.head_of()
    fftt = scenario.fftt_of()

    def _ff_time(o: str, d: str) -> float:
        dist = {o: 0.0}
        heap = [(0.0, o)]
        while heap:
            t, node = heapq.heappop(heap)
            if node == d:
                return t
            if t > dist.get(node, float("inf")):
                continue
            for e in out.get(node, ()):
                nt = t + fftt[e]
                if nt < dist.get(head[e], float("inf")):
                    dist[head[e]] = nt
                    heapq.heappush(heap, (nt, head[e]))
        return float("inf")

    ff_min = min(
        _ff_time(o, d)
        for o, d in set(zip(scenario.agent_origin, scenario.agent_dest, strict=True))
    )
    exit_check_from = 600 * _SIM_STEP_S  # the engine's t >= 600 threshold
    prev: float | None = None
    for d in sorted(float(x) for x in scenario.agent_depart):
        danger_from = exit_check_from if prev is None else max(
            exit_check_from, prev + ff_min
        )
        if d >= danger_from:
            raise ValueError(
                f"EdocScenario {scenario.name!r}: departure at {d:.0f} s allows an "
                f"all-completed lull from {danger_from:.0f} s — the engine's "
                "t>=600 early exit (TAPLite.cpp:5387) would silently DROP it at "
                "rc=0; compress the departure profile (adr-040 lull-drop gate)"
            )
        prev = d


def build_dtalite_corridor_scenario(
    name: str,
    *,
    seed: int = 42,
    n_agents: int = 1000,
    agents_per_slot: int = 2,
    slot_step: float = 6.0,
    fftt_a: float = 300.0,
    fftt_b: float = 420.0,
    bottleneck_lanes: int = 1,
    free_lanes: int = 2,
    dt: float = 6.0,
    n_intervals: int = 3600,
    separation_factor: float = 5.0,
    floor_seconds: float = 10.0,
    backlog_bound: float = 60.0,
    replay_deadline_s: float = 30.0,
    walk_bound: int = 2,
    engine_version: str | None = None,
) -> EdocScenario:
    """The dtalite-simulation row's CORRIDOR family: two parallel routes
    ``O ->a1-> MA ->a2-> D`` (fftt 2x300 s, capacity 600 veh/h) vs
    ``O ->b1-> MB ->b2-> D`` (2x420 s, 1200 veh/h nominal — which the ~600
    veh/h transfer-admission law caps anyway, the documented engine cost law).
    1000 agents depart 2-per-6-s-slot over 50 min (1200 veh/h aggregate vs
    the ~600 veh/h per-route admission), so the R9 step-0 FW split piles the
    transfer queue onto route A and separates from the MSA-converged blend.

    Family pins measured on 0.8.1 (adr-040): ``dt = 6 s`` (the engine dynamics
    grid); ``departure_quantum = 6 s`` with AT MOST 2 departures per grid
    instant — BURSTY profiles are refused by shape because the engine's
    transfer-entrance service is non-FIFO under simultaneous arrivals
    (measured: 10-per-30-s slots PARK ~2 agents/slot at the queue tail until
    the arrival stream ends, ~4100 s vs ~330 s for same-instant peers, which
    no entry-time interval-mean field can represent — delta 600 s, censored;
    at <= 2 per instant the queue is FIFO and delta is ~2 s). The engine's
    minute clock float-prints 6 s grid points that are not binary-exact in
    minutes with a <= 1 s floor residual (measured: sched 420.2 min ->
    printed entry 07:00:11), so depart_delay reads in [-1, 0] s — orders of
    magnitude below the floor, disclosed in adr-040. The engine period is
    7->13 h (boost onset 4.8 h; the whole family exits hours earlier,
    census-gated); ``seed`` is hashed but engine-INERT (deterministic track,
    LCG re-seeded per time step; ``seed_list=()``)."""
    if n_agents % agents_per_slot:
        raise ValueError("n_agents must be a multiple of agents_per_slot")
    if agents_per_slot > 2:
        raise ValueError(
            "at most 2 departures per grid instant: the engine's transfer-"
            "entrance service is non-FIFO under burst arrivals (measured: "
            "10-per-slot parks ~2 agents/slot until the stream ends — no "
            "entry-time interval-mean field can represent that; adr-040)"
        )
    depart = np.array(
        [slot_step * (i // agents_per_slot) for i in range(n_agents)],
        dtype=np.float64,
    )
    sc = EdocScenario(
        name=name,
        edge_ids=("a1", "a2", "b1", "b2"),
        edge_tail=("O", "MA", "O", "MB"),
        edge_head=("MA", "D", "MB", "D"),
        edge_fftt=np.array([fftt_a, fftt_a, fftt_b, fftt_b]),
        edge_lanes=np.array(
            [bottleneck_lanes, bottleneck_lanes, free_lanes, free_lanes]
        ),
        agent_ids=tuple(f"v{i}" for i in range(n_agents)),
        agent_origin=("O",) * n_agents,
        agent_dest=("D",) * n_agents,
        agent_depart=depart,
        engine=ENGINE,
        engine_version=(
            engine_version if engine_version is not None else installed_engine_version()
        ),
        seed=int(seed),
        semantic_config=_semantic_config(),
        dt=dt,
        n_intervals=n_intervals,
        departure_quantum=slot_step,
        backlog_bound=backlog_bound,
        separation_factor=separation_factor,
        floor_seconds=floor_seconds,
        replay_deadline_s=replay_deadline_s,
        walk_bound=walk_bound,
        seed_list=(),  # the deterministic track: no macroreps, disclosed
        family="dtalite-corridor",
    )
    _engine_period_hours(sc)  # integral-hour horizon (eager)
    _assert_no_lull_drop(sc)  # the t>=600 early-exit hazard (eager)
    _edge_order(sc)  # parallel-edge refusal (eager)
    return sc


def shared_bottleneck_scenario(
    *, seed: int = 42, n_agents: int = 1000, engine_version: str | None = None
) -> EdocScenario:
    """A DELIBERATELY non-separating topology for the refusal demonstration
    (pair 12): the transfer bottleneck sits on the SHARED first edge ``sh``
    (O -> J) whose queue both routes' costs carry identically (the parser
    charges the transfer queue to the upstream edge), and the two downstream
    routes are near-identical — driven and BR costs cancel, both anchors score
    ~0, and the floor-DISPLAYED separation reads ~1x < the declared factor, so
    :func:`negative_control_separation` REFUSES it (a config error, never a
    certified row; the raw ratio would degenerate 0/0 — the S3 ruling-4 basis
    adopted from birth)."""
    depart = np.array([6.0 * (i // 2) for i in range(n_agents)], dtype=np.float64)
    sc = EdocScenario(
        name="dtalite-shared-bottleneck",
        edge_ids=("sh", "p1", "p2", "q1", "q2"),
        edge_tail=("O", "J", "P", "J", "Q"),
        edge_head=("J", "P", "D", "Q", "D"),
        edge_fftt=np.array([300.0, 150.0, 150.0, 156.0, 156.0]),
        edge_lanes=np.array([1, 2, 2, 2, 2]),  # the drop is the SHARED transfer
        agent_ids=tuple(f"v{i}" for i in range(n_agents)),
        agent_origin=("O",) * n_agents,
        agent_dest=("D",) * n_agents,
        agent_depart=depart,
        engine=ENGINE,
        engine_version=(
            engine_version if engine_version is not None else installed_engine_version()
        ),
        seed=int(seed),
        semantic_config=_semantic_config(),
        dt=6.0,
        n_intervals=3600,
        departure_quantum=6.0,
        backlog_bound=60.0,
        separation_factor=5.0,
        floor_seconds=30.0,
        replay_deadline_s=30.0,
        walk_bound=3,
        seed_list=(),
        family="dtalite-shared-bottleneck",
    )
    _assert_no_lull_drop(sc)
    return sc


def reference_scenario() -> EdocScenario:
    """The pinned dtalite-simulation reference instance (adr-040). Family
    constants MEASURED with the SHIPPED estimator on the installed
    DTALite==0.8.1 wheel — the R4 re-derivation recorded in adr-040."""
    return build_dtalite_corridor_scenario("dtalite-simulation-ref")


# --------------------------------------------------------------------------
# row certification (vetting + substrate certificate + boost censor + the R3
# harness self-cross-check) and the negative-control separation gate
# --------------------------------------------------------------------------
def _field_selfcheck(
    scenario: EdocScenario,
    plans: dict[str, tuple[tuple[str, ...], float]],
    replay: ReplayResult,
    *,
    tolerance_s: float,
) -> dict[str, float]:
    """The R3 harness self-cross-check (adr-040 disclosure, the adr-039 shape):
    DTALite has NO standalone router artifact — ``simulation()`` never routes,
    ``assignment()`` never simulates — so R3's engine-router cross-check clause
    has nothing to bind to and the substrate TD-SP is normative-only on this
    row. Instead every driven cost is re-derived by an independently written
    field composition and compared to :func:`~tabench.edoc.tdsp.evaluate_route`
    under ``r3_tolerance_s`` — a field-arithmetic regression guard, infra
    RAISE, never a censor."""
    field = build_field_from_records(
        replay.field_records, scenario.fftt_of(), scenario.dt, scenario.n_intervals,
        scenario.field_semantics,
    )
    ow = build_origin_waits(
        [(a.first_edge, a.departure, a.depart_delay)
         for a in replay.agents.values() if a.experienced_time >= 0.0],
        scenario.dt,
        scenario.n_intervals,
    )
    diffs = []
    for route, dep in plans.values():
        mine = evaluate_route(field, ow, tuple(route), float(dep))
        # independent composition of the same frozen profiles
        tau = float(dep) + ow.wait(route[0], float(dep))
        for e in route:
            tau += field.traversal_time(e, tau)
        diffs.append(abs(mine - (tau - float(dep))))
    mean_d = sum(diffs) / len(diffs) if diffs else 0.0
    max_d = max(diffs, default=0.0)
    if max_d > tolerance_s:
        raise RuntimeError(
            f"R3 self-cross-check FAILED: evaluate_route vs direct field composition "
            f"differ by max {max_d:.3f}s (mean {mean_d:.3f}s) > tolerance {tolerance_s}s "
            "— the harness field arithmetic disagrees with itself"
        )
    return {"r3_mean_s": mean_d, "r3_max_s": max_d, "r3_tolerance_s": float(tolerance_s)}


def certify_emitted(
    scenario: EdocScenario,
    emitted: EmittedBundle,
    *,
    wall_seconds: float | None = None,
) -> dict[str, float]:
    """The ROW's full certification path: the separation-vetting assertion
    (F10, topology-digest-keyed), the G0-G4 + ``RG_D1`` substrate certificate,
    the certify-time BOOST-CENSUS CENSOR (pair 12/D3: computed from
    ``emitted.experienced``, which G1 has just proven equal to the replay
    tuple-for-tuple — the census on X IS the census on the replay, zero extra
    engine calls), and the R3 harness self-cross-check — all under one wall
    deadline."""
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
    # Boost census (pair 12/D3): surface the boost diagnostic whenever the
    # emitted X crosses the window, EVEN IF a downstream substrate gate already
    # censored (e.g. a too-short horizon also truncates late agents at G3) — the
    # boost window is the instance-design root cause and its count must reach
    # the metrics dict. Trustworthy on ``emitted.experienced`` because G1 has
    # bound X to the replay tuple-for-tuple before any post-G1 censor; a doctored
    # X is independently caught at G1. A no-crossing emission passes through
    # unchanged (the reference exits hours before its 4.8 h onset).
    metrics = _boost_censor(scenario, metrics, emitted.experienced)
    if metrics.get("feasible") == 1.0:
        try:
            replay = runner(scenario, emitted.plans)
        except PlanReplayFailure as exc:
            # S3 F4 typing from birth: G1 just replayed these SAME plans twice,
            # so a failure of this third (R3 self-check) replay is a
            # certifier-side fault, never an invalid emission.
            raise RuntimeError(
                "R3 self-cross-check replay failed after G1 replayed the same "
                f"plans twice — certifier-side fault, not an invalid emission: {exc}"
            ) from exc
        metrics.update(
            _field_selfcheck(
                scenario, emitted.plans, replay, tolerance_s=scenario.r3_tolerance_s
            )
        )
    return metrics


def negative_control_separation(
    scenario: EdocScenario,
    *,
    wall_seconds: float | None,
    control_iterations: int = 0,
    converged_iterations: int = 16,
) -> dict[str, float]:
    """The negative-control separation gate on the DETERMINISTIC track
    (adr-036 R4): control = the R9 step-0 FW split emitted as-is
    (``iterations=0``) vs the MSA-converged state, each a single emission (no
    macroreps — the deterministic track, disclosed). Separation is computed on
    FLOOR-DISPLAYED values (``max(rg, floor_gap)`` each side) — the adr-039
    ruling-4 basis adopted from birth, because the 0/0 shared-topology vacuous
    pass exists on this track too and adr-036's own leaderboard rule displays
    a sub-floor value AT the floor. A censored anchor RAISES ``ValueError`` —
    which automatically realizes the pair-12 constructor refusal for boost
    crossings (the anchor path applies the same certify-time boost censor).
    On success the TOPOLOGY is marked separation-vetted (F10)."""
    from ...metrics.edoc_gaps import EdocEvaluator

    deadline = time.perf_counter() + wall_seconds if wall_seconds is not None else None

    def _emit_certify(iters: int) -> dict[str, float]:
        adapter = DTALiteSimulationAdapter(iterations=iters)
        emitted = adapter.emit(scenario, wall_seconds=_remaining(deadline))
        runner = make_replay_runner(deadline=deadline)
        metrics = EdocEvaluator(scenario, runner).certify(emitted)
        # Same unconditional boost census as certify_emitted (a no-op on the
        # reference topology, which has no crossings).
        return _boost_censor(scenario, metrics, emitted.experienced)

    control = _emit_certify(control_iterations)
    converged = _emit_certify(converged_iterations)
    if control["feasible"] != 1.0 or converged["feasible"] != 1.0:
        raise ValueError(
            f"negative-control separation: an anchor censored (control feasible "
            f"{control['feasible']}, converged feasible {converged['feasible']}) — "
            "the separation anchors must be real feasible measurements"
        )
    a_rg = float(control["rg_d1"])
    c_rg = float(converged["rg_d1"])
    # displayed values: a sub-floor value reads AT the row floor (adr-036).
    a_disp = max(a_rg, float(control["floor_gap"]))
    c_disp = max(c_rg, float(converged["floor_gap"]))
    separation = a_disp / c_disp if c_disp > 0 else float("inf")
    if separation < scenario.separation_factor:
        raise ValueError(
            f"negative-control separation FAILED for {scenario.name!r}: displayed "
            f"step-0 control RG_D1 {a_disp:.5f} (raw {a_rg:.5f}) / displayed "
            f"converged {c_disp:.5f} (raw {c_rg:.5f}) = {separation:.2f}x < the "
            f"declared {scenario.separation_factor}x (a non-separating topology — "
            "e.g. a shared-transfer bottleneck, whose queue cancels between the "
            "anchors — is a construction error, never a certified row)"
        )
    # F10: mark the TOPOLOGY vetted (digest-keyed, S3 F3 from birth).
    _SEPARATION_VETTED_TOPOLOGIES.add(_topology_digest(scenario))
    return {
        "control_rg_d1": a_rg,
        "converged_rg_d1": c_rg,
        "control_displayed": a_disp,
        "converged_displayed": c_disp,
        "separation": separation,
        "separation_factor": float(scenario.separation_factor),
    }
