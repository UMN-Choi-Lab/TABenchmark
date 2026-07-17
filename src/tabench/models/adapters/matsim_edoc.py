"""MATSim as the second EDOC-1 row — the FIRST agent-based, FIRST
stochastic-track external engine (adr-036 / adr-039).

Like ``sumo-duaiterate`` (adr-037), MATSim has *no declared cost law*: the QSim
queue model's link time is constant below flow capacity (the adr-030 blocker),
so the engine IS the instance (adr-036) and the certifier re-derives every
scored number by re-running the pinned engine in zero-replanning replay
(``lastIteration = firstIteration``) on the model's emitted plans (G1). This
adapter is an **EDOC producer**, not a ``TrafficAssignmentModel``: it emits
plans ``P``, the door-to-door experienced record ``X`` (from its OWN pinned
replay of ``P`` — the adr-037 artifact-contract clarification), and provenance.

**Stochastic track (adr-036 R5).** ``global.randomSeed`` is outcome-bearing
(ChangeExpBeta selection + the qsim merge order), so the row is scored as P8
macroreps over the instance's pinned ``seed_list`` through
:func:`tabench.edoc.macrorep.certify_macroreps` — mean ``RG_D1`` + bootstrap
CI; single-seed readouts are structurally impossible (:func:`certify_row` is
the row's only score entry point).

**The corrected R10 record (adr-039, pilot-record correction).** The
same-timestamp event-tie permutation is a MULTITHREADING artifact, not a
replay-vs-original effect: measured, ``numberOfThreads=8`` breaks same-seed
events byte-stability (104/1400 lines permuted, multiset identical) while at
``numberOfThreads=1`` the replay is raw-byte-identical to the certified run
even with forced ties / reversed input order / a different replay seed. So (a)
``numberOfThreads=1`` is pinned in ``global``, ``qsim`` AND ``eventsManager``
— all inside the hashed ``semantic_config``; (b) the G1 certificate hash is
the R10-canonicalized stream hash (sorted same-timestamp runs, post-decompress
— invariant across thread counts / seeds / input order); raw-byte identity at
threads=1 is a stricter engine-gated bonus test, never the gate.

**Subprocess discipline (the S2 hazards, inherited verbatim):** java is
addressed ONLY via ``TABENCH_JAVA_HOME``/``JAVA_HOME`` and the engine ONLY via
``TABENCH_MATSIM_HOME`` (``_matsim_io``, F8); ``stdin=DEVNULL``; every call in
its own process group so ONE ``killpg(SIGKILL)`` reaps the whole JVM tree (F2
— an orphaned JVM idles at multi-hundred-MB); the hashed ``replay_deadline_s``
ALWAYS bounds a certifier replay (F3, the R6 fixed-JVM-startup-plus-multiple
form); ``rc`` is NEVER trusted (every step re-reads its artifact);
:class:`~tabench.edoc.replay.PlanReplayFailure` is raised ONLY by the
plan-replay java step (R6 first arm) — everything else RAISES un-laundered;
``removeStuckVehicles=false`` so gridlock censors as G3 incompletion, never
vanished demand; ``mkdtemp`` + ``rmtree`` in ``finally``.

**G0 pins.** The engine identity read at certify time is
``matsim-2025.0;jar-md5=…;jdk-major=…`` (:func:`installed_engine_version`,
inside the instance hash via ``engine_version``); the FULL JDK build
(``Temurin-21.0.11+10``) is a family-declared constant embedded in the hashed
``semantic_config`` and enforced with a G0 RAISE against ``java -version``
(a JDK patch drift is an uncontrolled G1 censor surface — adr-036 G0).

This module imports stdlib + numpy only (the engine is Java, not a wheel), so
it is importable everywhere; engine absence surfaces as the runtime
:func:`~tabench.models.adapters._matsim_io.matsim_available` probe / a G0
RAISE. Design: docs/design/adr-036 + docs/design/adr-039-matsim.md.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import signal
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable

import numpy as np

from ...edoc.canon import hash_matsim_artifacts
from ...edoc.field import build_field_from_records, build_origin_waits
from ...edoc.macrorep import MacrorepResult, certify_macroreps
from ...edoc.replay import (
    EmittedBundle,
    PlanReplayFailure,
    ReplayResult,
    assert_engine_pin,
)
from ...edoc.scenario import EdocScenario
from ...edoc.tdsp import evaluate_route
from ._matsim_io import (
    MATSIM_RELEASE,
    installed_engine_version,
    java_binary,
    java_version_string,
    matsim_jar,
    parse_events,
    parse_output_plans,
)

__all__ = [
    "ENGINE",
    "MatsimAdapter",
    "build_matsim_diamond_scenario",
    "certify_emitted",
    "certify_row",
    "installed_engine_version",
    "make_replay_runner",
    "matsim_reference_scenario",
    "matsim_shared_bottleneck_scenario",
    "negative_control_separation",
    "pinned_matsim_replay",
]

ENGINE = "matsim"

# --------------------------------------------------------------------------
# family-declared pinned constants (every outcome-bearing one rides in the
# hashed semantic_config below — a drift mints a new instance hash, MAJOR-5)
# --------------------------------------------------------------------------
# The FULL JDK build the certificate is defined on (adr-036 G0: a JDK
# minor/patch drift is an uncontrolled G1 censor surface). Enforced with a G0
# RAISE against `java -version` before any engine run.
_PINNED_FULL_JDK = "Temurin-21.0.11+10"
# The engine-side capacity dial: MATSim link flow capacity is EXPLICIT (veh/h
# per capperiod, unlike meso's emergent capacity), written as
# ``capacity = edge_lanes * _MATSIM_CAP_PER_LANE_VPH``; a capacity drop is
# fewer lanes on a route-distinguishing edge.
_MATSIM_CAP_PER_LANE_VPH = 600.0
# Fixed engine-clock offset: scenario time t maps to engine time t + _T0 (the
# writer adds it, the parsers subtract it), keeping every scenario departure on
# [0, horizon) while dodging any midnight-boundary special-casing in MATSim.
_MATSIM_T0 = 3600.0
# JVM invocation pins: bounded heap (the R7 CI sizing constraint — a
# pathological instance OOMs the child JVM, never the runner) + locale pins
# (java number formatting in XML writers must not float with the host locale).
_JVM_FLAGS = ("-Xmx1g", "-Duser.language=en", "-Duser.country=US")
# The pinned route/selection-ONLY replanning strategy set (G2 construction
# rule): a departure-time- or mode-mutating strategy would break the
# exact-departure bijection, so the writer refuses them eagerly.
_REPLANNING_STRATEGIES: tuple[tuple[str, float], ...] = (
    ("ChangeExpBeta", 0.7),
    ("ReRoute", 0.3),
)
_FORBIDDEN_STRATEGY_TOKENS = (
    "TimeAllocationMutator",
    "SubtourModeChoice",
    "ChangeTripMode",
    "ChangeLegMode",
    "ChangeSingleTripMode",
)
_PLAN_MEMORY = 5
# Certifier-written SCORING + ROUTER constants — outcome-bearing (the S3
# review's F2, executed: marginalUtilityOfTraveling -6.0 -> -6000.0 changed
# 13/100 selected routes and +609 s total experienced time under the same seed
# with the instance hash unmoved), so they ride in the hashed semantic_config
# like every other pinned option: ChangeExpBeta consumes the scores these
# produce, and the router is the ReRoute best-response oracle.
_ROUTING_ALGORITHM = "SpeedyALT"
_ACTIVITY_TYPICAL_DURATIONS: tuple[tuple[str, str], ...] = (
    ("home", "12:00:00"),
    ("work", "08:00:00"),
)
_MODE_MARGINAL_UTILITY_HR: tuple[tuple[str, float], ...] = (
    ("car", -6.0),
    ("pt", -6.0),
    ("walk", -6.0),
)
# The replay iteration: firstIteration == lastIteration derives from this ONE
# constant (forgery pair N6 — a replanning "replay" cannot be smuggled in).
_REPLAY_ITERATION = 0

_READBACK_RTOL = 1e-6  # output_network read-back tolerance (engine echoes doubles)

# Topologies separation-vetted by :func:`negative_control_separation` (F10):
# the row's certification path refuses instances whose TOPOLOGY digest is not
# in this set. Keyed on a topology digest, NOT the family string (the S3
# review's F3, executed: a never-vetted self-certifying shared-edge topology
# relabeled ``family='matsim-diamond'`` sailed past a name-keyed gate) —
# runtime state only, no instance hash involved.
_SEPARATION_VETTED_TOPOLOGIES: set[str] = set()


def _topology_digest(scenario: EdocScenario) -> str:
    """The F10 vetting key (S3 review F3): what the separation gate actually
    vets is a TOPOLOGY — the edge structure, lane (capacity) pattern, free-flow
    times and OD endpoints that make the negative control separate — so
    certification is keyed on this digest of exactly those fields, never on the
    forgeable ``family`` STRING."""
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


def _semantic_config() -> str:
    """The instance's semantic engine-config string, DERIVED from the actually
    pinned constants so a drift in ANY of them moves the instance hash (the
    ``_MESO_OPTS`` pattern of adr-037; hash-coverage rides on the scenario's
    ``semantic_config`` field)."""
    strategies = ",".join(f"{name}:{w:g}" for name, w in _REPLANNING_STRATEGIES)
    acts = ",".join(f"{a}:{d}" for a, d in _ACTIVITY_TYPICAL_DURATIONS)
    mode_utl = ",".join(f"{m}:{u:g}" for m, u in _MODE_MARGINAL_UTILITY_HR)
    return (
        f"{MATSIM_RELEASE};threads=1;jdk={_PINNED_FULL_JDK};"
        f"capPerLane={_MATSIM_CAP_PER_LANE_VPH:g};t0={_MATSIM_T0:g};"
        f"strategies={strategies};planMemory={_PLAN_MEMORY};removeStuck=false;"
        f"replayIt={_REPLAY_ITERATION};router={_ROUTING_ALGORITHM};"
        f"acts={acts};modeUtl={mode_utl};jvm={','.join(_JVM_FLAGS)}"
    )


def _assert_jdk_pin() -> None:
    """G0 full-JDK pin: the ADDRESSED java must be the family-declared build
    (:data:`_PINNED_FULL_JDK`, inside the hashed semantic_config). A mismatch
    is a configuration error and RAISES eagerly — never a censor."""
    banner = java_version_string()
    if _PINNED_FULL_JDK not in banner:
        raise ValueError(
            f"G0 full-JDK pin: `java -version` reports {banner.strip()!r} but the "
            f"family pins {_PINNED_FULL_JDK!r}; a JDK build drift is an uncontrolled "
            "G1 censor surface (adr-036 G0) — point TABENCH_JAVA_HOME at the pinned JDK"
        )


# --------------------------------------------------------------------------
# wall-deadline plumbing (the S2 discipline verbatim)
# --------------------------------------------------------------------------
def _remaining(deadline: float | None) -> float | None:
    """Seconds left on the single wall deadline, or ``None`` if unbudgeted.
    RAISES if the deadline already passed (a prior phase ate the budget)."""
    if deadline is None:
        return None
    left = deadline - time.perf_counter()
    if left <= 0.0:
        raise RuntimeError("matsim wall deadline exhausted before the next step")
    return left


def _reap_group(proc: subprocess.Popen) -> None:
    """SIGKILL the subprocess's whole process GROUP, then reap it (F2):
    ``subprocess`` times out only the direct child; a JVM (and any
    ``jspawnhelper`` children) otherwise orphans to init and keeps idling at
    multi-hundred-MB RSS. ``start_new_session=True`` makes one ``killpg`` take
    the whole tree down."""
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
    """The certifier's hard replay deadline (F3): the scenario-declared hashed
    ``replay_deadline_s`` measured from now, intersected with any tighter
    caller wall — so the hashed constant ALWAYS bounds a certifier replay and a
    head-blocking plan cannot hang the certifier unboundedly (adr-036 R6).

    Returns ``(deadline, clipped_by_caller)``. The flag carries the R6
    crash-vs-censor typing for a mid-replay timeout (the S3 review's F1,
    executed: a ``now + 1.5 s`` caller wall killed the replay JVM mid-startup
    and was laundered into ``feasible=0``): only an expiry of the
    SCENARIO-declared deadline is the model's fault (an unexecutable / hanging
    plan — censor); a caller wall clipping below it is a certifier-side budget
    exhaustion and must RAISE as infrastructure, never censor."""
    scen_deadline = time.perf_counter() + float(scenario.replay_deadline_s)
    if deadline is None or deadline >= scen_deadline:
        return scen_deadline, False
    return deadline, True


def _run_java(
    args: list[str], *, cwd: str, deadline: float | None, what: str,
    censor_on_fail: bool = False, censor_on_timeout: bool | None = None,
) -> None:
    """Run the pinned JVM: absolute java (F8), pinned JVM flags, the release
    jar on the classpath (its manifest ``Class-Path`` pulls the adjacent
    ``libs/``). Missing toolchain RAISES (infra, never a censor)."""
    java = java_binary()
    jar = matsim_jar()
    if java is None or jar is None:
        raise RuntimeError(
            "matsim toolchain unaddressed: set TABENCH_MATSIM_HOME and "
            "TABENCH_JAVA_HOME (adr-039 addressing rule)"
        )
    _run(
        [java, *_JVM_FLAGS, "-cp", jar, *args],
        cwd=cwd, deadline=deadline, what=what, censor_on_fail=censor_on_fail,
        censor_on_timeout=censor_on_timeout,
        env={**os.environ, "JAVA_HOME": os.path.dirname(os.path.dirname(java))},
    )


def _run(
    cmd: list[str], *, cwd: str, deadline: float | None, what: str,
    censor_on_fail: bool = False, censor_on_timeout: bool | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run one subprocess under the S2 discipline: ``stdin=DEVNULL``, its OWN
    process group (one killpg reaps the whole tree, F2), the single wall
    deadline as timeout.

    Crash-vs-censor (adr-036 R6): by default a timeout / OS error / nonzero rc
    is a certifier-side INFRASTRUCTURE ``RuntimeError``. For the ONE step that
    replays the MODEL's emitted plans, the caller passes ``censor_on_fail=True``
    so a genuine subprocess crash raises
    :class:`~tabench.edoc.replay.PlanReplayFailure` (the censor signal). The
    TIMEOUT typing is split out (the S3 review's F1): ``censor_on_timeout``
    (default: follows ``censor_on_fail``) is passed as ``False`` by the replay
    when a CALLER wall clipped below the scenario-declared deadline, so a
    certifier-side budget kill RAISES instead of censoring — only a
    scenario-deadline expiry blames the plan. A ``_remaining`` pre-exhaustion
    and a missing binary (``OSError``) stay infrastructure RAISEs on every
    step, replay included."""
    if censor_on_timeout is None:
        censor_on_timeout = censor_on_fail
    timeout = _remaining(deadline)  # pre-exhaustion -> plain RuntimeError (infra)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env if env is not None else dict(os.environ),
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
    # rc is NEVER trusted beyond this point: every caller re-reads its artifact.


# --------------------------------------------------------------------------
# instance -> MATSim inputs (deterministic, a function of hashed fields only)
# --------------------------------------------------------------------------
def _sec_to_hms(t: float) -> str:
    """Engine-grid time formatting (integer seconds — the family's declared
    ``departure_quantum`` is 1.0 s). A sub-second time is a config error."""
    r = int(round(t))
    if abs(t - r) > 1e-6 or r < 0:
        raise ValueError(f"matsim writer needs nonnegative integer-second times, got {t!r}")
    h, rem = divmod(r, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _node_coords(scenario: EdocScenario) -> dict[str, tuple[float, float]]:
    """A deterministic 2-D node layout. Coordinates carry NO cost (explicit
    lengths + freespeed define traversal; MATSim reads geometry only for
    routing landmarks); the grid keeps the writer reproducible."""
    nodes = sorted(set(scenario.edge_tail) | set(scenario.edge_head))
    w = max(1, math.ceil(math.sqrt(len(nodes))))
    return {n: (500.0 * (i // w), 500.0 * (i % w)) for i, n in enumerate(nodes)}


def _write_network(scenario: EdocScenario, workdir: str) -> str:
    """Write ``network.xml``: every scenario edge with ``length = fftt *
    canon_speed_mps`` (free-flow time = fftt exactly), ``freespeed =
    canon_speed_mps``, ``permlanes = edge_lanes``, ``capacity = edge_lanes *
    _MATSIM_CAP_PER_LANE_VPH`` (the explicit engine-side capacity dial, hashed
    via lanes + the semantic_config constant). MATSim's TripRouter aborts on a
    non-strongly-connected car network, so deterministic RETURN links (id
    prefix ``__ret``) connect every sink (out-degree 0) to every source
    (in-degree 0); they are engine-side plumbing — never scenario edges, never
    parsed into the field, and no agent can drive them (no activity sits on
    them and they lead only back to the entry)."""
    coords = _node_coords(scenario)
    lengths = scenario.length_of()
    lanes = scenario.lanes_of()
    v = float(scenario.canon_speed_mps)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE network SYSTEM "http://www.matsim.org/files/dtd/network_v2.dtd">',
        f'<network name="{scenario.name}">',
        "<nodes>",
    ]
    for n, (x, y) in coords.items():
        lines.append(f'\t<node id="{n}" x="{x:.1f}" y="{y:.1f}"/>')
    lines.append("</nodes>")
    lines.append('<links capperiod="01:00:00">')
    for eid, tail, head in zip(
        scenario.edge_ids, scenario.edge_tail, scenario.edge_head, strict=True
    ):
        cap = lanes[eid] * _MATSIM_CAP_PER_LANE_VPH
        lines.append(
            f'\t<link id="{eid}" from="{tail}" to="{head}" length="{lengths[eid]:.6f}" '
            f'capacity="{cap:.6f}" freespeed="{v:.6f}" permlanes="{lanes[eid]}" modes="car"/>'
        )
    # deterministic connectivity plumbing: sinks -> sources (sorted, so the
    # file is a pure function of the hashed graph).
    tails, heads = set(scenario.edge_tail), set(scenario.edge_head)
    sinks = sorted((heads | tails) - tails)  # nodes with no outgoing edge
    sources = sorted((heads | tails) - heads)  # nodes with no incoming edge
    ret = 0
    for snk in sinks:
        for src in sources:
            lines.append(
                f'\t<link id="__ret{ret}" from="{snk}" to="{src}" '
                f'length="{60.0 * v:.6f}" capacity="10000.000000" '
                f'freespeed="{v:.6f}" permlanes="1" modes="car"/>'
            )
            ret += 1
    lines.append("</links>")
    lines.append("</network>")
    path = os.path.join(workdir, "network.xml")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _readback_network(out_network: str, scenario: EdocScenario) -> None:
    """Read-back (F6, rc never trusted): re-parse the engine's echoed
    ``output_network.xml.gz`` and RAISE if any scenario link's
    length/freespeed/capacity/permlanes drifted beyond ``_READBACK_RTOL``
    relative — the compiled dynamics must be the hashed dynamics. Unparseable
    output RAISES the contract ``RuntimeError`` (F9a)."""
    import gzip as _gzip

    lengths = scenario.length_of()
    lanes = scenario.lanes_of()
    v = float(scenario.canon_speed_mps)
    try:
        with open(out_network, "rb") as fh:
            data = fh.read()
        if data[:2] == b"\x1f\x8b":
            data = _gzip.decompress(data)
        root = ET.fromstring(data)
    except (ET.ParseError, _gzip.BadGzipFile, OSError) as exc:
        raise RuntimeError(f"network read-back: {out_network} unparseable ({exc})") from exc
    seen: dict[str, tuple[float, float, float, float]] = {}
    links_el = root.find("links")
    for link in (links_el.findall("link") if links_el is not None else root.iter("link")):
        lid = link.get("id")
        if lid is None or lid not in lengths:
            continue  # __ret plumbing / foreign links are not scenario edges
        seen[lid] = (
            float(link.get("length")),
            float(link.get("freespeed")),
            float(link.get("capacity")),
            float(link.get("permlanes")),
        )
    for eid in scenario.edge_ids:
        if eid not in seen:
            raise RuntimeError(f"network read-back: engine dropped scenario link {eid!r}")
        length, speed, cap, perml = seen[eid]
        want = (
            ("length", lengths[eid], length),
            ("freespeed", v, speed),
            ("capacity", lanes[eid] * _MATSIM_CAP_PER_LANE_VPH, cap),
            ("permlanes", float(lanes[eid]), perml),
        )
        for label, declared, got in want:
            if abs(got - declared) > _READBACK_RTOL * max(abs(declared), 1e-9):
                raise RuntimeError(
                    f"network read-back: link {eid!r} {label} {declared!r} -> {got!r} "
                    "(the engine rewrote a hashed dynamic — infra RAISE, adr-039 F6)"
                )


def _terminal_links(scenario: EdocScenario) -> tuple[dict[str, str], dict[str, str]]:
    """Per-agent activity anchoring: MATSim activities sit ON links, so each
    origin node needs exactly ONE outgoing scenario edge (the departure link)
    and each destination node exactly ONE incoming scenario edge (the arrival
    link) — the plumbing-edge family pattern. Ambiguity would hand the ENGINE a
    choice the instance does not hash, so it RAISES (a config error)."""
    out_of: dict[str, list[str]] = {}
    into: dict[str, list[str]] = {}
    for eid, tail, head in zip(
        scenario.edge_ids, scenario.edge_tail, scenario.edge_head, strict=True
    ):
        out_of.setdefault(tail, []).append(eid)
        into.setdefault(head, []).append(eid)
    origin_link: dict[str, str] = {}
    dest_link: dict[str, str] = {}
    for o in set(scenario.agent_origin):
        cands = out_of.get(o, [])
        if len(cands) != 1:
            raise ValueError(
                f"matsim family contract: origin node {o!r} needs exactly one outgoing "
                f"edge (the departure link), found {sorted(cands)!r}"
            )
        origin_link[o] = cands[0]
    for d in set(scenario.agent_dest):
        cands = into.get(d, [])
        if len(cands) != 1:
            raise ValueError(
                f"matsim family contract: destination node {d!r} needs exactly one "
                f"incoming edge (the arrival link), found {sorted(cands)!r}"
            )
        dest_link[d] = cands[0]
    return origin_link, dest_link


def _write_plans(
    plans: dict[str, tuple[tuple[str, ...], float]] | None,
    scenario: EdocScenario,
    workdir: str,
) -> str:
    """Write ``plans.xml`` (population_v6). Two modes:

    * ``plans=None`` (emit-time INITIAL demand): one leg per agent with NO
      route — MATSim's PrepareForSim routes it at free flow, the deterministic
      AON iteration-0 state.
    * ``plans`` given (the replay input): each agent's selected plan carries
      the emitted route verbatim (``<route type="links">``) — zero replanning
      replays exactly this.

    Rows are written sorted by (departure, id); the replay output is measured
    invariant to input order (adr-039), the sort just keeps the writer a pure
    function of its arguments."""
    origin_link, dest_link = _terminal_links(scenario)
    trip = {
        aid: (o, d, float(dep))
        for aid, o, d, dep in zip(
            scenario.agent_ids, scenario.agent_origin, scenario.agent_dest,
            scenario.agent_depart, strict=True,
        )
    }
    rows = []
    if plans is None:
        for aid, (o, d, dep) in trip.items():
            rows.append((dep, aid, None, origin_link[o], dest_link[d]))
    else:
        for aid, (route, dep) in plans.items():
            if not route:
                raise ValueError(f"agent {aid!r}: empty route cannot be written")
            rows.append((float(dep), aid, tuple(route), route[0], route[-1]))
    rows.sort(key=lambda r: (r[0], r[1]))

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd">',
        "<population>",
    ]
    for dep, aid, route, first_link, last_link in rows:
        end_time = _sec_to_hms(dep + _MATSIM_T0)
        lines.append(f'<person id="{aid}">')
        lines.append('\t<plan selected="yes">')
        lines.append(f'\t\t<activity type="home" link="{first_link}" end_time="{end_time}"/>')
        if route is None:
            lines.append('\t\t<leg mode="car"/>')
        else:
            edges = " ".join(route)
            lines.append('\t\t<leg mode="car">')
            lines.append(
                f'\t\t\t<route type="links" start_link="{first_link}" '
                f'end_link="{last_link}">{edges}</route>'
            )
            lines.append("\t\t</leg>")
        lines.append(f'\t\t<activity type="work" link="{last_link}"/>')
        lines.append("\t</plan>")
        lines.append("</person>")
    lines.append("</population>")
    path = os.path.join(workdir, "plans.xml")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _write_config(
    scenario: EdocScenario,
    workdir: str,
    *,
    first_it: int,
    last_it: int,
    plans_name: str = "plans.xml",
    out_name: str = "out",
) -> str:
    """Write the certifier's ``config.xml`` — the certifier writes EVERY config
    itself; a model-supplied config is never accepted (replay-config forgery,
    pair N6, closed structurally). The pinned module set (all inside the hashed
    ``semantic_config``): ``numberOfThreads=1`` in global/qsim/eventsManager
    (the corrected-R10 determinism pin), ``randomSeed = scenario.seed``,
    route/selection-ONLY replanning (an eager gate refuses time/mode-mutating
    strategies — G2), scoring modeParams re-declared for car+pt+walk (the
    measured 2025.0 NPE gotcha), and NO stuck-vehicle removal (gridlock shows
    as G3 incompletion, never vanished demand)."""
    for name, _w in _REPLANNING_STRATEGIES:
        for token in _FORBIDDEN_STRATEGY_TOKENS:
            if token in name:
                raise ValueError(
                    f"pinned strategy {name!r} mutates departure time / mode — a "
                    "config error (adr-036 G2 construction rule): the exact-departure "
                    "bijection would break"
                )
    horizon = float(scenario.dt) * int(scenario.n_intervals)
    start = _sec_to_hms(_MATSIM_T0)
    end = _sec_to_hms(_MATSIM_T0 + horizon)
    strategy_sets = "\n".join(
        '\t\t<parameterset type="strategysettings">\n'
        f'\t\t\t<param name="strategyName" value="{name}"/>\n'
        f'\t\t\t<param name="weight" value="{w}"/>\n'
        "\t\t</parameterset>"
        for name, w in _REPLANNING_STRATEGIES
    )
    # scoring blocks DERIVED from the hashed module constants (F2): a drift in
    # any typicalDuration / marginal utility moves the instance hash.
    activity_sets = "\n".join(
        '\t\t\t<parameterset type="activityParams">\n'
        f'\t\t\t\t<param name="activityType" value="{act}"/>\n'
        f'\t\t\t\t<param name="typicalDuration" value="{dur}"/>\n'
        "\t\t\t</parameterset>"
        for act, dur in _ACTIVITY_TYPICAL_DURATIONS
    )
    mode_sets = "\n".join(
        '\t\t\t<parameterset type="modeParams">\n'
        f'\t\t\t\t<param name="mode" value="{mode}"/>\n'
        f'\t\t\t\t<param name="marginalUtilityOfTraveling_util_hr" value="{utl}"/>\n'
        "\t\t\t</parameterset>"
        for mode, utl in _MODE_MARGINAL_UTILITY_HR
    )
    cfg = f"""<?xml version="1.0" ?>
<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">
<config>
\t<module name="global">
\t\t<param name="randomSeed" value="{int(scenario.seed)}"/>
\t\t<param name="coordinateSystem" value="Atlantis"/>
\t\t<param name="numberOfThreads" value="1"/>
\t</module>
\t<module name="network">
\t\t<param name="inputNetworkFile" value="network.xml"/>
\t</module>
\t<module name="plans">
\t\t<param name="inputPlansFile" value="{plans_name}"/>
\t</module>
\t<module name="controller">
\t\t<param name="outputDirectory" value="{out_name}"/>
\t\t<param name="firstIteration" value="{int(first_it)}"/>
\t\t<param name="lastIteration" value="{int(last_it)}"/>
\t\t<param name="overwriteFiles" value="deleteDirectoryIfExists"/>
\t\t<param name="writeEventsInterval" value="1"/>
\t\t<param name="writePlansInterval" value="1"/>
\t\t<param name="createGraphsInterval" value="0"/>
\t\t<param name="writeTripsInterval" value="0"/>
\t\t<param name="writeSnapshotsInterval" value="0"/>
\t\t<param name="routingAlgorithmType" value="{_ROUTING_ALGORITHM}"/>
\t</module>
\t<module name="qsim">
\t\t<param name="startTime" value="{start}"/>
\t\t<param name="endTime" value="{end}"/>
\t\t<param name="numberOfThreads" value="1"/>
\t\t<param name="removeStuckVehicles" value="false"/>
\t</module>
\t<module name="eventsManager">
\t\t<param name="numberOfThreads" value="1"/>
\t</module>
\t<module name="scoring">
\t\t<parameterset type="scoringParameters">
{activity_sets}
{mode_sets}
\t\t</parameterset>
\t</module>
\t<module name="replanning">
\t\t<param name="maxAgentPlanMemorySize" value="{_PLAN_MEMORY}"/>
{strategy_sets}
\t</module>
</config>
"""
    path = os.path.join(workdir, "config.xml")
    with open(path, "w") as fh:
        fh.write(cfg)
    return path


# --------------------------------------------------------------------------
# the pinned zero-replanning replay (produces X + the field; the G1 runner)
# --------------------------------------------------------------------------
def pinned_matsim_replay(
    scenario: EdocScenario,
    plans: dict[str, tuple[tuple[str, ...], float]],
    *,
    deadline: float | None,
    workdir: str | None = None,
) -> ReplayResult:
    """One pinned zero-replanning replay of ``plans`` (``firstIteration ==
    lastIteration``, both derived from :data:`_REPLAY_ITERATION` — the G1
    matched object). Asserts the installed engine + full JDK against the
    instance pins BEFORE running (the runner contract), writes network / plans
    / config itself, runs the JVM as the ONE R6 censor step, re-reads every
    artifact (rc never trusted), read-backs the echoed network, and parses
    events into a :class:`ReplayResult` whose ``canon_hash`` is the
    R10-canonicalized allowlist hash (the G1 determinism object). The hashed
    ``replay_deadline_s`` ALWAYS bounds this call (F3) — but a timeout is
    typed :class:`PlanReplayFailure` (censor) ONLY when the SCENARIO deadline
    was the binding one; a tighter caller wall is certifier-side budget and
    RAISES ``RuntimeError`` instead (the S3 review's F1)."""
    assert_engine_pin(installed_engine_version(), scenario.engine_version)
    _assert_jdk_pin()
    deadline, clipped_by_caller = _intersect_replay_deadline(scenario, deadline)

    first_it = last_it = _REPLAY_ITERATION
    if first_it != last_it:  # pragma: no cover - self-assertion (pair N6)
        raise RuntimeError("replay writer drifted: firstIteration != lastIteration")

    own_tmp = workdir is None
    workdir = workdir or tempfile.mkdtemp(
        prefix=f"tabench-edoc-matsim-{os.getpid()}-replay-"
    )
    try:
        _write_network(scenario, workdir)
        _write_plans(plans, scenario, workdir)
        cfg = _write_config(scenario, workdir, first_it=first_it, last_it=last_it)
        _run_java(
            ["org.matsim.core.controler.Controler", cfg],
            cwd=workdir, deadline=deadline, what="matsim zero-replanning replay",
            # an engine CRASH here is always the plan's fault (R6 censor); a
            # TIMEOUT censors only when the SCENARIO deadline was binding — a
            # caller-clipped wall is a certifier budget fault, infra RAISE (F1).
            censor_on_fail=True,
            censor_on_timeout=not clipped_by_caller,
        )
        out = os.path.join(workdir, "out")
        artifacts: dict[str, bytes] = {}
        for base in (
            "output_events.xml.gz",
            "output_plans.xml.gz",
            "output_network.xml.gz",
            "output_config.xml",
        ):
            p = os.path.join(out, base)
            if not os.path.exists(p):  # rc is never trusted
                raise RuntimeError(f"matsim replay reported success but wrote no {base}")
            with open(p, "rb") as fh:
                artifacts[base] = fh.read()
        _readback_network(os.path.join(out, "output_network.xml.gz"), scenario)
        agents, field_records, flows = parse_events(
            os.path.join(out, "output_events.xml.gz"),
            dt=float(scenario.dt),
            n_intervals=int(scenario.n_intervals),
            edge_ids=set(scenario.edge_ids),
            t0=_MATSIM_T0,
        )
        return ReplayResult(
            canon_hash=hash_matsim_artifacts(artifacts),
            agents=agents,
            field_records=field_records,
            flows=flows,
            n_intervals=int(scenario.n_intervals),
        )
    finally:
        if own_tmp:
            shutil.rmtree(workdir, ignore_errors=True)


def make_replay_runner(*, deadline: float | None) -> Callable[..., ReplayResult]:
    """A :data:`~tabench.edoc.replay.ReplayRunner` bound to the single wall
    deadline, for injection into
    :class:`~tabench.metrics.edoc_gaps.EdocEvaluator`. Each call rebuilds every
    input fresh from the hashed scenario, so the emit-time replay and both
    certifier replays are the identical map."""
    def runner(
        scenario: EdocScenario, plans: dict[str, tuple[tuple[str, ...], float]]
    ) -> ReplayResult:
        return pinned_matsim_replay(scenario, plans, deadline=deadline)

    return runner


# --------------------------------------------------------------------------
# the adapter: emit the EDOC artifact bundle
# --------------------------------------------------------------------------
class MatsimAdapter:
    """Emit the EDOC-1 artifact contract for one :class:`EdocScenario` by
    running the pinned MATSim co-evolution loop (plans ``P`` = the final
    iterate's selected plans under ``scenario.seed``) and then THIS adapter's
    own pinned zero-replanning replay of ``P`` (the experienced record ``X`` +
    provenance — the adr-037 clarification: ``X`` is DEFINED by the pinned
    replay map, never scraped from the solver's internals). The certifier's G1
    re-runs the identical replay. ``iterations`` is the co-evolution length
    (``lastIteration``); iteration 0 executes the free-flow-routed initial
    plans, so ``iterations=0`` is the AON control state."""

    name = "matsim"
    track = "edoc-stochastic"
    seedable = True

    def __init__(self, *, iterations: int = 10, keep_files: bool = False) -> None:
        self.iterations = int(iterations)
        self.keep_files = bool(keep_files)
        self.last_workdir: str | None = None

    def emit(self, scenario: EdocScenario, *, wall_seconds: float | None = None) -> EmittedBundle:
        """Write inputs -> run the co-evolution -> read ``P`` -> pinned replay
        for ``X``, all under one wall deadline. RAISES on any engine/infra
        failure (never a partial or self-reported result)."""
        if scenario.engine != ENGINE:
            raise ValueError(
                f"matsim only runs matsim instances; scenario {scenario.name!r} pins "
                f"engine {scenario.engine!r}"
            )
        assert_engine_pin(installed_engine_version(), scenario.engine_version)
        _assert_jdk_pin()
        deadline = time.perf_counter() + wall_seconds if wall_seconds is not None else None
        keep = self.keep_files
        # pid-scoped prefix (S3 review F5): hygiene snapshots diff only their
        # own process's dirs, so concurrent sessions cannot cross-flake.
        workdir = tempfile.mkdtemp(
            prefix=f"tabench-edoc-matsim-{os.getpid()}-keep-" if keep
            else f"tabench-edoc-matsim-{os.getpid()}-"
        )
        self.last_workdir = workdir
        try:
            _write_network(scenario, workdir)
            _write_plans(None, scenario, workdir)  # initial demand, engine-routed AON
            cfg = _write_config(
                scenario, workdir, first_it=0, last_it=self.iterations
            )
            _run_java(
                ["org.matsim.core.controler.Controler", cfg],
                cwd=workdir, deadline=deadline, what="matsim co-evolution run",
            )
            out = os.path.join(workdir, "out")
            plans_path = os.path.join(out, "output_plans.xml.gz")
            if not os.path.exists(plans_path):  # rc is never trusted
                raise RuntimeError("matsim run reported success but wrote no output_plans")
            _readback_network(os.path.join(out, "output_network.xml.gz"), scenario)
            plans = parse_output_plans(plans_path, t0=_MATSIM_T0)
            # X + the field from THIS adapter's own pinned replay of P.
            replay = pinned_matsim_replay(scenario, plans, deadline=deadline)
            return EmittedBundle(
                plans=plans,
                experienced=replay.agents,
                engine_version=scenario.engine_version,
                seed=int(scenario.seed),
            )
        finally:
            if not keep:
                shutil.rmtree(workdir, ignore_errors=True)
                self.last_workdir = None


# --------------------------------------------------------------------------
# the scenario family (adr-039; constants measured with the SHIPPED estimator)
# --------------------------------------------------------------------------
def build_matsim_diamond_scenario(
    name: str,
    *,
    seed: int = 42,
    seed_list: tuple[int, ...] = (42, 7, 123, 2024, 31337),
    n_agents: int = 100,
    depart_spacing: float = 2.0,
    dt: float = 20.0,
    n_intervals: int = 90,
    fftt_entry: float = 1.0,
    fftt_a: float = 90.0,
    fftt_b: float = 100.0,
    fftt_exit: float = 10.0,
    bottleneck_lanes: int = 1,
    approach_lanes: int = 3,
    side_lanes: int = 2,
    terminal_lanes: int = 4,
    separation_factor: float = 5.0,
    floor_seconds: float = 15.0,
    backlog_bound: float = 60.0,
    replay_deadline_s: float = 60.0,
    r3_tolerance_s: float = 15.0,
    walk_bound: int = 4,
    engine_version: str | None = None,
) -> EdocScenario:
    """The matsim row's DIAMOND family: ``O0 ->home-> O``, then route A
    ``O ->a1-> N1 ->a2-> D`` vs route B ``O ->b1-> N2 ->b2-> D``, then
    ``D ->work-> D2``. The route-distinguishing capacity drop is ``a2``
    (``bottleneck_lanes`` — in MATSim the outflow queue sits ON the drop edge
    itself, so the cost signal lands exactly on the route-distinguishing edge);
    route A is uniquely free-flow-shorter (``fftt_a < fftt_b``) so the AON
    control piles onto the bottleneck. ``home``/``work`` are the single
    departure/arrival plumbing edges the family contract requires
    (:func:`_terminal_links`); both routes share them, so their cost cancels in
    the gap numerator. Exactly 5 pinned macrorep seeds (R5 floor == R7 bound).
    ``engine_version=None`` reads the installed pin (engine-gated); tests pass
    an explicit string to construct engine-free."""
    depart = np.array([i * depart_spacing for i in range(n_agents)], dtype=np.float64)
    return EdocScenario(
        name=name,
        edge_ids=("home", "a1", "a2", "b1", "b2", "work"),
        edge_tail=("O0", "O", "N1", "O", "N2", "D"),
        edge_head=("O", "N1", "D", "N2", "D", "D2"),
        edge_fftt=np.array([fftt_entry, fftt_a, fftt_a, fftt_b, fftt_b, fftt_exit]),
        edge_lanes=np.array(
            [terminal_lanes, approach_lanes, bottleneck_lanes,
             side_lanes, side_lanes, terminal_lanes]
        ),
        agent_ids=tuple(f"v{i}" for i in range(n_agents)),
        agent_origin=("O0",) * n_agents,
        agent_dest=("D2",) * n_agents,
        agent_depart=depart,
        engine=ENGINE,
        engine_version=(
            engine_version if engine_version is not None else installed_engine_version()
        ),
        seed=int(seed),
        semantic_config=_semantic_config(),
        dt=dt,
        n_intervals=n_intervals,
        departure_quantum=1.0,
        backlog_bound=backlog_bound,
        separation_factor=separation_factor,
        floor_seconds=floor_seconds,
        replay_deadline_s=replay_deadline_s,
        r3_tolerance_s=r3_tolerance_s,
        walk_bound=walk_bound,
        seed_list=seed_list,
        family="matsim-diamond",
    )


def matsim_shared_bottleneck_scenario(
    *, seed: int = 42, n_agents: int = 100, engine_version: str | None = None
) -> EdocScenario:
    """A DELIBERATELY non-separating topology for the refusal demonstration:
    the 1-lane drop sits on the SHARED edge ``sh`` (O -> J) that BOTH routes
    traverse before the split, so the downstream choice cannot relieve the
    bottleneck — control and converged states congest ``sh`` identically, the
    mean-vs-mean separation gate fails, and
    :func:`negative_control_separation` REFUSES the family (a config error,
    never a certified row)."""
    depart = np.array([i * 2.0 for i in range(n_agents)], dtype=np.float64)
    return EdocScenario(
        name="matsim-shared-bottleneck",
        edge_ids=("home", "sh", "p1", "p2", "q1", "q2", "work"),
        edge_tail=("O0", "O", "J", "M1", "J", "M2", "D"),
        edge_head=("O", "J", "M1", "D", "M2", "D", "D2"),
        edge_fftt=np.array([1.0, 90.0, 50.0, 50.0, 52.0, 52.0, 10.0]),
        edge_lanes=np.array([4, 1, 2, 2, 2, 2, 4]),  # the drop is on the SHARED sh
        agent_ids=tuple(f"v{i}" for i in range(n_agents)),
        agent_origin=("O0",) * n_agents,
        agent_dest=("D2",) * n_agents,
        agent_depart=depart,
        engine=ENGINE,
        engine_version=(
            engine_version if engine_version is not None else installed_engine_version()
        ),
        seed=int(seed),
        semantic_config=_semantic_config(),
        dt=20.0,
        n_intervals=90,
        departure_quantum=1.0,
        backlog_bound=60.0,
        separation_factor=5.0,
        floor_seconds=15.0,
        replay_deadline_s=60.0,
        r3_tolerance_s=15.0,
        walk_bound=5,
        seed_list=(42, 7, 123, 2024, 31337),
        family="matsim-shared-bottleneck",
    )


def matsim_reference_scenario() -> EdocScenario:
    """The pinned matsim reference instance (adr-039). Family constants
    MEASURED with the SHIPPED estimator on this toolchain (Temurin-21.0.11+10 +
    matsim-2025.0, 2026-07-17) — the R4 re-derivation recorded in adr-039."""
    return build_matsim_diamond_scenario("matsim-ref")


# --------------------------------------------------------------------------
# row certification (vetting + substrate certificate + self-cross-check)
# --------------------------------------------------------------------------
def _field_selfcheck(
    scenario: EdocScenario,
    plans: dict[str, tuple[tuple[str, ...], float]],
    replay: ReplayResult,
    *,
    tolerance_s: float,
) -> dict[str, float]:
    """The R3 harness self-cross-check (adr-039 ruling: MATSim has no
    standalone pinned router artifact — QSim routing is in-process — so the
    substrate TD-SP is normative-only and the mandatory cross-check clause has
    no engine router to bind to; disclosed in adr-039). Instead the row
    re-derives every driven cost by an INDEPENDENTLY written field composition
    and RAISES if :func:`~tabench.edoc.tdsp.evaluate_route` disagrees beyond
    ``r3_tolerance_s`` — a field-arithmetic regression guard, infra RAISE,
    never a censor."""
    field = build_field_from_records(
        replay.field_records, scenario.fftt_of(), scenario.dt, scenario.n_intervals,
        scenario.field_semantics,
    )
    ow = build_origin_waits(
        [(a.first_edge, a.departure, a.depart_delay) for a in replay.agents.values()],
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
        direct = tau - float(dep)
        diffs.append(abs(mine - direct))
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
    """The ROW's per-macrorep certification path: the G0-G4 + ``RG_D1``
    substrate certificate (:class:`EdocEvaluator`), the R3 harness
    self-cross-check (on a feasible run), and the separation-vetting assertion
    (F10) — all under one wall deadline. The row's SCORE lives one level up in
    :func:`certify_row` (R5: a single-seed readout is never the score)."""
    from ...metrics.edoc_gaps import EdocEvaluator

    if _topology_digest(scenario) not in _SEPARATION_VETTED_TOPOLOGIES:
        raise RuntimeError(
            f"certify_emitted refuses {scenario.name!r} (family {scenario.family!r}): "
            "its TOPOLOGY has not been separation-vetted — run "
            "negative_control_separation on this topology first (adr-036 R-control / "
            "F10; vetting is keyed on the topology digest, not the family string — "
            "adr-039 F3)"
        )
    deadline = time.perf_counter() + wall_seconds if wall_seconds is not None else None
    runner = make_replay_runner(deadline=deadline)
    metrics = EdocEvaluator(scenario, runner).certify(emitted)
    if metrics.get("feasible") == 1.0:
        try:
            replay = runner(scenario, emitted.plans)
        except PlanReplayFailure as exc:
            # F4 (S3 review): G1 just replayed these SAME plans twice, so a
            # failure of this third (R3 self-check) replay is a certifier-side
            # fault (budget/engine), not an invalid emission — re-typed to the
            # infra RAISE, never allowed to escape as the censor-signal type.
            # (Deliberate deferral, adr-039: reusing G1's replay here would
            # save one engine leg per feasible seed but needs an EdocEvaluator
            # interface change — not this batch.)
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


def certify_row(
    base: EdocScenario,
    *,
    iterations: int = 10,
    wall_seconds: float | None = None,
    b: int = 10000,
    level: float = 0.95,
) -> MacrorepResult:
    """The row's ONLY score entry point (R5): P8 macroreps over the pinned
    ``seed_list`` — each pinned seed gets its own emit (co-evolution under that
    seed) + :func:`certify_emitted`; the row score is the mean ``RG_D1`` with
    the house bootstrap CI (:mod:`tabench.edoc.macrorep`). Any censored
    macrorep censors the whole row; infra failures RAISE (R6)."""
    deadline = time.perf_counter() + wall_seconds if wall_seconds is not None else None

    def _emit(sc_s: EdocScenario) -> EmittedBundle:
        return MatsimAdapter(iterations=iterations).emit(
            sc_s, wall_seconds=_remaining(deadline)
        )

    def _certify(sc_s: EdocScenario, emitted: EmittedBundle) -> dict[str, float]:
        return certify_emitted(sc_s, emitted, wall_seconds=_remaining(deadline))

    return certify_macroreps(base, _emit, _certify, b=b, level=level)


def negative_control_separation(
    scenario: EdocScenario,
    *,
    wall_seconds: float | None,
    control_iterations: int = 0,
    converged_iterations: int = 10,
) -> dict[str, float]:
    """The negative-control separation gate on the STOCHASTIC track (adr-036
    R4; adr-039 ruling: mean-vs-mean over the SAME pinned seed list — a
    single-seed anchor would violate R5's single-readout ban). The control
    (``iterations=0``: the engine-routed free-flow AON state) and the converged
    state are each certified via :func:`certify_macroreps` over
    ``scenario.seed_list``, and the separation is computed on the
    **floor-DISPLAYED** values (``max(rg_mean, floor_gap_row)`` each side —
    adr-036's own leaderboard rule that a sub-floor value is displayed AT the
    floor): the gate requires ``displayed(control) / displayed(converged) >=
    scenario.separation_factor`` else RAISES ``ValueError``. The displayed
    basis is what makes the gate non-vacuous on the self-certifying shared-edge
    topology (measured, adr-039: the shared drop scores ``RG_D1 = 0`` exactly
    on BOTH anchors, so a raw ratio degenerates to ``0/0 -> inf`` and would
    vacuously pass — displayed values separate 1.0x and refuse it, exactly
    adr-036 forgery pair 12). A censored anchor also RAISES. On success the
    TOPOLOGY is marked separation-vetted (F10, digest-keyed — adr-039 F3:
    a relabeled family string cannot borrow the vetting), which :func:`certify_emitted`
    asserts before certifying."""
    from ...metrics.edoc_gaps import EdocEvaluator

    deadline = time.perf_counter() + wall_seconds if wall_seconds is not None else None

    def _anchor(iterations: int) -> MacrorepResult:
        def _emit(sc_s: EdocScenario) -> EmittedBundle:
            return MatsimAdapter(iterations=iterations).emit(
                sc_s, wall_seconds=_remaining(deadline)
            )

        def _certify(sc_s: EdocScenario, emitted: EmittedBundle) -> dict[str, float]:
            runner = make_replay_runner(deadline=deadline)
            return EdocEvaluator(sc_s, runner).certify(emitted)

        return certify_macroreps(scenario, _emit, _certify)

    control = _anchor(control_iterations)
    converged = _anchor(converged_iterations)
    if control.metrics["feasible"] != 1.0 or converged.metrics["feasible"] != 1.0:
        raise ValueError(
            f"negative-control separation: an anchor censored (control feasible "
            f"{control.metrics['feasible']}, converged feasible "
            f"{converged.metrics['feasible']}) — the separation anchors must be real "
            "feasible measurements over the full pinned seed list"
        )
    a_rg = float(control.metrics["rg_d1_mean"])
    c_rg = float(converged.metrics["rg_d1_mean"])
    # displayed values: a sub-floor mean reads AT the row floor (adr-036).
    a_disp = max(a_rg, float(control.metrics["floor_gap"]))
    c_disp = max(c_rg, float(converged.metrics["floor_gap"]))
    separation = a_disp / c_disp if c_disp > 0 else float("inf")
    if separation < scenario.separation_factor:
        raise ValueError(
            f"negative-control separation FAILED for {scenario.name!r}: displayed AON "
            f"control RG_D1 {a_disp:.5f} (raw {a_rg:.5f}) / displayed converged "
            f"{c_disp:.5f} (raw {c_rg:.5f}) = {separation:.2f}x < the declared "
            f"{scenario.separation_factor}x (a non-separating topology — e.g. a "
            "shared-edge bottleneck, which self-certifies RG_D1 = 0 on both anchors — "
            "is a construction error, never a certified row)"
        )
    # F10: mark the TOPOLOGY vetted (digest-keyed, adr-039 F3).
    _SEPARATION_VETTED_TOPOLOGIES.add(_topology_digest(scenario))
    return {
        "control_rg_d1": a_rg,
        "converged_rg_d1": c_rg,
        "control_displayed": a_disp,
        "converged_displayed": c_disp,
        "separation": separation,
        "separation_factor": float(scenario.separation_factor),
    }
