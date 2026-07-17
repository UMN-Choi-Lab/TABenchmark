"""MATSim toolchain addressing + artifact parsing for the ``matsim`` EDOC row.

Design: docs/design/adr-036-external-dynamic-observational-certificate.md
(the MATSim unblock record) + docs/design/adr-039-matsim.md (the row).

Unlike the SUMO/DTALite wheels, MATSim is a **Java-only** engine with no PyPI
artifact (the PyPI ``matsim`` package is an unrelated neuronal simulator,
adr-030), so this module imports NO optional python package and is importable
everywhere; engine availability is a RUNTIME probe, never an import guard.

**Addressing (the F8 anti-stale-toolchain rule).** The engine is located ONLY
through ``TABENCH_MATSIM_HOME`` (a directory containing ``matsim-2025.0.jar``
directly or as the unzipped ``matsim-2025.0/`` release layout — the release
layout must be preserved: the jar's manifest ``Class-Path`` references 154 jars
in the adjacent ``libs/`` dir) and java ONLY through an ABSOLUTE
``$TABENCH_JAVA_HOME/bin/java`` (falling back to ``JAVA_HOME``) — never a bare
``java`` from ``PATH``, so a stale system JDK/MATSim can never be silently
picked up. :func:`matsim_available` is **side-effect free** (the ``find_spec``
discipline: pure ``os.path`` checks, no JVM is ever started to probe).

**Event semantics parsed here (measured on 2025.0, adr-039):** per agent the
scheduled departure is the ``departure`` event; the insertion wait
``depart_delay`` is ``vehicle enters traffic`` minus ``departure`` (the MATSim
departDelay analogue — off-network waiting invisible to link records, forgery
pair 1); ``arrival`` is the ``arrival`` event; ``experienced_time = arrival -
departure`` (door-to-door, wait INCLUDED — G3). The driven route is the
departure link + the ``entered link`` sequence. Link traversal samples pair
``entered link``→``left link`` (departure link: ``vehicle enters traffic``→
``left link``; arrival link: ``entered link``→``vehicle leaves traffic``), so
the on-network experienced time decomposes EXACTLY into the per-link samples.
Flows count ``entered link`` + ``vehicle enters traffic`` as inflow and ``left
link`` + ``vehicle leaves traffic`` as outflow (the entered-link-only rule of
adr-036 undercounts the departure link — the measured correction recorded in
adr-039); ``output_links.csv`` is NEVER used for flows (it undercounts the
arrival link to zero, measured). Occupancy witnesses per interval come from the
on-link spans (cumulative entered-minus-left), exact for MATSim.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import re
import subprocess
import xml.etree.ElementTree as ET

from ...edoc.replay import ReplayAgent

__all__ = [
    "MATSIM_RELEASE",
    "installed_engine_version",
    "java_binary",
    "java_version_string",
    "matsim_available",
    "matsim_jar",
    "parse_events",
    "parse_output_plans",
]

MATSIM_RELEASE = "matsim-2025.0"
_JAR_BASENAME = f"{MATSIM_RELEASE}.jar"


# --------------------------------------------------------------------------
# toolchain addressing (env-var only; side-effect-free probes)
# --------------------------------------------------------------------------
def matsim_jar() -> str | None:
    """Absolute path to the pinned MATSim jar, addressed ONLY through
    ``TABENCH_MATSIM_HOME`` (the dir holding the jar, or its parent holding the
    ``matsim-2025.0/`` release layout). ``None`` when unset/absent."""
    home = os.environ.get("TABENCH_MATSIM_HOME")
    if not home:
        return None
    home = os.path.expanduser(home)
    for cand in (
        os.path.join(home, _JAR_BASENAME),
        os.path.join(home, MATSIM_RELEASE, _JAR_BASENAME),
    ):
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    return None


def java_binary() -> str | None:
    """Absolute ``bin/java`` from ``TABENCH_JAVA_HOME`` (fallback ``JAVA_HOME``);
    NEVER a bare ``java`` from PATH (F8). ``None`` when unset/absent."""
    home = os.environ.get("TABENCH_JAVA_HOME") or os.environ.get("JAVA_HOME")
    if not home:
        return None
    cand = os.path.join(os.path.expanduser(home), "bin", "java")
    if os.path.isfile(cand) and os.access(cand, os.X_OK):
        return os.path.abspath(cand)
    return None


def matsim_available() -> bool:
    """Runtime availability probe: jar + java both addressed. Side-effect free
    (no subprocess, no JVM — the ``find_spec`` discipline for a Java engine)."""
    return matsim_jar() is not None and java_binary() is not None


def java_version_string(java: str | None = None) -> str:
    """The full ``java -version`` banner (stderr text) of the ADDRESSED java —
    the G0 read for the full-JDK pin. One short subprocess; RAISES
    ``RuntimeError`` when java is unaddressed or the probe fails."""
    java = java or java_binary()
    if java is None:
        raise RuntimeError(
            "matsim toolchain unaddressed: set TABENCH_JAVA_HOME (or JAVA_HOME) to the "
            "pinned JDK root (Temurin 21.0.11+10, adr-039)"
        )
    try:
        proc = subprocess.run(
            [java, "-version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"java -version probe failed for {java!r} ({exc})") from exc
    return (proc.stderr or "") + (proc.stdout or "")


_JAVA_MAJOR = re.compile(r'version "(\d+)')


def installed_engine_version() -> str:
    """The engine identity string actually installed on this box (the G0 read):
    ``matsim-2025.0;jar-md5=<md5 of the addressed jar>;jdk-major=<N>``. The jar
    md5 is computed from the artifact on disk at certify time (engine-version
    drift, forgery pair 6); the FULL JDK build is enforced separately as a
    family-declared hashed constant with a G0 RAISE (adr-039). RAISES
    ``RuntimeError`` when the toolchain is unaddressed."""
    jar = matsim_jar()
    if jar is None:
        raise RuntimeError(
            "matsim engine unaddressed: set TABENCH_MATSIM_HOME to the directory "
            f"containing {_JAR_BASENAME} (or the {MATSIM_RELEASE}/ release layout)"
        )
    h = hashlib.md5()
    with open(jar, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    m = _JAVA_MAJOR.search(java_version_string())
    if m is None:
        raise RuntimeError("could not parse the JDK major from `java -version`")
    return f"{MATSIM_RELEASE};jar-md5={h.hexdigest()};jdk-major={m.group(1)}"


# --------------------------------------------------------------------------
# artifact parsers (F9a: every parse failure -> contract RuntimeError)
# --------------------------------------------------------------------------
def _open_maybe_gz(path: str) -> bytes:
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B:
        data = gzip.decompress(data)
    return data


def _hms_to_seconds(text: str) -> float:
    """MATSim time attribute: ``HH:MM:SS`` (hours may exceed 24) or a bare
    float-seconds string."""
    parts = text.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600.0 + float(m) * 60.0 + float(s)
    return float(text)


def parse_events(
    path: str,
    *,
    dt: float,
    n_intervals: int,
    edge_ids: frozenset[str] | set[str],
    t0: float = 0.0,
) -> tuple[
    dict[str, ReplayAgent],
    dict[str, dict[int, tuple[float, float]]],
    dict[str, dict[int, tuple[float, float]]],
]:
    """Parse a MATSim ``output_events.xml.gz`` into the certifier's model-blind
    material: ``(agents, field_records, flows)`` (see the module docstring for
    the measured event semantics). ``t0`` is the writer's fixed engine-clock
    offset (subtracted from every event time, so parsed times live on the
    scenario's ``[0, dt*n_intervals)`` axis). Events on non-scenario links
    (engine-side connectivity plumbing) are ignored. An agent with no
    ``arrival`` event (stuck at qsim end) is simply absent from ``agents`` —
    the G3 completion census censors it; nothing is invented here."""
    per_agent_dep: dict[str, float] = {}
    per_agent_dep_link: dict[str, str] = {}
    per_agent_enter: dict[str, float] = {}
    per_agent_arr: dict[str, float] = {}
    per_agent_route: dict[str, list[str]] = {}
    # current on-link span per vehicle: (link, entry_time)
    on_link: dict[str, tuple[str, float]] = {}
    spans: dict[str, list[tuple[float, float]]] = {}  # link -> [(t_in, t_out)]
    inflow: dict[str, dict[int, float]] = {}
    outflow: dict[str, dict[int, float]] = {}

    def _k(t: float) -> int:
        if t <= 0.0:
            return 0
        k = int(t // dt)
        return n_intervals - 1 if k >= n_intervals else k

    def _count(table: dict[str, dict[int, float]], link: str, t: float) -> None:
        per = table.setdefault(link, {})
        k = _k(t)
        per[k] = per.get(k, 0.0) + 1.0

    try:
        with gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb") as fh:
            for _, el in ET.iterparse(fh, events=("end",)):
                if el.tag != "event":
                    continue
                a = el.attrib
                etype = a.get("type")
                t = float(a["time"]) - t0
                if etype == "departure" and a.get("legMode") == "car":
                    person = a["person"]
                    link = a["link"]
                    per_agent_dep[person] = t
                    per_agent_dep_link[person] = link
                    per_agent_route[person] = [link]
                elif etype == "vehicle enters traffic":
                    veh = a["vehicle"]
                    link = a["link"]
                    per_agent_enter.setdefault(a["person"], t)
                    on_link[veh] = (link, t)
                    if link in edge_ids:
                        _count(inflow, link, t)
                elif etype == "entered link":
                    veh = a["vehicle"]
                    link = a["link"]
                    per_agent_route.setdefault(veh, []).append(link)
                    on_link[veh] = (link, t)
                    if link in edge_ids:
                        _count(inflow, link, t)
                elif etype == "left link":
                    veh = a["vehicle"]
                    link = a["link"]
                    span = on_link.pop(veh, None)
                    if span is not None and span[0] == link and link in edge_ids:
                        spans.setdefault(link, []).append((span[1], t))
                    if link in edge_ids:
                        _count(outflow, link, t)
                elif etype == "vehicle leaves traffic":
                    veh = a["vehicle"]
                    link = a["link"]
                    span = on_link.pop(veh, None)
                    if span is not None and span[0] == link and link in edge_ids:
                        spans.setdefault(link, []).append((span[1], t))
                    if link in edge_ids:
                        _count(outflow, link, t)
                elif etype == "arrival" and a.get("legMode") == "car":
                    per_agent_arr[a["person"]] = t
                el.clear()
    except (ET.ParseError, gzip.BadGzipFile, OSError, KeyError, ValueError) as exc:
        raise RuntimeError(f"matsim events at {path} unparseable ({exc})") from exc

    agents: dict[str, ReplayAgent] = {}
    for person, dep in per_agent_dep.items():
        arr = per_agent_arr.get(person)
        if arr is None:
            continue  # never arrived: censored by the G3 census, not invented here
        enter = per_agent_enter.get(person, dep)
        agents[person] = ReplayAgent(
            agent_id=person,
            departure=dep,
            arrival=arr,
            route=tuple(per_agent_route.get(person, ())),
            experienced_time=arr - dep,
            depart_delay=enter - dep,
        )

    # interval-mean traversal per link (samples keyed by ENTRY interval) and an
    # occupancy witness = number of vehicles present on the link during the
    # interval (exact from the spans; the burst-poisoning detection, R2).
    field_records: dict[str, dict[int, tuple[float, float]]] = {}
    occupancy: dict[str, dict[int, float]] = {}
    for link, link_spans in spans.items():
        for t_in, t_out in link_spans:
            k_lo = _k(t_in)
            k_hi = _k(max(t_in, t_out - 1e-9))
            for k in range(k_lo, k_hi + 1):
                per = occupancy.setdefault(link, {})
                per[k] = per.get(k, 0.0) + 1.0
    for link, link_spans in spans.items():
        sums: dict[int, list[float]] = {}
        for t_in, t_out in link_spans:
            sums.setdefault(_k(t_in), []).append(t_out - t_in)
        per_link = field_records.setdefault(link, {})
        for k, vals in sums.items():
            per_link[k] = (sum(vals) / len(vals), occupancy.get(link, {}).get(k, 0.0))

    flows: dict[str, dict[int, tuple[float, float]]] = {}
    for link in set(inflow) | set(outflow):
        per_in = inflow.get(link, {})
        per_out = outflow.get(link, {})
        flows[link] = {
            k: (per_in.get(k, 0.0), per_out.get(k, 0.0))
            for k in set(per_in) | set(per_out)
        }
    return agents, field_records, flows


def parse_output_plans(path: str, *, t0: float = 0.0) -> dict[str, tuple[tuple[str, ...], float]]:
    """Parse a MATSim ``output_plans.xml.gz`` into emitted plans ``{aid:
    (route_edges, depart)}`` from each person's SELECTED plan: the route is the
    ``<route type="links">`` link sequence of the (single) car leg, the
    departure is the first activity's ``end_time`` minus the writer clock
    offset ``t0``."""
    plans: dict[str, tuple[tuple[str, ...], float]] = {}
    try:
        root = ET.fromstring(_open_maybe_gz(path))
        for person in root.findall("person"):
            aid = person.get("id")
            for plan in person.findall("plan"):
                if plan.get("selected") != "yes":
                    continue
                acts = plan.findall("activity")
                legs = plan.findall("leg")
                if not acts or not legs:
                    raise ValueError(f"person {aid!r}: selected plan has no act/leg")
                end_time = acts[0].get("end_time")
                if end_time is None:
                    raise ValueError(f"person {aid!r}: first activity has no end_time")
                route_el = legs[0].find("route")
                if route_el is None or route_el.get("type") != "links":
                    raise ValueError(f"person {aid!r}: selected plan has no links route")
                edges = tuple((route_el.text or "").split())
                plans[aid] = (edges, _hms_to_seconds(end_time) - t0)
                break
    except (ET.ParseError, gzip.BadGzipFile, OSError, ValueError) as exc:
        raise RuntimeError(f"matsim output_plans at {path} unparseable ({exc})") from exc
    return plans
