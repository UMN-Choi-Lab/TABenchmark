"""Scenario <-> SUMO conversion for the ``sumo-marouter`` adapter (P4, P9).

Eclipse SUMO's macroscopic assignment tool ``marouter`` does NOT take a
user-supplied BPR volume-delay function. Its congestion law is **hardcoded per
road class** (``src/marouter/ROMAAssignments.cpp``: ``getCapacity`` +
``capacityConstraintFunction``, "based on the definitions in PTV-Validate and in
the VISUM-Cologne network"); with ``--capacities.default`` every edge uses one
default class whose latency is **linear in flow**

    t(f) = t0 * (1 + K * f / C),   t0 = length / speed  (free-flow time, seconds)

with ``(K, C)`` keyed on the declared speed band and lane count (measured on the
installed 1.27.1 binary, agreeing with the source to < 1e-9):

    speed > 26 m/s : C = 1400 * numLanes,  K = 2   (the "normal" band)
    speed <= 5 m/s : C =  200 * numLanes,  K = 6   (the "zero-intercept" band)

A repo link with a **linear** latency ``t(v) = A + B * v`` (only ``power == 1``
is representable; ``A = fft``, ``B = fft * b / cap``) therefore compiles to a
SUMO edge (speed band, numLanes, length) solving, at a chosen time scale ``tau``
(seconds per native cost unit) and flow scale ``s`` (veh/h per native flow unit):

    t0    = A * tau                          -> length = A * tau * speed
    slope = t0 * K / C = tau * B / s         -> numLanes = A * K * s / (B * cap_per_lane)

**THREE representability bounds** are unavoidable (slope and intercept are coupled
through ``t0``; lane counts and edge lengths are physically bounded):

* **zero-intercept links** (``A ~ 0``, ``B > 0``): the true intercept is 0 but
  every edge has ``t0 = tau * B * (C/K) / s > 0``, minimized by the ``speed <= 5``
  band (``C/K = 200*lanes/6``). The forced abstract intercept ``eps = B*(200*lanes/6)/s``
  must stay <= ``_INTERCEPT_FLOOR_TOL`` AND its edge length ``5*tau*eps`` must
  clear netconvert's silent 0.1 m minimum-length clamp; lanes are chosen to
  satisfy both, refusing when they cannot.
* **zero-slope links** (``B ~ 0``, ``A > 0``): every edge has ``slope > 0``; the
  parasitic abstract slope ``A*K*s/C`` is minimized with many lanes, bounded below
  ``_PARASITIC_SLOPE_TOL`` -- refused (not silently capped) when ``_MAX_LANES``
  cannot reach the tolerance.
* **lane quantization**: ``numLanes = round(A*K*s/(B*cap_per_lane))`` must be an
  exact positive integer <= ``_MAX_LANES``; the flow scale is rationalized with a
  bounded denominator and refused when it cannot reproduce the coefficient or
  when any link's lane count blows past the cap (generic float coefficients).

Two silent-failure hazards of the engine are neutralized: (1) netconvert's 0.1 m
minimum-length clamp is caught by a **compile read-back** that reparses
``net.net.xml`` and verifies every lane's ``length``/``speed``/``numLanes``
matches the declared values (mismatch -> RuntimeError, crash discipline); (2)
marouter reverts an edge to free-flow when a path's cumulative time exceeds the OD
time window, so the ``$OR;D2`` window is **sized from a worst-path bound** under
the mapped law and the netload is required to carry exactly one interval.

The certified metrics are recomputed repo-side from the returned link flows in
NATIVE units (``v = entered / s``); ``tau`` and ``s`` divide out of MASS exactly
(link flows are native), so they can only affect marouter's own logit dispersion
and the compiled edge geometry, never the mass conservation. See adr-027.

This module imports ``sumo`` (the ``eclipse-sumo`` wheel); the guarded import in
``models/__init__.py`` keeps the numpy/scipy core dependency-free.
"""

from __future__ import annotations

import math
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import sumo

from ...core.scenario import Network, Scenario

__all__ = [
    "SumoBuild",
    "scenario_to_sumo",
    "parse_netload",
    "parse_netload_attr",
    "netload_interval_count",
    "netload_matched_edge_count",
    "sumo_binary",
    "sumo_env",
    "REFUSED_TASK_FIELDS",
]

# --- marouter's hardcoded default-class vdf, measured on SUMO 1.27.1 -----------
_NORMAL_SPEED = 30.0  # m/s, the >26 band
_NORMAL_CAP_PER_LANE = 1400.0
_NORMAL_K = 2.0
_ZI_SPEED = 5.0  # m/s, the <=5 band (minimal C/K -> minimal forced intercept)
_ZI_CAP_PER_LANE = 200.0
_ZI_K = 6.0

# --- link-classification and representability thresholds ----------------------
# A repo link is "zero-intercept" when its free-flow time sits at/below the
# sentinel the builtin linear scenarios use for a true-zero intercept (fft=1e-6);
# "zero-slope" when its linear slope B = fft*b/cap is (near) zero.
_ZERO_INTERCEPT_A = 1e-4  # native cost units
_ZERO_SLOPE_B = 1e-12  # native cost per native flow unit
# Representability floors (abstract native units), documented in adr-027. The
# intercept floor bounds the forced zero-intercept perturbation (0.024 reproduces
# the Braess mapping floor ~1.7e-4 measured in the pilot); the parasitic floor
# bounds the residual slope on zero-slope links.
_INTERCEPT_FLOOR_TOL = 0.024
_PARASITIC_SLOPE_TOL = 0.004
_MAX_LANES = 4000  # hard cap on quantized lane counts (refuse past it)
# netconvert silently clamps edge lengths below this to it (SUMO default); the
# eps-edge lanes are chosen so the compiled length clears it, and the read-back
# below is the backstop. The small factor keeps a margin above the clamp.
_MIN_EDGE_LENGTH_M = 0.1
_LENGTH_TARGET_FACTOR = 1.05
# Compile read-back tolerances (netconvert writes --precision 6 lengths).
_READBACK_LENGTH_RTOL = 1e-3
_READBACK_SPEED_RTOL = 1e-4
# A representable link's lane coefficient must be an EXACT rational to this
# relative accuracy, else the flow scale cannot quantize it (refuse).
_RATIONAL_RTOL = 1e-9
_RATIONAL_MAX_DEN = 10**6  # bounded denominator: 1e12 chased binary-double noise
_OD_INTERVAL_SEC = 3600.0  # marouter's default weight-interval length (1 h)

#: Scenario task fields that make the instance non-representable in marouter's
#: hardcoded linear vdf: refused loudly (ValueError naming the field), never run
#: with a silently wrong cost model (adr-027).
REFUSED_TASK_FIELDS = (
    "sue_theta",
    "elastic_demand",
    "combined_demand",
    "br_epsilon",
    "side_capacities",
    "link_interaction",
    "multiclass",
)


@dataclass(frozen=True)
class _EdgeSpec:
    """One compiled SUMO edge: what the read-back must find in net.net.xml."""

    kind: int  # 0 normal, 1 zero-intercept, 2 zero-slope
    lanes: int
    length: float
    speed: float


@dataclass(frozen=True)
class SumoBuild:
    """The compiled artifacts + the flow scale needed to read flows back."""

    net_file: str
    taz_file: str
    od_file: str
    flow_scale: float  # s: veh/h per native flow unit (divides out of MASS)
    time_scale: float  # tau: seconds per native cost unit
    forced_intercept: float  # max abstract eps on zero-intercept links (0 if none)
    parasitic_slope: float  # max abstract residual slope on zero-slope links (0 if none)
    representable: np.ndarray  # bool (n_links,): links matched to marouter's vdf exactly
    od_window_hours: int  # $OR;D2 To-Time (sized from a worst-path bound)


def sumo_binary(name: str) -> str:
    """Absolute path to a SUMO tool inside the installed wheel.

    Binaries are addressed ONLY through ``sumo.SUMO_HOME`` (the wheel's own
    location), never PATH / ``shutil.which`` / the ambient ``SUMO_HOME`` env var
    -- this box ships a stale ``SUMO_HOME=/opt/sumo-1.12`` beside the 1.27.1
    binaries, exactly the discovery hazard the design names (adr-027)."""
    return os.path.join(sumo.SUMO_HOME, "bin", name)


def sumo_env() -> dict[str, str]:
    """Subprocess env with ``SUMO_HOME`` overridden to the wheel's own home AND the
    per-tool ``*_BINARY`` overrides pinned to the wheel's absolute binaries.

    ``sumolib.checkBinary`` (which ``duaIterate.py`` uses to locate ``sumo`` /
    ``duarouter`` / ``netconvert``) consults ``SUMO_BINARY`` / ``DUAROUTER_BINARY``
    / ``NETCONVERT_BINARY`` BEFORE ``SUMO_HOME`` / PATH, so an ambient poisoned
    ``SUMO_BINARY=/opt/sumo-1.12/bin/sumo`` would silently bypass the wheel-only
    rule inside the driver. Pinning those keys to the wheel closes that hole (F8);
    the adapter's own direct spawns already use absolute wheel paths."""
    env = {**os.environ, "SUMO_HOME": sumo.SUMO_HOME}
    for tool, var in (
        ("sumo", "SUMO_BINARY"),
        ("duarouter", "DUAROUTER_BINARY"),
        ("netconvert", "NETCONVERT_BINARY"),
    ):
        env[var] = os.path.join(sumo.SUMO_HOME, "bin", tool)
    return env


def _linear_params(network: Network) -> tuple[np.ndarray, np.ndarray]:
    """Per-link linear latency ``t(v) = A + B v`` (native units) for power==1.

    ``A = fft`` and ``B = fft * b / cap`` are exact for ``power == 1``:
    ``t = fft (1 + b (v/cap)^1) = fft + (fft b / cap) v``. Toll/distance fixed
    costs are NOT folded in here -- the model refuses tolled scenarios (adr-027),
    so ``A`` is the pure free-flow time."""
    a = np.asarray(network.free_flow_time, dtype=np.float64)
    b_slope = a * np.asarray(network.b, dtype=np.float64) / np.asarray(
        network.capacity, dtype=np.float64
    )
    return a, b_slope


def _lane_coeff(a: float, b: float) -> tuple[Fraction, float]:
    """Exact rational ``numLanes / s`` for a normal link (``A K / (B cap)``) and
    the same as a float, so the caller can verify the rationalization is exact."""
    coeff_float = (a * _NORMAL_K) / (b * _NORMAL_CAP_PER_LANE)
    frac = Fraction(coeff_float).limit_denominator(_RATIONAL_MAX_DEN)
    return frac, coeff_float


def _flow_scale(
    network: Network, min_lanes: int
) -> tuple[float, np.ndarray, np.ndarray]:
    """Choose the flow scale ``s`` and classify every link.

    Returns ``(s, kind, coeff)`` where ``kind[a]`` is 0 normal / 1 zero-intercept
    / 2 zero-slope and ``coeff[a]`` the exact ``numLanes/s`` rational for normal
    links (0 otherwise). ``s = s_base * m`` with ``s_base`` the smallest scale
    making every normal link's lane count integral and ``m`` the smallest integer
    bounding the forced intercepts and honoring ``min_lanes``. Refuses (ValueError
    naming the link) when a normal link's coefficient is not an exact bounded
    rational -- generic float coefficients cannot be exactly lane-quantized."""
    a, b = _linear_params(network)
    kind = np.where(
        a <= _ZERO_INTERCEPT_A, 1, np.where(b <= _ZERO_SLOPE_B, 2, 0)
    ).astype(np.int64)

    coeffs: list[Fraction] = []
    for i in range(len(a)):
        if kind[i] != 0:
            coeffs.append(Fraction(0))
            continue
        frac, cfloat = _lane_coeff(float(a[i]), float(b[i]))
        if cfloat <= 0 or abs(float(frac) - cfloat) > _RATIONAL_RTOL * cfloat:
            raise ValueError(
                f"sumo-marouter cannot lane-quantize link {i} (fft={a[i]}, "
                f"slope={b[i]}): its lane coefficient {cfloat!r} is not an exact "
                f"rational within {_RATIONAL_RTOL} (bounded denominator "
                f"{_RATIONAL_MAX_DEN}); the linear network is not representable "
                "by integer lane counts (adr-027)."
            )
        coeffs.append(frac)

    normal = [c for c, k in zip(coeffs, kind, strict=True) if k == 0]
    if normal:
        s_base = 1
        for c in normal:
            s_base = math.lcm(s_base, c.denominator)
        s_base = float(s_base)
    else:
        s_base = 1000.0

    m = 1
    zi = [float(b[i]) for i in range(len(a)) if kind[i] == 1]
    if zi:
        max_eps_base = max(zi) * (_ZI_CAP_PER_LANE / _ZI_K) / s_base
        m = max(m, math.ceil(max_eps_base / _INTERCEPT_FLOOR_TOL))
    if normal:
        min_lane_base = min(float(c) * s_base for c in normal)
        if min_lane_base > 0:
            m = max(m, math.ceil(min_lanes / min_lane_base))
    return s_base * m, kind, np.array([float(c) for c in coeffs], dtype=np.float64)


def _edge_specs(
    network: Network, s: float, kind: np.ndarray, coeff: np.ndarray, time_scale: float
) -> tuple[list[_EdgeSpec], float, float, np.ndarray]:
    """Solve (speed band, lanes, length) per link, validating all three
    representability bounds BEFORE any file is written (refuse, never silently
    clamp/cap). Returns (specs, forced_intercept, parasitic_slope, representable)."""
    a, b = _linear_params(network)
    specs: list[_EdgeSpec] = []
    forced_intercept = 0.0
    parasitic_slope = 0.0
    representable = np.zeros(network.n_links, dtype=bool)
    for i in range(network.n_links):
        ai, bi = float(a[i]), float(b[i])
        if kind[i] == 1:  # zero-intercept: forced intercept, speed=5, lanes clear clamp
            eps_1 = bi * (_ZI_CAP_PER_LANE / _ZI_K) / s
            min_eps = _MIN_EDGE_LENGTH_M * _LENGTH_TARGET_FACTOR / (_ZI_SPEED * time_scale)
            lanes = max(1, math.ceil(min_eps / eps_1))
            eps = lanes * eps_1
            if eps > _INTERCEPT_FLOOR_TOL:
                raise ValueError(
                    f"sumo-marouter cannot represent zero-intercept link {i} "
                    f"(slope={bi}): clearing netconvert's {_MIN_EDGE_LENGTH_M} m "
                    f"minimum edge length needs {lanes} lanes, forcing an intercept "
                    f"eps={eps:.5g} > _INTERCEPT_FLOOR_TOL={_INTERCEPT_FLOOR_TOL} "
                    "(raise time_scale or accept the refusal; adr-027)."
                )
            length = _ZI_SPEED * time_scale * eps
            speed = _ZI_SPEED
            forced_intercept = max(forced_intercept, eps)
        elif kind[i] == 2:  # zero-slope: minimize parasitic slope with many lanes
            need = ai * _NORMAL_K * s / (_NORMAL_CAP_PER_LANE * _PARASITIC_SLOPE_TOL)
            lanes = max(1, math.ceil(need))
            if lanes > _MAX_LANES:
                achievable = ai * _NORMAL_K * s / (_NORMAL_CAP_PER_LANE * _MAX_LANES)
                raise ValueError(
                    f"sumo-marouter cannot represent zero-slope link {i} (fft={ai}): "
                    f"holding the parasitic slope below _PARASITIC_SLOPE_TOL="
                    f"{_PARASITIC_SLOPE_TOL} needs {lanes} lanes > _MAX_LANES="
                    f"{_MAX_LANES}; the achievable floor is {achievable:.5g} (adr-027)."
                )
            length = ai * time_scale * _NORMAL_SPEED
            speed = _NORMAL_SPEED
            parasitic_slope = max(
                parasitic_slope, ai * _NORMAL_K * s / (_NORMAL_CAP_PER_LANE * lanes)
            )
        else:  # normal: exact integral lanes -> matches the repo BPR to machine eps
            lanes = int(round(float(coeff[i]) * s))
            length = ai * time_scale * _NORMAL_SPEED
            speed = _NORMAL_SPEED
            representable[i] = True
        if lanes > _MAX_LANES:
            raise ValueError(
                f"sumo-marouter cannot compile link {i}: the quantized lane count "
                f"{lanes} exceeds _MAX_LANES={_MAX_LANES} (the flow scale s={s:g} "
                "explodes the lane quantization; a representability limit, not an "
                "engine failure -- adr-027)."
            )
        specs.append(_EdgeSpec(kind=int(kind[i]), lanes=int(lanes), length=length, speed=speed))
    return specs, forced_intercept, parasitic_slope, representable


def _od_window_hours(
    specs: list[_EdgeSpec], total_demand: float, s: float
) -> int:
    """Hours the ``$OR;D2`` window must span so no path's cumulative (mapped) time
    exceeds it -- else marouter silently reverts over-window edges to free-flow
    (AON collapse). Upper-bounds any path cost by the sum of every edge's congested
    time at the full demand RATE ``total_demand*s`` veh/h (a safe over-estimate;
    the rate is window-invariant because the trip count is scaled by the window in
    ``scenario_to_sumo``). ``floor(.)+1`` guarantees ``H*3600 > bound`` strictly
    while leaving small-demand anchors (bound << 3600 s) at H=1."""
    f = total_demand * s
    total_sec = 0.0
    for spec in specs:
        cap_per_lane = _ZI_CAP_PER_LANE if spec.kind == 1 else _NORMAL_CAP_PER_LANE
        k = _ZI_K if spec.kind == 1 else _NORMAL_K
        c = cap_per_lane * spec.lanes
        t0_sec = spec.length / spec.speed
        total_sec += t0_sec * (1.0 + k * f / c)
    return max(1, math.floor(total_sec / _OD_INTERVAL_SEC) + 1)


def _node_names(network: Network) -> tuple[dict[int, str], dict[int, str]]:
    """Tail/head node names, splitting centroids (id < first_thru_node) into
    ``{id}src`` / ``{id}snk`` so no through path can traverse a zone centroid
    (the TNTP first_thru_node convention; verified mass-exact in the pilot).
    Non-centroid nodes are ``n{id}``; when first_thru_node == 1 nothing splits."""
    ftn = network.first_thru_node
    tail: dict[int, str] = {}
    head: dict[int, str] = {}
    for v in range(1, network.n_nodes + 1):
        if v < ftn:
            tail[v] = f"z{v}src"
            head[v] = f"z{v}snk"
        else:
            tail[v] = head[v] = f"n{v}"
    return tail, head


def _write_xml(path: str, root: ET.Element) -> None:
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=False)


def _run_netconvert(cmd: list[str], workdir: str, net_file: str, deadline: float | None) -> None:
    """Run netconvert under the remaining wall budget, mapping every failure surface
    (timeout, missing binary, nonzero exit) to the contract's RuntimeError."""
    timeout = None
    if deadline is not None:
        timeout = deadline - time.perf_counter()
        if timeout <= 0:
            raise RuntimeError(
                "wall_seconds budget exhausted before netconvert could run "
                f"(compile phase over budget):\n  cmd: {' '.join(cmd)}"
            )
    try:
        proc = subprocess.run(
            cmd, env=sumo_env(), capture_output=True, text=True, cwd=workdir, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "netconvert exceeded the wall_seconds budget and was killed:\n  cmd: "
            f"{' '.join(cmd)}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"netconvert could not be executed ({exc}):\n  cmd: {' '.join(cmd)}"
        ) from exc
    if proc.returncode != 0 or not os.path.exists(net_file):
        raise RuntimeError(
            f"netconvert failed (exit {proc.returncode}):\n  cmd: {' '.join(cmd)}\n"
            f"  stderr tail: {proc.stderr[-800:]}"
        )


def _verify_compiled(net_file: str, specs: list[_EdgeSpec]) -> None:
    """Read the compiled net.net.xml back and verify every ``e{i}`` edge's lane
    count, length and speed match the declared spec -- catching netconvert's silent
    0.1 m minimum-length clamp (and any other geometry rewrite) as a compile
    failure (RuntimeError), never a silently-wrong cost model (adr-027)."""
    compiled: dict[str, tuple[int, float, float]] = {}
    root = ET.parse(net_file).getroot()
    for edge in root.findall("edge"):
        eid = edge.get("id", "")
        if eid.startswith(":"):
            continue  # internal junction edge
        lanes = edge.findall("lane")
        if not lanes:
            continue
        compiled[eid] = (
            len(lanes),
            float(lanes[0].get("length", "nan")),
            float(lanes[0].get("speed", "nan")),
        )
    for i, spec in enumerate(specs):
        eid = f"e{i}"
        if eid not in compiled:
            raise RuntimeError(
                f"compile read-back: edge {eid} missing from {net_file} "
                "(netconvert dropped it); refusing a corrupted network (adr-027)."
            )
        n_lanes, length, speed = compiled[eid]
        if n_lanes != spec.lanes:
            raise RuntimeError(
                f"compile read-back: edge {eid} numLanes {n_lanes} != declared "
                f"{spec.lanes}; netconvert rewrote the network (adr-027)."
            )
        if abs(length - spec.length) > _READBACK_LENGTH_RTOL * max(spec.length, 1e-9):
            raise RuntimeError(
                f"compile read-back: edge {eid} length {length} != declared "
                f"{spec.length} (likely netconvert's {_MIN_EDGE_LENGTH_M} m minimum-"
                "length clamp corrupting a zero-intercept eps-edge); the certified "
                "mapping floor would be false -- refusing (adr-027)."
            )
        if abs(speed - spec.speed) > _READBACK_SPEED_RTOL * max(spec.speed, 1e-9):
            raise RuntimeError(
                f"compile read-back: edge {eid} speed {speed} != declared "
                f"{spec.speed}; netconvert rewrote the network (adr-027)."
            )


def scenario_to_sumo(
    scenario: Scenario,
    workdir: str,
    *,
    time_scale: float = 1.0,
    min_lanes: int = 1,
    deadline: float | None = None,
) -> SumoBuild:
    """Compile ``scenario`` into a SUMO net + TAZ + OD matrix under ``workdir``.

    Only fixed-demand ``power == 1`` scenarios are representable; the caller
    (``SumoMarouterModel.solve``) refuses everything else (incl. tolls). Edge ids
    are ``e{link_index}`` (bijective with the repo link order); TAZ sources/sinks
    are the zone's boundary edges (no connector edges -- verified mass-exact). All
    three representability bounds are validated BEFORE any file is written; drives
    ``netconvert`` (absolute wheel binary, ``SUMO_HOME`` overridden, ``--precision
    6``, remaining wall budget as timeout) and then reads the compiled net back to
    verify declared == compiled. ``deadline`` (a ``time.perf_counter`` instant)
    bounds the compile phase."""
    network = scenario.network
    if np.any(np.asarray(network.power, dtype=np.float64) != 1.0):
        raise ValueError(
            "sumo-marouter represents only power==1 (linear) latencies; "
            "marouter's hardcoded vdf cannot express BPR power!=1"
        )
    a, _ = _linear_params(network)
    s, kind, coeff = _flow_scale(network, min_lanes)
    specs, forced_intercept, parasitic_slope, representable = _edge_specs(
        network, s, kind, coeff, time_scale
    )
    tail, head = _node_names(network)

    # --- nodes (coords are scattered in 2D only so netconvert's junction
    #     geometry is non-degenerate; the explicit connection file below, not the
    #     geometry, decides connectivity, and marouter uses the explicit edge
    #     length, so positions never enter the certificate) -------------------
    used_nodes: dict[str, tuple[float, float]] = {}
    for v in range(1, network.n_nodes + 1):
        used_nodes.setdefault(tail[v], (float(v) * 100.0, float((v * 173) % 400)))
        used_nodes.setdefault(head[v], (float(v) * 100.0 + 40.0, float((v * 91) % 400) + 50.0))
    nod_root = ET.Element("nodes")
    for name, (x, y) in used_nodes.items():
        ET.SubElement(nod_root, "node", id=name, x=repr(x), y=repr(y), type="priority")

    # --- edges (from the validated specs) --------------------------------------
    edg_root = ET.Element("edges")
    for i, spec in enumerate(specs):
        ET.SubElement(
            edg_root,
            "edge",
            id=f"e{i}",
            attrib={"from": tail[int(network.init_node[i])], "to": head[int(network.term_node[i])]},
            numLanes=str(spec.lanes),
            speed=repr(float(spec.speed)),
            length=repr(float(spec.length)),
        )

    # --- connections: allow EVERY in->out movement at each physical junction
    #     (minus the immediate U-turn), independent of the scattered geometry.
    #     Without this, netconvert's angle-based pruning silently drops turns on
    #     an abstract net (verified: it collapsed Braess to a single path). Split
    #     centroids need none (their src/snk half-nodes are terminal). -------
    con_root = ET.Element("connections")
    init = np.asarray(network.init_node)
    term = np.asarray(network.term_node)
    for v in range(1, network.n_nodes + 1):
        if v < network.first_thru_node:
            continue  # split centroid: src/snk are terminal, no through movement
        for i in np.nonzero(term == v)[0]:
            for j in np.nonzero(init == v)[0]:
                if int(term[j]) == int(init[i]):
                    continue  # immediate U-turn back along the same pair
                ET.SubElement(
                    con_root, "connection", attrib={"from": f"e{int(i)}", "to": f"e{int(j)}"}
                )

    # --- TAZ: source = zone's out-edges, sink = zone's in-edges (boundary) -----
    taz_root = ET.Element("additional")
    for z in range(1, network.n_zones + 1):
        taz = ET.SubElement(taz_root, "taz", id=str(z))
        for i in np.nonzero(network.init_node == z)[0]:
            ET.SubElement(taz, "tazSource", id=f"e{int(i)}", weight="1")
        for i in np.nonzero(network.term_node == z)[0]:
            ET.SubElement(taz, "tazSink", id=f"e{int(i)}", weight="1")

    # --- OD matrix in $OR;D2 format, window sized to cover worst-path times -----
    # The trip count is scaled by the window so the flow RATE (trips/hour) -- which
    # drives marouter's congestion and thus the equilibrium -- is window-invariant;
    # native flows are recovered by dividing the netload ``entered`` (trips over the
    # whole interval) by ``flow_scale = s * window`` (adr-027 review MAJOR: a naive
    # widening halved the rate and collapsed Braess to AON).
    od = np.asarray(scenario.demand.matrix, dtype=np.float64)
    window = _od_window_hours(specs, float(scenario.demand.total), s)
    flow_scale = s * window
    lines = ["$OR;D2", "* From-Time  To-Time", f"0.00 {window}.00", "* Factor", "1.00"]
    n = od.shape[0]
    for o in range(n):
        for d in range(n):
            if o != d and od[o, d] > 0:
                lines.append(f"{o + 1} {d + 1} {od[o, d] * flow_scale:.9f}")

    nod_file = os.path.join(workdir, "net.nod.xml")
    edg_file = os.path.join(workdir, "net.edg.xml")
    con_file = os.path.join(workdir, "net.con.xml")
    taz_file = os.path.join(workdir, "net.taz.xml")
    od_file = os.path.join(workdir, "matrix.od")
    net_file = os.path.join(workdir, "net.net.xml")
    _write_xml(nod_file, nod_root)
    _write_xml(edg_file, edg_root)
    _write_xml(con_file, con_root)
    _write_xml(taz_file, taz_root)
    with open(od_file, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    cmd = [
        sumo_binary("netconvert"),
        "--node-files", nod_file,
        "--edge-files", edg_file,
        "--connection-files", con_file,
        "--output-file", net_file,
        "--precision", "6",
        "--no-turnarounds", "true",
        "--no-internal-links", "true",
        "--offset.disable-normalization", "true",
    ]
    _run_netconvert(cmd, workdir, net_file, deadline)
    _verify_compiled(net_file, specs)
    return SumoBuild(
        net_file=net_file,
        taz_file=taz_file,
        od_file=od_file,
        flow_scale=flow_scale,
        time_scale=time_scale,
        forced_intercept=forced_intercept,
        parasitic_slope=parasitic_slope,
        representable=representable,
        od_window_hours=window,
    )


def parse_netload_attr(path: str, n_links: int, attr: str) -> np.ndarray:
    """Per-edge float ``attr`` from a marouter ``--netload-output`` file, indexed
    by the ``e{link_index}`` edge id (missing edges -> 0.0). Used for ``entered``
    (parse_netload divides by the flow scale) and for ``traveltime`` (the A2
    cost-match check reads marouter's internal free-flow-plus-congestion time)."""
    values = np.zeros(n_links, dtype=np.float64)
    root = ET.parse(path).getroot()
    for edge in root.iter("edge"):
        eid = edge.get("id", "")
        raw = edge.get(attr)
        if eid.startswith("e") and raw is not None:
            try:
                idx = int(eid[1:])
            except ValueError:
                continue
            if 0 <= idx < n_links:
                values[idx] = float(raw)
    return values


def parse_netload(path: str, n_links: int, flow_scale: float) -> np.ndarray:
    """Repo-native link flows from a marouter ``--netload-output`` file.

    Reads the ``entered`` attribute (veh per aggregation interval) at the edge's
    ``e{link_index}`` id and divides by ``flow_scale`` -- NEVER the integerized
    route file (route ``<flow number=...>`` is rounded to integers; the
    macroscopic netload doubles are exact, the pilot's decisive finding)."""
    return parse_netload_attr(path, n_links, "entered") / flow_scale


def netload_interval_count(path: str) -> int:
    """Number of ``<interval>`` blocks in a netload file. The window is sized so
    exactly ONE interval is emitted; more would mean ``parse_netload`` silently
    keeps only the last interval's flows (dropping mass) -- the caller raises."""
    return len(ET.parse(path).getroot().findall("interval"))


def netload_matched_edge_count(path: str, n_links: int) -> int:
    """Number of distinct ``e{i}`` edges present in a netload file. Zero while
    demand is positive means the engine wrote no edge data at all (an engine
    failure the caller must RAISE, not launder into an all-zero censor row)."""
    seen: set[int] = set()
    root = ET.parse(path).getroot()
    for edge in root.iter("edge"):
        eid = edge.get("id", "")
        if eid.startswith("e"):
            try:
                idx = int(eid[1:])
            except ValueError:
                continue
            if 0 <= idx < n_links:
                seen.add(idx)
    return len(seen)
