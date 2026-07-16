"""EdocScenario: the frozen, content-hashed external-dynamic instance (adr-036).

Design: docs/design/adr-036-external-dynamic-observational-certificate.md.

The instance IS the engine (unlike the static external rows, where the engine's
cost law is matched against the declared BPR and its version is manifest
provenance — adr-027/029; here no declared cost law exists to certify against, so
engine identity + version sit INSIDE the instance hash). Per adr-036's
hash-everything ruling (MAJOR-5), the content hash covers **every certifier-side
constant that changes a censor / floor / score outcome**: the network + demand,
the engine identity + EXACT version pin, the pinned seed, the semantic engine
config, Δ, the backlog bound, the negative-control separation factor, the
resolution floor and replay hard-deadline, AND the R2 field-semantics +
completion-rule selections, the origin-wait convention, and the option-B walk
universe/length bound. Domain ``"tabench-edoc-scenario-v1;"``, length-framed (the
newell-3det defense-in-depth) — domain-separated so no existing scenario hash can
move.

Construction gates raise plain ``ValueError`` from ``__post_init__`` (the house
convention; engine-dependent refusals — negative-control non-separation, the
DTALite boost-clean census — belong to the row's construction, not this pure data
instance). Numeric arrays are frozen read-only (adr-020 precedent).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from dataclasses import field as _dc_field

import numpy as np

_SCENARIO_DOMAIN = b"tabench-edoc-scenario-v1;"

# Hard cap on per-edge lane count (the marouter ``_sumo_io._MAX_LANES`` precedent):
# adr-027's review measured ``netconvert`` HANGING on nets whose lane counts exploded
# past ~1e5, so a hostile or buggy instance must be REFUSED at construction — never
# discovered as a netconvert hang burning the row's wall deadline (adr-037 rider).
_MAX_LANES = 4000

FIELD_SEMANTICS = ("raw",)  # "monotonized" is a named future family (adr-036 R2)
COMPLETION_RULES = ("occupancy_aware",)
ORIGIN_WAIT_CONVENTIONS = ("profile", "agent_symmetric")


@dataclass(frozen=True)
class EdocScenario:
    """A frozen external-dynamic instance. Topology is an explicit directed edge
    graph (SUMO edge IDs are strings); demand is a fixed per-agent trip table
    (agent id, origin node, destination node, scheduled departure time)."""

    name: str
    # --- topology (parallel arrays over edges) ---
    edge_ids: tuple[str, ...]
    edge_tail: tuple[str, ...]
    edge_head: tuple[str, ...]
    edge_fftt: np.ndarray
    # Per-edge lane count: the ENGINE-side capacity dial. SUMO meso flow capacity
    # is a function of lanes + freespeed + the pinned meso config (measured ~1584
    # veh/h/lane at 13.89 m/s under the pinned config), so a capacity drop = fewer
    # lanes on a route-distinguishing edge (adr-036 line 557). Outcome-bearing ->
    # hashed. Geometry is otherwise a DETERMINISTIC function of the abstract graph:
    # canonical freespeed `canon_speed_mps`, edge length = fftt * canon_speed_mps
    # (so free-flow time = length / speed = fftt exactly), so no coordinate choice
    # can inject uncontrolled cost (junction control is pinned off in the adapter).
    edge_lanes: np.ndarray
    # --- demand trip table (parallel arrays over agents) ---
    agent_ids: tuple[str, ...]
    agent_origin: tuple[str, ...]
    agent_dest: tuple[str, ...]
    agent_depart: np.ndarray
    # --- engine identity (inside the hash — the NEW move vs adr-027/029) ---
    engine: str
    engine_version: str
    seed: int
    semantic_config: str
    # --- field / scoring constants (all hashed, MAJOR-5) ---
    dt: float
    n_intervals: int
    departure_quantum: float
    backlog_bound: float
    separation_factor: float
    floor_seconds: float
    replay_deadline_s: float
    canon_speed_mps: float = 13.89  # canonical freespeed (length = fftt * this)
    r3_tolerance_s: float = 15.0  # native-router cross-check agreement bound (R3 RAISE)
    field_semantics: str = "raw"
    completion_rule: str = "occupancy_aware"
    origin_wait_convention: str = "profile"
    walk_bound: int = 8
    walk_count_bound: int = 200_000
    family: str = _dc_field(default="")

    # -------------------------------------------------------------- gates
    def __post_init__(self) -> None:
        object.__setattr__(self, "edge_ids", tuple(map(str, self.edge_ids)))
        object.__setattr__(self, "edge_tail", tuple(map(str, self.edge_tail)))
        object.__setattr__(self, "edge_head", tuple(map(str, self.edge_head)))
        object.__setattr__(self, "agent_ids", tuple(map(str, self.agent_ids)))
        object.__setattr__(self, "agent_origin", tuple(map(str, self.agent_origin)))
        object.__setattr__(self, "agent_dest", tuple(map(str, self.agent_dest)))
        fftt = np.ascontiguousarray(self.edge_fftt, dtype=np.float64)
        lanes = np.ascontiguousarray(self.edge_lanes, dtype=np.int64)
        dep = np.ascontiguousarray(self.agent_depart, dtype=np.float64)
        fftt.flags.writeable = False
        lanes.flags.writeable = False
        dep.flags.writeable = False
        object.__setattr__(self, "edge_fftt", fftt)
        object.__setattr__(self, "edge_lanes", lanes)
        object.__setattr__(self, "agent_depart", dep)
        if not self.family:
            object.__setattr__(self, "family", self.name)

        n_e = len(self.edge_ids)
        if n_e == 0:
            raise ValueError(f"EdocScenario {self.name!r}: needs >= 1 edge")
        if not (len(self.edge_tail) == len(self.edge_head) == fftt.shape[0] == n_e):
            raise ValueError(f"EdocScenario {self.name!r}: edge arrays length-mismatched")
        if lanes.shape[0] != n_e:
            raise ValueError(f"EdocScenario {self.name!r}: edge_lanes length-mismatched")
        if len(set(self.edge_ids)) != n_e:
            raise ValueError(f"EdocScenario {self.name!r}: duplicate edge id")
        if not np.all(np.isfinite(fftt)) or fftt.min(initial=1.0) <= 0.0:
            raise ValueError(f"EdocScenario {self.name!r}: edge fftt must be finite and > 0")
        if lanes.min(initial=1) < 1:
            raise ValueError(f"EdocScenario {self.name!r}: edge_lanes must be >= 1 (SUMO numLanes)")
        if lanes.max(initial=1) > _MAX_LANES:
            raise ValueError(
                f"EdocScenario {self.name!r}: edge_lanes exceeds {_MAX_LANES} (netconvert hangs on "
                "exploded lane counts — refuse at construction, not as a wall-deadline hang)"
            )
        if not np.isfinite(self.canon_speed_mps) or self.canon_speed_mps <= 0.0:
            raise ValueError(f"EdocScenario {self.name!r}: canon_speed_mps must be finite and > 0")

        n_a = len(self.agent_ids)
        if n_a == 0:
            raise ValueError(f"EdocScenario {self.name!r}: needs >= 1 agent")
        if not (len(self.agent_origin) == len(self.agent_dest) == dep.shape[0] == n_a):
            raise ValueError(f"EdocScenario {self.name!r}: agent arrays length-mismatched")
        if len(set(self.agent_ids)) != n_a:
            raise ValueError(f"EdocScenario {self.name!r}: duplicate agent id")
        nodes = set(self.edge_tail) | set(self.edge_head)
        for o, d in zip(self.agent_origin, self.agent_dest, strict=True):
            if o not in nodes or d not in nodes:
                raise ValueError(f"EdocScenario {self.name!r}: agent OD {o}->{d} off the network")
        if not np.all(np.isfinite(dep)):
            raise ValueError(f"EdocScenario {self.name!r}: non-finite departure time")
        # fixed departures give ZERO timing freedom (G2): every departure must sit
        # on the declared engine grid, so no sub-quantum timing game exists.
        if self.departure_quantum <= 0.0:
            raise ValueError(f"EdocScenario {self.name!r}: departure_quantum must be > 0")
        q = self.departure_quantum
        if np.abs(dep / q - np.round(dep / q)).max(initial=0.0) > 1e-9:
            raise ValueError(
                f"EdocScenario {self.name!r}: departures not on the {q}s engine grid"
            )

        if self.dt <= 0.0 or self.n_intervals <= 0:
            raise ValueError(f"EdocScenario {self.name!r}: dt > 0 and n_intervals > 0 required")
        # Departure-WINDOW gate (adr-036 forgery pair 1 "quantize departures into
        # [start, end)"; pair 12 "unclearable horizons"): every departure must lie
        # in the field window [0, dt*n_intervals). A negative or beyond-horizon
        # departure is a CONSTRUCTION error caught eagerly (adr-020/036), not left
        # to surface as a late G3 completion censor. Clearing HEADROOM (an agent
        # actually completing before the horizon) is engine-dependent, so it stays a
        # certify-time G3 census concern (the replay agent set must equal the trip
        # table); this pure-data gate bounds only the departure window itself.
        horizon = float(self.dt) * int(self.n_intervals)
        if dep.size and (dep.min() < 0.0 or dep.max() >= horizon):
            raise ValueError(
                f"EdocScenario {self.name!r}: every departure must lie in the field "
                f"window [0, {horizon}) s (dt*n_intervals); got "
                f"[{dep.min()}, {dep.max()}] — a negative or beyond-horizon departure "
                "is a construction error (adr-036 forgery pair 1/12)"
            )
        if self.field_semantics not in FIELD_SEMANTICS:
            raise ValueError(
                f"EdocScenario {self.name!r}: field_semantics {self.field_semantics!r} "
                f"not in {FIELD_SEMANTICS} (monotonized is a named future family)"
            )
        if self.completion_rule not in COMPLETION_RULES:
            raise ValueError(
                f"EdocScenario {self.name!r}: completion_rule {self.completion_rule!r} "
                f"not in {COMPLETION_RULES}"
            )
        if self.origin_wait_convention not in ORIGIN_WAIT_CONVENTIONS:
            raise ValueError(
                f"EdocScenario {self.name!r}: origin_wait_convention "
                f"{self.origin_wait_convention!r} not in {ORIGIN_WAIT_CONVENTIONS}"
            )
        if self.walk_bound < 1:
            raise ValueError(f"EdocScenario {self.name!r}: walk_bound must be >= 1")
        if self.walk_count_bound < 1:
            raise ValueError(f"EdocScenario {self.name!r}: walk_count_bound must be >= 1")
        for label, val in (
            ("backlog_bound", self.backlog_bound),
            ("separation_factor", self.separation_factor),
            ("floor_seconds", self.floor_seconds),
            ("replay_deadline_s", self.replay_deadline_s),
            ("r3_tolerance_s", self.r3_tolerance_s),
        ):
            if not np.isfinite(val) or val < 0.0:
                raise ValueError(f"EdocScenario {self.name!r}: {label} must be finite and >= 0")
        if self.separation_factor < 1.0:
            raise ValueError(
                f"EdocScenario {self.name!r}: separation_factor must be >= 1 (declared "
                f"negative-control anchor)"
            )

    # -------------------------------------------------------------- derived
    @property
    def n_edges(self) -> int:
        return len(self.edge_ids)

    @property
    def n_agents(self) -> int:
        return len(self.agent_ids)

    def out_edges(self) -> dict[str, list[str]]:
        """Adjacency: node -> outgoing edge ids (certifier TD-SP input)."""
        adj: dict[str, list[str]] = {}
        for eid, tail in zip(self.edge_ids, self.edge_tail, strict=True):
            adj.setdefault(tail, []).append(eid)
        return adj

    def head_of(self) -> dict[str, str]:
        return dict(zip(self.edge_ids, self.edge_head, strict=True))

    def fftt_of(self) -> dict[str, float]:
        return {eid: float(f) for eid, f in zip(self.edge_ids, self.edge_fftt, strict=True)}

    def lanes_of(self) -> dict[str, int]:
        return {eid: int(n) for eid, n in zip(self.edge_ids, self.edge_lanes, strict=True)}

    def length_of(self) -> dict[str, float]:
        """Canonical SUMO edge length: fftt * canon_speed (free-flow time = fftt)."""
        v = float(self.canon_speed_mps)
        return {eid: float(f) * v for eid, f in zip(self.edge_ids, self.edge_fftt, strict=True)}

    # -------------------------------------------------------------- hash
    def content_hash(self) -> str:
        """SHA-256 over every scored-outcome-bearing field, domain-separated and
        length-framed (adr-036 MAJOR-5 hash-everything). A hash-coverage test
        (test_edoc) mutates each field and asserts the digest moves."""
        h = hashlib.sha256()
        h.update(_SCENARIO_DOMAIN)
        h.update(f"name={self.name};family={self.family};".encode())
        h.update(f"edges={self.n_edges};agents={self.n_agents};".encode())
        # string parallel arrays (topology + demand endpoints)
        for label, seq in (
            ("eid", self.edge_ids),
            ("etail", self.edge_tail),
            ("ehead", self.edge_head),
            ("aid", self.agent_ids),
            ("aorg", self.agent_origin),
            ("adst", self.agent_dest),
        ):
            joined = "\x1f".join(seq).encode()
            h.update(f"{label}:{len(joined)};".encode())
            h.update(joined)
        # numeric arrays, length-framed
        for label, arr in (("fftt", self.edge_fftt), ("adep", self.agent_depart)):
            framed = np.ascontiguousarray(arr, dtype=np.float64)
            h.update(f"{label}:{framed.size};".encode())
            h.update(framed.tobytes())
        # per-edge lane count (the engine-side capacity dial), framed as int64
        lanes = np.ascontiguousarray(self.edge_lanes, dtype=np.int64)
        h.update(f"lanes:{lanes.size};".encode())
        h.update(lanes.tobytes())
        # engine identity + all scored-outcome constants
        h.update(
            (
                f"engine={self.engine};ver={self.engine_version};seed={self.seed};"
                f"cfg={self.semantic_config};speed={float(self.canon_speed_mps)!r};"
                f"dt={float(self.dt)!r};"
                f"nint={self.n_intervals};dq={float(self.departure_quantum)!r};"
                f"backlog={float(self.backlog_bound)!r};"
                f"sep={float(self.separation_factor)!r};"
                f"floor={float(self.floor_seconds)!r};"
                f"deadline={float(self.replay_deadline_s)!r};"
                f"r3tol={float(self.r3_tolerance_s)!r};"
                f"fsem={self.field_semantics};compl={self.completion_rule};"
                f"owc={self.origin_wait_convention};wb={self.walk_bound};"
                f"wcb={self.walk_count_bound};"
            ).encode()
        )
        return h.hexdigest()
