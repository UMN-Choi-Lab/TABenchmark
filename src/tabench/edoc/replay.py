"""G1 replay-harness types: the pinned-engine replay result and the injected
runner protocol (adr-036 G1).

Design: docs/design/adr-036-external-dynamic-observational-certificate.md.

G1 (the A2 analogue) re-runs the pinned engine in zero-replanning replay on the
model's emitted plans; the replayed per-agent (departure, arrival, route,
experienced-time) tuples must equal the emitted record ``X`` exactly under the
pinned canonicalization, and the replay runs TWICE with identical canonicalized
output (the determinism double). The engine call is **injected** as a
``ReplayRunner`` so the substrate's core tests run engine-free against synthetic
replay fixtures; the real SUMO runner (adr-037) drives ``duaIterate``/``sumo``.

**Runner contract (G0 split).** A ``ReplayRunner`` MUST, before it replays,
assert that the engine actually installed on *this* box equals the instance's
pinned ``engine_version`` — call :func:`assert_engine_pin`, which RAISES a
``ValueError`` on a mismatch (a G0 config error, never a censor). The certifier
also checks the model's *emitted* provenance against the pin (that is the model's
self-report); the runner checking the box it runs on is what makes "the installed
version is read at certify time" true — the frozen field must be built by the
engine the instance pins, not merely by one the emission claims.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


class PlanReplayFailure(RuntimeError):
    """The ONE crash-vs-censor censor signal (adr-036 R6): the pinned engine
    subprocess crashed or timed out **while replaying the model's emitted plans**
    (an unexecutable / head-blocking plan is an invalid emission). A runner raises
    this — and ONLY this — for that case; the certifier catches ONLY this type and
    converts it to ``feasible=0``. Every other failure (deadline pre-exhaustion,
    a missing engine binary, a net-compile / read-back failure, a garbage/missing
    artifact after ``rc=0``, the determinism-double mismatch) is a certifier-side
    **infrastructure** fault that raises a plain ``RuntimeError``/``OSError`` and
    propagates — never laundered into a censor (R6's second arm)."""


@dataclass(frozen=True)
class ReplayAgent:
    """One agent's replayed experienced record: scheduled departure, arrival, the
    driven edge route, the door-to-door experienced time, and the off-network
    insertion (origin) wait ``depart_delay`` (INCLUDED in every cost, G3)."""

    agent_id: str
    departure: float
    arrival: float
    route: tuple[str, ...]
    experienced_time: float
    depart_delay: float

    @property
    def first_edge(self) -> str:
        return self.route[0]


@dataclass(frozen=True)
class ReplayResult:
    """The parsed, canonicalized output of one pinned-engine replay. ``canon_hash``
    is the object G1's determinism double compares; ``agents`` is the replayed
    per-agent record (compared to the emitted ``X``); ``field_records`` and
    ``flows`` are the model-blind material the certifier builds ``Ĉ`` and the G4
    conservation checks from."""

    canon_hash: str
    agents: dict[str, ReplayAgent]
    # edge -> interval -> (interval-mean traveltime, occupancy witness)
    field_records: dict[str, dict[int, tuple[float, float]]]
    # edge -> interval -> (entered, left) cumulative-flow counts (G4)
    flows: dict[str, dict[int, tuple[float, float]]] = field(default_factory=dict)
    n_intervals: int = 0


@dataclass(frozen=True)
class EmittedBundle:
    """What a model emits (adr-036 artifact contract): plans ``P`` (per-agent route
    + scheduled departure), the experienced record ``X``, and provenance (engine
    version + seed, gated at G0, never the engine's self-reported convergence)."""

    plans: dict[str, tuple[tuple[str, ...], float]]
    experienced: dict[str, ReplayAgent]
    engine_version: str
    seed: int


def assert_engine_pin(installed_version: str, pinned_version: str) -> None:
    """A ReplayRunner's mandatory pre-replay check (G0 split): the engine actually
    installed on this box must equal the instance's pinned ``engine_version``.

    A mismatch is a **configuration error and RAISES** ``ValueError`` eagerly (the
    adr-020 eager-config / adr-036 R6 crash-vs-censor map — an infra/config fault
    is never laundered into ``feasible=0``). The certifier separately checks the
    *emitted* provenance (the model's self-report); this is the runner verifying
    the box it is about to build the frozen field on, which is what the "read the
    installed version at certify time" clause of G0 actually requires."""
    if str(installed_version) != str(pinned_version):
        raise ValueError(
            f"G0 engine pin: installed engine {installed_version!r} != instance pin "
            f"{pinned_version!r}; refusing to replay (the frozen field would be built "
            "by a different engine than the instance pins)"
        )


# A ReplayRunner re-runs the pinned engine on the emitted plans and returns a
# parsed ReplayResult. Injected so the certifier is engine-agnostic. It MUST call
# assert_engine_pin before replaying (see the module docstring's runner contract).
ReplayRunner = Callable[["object", dict[str, tuple[tuple[str, ...], float]]], ReplayResult]
