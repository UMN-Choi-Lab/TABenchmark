"""The frozen experienced cost field ``Ĉ`` and per-first-edge origin-wait
profiles (rulings R1/R2/MAJOR-1/MAJOR-2 of adr-036).

Design: docs/design/adr-036-external-dynamic-observational-certificate.md.

From the G1 replay the certifier builds, per edge, the interval-mean experienced
traversal time at the pinned interval width Δ — the frozen field ``Ĉ`` the
best-response search and the driven-cost evaluation both compose (same basis, so
``c_drv >= c_br`` by construction). Two rulings shape the field:

* **R2 field completion is occupancy-aware.** A *never-loaded* edge sits at free
  flow (optimistic for the deviation — the plan-set-impoverishment defense). An
  *interior gap* of a *loaded* edge carries forward the last congested cost ONLY
  where the replay shows nonzero occupancy / a standing queue during that gap; a
  *zero-occupancy* interior gap falls back to free flow. A blind carry-forward
  would let two bracketing congested samples poison every interior interval of an
  alternative route (the burst-poisoning attack, forgery pair 8) — occupancy is
  the detection that closes it. For SUMO meandata, occupancy and a measured
  ``traveltime`` co-occur per interval, so a gap is simply an empty interval →
  free flow; the occupancy-aware path is exercised by the poisoned-alternative
  regression instance (a hand-built field with occupancy-without-traveltime gaps).

* **MAJOR-2 origin wait is a per-first-edge PROFILE**, not the agent's own wait:
  the same interval-mean construction and occupancy-aware completion, keyed by the
  first edge, so the best response is charged the *alternative* first edge's own
  measured wait and ``RG_D1`` scores entrance-choice disequilibrium.

This family pins the ``"raw"`` field semantics (label-correcting TD-SP — R2); the
``"monotonized"`` (label-setting) semantics is a named future family with its own
row, so this builder raises on it rather than shipping an untested code path.
"""

from __future__ import annotations

from dataclasses import dataclass

FIELD_SEMANTICS = ("raw", "monotonized")


def _interval(t: float, dt: float, n_intervals: int) -> int:
    """The interval index containing entry time ``t`` (relative to the field
    origin), clamped to ``[0, n_intervals-1]`` — a frozen field has no cost beyond
    its horizon, so a late entry reads the last interval."""
    if t <= 0.0:
        return 0
    k = int(t // dt)
    if k >= n_intervals:
        return n_intervals - 1
    return k


@dataclass(frozen=True)
class FrozenField:
    """Per-edge interval-mean experienced traversal-time field ``Ĉ`` with
    occupancy-aware completion (R2). ``traveltime[edge][k]`` is the measured mean
    cost on ``edge`` in interval ``k``; ``occupancy[edge][k]`` is its occupancy
    signal (a standing-queue witness for completion). ``fftt[edge]`` is the
    free-flow traversal time used for never-loaded edges and zero-occupancy gaps.
    """

    dt: float
    n_intervals: int
    fftt: dict[str, float]
    traveltime: dict[str, dict[int, float]]
    occupancy: dict[str, dict[int, float]]
    semantics: str = "raw"

    def __post_init__(self) -> None:
        if self.semantics != "raw":
            # The monotonized + label-setting family is named future work (adr-036
            # R2); shipping only its hashed selector without the smoothing code
            # would be an untested path.
            raise ValueError(
                f"FrozenField semantics {self.semantics!r} not implemented; the "
                "first EDOC family pins 'raw' (label-correcting TD-SP)"
            )
        if self.dt <= 0.0 or self.n_intervals <= 0:
            raise ValueError("FrozenField needs dt > 0 and n_intervals > 0")

    def is_loaded(self, edge: str) -> bool:
        """An edge is loaded if the replay put any occupancy or measured cost on
        it in any interval; a never-loaded edge is at free flow everywhere."""
        return bool(self.traveltime.get(edge)) or bool(self.occupancy.get(edge))

    def traversal_time(self, edge: str, entry_time: float) -> float:
        """The frozen cost of entering ``edge`` at ``entry_time`` (R2)."""
        ff = self.fftt[edge]
        if not self.is_loaded(edge):
            return ff
        k = _interval(entry_time, self.dt, self.n_intervals)
        tt = self.traveltime.get(edge, {})
        if k in tt:
            return tt[k]
        # gap on a loaded edge: occupancy-aware completion.
        occ = self.occupancy.get(edge, {}).get(k, 0.0)
        if occ > 0.0:
            # standing queue during the gap: carry forward the last congested cost.
            prior = [j for j in tt if j < k]
            if prior:
                return tt[max(prior)]
            return ff
        # zero-occupancy interior gap: free flow (deviation-optimistic, pair-3).
        return ff


@dataclass(frozen=True)
class OriginWaitProfile:
    """Per-first-edge interval-mean origin (insertion) wait, built from the
    replay's own ``departDelay`` samples with the same occupancy-aware completion
    as the link field (MAJOR-2). ``wait(edge, t)`` charges the measured entrance
    queue of ``edge`` for a departure in ``t``'s interval; a never-used first edge
    admits immediately (0 wait — the deviation-optimistic default). The
    ``agent_symmetric`` convention is the optional family alternative (disclosed
    entrance-choice blindness); it is handled by the certifier, not here."""

    dt: float
    n_intervals: int
    wait_mean: dict[str, dict[int, float]]
    occupancy: dict[str, dict[int, float]]

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.n_intervals <= 0:
            raise ValueError("OriginWaitProfile needs dt > 0 and n_intervals > 0")

    def wait(self, first_edge: str, depart_time: float) -> float:
        w = self.wait_mean.get(first_edge)
        if not w:
            return 0.0
        k = _interval(depart_time, self.dt, self.n_intervals)
        if k in w:
            return w[k]
        occ = self.occupancy.get(first_edge, {}).get(k, 0.0)
        if occ > 0.0:
            prior = [j for j in w if j < k]
            if prior:
                return w[max(prior)]
        return 0.0


# --------------------------------------------------------------------------
# Builders from parsed replay artifacts (engine-specific input, engine-agnostic
# field/profile output — the certifier is model-blind once these are built).
# --------------------------------------------------------------------------


def build_field_from_records(
    records: dict[str, dict[int, tuple[float, float]]],
    fftt: dict[str, float],
    dt: float,
    n_intervals: int,
    semantics: str = "raw",
) -> FrozenField:
    """Build a :class:`FrozenField` from per-edge ``{interval: (traveltime,
    occupancy)}`` records (SUMO meandata, or a synthetic fixture). An interval
    present with ``traveltime`` populates ``traveltime[edge][k]``; its occupancy
    populates ``occupancy[edge][k]``. Missing intervals are gaps (completed at
    read time, R2)."""
    tt: dict[str, dict[int, float]] = {}
    occ: dict[str, dict[int, float]] = {}
    for edge, per_k in records.items():
        for k, (traveltime, occupancy) in per_k.items():
            if traveltime is not None:
                tt.setdefault(edge, {})[k] = float(traveltime)
            if occupancy:
                occ.setdefault(edge, {})[k] = float(occupancy)
    return FrozenField(
        dt=dt,
        n_intervals=n_intervals,
        fftt=dict(fftt),
        traveltime=tt,
        occupancy=occ,
        semantics=semantics,
    )


def build_origin_waits(
    departures: list[tuple[str, float, float]],
    dt: float,
    n_intervals: int,
) -> OriginWaitProfile:
    """Build a per-first-edge origin-wait profile from replay departure samples
    ``(first_edge, depart_time, depart_delay)``: the interval-mean ``departDelay``
    per first edge, plus an occupancy witness (a nonzero mean wait in an interval
    marks a standing entrance queue there, so completion carries it forward across
    an interior gap; an interval with no departures has no witness → 0)."""
    sums: dict[str, dict[int, list[float]]] = {}
    for first_edge, depart_time, depart_delay in departures:
        k = _interval(depart_time, dt, n_intervals)
        sums.setdefault(first_edge, {}).setdefault(k, []).append(float(depart_delay))
    wait_mean: dict[str, dict[int, float]] = {}
    occ: dict[str, dict[int, float]] = {}
    for edge, per_k in sums.items():
        for k, vals in per_k.items():
            m = sum(vals) / len(vals)
            wait_mean.setdefault(edge, {})[k] = m
            if m > 0.0:
                occ.setdefault(edge, {})[k] = m
    return OriginWaitProfile(
        dt=dt, n_intervals=n_intervals, wait_mean=wait_mean, occupancy=occ
    )
