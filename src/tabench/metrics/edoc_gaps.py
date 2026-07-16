"""EdocEvaluator: the EDOC-1 certificate — G0-G4 gates + the frozen-field
best-response gap ``RG_D1`` (adr-036).

Design: docs/design/adr-036-external-dynamic-observational-certificate.md.

A pure function of ``(EdocScenario, EmittedBundle, ReplayRunner)``. The certifier
is model-blind: it re-runs the pinned engine on the emitted plans (G1), builds the
frozen field ``Ĉ`` and the per-first-edge origin-wait profiles from the replay,
and recomputes ``RG_D1`` itself. Crash-vs-censor (adr-036 R6, house convention):

* wrong SHAPES / G0 pin mismatch (a config error) → ``raise ValueError`` eagerly;
* the certifier's OWN replay disagreeing with itself (non-deterministic env), a
  net-compile / read-back failure, a deadline pre-exhaustion, a missing engine
  binary, or a garbage/missing artifact → ``raise`` (``RuntimeError``/``OSError``:
  infrastructure, never laundered into ``feasible=0`` — adr-036 R6's second arm);
* an emitted artifact that fails a gate (doctored ``X``, demand mismatch,
  over-floor field, non-clearing delivery) → CENSOR (``feasible = 0``, scored
  ``NaN``, ``logger.info``);
* the pinned engine subprocess crashing/timing out **while replaying the emitted
  plans** (an unexecutable / head-blocking plan is an invalid emission) → the
  runner raises :class:`~tabench.edoc.replay.PlanReplayFailure`, the ONE censor
  signal the certifier catches (R6's first arm). Every OTHER runner exception is
  an infrastructure RAISE.
"""

from __future__ import annotations

import logging

from ..edoc.field import build_field_from_records, build_origin_waits
from ..edoc.replay import (
    EmittedBundle,
    PlanReplayFailure,
    ReplayResult,
    ReplayRunner,
)
from ..edoc.scenario import EdocScenario
from ..edoc.tdsp import evaluate_route, td_shortest_path

logger = logging.getLogger(__name__)

_SCORED = (
    "rg_d1",
    "floor_gap",
    "sub_floor",
    "n_improvers",
    "tstt",
    "mean_backlog",
    "max_backlog",
    "delta",
    "br_coverage",
)

_ZERO_WAIT = build_origin_waits([], dt=1.0, n_intervals=1)


class EdocEvaluator:
    """The EDOC-1 certifier for one instance and one injected replay runner."""

    def __init__(
        self, scenario: EdocScenario, runner: ReplayRunner, *, tol: float = 1e-6
    ) -> None:
        self.scenario = scenario
        self.runner = runner
        self.tol = tol
        self._hash = scenario.content_hash()
        self._trip = {
            aid: (o, d, float(dep))
            for aid, o, d, dep in zip(
                scenario.agent_ids,
                scenario.agent_origin,
                scenario.agent_dest,
                scenario.agent_depart,
                strict=True,
            )
        }
        self._out = scenario.out_edges()
        self._head = scenario.head_of()
        self._tail = dict(zip(scenario.edge_ids, scenario.edge_tail, strict=True))
        self._fftt = scenario.fftt_of()

    # ------------------------------------------------------------- helpers
    def _censored(self, reason: str) -> dict[str, float]:
        logger.info("EDOC plan censored: %s", reason)
        metrics: dict[str, float] = dict.fromkeys(_SCORED, float("nan"))
        metrics["feasible"] = 0.0
        return metrics

    def _valid_route(self, route: tuple[str, ...], origin: str, dest: str) -> bool:
        """A route is a connected walk ``origin -> dest`` of ``<= walk_bound``
        existing edges (so it is in the TD-SP universe, keeping ``c_br <= c_drv``)."""
        if not route or len(route) > self.scenario.walk_bound:
            return False
        if self._tail.get(route[0]) != origin:
            return False
        for a, b in zip(route, route[1:], strict=False):
            if a not in self._head or self._head[a] != self._tail.get(b):
                return False
        last = route[-1]
        return last in self._head and self._head[last] == dest

    def _field_travel(self, field, route: tuple[str, ...], entry: float) -> float:
        """On-network traversal time of ``route`` on the RAW link field from
        ``entry`` (no origin wait) — the ``Ĉ``-evaluated driven cost the
        resolution-floor delta compares to the experienced on-network time."""
        tau = entry
        for e in route:
            tau += field.traversal_time(e, tau)
        return tau - entry

    # ------------------------------------------------------------- certify
    def certify(self, emitted: EmittedBundle) -> dict[str, float]:
        sc = self.scenario

        # G0 pin (config error -> RAISE): provenance must match the instance pin.
        if emitted.engine_version != sc.engine_version or int(emitted.seed) != int(sc.seed):
            raise ValueError(
                f"G0 pin mismatch: emitted {emitted.engine_version!r}/seed {emitted.seed} vs "
                f"instance {sc.engine_version!r}/seed {sc.seed}"
            )

        # G2 demand-match (two-sided bijection; fixed departures = zero timing
        # freedom). Agent set / OD / count EXACT; departures exact up to the
        # engine grid; routes valid walks. A shape/id mismatch CENSORS.
        if set(emitted.plans) != set(self._trip):
            return self._censored("G2: emitted agent set != instance trip table")
        for aid, (route, dep) in emitted.plans.items():
            o, d, want_dep = self._trip[aid]
            # Fixed departures give ZERO timing freedom (adr-036 G2 / forgery pair
            # 10): an honest emission carries the trip-table departure EXACTLY (the
            # engine writes it back at its own grid, ~centisecond for SUMO), so the
            # match is exact-within-tol — NOT a demand-quantum-proportional slack,
            # which would grant +/-0.5*quantum of de-peaking freedom into the field.
            if abs(float(dep) - want_dep) > self.tol:
                return self._censored(f"G2: agent {aid} departure {dep} off grid vs {want_dep}")
            if not self._valid_route(tuple(route), o, d):
                return self._censored(f"G2: agent {aid} route is not a valid {o}->{d} walk")

        # G1 replay fidelity (the A2 analogue). Replaying MODEL plans that
        # crash/timeout = CENSOR (invalid emission, R6 first arm) — the runner
        # signals exactly that with PlanReplayFailure, the ONLY exception censored
        # here. A net-compile / deadline-pre-exhaustion / missing-binary / garbage-
        # artifact / determinism-double failure is certifier-side infrastructure
        # and PROPAGATES (R6 second arm — never laundered into feasible=0).
        try:
            r1 = self.runner(sc, emitted.plans)
            r2 = self.runner(sc, emitted.plans)
        except PlanReplayFailure as exc:
            return self._censored(f"G1: engine crashed/timed out replaying emitted plans: {exc}")
        if not isinstance(r1, ReplayResult):
            raise RuntimeError("G1: replay runner did not return a ReplayResult")
        if r1.canon_hash != r2.canon_hash:
            raise RuntimeError(
                "G1 determinism double disagrees — the pinned replay is non-deterministic "
                "(threading/JDK not pinned?); infrastructure failure, not a censor"
            )
        # replay must reproduce the emitted X exactly (self-report substitution).
        if set(r1.agents) != set(emitted.experienced):
            return self._censored("G1: replay agent set != emitted X")
        for aid, xa in emitted.experienced.items():
            ra = r1.agents.get(aid)
            if ra is None or tuple(ra.route) != tuple(xa.route):
                return self._censored(f"G1: agent {aid} route diverges from replay")
            for f in ("departure", "arrival", "experienced_time", "depart_delay"):
                if abs(float(getattr(ra, f)) - float(getattr(xa, f))) > 1e-6:
                    return self._censored(f"G1: agent {aid} {f} diverges from replay (doctored X)")

        agents = r1.agents

        # G3 two-sided delivery: completion census + departDelay-in-cost + backlog.
        if set(agents) != set(self._trip):
            return self._censored("G3: replay agent set != trip table (demand loss/gain)")
        backlogs = []
        for aid, a in agents.items():
            if not (a.arrival >= a.departure and a.experienced_time >= 0.0):
                return self._censored(f"G3: agent {aid} did not complete (no valid arrival)")
            backlogs.append(float(a.depart_delay))
        max_backlog = max(backlogs, default=0.0)
        mean_backlog = sum(backlogs) / len(backlogs) if backlogs else 0.0
        if max_backlog > sc.backlog_bound + self.tol:
            return self._censored(
                f"G3: max insertion backlog {max_backlog:.3f}s exceeds bound {sc.backlog_bound}s"
            )

        # G4 conservation (C0/C1 shapes on the replay flows): per edge cumulative
        # entered >= left >= 0, so no vehicle leaves an edge it never entered.
        for edge, per_k in r1.flows.items():
            cum_in = cum_out = 0.0
            for k in sorted(per_k):
                ent, left = per_k[k]
                if ent < -self.tol or left < -self.tol:
                    return self._censored(f"G4: negative flow on edge {edge} interval {k}")
                cum_in += float(ent)
                cum_out += float(left)
                if cum_out > cum_in + self.tol:
                    return self._censored(
                        f"G4: edge {edge} interval {k} left>entered cumulatively (non-conserving)"
                    )

        # ---- build the frozen field + per-first-edge origin-wait profiles ----
        field = build_field_from_records(
            r1.field_records, self._fftt, sc.dt, sc.n_intervals, sc.field_semantics
        )
        ow_profile = build_origin_waits(
            [(a.first_edge, a.departure, a.depart_delay) for a in agents.values()],
            sc.dt,
            sc.n_intervals,
        )

        # Resolution-floor gate on the RAW field (aggregation fidelity, MAJOR-3):
        # mean |field-evaluated on-network driven cost - experienced on-network time|.
        deltas = []
        for a in agents.values():
            entry = a.departure + a.depart_delay
            field_travel = self._field_travel(field, tuple(a.route), entry)
            exp_travel = a.experienced_time - a.depart_delay
            deltas.append(abs(field_travel - exp_travel))
        delta = sum(deltas) / len(deltas) if deltas else 0.0
        if delta > sc.floor_seconds + self.tol:
            return self._censored(
                f"resolution floor: field-vs-experienced delta {delta:.3f}s exceeds "
                f"floor {sc.floor_seconds}s (field no longer represents experienced costs)"
            )

        # ---- RG_D1: the frozen-field best-response gap ----
        use_profile = sc.origin_wait_convention == "profile"
        ow = ow_profile if use_profile else _ZERO_WAIT
        sum_drv = sum_br = 0.0
        n_improvers = 0
        covered = total_hops = 0
        for aid, a in agents.items():
            o, d, dep = self._trip[aid]
            route = tuple(a.route)
            travel_drv = evaluate_route(field, ow, route, dep)
            travel_br = td_shortest_path(
                self._out, self._head, field, ow, o, d, dep,
                sc.walk_bound, sc.walk_count_bound,
            )
            # agent-symmetric: the same measured wait cancels in the numerator but
            # pads the denominator (disclosed entrance-choice blindness, MAJOR-2).
            pad = 0.0 if use_profile else float(a.depart_delay)
            c_drv = pad + travel_drv
            c_br = pad + travel_br
            if c_br > c_drv + 1e-6:
                # driven route is in the TD-SP universe, so this cannot happen on
                # valid inputs — an infra bug, not a solution property.
                raise RuntimeError(
                    f"c_br {c_br} > c_drv {c_drv} for agent {aid}: TD-SP universe does not "
                    "contain the driven route (walk_bound too small?)"
                )
            sum_drv += c_drv
            sum_br += c_br
            if c_drv - c_br > 1e-6:
                n_improvers += 1
            # Tier-B BR-path field coverage: share of driven route hops loaded.
            for e in route:
                total_hops += 1
                if field.is_loaded(e):
                    covered += 1
        if sum_drv <= self.tol:
            return self._censored("no scoreable driven cost (degenerate)")
        rg_d1 = (sum_drv - sum_br) / sum_drv
        floor_gap = sc.floor_seconds / (sum_drv / len(agents))
        # AVAILABILITY-based system time (>= 0): total door-to-door driven cost.
        tstt = sum_drv
        br_coverage = covered / total_hops if total_hops else 1.0

        return {
            "feasible": 1.0,
            "rg_d1": float(rg_d1),
            "floor_gap": float(floor_gap),
            "sub_floor": 1.0 if rg_d1 < floor_gap else 0.0,
            "n_improvers": float(n_improvers),
            "tstt": float(tstt),
            "mean_backlog": float(mean_backlog),
            "max_backlog": float(max_backlog),
            "delta": float(delta),
            "br_coverage": float(br_coverage),
        }
