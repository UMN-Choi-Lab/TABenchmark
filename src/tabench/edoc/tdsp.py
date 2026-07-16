"""Certifier-owned time-dependent shortest path (ruling R3 + the R2/MAJOR-3
non-FIFO ruling of adr-036).

Design: docs/design/adr-036-external-dynamic-observational-certificate.md.

The best response ``c_br`` is a certifier-owned TD-SP over the FULL network — the
counterfactual the certifier supplies so hidden cheap paths score AGAINST the
plan (forgery pair 3). Two properties are load-bearing:

* **Waiting-not-allowed** (R2, matching the engine routers): at every node the
  vehicle enters an outgoing edge immediately at its arrival time; only the ORIGIN
  incurs an off-network origin wait, and only on the first edge (MAJOR-2). A walk
  that returns to the origin leaves again wait-free (it is already inserted).

* **Non-FIFO soundness** (MAJOR-3): on a non-FIFO frozen field a later entry can
  arrive earlier, so a label-*setting* Dijkstra that keeps one earliest-arrival
  label per node is UNSOUND (it prunes the boundary-crossing path and deflates
  ``RG_D1`` — measured 380 s vs the true 115 s). This family pins the ``raw`` +
  label-correcting semantics, realized here as an **explicit walk universe**:
  every walk from origin to destination of length ``<= walk_bound`` (a hashed
  instance field) is enumerated with the same per-edge composition, and the
  minimum door-to-door time is taken. The driven route is a walk of length
  ``<= walk_bound`` (gated at construction), so it is IN the universe and
  ``c_br <= c_drv`` holds by construction. ``walk_count_bound`` is a hard safety
  cap enforced HERE, at certification time: the DFS counts every walk it pops and
  RAISES a ``RuntimeError`` (a certifier-side **infrastructure** guard — adr-036
  R6, never a censor) the instant the count exceeds the bound, so a pathologically
  dense net aborts promptly and bounded instead of looping. There is deliberately
  **no** construction-time DFS pre-count (it would double the enumeration cost for
  every scenario to guard a case the certify-time cap already bounds); the pop
  counter is prompt (measured sub-0.1 s to the raise on a dense K7).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .field import FrozenField, OriginWaitProfile

INF = float("inf")


def evaluate_route(
    field: FrozenField,
    origin_waits: OriginWaitProfile,
    route: Sequence[str],
    depart_time: float,
) -> float:
    """Door-to-door cost of the DRIVEN ``route`` from scheduled departure
    ``depart_time``: the first edge's origin-wait profile plus the time-dependent
    traversal of every edge on the frozen field (the same composition ``c_br``
    minimizes over, so the driven route is in the min set)."""
    if not route:
        raise ValueError("route must contain at least one edge")
    e0 = route[0]
    tau = depart_time + origin_waits.wait(e0, depart_time)
    tau += field.traversal_time(e0, tau)
    for e in route[1:]:
        tau += field.traversal_time(e, tau)
    return tau - depart_time


def td_shortest_path(
    out_edges: Mapping[str, Sequence[str]],
    edge_head: Mapping[str, str],
    field: FrozenField,
    origin_waits: OriginWaitProfile,
    origin: str,
    dest: str,
    depart_time: float,
    walk_bound: int,
    walk_count_bound: int = 200_000,
) -> float:
    """Minimum door-to-door cost from ``origin`` to ``dest`` for a departure at
    ``depart_time``, over every walk of ``<= walk_bound`` edges (waiting-not-
    allowed; origin wait on the first edge only). Returns ``inf`` if the
    destination is unreachable within the bound. Raises ``RuntimeError`` if the
    walk count exceeds ``walk_count_bound`` — a certify-time infrastructure guard
    (adr-036 R6, prompt + bounded, never a censor), not a solution property; there
    is no construction-time pre-count."""
    best = INF
    walks = 0
    # Seed hop 1: leave the origin via each outgoing edge, charging that edge's
    # own origin wait, then its frozen traversal.
    stack: list[tuple[str, float, int]] = []
    for e in out_edges.get(origin, ()):
        w = origin_waits.wait(e, depart_time)
        entry = depart_time + w
        arr = entry + field.traversal_time(e, entry)
        stack.append((edge_head[e], arr, 1))
    while stack:
        node, t, hops = stack.pop()
        walks += 1
        if walks > walk_count_bound:
            raise RuntimeError(
                f"TD-SP walk count exceeded {walk_count_bound} on OD {origin}->{dest}"
                f" at walk_bound={walk_bound}; raise walk_count_bound or lower "
                "walk_bound for this net (a certify-time infrastructure guard, R6)"
            )
        if node == dest:
            # Destination is absorbing: a vehicle that arrives is done (do not
            # extend past it).
            if t < best:
                best = t
            continue
        if hops >= walk_bound:
            continue
        for e in out_edges.get(node, ()):  # wait-free on-network hops
            arr = t + field.traversal_time(e, t)
            stack.append((edge_head[e], arr, hops + 1))
    if best is INF:
        return INF
    return best - depart_time
