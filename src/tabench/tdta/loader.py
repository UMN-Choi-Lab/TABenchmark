"""Per-path dynamic network loading for TD route-choice emissions (adr-031).

A thin runner over the DNL sending/receiving loop (the same
:class:`~tabench.dnl.link.LinkModel` S/R contract, the same
:class:`~tabench.dnl.node.TampereNode` junction physics) whose ONLY new code is
per-path first-link injection: each declared path owns a private first link
(guaranteed by the scenario's interior-diverge-free restriction), so its emitted
cumulative departures are injected into that link through a vertical origin queue
with NO multi-out origin split — sidestepping ADR-010's deferred ``OriginNode``
multi-out placeholder rather than reusing it. Interior merges/series nodes reuse
the shipped Tampere node model with the one-hot turns the path set implies.

Because the used-link graph is acyclic (paths branch only at origins, merge
downstream) with positive capacities, every conserving emission clears in finite
time; the runner therefore runs on an EXTENDED horizon (the original grid plus a
clearing pad, zero new departures beyond ``K``) so no vehicle's experienced time
is truncated — the DNL analogue of ``due_gaps``'s exact post-horizon clearing
chord. The emitted aggregate curves are a full-width :class:`DNLOutput`, so the
existing ``dnl_gaps`` C0-C8 certificate is a free correctness oracle on them.
"""

from __future__ import annotations

import numpy as np

from ..dnl.grid import TimeGrid
from ..dnl.node import TampereNode
from ..dnl.output import DNLOutput
from .scenario import TDTAScenario

__all__ = ["PathLoader"]


class PathLoader:
    """Deterministic per-path S/R loading of an emitted departure plan.

    ``departures`` is the ``(n_paths, K)`` emitted flow (aligned with
    ``scenario.paths`` and ``scenario.grid``); ``extra_steps`` extends the
    horizon so in-network vehicles clear. The loader itself never censors — a
    bad-but-honest split is loaded and scored; the evaluator owns clearance
    checking.
    """

    def __init__(
        self, scenario: TDTAScenario, departures: np.ndarray, extra_steps: int = 0
    ) -> None:
        self.scenario = scenario
        self.extra_steps = int(extra_steps)
        net = scenario.network
        dep = np.ascontiguousarray(departures, dtype=np.float64)
        if dep.shape != (scenario.n_paths, scenario.grid.n_steps):
            raise ValueError(
                f"PathLoader departures must have shape "
                f"({scenario.n_paths}, {scenario.grid.n_steps}), got {dep.shape}"
            )
        self.departures = dep
        self._grid = TimeGrid(dt=scenario.grid.dt, n_steps=scenario.grid.n_steps + self.extra_steps)
        self._first_link = scenario.first_link_of()
        # per-path cumulative desired departures on the extended grid (flat after K)
        k = scenario.grid.n_steps
        cum = np.zeros((scenario.n_paths, self._grid.n_steps + 1))
        np.cumsum(dep, axis=1, out=cum[:, 1 : k + 1])
        cum[:, k + 1 :] = cum[:, k : k + 1]  # flat after the original horizon
        self._path_cum = cum

        used = scenario.used_links()
        factory = scenario.link_factory
        self._used = used
        self._models = {
            a: factory(scenario.dynamics.fd(a), float(scenario.dynamics.length[a]), self._grid)
            for a in used
        }
        # interior node structures over USED links only (every link has a model)
        adj: dict[int, dict[int, int]] = {}
        for p in scenario.paths:
            for a, b in zip(p.links, p.links[1:], strict=False):
                adj.setdefault(int(net.term_node[a]), {})[a] = b
        self._nodes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for node in sorted(adj):
            ins = np.array(sorted(adj[node]), dtype=np.int64)
            outs = np.array(sorted({adj[node][int(a)] for a in ins}), dtype=np.int64)
            turns = np.zeros((ins.size, outs.size), dtype=np.float64)
            for i, a in enumerate(ins):
                turns[i, int(np.flatnonzero(outs == adj[node][int(a)])[0])] = 1.0
            self._nodes.append((ins, outs, turns))
        self._node_model = TampereNode()
        # sink links: used links whose head is a (destination) zone
        dests = set(scenario.destinations())
        self._sink_links = np.array(
            [a for a in used if int(net.term_node[a]) in dests], dtype=np.int64
        )
        # per-path origin zone (for aggregate origin_release bookkeeping)
        self._path_origin = np.array([p.origin for p in scenario.paths], dtype=np.int64)

    def run(self) -> DNLOutput:
        """Run the extended S/R loop; emit the full-width aggregate cumulative
        curves as a :class:`DNLOutput` (unused links stay all-zero)."""
        sc = self.scenario
        net = sc.network
        grid = self._grid
        dt = grid.dt
        n_links = net.n_links
        released = np.zeros(sc.n_paths)
        origin_release = np.zeros((net.n_zones, grid.n_steps + 1))
        released_by_zone = np.zeros(net.n_zones)

        for k in range(grid.n_steps):
            sending = np.zeros(n_links)
            receiving = np.zeros(n_links)
            for a, model in self._models.items():
                sending[a] = model.sending(k)
                receiving[a] = model.receiving(k)
            receiving_left = receiving.copy()
            inflow = np.zeros(n_links)
            outflow = np.zeros(n_links)

            # per-path first-link injection through a private vertical queue
            for p in range(sc.n_paths):
                link = int(self._first_link[p])
                waiting = self._path_cum[p, k + 1] - released[p]
                if waiting <= 0.0:
                    continue
                push = min(waiting, float(receiving_left[link]))
                if push <= 0.0:
                    continue
                inflow[link] += push
                receiving_left[link] -= push
                released[p] += push
                released_by_zone[int(self._path_origin[p]) - 1] += push

            # interior merges / series via the shipped Tampere node. Clamp the
            # S/R inputs at 0: link-model float noise (below the node's own 1e-12
            # tolerance) can leave a marginally negative sending/receiving, which
            # the node contract rejects; the clamp changes no feasible allocation.
            for ins, outs, turns in self._nodes:
                caps = sc.dynamics.capacity[ins] * dt
                q = self._node_model.transfer(
                    np.maximum(sending[ins], 0.0),
                    np.maximum(receiving_left[outs], 0.0),
                    turns,
                    caps,
                )
                outflow[ins] += q.sum(axis=1)
                got = q.sum(axis=0)
                inflow[outs] += got
                receiving_left[outs] -= got

            # destination absorption (r = +inf): sink links deliver their sending
            if self._sink_links.size:
                outflow[self._sink_links] += sending[self._sink_links]

            for a, model in self._models.items():
                model.advance(k, float(inflow[a]), float(outflow[a]))
            origin_release[:, k + 1] = released_by_zone

        n_in = np.zeros((n_links, grid.n_steps + 1))
        n_out = np.zeros((n_links, grid.n_steps + 1))
        for a, model in self._models.items():
            n_in[a] = model.cumulative_in
            n_out[a] = model.cumulative_out
        return DNLOutput(
            scenario_hash=sc.content_hash(),
            grid=grid,
            n_in=n_in,
            n_out=n_out,
            origin_release=origin_release,
            loader_version="tdta-path-loader-v1",
        )
