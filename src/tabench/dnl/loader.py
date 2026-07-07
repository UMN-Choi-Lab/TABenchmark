"""Deterministic sending/receiving runner for DNL link models."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .link import LinkModel, LinkModelFactory
from .node import NodeModel, OriginNode, SeriesNode
from .output import DNLOutput
from .scenario import DynamicScenario

__all__ = ["NetworkLoader"]


class NetworkLoader:
    """Deterministic S/R time loop over links and simple boundary nodes.

    The default runner handles origin boundaries, destination absorption, and
    1-in/1-out interior series nodes. Real merges/diverges require an explicit
    ``node_models`` entry; the core refuses to guess junction physics that belong
    to the later node-model sprint.
    """

    def __init__(
        self,
        scenario: DynamicScenario,
        link_factory: LinkModelFactory,
        node_models: Mapping[int, NodeModel] | None = None,
    ) -> None:
        self.scenario = scenario
        self.link_factory = link_factory
        self.node_models = dict(node_models or {})
        net = scenario.network
        self._in_links = tuple(np.flatnonzero(net.term_node == n) for n in range(net.n_nodes + 1))
        self._out_links = tuple(
            np.flatnonzero(net.init_node == n) for n in range(net.n_nodes + 1)
        )
        self.links: list[LinkModel] = [
            link_factory(scenario.dynamics.fd(a), float(scenario.dynamics.length[a]), scenario.grid)
            for a in range(net.n_links)
        ]
        self._turns = dict(scenario.turns.frac) if scenario.turns is not None else {}
        self._origin_node = OriginNode()
        self._interior_models: dict[int, NodeModel] = {}
        for node in range(net.n_zones + 1, net.n_nodes + 1):
            n_in = self._in_links[node].size
            n_out = self._out_links[node].size
            if n_in == 0 or n_out == 0:
                continue
            supplied = self.node_models.get(node)
            if supplied is not None:
                self._interior_models[node] = supplied
            elif n_in == 1 and n_out == 1:
                self._interior_models[node] = SeriesNode()
            else:
                raise ValueError(
                    f"NetworkLoader needs an explicit NodeModel for node {node} "
                    f"(n_in={n_in}, n_out={n_out}); dnl-core only defaults 1-in/1-out "
                    "series nodes"
                )

    def _origin_turns(self, node: int) -> np.ndarray:
        outs = self._out_links[node]
        if outs.size == 0:
            return np.zeros((1, 0), dtype=np.float64)
        if outs.size != 1:
            raise ValueError(
                f"NetworkLoader origin zone {node} has {outs.size} outgoing links; "
                "origin split policies are deferred beyond dnl-core"
            )
        return np.ones((1, 1), dtype=np.float64)

    def _interior_turns(self, node: int) -> np.ndarray:
        n_in = self._in_links[node].size
        n_out = self._out_links[node].size
        if node in self._turns:
            return self._turns[node]
        if n_out == 1:
            return np.ones((n_in, 1), dtype=np.float64)
        raise ValueError(f"missing turning fractions for interior node {node}")

    def run(self) -> DNLOutput:
        """Run the deterministic grid loop and emit canonical cumulative curves."""
        sc = self.scenario
        net = sc.network
        grid = sc.grid
        n_links = net.n_links
        release = np.zeros((net.n_zones, grid.n_steps + 1), dtype=np.float64)
        released = np.zeros(net.n_zones, dtype=np.float64)
        demand_edges = sc.demand.cumulative(grid.edges).sum(axis=2).T

        for k in range(grid.n_steps):
            sending = np.array([link.sending(k) for link in self.links], dtype=np.float64)
            receiving = np.array([link.receiving(k) for link in self.links], dtype=np.float64)
            receiving_left = receiving.copy()
            inflow = np.zeros(n_links, dtype=np.float64)
            outflow = np.zeros(n_links, dtype=np.float64)

            for zone in range(1, net.n_zones + 1):
                outs = self._out_links[zone]
                if outs.size == 0:
                    continue
                waiting = max(0.0, float(demand_edges[zone - 1, k + 1] - released[zone - 1]))
                q = self._origin_node.transfer(
                    np.array([waiting]),
                    receiving_left[outs],
                    self._origin_turns(zone),
                    np.array([np.inf]),
                )
                pushed = q[0]
                inflow[outs] += pushed
                receiving_left[outs] -= pushed
                released[zone - 1] += float(pushed.sum())

            for node in range(net.n_zones + 1, net.n_nodes + 1):
                model = self._interior_models.get(node)
                if model is None:
                    continue
                ins = self._in_links[node]
                outs = self._out_links[node]
                q = model.transfer(
                    sending[ins],
                    receiving_left[outs],
                    self._interior_turns(node),
                    sc.dynamics.capacity[ins] * grid.dt,
                )
                outflow[ins] += q.sum(axis=1)
                inflow[outs] += q.sum(axis=0)
                receiving_left[outs] -= q.sum(axis=0)

            sink = np.flatnonzero(net.term_node <= net.n_zones)
            outflow[sink] += sending[sink]

            for a, link in enumerate(self.links):
                link.advance(k, float(inflow[a]), float(outflow[a]))
            release[:, k + 1] = released

        return DNLOutput(
            scenario_hash=sc.content_hash(),
            grid=grid,
            n_in=np.vstack([link.cumulative_in for link in self.links]),
            n_out=np.vstack([link.cumulative_out for link in self.links]),
            origin_release=release,
        )
