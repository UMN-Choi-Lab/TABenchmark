"""Path-based gradient projection for deterministic user equilibrium.

Jayakrishnan, Tsai, Prashker & Rajadhyaksha (1994): per-OD path sets with
column generation, Newton-scaled flow shifts from each costlier path onto
the cheapest ("basic") path, projected at zero. The Newton denominator sums
cost derivatives over links on exactly ONE of the two paths (Boyles,
Lownes & Unnikrishnan, Transportation Network Analysis, sec. 6.3.1).

Two deliberate deviations from a literal reading of the 1994 paper, both
source-backed and empirically required:

* Costs and derivatives are refreshed after every OD's shifts (Gauss-Seidel,
  as in Boyles' worked example and Perederieieva et al. 2015). Freezing them
  for a whole sweep (Jacobi) diverges on Sioux Falls at alpha = 1.
* Link flows are used incrementally during sweeps but rebuilt exactly from
  path flows before every checkpoint, so emitted flows and stored path flows
  agree bitwise (guards the flow-drift failure mode Perederieieva documents).

Budget accounting matches the FW family: one batched all-origins Dijkstra
per iteration = 1 sp_call; flow-shift sweeps are numpy-only and cost none.
"""

from __future__ import annotations

import time

import numpy as np

from ..core.budget import Budget, BudgetCoords
from ..core.capabilities import Capabilities
from ..core.factors import FactorSpec
from ..core.results import ResultBundle, Trace
from ..core.rng import RngBundle
from ..core.scenario import Scenario
from ._paths import PathEngine
from .base import TrafficAssignmentModel, register_model

__all__ = ["GradientProjectionModel"]


@register_model
class GradientProjectionModel(TrafficAssignmentModel):
    """Path-based gradient projection (Jayakrishnan et al. 1994)."""

    name = "gp"
    capabilities = Capabilities(
        paradigm="static_ue",
        deterministic=True,
        provides_gap=True,
        seedable=True,
    )
    factors = {
        "alpha": FactorSpec(
            default=1.0,
            kind="float",
            bounds=(1e-6, 1.0),
            doc="Newton step scaling (Jayakrishnan et al. recommend 1; smaller "
            "values trade speed for robustness on hard instances).",
        ),
        "inner_iterations": FactorSpec(
            default=4,
            kind="int",
            bounds=(1, 16),
            doc="Flow-shift sweeps over all OD pairs per shortest-path call.",
        ),
    }

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        start = time.perf_counter()
        network = scenario.network
        engine = PathEngine(network)
        od = scenario.demand.matrix
        alpha = self.factor_values["alpha"]
        inner_iterations = self.factor_values["inner_iterations"]
        sp_calls = 0

        # Initialization: AON at empty-network costs, one path per OD.
        first, _ = engine.shortest_paths(
            network.link_cost(np.zeros(network.n_links)), scenario.demand
        )
        sp_calls += 1
        paths = {key: [p] for key, p in first.items()}
        flows = {key: [float(od[key[0], key[1]])] for key in first}

        def aggregate() -> np.ndarray:
            v = np.zeros(network.n_links)
            for key, plist in paths.items():
                for links, h in zip(plist, flows[key], strict=True):
                    v[links] += h
            return v

        v = aggregate()
        k = 0
        while True:
            k += 1
            costs = network.link_cost(v)
            shortest, sptt = engine.shortest_paths(costs, scenario.demand)
            sp_calls += 1
            tstt = float(v @ costs)
            gap = (tstt - sptt) / tstt if tstt > 0 else 0.0
            objective = float(network.link_cost_integral(v).sum())

            coords = BudgetCoords(
                iterations=k,
                sp_calls=sp_calls,
                wall_ms=1000.0 * (time.perf_counter() - start),
            )
            trace.record(
                v, coords, relative_gap=gap, tstt=tstt, sptt=sptt, beckmann=objective
            )

            if budget.exhausted(coords) or budget.target_met(gap):
                break

            # Column generation: add the current shortest path where new.
            for key, new_path in shortest.items():
                known = paths[key]
                if not any(
                    p.shape == new_path.shape and np.array_equal(p, new_path)
                    for p in known
                ):
                    known.append(new_path)
                    flows[key].append(0.0)

            # Newton-projected shifts, Gauss-Seidel over ODs.
            deriv = network.link_cost_derivative(v)
            for _ in range(inner_iterations):
                for key, plist in paths.items():
                    hlist = flows[key]
                    if len(plist) == 1:
                        continue
                    path_costs = [float(costs[p].sum()) for p in plist]
                    basic = int(np.argmin(path_costs))
                    basic_cost = path_costs[basic]
                    changed = False
                    for i in range(len(plist)):
                        if i == basic or hlist[i] <= 0.0:
                            continue
                        excess = path_costs[i] - basic_cost
                        if excess <= 0.0:
                            continue
                        distinct = np.setxor1d(
                            plist[i], plist[basic], assume_unique=True
                        )
                        denom = float(deriv[distinct].sum())
                        if denom <= 0.0:
                            # denom == 0 does NOT imply costs are constant
                            # along the shift direction: BPR links with
                            # power > 1 have zero derivative at v = 0.
                            # Re-probe at the would-be arrival flows; only a
                            # direction still flat there may shift entirely.
                            probe = v.copy()
                            probe[plist[i]] -= hlist[i]
                            probe[plist[basic]] += hlist[i]
                            denom = float(
                                network.link_cost_derivative(probe)[distinct].sum()
                            )
                        shift = (
                            min(hlist[i], alpha * excess / denom)
                            if denom > 0.0
                            else hlist[i]
                        )
                        if shift <= 0.0:
                            continue
                        hlist[i] -= shift
                        hlist[basic] += shift
                        v[plist[i]] -= shift
                        v[plist[basic]] += shift
                        changed = True
                    keep = [
                        i for i in range(len(plist)) if i == basic or hlist[i] > 0.0
                    ]
                    if len(keep) < len(plist):
                        paths[key] = [plist[i] for i in keep]
                        flows[key] = [hlist[i] for i in keep]
                    if changed:
                        costs = network.link_cost(v)
                        deriv = network.link_cost_derivative(v)

            v = aggregate()  # exact resync: emitted flows == path aggregation

        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
