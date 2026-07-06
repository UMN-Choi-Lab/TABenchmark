"""Boundedly-rational user equilibrium (Mahmassani & Chang 1987): band-relaxed
gradient projection.

A boundedly-rational traveler does not switch routes to save less than an
indifference band ``epsilon`` (in native cost units). A flow is an ``epsilon``-BRUE
iff, for every OD pair, every *used* route lies within the band of that OD's
minimum route cost: ``h_pi > 0  =>  c_pi <= kappa_rs + epsilon`` (Mahmassani &
Chang 1987; the rigorous static set formulation is Di, Liu, Pang & Ban 2013,
Transportation Research Part B 57:300-313). This relaxes Wardrop's *equality*
(``epsilon=0`` recovers UE, where every used route is exactly minimal) to a
one-sided band, so BR-UE is a **set** of flows, not a point, and the link flows
are non-unique (there is no Beckmann convex program).

We find one acceptable flow with a **band-thresholded gradient projection**
(reusing the ``gp`` skeleton -- per-OD path sets, column generation, exact
route->link resync): from a pinned free-flow all-or-nothing start, for each OD
pair shift flow off every route whose cost exceeds the OD minimum by more than the
band, onto the cheapest ("basic") route, by a Newton step sized to reduce that
route's excess to *exactly* the band ``epsilon`` -- not to zero. So the process
stops at the **band EDGE** (used-route excess ~ ``epsilon``), a genuine BR-UE that
is stable under bounded-rational behaviour, NOT a UE solver stopped early (whose
used-route excess is ~ 0). Routes within the band are never touched (the swap
incentive is boundedly-rational). Because the shift is a projected Newton step
(the ``gp`` denominator over links on exactly one of the two routes), convergence
is fast -- one step per OD on a linear network -- unlike a proportional route swap,
which drains low-flow out-of-band routes only at a rate proportional to their
(vanishing) flow.

Path-dependence/hysteresis is real (different starts land on different band edges);
the pinned free-flow-AON start makes the emitted flow deterministic and
reproducible. Emitted link flows are certified by the harness: the ordinary
fixed-demand relative gap plus ``br_acceptable = (AEC <= epsilon)``, a NECESSARY-
not-sufficient band check (adr-008) -- link flows cannot see the per-OD
concentration a fully sufficient check needs. Budget: one batched Dijkstra per
iteration = one sp_call (the FW/GP convention).

Sourcing. Mahmassani & Chang (1987, Transportation Science 21(2):89-99) is
paywalled and attributed unread (a behavioural paper). The static ``epsilon``-BRUE
condition, its non-uniqueness, and the ``epsilon``-monotone acceptable set are from
Di-Liu-Pang-Ban (2013) and cross-verified in Boyles, Lownes & Unnikrishnan,
*Transportation Network Analysis* ch. 5 (used-path band condition; "no convex
program; link-flow uniqueness not guaranteed"). No numeric result from the primary
is claimed.
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

__all__ = ["BoundedlyRationalUEModel"]


@register_model
class BoundedlyRationalUEModel(TrafficAssignmentModel):
    """Mahmassani & Chang (1987) BR-UE via band-thresholded gradient projection."""

    name = "br-ue"
    capabilities = Capabilities(
        paradigm="static_br_ue",
        deterministic=True,
        provides_gap=True,
        seedable=True,
    )
    factors = {
        "alpha": FactorSpec(
            default=1.0,
            kind="float",
            bounds=(1e-6, 1.0),
            doc="Newton step scaling (as in gp); 1 targets the band edge exactly on "
            "a linear network, smaller trades speed for robustness.",
        ),
        "inner_sweeps": FactorSpec(
            default=4,
            kind="int",
            bounds=(1, 16),
            doc="Band-thresholded flow-shift sweeps over all OD pairs per "
            "shortest-path call.",
        ),
    }

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        start = time.perf_counter()
        network = scenario.network
        eps = scenario.br_epsilon
        if eps is None:
            raise ValueError(
                "br-ue requires a scenario with br_epsilon set (the indifference "
                f"band); scenario '{scenario.name}' has none"
            )
        engine = PathEngine(network)
        od = scenario.demand.matrix
        alpha = self.factor_values["alpha"]
        inner_sweeps = self.factor_values["inner_sweeps"]
        sp_calls = 0

        # Day 0: free-flow all-or-nothing (pinned deterministic start).
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

            # BR rest measure: the largest per-OD used-route excess above the OD
            # minimum (kappa = the true shortest-path cost, which may be a route
            # not yet in the working set). <= epsilon means every used route is in
            # the band -- a genuine epsilon-BRUE (the disaggregate check; the
            # harness scores the aggregate AEC <= epsilon necessary condition).
            band_excess = 0.0
            for key, plist in paths.items():
                kappa = float(costs[shortest[key]].sum())
                for links, h in zip(plist, flows[key], strict=True):
                    if h > 0.0:
                        band_excess = max(band_excess, float(costs[links].sum()) - kappa)

            coords = BudgetCoords(
                iterations=k,
                sp_calls=sp_calls,
                wall_ms=1000.0 * (time.perf_counter() - start),
            )
            trace.record(
                v,
                coords,
                relative_gap=gap,
                tstt=tstt,
                sptt=sptt,
                beckmann=float(network.link_cost_integral(v).sum()),
                band_excess=band_excess,
            )
            if band_excess <= eps or budget.exhausted(coords) or budget.target_met(gap):
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

            # Band-thresholded Newton shifts, Gauss-Seidel over ODs. Only routes
            # whose excess exceeds the band are shifted, and only enough to bring
            # the excess DOWN TO epsilon (the band edge), never to zero.
            deriv = network.link_cost_derivative(v)
            for _ in range(inner_sweeps):
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
                        # Reduce the excess only by the amount OVER the band, so
                        # the route settles at cost = kappa + epsilon (band edge).
                        over_band = path_costs[i] - basic_cost - eps
                        if over_band <= 0.0:
                            continue
                        distinct = np.setxor1d(plist[i], plist[basic], assume_unique=True)
                        denom = float(deriv[distinct].sum())
                        if denom <= 0.0:
                            probe = v.copy()
                            probe[plist[i]] -= hlist[i]
                            probe[plist[basic]] += hlist[i]
                            denom = float(
                                network.link_cost_derivative(probe)[distinct].sum()
                            )
                        shift = (
                            min(hlist[i], alpha * over_band / denom)
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
                    keep = [i for i in range(len(plist)) if i == basic or hlist[i] > 0.0]
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
