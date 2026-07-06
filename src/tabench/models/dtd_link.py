"""He, Guo & Liu (2010) link-based day-to-day dynamics: a link-flow adjustment
process whose STATE is the aggregate link-flow vector, not per-OD route flows.

Where ``dtd-swap`` (Smith 1984) evolves *route* flows -- inheriting the
non-uniqueness of the path-flow representation -- He, Guo & Liu define the
adjustment directly on the polytope ``Omega`` of feasible LINK flows. The
continuous rational-behavior-adjustment / projected dynamical system
(Dupuis-Nagurney 1993; He et al. 2010) is

    dv/dt = x*(v) - v,   x*(v) = argmin_{x in Omega} <t(v), x> + (1/2)||x - v||^2
                                = Proj_Omega( v - t(v) ),

the FROZEN-cost Beckmann-type (proximal) target: the feasible link-flow pattern
"closest to today's" that is optimal at today's costs ``t(v)``. Because the
projection is nonexpansive, ``x*`` is a CONTINUOUS function of ``v`` (the whole
point vs the discontinuous all-or-nothing target), so the trajectory and its
rest point are well-defined in link space. The discrete update

    v_{k+1} = v_k + lambda_k (x*(v_k) - v_k),   lambda_k in (0, 1],

keeps ``v_{k+1}`` in the convex set ``Omega`` (both ``x*`` and ``v_k`` are in
``Omega``): the invariance principle -- the emitted link flows never leave the
OD-feasible set. The rest point ``x*(v*) = v*`` is exactly Wardrop UE (the VI
``<t(v*), x - v*> >= 0`` for all ``x in Omega``), so the equilibrium is the same
unique UE link-flow pattern that fw/cfw/bfw/gp/dtd-swap reach; route flows are
non-unique, so we certify only on link flows / the relative gap.

Implementable discretization (the one nontrivial piece). Exact link-space
projection onto the multicommodity polytope ``Omega`` is a network QP whose LP
oracle would need NEGATIVE-cost shortest paths (``v - t(v)`` is not a positive
cost). We instead compute the proximal target the way ``gp``/``dtd-swap`` already
compute UE-type subproblems -- column-generated per-OD working path sets grown by
one Dijkstra on the POSITIVE costs ``t(v)`` -- so no negative-cost oracle is ever
needed. With ``t(v)`` FROZEN, the proximal objective restricted to the working
set is minimized by per-OD pairwise flow shifts toward the cheapest working path,
each with a CLOSED FORM: shifting ``delta`` from path ``i`` onto the basic path
``j`` changes the proximal objective as a 1-D quadratic whose curvature is
exactly the number of links on which the two paths differ (the symmetric
difference ``|D|``) -- the proximal ``(1/2)||x-v||^2`` term contributes 1 per
distinct link and the frozen ``<a t, x>`` term is linear -- so the exact shift is

    delta* = ( C_i - C_j ) / |D|,   C_p = sum_{a' in p} ( a t_{a'} + (x_{a'} - v_{a'}) ),

clamped to ``[0, g_i]``. This is the direct proximal analogue of gp's
Newton shift, but the denominator is the distinct-link COUNT (the proximal
Hessian along the shift direction) rather than gp's sum of cost derivatives,
because the costs ``t`` are held fixed while the target is built. numpy-only,
zero shortest-path calls.

Step scaling (why the literal unit metric needs one number). The paper's unit
metric ``(1/2)||x-v||^2`` gives ``x*(v) = Proj_Omega(v - t(v))``, but when link
costs are O(10) (BPR at congestion) the ``<t,x>`` term dwarfs the proximal term
and the projected target COLLAPSES to all-or-nothing -- a discontinuous, badly
scaled direction that limit-cycles at ``lambda = 1`` and stalls under damping.
The standard Dupuis-Nagurney Euler cure is a cost step ``a`` in the projection,
``x*(v) = Proj_Omega(v - a t(v))`` (equivalently the proximal weight becomes
``1/(2a)``). We make ``a`` scale-free by normalizing by the local Lipschitz
constant of the Beckmann gradient, ``a = step_size / max_a t'_a(v)`` (one extra
numpy derivative call per day, no shortest-path call): then a single
``step_size ~ 1`` converges on every network (Braess costs O(10), Sioux Falls
costs O(1e-2)) instead of needing per-network tuning -- ``a < 2 / L`` is the
projected-gradient stability window and ``step_size = 1`` sits well inside it.

Monotone Beckmann descent (the RBAP / Lyapunov signature). The direction
``x*(v) - v`` is a Beckmann DESCENT direction: because ``v`` is itself feasible,
``<t(v), x*> + (1/2)||x* - v||^2 <= <t(v), v>``, so
``<t(v), x* - v> <= -(1/2)||x* - v||^2 <= 0`` (zero iff ``x* = v`` iff UE). The
unit proximal metric (``|D|`` curvature) ignores the true BPR cost curvature,
so a full step ``lambda = 1`` can still overshoot the Beckmann minimum along the
ray on high-curvature (BPR power > 1) congested links; we therefore reuse
``dtd-swap``'s Armijo backtracking ON THE BECKMANN OBJECTIVE (halve the step
until ``Z`` does not increase), which is guaranteed to terminate on a descent
direction and delivers the monotone Lyapunov descent He et al. prove for the
continuous system. This is the ``dtd-link`` analogue of ``dtd-swap``'s
"backtrack on Z" stable discretization; only the target/direction differs
(link-space projection vs route swap).

Path/column-generation machinery mirrors ``gp``/``dtd-swap``: per-OD working sets
grown by one Dijkstra per day, target path flows blended into the state path
flows (a linear per-path convex combination, so the aggregate exactly realizes
``v <- v + lambda(x* - v)``), and link flows rebuilt exactly from the state path
flows before every checkpoint (the flow-drift guard). Budget: one batched
all-origins Dijkstra per day = one sp_call; the proximal target and the step are
numpy-only and cost none (matching the FW/GP/dtd-swap convention).

Sourcing. He, Guo & Liu (2010, *Transportation Research Part B* 44(4):597-608)
is paywalled and attributed unread; the link-based projected dynamic ``dv/dt =
x*(v) - v`` with the Beckmann-type proximal target and its Lyapunov (Beckmann)
argument are cross-verified from the projected-dynamical-systems formulation
(Dupuis & Nagurney 1993; Nagurney & Zhang 1996) and the Beckmann convex program
(Boyles, Lownes & Unnikrishnan, *Transportation Network Analysis* ch. 4-5). The
proximal closed-form shift is re-derived here from the frozen-cost objective; the
Braess UE, route cost 92, and the Sioux Falls Beckmann optimum are the repo's
pinned constants, none fabricated.
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

__all__ = ["LinkBasedDTDModel"]


@register_model
class LinkBasedDTDModel(TrafficAssignmentModel):
    """He, Guo & Liu (2010) link-based day-to-day projected dynamical system."""

    name = "dtd-link"
    capabilities = Capabilities(
        paradigm="day_to_day",
        deterministic=True,
        provides_gap=True,
        seedable=True,
    )
    factors = {
        "step_size": FactorSpec(
            default=1.0,
            kind="float",
            bounds=(1e-6, 1e3),
            doc="Cost step of the projected target x*(v) = Proj_Omega(v - a t(v)), "
            "Lipschitz-normalized as a = step_size / max_a t'_a(v) so one value "
            "generalizes across networks (a < 2/L is the projected-gradient "
            "stability window; step_size ~ 1 sits inside it). Larger values look "
            "further ahead but may overshoot -- the Armijo backtracking then damps "
            "the day step to preserve monotone Beckmann descent.",
        ),
        "adjustment_rate": FactorSpec(
            default=1.0,
            kind="float",
            bounds=(1e-6, 1.0),
            doc="Day-to-day adjustment rate lambda in (0, 1]: the fraction of the "
            "way from today's link flows toward the proximal target x*(v) taken "
            "each day. 1 is the full projected step (nonexpansive, stays in Omega); "
            "smaller values relax the adjustment (slower, still monotone). The step "
            "is further capped by Armijo backtracking for guaranteed Beckmann "
            "descent, so lambda only matters when it is the binding (smaller) rate.",
        ),
        "inner_sweeps": FactorSpec(
            default=8,
            kind="int",
            bounds=(1, 32),
            doc="Per-day proximal flow-shift sweeps over all OD pairs used to solve "
            "the FROZEN-cost target x*(v) on the working path sets. The proximal "
            "objective is strictly convex, so a handful of sweeps closely realizes "
            "the link-space projection; costs zero shortest-path calls.",
        ),
        "prune_tol": FactorSpec(
            default=1e-14,
            kind="float",
            bounds=(0.0, 1e-6),
            doc="Path flows below this are treated as zero; zero-flow non-shortest "
            "columns are pruned to bound working-set growth (as in gp/dtd-swap).",
        ),
        "max_backtracks": FactorSpec(
            default=40,
            kind="int",
            bounds=(0, 100),
            doc="Maximum Armijo halvings of the day step to enforce monotone "
            "Beckmann (Lyapunov) descent. x*(v) - v is a descent direction, so a "
            "decreasing step always exists; 40 halvings reach ~1e-12 of the initial "
            "step. 0 disables backtracking (the raw projected step, which can "
            "overshoot Beckmann on high-curvature congested links).",
        ),
    }

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        start = time.perf_counter()
        network = scenario.network
        engine = PathEngine(network)
        od = scenario.demand.matrix
        step_size = self.factor_values["step_size"]
        lam = self.factor_values["adjustment_rate"]
        inner_sweeps = self.factor_values["inner_sweeps"]
        prune_tol = self.factor_values["prune_tol"]
        max_backtracks = self.factor_values["max_backtracks"]
        sp_calls = 0

        # Day 0: all-or-nothing on the free-flow shortest path, one route per OD.
        # The state is the link-flow vector v; per-OD path flows are bookkeeping
        # that aggregate to v exactly (so v stays in Omega by construction).
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

            # Proximal target x*(v) = Proj_Omega(v - a t(v)) = argmin_{x in Omega}
            # <a t(v), x> + (1/2)||x-v||^2, with t = t(v) FROZEN and the cost step
            # a = step_size / max_a t'_a(v) (Lipschitz-normalized so one step_size
            # generalizes across networks; one numpy derivative call, no sp_call).
            # Solved on the working path sets by per-OD pairwise shifts toward the
            # cheapest working path; each shift has the closed form delta =
            # (C_i - C_basic)/|distinct| (curvature = number of links the two paths
            # differ on -- the proximal Hessian along the shift, the frozen scaled
            # cost being linear). Target path flows `gflows` start from the state
            # path flows and never touch `flows`.
            lipschitz = float(network.link_cost_derivative(v).max())
            a = step_size / lipschitz if lipschitz > 0.0 else step_size
            scaled = a * costs
            gflows = {key: list(hlist) for key, hlist in flows.items()}
            x = v.copy()  # target link flows, incrementally updated
            for _ in range(inner_sweeps):
                for key, plist in paths.items():
                    if len(plist) < 2:
                        continue
                    glist = gflows[key]
                    # Proximal path cost C_p = sum_{a' in p} (a t_a' + x_a' - v_a'),
                    # recomputed per OD from the live target x (Gauss-Seidel).
                    pc = [float((scaled[p] + x[p] - v[p]).sum()) for p in plist]
                    basic = int(np.argmin(pc))
                    basic_cost = pc[basic]
                    for i in range(len(plist)):
                        if i == basic or glist[i] <= 0.0:
                            continue
                        reduced = pc[i] - basic_cost
                        if reduced <= 0.0:
                            continue
                        distinct = np.setxor1d(
                            plist[i], plist[basic], assume_unique=True
                        )
                        denom = float(distinct.size)  # proximal curvature = |D|
                        shift = (
                            min(glist[i], reduced / denom) if denom > 0.0 else glist[i]
                        )
                        if shift <= 0.0:
                            continue
                        glist[i] -= shift
                        glist[basic] += shift
                        x[plist[i]] -= shift
                        x[plist[basic]] += shift

            # Day step v <- v + lambda (x* - v). x* - v is a Beckmann descent
            # direction, so Armijo backtracking on the Beckmann objective (the
            # Lyapunov function) is guaranteed to find a monotone-descent step --
            # damping the overshoot the unit proximal metric ignores on
            # high-curvature links (the same "backtrack on Z" guarantee dtd-swap
            # uses). Blending the target path flows into the state path flows by
            # the SAME per-path lambda keeps the aggregate exact: v_new =
            # (1-step) v + step x*, still in Omega.
            dv = x - v
            z0 = float(network.link_cost_integral(v).sum())
            slack = 1e-12 * max(abs(z0), 1.0)
            step = lam
            for _ in range(max_backtracks):
                if float(network.link_cost_integral(v + step * dv).sum()) <= z0 + slack:
                    break
                step *= 0.5

            for key in flows:
                h = np.asarray(flows[key], dtype=np.float64)
                g = np.asarray(gflows[key], dtype=np.float64)
                h = h + step * (g - h)
                np.maximum(h, 0.0, out=h)  # clamp float dust at the boundary
                flows[key] = list(h)
                # Prune zero-flow columns that are not the current shortest path.
                plist = paths[key]
                sp = shortest.get(key)
                keep = [
                    i
                    for i in range(len(plist))
                    if flows[key][i] > prune_tol
                    or (
                        sp is not None
                        and plist[i].shape == sp.shape
                        and np.array_equal(plist[i], sp)
                    )
                ]
                if 0 < len(keep) < len(plist):
                    paths[key] = [plist[i] for i in keep]
                    flows[key] = [flows[key][i] for i in keep]

            v = aggregate()  # exact resync: emitted flows == working-set aggregation

        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
