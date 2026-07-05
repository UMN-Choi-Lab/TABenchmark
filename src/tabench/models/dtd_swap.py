"""Smith's (1984) route-swap day-to-day dynamics: the first ``day_to_day`` model.

Where the UE solvers *optimize* to the Wardrop fixed point, this models the
disequilibrium adjustment process that reaches it. Each "day", travelers swap
from costlier to cheaper routes at a rate proportional to (flow on the costlier
route) x (cost excess) -- Smith's proportional-switch adjustment process. Per OD
pair ``w`` with route flows ``h`` and route costs ``c`` (``[y]+`` = max(0, y)):

    h_k(n+1) = h_k(n) + a ( sum_p h_p [c_p - c_k]+  -  h_k sum_p [c_k - c_p]+ ),

the inflow being swaps ONTO route k and the outflow swaps OFF it. Swaps stay
within an OD pair, so demand is conserved structurally (no projection). The
fixed point is exactly Wardrop's first principle -- every used route at the
minimal OD cost -- so the equilibrium link flows are the unique UE flows the
harness already certifies (route flows are non-unique; we validate on link
flows / the certified gap, never on route flows).

Lyapunov / stability (Smith 1984; the distinctive validation). Along the swap
flow the Beckmann objective ``Z(x) = sum_a integral_0^{x_a} t_a`` decreases with
the exact identity ``Zdot = -a * V`` where

    V(h) = sum_w sum_{p,k in R_w} h_p ([c_p - c_k]+)^2  >= 0,  V = 0 iff UE

is Smith's flow-weighted disequilibrium function (the flow weight on the
higher-cost route is essential: without it V would vanish only if *unused* routes
were also equal-cost, which is stricter than Wardrop). We record ``beckmann``
(monotone non-increasing to the UE value the FW/GP solvers report) and the
route-level ``smith_disequilibrium`` V (-> 0) each day as provenance; neither is
scored -- the harness scores only the certified relative gap from link flows.

Discrete-time stability needs a step small enough for *monotone* descent, not
merely non-negative flows: the raw Euler step overshoots into a 2-day limit cycle
around the equilibrium (verified) well before flows go negative. We size the step
in two stages. First, Smith & Wisten's (1995) ``a <= 1/(B M)`` bound (B = largest
route cost, M = route count, scaled by ``step_safety``) as the initial step -- but
that bounds the cost *level*, not its *derivative*, so on high-curvature (BPR
power > 1) congested links a large flow shift can still spike a link cost and make
Z rise. Second, therefore, **Armijo backtracking on the Beckmann objective Z**:
the swap is a descent direction (``Zdot = -a V``), so halving the step until Z
actually decreases is guaranteed to terminate and *guarantees* the monotone
descent the Lyapunov theory promises -- adapting to the curvature the ``B M`` bound
ignores (full step where flat, damped where steep). This is the research-endorsed
"backtrack on Z" stable discretization of Smith's continuous dynamics; the
direction is exactly Smith's, only the step is chosen to descend.

Sourcing. Smith (1984, *Transportation Science* 18(3):245-252) is paywalled and
attributed unread; the swap update and the ``a <= 1/(BM)`` step bound are
cross-verified from open restatements (Peeta & Yang 2003 eqs 5-8; the DPAP /
Beckmann-Lyapunov identity in the NPSD paper, arXiv:1305.5046; Smith & Wisten
1995). The exact 1984 normalization of ``V`` is attributed, not verbatim-quoted;
it is used only as a monotone-decrease diagnostic, so scale is immaterial.

Path/column-generation machinery mirrors ``gp``: per-OD working route sets grown
by one Dijkstra per day, link flows rebuilt exactly from route flows before every
checkpoint. Budget: one batched all-origins Dijkstra per day = one sp_call; the
swap is numpy-only and costs none (matching the FW/GP convention).
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

__all__ = ["RouteSwapDTDModel"]


@register_model
class RouteSwapDTDModel(TrafficAssignmentModel):
    """Smith (1984) proportional route-swap day-to-day dynamical system."""

    name = "dtd-swap"
    capabilities = Capabilities(
        paradigm="day_to_day",
        deterministic=True,
        provides_gap=True,
        seedable=True,
    )
    factors = {
        "swap_rate": FactorSpec(
            default=1.0,
            kind="float",
            bounds=(1e-6, 1e6),
            doc="Smith's swap sensitivity a. The per-day step is adaptively capped "
            "below the over-swapping bound regardless, so a only matters when it is "
            "the binding (smaller) rate; larger a converges faster until the cap "
            "binds.",
        ),
        "step_safety": FactorSpec(
            default=0.5,
            kind="float",
            bounds=(1e-3, 0.99),
            doc="Fraction of the Smith & Wisten (1995) 1/(B M) monotone-descent "
            "bound used for the adaptive step. Smaller is more conservative (slower "
            "but strictly monotone); values near 1 approach the overshoot boundary.",
        ),
        "prune_tol": FactorSpec(
            default=1e-14,
            kind="float",
            bounds=(0.0, 1e-6),
            doc="Route flows below this are treated as zero; zero-flow non-shortest "
            "routes are pruned to bound working-set growth.",
        ),
        "max_backtracks": FactorSpec(
            default=40,
            kind="int",
            bounds=(0, 100),
            doc="Maximum Armijo halvings of the step to enforce monotone Beckmann "
            "descent. 40 halvings reach a step ~1e-12 of the initial; the descent "
            "direction guarantees a decreasing step exists well before that.",
        ),
    }

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        start = time.perf_counter()
        network = scenario.network
        engine = PathEngine(network)
        od = scenario.demand.matrix
        a = self.factor_values["swap_rate"]
        safety = self.factor_values["step_safety"]
        prune_tol = self.factor_values["prune_tol"]
        max_backtracks = self.factor_values["max_backtracks"]
        sp_calls = 0

        # Day 0: all-or-nothing on the free-flow shortest path, one route per OD.
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

            # Route-level Smith disequilibrium V = sum h_p ([c_p - c_k]+)^2, the
            # Lyapunov measure (-> 0 at UE); pure provenance, never scored.
            disequilibrium = 0.0
            for key, plist in paths.items():
                if len(plist) < 2:
                    continue
                c = np.array([float(costs[p].sum()) for p in plist])
                h = np.asarray(flows[key])
                excess = np.maximum(c[:, None] - c[None, :], 0.0)  # E[p,k]=[c_p-c_k]+
                disequilibrium += float(h @ (excess * excess).sum(axis=1))

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
                smith_disequilibrium=disequilibrium,
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

            # First pass: per-OD swap directions (a = 1), and the largest route
            # cost B and route-set size M for the step bound.
            directions: dict[tuple[int, int], np.ndarray] = {}
            b_max = 0.0
            m_max = 0
            for key, plist in paths.items():
                if len(plist) < 2:
                    continue
                c = np.array([float(costs[p].sum()) for p in plist])
                h = np.asarray(flows[key], dtype=np.float64)
                excess = np.maximum(c[:, None] - c[None, :], 0.0)  # E[p,k]=[c_p-c_k]+
                out_rate = excess.sum(axis=1)  # sum_p [c_k - c_p]+ per route k
                inflow = h @ excess  # sum_p h_p [c_p - c_k]+
                directions[key] = inflow - h * out_rate  # a=1 net swap
                b_max = max(b_max, float(c.max()))
                m_max = max(m_max, len(plist))

            # Initial step: the Smith & Wisten (1995) 1/(B M) bound (B = largest
            # route cost, M = route count) scaled by `step_safety`, keeping the
            # user's `a` when smaller. This bounds the cost LEVEL but not its
            # derivative, so on high-curvature (BPR power > 1) congested links it
            # can still overshoot -- so it is only the STARTING step.
            bound = b_max * m_max
            step = a if bound <= 0.0 else min(a, safety / bound)

            # Armijo backtracking on the Beckmann objective Z (the Lyapunov
            # function): the swap direction is a descent direction (Zdot = -a V),
            # so a small enough step always decreases Z; halve the step until it
            # does. This adapts to the cost curvature the B M bound ignores --
            # full step where flat, damped where steep -- and *guarantees* the
            # monotone descent the theory promises (Smith 1984; the research-blessed
            # "backtrack on Z" discretization of the continuous dynamics). Only
            # increasing routes ever exceed their flow, and step only shrinks, so no
            # route goes negative and the aggregate link-flow delta stays exact.
            dv = np.zeros(network.n_links)
            for key, delta in directions.items():
                for links, d in zip(paths[key], delta, strict=True):
                    dv[links] += d
            if directions.__len__():
                z0 = float(network.link_cost_integral(v).sum())
                slack = 1e-12 * max(abs(z0), 1.0)
                for _ in range(max_backtracks):
                    if float(network.link_cost_integral(v + step * dv).sum()) <= z0 + slack:
                        break
                    step *= 0.5

            # Second pass: apply the swap (Jacobi -- all ODs react to today's costs).
            for key, delta in directions.items():
                h = np.asarray(flows[key], dtype=np.float64) + step * delta
                np.maximum(h, 0.0, out=h)  # clamp float dust at the boundary
                flows[key] = list(h)
                # Prune zero-flow routes that are not the current shortest path.
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

            v = aggregate()  # exact resync: emitted flows == route aggregation

        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
