"""Friesz et al. (1994) day-to-day disequilibrium: a route-based PROJECTED
dynamical system whose state is per-OD route flows, evolved by the projection of
the negative route-cost vector onto the demand-feasible set.

Where ``dtd-swap`` (Smith 1984) moves route flows by proportional swaps and
``dtd-link`` (He, Guo & Liu 2010) moves the aggregate link vector by a link-space
projection, Friesz et al. cast day-to-day route adjustment as the projected
dynamical system (PDS)

    h-dot = P_K( h, -c(h) ),

"change route flows at a rate equal to the projection of the NEGATIVE route-cost
vector ``-c(h)`` onto the feasible set ``K``" (the per-OD demand simplices
``K_w = {x >= 0 : sum_{p in R_w} x_p = q_w}``). Its rest points solve the
route-flow variational inequality ``<c(h*), h - h*> >= 0`` for all ``h in K`` --
every used route at the OD-minimum cost, Wardrop's first principle. Because the
route-cost operator ``c(h) = Delta^T t(Delta h)`` is monotone (BPR ``t`` is
non-decreasing), the PDS converges globally; equivalently, since
``partial Z / partial h_p = sum_{a in p} t_a(v) = c_p`` EXACTLY for the Beckmann
objective ``Z(x) = sum_a integral_0^{x_a} t_a`` (``v = Delta h``), the operator
``-c(h)`` is literally ``-grad_h Z`` and the whole scheme is PROJECTED GRADIENT
DESCENT on Beckmann in ROUTE space. The emitted link flows ``v = Delta h*`` are
the unique UE link-flow pattern that fw/cfw/bfw/gp/dtd-swap/dtd-link reach; route
flows are non-unique (the very caveat He-Guo-Liu 2010 / dtd-link moved to link
space to sidestep), so we certify only on link flows / the relative gap.

Discretization (the one genuinely new numeric). The continuous PDS is discretized
by the classic Bertsekas & Gafni (1982) projection step

    h_{k+1} = P_K( h_k - alpha c(h_k) ),

a forward Euler / projected-gradient iteration. Per OD ``w`` this is a EUCLIDEAN
projection onto the scaled simplex ``K_w`` -- the "stay-feasible" step of the PDS,
conserving OD demand (``sum_p x_p = q_w``) and non-negativity EXACTLY. It is
solved in closed form by the sort-threshold algorithm (Held, Wolfe & Crowder
1974; Michelot 1986; Duchi et al. 2008): sort ``y = h - alpha c`` descending
``u_(1) >= ... >= u_(m)``, take ``rho = max{ j : u_(j) - (sum_{i<=j} u_(i) -
q_w)/j > 0 }``, ``tau = (sum_{i<=rho} u_(i) - q_w)/rho``, and ``x_p = max(y_p -
tau, 0)`` (:func:`_project_simplex`). Unlike ``gp``'s Gauss-Seidel pairwise
shifts, ALL routes react to today's FROZEN costs simultaneously (Jacobi) -- the
continuous-PDS reading in which the whole route-flow vector is projected at once.

Step scaling (why the cost step needs one number). The projected-gradient stability
window is ``alpha < 2 / L`` with ``L`` the Lipschitz constant of ``grad Z``; we
make the initial cost step scale-free by normalizing by the local Lipschitz
constant of the Beckmann gradient, ``alpha0 = step_size / max_a t'_a(v)`` (one
numpy derivative call per day, no shortest-path call), so a single
``step_size ~ 1`` sits inside the window on every network (Braess costs O(10),
Sioux Falls costs O(1e-2)) instead of needing per-network tuning -- exactly the
Lipschitz-normalized step ``dtd-link`` uses.

Monotone Beckmann descent (the Lyapunov signature the PDS proves). ``-c(h) =
-grad_h Z`` is a descent direction and the projection ``P_K`` is nonexpansive, so
a small enough ``alpha`` strictly decreases ``Z``; the local Lipschitz step can
still overshoot the Beckmann minimum along the projected ray on high-curvature
(BPR power > 1) congested links, so we reuse ``dtd-swap``/``dtd-link``'s ARMIJO
BACKTRACKING on the Beckmann objective (halve ``alpha`` and re-project until ``Z``
does not increase). Because the direction is a descent direction a decreasing
step always exists, so backtracking terminates and delivers the monotone Lyapunov
descent the continuous PDS proves -- the same "backtrack on Z" stable
discretization the sibling day-to-day models use, only the direction differs
(route-space projected gradient vs route swap vs link-space projection).

Path/column-generation machinery mirrors ``gp``/``dtd-swap``/``dtd-link``: per-OD
working route sets grown by one Dijkstra per day, an optional relaxation ``h <-
h + lambda (x - h)`` blended into the state route flows (a convex combination, so
it stays in ``K`` and, ``Z`` being convex, still descends), and link flows
rebuilt exactly from the route flows before every checkpoint (the flow-drift
guard). Budget: one batched all-origins Dijkstra per day = one sp_call; the
projection and backtracking are numpy-only and cost none (matching the
FW/GP/dtd convention).

Sourcing. Friesz et al. (1994, *Operations Research* 42(6):1120-1136) is
paywalled and attributed unread; the route-based PDS ``h-dot = P_K(h, -c(h))``,
the excess-cost tatonnement disequilibrium ``sum_p h_p (c_p - u_w)``, and the
Bertsekas & Gafni (1982) projection discretization are cross-verified from the
projected-dynamical-systems formulation (Dupuis & Nagurney 1993; Zhang & Nagurney
1996; Nagurney & Zhang 1996) and web-confirmed restatements; the exact
Euclidean-simplex projection is the classic Held-Wolfe-Crowder / Michelot / Duchi
algorithm; the Braess UE (link flows [4,2,2,2,4], route cost 92, Beckmann 386.0)
and the Sioux Falls Beckmann optimum are the repo's pinned constants, none
fabricated.
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

__all__ = ["FrieszDTDModel"]


def _project_simplex(y: np.ndarray, total: float) -> np.ndarray:
    """Exact Euclidean projection of ``y`` onto ``{x >= 0 : sum x = total}``.

    The classic sort-threshold algorithm (Held, Wolfe & Crowder 1974; Michelot
    1986; Duchi et al. 2008): the Bertsekas-Gafni projection step of the Friesz
    PDS onto one OD's scaled demand simplex. ``total = q_w >= 0`` is the OD
    demand, conserved EXACTLY (``sum x = total``) alongside non-negativity.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n == 1:
        return np.array([total], dtype=np.float64)
    u = np.sort(y)[::-1]
    css = np.cumsum(u) - total  # sum_{i<=j} u_(i) - q_w
    j = np.arange(1, n + 1)
    cond = u - css / j > 0.0
    rho = int(np.nonzero(cond)[0][-1])  # max j with the threshold condition
    tau = css[rho] / (rho + 1)
    return np.maximum(y - tau, 0.0)


def _projected_step(
    paths: dict,
    flows: dict,
    route_costs: dict,
    od: np.ndarray,
    n_links: int,
    alpha: float,
) -> tuple[dict, np.ndarray]:
    """One Bertsekas-Gafni step ``x = P_K(h - alpha c)`` over every OD.

    Returns the per-OD projected route flows and the aggregated link vector. The
    per-OD demand ``q_w`` (from ``od``) is conserved EXACTLY by each projection.
    """
    newflows: dict = {}
    x = np.zeros(n_links)
    for key, plist in paths.items():
        h = np.asarray(flows[key], dtype=np.float64)
        q = float(od[key[0], key[1]])  # OD demand q_w, conserved exactly
        xp = _project_simplex(h - alpha * route_costs[key], q)
        newflows[key] = xp
        for links, hp in zip(plist, xp, strict=True):
            x[links] += hp
    return newflows, x


@register_model
class FrieszDTDModel(TrafficAssignmentModel):
    """Friesz et al. (1994) route-based projected-dynamical-system day-to-day model."""

    name = "dtd-friesz"
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
            doc="Cost step of the projected-gradient iteration h_{k+1} = "
            "P_K(h_k - alpha c(h_k)), Lipschitz-normalized as alpha0 = "
            "step_size / max_a t'_a(v) so one value generalizes across networks "
            "(alpha < 2/L is the projected-gradient stability window; step_size ~ 1 "
            "sits inside it). Larger values look further ahead but may overshoot -- "
            "the Armijo backtracking then damps the step to preserve monotone "
            "Beckmann descent.",
        ),
        "adjustment_rate": FactorSpec(
            default=1.0,
            kind="float",
            bounds=(1e-6, 1.0),
            doc="Day-to-day relaxation lambda in (0, 1]: the fraction of the way "
            "from today's route flows toward the projected target x(alpha) taken "
            "each day (h <- h + lambda (x - h)). 1 is the full projected step; "
            "smaller values relax it (slower, still feasible and -- Z being convex "
            "-- still monotone). Parity with dtd-link's adjustment_rate.",
        ),
        "prune_tol": FactorSpec(
            default=1e-14,
            kind="float",
            bounds=(0.0, 1e-6),
            doc="Route flows below this are treated as zero; zero-flow non-shortest "
            "routes are pruned to bound working-set growth (as in gp/dtd-swap/"
            "dtd-link).",
        ),
        "max_backtracks": FactorSpec(
            default=40,
            kind="int",
            bounds=(0, 100),
            doc="Maximum Armijo halvings of the cost step to enforce monotone "
            "Beckmann (Lyapunov) descent. -c(h) is a descent direction and P_K is "
            "nonexpansive, so a decreasing step always exists; 40 halvings reach "
            "~1e-12 of the initial step. 0 disables backtracking (the raw projected "
            "step, which can overshoot Beckmann on high-curvature congested links).",
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
        prune_tol = self.factor_values["prune_tol"]
        max_backtracks = self.factor_values["max_backtracks"]
        sp_calls = 0

        # Day 0: all-or-nothing on the free-flow shortest path, one route per OD
        # (identical to dtd-swap/dtd-link/gp).
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

            # Route-flow excess-cost (tatonnement / VI disequilibrium)
            # G(h) = sum_w sum_{p in R_w} h_p (c_p - u_w), u_w = min_p c_p: the
            # TSTT - SPTT on the working set, -> 0 at UE. Pure provenance.
            excess_cost = 0.0
            for key, plist in paths.items():
                c = np.array([float(costs[p].sum()) for p in plist])
                h = np.asarray(flows[key], dtype=np.float64)
                excess_cost += float((h * (c - c.min())).sum())

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
                excess_cost=excess_cost,
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

            # Frozen-cost route costs c_p = sum_{a in p} t_a(v), the same for all
            # routes today (JACOBI: the continuous-PDS reading -- the whole
            # route-flow vector is projected against today's costs at once, unlike
            # gp's Gauss-Seidel refresh).
            route_costs = {
                key: np.array([float(costs[p].sum()) for p in plist])
                for key, plist in paths.items()
            }

            # Lipschitz-normalized initial cost step alpha0 = step_size / max t'(v)
            # (no shortest-path call); alpha < 2/L is the projected-gradient
            # stability window.
            lipschitz = float(network.link_cost_derivative(v).max())
            alpha0 = step_size / lipschitz if lipschitz > 0.0 else step_size

            # Armijo backtracking on the Beckmann objective Z (the Lyapunov
            # function): -c(h) is a descent direction and P_K is nonexpansive, so
            # halving alpha until Z does not increase terminates and guarantees
            # monotone descent -- damping the overshoot the local Lipschitz step
            # ignores on high-curvature links (the sibling "backtrack on Z"
            # guarantee).
            z0 = float(network.link_cost_integral(v).sum())
            slack = 1e-12 * max(abs(z0), 1.0)
            alpha = alpha0
            newflows, xv = _projected_step(
                paths, flows, route_costs, od, network.n_links, alpha
            )
            for _ in range(max_backtracks):
                if float(network.link_cost_integral(xv).sum()) <= z0 + slack:
                    break
                alpha *= 0.5
                newflows, xv = _projected_step(
                    paths, flows, route_costs, od, network.n_links, alpha
                )

            # Commit with optional relaxation h <- h + lambda (x - h). Both h and x
            # lie in K_w (sum = q_w), so the convex blend conserves demand exactly;
            # Z convex => the blend still descends. Then prune zero-flow non-shortest
            # routes.
            for key in flows:
                h = np.asarray(flows[key], dtype=np.float64)
                h = h + lam * (newflows[key] - h)
                np.maximum(h, 0.0, out=h)  # clamp float dust at the boundary
                flows[key] = list(h)
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
