"""Evans (1976) combined trip distribution + assignment.

The demand is *endogenous*: only the trip-end margins are fixed — productions
``O_i`` and attractions ``D_j`` — and the OD matrix ``d_ij`` is distributed by a
doubly-constrained gravity model at the equilibrium travel costs. The two
conditions couple (Evans 1976; Sheffi, *Urban Transportation Networks* 1985
ch. 6; Boyles, Lownes & Unnikrishnan, *Transportation Network Analysis* §6):

1. **Route equilibrium** given demand ``d`` — the ordinary Wardrop condition.
2. **Distribution consistency** — ``d_ij = A_i B_j exp(-β u_ij)`` (doubly
   constrained gravity) with ``u_ij`` the equilibrium min OD cost.

Both fall out of the single convex program

    min_{x,d}  Σ_a ∫_0^{x_a} t_a(w) dw  +  (1/β) Σ_ij d_ij (ln d_ij − 1)
    s.t.  Σ_j d_ij = O_i,  Σ_i d_ij = D_j,  d ≥ 0,  x = assign(d).

We solve it with **Evans' partial-linearization Frank-Wolfe**: linearize only
the assignment (Beckmann) term, keep the entropy term exact. At iterate
``(x, d)`` the subproblem

    min_y  Σ_ij u_ij y_ij + (1/β) Σ_ij y_ij(ln y_ij − 1)   s.t. margins

has the closed-form solution ``y = gravity(O, D, β, u)`` (the doubly-constrained
Furness balancing on ``CombinedDemand``); its all-or-nothing assignment ``w``
gives the descent direction, and an exact Brent line search on the combined
objective sets the step. ``x`` is kept a feasible assignment of ``d`` throughout
(``x_0 = AON(gravity at free-flow)`` and every update is a convex combination),
so the route-equilibrium gap of ``d`` stays ``≥ 0``.

Only real link flows are emitted; the harness recomputes ``d* = gravity(u(v))``
from those flows and certifies route equilibrium and demand consistency (P1,
adr-007) — exactly the elastic-demand recipe (adr-005) with the gravity in
place of the pointwise decay law ``D(u)``. Every positive-margin OD pair must be
reachable in the network (a disconnected instance raises during the shortest
path step, as for the other UE solvers).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import brentq

from ..core.budget import Budget, BudgetCoords
from ..core.capabilities import Capabilities
from ..core.factors import FactorSpec
from ..core.results import ResultBundle, Trace
from ..core.rng import RngBundle
from ..core.scenario import CombinedDemand, Demand, Network, Scenario
from ._paths import PathEngine
from .base import TrafficAssignmentModel, register_model

__all__ = ["EvansCombinedModel"]


@register_model
class EvansCombinedModel(TrafficAssignmentModel):
    """Evans (1976) partial-linearization Frank-Wolfe for the combined
    distribution + assignment problem.

    Requires a scenario with ``combined_demand`` set (the fixed trip-end margins
    and gravity dispersion ``β``). The self-monitored ``relative_gap`` is the
    combined route-equilibrium gap the harness scores (adr-007); the solver
    early-stops on ``max(route-gap, distribution-gap)``, both nonnegative, so
    ``Budget.target_relative_gap`` never trips on the transiently-negative
    combined gap a naive stop would.
    """

    name = "evans"
    capabilities = Capabilities(
        paradigm="static_ue_combined",
        deterministic=True,
        provides_gap=True,
        seedable=True,
        # solve() raises without scenario.combined_demand (the fixed margins).
        inputs_required=frozenset({"od_matrix", "combined_demand"}),
    )

    factors = {
        "line_search_xtol": FactorSpec(
            default=1e-12,
            kind="float",
            bounds=(1e-16, 1e-3),
            doc="Absolute tolerance of the Brent line search on the step size.",
        ),
    }

    @staticmethod
    def _line_search(
        network: Network,
        combined: CombinedDemand,
        x: np.ndarray,
        dx: np.ndarray,
        d: np.ndarray,
        dd: np.ndarray,
        support: np.ndarray,
        xtol: float,
    ) -> float:
        """Exact combined-objective step: root of

            g(a) = t(x + a dx) . dx + (1/β) Σ_ij ln(d_ij + a dd_ij) dd_ij

        on ``[0, 1]`` (sum over the gravity support). ``g`` is nondecreasing
        (link costs increase in flow and ``Σ dd²/(d + a dd) ≥ 0``), and
        ``g(0) ≤ 0`` at a non-optimal iterate, so a bracketed Brent root is
        exact. The auxiliary gravity demand is strictly positive on the support,
        so ``d + a dd = (1 - a) d + a y > 0`` for ``a ∈ (0, 1]`` and the log is
        finite (the floor only guards the ``a = 0`` corner)."""
        beta = combined.beta
        dd_s = dd[support]

        def g(alpha: float) -> float:
            real = float(network.link_cost(x + alpha * dx) @ dx)
            arg = (d + alpha * dd)[support]
            ent = float(np.sum(np.log(np.maximum(arg, 1e-300)) * dd_s) / beta)
            return real + ent

        if g(0.0) >= 0.0:
            return 0.0
        if g(1.0) <= 0.0:
            return 1.0
        return float(brentq(g, 0.0, 1.0, xtol=xtol))

    @staticmethod
    def _skim_and_load(
        t: np.ndarray,
        paths: dict[tuple[int, int], np.ndarray],
        demand: np.ndarray | None,
        n_zones: int,
        n_links: int,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """From one shortest-path tree return the OD-cost skim ``u`` and, if
        ``demand`` is given, its all-or-nothing link flows on those paths."""
        u = np.zeros((n_zones, n_zones), dtype=np.float64)
        flows = None if demand is None else np.zeros(n_links, dtype=np.float64)
        for (i, j), p in paths.items():
            u[i, j] = float(t[p].sum())
            if flows is not None:
                flows[p] += demand[i, j]
        return u, flows

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        start = time.perf_counter()
        network = scenario.network
        combined = scenario.combined_demand
        if combined is None:
            raise ValueError(
                "evans requires a scenario with combined_demand set "
                f"(scenario '{scenario.name}' has none)"
            )
        engine = PathEngine(network)
        xtol = self.factor_values["line_search_xtol"]
        m = network.n_links
        nz = network.n_zones
        support = combined.support()
        total = combined.total

        # Skim/assign are driven by the gravity support (the fixed margins),
        # independent of the reference matrix, so a reference entry that
        # underflowed to zero can never desync the solver from the certificate.
        support_demand = Demand(support.astype(np.float64))

        # Feasible start: gravity distribution at free-flow costs, assigned
        # all-or-nothing. x then routes d, an invariant every update preserves
        # (both x and d advance by the same step to convex combinations).
        t = network.link_cost(np.zeros(m))
        paths, _ = engine.shortest_paths(t, support_demand)
        sp_calls = 1
        u, _ = self._skim_and_load(t, paths, None, nz, m)
        d = combined.gravity(u)
        _, x = self._skim_and_load(t, paths, d, nz, m)

        k = 0
        while True:
            k += 1
            t = network.link_cost(x)
            paths, _ = engine.shortest_paths(t, support_demand)
            sp_calls += 1
            u, _ = self._skim_and_load(t, paths, None, nz, m)
            # Auxiliary = certificate demand: y = gravity(u) is both the FW
            # subproblem solution AND the demand the harness recomputes (adr-007).
            y = combined.gravity(u)
            _, w = self._skim_and_load(t, paths, y, nz, m)

            tstt = float(t @ x)
            # Route-equilibrium gap of the CURRENT demand d (x routes d, so this
            # is a standard nonnegative UE gap), and the harness-scored combined
            # gap against the gravity demand d* = y (may dip transiently negative
            # since x routes d, not y — hence it is reported, not stopped on).
            route_sptt = float((u * d).sum())
            route_gap = (tstt - route_sptt) / tstt if tstt > 0 else 0.0
            combined_sptt = float((u * y).sum())
            combined_gap = (tstt - combined_sptt) / tstt if tstt > 0 else 0.0
            # Distribution consistency: L1 distance of the current demand from
            # the gravity demand at current costs, per trip. Controls the
            # harness demand-consistency (node_balance) residual.
            dist_gap = float(np.abs(d - y).sum() / total) if total > 0 else 0.0
            stop_gap = max(route_gap, dist_gap)

            coords = BudgetCoords(
                iterations=k,
                sp_calls=sp_calls,
                wall_ms=1000.0 * (time.perf_counter() - start),
            )
            trace.record(
                x,
                coords,
                relative_gap=combined_gap,
                route_relative_gap=route_gap,
                distribution_gap=dist_gap,
                realized_demand=float(y.sum()),
                beckmann=float(network.link_cost_integral(x).sum()),
            )

            if budget.exhausted(coords) or budget.target_met(stop_gap):
                break

            dx = w - x
            dd = y - d
            alpha = self._line_search(network, combined, x, dx, d, dd, support, xtol)
            if alpha <= 0.0:
                break  # first-order optimal on the combined program
            x = x + alpha * dx
            d = d + alpha * dd

        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
