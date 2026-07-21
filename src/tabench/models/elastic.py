"""Elastic (variable) demand user equilibrium via the excess-demand transform.

The demand between each OD pair is a decreasing function of that pair's
equilibrium travel cost, ``d_rs = D_rs(u_rs)`` (docs/design/adr-005). We solve
it as an *ordinary fixed-demand UE on an augmented network* (Gartner 1980;
Sheffi 1985 sec. 6.3; Boyles et al., *Transportation Network Analysis*
sec. 9.1.2): for every OD pair add one direct "excess-demand" arc r->s whose
cost is the inverse demand ``W(e) = D_rs^{-1}(d0 - e)`` (increasing in its own
flow ``e``), fix the total OD demand at the reference ``d0 = D_rs(0)``, and run
Frank & Wolfe (1956). At equilibrium every used r->s path — real or dummy — has
the common cost ``u_rs``, so the dummy flow ``e_rs = d0 - D_rs(u_rs)`` is the
unmet demand and the realized demand is exactly ``D_rs(u_rs)``.

The seminal computational treatment is Florian & Nguyen (1974) (Generalized
Benders Decomposition — a different algorithm from the excess-demand transform
used here; the primary is paywalled/unread, cited for the problem, not the
method). Only real link flows are emitted; the harness reconstructs the
demand-consistent demand ``D(u(v))`` from those flows and certifies both route
equilibrium and demand consistency (P1, adr-005) — the excess-demand arcs are
an internal solver device the certificate never needs to see.
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
from ..core.scenario import ElasticDemand, Network, Scenario
from ._paths import PathEngine
from .base import TrafficAssignmentModel, register_model

__all__ = ["ElasticDemandFWModel"]


@register_model
class ElasticDemandFWModel(TrafficAssignmentModel):
    """Frank-Wolfe on the Gartner excess-demand augmented network.

    Emits real link flows only; the augmented (dummy) arcs are internal. The
    self-monitored ``relative_gap`` is the *real-route* relative gap — the same
    quantity the harness scores (adr-005) — so ``Budget.target_relative_gap``
    stops at a quality comparable to a fixed-demand solver's; the looser
    augmented-network gap is reported separately as ``augmented_relative_gap``.

    Like the other UE solvers this requires every positive-demand OD pair to be
    reachable in the real network; a disconnected instance (which the elastic
    formulation could in principle absorb as fully unmet demand) is out of scope
    for v1 and raises during the shortest-path step.
    """

    name = "fw-elastic"
    capabilities = Capabilities(
        paradigm="static_ue_elastic",
        deterministic=True,
        provides_gap=True,
        seedable=True,
        # solve() raises without scenario.elastic_demand (the decay law).
        inputs_required=frozenset({"od_matrix", "elastic_demand"}),
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
        elastic: ElasticDemand,
        x: np.ndarray,
        dx: np.ndarray,
        d0: np.ndarray,
        e: np.ndarray,
        de: np.ndarray,
        xtol: float,
    ) -> float:
        """Exact augmented-Beckmann step: root of
        ``g(a) = t(x + a dx) . dx + W(e + a de) . de`` on ``[0, 1]``.

        ``g`` is nondecreasing (real link costs and the excess-arc cost ``W``
        are both nondecreasing in their flow), and ``g(0) <= 0`` at a
        non-optimal iterate, so a bracketed Brent root is exact."""

        def g(alpha: float) -> float:
            real = float(network.link_cost(x + alpha * dx) @ dx)
            e_c = np.clip(e + alpha * de, 0.0, d0)
            dummy = float(elastic.excess_arc_cost(d0, e_c) @ de)
            return real + dummy

        if g(0.0) >= 0.0:
            return 0.0
        if g(1.0) <= 0.0:
            return 1.0
        return float(brentq(g, 0.0, 1.0, xtol=xtol))

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        start = time.perf_counter()
        network = scenario.network
        elastic = scenario.elastic_demand
        if elastic is None:
            raise ValueError(
                "fw-elastic requires a scenario with elastic_demand set "
                f"(scenario '{scenario.name}' has none)"
            )
        engine = PathEngine(network)
        xtol = self.factor_values["line_search_xtol"]
        m = network.n_links
        od = scenario.demand.matrix

        # One excess-demand arc per OD pair carrying (reference) demand.
        pairs = [
            (int(o), int(d))
            for o in range(network.n_zones)
            for d in np.nonzero(od[o] > 0)[0]
            if int(d) != o
        ]
        d0 = np.array([od[o, d] for (o, d) in pairs], dtype=np.float64)

        # Feasible, finite start: all demand routed on the real network
        # (excess e = 0). Starting at e = d0 (all demand unmet) is equally
        # feasible but hits the exponential form's D^{-1}(0) singularity.
        x, _ = engine.all_or_nothing(network.link_cost(np.zeros(m)), scenario.demand)
        e = np.zeros(len(pairs))
        sp_calls = 1

        k = 0
        while True:
            k += 1
            t = network.link_cost(x)
            paths, _ = engine.shortest_paths(t, scenario.demand)
            sp_calls += 1
            kappa = np.array([float(t[paths[p]].sum()) for p in pairs])
            w = elastic.excess_arc_cost(d0, e)  # current excess-arc costs

            # Real-route relative gap == the harness-scored metric (adr-005):
            # the excess-demand arcs are an internal device, so we certify and
            # early-stop on the REAL flows against the demand-consistent demand
            # d* = D(kappa). This keeps ``target_relative_gap`` on the same
            # scale as a fixed-demand solver's gap (the augmented gap below,
            # which counts the dummy arcs, is systematically looser). Both are
            # zero exactly at the elastic UE.
            d_star = elastic.realized_demand(d0, kappa)
            real_tstt = float(t @ x)
            gap = (real_tstt - float(kappa @ d_star)) / real_tstt if real_tstt > 0 else 0.0
            aug_tstt = real_tstt + float(w @ e)
            aug_sptt = float(np.minimum(kappa, w) @ d0)
            aug_gap = (aug_tstt - aug_sptt) / aug_tstt if aug_tstt > 0 else 0.0
            realized_total = float((d0 - e).sum())

            coords = BudgetCoords(
                iterations=k,
                sp_calls=sp_calls,
                wall_ms=1000.0 * (time.perf_counter() - start),
            )
            trace.record(
                x,
                coords,
                relative_gap=gap,
                augmented_relative_gap=aug_gap,
                realized_demand=realized_total,
                unmet_demand=float(e.sum()),
                beckmann=float(network.link_cost_integral(x).sum()),
            )

            # Stop on max(real, augmented) gap: the augmented gap is always >= 0
            # (the real-route gap can dip transiently negative while the solver
            # over-routes early, which would trip target_met prematurely), and
            # max >= real guarantees the real gap is <= target at the stop.
            if budget.exhausted(coords) or budget.target_met(max(gap, aug_gap)):
                break

            # All-or-nothing on the augmented network: each OD sends its full
            # d0 to the cheaper of its real shortest path or its excess arc.
            y_x = np.zeros(m)
            e_target = np.zeros(len(pairs))
            for i, p in enumerate(pairs):
                if kappa[i] <= w[i]:
                    y_x[paths[p]] += d0[i]
                else:
                    e_target[i] = d0[i]
            dx = y_x - x
            de = e_target - e
            alpha = self._line_search(network, elastic, x, dx, d0, e, de, xtol)
            if alpha <= 0.0:
                break  # first-order optimal on the augmented network
            x = x + alpha * dx
            e = np.clip(e + alpha * de, 0.0, d0)

        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
