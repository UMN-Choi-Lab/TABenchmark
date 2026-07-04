"""Frank-Wolfe family solvers for deterministic user equilibrium.

The link-based workhorse of four decades of practice and its conjugate-
direction accelerations:

* ``fw``  — Frank & Wolfe (1956); LeBlanc, Morlok & Pierskalla (1975):
  all-or-nothing subproblem + exact line search on the Beckmann objective.
* ``cfw`` / ``bfw`` — conjugate and bi-conjugate direction variants
  (Mitradjieva & Lindberg 2013): the search point is a convex combination of
  the AON solution and previous search points, chosen so the new direction is
  conjugate to the previous one(s) with respect to the diagonal Beckmann
  Hessian at the current flows.

All three share one loop; they differ only in the search point, so certified
comparisons at a fixed shortest-path-call budget isolate the direction rule.
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
from ..core.scenario import Network, Scenario
from ._paths import PathEngine
from .base import TrafficAssignmentModel, register_model

__all__ = ["FrankWolfeModel", "ConjugateFrankWolfeModel", "BiconjugateFrankWolfeModel"]


def _line_search(network: Network, v: np.ndarray, d: np.ndarray, xtol: float) -> float:
    """Exact step: root of g(a) = t(v + a d) . d on [0, 1].

    g is nondecreasing (link costs are nondecreasing in flow and d is fixed),
    g(0) = SPTT - TSTT <= 0; if g(1) <= 0 the full step is optimal.
    """

    def g(alpha: float) -> float:
        return float(network.link_cost(v + alpha * d) @ d)

    g0 = g(0.0)
    if g0 >= 0.0:
        return 0.0
    if g(1.0) <= 0.0:
        return 1.0
    return float(brentq(g, 0.0, 1.0, xtol=xtol))


class _FrankWolfeFamily(TrafficAssignmentModel):
    """Shared loop: AON subproblem -> search point -> exact line search.

    Subclasses override :meth:`_search_point` to choose the point ``s_k``
    toward which the line search moves (plain FW: the AON solution itself)
    and :meth:`_commit` to carry solver state between iterations.
    """

    factors = {
        "line_search_xtol": FactorSpec(
            default=1e-12,
            kind="float",
            bounds=(1e-16, 1e-3),
            doc="Absolute tolerance of the Brent line search on the step size.",
        ),
    }

    #: self-report key names at the trace.record site. Solvers that run the
    #: family on a TRANSFORMED network (e.g. system optimum on marginal
    #: costs) override these so self-reports stay truthfully labeled.
    _SELF_REPORT_KEYS = {
        "relative_gap": "relative_gap",
        "tstt": "tstt",
        "sptt": "sptt",
        "beckmann": "beckmann",
    }

    def _search_point(
        self, network: Network, v: np.ndarray, y: np.ndarray, state: dict
    ) -> np.ndarray:
        raise NotImplementedError

    def _commit(self, state: dict, s: np.ndarray, alpha: float) -> None:
        """Record post-step solver state (previous search points / step size)."""

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        start = time.perf_counter()
        network = scenario.network
        engine = PathEngine(network)
        xtol = self.factor_values["line_search_xtol"]
        sp_calls = 0
        state: dict = {}

        v, _ = engine.all_or_nothing(
            network.link_cost(np.zeros(network.n_links)), scenario.demand
        )
        sp_calls += 1

        k = 0
        while True:
            k += 1
            costs = network.link_cost(v)
            y, sptt = engine.all_or_nothing(costs, scenario.demand)
            sp_calls += 1
            tstt = float(v @ costs)
            gap = (tstt - sptt) / tstt if tstt > 0 else 0.0
            objective = float(network.link_cost_integral(v).sum())

            coords = BudgetCoords(
                iterations=k,
                sp_calls=sp_calls,
                wall_ms=1000.0 * (time.perf_counter() - start),
            )
            keys = self._SELF_REPORT_KEYS
            trace.record(
                v,
                coords,
                **{
                    keys["relative_gap"]: gap,
                    keys["tstt"]: tstt,
                    keys["sptt"]: sptt,
                    keys["beckmann"]: objective,
                },
            )

            if budget.exhausted(coords) or budget.target_met(gap):
                break
            s = self._search_point(network, v, y, state)
            d = s - v
            alpha = _line_search(network, v, d, xtol)
            if alpha <= 0.0 and s is not y:
                # Degenerate conjugate direction: restart with a plain FW step.
                state.clear()
                s = y
                d = y - v
                alpha = _line_search(network, v, d, xtol)
            if alpha <= 0.0:
                break  # first-order optimal at current point
            self._commit(state, s, alpha)
            v = v + alpha * d

        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )


@register_model
class FrankWolfeModel(_FrankWolfeFamily):
    """Link-based Frank-Wolfe with exact line search."""

    name = "fw"
    capabilities = Capabilities(
        paradigm="static_ue",
        deterministic=True,
        provides_gap=True,
        seedable=True,
    )

    def _search_point(
        self, network: Network, v: np.ndarray, y: np.ndarray, state: dict
    ) -> np.ndarray:
        return y


def _safe_ratio(numer: float, denom: float) -> float:
    """numer/denom with ~0 and non-finite denominators mapped to 0."""
    if not np.isfinite(numer) or not np.isfinite(denom) or abs(denom) <= 1e-30:
        return 0.0
    ratio = numer / denom
    return float(ratio) if np.isfinite(ratio) else 0.0


@register_model
class ConjugateFrankWolfeModel(_FrankWolfeFamily):
    """Conjugate-direction Frank-Wolfe (CFW, Mitradjieva & Lindberg 2013).

    The search point is ``s_k = a s_{k-1} + (1-a) y_k`` with ``a`` chosen so
    the new direction is H-conjugate to the previous one, where H is the
    diagonal Beckmann Hessian t'(v) at the current flows (paper eqs 2-6).
    A step of ~1 lands on the search point and restarts the method (FW step
    next); a = 0 recovers plain FW, so every safeguard degrades gracefully.
    """

    name = "cfw"
    capabilities = Capabilities(
        paradigm="static_ue",
        deterministic=True,
        provides_gap=True,
        seedable=True,
    )

    #: a = 1 would reproduce the just-optimized previous direction (paper
    #: eq 6 clamps to [0, 1 - delta]; delta 1e-4 as in TrafficAssignment.jl)
    _DELTA = 1e-4
    #: a full step collapses s_{k-1} onto v; restart from a plain FW step
    _RESTART_TOL = 1e-6

    def _conjugacy_coefficient(
        self, network: Network, v: np.ndarray, s_prev: np.ndarray, y: np.ndarray
    ) -> float:
        h = network.link_cost_derivative(v)
        d_bar = s_prev - v
        d_fw = y - v
        numer = float(d_bar @ (h * d_fw))
        denom = float(d_bar @ (h * (d_fw - d_bar)))
        return min(max(_safe_ratio(numer, denom), 0.0), 1.0 - self._DELTA)

    def _search_point(
        self, network: Network, v: np.ndarray, y: np.ndarray, state: dict
    ) -> np.ndarray:
        s_prev = state.get("s_prev")
        if s_prev is None:
            return y  # warm-up (or post-restart): plain FW step
        a = self._conjugacy_coefficient(network, v, s_prev, y)
        return a * s_prev + (1.0 - a) * y

    def _commit(self, state: dict, s: np.ndarray, alpha: float) -> None:
        if alpha >= 1.0 - self._RESTART_TOL:
            state.clear()
        else:
            state["s_prev"] = s


@register_model
class BiconjugateFrankWolfeModel(ConjugateFrankWolfeModel):
    """Bi-conjugate Frank-Wolfe (BFW, Mitradjieva & Lindberg 2013).

    The search point combines the AON solution with the previous TWO search
    points so the new direction is H-conjugate to both previous directions
    (paper eqs 8-9 and Appendix A). Ramps FW -> CFW -> BFW on warm-up and
    after every restart, which also guarantees the stored step size is < 1.
    """

    name = "bfw"
    capabilities = Capabilities(
        paradigm="static_ue",
        deterministic=True,
        provides_gap=True,
        seedable=True,
    )

    def _search_point(
        self, network: Network, v: np.ndarray, y: np.ndarray, state: dict
    ) -> np.ndarray:
        s_prev = state.get("s_prev")
        s_prev2 = state.get("s_prev2")
        if s_prev is None:
            return y  # warm-up step 1: plain FW
        if s_prev2 is None:
            a = self._conjugacy_coefficient(network, v, s_prev, y)
            return a * s_prev + (1.0 - a) * y  # warm-up step 2: CFW

        tau = state["alpha_prev"]  # step that produced v; < 1 by restart rule
        h = network.link_cost_derivative(v)
        d_fw = y - v
        # Residual previous directions at the current point (Appendix A):
        # d_bar  prop d_{k-1};  d_bbar prop d_{k-2} mapped through the last step.
        d_bar = s_prev - v
        d_bbar = tau * s_prev + (1.0 - tau) * s_prev2 - v

        mu = max(
            0.0,
            -_safe_ratio(
                float(d_bbar @ (h * d_fw)), float(d_bbar @ (h * (s_prev2 - s_prev)))
            ),
        )
        nu = max(
            0.0,
            -_safe_ratio(float(d_bar @ (h * d_fw)), float(d_bar @ (h * d_bar)))
            + mu * tau / (1.0 - tau),
        )
        beta0 = 1.0 / (1.0 + mu + nu)
        beta1 = nu * beta0
        beta2 = mu * beta0
        return beta0 * y + beta1 * s_prev + beta2 * s_prev2

    def _commit(self, state: dict, s: np.ndarray, alpha: float) -> None:
        if alpha >= 1.0 - self._RESTART_TOL:
            state.clear()  # re-enter the FW -> CFW -> BFW ramp
        else:
            state["s_prev2"] = state.get("s_prev")
            state["s_prev"] = s
            state["alpha_prev"] = alpha
