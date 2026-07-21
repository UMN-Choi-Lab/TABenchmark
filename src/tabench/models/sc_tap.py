"""Side-constrained traffic assignment (Larsson & Patriksson 1995): UE under hard
link-capacity constraints, by an augmented-Lagrangian dual.

Ordinary UE minimizes the Beckmann objective; SC-TAP adds hard **side constraints**
``v_a <= u_a`` (per-link physical throughput caps, distinct from the BPR reference
``Network.capacity``). Its KKT conditions are a Wardrop equilibrium on the
capacity-**augmented** cost

    c_a(v) = t_a(v_a) + beta_a,   beta_a >= 0,   beta_a (u_a - v_a) = 0,

i.e. each link carries a multiplier ``beta_a`` that is zero unless its capacity
binds and, where it binds, is the **queueing delay / congestion toll** that stops
travelers piling onto the (physically cheap but full) link -- the extra
generalized cost that equalizes used-route costs (Larsson & Patriksson 1995/1999;
the multiplier-as-toll reading is verbatim in the 1999 companion).

We solve it by the **method of multipliers** (augmented Lagrangian): the inner
problem, for fixed ``(beta, rho)``, is an ordinary UE with the modified, still
non-decreasing link cost

    t~_a(v) = t_a(v_a) + max{0, beta_a + rho (v_a - u_a)},

solved by Frank-Wolfe (exact Brent line search on the augmented Beckmann
objective); the outer loop updates ``beta_a <- max{0, beta_a + rho (v_a - u_a)}``
and grows ``rho`` when the worst capacity violation stops shrinking. At the fixed
point the constraints hold exactly (``v_a <= u_a``) and the recovered ``beta_a`` is
the true multiplier -- unlike a fixed large-penalty solve, which only satisfies the
constraints in the limit ``rho -> infinity``. When no capacity binds SC-TAP reduces
*exactly* to plain UE (same convex program), so it matches the shipped FW solver.

The harness certifies from the emitted link flows (adr-009): **capacity
feasibility ``v_a <= u_a``** is link-visible and checked PER LINK to a tight
relative tolerance (the scored SC quantity), and the raw-cost relative gap is
reported for provenance -- it is *positive* at a correct SC equilibrium (binding
links carry flow that would prefer to grow), so it is not the acceptance criterion.
This certifies capacity FEASIBILITY, not the full SC equilibrium: the model
self-reports the recovered ``beta_a`` and the augmented-cost gap, and a fully
harness-recomputed augmented-cost equilibrium gap (recovering ``beta`` as shadow
prices on the binding set by a small convex program) is a documented enhancement.

Convergence STOP (deliberate ``Budget.target_relative_gap`` bypass). Because that
raw relative gap is *positive* at a correct SC equilibrium (above), sc-tap does NOT
honor a caller ``Budget.target_relative_gap`` -- a target on the raw gap would be
met spuriously or never. The outer loop's convergence test is instead the
augmented-cost inner gap plus capacity feasibility (``aug_gap <= 1e-10`` and
``violation <= feas_tol``); absent that, it runs to the ``Budget`` iteration/wall
envelope. (br-ue documents an analogous bypass; sc-tap's had been left implicit.)

Sourcing. Larsson & Patriksson (1995, Transportation Research Part B 29(6):433-455)
is paywalled and attributed unread; the augmented-cost equilibrium, the
multiplier-as-toll interpretation, and the augmented-Lagrangian form are
cross-verified from the 1999 companion, Nie-Zhang-Lee (2004), and standard
augmented-Lagrangian theory (Bertsekas 1982). The analytic anchor numbers are
derived here, not quoted.
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
from ..core.scenario import Scenario
from ._paths import PathEngine
from .base import TrafficAssignmentModel, register_model

__all__ = ["SideConstrainedModel"]


@register_model
class SideConstrainedModel(TrafficAssignmentModel):
    """Larsson & Patriksson (1995) side-constrained UE via augmented Lagrangian."""

    name = "sc-tap"
    capabilities = Capabilities(
        paradigm="static_sc_ue",
        deterministic=True,
        provides_gap=True,
        seedable=True,
    )
    factors = {
        "rho0": FactorSpec(
            default=1.0,
            kind="float",
            bounds=(1e-6, 1e6),
            doc="Initial augmented-Lagrangian penalty rho; grown by rho_growth when "
            "the worst capacity violation stops shrinking (method of multipliers).",
        ),
        "rho_growth": FactorSpec(
            default=4.0,
            kind="float",
            bounds=(1.0, 100.0),
            doc="Factor rho is multiplied by when feasibility stalls (Bertsekas).",
        ),
        "inner_iters": FactorSpec(
            default=20,
            kind="int",
            bounds=(1, 200),
            doc="Frank-Wolfe steps on the augmented cost per multiplier update.",
        ),
        "feas_tol": FactorSpec(
            default=1e-9,
            kind="float",
            bounds=(1e-14, 1e-2),
            doc="Capacity-violation tolerance max(v_a - u_a)+ for the outer stop.",
        ),
        "line_search_xtol": FactorSpec(
            default=1e-13,
            kind="float",
            bounds=(1e-16, 1e-3),
            doc="Absolute tolerance of the Brent line search on the step size.",
        ),
    }

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        start = time.perf_counter()
        network = scenario.network
        u = scenario.side_capacities
        if u is None:
            raise ValueError(
                "sc-tap requires a scenario with side_capacities set (per-link hard "
                f"capacities); scenario '{scenario.name}' has none"
            )
        engine = PathEngine(network)
        rho = self.factor_values["rho0"]
        rho_growth = self.factor_values["rho_growth"]
        inner_iters = self.factor_values["inner_iters"]
        feas_tol = self.factor_values["feas_tol"]
        line_search_xtol = self.factor_values["line_search_xtol"]
        m = network.n_links

        def aug_cost(w: np.ndarray, beta: np.ndarray, r: float) -> np.ndarray:
            # t~_a(w) = t_a(w) + max{0, beta_a + rho (w_a - u_a)}: the true BPR
            # latency plus the augmented-Lagrangian penalty gradient. Strictly
            # positive (link_cost > 0), so shortest paths are well-defined.
            return network.link_cost(w) + np.maximum(0.0, beta + r * (w - u))

        beta = np.zeros(m)
        # Feasible finite start: AON at free-flow augmented cost (beta=0).
        v, _ = engine.all_or_nothing(aug_cost(np.zeros(m), beta, rho), scenario.demand)
        sp_calls = 1

        prev_violation = np.inf
        k = 0
        while True:
            k += 1
            # Inner Frank-Wolfe on the augmented cost (fixed beta, rho). On an
            # INFEASIBLE-capacity instance the penalties can distort costs enough
            # that a shortest-path step fails (a zone becomes unreachable): stop
            # gracefully and emit the current flow, which the certificate then
            # reports as capacity-infeasible, rather than crashing.
            try:
                for _ in range(inner_iters):
                    t_aug = aug_cost(v, beta, rho)
                    y, _ = engine.all_or_nothing(t_aug, scenario.demand)
                    sp_calls += 1
                    dx = y - v

                    def g(a: float, _dx=dx, _v=v, _beta=beta, _rho=rho) -> float:
                        return float(aug_cost(_v + a * _dx, _beta, _rho) @ _dx)

                    if g(0.0) >= 0.0:
                        break  # augmented UE reached for this (beta, rho)
                    alpha = (
                        1.0 if g(1.0) <= 0.0 else float(brentq(g, 0.0, 1.0, xtol=line_search_xtol))
                    )
                    if alpha <= 0.0:
                        break
                    v = v + alpha * dx
            except (RuntimeError, ValueError):
                break

            # Multiplier update and penalty growth (method of multipliers). Cap
            # beta at 1e8 -- far above any real queueing toll for the benchmark's
            # cost scales, so it never binds on a realistic feasible instance. On an
            # INFEASIBLE one (a capacity below a cut link's forced flow, where no SC
            # solution exists) it stops beta diverging to overflow / distorting the
            # graph so a zone becomes unreachable -- the solver then stops with the
            # constraint reported violated instead of crashing. If a feasible
            # instance genuinely needed beta > 1e8 (astronomical), the cap would
            # bind on the SAFE side: feasibility under-reported, never over-reported
            # (adversarial-review MINOR 2).
            beta = np.minimum(np.maximum(0.0, beta + rho * (v - u)), 1e8)
            violation = float(np.maximum(v - u, 0.0).max())
            if violation > 0.25 * prev_violation and violation > feas_tol:
                rho = min(rho * rho_growth, 1e10)  # cap: an INFEASIBLE instance (a
                #   capacity below a cut link's forced flow) has no SC solution, so
                #   beta/rho would otherwise diverge; cap them and stop gracefully
                #   with the constraint reported violated rather than overflowing.
            prev_violation = violation
            if not np.all(np.isfinite(beta)) or not np.all(
                np.isfinite(aug_cost(v, beta, rho))
            ):
                break  # penalties diverged (likely an infeasible-capacity instance)

            # Scored raw-cost gap (positive at a binding SC equilibrium) + the
            # augmented-cost inner gap and the recovered multipliers (provenance).
            raw = network.link_cost(v)
            _, sptt = engine.all_or_nothing(raw, scenario.demand)
            sp_calls += 1
            tstt = float(v @ raw)
            gap = (tstt - sptt) / tstt if tstt > 0 else 0.0
            t_aug = aug_cost(v, beta, rho)
            _, aug_sptt = engine.all_or_nothing(t_aug, scenario.demand)
            sp_calls += 1
            aug_tstt = float(v @ t_aug)
            aug_gap = (aug_tstt - aug_sptt) / aug_tstt if aug_tstt > 0 else 0.0

            coords = BudgetCoords(
                iterations=k,
                sp_calls=sp_calls,
                wall_ms=1000.0 * (time.perf_counter() - start),
            )
            trace.record(
                v,
                coords,
                relative_gap=gap,
                augmented_relative_gap=aug_gap,
                max_capacity_violation=violation,
                max_multiplier=float(beta.max()),
                beckmann=float(network.link_cost_integral(v).sum()),
            )
            # Converged: constraints satisfied and the augmented UE is tight.
            if budget.exhausted(coords) or (violation <= feas_tol and aug_gap <= 1e-10):
                break

        # Guard: if the very first inner solve failed before any checkpoint (only
        # on pathological input, e.g. costs overflowing to non-finite), still emit
        # the start flow instead of returning an empty trace (adversarial-review
        # MINOR 1) -- the certificate scores it and reports the constraint state.
        if len(trace) == 0:
            trace.record(
                v,
                BudgetCoords(
                    iterations=k, sp_calls=sp_calls, wall_ms=1000.0 * (time.perf_counter() - start)
                ),
                relative_gap=float("nan"),
                # Key parity with the normal branch: emit augmented_relative_gap so a
                # consumer reading the union of self_report keys does not see it vanish
                # on pathological input. This guard is entered precisely because the
                # augmented-cost all-or-nothing was not computable (its costs went
                # non-finite / non-positive -- the same reason beckmann is NaN here),
                # so the value is the not-computable NaN sentinel, not an invented gap.
                augmented_relative_gap=float("nan"),
                max_capacity_violation=float(np.maximum(v - u, 0.0).max()),
                max_multiplier=float(beta.max()),
                beckmann=float("nan"),
            )

        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
