"""Multiclass-user traffic assignment (Dafermos 1972) by diagonalization.

Ordinary UE routes ONE class of travelers. Dafermos (1972) generalizes it to
``K`` user classes that share the physical network but perceive class-specific,
mutually-coupled link costs. Each class ``i`` routes its own demand ``g^i`` to a
Wardrop equilibrium in ITS cost, but every class's cost depends on the joint
flow, so the equilibria are simultaneous. This ships the linear-coupling case

    t_a^i(V) = t_a^BPR(v_a) + sum_j M_ij v_a^j,
    v_a = sum_j v_a^j (total link flow),   M = scenario.multiclass.interaction,

with per-class flows ``v_a^j`` and a ``(K, K)`` class-interaction ``M`` applied
per link. Stacking the class-indexed flows ``V = (v^1, ..., v^K)`` this is the
block-structured single-class asymmetric VI

    find V* in K = K_1 x ... x K_K  s.t.  <T(V*), V - V*> >= 0  for all V in K,

i.e. exactly ``vi-asym`` promoted one level up (``K = 1`` recovers it): the
feasible set is a PRODUCT of per-class demand polytopes, so routing is done per
class, and the cost operator is affine with the block interaction ``M``. A
SYMMETRIC ``M`` is the integrable case (the equilibrium minimizes a convex
multiclass-Beckmann potential; Dafermos 1972); an ASYMMETRIC ``M`` is a genuine
VI with no equivalent optimization (Smith 1979; Dafermos 1980), reachable by no
Beckmann / Frank-Wolfe minimizer.

Solver -- multiclass diagonalization (nonlinear Gauss-Seidel). Sweep the
classes; for class ``i`` freeze the other classes' flows, which makes class
``i``'s cost separable in its own flow

    c_a^i(w) = t_a^BPR(w_a + o_a) + M_ii w_a + sum_{j != i} M_ij v_a^j,
    o_a = sum_{j != i} v_a^j  (frozen other-class total on link a),

an ordinary single-class UE that Frank-Wolfe (exact Brent line search on the
diagonalized cost) solves; relax ``v^i <- v^i + step (v^i_inner - v^i)`` and use
each class's updated flow immediately (Gauss-Seidel). At a fixed point every
class is at Wardrop equilibrium in the TRUE coupled cost, i.e. ``V`` solves the
VI. Convergence follows the Dafermos (1982) / Florian & Spiess (1982) diagonal-
dominance condition (own-class sensitivity dominates the cross-class coupling),
NOT from monotonicity alone; a coupling that drives a cost non-positive stops
the sweep and emits a flow the certificate then CENSORS (never a false accept).
The outer loop stops on EITHER convergence channel: the caller-facing
``Budget.target_relative_gap`` (via ``budget.target_met``, parity with the
sibling solvers) OR the model-owned ``target_gap`` factor, whichever the
class-summed VI relative gap satisfies first (in addition to the budget resource
axes).

Certificate (P1; adr-013). Each checkpoint emits the per-class link flows
``V`` as a first-class object (``FlowState.class_link_flows``), and the harness
recomputes the class-summed VI residual from ``V`` -- the product feasible set
makes the VI gap decompose into per-class all-or-nothing minima,

    gap = (sum_i <t^i(V), v^i> - sum_i min_{y^i in K_i} <t^i(V), y^i>) / sum_i <t^i(V), v^i>,

so it is the ordinary relative gap summed over classes, evaluated at the coupled
cost. It is 0 iff ``V`` solves the multiclass VI -- a sound, harness-recomputed
residual, not a self-report. ``beckmann_objective`` is NaN (no potential for an
asymmetric ``M``). The per-class feasibility audit checks each class conserves
its own demand.

Sourcing. Dafermos (1972, *Transportation Science* 6(1):73-87,
``dafermos1972traffic``) is the multiclass model and the symmetry/integrability
condition; Smith (1979, ``smith1979existence``) and Dafermos (1980,
``dafermos1980traffic``) are the asymmetric-VI characterization, and the
diagonalization / relaxation convergence is Dafermos (1982) / Florian & Spiess
(1982). The block-VI reading and the analytic anchor are hand-derived here
(cross-verified against an independent multiclass diagonalization); no number
from any paper is reproduced.
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
from ..core.scenario import Demand, Scenario
from ._paths import PathEngine
from .base import TrafficAssignmentModel, register_model

__all__ = ["MulticlassModel"]


@register_model
class MulticlassModel(TrafficAssignmentModel):
    """Dafermos (1972) multiclass-user UE with a linear class interaction, by
    diagonalization."""

    name = "multiclass"
    capabilities = Capabilities(
        paradigm="static_ue_multiclass",
        deterministic=True,
        provides_gap=True,
        seedable=True,
        # solve() raises without scenario.multiclass; it emits per-class flows.
        inputs_required=frozenset({"od_matrix", "multiclass"}),
        outputs=frozenset({"link_flows", "class_link_flows"}),
    )
    factors = {
        "outer_iters": FactorSpec(
            default=200,
            kind="int",
            bounds=(1, 10000),
            doc="Outer diagonalization sweeps over the classes (freeze the other "
            "classes, re-solve each class's separable UE, repeat).",
        ),
        "inner_iterations": FactorSpec(
            default=20,
            kind="int",
            bounds=(1, 500),
            doc="Frank-Wolfe steps on each class's diagonalized separable cost per "
            "sweep.",
        ),
        "relaxation": FactorSpec(
            default=0.7,
            kind="float",
            bounds=(1e-3, 1.0),
            doc="Per-class outer relaxation step in (0,1]: v^i <- v^i + step*"
            "(v^i_inner - v^i). 1.0 is plain diagonalization; smaller damps "
            "oscillation when the cross-class coupling is strong.",
        ),
        "target_gap": FactorSpec(
            default=1e-10,
            kind="float",
            bounds=(0.0, 1e-1),
            doc="Outer stop tolerance on the class-summed VI relative gap at the "
            "full coupled cost.",
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
        mc = scenario.multiclass
        if mc is None:
            raise ValueError(
                "multiclass requires a scenario with multiclass demand set "
                f"(MulticlassDemand); scenario '{scenario.name}' has none"
            )
        engine = PathEngine(network)
        outer_iters = int(self.factor_values["outer_iters"])
        inner_iterations = int(self.factor_values["inner_iterations"])
        relaxation = float(self.factor_values["relaxation"])
        target_gap = float(self.factor_values["target_gap"])
        line_search_xtol = float(self.factor_values["line_search_xtol"])
        m = network.n_links
        k = mc.n_classes
        interaction = mc.interaction
        class_demands = [Demand(mc.matrices[i]) for i in range(k)]

        def class_cost(w: np.ndarray, others: np.ndarray, i: int, offset: np.ndarray) -> np.ndarray:
            # Diagonalized SEPARABLE cost for class i with the other classes'
            # flow FROZEN: BPR of the total (own w + frozen others), own-class
            # self-interaction M_ii w, and the frozen cross-class coupling offset.
            return network.link_cost(w + others) + interaction[i, i] * w + offset

        def full_class_cost(state: np.ndarray, i: int) -> np.ndarray:
            # The true coupled cost of class i at the joint flow `state` (K, m):
            # t_a^i = t_BPR(total) + sum_j M_ij v_a^j.
            total = state.sum(axis=0)
            return network.link_cost(total) + interaction[i] @ state

        # Feasible finite start: each class routes at the free-flow BPR cost.
        v0 = np.zeros((k, m), dtype=np.float64)
        base = network.link_cost(np.zeros(m))
        sp_calls = 0
        for i in range(k):
            v0[i], _ = engine.all_or_nothing(base, class_demands[i])
            sp_calls += 1
        state = v0

        def vi_gap(cur: np.ndarray) -> float:
            nonlocal sp_calls
            tstt = 0.0
            sptt = 0.0
            for i in range(k):
                cost_i = full_class_cost(cur, i)
                if not np.all(np.isfinite(cost_i)) or cost_i.min() <= 0.0:
                    return float("nan")
                tstt += float(cur[i] @ cost_i)
                _, s_i = engine.all_or_nothing(cost_i, class_demands[i])
                sp_calls += 1
                sptt += s_i
            return (tstt - sptt) / tstt if tstt > 0 else 0.0

        outer = 0
        while outer < outer_iters:
            outer += 1
            # One Gauss-Seidel sweep over the classes.
            try:
                for i in range(k):
                    others = state.sum(axis=0) - state[i]
                    offset = interaction[i] @ state - interaction[i, i] * state[i]
                    w = state[i].copy()
                    for _ in range(inner_iterations):
                        c = class_cost(w, others, i, offset)
                        if not np.all(np.isfinite(c)) or c.min() <= 0.0:
                            raise ValueError("non-positive diagonalized cost")
                        y, _ = engine.all_or_nothing(c, class_demands[i])
                        sp_calls += 1
                        dx = y - w

                        def g(a: float, _dx=dx, _w=w, _o=others, _i=i, _off=offset) -> float:
                            return float(class_cost(_w + a * _dx, _o, _i, _off) @ _dx)

                        if g(0.0) >= 0.0:
                            break  # class-i diagonalized UE reached
                        alpha = (
                            1.0
                            if g(1.0) <= 0.0
                            else float(brentq(g, 0.0, 1.0, xtol=line_search_xtol))
                        )
                        if alpha <= 0.0:
                            break
                        w = w + alpha * dx
                    # Gauss-Seidel: commit the relaxed class-i flow immediately.
                    state[i] = state[i] + relaxation * (w - state[i])
            except (RuntimeError, ValueError):
                break

            gap = vi_gap(state)
            coords = BudgetCoords(
                iterations=outer,
                sp_calls=sp_calls,
                wall_ms=1000.0 * (time.perf_counter() - start),
            )
            trace.record(
                state.sum(axis=0), coords, class_link_flows=state, relative_gap=gap
            )
            # Two convergence stop channels (parity with the sibling solvers,
            # e.g. frank_wolfe/msa): the caller-facing Budget.target_relative_gap
            # via budget.target_met (already NaN/None-safe), AND the model-owned
            # target_gap factor (kept under its np.isfinite guard). Both are
            # honored so a caller's Budget target is not a silent no-op here.
            if (
                budget.exhausted(coords)
                or budget.target_met(gap)
                or (np.isfinite(gap) and gap <= target_gap)
            ):
                break

        if len(trace) == 0:
            trace.record(
                state.sum(axis=0),
                BudgetCoords(
                    iterations=outer,
                    sp_calls=sp_calls,
                    wall_ms=1000.0 * (time.perf_counter() - start),
                ),
                class_link_flows=state,
                relative_gap=float("nan"),
            )

        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
