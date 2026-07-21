"""Asymmetric variational-inequality traffic assignment (Dafermos 1980; Smith 1979).

Ordinary UE assumes SEPARABLE link costs ``t_a = t_a(v_a)`` -- each link's cost
depends only on its own flow -- which makes the equilibrium the minimizer of the
Beckmann potential ``sum_a integral_0^{v_a} t_a``. Dafermos (1980) and Smith
(1979) drop that assumption: with NON-separable costs the equilibrium is a
**variational inequality** (VI)

    find v* in the demand-feasible set K s.t.  <t(v*), v - v*> >= 0  for all v in K,

which has a Beckmann potential ONLY when the Jacobian ``nabla t`` is symmetric.
This model ships the genuinely asymmetric case: an affine non-separable cost

    t(v) = t_BPR(v) + C v,     C = scenario.link_interaction (possibly asymmetric),

so link ``a``'s cost picks up ``(C v)_a = sum_b C_ab v_b`` -- a flow interaction
(e.g. opposing-movement / merge spillover, or a shared-resource externality)
between links. When ``C`` is asymmetric there is NO potential, so no Beckmann /
Frank-Wolfe / gradient-projection solver can find ``v*`` by minimizing an
objective; the equilibrium is defined only by the VI. (When ``C`` is symmetric
the problem collapses back to a Beckmann UE with the quadratic term
``(1/2) v^T C v`` folded in; the interesting, distinct case is ``C != C^T``.)

Solver -- Dafermos diagonalization. We solve the VI by the classical fixed-point
scheme (Dafermos 1982): freeze the interaction term at the current iterate,
``offset = C v``, which turns the cost SEPARABLE (``t_diag_a(w) = t_BPR_a(w) +
offset_a``, a per-link constant), solve the resulting ordinary UE by Frank-Wolfe
(exact Brent line search on the diagonalized Beckmann objective), then re-freeze
the interaction at the new flow and repeat. An outer relaxation
``v <- v + step*(v_inner - v)`` (``step=1`` is plain diagonalization) damps
oscillation when the interaction is strong. The fixed point satisfies
``v = UE(t_BPR(v) + C v)``, i.e. ``<t_BPR(v) + C v, y - v> >= 0`` for all feasible
``y`` -- exactly the VI. Strict monotonicity (``nabla t = diag(t_BPR') + C`` has
positive-definite symmetric part; Dafermos 1980) guarantees the VI SOLUTION
exists and is unique, but that alone does NOT guarantee the diagonalization
ALGORITHM converges: convergence needs the stronger contraction / diagonal-
dominance condition of Dafermos (1982) -- roughly, the own-link sensitivity must
dominate the cross-effects AND the interaction must keep augmented costs positive
along the iteration. Positive, diagonally-dominant ``C`` (like the shipped anchor)
is in this convergent regime; a competitive/skew ``C`` with negative off-diagonal
entries can drive an augmented cost non-positive from the (route-concentrated)
free-flow start, at which point the diagonalization stops and emits a flow the
certificate then CENSORS -- never a false accept, but not a solution. The
always-reported VI residual makes non-convergence visible rather than hiding it.
When the interaction vanishes (``C = 0``) the outer loop is a no-op and the model
reduces EXACTLY to ordinary Frank-Wolfe UE, matching the shipped solvers on the
separable case. The outer loop stops on EITHER convergence channel: the
caller-facing ``Budget.target_relative_gap`` (via ``budget.target_met``, parity
with the sibling solvers) OR the model-owned ``target_gap`` factor, whichever the
VI relative gap satisfies first (in addition to the budget resource axes).

Certificate (P1; adr-011). The scored quantity is the **VI residual** -- the
normalized gap ``(<t(v), v> - min_{y in K} <t(v), y>) / <t(v), v>`` -- which the
harness recomputes from the emitted flows exactly as the ordinary ``relative_gap``,
only evaluating the cost at the asymmetric ``t(v) = t_BPR(v) + C v`` and running
all-or-nothing against it. A VI gap needs no potential, so this is a sound,
harness-recomputed necessary-and-sufficient equilibrium residual (it is 0 iff
``v`` solves the VI), NOT a self-report. ``beckmann_objective`` is reported NaN
(no potential exists). The fixed-demand feasibility / conservation audit is
unchanged.

Sourcing. Dafermos (1980, *Transportation Science* 14(1):42-54,
``dafermos1980traffic``) is the VI formulation; Smith (1979, *Transportation
Research Part B* 13(4):295-304, ``smith1979existence``) is the equivalent
equilibrium-existence characterization, and the diagonalization algorithm is
Dafermos (1982). Both primaries attributed; the VI condition, the monotonicity
uniqueness result, and the diagonalization scheme are cross-verified from the
open Boyles et al. TNA (non-separable costs / VI chapter). The analytic anchor
numbers are hand-derived here, not quoted.
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

__all__ = ["AsymmetricVIModel"]


@register_model
class AsymmetricVIModel(TrafficAssignmentModel):
    """Dafermos (1980) asymmetric-VI UE with non-separable costs, by diagonalization."""

    name = "vi-asym"
    capabilities = Capabilities(
        paradigm="static_ue_vi",
        deterministic=True,
        provides_gap=True,
        seedable=True,
    )
    factors = {
        "outer_iters": FactorSpec(
            default=60,
            kind="int",
            bounds=(1, 5000),
            doc="Outer diagonalization sweeps (freeze C v, re-solve the separable "
            "UE, re-freeze; the count converges the interaction fixed point).",
        ),
        "inner_iters": FactorSpec(
            default=20,
            kind="int",
            bounds=(1, 500),
            doc="Frank-Wolfe steps on the diagonalized separable cost per sweep.",
        ),
        "relaxation": FactorSpec(
            default=1.0,
            kind="float",
            bounds=(1e-3, 1.0),
            doc="Outer relaxation step in (0,1]: v <- v + step*(v_inner - v). 1.0 "
            "is plain diagonalization; smaller values damp oscillation when the "
            "interaction is strong relative to the BPR slopes.",
        ),
        "target_gap": FactorSpec(
            default=1e-10,
            kind="float",
            bounds=(0.0, 1e-1),
            doc="Outer stop tolerance on the VI relative gap at the full asymmetric "
            "cost t(v) = t_BPR(v) + C v.",
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
        c_mat = scenario.link_interaction
        if c_mat is None:
            raise ValueError(
                "vi-asym requires a scenario with link_interaction set (the "
                f"non-separable cost operator C); scenario '{scenario.name}' has none"
            )
        engine = PathEngine(network)
        outer_iters = self.factor_values["outer_iters"]
        inner_iters = self.factor_values["inner_iters"]
        relaxation = self.factor_values["relaxation"]
        target_gap = self.factor_values["target_gap"]
        line_search_xtol = self.factor_values["line_search_xtol"]
        m = network.n_links

        def full_cost(w: np.ndarray) -> np.ndarray:
            # The true (possibly asymmetric) VI cost t(w) = t_BPR(w) + C w.
            return network.link_cost(w) + c_mat @ w

        def diag_cost(w: np.ndarray, offset: np.ndarray) -> np.ndarray:
            # Diagonalized SEPARABLE cost: BPR own-link latency + the FROZEN
            # interaction contribution as a per-link constant.
            return network.link_cost(w) + offset

        # Feasible finite start: AON at free-flow (interaction offset C*0 = 0).
        v, _ = engine.all_or_nothing(network.link_cost(np.zeros(m)), scenario.demand)
        sp_calls = 1

        k = 0
        while k < int(outer_iters):
            k += 1
            offset = c_mat @ v  # freeze the interaction at the current outer iterate
            v_inner = v.copy()
            # Inner Frank-Wolfe on the diagonalized separable cost (fixed offset).
            # A factor-reachable regime where the frozen offset drives a cost
            # non-positive would break AON: stop gracefully and emit the current
            # flow, which the certificate then censors, rather than crashing.
            try:
                for _ in range(int(inner_iters)):
                    t_diag = diag_cost(v_inner, offset)
                    if not np.all(np.isfinite(t_diag)) or t_diag.min() <= 0.0:
                        raise ValueError("non-positive diagonalized cost")
                    y, _ = engine.all_or_nothing(t_diag, scenario.demand)
                    sp_calls += 1
                    dx = y - v_inner

                    def g(a: float, _dx=dx, _v=v_inner, _off=offset) -> float:
                        return float(diag_cost(_v + a * _dx, _off) @ _dx)

                    if g(0.0) >= 0.0:
                        break  # diagonalized UE reached for this frozen offset
                    alpha = (
                        1.0 if g(1.0) <= 0.0 else float(brentq(g, 0.0, 1.0, xtol=line_search_xtol))
                    )
                    if alpha <= 0.0:
                        break
                    v_inner = v_inner + alpha * dx
            except (RuntimeError, ValueError):
                break

            # Outer relaxation toward the diagonalized solution.
            v = v + relaxation * (v_inner - v)

            # Certificate: the VI relative gap at the FULL asymmetric cost.
            t = full_cost(v)
            if not np.all(np.isfinite(t)) or t.min() <= 0.0:
                break  # emit what we have; the harness censors a non-positive-cost flow
            _, sptt = engine.all_or_nothing(t, scenario.demand)
            sp_calls += 1
            tstt = float(v @ t)
            gap = (tstt - sptt) / tstt if tstt > 0 else 0.0

            coords = BudgetCoords(
                iterations=k,
                sp_calls=sp_calls,
                wall_ms=1000.0 * (time.perf_counter() - start),
            )
            trace.record(v, coords, relative_gap=gap)
            # Two convergence stop channels (parity with the sibling solvers,
            # e.g. frank_wolfe/msa): the caller-facing Budget.target_relative_gap
            # via budget.target_met, AND the model-owned target_gap factor. Both
            # are honored so a caller's Budget target is not a silent no-op here.
            if budget.exhausted(coords) or budget.target_met(gap) or gap <= target_gap:
                break

        # Guard: if the very first inner solve failed before any checkpoint, still
        # emit the start flow instead of an empty trace (the certificate scores it).
        if len(trace) == 0:
            trace.record(
                v,
                BudgetCoords(
                    iterations=k,
                    sp_calls=sp_calls,
                    wall_ms=1000.0 * (time.perf_counter() - start),
                ),
                relative_gap=float("nan"),
            )

        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
