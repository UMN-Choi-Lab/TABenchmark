"""Horowitz's (1984) perceived-cost-state day-to-day SUE — the logit variant.

The first stability analysis of stochastic user equilibrium: Horowitz embedded
the (S)UE fixed point inside an explicit discrete-time day-to-day process in
which travelers **exponentially smooth previously experienced costs** and
re-choose routes by a probabilistic (logit) rule, then asked whether the SUE
fixed point is a stable *attractor*. This is the day-to-day sibling of
``sue-msa`` (the MSA SUE solver) and ``dtd-swap-sue`` (the route-swap SUE
dynamics), and it converges to the SAME Dial-STOCH logit fixed point — but it
carries a *distinctive state* and can *fail to converge*, which is the whole
point of the model.

State (the distinctive difference from every shipped day-to-day model). Where
``dtd-swap``/``dtd-link`` carry link/route FLOWS and ``dtd-swap-sue`` carries
route flows, Horowitz's traveler carries a **perceived LINK-cost vector**
``p`` (shape ``(n_links,)``): the exponentially smoothed memory of the costs
experienced on previous days. Each day travelers load the network with a logit
route choice made at the *perceived* costs, experience the *actual* costs the
resulting flow induces, and update their memory toward those actual costs
(the perceived-cost-smoothing + logit-load day-to-day process reviewed in the
open Watling & Hazelton 2003, *Networks and Spatial Economics* 3(3); the
``horowitz1984stability`` spec math_sketch):

    v_n     = D * P_logit(p_n)          [Dial-STOCH loading at perceived costs]
    p_{n+1} = (1 - w) * p_n + w * t(v_n)  [exponential cost smoothing, weight w]

There are NO per-OD route sets and NO column generation — the loading is the
pinned Dial map, so the model is much simpler than ``dtd-swap-sue`` and reuses
``StochEngine.load`` directly exactly as ``sue-msa`` does. "Stochastic" refers
to traveler perception; Dial's loading is closed-form and deterministic, so
this runs on the deterministic track (no RNG).

Equilibrium & certificate (P1; docs/design/adr-001). The rest point is the
logit stochastic user equilibrium: the Dial-STOCH fixed point ``v = L(t(v),
theta)`` — identical to what ``sue-msa`` and ``dtd-swap-sue`` reach. At the
fixed point the perceived costs equal the experienced costs (``p = t(v)``), so
``v = L(p) = L(t(v))`` and the residual is zero. The scored quantity is the
EXISTING logit-SUE certificate: the harness's fixed-point residual ``||v -
L(t(v), theta)||_1 / D`` with ``L`` the pinned Dial-STOCH loading map, gated on
``scenario.sue_theta`` with ``sue_family == "logit"`` — no new certificate and
no new scenario field. The model self-reports the SAME residual computed with
the SAME ``StochEngine.load`` map (loaded a second time each day at the ACTUAL
costs), so the P1 honesty check (self-report == harness-recomputed) passes to
float precision, exactly the mechanism ``sue-msa`` relies on. The
perceived-cost state changes nothing for certification: the harness certifies
the emitted physical flow ``v = L(p)`` (always demand-feasible — Dial routes
all demand), and the residual reads ~0 at the SUE and stays bounded away from 0
on an oscillating (non-converged) trajectory, so it faithfully witnesses
Horowitz's instability. It is NOT deterministic Wardrop UE: the certified
relative gap stays strictly positive at SUE (a descriptive column, like
``sue-msa``).

Distinctive dynamical property — stability vs. oscillation. Unlike the
always-convergent ``sue-msa`` (MSA step ``1/k``) and the always-stabilized
``dtd-swap``/``dtd-swap-sue`` (Armijo backtracking), the constant-weight
smoothing map is a genuine nonlinear discrete dynamical system whose SUE fixed
point is a stable attractor **only when the day-map Jacobian at the fixed point
is a contraction**. NO damping is added and NO backtracking/step-capping is
applied (unlike the three shipped day-to-day models) — preserving the
possibility of divergence is the whole point of the model. On the two-route
anchor the linearized multiplier of the perceived route-cost-difference map
``Delta_{n+1} = (1 - w) Delta_n + w * phi'(Delta_n)`` is ``M(w) = (1 - w) + w
phi'`` with ``phi' = -theta * s * f_A* f_B* / D`` (``s = dc_A/df_A + dc_B/df_B``
the summed route-cost slope), so the process is stable iff ``|M| < 1``, i.e.
``0 < w < w* = 2 / (1 - phi')`` — the forward-Euler stability limit of the
stable continuous learning ODE ``dp/dt = alpha (t(v) - p)``. On the anchor
``w* ~ 0.81``: below it the residual drives to ~0; above it the process settles
into a period-2 limit cycle and the residual stays O(1). ``w -> 0`` is heavy
memory / slow learning (very stable); ``w = 1`` is Horowitz's naive
"use yesterday's cost" current-cost adjustment (the least stable).

Sourcing. Horowitz (1984, *Transportation Science* 18(3):200-221) is a
two-link analytic/illustrative study with no standard-network reproducible
numerics; it is attributed unread. The perceived-cost smoothing update and the
logit day-loading are cross-verified from the open Watling & Hazelton (2003,
*Networks and Spatial Economics* 3(3):349-370, ``watling2003dynamics`` in the
canon) day-to-day assignment survey and the Sheffi (1985, *Urban Transportation
Networks* ch. 11-12) Dial-STOCH loading already shipped as
``_stoch.StochEngine``; the linearized
stability threshold ``w* = 2/(1 - phi')`` is re-derived here (forward-Euler of
the smoothing ODE) and numerically confirmed against the model's own trajectory
on the two-route anchor. Descends from Daganzo & Sheffi (1977) stochastic UE.
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
from ._stoch import StochEngine
from .base import TrafficAssignmentModel, register_model

__all__ = ["CostSmoothingSUEModel"]


@register_model
class CostSmoothingSUEModel(TrafficAssignmentModel):
    """Horowitz (1984) exponential cost-smoothing day-to-day dynamics for logit SUE.

    ``p_{n+1} = (1 - w) p_n + w t(L(p_n))``: travelers logit-load at their
    perceived link costs ``p`` and smooth those toward the experienced costs.
    """

    name = "dtd-horowitz"
    capabilities = Capabilities(
        paradigm="day_to_day",
        deterministic=True,
        provides_gap=True,
        seedable=True,
        # solve() raises without scenario.sue_theta (the logit-SUE task dial).
        inputs_required=frozenset({"od_matrix", "sue_theta"}),
    )
    factors = {
        "smoothing_weight": FactorSpec(
            default=0.5,
            kind="float",
            bounds=(1e-6, 1.0),
            doc="Exponential-smoothing weight w of the perceived-cost update "
            "p <- (1 - w) p + w t(v). w -> 0 is heavy memory / slow learning "
            "(very stable); w = 1 is the naive 'use yesterday's cost' model "
            "(Horowitz's current-cost adjustment). NO damping is added, so above "
            "the task-dependent stability threshold (~0.81 on the two-route "
            "anchor) the process oscillates -- that instability is the model's "
            "purpose, so unlike the other day-to-day models the step is NEVER "
            "capped or backtracked.",
        ),
    }

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        theta = scenario.sue_theta
        if theta is None:
            raise ValueError(
                "dtd-horowitz requires an SUE scenario (scenario.sue_theta is "
                "None); theta is task data, not a model factor"
            )
        if scenario.sue_family != "logit":
            raise ValueError(
                f"dtd-horowitz is the logit-SUE cost-smoothing model but scenario "
                f"'{scenario.name}' declares sue_family={scenario.sue_family!r}; "
                "use sue-probit-msa for the probit-SUE task"
            )
        start = time.perf_counter()
        network = scenario.network
        engine = StochEngine(network)
        total = scenario.demand.total
        w = self.factor_values["smoothing_weight"]
        sp_calls = 0

        # Day 0: perceived costs start at the free-flow costs (t(0)), matching
        # sue-msa's free-flow day-0. A convex combination of positive costs, so
        # p stays strictly positive forever and StochEngine._graph never rejects
        # it (Dial requires costs > 0).
        p = network.link_cost(np.zeros(network.n_links))

        k = 0
        while True:
            k += 1
            # Emit today's physical flow: logit (Dial-STOCH) loading at the
            # PERCEIVED costs. Always demand-feasible (Dial routes all demand,
            # node-balance ~ 0), so this is what the harness certifies.
            v = engine.load(p, scenario.demand, theta)
            sp_calls += 1

            # Experienced (actual) link costs the emitted flow induces (numpy,
            # free) -- what travelers smooth their memory toward.
            costs = network.link_cost(v)

            # Convergence measure == the harness certificate (P1): the logit-SUE
            # fixed-point residual computed with the SAME pinned Dial-STOCH map
            # at the ACTUAL costs, so the self-report equals the recomputed score
            # to float precision (exactly sue-msa's mechanism).
            y = engine.load(costs, scenario.demand, theta)
            sp_calls += 1
            residual = float(np.abs(v - y).sum() / total) if total > 0 else 0.0

            coords = BudgetCoords(
                iterations=k,
                sp_calls=sp_calls,  # two Dial loads/day (emit + certify)
                wall_ms=1000.0 * (time.perf_counter() - start),
            )
            # perceived_cost_gap = ||p - t(v)||_1 -> 0 at the fixed point
            # (perception matches experience); pure provenance, never scored.
            trace.record(
                v,
                coords,
                sue_fixed_point_residual=residual,
                perceived_cost_gap=float(np.abs(p - costs).sum()),
            )

            # The convergence target applies to this model's self-monitored
            # convergence measure: the SUE fixed-point residual (ADR-001).
            if budget.exhausted(coords) or budget.target_met(residual):
                break

            # The learning update: exponentially smooth the perceived costs
            # toward today's experienced costs. NO damping, NO step cap -- above
            # the task-dependent stability threshold this diverges into a limit
            # cycle, which is the phenomenon the model exists to exhibit.
            p = (1.0 - w) * p + w * costs

        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
