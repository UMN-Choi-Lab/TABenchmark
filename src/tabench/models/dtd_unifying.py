"""Cantarella & Cascetta's (1995) unifying deterministic/stochastic day-to-day process.

The canonical unifying-theory node of the day-to-day family: Cantarella &
Cascetta synthesized the previously separate deterministic (Smith/Friesz) and
stochastic-perception (Horowitz/Cascetta) strands into ONE discrete-time
process built from two coupled updates — an explicit **cost-learning filter**
(travelers forecast tomorrow's costs by exponentially smoothing experienced
costs) and a **choice update** (each day only a fraction of travelers
reconsiders their route at the forecast costs, the rest repeating yesterday's
choice). The same two equations produce a deterministic process converging to
Wardrop UE when the choice map is the all-or-nothing best response, and to
logit SUE when it is a probabilistic (logit) choice map — the unification this
model realizes as a per-scenario mode gate, with NO new scenario field.

State and day loop (link space — no route sets, no column generation; the open
Cantarella & Watling restatement licenses the arc-variable formulation
verbatim: the process "may be formulated with respect to arc or path
variables" with the same stability conditions). The state is a forecast
(filtered) link-cost vector ``p`` (shape ``(n_links,)``) plus link flows
``v``; ``p`` and ``v`` start at the free-flow costs ``t(0)`` and the choice
map's load at them (matching ``msa``/``sue-msa``'s day 0). Day ``n`` then runs
(C&C eqs. 4.1a/4.1b, cross-verified from the open restatement):

    p_n = (1 - w) p_{n-1} + w t(v_{n-1})        [cost updating, memory w]
    y_n = ChoiceLoad(p_n)                       [choice map at forecast costs]
    v_n = v_{n-1} + alpha_n (y_n - v_{n-1})     [choice updating: fraction
                                                 alpha_n reconsiders]

``v`` stays demand-feasible at every day — a convex combination of loads that
each route all demand — so every checkpoint passes the harness feasibility
audit (node balance ~ float noise).

Mode gate (per scenario, no new field; probit tasks are refused exactly as
``sue-msa`` refuses them):

* ``scenario.sue_theta is None`` — DETERMINISTIC process: the choice map is
  the all-or-nothing best response ``PathEngine.all_or_nothing`` (the
  ``theta -> inf`` limit of the choice model); fixed point = Wardrop UE, the
  same unique UE link flows ``fw``/``gp``/``dtd-swap`` reach.
* ``sue_theta`` set with ``sue_family == "logit"`` — STOCHASTIC process: the
  choice map is the pinned Dial-STOCH logit load ``StochEngine.load``; fixed
  point ``p* = t(v*), v* = L(t(v*), theta)`` = the logit SUE, identical to
  ``sue-msa``/``dtd-horowitz``/``dtd-swap-sue`` (the fixed-point <-> (S)UE
  equivalence is C&C eq. 4.2). "Stochastic" refers to traveler perception;
  Dial's loading is closed-form, so both modes run on the deterministic track.

Factors (the two C&C inertia axes — existence/uniqueness of the fixed point
are independent of them, stability is NOT, which is the model's point):
``memory_weight`` w in (0, 1] (C&C's cost-updating weight; w -> 0 is heavy
memory) and ``reconsideration_rate`` alpha in (0, 1] (the fraction of
travelers who reconsider each day; the open restatement notes values "in the
range [0.4, 0.6] seem likely", hence the (0.5, 0.5) defaults, flip-stable on
the anchor with wide margin). In stochastic mode ``alpha_n = alpha`` is
CONSTANT — the faithful C&C deterministic process (in their sense: a
deterministic recursion whose choice map is probabilistic). Flow inertia
``alpha < 1`` is the genuinely new axis over ``dtd-horowitz``, which is
exactly this model's ``alpha = 1`` corner.

Exact reductions (regression-tested to float precision): stochastic mode at
``alpha = 1`` reproduces ``dtd-horowitz``'s emitted flow trajectory verbatim
at the same ``w`` (one-day index offset from init bookkeeping); deterministic
mode at ``w = 1, alpha = 1`` IS ``msa``'s iterate sequence exactly, including
the AON-at-free-flow init (recorded-flow index offset by one).

Stability — the joint (alpha, w) flip boundary (re-derived here, numerically
confirmed; the distinctive validation). Like ``dtd-horowitz`` and unlike every
other shipped day-to-day model, NO damping/backtracking is added: the
constant-step stochastic process is a genuine nonlinear dynamical system that
converges only when the day-map Jacobian at the SUE is a contraction. On the
two-route anchor the linearized 2x2 day map in (perceived route-cost
difference, route-A flow) has determinant ``(1 - w)(1 - alpha)`` and
period-doubling (flip) boundary

    (2 - w)(2 - alpha) = alpha w |phi'|,   phi' = -theta s f_A* f_B* / D

(``s`` = summed route-cost slope; the fold boundary ``alpha w (1 - phi') = 0``
never binds, so flip is the only instability). At ``alpha = 1`` this reduces
to ``dtd-horowitz``'s documented threshold ``w* = 2/(1 - phi') ~ 0.81`` on the
anchor — an independent consistency check — and (1, 1) is unstable (multiplier
``|1 - w + w phi'| ~ 1.47``, period-2 limit cycle, residual O(1)) while the
(0.5, 0.5) default is stable (margin 2.25 vs ~0.37). C&C's headline follows by
hand from the formula: EITHER form of inertia — cost memory (small w) OR
choice inertia (small alpha) — restores stability.

Certificates (P1; EXISTING only, no new certificate and no new scenario
field). Deterministic mode: the standard harness-recomputed relative gap
``(TSTT - SPTT)/TSTT`` from emitted link flows — exactly what certifies
``dtd-swap``/``dtd-link``/``dtd-friesz``. Stochastic mode: the ADR-001
logit-SUE fixed-point residual ``||v - L(t(v), theta)||_1 / D`` with the
pinned Dial-STOCH map, gated on ``scenario.sue_theta`` — exactly what
certifies ``sue-msa``/``dtd-horowitz``/``dtd-stochastic``. The harness selects
the certified mode automatically because the Evaluator already branches on
``sue_theta``; the model self-monitors the SAME quantity computed with the
SAME engines (one extra AON/Dial load per day at the ACTUAL costs — the
``dtd-horowitz`` two-loads-per-day convention), so the P1 honesty diff passes
to float precision. The provenance column ``perceived_cost_gap = ||p -
t(v)||_1`` (forecast minus experience, -> 0 at any rest point) is never scored.

Flagged variants (repo convention — honesty about what is and is not the 1995
paper's letter): (i) the scalar exponential filter is the canonical special
case of C&C's general matrix learning filter — the same documented flag
``dtd-stochastic`` carries; (ii) the deterministic branch anneals the
reconsideration step ``alpha_n = alpha / n`` — an algorithmic selection, not
C&C's constant-step process: the AON best-response map is discontinuous
(set-valued at cost ties), so the constant-alpha deterministic-choice process
generically limit-cycles at O(alpha) around UE, and the open Cantarella &
Watling restatement confirms discrete-time deterministic-choice processes need
"a sufficiently slow rate of adjustment". ``alpha / n`` satisfies the
Blum/Powell-Sheffi step conditions (Powell & Sheffi 1982), UE remains the
fixed point of the set-valued inclusion, and at the ``w = 1, alpha = 1``
corner the scheme IS MSA exactly; for ``w < 1`` the filter lags ``t(v_n)`` by
``O(1/(w n))`` (v moves O(1/n) per day), so it is asymptotically MSA at true
costs — numerically validated on the anchors, no formal proof claimed.

Sourcing. Cantarella & Cascetta (1995, *Transportation Science* 29(4):305-329,
DOI 10.1287/trsc.29.4.305) is paywalled and attributed unread — no number from
it is reproduced. The two-equation process, the arc-variable license, the
fixed-point <-> (S)UE equivalence, and the either-form-of-inertia stability
taxonomy are cross-verified verbatim from the open-access Cantarella & Watling
(2016, *EJTL* 5(3)) restatement and the open Watling & Hazelton (2003, *NSE*
3(3):349-370) survey; the annealed-step conditions are Powell & Sheffi (1982);
the joint flip boundary is re-derived here and numerically confirmed against
the model's own trajectory (and against ``dtd-horowitz``'s shipped ``w*``).
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
from ._stoch import StochEngine
from .base import TrafficAssignmentModel, register_model

__all__ = ["UnifyingDTDModel"]


@register_model
class UnifyingDTDModel(TrafficAssignmentModel):
    """Cantarella & Cascetta (1995) unified cost-learning + choice-inertia dynamics.

    ``p <- (1 - w) p + w t(v); v <- v + alpha_n (ChoiceLoad(p) - v)``: the
    choice map is all-or-nothing (deterministic scenarios -> Wardrop UE) or
    the pinned Dial-STOCH logit load (SUE scenarios -> logit SUE), gated on
    ``scenario.sue_theta``.
    """

    name = "dtd-unifying"
    capabilities = Capabilities(
        paradigm="day_to_day",
        deterministic=True,
        provides_gap=True,
        seedable=True,
    )
    factors = {
        "memory_weight": FactorSpec(
            default=0.5,
            kind="float",
            bounds=(1e-6, 1.0),
            doc="Cost-updating (learning-filter) weight w of the forecast-cost "
            "update p <- (1 - w) p + w t(v). w -> 0 is heavy memory / slow "
            "learning (very stable); w = 1 forecasts with yesterday's costs "
            "alone. NO damping is added: in stochastic mode the joint flip "
            "boundary (2 - w)(2 - alpha) = alpha w |phi'| is task-dependent, "
            "and above it the process settles into a period-2 limit cycle -- "
            "exhibiting that boundary is the model's purpose.",
        ),
        "reconsideration_rate": FactorSpec(
            default=0.5,
            kind="float",
            bounds=(1e-6, 1.0),
            doc="Choice-updating rate alpha: the fraction of travelers who "
            "reconsider their route at the forecast costs each day (the rest "
            "repeat yesterday's choice), so v <- v + alpha_n (y - v). Constant "
            "alpha_n = alpha in stochastic mode (the faithful C&C process; "
            "alpha = 1 recovers dtd-horowitz exactly); annealed alpha_n = "
            "alpha / n in deterministic mode (flagged algorithmic selection -- "
            "the AON best response is discontinuous, so a constant step "
            "generically limit-cycles at O(alpha) around UE; alpha = 1 with "
            "memory_weight = 1 is exactly msa).",
        ),
    }

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        theta = scenario.sue_theta
        stochastic = theta is not None
        if stochastic and scenario.sue_family != "logit":
            raise ValueError(
                f"dtd-unifying's stochastic branch is the logit-SUE process but "
                f"scenario '{scenario.name}' declares "
                f"sue_family={scenario.sue_family!r}; use sue-probit-msa for the "
                "probit-SUE task"
            )
        start = time.perf_counter()
        network = scenario.network
        demand = scenario.demand
        total = demand.total
        w = self.factor_values["memory_weight"]
        alpha = self.factor_values["reconsideration_rate"]
        sp_calls = 0

        # The mode gate: the choice map is the pinned Dial-STOCH logit load on
        # SUE scenarios, the all-or-nothing best response otherwise. Both are
        # the SAME engines the harness certificate uses (P1).
        stoch = StochEngine(network) if stochastic else None
        paths = PathEngine(network) if not stochastic else None

        def choice_load(costs: np.ndarray) -> np.ndarray:
            if stochastic:
                return stoch.load(costs, demand, theta)
            return paths.all_or_nothing(costs, demand)[0]

        # Day 0: forecast costs start at the free-flow costs t(0) and the flow
        # at the choice map's load there -- matching msa/sue-msa's day 0 (the
        # exact-reduction corners). Every later p is a convex combination of
        # strictly positive cost vectors, so Dijkstra/Dial never reject it.
        p = network.link_cost(np.zeros(network.n_links))
        v = choice_load(p)
        sp_calls += 1
        costs = network.link_cost(v)  # experienced costs (numpy, free)

        n = 0
        while True:
            n += 1
            # (1) Cost updating (C&C eq. 4.1a): exponentially smooth the
            # forecast toward yesterday's experienced costs (reused from step
            # (3) of the previous day -- free).
            p = (1.0 - w) * p + w * costs

            # (2) Choice updating (C&C eq. 4.1b): a fraction alpha_n of
            # travelers reconsiders at the forecast costs. Constant alpha in
            # stochastic mode (the faithful C&C process); annealed alpha / n
            # in deterministic mode (flagged: the AON map is discontinuous, so
            # a Blum/Powell-Sheffi step is needed for convergence). v stays a
            # convex combination of full-demand loads => demand-feasible.
            y = choice_load(p)
            sp_calls += 1
            step = alpha if stochastic else alpha / n
            v = v + step * (y - v)

            # (3) Self-monitored convergence measure == the harness
            # certificate (P1), computed with the SAME engine at the ACTUAL
            # costs the emitted flow induces (the dtd-horowitz second load).
            costs = network.link_cost(v)
            if stochastic:
                y_cert = stoch.load(costs, demand, theta)
                sp_calls += 1
                residual = float(np.abs(v - y_cert).sum() / total) if total > 0 else 0.0
                measure = residual
            else:
                _, sptt = paths.all_or_nothing(costs, demand)
                sp_calls += 1
                tstt = float(v @ costs)
                gap = (tstt - sptt) / tstt if tstt > 0 else 0.0
                measure = gap

            coords = BudgetCoords(
                iterations=n,
                sp_calls=sp_calls,  # one init load + two loads/day (choice + certify)
                wall_ms=1000.0 * (time.perf_counter() - start),
            )
            # perceived_cost_gap = ||p - t(v)||_1 -> 0 at any rest point
            # (forecast matches experience); pure provenance, never scored.
            perceived_gap = float(np.abs(p - costs).sum())
            if stochastic:
                trace.record(
                    v,
                    coords,
                    sue_fixed_point_residual=residual,
                    perceived_cost_gap=perceived_gap,
                )
            else:
                trace.record(
                    v,
                    coords,
                    relative_gap=gap,
                    tstt=tstt,
                    sptt=sptt,
                    beckmann=float(network.link_cost_integral(v).sum()),
                    perceived_cost_gap=perceived_gap,
                )

            # The convergence target applies to this model's self-monitored
            # measure: the UE relative gap (deterministic) or the ADR-001
            # SUE fixed-point residual (stochastic).
            if budget.exhausted(coords) or budget.target_met(measure):
                break

        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
