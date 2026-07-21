"""Li, Wang & Nie (2024) cumulative-logit (CumLog) day-to-day dynamics: the
first boundedly-rational logit route-choice process whose limit is EXACT
deterministic Wardrop UE at a *finite* exploitation parameter.

Every shipped logit-choice day-to-day model rests at STOCHASTIC user
equilibrium -- ``dtd-horowitz`` (perceived-cost smoothing + logit),
``dtd-stochastic`` (Cascetta's finite-population Markov chain),
``dtd-swap-sue`` (Smith-Watling swap on the Fisk cost) -- and every WE-limiting
day-to-day model uses a PERFECTLY-RATIONAL adjustment direction (``dtd-swap``'s
proportional swap, ``dtd-friesz``'s projected gradient, ``dtd-link``'s link
projection, ``dtd-unifying``'s AON best response). CumLog fills the empty cell:
a boundedly-rational logit choice map -- travelers assign strictly positive
probability to acceptable suboptimal routes every day -- whose global limit is
nonetheless the exact deterministic WE, with no equilibrium-concept relaxation,
no indifference band, and a finite ``r``. Li, Wang & Nie's bounded rationality
is thus PROCESS-level (the adjustment path is imperfect) rather than
CONCEPT-level (the ``br-ue`` / Mahmassani-Chang indifference band of
docs/design/adr-008 relaxes the equilibrium itself); the two rows are
complementary bookends and never interact through the certificate.

State -- cumulative route valuations (the distinctive difference). Where
``dtd-swap``/``dtd-link`` carry route/link FLOWS and ``dtd-horowitz`` carries a
smoothed perceived-cost vector, CumLog carries a per-OD route-VALUATION vector
``s`` over gp/dtd-swap-style column-generated working route sets. The day map is
the logit model on those valuations (paper Eq. 2),

    p_k = exp(-r s_k) / sum_{k' in K_w} exp(-r s_k'),   r > 0,

and the emitted flow loads the full OD demand over the working set,
``f_k = d_w p_k`` (demand-feasible every day, node balance ~ 0), aggregated to
link flows ``v``. The crucial one-line change from the classical
successive-average (SA) scheme ``s <- (1-eta) s + eta c`` (Horowitz 1984; the
``dtd-horowitz`` rest point is SUE) is that the experienced route cost
``c(p) = L^T u(v)`` is ACCUMULATED, not averaged (paper Eq. 6):

    s_t = s_{t-1} + eta_t c(p_{t-1}).

On the WE support the accumulated valuation DIFFERENCES converge to finite
nonzero constants -- explaining why equal-cost routes at WE carry unequal
probabilities (the valuation gap is ``-(log p*_k - log p*_k')/r``, resolving
Harsanyi's instability) -- while routes no WE strategy uses have valuations that
diverge to +inf, so their logit share vanishes and CumLog eliminates non-WE
routes even at finite ``r`` (a logit model with averaged costs cannot, unless
``r -> inf``; its limit is SUE). Valuations are stored min-normalized per OD
(``s <- s - min(s)``, paper Variant 1 / Remark 2 -- mathematically identical to
the raw scheme because the logit map depends only on valuation differences, and
it keeps the used-route floats finite while the dropped-route valuations still
diverge). ``accumulate=False`` recovers the SA scheme verbatim on the SAME
machinery -- the executable form of Remark 3's accumulation-vs-averaging
contrast (its rest point is the path-flow logit SUE, whose UE relative gap stays
strictly positive) and NOT a supported production mode.

Equilibrium & certificate (P1; docs/design/adr-001) -- EXISTING and unchanged.
The rest point is deterministic Wardrop UE (Prop. 1's VI ``<c(p*), p - p*> >=
0``), so the scored quantity is the harness's ordinary UE relative gap
``(TSTT - SPTT)/TSTT`` from the emitted link flows -- identical to
``dtd-swap``/``dtd-friesz``/``dtd-link``/``dtd-unifying`` (det.), and to the
paper's own convergence measure (its Eq. 11 is the same normalized relative gap
against the AON minimizer). No new certificate and no new scenario field: ``r``,
``eta0``, the ``eta_schedule`` and the ``accumulate``/``init_valuation_scale``
knobs are model factors like ``dtd-swap``'s ``swap_rate`` or ``dtd-link``'s
``step_size``, so the golden Braess content hash is byte-identical by
construction. The model self-reports the SAME relative gap the harness recomputes
(both ``(v @ t(v) - SPTT)/(v @ t(v))`` with ``SPTT`` from one batched Dijkstra),
so the P1 honesty check passes to float precision. SUE/elastic/combined/BR
scenarios are refused -- the limit is WE regardless of ``r``, and in particular
``scenario.sue_theta`` is task data that must NEVER be mapped to the model's
exploitation parameter ``r``.

Convergence conditions (Theorem 1; the distinctive validation). Under
Assumptions 1-2 (``u`` twice continuously differentiable; the symmetric parts of
``grad u`` and ``(grad u)^2`` PSD -- satisfied by any monotone ``u`` with
symmetric Jacobian and by "not-too-asymmetric" ``grad u``; NO strong
monotonicity, NO separability required) and any finite ``s0``, ``p_t`` converges
to a point in the WE solution set if EITHER

  (i)  ``eta_t -> 0`` with ``sum eta_t = inf`` (the ``"harmonic"`` schedule
       ``eta_t = eta0/(t+1)`` is the canonical example) -- convergence for ANY
       ``r``, the safe default; or
  (ii) ``eta_t = eta0`` constant with ``eta0 < 1/(2 r L)`` (the ``"constant"``
       schedule), ``c(p)`` being ``1/(4L)``-cocoercive.

The ``"harmonic"`` default converges regardless of ``r`` (paper Sec. 6.1, robust
where the SA model's WE coupling is a knife-edge: perturbing either the ``eta``
or ``r`` exponent by 0.01 destroys SA convergence, Sec. 6.2). Convergence is
Theorem 1(i)-ASYMPTOTIC, NOT unconditionally fast: when route costs are large the
logit saturates while the accumulated valuation differences slowly build up, and
the time to de-saturate scales like ``r * eta0 * cost-scale`` even on tiny
networks -- on high-cost instances pick ``eta0 ~ 1/cost-scale`` (or the constant
schedule with a small step). These are transients, not non-convergence: the
certified gap is honest throughout. The ``"constant"`` schedule uses ``eta0``
LITERALLY and converges under Theorem 1(ii)'s SUFFICIENT condition
``eta0 < 1/(2 r L)``, whose ``L`` is the demand-scaled Lipschitz constant of the
ROUTE-cost map ``c(p)`` on ``P`` (asymmetric case ``L = max_w d_w * H *
||Lambda||^2``) -- NOT computed here, and the paper never claims divergence above
it. The provenance column ``eta_heuristic_scale`` reports only the house
step-scale heuristic ``1/(2 r max_a t'_a(v))`` (the ``dtd-link``/``dtd-friesz``
step-normalization precedent), which is FLOW-INDEPENDENT for linear costs and is
NOT the Theorem 1(ii) bound in EITHER direction: on demand-6 Braess constant
steps well ABOVE it still converge to machine precision, and on demand-60 Braess
constant steps well BELOW it diverge (true stability is flow-dependent; the
heuristic is a step-scale reference only). Above a genuinely too-large constant
step the process diverges, and that divergence is PRESERVED, not damped (as in
``dtd-horowitz``). Global stability holds from ANY finite ``s0``: different
initializations reach the SAME unique WE link flow through DIFFERENT route
strategies (paper Sec. 6.4), which ``init_valuation_scale`` exposes by seeding
entering routes with N(0, scale) valuations.

Sourcing -- the strongest in the dtd family. Li, Wang & Nie (2024,
*Transportation Science* 58(5):973-994, DOI 10.1287/trsc.2023.0132) was READ IN
FULL from the published PDF: the model (Eqs. 2, 6), Assumptions 1-2, Theorem 1
with its KL-divergence / dual-averaging proof (Lemmas 1-2, Props. 2-3, Remarks
2-4), and all four experiment sets (Sec. 6) are used verbatim, so NO
attributed-unread caveat applies (unlike every other dtd primary). Open anchors
for the equations/theorem/numerics: arXiv:2304.02500 (v2, 2024-02-06) and the
NSF PAGES accepted manuscript (par.nsf.gov/servlets/purl/10537406). See
docs/design/adr-038-dtd-cumlog.md.

Path/column-generation machinery mirrors ``gp``/``dtd-swap``: per-OD working
route sets grown by one batched Dijkstra per day (which also supplies ``SPTT``
for the gap); a newly generated route enters at the per-OD benchmark valuation
``min(s)`` (plus the optional ``init_valuation_scale`` seed) so a freshly cheapest
route is immediately competitive; routes are never pruned (a dropped route's
valuation must be free to diverge -- that divergence is the WE-support signal),
and link flows are rebuilt exactly from the softmax load before every checkpoint.
Budget: one batched all-origins Dijkstra per day = one sp_call; the softmax load
and the cumulative update are numpy-only and cost none (the FW/GP/dtd-swap
convention).
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
from ._numerics import softmax
from ._paths import PathEngine
from .base import TrafficAssignmentModel, register_model

__all__ = ["CumLogDTDModel"]

_ACTIVE_TOL = 1e-6  # a route is "active" if its choice probability is >= this


@register_model
class CumLogDTDModel(TrafficAssignmentModel):
    """Li, Wang & Nie (2024) cumulative-logit day-to-day dynamics for Wardrop UE.

    ``s_t = s_{t-1} + eta_t c(p_{t-1})`` with ``p = softmax(-r s)`` over
    column-generated working route sets: boundedly-rational logit choices whose
    global limit is the exact deterministic Wardrop UE (finite ``r``).
    """

    name = "dtd-cumlog"
    capabilities = Capabilities(
        paradigm="day_to_day",
        deterministic=True,
        provides_gap=True,
        seedable=True,
    )
    factors = {
        "r": FactorSpec(
            default=1.0,
            kind="float",
            bounds=(1e-6, 1e6),
            doc="Exploitation parameter r of the logit day map p_k = "
            "exp(-r s_k)/sum exp(-r s_k'). Larger r is more exploitative (choices "
            "concentrate on the best-valued route); smaller r explores acceptable "
            "suboptimal routes more. Under the 'harmonic' schedule the WE limit "
            "holds for ANY r (Theorem 1(i)); under 'constant' Theorem 1(ii) gives "
            "the SUFFICIENT condition eta0 < 1/(2 r L) (L the demand-scaled "
            "route-cost Lipschitz constant, not computed here). r is a behavioral "
            "model factor and is NEVER derived from scenario.sue_theta.",
        ),
        "eta_schedule": FactorSpec(
            default="harmonic",
            kind="str",
            doc="Proactivity (step) schedule for the valuation update. 'harmonic' "
            "(default) uses eta_t = eta0/(t+1): eta_t -> 0 with sum eta_t = inf, so "
            "it converges to WE for ANY r (Theorem 1(i)) -- the safe default. "
            "'constant' uses eta_t = eta0 literally: Theorem 1(ii) gives the "
            "SUFFICIENT condition eta0 < 1/(2 r L) (L the demand-scaled route-cost "
            "Lipschitz constant, not computed here -- the reported "
            "eta_heuristic_scale is a step-scale reference, NOT this bound). Above a "
            "genuinely too-large step it diverges, and the divergence is preserved "
            "(never damped) because exhibiting the boundary is the model's purpose "
            "(cf. dtd-horowitz).",
        ),
        "eta0": FactorSpec(
            default=1.0,
            kind="float",
            bounds=(1e-9, 1e6),
            doc="Proactivity measure eta0 (paper's eta). 'harmonic': the numerator "
            "of eta_t = eta0/(t+1) (eta0 = 1 is the paper's eta_t = 1/(t+1)). "
            "'constant': the fixed step eta_t = eta0. Larger eta0 makes travelers "
            "settle faster but, under the constant schedule, risks crossing "
            "Theorem 1(ii)'s sufficient step ceiling 1/(2 r L).",
        ),
        "accumulate": FactorSpec(
            default=True,
            kind="bool",
            doc="Cost-accruement rule. True (default, CumLog): valuations ACCUMULATE "
            "experienced costs, s <- s + eta_t c(p) (Eq. 6), whose limit is exact "
            "Wardrop UE. False: the classical successive-average rule s <- "
            "(1-eta_t) s + eta_t c(p) (Eq. 4), whose limit is the path-flow logit "
            "SUE -- the one-line Remark 3 contrast, held on identical machinery. "
            "False is a regression/comparison knob for that contrast, NEVER a "
            "shipped SUE mode (dtd-horowitz remains the benchmark's logit-SUE "
            "day-to-day row); it does NOT reach deterministic UE, so the UE "
            "relative gap stays strictly positive.",
        ),
        "init_valuation_scale": FactorSpec(
            default=0.0,
            kind="float",
            bounds=(0.0, 1e6),
            doc="Standard deviation of the N(0, scale) valuation each entering route "
            "receives on top of the per-OD benchmark min(s). 0 (default) is the "
            "paper's deterministic s0 = 0. A positive scale seeds different initial "
            "valuations (paper Sec. 6.4): different finite s0 reach the SAME unique "
            "WE link flow through different route strategies (drawn reproducibly "
            "from the seeded RNG).",
        ),
    }

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        if scenario.sue_theta is not None:
            raise ValueError(
                "dtd-cumlog is a deterministic Wardrop-UE day-to-day model; its "
                f"limit is WE regardless of r, so it refuses the SUE scenario "
                f"'{scenario.name}' (scenario.sue_theta is set). theta is task data "
                "and must never be mapped to the model's exploitation parameter r; "
                "use dtd-horowitz / sue-msa for the logit-SUE task."
            )
        if scenario.elastic_demand is not None:
            raise ValueError(
                "dtd-cumlog assumes fixed demand; it refuses the elastic-demand "
                f"scenario '{scenario.name}' (use elastic-fw)."
            )
        if scenario.combined_demand is not None:
            raise ValueError(
                "dtd-cumlog assumes fixed demand; it refuses the combined "
                f"distribution+assignment scenario '{scenario.name}' (use evans)."
            )
        if scenario.br_epsilon is not None:
            raise ValueError(
                "dtd-cumlog reaches point-set Wardrop UE, not the boundedly-rational "
                f"indifference-band equilibrium; it refuses the BR-UE scenario "
                f"'{scenario.name}' (use br-ue). Its bounded rationality is in the "
                "adjustment process, not the equilibrium concept (adr-038 vs adr-008)."
            )

        start = time.perf_counter()
        network = scenario.network
        engine = PathEngine(network)
        od = scenario.demand.matrix
        r = self.factor_values["r"]
        schedule = self.factor_values["eta_schedule"]
        eta0 = self.factor_values["eta0"]
        accumulate = self.factor_values["accumulate"]
        init_scale = self.factor_values["init_valuation_scale"]
        if schedule not in ("harmonic", "constant"):
            raise ValueError(
                f"eta_schedule must be 'harmonic' or 'constant', got {schedule!r}"
            )
        gen = rng.generator(source=0) if init_scale > 0.0 else None
        sp_calls = 0

        def seed() -> float:
            """A one-time valuation offset for an entering route (0 unless seeded)."""
            return float(gen.normal(0.0, init_scale)) if gen is not None else 0.0

        # Day 0: all-or-nothing on the free-flow shortest path, one route per OD.
        first, _ = engine.shortest_paths(
            network.link_cost(np.zeros(network.n_links)), scenario.demand
        )
        sp_calls += 1
        paths = {key: [p] for key, p in first.items()}
        demand_rs = {key: float(od[key[0], key[1]]) for key in first}
        # Route valuations s_0: benchmark 0 (+ optional seed), min-normalized per OD.
        s: dict[tuple[int, int], np.ndarray] = {}
        for key in first:
            s0 = np.array([seed()], dtype=np.float64)
            s[key] = s0 - s0.min()

        def route_probs(key: tuple[int, int]) -> np.ndarray:
            """Logit choice probabilities p_k = softmax(-r s_k) over the OD's set."""
            return softmax(-r * s[key])

        def emitted() -> np.ndarray:
            """Softmax-loaded link flows: f_k = d_w p_k aggregated over routes."""
            v = np.zeros(network.n_links)
            for key, plist in paths.items():
                p = route_probs(key)
                d = demand_rs[key]
                for links, pk in zip(plist, p, strict=True):
                    v[links] += d * pk
            return v

        v = emitted()
        k = 0
        while True:
            k += 1
            costs = network.link_cost(v)
            shortest, sptt = engine.shortest_paths(costs, scenario.demand)
            sp_calls += 1
            tstt = float(v @ costs)
            gap = (tstt - sptt) / tstt if tstt > 0 else 0.0

            # Provenance (never scored): the valuation-divergence signature and the
            # WE-support descent. Used-route valuations stabilize to finite spreads
            # while dropped-route valuations diverge; the active-route count
            # descends to the WE support (paper Figs. 2, 11).
            used_spread = 0.0
            valuation_max = 0.0
            active_routes = 0
            strategy_entropy = 0.0
            for key in paths:
                p = route_probs(key)
                sv = s[key]
                valuation_max = max(valuation_max, float(sv.max()))
                active = p >= _ACTIVE_TOL
                active_routes += int(active.sum())
                if active.any():
                    used = sv[active]
                    used_spread = max(used_spread, float(used.max() - used.min()))
                pos = p[p > 0.0]
                strategy_entropy += float(-(pos * np.log(pos)).sum())

            # House step-scale heuristic 1/(2 r max_a t'_a(v)) (the
            # dtd-link/dtd-friesz step-normalization precedent) -- a provenance
            # reference for the constant schedule, never used to alter the step. It
            # is NOT the Theorem 1(ii) bound (whose L is the demand-scaled route-cost
            # Lipschitz constant): flow-independent for linear costs, so constant
            # steps above it can still converge and steps below it can still diverge.
            lipschitz = float(network.link_cost_derivative(v).max())
            eta_scale = 1.0 / (2.0 * r * lipschitz) if lipschitz > 0.0 else float("inf")

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
                used_valuation_spread=used_spread,
                valuation_max=valuation_max,
                active_routes=float(active_routes),
                strategy_entropy=strategy_entropy,
                eta_heuristic_scale=eta_scale,
            )
            if budget.exhausted(coords) or budget.target_met(gap):
                break

            # Column generation: add today's shortest path where new, entering at
            # the per-OD benchmark valuation min(s) (+ optional seed) so a freshly
            # cheapest route is immediately competitive.
            for key, new_path in shortest.items():
                known = paths[key]
                if not any(
                    p.shape == new_path.shape and np.array_equal(p, new_path)
                    for p in known
                ):
                    known.append(new_path)
                    s[key] = np.append(s[key], float(s[key].min()) + seed())

            # The learning update on the EXPERIENCED route costs. eta_t is the
            # schedule's step for this day. accumulate=True is CumLog (Eq. 6, the
            # WE limit); accumulate=False is the successive-average scheme (Eq. 4,
            # the SUE limit) held on identical machinery -- the Remark 3 contrast.
            eta_t = eta0 / (k + 1) if schedule == "harmonic" else eta0
            for key, plist in paths.items():
                c = np.array([float(costs[p].sum()) for p in plist])
                if accumulate:
                    s[key] = s[key] + eta_t * c
                else:
                    s[key] = (1.0 - eta_t) * s[key] + eta_t * c
                # Variant 1 (Remark 2): renormalize the best route to 0. The logit
                # map depends only on valuation differences, so this is exact; it
                # keeps used-route floats finite while dropped valuations diverge.
                s[key] = s[key] - s[key].min()

            v = emitted()

        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
