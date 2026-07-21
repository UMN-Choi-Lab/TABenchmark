"""Cascetta's (1989) stochastic-process day-to-day model — logit SUE as a mean.

Where every shipped day-to-day model (``dtd-swap``, ``dtd-link``,
``dtd-swap-sue``, ``dtd-horowitz``, ``dtd-friesz``) is deterministic-in-the-mean,
Cascetta recast day-to-day dynamics as a **genuine finite-population Markov
stochastic process**: each day a finite number of travelers each draws a route
at random from the logit choice probabilities, so daily flows NEVER converge —
"equilibrium" is the unique stationary probability distribution of the chain,
not a fixed point. The persistent day-to-day variability (order ``1/sqrt(N)``
in the traveler count ``N``) is the model's point, and its stationary MEAN
approximates the logit SUE with a finite-population bias that vanishes as the
population grows (Davis & Nihan 1993, *Operations Research* 41(1):169-178, the
large-population limit; Hazelton & Watling 2004, *Transportation Science*
38(3):331-342, computation of the stationary distribution).

State and day loop. Travelers carry ``dtd-horowitz``'s perceived link-cost
memory ``p`` (shape ``(n_links,)``, initialized at the free-flow costs — a
convex combination of strictly positive costs forever, so Dial never rejects
it). Day ``n`` then runs:

    v_n     ~ SampledDial(p_n, N)           [multinomial Dial load, N travelers]
    p_{n+1} = (1 - w) p_n + w t(v_n)        [exponential smoothing of the
                                             EXPERIENCED costs of the REALIZED
                                             daily flow]

The sampled load reuses ``_stoch.StochEngine``'s expanded-graph forward pass
verbatim (Dijkstra labels, efficient links, log-domain weights ``b``), then per
destination runs the backward pass on INTEGER traveler counts: ``N_od =
max(1, round(population_scale * d_od))`` travelers of weight ``d_od / N_od``
start at the destination and each node's count splits over incoming efficient
links by a multinomial draw with the SAME renormalized fractions ``phi =
exp(x + b_tail - b_j)`` the deterministic pass uses. Multinomial means
telescope to the deterministic recursion, so ``E[v_n | p_n] =
StochEngine.load(p_n, demand, theta)`` EXACTLY, and every traveler is routed,
so ``v_n`` is demand-feasible every day (node balance ~ float noise). Because
the update is driven by the sampled realization rather than its mean, ``{p_n}``
is a continuous-state Markov chain (an iterated random function: contraction
``(1 - w)`` plus bounded multinomial noise, geometrically ergodic for
``w < 1`` whenever the mean map is stable — the mean map is ``dtd-horowitz``'s,
with the same TASK-DEPENDENT stability threshold ``w* = 2 / (1 - phi')``:
~0.81 on the two-route anchor but far lower on congested networks — below the
0.3 default on the congested all-BPR-power-4 instance, where ``w`` must be
tuned down; see the ``smoothing_weight`` factor doc).

Emitted flow — the burnt-in time average with a soft handoff. Daily flows do
not converge, so the model emits the ergodic-theorem object instead: the
cumulative mean of the post-burn-in daily flows, which converges to the
stationary mean. The stationary average only takes over the emission once its
window is at least ``burn_in_days`` long (matching the statistical weight of
the day-1 running mean it replaces); until then the running mean from day 1 is
emitted (pre-stationary, ``window_days = 0`` marks it as provenance). The
handoff is why there is no reset discontinuity: a hard restart at day
``burn_in + 1`` would emit a single day's multinomial sample, collapsing
certified quality ~38x on the anchor and scoring WORSE than the
day-``burn_in`` emission for ~``burn_in`` further days — a budget-quality
inversion (adversarial-review Major 3, regression-tested). A convex
combination of demand-feasible daily loads is demand-feasible, so every
checkpoint passes the harness feasibility audit.

Certificate (P1; docs/design/adr-001) — EXISTING and unchanged: the harness
recomputes the logit-SUE fixed-point residual ``||v - L(t(v), theta)||_1 / D``
of the emitted time average with the pinned Dial-STOCH map, gated on
``scenario.sue_theta`` with ``sue_family == "logit"``. The model self-reports
the SAME residual via one extra ``StochEngine.load`` per day (exactly
``dtd-horowitz``'s two-loads-per-day pattern, ``sp_calls == 2n``), so the P1
honesty diff passes to float precision. Honest caveat: at finite population the
stationary mean is only APPROXIMATELY the SUE fixed point, so the certified
residual decays to a floor of order (finite-population bias + time-average
standard error) TIMES the certificate map's local L1 amplification — and BOTH
factors are instance-dependent. On the two-route anchor at ``population_scale
= 25`` (N = 100 travelers) a hand-derived delta-method ESTIMATE puts the
logit-curvature bias near 6e-4 in ``f_A`` and the certified floor near 0.015
after ~3000 days, not 1e-8. That anchor calibration does NOT transfer to
congested instances: on the all-BPR-power-4 day-to-day net the pinned Dial map
amplifies an L1-normalized flow perturbation of the fixed point by a measured
~5-54x (directions and seeds at 1e-3..1e-2), so the certified floor there is
O(0.1-1) even when the emitted time average is within ~1e-3 of the fixed point
(adversarial-review Major 2, regression-tested). The certificate itself stays
sound — the residual AT the (numerically converged) fixed point reads ~1e-3
and there is no false accept — but on high-curvature tasks the scored column
saturates at the amplified floor: compare dtd-stochastic runs on matched
instances, and read the floor as the price of emitting a finite-population
average, not as headroom the model could reach with more days.

Randomness (P8). ``deterministic=False`` routes the model onto the EXISTING
stochastic track (macroreps + percentile-bootstrap CI, adr-003 Decision 4).
Each day draws from ``rng.generator(source=0, replication=day)`` — the
``sue-probit-msa`` seeding pattern — so the same ``(root_seed, macrorep)`` is
byte-reproducible and distinct macroreps are independent trajectories.

Sourcing. Cascetta (1989, *Transportation Research Part B* 23(1):1-17, DOI
10.1016/0191-2615(89)90019-2) proves ergodicity for the original finite-memory,
finite-state chain; it is paywalled and attributed unread — no number from it
is reproduced. FLAGGED VARIANT: the original filters costs over an m-day
weighted moving average; this implementation uses the exponential filter,
the canonical special case formalized by Cantarella & Cascetta (1995,
*Transportation Science* 29(4):305-329) in the unified day-to-day framework,
cross-verified from the open Watling & Hazelton (2003, *Networks and Spatial
Economics* 3(3):349-370) survey of day-to-day dynamics — a documented,
well-established variant, not a verbatim reproduction. The stationary-mean ~
SUE approximation and its large-population limit are Davis & Nihan (1993) and
Hazelton & Watling (2004), both in the verified reference canon.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.sparse.csgraph import dijkstra

from ..core.budget import Budget, BudgetCoords
from ..core.capabilities import Capabilities
from ..core.factors import FactorSpec
from ..core.results import ResultBundle, Trace
from ..core.rng import RngBundle
from ..core.scenario import Demand, Scenario
from ._numerics import logsumexp
from ._stoch import StochEngine
from .base import TrafficAssignmentModel, register_model

__all__ = ["CascettaStochasticProcessModel"]


def _sampled_dial_load(
    engine: StochEngine,
    costs: np.ndarray,
    demand: Demand,
    theta: float,
    gen: np.random.Generator,
    population_scale: float,
) -> np.ndarray:
    """One day's SAMPLED Dial-STOCH load: integer travelers, multinomial splits.

    The forward pass (Dijkstra labels, efficient-link likelihoods ``x``,
    log-domain node weights ``b``) is ``StochEngine.load`` verbatim. The
    backward pass runs per destination on INTEGER traveler counts: ``N_od =
    max(1, round(population_scale * d_od))`` travelers (each of weight
    ``d_od / N_od``) start at the destination node; sweeping nodes by
    descending label, each node's count splits over its incoming efficient
    links by ``gen.multinomial(count, phi)`` with the SAME renormalized
    fractions ``phi = exp(x + b_tail - b_j)`` the deterministic pass uses.

    Unbiased by construction — multinomial means telescope node by node to the
    deterministic backward recursion, so ``E[result] = engine.load(costs,
    demand, theta)`` exactly — and every traveler reaches the origin, so the
    result routes all demand (node balance ~ float noise). Cost: one Dijkstra
    sweep per origin, the same unit as one deterministic Dial load (1 sp_call).
    """
    if not np.isfinite(theta) or theta <= 0:
        raise ValueError(f"theta must be finite and > 0, got {theta!r}")
    costs = np.asarray(costs, dtype=np.float64)
    graph = engine._paths._graph(costs)  # validates costs > 0 and finite
    od = demand.matrix
    origins = np.nonzero(od.sum(axis=1) > 0)[0]  # 0-based zone indices
    n_links = engine.network.n_links
    flows = np.zeros(n_links, dtype=np.float64)
    if origins.size == 0:
        return flows

    dist = dijkstra(graph, directed=True, indices=origins)
    tails, heads = engine._tails, engine._heads

    for row, o in enumerate(origins):
        r = dist[row]
        # Efficient links, likelihood exponents, and log-domain node weights:
        # byte-identical to StochEngine.load's forward pass.
        finite = np.isfinite(r[tails]) & np.isfinite(r[heads])
        efficient = finite & (r[tails] < r[heads])
        x = np.full(n_links, -np.inf)
        x[efficient] = theta * (
            r[heads[efficient]] - r[tails[efficient]] - costs[efficient]
        )

        reachable = np.nonzero(np.isfinite(r))[0]
        order = reachable[np.argsort(r[reachable], kind="stable")]
        origin_index = int(o)  # origin's tail role keeps its original index

        b = np.full(r.size, -np.inf)
        b[origin_index] = 0.0
        for j in order:
            if j == origin_index:
                continue
            terms = x[engine._incoming(j)] + b[tails[engine._incoming(j)]]
            b[j] = logsumexp(terms)  # stays -inf if no finite incoming term

        # Backward pass per destination on integer traveler counts (each OD
        # pair has its own per-traveler weight, so counts are never mixed
        # across destinations; expectations still sum to the merged
        # deterministic pass).
        for d in np.nonzero(od[o] > 0)[0]:
            if d == o:
                continue  # intrazonal demand never enters the network
            di = engine._paths._dest_index(d + 1)
            if not np.isfinite(r[di]):
                raise RuntimeError(
                    f"Zone {d + 1} unreachable from zone {o + 1} at current costs"
                )
            n_od = max(1, int(round(population_scale * od[o, d])))
            weight = od[o, d] / n_od
            count = np.zeros(r.size, dtype=np.int64)
            count[di] = n_od
            for j in order[::-1]:
                c = int(count[j])
                if c <= 0 or j == origin_index:
                    continue
                if not np.isfinite(b[j]):
                    raise RuntimeError(
                        f"No efficient path reaches loaded node "
                        f"{engine._node_id[j]} from zone {o + 1} (label ties "
                        "can sever efficient paths when costs saturate float64 "
                        "resolution)"
                    )
                links = engine._incoming(j)
                lw = x[links] + b[tails[links]] - b[j]
                mask = np.isfinite(lw)
                links, lw = links[mask], lw[mask]
                phi = np.exp(lw)  # each lw <= 0 by definition of b(j)
                phi /= phi.sum()  # renormalize: conserve travelers exactly
                draw = gen.multinomial(c, phi)
                flows[links] += weight * draw
                np.add.at(count, tails[links], draw)

    return flows


@register_model
class CascettaStochasticProcessModel(TrafficAssignmentModel):
    """Cascetta (1989) finite-population stochastic-process day-to-day dynamics.

    ``v_n ~ SampledDial(p_n, N); p_{n+1} = (1 - w) p_n + w t(v_n)``: daily flows
    never converge; the emitted burnt-in time average converges (ergodic
    theorem) to the stationary mean, which approximates the logit SUE.
    """

    name = "dtd-stochastic"
    capabilities = Capabilities(
        paradigm="day_to_day",
        deterministic=False,
        provides_gap=True,
        seedable=True,
        # solve() raises without scenario.sue_theta (the logit-SUE task dial).
        inputs_required=frozenset({"od_matrix", "sue_theta"}),
    )
    factors = {
        "smoothing_weight": FactorSpec(
            default=0.3,
            kind="float",
            bounds=(1e-6, 1.0),
            doc="Exponential-smoothing weight w of the perceived-cost update "
            "p <- (1 - w) p + w t(v) -- dtd-horowitz's filter verbatim, but "
            "driven by the SAMPLED daily flow, making {p} a genuine Markov "
            "chain. The chain is ergodic only below the mean map's "
            "TASK-DEPENDENT stability threshold w* = 2 / (1 - phi') "
            "(dtd-horowitz's): ~0.81 on the two-route anchor but below 0.03 "
            "on the congested all-BPR-power-4 instance, where the 0.3 default "
            "orbits far from the SUE and the time average certifies at O(1) "
            "-- on congested tasks w must be tuned DOWN (w = 0.01 is stable "
            "on that instance; regression-tested). w -> 0 is heavy memory / "
            "slow mixing; at w = 1.0 (memoryless) the mean map is unstable "
            "even on the anchor and the time average fails to certify small.",
        ),
        "population_scale": FactorSpec(
            default=1.0,
            kind="float",
            bounds=(1e-3, 1e6),
            doc="Travelers per demand unit: each positive interzonal OD pair "
            "carries N_od = max(1, round(scale * d_od)) travelers of weight "
            "d_od / N_od. Daily variability is O(1/sqrt(N)); scale -> inf "
            "recovers the deterministic-in-the-mean dtd-horowitz limit "
            "(Davis & Nihan 1993).",
        ),
        "burn_in_days": FactorSpec(
            default=200,
            kind="int",
            bounds=(0, 100000),
            doc="Days discarded before the stationary averaging window: the "
            "emitted flow is the cumulative mean of the daily flows on days "
            "burn_in+1..n, which takes over the emission only once that "
            "window is at least burn_in days long (soft handoff -- no reset "
            "discontinuity). Until then the running mean from day 1 is "
            "emitted (pre-stationary; window_days = 0 marks it).",
        ),
    }

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        theta = scenario.sue_theta
        if theta is None:
            raise ValueError(
                "dtd-stochastic requires an SUE scenario (scenario.sue_theta is "
                "None); theta is task data, not a model factor"
            )
        if scenario.sue_family != "logit":
            raise ValueError(
                f"dtd-stochastic is the logit-SUE stochastic-process model but "
                f"scenario '{scenario.name}' declares "
                f"sue_family={scenario.sue_family!r}; "
                "use sue-probit-msa for the probit-SUE task"
            )
        start = time.perf_counter()
        network = scenario.network
        engine = StochEngine(network)
        total = scenario.demand.total
        w = self.factor_values["smoothing_weight"]
        scale = self.factor_values["population_scale"]
        burn_in = self.factor_values["burn_in_days"]
        sp_calls = 0

        # Day 0: perceived costs start at the free-flow costs (t(0)), exactly
        # dtd-horowitz. A convex combination of positive costs, so p stays
        # strictly positive forever and Dial never rejects it.
        p = network.link_cost(np.zeros(network.n_links))

        # Two time averages (adversarial-review Major 3 fix): vbar_all is the
        # running mean of ALL days from day 1; vbar_stat is the stationary
        # average over the post-burn-in days only. The emission hands off from
        # vbar_all to vbar_stat only once the stationary window is at least
        # burn_in days long -- a hard reset at day burn_in + 1 would emit a
        # single day's multinomial sample, collapsing certified quality ~38x
        # and scoring WORSE than the day-burn_in emission for ~burn_in further
        # days (a budget-quality inversion).
        vbar_all = np.zeros(network.n_links)
        vbar_stat = np.zeros(network.n_links)
        window = 0  # days accumulated into the stationary average
        n = 0
        while True:
            n += 1
            # Per-day Philox stream from the harness RngBundle (P8): the same
            # (root_seed, macrorep) replays byte-identically; distinct
            # macroreps are independent trajectories (sue-probit-msa pattern).
            gen = rng.generator(source=0, replication=n)

            # (1) Today's realized flow: SAMPLED Dial load at the perceived
            # costs -- N integer travelers, multinomial route draws. Always
            # demand-feasible (every traveler is routed).
            v = _sampled_dial_load(engine, p, scenario.demand, theta, gen, scale)
            sp_calls += 1

            # (2) Emitted flow: the burnt-in cumulative time average with a
            # soft handoff. Both accumulators are convex combinations of
            # demand-feasible daily loads, so the emission stays
            # demand-feasible at every checkpoint.
            vbar_all = vbar_all + (v - vbar_all) / n
            if n > burn_in:
                window += 1
                vbar_stat = vbar_stat + (v - vbar_stat) / window
            stationary = window > 0 and window >= burn_in
            vbar = vbar_stat if stationary else vbar_all

            # (3) Convergence measure == the harness certificate (P1): the
            # logit-SUE fixed-point residual of the EMITTED time average,
            # computed with the SAME pinned Dial-STOCH map, so the self-report
            # equals the recomputed score to float precision.
            y = engine.load(network.link_cost(vbar), scenario.demand, theta)
            sp_calls += 1
            residual = float(np.abs(vbar - y).sum() / total) if total > 0 else 0.0

            coords = BudgetCoords(
                iterations=n,
                sp_calls=sp_calls,  # two Dial-unit loads/day (sample + certify)
                wall_ms=1000.0 * (time.perf_counter() - start),
            )
            # daily_flow_deviation = ||v_n - vbar_n||_1 / D: the persistent
            # day-to-day variability (bounded away from 0 at finite N even as
            # the time-average residual settles); window_days = 0 flags the
            # pre-handoff emissions (the running mean from day 1). Pure
            # provenance, never scored.
            deviation = float(np.abs(v - vbar).sum() / total) if total > 0 else 0.0
            trace.record(
                vbar,
                coords,
                sue_fixed_point_residual=residual,
                daily_flow_deviation=deviation,
                window_days=float(window if stationary else 0),
            )

            # The convergence target applies to this model's self-monitored
            # convergence measure: the SUE fixed-point residual of the time
            # average (ADR-001) -- which floors at O(bias + SE), not 0.
            if budget.exhausted(coords) or budget.target_met(residual):
                break

            # (4) Memory update on the EXPERIENCED costs of the REALIZED daily
            # flow -- dtd-horowitz's exponential smoothing verbatim, driven by
            # the sampled realization: {p_n} is a genuine Markov chain.
            p = (1.0 - w) * p + w * network.link_cost(v)

        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
