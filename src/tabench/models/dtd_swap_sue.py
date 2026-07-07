"""Smith & Watling (2016) route-swap dynamics for logit stochastic user equilibrium.

The SUE sibling of ``dtd-swap`` (Smith 1984): the SAME proportional route-swap
day-to-day dynamical system, but with the deterministic route cost ``c_k``
replaced by the FISK-GENERALIZED (perceived) cost

    C_k = c_k + (1/theta) ln h_k,     theta = scenario.sue_theta,

so the swap is driven by the entropy-augmented utility rather than the raw
travel time. Per OD pair ``w`` with working-set route flows ``h`` and
generalized costs ``C`` (``[y]+`` = max(0, y)):

    h_k(n+1) = h_k(n) + a ( sum_p h_p [C_p - C_k]+  -  h_k sum_p [C_k - C_p]+ ),

the inflow being swaps ONTO route k and the outflow swaps OFF it. Swaps stay
within an OD pair, so demand is conserved structurally. The unique globally
stable REST POINT equalizes ``C_k`` across all used routes of each OD, i.e.
``h_k proportional to exp(-theta c_k)`` over the working route set -- a
path-flow logit (Fisk 1980), NOT deterministic Wardrop UE. Dial's
efficient-link weights telescope so that each efficient path carries weight
``exp(-theta c_path)`` (Sheffi 1985 sec. 11.2), so this path-flow logit
coincides EXACTLY with Dial's efficient-link logit -- the ``sue-msa`` /
Dial-STOCH fixed point -- *provided the working route set is the full set of
Dial-efficient paths at the equilibrium costs*. That proviso holds on ANY
network, overlapping routes included, not only when routes are link-disjoint;
what breaks it is an incomplete route set. The model therefore column-generates
the Dial-efficient route set each day (see below), so on networks whose
efficient DAG is enumerable and stable the coincidence holds generally and not
merely on the two-route anchor. Two things bound that reach on hard networks --
both scoped honestly under "Certificate" below, and NEITHER a certificate
defect: a per-OD enumeration cap that dense efficient DAGs exceed, and
never-pruned stale routes on strongly congested multi-OD instances. The
deterministic-UE limit ``theta -> infinity`` recovers dtd-swap's Wardrop rest
point.

Lyapunov / stability (Smith & Watling 2016; the distinctive validation). The
C-swap is a descent direction for Fisk's (1980) SUE convex objective

    F(h) = sum_a integral_0^{v_a} t_a(s) ds  +  (1/theta) sum_routes h_k (ln h_k - 1),

whose route-flow gradient is exactly the generalized cost, ``dF/dh_k = C_k`` (the
Beckmann link term contributes the route cost ``c_k = sum_{a in k} t_a``, the
entropy term contributes ``(1/theta) ln h_k``). Along the swap flow ``Fdot =
-a V <= 0`` with

    V(h) = sum_w sum_{p,k} h_p ([C_p - C_k]+)^2  >= 0,   V = 0 iff logit SUE,

the SUE analog of Smith's flow-weighted disequilibrium (the generalized cost in
place of the travel time). F decreases monotonically to the logit-SUE minimum.
We record ``fisk_objective`` F (the Lyapunov function, monotone non-increasing)
and the generalized-cost ``sue_disequilibrium`` V (-> 0) each day as provenance;
neither is scored -- the harness scores only the certified SUE residual.

Certificate (P1; docs/design/adr-001). The scored quantity is the harness's
logit-SUE fixed-point residual ``||v - L(t(v), theta)||_1 / D`` with ``L`` the
pinned Dial-STOCH loading map, gated on ``scenario.sue_theta`` with
``sue_family == "logit"`` -- no new certificate and no new scenario field. This
model self-reports the SAME residual computed with the SAME ``StochEngine.load``
map, so the P1 honesty check (self-report == recomputed) passes to float
precision, exactly the mechanism ``sue-msa`` relies on. When the working set is
the full Dial-efficient route set the path-flow logit equals Dial's
efficient-link logit, so the certified residual -> 0 to the same solver
tolerance ``sue-msa`` reaches. This is the point of enumerating the efficient
set each day rather than one shortest path: it is the SUE analog of dtd-swap's
UE gap reaching the Wardrop value -- both need the pricing (column generation)
to supply the equilibrium's support, and for logit SUE that support is the
ENTIRE efficient set (every efficient route carries flow), not just the min-cost
routes UE needs. On tasks where the FULL efficient set is enumerable (fits under
the cap) AND stays efficient as flows settle -- the two-route anchor,
link-disjoint route sets (including the K>=3 disjoint case where a route is
never the strict shortest path), and single-OD grids small enough that the
efficient DAG fits -- the residual drives to machine precision, regression-tested
here against ``sue-msa``.

The residual is still reported as a descriptive convergence column, because on
hard networks it plateaus ABOVE that tolerance -- always a convergence property
of the dynamics, NEVER a certificate defect (the harness recomputes it from the
emitted flows; self-report == recomputed to float precision and feasible = 1, so
nothing false is accepted). Two mechanisms drive the plateau, both
regression-pinned below:

(1) Enumeration cap. ``PathEngine.efficient_paths`` caps each OD's efficient-DAG
enumeration at ``max_routes`` (a dense DAG is exponential in the network size).
On a network whose efficient set exceeds the cap -- e.g. a single-OD 10x10 grid,
~24k efficient paths against a 4096 default -- the working set can never contain
the Dial support, the truncation emits a ``RuntimeWarning`` (never silent), and
the certified residual plateaus at an O(10) value that IS a route-set/logit
mismatch the truncation creates.

(2) Never-pruned stale routes. On STRONGLY CONGESTED MULTI-OD networks a route
enters the efficient set at one flow and leaves it as flows congest; because
routes are never pruned (the entropy floor keeps every working-set route at a
positive share), it keeps an ``exp(-theta c)`` share Dial does not assign. The
swap converges to an EXACT rest point -- the path-flow logit over the
accumulated, stale-inclusive working set -- with disequilibrium ``V ~ 0`` and a
certified residual that can be O(1), orders of magnitude above what ``sue-msa``
reaches on the SAME instance, and (being a true rest point, not slow
convergence) can even sit above an earlier iterate's residual. This is intrinsic
to keeping the pure Smith & Watling route-swap: matching Dial here would require
dropping stale columns, which forfeits the monotone-Fisk Lyapunov descent that
IS this model's defining validation. It is therefore scoped, not silently
certified -- ``dtd-swap-sue`` targets logit SUE where the efficient set is
enumerable and stable; on strongly congested multi-OD instances ``sue-msa``
reaches a tight residual and this model does not.

Discretization. Like dtd-swap, the raw Euler step overshoots, so it is sized in
two stages. First, the Smith & Wisten (1995) ``a <= 1/(B M)`` level bound on the
GENERALIZED cost (B = largest |C|, M = route count, scaled by ``step_safety``) as
the initial step. Second, **Armijo backtracking on the Fisk Lyapunov F**: the
C-swap is a descent direction (``Fdot = -a V``), so halving the step until F
actually decreases is guaranteed to terminate and *guarantees* the monotone
descent the theory promises (Smith & Watling 2016) -- the SUE analog of
dtd-swap's backtrack-on-Beckmann.

Entropy floor. The ``(1/theta) ln h_k`` term is -inf at ``h_k = 0``, so a newly
column-generated route enters with a small positive seed (not 0). The seed is
backtracked against the Fisk objective and paid for by scaling the existing
routes of the same OD pair, so route expansion itself is demand-feasible and
does not sit outside the Lyapunov check. Route flows are clamped to a per-OD
relative floor ``h_floor_rel * D_rs`` before taking the log, and each OD's flows
are renormalized to sum to its demand after every swap (exact conservation ->
node-balance ~ 0). Routes are never pruned to 0: logit SUE uses every working-set
route at a positive share, so the entropy term keeps flows away from 0 and the
clamp rarely binds.

Sourcing. Smith & Watling (2016, *Transportation Research Part B* 85:132-141,
DOI 10.1016/j.trb.2015.12.015) is paywalled and attributed unread; the modified
route-swap dynamic and its Fisk-objective Lyapunov function are cross-verified
from the open Fisk (1980) SUE convex program (Sheffi 1985 ch. 11-12) and the
Smith (1984) swap + Lyapunov identity already shipped as ``dtd-swap``. The
``a <= 1/(BM)`` step bound is Smith & Wisten (1995), reused verbatim from
``dtd-swap`` with the generalized cost in place of the travel time; the exact
1984/2016 normalization of ``V`` is attributed, not verbatim-quoted, and used
only as a monotone-decrease diagnostic, so its scale is immaterial.

Path/column-generation machinery mirrors ``dtd-swap`` but enumerates the
Dial-efficient route set each day (``PathEngine.efficient_paths``, up to its
``max_routes`` cap) instead of the single shortest path -- the SUE equilibrium
loads every efficient route, so the pricing must supply every efficient route
(Sheffi 1985 sec. 11.2). Routes are never pruned (logit SUE keeps each at a
positive share), so the per-OD working set grows monotonically, accumulating
every route that was EVER Dial-efficient -- including ones that later leave the
efficient set as flows congest (the never-pruned-stale-route plateau scoped
above); link flows are rebuilt exactly from route flows before every checkpoint.
Budget: one batched Dijkstra (the efficient-set enumeration) + one Dial-STOCH
load (the certified residual) per day, each counted as one sp_call (dtd-swap
counts the Dijkstra; sue-msa counts each Dial load -- same unit).
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

__all__ = ["RouteSwapSUEModel"]

_TINY = float(np.finfo(np.float64).tiny)


@register_model
class RouteSwapSUEModel(TrafficAssignmentModel):
    """Smith & Watling (2016) generalized-cost route-swap dynamics for logit SUE."""

    name = "dtd-swap-sue"
    capabilities = Capabilities(
        paradigm="day_to_day",
        deterministic=True,
        provides_gap=True,
        seedable=True,
    )
    factors = {
        "swap_rate": FactorSpec(
            default=1.0,
            kind="float",
            bounds=(1e-6, 1e6),
            doc="Smith's swap sensitivity a. The per-day step is adaptively capped "
            "below the over-swapping bound on the GENERALIZED cost regardless, so a "
            "only matters when it is the binding (smaller) rate; larger a converges "
            "faster until the cap binds.",
        ),
        "step_safety": FactorSpec(
            default=0.5,
            kind="float",
            bounds=(1e-3, 0.99),
            doc="Fraction of the Smith & Wisten (1995) 1/(B M) monotone-descent "
            "bound used for the adaptive step, with B the largest |generalized route "
            "cost| and M the route count. Smaller is more conservative (slower but "
            "strictly monotone); values near 1 approach the overshoot boundary.",
        ),
        "max_backtracks": FactorSpec(
            default=40,
            kind="int",
            bounds=(0, 100),
            doc="Maximum Armijo halvings of the step to enforce monotone descent of "
            "the Fisk Lyapunov objective. 40 halvings reach a step ~1e-12 of the "
            "initial; the descent direction guarantees a decreasing step exists well "
            "before that.",
        ),
        "seed_frac": FactorSpec(
            default=1e-2,
            kind="float",
            bounds=(1e-9, 0.5),
            doc="Positive flow (as a fraction of the OD demand) a newly "
            "column-generated route enters with. It must be > 0 because the entropy "
            "term (1/theta) ln h is -inf at h = 0; the end-of-day renormalization "
            "restores exact demand conservation.",
        ),
        "h_floor_rel": FactorSpec(
            default=1e-12,
            kind="float",
            bounds=(0.0, 1e-3),
            doc="Per-OD relative lower clamp on route flows (as a fraction of the OD "
            "demand) keeping ln h finite. The entropy term keeps used routes away "
            "from 0 at a logit SUE, so this rarely binds.",
        ),
    }

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        theta = scenario.sue_theta
        if theta is None:
            raise ValueError(
                "dtd-swap-sue requires an SUE scenario (scenario.sue_theta is None); "
                "theta is task data, not a model factor"
            )
        if scenario.sue_family != "logit":
            raise ValueError(
                f"dtd-swap-sue is the logit-SUE route-swap model but scenario "
                f"'{scenario.name}' declares sue_family={scenario.sue_family!r}; "
                "use sue-probit-msa for the probit-SUE task"
            )
        start = time.perf_counter()
        network = scenario.network
        engine = PathEngine(network)
        stoch = StochEngine(network)
        od = scenario.demand.matrix
        total = scenario.demand.total
        inv_theta = 1.0 / theta
        a = self.factor_values["swap_rate"]
        safety = self.factor_values["step_safety"]
        max_backtracks = self.factor_values["max_backtracks"]
        seed_frac = self.factor_values["seed_frac"]
        h_floor_rel = self.factor_values["h_floor_rel"]
        sp_calls = 0

        # Day 0: all-or-nothing on the free-flow shortest path, one route per OD.
        first, _ = engine.shortest_paths(
            network.link_cost(np.zeros(network.n_links)), scenario.demand
        )
        sp_calls += 1
        paths = {key: [p] for key, p in first.items()}
        demand_rs = {key: float(od[key[0], key[1]]) for key in first}
        flows = {key: [demand_rs[key]] for key in first}

        def aggregate_from(
            route_paths: dict[tuple[int, int], list[np.ndarray]],
            route_flows: dict[tuple[int, int], list[float] | np.ndarray],
        ) -> np.ndarray:
            v = np.zeros(network.n_links)
            for key, plist in route_paths.items():
                for links, h in zip(plist, route_flows[key], strict=True):
                    v[links] += h
            return v

        def aggregate() -> np.ndarray:
            return aggregate_from(paths, flows)

        def clamp_log(key: tuple[int, int], h: np.ndarray) -> np.ndarray:
            """``ln h`` with h clamped to the OD floor so the log stays finite."""
            floor = h_floor_rel * demand_rs[key]
            return np.log(np.maximum(h, floor if floor > 0.0 else _TINY))

        def entropy(key: tuple[int, int], h: np.ndarray) -> float:
            """``(1/theta) sum_k h_k (ln h_k - 1)`` -- the Fisk entropy term."""
            hc = np.maximum(h, h_floor_rel * demand_rs[key] or _TINY)
            return float((hc * (np.log(hc) - 1.0)).sum())

        def fisk_objective(
            route_paths: dict[tuple[int, int], list[np.ndarray]],
            route_flows: dict[tuple[int, int], list[float] | np.ndarray],
            link_flows: np.ndarray,
        ) -> float:
            """Fisk SUE objective for a consistent path-flow state."""
            out = float(network.link_cost_integral(link_flows).sum())
            for key in route_paths:
                out += inv_theta * entropy(
                    key, np.asarray(route_flows[key], dtype=np.float64)
                )
            return out

        def path_known(known: list[np.ndarray], new_path: np.ndarray) -> bool:
            return any(
                p.shape == new_path.shape and np.array_equal(p, new_path)
                for p in known
            )

        v = aggregate()
        k = 0
        while True:
            k += 1
            costs = network.link_cost(v)

            # Convergence measure == the harness certificate (P1): the logit-SUE
            # fixed-point residual computed with the SAME pinned Dial-STOCH map,
            # so the self-report equals the recomputed score to float precision.
            y = stoch.load(costs, scenario.demand, theta)
            sp_calls += 1
            residual = float(np.abs(y - v).sum() / total) if total > 0 else 0.0

            # Provenance: Fisk Lyapunov objective F (Beckmann link term + entropy,
            # monotone non-increasing) and the generalized-cost disequilibrium
            # V (-> 0 at logit SUE). Neither is scored.
            fisk = fisk_objective(paths, flows, v)
            disequilibrium = 0.0
            for key, plist in paths.items():
                h = np.asarray(flows[key], dtype=np.float64)
                if len(plist) < 2:
                    continue
                c = np.array([float(costs[p].sum()) for p in plist])
                cgen = c + inv_theta * clamp_log(key, h)  # C_k = c_k + (1/theta) ln h_k
                excess = np.maximum(cgen[:, None] - cgen[None, :], 0.0)
                disequilibrium += float(h @ (excess * excess).sum(axis=1))

            coords = BudgetCoords(
                iterations=k,
                sp_calls=sp_calls,
                wall_ms=1000.0 * (time.perf_counter() - start),
            )
            trace.record(
                v,
                coords,
                sue_fixed_point_residual=residual,
                fisk_objective=fisk,
                sue_disequilibrium=disequilibrium,
            )
            if budget.exhausted(coords) or budget.target_met(residual):
                break

            # Column generation: add EVERY currently Dial-efficient route not yet
            # in the working set. New routes need positive flow because ln(0) is
            # -inf, but that seed must be part of the Lyapunov step: it is paid
            # for by scaling the OD's existing routes and halved until the actual
            # Fisk objective of the expanded, demand-feasible state does not rise.
            # Enumerating the WHOLE efficient set (not one shortest path per day)
            # is what makes the path-flow logit rest point coincide with Dial.
            efficient = engine.efficient_paths(costs, scenario.demand)
            sp_calls += 1
            new_routes: dict[tuple[int, int], list[np.ndarray]] = {}
            for key, eff_routes in efficient.items():
                known = paths[key]
                for new_path in eff_routes:
                    if not path_known(known, new_path):
                        new_routes.setdefault(key, []).append(new_path)

            if new_routes:
                seed_scale = 1.0
                slack = 1e-12 * max(abs(fisk), 1.0)
                accepted_seed: tuple[
                    dict[tuple[int, int], list[np.ndarray]],
                    dict[tuple[int, int], np.ndarray],
                    np.ndarray,
                ] | None = None
                for _ in range(max_backtracks + 1):
                    trial_paths = {key: list(plist) for key, plist in paths.items()}
                    trial_flows = {
                        key: np.asarray(h, dtype=np.float64).copy()
                        for key, h in flows.items()
                    }
                    for key, additions in new_routes.items():
                        demand = demand_rs[key]
                        n_new = len(additions)
                        seed = min(seed_frac * seed_scale * demand, 0.5 * demand / n_new)
                        old = trial_flows[key]
                        old_sum = float(old.sum())
                        total_seed = seed * n_new
                        if old_sum > 0.0 and total_seed < demand:
                            old *= (demand - total_seed) / old_sum
                        else:
                            old.fill(0.0)
                        trial_paths[key].extend(additions)
                        trial_flows[key] = np.concatenate(
                            [old, np.full(n_new, seed, dtype=np.float64)]
                        )
                    trial_v = aggregate_from(trial_paths, trial_flows)
                    if fisk_objective(trial_paths, trial_flows, trial_v) <= fisk + slack:
                        accepted_seed = (trial_paths, trial_flows, trial_v)
                        break
                    seed_scale *= 0.5
                if accepted_seed is not None:
                    paths, seed_flows, v = accepted_seed
                    flows = {key: list(h) for key, h in seed_flows.items()}
                    costs = network.link_cost(v)

            # Per-OD swap directions on the GENERALIZED cost C_k = c_k + (1/theta)
            # ln h_k (a = 1 here; a enters only through the step cap below), plus
            # the largest |C| (B) and route-set size (M) for the step bound.
            directions: dict[tuple[int, int], np.ndarray] = {}
            b_max = 0.0
            m_max = 0
            for key, plist in paths.items():
                if len(plist) < 2:
                    continue
                c = np.array([float(costs[p].sum()) for p in plist])
                h = np.asarray(flows[key], dtype=np.float64)
                cgen = c + inv_theta * clamp_log(key, h)
                excess = np.maximum(cgen[:, None] - cgen[None, :], 0.0)  # E[p,k]=[C_p-C_k]+
                out_rate = excess.sum(axis=1)  # sum_p [C_k - C_p]+ per route k
                inflow = h @ excess  # sum_p h_p [C_p - C_k]+
                directions[key] = inflow - h * out_rate  # a=1 net swap, sum_k = 0
                b_max = max(b_max, float(np.abs(cgen).max()))
                m_max = max(m_max, len(plist))

            # Initial step: the Smith & Wisten (1995) 1/(B M) level bound on the
            # GENERALIZED cost, keeping the user's a when smaller.
            bound = b_max * m_max
            step = a if bound <= 0.0 else min(a, safety / bound)

            # Armijo backtracking on the Fisk Lyapunov F: check the actual emitted
            # state after clamping and per-OD renormalization, not the unprojected
            # Euler state. This keeps the discrete trace faithful to the claimed
            # monotone Lyapunov objective even when new columns were just seeded.
            dv = np.zeros(network.n_links)
            for key, delta in directions.items():
                for links, d in zip(paths[key], delta, strict=True):
                    dv[links] += d
            accepted_swap: tuple[dict[tuple[int, int], np.ndarray], np.ndarray] | None = None
            if directions:
                f0 = fisk_objective(paths, flows, v)
                slack = 1e-12 * max(abs(f0), 1.0)
                for _ in range(max_backtracks + 1):
                    trial_flows = {
                        key: np.asarray(h, dtype=np.float64).copy()
                        for key, h in flows.items()
                    }
                    for key, delta in directions.items():
                        h = trial_flows[key] + step * delta
                        np.maximum(h, h_floor_rel * demand_rs[key], out=h)
                        s = float(h.sum())
                        if s > 0.0:
                            h *= demand_rs[key] / s
                        trial_flows[key] = h
                    trial_v = aggregate_from(paths, trial_flows)
                    if fisk_objective(paths, trial_flows, trial_v) <= f0 + slack:
                        accepted_swap = (trial_flows, trial_v)
                        break
                    step *= 0.5

            # Apply the swap, clamp to the entropy floor, and renormalize each OD
            # to its exact demand (node-balance ~ 0). Routes are never pruned:
            # logit SUE keeps every working-set route at a positive share.
            if accepted_swap is not None:
                swap_flows, v = accepted_swap
                flows = {key: list(h) for key, h in swap_flows.items()}
            else:
                v = aggregate()  # exact resync: emitted flows == route aggregation

        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
