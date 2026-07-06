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
Dial-efficient paths*. That proviso holds on ANY network, overlapping routes
included, not only when routes are link-disjoint; what breaks it is an
incomplete route set. The model therefore column-generates the WHOLE efficient
route set each day (see below), so the coincidence holds on general networks
and not merely on the two-route anchor. The deterministic-UE limit
``theta -> infinity`` recovers dtd-swap's Wardrop rest point.

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
precision, exactly the mechanism ``sue-msa`` relies on. Once the working set is
the full Dial-efficient route set the path-flow logit equals Dial's
efficient-link logit, so the certified residual -> 0 to the same solver
tolerance ``sue-msa`` reaches. This is the point of enumerating the WHOLE
efficient set each day rather than one shortest path: it is the SUE analog of
dtd-swap's UE gap reaching the Wardrop value -- both need the pricing (column
generation) to supply the equilibrium's support, and for logit SUE that support
is the ENTIRE efficient set (every efficient route carries flow), not just the
min-cost routes UE needs. On tasks where the enumerated columns stay efficient
as the flows settle -- the two-route anchor, link-disjoint route sets (including
the K>=3 disjoint case where a route is never the strict shortest path), and
single-OD grids with overlapping routes -- the residual drives to machine
precision, regression-tested here against ``sue-msa``.

The residual is still reported as a descriptive convergence column, because two
things can leave it short of that tolerance, neither a route-set/logit-mismatch
(the old, wrong story) nor a certificate defect. (1) On STRONGLY CONGESTED
MULTI-OD networks a route can enter the efficient set at one flow and leave it
at another; because routes are never pruned (the entropy floor keeps every
working-set route at a positive share), such a route keeps a small logit share
that Dial does not assign, leaving a small residual floor -- far below the O(1)
residual one-shortest-path-per-day column generation left, but not tight. (2)
The proportional swap is a first-order day-to-day adjustment, so like ``sue-msa``
(and like dtd-swap's UE gap) it merely trends downward on stiff, high-demand
instances within a fixed horizon. Both are convergence properties of the
dynamics on hard networks, honestly scoped -- and distinct from the anchor and
disjoint/grid tasks, where the fixed point is reached outright.

Discretization. Like dtd-swap, the raw Euler step overshoots, so it is sized in
two stages. First, the Smith & Wisten (1995) ``a <= 1/(B M)`` level bound on the
GENERALIZED cost (B = largest |C|, M = route count, scaled by ``step_safety``) as
the initial step. Second, **Armijo backtracking on the Fisk Lyapunov F**: the
C-swap is a descent direction (``Fdot = -a V``), so halving the step until F
actually decreases is guaranteed to terminate and *guarantees* the monotone
descent the theory promises (Smith & Watling 2016) -- the SUE analog of
dtd-swap's backtrack-on-Beckmann.

Entropy floor. The ``(1/theta) ln h_k`` term is -inf at ``h_k = 0``, so a newly
column-generated route enters with a small positive seed ``seed_frac * D_rs``
(not 0), route flows are clamped to a per-OD relative floor ``h_floor_rel *
D_rs`` before taking the log, and each OD's flows are renormalized to sum to its
demand after every swap (exact conservation -> node-balance ~ 0). Routes are
never pruned to 0: logit SUE uses every working-set route at a positive share,
so the entropy term keeps flows away from 0 and the clamp rarely binds.

Sourcing. Smith & Watling (2016, *Transportation Research Part B* 85:132-141,
DOI 10.1016/j.trb.2015.12.015) is paywalled and attributed unread; the modified
route-swap dynamic and its Fisk-objective Lyapunov function are cross-verified
from the open Fisk (1980) SUE convex program (Sheffi 1985 ch. 11-12) and the
Smith (1984) swap + Lyapunov identity already shipped as ``dtd-swap``. The
``a <= 1/(BM)`` step bound is Smith & Wisten (1995), reused verbatim from
``dtd-swap`` with the generalized cost in place of the travel time; the exact
1984/2016 normalization of ``V`` is attributed, not verbatim-quoted, and used
only as a monotone-decrease diagnostic, so its scale is immaterial.

Path/column-generation machinery mirrors ``dtd-swap`` but enumerates the FULL
Dial-efficient route set each day (``PathEngine.efficient_paths``) instead of the
single shortest path -- the SUE equilibrium loads every efficient route, so the
pricing must supply every efficient route (Sheffi 1985 sec. 11.2). Routes are
never pruned (logit SUE keeps each at a positive share), so the per-OD working
set grows monotonically to the equilibrium efficient set; link flows are rebuilt
exactly from route flows before every checkpoint. Budget: one batched Dijkstra
(the efficient-set enumeration) + one Dial-STOCH load (the certified residual)
per day, each counted as one sp_call (dtd-swap counts the Dijkstra; sue-msa
counts each Dial load -- same unit).
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

        def aggregate() -> np.ndarray:
            v = np.zeros(network.n_links)
            for key, plist in paths.items():
                for links, h in zip(plist, flows[key], strict=True):
                    v[links] += h
            return v

        def clamp_log(key: tuple[int, int], h: np.ndarray) -> np.ndarray:
            """``ln h`` with h clamped to the OD floor so the log stays finite."""
            floor = h_floor_rel * demand_rs[key]
            return np.log(np.maximum(h, floor if floor > 0.0 else _TINY))

        def entropy(key: tuple[int, int], h: np.ndarray) -> float:
            """``(1/theta) sum_k h_k (ln h_k - 1)`` -- the Fisk entropy term."""
            hc = np.maximum(h, h_floor_rel * demand_rs[key] or _TINY)
            return float((hc * (np.log(hc) - 1.0)).sum())

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
            fisk = float(network.link_cost_integral(v).sum())
            disequilibrium = 0.0
            for key, plist in paths.items():
                h = np.asarray(flows[key], dtype=np.float64)
                fisk += inv_theta * entropy(key, h)
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
            # in the working set, each entering with a positive seed (NOT 0 --
            # ln 0 = -inf); the end-of-day renormalization restores the exact OD
            # demand. Enumerating the WHOLE efficient set (not one shortest path
            # per day) is what makes the path-flow logit rest point coincide with
            # the certified Dial link-logit on general networks: an efficient
            # route that is never the strict shortest path would otherwise never
            # be generated -- and so never loaded -- yet Dial loads it, which is
            # exactly the residual plateau this replaces.
            efficient = engine.efficient_paths(costs, scenario.demand)
            sp_calls += 1
            for key, eff_routes in efficient.items():
                known = paths[key]
                for new_path in eff_routes:
                    if not any(
                        p.shape == new_path.shape and np.array_equal(p, new_path)
                        for p in known
                    ):
                        known.append(new_path)
                        flows[key].append(seed_frac * demand_rs[key])

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

            # Armijo backtracking on the Fisk Lyapunov F: the C-swap is a descent
            # direction (Fdot = -a V), so a small enough step always decreases F;
            # halve until it does. Only the swapping ODs' entropy and the link
            # Beckmann term change, so F0 and the trial share every constant term.
            dv = np.zeros(network.n_links)
            for key, delta in directions.items():
                for links, d in zip(paths[key], delta, strict=True):
                    dv[links] += d
            if directions:
                f0 = float(network.link_cost_integral(v).sum())
                for key in directions:
                    f0 += inv_theta * entropy(key, np.asarray(flows[key]))
                slack = 1e-12 * max(abs(f0), 1.0)
                for _ in range(max_backtracks):
                    f_trial = float(network.link_cost_integral(v + step * dv).sum())
                    for key, delta in directions.items():
                        f_trial += inv_theta * entropy(
                            key, np.asarray(flows[key]) + step * delta
                        )
                    if f_trial <= f0 + slack:
                        break
                    step *= 0.5

            # Apply the swap, clamp to the entropy floor, and renormalize each OD
            # to its exact demand (node-balance ~ 0). Routes are never pruned:
            # logit SUE keeps every working-set route at a positive share.
            for key, delta in directions.items():
                h = np.asarray(flows[key], dtype=np.float64) + step * delta
                np.maximum(h, h_floor_rel * demand_rs[key], out=h)
                s = float(h.sum())
                if s > 0.0:
                    h *= demand_rs[key] / s
                flows[key] = list(h)

            v = aggregate()  # exact resync: emitted flows == route aggregation

        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
