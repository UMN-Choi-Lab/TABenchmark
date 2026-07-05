"""TAPAS (Bar-Gera 2010): traffic assignment by paired alternative segments.

TAPAS equilibrates a global pool of *paired alternative segments* (PASs). A PAS
is two disjoint path segments sharing a divergence node and a merge node, plus
the set of *relevant origins* whose bushes carry flow on both segments. Unlike
Algorithm B -- which shifts flow node-by-node within each origin's bush
independently -- one TAPAS PAS is shifted once for ALL its relevant origins with
a single aggregate Newton step, and TAPAS additionally performs *proportionality
adjustments*: it redistributes each origin's flow between a PAS's two segments
toward a common ratio, holding total link flows fixed. That last step is what
sets TAPAS apart: it drives the route flows to the (entropy-consistent)
*proportional* solution, whereas plain user-equilibrium fixes only link flows
(which are unique) and leaves route flows anywhere in a polyhedron.

Three components, exactly as Boyles/Lownes/Unnikrishnan lay them out (see
sourcing below), run each iteration: PAS management (identify), flow shifts
(cost equilibration), and proportionality adjustments.

Sourcing. Bar-Gera, *Transportation Research Part B* 44(8-9):1022-1046 (2010) is
the primary text and is paywalled; it was not read directly. Every load-bearing
formula was cross-verified against three accessible sources that agree:

* Boyles, Lownes & Unnikrishnan, *Transportation Network Analysis* v1.0
  (sboyles.github.io, Jan 2025) sec. 6.5.3 "Traffic assignment by paired
  alternate segments" (pp. 215-230) -- the SAME open textbook this repo already
  cites for Algorithm B (sec. 6.4). It gives: the PAS definition (p. 218); the
  Newton cost-equilibrating shift ``dH = (sum t_seg1 - sum t_seg2) / (sum t'_seg1
  + sum t'_seg2)`` (eq. 6.86, p. 222); the per-origin feasibility split
  proportional to each origin's bottleneck ``dhbar^r`` (eqs. 6.87-6.89, p. 223,
  "to help maintain proportionality"); the segment-flow proportionality condition
  (eq. 6.94, p. 226); and the linearized proportionality-restoring shift
  ``dh^r = g0^r(s1) - [g0^r(s1)+g0^r(s2)] * (sum_r' g0^r'(s1) / sum_r'
  [g0^r'(s1)+g0^r'(s2)])`` (eq. 6.100, p. 227), which the book flags as a
  heuristic -- exact for an "isolated" PAS, convergent under repetition for a
  non-isolated one.
* TAsK (olga-perederieieva/TAsK, PAS.cpp getFlowShift/calculateFlowShift, the
  code behind Perederieieva et al. 2015) -- the same cost-shift and the same
  ``minShift^r / totalShift`` per-origin split. TAsK's own comment states it does
  NOT implement the proportionality condition, so it verifies the cost half only.
* iTAPAS (hanqiu92/itapas, itapas.py shift(), after Xie & Xie 2014/2016) -- the
  identical Newton shift ``delta = (t2 - t1)/(dt1 + dt2)`` with the same
  losing-segment feasibility cap, from an independent author and library.

The proportionality shift itself (eq. 6.100) is verified only against Boyles;
neither reference implementation carries it, and the primary is unread -- flagged
here per the same honesty discipline as algb.py's Dial-2006 attribution.

The acyclic-bush machinery (initial bushes, min/max DAG labels, drop/add bush
update, the Kahn cyclicity guard) is inherited from ``_bush._BushMachinery`` and
shared bit-for-bit with Algorithm B, so bushes stay acyclic and TAPAS needs no
separate cycle-removal shift (Boyles' fourth, optional component): the shifts
only move flow along existing bush segments, adding no links.

Certificate (P1). The harness scores TAPAS on the UE ``relative_gap`` recomputed
from emitted link flows, exactly like every other UE solver -- no new scoring
code. TAPAS additionally self-reports ``proportionality_residual`` (eq. 6.94,
L1-normalized by total demand) and ``pas_proportionality_max``; these are
route-flow properties INVISIBLE to link flows (two solvers can emit identical
link flows with different segment splits), so they are provenance diagnostics
only, never scored. Making proportionality a scored certificate would require
emitting the origin-disaggregated flow decomposition -- a ``FlowState`` schema
change deferred to the PI in docs/design/adr-004.

Budget accounting (P6), consistent with algb. One all-origins bush-scan round
(here: one PAS-identification pass, min/max labels per origin) ~ one AON sweep =
one sp_call, as does each bush-update round and the honest-gap all-or-nothing.
Proportionality adjustments do NO shortest-path or cost work (pure per-origin
redistribution), so they are charged nothing and reported separately as
``pas_prop_rounds`` for transparency, not made to look like free label passes.
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
from ._bush import _BushMachinery, _BushState, walk_to_divergence
from .base import TrafficAssignmentModel, register_model

__all__ = ["TapasModel"]


@register_model
class TapasModel(_BushMachinery, TrafficAssignmentModel):
    """Bar-Gera (2010) TAPAS: global PAS pool + proportionality adjustments."""

    name = "tapas"
    capabilities = Capabilities(
        paradigm="static_ue",
        deterministic=True,
        provides_gap=True,
        seedable=True,
    )
    factors = {
        "inner_rounds": FactorSpec(
            default=10,
            kind="int",
            bounds=(1, 64),
            doc="Cost-equilibrating rounds per iteration; each round re-identifies "
            "the PAS pool and shifts every PAS once (mirrors algb inner_rounds).",
        ),
        "prop_rounds": FactorSpec(
            default=5,
            kind="int",
            bounds=(0, 64),
            doc="Proportionality-adjustment sub-rounds per iteration (Boyles eq. "
            "6.100 is a heuristic; repeated application converges for non-isolated "
            "PASs). 0 disables proportionality (pure UE, route flows arbitrary).",
        ),
        "newton_shifts": FactorSpec(
            default=1,
            kind="int",
            bounds=(1, 8),
            doc="Repeated Newton steps per PAS per round (TAsK numNewtonShifts).",
        ),
        "bush_update_every": FactorSpec(
            default=1,
            kind="int",
            bounds=(1, 8),
            doc="Bush improvement (drop/add links) every k-th iteration.",
        ),
        "alpha": FactorSpec(
            default=1.0,
            kind="float",
            bounds=(1e-6, 1.0),
            doc="Newton step scaling; 1 is stable thanks to the per-shift cost refresh.",
        ),
        "drop_tol": FactorSpec(
            default=1e-13,
            kind="float",
            bounds=(1e-16, 1e-6),
            doc="Bush-flow threshold for dropping links and used-link eligibility.",
        ),
        "dir_tol": FactorSpec(
            default=1e-13,
            kind="float",
            bounds=(1e-16, 1e-6),
            doc="Segment cost-gap below which a PAS is treated as balanced and not "
            "shifted (TAsK dirTolerance).",
        ),
    }

    # --------------------------------------------------------------- setup
    def _setup(self, scenario: Scenario) -> None:
        self._setup_bush_graph(scenario)  # shared expanded-graph state (_bush)
        self._drop_tol = self.factor_values["drop_tol"]
        self._alpha = self.factor_values["alpha"]
        self._newton_shifts = self.factor_values["newton_shifts"]
        self._dir_tol = self.factor_values["dir_tol"]

    # ------------------------------------------------------- PAS management
    def _identify_pas(self, bushes: list[_BushState], t: np.ndarray) -> list[dict]:
        """Rebuild the global PAS pool from every origin's current bush.

        For each origin, scan min (L, SP-tree) and max (U, longest-used) labels;
        at every node reached both by its shortest-path link and a distinct
        longer used link (U > L), walk both chains back to their common
        divergence node (:func:`walk_to_divergence`) to form a PAS. PASs with
        identical link sets discovered from different origins are one pooled PAS
        with several relevant origins. Rebuilding fresh each round means a PAS
        can never reference a link a bush update has since dropped (the main
        TAPAS bookkeeping hazard is designed out rather than patched).
        """
        pool: dict[frozenset, dict] = {}
        for oi, o in enumerate(self._origins):
            bush = bushes[oi]
            L, U, minp, maxp = self._scan(bush, int(o), t, "used")
            pos = bush.topo_pos
            for j in bush.topo:
                jj = int(j)
                if jj == int(o):
                    continue
                kmin = int(minp[jj])
                kmax = int(maxp[jj])
                if kmin < 0 or kmax < 0 or kmin == kmax:
                    continue
                if not (U[jj] - L[jj] > 0.0):
                    continue
                walked = walk_to_divergence(pos, minp, maxp, self._tails, jj, kmin, kmax)
                if walked is None:
                    continue
                seg_min, seg_max = walked
                sig = frozenset(
                    (
                        frozenset(int(x) for x in seg_min),
                        frozenset(int(x) for x in seg_max),
                    )
                )
                rec = pool.get(sig)
                if rec is None:
                    pool[sig] = {"segA": seg_min, "segB": seg_max, "origins": [oi]}
                elif oi not in rec["origins"]:
                    rec["origins"].append(oi)
        return list(pool.values())

    # ---------------------------------------------------- cost equilibration
    def _cost_equilibrate(
        self,
        pas: dict,
        bushes: list[_BushState],
        v: np.ndarray,
        t: np.ndarray,
        dt: np.ndarray,
    ) -> bool:
        """One aggregate Newton shift on a PAS across all relevant origins.

        Boyles eq. 6.86 (shift = segment cost gap / summed derivatives) with the
        per-origin split proportional to each origin's expensive-segment
        bottleneck (eqs. 6.87-6.89), capped so no origin's flow goes negative.
        Costs/derivatives are refreshed on the two segments after the shift.
        """
        network = self._network
        segA = pas["segA"]
        segB = pas["segB"]
        origins = pas["origins"]
        moved = False
        for _ in range(self._newton_shifts):
            # (Re)label cheap/expensive by current cost -- roles can flip as
            # other PASs shift within the same round (TAsK recalcPASCosts).
            if float(t[segA].sum()) <= float(t[segB].sum()):
                cheap, exp = segA, segB
            else:
                cheap, exp = segB, segA
            excess = float(t[exp].sum()) - float(t[cheap].sum())
            if excess <= self._dir_tol:
                break
            # Per-origin expensive-segment bottleneck (donor capacity).
            min_shift = [float(bushes[oi].x[exp].min()) for oi in origins]
            total_shift = float(sum(min_shift))
            if total_shift <= 0.0:
                break
            denom = float(dt[cheap].sum() + dt[exp].sum())
            if denom <= 0.0:
                # Zero derivative need not mean a flat direction (BPR power > 1
                # has t' = 0 only at v = 0): re-probe at the arrival flows, then
                # full-shift only if still flat (algb's fallback).
                probe = v.copy()
                probe[exp] -= total_shift
                probe[cheap] += total_shift
                np.maximum(probe, 0.0, out=probe)
                pd = network.link_cost_derivative(probe)
                denom = float(pd[cheap].sum() + pd[exp].sum())
                d_flow = (
                    total_shift if denom <= 0.0 else min(total_shift, self._alpha * excess / denom)
                )
            else:
                d_flow = min(total_shift, self._alpha * excess / denom)
            if d_flow <= 0.0:
                break
            for oi, m in zip(origins, min_shift, strict=True):
                share = m / total_shift * d_flow
                if share <= 0.0:
                    continue
                bx = bushes[oi].x
                bx[cheap] += share
                bx[exp] -= share
                np.maximum(bx, 0.0, out=bx)
            aff = np.concatenate([cheap, exp])
            v[cheap] += d_flow
            v[exp] -= d_flow
            v[aff] = np.maximum(v[aff], 0.0)
            t[aff] = network.link_cost(v)[aff]
            dt[aff] = network.link_cost_derivative(v)[aff]
            moved = True
        return moved

    # ------------------------------------------------ proportionality (6.100)
    @staticmethod
    def _segment_flows(pas: dict, bushes: list[_BushState]) -> list[tuple[int, float, float]]:
        """Per-origin (oi, segA-flow, segB-flow) using the segment bottleneck.

        The through-flow of an origin on a segment is the min over its links
        (equal on every link of an isolated PAS; the bottleneck otherwise -- the
        choice Boyles' heuristic assumes).
        """
        out = []
        for oi in pas["origins"]:
            bx = bushes[oi].x
            a = float(bx[pas["segA"]].min())
            b = float(bx[pas["segB"]].min())
            out.append((oi, a, b))
        return out

    def _proportionality_adjust(self, pas: dict, bushes: list[_BushState]) -> None:
        """Redistribute flow across origins toward the common segment ratio.

        Boyles eq. 6.100: shift ``dh^r = a^r - w^r * pi`` from segA to segB for
        each relevant origin, where ``pi`` is the flow-weighted aggregate segA
        share and ``w^r = a^r + b^r``. Because ``sum_r dh^r = 0`` (eq. 6.90),
        total segment flows -- and therefore all link flows and costs -- are
        unchanged; only the split across origins moves. ``dh^r`` lies in
        ``[-b^r, a^r]``, so every segment flow stays nonnegative by construction.
        """
        if len(pas["origins"]) < 2:
            return
        rows = self._segment_flows(pas, bushes)
        a_sum = sum(a for _, a, _ in rows)
        w_sum = sum(a + b for _, a, b in rows)
        if w_sum <= 0.0:
            return
        pi = a_sum / w_sum
        segA = pas["segA"]
        segB = pas["segB"]
        for oi, a, b in rows:
            dh = a - (a + b) * pi
            if dh == 0.0:
                continue
            bx = bushes[oi].x
            bx[segA] -= dh
            bx[segB] += dh
            np.maximum(bx, 0.0, out=bx)

    def _proportionality_residual(
        self, pool: list[dict], bushes: list[_BushState], demand_total: float
    ) -> tuple[float, float]:
        """Scenario proportionality diagnostic (Boyles eq. 6.94).

        ``sum over PASs and relevant origins of |a^r - w^r * pi_p|`` normalized by
        total demand (an intensive per-traveler misallocation, matching the UE
        relative gap's convention), plus the max per-origin ratio deviation. Zero
        iff every relevant origin matches its PAS's aggregate segment split.
        """
        total = 0.0
        max_dev = 0.0
        for pas in pool:
            if len(pas["origins"]) < 2:
                continue
            rows = self._segment_flows(pas, bushes)
            a_sum = sum(a for _, a, _ in rows)
            w_sum = sum(a + b for _, a, b in rows)
            if w_sum <= 0.0:
                continue
            pi = a_sum / w_sum
            for _, a, b in rows:
                w = a + b
                total += abs(a - w * pi)
                if w > 0.0:
                    max_dev = max(max_dev, abs(a / w - pi))
        resid = total / demand_total if demand_total > 0.0 else 0.0
        return resid, max_dev

    # ---------------------------------------------------------------- solve
    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        start = time.perf_counter()
        self._setup(scenario)
        network = self._network
        engine = self._engine
        inner_rounds = self.factor_values["inner_rounds"]
        prop_rounds = self.factor_values["prop_rounds"]
        bush_update_every = self.factor_values["bush_update_every"]
        demand_total = float(scenario.demand.total)

        v = np.zeros(self._n_links)
        bushes, v, sp_calls = self._initial_bushes(v)
        t = network.link_cost(v)
        dt = network.link_cost_derivative(v)

        k = 0
        while True:
            k += 1
            rounds = 0
            if (k - 1) % bush_update_every == 0:
                for bush, o in zip(bushes, self._origins, strict=True):
                    self._update_bush(bush, int(o), t)
                rounds += 1
            # Component 1+2: identify the PAS pool and cost-equilibrate it, once
            # per round, fine-interleaved (rebuild-each-round designs out stale
            # PASs). pool holds the last round's identification for the
            # proportionality pass and the residual.
            pool: list[dict] = []
            cost_rounds = 0
            for _ in range(inner_rounds):
                pool = self._identify_pas(bushes, t)
                moved_any = False
                for pas in pool:
                    if self._cost_equilibrate(pas, bushes, v, t, dt):
                        moved_any = True
                rounds += 1
                cost_rounds += 1
                if not moved_any:
                    break
            # Component 3: proportionality adjustments (SP-free redistribution;
            # leaves link flows and the UE gap unchanged, only the route split).
            prop_applied = 0
            for _ in range(prop_rounds):
                for pas in pool:
                    self._proportionality_adjust(pas, bushes)
                prop_applied += 1
            # Exact resync: emitted flows equal the bush aggregation bitwise.
            v = np.zeros(self._n_links)
            for bush in bushes:
                v += bush.x
            t = network.link_cost(v)
            dt = network.link_cost_derivative(v)

            sp_calls += rounds  # one all-origins scan / update / AON ~ one sp_call
            _, sptt = engine.all_or_nothing(t, scenario.demand)
            sp_calls += 1  # honest self-report needs one global Dijkstra
            tstt = float(v @ t)
            gap = (tstt - sptt) / tstt if tstt > 0 else 0.0
            prop_resid, prop_max = self._proportionality_residual(pool, bushes, demand_total)

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
                proportionality_residual=prop_resid,
                pas_proportionality_max=prop_max,
                pas_pool_size=float(len(pool)),
                pas_cost_shift_rounds=float(cost_rounds),
                pas_prop_rounds=float(prop_applied),
                bush_scan_rounds=float(rounds),
            )
            if budget.exhausted(coords) or budget.target_met(gap):
                break

        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
