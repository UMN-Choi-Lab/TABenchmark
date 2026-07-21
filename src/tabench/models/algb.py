"""Dial's Algorithm B (2006): bush-based deterministic user equilibrium.

Per origin, traffic is confined to an acyclic sub-network (a "bush") of the
full graph. Each bush is equilibrated by Newton flow shifts between the
longest- and shortest-used path segments into each node, and periodically
improved (drop unused links; add links that provably keep it acyclic).
Origins are swept Gauss-Seidel with *fine* interleaving: one shift pass per
origin per round, several rounds per iteration -- coarse interleaving
(equilibrating each bush fully before moving on) measurably stalls, because
the cross-origin coupling cycles even when every bush alone is equilibrated.

Dial, Transportation Research Part B 40(10):917-936 (2006) is the algorithm;
Nie, TR-B 44(1):73-89 (2010) is the bush-update rule used here (strict
min-and-max-label improvement with a max-label fallback). Both primary texts
are paywalled and were not read directly; every load-bearing formula was
cross-verified against four accessible sources that agree on all of them:
Boyles, Lownes & Unnikrishnan, *Transportation Network Analysis* v1.0 sec.
6.4.1-6.4.3 (labels eq. 6.49-6.54, Newton shift eq. 6.61, and the max-label
"modified-shortcut" add criterion U_i + t_ij < U_j with its acyclicity
proof); the CE392C slides 10-bushbased.pdf (Dial's longest-*used*-path
labels, U_i = max over used in-links); and two reference implementations,
TAP-B (spartalab/tap-b, src/bush.c) and TAsK (olga-perederieieva/TAsK,
DAGraphB.cpp/DAGraph.cpp, the code behind Perederieieva et al. 2015). The
max-label add criterion is attributed here to Boyles sec. 6.4.3 as its
"modified shortcut", not to Dial's primary text, pending a maintainer
spot-check of the two paywalled PDFs.

Bushes live on the PathEngine expanded graph, so restricted centroids (nodes
below ``first_thru_node``) can never be through-nodes of any bush path: the
node-split shadow head has no out-arcs, so no shift or bush-update candidate
can route through a centroid, with no runtime first-thru-node check needed.

Budget accounting (P6). One sp_call = one batched all-origins tree
computation. A bush-scan round (all origins; min and max DAG labels on the
restricted graph) IS such a computation, so each shift round, each
bush-update round, the initial Dijkstra, and the per-iteration honest-gap
all-or-nothing each cost one sp_call -- 12 per iteration at defaults (one
update + ten shift rounds + one gap). This is deliberately STRICTER than the
FW/gp precedent, which charges within-iteration flow-shift / path-re-pricing
sweeps nothing; under it algb's sp_calls axis is *not* comparable to fw/gp's
(it overcounts algb by ~11 sp_calls per iteration). Algorithm B's advantage
shows on the iterations axis instead (~5x fewer iterations than gp to a given
certified gap on Sioux Falls); sp_calls is reported honestly rather than
gamed by making label passes free (P6). The number of bush-scan rounds per
iteration is recorded separately as ``bush_scan_rounds`` for transparency.

The relative gap self-reported each iteration comes from the model's own
batched all-or-nothing at the resynced flows -- the same formula the harness
evaluator uses -- so it is byte-comparable to the certified value (the
FW-style honesty regression). Emitted link flows are rebuilt exactly from the
sum of bush flows before every checkpoint (the drift guard gp documents).

The acyclic-bush primitives (``_BushState``, ``_kahn``, ``_scan``,
``_initial_bushes``, ``_update_bush``) live in ``_bush._BushMachinery`` and are
shared with TAPAS; ``_shift_pass`` -- the Newton flow-shift on the
BLAS-sensitive tail-convergence path -- stays here. ``_BushState`` is re-exported
below so ``from tabench.models.algb import _BushState`` keeps working.
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
from ._bush import _BushMachinery, _BushState
from .base import TrafficAssignmentModel, register_model

__all__ = ["AlgorithmBModel", "_BushState"]


@register_model
class AlgorithmBModel(_BushMachinery, TrafficAssignmentModel):
    """Dial (2006) Algorithm B, origin-by-origin Gauss-Seidel over bushes."""

    name = "algb"
    capabilities = Capabilities(
        paradigm="static_ue",
        deterministic=True,
        provides_gap=True,
        seedable=True,
    )
    factors = {
        "inner_iterations": FactorSpec(
            default=10,
            kind="int",
            bounds=(1, 64),
            doc="Fine-interleaved shift rounds per iteration (one pass per origin "
            "per round; TAP-B innerIterations). Fewer rounds measurably stall: "
            "10 reaches 1e-14 in 28 iterations, while 3 and 2 plateau near 1e-9.",
        ),
        "newton_shifts": FactorSpec(
            default=1,
            kind="int",
            bounds=(1, 8),
            doc="Repeated Newton steps per divergent node per pass (TAP-B "
            "numNewtonShifts); >1 does not pay at inner_iterations=10 on Sioux Falls.",
        ),
        "bush_update_every": FactorSpec(
            default=1,
            kind="int",
            bounds=(1, 8),
            doc="Bush improvement (drop/add links) every k-th iteration; TAP-B "
            "updates every main iteration.",
        ),
        "alpha": FactorSpec(
            default=1.0,
            kind="float",
            bounds=(1e-6, 1.0),
            doc="Newton step scaling (TAP-B newtonStep); 1 is Dial's choice and is "
            "stable here thanks to the per-shift cost refresh.",
        ),
        "drop_tol": FactorSpec(
            default=1e-13,
            kind="float",
            bounds=(1e-16, 1e-6),
            doc="Bush-flow threshold for dropping links and for the used-link "
            "eligibility of max labels (TAP-B minLinkFlow).",
        ),
    }

    # --------------------------------------------------------------- setup
    def _setup(self, scenario: Scenario) -> None:
        self._setup_bush_graph(scenario)  # shared expanded-graph state (_bush)
        self._drop_tol = self.factor_values["drop_tol"]
        self._alpha = self.factor_values["alpha"]
        self._newton_shifts = self.factor_values["newton_shifts"]

    # --------------------------------------------------------------- shifts
    def _shift_pass(
        self, bush: _BushState, origin_idx: int, v: np.ndarray, t: np.ndarray, dt: np.ndarray
    ) -> bool:
        """One reverse-topological Newton pass (Boyles eq. 6.61; TAP-B pass).

        Costs and derivatives are refreshed on the affected links after every
        pair shift (TAP-B exactCostUpdate / TAsK link->updateTime); the L/U
        trees stay as scanned for the whole pass. Returns True if flow moved.
        """
        L, U, minp, maxp = self._scan(bush, origin_idx, t, "used")
        network = self._network
        pos = bush.topo_pos
        moved = False
        for j in bush.topo[::-1]:
            if j == origin_idx:
                continue
            kmin = minp[j]
            kmax = maxp[j]
            if kmin < 0 or kmax < 0 or kmin == kmax:
                continue
            if not (U[j] - L[j] > 0.0):
                continue
            # Walk back to the divergence node, advancing the topologically
            # later chain so the two chains provably meet (TAsK performFlowMove).
            seg_min: list[int] = []
            seg_max: list[int] = []
            i_min = i_max = int(j)
            first = True
            while i_min != i_max or first:
                if first:
                    seg_min.append(int(kmin))
                    seg_max.append(int(kmax))
                    i_min = int(self._tails[kmin])
                    i_max = int(self._tails[kmax])
                    first = False
                    continue
                if pos[i_max] > pos[i_min]:
                    k = maxp[i_max]
                    if k < 0:
                        break  # max chain exhausted (no used approach)
                    seg_max.append(int(k))
                    i_max = int(self._tails[k])
                else:
                    k = minp[i_min]
                    if k < 0:
                        break
                    seg_min.append(int(k))
                    i_min = int(self._tails[k])
            if i_min != i_max:
                continue
            sm = np.asarray(seg_min, dtype=np.int64)
            sx = np.asarray(seg_max, dtype=np.int64)
            aff = np.concatenate([sm, sx])
            for _ in range(self._newton_shifts):
                c_min = float(t[sm].sum())
                c_max = float(t[sx].sum())
                excess = c_max - c_min
                if excess <= 0.0:
                    break
                cap = float(bush.x[sx].min())  # only this origin's flow may move
                if cap <= 0.0:
                    break
                denom = float(dt[sm].sum() + dt[sx].sum())
                if denom <= 0.0:
                    # A zero derivative need not mean a flat direction: BPR
                    # links with power > 1 have t' = 0 only at v = 0. Re-probe
                    # at the would-be arrival flows (gp's fix); full-shift only
                    # if the direction is still flat there.
                    probe = v.copy()
                    probe[sx] -= cap
                    probe[sm] += cap
                    pd = network.link_cost_derivative(probe)
                    denom = float(pd[sm].sum() + pd[sx].sum())
                    shift = cap if denom <= 0.0 else min(cap, self._alpha * excess / denom)
                else:
                    shift = min(cap, self._alpha * excess / denom)
                if shift <= 0.0:
                    break
                bush.x[sx] -= shift
                bush.x[sm] += shift
                np.maximum(bush.x, 0.0, out=bush.x)  # clamp float dust at caps
                v[sx] -= shift
                v[sm] += shift
                v[aff] = np.maximum(v[aff], 0.0)
                t[aff] = network.link_cost(v)[aff]
                dt[aff] = network.link_cost_derivative(v)[aff]
                moved = True
        return moved

    # ---------------------------------------------------------------- solve
    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        start = time.perf_counter()
        self._setup(scenario)
        network = self._network
        engine = self._engine
        inner_iterations = self.factor_values["inner_iterations"]
        bush_update_every = self.factor_values["bush_update_every"]

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
            # Fine-interleaved Gauss-Seidel: one shift pass per origin per
            # round (TAP-B updateBatchFlows structure). Coarse interleaving
            # (many consecutive passes on one origin) measurably stalls.
            for _ in range(inner_iterations):
                moved_any = False
                for bush, o in zip(bushes, self._origins, strict=True):
                    if self._shift_pass(bush, int(o), v, t, dt):
                        moved_any = True
                rounds += 1
                if not moved_any:
                    break
            # Exact resync: emitted flows equal the bush aggregation bitwise.
            v = np.zeros(self._n_links)
            for bush in bushes:
                v += bush.x
            t = network.link_cost(v)
            dt = network.link_cost_derivative(v)

            sp_calls += rounds  # one all-origins bush-scan round ~ one AON sweep
            _, sptt = engine.all_or_nothing(t, scenario.demand)
            sp_calls += 1  # honest self-report needs one global Dijkstra
            tstt = float(v @ t)
            gap = (tstt - sptt) / tstt if tstt > 0 else 0.0

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
