"""Harness-side certification of emitted transit strategies (P1 for transit).

Every scored quantity is recomputed here as a pure function of
``(TransitScenario bytes, TransitStrategy arc volumes)``; the model's
self-reported labels/costs are never trusted (docs/design/adr-014). Semantics
mirror ``metrics/gaps.py`` and ``metrics/dnl_gaps.py``:

* the optimal-strategy assignment is a convex LP with objective
  ``Z = sum_a c_a v_a + sum_i w_i`` (in-vehicle time + total expected wait);
* the harness recomputes the LP optimum ``Z*`` independently from the scenario
  (Spiess & Florian 1989, the label-setting algorithm), exactly as the road
  certifier recomputes an all-or-nothing bound;
* the SCORED gap is the emitted primal cost ``Z_emitted`` (recomputed from the
  emitted arc volumes) versus ``Z*``: ``optimality_gap = (Z_emitted - Z*)/Z*``,
  ``>= 0`` and ``0`` iff the emitted assignment is optimal.

The wait is computed per commodity: it is per-(node, destination), and at each
node the LP-minimal feasible wait for a fixed arc-volume split is
``w_i = max_a v_a / f_a`` (every out-arc must satisfy ``f_a w_i >= v_a``), which
keeps ``Z_emitted >= Z*`` — so ``optimality_gap`` is provably ``>= 0`` even for a
non-proportional split (that split is feasible but suboptimal: its larger ``w_i``
raises the gap rather than being censored). Certification therefore gates on
demand feasibility alone — nonnegative arc volumes that conserve each
destination's demand; a non-conserving flow is censored (``feasible = 0``, NaN
gap). Because the wait is per-destination, certifying a multi-destination
scenario needs the per-destination arc-volume decomposition
(``TransitStrategy.dest_arc_volumes``); the summed arc volumes suffice only for a
single-destination scenario.
"""

from __future__ import annotations

import numpy as np

from ..transit.network import TransitScenario, TransitStrategy
from ..transit.strategy import optimal_strategy

__all__ = ["TransitEvaluator"]


class TransitEvaluator:
    """Model-blind certifier for one transit scenario."""

    _CLIP_TOL = 1e-9

    def __init__(self, scenario: TransitScenario, feasibility_tol: float = 1e-6) -> None:
        self.scenario = scenario
        self.feasibility_tol = feasibility_tol
        net = scenario.network
        dem = scenario.demand
        self._total_demand = dem.total
        # Per-DESTINATION net demand: production (origin) minus attraction, one
        # vector per distinct destination. The wait term is per-(node,
        # destination), so certification must be done per commodity, not on the
        # destination-summed arc volumes.
        self._dest_net_demand: dict[int, np.ndarray] = {}
        for d in np.unique(dem.destinations):
            d = int(d)
            mask = dem.destinations == d
            nd = np.bincount(dem.origins[mask], weights=dem.volumes[mask], minlength=net.n_nodes)
            nd[d] -= float(dem.volumes[mask].sum())  # all attraction is at the destination
            self._dest_net_demand[d] = nd
        # Out-arcs per node (for the per-node split / wait recomputation).
        self._out_arcs: list[list[int]] = [[] for _ in range(net.n_nodes)]
        for a in range(net.n_arcs):
            self._out_arcs[int(net.tail[a])].append(a)
        # Harness-recomputed LP optimum (from public content-hashed scenario data).
        opt = optimal_strategy(scenario)
        self._z_star = float((dem.volumes * opt.pair_costs).sum())

    def _censored(self, reason: str) -> dict[str, float]:
        return {
            "feasible": 0.0,
            "expected_cost": self._z_star / self._total_demand if self._total_demand > 0 else 0.0,
            "optimal_total_cost": self._z_star,
            "total_expected_cost": float("nan"),
            "optimality_gap": float("nan"),
            "conservation_residual": float("inf"),
        }

    def _dest_wait(self, vd: np.ndarray) -> float:
        """Total expected wait for one destination's arc volumes (the wait is
        per-(node, destination)).

        The LP waiting constraint ``f_a w_i >= v_a`` must hold for EVERY out-arc,
        so the minimal feasible wait at node ``i`` is ``w_i = max_a v_a / f_a``
        (a deterministic ``f_a = inf`` arc contributes 0). Using this max — not
        ``V_i / F_i`` over a tolerance-thresholded subset of arcs — keeps the
        emitted primal cost, and hence ``optimality_gap``, provably ``>= 0`` even
        when an arc carries a sub-tolerance sliver on a near-zero-frequency line.
        A non-proportional split is then feasible but suboptimal (a larger ``w_i``
        raises the gap), rather than being censored.
        """
        net = self.scenario.network
        total_wait = 0.0
        for i in range(net.n_nodes):
            arcs = self._out_arcs[i]
            if arcs:
                total_wait += max(float(vd[a]) / float(net.freq[a]) for a in arcs)  # inf -> 0
        return total_wait

    def certify(self, strategy: TransitStrategy) -> dict[str, float]:
        net = self.scenario.network
        v = np.asarray(strategy.arc_volumes, dtype=np.float64)
        if v.shape != (net.n_arcs,):
            raise ValueError(f"arc_volumes shape {v.shape} != ({net.n_arcs},)")
        if not np.all(np.isfinite(v)):
            return self._censored("non-finite arc volumes")
        scale = max(1.0, float(np.abs(v).max()))
        if v.min() < -self._CLIP_TOL * scale:
            return self._censored("negative arc volumes")
        v = np.maximum(v, 0.0)

        tol = self.feasibility_tol * max(1.0, self._total_demand)

        # Per-destination decomposition (the wait is per-(node, destination), so
        # the summed arc_volumes cannot certify a multi-destination scenario). Fall
        # back to the aggregate only when the scenario has a single destination.
        n_dest = len(self._dest_net_demand)
        dav = strategy.dest_arc_volumes
        if not dav:
            if n_dest == 0:
                # No demand: the only feasible flow is empty; phantom flow censored.
                if float(np.abs(v).max()) > tol:
                    return self._censored("phantom flow on a zero-demand scenario")
                dav = ()
            elif n_dest == 1:
                dav = ((next(iter(self._dest_net_demand)), v),)
            else:
                return self._censored("multi-destination scenario requires per-destination volumes")
        else:
            if {int(d) for d, _ in dav} != set(self._dest_net_demand):
                return self._censored("dest_arc_volumes destinations mismatch the scenario")
            stacked = np.zeros(net.n_arcs)
            for _, vd in dav:
                stacked += np.asarray(vd, dtype=np.float64)
            if not np.allclose(stacked, v, atol=tol, rtol=0.0):
                return self._censored("aggregate arc_volumes != sum of per-destination volumes")

        z_emitted = 0.0
        conservation_residual = 0.0
        for d, vd in dav:
            vd = np.asarray(vd, dtype=np.float64)
            if vd.shape != (net.n_arcs,) or not np.all(np.isfinite(vd)):
                return self._censored("bad per-destination arc volumes")
            if vd.min() < -self._CLIP_TOL * scale:
                return self._censored("negative per-destination arc volumes")
            vd = np.maximum(vd, 0.0)
            # Per-commodity conservation: v^d routes exactly destination d's demand.
            outflow = np.bincount(net.tail, weights=vd, minlength=net.n_nodes)
            inflow = np.bincount(net.head, weights=vd, minlength=net.n_nodes)
            bal = float(np.abs(outflow - inflow - self._dest_net_demand[int(d)]).max())
            conservation_residual = max(conservation_residual, bal)
            z_emitted += float((net.time * vd).sum()) + self._dest_wait(vd)

        # Feasibility gates on demand conservation only; a non-proportional split
        # is feasible-but-suboptimal (the LP-minimal wait charges it a positive
        # gap), not censored.
        if conservation_residual > tol:
            metrics = self._censored("failed transit feasibility audit")
            metrics["conservation_residual"] = conservation_residual
            metrics["total_expected_cost"] = z_emitted
            return metrics

        return {
            "feasible": 1.0,
            "expected_cost": self._z_star / self._total_demand if self._total_demand > 0 else 0.0,
            "optimal_total_cost": self._z_star,
            "total_expected_cost": z_emitted,
            "optimality_gap": (
                (z_emitted - self._z_star) / self._z_star if self._z_star > 0 else 0.0
            ),
            "conservation_residual": conservation_residual,
        }
