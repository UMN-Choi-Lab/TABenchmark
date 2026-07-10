"""Harness-side certification of emitted Merchant-Nemhauser trajectories (P1).

Every scored quantity is recomputed here as a pure function of
``(SODTAScenario, DTATrajectory)`` — the solver's self-reported objective is
never trusted. Feasibility gates recompute, from the emitted
inflow/exit/occupancy arrays alone: link conservation
``x(t+1) = x(t) + u(t) - e(t)``, per-period node balance
``sum u_out = d + sum e_in`` at every non-destination node, the exit-function
bound ``e_a(t) <= g_a(x_a(t))`` (``g`` re-evaluated from the scenario's pieces),
the empty initial network, nonnegativity, terminal clearance ``x(T) = 0``
(stranded flow makes total cost ill-posed and is CENSORED), and full delivery
into the destination. Gates run at TWO scales — the adversarial review caught
that a single per-cell tolerance scaled by total demand lets ~eps-sized
residuals aggregate over the ``T * (links + nodes)`` cells into a material
teleport that certified below the true optimum. So per-cell residuals are
bounded by ``tol * max(1, demand.max())`` (a local scale) AND each violation
family's network-wide ABSOLUTE sum is bounded by the aggregate mass budget
``tol * max(1, demand.sum())``, capping total conjured/vanished mass at the
same relative ``tol`` the score claims. Costs are computed on occupancies
clamped at zero so sub-zero noise can never buy cost, and — the weak-duality
backstop — a trajectory whose recomputed cost undercuts the harness's own LP
optimum ``Z*`` by more than ``tol`` is CENSORED: no feasible plan can beat
``Z*``, so undercutting it is itself a proof of infeasibility. Semantics mirror
``gaps.py``/``transit_gaps.py``/``bottleneck_gaps.py``: infeasible trajectories
are censored (``feasible = 0``, scored quantities NaN); only wrong shapes raise.

Scoring: ``total_cost = sum_t sum_a w_a * max(x_a(t), 0)`` and

    so_optimality_gap = (total_cost - Z*) / Z*

where ``Z*`` is the canonical-LP optimum the harness resolves itself (HiGHS,
eagerly at construction — an unclearable scenario is a configuration error
raised from ``__init__``, never a crash at scoring time; ``Z* > 0`` always,
since every vehicle spends at least one period on a positively-weighted link).
If the trajectory carries an LP dual certificate the harness ADDITIONALLY
verifies global optimality by pure arithmetic — dual feasibility
(``y_ub <= 0``, reduced costs ``c - A_eq'y_eq - A_ub'y_ub >= 0``) plus the
duality gap against the RECOMPUTED primal cost — reported as ``dual_gap`` /
``dual_infeasibility`` (NaN when no certificate is emitted; wrong certificates
show up as large values, they are never believed). ``exit_slack_max`` is a
Tier-B diagnostic: the largest ``g_a(x) - e_a`` on occupied links, i.e. how
much the plan "holds back" flow — legitimate, sometimes strictly optimal
SO behaviour under the Carey relaxation, not an error.
"""

from __future__ import annotations

import logging

import numpy as np

from ..dta.scenario import SODTAScenario
from ..dta.solve import DTATrajectory, canonical_lp

__all__ = ["SODTAEvaluator"]

logger = logging.getLogger(__name__)

_SCORED = (
    "so_optimality_gap",
    "total_cost",
    "max_occupancy",
    "exit_slack_max",
    "dual_gap",
    "dual_infeasibility",
)


class SODTAEvaluator:
    """Model-blind M-N certifier: pure function of ``(scenario, trajectory)``.

    Raises ``ValueError`` at construction if the scenario's canonical LP cannot
    be solved (horizon too short to clear the demand) — a configuration error,
    surfaced before any model output is scored.
    """

    def __init__(self, scenario: SODTAScenario, tol: float = 1e-6) -> None:
        from scipy.optimize import linprog

        self.scenario = scenario
        self.tol = float(tol)
        self._hash = scenario.content_hash()
        self._lp = canonical_lp(scenario)
        c, a_eq, b_eq, a_ub, b_ub = self._lp
        res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, method="highs")
        if res.status != 0:
            raise ValueError(
                f"canonical LP unsolvable for '{scenario.name}' (status {res.status}: "
                f"{res.message}) — the horizon T cannot clear the demand"
            )
        self._z_star = float(res.fun)

    def _censored(self, reason: str) -> dict[str, float]:
        logger.info("SO-DTA trajectory censored: %s", reason)
        metrics = dict.fromkeys(_SCORED, float("nan"))
        metrics["feasible"] = 0.0
        return metrics

    def certify(self, trajectory: DTATrajectory) -> dict[str, float]:
        sc = self.scenario
        u, e, x = trajectory.inflows, trajectory.exits, trajectory.occupancies
        n_t, n_l = sc.n_periods, sc.n_links
        if u.shape != (n_t, n_l) or x.shape != (n_t + 1, n_l):
            raise ValueError(
                f"DTATrajectory shape mismatch: inflows {u.shape}, occupancies "
                f"{x.shape}, scenario wants T={n_t}, n_links={n_l}"
            )
        if trajectory.scenario_hash != self._hash:
            return self._censored(
                f"wrong scenario hash: trajectory ran on {trajectory.scenario_hash!r}, "
                f"this instance is {self._hash!r}"
            )
        if not (np.isfinite(u).all() and np.isfinite(e).all() and np.isfinite(x).all()):
            return self._censored("non-finite inflows/exits/occupancies")

        # Two-scale gates: per-cell residuals at the LOCAL scale, each violation
        # family's absolute network-wide sum at the aggregate mass budget.
        eps = self.tol * max(1.0, float(sc.demand.max()))
        budget = self.tol * max(1.0, float(sc.demand.sum()))
        if min(u.min(), e.min(), x.min()) < -eps:
            return self._censored("negative inflow/exit/occupancy")
        if np.abs(x[0]).max() > eps:
            return self._censored("initial network must be empty (x(0) = 0)")
        cons = x[1:] - x[:-1] - u + e
        if np.abs(cons).max() > eps or np.abs(cons).sum() > budget:
            return self._censored(
                f"link conservation violated (max residual {np.abs(cons).max():.3e}, "
                f"total {np.abs(cons).sum():.3e})"
            )
        node_total = 0.0
        for j in range(sc.n_nodes):
            if j == sc.destination:
                continue
            u_out = u[:, sc.link_tail == j].sum(axis=1)
            e_in = e[:, sc.link_head == j].sum(axis=1)
            resid = np.abs(u_out - sc.demand[:, j] - e_in)
            node_total += float(resid.sum())
            if resid.max() > eps:
                return self._censored(
                    f"node balance violated at node {j} (max residual {resid.max():.3e})"
                )
        if node_total > budget:
            return self._censored(
                f"node balance violated in aggregate (total residual {node_total:.3e})"
            )
        g = sc.exit_flow(x[:-1])  # exit bound from START-of-period occupancy
        excess = np.clip(e - g, 0.0, None)
        if excess.max() > eps or excess.sum() > budget:
            return self._censored(
                f"exit-function bound violated (max excess {excess.max():.3e}, "
                f"total {excess.sum():.3e})"
            )
        if x[-1].max() > eps or x[-1].sum() > budget:
            return self._censored(
                f"stranded flow: x(T) has {x[-1].sum():.3e} vehicles left in the network"
            )
        delivered = float(e[:, sc.link_head == sc.destination].sum())
        if abs(delivered - float(sc.demand.sum())) > budget:
            return self._censored(
                f"delivery mismatch: {delivered!r} arrived, demand is {sc.demand.sum()!r}"
            )

        # Score on clamped occupancies (sub-zero noise never buys cost) against
        # the harness's own LP optimum; Z* > 0 is guaranteed by validation.
        total_cost = float(np.sum(np.maximum(x[1:], 0.0) * sc.cost_weights))
        z_star = self._z_star
        gap = (total_cost - z_star) / z_star
        if gap < -self.tol:
            return self._censored(
                f"cost {total_cost!r} undercuts the certified LP optimum {z_star!r} "
                "beyond tolerance — by weak duality no feasible plan can, so this is "
                "a proof of infeasibility"
            )
        occupied = x[:-1] > eps
        slack = np.where(occupied, g - e, 0.0)
        metrics = {
            "feasible": 1.0,
            "so_optimality_gap": gap,
            "total_cost": total_cost,
            "max_occupancy": float(x.max()),
            "exit_slack_max": float(slack.max(initial=0.0)),
            "dual_gap": float("nan"),
            "dual_infeasibility": float("nan"),
        }
        if trajectory.duals is not None:
            metrics.update(self._verify_dual_certificate(trajectory, total_cost))
        return metrics

    def _verify_dual_certificate(
        self, trajectory: DTATrajectory, total_cost: float
    ) -> dict[str, float]:
        """Pure-arithmetic weak-duality check of the emitted certificate: if
        ``y_ub <= 0`` and reduced costs are >= 0, then ``b_eq'y_eq + b_ub'y_ub``
        lower-bounds EVERY feasible cost; a zero gap against the recomputed
        primal cost therefore certifies global optimality without solving."""
        assert trajectory.duals is not None
        c, a_eq, b_eq, a_ub, b_ub = self._lp
        y_eq, y_ub = trajectory.duals["eq"], trajectory.duals["ub"]
        if y_eq.shape != b_eq.shape or y_ub.shape != b_ub.shape:
            return {"dual_gap": float("inf"), "dual_infeasibility": float("inf")}
        if not (np.isfinite(y_eq).all() and np.isfinite(y_ub).all()):
            return {"dual_gap": float("inf"), "dual_infeasibility": float("inf")}
        reduced = c - a_eq.T @ y_eq - a_ub.T @ y_ub
        infeas = max(0.0, float(y_ub.max(initial=0.0)), float(-reduced.min()))
        dual_obj = float(b_eq @ y_eq + b_ub @ y_ub)
        return {
            "dual_gap": (total_cost - dual_obj) / self._z_star,
            "dual_infeasibility": infeas,
        }
