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

from ..dta.cells import CellSODTAScenario, CellTrajectory, cell_canonical_lp
from ..dta.scenario import SODTAScenario
from ..dta.solve import DTATrajectory, canonical_lp

__all__ = ["SODTAEvaluator", "CellSODTAEvaluator"]

logger = logging.getLogger(__name__)

_SCORED = (
    "so_optimality_gap",
    "total_cost",
    "max_occupancy",
    "exit_slack_max",
    "dual_gap",
    "dual_infeasibility",
)

_CELL_SCORED = (
    "so_optimality_gap",
    "total_cost",
    "max_occupancy",
    "holding_max",
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
        # Clip y_ub at 0 BEFORE computing the bound: a sign violation must not
        # inflate the dual objective through large b entries (the review's
        # scale-blindness finding — +1e-12 on a 1e12-sized spillback row moved
        # the "certified" bound by 56x tol while infeasibility read 1e-12).
        # The clipped vector is sign-feasible by construction, the bound it
        # certifies is conservative, and the raw violation is still reported.
        y_ub_eff = np.minimum(y_ub, 0.0)
        reduced = c - a_eq.T @ y_eq - a_ub.T @ y_ub_eff
        infeas = max(0.0, float(y_ub.max(initial=0.0)), float(-reduced.min()))
        dual_obj = float(b_eq @ y_eq + b_ub @ y_ub_eff)
        return {
            "dual_gap": (total_cost - dual_obj) / self._z_star,
            "dual_infeasibility": infeas,
        }


class CellSODTAEvaluator:
    """Model-blind Ziliaskopoulos cell certifier: pure function of
    ``(scenario, trajectory)`` — same two-scale, weak-duality-backstopped
    design as :class:`SODTAEvaluator` (adr-021 inherits the adr-020 review
    hardening). Gates recompute conservation (with the exogenous demand and
    the absorbing sink), the four aggregate CTM bound families, the ``x <= N``
    envelope, the initial condition, terminal clearance, and delivery into the
    sink; scored quantities are the clamped-occupancy total cost, the gap
    against an eagerly harness-resolved LP optimum ``Z*``, and the
    pure-arithmetic dual-certificate checks. ``holding_max`` is a Tier-B
    diagnostic — the largest headroom left unused by a connector whose tail
    still queues (LP "traffic holding"): legitimate on the optimal face for a
    single destination, never an error.

    Raises ``ValueError`` at construction if the canonical LP is unsolvable
    (the horizon cannot clear the demand) — a configuration error, never a
    scoring-time crash.
    """

    def __init__(self, scenario: CellSODTAScenario, tol: float = 1e-6) -> None:
        from scipy.optimize import linprog

        self.scenario = scenario
        self.tol = float(tol)
        self._hash = scenario.content_hash()
        self._lp = cell_canonical_lp(scenario)
        c, a_eq, b_eq, a_ub, b_ub = self._lp
        res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, method="highs")
        if res.status != 0:
            raise ValueError(
                f"canonical cell LP unsolvable for '{scenario.name}' (status "
                f"{res.status}: {res.message}) — the horizon T cannot clear the demand"
            )
        self._z_star = float(res.fun)

    def _censored(self, reason: str) -> dict[str, float]:
        logger.info("cell SO-DTA trajectory censored: %s", reason)
        metrics = dict.fromkeys(_CELL_SCORED, float("nan"))
        metrics["feasible"] = 0.0
        return metrics

    def certify(self, trajectory: CellTrajectory) -> dict[str, float]:
        sc = self.scenario
        x, y = trajectory.occupancies, trajectory.flows
        n_t, n_c, n_e = sc.n_periods, sc.n_cells, sc.n_conns
        if x.shape != (n_t + 1, n_c) or y.shape != (n_t, n_e):
            raise ValueError(
                f"CellTrajectory shape mismatch: occupancies {x.shape}, flows "
                f"{y.shape}, scenario wants T={n_t}, n_cells={n_c}, n_conns={n_e}"
            )
        if trajectory.scenario_hash != self._hash:
            return self._censored(
                f"wrong scenario hash: trajectory ran on {trajectory.scenario_hash!r}, "
                f"this instance is {self._hash!r}"
            )
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            return self._censored("non-finite occupancies/flows")

        total_mass = float(sc.demand.sum() + sc.initial_occupancy.sum())
        eps = self.tol * max(
            1.0, float(sc.demand.max(initial=0.0)), float(sc.initial_occupancy.max())
        )
        budget = self.tol * max(1.0, total_mass)
        neg = float(np.clip(-x, 0.0, None).sum() + np.clip(-y, 0.0, None).sum())
        if min(x.min(), y.min()) < -eps or neg > budget:
            return self._censored("negative occupancy/flow")
        # Two-scale here too: the adversarial review caught that a per-cell-only
        # initial gate was the ONE unbudgeted door — with eps scaled by a large
        # x0, whole vehicles could be deleted at loaded cells and conjured
        # beside the sink (delivery nets out, conservation sees the CLAIMED
        # x[0]), resurrecting the adr-020 teleport at ~500x tol.
        init_diff = np.abs(x[0] - sc.initial_occupancy)
        if init_diff.max() > eps or init_diff.sum() > budget:
            return self._censored("occupancies[0] must equal the initial condition")

        inflow = np.zeros((n_t, n_c))
        outflow = np.zeros((n_t, n_c))
        for c in range(n_e):
            outflow[:, sc.conn_tail[c]] += y[:, c]
            inflow[:, sc.conn_head[c]] += y[:, c]
        cons = x[1:] - x[:-1] - sc.demand - inflow + outflow
        if np.abs(cons).max() > eps or np.abs(cons).sum() > budget:
            return self._censored(
                f"cell conservation violated (max residual {np.abs(cons).max():.3e}, "
                f"total {np.abs(cons).sum():.3e})"
            )
        # the four aggregate CTM bound families, each at both scales
        send_occ = np.clip(outflow - x[:-1], 0.0, None)
        send_cap = np.clip(outflow - sc.capacity, 0.0, None)
        recv_cap = np.clip(inflow - sc.capacity, 0.0, None)
        finite_n = np.isfinite(sc.storage)
        space = np.where(finite_n, sc.delta * (sc.storage - x[:-1]), np.inf)
        recv_space = np.clip(inflow - space, 0.0, None)
        for label, exc in (
            ("sending-occupancy", send_occ),
            ("sending-capacity", send_cap),
            ("receiving-capacity", recv_cap),
            ("receiving-space (spillback)", recv_space),
        ):
            if exc.max() > eps or exc.sum() > budget:
                return self._censored(
                    f"{label} bound violated (max excess {exc.max():.3e}, "
                    f"total {exc.sum():.3e})"
                )
        overfill = np.clip(x - np.where(finite_n, sc.storage, np.inf), 0.0, None)
        if overfill.max() > eps:
            return self._censored(
                f"occupancy exceeds storage (max excess {overfill.max():.3e})"
            )
        non_sink = np.arange(n_c) != sc.sink
        if x[-1, non_sink].max() > eps or x[-1, non_sink].sum() > budget:
            return self._censored(
                f"stranded flow: non-sink x(T) totals {x[-1, non_sink].sum():.3e}"
            )
        if abs(float(x[-1, sc.sink]) - total_mass) > budget:
            return self._censored(
                f"delivery mismatch: sink holds {x[-1, sc.sink]!r} at T, total "
                f"demand is {total_mass!r}"
            )

        total_cost = float(np.sum(np.maximum(x[:-1, non_sink], 0.0)))
        z_star = self._z_star
        gap = (total_cost - z_star) / z_star
        if gap < -self.tol:
            return self._censored(
                f"cost {total_cost!r} undercuts the certified LP optimum {z_star!r} "
                "beyond tolerance — by weak duality no feasible plan can, so this "
                "is a proof of infeasibility"
            )
        # Tier-B holding diagnostic: headroom below ALL four bounds on a
        # connector whose tail still queues after this interval's outflow.
        holding = 0.0
        for c in range(n_e):
            i, j = int(sc.conn_tail[c]), int(sc.conn_head[c])
            oth_out = outflow[:, i] - y[:, c]
            oth_in = inflow[:, j] - y[:, c]
            room = np.minimum(x[:-1, i] - oth_out, sc.capacity[i] - oth_out)
            room = np.minimum(room, sc.capacity[j] - oth_in)
            if finite_n[j]:
                room = np.minimum(room, space[:, j] - oth_in)
            queued = x[:-1, i] - outflow[:, i] > eps
            held = np.where(queued, room - y[:, c], 0.0)
            holding = max(holding, float(held.max(initial=0.0)))
        metrics = {
            "feasible": 1.0,
            "so_optimality_gap": gap,
            "total_cost": total_cost,
            "max_occupancy": float(x[:, non_sink].max()),
            "holding_max": holding,
            "dual_gap": float("nan"),
            "dual_infeasibility": float("nan"),
        }
        if trajectory.duals is not None:
            metrics.update(self._verify_dual_certificate(trajectory, total_cost))
        return metrics

    def _verify_dual_certificate(
        self, trajectory: CellTrajectory, total_cost: float
    ) -> dict[str, float]:
        """Identical pure-arithmetic weak-duality check to
        :meth:`SODTAEvaluator._verify_dual_certificate`, against the cell LP."""
        assert trajectory.duals is not None
        c, a_eq, b_eq, a_ub, b_ub = self._lp
        y_eq, y_ub = trajectory.duals["eq"], trajectory.duals["ub"]
        if y_eq.shape != b_eq.shape or y_ub.shape != b_ub.shape:
            return {"dual_gap": float("inf"), "dual_infeasibility": float("inf")}
        if not (np.isfinite(y_eq).all() and np.isfinite(y_ub).all()):
            return {"dual_gap": float("inf"), "dual_infeasibility": float("inf")}
        # Clip y_ub at 0 BEFORE computing the bound: a sign violation must not
        # inflate the dual objective through large b entries (the review's
        # scale-blindness finding — +1e-12 on a 1e12-sized spillback row moved
        # the "certified" bound by 56x tol while infeasibility read 1e-12).
        # The clipped vector is sign-feasible by construction, the bound it
        # certifies is conservative, and the raw violation is still reported.
        y_ub_eff = np.minimum(y_ub, 0.0)
        reduced = c - a_eq.T @ y_eq - a_ub.T @ y_ub_eff
        infeas = max(0.0, float(y_ub.max(initial=0.0)), float(-reduced.min()))
        dual_obj = float(b_eq @ y_eq + b_ub @ y_ub_eff)
        return {
            "dual_gap": (total_cost - dual_obj) / self._z_star,
            "dual_infeasibility": infeas,
        }
