"""Harness-side certification of emitted flows (P1).

Every scored metric is recomputed here from ``(scenario, link_flows)``; model
self-reports are never trusted. Definitions (single source of truth — see
docs/ARCHITECTURE.md section 2):

* ``TSTT(v) = sum_a v_a t_a(v_a)``
* ``SPTT(v) = sum_a y_a t_a(v_a)`` with ``y`` the all-or-nothing assignment
  at the costs induced by ``v``
* relative gap ``RG = (TSTT - SPTT) / TSTT``
* average excess cost ``AEC = (TSTT - SPTT) / total demand`` (the convention
  used by the TransportationNetworks best-known solutions)
* Beckmann objective ``B(v) = sum_a integral_0^{v_a} t_a(s) ds``

Certification is gated by a **demand-aware feasibility audit** (P7): a flow
vector only receives a gap if it (a) is finite and nonnegative, (b) conserves
flow at every intersection, AND (c) actually routes the scenario's demand —
each zone's net flow must match its productions/attractions from the OD
matrix. Flows failing the audit are *censored*: ``feasible = 0`` and the gap
metrics are NaN. Without (c), an all-zero "model" would certify with a
perfect gap; with it, unrouted or phantom demand is caught.

The audit checks the aggregate (single-commodity) flow-conservation
conditions, which are necessary but not sufficient for multi-OD feasibility;
as an additional necessary condition, a negative excess cost
(``SPTT > TSTT``) — impossible for truly demand-feasible flows — is also
censored.
"""

from __future__ import annotations

import numpy as np

from ..core.scenario import Scenario
from ..models._paths import PathEngine

__all__ = ["Evaluator", "node_balance_residual"]


def node_balance_residual(scenario: Scenario, link_flows: np.ndarray) -> float:
    """Maximum absolute demand-aware flow-conservation residual over all nodes.

    Non-zone nodes must conserve flow exactly. Zone node ``i`` must satisfy
    ``inflow_i - outflow_i = attractions_i - productions_i`` where productions
    and attractions are the off-diagonal row/column sums of the OD matrix
    (intrazonal demand never enters the network).
    """
    net = scenario.network
    od = scenario.demand.matrix
    v = np.asarray(link_flows, dtype=np.float64)
    inflow = np.bincount(net.term_node - 1, weights=v, minlength=net.n_nodes)
    outflow = np.bincount(net.init_node - 1, weights=v, minlength=net.n_nodes)
    balance = inflow - outflow

    off_diagonal = od - np.diag(np.diag(od))
    productions = off_diagonal.sum(axis=1)
    attractions = off_diagonal.sum(axis=0)
    expected = np.zeros(net.n_nodes)
    expected[: net.n_zones] = attractions - productions

    residual = np.abs(balance - expected)
    return float(residual.max()) if residual.size else 0.0


class Evaluator:
    """Model-blind scorer for one scenario. Reuse across checkpoints."""

    #: negative flows within this (relative) tolerance are clipped as noise
    _CLIP_TOL = 1e-9

    def __init__(self, scenario: Scenario, feasibility_tol: float = 1e-6) -> None:
        self.scenario = scenario
        self.feasibility_tol = feasibility_tol
        self._engine = PathEngine(scenario.network)
        self._total_demand = scenario.demand.total

    def _censored(self, reason: str) -> dict[str, float]:
        return {
            "tstt": float("nan"),
            "sptt": float("nan"),
            "relative_gap": float("nan"),
            "average_excess_cost": float("nan"),
            "beckmann_objective": float("nan"),
            "node_balance_residual": float("inf"),
            "feasible": 0.0,
        }

    def evaluate(self, link_flows: np.ndarray) -> dict[str, float]:
        """Certified metrics for one emitted flow state.

        Infeasible or invalid flows are censored (``feasible=0``, NaN gaps),
        never scored and never raised out of the scoring loop — a black box
        emitting garbage must not crash the experiment nor top a leaderboard.
        Only a wrong-shaped array raises, since that is a programming error
        in the wrapper, not a property of the solution.
        """
        net = self.scenario.network
        v = np.asarray(link_flows, dtype=np.float64)
        if v.shape != (net.n_links,):
            raise ValueError(f"link_flows shape {v.shape} != ({net.n_links},)")

        if not np.all(np.isfinite(v)):
            return self._censored("non-finite flows")
        scale = max(1.0, float(np.abs(v).max()))
        if v.min() < -self._CLIP_TOL * scale:
            return self._censored("negative flows")
        v = np.maximum(v, 0.0)

        costs = net.link_cost(v)
        tstt = float(v @ costs)
        _, sptt = self._engine.all_or_nothing(costs, self.scenario.demand)
        excess = tstt - sptt

        balance = node_balance_residual(self.scenario, v)
        demand_scale = max(1.0, self._total_demand)
        conserves = balance <= self.feasibility_tol * demand_scale
        # SPTT > TSTT is impossible for demand-feasible flows: censor it too.
        nonnegative_excess = excess >= -self.feasibility_tol * max(tstt, 1.0)
        feasible = conserves and nonnegative_excess

        if not feasible:
            metrics = self._censored("failed feasibility audit")
            metrics["node_balance_residual"] = balance
            # Report raw totals for diagnosis; the *scored* gaps stay censored.
            metrics["tstt"] = tstt
            metrics["sptt"] = sptt
            metrics["beckmann_objective"] = float(net.link_cost_integral(v).sum())
            return metrics

        return {
            "tstt": tstt,
            "sptt": sptt,
            "relative_gap": excess / tstt if tstt > 0 else 0.0,
            "average_excess_cost": excess / self._total_demand
            if self._total_demand > 0
            else 0.0,
            "beckmann_objective": float(net.link_cost_integral(v).sum()),
            "node_balance_residual": balance,
            "feasible": 1.0,
        }
