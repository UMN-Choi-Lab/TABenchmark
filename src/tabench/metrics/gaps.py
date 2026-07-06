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
* SUE fixed-point residual (scenarios with ``sue_theta`` only):
  ``||v - L(t(v), theta)||_1 / total demand`` with ``L`` the pinned
  Dial-STOCH loading map — misallocated link-traversals per traveler
  (docs/design/adr-001). On SUE tasks this is the ranking metric; the UE
  columns remain descriptive (they are strictly positive at SUE by design).

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

from ..core.rng import SOURCE_EVALUATION, RngBundle
from ..core.scenario import Demand, Scenario
from ..models._paths import PathEngine
from ..models._probit import ProbitEngine
from ..models._stoch import StochEngine
from ..models.so import marginal_network

__all__ = ["Evaluator", "node_balance_residual"]


def _probit_certificate(
    v: np.ndarray, samples: np.ndarray, total_demand: float
) -> tuple[float, float, float]:
    """Certified probit residual, jackknife SE, and CLT noise floor (adr-003).

    ``samples`` are the ``R_cert`` per-draw AON link flows at the pinned
    perturbations. Returns ``(residual, se, floor)`` where

    * residual ``= ||v - mean(samples)||_1 / D`` — the ranking column;
    * se ``=`` jackknife standard error of that residual over the samples
      (conservative at the fixed point, where the |.| kink makes it
      inconsistent — hence the floor);
    * floor ``= sum_a sqrt(2 s_a^2 / (pi R_cert)) / D`` with ``s_a^2`` the
      across-sample link-flow variance — the expected residual when ``v`` is
      exactly the fixed point (the certificate's own O(1/sqrt(R_cert)) bias).
    """
    r_cert = samples.shape[0]
    d = total_demand if total_demand > 0 else 1.0
    vhat = samples.mean(axis=0)
    residual = float(np.abs(v - vhat).sum() / d)
    # Jackknife over the R_cert samples: leave-one-out residuals.
    total = samples.sum(axis=0)
    jackknife = np.abs(v[None, :] - (total - samples) / (r_cert - 1)).sum(axis=1) / d
    se = float(np.sqrt((r_cert - 1) / r_cert * ((jackknife - jackknife.mean()) ** 2).sum()))
    variance = samples.var(axis=0, ddof=1)
    floor = float(np.sqrt(2.0 * variance / (np.pi * r_cert)).sum() / d)
    return residual, se, floor


def node_balance_residual(
    scenario: Scenario,
    link_flows: np.ndarray,
    demand_matrix: np.ndarray | None = None,
) -> float:
    """Maximum absolute demand-aware flow-conservation residual over all nodes.

    Non-zone nodes must conserve flow exactly. Zone node ``i`` must satisfy
    ``inflow_i - outflow_i = attractions_i - productions_i`` where productions
    and attractions are the off-diagonal row/column sums of the OD matrix
    (intrazonal demand never enters the network).

    ``demand_matrix`` overrides the scenario's fixed OD matrix — used by the
    elastic-demand certificate, which audits conservation against the
    *demand-consistent* demand ``d* = D(u(v))`` recomputed from the flows,
    not against a fixed matrix (adr-005).
    """
    net = scenario.network
    od = (
        scenario.demand.matrix
        if demand_matrix is None
        else np.asarray(demand_matrix, dtype=np.float64)
    )
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

    def __init__(
        self,
        scenario: Scenario,
        feasibility_tol: float = 1e-6,
        so_metrics: bool = False,
        root_seed: int = 0,
        r_cert: int = 2000,
    ) -> None:
        self.scenario = scenario
        self.feasibility_tol = feasibility_tol
        self._engine = PathEngine(scenario.network)
        self._total_demand = scenario.demand.total
        self._theta = scenario.sue_theta
        # Elastic (variable) demand: the certificate recomputes the
        # demand-consistent demand d* = D(u(v)) from the flows and audits both
        # route equilibrium and demand consistency against it (adr-005). Gated
        # on the scenario field, exactly like sue_theta.
        self._elastic = scenario.elastic_demand
        # Combined trip-distribution + assignment (Evans 1976, adr-007): the
        # demand is endogenous too, but distributed from fixed trip-end margins
        # by a doubly-constrained gravity model rather than a pointwise decay
        # law. The certificate recomputes d* = gravity(u(v)) from the flows and
        # audits route equilibrium + demand consistency against it — the same
        # P1-pure recipe as elastic, with the gravity in place of D(u). The
        # OD-cost skim is driven by the gravity *support* (interzonal pairs with
        # positive margins), not the reference matrix's nonzeros, so it stays
        # exact even if a reference entry underflowed to zero.
        self._combined = scenario.combined_demand
        self._combined_support_demand = (
            Demand(self._combined.support().astype(np.float64))
            if self._combined is not None
            else None
        )
        # Boundedly-rational UE (adr-008): a fixed-demand deterministic route
        # equilibrium relaxed to an indifference band epsilon. It uses the ordinary
        # fixed-demand certificate; the band adds one scored flag, br_acceptable =
        # (AEC <= epsilon). Because TSTT - SPTT = D * AEC is the demand-weighted
        # MEAN per-traveler excess, AEC <= epsilon is NECESSARY for BR-UE (every
        # used route within epsilon of its OD min => mean excess <= epsilon) but
        # NOT sufficient: a flow can concentrate a little traffic on a route far
        # outside the band and still average under it. This is the same
        # aggregate-vs-disaggregate limitation the node-balance audit documents;
        # link flows cannot exclude it (route flows are never emitted).
        self._br_epsilon = scenario.br_epsilon
        # Side-constrained UE (adr-009): hard per-link capacities v_a <= u_a. The
        # SC-specific scored quantity is capacity feasibility -- link-visible and
        # checked per-link to a tight relative tolerance (unlike the augmented-cost
        # multipliers, which are duals). The raw relative gap stays positive at a
        # correct SC equilibrium (binding links carry flow that would prefer to
        # grow), so it is reported but is not the acceptance criterion.
        self._side_capacities = scenario.side_capacities
        # Logit SUE certifies through the closed-form Dial-STOCH map; probit
        # SUE has no closed form, so the harness pins ONE Monte Carlo
        # perturbation matrix E per task, drawn from the reserved evaluation
        # stream under macrorep=0 (adr-003 Decision 1). Pinning E is legal
        # because the perception variance is flow-independent, so E never
        # depends on t(v): every macrorep and checkpoint shares one sampled
        # map (common random numbers), and the certificate stays a pure
        # function of (link_flows, scenario, root_seed).
        self._stoch = None
        self._probit = None
        self._probit_perturbations = None
        if self._theta is not None:
            if scenario.sue_family == "probit":
                if r_cert < 2:
                    # The jackknife SE divides by r_cert-1 and the CLT floor by
                    # a ddof=1 variance; below 2 both are NaN, which is the
                    # censoring signal — a feasible row must never emit it.
                    raise ValueError(
                        "probit certificate needs r_cert >= 2 (jackknife SE and "
                        f"CLT floor are undefined below 2), got {r_cert}"
                    )
                self._probit = ProbitEngine(scenario.network)
                gen = RngBundle(root_seed, macrorep=0).generator(SOURCE_EVALUATION)
                self._probit_perturbations = self._probit.perturbations(
                    self._theta, gen, r_cert
                )
            else:
                self._stoch = StochEngine(scenario.network)
        # Certified SO gap columns (one extra AON per checkpoint): enabled by
        # the runner when the grid contains a static_so model, or explicitly.
        # No scenario field: the SO gap needs no instance data — UE and SO
        # runs answer two questions about ONE hashed instance.
        self.so_metrics = so_metrics
        self._marginal = marginal_network(scenario.network) if so_metrics else None

    def _censored(self, reason: str) -> dict[str, float]:
        metrics = {
            "tstt": float("nan"),
            "sptt": float("nan"),
            "relative_gap": float("nan"),
            "average_excess_cost": float("nan"),
            "beckmann_objective": float("nan"),
            "node_balance_residual": float("inf"),
            "feasible": 0.0,
        }
        if self._theta is not None:
            metrics["sue_fixed_point_residual"] = float("nan")
        if self._probit is not None:
            metrics["sue_residual_se"] = float("nan")
            metrics["sue_residual_floor"] = float("nan")
        if self._elastic is not None or self._combined is not None:
            metrics["realized_demand"] = float("nan")
        if self._br_epsilon is not None:
            metrics["br_acceptable"] = 0.0  # a censored flow is not BR-acceptable
        if self._side_capacities is not None:
            metrics["sc_capacity_feasible"] = 0.0
            metrics["max_capacity_violation"] = float("inf")
        if self.so_metrics:
            for key in ("so_relative_gap", "so_average_excess_cost", "tstt_mc", "sptt_mc"):
                metrics[key] = float("nan")
        return metrics

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
        realized_total: float | None = None
        if self._elastic is not None:
            # Elastic certificate (adr-005): recompute the demand-consistent
            # demand d* = D(u(v)) from the emitted flows (u = per-OD shortest
            # path cost) and audit BOTH conditions against it — route
            # equilibrium (relative_gap below) AND demand consistency
            # (node_balance vs d*, which pins the per-node demand v must route,
            # so a phantom/circulation flow carrying no OD traffic is censored
            # even on all-zone networks). Everything is a pure function of v and
            # the content-hashed demand law — no self-report is trusted (P1).
            # Unlike fixed demand, node_balance is a *convergence* quantity here
            # (the real flows route d* only at equilibrium), so it doubles as
            # the feasibility gate: an off-equilibrium flow is not a valid
            # elastic solution — there is no fixed demand for it to be
            # feasible-but-suboptimal against.
            try:
                kappa = self._engine.od_cost_matrix(costs, self.scenario.demand)
            except RuntimeError:
                return self._censored("unreachable OD pair at current costs")
            d_star = self._elastic.realized_demand(self.scenario.demand.matrix, kappa)
            np.fill_diagonal(d_star, 0.0)  # intrazonal demand never enters the network
            realized_total = float(d_star.sum())
            sptt = float((kappa * d_star).sum())  # SPTT of demand d* at costs
            balance = node_balance_residual(self.scenario, v, demand_matrix=d_star)
            demand_scale = max(1.0, realized_total)
        elif self._combined is not None:
            # Combined distribution + assignment certificate (adr-007): recompute
            # the gravity-consistent demand d* = gravity(u(v)) from the emitted
            # flows (u = per-OD shortest-path cost over the reference support) and
            # audit BOTH conditions against it — route equilibrium (relative_gap)
            # AND demand consistency (node_balance vs d*, which also gates
            # feasibility). Like elastic, an off-equilibrium flow routes the
            # solver's intermediate demand, not d*, so node_balance is a
            # convergence quantity: a checkpoint certifies only once the combined
            # equilibrium is (nearly) reached. Everything is a pure function of v
            # and the content-hashed margins/beta — no self-report is trusted.
            try:
                kappa = self._engine.od_cost_matrix(costs, self._combined_support_demand)
            except RuntimeError:
                return self._censored("unreachable OD pair at current costs")
            d_star = self._combined.gravity(kappa)
            np.fill_diagonal(d_star, 0.0)  # intrazonal demand never enters the network
            realized_total = float(d_star.sum())
            sptt = float((kappa * d_star).sum())  # SPTT of gravity demand d* at costs
            balance = node_balance_residual(self.scenario, v, demand_matrix=d_star)
            demand_scale = max(1.0, realized_total)
        else:
            _, sptt = self._engine.all_or_nothing(costs, self.scenario.demand)
            balance = node_balance_residual(self.scenario, v)
            demand_scale = max(1.0, self._total_demand)
        excess = tstt - sptt
        conserves = balance <= self.feasibility_tol * demand_scale

        # SPTT > TSTT is impossible for demand-feasible flows: censor it too.
        nonnegative_excess = excess >= -self.feasibility_tol * max(tstt, 1.0)
        feasible = conserves and nonnegative_excess

        tstt_mc = sptt_mc = None
        if self._marginal is not None and feasible:
            t_mc = self._marginal.link_cost(v)
            tstt_mc = float(v @ t_mc)
            _, sptt_mc = self._engine.all_or_nothing(t_mc, self.scenario.demand)
            # SPTT_mc > TSTT_mc is equally impossible for demand-feasible
            # flows — without this, under-scaled flows would earn a negative
            # certified SO gap and top an SO leaderboard.
            feasible = (
                tstt_mc - sptt_mc >= -self.feasibility_tol * max(tstt_mc, 1.0)
            )

        if not feasible:
            metrics = self._censored("failed feasibility audit")
            metrics["node_balance_residual"] = balance
            # Report raw totals for diagnosis; the *scored* gaps stay censored.
            metrics["tstt"] = tstt
            metrics["sptt"] = sptt
            metrics["beckmann_objective"] = float(net.link_cost_integral(v).sum())
            return metrics

        # AEC divides the excess by the number of trips actually made — the
        # realized demand on elastic/combined tasks, the fixed total otherwise.
        aec_denom = realized_total if realized_total is not None else self._total_demand
        metrics = {
            "tstt": tstt,
            "sptt": sptt,
            "relative_gap": excess / tstt if tstt > 0 else 0.0,
            "average_excess_cost": excess / aec_denom if aec_denom and aec_denom > 0 else 0.0,
            "beckmann_objective": float(net.link_cost_integral(v).sum()),
            "node_balance_residual": balance,
            "feasible": 1.0,
        }
        if self._br_epsilon is not None:
            # Boundedly-rational acceptability (adr-008): AEC <= epsilon. Necessary
            # (a true BR-UE always passes) but not sufficient (concentration can
            # hide an out-of-band route) — the documented aggregate limitation.
            metrics["br_acceptable"] = (
                1.0 if metrics["average_excess_cost"] <= self._br_epsilon else 0.0
            )
        if self._side_capacities is not None:
            # SC-TAP scored quantity: hard capacity feasibility v_a <= u_a,
            # link-visible. A hard cap is a PER-LINK quantity, so the tolerance is
            # relative to each link's OWN capacity -- scaling by total demand (as
            # the demand-feasibility audit does) would let a fixed absolute overload
            # certify on a high-demand network (adversarial-review MAJOR 1). The
            # recovered multipliers / augmented-cost gap are a model self-report; a
            # harness-recomputed augmented-cost equilibrium gap is future work, so
            # this certifies capacity FEASIBILITY, not the full SC equilibrium.
            overload = np.maximum(v - self._side_capacities, 0.0)
            metrics["max_capacity_violation"] = float(overload.max())
            rel_overload = float((overload / self._side_capacities).max())
            metrics["sc_capacity_feasible"] = 1.0 if rel_overload <= self.feasibility_tol else 0.0
        if realized_total is not None:
            # Certified realized demand — the endogenous-demand scored quantity
            # (how much travel the equilibrium induces): Sum_rs D_rs(u_rs) for
            # elastic, Sum_ij gravity_ij(u) for combined distribution+assignment.
            metrics["realized_demand"] = realized_total
        if self._marginal is not None:
            excess_mc = tstt_mc - sptt_mc
            metrics["tstt_mc"] = tstt_mc
            metrics["sptt_mc"] = sptt_mc
            metrics["so_relative_gap"] = excess_mc / tstt_mc if tstt_mc > 0 else 0.0
            metrics["so_average_excess_cost"] = (
                excess_mc / self._total_demand if self._total_demand > 0 else 0.0
            )
        if self._probit is not None:
            # Probit certificate: MC loading at the pinned perturbations, scored
            # with its own uncertainty (residual ranks; se + floor bound the
            # certificate's own noise — adr-003 Decision 1). The pinned E makes
            # this a pure, byte-reproducible function of (v, scenario, root_seed).
            _, samples = self._probit.load_perturbed(
                costs, self.scenario.demand, self._probit_perturbations, return_samples=True
            )
            residual, se, floor = _probit_certificate(v, samples, self._total_demand)
            metrics["sue_fixed_point_residual"] = residual
            metrics["sue_residual_se"] = se
            metrics["sue_residual_floor"] = floor
        elif self._stoch is not None:
            try:
                v_hat = self._stoch.load(costs, self.scenario.demand, self._theta)
            except RuntimeError:
                # The pinned loader cannot certify at these costs (e.g.
                # float-saturated labels severing every Dial-efficient path):
                # censor the residual, never raise out of the scoring loop.
                metrics["sue_fixed_point_residual"] = float("nan")
            else:
                metrics["sue_fixed_point_residual"] = (
                    float(np.abs(v - v_hat).sum() / self._total_demand)
                    if self._total_demand > 0
                    else 0.0
                )
        return metrics
