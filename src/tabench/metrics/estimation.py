"""Harness-side certification of emitted OD matrices (P1, ADR-002 Decision 2).

The T2 analogue of ``metrics.gaps.Evaluator``: model-blind, it runs a **pinned
reference assignment** on each emitted OD matrix and scores the resulting link
flows. UE flows are unique under strictly increasing BPR costs, so the map
``Q -> v`` is well defined; the pin (``bfw``, cold start, target relative gap
``1e-6``, iteration cap, line-search tolerance) only fixes the finite-budget
approximation and the tie-breaking bytes, and is part of the task definition
(it feeds ``EstimationTask.content_hash`` and lands in the manifest).

Scoring is a **pair, never collapsed** (Hazelton 2015 transported to T2):

* count-fit — ``obs_count_rmse`` on the sensors the estimator saw (with the
  ``oracle_*`` floor = the same metric at ``UE(Q_true)``), its fit-to-period-mean
  companion ``obs_mean_count_rmse`` (the P1 honesty-diff target against the
  estimator's self-report, which also measures fit to the mean count — the two
  agree only in that reduction, not against per-period counts under noise), and,
  as the *ranking* column on every task, ``heldout_count_rmse`` on a disjoint
  held-out sensor set whose counts the harness draws from truth on
  ``SOURCE_EVALUATION`` and never shows the estimator, plus ``heldout_flow_rmse``;
* OD-fit — ``od_rmse`` / ``od_nrmse`` / signed ``total_demand_error`` over
  **off-diagonal** cells only, always reported but ranking nothing, flagged
  ``od_identifiable = 0`` when the task's identifiability report is negative.

Censoring mirrors ``Evaluator.evaluate``: a wrong-shaped matrix raises (a wrapper
programming error); non-finite entries, sub-tolerance negatives, and demand the
pinned assignment cannot route are censored (``od_feasible = 0``, NaN metrics),
never crashing the run. A **zero matrix is not censored** — it is a legitimate,
terrible estimate that certifies with catastrophic count-fit.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..core.budget import Budget
from ..core.results import Trace
from ..core.rng import RngBundle
from ..core.scenario import Demand, Scenario
from ..models.frank_wolfe import BiconjugateFrankWolfeModel
from ._feasibility import clip_negatives

__all__ = ["ODCertifier", "CERTIFICATE_DEFAULTS"]

CERTIFICATE_DEFAULTS: dict[str, Any] = {
    "assignment": "bfw",
    "target_relative_gap": 1.0e-6,
    "max_iterations": 5000,
    "line_search_xtol": 1.0e-12,
}

_METRIC_KEYS = (
    "od_feasible",
    "obs_count_rmse",
    "obs_mean_count_rmse",
    "oracle_obs_count_rmse",
    "heldout_count_rmse",
    "oracle_heldout_count_rmse",
    "heldout_flow_rmse",
    "od_rmse",
    "od_nrmse",
    "total_demand_error",
    "od_identifiable",
    "certificate_gap",
    "certificate_converged",
)


def _rmse_counts(flows: np.ndarray, sensors: np.ndarray, counts: np.ndarray) -> float:
    """RMSE over (period, sensor) of modeled flow minus observed count."""
    if sensors.size == 0 or counts.size == 0:
        return float("nan")
    modeled = flows[sensors][None, :]
    return float(np.sqrt(np.mean((modeled - counts) ** 2)))


class ODCertifier:
    """Model-blind pinned-assignment scorer for one T2 task. Reuse across checkpoints."""

    def __init__(
        self,
        scenario: Scenario,
        obs_sensors: np.ndarray,
        heldout_sensors: np.ndarray,
        obs_counts: np.ndarray,
        heldout_counts: np.ndarray,
        oracle_flows: np.ndarray,
        identifiability: Mapping[str, Any],
        certificate: Mapping[str, Any] | None = None,
    ) -> None:
        self.scenario = scenario
        self.network = scenario.network
        self.truth_od = scenario.demand.matrix
        self.obs_sensors = np.asarray(obs_sensors, dtype=np.int64)
        self.heldout_sensors = np.asarray(heldout_sensors, dtype=np.int64)
        self.obs_counts = np.asarray(obs_counts, dtype=np.float64)
        self.heldout_counts = np.asarray(heldout_counts, dtype=np.float64)
        self.oracle_flows = np.asarray(oracle_flows, dtype=np.float64)
        self.identifiability = identifiability
        self.certificate = dict(CERTIFICATE_DEFAULTS)
        if certificate:
            self.certificate.update(certificate)
        if str(self.certificate["assignment"]) != "bfw":
            # The pin's model component is hashed and recorded; enforce it rather
            # than silently running bfw under a different label (ADR-002 Dec. 2).
            raise ValueError(
                f"unsupported certificate assignment {self.certificate['assignment']!r}: "
                "only 'bfw' is a supported certificate pin this sprint (SUE-pinned "
                "certificates are deferred, ADR-002)"
            )

        self._off = ~np.eye(self.truth_od.shape[0], dtype=bool)
        truth_off = self.truth_od[self._off]
        self._truth_off_sum = float(truth_off.sum())
        positive = truth_off[truth_off > 0]
        self._truth_off_mean = float(positive.mean()) if positive.size else 0.0
        self._identifiable = bool(identifiability.get("linear_identifiable", False))

        # Constant oracle floors (independent of the estimate).
        self._oracle_obs = _rmse_counts(self.oracle_flows, self.obs_sensors, self.obs_counts)
        self._oracle_ho = _rmse_counts(
            self.oracle_flows, self.heldout_sensors, self.heldout_counts
        )

        pin_gap = float(self.certificate["target_relative_gap"])
        self._pin_budget = Budget(
            iterations=int(self.certificate["max_iterations"]),
            target_relative_gap=pin_gap,
        )
        self._pin_gap = pin_gap
        self._solver = BiconjugateFrankWolfeModel(
            line_search_xtol=float(self.certificate["line_search_xtol"])
        )

    def _censored(self) -> dict[str, float]:
        metrics = {key: float("nan") for key in _METRIC_KEYS}
        metrics["od_feasible"] = 0.0
        metrics["oracle_obs_count_rmse"] = self._oracle_obs
        metrics["oracle_heldout_count_rmse"] = self._oracle_ho
        metrics["od_identifiable"] = 1.0 if self._identifiable else 0.0
        return metrics

    def _pinned_ue(self, od_matrix: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Run the pinned bfw assignment on ``od_matrix``; may raise RuntimeError."""
        scen = dataclasses.replace(
            self.scenario, demand=Demand(matrix=od_matrix), reference=None
        )
        trace = Trace()
        self._solver.solve(scen, self._pin_budget, RngBundle(0), trace)
        flows = trace.final.link_flows
        gap = float(trace.final.self_report.get("relative_gap", float("nan")))
        converged = 1.0 if (np.isfinite(gap) and gap <= self._pin_gap) else 0.0
        return flows, gap, converged

    def certify(self, od_matrix: np.ndarray) -> dict[str, float]:
        """Certified metric dict for one emitted OD estimate."""
        q = np.asarray(od_matrix, dtype=np.float64)
        if q.shape != self.truth_od.shape:
            raise ValueError(
                f"OD estimate shape {q.shape} != {self.truth_od.shape}"
            )
        if not np.all(np.isfinite(q)):
            return self._censored()
        # off-diagonal scale only: a huge intrazonal (assignment-ignored) cell
        # must not inflate the tolerance under which a genuinely negative
        # inter-zonal cell escapes censoring (adr-023 review parity fix)
        scale = max(1.0, float(np.abs(q[self._off]).max(initial=0.0)))
        clipped = clip_negatives(q, scale)
        if clipped is None:
            return self._censored()
        q = clipped

        try:
            flows, cert_gap, cert_conv = self._pinned_ue(q)
        except RuntimeError:
            return self._censored()

        metrics = {
            "od_feasible": 1.0,
            "obs_count_rmse": _rmse_counts(flows, self.obs_sensors, self.obs_counts),
            "obs_mean_count_rmse": _rmse_counts(
                flows, self.obs_sensors, self.obs_counts.mean(axis=0, keepdims=True)
            ),
            "oracle_obs_count_rmse": self._oracle_obs,
            "heldout_count_rmse": _rmse_counts(
                flows, self.heldout_sensors, self.heldout_counts
            ),
            "oracle_heldout_count_rmse": self._oracle_ho,
            "heldout_flow_rmse": (
                float(
                    np.sqrt(
                        np.mean(
                            (flows[self.heldout_sensors] - self.oracle_flows[self.heldout_sensors])
                            ** 2
                        )
                    )
                )
                if self.heldout_sensors.size
                else float("nan")
            ),
            "certificate_gap": cert_gap,
            "certificate_converged": cert_conv,
            "od_identifiable": 1.0 if self._identifiable else 0.0,
        }
        diff_off = (q - self.truth_od)[self._off]
        metrics["od_rmse"] = float(np.sqrt(np.mean(diff_off**2)))
        metrics["od_nrmse"] = (
            metrics["od_rmse"] / self._truth_off_mean if self._truth_off_mean > 0 else float("nan")
        )
        metrics["total_demand_error"] = (
            (float(q[self._off].sum()) - self._truth_off_sum) / self._truth_off_sum
            if self._truth_off_sum > 0
            else float("nan")
        )
        return metrics
