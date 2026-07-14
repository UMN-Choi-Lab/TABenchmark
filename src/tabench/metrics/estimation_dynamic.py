"""Harness-side certification of emitted within-day OD profiles (P1, ADR-023).

The dynamic-T2 analogue of :class:`~tabench.metrics.estimation.ODCertifier`, but
**cheaper and exact**: the map from a time-slice profile to link counts is the
*linear* exogenous lag tensor, so there is no pinned assignment to run — the
certifier regenerates the **full-network** lag tensor deterministically from the
hashed recipe (never the estimator's payload) and scores the emitted ``(H, Z, Z)``
profile by exact linear algebra.

Scoring is a **pair, never collapsed over ``t``** (Hazelton 2015 transported to
the within-day setting):

* count-fit — ``obs_count_rmse`` over every ``(day, interval, obs sensor)`` (the
  per-interval residual; collapsing over ``t`` is exactly what would make this
  ``gls``), its fit-to-day-mean companion ``obs_mean_count_rmse`` (the P1
  honesty-diff target), the ``oracle_*`` floors at the planted truth, and — the
  **ranking** column — ``heldout_count_rmse`` on a disjoint held-out sensor set,
  plus ``heldout_flow_rmse`` vs the noise-free held-out crossings;
* OD-fit — ``od_rmse`` / ``od_nrmse`` over off-diagonal cells **and slices**,
  signed ``total_demand_error``, and the descriptive ``profile_rmse`` over
  normalized slice totals (right-total-wrong-timing made visible). Always
  reported, ranking nothing, flagged ``od_identifiable = 0`` when the task's
  identifiability report is negative (the new false-accept surface: held-out
  sensors *share* the lag structure, so a count-invariant cross-slice shift fools
  the held-out column too — OD columns must never rank).

Censoring mirrors ``ODCertifier``: a wrong-shaped profile raises; non-finite
entries and sub-tolerance negatives are censored (``od_feasible = 0``, NaN
metrics); a zero profile is **not** censored (a legitimate, terrible estimate).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..core.scenario import Scenario
from ..estimation._dynamic_map import lagged_assignment_tensor, predict_interval_counts
from ..estimation._proportions import active_pairs

__all__ = ["DynamicODCertifier", "DYNAMIC_METRIC_KEYS"]

DYNAMIC_METRIC_KEYS = (
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
    "profile_rmse",
    "od_identifiable",
)

_CLIP_TOL = 1e-9


def _count_rmse(pred: np.ndarray, counts: np.ndarray) -> float:
    """RMSE of predicted interval counts ``(T, S)`` vs observed counts ``(n_days, T, S)``."""
    if pred.size == 0 or counts.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred[None, :, :] - counts) ** 2)))


class DynamicODCertifier:
    """Model-blind exact scorer for one within-day dynamic T2 task. Reuse across checkpoints.

    The full-network lag tensor is regenerated here from ``(network, pairs, slice
    length, n_lags)`` — nothing model-attested is trusted, so map forgery is
    impossible. ``truth_profile`` is harness-only (the estimator never sees it).
    """

    def __init__(
        self,
        scenario: Scenario,
        obs_sensors: np.ndarray,
        heldout_sensors: np.ndarray,
        obs_counts: np.ndarray,
        heldout_counts: np.ndarray,
        truth_profile: np.ndarray,
        slice_length: float,
        n_lags: int,
        n_intervals: int,
        identifiability: Mapping[str, Any],
    ) -> None:
        self.scenario = scenario
        self.network = scenario.network
        self.pairs = active_pairs(scenario.demand.matrix)
        self.obs_sensors = np.asarray(obs_sensors, dtype=np.int64)
        self.heldout_sensors = np.asarray(heldout_sensors, dtype=np.int64)
        self.obs_counts = np.asarray(obs_counts, dtype=np.float64)
        self.heldout_counts = np.asarray(heldout_counts, dtype=np.float64)
        self.truth_profile = np.asarray(truth_profile, dtype=np.float64)
        self.n_intervals = int(n_intervals)
        self.identifiability = identifiability
        self._identifiable = bool(identifiability.get("linear_identifiable", False))

        # Regenerate the FULL-network exogenous map from the hashed recipe.
        full_map = lagged_assignment_tensor(
            self.network, self.pairs, float(slice_length), int(n_lags)
        )
        self._obs_map = full_map[:, self.obs_sensors, :]
        self._ho_map = full_map[:, self.heldout_sensors, :]

        n_zones = self.truth_profile.shape[1]
        self._off = ~np.eye(n_zones, dtype=bool)
        # Support = active pairs + diagonal (intrazonal pass-through). Emitted
        # mass on any OTHER cell has no lag column, so it is invisible to every
        # harness-recomputed count — obs AND held-out — while moving only the
        # descriptive OD columns (review MAJOR: an off-support dump certified
        # byte-identical count columns to the truth). Such profiles are
        # censored, mirroring the static track where the pinned assignment
        # loads the FULL emitted matrix and unroutable demand raises.
        support = np.eye(n_zones, dtype=bool)
        for i, j in self.pairs:
            support[i, j] = True
        self._offsupport = ~support
        truth_off = self.truth_profile[:, self._off]
        self._truth_off_sum = float(truth_off.sum())
        positive = truth_off[truth_off > 0]
        self._truth_off_mean = float(positive.mean()) if positive.size else 0.0
        self._truth_slice_frac = self._slice_fractions(self.truth_profile)

        truth_pairs = self._pairs_of(self.truth_profile)
        oracle_obs = predict_interval_counts(self._obs_map, truth_pairs, self.n_intervals)
        oracle_ho = predict_interval_counts(self._ho_map, truth_pairs, self.n_intervals)
        self._oracle_obs = _count_rmse(oracle_obs, self.obs_counts)
        self._oracle_ho = _count_rmse(oracle_ho, self.heldout_counts)
        self._ho_flow_ref = oracle_ho  # noise-free held-out crossings at truth

    def _pairs_of(self, profile: np.ndarray) -> np.ndarray:
        return np.array(
            [[profile[h, i, j] for (i, j) in self.pairs] for h in range(profile.shape[0])],
            dtype=np.float64,
        ).reshape(profile.shape[0], len(self.pairs))

    def _slice_fractions(self, profile: np.ndarray) -> np.ndarray:
        totals = profile[:, self._off].sum(axis=1)
        grand = float(totals.sum())
        return totals / grand if grand > 0 else np.zeros_like(totals)

    def _censored(self) -> dict[str, float]:
        metrics = {key: float("nan") for key in DYNAMIC_METRIC_KEYS}
        metrics["od_feasible"] = 0.0
        metrics["oracle_obs_count_rmse"] = self._oracle_obs
        metrics["oracle_heldout_count_rmse"] = self._oracle_ho
        metrics["od_identifiable"] = 1.0 if self._identifiable else 0.0
        return metrics

    def certify(self, profile: np.ndarray) -> dict[str, float]:
        """Certified metric dict for one emitted ``(H, Z, Z)`` profile."""
        q = np.asarray(profile, dtype=np.float64)
        if q.shape != self.truth_profile.shape:
            raise ValueError(
                f"OD profile shape {q.shape} != {self.truth_profile.shape}"
            )
        if not np.all(np.isfinite(q)):
            return self._censored()
        # negativity gate toleranced against the OFF-DIAGONAL scale only — a
        # huge intrazonal (metric-ignored) cell must not inflate the tolerance
        # under which a genuinely negative active cell escapes censoring
        # (review MINOR, fixed in the static ODCertifier for parity too)
        scale = max(1.0, float(np.abs(q[:, self._off]).max(initial=0.0)))
        if q.min() < -_CLIP_TOL * scale:
            return self._censored()
        q = np.maximum(q, 0.0)
        # off-support censor (review MAJOR): demand on cells with no lag
        # column is count-invisible by construction — see __init__
        off_support_mass = float(q[:, self._offsupport].sum())
        if off_support_mass > _CLIP_TOL * max(1.0, float(q[:, self._off].sum())):
            return self._censored()

        d_pairs = self._pairs_of(q)
        pred_obs = predict_interval_counts(self._obs_map, d_pairs, self.n_intervals)
        pred_ho = predict_interval_counts(self._ho_map, d_pairs, self.n_intervals)

        metrics: dict[str, float] = {
            "od_feasible": 1.0,
            "obs_count_rmse": _count_rmse(pred_obs, self.obs_counts),
            "obs_mean_count_rmse": (
                float(np.sqrt(np.mean((pred_obs - self.obs_counts.mean(axis=0)) ** 2)))
                if self.obs_counts.size
                else float("nan")
            ),
            "oracle_obs_count_rmse": self._oracle_obs,
            "heldout_count_rmse": _count_rmse(pred_ho, self.heldout_counts),
            "oracle_heldout_count_rmse": self._oracle_ho,
            "heldout_flow_rmse": (
                float(np.sqrt(np.mean((pred_ho - self._ho_flow_ref) ** 2)))
                if self.heldout_sensors.size
                else float("nan")
            ),
            "od_identifiable": 1.0 if self._identifiable else 0.0,
        }
        diff_off = (q - self.truth_profile)[:, self._off]
        metrics["od_rmse"] = float(np.sqrt(np.mean(diff_off**2)))
        metrics["od_nrmse"] = (
            metrics["od_rmse"] / self._truth_off_mean
            if self._truth_off_mean > 0
            else float("nan")
        )
        metrics["total_demand_error"] = (
            (float(q[:, self._off].sum()) - self._truth_off_sum) / self._truth_off_sum
            if self._truth_off_sum > 0
            else float("nan")
        )
        metrics["profile_rmse"] = float(
            np.sqrt(np.mean((self._slice_fractions(q) - self._truth_slice_frac) ** 2))
        )
        return metrics
