"""Harness-side certification of emitted Vickrey bottleneck schedules (P1).

Every scored quantity is recomputed here as a pure function of
``(BottleneckScenario, BottleneckSchedule cumulative departures)`` — the model's
self-reported ``r_early``/``t1``/``C*`` provenance is never trusted. From the
emitted cumulative departure curve the harness simulates the deterministic
point queue, recomputes each used departure time's generalized cost
``c(t) = alpha*T(t) + beta*[t* - (t+T)]+ + gamma*[(t+T) - t*]+``, and scores

    equilibrium_gap = (max c - min c) / C*   over used departure times

which is ``0`` iff the schedule is a user equilibrium (no traveler can improve by
shifting) and grows with the incentive to deviate — the SO schedule, which queues
no one but spreads schedule delay, scores a large positive gap. Semantics mirror
``metrics/gaps.py``/``transit_gaps.py``: an infeasible schedule (non-conserving,
non-monotone, wrong hash) is CENSORED (``feasible = 0``, scored quantities NaN);
only wrong shapes raise.
"""

from __future__ import annotations

import logging

import numpy as np

from ..bottleneck.scenario import BottleneckScenario
from ..bottleneck.solve import BottleneckSchedule

__all__ = ["BottleneckEvaluator"]

logger = logging.getLogger(__name__)

_SCORED = ("equilibrium_gap", "total_cost", "expected_cost", "max_queue", "total_travel_delay")


def _invert(curve: np.ndarray, times: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Earliest times at which the nondecreasing ``curve`` (sampled at ``times``,
    piecewise-linear) reaches each of ``levels`` — an exact inversion for the
    piecewise-linear cumulative curves."""
    j = np.clip(np.searchsorted(curve, levels, side="left"), 1, curve.shape[0] - 1)
    lo, hi = curve[j - 1], curve[j]
    span = hi - lo
    frac = np.where(span > 0.0, (levels - lo) / np.where(span > 0.0, span, 1.0), 0.0)
    return times[j - 1] + frac * (times[j] - times[j - 1])


class BottleneckEvaluator:
    """Model-blind bottleneck certifier: pure function of ``(scenario, schedule)``."""

    def __init__(self, scenario: BottleneckScenario, tol: float = 1e-6) -> None:
        self.scenario = scenario
        self.tol = float(tol)
        self._hash = scenario.content_hash()
        self._cstar = scenario.equilibrium_cost

    def _censored(self, reason: str) -> dict[str, float]:
        logger.info("bottleneck schedule censored: %s", reason)
        metrics = dict.fromkeys(_SCORED, float("nan"))
        metrics["feasible"] = 0.0
        return metrics

    def certify(self, schedule: BottleneckSchedule) -> dict[str, float]:
        sc = self.scenario
        t = schedule.times
        cum = schedule.cumulative
        if t.shape != cum.shape or t.ndim != 1:
            raise ValueError("BottleneckSchedule times/cumulative shape mismatch")
        if schedule.scenario_hash != self._hash:
            return self._censored(
                f"wrong scenario hash: schedule ran on {schedule.scenario_hash!r}, "
                f"this instance is {self._hash!r}"
            )
        if not (np.isfinite(t).all() and np.isfinite(cum).all()):
            return self._censored("non-finite times/cumulative")
        if np.any(np.diff(t) <= 0.0):
            return self._censored("times must be strictly increasing")

        eps = self.tol * max(1.0, sc.n_travelers)
        arrivals = cum  # cumulative arrivals to the bottleneck (= cumulative departures)
        if cum[0] != 0.0 or float(np.diff(cum).min(initial=0.0)) < -eps:
            return self._censored("cumulative departures must start at 0 and be nondecreasing")
        if abs(cum[-1] - sc.n_travelers) > eps:
            return self._censored(
                f"conservation: cumulative ends at {cum[-1]!r}, need N={sc.n_travelers!r}"
            )
        arrivals = np.maximum.accumulate(np.minimum(arrivals, sc.n_travelers))

        # Deterministic point-queue OUTPUT curve D(t): the bottleneck serves at
        # rate s whenever a queue is present, never more than has arrived —
        # D[k+1] = min(A[k+1], D[k] + s*dt). Queue n(t) = A(t) - D(t).
        s = sc.capacity
        dt = np.diff(t)
        served = np.empty_like(arrivals)
        served[0] = 0.0
        for k in range(dt.shape[0]):
            served[k + 1] = min(arrivals[k + 1], served[k] + s * dt[k])
        served = np.maximum.accumulate(served)

        # Score PER TRAVELER (per count level), not per grid step: invert both
        # curves so each traveler is charged the cost they actually experience
        # (departure time from A, exit/arrival time from D) — a start-of-step
        # sample would let a burst dump its whole mass at one cheap sampled time.
        m = int(max(8000, 4 * t.shape[0]))
        kinks = np.unique(np.concatenate([arrivals, served]))
        levels = np.unique(
            np.concatenate([np.linspace(0.0, sc.n_travelers, m + 1), kinks])
        )
        levels = levels[(levels > eps) & (levels < sc.n_travelers - eps)]
        if levels.size == 0:
            levels = np.array([0.5 * sc.n_travelers])
        t_dep = _invert(arrivals, t, levels)
        t_exit = _invert(served, t, levels)
        travel = np.maximum(0.0, t_exit - t_dep)
        cost = (
            sc.alpha * travel
            + sc.beta * np.maximum(0.0, sc.t_star - t_exit)
            + sc.gamma * np.maximum(0.0, t_exit - sc.t_star)
        )
        gap = float(cost.max() - cost.min()) / self._cstar
        # integrate cost / delay over ALL travelers m in [0, N]: extend the sampled
        # (interior) levels by continuity to the endpoints, then trapezoid in the
        # count level — this IS the system total (sum over travelers).
        lv = np.concatenate([[0.0], levels, [sc.n_travelers]])
        cst = np.concatenate([cost[:1], cost, cost[-1:]])
        trv = np.concatenate([travel[:1], travel, travel[-1:]])
        dm = np.diff(lv)
        total_cost = float(np.sum(0.5 * (cst[:-1] + cst[1:]) * dm))
        total_delay = float(np.sum(0.5 * (trv[:-1] + trv[1:]) * dm))
        return {
            "feasible": 1.0,
            "equilibrium_gap": gap,
            "total_cost": total_cost,
            "expected_cost": total_cost / sc.n_travelers,
            "max_queue": float((arrivals - served).max()),
            "total_travel_delay": total_delay,
        }
