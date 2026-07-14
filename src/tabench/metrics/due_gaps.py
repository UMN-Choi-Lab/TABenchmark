"""Harness-side certification of emitted Friesz SRDC-DUE profiles (P1).

Every scored quantity is recomputed as a pure function of
``(DUEScenario, DUEProfile)`` — the solver's self-reported ``C``/split/window
provenance is never trusted. From each route's emitted cumulative departure
curve the harness shifts by the free-flow time, reconstructs the EXACT
deterministic point-queue served curve (interior queue-clearing kinks and the
post-horizon clearing chord inserted analytically, so the score is invariant
to any regridding of the same piecewise-linear plan), and scores PER TRAVELER
by level inversion of BOTH curves (the adr-019 lesson: a start-of-step cost
sample lets a burst dump certify a false equilibrium; levels never do). The route axis
adds a genuinely new failure mode the single-route certifier cannot see: an
all-on-one-route profile equalizes its OWN used costs while the idle route is
strictly cheaper — so the reference minimum scans the MARGINAL-INSERTION cost
of every route at a dense harness-chosen set of candidate times (an
infinitesimal traveler does not move the curves, so this too is a pure
function of the emitted profile), and

    due_gap = (max cost over used travelers - min marginal cost anywhere) / C

which is 0 iff no traveler can improve by shifting departure time OR route —
the discrete Friesz (1993) DUE conditions — and positive otherwise. ``C`` is
the scenario's analytic equilibrium cost (the ``C*`` normalizer of
``bottleneck_gaps``). Infeasible profiles (wrong hash, non-conserving,
non-monotone, non-finite) are CENSORED (``feasible = 0``, scored NaN); only
wrong shapes raise.
"""

from __future__ import annotations

import logging

import numpy as np

from ..bottleneck.due import DUEProfile, DUEScenario

__all__ = ["DUEEvaluator"]

logger = logging.getLogger(__name__)

_SCORED = ("due_gap", "total_cost", "expected_cost", "max_queue", "total_travel_delay")


def _invert(curve: np.ndarray, times: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Earliest times at which the nondecreasing piecewise-linear ``curve``
    reaches each of ``levels`` (exact; same kernel as ``bottleneck_gaps``)."""
    j = np.clip(np.searchsorted(curve, levels, side="left"), 1, curve.shape[0] - 1)
    lo, hi = curve[j - 1], curve[j]
    span = hi - lo
    frac = np.where(span > 0.0, (levels - lo) / np.where(span > 0.0, span, 1.0), 0.0)
    return times[j - 1] + frac * (times[j] - times[j - 1])


class DUEEvaluator:
    """Model-blind SRDC-DUE certifier: pure function of ``(scenario, profile)``."""

    def __init__(self, scenario: DUEScenario, tol: float = 1e-6) -> None:
        self.scenario = scenario
        self.tol = float(tol)
        self._hash = scenario.content_hash()
        self._c_star = scenario.equilibrium_cost()

    def _censored(self, reason: str) -> dict[str, float]:
        logger.info("DUE profile censored: %s", reason)
        metrics = dict.fromkeys(_SCORED, float("nan"))
        metrics["feasible"] = 0.0
        return metrics

    def _schedule_cost(self, t_dep: np.ndarray, t_exit: np.ndarray) -> np.ndarray:
        sc = self.scenario
        return (
            sc.alpha * np.maximum(0.0, t_exit - t_dep)
            + sc.beta * np.maximum(0.0, sc.t_star - t_exit)
            + sc.gamma * np.maximum(0.0, t_exit - sc.t_star)
        )

    def certify(self, profile: DUEProfile) -> dict[str, float]:
        sc = self.scenario
        t = profile.times
        cum = profile.cumulative
        if cum.shape[0] != sc.n_routes:
            raise ValueError(
                f"DUEProfile has {cum.shape[0]} routes, scenario has {sc.n_routes}"
            )
        if profile.scenario_hash != self._hash:
            return self._censored(
                f"wrong scenario hash: profile ran on {profile.scenario_hash!r}, "
                f"this instance is {self._hash!r}"
            )
        if not (np.isfinite(t).all() and np.isfinite(cum).all()):
            return self._censored("non-finite times/cumulative")
        if np.any(np.diff(t) <= 0.0):
            return self._censored("times must be strictly increasing")
        eps = self.tol * max(1.0, sc.n_travelers)
        if np.any(cum[:, 0] > eps) or np.any(cum[:, 0] < -eps):
            return self._censored("each route's cumulative departures must start at 0")
        # running-max retraction gate: per-STEP tolerance would let T tiny
        # sub-eps retractions accumulate into a real rollback (the DTA
        # eps-accumulation family) — gate the total drop from the high-water
        # mark instead
        if float((np.maximum.accumulate(cum, axis=1) - cum).max(initial=0.0)) > eps:
            return self._censored("cumulative departures must be nondecreasing")
        n_r = np.maximum(cum[:, -1], 0.0)
        if abs(float(n_r.sum()) - sc.n_travelers) > eps:
            return self._censored(
                f"conservation: route volumes sum to {n_r.sum()!r}, need "
                f"N={sc.n_travelers!r}"
            )

        # Global insertion candidates: the profile's own grid, a FIXED-window
        # sweep of the analytic departure window (never stretched by the
        # emitted grid's hull — round-2 review CRITICAL: one far flat pad
        # point diluted a hull-spanning sweep until a 0.77-gap profile
        # certified ~0), and each route's on-time departure {t* - f_r}. The
        # sweep is belt-and-braces only: the per-route kink enumeration below
        # makes the reference minimum EXACT for any piecewise-linear plan.
        c_ref = self._c_star
        span_lo = sc.t_star - c_ref / sc.beta - float(sc.route_free_flow.max()) - 1.0
        span_hi = sc.t_star + c_ref / sc.gamma + 1.0
        eval_t = np.unique(
            np.concatenate(
                [
                    t,
                    np.linspace(span_lo, span_hi, 4001),
                    sc.t_star - sc.route_free_flow,
                ]
            )
        )

        max_used = -np.inf
        min_ref = np.inf
        total_cost = 0.0
        total_delay = 0.0
        max_queue = 0.0
        for r in range(sc.n_routes):
            f_r = float(sc.route_free_flow[r])
            s_r = float(sc.route_capacity[r])
            # bottleneck arrival curve on the shifted grid, then the EXACT
            # served curve: within a segment the arrival rate is constant, so
            # the point queue either serves at exactly s_r (queue up) or
            # tracks the arrival curve (queue empty), with at most ONE
            # interior kink — where the queue empties mid-segment — computed
            # analytically and inserted. This makes the score invariant to
            # ANY regridding that leaves the piecewise-linear plan unchanged
            # (the adr-022 review MAJOR: an interpolated extension bent the
            # clearing kink, so the same plan scored 25.6x differently
            # depending on where its emitted grid happened to end).
            tau = t + f_r
            arrive = np.maximum.accumulate(np.clip(cum[r], 0.0, None))
            tau_pts = [float(tau[0])]
            arr_pts = [float(arrive[0])]
            srv_pts = [0.0]
            for k in range(tau.size - 1):
                t0, t1 = float(tau[k]), float(tau[k + 1])
                a0, a1 = float(arrive[k]), float(arrive[k + 1])
                dt = t1 - t0  # can round to 0 when f_r >> grid spacing
                q0 = a0 - srv_pts[-1]
                if dt > 0.0:
                    lam = (a1 - a0) / dt
                    if q0 > 0.0 and lam < s_r:
                        t_e = t0 + q0 / (s_r - lam)  # queue-empties instant
                        if t0 < t_e < t1:
                            a_e = a0 + lam * (t_e - t0)
                            tau_pts += [t_e, t1]
                            arr_pts += [a_e, a1]
                            srv_pts += [a_e, a1]  # served == arrival past t_e
                            continue
                tau_pts.append(t1)
                arr_pts.append(a1)
                srv_pts.append(min(a1, srv_pts[-1] + s_r * dt))
            # past the horizon any residual queue drains at exactly s_r: the
            # true served curve is the chord to (t_clear, arrive[-1]), flat on
            q_h = arr_pts[-1] - srv_pts[-1]  # >= 0: served <= arrive
            t_clear = tau_pts[-1] + q_h / s_r
            if t_clear > tau_pts[-1]:
                tau_pts += [t_clear, t_clear + 1.0]
                arr_pts += [arr_pts[-1], arr_pts[-1]]
                srv_pts += [arr_pts[-1], arr_pts[-1]]
            else:  # already clear at the horizon: just a flat tail
                tau_pts.append(tau_pts[-1] + 1.0)
                arr_pts.append(arr_pts[-1])
                srv_pts.append(arr_pts[-1])
            tau_ext = np.array(tau_pts)
            arr_ext = np.array(arr_pts)
            served = np.maximum.accumulate(np.array(srv_pts))
            max_queue = max(max_queue, float((arr_ext - served).max()))

            # Marginal-insertion cost at candidate times (all routes, used or
            # not): a deviator at t joins behind A(t) = R_r(t) travelers and
            # exits at max(t + f_r, S^{-1}(A(t))) (t + f_r alone while A = 0:
            # the bottleneck is provably empty before the first departure).
            # The cost is piecewise linear in t with kinks ONLY at (a) the
            # profile's grid points, (b) pullbacks A^{-1} of the served
            # curve's kink levels, (c) the pullback of level S(t*) (the
            # queued exit crossing t*; the free-flow crossing is t* - f_r,
            # already a global candidate), and (d) the queue-vanishing zeros
            # of g(t) = A(t) - S(t + f_r), which is linear between the merged
            # kink grids. Evaluating every kink makes min_ref EXACT — no
            # sweep resolution to exploit.
            a_end = float(arrive[-1])
            s_end = float(served[-1])
            srv_levels = np.unique(served)
            pulls = _invert(arrive, t, srv_levels)
            s_tstar = float(np.interp(sc.t_star, tau_ext, served))
            kg = np.unique(np.concatenate([t, tau_ext - f_r]))
            g = np.interp(kg, t, arrive, left=0.0, right=a_end) - np.interp(
                kg + f_r, tau_ext, served, left=0.0, right=s_end
            )
            cross = np.nonzero(g[:-1] * g[1:] < 0.0)[0]
            g_zeros = kg[cross] + (kg[cross + 1] - kg[cross]) * (
                g[cross] / (g[cross] - g[cross + 1])
            )
            cand = np.concatenate(
                [eval_t, pulls, _invert(arrive, t, np.array([s_tstar])), kg, g_zeros]
            )
            level_at = np.interp(cand, t, arrive, left=0.0, right=a_end)
            queued = np.where(
                level_at > 0.0, _invert(served, tau_ext, level_at), -np.inf
            )
            t_exit_ins = np.maximum(cand + f_r, queued)
            min_ref = min(min_ref, float(self._schedule_cost(cand, t_exit_ins).min()))

            if n_r[r] <= eps:
                continue
            # Per-traveler costs on the used route: invert both curves. The
            # level set holds every kink of cost(L): the curves' own values,
            # the free-flow/queued switch levels (zeros of the piecewise-
            # linear S^{-1}(L) - A^{-1}(L) - f_r), and the t*-crossing levels
            # S(t*) and A(t* - f_r) — plus a FIXED-size linspace (level
            # resolution must not depend on the emitted grid size) and the
            # tolerance-boundary travelers at eps and n_r - eps (the cost
            # supremum is often attained by the first/last traveler; only
            # sub-eps dirt is excluded, so any residual understatement is
            # bounded by eps times the cost slope in level).
            kinks = np.unique(np.concatenate([arr_ext, served, [s_tstar]]))
            inv_gap = _invert(served, tau_ext, kinks) - (_invert(arrive, t, kinks) + f_r)
            sw = np.nonzero(inv_gap[:-1] * inv_gap[1:] < 0.0)[0]
            switch_levels = kinks[sw] + (kinks[sw + 1] - kinks[sw]) * (
                inv_gap[sw] / (inv_gap[sw] - inv_gap[sw + 1])
            )
            a_tstar = float(np.interp(sc.t_star - f_r, t, arrive, left=0.0, right=a_end))
            m = 8000
            levels = np.unique(
                np.concatenate(
                    [np.linspace(0.0, n_r[r], m + 1), kinks, switch_levels, [a_tstar]]
                )
            )
            levels = levels[(levels > eps) & (levels < n_r[r] - eps)]
            if n_r[r] > 2.0 * eps:
                levels = np.unique(np.concatenate([levels, [eps, n_r[r] - eps]]))
            if levels.size == 0:
                levels = np.array([0.5 * n_r[r]])
            t_dep = _invert(arrive, t, levels)
            t_exit = np.maximum(t_dep + f_r, _invert(served, tau_ext, levels))
            cost = self._schedule_cost(t_dep, t_exit)
            max_used = max(max_used, float(cost.max()))
            # integrate per-route cost/delay over the count level (trapezoid,
            # extended by continuity to [0, N_r]) — the system totals
            lv = np.concatenate([[0.0], levels, [n_r[r]]])
            cst = np.concatenate([cost[:1], cost, cost[-1:]])
            trv = np.concatenate(
                [(t_exit - t_dep)[:1], t_exit - t_dep, (t_exit - t_dep)[-1:]]
            )
            dm = np.diff(lv)
            total_cost += float(np.sum(0.5 * (cst[:-1] + cst[1:]) * dm))
            total_delay += float(np.sum(0.5 * (trv[:-1] + trv[1:]) * dm))

        if max_used == -np.inf:
            # round-2 review MAJOR: with every route below tolerance the old
            # code scored due_gap = -inf for arbitrary conserving garbage
            return self._censored(
                "no route carries scoreable volume (all n_r <= tolerance) — "
                "the per-traveler DUE conditions are unresolvable at this scale"
            )
        gap = (max_used - min_ref) / self._c_star
        return {
            "feasible": 1.0,
            "due_gap": gap,
            "total_cost": total_cost,
            "expected_cost": total_cost / sc.n_travelers,
            "max_queue": max_queue,
            "total_travel_delay": total_delay,
        }
