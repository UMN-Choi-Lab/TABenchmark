"""Harness-side certification of emitted TD route-choice plans (Peeta & Mahmassani
1995, adr-031).

Every scored quantity is a pure function of ``(TDTAScenario, TDPathFlows)`` — the
solver's self-reports are never trusted. The model emits only its decision
variable (per-path, per-departure-interval flow, ``TDPathFlows``); the harness
runs its OWN dynamic network loading of those departures (the scenario-declared
``ctm``/``ltm`` kernel over the repo's DNL S/R loop) and recomputes:

* **TD-UE** (``tdue_gap``): the discrete experienced-time route-swap residual —
  the relative average excess cost between the total experienced travel time the
  emission actually incurs and the total it would incur if every traveler took
  their OD's cheapest available path at their own departure time, both measured
  by FIFO level composition on the harness's realized curves. It is 0 iff no
  traveler can lower their EXPERIENCED time by swapping routes (the discrete
  Wardrop conditions (5a)/(5b) of the paper), positive otherwise. This is the
  standard simulation-DTA gap w.r.t. the frozen realized times (a re-simulated
  deviation would differ — a Tier-B caveat, disclosed, not scored).
* **TD-SO** (``so_bound_gap``): ``(TSTT - Z*) / Z*`` where ``Z*`` is the
  lp-so-dta LP optimum on the CTM-cell instance derived from the same grid (a
  single destination). The LP relaxes the CTM Godunov flux to its four linear
  bounds, so its optimum PROVABLY lower-bounds every strict-CTM loading:
  ``so_bound_gap >= -tol`` always, and an undercut is an assert-level anomaly,
  not a censoring rule (the primal here is harness-computed, so the ADR-020
  weak-duality censor is unnecessary — kept only as a monitor).

Gates mirror ``dta_gaps``/``due_gaps``: fixed departure times give the model ZERO
timing freedom, so the demand-match gate closes every departure-time-gaming door
at TWO scales (per-edge ``eps`` + aggregate ``budget``); infeasible plans are
CENSORED (``feasible = 0``, scored NaN), only wrong shapes raise. The reference
minimum scans EVERY declared path (used or not) at every traveler's departure
time, so hiding the cheap path cannot dilute it.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from ..dnl.link import interp_curve
from ..tdta.loader import PathLoader
from ..tdta.scenario import TDTAScenario

__all__ = ["TDTAEvaluator"]

logger = logging.getLogger(__name__)

_SCORED = (
    "tdue_gap",
    "tdue_gap_max",
    "so_bound_gap",
    "tstt",
    "total_experienced_time",
    "max_experienced_time",
    "max_queue",
    "z_star",
)


def _earliest_time(curve: np.ndarray, level: float, dt: float) -> float:
    """Earliest ``t`` with the nondecreasing piecewise-linear ``curve`` (sampled
    at grid edges) ``>= level``; ``+inf`` if never reached in-horizon (same
    kernel as ``DNLOutput.travel_time`` / ``due_gaps._invert``)."""
    if level <= 0.0:
        return 0.0
    if curve[-1] < level:
        return math.inf
    j = int(np.searchsorted(curve, level, side="left"))
    if j == 0:
        return 0.0
    lo, hi = float(curve[j - 1]), float(curve[j])
    if hi <= lo:
        return dt * j
    return dt * (j - 1 + (level - lo) / (hi - lo))


def _interp(curve: np.ndarray, t: float, dt: float) -> float:
    """``interp_curve`` guarded against a ``+inf`` query time (a marginal
    traveler who never clears reads the curve's final level)."""
    if not math.isfinite(t):
        return float(curve[-1]) if t > 0 else 0.0
    return interp_curve(curve, t, dt)


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoid integral of ``y`` over ``x`` (numpy-version-agnostic; ``np.trapz``
    is deprecated in numpy 2.x and absent from the >=1.24 floor's successor name)."""
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


class TDTAEvaluator:
    """Model-blind TD-UE / TD-SO certifier: a pure function of
    ``(scenario, TDPathFlows)``.

    Raises ``ValueError`` at construction only if a single-destination SO cell LP
    is requested but unclearable; a multi-destination (UE-only) scenario simply
    reports ``so_bound_gap = NaN`` and never fakes a bound.
    """

    def __init__(self, scenario: TDTAScenario, tol: float = 1e-6, n_levels: int = 2000) -> None:
        self.scenario = scenario
        self.tol = float(tol)
        self.n_levels = int(n_levels)
        self._hash = scenario.content_hash()
        self._grid = scenario.grid
        self._dt = scenario.grid.dt
        self._edges = scenario.grid.edges
        self._paths_by_od = scenario.paths_by_od()
        self._first_link = scenario.first_link_of()
        self._ff = scenario.dynamics.length / scenario.dynamics.free_speed  # L/vf per link
        # per-OD cumulative desired departures at ORIGINAL grid edges (the gate ref)
        dcum = scenario.demand.cumulative(self._edges)  # (K+1, Z, Z)
        self._od_cum = {
            od: np.ascontiguousarray(dcum[:, od[0] - 1, od[1] - 1]) for od in self._paths_by_od
        }
        self._od_total = {od: float(self._od_cum[od][-1]) for od in self._od_cum}
        self._V = float(scenario.demand.total())
        # SO bound: derive the cell LP where the instance admits one (single
        # destination, CTM kernel, burst demand); a structural non-fit simply
        # leaves so_bound_gap unreported (NaN), never faked. But once a cell LP
        # IS derivable, Z* is resolved EAGERLY and an unclearable horizon raises
        # at construction (the ADR-020/021 discipline — a config error surfaced
        # before any output is scored, never a scoring-time crash).
        self._z_star_time: float | None = None
        try:
            cell_sc = scenario.derive_cell_scenario()
        except ValueError:
            cell_sc = None
        if cell_sc is not None:
            self._z_star_time = self._resolve_z_star(cell_sc) * self._dt

    @staticmethod
    def _resolve_z_star(cell_sc) -> float:
        from scipy.optimize import linprog

        from ..dta.cells import cell_canonical_lp

        c, a_eq, b_eq, a_ub, b_ub = cell_canonical_lp(cell_sc)
        res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, method="highs")
        if res.status != 0:
            raise ValueError(
                f"derived SO cell LP unsolvable (status {res.status}: {res.message}) — "
                "the horizon cannot clear the demand"
            )
        return float(res.fun)

    def _censored(self, reason: str) -> dict[str, float]:
        logger.info("TD path plan censored: %s", reason)
        metrics = dict.fromkeys(_SCORED, float("nan"))
        metrics["feasible"] = 0.0
        return metrics

    def _clearing_pad(self) -> int:
        """A generous bound on the extra steps needed to clear the network: the
        longest free-flow path traversal plus a queue-drain estimate at the
        slowest USED-link capacity (NOT just the sink links — an interior
        bottleneck behind a wide sink dominates the drain time, and reading only
        sink capacities false-censors honest finite-clearing plans; review MAJOR).
        The used-link graph is acyclic with positive capacities, so clearing is
        finite; the pad is HARD-CAPPED at 20000 so a pathological instance
        censors (never crashes/loops) — a bounded safety that could in principle
        bite a legitimately enormous instance (documented in adr-031)."""
        sc = self.scenario
        dt = self._dt
        ff_steps = 0
        for p in sc.paths:
            ff_steps = max(ff_steps, int(math.ceil(sum(self._ff[a] for a in p.links) / dt)) + 1)
        used_caps = [float(sc.dynamics.capacity[a]) for a in sc.used_links()]
        min_cap = min(used_caps) if used_caps else float(sc.dynamics.capacity.min())
        drain = int(math.ceil(self._V / (min_cap * dt))) + 1 if min_cap > 0 else 0
        return min(ff_steps + drain + 4, 20000)

    def certify(self, flows) -> dict[str, float]:
        sc = self.scenario
        dep = flows.departures
        if dep.shape != (sc.n_paths, self._grid.n_steps):
            raise ValueError(
                f"TDPathFlows shape mismatch: departures {dep.shape}, scenario wants "
                f"({sc.n_paths}, {self._grid.n_steps})"
            )
        if flows.scenario_hash != self._hash:
            return self._censored(
                f"wrong scenario hash: plan ran on {flows.scenario_hash!r}, this "
                f"instance is {self._hash!r}"
            )
        if not np.isfinite(dep).all():
            return self._censored("non-finite departures")

        budget = self.tol * max(1.0, self._V)
        # PER-OD tolerance scales (review MAJOR): a single global eps scaled by the
        # LARGEST OD lets a tiny OD shift its whole demand or retract a real
        # vehicle beside a huge OD (global eps ~ tol*V_huge >> 1) and still
        # certify. Each OD's gate is scaled by its OWN total — the ADR-020
        # two-scale discipline at the right granularity.
        cum = flows.cumulative()  # (n_paths, K+1)
        for od, plist in self._paths_by_od.items():
            eps_od = self.tol * max(1.0, self._od_total[od])
            # negative departures on this OD's paths = cumulative retraction
            neg = np.clip(-dep[plist], 0.0, None)
            if neg.max(initial=0.0) > eps_od or neg.sum() > eps_od:
                return self._censored(
                    f"negative departures on OD {od} (cumulative retraction)"
                )
            # demand-match: fixed departure times -> the model chooses only the
            # split, so per grid edge the OD's cumulative emitted departures must
            # equal its demand curve (per-cell eps AND aggregate mass, both per-OD).
            emitted = cum[plist].sum(axis=0)
            diff = np.abs(emitted - self._od_cum[od])
            if diff.max(initial=0.0) > eps_od or diff.sum() > eps_od:
                return self._censored(
                    f"demand-match violated for OD {od} (max {diff.max(initial=0.0):.3e}, "
                    f"total {diff.sum():.3e}) — departures are scenario-fixed"
                )
        if self._V <= self.tol:
            return self._censored("no OD carries scoreable volume")

        # ---- harness loading (extended so every vehicle clears in-horizon)
        pad = self._clearing_pad()
        out = PathLoader(sc, dep, extra_steps=pad).run()
        dt = self._dt
        n_in, n_out = out.n_in, out.n_out
        dests = set(sc.destinations())
        sink_links = [a for a in sc.used_links() if int(sc.network.term_node[a]) in dests]
        arrivals = n_out[sink_links].sum(axis=0) if sink_links else np.zeros(n_out.shape[1])

        # per-path desired cumulative on the EXTENDED grid (flat after K)
        k = self._grid.n_steps
        path_cum = np.zeros((sc.n_paths, out.grid.n_steps + 1))
        np.cumsum(dep, axis=1, out=path_cum[:, 1 : k + 1])
        path_cum[:, k + 1 :] = path_cum[:, k : k + 1]
        emitted_total = path_cum.sum(axis=0)  # total cumulative departures at edges

        # Delivery/clearing gate, TWO-SIDED against the EMITTED total (review
        # MAJOR): under-delivery = stranded flow (experienced time truncated);
        # over-delivery is impossible under conservation, so the abs form is a
        # belt-and-braces anomaly check. Gating against the emitted total (not V)
        # is robust to sub-budget over-emission that the demand-match gate allows.
        if abs(float(arrivals[-1]) - float(emitted_total[-1])) > budget:
            return self._censored(
                f"clearing failed: {emitted_total[-1] - arrivals[-1]:.3e} vehicles "
                f"remain in-network after the extended horizon (pad={pad})"
            )

        result = self._score_ue(sc, n_in, n_out, path_cum, dt, budget)
        if result is None:
            return self._censored("stranded traveler: experienced time is unbounded in-horizon")
        tdue_gap, tdue_gap_max, tc_used, max_eta = result

        # TSTT (system time incl. origin queue) as the AVAILABILITY-based occupancy
        # area: dt * sum_k (D(t_{k+1}) - A(t_k)), D the total cumulative emitted
        # departures. This (i) counts each vehicle only from when it is generated,
        # not from t=0 (review MAJOR: the V-based form charged the late-departing
        # tail pre-departure waiting), and (ii) is >= 0 by construction (arrivals
        # never exceed departures-so-far), so a sub-budget over-emission can no
        # longer forge an undercut. On a first-interval burst it coincides with the
        # LP's initial-occupancy convention (corridor stays 33 = Z*), which is the
        # convention the LP lower-bounds; the per-traveler EXPERIENCED total is
        # reported separately as total_experienced_time.
        tstt = float(dt * (emitted_total[1:].sum() - arrivals[:-1].sum()))
        max_queue = float((n_in - n_out).max(initial=0.0))
        so_gap = float("nan")
        if self._z_star_time is not None and self._z_star_time > 0:
            so_gap = (tstt - self._z_star_time) / self._z_star_time
            if so_gap < -self.tol:
                # By weak duality no conforming loading can beat Z* in this
                # convention; an undercut is a proof of infeasibility -> CENSOR
                # (the ADR-020/021 weak-duality-undercut discipline, not a warning).
                return self._censored(
                    f"TSTT {tstt!r} undercuts the certified LP bound "
                    f"{self._z_star_time!r} beyond tolerance (gap {so_gap:.3e}) — by "
                    "weak duality no conforming loading can, so this is infeasible"
                )
        return {
            "feasible": 1.0,
            "tdue_gap": tdue_gap,
            "tdue_gap_max": tdue_gap_max,
            "so_bound_gap": so_gap,
            "tstt": tstt,
            "total_experienced_time": tc_used,
            "max_experienced_time": max_eta,
            "max_queue": max_queue,
            "z_star": self._z_star_time if self._z_star_time is not None else float("nan"),
        }

    # ------------------------------------------------------------------ TD-UE

    def _marginal_time(self, path, t: float, n_in, n_out, path_cum, dt: float) -> float:
        """Experienced time of an infinitesimal traveler entering ``path`` at
        departure time ``t`` on the FROZEN loaded curves: origin-queue wait on the
        private first link (behind the desired level at ``t``) plus FIFO traversal
        of each subsequent link, floored by free flow. A pure function of the
        emitted plan (the marginal traveler does not move the curves)."""
        pi = path[0]
        links = path[1]
        a0 = int(links[0])
        level = _interp(path_cum[pi], t, dt)  # desired cumulative on this path at t
        exit_t = max(t + float(self._ff[a0]), _earliest_time(n_out[a0], level, dt))
        for b in links[1:]:
            if not math.isfinite(exit_t):
                return math.inf
            b = int(b)
            lvl = _interp(n_in[b], exit_t, dt)
            exit_t = max(exit_t + float(self._ff[b]), _earliest_time(n_out[b], lvl, dt))
        return exit_t - t

    def _score_ue(self, sc, n_in, n_out, path_cum, dt, budget):
        # index paths within each OD for the reference-min scan (used or not)
        od_of = [p.od for p in sc.paths]
        od_paths = {
            od: [(pj, sc.paths[pj].links) for pj in plist]
            for od, plist in self._paths_by_od.items()
        }
        tc_used = 0.0
        tc_min = 0.0
        max_eta = -math.inf
        max_excess = -math.inf
        for pi, p in enumerate(sc.paths):
            n_p = float(path_cum[pi, -1])
            if n_p <= self.tol:
                continue
            a0 = int(p.links[0])
            eps_lv = self.tol * max(1.0, n_p)
            # candidate levels: dense linspace + EVERY kink of the composed cost
            # profiles. The used/reference cost of a rank-`lv` traveler is
            # piecewise-linear in `lv` with breakpoints wherever any curve it
            # touches kinks — the private first link, but also every DOWNSTREAM
            # link on this path AND on every same-OD candidate path (the reference
            # minimum composes over all of them). Enumerating those level values
            # makes the max-form exact rather than resolution-limited (review
            # MAJOR; the vi-due exact-kink-enumeration lesson). Interior only: the
            # exact endpoints 0 / n_p are float-fragile (the private first link's
            # cumulative outflow tops out at n_p - O(1e-15), so the last-rank
            # traveler reads +inf); the trapezoid over the full [0, n_p] is closed
            # by continuity below (the due_gaps pattern).
            cand_links: set[int] = set()
            for _, links in od_paths[p.od]:
                cand_links.update(int(x) for x in links)
            kink_arrays = [path_cum[pj] for pj, _ in od_paths[p.od]]
            kink_arrays += [n_in[a] for a in cand_links] + [n_out[a] for a in cand_links]
            base = np.linspace(0.0, n_p, self.n_levels + 1)
            kinks = np.unique(np.concatenate(kink_arrays))
            levels = np.unique(np.concatenate([base, kinks]))
            levels = levels[(levels > eps_lv) & (levels < n_p - eps_lv)]
            if levels.size == 0:
                levels = np.array([0.5 * n_p])
            used = np.empty(levels.shape[0])
            minmarg = np.empty(levels.shape[0])
            eta_act = np.empty(levels.shape[0])
            for i, lv in enumerate(levels):
                t_dep = _earliest_time(path_cum[pi], lv, dt)
                # actual FIFO-composed experienced time (reporting); the private
                # first link's level IS this traveler's rank.
                exit_t = _earliest_time(n_out[a0], lv, dt)
                for b in p.links[1:]:
                    if not math.isfinite(exit_t):
                        break
                    b = int(b)
                    l_b = _interp(n_in[b], exit_t, dt)
                    exit_t = max(exit_t + float(self._ff[b]), _earliest_time(n_out[b], l_b, dt))
                if not math.isfinite(exit_t):
                    return None  # stranded on the used path -> censor
                eta_act[i] = exit_t - t_dep
                # cost for the GAP: the used cost and the reference minimum are the
                # SAME marginal-insertion composition, so the used path is in the
                # min set and the excess is >= 0 by construction (exact 0 when the
                # used route is the cheapest available at t_dep).
                costs = [
                    self._marginal_time(cand, t_dep, n_in, n_out, path_cum, dt)
                    for cand in od_paths[od_of[pi]]
                ]
                used[i] = self._marginal_time((pi, p.links), t_dep, n_in, n_out, path_cum, dt)
                minmarg[i] = min(costs)
            # trapezoid over the count level [0, n_p], extended by continuity at
            # the excluded endpoints (the per-traveler integral)
            lv_ext = np.concatenate([[0.0], levels, [n_p]])
            tc_used += _trapz(np.concatenate([used[:1], used, used[-1:]]), lv_ext)
            tc_min += _trapz(np.concatenate([minmarg[:1], minmarg, minmarg[-1:]]), lv_ext)
            max_eta = max(max_eta, float(eta_act.max()))
            # Max-form (Tier-B) — resolve the PEAK, conservatively (review MAJOR:
            # it was systematically under-reported, model-flattering). The max is
            # over ACTUAL travelers, i.e. over the count LEVEL (a departure-time
            # sweep would spuriously score hypothetical travelers inside departure
            # plateaus). The excess is piecewise-linear in the level but its
            # breakpoints are PULLBACKS of the composed cost's departure-time kinks
            # (each free-flow entry crossing a grid edge, t = edge - free-flow
            # prefix) mapped to the traveler level D(t) that departs then — the raw
            # curve values the integral kink set enumerates miss these. Add those
            # pullback levels, then BISECT the top corners for the queue-clearing
            # kinks a free-flow model cannot place; the peak converges to the dense
            # level reference at negligible cost (the vi-due kink lesson).
            cands = od_paths[od_of[pi]]

            def _excess_lv(lv: float, _cands=cands, _pi=pi, _links=p.links) -> float:
                t = _earliest_time(path_cum[_pi], lv, dt)
                u = self._marginal_time((_pi, _links), t, n_in, n_out, path_cum, dt)
                return u - min(
                    self._marginal_time(c, t, n_in, n_out, path_cum, dt) for c in _cands
                )

            edges = np.arange(path_cum.shape[1]) * dt
            tkinks = [edges]
            for _, links in cands:
                pref = 0.0
                for a in links:
                    tkinks.append(edges - pref)
                    pref += float(self._ff[int(a)])
            tk = np.unique(np.concatenate(tkinks))
            lv_kinks = np.array([_interp(path_cum[pi], float(t), dt) for t in tk])
            max_lv = np.unique(np.concatenate([levels, lv_kinks]))
            max_lv = max_lv[(max_lv > eps_lv) & (max_lv < n_p - eps_lv)]
            ex = np.array([_excess_lv(float(lv)) for lv in max_lv])
            max_excess = max(max_excess, float(ex.max(initial=-math.inf)))
            for j_star in np.argsort(ex)[-min(3, ex.size) :]:
                lo = float(max_lv[max(0, int(j_star) - 1)])
                hi = float(max_lv[min(max_lv.size - 1, int(j_star) + 1)])
                for _ in range(4):  # bisection zoom into the corner
                    if hi <= lo:
                        break
                    grid = np.linspace(lo, hi, 65)
                    vals = np.array([_excess_lv(float(lv)) for lv in grid])
                    jm = int(np.argmax(vals))
                    max_excess = max(max_excess, float(vals[jm]))
                    lo = float(grid[max(0, jm - 1)])
                    hi = float(grid[min(grid.size - 1, jm + 1)])

        if not math.isfinite(max_eta):
            return None
        denom = tc_min if tc_min > budget else max(tc_min, 1.0)
        tdue_gap = (tc_used - tc_min) / denom
        avg_min = tc_min / self._V if self._V > 0 else 1.0
        tdue_gap_max = max_excess / (avg_min if avg_min > 0 else 1.0)
        return tdue_gap, tdue_gap_max, tc_used, max_eta
