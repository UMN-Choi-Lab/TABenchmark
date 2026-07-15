"""Reference MSA solvers for TD-UE and TD-SO (Peeta & Mahmassani 1995, adr-031).

NON-certified research code: the certifier (:class:`~tabench.metrics.tdta_gaps.
TDTAEvaluator`), never the solver's claim, is the arbiter (the vi-due lesson).
Both solvers are the paper's method of successive averages with the fixed
predetermined step ``1/(l+1)`` (Eq. 16), sharing every component except the path
cost the all-or-nothing (AON) direction minimizes — average experienced time for
UE (Eq. 5, §4.3) versus time-dependent path MARGINAL time for SO (Eqs. 6/18,
§4.1) — exactly the paper's UE/SO symmetry. Because the path set is ENUMERATED in
the scenario (P2), the paper's Ziliaskopoulos-Mahmassani time-dependent shortest
path + column generation collapses to evaluating each declared path's cost and
AON-ing onto the minimum per OD and departure interval.

Deviation from the paper, disclosed (adr-031): the stopping rule is a budgeted
gap-aware iterate selection (emit the best-certified iterate), NOT the paper's
solution-stability count ``N(eps) <= Omega`` — a simulation-based MSA has no
convergence guarantee (the paper says so), so the benchmark reports the certified
gap of the best iterate, never a convergence claim.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from ..dnl.link import interp_curve
from ..dnl.output import DNLOutput
from .artifact import TDPathFlows
from .loader import PathLoader
from .scenario import TDTAScenario

__all__ = ["solve_td_ue", "solve_td_so"]


def _earliest_time(curve: np.ndarray, level: float, dt: float) -> float:
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
    if not math.isfinite(t):
        return float(curve[-1]) if t > 0 else 0.0
    return interp_curve(curve, t, dt)


def _od_step_demand(scenario: TDTAScenario) -> dict[tuple[int, int], np.ndarray]:
    """Per-OD vehicles departing in each grid step (fixed, the demand table)."""
    edges = scenario.grid.edges
    dcum = scenario.demand.cumulative(edges)  # (K+1, Z, Z)
    out: dict[tuple[int, int], np.ndarray] = {}
    for od in scenario.paths_by_od():
        col = dcum[:, od[0] - 1, od[1] - 1]
        out[od] = np.diff(col)
    return out


def _path_average_cost(
    scenario: TDTAScenario, out: DNLOutput, path, t: float, path_cum_p: np.ndarray, ff
) -> float:
    """Average experienced time of a marginal traveler entering ``path`` at
    departure time ``t`` on the frozen loaded curves (the UE AON cost)."""
    dt = out.grid.dt
    links = path.links
    a0 = int(links[0])
    level = _interp(path_cum_p, t, dt)
    exit_t = max(t + float(ff[a0]), _earliest_time(out.n_out[a0], level, dt))
    for b in links[1:]:
        if not math.isfinite(exit_t):
            return math.inf
        b = int(b)
        lvl = _interp(out.n_in[b], exit_t, dt)
        exit_t = max(exit_t + float(ff[b]), _earliest_time(out.n_out[b], lvl, dt))
    return exit_t - t


def _link_marginals(scenario: TDTAScenario, out: DNLOutput):
    """Per used link, per edge: (average travel time ``T_a[k]``, local marginal
    ``T_a + x_a dT/dx`` via the paper's 3-point quadratic fit, Fig. 3)."""
    dt = out.grid.dt
    n_edges = out.grid.n_steps + 1
    avg: dict[int, np.ndarray] = {}
    marg: dict[int, np.ndarray] = {}
    for a in scenario.used_links():
        n_in, n_out = out.n_in[a], out.n_out[a]
        x = np.clip(n_in - n_out, 0.0, None)  # storage (occupancy)
        ff = float(scenario.dynamics.length[a] / scenario.dynamics.free_speed[a])
        t_a = np.full(n_edges, ff)
        for k in range(n_edges):
            lvl = float(n_in[k])
            if lvl <= 0.0:
                continue
            te = _earliest_time(n_out, lvl, dt)
            if math.isfinite(te):
                t_a[k] = max(ff, te - k * dt)
        # dT/dx by a quadratic through the 3 most recent (x, T) points (Fig. 3);
        # fall back to a 2-point slope, then 0, when points coincide.
        dtdx = np.zeros(n_edges)
        for k in range(n_edges):
            lo = max(0, k - 2)
            xs = x[lo : k + 1]
            ts = t_a[lo : k + 1]
            dtdx[k] = _local_slope(xs, ts, x[k])
        avg[a] = t_a
        marg[a] = t_a + x * dtdx
    return avg, marg


def _local_slope(xs: np.ndarray, ts: np.ndarray, x0: float) -> float:
    """Slope dT/dx at ``x0`` from up to 3 (x, T) samples: quadratic fit if the
    three abscissae are distinct, else a finite difference, else 0."""
    ux, idx = np.unique(xs, return_index=True)
    if ux.size >= 3:
        # a nearly-collinear triple is fine here (the fit degrades gracefully to
        # the line it should be); the ill-conditioning warning is expected noise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.exceptions.RankWarning)
            coeffs = np.polyfit(xs, ts, 2)
        return float(2.0 * coeffs[0] * x0 + coeffs[1])
    if ux.size == 2:
        tu = ts[idx]
        return float((tu[1] - tu[0]) / (ux[1] - ux[0]))
    return 0.0


def _path_marginal_cost(
    scenario: TDTAScenario, out: DNLOutput, path, t: float, path_cum_p, ff, marg
) -> float:
    """Time-dependent path marginal cost: accumulate each link's local MARGINAL
    time while advancing the arrival clock by AVERAGE times (the paper's
    penalty-vs-movement split, §4.2.4). Each link's marginal is sampled at the
    interval the traveler EXITS it — the congestion (and hence the externality
    ``x_a dT/dx``) the marginal vehicle actually contributes to lives in the queue
    it sits through, not the empty cell it enters at ``t=0`` (the cold-start trap
    that a start-of-traversal sample falls into)."""
    dt = out.grid.dt
    links = path.links
    a0 = int(links[0])
    level = _interp(path_cum_p, t, dt)
    exit_t = max(t + float(ff[a0]), _earliest_time(out.n_out[a0], level, dt))
    k = min(int(exit_t / dt), out.grid.n_steps) if math.isfinite(exit_t) else out.grid.n_steps
    cost = float(marg[a0][k])
    for b in links[1:]:
        b = int(b)
        lvl = _interp(out.n_in[b], exit_t, dt)
        exit_t = max(exit_t + float(ff[b]), _earliest_time(out.n_out[b], lvl, dt))
        k = min(int(exit_t / dt), out.grid.n_steps) if math.isfinite(exit_t) else out.grid.n_steps
        cost += float(marg[b][k])
    return cost


def _msa(scenario: TDTAScenario, iters: int, mode: str) -> TDPathFlows:
    from ..metrics.tdta_gaps import TDTAEvaluator

    K = scenario.grid.n_steps
    ff = scenario.dynamics.length / scenario.dynamics.free_speed
    paths_by_od = scenario.paths_by_od()
    step_dem = _od_step_demand(scenario)
    # initialize: all of each OD's per-step demand on its first declared path
    dep = np.zeros((scenario.n_paths, K))
    for od, plist in paths_by_od.items():
        dep[plist[0]] = step_dem[od]

    evaluator = TDTAEvaluator(scenario)
    best_dep = dep.copy()
    best_gap = math.inf
    trajectory: list[float] = []
    for it in range(iters):
        out = PathLoader(scenario, dep, extra_steps=evaluator._clearing_pad()).run()
        path_cum = _path_cum(dep, out.grid.n_steps, K)
        marg = _link_marginals(scenario, out)[1] if mode == "so" else None
        y = np.zeros_like(dep)
        for od, plist in paths_by_od.items():
            for k in range(K):
                if step_dem[od][k] <= 0.0:
                    continue
                t = (k + 0.5) * scenario.grid.dt
                if mode == "so":
                    costs = [
                        _path_marginal_cost(
                            scenario, out, scenario.paths[p], t, path_cum[p], ff, marg
                        )
                        for p in plist
                    ]
                else:
                    costs = [
                        _path_average_cost(scenario, out, scenario.paths[p], t, path_cum[p], ff)
                        for p in plist
                    ]
                y[plist[int(np.argmin(costs))], k] = step_dem[od][k]
        metrics = evaluator.certify(TDPathFlows(scenario.content_hash(), dep))
        score = metrics["so_bound_gap"] if mode == "so" else metrics["tdue_gap"]
        if metrics["feasible"] == 1.0 and math.isfinite(score) and score < best_gap:
            best_gap, best_dep = score, dep.copy()
        trajectory.append(float(score) if math.isfinite(score) else math.nan)
        dep = dep + (1.0 / (it + 2.0)) * (y - dep)  # MSA step 1/(l+1) (Eq. 16)

    return TDPathFlows(
        scenario.content_hash(),
        best_dep,
        provenance={"mode": mode, "iters": iters, "best_gap": best_gap, "trajectory": trajectory},
    )


def _path_cum(dep: np.ndarray, n_ext: int, K: int) -> np.ndarray:
    cum = np.zeros((dep.shape[0], n_ext + 1))
    np.cumsum(dep, axis=1, out=cum[:, 1 : K + 1])
    cum[:, K + 1 :] = cum[:, K : K + 1]
    return cum


def solve_td_ue(scenario: TDTAScenario, iters: int = 40) -> TDPathFlows:
    """MSA on least-experienced-time paths (the paper's UE algorithm, §4.3).
    Returns the best-certified iterate (lowest ``tdue_gap``)."""
    return _msa(scenario, iters, mode="ue")


def solve_td_so(scenario: TDTAScenario, iters: int = 40) -> TDPathFlows:
    """MSA on least-path-MARGINAL-time paths (the paper's SO algorithm, §4.1)
    with local link marginals from the 3-point quadratic fit (Fig. 3). Returns
    the best-certified iterate (lowest ``so_bound_gap``)."""
    return _msa(scenario, iters, mode="so")
