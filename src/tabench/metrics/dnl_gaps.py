"""Harness-side certification of dynamic-network-loading outputs (P1, adr-010).

Every scored/certified quantity is recomputed here as a pure function of
``(DynamicScenario bytes, DNLOutput arrays)`` — the emitted time-indexed
per-link cumulative counts ``n_in``/``n_out`` and the per-zone
``origin_release`` curve. Model self-reports are provenance only, never
trusted. Semantics mirror ``metrics/gaps.py``: invalid outputs are CENSORED
(``dnl_feasible = 0.0``, scored quantities NaN, residual columns populated
for diagnosis), never raised out of a scoring loop; only wrong shapes raise
(programming errors in the wrapper, not solution properties). The censor
reason goes to a module logger; the returned dict stays ``dict[str, float]``.

GATING certificates (any failure censors):

* **C0 shape & validity** — finite counts, exact zero start, monotone curves,
  matching time grid. A ``scenario_hash`` mismatch also censors (the run
  answers a different instance; hash-blind adapters must not be crashable).
* **C1 conservation** — per interior node and per step, vehicles leaving
  incoming links equal vehicles entering outgoing links; per origin zone,
  link inflow equals the released count; globally, released = arrived +
  in-network at every edge.
* **C2 capacity respect** — both boundary fluxes of every link are bounded
  by ``q_max * dt`` per step (the capped capacity under any single-regime FD).
* **C3 storage bounds** — link storage nonnegative everywhere and at most
  ``kappa * L`` on finite-jam links.
* **C4 free-flow causality** — Newell's upper cumulative envelope
  ``N_out(t) <= N_in(t - L/vf)``, relaxed to the at-or-after grid edge:
  sound on any emission grid, conservative by at most one step of slack,
  exact when ``L/(vf*dt)`` is an integer (the sanctioned CFL = 1 cell-aligned
  operating point; CFL < 1 cell schemes are NOT promised to pass and are
  answerable to the always-reported raw residual).
* **C6 FIFO / travel-time consistency** — level-matched inverse interpolation
  of both cumulative curves; no count level may traverse faster than free
  flow (the travel-time face of C4, sensitive to sub-step violations the
  grid-edge relaxation of C4 admits). Honesty note: with single-commodity
  cumulative counts, within-link FIFO is definitional — physical overtaking
  is not observable from aggregate counts (the same aggregate-vs-disaggregate
  limitation the static node-balance audit documents); per-commodity FIFO
  certification requires per-commodity emissions, a reserved additive
  extension.
* **C7 demand coupling** — no origin releases more than its cumulative
  demand (no phantom vehicles).

TIER B (non-gating; raw residuals ALWAYS reported):

* **C5 backward-wave envelope** — exact kinematic-wave theory requires
  ``N_in(t) <= N_out(t - L/w) + kappa*L`` on finite-jam links. The bound is
  necessary for the EXACT KW solution, but standard CTM at CFL = 1 under
  spillback violates it: hole (rarefaction) information numerically
  propagates at up to one cell per step (= vf > w), overshooting by
  ~(w/vf)^n_cells * q_max * dt per wave arrival — orders above tolerance.
  Gating C5 would falsely censor a correct convergent scheme; a tolerance
  loose enough to admit CTM would also excuse real bugs. The residual is the
  science; the flags are the convenience. Consumers ranking on KW fidelity
  gate on the flags/residuals explicitly. A teleporting or storage-violating
  model is still censored by the GATING C2/C3/C4 — demoting C5 opens no
  feasibility hole for free-flow physics; what it stops gating is only
  spillback-timing fidelity, exactly what first-order schemes legitimately
  differ on at finite resolution.

Envelope parameters ``(vf, w, kappa)`` come from each link's
``FundamentalDiagram.envelope_params()`` triangular majorant, so every
certificate stays sound (a necessary condition) for any concave FD subclass.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from ..dnl.output import DNLOutput
from ..dnl.scenario import DynamicScenario

__all__ = ["DNLEvaluator"]

logger = logging.getLogger(__name__)

#: scored quantities (NaN when censored; dnl_cleared is a flag, 0.0 when censored
#: — a censored run is not cleared, mirroring gaps.py's br_acceptable convention)
_SCORED_KEYS = (
    "tstt",
    "total_delay",
    "unserved_demand",
    "vehicles_completed",
    "vehicles_in_network",
)
#: residual/diagnostic columns, always reported (populated even on censored runs)
_RESIDUAL_KEYS = (
    "conservation_residual",
    "capacity_residual",
    "storage_residual",
    "causality_residual",
    "fifo_residual",
    "demand_coupling_residual",
    "kw_backward_residual",
    "kw_backward_residual_rel",
    "kw_backward_exact",
    "kw_backward_at_resolution",
)


def _earliest_times(curve: np.ndarray, levels: np.ndarray, dt: float) -> np.ndarray:
    """Earliest times at which the piecewise-linear ``curve`` (sampled at grid
    edges, nondecreasing) reaches each of the sorted ``levels``; ``+inf``
    where a level is never reached within the horizon. Independent
    re-implementation of the inversion (the evaluator never reuses the
    output-side convenience helpers, P1)."""
    times = np.full(levels.shape, math.inf)
    j = np.searchsorted(curve, levels, side="left")
    at_start = j == 0
    times[at_start] = 0.0
    mid = ~at_start & (j < curve.shape[0])
    jm = j[mid]
    lo = curve[jm - 1]
    hi = curve[jm]
    # side="left" guarantees lo < level <= hi, so hi - lo > 0.
    times[mid] = dt * (jm - 1 + (levels[mid] - lo) / (hi - lo))
    return times


class DNLEvaluator:
    """Model-blind DNL certifier. Pure function of ``(DynamicScenario, DNLOutput)``.

    Master tolerances (constructor-overridable): with vehicle scale
    ``V = max(1, total demand)`` and per-link per-step flow scale
    ``F_a = q_max_a * dt``,

    * ``eps_count = tol * V`` (absolute count comparisons),
    * ``eps_flow_a = tol * F_a + eps_count`` (per-step flow comparisons).

    Rationale: float64 accumulates <= K ~ 1e4 additions of O(F) numbers, a
    relative error ~ K * 2^-52 ~ 2e-12; the default ``tol = 1e-9`` sits three
    orders above rounding and six or more below any physical violation.
    ``kw_tol_factor`` sets the Tier-B ``kw_backward_at_resolution`` flag
    threshold in units of one step of capacity flow per link.
    """

    def __init__(
        self, scenario: DynamicScenario, tol: float = 1e-9, kw_tol_factor: float = 1.0
    ) -> None:
        self.scenario = scenario
        self.tol = float(tol)
        self.kw_tol_factor = float(kw_tol_factor)
        self._hash = scenario.content_hash()

        net = scenario.network
        dyn = scenario.dynamics
        grid = scenario.grid
        self._V = max(1.0, scenario.demand.total())
        self._eps_count = self.tol * self._V
        self._F = dyn.capacity * grid.dt  # (n_links,) veh per step

        # Triangular-majorant envelope parameters per link (G3): sound
        # (necessary-condition) certificates for any concave FD subclass.
        env = np.array(
            [dyn.fd(a).envelope_params() for a in range(dyn.n_links)], dtype=np.float64
        ).reshape(-1, 3)
        self._tau_ff = dyn.length / env[:, 0]  # L / vf
        self._tau_bw = dyn.length / env[:, 1]  # L / w; 0.0 where w = inf (unused there)
        self._J = env[:, 2] * dyn.length  # kappa * L; inf on point-queue links
        self._finite_mass = np.isfinite(self._J) & np.isfinite(env[:, 1])

        # Node incidence (1-based ids; index 0 unused) and sink links (heads
        # at zone nodes absorb: every vehicle exiting there has arrived).
        self._in_links = tuple(
            np.flatnonzero(net.term_node == n) for n in range(net.n_nodes + 1)
        )
        self._out_links = tuple(
            np.flatnonzero(net.init_node == n) for n in range(net.n_nodes + 1)
        )
        self._sink_links = np.flatnonzero(net.term_node <= net.n_zones)

        # Exact piecewise-linear cumulative demand per origin zone at the grid
        # edges, (n_zones, K+1), and per-origin totals (the full-demand scale
        # used by unserved_demand — released vehicles are compared against
        # everything the instance ever demands, so a too-short horizon reports
        # honestly instead of clearing, G7).
        self._D_edges = scenario.demand.cumulative(grid.edges).sum(axis=2).T
        last = float(scenario.demand.breakpoints[-1])
        self._D_total = scenario.demand.cumulative(np.array([last]))[0].sum(axis=1)

    # ------------------------------------------------------------------
    # censoring
    # ------------------------------------------------------------------

    def _censored(self, reason: str) -> dict[str, float]:
        logger.info("DNL output censored: %s", reason)
        metrics = dict.fromkeys(_SCORED_KEYS, float("nan"))
        metrics["dnl_cleared"] = 0.0
        for key in _RESIDUAL_KEYS:
            metrics[key] = float("inf")
        metrics["kw_backward_exact"] = 0.0
        metrics["kw_backward_at_resolution"] = 0.0
        metrics["dnl_feasible"] = 0.0
        return metrics

    # ------------------------------------------------------------------
    # certificates (each returns (residual, ok); pure, inputs never mutated)
    # ------------------------------------------------------------------

    def _conservation(
        self,
        d_in: np.ndarray,
        d_out: np.ndarray,
        d_rel: np.ndarray,
        n_out: np.ndarray,
        release: np.ndarray,
        storage: np.ndarray,
    ) -> tuple[float, bool]:
        """C1: per-node per-step conservation, origin coupling, global identity."""
        net = self.scenario.network
        resid, ok = 0.0, True
        for node in range(net.n_zones + 1, net.n_nodes + 1):
            ins, outs = self._in_links[node], self._out_links[node]
            if ins.size == 0 and outs.size == 0:
                continue
            r = float(np.abs(d_out[ins].sum(axis=0) - d_in[outs].sum(axis=0)).max(initial=0.0))
            resid = max(resid, r)
            ok = ok and r <= self.tol * max(1.0, float(self._F[ins].sum()))
        for o in range(1, net.n_zones + 1):
            outs = self._out_links[o]
            r = float(np.abs(d_in[outs].sum(axis=0) - d_rel[o - 1]).max(initial=0.0))
            resid = max(resid, r)
            ok = ok and r <= self.tol * max(1.0, float(self._F[outs].sum()))
        arrived = n_out[self._sink_links].sum(axis=0)
        g = float(np.abs(release.sum(axis=0) - arrived - storage.sum(axis=0)).max(initial=0.0))
        resid = max(resid, g)
        ok = ok and g <= self.tol * max(1.0, self._V)
        return resid, ok

    def _capacity(self, d_in: np.ndarray, d_out: np.ndarray) -> tuple[float, bool]:
        """C2: both boundary fluxes bounded by F_a = q_max_a * dt every step."""
        flux = np.maximum(d_in, d_out)
        resid = float(np.maximum(flux - self._F[:, None], 0.0).max(initial=0.0))
        bound = (self._F + (self.tol * self._F + self._eps_count))[:, None]
        return resid, bool((flux <= bound).all())

    def _storage(self, storage: np.ndarray) -> tuple[float, bool]:
        """C3: storage nonnegative everywhere; <= J_a = kappa_a * L_a where finite."""
        resid = float(np.maximum(-storage, 0.0).max(initial=0.0))
        ok = bool((storage >= -self._eps_count).all())
        fin = self._finite_mass
        if fin.any():
            over = storage[fin] - self._J[fin, None]
            resid = max(resid, float(np.maximum(over, 0.0).max(initial=0.0)))
            j_eps = self.tol * np.maximum(1.0, self._J[fin])
            ok = ok and bool((over <= j_eps[:, None]).all())
        return resid, ok

    def _envelope_indices(self, tau: float) -> np.ndarray:
        """Grid-edge relaxation j+(k) = index_at_or_after(t_k - tau) for all k:
        using the LATER edge under a nondecreasing n_in/n_out makes the bound a
        relaxation of the exact envelope — sound on any grid, conservative by
        at most one step, exact when tau/dt is an integer."""
        k = np.arange(self.scenario.grid.n_steps + 1)
        j = np.ceil(k - tau / self.scenario.grid.dt - 1e-12).astype(np.int64)
        return np.clip(j, 0, self.scenario.grid.n_steps)

    def _causality(self, n_in: np.ndarray, n_out: np.ndarray) -> tuple[float, bool]:
        """C4: N_out(t_k) <= N_in at the edge at-or-after t_k - L/vf (with
        n_in[:, 0] = 0, the j+ = 0 clip reproduces the t_k < L/vf branch)."""
        resid = 0.0
        for a in range(n_in.shape[0]):
            j = self._envelope_indices(self._tau_ff[a])
            resid = max(resid, float((n_out[a] - n_in[a, j]).max()))
        return max(resid, 0.0), resid <= self._eps_count

    def _fifo(self, n_in: np.ndarray, n_out: np.ndarray) -> tuple[float, bool]:
        """C6: level-matched curve inversion (earliest-time convention on
        plateaus); every entered count level must travel >= L/vf. Levels never
        exiting in-horizon are unreported, not violations. Curves are
        monotone-regularized by running max first (deviation <= eps_count once
        C0 passes; keeps the inversion well-posed on eps-scale dips)."""
        dt = self.scenario.grid.dt
        resid, ok = 0.0, True
        for a in range(n_in.shape[0]):
            cin = np.maximum.accumulate(n_in[a])
            cout = np.maximum.accumulate(n_out[a])
            levels = np.unique(cin)
            levels = levels[levels > 0.0]
            if levels.size == 0:
                continue
            t_entry = _earliest_times(cin, levels, dt)
            t_exit = _earliest_times(cout, levels, dt)
            reached = np.isfinite(t_exit)
            if not reached.any():
                continue
            deficit = self._tau_ff[a] - (t_exit[reached] - t_entry[reached])
            m = float(deficit.max())
            resid = max(resid, m)
            ok = ok and m <= self.tol * max(1.0, float(self._tau_ff[a]))
            # Belt-and-braces: exit times nondecreasing in level — implied by
            # C0 monotonicity (post-regularization the inversion is exactly
            # monotone), asserted as a documented sanity check.
            ok = ok and bool((np.diff(t_exit[reached]) >= 0.0).all())
        return max(resid, 0.0), ok

    def _demand_coupling(self, release: np.ndarray) -> tuple[float, bool]:
        """C7: Release_o(t_k) <= D_o(t_k) at every origin and edge."""
        resid = float(np.maximum(release - self._D_edges, 0.0).max(initial=0.0))
        return resid, resid <= self._eps_count

    def _kw_backward(
        self, n_in: np.ndarray, n_out: np.ndarray
    ) -> tuple[float, float, float, float]:
        """C5 (Tier B, non-gating): N_in(t) <= N_out(t - L/w) + kappa*L on
        finite-jam links, grid-edge relaxed like C4 (n_out[:, 0] = 0 makes the
        j+ = 0 clip reproduce the t_k < L/w branch). Returns
        ``(residual, residual_rel, exact_flag, at_resolution_flag)``; exempt
        (point-queue) links contribute 0, and if every link is exempt the
        residuals are 0.0 and both flags 1.0."""
        resid, rel, at_res = 0.0, 0.0, True
        for a in np.flatnonzero(self._finite_mass):
            j = self._envelope_indices(self._tau_bw[a])
            r = float(np.maximum(n_in[a] - n_out[a, j] - self._J[a], 0.0).max())
            resid = max(resid, r)
            rel = max(rel, r / max(1.0, float(self._J[a])))
            at_res = at_res and r <= self.kw_tol_factor * self._F[a] + self._eps_count
        exact = 1.0 if resid <= self._eps_count else 0.0
        return resid, rel, exact, 1.0 if at_res else 0.0

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    def evaluate(self, out: DNLOutput) -> dict[str, float]:
        """Certified metrics for one emitted DNL output.

        Infeasible or invalid outputs are censored (``dnl_feasible = 0``, NaN
        scored quantities, residual columns populated), never scored and never
        raised out of the scoring loop. Only wrong-shaped arrays raise, since
        that is a programming error in the wrapper, not a solution property.
        """
        net = self.scenario.network
        grid = self.scenario.grid
        if out.grid != grid:
            return self._censored(
                f"time grid mismatch: output has {out.grid}, scenario has {grid} (C0)"
            )
        edges = grid.n_steps + 1
        if out.n_in.shape != (net.n_links, edges):
            raise ValueError(
                f"DNLOutput n_in/n_out shape {out.n_in.shape} != ({net.n_links}, {edges})"
            )
        if out.origin_release.shape != (net.n_zones, edges):
            raise ValueError(
                f"DNLOutput origin_release shape {out.origin_release.shape} "
                f"!= ({net.n_zones}, {edges})"
            )
        n_in, n_out, release = out.n_in, out.n_out, out.origin_release
        if not (
            np.isfinite(n_in).all() and np.isfinite(n_out).all() and np.isfinite(release).all()
        ):
            return self._censored("non-finite cumulative counts (C0)")
        if out.scenario_hash != self._hash:
            return self._censored(
                f"wrong scenario hash: output was run on {out.scenario_hash!r}, "
                f"this instance is {self._hash!r} (C0)"
            )

        d_in = np.diff(n_in, axis=1)
        d_out = np.diff(n_out, axis=1)
        d_rel = np.diff(release, axis=1)
        storage = n_in - n_out

        failures: list[str] = []
        starts_zero = bool(
            (n_in[:, 0] == 0.0).all()
            and (n_out[:, 0] == 0.0).all()
            and (release[:, 0] == 0.0).all()
        )
        monotone = bool(
            min(
                d_in.min(initial=0.0), d_out.min(initial=0.0), d_rel.min(initial=0.0)
            )
            >= -self._eps_count
        )
        if not (starts_zero and monotone):
            failures.append("C0 validity (zero start / monotone counts)")

        cons_resid, c1_ok = self._conservation(d_in, d_out, d_rel, n_out, release, storage)
        cap_resid, c2_ok = self._capacity(d_in, d_out)
        sto_resid, c3_ok = self._storage(storage)
        caus_resid, c4_ok = self._causality(n_in, n_out)
        fifo_resid, c6_ok = self._fifo(n_in, n_out)
        dc_resid, c7_ok = self._demand_coupling(release)
        kw_resid, kw_rel, kw_exact, kw_at_res = self._kw_backward(n_in, n_out)
        for ok, label in (
            (c1_ok, "C1 conservation"),
            (c2_ok, "C2 capacity"),
            (c3_ok, "C3 storage"),
            (c4_ok, "C4 free-flow causality"),
            (c6_ok, "C6 FIFO travel time"),
            (c7_ok, "C7 demand coupling"),
        ):
            if not ok:
                failures.append(label)

        diagnostics = {
            "conservation_residual": cons_resid,
            "capacity_residual": cap_resid,
            "storage_residual": sto_resid,
            "causality_residual": caus_resid,
            "fifo_residual": fifo_resid,
            "demand_coupling_residual": dc_resid,
            "kw_backward_residual": kw_resid,
            "kw_backward_residual_rel": kw_rel,
            "kw_backward_exact": kw_exact,
            "kw_backward_at_resolution": kw_at_res,
        }
        if failures:
            metrics = self._censored(", ".join(failures))
            metrics.update(diagnostics)
            return metrics

        # Scored / derived quantities (recomputed, never trusted). The origin
        # queue D_o(t) - Release_o(t) is exact piecewise-linear, so the
        # trapezoid is exact for piecewise-linear curves with aligned kinks.
        queue = self._D_edges - release
        dt = grid.dt
        tstt = float(
            0.5
            * dt
            * ((storage[:, :-1] + storage[:, 1:]).sum() + (queue[:, :-1] + queue[:, 1:]).sum())
        )
        total_delay = tstt - float((self._tau_ff * n_in[:, -1]).sum())
        unserved = float((self._D_total - release[:, -1]).sum())
        completed = float(n_out[self._sink_links, -1].sum())
        in_network = float(storage[:, -1].sum())
        metrics: dict[str, float] = {
            "tstt": tstt,
            "total_delay": total_delay,
            "unserved_demand": unserved,
            "vehicles_completed": completed,
            "vehicles_in_network": in_network,
            "dnl_cleared": (
                1.0 if in_network <= self._eps_count and unserved <= self._eps_count else 0.0
            ),
        }
        metrics.update(diagnostics)
        metrics["dnl_feasible"] = 1.0
        return metrics
