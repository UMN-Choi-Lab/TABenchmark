"""Harness-side certification of emitted Newell three-detector fields (P1, adr-024).

Every scored quantity is a pure function of ``(ThreeDetectorScenario bytes,
emitted (m, K+1) interior field)``. The certifier REGENERATES the clean detector
curves from the hashed recipe by running LTM on the corridor (never a stored
array — the adr-023 discipline) and evaluates Newell's closed-form min at the
fixed hashed query grid to get the reference field ``M``; the emission's
``provenance`` is never trusted. Semantics mirror ``metrics/dnl_gaps.py``:
malformed emissions are CENSORED (``feasible = 0``, scored NaN, residual columns
populated); only wrong shapes raise.

GATING certificates (any failure censors) — the "valid cumulative field"
properties, guaranteed by ``reconstruct_field`` (whose suffix-min repair keeps the
reconstruction monotone even when masked windows drop a branch — adr-024 review),
so a legitimate noisy/partial estimate passes while garbage is caught:

* **C0 shape / hash / finiteness** — wrong array shape raises; a hash or query-grid
  mismatch censors; non-finite entries censor.
* **C1 zero start** — ``N(x_i, t_0) = 0`` at every queried position (the empty-start
  record).
* **C2 non-negativity** — counts are nonnegative (honest zero is NOT censored;
  only negatives beyond tolerance are).
* **C3 monotonicity** — each position's curve is nondecreasing in ``t``, gated on
  the maximum TOTAL DROP from the running high-water mark (the adr-022 convention
  — never per-step, so an eps-creep dip accumulated over K steps is caught by its
  total depth, while a sustained sub-eps dip is tolerated rather than
  duration-amplified).

TIER B (non-gating; raw residuals ALWAYS reported) — the pairwise two-sided Newell
envelopes between every emitted position pair (``x_i < x_j``), grid-edge relaxed
like dnl_gaps C4/C5:

* **forward free-flow** ``N(x_j, t) <= N(x_i, t - (x_j-x_i)/vf)``;
* **backward storage** ``N(x_i, t) <= N(x_j, t - (x_j-x_i)/w) + kappa*(x_j-x_i)``.

Both are necessary conditions for the EXACT LWR field (the reference ``M`` passes at
0), but a legitimate NOISY reconstruction violates them at noise scale — exactly as
CTM legitimately violates the backward-wave envelope C5 (dnl_gaps §Tier B). Gating
them would false-censor the honest naive baseline; a tolerance loose enough to admit
honest noise would also excuse a teleport. So the residual is the science and the
``interior_rmse`` ranking is the plausibility check (a physics-violating field is far
from the physical truth, so it cannot both cheat the ranking and pass). Reported with
the adr-020 two-scale split — a per-cell max residual AND the aggregate violated mass
— plus ``envelope_exact`` / ``envelope_at_resolution`` flags.

RANKING: ``interior_rmse`` = RMSE(emission - ``M``) over the fixed grid, on
NOISY/partial levels only (``rankable = 1``). The clean level is an oracle/validity
row where the min formula reproduces ``M`` by construction and is NEVER ranked
(``rankable = 0``) — the formula-evaluation triviality trap. Descriptive:
per-position RMSE, congested/free-flow region RMSE, and the self-report honesty diff.
"""

from __future__ import annotations

import logging

import numpy as np

from ..newell.scenario import ThreeDetectorScenario, _interp_curve, reconstruct_field
from ..newell.solve import ThreeDetectorField

__all__ = ["ThreeDetectorEvaluator"]

logger = logging.getLogger(__name__)

_SCORED_KEYS = ("interior_rmse", "congested_rmse", "freeflow_rmse", "max_abs_error")
_FLAG_KEYS = ("feasible", "rankable", "envelope_exact", "envelope_at_resolution")
_RESIDUAL_KEYS = (
    "zero_start_residual",
    "negativity_residual",
    "retraction_residual",
    "envelope_forward_residual",
    "envelope_backward_residual",
    "envelope_mass",
    "self_report_diff",
)


class ThreeDetectorEvaluator:
    """Model-blind three-detector certifier. Pure function of ``(scenario, field)``.

    Reuse across submissions on one scenario: the reference field ``M`` and the
    clean branch structure are regenerated once at construction. ``tol`` sets the
    absolute count tolerance ``eps_count = tol * V`` (``V`` = vehicle scale);
    ``env_tol_factor`` sets the Tier-B ``envelope_at_resolution`` threshold in
    units of one step of link capacity.
    """

    def __init__(
        self,
        scenario: ThreeDetectorScenario,
        tol: float = 1e-9,
        env_tol_factor: float = 1.0,
    ) -> None:
        self.scenario = scenario
        self.tol = float(tol)
        self.env_tol_factor = float(env_tol_factor)
        self._hash = scenario.content_hash()
        self._dt = scenario.grid.dt
        self._n_steps = scenario.grid.n_steps
        self._x = scenario.x_query
        self._m = self._x.shape[0]

        self._V = max(1.0, scenario.total_demand)
        self._eps_count = self.tol * self._V
        apex = scenario.vf * scenario.w * scenario.kappa / (scenario.vf + scenario.w)
        cap0 = apex if scenario.capacity is None else min(apex, scenario.capacity)
        self._F = cap0 * self._dt  # per-step flow scale

        # Regenerate the clean detector curves and the reference field M, and mark
        # which cells the congested branch binds in the exact solution (the
        # queued region where reconstruction is hardest — descriptive columns).
        times, n_up, n_dn = scenario.truth_boundary_curves()
        self._M = reconstruct_field(
            scenario.vf, scenario.w, scenario.kappa, scenario.length,
            times, n_up, n_dn, self._x, self._dt,
        )
        self._cong_active = self._congested_mask(times, n_up, n_dn)
        self._rankable = scenario.noise != "none"

        # sorted position pairs for the pairwise envelopes
        order = np.argsort(self._x, kind="stable")
        self._pairs = [
            (int(order[a]), int(order[b]))
            for a in range(self._m)
            for b in range(a + 1, self._m)
            if self._x[order[a]] < self._x[order[b]]
        ]

    def _congested_mask(self, times, n_up, n_dn) -> np.ndarray:
        sc = self.scenario
        mask = np.zeros((self._m, times.shape[0]), dtype=bool)
        for i in range(self._m):
            x = float(self._x[i])
            free = _interp_curve(n_up, times - x / sc.vf, self._dt)
            cong = _interp_curve(n_dn, times - (sc.length - x) / sc.w, self._dt) + sc.kappa * (
                sc.length - x
            )
            mask[i] = cong <= free
        return mask

    # ------------------------------------------------------------------

    def _censored(self, reason: str) -> dict[str, float]:
        logger.info("three-detector field censored: %s", reason)
        metrics = dict.fromkeys(_SCORED_KEYS, float("nan"))
        for key in _FLAG_KEYS:
            metrics[key] = 0.0
        for key in _RESIDUAL_KEYS:
            metrics[key] = float("inf")
        for i in range(self._m):
            metrics[f"interior_rmse_x{i}"] = float("nan")
        metrics["rankable"] = 1.0 if self._rankable else 0.0
        return metrics

    def _aoa(self, tau: float) -> np.ndarray:
        """Grid-edge relaxation j+(k) = index_at_or_after(t_k - tau): the LATER
        edge under a nondecreasing curve makes the envelope a sound relaxation
        (dnl_gaps._envelope_indices)."""
        k = np.arange(self._n_steps + 1)
        j = np.ceil(k - tau / self._dt - 1e-12).astype(np.int64)
        return np.clip(j, 0, self._n_steps)

    def _envelopes(self, field: np.ndarray) -> tuple[float, float, float]:
        """Tier-B pairwise Newell envelope residuals (forward, backward) and the
        aggregate violated mass across all pairs and both directions."""
        sc = self.scenario
        fwd = bwd = mass = 0.0
        for i, j in self._pairs:
            d = float(self._x[j] - self._x[i])
            rf = field[j] - field[i][self._aoa(d / sc.vf)]
            rb = field[i] - field[j][self._aoa(d / sc.w)] - sc.kappa * d
            fwd = max(fwd, float(rf.max(initial=0.0)))
            bwd = max(bwd, float(rb.max(initial=0.0)))
            mass += float(np.maximum(rf, 0.0).sum() + np.maximum(rb, 0.0).sum())
        return fwd, bwd, mass

    # ------------------------------------------------------------------

    def evaluate(self, field: ThreeDetectorField) -> dict[str, float]:
        """Certified metric dict for one emitted interior field."""
        arr = np.asarray(field.field, dtype=np.float64)
        if arr.shape != (self._m, self._n_steps + 1):
            raise ValueError(
                f"ThreeDetectorField shape {arr.shape} != "
                f"(m, K+1) = ({self._m}, {self._n_steps + 1})"
            )
        if field.scenario_hash != self._hash:
            return self._censored(
                f"wrong scenario hash: field ran on {field.scenario_hash!r}, "
                f"this instance is {self._hash!r} (C0)"
            )
        if not (
            np.array_equal(np.asarray(field.x_query, dtype=np.float64), self._x)
            and np.array_equal(np.asarray(field.times, dtype=np.float64), self.scenario.grid.edges)
        ):
            return self._censored("emitted query grid differs from the scenario grid (C0)")
        if not np.isfinite(arr).all():
            return self._censored("non-finite interior field (C0)")

        zero_start = float(np.abs(arr[:, 0]).max(initial=0.0))
        negativity = float(np.maximum(-arr, 0.0).max(initial=0.0))
        # max drop from the high-water mark (the due_gaps/adr-022 convention):
        # a SUM over cells would weight each dip by its duration, censoring a
        # sustained sub-eps dip that the convention deliberately tolerates
        # (adr-024 review)
        running = np.maximum.accumulate(arr, axis=1)
        retraction = float((running - arr).max(initial=0.0))
        fwd, bwd, mass = self._envelopes(arr)

        diagnostics = {
            "zero_start_residual": zero_start,
            "negativity_residual": negativity,
            "retraction_residual": retraction,
            "envelope_forward_residual": fwd,
            "envelope_backward_residual": bwd,
            "envelope_mass": mass,
            "self_report_diff": float("nan"),
        }
        env_max = max(fwd, bwd)
        diagnostics_flags = {
            "envelope_exact": 1.0 if env_max <= self._eps_count else 0.0,
            "envelope_at_resolution": (
                1.0 if env_max <= self.env_tol_factor * self._F + self._eps_count else 0.0
            ),
        }

        failures: list[str] = []
        if zero_start > self._eps_count:
            failures.append("C1 zero start")
        if negativity > self._eps_count:
            failures.append("C2 non-negativity")
        if retraction > self._eps_count:
            failures.append("C3 monotonicity (running-max total retraction)")
        if failures:
            metrics = self._censored(", ".join(failures))
            metrics.update(diagnostics)
            metrics.update(diagnostics_flags)
            return metrics

        err = arr - self._M
        interior_rmse = float(np.sqrt(np.mean(err**2)))
        cong = self._cong_active
        congested_rmse = (
            float(np.sqrt(np.mean(err[cong] ** 2))) if cong.any() else float("nan")
        )
        freeflow_rmse = (
            float(np.sqrt(np.mean(err[~cong] ** 2))) if (~cong).any() else float("nan")
        )
        self_report = field.provenance.get("interior_rmse")
        if self_report is not None:
            diagnostics["self_report_diff"] = abs(float(self_report) - interior_rmse)

        metrics: dict[str, float] = {
            "feasible": 1.0,
            "rankable": 1.0 if self._rankable else 0.0,
            "interior_rmse": interior_rmse,
            "congested_rmse": congested_rmse,
            "freeflow_rmse": freeflow_rmse,
            "max_abs_error": float(np.abs(err).max(initial=0.0)),
        }
        for i in range(self._m):
            metrics[f"interior_rmse_x{i}"] = float(np.sqrt(np.mean(err[i] ** 2)))
        metrics.update(diagnostics)
        metrics.update(diagnostics_flags)
        return metrics
