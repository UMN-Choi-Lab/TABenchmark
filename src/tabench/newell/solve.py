"""Reference three-detector estimators and the emitted interior-field artifact.

``ThreeDetectorField`` is the P1-certifiable artifact (the analogue of
``DNLOutput`` / ``BottleneckSchedule``): the reconstructed interior cumulative
curves ``N_hat(x_i, t_k)`` on the scenario's fixed hashed query grid. The
certifier (``metrics/newell_gaps.py``) recomputes the reference field from the
hashed recipe and scores the emission alone — the ``provenance`` is never trusted.

Two reference estimators bracket the ranked task on a noisy detector level:

* ``newell_min`` (naive) — Newell's min on the observed day-mean curves after a
  running-max pass, the CHEAPEST way to make the emission monotone (C0). On a
  non-monotone Gaussian-reading level the running-max keeps the upward
  excursions, so it is biased high.
* ``newell_min_isotonic`` (denoised) — the L2-optimal isotonic (pool-adjacent-
  violators) monotone regression of each detector curve BEFORE the min, which
  averages the reading noise down and strictly beats the naive baseline in
  interior RMSE (anchor A4).

Both are pure functions of the model-visible :class:`ThreeDetectorProblem` (public
physics + grid + observed detectors), so neither can see the truth recipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .observe import DetectorObservation, observe_detectors
from .scenario import ThreeDetectorScenario, reconstruct_field

__all__ = [
    "ThreeDetectorProblem",
    "ThreeDetectorField",
    "problem_from_scenario",
    "newell_min",
    "newell_min_isotonic",
]


@dataclass(frozen=True)
class ThreeDetectorProblem:
    """The MODEL-VISIBLE task: public link physics, the record grid, the fixed
    interior query positions, and the seeded detector observation. Carries no
    truth recipe (demand, metering) — a submission cannot regenerate ground
    truth from it (the adr-023 information boundary)."""

    scenario_hash: str
    vf: float
    w: float
    kappa: float
    length: float
    times: np.ndarray  # (K+1,) grid edges
    dt: float
    x_query: np.ndarray  # (m,) interior query positions
    observation: DetectorObservation


@dataclass(frozen=True)
class ThreeDetectorField:
    """Emitted interior reconstruction: cumulative curves ``field[i, k]`` at query
    position ``x_query[i]`` and grid edge ``times[k]`` (``(m, K+1)``). ``provenance``
    is inspection-only; the certifier ignores it."""

    scenario_hash: str
    x_query: np.ndarray  # (m,)
    times: np.ndarray  # (K+1,)
    field: np.ndarray  # (m, K+1)
    provenance: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_query", np.ascontiguousarray(self.x_query, dtype=np.float64))
        object.__setattr__(self, "times", np.ascontiguousarray(self.times, dtype=np.float64))
        object.__setattr__(self, "field", np.ascontiguousarray(self.field, dtype=np.float64))
        if self.field.ndim != 2:
            raise ValueError("ThreeDetectorField field must be 2-D (m, K+1)")
        if self.field.shape != (self.x_query.shape[0], self.times.shape[0]):
            raise ValueError(
                f"ThreeDetectorField field shape {self.field.shape} != "
                f"(m, K+1) = ({self.x_query.shape[0]}, {self.times.shape[0]})"
            )


def problem_from_scenario(scenario: ThreeDetectorScenario) -> ThreeDetectorProblem:
    """Build the model-visible problem (public physics + seeded observation)."""
    obs = observe_detectors(scenario)
    return ThreeDetectorProblem(
        scenario_hash=scenario.content_hash(),
        vf=scenario.vf,
        w=scenario.w,
        kappa=scenario.kappa,
        length=scenario.length,
        times=scenario.grid.edges,
        dt=scenario.grid.dt,
        x_query=scenario.x_query.copy(),
        observation=obs,
    )


def _fill_nan(curve: np.ndarray) -> np.ndarray:
    """Linearly bridge NaN gaps (masked windows) using the surrounding finite
    samples so interpolation near a window edge stays well-posed. A shifted read
    landing INSIDE a window is dropped by the reconstruction, but a read within
    one grid step OUTSIDE a window edge interpolates against a bridged sample —
    those boundary-stencil reads are scored (adr-024 review), which is exactly
    why the bridge is linear from the surrounding true samples."""
    curve = np.asarray(curve, dtype=np.float64).copy()
    finite = np.isfinite(curve)
    if finite.all():
        return curve
    if not finite.any():
        return np.zeros_like(curve)
    idx = np.arange(curve.shape[0], dtype=np.float64)
    curve[~finite] = np.interp(idx[~finite], idx[finite], curve[finite])
    return curve


def _isotonic(y: np.ndarray) -> np.ndarray:
    """L2-optimal nondecreasing fit by pool-adjacent-violators (unit weights)."""
    y = np.asarray(y, dtype=np.float64)
    vals: list[float] = []
    wts: list[float] = []
    sizes: list[int] = []
    for value in y:
        v, wsum, size = float(value), 1.0, 1
        while vals and vals[-1] > v:
            pv, pw, ps = vals.pop(), wts.pop(), sizes.pop()
            v = (pv * pw + v * wsum) / (pw + wsum)
            wsum += pw
            size += ps
        vals.append(v)
        wts.append(wsum)
        sizes.append(size)
    out = np.empty(y.shape[0], dtype=np.float64)
    k = 0
    for v, size in zip(vals, sizes, strict=True):
        out[k : k + size] = v
        k += size
    return out


def _day_mean(curves: np.ndarray) -> np.ndarray:
    """Mean over days, ignoring NaN (masked) edges; all-NaN edges stay NaN."""
    finite = np.isfinite(curves)
    counts = finite.sum(axis=0)
    total = np.where(finite, curves, 0.0).sum(axis=0)
    return np.where(counts > 0, total / np.maximum(counts, 1), np.nan)


def _emit(problem: ThreeDetectorProblem, mono) -> ThreeDetectorField:
    obs = problem.observation
    up = _day_mean(obs.up)
    dn = _day_mean(obs.dn)
    # bridge masked windows, force the known empty start, then monotonize
    up = mono(np.maximum(_fill_nan(up), 0.0))
    dn = mono(np.maximum(_fill_nan(dn), 0.0))
    up[0] = 0.0
    dn[0] = 0.0
    field = reconstruct_field(
        problem.vf, problem.w, problem.kappa, problem.length,
        problem.times, up, dn, problem.x_query, problem.dt,
        up_windows=obs.up_windows, dn_windows=obs.dn_windows,
    )
    return ThreeDetectorField(
        scenario_hash=problem.scenario_hash,
        x_query=problem.x_query.copy(),
        times=problem.times.copy(),
        field=field,
    )


def newell_min(problem: ThreeDetectorProblem) -> ThreeDetectorField:
    """Naive baseline: running-max monotonization then Newell's min."""
    return _emit(problem, np.maximum.accumulate)


def newell_min_isotonic(problem: ThreeDetectorProblem) -> ThreeDetectorField:
    """Denoised baseline: isotonic regression then Newell's min (A4 winner)."""
    return _emit(problem, _isotonic)
