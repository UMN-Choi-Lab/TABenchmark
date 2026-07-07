"""Time-grid primitives for dynamic network loading (DNL).

Design: docs/design/adr-010-dnl-core.md. The grid is model-agnostic: each
link-model sprint enforces its own stability (CFL) condition at construction
(CTM: ``vf*dt <= cell_len``; LTM: ``dt <= min(L/vf, L/w)``).
:func:`assert_wave_resolved` is the weakest condition every KW-consistent
discretization shares: one step must never outrun the fastest characteristic
across a whole link.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["TimeGrid", "assert_wave_resolved"]


@dataclass(frozen=True)
class TimeGrid:
    """Uniform time grid. Edges ``t_k = k * dt``, ``k = 0..n_steps``; native
    time units (P9).

    ``t0 = 0`` always; demand breakpoints share the clock. Uniform grid only
    in v1.
    """

    dt: float  # step length, > 0, finite
    n_steps: int  # K >= 1

    def __post_init__(self) -> None:
        if isinstance(self.n_steps, bool) or not isinstance(self.n_steps, (int, np.integer)):
            raise ValueError(f"TimeGrid n_steps must be an int, got {self.n_steps!r}")
        object.__setattr__(self, "n_steps", int(self.n_steps))
        object.__setattr__(self, "dt", float(self.dt))
        if not (math.isfinite(self.dt) and self.dt > 0):
            raise ValueError(f"TimeGrid dt must be finite and > 0, got {self.dt!r}")
        if self.n_steps < 1:
            raise ValueError(f"TimeGrid n_steps must be >= 1, got {self.n_steps}")

    @property
    def horizon(self) -> float:
        """Horizon ``T = dt * n_steps``."""
        return self.dt * self.n_steps

    @property
    def edges(self) -> np.ndarray:
        """Edge times, ``(n_steps + 1,)`` float64 ``= dt * arange(n_steps + 1)``."""
        return self.dt * np.arange(self.n_steps + 1, dtype=np.float64)

    def index_at_or_after(self, t: float) -> int:
        """Smallest ``k`` with ``t_k >= t - 1e-12*dt``, clipped to ``[0, n_steps]``.

        The ``1e-12*dt`` slack absorbs float rounding when ``t`` is meant to
        BE a grid edge (an edge lying just below ``t`` still counts as at-or-
        after). Used by the certificate envelopes (C4/C5 grid-edge relaxation).
        """
        t = float(t)
        if math.isnan(t):
            raise ValueError("TimeGrid.index_at_or_after: t must not be NaN")
        if math.isinf(t):
            return 0 if t < 0 else self.n_steps
        k = math.ceil(t / self.dt - 1e-12)
        return min(max(k, 0), self.n_steps)


def assert_wave_resolved(
    grid: TimeGrid,
    length: np.ndarray,
    free_speed: np.ndarray,
    wave_speed: np.ndarray,
) -> None:
    """Raise ValueError unless ``dt <= min_a min(L_a/vf_a, L_a/w_a)``
    (finite-``w`` terms only): one step never outruns the fastest
    characteristic across a whole link.

    Takes plain ``(n_links,)`` arrays (NOT ``LinkDynamics``) so primitives
    stay layer-1 pure. Infinite wave speeds (point-queue links, jam density
    inf) have no backward characteristic and contribute no ``L/w`` term.
    Equality holds at the sanctioned CFL = 1 operating point, so the bound is
    checked with a 1e-12 relative slack (float rounding, never physics).
    """
    length = np.ascontiguousarray(length, dtype=np.float64)
    free_speed = np.ascontiguousarray(free_speed, dtype=np.float64)
    wave_speed = np.ascontiguousarray(wave_speed, dtype=np.float64)
    if length.ndim != 1 or free_speed.shape != length.shape or wave_speed.shape != length.shape:
        raise ValueError("assert_wave_resolved: arrays must be 1-D with equal shapes")
    if length.size == 0:
        return
    if np.any(~np.isfinite(length)) or np.any(length <= 0):
        raise ValueError("assert_wave_resolved: length must be finite and > 0")
    if np.any(~np.isfinite(free_speed)) or np.any(free_speed <= 0):
        raise ValueError("assert_wave_resolved: free_speed must be finite and > 0")
    if np.any(np.isnan(wave_speed)) or np.any(wave_speed <= 0):
        raise ValueError("assert_wave_resolved: wave_speed must be > 0 (inf allowed)")
    bound = length / free_speed
    finite_w = np.isfinite(wave_speed)
    # length / inf == 0.0 exactly; the where() keeps only finite-w backward terms.
    bound = np.where(finite_w, np.minimum(bound, length / wave_speed), bound)
    a = int(np.argmin(bound))
    t_min = float(bound[a])
    if grid.dt > t_min * (1.0 + 1e-12):
        raise ValueError(
            f"time grid is not wave-resolved: dt={grid.dt} exceeds the fastest "
            f"characteristic crossing time min(L/vf, L/w) = {t_min} (link index {a})"
        )
