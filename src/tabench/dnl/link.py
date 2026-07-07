"""Link-model interface and cumulative-curve machinery (dnl-core, adr-010).

Every DNL link model — the ctm, ltm, newell-kw, and godunov sprints, and the
test-only point-queue reference — is a :class:`LinkModel` subclass behind the
generic sending/receiving interface. All quantities are VEHICLES PER STEP
(counts), not rates: node models never see ``dt``, so min-allocations and
conservation sums are pure counts (Yperman 2007, the open primary for the
per-step vehicle-count convention; Daganzo's 1994/95 CTM papers are paywalled
and attributed unread, cross-verified from open restatements).

The base class owns the canonical emitted state — the cumulative inflow and
outflow curves ``n_in``/``n_out`` at grid edges with ``N[0] = 0`` — which is
exactly what :class:`~tabench.dnl.output.DNLOutput` carries and the harness
recomputes certificates from (P1). Subclasses own any internal state (cells,
queues) via the :meth:`LinkModel._advance_state` hook.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from .fd import FundamentalDiagram, TriangularFD
from .grid import TimeGrid

__all__ = ["interp_curve", "LinkModel", "LinkModelFactory"]


def interp_curve(curve: np.ndarray, t: float, dt: float) -> float:
    """Linear interpolation of a cumulative curve sampled on grid edges (G8).

    ``curve[j]`` is the value at ``t_j = j*dt``; ``t < 0`` returns 0.0 (the
    curve starts at ``N(0) = 0`` and nothing precedes it); ``t`` beyond the
    last edge returns ``curve[-1]`` (constant continuation). One audited
    interpolation reused by the point-queue reference and the ltm/newell
    sprints instead of three private copies.
    """
    t = float(t)
    if t < 0.0:
        return 0.0
    x = t / float(dt)
    j = int(x)
    if j >= curve.shape[0] - 1:
        return float(curve[-1])
    frac = x - j
    return float(curve[j] + frac * (curve[j + 1] - curve[j]))


class LinkModel(ABC):
    """One link's loading dynamics behind the generic S/R interface.

    Owns the canonical cumulative curves; subclasses own internal state.
    The loader calls, per step ``k``: :meth:`sending`/:meth:`receiving`
    (pure), lets the node models allocate transfer counts, then commits them
    via :meth:`advance`. Stability (CFL) conditions are each subclass's
    responsibility at construction (see grid.py).
    """

    def __init__(self, fd: FundamentalDiagram, length: float, grid: TimeGrid) -> None:
        self.fd = fd
        self.length = float(length)
        self.grid = grid
        # canonical emitted state: values at grid EDGES, N[0] = 0
        self.n_in = np.zeros(grid.n_steps + 1)  # (K+1,) float64
        self.n_out = np.zeros(grid.n_steps + 1)  # (K+1,) float64

    @abstractmethod
    def sending(self, k: int) -> float:
        """``S_a(k) >= 0``: max vehicles this link can DELIVER to its head
        node during step ``k`` (over ``[t_k, t_{k+1})``), given state at
        ``t_k``. Contract: ``S_a(k) <= fd.capacity * grid.dt`` (+ float
        slack). Pure (no mutation)."""

    @abstractmethod
    def receiving(self, k: int) -> float:
        """``R_a(k) >= 0``: max vehicles this link can ACCEPT from its tail
        node during step ``k``, given state at ``t_k``. Contract:
        ``R_a(k) <= fd.capacity * grid.dt``. Pure (no mutation)."""

    def advance(self, k: int, inflow: float, outflow: float) -> None:
        """Commit node-allocated transfer counts for step ``k``.

        Preconditions (loader-guaranteed; asserted in debug builds):
        ``0 <= outflow <= sending(k) + eps`` and
        ``0 <= inflow <= receiving(k) + eps``. The curve commit is FINAL —
        subclasses extend :meth:`_advance_state`, never this method::

            n_in[k+1]  = n_in[k]  + inflow
            n_out[k+1] = n_out[k] + outflow
        """
        if not 0 <= k < self.grid.n_steps:
            raise ValueError(
                f"LinkModel.advance: step {k} outside 0..{self.grid.n_steps - 1}"
            )
        if __debug__:
            eps = 1e-9 * max(1.0, self.fd.capacity * self.grid.dt)
            assert -eps <= inflow <= self.receiving(k) + eps, (
                f"advance step {k}: inflow {inflow} violates 0 <= inflow <= receiving"
            )
            assert -eps <= outflow <= self.sending(k) + eps, (
                f"advance step {k}: outflow {outflow} violates 0 <= outflow <= sending"
            )
        self.n_in[k + 1] = self.n_in[k] + inflow
        self.n_out[k + 1] = self.n_out[k] + outflow
        self._advance_state(k, inflow, outflow)

    def _advance_state(self, k: int, inflow: float, outflow: float) -> None:  # noqa: B027
        """Subclass hook for internal state (cells, queues). Default: no-op —
        deliberately NOT abstract (stateless models like the point-queue
        reference and LTM need no override)."""

    @property
    def cumulative_in(self) -> np.ndarray:
        """``(K+1,)`` cumulative inflow counts at grid edges (copy-on-emit)."""
        return self.n_in.copy()

    @property
    def cumulative_out(self) -> np.ndarray:
        """``(K+1,)`` cumulative outflow counts at grid edges (copy-on-emit)."""
        return self.n_out.copy()


# What a DNL "model" registers per link; the loader builds one LinkModel per
# network link from its canonical TriangularFD (LinkDynamics.fd(a)).
LinkModelFactory = Callable[[TriangularFD, float, TimeGrid], LinkModel]
