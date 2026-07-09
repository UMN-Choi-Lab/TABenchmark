"""Cell Transmission Model link (Daganzo 1994/1995) — first shipped DNL link model.

A :class:`~tabench.dnl.link.LinkModel` subclass on the generic sending/receiving
interface (adr-010): the loader and node models are unchanged, and a CTM link
carries NO turning logic (merges/diverges are the node models' job — Daganzo
1995's network extension lives there, not here). One link is discretised into
``n = L / (vf*dt)`` equal cells of length ``dx = vf*dt`` at the sanctioned CFL = 1
operating point, so a free-flow vehicle crosses exactly one cell per step.

Per step, using the Lebacque demand/supply the FD already exposes
(``demand_at`` = Delta, ``supply_at`` = Sigma), the inter-cell Godunov flux is

    y_{i->i+1} = min( demand_at(k_i), supply_at(k_{i+1}) ) * dt          (veh)

with the boundary fluxes supplied by the node allocation: ``sending`` is the last
cell's demand (what the link can deliver to its head), ``receiving`` the first
cell's supply (what it can accept). Cell occupancies then update by conservation.

Exactness (docs/design/adr-010, cross-verified from Boyles TNA §10.5 and Daganzo's
open ISTTT 1999 restatement; the 1994/1995 TR-B primaries are paywalled, attributed
unread): at CFL = 1 the free-flow branch is linear advection at Courant number 1 —
zero numerical diffusion, so ``n_out(t) = n_in(t - L/vf)`` is bit-exact. The
congested branch has Courant number ``w/vf < 1``, so a backward shock spreads by
O(one cell) — expected scheme physics, which is exactly why the harness demotes the
backward-wave envelope (C5) to a non-gating Tier-B residual for CTM.
"""

from __future__ import annotations

import math

import numpy as np

from .fd import FundamentalDiagram
from .grid import TimeGrid
from .link import LinkModel

__all__ = ["CTMLink"]


class CTMLink(LinkModel):
    """Godunov cell-transmission link at CFL = 1 (cell length ``vf*dt``).

    Requires a FINITE jam density (CTM models bounded storage / spillback; the
    unbounded point queue is the test-only reference, not CTM), and a
    cell-aligned link length ``L = n*vf*dt`` for integer ``n >= 1`` — the
    sanctioned operating point the certificates gate at. Internal state is the
    per-cell occupancy ``occ[i]`` (vehicles), all zero at ``t = 0``.
    """

    def __init__(self, fd: FundamentalDiagram, length: float, grid: TimeGrid) -> None:
        if math.isinf(fd.jam_density):
            raise ValueError(
                "CTMLink requires a finite jam density (bounded storage); an "
                f"unbounded point queue is the test-only reference, got kappa="
                f"{fd.jam_density!r}"
            )
        # Backward-wave CFL: at CFL = 1 the cell is dx = vf*dt, so the standard
        # CTM stability requirement dt <= dx/max(vf, w) reduces to w <= vf — the
        # congested-branch flux supply_at(k)*dt = w*(kappa-k)*dt could otherwise
        # overfill a cell past kappa*dx (physically impossible, C3-censored). A
        # faster backward wave needs a finer grid (dx = w*dt), off this forward-
        # CFL=1 operating point; the scenario-level assert_wave_resolved uses the
        # whole-link length, so it does NOT catch this per-cell condition.
        if fd.wave_speed > fd.free_speed * (1.0 + 1e-9):
            raise ValueError(
                f"CTMLink at CFL = 1 (cell length vf*dt) requires the backward wave "
                f"to be resolved per cell, i.e. w <= vf; got w={fd.wave_speed!r} > "
                f"vf={fd.free_speed!r} (a faster backward wave needs dx = w*dt, off "
                "the sanctioned forward-CFL=1 operating point)"
            )
        super().__init__(fd, length, grid)
        dx = fd.free_speed * grid.dt  # cell length at CFL = 1
        n_float = self.length / dx
        n_cells = int(round(n_float))
        if n_cells < 1 or abs(n_float - n_cells) > 1e-9 * max(1.0, n_float):
            raise ValueError(
                f"CTMLink needs a cell-aligned length L = n*vf*dt (CFL = 1): "
                f"L={self.length!r}, vf*dt={dx!r} give {n_float!r} cells, not an "
                "integer >= 1"
            )
        self._dx = dx
        self._n_cells = n_cells
        self._occ = np.zeros(n_cells, dtype=np.float64)

    def sending(self, k: int) -> float:
        """Last cell's Lebacque demand this step, capped by what it holds."""
        dens = self._occ[-1] / self._dx
        flux = float(self.fd.demand_at(np.array([dens]))[0]) * self.grid.dt
        return min(flux, float(self._occ[-1]))

    def receiving(self, k: int) -> float:
        """First cell's Lebacque supply this step."""
        dens = self._occ[0] / self._dx
        return float(self.fd.supply_at(np.array([dens]))[0]) * self.grid.dt

    def _advance_state(self, k: int, inflow: float, outflow: float) -> None:
        occ = self._occ
        if self._n_cells > 1:
            dens = occ / self._dx
            demand = self.fd.demand_at(dens)
            supply = self.fd.supply_at(dens)
            # y[i] = flux from cell i to cell i+1 (i = 0..n-2), in vehicles
            y = np.minimum(demand[:-1], supply[1:]) * self.grid.dt
            occ[:-1] -= y
            occ[1:] += y
        occ[0] += inflow
        occ[-1] -= outflow

    @property
    def n_cells(self) -> int:
        return self._n_cells

    @property
    def occupancy(self) -> np.ndarray:
        """Per-cell occupancy (vehicles), copy-on-read."""
        return self._occ.copy()
