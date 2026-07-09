"""Link Transmission Model link (Yperman 2007) — the Newell-Daganzo cumulative
-curve loading method as a DNL LinkModel.

Like :class:`~tabench.dnl.ctm.CTMLink`, an additive :class:`LinkModel` on the
generic sending/receiving interface — but STATELESS beyond the base cumulative
curves ``n_in`` (upstream) / ``n_out`` (downstream): LTM reads Newell's shifted
cumulative curves directly, so there are no cells and ``_advance_state`` is a
no-op. It carries NO turning logic (node models handle junctions).

Sending is the point-queue's look-ahead convention (a vehicle entering the
upstream end is free-flow-eligible to leave the downstream end ``L/vf`` later);
receiving adds the finite backward wave the point queue lacks — space freed by a
downstream departure reappears at the upstream end ``L/w`` later:

    sending(k)   = min( N_up(t_{k+1} - L/vf) - N_dn(t_k),  q_max*dt )
    receiving(k) = min( N_dn(t_{k+1} - L/w) + kappa*L - N_up(t_k),  q_max*dt )

``sending`` is byte-identical to the point-queue reference's; the ``kappa*L``
storage term in ``receiving`` is exactly what turns the point queue's
unconstrained receiving into a finite-storage backward wave (so LTM requires a
finite jam density, mirror-image of the point queue's ``kappa = inf``).

Because these are exact evaluations of the piecewise-linear cumulative curves at
the Newell shifts (not a cell average), LTM has no interior discretisation and
in principle no numerical diffusion for the backward wave (Boyles TNA §9.5.4) —
though on the small single-shock anchors here it and CTM agree to machine
precision (CTM's O((w/vf)^n_cells) spreading stays below the certificate
tolerance at that scale). Its concrete advantage over CTM is grid FLEXIBILITY:
LTM needs no CFL=1 cell alignment, only a wave-resolved grid
``dt <= min(L/vf, L/w)`` (the existing ``assert_wave_resolved`` bound, also the
causality guarantee that the look-ahead never reads a future value), so it runs
on coarser / non-cell-aligned grids that CTMLink rejects at construction.

Sourcing: Yperman (2007) PhD thesis (KU Leuven), eq. 4.31/4.35, and Boyles TNA
§9.5.2 eq. 9.65/9.67 — both OPEN and read. NOTE a sign convention: Yperman
typesets the backward term ``+L/w`` because his ``w`` is the SIGNED (negative)
backward-wave velocity; this repo (and Boyles) use ``wave_speed > 0`` as a
magnitude, so the equivalent form is ``- L/w`` — which is what this module uses,
verified against Boyles' worked example (Table 9.6, ``R(10) = 5``).
"""

from __future__ import annotations

import math

from .fd import FundamentalDiagram
from .grid import TimeGrid
from .link import LinkModel, interp_curve

__all__ = ["LTMLink"]


class LTMLink(LinkModel):
    """Newell-Daganzo link transmission model (finite jam density required)."""

    def __init__(self, fd: FundamentalDiagram, length: float, grid: TimeGrid) -> None:
        if math.isinf(fd.jam_density):
            raise ValueError(
                "LTMLink requires a finite jam density (the backward-wave receiving "
                f"term needs kappa*L); the unbounded point queue is the reference, "
                f"got kappa={fd.jam_density!r}"
            )
        super().__init__(fd, length, grid)
        self._tau_ff = self.length / fd.free_speed  # L / vf
        self._tau_bw = self.length / fd.wave_speed  # L / w
        self._jam = fd.jam_density * self.length  # kappa * L (max vehicles on the link)
        self._cap_step = fd.capacity * grid.dt

    def sending(self, k: int) -> float:
        dt = self.grid.dt
        eligible = interp_curve(self.n_in, dt * (k + 1) - self._tau_ff, dt)
        return max(0.0, min(eligible - self.n_out[k], self._cap_step))

    def receiving(self, k: int) -> float:
        dt = self.grid.dt
        space = interp_curve(self.n_out, dt * (k + 1) - self._tau_bw, dt) + self._jam
        return max(0.0, min(space - self.n_in[k], self._cap_step))
