"""Test-only point-queue reference link (dnl-core, adr-010).

Test scaffolding — NOT a benchmark model, not the vickrey sprint (that is a
departure-time equilibrium, not a link kernel), excluded from MODELS.md.
Never registered, never exported from ``tabench.dnl``; importable by tests
and by later sprints' tests as a comparison baseline (mirrors
``models/_stoch.py``).

Why the point queue cannot pre-empt any sprint: no backward wave / spillback
(that IS ctm/ltm's distinguishing physics), no cells (ctm/godunov), no event
construction (newell), no merge/diverge (node-model). Yet it exercises every
LinkModel interface end-to-end. Lineage: Vickrey (1969), attribution only;
the look-ahead sending convention follows Yperman's (2007) open thesis.
"""

from __future__ import annotations

import math

from .fd import FundamentalDiagram
from .grid import TimeGrid
from .link import LinkModel, interp_curve

__all__ = ["PointQueueLink"]


class PointQueueLink(LinkModel):
    """Point (vertical) queue with free-flow lag and capacity-limited exit.
    TEST/REFERENCE ONLY.

    Physics: a vehicle entering at ``t`` becomes exit-eligible at
    ``t + L/vf``; the exit server discharges at most ``q_max`` per tu;
    storage is unbounded (``fd.jam_density`` must be inf — validated at
    construction, so ``fd.q_cap`` IS the capacity); entry is capacity-capped::

        sending(k)   = min( interp_curve(n_in, t_{k+1} - L/vf, dt) - n_out[k],
                            q_max * dt )
        receiving(k) = q_max * dt

    Only PAST values are read: ``assert_wave_resolved`` at scenario
    construction guarantees ``dt <= L/vf``, hence ``t_{k+1} - L/vf <= t_k``.
    The look-ahead-by-one-step convention (vehicles reaching the exit DURING
    step ``k`` may leave within step ``k``) is the standard LTM/Newell
    sending convention (Yperman 2007) and makes the discrete solution EXACT
    at grid edges for piecewise-linear inputs with grid-aligned kinks. With
    inflow rate <= ``q_max`` at all times, ``n_out(t) = n_in(t - L/vf)``
    exactly (free-flow translation identity, asserted in tests).
    """

    def __init__(self, fd: FundamentalDiagram, length: float, grid: TimeGrid) -> None:
        if not math.isinf(fd.jam_density):
            raise ValueError(
                "PointQueueLink requires unbounded storage (fd.jam_density = inf); "
                f"got jam_density = {fd.jam_density!r} — finite-kappa physics is "
                "the ctm/ltm sprints' job, not the reference's"
            )
        super().__init__(fd, length, grid)
        self._lag = self.length / fd.free_speed  # L/vf, the free-flow time

    def sending(self, k: int) -> float:
        eligible = interp_curve(self.n_in, self.grid.dt * (k + 1) - self._lag, self.grid.dt)
        cap = self.fd.capacity * self.grid.dt
        # max() guards the float dust of (eligible - n_out) at emptiness, never physics
        return max(0.0, min(eligible - self.n_out[k], cap))

    def receiving(self, k: int) -> float:
        return self.fd.capacity * self.grid.dt
