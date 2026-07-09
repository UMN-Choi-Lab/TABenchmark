"""Godunov-scheme link for general first-order (LWR) traffic flow (Lebacque 1996).

The Cell Transmission Model (``ctm``) is the Godunov scheme for the TRIANGULAR
fundamental diagram; ``GodunovLink`` runs the SAME ``min(demand, supply)`` Godunov
flux on a GENERAL concave FD — here the smooth :class:`~tabench.dnl.fd.GreenshieldsFD`
parabola ``Q(k) = vf*k*(1 - k/kappa)`` built from the link's ``(vf, kappa)``. That
makes it the first DNL link with a non-triangular FD, so it captures RAREFACTION
fans (smooth acceleration waves) a triangular FD cannot produce — the flux at a
transonic interface ``k_L > k_c > k_R`` is the sonic-point capacity ``q_max``, the
entropy-correct rarefaction value, exactly what ``min(demand, supply)`` returns.

It reuses the verified :class:`~tabench.dnl.ctm.CTMLink` cell update unchanged (the
scheme is identical — only the FD differs), and the certifier's triangular-majorant
envelopes remain sound (necessary conditions) for the concave Greenshields FD:
``Q(k) <= min(vf*k, vf*(kappa - k))``, so a Greenshields loading certifies with no
certificate change. A Greenshields link's ``LinkDynamics`` must be consistent with
the parabola — ``wave_speed = free_speed`` and ``capacity = vf*kappa/4`` — so that
``dynamics.fd(a)``'s triangular majorant and capacity match what the scheme loads.

Sourcing: Lebacque (1996, ISTTT 13) demand/supply Godunov formalism (the ``fd.py``
``demand_at``/``supply_at`` interface); Greenshields (1935) FD; Godunov (1959) /
LeVeque for the exact-Riemann-solver flux. Open restatements; no DOIs reproduced.
"""

from __future__ import annotations

from .ctm import CTMLink
from .fd import FundamentalDiagram, GreenshieldsFD
from .grid import TimeGrid

__all__ = ["GodunovLink"]


class GodunovLink(CTMLink):
    """Godunov cell scheme on the smooth Greenshields FD (built from ``vf, kappa``)."""

    def __init__(self, fd: FundamentalDiagram, length: float, grid: TimeGrid) -> None:
        greenshields = GreenshieldsFD(vf=fd.free_speed, kappa=fd.jam_density)
        # The scenario's LinkDynamics must be Greenshields-consistent, or the
        # certifier (which reads dynamics.capacity / envelope_params, NOT this FD)
        # would gate against the wrong capacity: require wave_speed == free_speed
        # and capacity == vf*kappa/4. A tiny relative slack absorbs float rounding.
        rel = 1e-9 * max(1.0, greenshields.capacity)
        if abs(fd.wave_speed - fd.free_speed) > 1e-9 * max(1.0, fd.free_speed):
            raise ValueError(
                f"GodunovLink needs a Greenshields-consistent link: wave_speed "
                f"({fd.wave_speed!r}) must equal free_speed ({fd.free_speed!r})"
            )
        if abs(fd.capacity - greenshields.capacity) > rel:
            raise ValueError(
                f"GodunovLink needs a Greenshields-consistent capacity vf*kappa/4 = "
                f"{greenshields.capacity!r}; the link declares {fd.capacity!r}"
            )
        # substitute the smooth Greenshields FD for the link's triangular one, then
        # run the shared CTMLink cell setup + guards (finite jam, w <= vf, alignment).
        super().__init__(greenshields, length, grid)
