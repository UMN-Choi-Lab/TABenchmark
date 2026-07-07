"""Fundamental diagrams and per-link kinematic-wave attributes (DNL primitives).

The interface is Lebacque's (1996) equilibrium demand/supply formalism:
``demand_at(k)`` = Delta(k) = Q(min(k, k_c)) and ``supply_at(k)`` = Sigma(k)
= Q(max(k, k_c)), so a Godunov cell flux is
``min(demand_at(k_i), supply_at(k_{i+1}))`` — the ctm and godunov sprints call
these per cell with zero interface change.

Sourcing (docs/design/adr-010): Daganzo's (1994, 1995) CTM papers and the
(vf, w, kappa, q_max) trapezoidal form are paywalled — attributed unread, the
formulas cross-verified from open restatements (Yperman's 2007 thesis is
open). Lebacque (1996) demand/supply likewise from open restatements.

Units (P9, native per network): densities [veh/du], speeds [du/tu],
flows [veh/tu].
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

__all__ = ["FundamentalDiagram", "TriangularFD", "LinkDynamics"]


class FundamentalDiagram(ABC):
    """Single-regime concave flow-density relation ``q = Q(k)`` on ``[0, kappa]``."""

    @property
    @abstractmethod
    def capacity(self) -> float:
        """``q_max = max_k Q(k)`` [veh/tu], finite."""

    @property
    @abstractmethod
    def critical_density(self) -> float:
        """Smallest density attaining ``q_max`` [veh/du]."""

    @property
    @abstractmethod
    def jam_density(self) -> float:
        """``kappa``; ``math.inf`` allowed (point queue)."""

    @property
    @abstractmethod
    def free_speed(self) -> float:
        """``vf = Q'(0+)`` [du/tu]."""

    @property
    @abstractmethod
    def wave_speed(self) -> float:
        """``w = |Q'(kappa-)|`` [du/tu]; inf iff ``kappa = inf``."""

    @abstractmethod
    def flow_at(self, k: np.ndarray) -> np.ndarray:
        """``Q(k)``, vectorized."""

    @abstractmethod
    def demand_at(self, k: np.ndarray) -> np.ndarray:
        """Lebacque demand ``Delta(k) = Q(min(k, k_c))``, vectorized."""

    @abstractmethod
    def supply_at(self, k: np.ndarray) -> np.ndarray:
        """Lebacque supply ``Sigma(k) = Q(max(k, k_c))``, vectorized."""

    def envelope_params(self) -> tuple[float, float, float]:
        """``(vf, w, kappa)`` of the TRIANGULAR MAJORANT of ``Q`` (G3): for any
        concave ``Q`` with ``Q(0) = 0`` and ``Q(kappa) = 0``,
        ``Q(k) <= min(vf*k, w*(kappa - k))``, so certificates built from these
        parameters remain sound (necessary conditions) for EVERY subclass.
        Default implementation returns ``(free_speed, wave_speed,
        jam_density)``; the godunov sprint's non-triangular FDs inherit it
        unchanged (tangent slopes at the ends majorize a concave function)."""
        return (self.free_speed, self.wave_speed, self.jam_density)


@dataclass(frozen=True)
class TriangularFD(FundamentalDiagram):
    """Triangular FD with optional trapezoidal capacity cap (G4, Daganzo's
    (vf, w, kappa, q_max) form): ``Q(k) = min(vf*k, q_cap, w*(kappa - k))``.

        capacity  q_max = min(vf*w*kappa/(vf + w), q_cap)      (finite kappa)
                  q_max = q_cap                                 (kappa = inf)
        critical_density k_c = q_max / vf   (smallest density attaining q_max)

    ``kappa = math.inf`` allowed => point-queue semantics: ``supply_at`` is
    ``q_max`` everywhere, ``wave_speed == inf``, and the storage certificate
    (C3 upper) and Tier-B backward-wave residual (C5) are skipped for that
    link. ``q_cap`` is REQUIRED iff ``kappa == inf``.
    """

    vf: float
    w: float
    kappa: float
    q_cap: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "vf", float(self.vf))
        object.__setattr__(self, "w", float(self.w))
        object.__setattr__(self, "kappa", float(self.kappa))
        if self.q_cap is not None:
            object.__setattr__(self, "q_cap", float(self.q_cap))
        if not (math.isfinite(self.vf) and self.vf > 0):
            raise ValueError(f"TriangularFD vf must be finite and > 0, got {self.vf!r}")
        if math.isnan(self.kappa) or self.kappa <= 0:
            raise ValueError(f"TriangularFD kappa must be > 0 (inf allowed), got {self.kappa!r}")
        if math.isnan(self.w) or self.w <= 0:
            raise ValueError(f"TriangularFD w must be > 0, got {self.w!r}")
        if math.isinf(self.w) != math.isinf(self.kappa):
            raise ValueError(
                "TriangularFD wave speed w must be infinite iff kappa is infinite "
                f"(point-queue semantics), got w={self.w!r}, kappa={self.kappa!r}"
            )
        if self.q_cap is not None and not (math.isfinite(self.q_cap) and self.q_cap > 0):
            raise ValueError(f"TriangularFD q_cap must be finite and > 0, got {self.q_cap!r}")
        if math.isinf(self.kappa):
            if self.q_cap is None:
                raise ValueError(
                    "TriangularFD with kappa = inf (point queue) requires q_cap "
                    "(it IS the capacity / exit service rate)"
                )
        elif self.q_cap is not None and self.q_cap > self._apex * (1.0 + 1e-9):
            raise ValueError(
                f"TriangularFD q_cap={self.q_cap!r} exceeds the geometric apex "
                f"vf*w*kappa/(vf+w)={self._apex!r}; a cap above the apex is a "
                "non-canonical byte-representation of the same FD (P2) — omit q_cap"
            )

    @property
    def _apex(self) -> float:
        """Uncapped triangle apex ``vf*w*kappa/(vf+w)`` (finite kappa only)."""
        return self.vf * self.w * self.kappa / (self.vf + self.w)

    @property
    def capacity(self) -> float:
        if math.isinf(self.kappa):
            return self.q_cap  # validated non-None for point queues
        return self._apex if self.q_cap is None else min(self._apex, self.q_cap)

    @property
    def critical_density(self) -> float:
        return self.capacity / self.vf

    @property
    def jam_density(self) -> float:
        return self.kappa

    @property
    def free_speed(self) -> float:
        return self.vf

    @property
    def wave_speed(self) -> float:
        return self.w

    def flow_at(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=np.float64)
        q = self.vf * k
        if self.q_cap is not None:
            q = np.minimum(q, self.q_cap)
        if math.isfinite(self.kappa):
            q = np.minimum(q, self.w * (self.kappa - k))
        return q

    def demand_at(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=np.float64)
        return np.minimum(self.vf * k, self.capacity)

    def supply_at(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=np.float64)
        if math.isinf(self.kappa):
            return np.full(k.shape, self.capacity, dtype=np.float64)
        return np.clip(np.minimum(self.capacity, self.w * (self.kappa - k)), 0.0, None)

    def envelope_params(self) -> tuple[float, float, float]:
        """``(vf, w, kappa)`` — the UNCAPPED tangents (capping only lowers
        flows, so the envelope still majorizes; the C2 certificate uses the
        capped :attr:`capacity`, staying tight at bottlenecks)."""
        return (self.vf, self.w, self.kappa)


@dataclass(frozen=True)
class LinkDynamics:
    """Per-link KW attributes, ``(n_links,)`` float64 arrays aligned with the
    ``Network`` link order.

    ``jam_density = inf`` means unbounded storage (point queue); on such links
    ``wave_speed`` must also be inf (and vice versa) and ``capacity`` is the
    point-queue exit service rate. On finite-jam links ``capacity`` must not
    exceed the triangular apex ``vf*w*kappa/(vf+w)`` (G4: a lower value is a
    trapezoidal cap). NaN is rejected everywhere; +inf is the only sanctioned
    non-finite, jointly for ``wave_speed``/``jam_density``.
    """

    length: np.ndarray  # L_a > 0 finite [du]
    free_speed: np.ndarray  # vf_a > 0 finite [du/tu]
    wave_speed: np.ndarray  # w_a > 0 [du/tu]; inf allowed iff jam_density[a] is inf
    jam_density: np.ndarray  # kappa_a > 0 [veh/du]; +inf = unbounded storage
    capacity: np.ndarray  # q_max_a > 0 finite [veh/tu]

    def __post_init__(self) -> None:
        names = ("length", "free_speed", "wave_speed", "jam_density", "capacity")
        for name in names:
            object.__setattr__(
                self, name, np.ascontiguousarray(getattr(self, name), dtype=np.float64)
            )
        if self.length.ndim != 1 or any(
            getattr(self, name).shape != self.length.shape for name in names
        ):
            raise ValueError("LinkDynamics arrays must be 1-D with equal shapes")
        for name in names:
            if np.any(np.isnan(getattr(self, name))):
                raise ValueError(f"LinkDynamics {name} must not contain NaN")
        if np.any(~np.isfinite(self.length)) or np.any(self.length <= 0):
            raise ValueError("LinkDynamics length must be finite and > 0")
        if np.any(~np.isfinite(self.free_speed)) or np.any(self.free_speed <= 0):
            raise ValueError("LinkDynamics free_speed must be finite and > 0")
        if np.any(self.wave_speed <= 0):
            raise ValueError("LinkDynamics wave_speed must be > 0 (inf allowed)")
        if np.any(self.jam_density <= 0):
            raise ValueError("LinkDynamics jam_density must be > 0 (inf allowed)")
        if not np.array_equal(np.isinf(self.wave_speed), np.isinf(self.jam_density)):
            raise ValueError(
                "LinkDynamics wave_speed may be infinite exactly where jam_density "
                "is infinite (point-queue links), and nowhere else"
            )
        if np.any(~np.isfinite(self.capacity)) or np.any(self.capacity <= 0):
            raise ValueError("LinkDynamics capacity must be finite and > 0")
        finite = np.isfinite(self.jam_density)
        if np.any(finite):
            vf, w, kj = self.free_speed[finite], self.wave_speed[finite], self.jam_density[finite]
            apex = vf * w * kj / (vf + w)
            if np.any(self.capacity[finite] > apex * (1.0 + 1e-9)):
                raise ValueError(
                    "LinkDynamics capacity exceeds the triangular apex "
                    "vf*w*kappa/(vf+w) on a finite-jam link (G4: capacity may only "
                    "cap the triangle, never raise it)"
                )

    @property
    def n_links(self) -> int:
        return self.length.shape[0]

    def fd(self, a: int) -> TriangularFD:
        """:class:`TriangularFD` for link ``a`` with a canonical cap:
        ``q_cap`` is set iff ``capacity[a] < geometric apex * (1 - 1e-12)`` or
        ``jam_density[a]`` is inf, else ``None`` (canonical: the same physics
        always builds the same FD object, P2)."""
        vf = float(self.free_speed[a])
        w = float(self.wave_speed[a])
        kj = float(self.jam_density[a])
        q = float(self.capacity[a])
        if math.isinf(kj):
            return TriangularFD(vf, w, kj, q_cap=q)
        apex = vf * w * kj / (vf + w)
        return TriangularFD(vf, w, kj, q_cap=q if q < apex * (1.0 - 1e-12) else None)
