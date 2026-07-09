"""BottleneckScenario: the Vickrey (1969) single-bottleneck departure-time model.

A departure-TIME equilibrium — a different paradigm from the repo's route-choice
and link-loading models — so it lives in its own parallel module (like
``transit/`` and ``dnl/``), touching no road/DNL code. ``N`` travelers each choose
a departure time to trade off point-queue delay at a bottleneck of capacity ``s``
against schedule delay relative to a desired arrival time ``t_star`` (early penalty
``beta``, late penalty ``gamma``, travel-time value ``alpha``, standard ordering
``0 < beta < alpha < gamma``). The frozen, content-hashed scenario is the six
scalars only — no network topology.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

__all__ = ["BottleneckScenario"]


@dataclass(frozen=True)
class BottleneckScenario:
    """Frozen, content-hashed single-bottleneck instance (P2).

    ``n_travelers`` total demand ``N``; ``capacity`` bottleneck service rate ``s``
    (veh per time unit); ``alpha`` value of travel time; ``beta`` early-arrival
    penalty; ``gamma`` late-arrival penalty; ``t_star`` desired arrival time (same
    clock as departures). ``family`` is P7 lineage (defaults to ``name``,
    provenance only, unhashed).
    """

    name: str
    n_travelers: float
    capacity: float
    alpha: float
    beta: float
    gamma: float
    t_star: float
    family: str = ""

    def __post_init__(self) -> None:
        for field in ("n_travelers", "capacity", "alpha", "beta", "gamma", "t_star"):
            object.__setattr__(self, field, float(getattr(self, field)))
        if not self.family:
            object.__setattr__(self, "family", self.name)
        finite = (
            self.n_travelers,
            self.capacity,
            self.alpha,
            self.beta,
            self.gamma,
            self.t_star,
        )
        if any(not math.isfinite(v) for v in finite):
            raise ValueError(f"BottleneckScenario '{self.name}': all parameters must be finite")
        if self.n_travelers <= 0:
            raise ValueError(f"BottleneckScenario '{self.name}': n_travelers must be > 0")
        if self.capacity <= 0:
            raise ValueError(f"BottleneckScenario '{self.name}': capacity must be > 0")
        # The bottleneck equilibrium requires 0 < beta < alpha (an early minute is
        # cheaper than a travel minute, else no one departs early) and gamma > 0.
        if not (0.0 < self.beta < self.alpha):
            raise ValueError(
                f"BottleneckScenario '{self.name}': need 0 < beta < alpha "
                f"(got beta={self.beta}, alpha={self.alpha})"
            )
        if self.gamma <= 0.0:
            raise ValueError(f"BottleneckScenario '{self.name}': gamma must be > 0")

    @property
    def equilibrium_cost(self) -> float:
        """The analytic equilibrium generalized cost ``C* = beta*gamma/(beta+gamma)*N/s``."""
        return (self.beta * self.gamma / (self.beta + self.gamma)) * (
            self.n_travelers / self.capacity
        )

    def content_hash(self) -> str:
        """SHA-256 over the six scored scalars, domain-separated from every other
        scenario space (``"tabench-bottleneck-scenario-v1;"`` prefix)."""
        h = hashlib.sha256()
        h.update(b"tabench-bottleneck-scenario-v1;")
        for field in ("n_travelers", "capacity", "alpha", "beta", "gamma", "t_star"):
            h.update(f"{field}={float(getattr(self, field))!r};".encode())
        return h.hexdigest()
