"""Closed-form Vickrey bottleneck solutions and the emitted schedule artifact.

``BottleneckSchedule`` is the emitted, P1-certifiable artifact — a cumulative
departure curve ``R(t)`` on a time grid (the analogue of ``FlowState`` /
``TransitStrategy``). The closed-form UE and SO builders emit one; the certifier
(``metrics/bottleneck_gaps.py``) recomputes the queue and generalized costs from
the emitted curve alone, never trusting the ``r_early``/``t1``/``C*`` provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scenario import BottleneckScenario

__all__ = ["BottleneckSchedule", "ue_closed_form", "so_closed_form"]


@dataclass(frozen=True)
class BottleneckSchedule:
    """Emitted departure plan: cumulative departures ``cumulative[k]`` at grid
    edges ``times[k]`` (nondecreasing, starts at 0, ends at ``N``).

    ``provenance`` carries the solver's self-reported analytic quantities
    (``r_early``, ``t1``, ``C*`` ...) for inspection; the certifier ignores it.
    """

    scenario_hash: str
    times: np.ndarray  # (K+1,) float64, strictly increasing
    cumulative: np.ndarray  # (K+1,) float64, nondecreasing, [0] = 0
    provenance: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "times", np.ascontiguousarray(self.times, dtype=np.float64))
        object.__setattr__(
            self, "cumulative", np.ascontiguousarray(self.cumulative, dtype=np.float64)
        )
        if self.times.ndim != 1 or self.times.shape != self.cumulative.shape:
            raise ValueError("BottleneckSchedule times/cumulative must be 1-D of equal length")
        if self.times.shape[0] < 2:
            raise ValueError("BottleneckSchedule needs >= 2 grid edges")


def _grid(scenario: BottleneckScenario, t1: float, t2: float, n_steps: int) -> np.ndarray:
    """Uniform grid padded one step beyond the departure window on each side, so
    the endpoints carry an empty queue and the certifier sees the full support."""
    dt = (t2 - t1) / n_steps
    return np.linspace(t1 - dt, t2 + dt, n_steps + 3)


def ue_closed_form(scenario: BottleneckScenario, n_steps: int = 2000) -> BottleneckSchedule:
    """The Vickrey user-equilibrium departure schedule.

    Queue-building rate ``r_early = s*alpha/(alpha-beta)`` on ``[t1, t_n]`` then
    queue-dissipating ``r_late = s*alpha/(alpha+gamma)`` on ``[t_n, t2]``, with
    ``t1 = t* - C*/beta``, ``t2 = t* + C*/gamma``, ``t_n = t* - C*/alpha`` — every
    used departure time yields the same generalized cost ``C*``.
    """
    s, a, b, g = scenario.capacity, scenario.alpha, scenario.beta, scenario.gamma
    cstar = scenario.equilibrium_cost
    t1 = scenario.t_star - cstar / b
    t2 = scenario.t_star + cstar / g
    t_n = scenario.t_star - cstar / a
    r_early = s * a / (a - b)
    r_late = s * a / (a + g)
    times = _grid(scenario, t1, t2, n_steps)
    cum = np.where(
        times <= t1,
        0.0,
        np.where(
            times <= t_n,
            r_early * (times - t1),
            r_early * (t_n - t1) + r_late * (times - t_n),
        ),
    )
    cum = np.clip(cum, 0.0, scenario.n_travelers)
    return BottleneckSchedule(
        scenario_hash=scenario.content_hash(),
        times=times,
        cumulative=cum,
        provenance={
            "r_early": r_early,
            "r_late": r_late,
            "t1": t1,
            "t2": t2,
            "t_n": t_n,
            "equilibrium_cost": cstar,
        },
    )


def so_closed_form(scenario: BottleneckScenario, n_steps: int = 2000) -> BottleneckSchedule:
    """The system-optimum departure schedule: no queue — depart uniformly at the
    bottleneck capacity ``s`` over the SAME window ``[t1, t2]`` as the UE, paying
    schedule delay only. Total cost is exactly half the UE's (PoA = 2)."""
    s, b, g = scenario.capacity, scenario.beta, scenario.gamma
    cstar = scenario.equilibrium_cost
    t1 = scenario.t_star - cstar / b
    t2 = scenario.t_star + cstar / g
    times = _grid(scenario, t1, t2, n_steps)
    cum = np.clip(s * (times - t1), 0.0, scenario.n_travelers)
    return BottleneckSchedule(
        scenario_hash=scenario.content_hash(),
        times=times,
        cumulative=cum,
        provenance={"rate": s, "t1": t1, "t2": t2, "queue": 0.0},
    )
