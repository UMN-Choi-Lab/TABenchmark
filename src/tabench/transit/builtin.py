"""Built-in analytic transit scenarios (Spiess & Florian 1989 common-lines oracle).

The classic common-lines example: one boarding stop (node 0) and one destination
(node 1) joined by two parallel lines (arcs). The optimal-strategy expected cost
and the frequency-share split are closed forms, recomputed here — no trusted
digits.
"""

from __future__ import annotations

import numpy as np

from .network import (
    TransitDemand,
    TransitNetwork,
    TransitReference,
    TransitScenario,
)

__all__ = [
    "common_lines_scenario",
    "common_lines_unattractive_scenario",
    "common_lines_expected_cost",
]


def common_lines_expected_cost(lines: list[tuple[float, float]]) -> tuple[float, list[int]]:
    """Closed-form optimal-strategy expected cost for parallel lines to the sink.

    ``lines`` is ``[(frequency, in_vehicle_time), ...]``. Returns ``(C*, attractive)``
    where ``C* = (1 + sum_{l in A} f_l t_l) / sum_{l in A} f_l`` (expected wait
    ``1/sum f`` + frequency-weighted ride) and ``attractive`` are the indices of
    the lines in the optimal strategy: sort by time, greedily include line ``l``
    while its onward cost ``t_l`` is strictly below the running expected cost.
    """
    order = sorted(range(len(lines)), key=lambda k: lines[k][1])
    f_sum = 0.0
    ft_sum = 0.0
    cost = np.inf
    attractive: list[int] = []
    for k in order:
        f, t = lines[k]
        if t < cost:  # strictly attractive (ties never lower the cost)
            f_sum += f
            ft_sum += f * t
            cost = (1.0 + ft_sum) / f_sum
            attractive.append(k)
        else:
            break  # sorted by time, so no later line is attractive either
    return cost, sorted(attractive)


def _common_lines_scenario(
    name: str, lines: list[tuple[float, float]], demand: float
) -> TransitScenario:
    n_arcs = len(lines)
    freq = np.array([f for f, _ in lines], dtype=np.float64)
    time = np.array([t for _, t in lines], dtype=np.float64)
    network = TransitNetwork(
        n_nodes=2,
        tail=np.zeros(n_arcs, dtype=np.int64),  # all board at stop 0
        head=np.ones(n_arcs, dtype=np.int64),  # all go to destination 1
        time=time,
        freq=freq,
    )
    dem = TransitDemand(
        origins=np.array([0]), destinations=np.array([1]), volumes=np.array([demand])
    )
    cost, attractive = common_lines_expected_cost(lines)
    reference = TransitReference(
        expected_total_cost=demand * cost,
        source="analytic",
        note=(
            f"Optimal-strategy expected cost {cost} per trip; attractive lines "
            f"{attractive}; frequency-share split over them."
        ),
    )
    return TransitScenario(
        name=name, network=network, demand=dem, family=f"builtin-{name}", reference=reference
    )


def common_lines_scenario(demand: float = 1000.0) -> TransitScenario:
    """Two attractive common lines (both boarded).

    Line 0: frequency 1/6 (6-min headway), in-vehicle 21 min; line 1: frequency
    1/12, in-vehicle 18 min. Combined frequency 1/4, optimal expected cost
    ``(1 + 1/6*21 + 1/12*18)/(1/4) = 6/(1/4) = 24`` min (wait 4 + ride 20); the
    two lines split the demand 2:1 by frequency (``v0 = 2/3 D``, ``v1 = 1/3 D``).
    """
    return _common_lines_scenario(
        "transit-common-lines", [(1.0 / 6.0, 21.0), (1.0 / 12.0, 18.0)], demand
    )


def common_lines_unattractive_scenario(demand: float = 1000.0) -> TransitScenario:
    """One attractive line, one excluded — the attractiveness-threshold test.

    Line 0: frequency 1/6, in-vehicle 15 min → alone gives ``1/(1/6) + 15 = 21``
    min. Line 1: frequency 1/12, in-vehicle 40 min; its onward cost 40 is NOT
    below 21, so it is excluded. Optimal cost 21 min, all demand on line 0.
    """
    return _common_lines_scenario(
        "transit-common-lines-unattractive", [(1.0 / 6.0, 15.0), (1.0 / 12.0, 40.0)], demand
    )
