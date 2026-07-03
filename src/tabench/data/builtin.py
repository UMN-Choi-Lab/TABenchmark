"""Built-in synthetic scenarios with analytically known equilibria.

These require no download and anchor the test suite: if the harness cannot
reproduce a hand-checkable equilibrium, nothing else matters.
"""

from __future__ import annotations

import numpy as np

from ..core.scenario import Demand, Network, ReferenceSolution, Scenario

__all__ = ["braess_scenario"]


def braess_scenario(demand: float = 6.0) -> Scenario:
    """The classic Braess (1968) paradox network **with** the bypass link.

    Nodes: 1 = origin zone, 2 = destination zone, 3 and 4 = intersections.
    Latency functions (linear, emulated exactly via BPR with ``power=1``):

    * 1->3 and 4->2: t(v) = 10 v          (steep, flow-dependent)
    * 1->4 and 3->2: t(v) = 50 + v        (flat)
    * 3->4:          t(v) = 10 + v        (the paradox-inducing bypass)

    With total demand 6 from zone 1 to zone 2 the unique user equilibrium
    splits 2 units on each of the three routes, giving link flows
    (4, 2, 2, 2, 4) in the link order below and a common route travel time
    of 92 (the paradox: removing the bypass would lower it to 83).

    The linear latencies are represented exactly: for ``power=1``,
    ``t(v) = fft + (fft*b/cap) v``, so slope and intercept are free choices.
    For the zero-intercept links a tiny ``fft = 1e-6`` is used; the induced
    error in the equilibrium is far below test tolerances.
    """
    eps = 1e-6

    def bpr_linear(intercept: float, slope: float) -> tuple[float, float, float]:
        """Return (fft, b, capacity) so that t(v) = intercept + slope*v (power=1)."""
        fft = intercept if intercept > 0 else eps
        cap = 1.0
        b = slope * cap / fft
        return fft, b, cap

    # Link order: 1->3, 1->4, 3->4, 3->2, 4->2
    init = np.array([1, 1, 3, 3, 4], dtype=np.int64)
    term = np.array([3, 4, 4, 2, 2], dtype=np.int64)
    params = [
        bpr_linear(0.0, 10.0),  # 1->3: 10v
        bpr_linear(50.0, 1.0),  # 1->4: 50 + v
        bpr_linear(10.0, 1.0),  # 3->4: 10 + v
        bpr_linear(50.0, 1.0),  # 3->2: 50 + v
        bpr_linear(0.0, 10.0),  # 4->2: 10v
    ]
    fft = np.array([p[0] for p in params])
    b = np.array([p[1] for p in params])
    cap = np.array([p[2] for p in params])

    network = Network(
        name="braess",
        n_nodes=4,
        n_zones=2,
        first_thru_node=1,
        init_node=init,
        term_node=term,
        capacity=cap,
        length=np.zeros(5),
        free_flow_time=fft,
        b=b,
        power=np.ones(5),
        toll=np.zeros(5),
        link_type=np.ones(5, dtype=np.int64),
        units=(("time", "abstract"), ("flow", "abstract")),
    )

    od = np.zeros((2, 2))
    od[0, 1] = demand
    reference = None
    if demand == 6.0:
        reference = ReferenceSolution(
            link_flows=np.array([4.0, 2.0, 2.0, 2.0, 4.0]),
            source="analytic",
            note="Unique UE: 2 units on each of the three routes; route time 92.",
        )

    return Scenario(
        name="braess",
        network=network,
        demand=Demand(matrix=od),
        reference=reference,
        family="builtin-braess",
    )
