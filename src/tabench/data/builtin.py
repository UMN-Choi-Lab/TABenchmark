"""Built-in synthetic scenarios with analytically known equilibria.

These require no download and anchor the test suite: if the harness cannot
reproduce a hand-checkable equilibrium, nothing else matters.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from ..core.scenario import (
    CombinedDemand,
    Demand,
    ElasticDemand,
    Network,
    ReferenceSolution,
    Scenario,
)

__all__ = [
    "braess_scenario",
    "two_route_scenario",
    "elastic_two_route_scenario",
    "evans_symmetric_scenario",
    "br_two_route_scenario",
    "sc_two_route_scenario",
]

_EPS = 1e-6


def _bpr_linear(intercept: float, slope: float) -> tuple[float, float, float]:
    """Return (fft, b, capacity) so that t(v) = intercept + slope*v (power=1).

    Zero intercepts use a tiny fft = 1e-6 (Network requires fft > 0); the
    induced equilibrium error is far below test tolerances.
    """
    fft = intercept if intercept > 0 else _EPS
    cap = 1.0
    b = slope * cap / fft
    return fft, b, cap


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
    # Link order: 1->3, 1->4, 3->4, 3->2, 4->2
    init = np.array([1, 1, 3, 3, 4], dtype=np.int64)
    term = np.array([3, 4, 4, 2, 2], dtype=np.int64)
    params = [
        _bpr_linear(0.0, 10.0),  # 1->3: 10v
        _bpr_linear(50.0, 1.0),  # 1->4: 50 + v
        _bpr_linear(10.0, 1.0),  # 3->4: 10 + v
        _bpr_linear(50.0, 1.0),  # 3->2: 50 + v
        _bpr_linear(0.0, 10.0),  # 4->2: 10v
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


def two_route_scenario(
    demand: float = 4.0,
    sue_theta: float | None = 0.5,
    sue_family: str = "logit",
) -> Scenario:
    """Two disjoint 2-link routes: the analytic anchor for the SUE tasks.

    Nodes: 1 = origin zone, 2 = destination zone, 3 and 4 = intersections.
    Route A = 1->3->2 with cost ``c_A = 2 + f_A``; route B = 1->4->2 with
    cost ``c_B = 1.5 + 2 f_B`` (linear latencies via BPR power=1; parallel
    links are forbidden by Network validation, hence the 2-link routes).

    The first legs cost a constant 1, so ``r(3) = r(4) = 1`` is below every
    route cost and BOTH routes are Dial-efficient at all nonnegative flows:
    Dial's loading reduces exactly to a binary logit, and the SUE fixed point
    to the scalar equation ``f_A = D / (1 + exp(theta (c_A(f_A) -
    c_B(D - f_A))))`` — solvable by brentq in tests, no trusted digits.
    The deterministic UE (theta -> infinity limit) puts f_A = 2.5 at D = 4.

    ``sue_family="probit"`` selects the probit task (adr-003) on the same
    network: disjoint routes make the perceived route costs independent
    normals, so ``P(A) = Phi((c_B - c_A)/sqrt(3.5 beta))`` — again a scalar
    fixed point tests recompute via brentq. The default ``"logit"`` leaves
    every existing scenario (and its content hash) untouched.
    """
    # Link order: 1->3, 3->2, 1->4, 4->2
    init = np.array([1, 3, 1, 4], dtype=np.int64)
    term = np.array([3, 2, 4, 2], dtype=np.int64)
    params = [
        _bpr_linear(1.0, 0.0),  # 1->3: constant 1
        _bpr_linear(1.0, 1.0),  # 3->2: 1 + f
        _bpr_linear(1.0, 0.0),  # 1->4: constant 1
        _bpr_linear(0.5, 2.0),  # 4->2: 0.5 + 2f
    ]
    fft = np.array([p[0] for p in params])
    b = np.array([p[1] for p in params])
    cap = np.array([p[2] for p in params])

    network = Network(
        name="two-route",
        n_nodes=4,
        n_zones=2,
        first_thru_node=1,
        init_node=init,
        term_node=term,
        capacity=cap,
        length=np.zeros(4),
        free_flow_time=fft,
        b=b,
        power=np.ones(4),
        toll=np.zeros(4),
        link_type=np.ones(4, dtype=np.int64),
        units=(("time", "abstract"), ("flow", "abstract")),
    )

    od = np.zeros((2, 2))
    od[0, 1] = demand
    reference = None
    if demand == 4.0 and sue_theta == 0.5 and sue_family == "logit":
        # The analytic oracle is the binary-LOGIT fixed point; it must not be
        # attached to a probit instance (whose fixed point differs: f_A=2.444
        # at beta=0.1), or flow_rmse_vs_reference would score against the wrong
        # equilibrium. Probit tasks certify through the ADR-003 MC residual.
        f_a = 2.2990959494  # scalar logit fixed point; tests recompute via brentq
        reference = ReferenceSolution(
            link_flows=np.array([f_a, f_a, 4.0 - f_a, 4.0 - f_a]),
            source="analytic",
            note="Binary-logit SUE fixed point at theta=0.5, demand=4.",
        )

    return Scenario(
        name="tworoute",
        network=network,
        demand=Demand(matrix=od),
        reference=reference,
        family="builtin-tworoute",
        sue_theta=sue_theta,
        sue_family=sue_family,
    )


def elastic_two_route_scenario(d0: float = 10.0, u0: float = 10.0) -> Scenario:
    """Two disjoint 2-link routes with **linear elastic demand**: the analytic
    anchor for the variable-demand UE task (Florian & Nguyen 1974; adr-005).

    Nodes: 1 = origin zone, 2 = destination zone, 3 and 4 = intersections.
    Route A = 1->3->2 with cost ``c_A = 2 + f_A``; route B = 1->4->2 with cost
    ``c_B = 3 + f_B`` (linear latencies via BPR ``power=1``; the first legs are
    constant 1, the second legs ``1 + f_A`` and ``2 + f_B``). The reference
    demand ``d0`` is the demand at zero cost; the realized demand follows the
    linear law ``D(u) = d0 * max(0, 1 - u/u0)``.

    At ``d0 = 10, u0 = 10`` both routes are used and the elastic UE is exact
    and rational: ``c_A = c_B = u`` gives ``f_A = 1 + f_B``; demand consistency
    ``f_A + f_B = 10 - u`` with ``u = 2 + f_A`` yields **``u = 5``,
    ``f_A = 3``, ``f_B = 2``**, realized demand ``5``, and link flows
    ``(3, 3, 2, 2)`` in the order below. The excess-demand arc carries the
    unmet ``10 - 5 = 5`` at cost ``u0 * e/d0 = 5 = u``. These integers make the
    scenario a hand-checkable oracle for both the solver and the certificate.
    """
    # Link order: 1->3, 3->2, 1->4, 4->2
    init = np.array([1, 3, 1, 4], dtype=np.int64)
    term = np.array([3, 2, 4, 2], dtype=np.int64)
    params = [
        _bpr_linear(1.0, 0.0),  # 1->3: constant 1
        _bpr_linear(1.0, 1.0),  # 3->2: 1 + f
        _bpr_linear(1.0, 0.0),  # 1->4: constant 1
        _bpr_linear(2.0, 1.0),  # 4->2: 2 + f
    ]
    fft = np.array([p[0] for p in params])
    b = np.array([p[1] for p in params])
    cap = np.array([p[2] for p in params])

    network = Network(
        name="elastic-two-route",
        n_nodes=4,
        n_zones=2,
        first_thru_node=1,
        init_node=init,
        term_node=term,
        capacity=cap,
        length=np.zeros(4),
        free_flow_time=fft,
        b=b,
        power=np.ones(4),
        toll=np.zeros(4),
        link_type=np.ones(4, dtype=np.int64),
        units=(("time", "abstract"), ("flow", "abstract")),
    )

    od = np.zeros((2, 2))
    od[0, 1] = d0  # reference demand d0 = D(0)
    reference = None
    if d0 == 10.0 and u0 == 10.0:
        reference = ReferenceSolution(
            link_flows=np.array([3.0, 3.0, 2.0, 2.0]),
            source="analytic",
            note="Linear elastic UE at d0=10, u0=10: f_A=3, f_B=2, realized demand 5.",
        )

    return Scenario(
        name="elastic-tworoute",
        network=network,
        demand=Demand(matrix=od),
        reference=reference,
        family="builtin-elastic",
        elastic_demand=ElasticDemand(form="linear", param=u0),
    )


def evans_symmetric_scenario(trips: float = 10.0, beta: float = 0.5) -> Scenario:
    """Symmetric bipartite **combined distribution + assignment** anchor for the
    Evans (1976) task (adr-007).

    Two origin zones (1, 2) and two destination zones (3, 4); each origin
    produces ``trips`` and each destination attracts ``trips`` (total ``2 trips``
    both ways, so the doubly-constrained gravity is feasible). One congestible
    link per OD pair, symmetric under swapping ``1<->2`` and ``3<->4``:

    * "near" links ``1->3`` and ``2->4``: ``t(f) = 1 + 0.1 f``
    * "far"  links ``1->4`` and ``2->3``: ``t(f) = 3 + 0.1 f``

    The demand is endogenous: ``d_ij`` is the doubly-constrained gravity at the
    equilibrium costs. By symmetry the equilibrium sets ``d_13 = d_24 = p`` and
    ``d_14 = d_23 = q = trips - p``, and the balancing factors cancel, so the
    gravity collapses to a binary **logit split**

        p = trips / (1 + exp(beta (c_near(p) - c_far(q)))),
        c_near(p) = 1 + 0.1 p,   c_far(q) = 3 + 0.1 q,

    a single scalar fixed point (recomputed with brentq in tests, no trusted
    digits). The equilibrium link flows are ``(p, q, q, p)`` in the link order
    below (``p ~ 6.92`` at ``beta = 0.5``); ``beta`` genuinely bites
    (``p != trips/2``). The near/far intercepts are spaced (1 vs 3) so
    ``c_near(s) - c_far(trips - s) = 0.2 s - 3 < 0`` for every feasible split
    ``s in [0, trips]`` — the costs never equalize, so the ONLY margin-feasible
    flow with a zero certified gap is the true equilibrium (any other split is
    censored by the negative-excess guard or carries a strictly positive gap).
    That makes this anchor degeneracy-free: it does not itself exhibit the
    aggregate-multicommodity certificate limitation (adr-007), which is pinned
    separately on a deliberately cost-degenerate instance in the tests.
    """
    # Link order: 1->3 (near), 1->4 (far), 2->3 (far), 2->4 (near)
    init = np.array([1, 1, 2, 2], dtype=np.int64)
    term = np.array([3, 4, 3, 4], dtype=np.int64)
    a_near, s_near = 1.0, 0.1
    a_far, s_far = 3.0, 0.1
    params = [
        _bpr_linear(a_near, s_near),  # 1->3 near
        _bpr_linear(a_far, s_far),  # 1->4 far
        _bpr_linear(a_far, s_far),  # 2->3 far
        _bpr_linear(a_near, s_near),  # 2->4 near
    ]
    fft = np.array([p[0] for p in params])
    b = np.array([p[1] for p in params])
    cap = np.array([p[2] for p in params])

    network = Network(
        name="evans-symmetric",
        n_nodes=4,
        n_zones=4,
        first_thru_node=1,
        init_node=init,
        term_node=term,
        capacity=cap,
        length=np.zeros(4),
        free_flow_time=fft,
        b=b,
        power=np.ones(4),
        toll=np.zeros(4),
        link_type=np.ones(4, dtype=np.int64),
        units=(("time", "abstract"), ("flow", "abstract")),
    )

    productions = np.array([trips, trips, 0.0, 0.0])  # zones 1, 2 originate
    attractions = np.array([0.0, 0.0, trips, trips])  # zones 3, 4 attract
    combined = CombinedDemand(productions=productions, attractions=attractions, beta=beta)

    # Reference demand = free-flow gravity (the uncongested-equilibrium OD
    # matrix): a deterministic, meaningful reference with the right margins and
    # full support. Costs at zero flow are the link free-flow times; each OD is
    # a single link, so the skim is a direct lookup.
    free_costs = network.link_cost(np.zeros(network.n_links))
    link_of = {
        (int(a), int(t)): k
        for k, (a, t) in enumerate(zip(init.tolist(), term.tolist(), strict=True))
    }
    u0 = np.zeros((4, 4))
    for i, j in ((0, 2), (0, 3), (1, 2), (1, 3)):
        u0[i, j] = free_costs[link_of[(i + 1, j + 1)]]
    d_ref = combined.gravity(u0)

    # Analytic equilibrium split (the symmetric logit fixed point).
    def _split(p: float) -> float:
        c_near = a_near + s_near * p
        c_far = a_far + s_far * (trips - p)
        return p - trips / (1.0 + np.exp(beta * (c_near - c_far)))

    p_star = float(brentq(_split, 0.0, trips))
    q_star = trips - p_star
    reference = ReferenceSolution(
        link_flows=np.array([p_star, q_star, q_star, p_star]),
        source="analytic",
        note=(
            "Symmetric doubly-constrained Evans equilibrium: gravity collapses to "
            "a binary logit split, a scalar fixed point recomputed via brentq."
        ),
    )

    return Scenario(
        name="evans",
        network=network,
        demand=Demand(matrix=d_ref),
        reference=reference,
        family="builtin-evans",
        combined_demand=combined,
    )


def br_two_route_scenario(demand: float = 10.0, epsilon: float = 1.0) -> Scenario:
    """Two disjoint 2-link routes: the analytic anchor for boundedly-rational UE
    (Mahmassani & Chang 1987; adr-008).

    Route A = 1->3->2 with cost ``c_A = 2 + f_A``; route B = 1->4->2 with cost
    ``c_B = 3 + f_B`` (linear via BPR ``power=1``; first legs constant 1, second
    legs ``1 + f_A`` and ``2 + f_B``). Both slopes are 1, so the Wardrop split is
    ``f_A* = (a_B + b_B D - a_A)/(b_A + b_B) = (D + 1)/2`` and the ``epsilon``-BRUE
    acceptable set is the exact interval ``f_A in [f_A* - epsilon/2, f_A* +
    epsilon/2]`` clamped to ``[0, D]`` (band half-width ``epsilon/(b_A+b_B) =
    epsilon/2``).

    Route A is cheaper at free flow (``c_A(0)=2 < c_B(0)=3``), so the free-flow
    all-or-nothing start loads everyone on A; the band-relaxed swap then bleeds
    flow to B and stops at the **band edge** ``f_A = f_A* + epsilon/2`` (used-route
    excess exactly ``epsilon``) -- NOT the Wardrop point (excess 0). At ``D=10,
    epsilon=1`` that edge is ``f_A = 6``, link flows ``(6, 6, 4, 4)``; these
    hand-checkable numbers make it an oracle for both the model (band edge) and the
    certificate (``AEC = (6/10)*1 = 0.6 <= 1``). ``epsilon -> 0`` recovers the
    Wardrop split ``f_A = 5.5``; a huge ``epsilon`` leaves the AON start unchanged.
    """
    # Link order: 1->3, 3->2, 1->4, 4->2
    init = np.array([1, 3, 1, 4], dtype=np.int64)
    term = np.array([3, 2, 4, 2], dtype=np.int64)
    params = [
        _bpr_linear(1.0, 0.0),  # 1->3: constant 1
        _bpr_linear(1.0, 1.0),  # 3->2: 1 + f
        _bpr_linear(1.0, 0.0),  # 1->4: constant 1
        _bpr_linear(2.0, 1.0),  # 4->2: 2 + f
    ]
    fft = np.array([p[0] for p in params])
    b = np.array([p[1] for p in params])
    cap = np.array([p[2] for p in params])

    network = Network(
        name="br-two-route",
        n_nodes=4,
        n_zones=2,
        first_thru_node=1,
        init_node=init,
        term_node=term,
        capacity=cap,
        length=np.zeros(4),
        free_flow_time=fft,
        b=b,
        power=np.ones(4),
        toll=np.zeros(4),
        link_type=np.ones(4, dtype=np.int64),
        units=(("time", "abstract"), ("flow", "abstract")),
    )

    od = np.zeros((2, 2))
    od[0, 1] = demand
    reference = None
    f_edge = 0.5 * (demand + 1.0) + 0.5 * epsilon  # band edge f_A* + epsilon/2
    if 0.0 <= f_edge <= demand:
        reference = ReferenceSolution(
            link_flows=np.array([f_edge, f_edge, demand - f_edge, demand - f_edge]),
            source="analytic",
            note=(
                "Boundedly-rational UE band edge from the free-flow-AON start: "
                f"f_A = (D+1)/2 + epsilon/2 (used-route excess = epsilon = {epsilon})."
            ),
        )

    return Scenario(
        name="br-tworoute",
        network=network,
        demand=Demand(matrix=od),
        reference=reference,
        family="builtin-br",
        br_epsilon=epsilon,
    )


def sc_two_route_scenario(demand: float = 10.0, cap: float = 4.0) -> Scenario:
    """Two disjoint 2-link routes with a **link capacity**: the analytic anchor for
    side-constrained UE (Larsson & Patriksson 1995; adr-009).

    Route A = 1->3->2 with cost ``c_A = 1 + f_A`` (legs: 1->3 constant 1, 3->2 =
    ``f_A``); route B = 1->4->2 with ``c_B = 2 + f_B`` (1->4 constant 1, 4->2 =
    ``1 + f_B``). A hard capacity ``cap`` is placed on link 3->2 (which carries the
    route-A flow); the other links are effectively uncapacitated.

    Plain UE (no capacity) equalizes ``1 + f_A = 2 + f_B`` with ``f_A + f_B = D``,
    giving ``f_A* = (D + 1)/2 = 5.5`` at ``D = 10``. When ``cap >= f_A*`` the
    constraint is slack and SC-TAP reduces EXACTLY to plain UE (link flows
    ``(5.5, 5.5, 4.5, 4.5)``). When ``cap < f_A*`` link 3->2 saturates: ``f_A =
    cap``, ``f_B = D - cap``, and the queueing multiplier is ``beta = c_B(D-cap) -
    c_A(cap) = (2 + D - cap) - (1 + cap) = 1 + D - 2 cap``. At ``D = 10, cap = 4``:
    ``f_A = 4``, ``f_B = 6``, ``beta = 3``, augmented costs equalized at 8, link
    flows ``(4, 4, 6, 6)`` -- a hand-checkable oracle for the flows and the
    multiplier. Tightening ``cap`` pushes ``f_A`` down and ``beta`` up (monotone).
    """
    # Link order: 1->3, 3->2, 1->4, 4->2
    init = np.array([1, 3, 1, 4], dtype=np.int64)
    term = np.array([3, 2, 4, 2], dtype=np.int64)
    params = [
        _bpr_linear(1.0, 0.0),  # 1->3: constant 1
        _bpr_linear(0.0, 1.0),  # 3->2: f_A            (capacitated link)
        _bpr_linear(1.0, 0.0),  # 1->4: constant 1
        _bpr_linear(1.0, 1.0),  # 4->2: 1 + f_B
    ]
    fft = np.array([p[0] for p in params])
    b = np.array([p[1] for p in params])
    capacity = np.array([p[2] for p in params])

    network = Network(
        name="sc-two-route",
        n_nodes=4,
        n_zones=2,
        first_thru_node=1,
        init_node=init,
        term_node=term,
        capacity=capacity,
        length=np.zeros(4),
        free_flow_time=fft,
        b=b,
        power=np.ones(4),
        toll=np.zeros(4),
        link_type=np.ones(4, dtype=np.int64),
        units=(("time", "abstract"), ("flow", "abstract")),
    )

    od = np.zeros((2, 2))
    od[0, 1] = demand
    big = 1e6  # effectively uncapacitated on the other links
    side = np.array([big, cap, big, big])

    reference = None
    f_a_ue = 0.5 * (demand + 1.0)
    if cap < f_a_ue:  # binding: hand-checked oracle
        reference = ReferenceSolution(
            link_flows=np.array([cap, cap, demand - cap, demand - cap]),
            source="analytic",
            note=(
                f"Side-constrained UE, link 3->2 capacity {cap} binds: f_A=cap, "
                f"queue multiplier beta = 1 + D - 2*cap = {1.0 + demand - 2.0 * cap}."
            ),
        )

    return Scenario(
        name="sc-tworoute",
        network=network,
        demand=Demand(matrix=od),
        reference=reference,
        family="builtin-sc",
        side_capacities=side,
    )
