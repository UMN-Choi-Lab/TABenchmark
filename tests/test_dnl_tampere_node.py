"""TampereNode (2011) generic first-order merge/diverge node model — anchors,
axioms, N6 invariance, and end-to-end network loading through the loader.

Algorithm restated from Boyles TNA §9.6.2 + Yperman 2007 thesis Ch. 5 (both
open); anchors hand-computed as exact fractions (adr-017).
"""

import numpy as np
import pytest

from tabench.core.scenario import Network
from tabench.dnl import (
    CTMLink,
    DynamicDemand,
    DynamicScenario,
    LinkDynamics,
    NetworkLoader,
    TampereNode,
    TimeGrid,
    TurningFractions,
)
from tabench.dnl.node import assert_node_axioms
from tabench.metrics import DNLEvaluator


def _q(s, r, turns, caps):
    s, r = np.array(s, float), np.array(r, float)
    turns, caps = np.array(turns, float), np.array(caps, float)
    out = TampereNode().transfer(s, r, turns, caps)
    assert_node_axioms(out, s, r, turns, eps=1e-9)  # N1-N5 on every anchor
    return out


# ---------------------------------------------------------------------------
# Anchors (exact fractions).
# ---------------------------------------------------------------------------


def test_merge_capacity_proportional() -> None:
    # 2->1, both want 1, out-link supplies 1: capacity-proportional share.
    np.testing.assert_allclose(_q([1, 1], [1], [[1], [1]], [1, 1]), [[0.5], [0.5]])
    np.testing.assert_allclose(_q([1, 1], [1], [[1], [1]], [2, 1]), [[2 / 3], [1 / 3]])


def test_diverge_fifo_holdback() -> None:
    # 1->2, split 0.6/0.4, out-link 2 supplies only 0.4: the WHOLE approach is
    # throttled (phi=min(1e6/1.2, 0.4/0.8, 1)=0.5) so only 1 of 2 veh transfer.
    np.testing.assert_allclose(_q([2], [1e6, 0.4], [[0.6, 0.4]], [1]), [[0.6, 0.4]])


def test_two_by_two_binding_out_link() -> None:
    q = _q([10, 10], [5, 100], [[0.7, 0.3], [0.4, 0.6]], [6, 8])
    np.testing.assert_allclose(q, [[105 / 37, 45 / 37], [80 / 37, 120 / 37]], atol=1e-12)
    np.testing.assert_allclose(q.sum(axis=0), [5.0, 165 / 37], atol=1e-12)  # out-link A saturated


def test_reduces_to_min_at_series_node() -> None:
    np.testing.assert_allclose(_q([3], [2], [[1]], [5]), [[2.0]])  # min(s, r)
    np.testing.assert_allclose(_q([1], [4], [[1]], [5]), [[1.0]])


def test_infinite_receiving_absorbs_everything() -> None:
    # a destination-style out-link (r=inf) never binds: full sending transfers.
    np.testing.assert_allclose(_q([2, 3], [np.inf], [[1], [1]], [5, 5]), [[2.0], [3.0]])


# ---------------------------------------------------------------------------
# N6 invariance (inflating a non-binding sending flow leaves q unchanged).
# ---------------------------------------------------------------------------


def test_n6_invariance_diverge() -> None:
    args = ([1e6, 0.4], [[0.6, 0.4]], [1])
    q_small = TampereNode().transfer(np.array([2.0]), *[np.array(a, float) for a in args])
    q_big = TampereNode().transfer(np.array([200.0]), *[np.array(a, float) for a in args])
    np.testing.assert_allclose(q_small, q_big)  # s was FIFO-blocked, never binding


def test_n6_invariance_merge() -> None:
    args = ([1.0], [[1.0], [1.0]], [1.0, 1.0])
    q_small = TampereNode().transfer(np.array([0.3, 5.0]), *[np.array(a, float) for a in args])
    q_big = TampereNode().transfer(np.array([0.3, 500.0]), *[np.array(a, float) for a in args])
    np.testing.assert_allclose(q_small, q_big)
    np.testing.assert_allclose(q_small, [[0.3], [0.7]])  # link 1 receiving-limited


def test_axioms_hold_on_random_cases() -> None:
    rng = np.random.default_rng(20260709)
    for _ in range(300):
        n_in = int(rng.integers(1, 5))
        n_out = int(rng.integers(1, 5))
        caps = rng.uniform(0.1, 5.0, n_in)
        s = rng.uniform(0.0, 1.0, n_in) * caps  # s <= caps (link contract)
        turns = rng.uniform(0.0, 1.0, (n_in, n_out))
        turns /= turns.sum(axis=1, keepdims=True)
        r = rng.uniform(0.0, 3.0, n_out)
        if rng.random() < 0.3:
            r[int(rng.integers(n_out))] = np.inf
        q = TampereNode().transfer(s, r, turns, caps)
        assert_node_axioms(q, s, r, turns, eps=1e-9)


def test_tiny_approach_not_dropped_at_extreme_capacity_ratio() -> None:
    """A tiny sending flow co-incident with a huge one at a merge must still
    transfer against an unconstrained supply — a global s.sum()-scaled tolerance
    dropped it to zero (N5 violation, adversarial-review regression). Per-element
    tolerances keep it. Also fuzzes extreme (1e-6..1e6) capacity ratios."""
    s = np.array([1.23186443e-06, 1.49682256e06])
    r, turns, caps = np.array([np.inf]), np.array([[1.0], [1.0]]), np.array([3.045e-06, 2.787e06])
    q = TampereNode().transfer(s, r, turns, caps)
    assert_node_axioms(q, s, r, turns, eps=1e-9)
    assert q[0, 0] == pytest.approx(1.23186443e-06)  # not swallowed
    rng = np.random.default_rng(7)
    for _ in range(2000):
        n_in, n_out = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        caps = 10.0 ** rng.uniform(-6, 6, n_in)
        s = rng.uniform(0.0, 1.0, n_in) * caps
        turns = rng.uniform(0.0, 1.0, (n_in, n_out))
        turns /= turns.sum(axis=1, keepdims=True)
        r = 10.0 ** rng.uniform(-3, 6, n_out)
        if rng.random() < 0.3:
            r[int(rng.integers(n_out))] = np.inf
        assert_node_axioms(TampereNode().transfer(s, r, turns, caps), s, r, turns, eps=1e-9)


# ---------------------------------------------------------------------------
# End-to-end: the loader now handles merges/diverges (previously raised).
# ---------------------------------------------------------------------------


def _dyn(n: int, cap: np.ndarray | None = None) -> LinkDynamics:
    return LinkDynamics(
        length=np.full(n, 4.0), free_speed=np.full(n, 1.0), wave_speed=np.full(n, 1.0),
        jam_density=np.full(n, 4.0), capacity=np.full(n, 2.0) if cap is None else cap,
    )


def _diverge_scenario() -> DynamicScenario:
    net = Network(
        name="dv", n_nodes=4, n_zones=3, first_thru_node=4,
        init_node=np.array([1, 4, 4]), term_node=np.array([4, 2, 3]),
        capacity=np.ones(3), length=np.zeros(3), free_flow_time=np.ones(3),
        b=np.zeros(3), power=np.ones(3), toll=np.zeros(3), link_type=np.ones(3, dtype=np.int64),
    )
    rates = np.zeros((1, 3, 3))
    rates[0, 0, 1] = 0.9
    rates[0, 0, 2] = 0.6
    return DynamicScenario(
        name="dv", network=net, dynamics=_dyn(3),
        demand=DynamicDemand(breakpoints=np.array([0.0, 10.0]), rates=rates),
        grid=TimeGrid(dt=1.0, n_steps=16),
        turns=TurningFractions(frac=((4, np.array([[0.6, 0.4]])),)),
    )


def test_loader_diverge_certifies_with_turn_fidelity() -> None:
    scenario = _diverge_scenario()
    metrics = DNLEvaluator(scenario).evaluate(NetworkLoader(scenario, CTMLink).run())
    assert metrics["dnl_feasible"] == 1.0
    assert metrics["turn_residual"] <= 1e-9  # C8: realized split obeys the 0.6/0.4 mandate
    assert metrics["conservation_residual"] <= 1e-9


def test_loader_merge_shares_bottleneck_equally() -> None:
    net = Network(
        name="mg", n_nodes=4, n_zones=3, first_thru_node=4,
        init_node=np.array([1, 2, 4]), term_node=np.array([4, 4, 3]),
        capacity=np.ones(3), length=np.zeros(3), free_flow_time=np.ones(3),
        b=np.zeros(3), power=np.ones(3), toll=np.zeros(3), link_type=np.ones(3, dtype=np.int64),
    )
    rates = np.zeros((1, 3, 3))
    rates[0, 0, 2] = 1.0  # zone 1 -> zone 3
    rates[0, 1, 2] = 1.0  # zone 2 -> zone 3, total 2.0 > the out-link cap 1.0
    scenario = DynamicScenario(
        name="mg", network=net,
        dynamics=_dyn(3, cap=np.array([2.0, 2.0, 1.0])),  # out-link (idx 2) is the merge bottleneck
        demand=DynamicDemand(breakpoints=np.array([0.0, 10.0]), rates=rates),
        grid=TimeGrid(dt=1.0, n_steps=20),
    )
    out = NetworkLoader(scenario, CTMLink).run()
    metrics = DNLEvaluator(scenario).evaluate(out)
    assert metrics["dnl_feasible"] == 1.0
    assert metrics["conservation_residual"] <= 1e-9
    # equal caps -> the two approaches share the saturated out-link equally.
    assert out.n_out[0, -1] == pytest.approx(out.n_out[1, -1], abs=1e-9)


def test_loader_no_longer_raises_on_junctions() -> None:
    """dnl-core used to raise for any non-1x1 interior node; the node-model
    sprint makes the loader default to TampereNode."""
    scenario = _diverge_scenario()
    loader = NetworkLoader(scenario, CTMLink)
    assert isinstance(loader._interior_models[4], TampereNode)
