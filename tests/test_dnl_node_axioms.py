"""DNL node component tests: NodeTopology validation, axiom checker N1-N5
branch coverage, shipped Series/Origin/Destination nodes over seeded random
(s, r, turns, caps) grids, N6 behavioral pattern, caps ignored (adr-010 layer 2)."""

import numpy as np
import pytest

from tabench.dnl.node import (
    DestinationNode,
    NodeModel,
    NodeTopology,
    OriginNode,
    SeriesNode,
    assert_node_axioms,
)

EPS = 1e-9


def rng() -> np.random.Generator:
    """Seeded generator from a fixed literal SeedSequence (repo rule: never a global seed)."""
    return np.random.default_rng(np.random.SeedSequence(20260707))


# ---------------------------------------------------------------- NodeTopology


def test_node_topology_coerces_and_counts():
    top = NodeTopology(node_id=np.int64(3), in_links=[0, 2], out_links=np.array([1, 4, 5]))
    assert top.node_id == 3 and isinstance(top.node_id, int)
    assert top.in_links.dtype == np.int64 and top.out_links.dtype == np.int64
    assert top.n_in == 2 and top.n_out == 3
    assert np.array_equal(top.in_links, [0, 2])


def test_node_topology_allows_boundary_empties():
    origin = NodeTopology(node_id=1, in_links=[], out_links=[0])
    dest = NodeTopology(node_id=2, in_links=[0], out_links=[])
    assert origin.n_in == 0 and origin.in_links.dtype == np.int64
    assert dest.n_out == 0


def test_node_topology_rejections():
    with pytest.raises(ValueError, match="ascending"):
        NodeTopology(node_id=1, in_links=[2, 0], out_links=[1])
    with pytest.raises(ValueError, match="ascending"):
        NodeTopology(node_id=1, in_links=[0, 0], out_links=[1])
    with pytest.raises(ValueError, match="nonnegative"):
        NodeTopology(node_id=1, in_links=[-1], out_links=[0])
    with pytest.raises(ValueError, match="integer"):
        NodeTopology(node_id=1, in_links=[0.5], out_links=[1])
    with pytest.raises(ValueError, match="1-D"):
        NodeTopology(node_id=1, in_links=[[0]], out_links=[1])
    with pytest.raises(ValueError, match="node_id"):
        NodeTopology(node_id="n1", in_links=[0], out_links=[1])
    with pytest.raises(ValueError, match="node_id"):
        NodeTopology(node_id=True, in_links=[0], out_links=[1])


# ---------------------------------------------------------------- assert_node_axioms


def test_axioms_pass_on_consistent_allocation():
    q = np.array([[1.0, 1.0], [2.0, 0.0]])
    s = np.array([2.0, 2.0])
    r = np.array([3.0, 1.0])
    turns = np.array([[0.5, 0.5], [1.0, 0.0]])
    assert_node_axioms(q, s, r, turns, eps=EPS)  # binding columns, no undersend


def test_axiom_n1_negativity_raises():
    with pytest.raises(ValueError, match="N1"):
        assert_node_axioms([[-1.0]], [1.0], [1.0], [[1.0]], eps=EPS)


def test_axiom_n2_demand_respect_raises():
    with pytest.raises(ValueError, match="N2"):
        assert_node_axioms([[3.0]], [2.0], [5.0], [[1.0]], eps=EPS)


def test_axiom_n3_supply_respect_raises():
    with pytest.raises(ValueError, match="N3"):
        assert_node_axioms([[3.0]], [3.0], [2.0], [[1.0]], eps=EPS)


def test_axiom_n4_turning_fraction_conservation_raises():
    # row sends 2 total but all through column 0 despite a 50/50 split
    with pytest.raises(ValueError, match="N4"):
        assert_node_axioms([[2.0, 0.0]], [2.0], [5.0, 5.0], [[0.5, 0.5]], eps=EPS)


def test_axiom_n5_holdback_without_saturation_raises():
    with pytest.raises(ValueError, match="N5"):
        assert_node_axioms([[0.5]], [2.0], [5.0], [[1.0]], eps=EPS)


def test_axiom_n5_infinite_supply_column_is_never_saturated():
    with pytest.raises(ValueError, match="N5"):
        assert_node_axioms([[0.5]], [2.0], [np.inf], [[1.0]], eps=EPS)


def test_axiom_n5_holdback_legal_when_eligible_column_saturated():
    assert_node_axioms([[1.0]], [2.0], [1.0], [[1.0]], eps=EPS)


def test_axioms_eps_boundary():
    # violations within eps pass; violations above eps raise
    assert_node_axioms([[-0.5 * EPS]], [0.0], [0.0], [[1.0]], eps=EPS)
    with pytest.raises(ValueError, match="N1"):
        assert_node_axioms([[-2.0 * EPS]], [0.0], [0.0], [[1.0]], eps=EPS)
    assert_node_axioms([[1.0 + 0.5 * EPS]], [1.0], [2.0], [[1.0]], eps=EPS)
    with pytest.raises(ValueError, match="N2"):
        assert_node_axioms([[1.0 + 2.0 * EPS]], [1.0], [2.0], [[1.0]], eps=EPS)


def test_axioms_shape_and_validity_rejections():
    with pytest.raises(ValueError, match="shape"):
        assert_node_axioms([[1.0]], [1.0, 1.0], [1.0], [[1.0]], eps=EPS)
    with pytest.raises(ValueError, match="2-D"):
        assert_node_axioms([1.0], [1.0], [1.0], [[1.0]], eps=EPS)
    with pytest.raises(ValueError, match="finite"):
        assert_node_axioms([[np.nan]], [1.0], [1.0], [[1.0]], eps=EPS)
    with pytest.raises(ValueError, match="eps"):
        assert_node_axioms([[1.0]], [1.0], [1.0], [[1.0]], eps=-1.0)


# ---------------------------------------------------------------- SeriesNode


def test_series_node_is_min_and_satisfies_axioms_on_seeded_grid():
    node = SeriesNode()
    turns = np.array([[1.0]])
    g = rng()
    for _ in range(200):
        s0, r0, cap = g.uniform(0.0, 10.0, size=3)
        q = node.transfer([s0], [r0], turns, [cap])
        assert q.shape == (1, 1) and q.dtype == np.float64
        assert q[0, 0] == min(s0, r0)
        assert_node_axioms(q, [s0], [r0], turns, eps=EPS)


def test_series_node_n6_invariance_behavioral_pattern():
    node = SeriesNode()
    turns, caps = [[1.0]], [1.0]
    # supply-constrained: inflating the constrained sending flow changes nothing
    q = node.transfer([5.0], [2.0], turns, caps)
    q_inflated = node.transfer([50.0], [2.0], turns, caps)
    assert np.array_equal(q, q_inflated) and q[0, 0] == 2.0
    # demand-constrained: inflating the constrained receiving flow changes nothing
    q = node.transfer([2.0], [5.0], turns, caps)
    q_inflated = node.transfer([2.0], [500.0], turns, caps)
    assert np.array_equal(q, q_inflated) and q[0, 0] == 2.0


def test_series_node_rejects_non_1x1():
    node = SeriesNode()
    with pytest.raises(ValueError, match="1-in/1-out"):
        node.transfer([1.0, 2.0], [1.0], [[1.0], [1.0]], [1.0, 1.0])
    with pytest.raises(ValueError, match="1-in/1-out"):
        node.transfer([1.0], [1.0, 2.0], [[0.5, 0.5]], [1.0])


# ---------------------------------------------------------------- OriginNode


def test_origin_node_single_out_seeded_grid():
    node = OriginNode()
    turns = np.array([[1.0]])
    g = rng()
    for _ in range(200):
        waiting, r0 = g.uniform(0.0, 10.0, size=2)
        q = node.transfer([waiting], [r0], turns, [np.inf])
        assert q.shape == (1, 1)
        assert q[0, 0] == min(waiting, r0)
        assert_node_axioms(q, [waiting], [r0], turns, eps=EPS)


def test_origin_node_multi_out_unblocked_satisfies_all_axioms():
    node = OriginNode()
    g = rng()
    for _ in range(100):
        n_out = int(g.integers(2, 5))
        split = g.dirichlet(np.ones(n_out))
        waiting = g.uniform(0.0, 10.0)
        r = waiting * split + g.uniform(0.1, 1.0, size=n_out)  # no split is blocked
        q = node.transfer([waiting], r, split[None, :], [np.inf])
        assert q.shape == (1, n_out)
        assert np.allclose(q[0], waiting * split, atol=EPS)
        assert_node_axioms(q, [waiting], r, split[None, :], eps=EPS)


def test_origin_node_multi_out_blocked_split_is_documented_placeholder():
    # waiting = 10, 50/50 split, column 0 blocked at 2: q = [[2, 5]].
    node = OriginNode()
    waiting, r = 10.0, np.array([2.0, 10.0])
    turns = np.array([[0.5, 0.5]])
    q = node.transfer([waiting], r, turns, [np.inf])
    assert np.array_equal(q, [[2.0, 5.0]])
    # per-column interpretation (the documented placeholder policy): each
    # column takes min(waiting * split_j, r_j) — N1-N3 hold ...
    assert np.all(q >= 0.0)
    assert q.sum() <= waiting + EPS
    assert np.all(q.sum(axis=0) <= r + EPS)
    for j in range(2):
        assert q[0, j] == min(waiting * turns[0, j], r[j])
    # ... but the JOINT N4 (CTF) does not: no re-normalization across blocked
    # splits. Pinned so nobody silently "fixes" the placeholder.
    with pytest.raises(ValueError, match="N4"):
        assert_node_axioms(q, [waiting], r, turns, eps=EPS)


def test_origin_node_rejects_multiple_in_entries():
    with pytest.raises(ValueError, match="one synthetic waiting"):
        OriginNode().transfer([1.0, 2.0], [1.0], [[1.0], [1.0]], [np.inf, np.inf])


# ---------------------------------------------------------------- DestinationNode


def test_destination_node_absorbs_everything_seeded_grid():
    node = DestinationNode()
    g = rng()
    for _ in range(100):
        n_in = int(g.integers(1, 5))
        n_out = int(g.integers(1, 4))
        turns = g.uniform(0.1, 1.0, size=(n_in, n_out))
        turns /= turns.sum(axis=1, keepdims=True)
        s = g.uniform(0.0, 10.0, size=n_in)
        r = np.full(n_out, np.inf)
        q = node.transfer(s, r, turns, np.full(n_in, np.inf))
        assert q.shape == (n_in, n_out)
        assert np.allclose(q.sum(axis=1), s, atol=EPS)  # q[i, :].sum() == s_i
        assert_node_axioms(q, s, r, turns, eps=EPS)


def test_destination_node_ignores_r_by_convention():
    # the loader's convention is r = +inf at destinations; the class pins that
    # r's VALUES are never consulted (finite r does not throttle absorption)
    q = DestinationNode().transfer([3.0], [0.0], [[1.0]], [np.inf])
    assert np.array_equal(q, [[3.0]])


# ---------------------------------------------------------------- shared contract


def test_node_model_is_abstract():
    with pytest.raises(TypeError):
        NodeModel()


@pytest.mark.parametrize("node_cls", [SeriesNode, OriginNode, DestinationNode])
def test_caps_ignored_by_all_shipped_nodes(node_cls):
    node = node_cls()
    for caps in ([0.001], [7.5], [np.inf]):
        q = node.transfer([4.0], [3.0], [[1.0]], caps)
        assert np.array_equal(q, node.transfer([4.0], [3.0], [[1.0]], [np.inf]))
    assert q.shape == (1, 1)


@pytest.mark.parametrize("node_cls", [SeriesNode, OriginNode, DestinationNode])
def test_transfer_returns_fresh_array_and_never_mutates_inputs(node_cls):
    node = node_cls()
    s = np.array([4.0])
    r = np.array([3.0])
    turns = np.array([[1.0]])
    caps = np.array([2.0])
    q1 = node.transfer(s, r, turns, caps)
    q1[0, 0] = -99.0  # clobber the returned array
    q2 = node.transfer(s, r, turns, caps)
    assert q2[0, 0] != -99.0
    assert np.array_equal(s, [4.0]) and np.array_equal(r, [3.0])
    assert np.array_equal(turns, [[1.0]]) and np.array_equal(caps, [2.0])


@pytest.mark.parametrize("node_cls", [SeriesNode, OriginNode, DestinationNode])
def test_transfer_input_validation_shared(node_cls):
    node = node_cls()
    with pytest.raises(ValueError, match="turns"):
        node.transfer([1.0], [1.0], [[0.5, 0.5]], [1.0])
    with pytest.raises(ValueError, match="caps"):
        node.transfer([1.0], [1.0], [[1.0]], [1.0, 1.0])
    with pytest.raises(ValueError, match="NaN"):
        node.transfer([np.nan], [1.0], [[1.0]], [1.0])
    with pytest.raises(ValueError, match="nonnegative"):
        node.transfer([-1.0], [1.0], [[1.0]], [1.0])
    with pytest.raises(ValueError, match="nonempty"):
        node.transfer(np.empty(0), [1.0], np.empty((0, 1)), np.empty(0))
