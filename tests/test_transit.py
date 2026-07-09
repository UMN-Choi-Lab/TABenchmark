"""Transit assignment — Spiess & Florian (1989) optimal strategies (adr-014).

The optimal-strategy expected cost and the frequency-share split are closed
forms; every scored quantity is recomputed by the harness (P1). Anchors are
hand-derived (Instance 1 both lines attractive → 24 min; Instance 2 one line
excluded → 21 min), recomputed here — no trusted digits.
"""

from __future__ import annotations

import numpy as np
import pytest

import tabench as tb
from tabench.metrics.transit_gaps import TransitEvaluator
from tabench.transit import (
    TransitDemand,
    TransitNetwork,
    TransitReference,
    TransitScenario,
    TransitStrategy,
    common_lines_expected_cost,
    common_lines_scenario,
    common_lines_unattractive_scenario,
    optimal_strategy,
)

# ------------------------------------------------------------- closed form

def test_common_lines_formula_both_attractive() -> None:
    cost, attractive = common_lines_expected_cost([(1 / 6, 21.0), (1 / 12, 18.0)])
    assert cost == pytest.approx(24.0)
    assert attractive == [0, 1]


def test_common_lines_formula_threshold_excludes() -> None:
    cost, attractive = common_lines_expected_cost([(1 / 6, 15.0), (1 / 12, 40.0)])
    assert cost == pytest.approx(21.0)
    assert attractive == [0]  # line 1 (40 min) not below 21


# ------------------------------------------------------------- anchor recovery

def test_instance1_recovery() -> None:
    sc = common_lines_scenario(1000.0)
    strat = optimal_strategy(sc)
    # 2:1 frequency split (f0 = 2 f1) over both attractive lines.
    np.testing.assert_allclose(strat.arc_volumes, [2000 / 3, 1000 / 3], atol=1e-9)
    assert strat.pair_costs[0] == pytest.approx(24.0)


def test_instance2_recovery_unattractive_line() -> None:
    sc = common_lines_unattractive_scenario(1000.0)
    strat = optimal_strategy(sc)
    np.testing.assert_allclose(strat.arc_volumes, [1000.0, 0.0], atol=1e-9)
    assert strat.pair_costs[0] == pytest.approx(21.0)


# ------------------------------------------------------------- certificate

def test_certificate_optimal_zero_gap() -> None:
    for sc in (common_lines_scenario(1000.0), common_lines_unattractive_scenario(1000.0)):
        m = TransitEvaluator(sc).certify(optimal_strategy(sc))
        assert m["feasible"] == 1.0
        assert m["optimality_gap"] == pytest.approx(0.0, abs=1e-12)
        assert m["conservation_residual"] == pytest.approx(0.0, abs=1e-9)
        # Primal cost recomputed from emitted arcs == harness LP optimum.
        assert m["total_expected_cost"] == pytest.approx(m["optimal_total_cost"])


def test_certificate_matches_reference() -> None:
    sc = common_lines_scenario(1000.0)
    m = TransitEvaluator(sc).certify(optimal_strategy(sc))
    assert m["optimal_total_cost"] == pytest.approx(sc.reference.expected_total_cost)
    assert m["optimal_total_cost"] == pytest.approx(24000.0)


def test_certificate_penalizes_feasible_but_suboptimal() -> None:
    """All demand on the slower single line: feasible (one arc, trivially
    proportional) but suboptimal (30 > 24) → positive optimality gap."""
    sc = common_lines_scenario(1000.0)
    labels = (1, np.array([np.nan, 0.0]))  # unused by the certifier
    bad = TransitStrategy(
        arc_volumes=np.array([0.0, 1000.0]), labels=(labels,), pair_costs=np.array([30.0])
    )
    m = TransitEvaluator(sc).certify(bad)
    assert m["feasible"] == 1.0
    assert m["optimality_gap"] == pytest.approx(0.25)  # (30000 - 24000)/24000


def test_certificate_censors_nonconserving() -> None:
    sc = common_lines_scenario(1000.0)
    labels = (1, np.array([np.nan, 0.0]))
    # Volumes do not route the demand (both lines under-loaded).
    bad = TransitStrategy(
        arc_volumes=np.array([100.0, 100.0]), labels=(labels,), pair_costs=np.array([24.0])
    )
    m = TransitEvaluator(sc).certify(bad)
    assert m["feasible"] == 0.0
    assert np.isnan(m["optimality_gap"])


def test_certificate_censors_negative() -> None:
    sc = common_lines_scenario(1000.0)
    labels = (1, np.array([np.nan, 0.0]))
    bad = TransitStrategy(
        arc_volumes=np.array([-50.0, 1050.0]), labels=(labels,), pair_costs=np.array([24.0])
    )
    assert TransitEvaluator(sc).certify(bad)["feasible"] == 0.0


def test_certificate_nonproportional_split_is_suboptimal() -> None:
    """Both lines used but NOT in frequency proportion: feasible but suboptimal.
    The LP-minimal wait w_i = max_a(v_a/f_a) charges the excess, so the gap is
    positive (not censored, and never negative)."""
    sc = common_lines_scenario(1000.0)
    bad = TransitStrategy(
        arc_volumes=np.array([500.0, 500.0]), labels=(), pair_costs=np.array([24.0])
    )
    m = TransitEvaluator(sc).certify(bad)
    assert m["feasible"] == 1.0
    # w0 = max(500/(1/6), 500/(1/12)) = 6000; Z = 21*500 + 18*500 + 6000 = 25500.
    assert m["optimality_gap"] == pytest.approx(1500 / 24000)  # 0.0625


def test_certificate_near_zero_frequency_no_negative_gap() -> None:
    """A sub-tolerance sliver routed through a near-zero-frequency parasite arc
    must NOT dodge its (enormous) wait charge and drive the gap negative — the
    LP-minimal wait w_i = max_a(v_a/f_a) charges it (regression for the
    adversarial re-review CRITICAL)."""
    net = TransitNetwork(
        n_nodes=2,
        tail=np.array([0, 0]),
        head=np.array([1, 1]),
        time=np.array([10.0, 10.0]),
        freq=np.array([0.5, 1e-12]),  # bulk line + a near-zero-frequency parasite
    )
    dem = TransitDemand(np.array([0]), np.array([1]), np.array([1000.0]))
    sc = TransitScenario(name="parasite", network=net, demand=dem)
    eps = 0.999e-3  # just under the feasibility tolerance 1e-6 * 1000
    bad = TransitStrategy(
        arc_volumes=np.array([1000.0 - eps, eps]), labels=(), pair_costs=np.array([12.0])
    )
    m = TransitEvaluator(sc).certify(bad)
    assert m["feasible"] == 1.0
    assert m["optimality_gap"] >= 0.0  # never negative
    assert m["optimality_gap"] > 1e3  # the parasite arc's wait is astronomically charged


def test_empty_demand_certifies() -> None:
    """A zero-demand scenario's (empty) optimal solution certifies trivially."""
    net = TransitNetwork(
        n_nodes=2, tail=np.array([0]), head=np.array([1]),
        time=np.array([5.0]), freq=np.array([1.0]),
    )
    dem = TransitDemand(np.array([], dtype=int), np.array([], dtype=int), np.array([]))
    sc = TransitScenario(name="empty", network=net, demand=dem)
    m = TransitEvaluator(sc).certify(optimal_strategy(sc))
    assert m["feasible"] == 1.0
    assert m["optimality_gap"] == pytest.approx(0.0)


def test_certificate_wrong_shape_raises() -> None:
    sc = common_lines_scenario(1000.0)
    labels = (1, np.array([np.nan, 0.0]))
    bad = TransitStrategy(arc_volumes=np.zeros(3), labels=(labels,), pair_costs=np.array([24.0]))
    with pytest.raises(ValueError, match="arc_volumes shape"):
        TransitEvaluator(sc).certify(bad)


# ------------------------------------------------------------- multi-leg / interchange

def test_interchange_label_propagation() -> None:
    """origin 0 -> interchange 1 -> destination 2. At node 1 the two common lines
    give u[1]=24; a single line 0->1 (f=1/10, t=5) adds wait 10 + 5, so u[0]=39."""
    net = TransitNetwork(
        n_nodes=3,
        tail=np.array([0, 1, 1]),
        head=np.array([1, 2, 2]),
        time=np.array([5.0, 21.0, 18.0]),
        freq=np.array([1 / 10, 1 / 6, 1 / 12]),
    )
    dem = TransitDemand(np.array([0]), np.array([2]), np.array([100.0]))
    sc = TransitScenario(name="interchange", network=net, demand=dem)
    strat = optimal_strategy(sc)
    assert strat.pair_costs[0] == pytest.approx(39.0)
    # arc 0->1 carries all 100; then split 2:1 at node 1.
    np.testing.assert_allclose(strat.arc_volumes, [100.0, 200 / 3, 100 / 3], atol=1e-9)
    assert TransitEvaluator(sc).certify(strat)["optimality_gap"] == pytest.approx(0.0, abs=1e-12)


def test_deterministic_arc_dominates() -> None:
    """A deterministic walk (freq=inf, 25 min) beats a bus with expected cost 27,
    so the optimal strategy walks (cost 25) and the bus carries nobody."""
    net = TransitNetwork(
        n_nodes=2,
        tail=np.array([0, 0]),
        head=np.array([1, 1]),
        time=np.array([21.0, 25.0]),  # bus 21 (+wait 6 => 27), walk 25
        freq=np.array([1 / 6, np.inf]),
    )
    dem = TransitDemand(np.array([0]), np.array([1]), np.array([100.0]))
    sc = TransitScenario(name="walk", network=net, demand=dem)
    strat = optimal_strategy(sc)
    assert strat.pair_costs[0] == pytest.approx(25.0)
    np.testing.assert_allclose(strat.arc_volumes, [0.0, 100.0], atol=1e-9)
    assert TransitEvaluator(sc).certify(strat)["feasible"] == 1.0


def test_multi_destination_shared_origin() -> None:
    """Two destinations share origin 0's out-arcs. The wait term is per-(node,
    destination), so the certificate must decompose by destination — recomputing
    it from the summed arc volumes would report a negative optimality gap
    (regression for the adversarial-review CRITICAL)."""
    net = TransitNetwork(
        n_nodes=4,
        tail=np.array([0, 0]),
        head=np.array([2, 3]),
        time=np.array([5.0, 5.0]),
        freq=np.array([1 / 10, 1 / 20]),
    )
    dem = TransitDemand(np.array([0, 0]), np.array([2, 3]), np.array([100.0, 50.0]))
    sc = TransitScenario(name="multidest", network=net, demand=dem)
    strat = optimal_strategy(sc)
    np.testing.assert_allclose(strat.arc_volumes, [100.0, 50.0], atol=1e-9)
    m = TransitEvaluator(sc).certify(strat)
    assert m["optimal_total_cost"] == pytest.approx(2750.0)  # 100*15 + 50*25
    assert m["total_expected_cost"] == pytest.approx(2750.0)
    assert m["optimality_gap"] == pytest.approx(0.0, abs=1e-12)  # never negative
    assert m["feasible"] == 1.0


def test_certificate_censors_multidest_without_decomposition() -> None:
    """A multi-destination scenario cannot be certified from the summed arc
    volumes alone (no per-destination decomposition)."""
    net = TransitNetwork(
        n_nodes=4,
        tail=np.array([0, 0]),
        head=np.array([2, 3]),
        time=np.array([5.0, 5.0]),
        freq=np.array([1 / 10, 1 / 20]),
    )
    dem = TransitDemand(np.array([0, 0]), np.array([2, 3]), np.array([100.0, 50.0]))
    sc = TransitScenario(name="multidest", network=net, demand=dem)
    agg = optimal_strategy(sc).arc_volumes
    stripped = TransitStrategy(
        arc_volumes=agg, labels=(), pair_costs=np.array([15.0, 25.0])
    )  # dest_arc_volumes defaults to ()
    assert TransitEvaluator(sc).certify(stripped)["feasible"] == 0.0


# ------------------------------------------------------------- P8 + hashing

def test_determinism() -> None:
    sc = common_lines_scenario(777.0)
    np.testing.assert_array_equal(
        optimal_strategy(sc).arc_volumes, optimal_strategy(sc).arc_volumes
    )


def test_content_hash_stable_and_sensitive() -> None:
    sc = common_lines_scenario(1000.0)
    assert sc.content_hash() == common_lines_scenario(1000.0).content_hash()
    assert sc.content_hash() != common_lines_scenario(999.0).content_hash()
    assert sc.content_hash() != common_lines_unattractive_scenario(1000.0).content_hash()
    # Domain-separated: the transit hash carries its own prefix.
    assert sc.content_hash().isalnum()


def test_content_hash_domain_prefix_distinct_from_road() -> None:
    """A transit scenario's hash must not collide with any road/DNL hash space."""
    sc = common_lines_scenario()
    braess_hash = tb.braess_scenario().content_hash()
    assert sc.content_hash() != braess_hash


# ------------------------------------------------------------- validation

def test_network_rejects_bad_frequency() -> None:
    with pytest.raises(ValueError, match="frequencies must be > 0"):
        TransitNetwork(2, np.array([0]), np.array([1]), np.array([5.0]), np.array([0.0]))


def test_network_rejects_negative_time() -> None:
    with pytest.raises(ValueError, match="times must be finite and >= 0"):
        TransitNetwork(2, np.array([0]), np.array([1]), np.array([-1.0]), np.array([1.0]))


def test_network_rejects_out_of_range_endpoint() -> None:
    with pytest.raises(ValueError, match="out of range"):
        TransitNetwork(2, np.array([0]), np.array([5]), np.array([5.0]), np.array([1.0]))


def test_demand_rejects_intrazonal() -> None:
    with pytest.raises(ValueError, match="intrazonal"):
        TransitDemand(np.array([1]), np.array([1]), np.array([10.0]))


def test_demand_rejects_negative_node_id() -> None:
    """A negative id would wrap to the wrong node under numpy indexing."""
    with pytest.raises(ValueError, match=">= 0"):
        TransitDemand(np.array([-1]), np.array([1]), np.array([10.0]))


def test_demand_rejects_fractional_node_id() -> None:
    """A fractional id would be silently truncated by the int64 cast."""
    with pytest.raises(ValueError, match="integer node ids"):
        TransitDemand(np.array([0.9]), np.array([1.0]), np.array([10.0]))


def test_golden_braess_hash_unaffected() -> None:
    """The parallel transit module touches no road code; the golden hash holds."""
    assert (
        tb.braess_scenario().content_hash()
        == "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"
    )


def test_reference_dataclass() -> None:
    ref = TransitReference(expected_total_cost=24000.0, source="analytic", note="x")
    assert ref.expected_total_cost == 24000.0
