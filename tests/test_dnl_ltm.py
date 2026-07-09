"""LTMLink (Yperman 2007 link transmission model) — anchors + CTM cross-check.

The Newell-Daganzo cumulative-curve method: LTM matches CTM's boundary curves
where both are exact, and runs on wave-resolved grids CTM's cell-alignment
rejects. Anchors hand-derived from Yperman (2007) §4.6 / Boyles TNA §9.5.2 (both
open, read) and machine-verified (adr-016).
"""

import math

import numpy as np
import pytest

from tabench.core.scenario import Network
from tabench.dnl import (
    CTMLink,
    DynamicDemand,
    DynamicScenario,
    LinkDynamics,
    LinkModel,
    LTMLink,
    NetworkLoader,
    TimeGrid,
    TriangularFD,
)
from tabench.metrics import DNLEvaluator


def _single(name: str) -> Network:
    return Network(
        name=name, n_nodes=2, n_zones=2, first_thru_node=1,
        init_node=np.array([1], dtype=np.int64), term_node=np.array([2], dtype=np.int64),
        capacity=np.ones(1), length=np.zeros(1), free_flow_time=np.ones(1),
        b=np.zeros(1), power=np.ones(1), toll=np.zeros(1), link_type=np.ones(1, dtype=np.int64),
    )


def _corridor(name: str) -> Network:
    return Network(
        name=name, n_nodes=3, n_zones=2, first_thru_node=3,
        init_node=np.array([1, 3], dtype=np.int64), term_node=np.array([3, 2], dtype=np.int64),
        capacity=np.ones(2), length=np.zeros(2), free_flow_time=np.ones(2),
        b=np.zeros(2), power=np.ones(2), toll=np.zeros(2), link_type=np.ones(2, dtype=np.int64),
    )


def _bottleneck_scenario() -> DynamicScenario:
    """Same symmetric bottleneck as CTM anchor (b): vf=w=1, kappa=4, cap 2 link
    feeding a 0.5 bottleneck at inflow 1.5."""
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 1.5
    return DynamicScenario(
        name="ltm-bottleneck", network=_corridor("ltm-bottleneck"),
        dynamics=LinkDynamics(
            length=np.array([4.0, 1.0]), free_speed=np.array([1.0, 1.0]),
            wave_speed=np.array([1.0, 1.0]), jam_density=np.array([4.0, 4.0]),
            capacity=np.array([2.0, 0.5]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 12.0]), rates=rates),
        grid=TimeGrid(dt=1.0, n_steps=12),
    )


def test_ltm_free_flow_translation_is_exact() -> None:
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 1.0
    scenario = DynamicScenario(
        name="ltm-ff", network=_single("ltm-ff"),
        dynamics=LinkDynamics(
            length=np.array([4.0]), free_speed=np.array([1.0]), wave_speed=np.array([1.0]),
            jam_density=np.array([4.0]), capacity=np.array([2.0]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 4.0]), rates=rates),
        grid=TimeGrid(dt=1.0, n_steps=10),
    )
    out = NetworkLoader(scenario, LTMLink).run()
    metrics = DNLEvaluator(scenario).evaluate(out)
    edges = scenario.grid.edges
    assert metrics["dnl_feasible"] == 1.0
    expected = np.minimum(np.maximum(edges - 4.0, 0.0), 4.0)
    np.testing.assert_allclose(out.n_out[0], expected, atol=1e-12)
    assert metrics["tstt"] == pytest.approx(16.0)
    assert metrics["total_delay"] == pytest.approx(0.0)


def test_ltm_bottleneck_matches_ctm_exactly() -> None:
    """On the CFL=1 symmetric bottleneck both models are exact, so LTM must
    reproduce CTM's cumulative curves byte-for-byte (RH shock, storage 14)."""
    scenario = _bottleneck_scenario()
    ltm = NetworkLoader(scenario, LTMLink).run()
    ctm = NetworkLoader(scenario, CTMLink).run()
    metrics = DNLEvaluator(scenario).evaluate(ltm)
    edges = scenario.grid.edges
    assert metrics["dnl_feasible"] == 1.0
    np.testing.assert_allclose(ltm.n_in[0], 1.5 * edges, atol=1e-9)
    np.testing.assert_allclose(ltm.n_out[0], np.maximum(0.0, 0.5 * (edges - 4.0)), atol=1e-9)
    assert ltm.n_in[0, -1] - ltm.n_out[0, -1] == pytest.approx(14.0, abs=1e-9)
    np.testing.assert_allclose(ltm.n_in, ctm.n_in, atol=1e-9)
    np.testing.assert_allclose(ltm.n_out, ctm.n_out, atol=1e-9)


def test_ltm_asymmetric_wave_spillback() -> None:
    """w < vf (backward wave slower than free flow, the physical norm): vf=2,
    w=1, kappa=3, cap 2 link feeding a 0.5 bottleneck at inflow 1.0. RH shock
    speed s=(1-0.5)/(0.5-2.5)=-0.25 reaches x=0 at t=18; n_out=max(0,0.5(t-2)),
    storage k_B*L = 2.5*4 = 10 (Yperman receiving-recursion, machine-verified)."""
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 1.0
    scenario = DynamicScenario(
        name="ltm-asym", network=_corridor("ltm-asym"),
        dynamics=LinkDynamics(
            length=np.array([4.0, 2.0]), free_speed=np.array([2.0, 2.0]),
            wave_speed=np.array([1.0, 1.0]), jam_density=np.array([3.0, 3.0]),
            capacity=np.array([2.0, 0.5]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 22.0]), rates=rates),
        grid=TimeGrid(dt=1.0, n_steps=22),
    )
    out = NetworkLoader(scenario, LTMLink).run()
    metrics = DNLEvaluator(scenario).evaluate(out)
    edges = scenario.grid.edges
    assert metrics["dnl_feasible"] == 1.0
    np.testing.assert_allclose(out.n_out[0], np.maximum(0.0, 0.5 * (edges - 2.0)), atol=1e-9)
    # spillback reaches x=0 at t=18: n_in tracks the 1.0 inflow until then.
    assert out.n_in[0, 18] == pytest.approx(18.0, abs=1e-9)
    assert out.n_in[0, 18] - out.n_out[0, 18] == pytest.approx(10.0, abs=1e-9)  # k_B*L = 2.5*4


def test_ltm_runs_on_unaligned_grid_where_ctm_raises() -> None:
    """LTM's chief advantage: no CFL=1 cell alignment. L=3, vf=2, dt=1 gives
    L/vf=1.5 (non-integer) — CTMLink raises, LTM free-flow-translates exactly by
    the 1.5 lag via the exact cumulative-curve interpolation."""
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 1.0
    scenario = DynamicScenario(
        name="ltm-unaligned", network=_single("ltm-unaligned"),
        dynamics=LinkDynamics(
            length=np.array([3.0]), free_speed=np.array([2.0]), wave_speed=np.array([1.0]),
            jam_density=np.array([3.0]), capacity=np.array([2.0]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 4.0]), rates=rates),
        grid=TimeGrid(dt=1.0, n_steps=10),
    )
    out = NetworkLoader(scenario, LTMLink).run()
    metrics = DNLEvaluator(scenario).evaluate(out)
    edges = scenario.grid.edges
    assert metrics["dnl_feasible"] == 1.0
    expected = np.minimum(np.maximum(edges - 1.5, 0.0), 4.0)  # free-flow lag L/vf = 1.5
    np.testing.assert_allclose(out.n_out[0], expected, atol=1e-12)
    with pytest.raises(ValueError, match="cell-aligned"):
        NetworkLoader(scenario, CTMLink).run()


def test_ltm_is_stateless() -> None:
    """LTM keeps no interior state — it does not override the base no-op
    _advance_state (all state is the base cumulative curves)."""
    assert LTMLink._advance_state is LinkModel._advance_state


def test_ltm_requires_finite_jam_density() -> None:
    with pytest.raises(ValueError, match="finite jam density"):
        LTMLink(TriangularFD(vf=1.0, w=math.inf, kappa=math.inf, q_cap=2.0), 4.0, TimeGrid(1.0, 10))


def test_ltm_deterministic() -> None:
    scenario = _bottleneck_scenario()
    a = NetworkLoader(scenario, LTMLink).run()
    b = NetworkLoader(scenario, LTMLink).run()
    np.testing.assert_array_equal(a.n_in, b.n_in)
    np.testing.assert_array_equal(a.n_out, b.n_out)


def test_ltm_does_not_move_the_golden_braess_hash() -> None:
    from tabench.data.builtin import braess_scenario

    assert (
        braess_scenario().content_hash()
        == "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"
    )
