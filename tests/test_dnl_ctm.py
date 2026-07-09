"""CTMLink (Daganzo 1994/1995 cell transmission model) — anchors + certification.

All three analytic anchors are hand-derived (docs/design/adr-015-ctm) and
machine-verified here against the shipped ``DNLEvaluator``, mirroring the
point-queue reference's ``test_dnl_link_reference.py``. Common FD: triangular
``vf = w = 1``, ``kappa = 4`` -> ``capacity = vf*w*kappa/(vf+w) = 2``,
``k_c = 2``; ``dt = 1`` -> cell length ``dx = vf*dt = 1``.
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
    NetworkLoader,
    TimeGrid,
    TriangularFD,
)
from tabench.metrics import DNLEvaluator


def _single_link_network(name: str) -> Network:
    return Network(
        name=name,
        n_nodes=2,
        n_zones=2,
        first_thru_node=1,
        init_node=np.array([1], dtype=np.int64),
        term_node=np.array([2], dtype=np.int64),
        capacity=np.ones(1),
        length=np.zeros(1),
        free_flow_time=np.ones(1),
        b=np.zeros(1),
        power=np.ones(1),
        toll=np.zeros(1),
        link_type=np.ones(1, dtype=np.int64),
    )


def _corridor_network(name: str) -> Network:
    """Origin zone 1 -> interior node 3 -> dest zone 2 (two links, a series node)."""
    return Network(
        name=name,
        n_nodes=3,
        n_zones=2,
        first_thru_node=3,
        init_node=np.array([1, 3], dtype=np.int64),
        term_node=np.array([3, 2], dtype=np.int64),
        capacity=np.ones(2),
        length=np.zeros(2),
        free_flow_time=np.ones(2),
        b=np.zeros(2),
        power=np.ones(2),
        toll=np.zeros(2),
        link_type=np.ones(2, dtype=np.int64),
    )


def _free_flow_scenario() -> DynamicScenario:
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 1.0  # rate 1.0 < capacity 2 -> uncongested throughout
    return DynamicScenario(
        name="ctm-free-flow",
        network=_single_link_network("ctm-free-flow"),
        dynamics=LinkDynamics(
            length=np.array([4.0]),
            free_speed=np.array([1.0]),
            wave_speed=np.array([1.0]),
            jam_density=np.array([4.0]),
            capacity=np.array([2.0]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 4.0]), rates=rates),
        grid=TimeGrid(dt=1.0, n_steps=10),
    )


def _bottleneck_scenario() -> DynamicScenario:
    """CTM link (L=4, cap 2) feeding a low-capacity (0.5) sink link: a backward
    shock at RH speed s = -0.5 builds from t = L/vf = 4."""
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 1.5  # arrival state k_A = 1.5 < k_c = 2 (uncongested inflow)
    return DynamicScenario(
        name="ctm-bottleneck",
        network=_corridor_network("ctm-bottleneck"),
        dynamics=LinkDynamics(
            length=np.array([4.0, 1.0]),
            free_speed=np.array([1.0, 1.0]),
            wave_speed=np.array([1.0, 1.0]),
            jam_density=np.array([4.0, 4.0]),
            capacity=np.array([2.0, 0.5]),  # link 2 = the 0.5 bottleneck
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 12.0]), rates=rates),
        grid=TimeGrid(dt=1.0, n_steps=12),
    )


# ---------------------------------------------------------------------------
# Anchor (a): free-flow translation is bit-exact at CFL = 1.
# ---------------------------------------------------------------------------


def test_ctm_free_flow_translation_is_exact() -> None:
    scenario = _free_flow_scenario()
    out = NetworkLoader(scenario, CTMLink).run()
    metrics = DNLEvaluator(scenario).evaluate(out)
    edges = scenario.grid.edges

    assert metrics["dnl_feasible"] == 1.0
    # n_out(t) = n_in(t - L/vf) exactly (Courant number 1 advection, zero diffusion).
    np.testing.assert_allclose(out.n_in[0], np.minimum(edges, 4.0), atol=1e-12)
    np.testing.assert_allclose(
        out.n_out[0], np.minimum(np.maximum(edges - 4.0, 0.0), 4.0), atol=1e-12
    )
    assert metrics["tstt"] == pytest.approx(16.0)  # (L/vf) * D = 4 * 4
    assert metrics["total_delay"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Anchor (b): queue spillback — RH shock, exact bottleneck boundary curves.
# ---------------------------------------------------------------------------


def test_ctm_bottleneck_shock_boundary_curves() -> None:
    scenario = _bottleneck_scenario()
    out = NetworkLoader(scenario, CTMLink).run()
    metrics = DNLEvaluator(scenario).evaluate(out)
    edges = scenario.grid.edges

    assert metrics["dnl_feasible"] == 1.0
    # Cell 0 stays uncongested until full spillback at t=12, so inflow = 1.5/step.
    np.testing.assert_allclose(out.n_in[0], 1.5 * edges, atol=1e-9)
    # Exit is bottleneck-capped at 0.5 from t = L/vf = 4 (free-flow front arrival).
    np.testing.assert_allclose(out.n_out[0], np.maximum(0.0, 0.5 * (edges - 4.0)), atol=1e-9)
    # Storage at t=12 = k_B * L = 3.5 * 4 = 14 (RH congested density k_B = kappa - q_B/w).
    assert out.n_in[0, -1] - out.n_out[0, -1] == pytest.approx(14.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Anchor (c): the congested cells settle at the supply-side root k = kappa - q_B/w.
# ---------------------------------------------------------------------------


def test_ctm_congested_density_is_supply_root() -> None:
    scenario = _bottleneck_scenario()
    loader = NetworkLoader(scenario, CTMLink)
    loader.run()
    fd = scenario.dynamics.fd(0)
    dx = fd.free_speed * scenario.grid.dt
    occ = loader.links[0].occupancy
    # every cell fully engulfed by the queue holds density k_B = 4 - 0.5/1 = 3.5,
    # the root of supply_at(k) = q_B on the congested branch.
    k_b = fd.jam_density - 0.5 / fd.wave_speed
    np.testing.assert_allclose(occ / dx, k_b, atol=1e-9)
    assert float(fd.supply_at(np.array([k_b]))[0]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Interface / structure.
# ---------------------------------------------------------------------------


def test_ctm_single_cell_link_is_a_one_step_lag() -> None:
    """L = vf*dt is a single cell: uncongested it is an exact one-step free-flow
    lag (n_out(t) = n_in(t - dt))."""
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 1.0
    scenario = DynamicScenario(
        name="ctm-single-cell",
        network=_single_link_network("ctm-single-cell"),
        dynamics=LinkDynamics(
            length=np.array([1.0]),
            free_speed=np.array([1.0]),
            wave_speed=np.array([1.0]),
            jam_density=np.array([4.0]),
            capacity=np.array([2.0]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 4.0]), rates=rates),
        grid=TimeGrid(dt=1.0, n_steps=8),
    )
    loader = NetworkLoader(scenario, CTMLink)
    out = loader.run()
    assert loader.links[0].n_cells == 1
    assert DNLEvaluator(scenario).evaluate(out)["dnl_feasible"] == 1.0
    edges = scenario.grid.edges
    np.testing.assert_allclose(
        out.n_out[0], np.minimum(np.maximum(edges - 1.0, 0.0), 4.0), atol=1e-12
    )


def test_ctm_requires_finite_jam_density() -> None:
    with pytest.raises(ValueError, match="finite jam density"):
        CTMLink(TriangularFD(vf=1.0, w=math.inf, kappa=math.inf, q_cap=2.0), 4.0, TimeGrid(1.0, 10))


def test_ctm_requires_cell_aligned_length() -> None:
    # L = 3.5, vf*dt = 1.0 -> 3.5 cells, not an integer.
    with pytest.raises(ValueError, match="cell-aligned"):
        CTMLink(TriangularFD(vf=1.0, w=1.0, kappa=4.0), 3.5, TimeGrid(1.0, 10))


def test_ctm_rejects_faster_backward_wave() -> None:
    """w > vf is a legal FD but unresolvable at CFL = 1 (cell = vf*dt): the
    congested-branch flux would overfill a cell past jam. CTMLink must raise
    rather than silently mis-simulate (adversarial-review regression: 66% of
    such configs otherwise produce C3-censored, physically impossible output)."""
    with pytest.raises(ValueError, match="w <= vf|backward wave"):
        CTMLink(TriangularFD(vf=1.0, w=3.0, kappa=4.0), 3.0, TimeGrid(1.0, 10))


def test_ctm_asymmetric_fd_spillback_certifies() -> None:
    """A genuine w < vf FD (backward wave slower than free flow, the physical
    norm) under spillback still certifies — the anchors above use w = vf = 1,
    a symmetric special case, so this exercises the general regime."""
    net = _corridor_network("ctm-asym")
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 3.0  # above the 1.0 bottleneck -> spillback
    scenario = DynamicScenario(
        name="ctm-asym",
        network=net,
        dynamics=LinkDynamics(
            length=np.array([4.0, 2.0]),
            free_speed=np.array([2.0, 2.0]),  # vf = 2
            wave_speed=np.array([1.0, 1.0]),  # w = 1 < vf
            jam_density=np.array([6.0, 6.0]),
            capacity=np.array([4.0, 1.0]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 20.0]), rates=rates),
        grid=TimeGrid(dt=1.0, n_steps=20),
    )
    metrics = DNLEvaluator(scenario).evaluate(NetworkLoader(scenario, CTMLink).run())
    assert metrics["dnl_feasible"] == 1.0
    assert metrics["conservation_residual"] <= 1e-9
    assert metrics["storage_residual"] <= 1e-9  # no jam overshoot in the well-posed regime


def test_ctm_deterministic() -> None:
    scenario = _bottleneck_scenario()
    a = NetworkLoader(scenario, CTMLink).run()
    b = NetworkLoader(scenario, CTMLink).run()
    np.testing.assert_array_equal(a.n_in, b.n_in)
    np.testing.assert_array_equal(a.n_out, b.n_out)


def test_ctm_does_not_move_the_golden_braess_hash() -> None:
    from tabench.data.builtin import braess_scenario

    assert (
        braess_scenario().content_hash()
        == "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"
    )
