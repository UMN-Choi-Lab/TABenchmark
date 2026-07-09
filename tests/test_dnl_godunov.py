"""GodunovLink (Lebacque 1996) + GreenshieldsFD — the first non-triangular FD and
the first rarefaction physics in the benchmark.

The Godunov scheme is CTM's cell update run on a GENERAL concave FD; the smooth
Greenshields parabola produces rarefaction fans a triangular FD cannot. Anchors
hand-derived + machine-verified (adr-018).
"""

import math

import numpy as np
import pytest

from tabench.core.scenario import Network
from tabench.dnl import (
    DynamicDemand,
    DynamicScenario,
    GodunovLink,
    GreenshieldsFD,
    LinkDynamics,
    NetworkLoader,
    TimeGrid,
)
from tabench.metrics import DNLEvaluator

# ---------------------------------------------------------------------------
# GreenshieldsFD: the first smooth, strictly concave (non-triangular) FD.
# ---------------------------------------------------------------------------


def test_greenshields_derived_quantities() -> None:
    fd = GreenshieldsFD(vf=2.0, kappa=4.0)
    assert fd.capacity == pytest.approx(2.0)  # vf*kappa/4
    assert fd.critical_density == pytest.approx(2.0)  # kappa/2
    assert fd.free_speed == pytest.approx(2.0)
    assert fd.wave_speed == pytest.approx(2.0)  # |Q'(kappa)| = vf (symmetric)
    assert fd.jam_density == pytest.approx(4.0)


def test_greenshields_demand_supply_branches() -> None:
    fd = GreenshieldsFD(vf=2.0, kappa=4.0)  # k_c = 2, q_max = 2
    k = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(fd.flow_at(k), [0.0, 1.5, 2.0, 1.5, 0.0])  # parabola
    np.testing.assert_allclose(fd.demand_at(k), [0.0, 1.5, 2.0, 2.0, 2.0])  # Q(min(k,k_c))
    np.testing.assert_allclose(fd.supply_at(k), [2.0, 2.0, 2.0, 1.5, 0.0])  # Q(max(k,k_c))


def test_greenshields_is_majorized_by_its_triangular_envelope() -> None:
    """Q(k) <= min(vf*k, vf*(kappa-k)) — the envelope_params majorant the
    certificates rely on, sound (necessary) for the concave FD."""
    fd = GreenshieldsFD(vf=1.5, kappa=6.0)
    vf, w, kappa = fd.envelope_params()
    assert (vf, w, kappa) == pytest.approx((1.5, 1.5, 6.0))
    k = np.linspace(0.0, 6.0, 200)
    assert np.all(fd.flow_at(k) <= np.minimum(vf * k, w * (kappa - k)) + 1e-12)


def test_greenshields_validation() -> None:
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            GreenshieldsFD(vf=bad, kappa=4.0)
        with pytest.raises(ValueError):
            GreenshieldsFD(vf=1.0, kappa=bad)


def test_godunov_flux_is_entropy_correct_at_a_transonic_rarefaction() -> None:
    """The distinguishing physics: at a transonic interface k_L > k_c > k_R the
    Godunov flux min(demand(k_L), supply(k_R)) is the sonic-point capacity q_max
    (the entropy-correct rarefaction value) — not the shock value."""
    fd = GreenshieldsFD(vf=2.0, kappa=4.0)  # q_max = 2
    flux = min(float(fd.demand_at(np.array([3.5]))[0]), float(fd.supply_at(np.array([0.5]))[0]))
    assert flux == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# GodunovLink: the general-FD Godunov cell scheme.
# ---------------------------------------------------------------------------


def _greenshields_dynamics(n: int, vf: float, kappa: float, length: float) -> LinkDynamics:
    """Greenshields-consistent LinkDynamics: w = vf and capacity = vf*kappa/4 so
    dynamics.fd(a)'s triangular majorant + capacity match what GodunovLink loads."""
    return LinkDynamics(
        length=np.full(n, length), free_speed=np.full(n, vf), wave_speed=np.full(n, vf),
        jam_density=np.full(n, kappa), capacity=np.full(n, vf * kappa / 4.0),
    )


def test_godunov_link_uses_a_greenshields_fd() -> None:
    dyn = _greenshields_dynamics(1, 1.0, 4.0, 8.0)
    link = GodunovLink(dyn.fd(0), 8.0, TimeGrid(1.0, 10))
    assert isinstance(link.fd, GreenshieldsFD)
    assert link.n_cells == 8  # dx = vf*dt = 1


def test_godunov_rejects_inconsistent_link_dynamics() -> None:
    """A LinkDynamics whose capacity isn't the Greenshields vf*kappa/4 would make
    the certifier gate against the wrong capacity — GodunovLink rejects it."""
    bad = LinkDynamics(
        length=np.array([8.0]), free_speed=np.array([1.0]), wave_speed=np.array([1.0]),
        jam_density=np.array([4.0]), capacity=np.array([1.5]),  # != vf*kappa/4 = 1.0
    )
    with pytest.raises(ValueError, match="Greenshields-consistent capacity"):
        GodunovLink(bad.fd(0), 8.0, TimeGrid(1.0, 10))


def test_godunov_loads_greenshields_and_certifies() -> None:
    """A Greenshields loading through the loader certifies: the certifier's
    triangular-majorant envelopes are sound (necessary) for the smooth concave FD
    even though the emitted flux is parabolic."""
    net = Network(
        name="gd", n_nodes=2, n_zones=2, first_thru_node=1,
        init_node=np.array([1]), term_node=np.array([2]),
        capacity=np.ones(1), length=np.zeros(1), free_flow_time=np.ones(1),
        b=np.zeros(1), power=np.ones(1), toll=np.zeros(1), link_type=np.ones(1, dtype=np.int64),
    )
    rates = np.zeros((1, 2, 2))
    rates[0, 0, 1] = 0.6  # < capacity 1.0
    scenario = DynamicScenario(
        name="gd", network=net, dynamics=_greenshields_dynamics(1, 1.0, 4.0, 8.0),
        demand=DynamicDemand(breakpoints=np.array([0.0, 20.0]), rates=rates),
        grid=TimeGrid(dt=1.0, n_steps=30),
    )
    metrics = DNLEvaluator(scenario).evaluate(NetworkLoader(scenario, GodunovLink).run())
    assert metrics["dnl_feasible"] == 1.0
    assert metrics["conservation_residual"] <= 1e-9


def _rarefaction_l1_error(n_cells: int) -> float:
    """Dam-break Riemann problem (jam left, empty right) stepped by the Godunov
    flux; L1 error of the interior fan vs the analytic self-similar solution
    k(x,t) = (kappa/2)(1 - (x-x0)/(vf*t))."""
    vf, kappa, dt = 1.0, 4.0, 1.0
    dx = vf * dt  # CFL = 1
    link = GodunovLink(
        LinkDynamics(
            length=np.array([n_cells * dx]), free_speed=np.array([vf]), wave_speed=np.array([vf]),
            jam_density=np.array([kappa]), capacity=np.array([vf * kappa / 4.0]),
        ).fd(0),
        n_cells * dx, TimeGrid(dt, n_cells),
    )
    occ = link._occ
    occ[: n_cells // 2] = kappa * dx  # jam on the left half
    occ[n_cells // 2:] = 0.0  # empty on the right
    steps = n_cells // 4
    for _ in range(steps):  # interior Godunov flux (boundaries drift but stay far)
        dens = occ / dx
        y = np.minimum(link.fd.demand_at(dens[:-1]), link.fd.supply_at(dens[1:])) * dt
        occ[:-1] -= y
        occ[1:] += y
    x0 = (n_cells // 2) * dx
    xc = (np.arange(n_cells) + 0.5) * dx
    k_ana = np.clip((kappa / 2.0) * (1.0 - (xc - x0) / (vf * steps)), 0.0, kappa)
    lo, hi = n_cells // 2 - steps + 1, n_cells // 2 + steps - 1  # fan interior, off boundaries
    return float(np.abs(occ[lo:hi] / dx - k_ana[lo:hi]).mean())


def test_godunov_rarefaction_converges_to_the_analytic_fan() -> None:
    """The Godunov scheme resolves a rarefaction fan (jam clearing) that a
    triangular FD cannot produce; first-order convergence — the L1 error shrinks
    monotonically as the cells are refined."""
    errors = [_rarefaction_l1_error(n) for n in (20, 40, 80, 160)]
    assert all(errors[i + 1] < errors[i] for i in range(len(errors) - 1))  # monotone
    assert errors[-1] < 0.5 * errors[0]  # materially converging
    assert errors[-1] < 0.06


def test_godunov_does_not_move_the_golden_braess_hash() -> None:
    from tabench.data.builtin import braess_scenario

    assert (
        braess_scenario().content_hash()
        == "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"
    )
