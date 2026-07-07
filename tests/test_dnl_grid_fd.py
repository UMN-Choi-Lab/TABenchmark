"""DNL primitives tests: TimeGrid, assert_wave_resolved, TriangularFD, LinkDynamics."""

import math
from dataclasses import dataclass

import numpy as np
import pytest

from tabench.dnl.fd import FundamentalDiagram, LinkDynamics, TriangularFD
from tabench.dnl.grid import TimeGrid, assert_wave_resolved

# ---------------------------------------------------------------------------
# TimeGrid
# ---------------------------------------------------------------------------


def test_time_grid_validation():
    for dt in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            TimeGrid(dt=dt, n_steps=4)
    with pytest.raises(ValueError):
        TimeGrid(dt=0.5, n_steps=0)
    with pytest.raises(ValueError):
        TimeGrid(dt=0.5, n_steps=-3)
    with pytest.raises(ValueError):
        TimeGrid(dt=0.5, n_steps=2.0)  # non-int
    with pytest.raises(ValueError):
        TimeGrid(dt=0.5, n_steps=True)  # bool is not a step count


def test_time_grid_edges_and_horizon():
    g = TimeGrid(dt=0.5, n_steps=24)
    assert g.horizon == 12.0
    assert g.edges.dtype == np.float64
    assert g.edges.shape == (25,)
    assert np.array_equal(g.edges, 0.5 * np.arange(25))
    # numpy step counts coerce to a plain int, so grid equality (used by the
    # C0 certificate) compares canonical values.
    assert TimeGrid(dt=0.5, n_steps=np.int64(24)) == g


def test_index_at_or_after_boundaries():
    g = TimeGrid(dt=0.5, n_steps=10)
    # exact hit on an edge
    assert g.index_at_or_after(1.5) == 3
    assert g.index_at_or_after(0.0) == 0
    # mid-segment rounds up to the next edge
    assert g.index_at_or_after(1.3) == 3
    # an edge just below t (within the 1e-12*dt slack) still counts
    assert g.index_at_or_after(1.5 + 0.4e-12 * 0.5) == 3
    # beyond the slack the next edge is required
    assert g.index_at_or_after(1.5 + 1e-9) == 4
    # t just below an edge maps to that edge (smallest t_k >= t)
    assert g.index_at_or_after(1.5 - 1e-9) == 3
    # clipping
    assert g.index_at_or_after(-2.0) == 0
    assert g.index_at_or_after(99.0) == 10
    assert g.index_at_or_after(-math.inf) == 0
    assert g.index_at_or_after(math.inf) == 10
    with pytest.raises(ValueError):
        g.index_at_or_after(math.nan)


# ---------------------------------------------------------------------------
# assert_wave_resolved
# ---------------------------------------------------------------------------


def test_assert_wave_resolved_passes_at_equality():
    grid = TimeGrid(dt=0.5, n_steps=4)
    # link 0: L/vf = 0.5 exactly (CFL = 1); link 1: min(1, 1) = 1
    assert_wave_resolved(
        grid,
        length=np.array([1.0, 1.0]),
        free_speed=np.array([2.0, 1.0]),
        wave_speed=np.array([math.inf, 1.0]),
    )


def test_assert_wave_resolved_raises_on_fast_free_flow():
    grid = TimeGrid(dt=0.5, n_steps=4)
    with pytest.raises(ValueError, match="wave-resolved"):
        assert_wave_resolved(
            grid,
            length=np.array([1.0]),
            free_speed=np.array([3.0]),  # L/vf = 1/3 < dt
            wave_speed=np.array([1.0]),
        )


def test_assert_wave_resolved_finite_wave_binds_but_inf_is_exempt():
    grid = TimeGrid(dt=0.5, n_steps=4)
    length = np.array([1.0])
    free_speed = np.array([1.0])
    with pytest.raises(ValueError, match="wave-resolved"):
        assert_wave_resolved(grid, length, free_speed, np.array([3.0]))  # L/w = 1/3 < dt
    # an infinite wave speed (point queue) contributes no backward term
    assert_wave_resolved(grid, length, free_speed, np.array([math.inf]))


def test_assert_wave_resolved_input_validation():
    grid = TimeGrid(dt=0.5, n_steps=4)
    with pytest.raises(ValueError, match="equal shapes"):
        assert_wave_resolved(grid, np.array([1.0, 1.0]), np.array([1.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="length"):
        assert_wave_resolved(grid, np.array([0.0]), np.array([1.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="free_speed"):
        assert_wave_resolved(grid, np.array([1.0]), np.array([-1.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="wave_speed"):
        assert_wave_resolved(grid, np.array([1.0]), np.array([1.0]), np.array([0.0]))


# ---------------------------------------------------------------------------
# TriangularFD
# ---------------------------------------------------------------------------


def test_triangular_fd_derived_quantities():
    # C5-fixture geometry: apex = 1 * (1/3) * 180 / (1 + 1/3) = 45
    fd = TriangularFD(vf=1.0, w=1.0 / 3.0, kappa=180.0)
    assert fd.capacity == pytest.approx(45.0, rel=1e-15)
    assert fd.critical_density == pytest.approx(45.0, rel=1e-15)  # k_c = q_max / vf
    assert fd.jam_density == 180.0
    assert fd.free_speed == 1.0
    assert fd.wave_speed == 1.0 / 3.0


def test_triangular_fd_trapezoidal_cap():
    # repaired C5-fixture geometry: apex = 100, capped at 45
    fd = TriangularFD(vf=1.0, w=1.0 / 3.0, kappa=400.0, q_cap=45.0)
    assert fd.capacity == 45.0
    assert fd.critical_density == 45.0
    # flow on the cap plateau: min(vf*k, q_cap, w*(kappa - k))
    assert fd.flow_at(np.array([100.0]))[0] == 45.0


def test_demand_supply_branch_values():
    fd = TriangularFD(vf=2.0, w=1.0, kappa=150.0)  # apex = 100, k_c = 50
    assert fd.capacity == 100.0
    assert fd.critical_density == 50.0
    k = np.array([0.0, 25.0, 50.0, 100.0, 150.0])  # {0, k_c/2, k_c, (k_c+kappa)/2, kappa}
    assert np.array_equal(fd.flow_at(k), [0.0, 50.0, 100.0, 50.0, 0.0])
    assert np.array_equal(fd.demand_at(k), [0.0, 50.0, 100.0, 100.0, 100.0])
    assert np.array_equal(fd.supply_at(k), [100.0, 100.0, 100.0, 50.0, 0.0])
    # supply never negative, even past kappa
    assert fd.supply_at(np.array([200.0]))[0] == 0.0


def test_point_queue_fd_semantics():
    fd = TriangularFD(vf=1.0, w=math.inf, kappa=math.inf, q_cap=2.0)
    assert fd.capacity == 2.0
    assert fd.critical_density == 2.0
    assert math.isinf(fd.jam_density) and math.isinf(fd.wave_speed)
    k = np.array([0.0, 1.0, 2.0, 1e9])
    assert np.array_equal(fd.supply_at(k), [2.0, 2.0, 2.0, 2.0])  # q_max everywhere
    assert np.array_equal(fd.demand_at(k), [0.0, 1.0, 2.0, 2.0])
    assert np.array_equal(fd.flow_at(k), [0.0, 1.0, 2.0, 2.0])


def test_triangular_fd_validation():
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            TriangularFD(vf=bad, w=1.0, kappa=100.0)
    with pytest.raises(ValueError):
        TriangularFD(vf=1.0, w=0.0, kappa=100.0)
    with pytest.raises(ValueError):
        TriangularFD(vf=1.0, w=math.nan, kappa=100.0)
    with pytest.raises(ValueError):
        TriangularFD(vf=1.0, w=1.0, kappa=-5.0)
    # w and kappa must be infinite together
    with pytest.raises(ValueError, match="iff"):
        TriangularFD(vf=1.0, w=math.inf, kappa=100.0)
    with pytest.raises(ValueError, match="iff"):
        TriangularFD(vf=1.0, w=1.0, kappa=math.inf, q_cap=1.0)


def test_q_cap_required_iff_point_queue_and_rejected_above_apex():
    with pytest.raises(ValueError, match="requires q_cap"):
        TriangularFD(vf=1.0, w=math.inf, kappa=math.inf)
    # apex = 45 for (vf=1, w=1/3, kappa=180): a cap above it is non-canonical
    with pytest.raises(ValueError, match="apex"):
        TriangularFD(vf=1.0, w=1.0 / 3.0, kappa=180.0, q_cap=46.0)
    # a cap AT the apex is within the 1e-9 relative slack
    assert TriangularFD(vf=1.0, w=1.0 / 3.0, kappa=180.0, q_cap=45.0).capacity == 45.0
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            TriangularFD(vf=1.0, w=1.0, kappa=100.0, q_cap=bad)


@dataclass(frozen=True)
class _GreenshieldsFD(FundamentalDiagram):
    """Inline quadratic FD: Q(k) = vf * k * (1 - k/kappa) (Greenshields-style)."""

    vf: float
    kappa: float

    @property
    def capacity(self) -> float:
        return self.vf * self.kappa / 4.0

    @property
    def critical_density(self) -> float:
        return self.kappa / 2.0

    @property
    def jam_density(self) -> float:
        return self.kappa

    @property
    def free_speed(self) -> float:
        return self.vf  # Q'(0+)

    @property
    def wave_speed(self) -> float:
        return self.vf  # |Q'(kappa-)|

    def flow_at(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=np.float64)
        return self.vf * k * (1.0 - k / self.kappa)

    def demand_at(self, k: np.ndarray) -> np.ndarray:
        return self.flow_at(np.minimum(np.asarray(k, dtype=np.float64), self.critical_density))

    def supply_at(self, k: np.ndarray) -> np.ndarray:
        return self.flow_at(np.maximum(np.asarray(k, dtype=np.float64), self.critical_density))


def test_envelope_params_majorizes_quadratic_fd():
    fd = _GreenshieldsFD(vf=2.0, kappa=100.0)
    vf, w, kappa = fd.envelope_params()  # default ABC implementation (G3)
    assert (vf, w, kappa) == (2.0, 2.0, 100.0)
    k = np.linspace(0.0, 100.0, 501)
    envelope = np.minimum(vf * k, w * (kappa - k))
    assert np.all(fd.flow_at(k) <= envelope + 1e-12)


def test_envelope_params_returns_uncapped_tangents():
    capped = TriangularFD(vf=1.0, w=1.0 / 3.0, kappa=400.0, q_cap=45.0)
    assert capped.envelope_params() == (1.0, 1.0 / 3.0, 400.0)
    pq = TriangularFD(vf=1.0, w=math.inf, kappa=math.inf, q_cap=2.0)
    assert pq.envelope_params() == (1.0, math.inf, math.inf)


# ---------------------------------------------------------------------------
# LinkDynamics
# ---------------------------------------------------------------------------


def _dynamics(**overrides) -> LinkDynamics:
    """Two links: a finite-jam triangle (apex 100) and a point queue."""
    fields = dict(
        length=[1.0, 1.0],
        free_speed=[2.0, 1.0],
        wave_speed=[1.0, math.inf],
        jam_density=[150.0, math.inf],
        capacity=[50.0, 1.0],
    )
    fields.update(overrides)
    return LinkDynamics(**{name: np.asarray(v) for name, v in fields.items()})


def test_link_dynamics_valid_construction_and_coercion():
    dyn = _dynamics()
    assert dyn.n_links == 2
    for name in ("length", "free_speed", "wave_speed", "jam_density", "capacity"):
        assert getattr(dyn, name).dtype == np.float64


def test_link_dynamics_shape_validation():
    with pytest.raises(ValueError, match="equal shapes"):
        _dynamics(length=[1.0])
    with pytest.raises(ValueError, match="equal shapes"):
        _dynamics(
            length=[[1.0, 1.0]],
            free_speed=[[2.0, 1.0]],
            wave_speed=[[1.0, math.inf]],
            jam_density=[[150.0, math.inf]],
            capacity=[[50.0, 1.0]],
        )


def test_link_dynamics_positivity_and_nan_rejection():
    with pytest.raises(ValueError, match="length"):
        _dynamics(length=[0.0, 1.0])
    with pytest.raises(ValueError, match="length"):
        _dynamics(length=[math.inf, 1.0])
    with pytest.raises(ValueError, match="free_speed"):
        _dynamics(free_speed=[2.0, -1.0])
    with pytest.raises(ValueError, match="wave_speed"):
        _dynamics(wave_speed=[0.0, math.inf])
    with pytest.raises(ValueError, match="jam_density"):
        _dynamics(jam_density=[-150.0, math.inf])
    with pytest.raises(ValueError, match="capacity"):
        _dynamics(capacity=[50.0, math.inf])
    with pytest.raises(ValueError, match="NaN"):
        _dynamics(capacity=[math.nan, 1.0])


def test_link_dynamics_inf_wave_iff_inf_jam():
    with pytest.raises(ValueError, match="infinite"):
        _dynamics(wave_speed=[math.inf, math.inf])  # inf wave over finite jam
    with pytest.raises(ValueError, match="infinite"):
        _dynamics(wave_speed=[1.0, 1.0])  # finite wave over inf jam


def test_link_dynamics_capacity_vs_apex_rule():
    # link 0 apex = 2*1*150/3 = 100: at the apex is fine, above raises (G4)
    assert _dynamics(capacity=[100.0, 1.0]).capacity[0] == 100.0
    with pytest.raises(ValueError, match="apex"):
        _dynamics(capacity=[101.0, 1.0])


def test_link_dynamics_fd_is_canonical():
    dyn = _dynamics()
    capped = dyn.fd(0)  # capacity 50 < apex 100 => trapezoidal cap
    assert capped == TriangularFD(vf=2.0, w=1.0, kappa=150.0, q_cap=50.0)
    assert capped.capacity == 50.0
    pq = dyn.fd(1)  # point queue: q_cap IS the capacity
    assert pq == TriangularFD(vf=1.0, w=math.inf, kappa=math.inf, q_cap=1.0)
    assert pq.capacity == 1.0
    # capacity at the geometric apex builds the UNCAPPED triangle (q_cap=None)
    uncapped = _dynamics(capacity=[100.0, 1.0]).fd(0)
    assert uncapped == TriangularFD(vf=2.0, w=1.0, kappa=150.0)
    assert uncapped.q_cap is None
