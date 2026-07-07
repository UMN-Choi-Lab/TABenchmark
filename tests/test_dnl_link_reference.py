"""Link-component tests: interp_curve, LinkModel base machinery, the
test-only PointQueueLink reference (driven by hand, no loader), and the
DNLOutput artifact helpers."""

import math

import numpy as np
import pytest

from tabench.dnl._reference import PointQueueLink
from tabench.dnl.fd import TriangularFD
from tabench.dnl.grid import TimeGrid
from tabench.dnl.link import LinkModel, LinkModelFactory, interp_curve
from tabench.dnl.output import DNLOutput

# ---------------------------------------------------------------------------
# interp_curve
# ---------------------------------------------------------------------------


def test_interp_curve_values():
    curve = np.array([0.0, 1.0, 3.0])  # edges at t = 0, 0.5, 1.0
    dt = 0.5
    assert interp_curve(curve, -0.1, dt) == 0.0  # before the clock starts
    assert interp_curve(curve, 0.0, dt) == 0.0
    assert interp_curve(curve, 0.25, dt) == 0.5  # mid-segment, first piece
    assert interp_curve(curve, 0.5, dt) == 1.0  # exact interior edge
    assert interp_curve(curve, 0.75, dt) == 2.0  # mid-segment, second piece
    assert interp_curve(curve, 1.0, dt) == 3.0  # last edge
    assert interp_curve(curve, 5.0, dt) == 3.0  # beyond the end: constant


# ---------------------------------------------------------------------------
# PointQueueLink — driven BY HAND (no loader)
# ---------------------------------------------------------------------------


def _pq_link(q_cap=1.0, n_steps=8):
    fd = TriangularFD(vf=1.0, w=math.inf, kappa=math.inf, q_cap=q_cap)
    return PointQueueLink(fd, 1.0, TimeGrid(dt=0.5, n_steps=n_steps))


def test_point_queue_rejects_finite_jam_density():
    fd = TriangularFD(vf=1.0, w=1.0 / 3.0, kappa=180.0)
    with pytest.raises(ValueError, match="jam_density"):
        PointQueueLink(fd, 1.0, TimeGrid(dt=0.5, n_steps=4))


def test_point_queue_matches_link_model_factory_signature():
    factory: LinkModelFactory = PointQueueLink
    link = factory(TriangularFD(1.0, math.inf, math.inf, q_cap=1.0), 1.0, TimeGrid(0.5, 4))
    assert isinstance(link, LinkModel)
    assert np.array_equal(link.cumulative_in, np.zeros(5))  # N[0] = 0, all edges


def test_point_queue_scripted_episode():
    """Hand-checked 8-step episode (dt=0.5, L/vf=1 -> 2-step lag, q_max*dt=0.5):
    inflow 0.5/step for 4 steps; head node blocked through step 2, then serves.
    Exercises empty (S=0), uncapped, cap-binding, drain, and empty-again."""
    link = _pq_link()
    inflow = [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
    outflow = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0]
    expected_s = [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0]
    cap = link.fd.capacity * link.grid.dt
    for k in range(8):
        s = link.sending(k)
        assert s == link.sending(k)  # pure: no mutation on evaluation
        assert s == pytest.approx(expected_s[k], abs=1e-12)
        assert 0.0 <= s <= cap + 1e-12  # sending <= q_max*dt always
        assert link.receiving(k) == cap  # unbounded storage: R constant
        link.advance(k, inflow[k], outflow[k])
    assert np.allclose(link.cumulative_in, [0, 0.5, 1, 1.5, 2, 2, 2, 2, 2], atol=1e-12)
    assert np.allclose(link.cumulative_out, [0, 0, 0, 0, 0.5, 1, 1.5, 2, 2], atol=1e-12)
    # the cap genuinely bound at k = 3: eligible backlog exceeded one step of capacity
    assert interp_curve(link.cumulative_in, 1.0, 0.5) - link.cumulative_out[3] > cap


def test_point_queue_free_flow_translation_identity():
    """With inflow rate <= q_max at all times and outflow = sending(k), the
    discrete solution is the exact translation n_out(t) = n_in(t - L/vf)."""
    link = _pq_link()
    lag_steps = 2  # (L/vf) / dt
    for k in range(8):
        inflow = 0.25 if k < 4 else 0.0
        link.advance(k, inflow, link.sending(k))
    n_in, n_out = link.cumulative_in, link.cumulative_out
    assert np.array_equal(n_out[:lag_steps], np.zeros(lag_steps))
    assert np.array_equal(n_out[lag_steps:], n_in[:-lag_steps])


def test_advance_commits_curves_and_state_hook_is_noop():
    calls = []

    class _Recorder(PointQueueLink):
        def _advance_state(self, k, inflow, outflow):
            calls.append((k, inflow, outflow))

    link = _recorder = _Recorder(
        TriangularFD(1.0, math.inf, math.inf, q_cap=1.0), 1.0, TimeGrid(0.5, 4)
    )
    _recorder.advance(0, 0.5, 0.0)
    assert calls == [(0, 0.5, 0.0)]
    assert link.n_in[1] == 0.5 and link.n_out[1] == 0.0
    # the base hook itself is a no-op returning None
    plain = _pq_link()
    assert plain._advance_state(0, 0.1, 0.0) is None


def test_advance_step_bounds_and_debug_preconditions():
    link = _pq_link(n_steps=4)
    with pytest.raises(ValueError, match="step"):
        link.advance(4, 0.0, 0.0)
    with pytest.raises(ValueError, match="step"):
        link.advance(-1, 0.0, 0.0)
    with pytest.raises(AssertionError):
        link.advance(0, 0.6, 0.0)  # inflow > receiving = 0.5
    with pytest.raises(AssertionError):
        link.advance(0, -0.1, 0.0)
    with pytest.raises(AssertionError):
        link.advance(0, 0.0, 0.1)  # outflow > sending = 0 (empty link)


def test_cumulative_properties_are_copies():
    link = _pq_link()
    emitted = link.cumulative_in
    emitted[0] = 99.0
    assert link.n_in[0] == 0.0
    emitted_out = link.cumulative_out
    emitted_out[0] = 99.0
    assert link.n_out[0] == 0.0


# ---------------------------------------------------------------------------
# DNLOutput — artifact shape contract and derived helpers
# ---------------------------------------------------------------------------


def _anchor1_output() -> DNLOutput:
    """Anchor-1 closed forms, written down directly (no loader): 1 link,
    dt=0.5, K=24; N_in(t) = 0.5*min(t, 10), N_out(t) = N_in(t - 1)."""
    grid = TimeGrid(dt=0.5, n_steps=24)
    t = grid.edges
    n_in = 0.5 * np.minimum(t, 10.0)
    n_out = 0.5 * np.minimum(np.maximum(t - 1.0, 0.0), 10.0)
    return DNLOutput(
        scenario_hash="a" * 64,
        grid=grid,
        n_in=n_in[None, :],
        n_out=n_out[None, :],
        origin_release=np.vstack([n_in, np.zeros_like(n_in)]),
    )


def test_output_shape_validation_raises():
    grid = TimeGrid(dt=0.5, n_steps=4)
    good = np.zeros((1, 5))
    with pytest.raises(ValueError, match="shape"):
        DNLOutput("h", grid, np.zeros((1, 4)), good, np.zeros((2, 5)))
    with pytest.raises(ValueError, match="shape"):
        DNLOutput("h", grid, good, good, np.zeros((2, 4)))
    with pytest.raises(ValueError, match="shapes differ"):
        DNLOutput("h", grid, good, np.zeros((2, 5)), np.zeros((2, 5)))
    with pytest.raises(ValueError, match="shape"):
        DNLOutput("h", grid, np.zeros(5), np.zeros(5), np.zeros((2, 5)))  # 1-D


def test_output_link_storage_travel_time_and_tstt():
    out = _anchor1_output()
    storage = out.link_storage()
    assert storage.shape == (1, 25)
    assert storage[0, 4] == 0.5  # steady state: 0.5 veh on the link
    assert storage[0, 24] == 0.0  # cleared
    tt = out.travel_time(0)
    assert math.isnan(tt[0])  # level 0: no vehicle
    assert np.allclose(tt[1:21], 1.0, atol=1e-12)  # free-flow time L/vf = 1
    assert np.all(np.isnan(tt[21:]))  # plateau-repeated levels
    assert out.tstt() == pytest.approx(5.0, abs=1e-12)  # anchor-1 pinned TSTT


def test_output_travel_time_unreported_when_level_never_exits():
    grid = TimeGrid(dt=0.5, n_steps=4)
    n_in = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])
    n_out = np.array([[0.0, 0.0, 1.0, 2.0, 3.0]])
    out = DNLOutput("h", grid, n_in, n_out, np.zeros((1, 5)))
    tt = out.travel_time(0)
    assert tt[1] == pytest.approx(0.5)  # level 1 enters at 0.5, exits at 1.0
    assert math.isnan(tt[4])  # level 4 never exits in-horizon: unreported


def test_output_npz_round_trip(tmp_path):
    out = _anchor1_output()
    path = tmp_path / "anchor1.npz"
    out.save_npz(path)
    loaded = DNLOutput.load_npz(path)
    assert loaded.scenario_hash == out.scenario_hash
    assert loaded.loader_version == out.loader_version
    assert loaded.grid == out.grid
    assert np.array_equal(loaded.n_in, out.n_in)
    assert np.array_equal(loaded.n_out, out.n_out)
    assert np.array_equal(loaded.origin_release, out.origin_release)


def test_output_npz_missing_key_raises(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, n_in=np.zeros((1, 5)))
    with pytest.raises(ValueError, match="missing keys"):
        DNLOutput.load_npz(path)
