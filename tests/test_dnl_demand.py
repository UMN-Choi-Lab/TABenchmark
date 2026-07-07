"""DNL demand component tests: DynamicDemand cumulative exactness + validation,
TurningFractions row-stochastic validation and ordering canonicality (adr-010 layer 1)."""

import numpy as np
import pytest

from tabench.dnl.demand import DynamicDemand, TurningFractions


def two_period_demand() -> DynamicDemand:
    """Rates 1.5/0.5 (zone 1 -> 2) and 0.5/0.25 (zone 2 -> 1) on [0, 4) / [4, 10)."""
    return DynamicDemand(
        breakpoints=np.array([0.0, 4.0, 10.0]),
        rates=np.array(
            [
                [[0.0, 1.5], [0.5, 0.0]],
                [[0.0, 0.5], [0.25, 0.0]],
            ]
        ),
    )


# ---------------------------------------------------------------- DynamicDemand


def test_cumulative_closed_form_on_and_off_breakpoints():
    d = two_period_demand()
    t = np.array([0.0, 2.0, 4.0, 7.0, 10.0])
    out = d.cumulative(t)
    assert out.shape == (5, 2, 2)
    # OD 1->2: integral of 1.5 on [0,4) then 0.5 on [4,10)
    assert np.array_equal(out[:, 0, 1], [0.0, 3.0, 6.0, 7.5, 9.0])
    # OD 2->1: integral of 0.5 then 0.25
    assert np.array_equal(out[:, 1, 0], [0.0, 1.0, 2.0, 2.75, 3.5])
    # intrazonal stays identically zero
    assert np.array_equal(out[:, 0, 0], np.zeros(5))
    assert np.array_equal(out[:, 1, 1], np.zeros(5))


def test_cumulative_is_continuous_across_breakpoints():
    d = two_period_demand()
    eps = 1e-9
    below, at, above = d.cumulative(np.array([4.0 - eps, 4.0, 4.0 + eps]))
    assert np.allclose(below, at, atol=1e-8)
    assert np.allclose(above, at, atol=1e-8)


def test_cumulative_constant_after_last_breakpoint_and_zero_before_zero():
    d = two_period_demand()
    end = d.cumulative(np.array([10.0]))[0]
    for t in (10.5, 1e9, np.inf):
        assert np.array_equal(d.cumulative(np.array([t]))[0], end)
    assert np.array_equal(d.cumulative(np.array([-3.0, -np.inf])), np.zeros((2, 2, 2)))
    assert float(end.sum()) == d.total()


def test_cumulative_accepts_scalar_and_rejects_nan_and_2d():
    d = two_period_demand()
    assert d.cumulative(2.0).shape == (1, 2, 2)
    assert d.cumulative(np.float64(2.0))[0, 0, 1] == 3.0
    with pytest.raises(ValueError):
        d.cumulative(np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        d.cumulative(np.zeros((2, 2)))


def test_total_matches_hand_sum():
    d = two_period_demand()
    # 4*(1.5 + 0.5) + 6*(0.5 + 0.25) = 8 + 4.5
    assert d.total() == 12.5


def test_arrays_are_coerced_to_float64():
    d = DynamicDemand(breakpoints=[0, 1], rates=[[[0, 2], [1, 0]]])
    assert d.breakpoints.dtype == np.float64
    assert d.rates.dtype == np.float64
    assert d.n_zones == 2
    assert d.total() == 3.0


def test_rejects_nonzero_diagonal():
    with pytest.raises(ValueError, match="diagonal"):
        DynamicDemand(
            breakpoints=np.array([0.0, 1.0]),
            rates=np.array([[[0.1, 1.0], [1.0, 0.0]]]),
        )


def test_rejects_negative_rates():
    with pytest.raises(ValueError, match="nonnegative"):
        DynamicDemand(
            breakpoints=np.array([0.0, 1.0]),
            rates=np.array([[[0.0, -1.0], [1.0, 0.0]]]),
        )


def test_rejects_non_increasing_breakpoints():
    rates = np.array([[[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]])
    with pytest.raises(ValueError, match="strictly increasing"):
        DynamicDemand(breakpoints=np.array([0.0, 2.0, 2.0]), rates=rates)
    with pytest.raises(ValueError, match="strictly increasing"):
        DynamicDemand(breakpoints=np.array([0.0, 2.0, 1.0]), rates=rates)


def test_rejects_breakpoints_not_starting_at_zero():
    with pytest.raises(ValueError, match="start at 0.0"):
        DynamicDemand(
            breakpoints=np.array([1.0, 2.0]),
            rates=np.array([[[0.0, 1.0], [0.0, 0.0]]]),
        )


def test_rejects_malformed_shapes_and_nonfinite():
    ok_rates = np.array([[[0.0, 1.0], [0.0, 0.0]]])
    with pytest.raises(ValueError, match="P\\+1 >= 2"):
        DynamicDemand(breakpoints=np.array([0.0]), rates=ok_rates[:0])
    with pytest.raises(ValueError, match="periods"):
        DynamicDemand(breakpoints=np.array([0.0, 1.0, 2.0]), rates=ok_rates)
    with pytest.raises(ValueError, match="n_zones"):
        DynamicDemand(breakpoints=np.array([0.0, 1.0]), rates=np.zeros((1, 2, 3)))
    with pytest.raises(ValueError, match="n_zones"):
        DynamicDemand(breakpoints=np.array([0.0, 1.0]), rates=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        DynamicDemand(
            breakpoints=np.array([0.0, 1.0]),
            rates=np.array([[[0.0, np.inf], [0.0, 0.0]]]),
        )
    with pytest.raises(ValueError, match="finite"):
        DynamicDemand(breakpoints=np.array([0.0, np.inf]), rates=ok_rates)


# ------------------------------------------------------------- TurningFractions


def test_turning_fractions_valid_and_coerced():
    tf = TurningFractions(
        frac=(
            (3, [[0.25, 0.75]]),  # 1 in, 2 out
            (7, np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])),  # 2 in, 3 out
        )
    )
    assert [nid for nid, _ in tf.frac] == [3, 7]
    for _, m in tf.frac:
        assert m.dtype == np.float64
        assert np.allclose(m.sum(axis=1), 1.0, atol=1e-12)


def test_turning_fractions_rejects_non_row_stochastic():
    with pytest.raises(ValueError, match="sum to 1"):
        TurningFractions(frac=((3, np.array([[0.5, 0.4]])),))
    # nonnegativity checked before the (compensating) row sum
    with pytest.raises(ValueError, match="nonnegative"):
        TurningFractions(frac=((3, np.array([[1.5, -0.5]])),))
    with pytest.raises(ValueError, match="finite"):
        TurningFractions(frac=((3, np.array([[np.inf, 0.0]])),))


def test_turning_fractions_ordering_canonicality():
    m = np.array([[0.5, 0.5]])
    with pytest.raises(ValueError, match="strictly increasing"):
        TurningFractions(frac=((7, m), (3, m)))
    with pytest.raises(ValueError, match="strictly increasing"):
        TurningFractions(frac=((3, m), (3, m)))


def test_turning_fractions_rejects_empty_and_malformed():
    with pytest.raises(ValueError, match="turns=None"):
        TurningFractions(frac=())
    with pytest.raises(ValueError, match="2-D"):
        TurningFractions(frac=((3, np.array([0.5, 0.5])),))
    with pytest.raises(ValueError, match="2-D"):
        TurningFractions(frac=((3, np.zeros((0, 2))),))
