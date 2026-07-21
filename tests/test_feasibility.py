"""Unit tests for the shared negativity-clip kernel (``metrics/_feasibility.py``).

The five harness certifiers (road gaps, transit, static/dynamic/BO4Mob OD) share
ONE negativity rule through :func:`clip_negatives`; this file pins the boundary
(the load-bearing ``<`` at the threshold), the scale-invariance of the decision,
and the clip output, so a future drift of the consolidated kernel is caught here
rather than only via each certifier's own regression file.
"""

import numpy as np

from tabench.metrics._feasibility import _CLIP_TOL, clip_negatives


def test_clip_output_zeros_the_negative_dust():
    """A within-tolerance negative is clipped to exactly zero; positives untouched."""
    v = np.array([3.0, -1e-13, 0.0, 5.0])
    out = clip_negatives(v, scale=1.0)
    assert out is not None
    assert out[1] == 0.0
    assert np.array_equal(out, np.array([3.0, 0.0, 0.0, 5.0]))


def test_boundary_exactly_at_threshold_is_clipped_not_censored():
    """``<`` is load-bearing: an entry EXACTLY at ``-_CLIP_TOL * scale`` is clipped,
    not censored (the boundary convention every certifier shares)."""
    scale = 7.0
    v = np.array([1.0, -_CLIP_TOL * scale])  # v.min() == -_CLIP_TOL * scale exactly
    out = clip_negatives(v, scale)
    assert out is not None  # not censored
    assert out[1] == 0.0


def test_boundary_just_beyond_threshold_is_censored():
    """One float64 step past the threshold flips the decision to censor (``None``)."""
    scale = 7.0
    just_beyond = np.nextafter(-_CLIP_TOL * scale, -np.inf)  # slightly more negative
    v = np.array([1.0, just_beyond])
    assert clip_negatives(v, scale) is None


def test_scale_invariance_of_the_decision_and_output():
    """The tolerance is relative to ``scale``: scaling both the array and the scale
    by the same ``k > 0`` preserves the clip-vs-censor decision, and the clipped
    output scales with ``k``."""
    clip_case = np.array([2.0, -1e-10])  # within tol at scale 2.0 -> clipped
    censor_case = np.array([2.0, -3e-9])  # beyond tol at scale 2.0 -> censored
    for v in (clip_case, censor_case):
        base = clip_negatives(v, scale=2.0)
        for k in (0.5, 10.0, 1e6):
            scaled = clip_negatives(k * v, scale=k * 2.0)
            assert (base is None) == (scaled is None)
            if base is not None:
                assert np.allclose(scaled, k * base)
    # The two cases actually exercise both branches (guard against a silent
    # tolerance change that collapses them onto one side).
    assert clip_negatives(clip_case, scale=2.0) is not None
    assert clip_negatives(censor_case, scale=2.0) is None


def test_material_negative_is_censored():
    """A clearly-negative entry (well past the tolerance) is censored regardless."""
    assert clip_negatives(np.array([1.0, -0.5]), scale=1.0) is None
