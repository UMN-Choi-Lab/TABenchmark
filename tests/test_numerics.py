"""Unit tests for the shared log-domain primitives (``models/_numerics.py``) plus
a drift guard that the four logit-family models import them rather than carrying a
re-inlined copy that could silently diverge (the hazard the module consolidates)."""

from __future__ import annotations

import numpy as np
import pytest

from tabench.models import _numerics
from tabench.models._numerics import logsumexp, softmax

# ------------------------------------------------------------------ logsumexp


def test_logsumexp_matches_reference() -> None:
    x = np.array([0.3, -1.2, 2.5, 0.0])
    assert logsumexp(x) == pytest.approx(float(np.log(np.exp(x).sum())))


def test_logsumexp_shift_invariance() -> None:
    x = np.array([1.0, 2.0, 3.0])
    for c in (-5.0, 10.0, 1e3):
        assert logsumexp(x + c) == pytest.approx(logsumexp(x) + c)


def test_logsumexp_single_element() -> None:
    assert logsumexp(np.array([4.25])) == pytest.approx(4.25)


def test_logsumexp_large_magnitude_stable() -> None:
    # exp(1000) overflows; the max-shift keeps logsumexp finite and exact.
    val = logsumexp(np.array([1000.0, 1000.0]))
    assert np.isfinite(val)
    assert val == pytest.approx(1000.0 + np.log(2.0))


def test_logsumexp_drops_neg_inf_entries() -> None:
    # -inf entries are dropped, so a finite/-inf mix equals the finite-only result
    # (the load-bearing filter: an all-(-inf) set must not poison b with nan).
    mixed = np.array([-np.inf, 2.0, -np.inf, 0.5])
    assert logsumexp(mixed) == pytest.approx(logsumexp(np.array([2.0, 0.5])))


def test_logsumexp_all_neg_inf_is_neg_inf() -> None:
    assert logsumexp(np.array([-np.inf, -np.inf])) == float("-inf")


def test_logsumexp_empty_is_neg_inf() -> None:
    assert logsumexp(np.array([])) == float("-inf")


# -------------------------------------------------------------------- softmax


def test_softmax_sums_to_one() -> None:
    p = softmax(np.array([0.1, -2.0, 3.5, 1.0]))
    assert p.sum() == pytest.approx(1.0)
    assert (p > 0).all()


def test_softmax_shift_invariance() -> None:
    z = np.array([1.0, 2.0, 3.0])
    for c in (-4.0, 7.0, 500.0):
        np.testing.assert_allclose(softmax(z + c), softmax(z))


def test_softmax_single_element() -> None:
    np.testing.assert_array_equal(softmax(np.array([9.9])), np.array([1.0]))


def test_softmax_large_magnitude_stable() -> None:
    # Without the max-shift exp(2000) overflows; the shift keeps it finite.
    p = softmax(np.array([2000.0, 2000.0, 1999.0]))
    assert np.isfinite(p).all()
    assert p.sum() == pytest.approx(1.0)
    assert p[0] == pytest.approx(p[1])


def test_softmax_matches_reference() -> None:
    z = np.array([0.5, -1.0, 2.0])
    e = np.exp(z - z.max())
    np.testing.assert_allclose(softmax(z), e / e.sum())


def test_softmax_does_not_mutate_input() -> None:
    z = np.array([1.0, 2.0, 3.0])
    z_copy = z.copy()
    softmax(z)
    np.testing.assert_array_equal(z, z_copy)


def test_softmax_fused_scale_matches_head_association() -> None:
    """softmax(z, scale=d) fuses the scale INSIDE the division -- (d*e)/s, the
    association implicit_ue's numpy logit loader used before the softmax was
    shared -- and is bitwise DIFFERENT from d*softmax(z) == d*(e/s) on a draw
    where float non-associativity bites. This pin catches any regression back to
    the wrong association (the confirmed B5-review MAJOR)."""
    rng = np.random.default_rng(20260721)
    for _ in range(1000):
        z = rng.normal(size=5) * 10.0
        d = float(rng.uniform(1.0, 1e4))
        e = np.exp(z - z.max())
        head = (d * e) / e.sum()  # implicit_ue's original (scale * e) / s
        if not np.array_equal(head, d * (e / e.sum())):
            break
    else:  # pragma: no cover - a differing draw exists for ~86% of draws
        raise AssertionError("no float-nonassociative draw found in 1000 tries")
    # fused-scale softmax reproduces HEAD's association bitwise ...
    np.testing.assert_array_equal(softmax(z, scale=d), head)
    # ... while d * softmax(z) uses the OTHER association and differs in the last bit.
    assert not np.array_equal(d * softmax(z), head)


# --------------------------------------------- import-identity drift guards
# A future local re-copy of either primitive would break the `is` identity and
# fail here, which is exactly the drift the shared module exists to prevent.


def test_stoch_imports_shared_logsumexp() -> None:
    from tabench.models import _stoch

    assert _stoch.logsumexp is _numerics.logsumexp


def test_dtd_stochastic_imports_shared_logsumexp() -> None:
    from tabench.models import dtd_stochastic

    assert dtd_stochastic.logsumexp is _numerics.logsumexp


def test_dtd_cumlog_imports_shared_softmax() -> None:
    from tabench.models import dtd_cumlog

    assert dtd_cumlog.softmax is _numerics.softmax


def test_implicit_ue_imports_shared_softmax() -> None:
    pytest.importorskip("torch")
    from tabench.models import implicit_ue

    assert implicit_ue.softmax is _numerics.softmax
