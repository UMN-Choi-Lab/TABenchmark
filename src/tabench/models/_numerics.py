"""Shared log-domain numerics for the logit-family models.

Two overflow-safe primitives had been hand-inlined at four model sites; this
module gives each a single home so the copies cannot silently drift apart
(exactly as :mod:`..metrics._feasibility` does for the negativity clip). Both
are pure and operate on 1-D float64 arrays.

* ``logsumexp`` — the max-shifted ``log sum exp`` over the *finite* entries of a
  vector, used for Sheffi's log-domain node-weight recursion ``b(j) = log W(j)``
  in BOTH the deterministic Dial-STOCH forward pass (:mod:`._stoch`) and its
  sampled Markov-chain twin (:mod:`.dtd_stochastic`). Its two inlined copies were
  byte-for-byte identical, and they MUST stay bit-identical: ``dtd-stochastic``'s
  multinomial means telescope to the deterministic recursion only if both share
  the SAME weight map (the model's ``E[v_n | p_n] = StochEngine.load``
  unbiasedness claim, adr-001). The finite-filter is load-bearing, not cosmetic:
  an all-``-inf`` incoming set (a node no efficient link reaches) must yield
  ``-inf``, whereas an unfiltered ``max`` of ``-inf`` would make ``terms - max``
  be ``nan`` and poison ``b``.

* ``softmax`` — the max-shifted logit choice map, used for the route-choice
  probabilities in :mod:`.dtd_cumlog` (bare, ``e / s``) and the numpy logit
  loader in :mod:`.implicit_ue` (demand-scaled). Its two inlined copies were NOT
  byte-for-byte: ``implicit_ue`` fused the per-OD demand INSIDE the division,
  ``(demand * e) / s``, while ``dtd_cumlog`` is bare, ``e / s``. Because float
  ``*`` / ``/`` do not associate, ``(demand * e) / s`` and ``demand * (e / s)``
  differ in the last bit. The optional ``scale`` argument exists precisely so the
  ``implicit_ue`` call keeps its original ``(demand * e) / s`` association bitwise
  while sharing this one implementation; ``dtd_cumlog`` calls it bare.

A fix to one copy's max-shift (or a dropped finite-filter) would silently diverge
the pinned Dial map from its sampled twin, or the two softmaxes; the single home
plus the import-identity guards in ``tests/test_numerics.py`` kill that drift.
"""

from __future__ import annotations

import numpy as np

__all__ = ["logsumexp", "softmax"]


def logsumexp(values: np.ndarray) -> float:
    """``log sum exp`` over the FINITE entries of ``values``, max-shifted.

    Non-finite (``-inf``) entries are dropped first, so a vector with no finite
    entry (or an empty vector) returns ``-inf`` — the identity element — rather
    than the ``nan`` an unfiltered ``max``-shift would produce. The max
    subtraction keeps ``exp`` from overflowing. Returns a Python ``float``.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("-inf")
    m = float(finite.max())
    return m + float(np.log(np.exp(finite - m).sum()))


def softmax(z: np.ndarray, scale: float | np.ndarray | None = None) -> np.ndarray:
    """Numerically stable softmax over a 1-D array (max-shift for overflow).

    With ``scale is None`` returns ``exp(z - max z) / sum(...)``, which sums to 1.
    With ``scale`` given, returns ``(scale * exp(z - max z)) / sum(...)`` — the
    scale is fused INSIDE the division, ``(scale * e) / s`` rather than
    ``scale * (e / s)``, so a caller loading demand over the softmax
    (:mod:`.implicit_ue`) keeps its original association bitwise (float ``*`` / ``/``
    do not associate). ``z`` is not mutated; requires ``z`` non-empty (every
    caller's OD group has at least one route), matching the inlined originals.
    """
    z = z - z.max()
    e = np.exp(z)
    s = e.sum()
    if scale is None:
        return e / s
    return (scale * e) / s
