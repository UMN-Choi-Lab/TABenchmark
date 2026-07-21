"""Shared negativity-clip feasibility kernel for the harness certifiers (P1, P7).

Every certifier — road gaps (:mod:`.gaps`), transit (:mod:`.transit_gaps`), and
static / dynamic / BO4Mob OD estimation (:mod:`.estimation`,
:mod:`.estimation_dynamic`, :mod:`.estimation_bo4mob`) — gates an emitted
flow/matrix through the SAME demand-blind negativity rule: an entry negative only
by floating-point dust is CLIPPED to zero (a solver's 1e-16 sliver is not a
violation), while an entry negative beyond a relative tolerance is a real
infeasibility and the caller CENSORS the emission. This rule had been copy-pasted
into five modules and drifted twice historically (see the scale-formula notes at
``estimation.py`` and ``estimation_dynamic.py``), so the single tolerance and the
exact comparison now live here once.

The *scale* the tolerance multiplies is deliberately NOT computed here: each
certificate scales differently — the whole-array max (road / transit / BO4Mob)
versus the OFF-DIAGONAL max (static / dynamic OD, where a huge intrazonal cell
must never buy negativity tolerance for an inter-zonal cell, adr-023), some with
an ``initial=0.0`` guarding a possibly-empty off-diagonal slice. The caller
computes its own ``scale`` and passes it in; this kernel owns only the invariant
part: the strict ``<`` comparison at ``-_CLIP_TOL * scale`` and the
``np.maximum(_, 0.0)`` clip. The ``<`` is load-bearing — an entry EXACTLY at
``-_CLIP_TOL * scale`` is clipped, not censored (the boundary convention every
certifier shares).
"""

from __future__ import annotations

import numpy as np

__all__ = ["clip_negatives"]

#: negative entries within this (relative-to-scale) tolerance are clipped as noise
_CLIP_TOL = 1e-9


def clip_negatives(array: np.ndarray, scale: float) -> np.ndarray | None:
    """Clip dust-negative entries to zero, or signal a censor.

    Returns ``np.maximum(array, 0.0)`` when the most-negative entry is within the
    relative clip tolerance (``array.min() >= -_CLIP_TOL * scale``); returns
    ``None`` — the caller's signal to censor — when any entry is negative beyond
    it. The comparison is strict (``<``): an entry exactly at ``-_CLIP_TOL *
    scale`` is clipped, not censored. ``scale`` is supplied by the caller because
    each certificate scales differently (see the module docstring).
    """
    if array.min() < -_CLIP_TOL * scale:
        return None
    return np.maximum(array, 0.0)
