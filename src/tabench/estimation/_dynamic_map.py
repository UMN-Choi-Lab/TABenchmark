"""Exogenous lagged assignment map for within-day dynamic ODME (ADR-023).

Cascetta, Inaudi & Marquis (1993) link time-sliced demand ``d_h`` (departures in
slice ``h``) to time-sliced link counts ``c_t`` through a lagged, **exogenous**
assignment map: a slice-``h`` trip crosses a link during a *later* interval
``t >= h`` set by the travel time to that link. The paper takes the map as given
(computed from a path-choice model and known/measured link travel times; valid on
uncongested networks or congested networks with known costs) — congestion
feedback ``M(d)`` is explicitly out of scope (that is ``cascetta2001fixed``).

This module realizes that map as a **frozen free-flow, two-interval-split** lag
tensor, a deterministic RNG-free function of the network free-flow times, the
slice length ``Delta``, and free-flow path choice:

* per pair ``(i, j)``, take the free-flow shortest path (link costs at ``v = 0``,
  tolls included — route choice is a cost decision);
* for each on-path link ``a`` the entry offset ``tau_a`` = sum of the **pure
  free-flow travel times** of the strictly-upstream on-path links (time from
  departure to entering ``a``; tolls are money, not minutes, so they never
  shift a crossing interval);
* departures are uniform over a slice, so writing ``tau_a = q*Delta + r`` with
  ``0 <= r < Delta`` splits a slice's crossings of ``a`` exactly two ways: a
  fraction ``1 - r/Delta`` at lag ``q`` and ``r/Delta`` at lag ``q + 1``.

The result is a **time-invariant** lag tensor ``M`` of shape
``(L + 1, n_links, n_pairs)`` with ``M[l, a, k]`` = the fraction of pair-``k``
demand that crosses link ``a`` exactly ``l`` slices after departing. Predicted
counts are ``c_t = sum_l M[l] @ d_{t-l}`` (zero for ``t < h`` by construction —
causality). The time-invariant form (``M_{h,t} = M[t-h]``, one crossing profile
reused for every departure slice) is a documented restriction of the paper's
general time-varying fractions ``m^a_ij(t, h)``, and the memory-safe
representation (``(L+1) x n_links x n_pairs`` rather than ``T x H x n_links x
n_pairs``); it is hashed under a ``v1`` recipe id so a time-varying variant is a
clean bump. See docs/design/adr-023-od-dynamic.md.
"""

from __future__ import annotations

import numpy as np

from ..core.scenario import Demand, Network
from ..models._paths import PathEngine

__all__ = [
    "lagged_assignment_tensor",
    "predict_interval_counts",
    "tensor_blocks",
    "stacked_tensor_map",
    "MAP_RECIPE",
]

MAP_RECIPE = "frozen_freeflow_v1"

# Split tolerance: a remainder within this (relative to Delta) of an interval
# boundary is treated as an exact integer lag, so tau = Delta lands entirely at
# lag q+1 with no spurious 1e-16 mass one lag further (keeps L tight).
_SPLIT_TOL = 1e-9


def lagged_assignment_tensor(
    network: Network,
    pairs: list[tuple[int, int]],
    slice_length: float,
    n_lags: int | None = None,
    engine: PathEngine | None = None,
) -> np.ndarray:
    """Frozen free-flow two-interval-split lag tensor ``M`` (``(L+1, n_links, P)``).

    ``M[l, a, k]`` is the fraction of pair-``pairs[k]`` demand crossing link ``a``
    exactly ``l`` slices after departure, under free-flow path choice and uniform
    within-slice departures. ``slice_length`` is ``Delta`` (native time units).
    ``L`` is the largest lag any path induces; pass ``n_lags`` to fix a larger
    horizon (padded with zeros) — a smaller one raises, since dropping a
    contribution would break the column-mass identity. Deterministic and RNG-free.
    """
    if not (np.isfinite(slice_length) and slice_length > 0):
        raise ValueError(f"slice_length must be finite and > 0, got {slice_length!r}")
    engine = engine or PathEngine(network)
    n_links = network.n_links
    n_pairs = len(pairs)
    # Path CHOICE uses the free-flow generalized cost (tolls included, as route
    # choice does); the crossing-time offsets tau accumulate the PURE free-flow
    # travel time — a toll is money, not minutes, so it must not shift which
    # interval a trip crosses a counter in (adr-023 review fix).
    fft = network.link_cost(np.zeros(n_links))
    travel_time = np.asarray(network.free_flow_time, dtype=np.float64)

    # Free-flow shortest path per pair (probe demand of 1 on each requested pair).
    probe = np.zeros((network.n_zones, network.n_zones), dtype=np.float64)
    for i, j in pairs:
        probe[i, j] = 1.0
    paths, _ = engine.shortest_paths(fft, Demand(probe))

    # Collect (lag, link, pair, fraction) contributions and the max lag first.
    contributions: list[tuple[int, int, int, float]] = []
    for k, pair in enumerate(pairs):
        links = paths.get(pair)
        if links is None:
            raise ValueError(f"pair {pair} has no free-flow path (disconnected)")
        tau = 0.0
        for a in np.asarray(links, dtype=np.int64):
            q = int(np.floor(tau / slice_length))
            r = tau - q * slice_length
            frac_next = r / slice_length
            if frac_next <= _SPLIT_TOL:
                contributions.append((q, int(a), k, 1.0))
            elif frac_next >= 1.0 - _SPLIT_TOL:
                contributions.append((q + 1, int(a), k, 1.0))
            else:
                contributions.append((q, int(a), k, 1.0 - frac_next))
                contributions.append((q + 1, int(a), k, frac_next))
            tau += float(travel_time[a])

    max_lag = max((c[0] for c in contributions), default=0)
    if n_lags is None:
        n_lags = max_lag
    elif n_lags < max_lag:
        raise ValueError(
            f"n_lags={n_lags} too small for this map: a path induces lag {max_lag} "
            "(a dropped crossing would break the column-mass identity)"
        )
    m = np.zeros((int(n_lags) + 1, n_links, n_pairs), dtype=np.float64)
    for lag, a, k, frac in contributions:
        m[lag, a, k] += frac
    return m


def predict_interval_counts(
    m: np.ndarray, profile: np.ndarray, n_intervals: int
) -> np.ndarray:
    """Expected crossing counts ``c_t = sum_l M[l] @ d_{t-l}`` (shape ``(T, R)``).

    ``m`` is ``(L+1, R, P)`` (``R`` = n_links or a sensor subset); ``profile`` is
    the ``(H, P)`` per-slice per-pair demand. Slices outside ``0..H-1`` contribute
    nothing (causality and horizon truncation), so counts are exact linear algebra.
    """
    l1, r, _p = m.shape
    n_slices = profile.shape[0]
    out = np.zeros((int(n_intervals), r), dtype=np.float64)
    for t in range(int(n_intervals)):
        for lag in range(l1):
            h = t - lag
            if 0 <= h < n_slices:
                out[t] += m[lag] @ profile[h]
    return out


def tensor_blocks(
    m: np.ndarray, n_slices: int, n_intervals: int
) -> list[dict[int, np.ndarray]]:
    """Time-invariant tensor -> per-interval block map ``blocks[t] = {h: M[t-h]}``.

    The general (possibly time-varying) representation the GLS solvers consume:
    ``blocks[t][h]`` is the ``(R, P)`` block mapping slice ``h`` demand to interval
    ``t`` counts. For the frozen tensor it is simply ``M[t-h]`` on the causal band.
    """
    l1 = m.shape[0]
    blocks: list[dict[int, np.ndarray]] = []
    for t in range(int(n_intervals)):
        row: dict[int, np.ndarray] = {}
        for h in range(int(n_slices)):
            lag = t - h
            if 0 <= lag < l1:
                row[h] = m[lag]
        blocks.append(row)
    return blocks


def stacked_tensor_map(m: np.ndarray, n_slices: int, n_intervals: int) -> np.ndarray:
    """Stacked observation map ``A`` (``(T*R, H*P)``) from a time-invariant tensor.

    Block ``(t, h)`` is ``M[t-h]`` on the causal band ``0 <= t-h <= L`` and zero
    elsewhere, so ``A`` is block-lower-banded. Row-major over ``(t, sensor)`` and
    ``(h, pair)`` — the layout the whitened simultaneous GLS and the
    identifiability rank test both use.
    """
    l1, r, p = m.shape
    n_slices = int(n_slices)
    n_intervals = int(n_intervals)
    a = np.zeros((n_intervals * r, n_slices * p), dtype=np.float64)
    for t in range(n_intervals):
        for h in range(n_slices):
            lag = t - h
            if 0 <= lag < l1:
                a[t * r:(t + 1) * r, h * p:(h + 1) * p] = m[lag]
    return a
