"""Seeded detector observation: degraded projections of the clean boundary
curves (P3, the ``observe/levels.py`` convention, module-local).

Ground truth (the two clean detector cumulative curves) is regenerated from the
hashed recipe; each observation derives a model-visible dataset from it
deterministically given the scenario seed. Three noise kinds:

* ``"none"`` — the exact clean curves repeated ``n_days`` times (the oracle /
  validity level; the min formula reproduces the reference exactly, so it is
  reported but NEVER ranked, mirroring adr-023's clean oracle row).
* ``"poisson"`` — per-interval Poisson counts (the ``LinkCounts`` precedent). The
  cumulative stays monotone (counts are nonnegative), a faithful low-count level.
* ``"gaussian"`` — a Gaussian cumulative READING error ``N(0, sigma^2)`` per edge
  (the ``DayToDayCounts`` large-population Gaussian limit, which likewise admits
  negative fluctuations). This is the level that DISCRIMINATES denoisers: a
  Poisson-increment cumulative is already monotone, so isotonic regression and a
  running-max both reduce to the identity and cannot separate the two reference
  estimators; the Gaussian reading error makes the observed cumulative
  non-monotone, so the L2-optimal isotonic fit strictly beats the running-max
  envelope (anchor A4).

``up_windows`` / ``dn_windows`` mark unobserved time windows (the missing-detector
dials): the observed cumulative is NaN there, and the reconstruction drops the
branch whose shifted source time lands inside a window (anchor A5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .scenario import ThreeDetectorScenario

__all__ = ["DetectorObservation", "observe_detectors"]


@dataclass(frozen=True)
class DetectorObservation:
    """One seeded detector dataset — the MODEL-VISIBLE projection of ground truth.

    Carries ONLY observables: the record grid, the per-day noisy cumulative
    detector curves, and the unobserved windows. It deliberately holds none of the
    truth recipe (demand, metering, reference field), so a submission built on it
    cannot regenerate the ground truth (the adr-023 information boundary).
    """

    scenario_hash: str
    times: np.ndarray  # (K+1,) grid edges
    up: np.ndarray  # (n_days, K+1) observed upstream cumulative (NaN in up_windows)
    dn: np.ndarray  # (n_days, K+1) observed downstream cumulative (NaN in dn_windows)
    up_windows: tuple[tuple[float, float], ...] = ()
    dn_windows: tuple[tuple[float, float], ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


def _window_mask(times: np.ndarray, windows: tuple[tuple[float, float], ...]) -> np.ndarray:
    mask = np.zeros(times.shape, dtype=bool)
    for t0, t1 in windows:
        mask |= (times >= t0) & (times <= t1)
    return mask


def _project(
    clean: np.ndarray, noise: str, sigma: float, drift: float, n_days: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``n_days`` observed cumulative curves from one clean cumulative."""
    k = clean.shape[0]
    if noise == "none":
        obs = np.tile(clean, (n_days, 1))
    elif noise == "poisson":
        incr = np.clip(np.diff(clean), 0.0, None)
        draws = rng.poisson(lam=np.tile(incr, (n_days, 1))).astype(np.float64)
        obs = np.zeros((n_days, k), dtype=np.float64)
        obs[:, 1:] = np.cumsum(draws, axis=1)
    else:  # gaussian cumulative reading error
        obs = clean[None, :] + rng.normal(0.0, sigma, size=(n_days, k))
        obs[:, 0] = 0.0  # the record starts empty and that start is known
    if drift != 0.0:
        # a systematic linear cumulative miscount (detector gain drift), zero at
        # the known empty start
        obs = obs + drift * np.arange(k, dtype=np.float64)[None, :]
        obs[:, 0] = 0.0
    return obs


def observe_detectors(scenario: ThreeDetectorScenario) -> DetectorObservation:
    """Project the clean detector curves into the seeded model-visible dataset."""
    times, n_up, n_dn = scenario.truth_boundary_curves()
    up_gen, dn_gen = (
        np.random.default_rng(s) for s in np.random.SeedSequence(scenario.seed).spawn(2)
    )
    args = (scenario.noise, scenario.read_sigma, scenario.drift, scenario.n_days)
    up = _project(n_up, *args, up_gen)
    dn = _project(n_dn, *args, dn_gen)
    if scenario.up_windows:
        up[:, _window_mask(times, scenario.up_windows)] = np.nan
    if scenario.dn_windows:
        dn[:, _window_mask(times, scenario.dn_windows)] = np.nan
    return DetectorObservation(
        scenario_hash=scenario.content_hash(),
        times=times.copy(),
        up=up,
        dn=dn,
        up_windows=scenario.up_windows,
        dn_windows=scenario.dn_windows,
        meta={
            "noise": scenario.noise,
            "read_sigma": scenario.read_sigma,
            "drift": scenario.drift,
            "n_days": scenario.n_days,
            "rankable": scenario.noise != "none",
        },
    )
