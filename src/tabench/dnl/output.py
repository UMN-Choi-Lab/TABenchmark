"""DNLOutput: the P1 artifact a dynamic-loading run emits (dnl-core, adr-010).

Everything the harness needs to recompute every certificate lives here —
time-indexed cumulative counts at grid edges, float64, counts not rates. The
evaluator (``metrics/dnl_gaps.py``) is a pure function of
``(DynamicScenario bytes, these arrays)``; the derived helpers below are
convenience only and are recomputed independently on the harness side.

Construction validates SHAPES only (wrong shapes are programming errors and
raise); array VALUES are deliberately not validated — garbage curves must be
representable so the evaluator can censor them (C0), never crash on them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .grid import TimeGrid

__all__ = ["DNLOutput"]

_NPZ_KEYS = ("scenario_hash", "loader_version", "dt", "n_steps", "n_in", "n_out", "origin_release")


def _earliest_time(curve: np.ndarray, level: float, dt: float) -> float:
    """Earliest ``t`` with the piecewise-linear ``curve`` (sampled at grid
    edges) ``>= level``; ``+inf`` if the level is never reached in-horizon.
    Assumes a nondecreasing curve (C0-valid outputs)."""
    if curve[-1] < level:
        return math.inf
    j = int(np.searchsorted(curve, level, side="left"))
    if j == 0:
        return 0.0
    lo, hi = float(curve[j - 1]), float(curve[j])
    return dt * (j - 1 + (level - lo) / (hi - lo))


@dataclass(frozen=True)
class DNLOutput:
    """Everything the harness needs to recompute every certificate (P1).

    Time-indexed at grid EDGES; float64; counts, not rates. ``scenario_hash``
    binds the run to one :class:`~tabench.dnl.scenario.DynamicScenario`
    instance (a mismatch is censored by the evaluator, not raised);
    ``loader_version`` is provenance only, never scored.
    """

    scenario_hash: str
    grid: TimeGrid
    n_in: np.ndarray  # (n_links, K+1) cumulative link inflow;  [:, 0] = 0
    n_out: np.ndarray  # (n_links, K+1) cumulative link outflow; [:, 0] = 0
    origin_release: np.ndarray  # (n_zones, K+1) cumulative vehicles released
    loader_version: str = field(default="dnl-core-v1")

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_hash, str):
            raise ValueError("DNLOutput scenario_hash must be a str")
        if not isinstance(self.loader_version, str):
            raise ValueError("DNLOutput loader_version must be a str")
        edges = self.grid.n_steps + 1
        for name in ("n_in", "n_out", "origin_release"):
            arr = np.ascontiguousarray(np.asarray(getattr(self, name), dtype=np.float64))
            object.__setattr__(self, name, arr)
            if arr.ndim != 2 or arr.shape[1] != edges:
                raise ValueError(
                    f"DNLOutput {name} must have shape (n, K+1) = (n, {edges}), "
                    f"got {arr.shape}"
                )
        if self.n_in.shape != self.n_out.shape:
            raise ValueError(
                f"DNLOutput n_in and n_out shapes differ: "
                f"{self.n_in.shape} vs {self.n_out.shape}"
            )

    # -- derived helpers (convenience; the EVALUATOR recomputes independently)

    def link_storage(self) -> np.ndarray:
        """Vehicles on each link at each edge: ``n_in - n_out``, (n_links, K+1)."""
        return self.n_in - self.n_out

    def travel_time(self, a: int) -> np.ndarray:
        """``(K+1,)`` FIFO travel time of link ``a`` at the count levels
        ``n = n_in[a, k]``: ``te(n) - ti(n)`` via level-matched inverse linear
        interpolation of BOTH cumulative curves (earliest-time convention on
        plateaus; same formula as certificate C6). NaN where the level is 0
        (no vehicle has entered), where it repeats an earlier level (plateau),
        and where it never exits within the horizon."""
        dt = self.grid.dt
        cin = self.n_in[a]
        cout = self.n_out[a]
        tt = np.full(cin.shape[0], np.nan)
        for k in range(cin.shape[0]):
            level = float(cin[k])
            if level <= 0.0 or (k > 0 and level <= cin[k - 1]):
                continue
            te = _earliest_time(cout, level, dt)
            if math.isinf(te):
                continue
            tt[k] = te - _earliest_time(cin, level, dt)
        return tt

    def tstt(self) -> float:
        """Trapezoid integral of link storage over the horizon [veh*tu].

        NOTE: origin-queue time needs the scenario's demand curve, so the
        FULL total system travel time (incl. origin queues) is the
        evaluator's; this is the on-link part only.
        """
        storage = self.link_storage()
        return float(0.5 * self.grid.dt * (storage[:, :-1] + storage[:, 1:]).sum())

    # -- npz round-trip (exact keys per adr-010 so cross-process adapters
    #    can emit the artifact without importing this class)

    def save_npz(self, path: str | Path) -> None:
        """Write the artifact to ``path`` with the exact adr-010 field spec:
        ``scenario_hash`` (str), ``loader_version`` (str), ``dt`` (float64
        scalar), ``n_steps`` (int64 scalar), ``n_in``, ``n_out``,
        ``origin_release`` (float64 arrays)."""
        np.savez(
            Path(path),
            scenario_hash=self.scenario_hash,
            loader_version=self.loader_version,
            dt=np.float64(self.grid.dt),
            n_steps=np.int64(self.grid.n_steps),
            n_in=self.n_in,
            n_out=self.n_out,
            origin_release=self.origin_release,
        )

    @staticmethod
    def load_npz(path: str | Path) -> DNLOutput:
        """Load and validate an artifact written by :meth:`save_npz` (or a
        conforming cross-process adapter); reconstructs ``TimeGrid(dt,
        n_steps)``. Missing keys and shape mismatches raise (programming
        errors, not solution properties)."""
        with np.load(Path(path), allow_pickle=False) as npz:
            missing = [key for key in _NPZ_KEYS if key not in npz.files]
            if missing:
                raise ValueError(f"DNLOutput.load_npz: missing keys {missing}")
            grid = TimeGrid(dt=float(npz["dt"]), n_steps=int(npz["n_steps"]))
            return DNLOutput(
                scenario_hash=str(npz["scenario_hash"]),
                grid=grid,
                n_in=npz["n_in"],
                n_out=npz["n_out"],
                origin_release=npz["origin_release"],
                loader_version=str(npz["loader_version"]),
            )
