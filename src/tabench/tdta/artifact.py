"""TDPathFlows: the P1 artifact a time-dependent route-choice model emits (adr-031).

The Peeta & Mahmassani (1995) decision variable is the per-departure-interval
path split ``r_ijk^tau`` — with departure times FIXED, the ONLY thing the model
chooses. So this artifact carries that and nothing else: cumulative link curves,
experienced path times, and total system travel time are all CONSEQUENCES that
:class:`~tabench.metrics.tdta_gaps.TDTAEvaluator` recomputes by running the
harness's own dynamic network loading of the emitted departures (the ADR-022
lesson, sharpened — there are no emitted curves for the flows to be inconsistent
with, so the whole aggregate-observability attack class of ADR-010's C8 limit is
structurally absent).

``departures[p, k]`` = vehicles assigned to declared path ``p`` departing during
grid step ``k`` (over ``[t_k, t_{k+1})``); ``scenario_hash`` binds the emission
to one :class:`~tabench.tdta.scenario.TDTAScenario` (a mismatch is censored by
the evaluator, not raised). Construction validates SHAPES only — garbage values
must be representable so the evaluator can censor them, never crash (the
DNLOutput convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["TDPathFlows"]

_NPZ_KEYS = ("scenario_hash", "solver_version", "departures")


@dataclass(frozen=True)
class TDPathFlows:
    """Emitted per-path, per-departure-interval flow plan (P1, decisions only).

    ``departures`` is ``(n_paths, K)`` float64, aligned with the scenario's
    declared ``paths`` tuple and its ``grid.n_steps``. ``solver_version`` is
    provenance only, never scored; ``provenance`` is a free-form dict the
    reference solver may attach (self-reported iterate counts, gaps, an honesty
    :class:`~tabench.dnl.output.DNLOutput`), never certified.
    """

    scenario_hash: str
    departures: np.ndarray  # (n_paths, K) vehicles on path p departing in step k
    solver_version: str = field(default="tdta-msa-v1")
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_hash, str):
            raise ValueError("TDPathFlows scenario_hash must be a str")
        if not isinstance(self.solver_version, str):
            raise ValueError("TDPathFlows solver_version must be a str")
        arr = np.ascontiguousarray(np.asarray(self.departures, dtype=np.float64))
        object.__setattr__(self, "departures", arr)
        if arr.ndim != 2:
            raise ValueError(
                f"TDPathFlows departures must be 2-D (n_paths, K), got shape {arr.shape}"
            )

    @property
    def n_paths(self) -> int:
        return int(self.departures.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.departures.shape[1])

    def cumulative(self) -> np.ndarray:
        """``(n_paths, K+1)`` cumulative emitted departures at grid edges, with a
        leading zero column (``D_p(t_0) = 0``). The evaluator's demand-match gate
        and per-path origin queues read this."""
        cum = np.zeros((self.departures.shape[0], self.departures.shape[1] + 1))
        np.cumsum(self.departures, axis=1, out=cum[:, 1:])
        return cum

    # -- npz round-trip (mirrors DNLOutput.save_npz so cross-process adapters can
    #    emit the artifact without importing this class)

    def save_npz(self, path: str | Path) -> None:
        """Write the artifact to ``path``: ``scenario_hash`` (str),
        ``solver_version`` (str), ``departures`` (float64 array)."""
        np.savez(
            Path(path),
            scenario_hash=self.scenario_hash,
            solver_version=self.solver_version,
            departures=self.departures,
        )

    @staticmethod
    def load_npz(path: str | Path) -> TDPathFlows:
        """Load and validate an artifact written by :meth:`save_npz`. Missing
        keys and wrong shapes raise (programming errors, not solution
        properties)."""
        with np.load(Path(path), allow_pickle=False) as npz:
            missing = [key for key in _NPZ_KEYS if key not in npz.files]
            if missing:
                raise ValueError(f"TDPathFlows.load_npz: missing keys {missing}")
            return TDPathFlows(
                scenario_hash=str(npz["scenario_hash"]),
                departures=npz["departures"],
                solver_version=str(npz["solver_version"]),
            )
