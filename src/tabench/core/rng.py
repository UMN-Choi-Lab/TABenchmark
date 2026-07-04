"""Reproducible randomness with a fixed spawn-key schema (P8).

SimOpt uses MRG32k3a streams indexed (stream, substream, subsubstream) =
(macroreplication, randomness source, replication). TABenchmark realizes the
same design with NumPy's counter-based Philox generator seeded through
``SeedSequence`` spawn keys ``(macrorep, source, replication)``: streams are
statistically independent by construction, order-independent, and identical
across machines and runs for the same root seed.

Reserved source ids (documented, never used by models):

* ``SOURCE_OBSERVATION`` — observation-process generation (data levels)
* ``SOURCE_EVALUATION`` — harness post-evaluations of stochastic responses
  (e.g. held-out sensor counts on the T2 estimation track)
* ``SOURCE_BOOTSTRAP`` — bootstrap confidence intervals
* ``SOURCE_PRIOR`` — prior/seed-matrix generation for T2 estimation
  (``StalePriorOD``); drawn independently of the observed counts so a stale
  prior never leaks the same noise realization the counts carry
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "RngBundle",
    "SOURCE_OBSERVATION",
    "SOURCE_EVALUATION",
    "SOURCE_BOOTSTRAP",
    "SOURCE_PRIOR",
]

SOURCE_OBSERVATION = 1_000_000
SOURCE_EVALUATION = 1_000_001
SOURCE_BOOTSTRAP = 1_000_002
SOURCE_PRIOR = 1_000_003


class RngBundle:
    """Factory of independent generators keyed by (macrorep, source, replication)."""

    def __init__(self, root_seed: int, macrorep: int = 0) -> None:
        self.root_seed = int(root_seed)
        self.macrorep = int(macrorep)

    def generator(self, source: int, replication: int = 0) -> np.random.Generator:
        """Return the deterministic generator for one randomness source.

        The same (root_seed, macrorep, source, replication) always yields an
        identical stream; distinct tuples yield independent streams.
        """
        ss = np.random.SeedSequence(
            entropy=self.root_seed,
            spawn_key=(self.macrorep, int(source), int(replication)),
        )
        return np.random.Generator(np.random.Philox(ss))

    def describe(self) -> dict[str, int]:
        """Provenance snapshot for manifests."""
        return {"root_seed": self.root_seed, "macrorep": self.macrorep}
