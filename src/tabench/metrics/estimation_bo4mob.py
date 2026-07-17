"""Harness-side certification of emitted BO4Mob OD vectors (D2 observational; adr-041).

The BO4Mob T2 analogue of :class:`~tabench.metrics.estimation.ODCertifier`, but a
**different scored object** (adr-036 R11's D2 boundary): there is no true OD and
no ``bfw``-certifiable BPR network, so nothing is compared to a planted truth and
**equilibrium is never claimed**. Instead the certifier re-runs the pinned
``eclipse-sumo`` engine ONCE on the emitted OD vector (od2trips ``--spread.uniform``
+ the fixed ``routes_single`` table + a mesoscopic ``sumo`` run — BO4Mob's own
single-run pipeline, promoted from the stage-1 smoke test per adr-034 Decision 2)
and scores the resulting link counts against real Caltrans PeMS panels:

* ``obs_nrmse`` — the in-sample count NRMSE on the TRAIN anchor date (the honesty
  diff a self-report is measured against), and
* ``heldout_nrmse`` — **the ranking column** — the MEAN over a temporally-disjoint,
  same-hour held-out set of dates of BO4Mob's own count NRMSE (framing b: ONE meso
  run per certify; only the pure-numpy ground-truth comparison varies per date).

BO4Mob is the lab's OWN benchmark, hosted here as scenarios/tasks/certificates —
never as validation of TABench methods. This D2 held-out NRMSE is NOT comparable to
the static/dynamic T2 ``heldout_count_rmse`` scale (it is BO4Mob's n-scaled NRMSE, a
same-lab scenario), and it does NOT reproduce BO4Mob's own SPSA/BO leaderboard
rankings (adr-034 forbidden clause 3).

**Subprocess discipline (adr-027/029 verbatim).** Binary discovery via
``sumo.SUMO_HOME`` ONLY (never PATH, never the ambient ``$SUMO_HOME`` — the box's
ambient value points at a stale non-existent path); ``stdin=DEVNULL``; ONE wall
deadline threaded across both subprocesses; a tempdir per run with a
``finally``-guaranteed cleanup; ``rc`` is NEVER trusted (success is DEFINED by the
read-back of the produced artifact); an engine crash / timeout / missing-or-garbage
artifact after ``rc=0`` RAISES ``RuntimeError`` that PROPAGATES — it is
infrastructure, NEVER laundered to ``od_feasible=0``. ``od_feasible=0`` is reserved
for a well-formed OD that fails the certificate's OWN validity gates (shape /
finite / non-negativity). A zero OD is **not** censored (a legitimate, terrible
estimate that certifies with catastrophic-but-finite NRMSE).

**Engine pin.** :func:`~tabench.edoc.replay.assert_engine_pin` (reused verbatim,
the tiny G0 helper) checks the ``eclipse-sumo`` actually installed on this box
against the instance's pin at certify time and RAISES ``ValueError`` on a mismatch
— never silently scoring under a different engine.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..data.bo4mob import (
    bo4mob_nrmse,
    edgedata_counts,
    fill_od_from_vector,
    fix_routes_single,
    local_edgedata_additional,
)
from ..edoc.replay import assert_engine_pin

__all__ = ["Bo4MobODCertifier", "BO4MOB_METRIC_KEYS"]

BO4MOB_METRIC_KEYS = (
    "od_feasible",
    "obs_nrmse",
    "heldout_nrmse",
    "heldout_nrmse_min",
    "heldout_nrmse_max",
    "n_heldout_dates",
)

_CLIP_TOL = 1e-9
# Default wall deadline threaded across the certify's od2trips + meso pair. 4smallRegion's
# meso alone measures ~126 s (adr-041 pilot), so the default clears the slow instance;
# the fast three (1ramp 0.4 s, 2corridor 9 s, 3junction 14 s) never approach it.
_DEFAULT_WALL_SECONDS = 300.0

# A counts producer: emitted OD vector -> {edge_id: count}. Injected so the certifier's
# control flow (censor / raise / score) is testable engine-free (the edoc ReplayRunner
# injection precedent). A real crash is a runner that raises RuntimeError, which PROPAGATES.
CountsRunner = Callable[[np.ndarray], dict[str, float]]


class Bo4MobODCertifier:
    """Pinned-engine held-out-count scorer for one BO4Mob T2 task. Reuse across checkpoints.

    ``train_sensor`` is the anchor GT CSV; ``heldout_sensors`` maps held-out date ->
    GT CSV path (fetched into the certifier's closure ONLY — never exposed to the
    estimator, P7). ``paths`` are the instance's static engine inputs (net / taz / od
    template / additional / routes_single). When ``runner`` is injected, ``paths`` may
    be ``None`` (engine-free tests).
    """

    def __init__(
        self,
        instance_key: str,
        pairs: tuple[tuple[str, str], ...],
        train_sensor: str | Path,
        heldout_sensors: Mapping[str, str | Path],
        paths: Mapping[str, Path] | None,
        od_end_time: int,
        sim_end_time: float,
        sensor_start_time: float,
        sensor_end_time: float,
        engine_version: str,
        certificate: Mapping[str, Any] | None = None,
        identifiability: Mapping[str, Any] | None = None,
        installed_version: str | None = None,
        runner: CountsRunner | None = None,
    ) -> None:
        self.instance_key = instance_key
        self.pairs = tuple(pairs)
        self.train_sensor = Path(train_sensor)
        self.heldout_sensors = {str(d): Path(p) for d, p in heldout_sensors.items()}
        self.paths = paths
        self.od_end_time = int(od_end_time)
        self.sim_end_time = float(sim_end_time)
        self.sensor_start_time = float(sensor_start_time)
        self.sensor_end_time = float(sensor_end_time)
        self.engine_version = str(engine_version)
        self.certificate = dict(certificate or {})
        self.identifiability = dict(identifiability or {})
        self._installed_override = installed_version
        self._runner = runner
        self._n_heldout = len(self.heldout_sensors)

    # -- engine version (read at certify time; overridable for engine-free tests) --
    def _installed_version(self) -> str:
        if self._installed_override is not None:
            return self._installed_override
        import importlib.metadata as im

        return im.version("eclipse-sumo")

    def _censored(self) -> dict[str, float]:
        metrics = {key: float("nan") for key in BO4MOB_METRIC_KEYS}
        metrics["od_feasible"] = 0.0
        metrics["n_heldout_dates"] = float(self._n_heldout)
        return metrics

    def certify(self, od_vector: np.ndarray) -> dict[str, float]:
        """Certified metric dict for one emitted OD vector over ``self.pairs``."""
        q = np.asarray(od_vector, dtype=np.float64)
        if q.shape != (len(self.pairs),):
            # A wrong-shaped emission is a wrapper programming error (mirrors
            # ODCertifier.certify's shape RAISE) — not a censor.
            raise ValueError(
                f"OD estimate shape {q.shape} != ({len(self.pairs)},) for the pair layout"
            )
        if not np.all(np.isfinite(q)):
            return self._censored()
        scale = max(1.0, float(np.abs(q).max(initial=0.0)))
        if q.min() < -_CLIP_TOL * scale:
            return self._censored()
        q = np.maximum(q, 0.0)

        # Engine pin: read the installed version and RAISE on mismatch (never a
        # censor). Checked before any scoring so a feasible certification always
        # runs under the pinned engine, whether or not it short-circuits below.
        assert_engine_pin(self._installed_version(), self.engine_version)

        # Crash-vs-censor (adr-041 ruling 7): an engine crash/timeout/read-back
        # failure inside our OWN pipeline RAISES RuntimeError and propagates.
        counts = self._counts(q)

        obs = bo4mob_nrmse(self.train_sensor, counts)
        per_date = [
            bo4mob_nrmse(path, counts)
            for _date, path in sorted(self.heldout_sensors.items())
        ]
        heldout = np.asarray(per_date, dtype=np.float64)
        metrics: dict[str, float] = {
            "od_feasible": 1.0,
            "obs_nrmse": float(obs),
            "heldout_nrmse": float(heldout.mean()) if heldout.size else float("nan"),
            "heldout_nrmse_min": float(heldout.min()) if heldout.size else float("nan"),
            "heldout_nrmse_max": float(heldout.max()) if heldout.size else float("nan"),
            "n_heldout_dates": float(self._n_heldout),
        }
        return metrics

    def per_date_heldout_nrmse(self, od_vector: np.ndarray) -> dict[str, float]:
        """The per-held-out-date NRMSE for one OD vector (manifest transparency).

        Runs the SAME single pipeline as :meth:`certify` and reports each date's
        NRMSE, so the aggregated ``heldout_nrmse`` is never a black box. Raises the
        same infrastructure errors; assumes a feasible (finite, non-negative) OD.
        """
        q = np.maximum(np.asarray(od_vector, dtype=np.float64), 0.0)
        assert_engine_pin(self._installed_version(), self.engine_version)
        counts = self._counts(q)
        return {
            date: float(bo4mob_nrmse(path, counts))
            for date, path in sorted(self.heldout_sensors.items())
        }

    # -- counts producer -------------------------------------------------------
    def _counts(self, q: np.ndarray) -> dict[str, float]:
        if self._runner is not None:
            return self._runner(q)
        if float(q.sum()) <= 0.0:
            # Zero-demand fast path (adr-027): no trips, so the simulated counts
            # are empty; bo4mob_nrmse fills absent sensors with 0 -> a catastrophic
            # but FINITE NRMSE. Not censored, not an engine call.
            return {}
        return self._run_pipeline(q)

    def _run_pipeline(self, q: np.ndarray) -> dict[str, float]:
        """od2trips + route-fix + mesoscopic SUMO on ``q``; returns per-edge counts.

        RAISES ``RuntimeError`` on any engine crash, timeout, or read-back failure
        (infrastructure — never ``od_feasible=0``). The binary is discovered via
        ``sumo.SUMO_HOME`` ONLY (never PATH/ambient), imported lazily so this module
        loads on a core (sumo-free) install.
        """
        if self.paths is None:
            raise RuntimeError(
                "Bo4MobODCertifier has no instance paths and no injected runner; "
                "cannot run the engine pipeline"
            )
        import sumo  # lazy: the module imports without the wheel

        sumo_home = sumo.SUMO_HOME
        env = {**os.environ, "SUMO_HOME": sumo_home}

        def sbin(name: str) -> str:
            return os.path.join(sumo_home, "bin", name)

        wall = float(self.certificate.get("wall_deadline_seconds", _DEFAULT_WALL_SECONDS))
        deadline = time.monotonic() + wall
        seed = int(self.certificate.get("seed", 0))
        tmpdir = tempfile.mkdtemp(prefix="bo4mob-certify-")
        work = Path(tmpdir)

        def run(cmd: list[str], produces: Path) -> None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"bo4mob certify exceeded the wall budget before {cmd[0]}"
                )
            try:
                proc = subprocess.run(
                    cmd, env=env, cwd=work, stdin=subprocess.DEVNULL,
                    capture_output=True, text=True, timeout=remaining,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"{cmd[0]} exceeded the wall budget") from exc
            # rc is NOT trusted: success is DEFINED by the read-back of `produces`
            # (a future engine exiting 0 without writing must not pass silently).
            if proc.returncode != 0:
                raise RuntimeError(
                    f"{cmd[0]} failed (rc={proc.returncode}): {proc.stderr[-800:]}"
                )
            if not produces.exists():
                raise RuntimeError(
                    f"{cmd[0]} produced no {produces.name} despite rc=0 (read-back failed)"
                )

        try:
            od_filled = work / "od_filled.xml"
            fill_od_from_vector(self.paths["od"], self.pairs, q, od_filled, self.od_end_time)
            trips_before = work / "trips_before.xml"
            run(
                [
                    sbin("od2trips"), "--spread.uniform",
                    "--taz-files", str(self.paths["taz"]),
                    "--tazrelation-files", str(od_filled), "-o", str(trips_before),
                ],
                trips_before,
            )
            trips_fixed = work / "trips_fixed.xml"
            fix_routes_single(trips_before, self.paths["routes_single"], trips_fixed)
            edge_data_name = "edge_data.xml"
            add_local = work / "additional_local.xml"
            local_edgedata_additional(self.paths["additional"], add_local, edge_data_name)
            edge_data = work / edge_data_name
            run(
                [
                    sbin("sumo"), "--mesosim", "true",
                    "--net-file", str(self.paths["net"]), "--routes", str(trips_fixed),
                    "-b", "0", "-e", str(int(self.sim_end_time)),
                    "--additional-files", str(add_local), "--ignore-route-errors", "true",
                    "--xml-validation", "never", "--no-warnings", "--seed", str(seed),
                ],
                edge_data,
            )
            try:
                return edgedata_counts(edge_data, self.sensor_start_time, self.sensor_end_time)
            except ET.ParseError as exc:
                raise RuntimeError(
                    f"bo4mob certify: {edge_data.name} unparseable after rc=0 ({exc})"
                ) from exc
        finally:  # tempdir-per-run, always cleaned (adr-027 discipline)
            shutil.rmtree(tmpdir, ignore_errors=True)
