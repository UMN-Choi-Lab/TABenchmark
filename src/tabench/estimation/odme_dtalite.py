"""odme-dtalite: DTALite 0.8.1's static ODME wrapped as a GUARDED T2 estimator (adr-042).

DTALite's PyPI wheel hides an Origin-Destination Matrix Estimation routine
(``performODME``) *inside* the same static ``assignment()`` entry the ``dtalite-tap``
T1 adapter already wraps (TAPLite.cpp, called right after the Frank-Wolfe loop when
``settings.csv`` carries ``odme_mode=1``). It is a gradient-descent calibrator: given a
seed OD, a target OD, and sensor link counts (``obs_volume``), it re-weights the
route-flows the base assignment discovered so the modeled link volumes approach the
counts. There is NO separate ODME API -- ``assignment()`` alone triggers it.

**Guarded static estimator, ZERO certifier change (the adr-028 ideal).** This class
emits an OD matrix through ``ODResultBundle.final.od_matrix`` -- the SAME channel
``spsa``/``gls``/``spsa-sumo`` use -- and the EXISTING pinned-bfw ``ODCertifier`` scores
it unchanged, re-running its own independent ``bfw`` solve under the scenario's declared
BPR. No task/runner/certifier code is touched. The anti-laundering property holds by
construction: ``ODCertifier.certify()`` never imports or trusts DTALite; a buggy or
adversarial ODME run can at worst emit a *bad* OD that certifies honestly as a
no-improvement row -- it cannot forge a good certificate (pinned by the negative-control
test, the adr-028 poisson precedent).

**Marquee anchor (measured, siouxfalls, adr-042).** Sensors random cov=0.5, held-out
cov=0.2, clean counts, stale prior cv=0.3, seed 7: the pinned-bfw certifier reports
``obs_count_rmse`` 994.8 -> 365.3 (0.37x) and the ranking ``heldout_count_rmse``
816.5 -> 556.6 (0.68x) over the prior baseline -- a real "estimator beats prior" row --
with the ODME descent running 69 real gradient iterations (never a "0 iterations"
no-op). ``od_rmse`` barely moves (267.4 -> 264.5): ODME is a count-matcher, and Sioux
Falls OD is not identifiable from these sensors -- honest, disclosed.

**Disclosed engine envelope (part of the pinned 0.8.1 identity, like marouter's vdf).**

* ``route_output=1`` is REQUIRED, not cosmetic. ODME reconstructs each link's modeled
  volume from the stored route/path history; with ``route_output=0`` (dtalite-tap's lean
  default) that history is never populated, the reconstruction collapses to ~0 on every
  sensor, and ODME inflates every OD cell to its box ceiling (measured +40% total demand,
  a degenerate estimate). This module pins ``route_output=1`` (adr-042).
* The OD estimate is sourced EXCLUSIVELY from ``od_performance.csv``'s ``volume`` column
  (``= MDODflow``). ``link_performance.csv`` is measurably CORRUPTED under ``odme_mode=1``
  (a lossy post-ODME "final synchronization" overwrites the link volumes so they no longer
  conserve OD demand), so it is NEVER read here, and the ``dtalite-tap`` A2 cost-match /
  per-origin mass-gate / echo-check (validated only for ``odme_mode=0``) are NEVER reused.
* DTALite hard-clamps every emitted cell to ``[0.5*min(seed,target), 1.5*max(seed,target)]``
  -- a hardcoded box baked into the line search, no settings-file exposure. With
  ``demand_target = prior`` the estimator can only recover truth within ~2x of the prior.
  ``demand_target_frac`` (default 1.0) is a documented ONE-SIDED dial: ``>1`` raises the
  upper bound, ``<1`` lowers the lower bound (never both), at the cost of biasing the OD
  regularization pull toward the inflated target. It is a hashed estimator-identity factor.
* The ODME penalty weights (``w_link=0.1``, ``w_od=0.01``, ``w_vmt=1e-6``), the ABSOLUTE
  gradient-norm tolerance (``tol=1``), and the fixed 400-iteration descent are hardcoded
  C++ constants (only ``odme_mode``/``odme_vmt`` are settings-configurable). The ``tol=1``
  floor means small-demand anchors (Braess ~6) no-op with "0 iterations"; the marquee
  anchor MUST be Sioux-Falls-scale so realistic deviations clear the floor.

**Scope (adr-042): single-mode "auto" demand-only**, matching ``dtalite-tap``'s sprint-1
scope. The scenario envelope IS the T1 adapter's -- refusals (power!=1 is fine here; toll,
sub-0.1 capacity, and the SUE-family task fields) are delegated to
``DTALiteTapModel._refuse_unrepresentable`` verbatim. Native multiclass ODME
(``mode_type.csv``) is a named deferred follow-up.

**Determinism / budget / crash discipline (adr-029 verbatim).** The assignment+ODME path
is byte-deterministic under ``OMP_NUM_THREADS=1`` (md5-identical reruns, measured); the
estimator is ``deterministic=True``, ``seedable=False`` (no engine seed to pin). One
subprocess is one solve (``stdin=DEVNULL``, tempdir-per-run with ``finally`` cleanup);
``returncode`` is NEVER trusted (success = read-back of ``od_performance.csv``); a nonzero
exit / timeout / missing output RAISES ``RuntimeError`` (never a false estimate). An
``sp_calls``-only budget cannot bound the run (the engine hides its shortest-path count)
and is refused up front; ``budget.iterations`` maps to the base FW ``number_of_iterations``
(the ODME descent itself is the engine's fixed 400, not budget-controllable -- disclosed),
and ``wall_seconds`` is one deadline threaded across write -> subprocess -> parse.

``DTALite`` is an optional extra (``pip install tabench[dtalite]``); this module never
imports it at scope (the wheel prints a banner and ctypes-loads an OpenMP engine on import)
and is guarded in ``estimation/__init__.py`` so the numpy/scipy core stays dependency-free.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

# Probe availability WITHOUT importing the package: `import DTALite` prints a banner and
# ctypes-loads the engine .so + libgomp into the host (adr-029). find_spec touches neither.
# Absent -> ModuleNotFoundError(name="DTALite"), swallowed by exact name in
# estimation/__init__.py (the capital-`DTALite` guard shape, matching the T1 model guard).
if importlib.util.find_spec("DTALite") is None:  # pragma: no cover - core-install leg
    raise ModuleNotFoundError("No module named 'DTALite'", name="DTALite")

from ..core.budget import Budget, BudgetCoords
from ..core.factors import FactorSpec
from ..core.rng import RngBundle
from ..core.scenario import Scenario
from ..models.adapters.dtalite_tap import DTALiteTapModel
from ._proportions import active_pairs, od_from_pairs
from .base import (
    EstimationTask,
    ODEstimator,
    ODResultBundle,
    ODTrace,
    _estimation_capabilities,
    register_estimator,
)

__all__ = ["DtaliteODMEEstimator"]

# The demand period is fixed to one hour (07:00-08:00), so the engine's VDF makes flow V =
# demand I and the ratio V/cap the textbook v/c (the dtalite-tap convention, adr-029).
_PERIOD_START_HOUR = 7
_PERIOD_END_HOUR = 8
_DEFAULT_ITERATIONS = 100  # base FW number_of_iterations when the budget bounds only wall/None
_OD_FILE = "od_performance.csv"  # the ONLY trustworthy ODME output (volume = MDODflow)
_ODME_LOG = "ODME_log.txt"  # per-iteration objective/deviation trace (provenance only)
# The load-bearing hardcoded settings, as a SINGLE source of truth: the settings writer AND
# the recorded bundle identity (seed_info) both read these, so a change to either flips the
# written engine input AND the recorded provenance together (F4). ``route_output=1`` is the
# non-obvious one -- flipping it to 0 is the measured +40% degeneration (adr-042 Decision 2).
_ODME_MODE = 1  # settings.csv odme_mode (ODME is OFF at 0)
_ROUTE_OUTPUT = 1  # REQUIRED: ODME reconstructs modeled link volume from the route history
# The engine's non-configurable ODME constants, recorded as an identity tag (never scored):
# the [0.5,1.5]x-(seed,target) box, the absolute tol=1 gradient-norm floor, and the
# w_link/w_od/w_vmt penalty weights (adr-042 Decision 4).
_ODME_CFG_TAG = "box0.5-1.5;tol1;w0.1/0.01/1e-6"


@register_estimator
class DtaliteODMEEstimator(ODEstimator):
    """DTALite 0.8.1 static ODME as a guarded T2 estimator (Zhou & Taylor 2014; adr-042).

    Rides the UNCHANGED pinned-bfw ``ODCertifier`` (the adr-028 zero-certifier-change
    ideal). ``deterministic=True`` (byte-identical reruns at ``OMP_NUM_THREADS=1``);
    ``seedable=False`` (the engine exposes no seed -- the RngBundle root seed still lands in
    the bundle as provenance). ``provides_gap=False`` (the harness certifies).
    """

    name = "odme-dtalite"
    capabilities = _estimation_capabilities(deterministic=True, seedable=False)
    factors = {
        "demand_target_frac": FactorSpec(
            default=1.0, kind="float", bounds=(1e-6, 1e6),
            doc="One-sided widening of DTALite's hardcoded [0.5*min, 1.5*max]-of-"
            "(seed,target) box: demand_target.csv = prior * frac. >1 raises the upper "
            "recovery bound, <1 lowers the lower bound (never both), at the cost of biasing "
            "the (also hardcoded) OD-regularization pull toward the inflated target. 1.0 "
            "sets target == seed == prior, the symmetric [0.5,1.5]x-prior envelope (adr-042).",
        ),
        "keep_files": FactorSpec(
            default=False, kind="bool",
            doc="Keep the generated DTALite working directory for debugging instead of "
            "deleting it (path stored on the estimator as ``last_workdir``).",
        ),
    }

    def __init__(self, **factor_overrides: object) -> None:
        super().__init__(**factor_overrides)
        self.last_command: list[str] = []  # provenance / test inspection
        self.last_workdir: str | None = None

    def estimate(
        self, task: EstimationTask, budget: Budget, rng: RngBundle, trace: ODTrace
    ) -> ODResultBundle:
        start = time.perf_counter()

        # sp_calls is unmappable (the engine hides its Dijkstra count): an sp_calls-only
        # budget cannot bound the run and is refused up front rather than silently ignored
        # while the fixed 400-iteration ODME descent runs anyway (the adr-027/028/029
        # pattern; the inverted adr-025 lesson).
        if budget.iterations is None and budget.wall_seconds is None:
            raise ValueError(
                "odme-dtalite cannot honor an sp_calls-only budget (the DTALite engine "
                "exposes no shortest-path count); constrain iterations or wall_seconds so "
                "the run is bounded (adr-042). Note the ODME gradient descent itself is the "
                "engine's hardcoded 400 iterations, not budget-controllable."
            )
        n_iterations = (
            max(1, int(budget.iterations))
            if budget.iterations is not None
            else _DEFAULT_ITERATIONS
        )
        deadline = (
            start + budget.wall_seconds if budget.wall_seconds is not None else None
        )

        # Delegate the T1 adapter's unrepresentability envelope (SUE-family task fields,
        # nonzero fixed cost, sub-0.1 capacity clamp), surfaced fast on a probe scenario
        # BEFORE any engine run. The estimator's scenario envelope IS the adapter's, by
        # construction (the spsa_sumo.py:229-230 precedent). power!=1 is representable here
        # (DTALite's VDF is the repo BPR exactly, unlike marouter's linear law).
        probe = Scenario(name=task.name, network=task.network, demand=task.prior)
        DTALiteTapModel()._refuse_unrepresentable(probe)

        prior = np.asarray(task.prior.matrix, dtype=np.float64)
        pairs = active_pairs(prior)
        # No routable prior demand: nothing to calibrate. Emit the prior unchanged before
        # writing any file or spawning the engine (the dtalite-tap short-circuit); the
        # certifier scores it as a legitimate (terrible) estimate. last_command stays empty.
        if not pairs:
            coords = BudgetCoords(
                iterations=0, sp_calls=0, wall_ms=1000.0 * (time.perf_counter() - start)
            )
            trace.record(prior, coords, engine_odme_iterations=0.0)
            return self._bundle(trace, rng)

        frac = float(self.factor_values["demand_target_frac"])
        keep = bool(self.factor_values["keep_files"])
        workdir = tempfile.mkdtemp(
            prefix="tabench-odme-dtalite-keep-" if keep else "tabench-odme-dtalite-"
        )
        self.last_workdir = workdir
        try:
            # Phase 1 (in-host): write the GMNS CSVs + obs_volume + demand_target.csv.
            # Counts against the SAME wall deadline as the engine run (no phase unbounded).
            _write_odme_inputs(task, workdir, n_iterations, frac)

            cmd = [sys.executable, "-c", "import DTALite; DTALite.assignment()"]
            self.last_command = cmd
            timeout = None
            if deadline is not None:
                timeout = deadline - time.perf_counter()
                if timeout <= 0:
                    raise RuntimeError(
                        "wall_seconds budget exhausted while writing the engine inputs, "
                        f"before DTALite could run:\n  cmd: {' '.join(cmd)}"
                    )
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=workdir,
                    stdin=subprocess.DEVNULL,  # getchar() sees EOF, never blocks (adr-029)
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env={**os.environ, "OMP_NUM_THREADS": "1"},
                )
            except subprocess.TimeoutExpired as exc:
                # A wall-budget kill is an infrastructure outcome, not an infeasible
                # estimate: RAISE, never launder into a bad OD (crash discipline).
                raise RuntimeError(
                    "DTALite exceeded the wall_seconds budget and was killed:\n  "
                    f"cmd: {' '.join(cmd)}"
                ) from exc
            except OSError as exc:
                raise RuntimeError(
                    f"DTALite could not be executed ({exc}):\n  cmd: {' '.join(cmd)}"
                ) from exc

            od_path = os.path.join(workdir, _OD_FILE)
            # NEVER trust returncode alone (the engine exits 0 on missing files / garbage,
            # adr-029). A nonzero exit / missing-or-empty od_performance.csv IS a failure
            # (RuntimeError with tails). A MISSING demand_target.csv is a clean exit(1) here
            # -- but we always write it, so that path is defensive, pinned by a test.
            if proc.returncode != 0:
                raise RuntimeError(
                    f"DTALite ODME failed (exit {proc.returncode}):\n  cmd: {' '.join(cmd)}\n"
                    f"  stdout tail: {proc.stdout[-600:]}\n"
                    f"  stderr tail: {proc.stderr[-600:]}"
                )
            if not os.path.exists(od_path) or os.path.getsize(od_path) == 0:
                raise RuntimeError(
                    "DTALite ODME produced no od_performance.csv (or an empty one) while "
                    f"prior demand is positive:\n  cmd: {' '.join(cmd)}\n"
                    f"  stdout tail: {proc.stdout[-600:]}\n"
                    f"  stderr tail: {proc.stderr[-600:]}"
                )
            _check_deadline(deadline, cmd, "after the engine run, before read-back")
            try:
                od = _read_od_estimate(od_path, prior, pairs)
            except _ODMEReadError as exc:
                raise RuntimeError(
                    f"DTALite ODME output read-back failed ({exc}); the engine did not "
                    f"emit a well-formed od_performance.csv:\n  cmd: {' '.join(cmd)}\n"
                    f"  stdout tail: {proc.stdout[-400:]}\n"
                    f"  stderr tail: {proc.stderr[-400:]}"
                ) from exc
            _check_deadline(deadline, cmd, "during the read-back/parse phase")
            # Provenance ONLY (never gated, never scored): the ODME descent's iteration
            # count + a self-reported count fit from ITS OWN predicted volumes (the
            # route-history reconstruction, NOT the corrupted link_performance.csv). The
            # self-vs-certified diff MEASURES the engine-in-the-loop bias, exactly like
            # spsa-sumo -- provenance, not a bound.
            odme_iters, self_obs_rmse = _parse_odme_log(os.path.join(workdir, _ODME_LOG))
        finally:
            if not keep:
                shutil.rmtree(workdir, ignore_errors=True)
                self.last_workdir = None

        self_report = {"engine_odme_iterations": float(odme_iters)}
        if self_obs_rmse is not None:
            self_report["obs_count_rmse"] = float(self_obs_rmse)
        coords = BudgetCoords(
            iterations=n_iterations,  # base FW iterations (the ODME descent is the fixed 400)
            sp_calls=0,  # the engine exposes no shortest-path count (disclosed, not hidden)
            wall_ms=1000.0 * (time.perf_counter() - start),
        )
        # A single-checkpoint emit: the entire 400-iteration ODME gradient descent happens
        # INSIDE one DTALite subprocess call, so there is one OD to certify (the GLSEstimator
        # one-call shape, not the SPSAEstimator external multi-checkpoint loop).
        trace.record(od, coords, **self_report)
        return self._bundle(trace, rng)

    def _bundle(self, trace: ODTrace, rng: RngBundle) -> ODResultBundle:
        from ..models.adapters.dtalite_tap import _engine_version

        return ODResultBundle(
            estimator_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            # Record the load-bearing engine config in the bundle identity, not just the
            # structural settings test: odme_mode + route_output (the pilot-caught flip) +
            # the box/tol/weights tag, so a config change shows in the RECORDED provenance
            # (never scored -- seed_info is provenance only, F4).
            seed_info={
                **rng.describe(),
                "engine": _engine_version(),
                "odme_mode": _ODME_MODE,
                "route_output": _ROUTE_OUTPUT,
                "odme_cfg": _ODME_CFG_TAG,
            },
        )


# --------------------------------------------------------------------------------------
# GMNS + ODME compilation and read-back (adr-042)
# --------------------------------------------------------------------------------------


class _ODMEReadError(Exception):
    """od_performance.csv is missing, malformed, inconsistent, or incomplete."""


def _write_odme_inputs(
    task: EstimationTask, workdir: str, n_iterations: int, demand_target_frac: float
) -> None:
    """Write node/link/demand/demand_target/settings CSVs for one ODME solve.

    Mirrors ``dtalite_tap._write_gmns``'s identity compile map (lanes=1, capacity = total
    link capacity, VDF columns carrying the repo BPR verbatim, links SORTED by
    ``(from_node_id, to_node_id)`` -- an ungrouped link.csv silently corrupts the engine's
    adjacency, measured CRITICAL, adr-029) and adds the two ODME-only pieces: an
    ``obs_volume`` column (period-mean sensor count on each observed link, blank elsewhere)
    and a ``<demand>_target.csv`` regularization anchor. ``settings.csv`` sets
    ``odme_mode=1`` AND ``route_output=1`` (REQUIRED -- ODME reconstructs modeled link
    volume from the stored route history; without it the reconstruction collapses to ~0 and
    ODME inflates every cell to its box ceiling, measured +40% demand -- adr-042).
    """
    net = task.network
    prior = np.asarray(task.prior.matrix, dtype=np.float64)
    n = prior.shape[0]

    # node.csv: zones are nodes 1..n_zones with zone_id == node_id; other nodes zone_id 0.
    with open(os.path.join(workdir, "node.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["node_id", "zone_id", "x_coord", "y_coord"])
        for nid in range(1, net.n_nodes + 1):
            zid = nid if nid <= net.n_zones else 0
            w.writerow([nid, zid, float(nid), float((nid * 7) % 13)])

    # Period-mean sensor counts -> obs_volume (the P1 obs_mean_count_rmse companion the
    # certifier scores against). counts is (n_periods, n_sensors); sensor_links is 0-based
    # link indices. Only rows with obs_volume>1 are used by the engine (measured).
    sensors = np.asarray(task.dataset.payload["sensor_links"], dtype=np.int64)
    counts = np.asarray(task.dataset.payload["counts"], dtype=np.float64)
    mean_counts = counts.mean(axis=0) if counts.size else np.zeros(sensors.shape)
    obs_by_link = {int(sensors[k]): float(mean_counts[k]) for k in range(sensors.size)}

    order = sorted(
        range(net.n_links),
        key=lambda i: (int(net.init_node[i]), int(net.term_node[i])),
    )
    with open(os.path.join(workdir, "link.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["from_node_id", "to_node_id", "link_id", "lanes", "capacity", "free_speed",
             "length", "vdf_fftt", "vdf_alpha", "vdf_beta", "vdf_plf", "toll", "obs_volume"]
        )
        for i in order:
            obs = obs_by_link.get(i)
            w.writerow([
                int(net.init_node[i]), int(net.term_node[i]), i + 1, 1,
                repr(float(net.capacity[i])), 60.0, 0.0,
                repr(float(net.free_flow_time[i])), repr(float(net.b[i])),
                repr(float(net.power[i])), 1, 0,
                repr(obs) if obs is not None else "",
            ])

    # demand.csv (seed) and demand_target.csv (regularization target / box anchor); one row
    # per positive off-diagonal prior cell. demand_target scales the seed by the one-sided
    # frac dial (frac=1.0 -> target == seed -> the symmetric [0.5,1.5]x-prior box).
    for fname, scale in (("demand.csv", 1.0), ("demand_target.csv", demand_target_frac)):
        with open(os.path.join(workdir, fname), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["o_zone_id", "d_zone_id", "volume"])
            for o in range(n):
                for d in range(n):
                    if o != d and prior[o, d] > 0:
                        w.writerow([o + 1, d + 1, repr(float(prior[o, d] * scale))])

    # settings.csv: single 1 h period, ODME ON (odme_mode=1), route_output=1 (REQUIRED for
    # the ODME reconstruction -- adr-042), lean otherwise. odme_vmt=0 (the VMT term is inert;
    # there is no system-VMT concept in the T2 contract).
    with open(os.path.join(workdir, "settings.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "number_of_iterations", "number_of_processors",
            "demand_period_starting_hours", "demand_period_ending_hours",
            "first_through_node_id", "base_demand_mode", "route_output",
            "vehicle_output", "log_file", "odme_mode", "odme_vmt",
        ])
        w.writerow([
            int(n_iterations), 1, _PERIOD_START_HOUR, _PERIOD_END_HOUR,
            int(net.first_thru_node), 0, _ROUTE_OUTPUT, 0, 0, _ODME_MODE, 0,
        ])


def _read_od_estimate(
    od_path: str, prior: np.ndarray, pairs: list[tuple[int, int]]
) -> np.ndarray:
    """Parse ``od_performance.csv`` into a full ``(Z,Z)`` OD estimate.

    The OD estimate is sourced EXCLUSIVELY from the ``volume`` column (= MDODflow), keyed
    by ``(o_zone_id, d_zone_id)`` (1-based). ``link_performance.csv`` is corrupted under
    ``odme_mode=1`` and is NEVER read (adr-042). Discipline transported from
    ``dtalite_tap``'s "every repo link matched exactly once" to the OD-pair domain: every
    prior-support pair must appear (completeness), a pair's rows must all report the same
    volume (route-split consistency), no off-support/phantom pair may appear, and every
    volume must be finite and non-negative -- else ``_ODMEReadError`` (mapped to the
    contract's RuntimeError). Diagonal and off-support cells carry the prior via
    ``od_from_pairs``.
    """
    support = {p for p in pairs}
    by_pair: dict[tuple[int, int], float] = {}
    try:
        fh = open(od_path, newline="", encoding="utf-8")
    except OSError as exc:  # pragma: no cover - the caller already checked existence
        raise _ODMEReadError(f"cannot open od_performance.csv: {exc}") from exc
    try:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or [])
        required = {"o_zone_id", "d_zone_id", "volume"}
        missing_cols = required - fieldnames
        if missing_cols:
            raise _ODMEReadError(f"od_performance.csv missing columns {sorted(missing_cols)}")
        try:
            rows = list(reader)
        except (UnicodeDecodeError, csv.Error) as exc:
            raise _ODMEReadError(f"undecodable od_performance.csv body: {exc}") from exc
    finally:
        fh.close()

    for row in rows:
        try:
            o = int(row["o_zone_id"]) - 1
            d = int(row["d_zone_id"]) - 1
            vol = float(row["volume"])
        except (TypeError, ValueError) as exc:
            raise _ODMEReadError(
                f"unparseable od_performance row {row!r}: {exc}"
            ) from exc
        key = (o, d)
        if key not in support:
            # An off-diagonal pair the engine emitted that is absent from the prior support
            # (phantom) -- the OD-pair analogue of dtalite-tap's phantom-link check. The
            # diagonal (o == d) is intrazonal and legitimately ignored.
            if o != d:
                raise _ODMEReadError(
                    f"engine emitted OD pair {(o + 1, d + 1)} absent from the prior support"
                )
            continue
        if not (np.isfinite(vol) and vol >= 0.0):
            raise _ODMEReadError(f"OD pair {(o + 1, d + 1)} has a bad volume {vol!r}")
        if key in by_pair and abs(by_pair[key] - vol) > 1e-6:
            raise _ODMEReadError(
                f"OD pair {(o + 1, d + 1)} reported inconsistent volumes "
                f"{by_pair[key]!r} vs {vol!r} across route rows"
            )
        by_pair[key] = vol

    missing = [p for p in pairs if p not in by_pair]
    if missing:
        raise _ODMEReadError(
            f"od_performance.csv omitted {len(missing)} prior-support OD pair(s), "
            f"e.g. {[(o + 1, d + 1) for (o, d) in missing[:8]]}"
        )
    g = np.array([by_pair[p] for p in pairs], dtype=np.float64)
    return od_from_pairs(prior, pairs, g)


def _check_deadline(deadline: float | None, cmd: list[str], phase: str) -> None:
    """Raise the contract's RuntimeError if the wall deadline has passed. Threaded through
    every post-subprocess phase so a slow read-back/parse cannot silently overrun the
    ``wall_seconds`` budget (the adr-029 discipline)."""
    if deadline is not None and time.perf_counter() > deadline:
        raise RuntimeError(
            f"wall_seconds budget exhausted {phase}:\n  cmd: {' '.join(cmd)}"
        )


def _parse_odme_log(log_path: str) -> tuple[int, float | None]:
    """Return ``(odme_iterations, self_obs_count_rmse)`` from ``ODME_log.txt`` -- PROVENANCE
    ONLY (never scored, never gated).

    ``odme_iterations`` is the highest ``Iteration k`` line + 1 (the descent length, > 0 on
    a real calibration, 0 on a ``tol=1``-floor no-op). ``self_obs_count_rmse`` is the RMSE
    of the engine's OWN "Deviation Log" (``Predicted volume`` vs ``Observed volume`` over the
    sensor links) -- reconstructed from the route history (NOT the corrupted
    link_performance.csv); the self-vs-certified diff MEASURES the engine-in-the-loop bias
    (the spsa-sumo reframing). Missing/short/format-drift file -> ``(0, None)``, never an
    error: the OD estimate is already validated by the read-back."""
    if not os.path.exists(log_path):
        return 0, None
    max_iter = -1
    sq = 0.0
    n = 0
    try:
        with open(log_path, encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("Iteration"):
                    try:
                        max_iter = max(max_iter, int(s.split("Iteration")[1].split(":")[0]))
                    except (IndexError, ValueError):  # pragma: no cover - defensive
                        continue
                elif "Predicted volume:" in s and "Observed volume:" in s:
                    try:
                        pred = float(s.split("Predicted volume:")[1].split(",")[0])
                        obs = float(s.split("Observed volume:")[1].split(",")[0])
                    except (IndexError, ValueError):  # pragma: no cover - defensive
                        continue
                    sq += (pred - obs) ** 2
                    n += 1
    except OSError:  # pragma: no cover - defensive
        return (max_iter + 1 if max_iter >= 0 else 0), None
    iters = max_iter + 1 if max_iter >= 0 else 0
    self_rmse = float(np.sqrt(sq / n)) if n else None
    return iters, self_rmse
