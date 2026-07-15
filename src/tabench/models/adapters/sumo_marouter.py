"""SUMO ``marouter`` macroscopic assignment as a benchmark adapter (Lopez et al. 2018).

``marouter`` (shipped in the ``eclipse-sumo`` wheel the Lopez et al. 2018 ITSC
tool paper describes) solves a macroscopic stochastic user equilibrium and emits
per-edge flows directly via ``--netload-output``. This adapter compiles a
fixed-demand ``power == 1`` scenario into a SUMO network (``_sumo_io.py``), runs
``marouter`` to convergence under ITS hardcoded linear cost law, reads the edge
flows back into repo-native units, and lets the harness certify the equilibrium
gap under the scenario's DECLARED BPR costs (P1).

**What the certified row means (adr-027).** marouter equilibrates its OWN cost
model (a hardcoded linear-in-v/c class law, NOT the scenario's BPR); on a linear
scenario the adapter maps that law near-exactly, so the certified gap is a real
but small mapping floor (Braess ~1.7e-4). The row measures ADAPTER + marouter
fidelity to the repo's equilibrium, never the ITSC paper's numerics; the cost-law
provenance is PTV-Validate / VISUM-Cologne per the SUMO source comment.

**Honest traps this adapter refuses (measured on 1.27.1), never launders:**

* ``--assignment-method UE`` silently falls back to SUE with only a warning -> a
  declared "UE" run would poison the manifest; REFUSED with ValueError.
* the default ``--assignment-method incremental`` is NOT an equilibrium (RG ~1e-1
  on Braess) -> the adapter always requests ``SUE`` explicitly.
* ``gawron`` / ``lohse`` route choice emit all-zero flows in this macroscopic
  setting (the harness would censor them feasible=0) -> the factor space is
  RESTRICTED to ``logit``.
* BPR ``power != 1`` (e.g. Sioux Falls power-4) is UNREPRESENTABLE in marouter's
  hardcoded linear vdf -> refused loudly, a documented capability limit rather
  than a silently wrong-cost run.

``eclipse-sumo`` is an optional extra (``pip install tabench[sumo]``); this module
imports ``sumo`` and is guarded in ``models/__init__.py`` so the numpy/scipy core
stays dependency-free.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time

import numpy as np

from ...core.budget import Budget, BudgetCoords
from ...core.capabilities import Capabilities
from ...core.factors import FactorSpec
from ...core.results import ResultBundle, Trace
from ...core.rng import RngBundle
from ...core.scenario import Scenario
from ..base import TrafficAssignmentModel, register_model
from . import _sumo_io as sio

__all__ = ["SumoMarouterModel"]

_DEFAULT_MAX_ITERATIONS = 50  # marouter outer-assignment iterations when unbudgeted
_ENGINE_VERSION: str | None = None  # cached `marouter --version` string (provenance)


def _engine_version() -> str:
    """Cached ``marouter --version`` string, recorded as manifest provenance.

    The vdf/capacity tables are hardcoded in the SUMO source and could change
    between releases, so the running engine version is pinned in CI and recorded
    here (adr-027)."""
    global _ENGINE_VERSION
    if _ENGINE_VERSION is None:
        try:
            out = subprocess.run(
                [sio.sumo_binary("marouter"), "--version"],
                env=sio.sumo_env(), capture_output=True, text=True, timeout=30,
            )
            _ENGINE_VERSION = next(
                (ln.strip() for ln in out.stdout.splitlines() if "marouter" in ln.lower()),
                "unknown",
            )
        except (OSError, subprocess.SubprocessError):
            _ENGINE_VERSION = "unknown"
    return _ENGINE_VERSION


@register_model
class SumoMarouterModel(TrafficAssignmentModel):
    """SUMO ``marouter`` macroscopic SUE assignment (Lopez et al. 2018; ``heuristic``).

    A registered, CI-validated external-simulator adapter. It reuses the
    ``heuristic`` paradigm (the ``aon`` precedent): the Evaluator branches on the
    scenario's task fields, not the paradigm, so a heuristic-paradigm model earns
    the certified ``relative_gap`` on static scenarios exactly as ``aon`` does.
    Deterministic (marouter's SUE/logit path uses no RNG -- byte-identical netload
    across seeds), so the seed is pinned on the command line for provenance and
    ``--routing-threads 1`` is unconditional (byte-determinism verified only
    single-threaded). ``provides_gap=False``: the harness recomputes the gap.
    """

    name = "sumo-marouter"
    capabilities = Capabilities(
        paradigm="heuristic",
        deterministic=True,
        provides_gap=False,
        seedable=True,
    )
    factors = {
        "assignment_method": FactorSpec(
            default="SUE", kind="str",
            doc="marouter --assignment-method: 'SUE' (equilibrium) or 'incremental' "
            "(non-equilibrium, honest worse-gap control). 'UE' is REFUSED (marouter "
            "silently falls back to SUE).",
        ),
        "route_choice": FactorSpec(
            default="logit", kind="str",
            doc="Route-choice family; RESTRICTED to 'logit' (gawron/lohse emit "
            "all-zero flows in this macroscopic setting).",
        ),
        "logit_theta": FactorSpec(
            default=200.0, kind="float", bounds=(0.0, 1e6),
            doc="Logit dispersion in 1/(native cost unit); large -> deterministic-UE "
            "approximation. Calibrated on the ASYMMETRIC two-route anchor (never "
            "symmetric Braess, whose UE is the equal split at any theta). Passed as "
            "--logit.theta = logit_theta / time_scale (per second).",
        ),
        "paths": FactorSpec(
            default=4, kind="int", bounds=(1, 64),
            doc="marouter --paths: k-shortest paths enumerated per OD for the SUE "
            "route set.",
        ),
        "max_inner_iterations": FactorSpec(
            default=5000, kind="int", bounds=(1, 100000),
            doc="marouter --max-inner-iterations: inner MSA flow-averaging cap.",
        ),
        "tolerance": FactorSpec(
            default=1e-7, kind="float", bounds=(0.0, 1.0),
            doc="marouter --tolerance: per-edge SUE stability tolerance under ITS "
            "own costs (NOT the repo gap; target_relative_gap is ignored).",
        ),
        "time_scale": FactorSpec(
            default=1.0, kind="float", bounds=(0.2, 30.0),
            doc="tau: seconds per native cost unit. Divides out of MASS (link flows "
            "are native), NOT the certified gap: it rescales marouter's internal "
            "seconds, which sets both the compiled edge lengths (too small -> the "
            "netconvert 0.1 m clamp) and the path times vs the OD window (too large "
            "-> free-flow revert). Bounds are the validated envelope; the compile "
            "read-back + window sizing catch any residual out-of-envelope corruption.",
        ),
        "min_lanes": FactorSpec(
            default=1, kind="int", bounds=(1, 1000),
            doc="Minimum quantized lane count on representable links (flow-scale "
            "resolution). Larger values shrink the forced-intercept floor at the "
            "cost of bigger nets.",
        ),
        "keep_files": FactorSpec(
            default=False, kind="bool",
            doc="Keep the generated SUMO working directory for debugging instead of "
            "deleting it (path stored on the model as ``last_workdir``).",
        ),
    }

    def __init__(self, **factor_overrides: object) -> None:
        super().__init__(**factor_overrides)
        self.last_command: list[str] = []  # for provenance / test inspection
        self.last_workdir: str | None = None

    def _refuse_unrepresentable(self, scenario: Scenario) -> None:
        """Raise ValueError naming the first field that makes the instance
        non-representable in marouter's hardcoded linear vdf (adr-027)."""
        for field in sio.REFUSED_TASK_FIELDS:
            if getattr(scenario, field, None) is not None:
                raise ValueError(
                    f"sumo-marouter accepts only fixed-demand power==1 UE scenarios; "
                    f"scenario '{scenario.name}' sets '{field}', which marouter's "
                    "hardcoded linear cost law cannot represent (adr-027)."
                )
        # Generalized-cost fixed terms (toll_weight*toll + distance_weight*length)
        # enter the certified link_cost but marouter has no per-edge cost hook, so
        # they would be SILENTLY dropped and a tolled instance scored under the
        # wrong cost model. Refuse loudly (adr-027 review CRITICAL/MAJOR).
        if np.any(np.asarray(scenario.network.fixed_cost, dtype=np.float64) != 0.0):
            raise ValueError(
                f"sumo-marouter cannot represent generalized-cost fixed terms; "
                f"scenario '{scenario.name}' has a nonzero fixed cost "
                "(toll_weight*toll + distance_weight*length) that marouter's vdf "
                "would silently drop -- set toll_weight and distance_weight to 0 "
                "or use a white-box solver (adr-027)."
            )
        if np.any(np.asarray(scenario.network.power, dtype=np.float64) != 1.0):
            raise ValueError(
                f"sumo-marouter accepts only power==1 (linear) latencies; scenario "
                f"'{scenario.name}' has power != 1 (e.g. Sioux Falls power-4 is "
                "UNREPRESENTABLE in marouter's linear vdf -- a capability limit, "
                "adr-027)."
            )

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        start = time.perf_counter()
        self._refuse_unrepresentable(scenario)

        method = self.factor_values["assignment_method"]
        if method == "UE":
            raise ValueError(
                "assignment_method='UE' is refused: marouter does not implement "
                "deterministic UE and silently falls back to SUE (verified on "
                "1.27.1) -- declaring UE would poison the manifest. Use 'SUE'."
            )
        if method not in ("SUE", "incremental"):
            raise ValueError(
                f"assignment_method must be 'SUE' or 'incremental', got {method!r}"
            )
        route_choice = self.factor_values["route_choice"]
        if route_choice != "logit":
            raise ValueError(
                f"route_choice is restricted to 'logit' (gawron/lohse emit all-zero "
                f"flows in this macroscopic setting); got {route_choice!r}."
            )

        # Budget mapping (P6). marouter exposes no shortest-path (Dijkstra) count,
        # so sp_calls is unmappable: an sp_calls-ONLY budget cannot bound the run
        # and is refused up front rather than silently running unbounded (the
        # inverted adr-025 wall_seconds lesson). iterations -> --max-iterations;
        # wall_seconds -> subprocess timeout; target_relative_gap ignored (single
        # shot; --tolerance is SUE stability under ITS costs, not the repo gap).
        if budget.iterations is None and budget.wall_seconds is None:
            raise ValueError(
                "sumo-marouter cannot honor an sp_calls-only budget (marouter "
                "exposes no shortest-path count); constrain iterations or "
                "wall_seconds so the run is bounded (adr-027)."
            )
        # Floor --max-iterations at 1: --max-iterations 0 makes marouter emit an
        # all-zero flow (censored feasible=0) -- the exact outcome gawron/lohse are
        # refused for, reachable through a budget sweep otherwise (adr-027 review).
        max_iters = (
            max(1, int(budget.iterations))
            if budget.iterations is not None
            else _DEFAULT_MAX_ITERATIONS
        )
        # The wall budget must cover BOTH the netconvert compile phase and the
        # marouter run: thread a single deadline through both (adr-027 review MAJOR
        # -- the compile phase was silently unbounded).
        deadline = start + budget.wall_seconds if budget.wall_seconds is not None else None

        # Zero routable demand: the exact equilibrium is the zero flow. Short-
        # circuit before compiling the network or running the engine (the
        # implicit_ue precedent); `last_command` stays empty (no marouter run).
        od = np.asarray(scenario.demand.matrix, dtype=np.float64)
        if not np.any(od[~np.eye(od.shape[0], dtype=bool)] > 0):
            flows = np.zeros(scenario.network.n_links)
            coords = BudgetCoords(
                iterations=0, sp_calls=0, wall_ms=1000.0 * (time.perf_counter() - start)
            )
            trace.record(flows, coords, engine_iterations_executed_unknown=1.0)
            return self._bundle(trace, rng)

        # Seed is drawn from the RngBundle stream and pinned on the command line
        # (deterministic given root_seed) even though marouter's SUE path is
        # RNG-free -- so the manifest records a real, reproducible seed and the
        # single-threaded byte-determinism declaration holds unconditionally.
        seed = int(rng.generator(source=0).integers(1, 2**31 - 1))
        theta_cmd = self.factor_values["logit_theta"] / self.factor_values["time_scale"]

        keep = bool(self.factor_values["keep_files"])
        workdir = tempfile.mkdtemp(prefix="tabench-sumo-keep-" if keep else "tabench-sumo-")
        self.last_workdir = workdir
        try:
            build = sio.scenario_to_sumo(
                scenario, workdir,
                time_scale=self.factor_values["time_scale"],
                min_lanes=int(self.factor_values["min_lanes"]),
                deadline=deadline,
            )
            netload = os.path.join(workdir, "netload.xml")
            cmd = [
                sio.sumo_binary("marouter"),
                "--net-file", build.net_file,
                "--additional-files", build.taz_file,
                "--od-matrix-files", build.od_file,
                "--netload-output", netload,
                "--output-file", os.devnull,
                "--precision", "9",
                "--weights.minor-penalty", "0",
                "--left-turn-penalty", "0",
                "--capacities.default", "true",
                "--assignment-method", method,
                "--route-choice-method", route_choice,
                "--logit.beta", "0",
                "--logit.gamma", "0",
                "--logit.theta", repr(float(theta_cmd)),
                "--paths", str(int(self.factor_values["paths"])),
                "--max-iterations", str(int(max_iters)),
                "--max-inner-iterations", str(int(self.factor_values["max_inner_iterations"])),
                "--tolerance", repr(float(self.factor_values["tolerance"])),
                "--seed", str(seed),
                "--routing-threads", "1",
            ]
            self.last_command = cmd
            marouter_timeout = None
            if deadline is not None:
                marouter_timeout = deadline - time.perf_counter()
                if marouter_timeout <= 0:
                    raise RuntimeError(
                        "wall_seconds budget exhausted during the compile phase, "
                        f"before marouter could run:\n  cmd: {' '.join(cmd)}"
                    )
            try:
                proc = subprocess.run(
                    cmd, env=sio.sumo_env(), capture_output=True, text=True,
                    cwd=workdir, timeout=marouter_timeout,
                )
            except subprocess.TimeoutExpired as exc:
                # A wall-budget kill is an infrastructure outcome, not an
                # infeasible solution: RAISE, never launder into feasible=0.
                raise RuntimeError(
                    "marouter exceeded the wall_seconds budget and was killed:\n  "
                    f"cmd: {' '.join(cmd)}"
                ) from exc
            except OSError as exc:
                # Missing/undiscoverable binary etc.: the contract's RuntimeError
                # with diagnostics, not a bare FileNotFoundError (adr-027).
                raise RuntimeError(
                    f"marouter could not be executed ({exc}):\n  cmd: {' '.join(cmd)}"
                ) from exc
            if proc.returncode != 0 or not os.path.exists(netload):
                raise RuntimeError(
                    f"marouter failed (exit {proc.returncode}):\n  cmd: "
                    f"{' '.join(cmd)}\n  stderr tail: {proc.stderr[-800:]}"
                )
            if "No interval matches" in proc.stderr:
                # A path's cumulative time exceeded the OD window and marouter
                # reverted the edge to free-flow -- an engine artifact, never a
                # feasible row (the window is sized to avoid this; adr-027 review).
                raise RuntimeError(
                    "marouter reverted an edge to free-flow (OD time window too "
                    f"short for a path):\n  cmd: {' '.join(cmd)}\n  stderr tail: "
                    f"{proc.stderr[-400:]}"
                )
            try:
                n_intervals = sio.netload_interval_count(netload)
                n_matched = sio.netload_matched_edge_count(netload, scenario.network.n_links)
                flows = sio.parse_netload(netload, scenario.network.n_links, build.flow_scale)
            except Exception as exc:  # noqa: BLE001 - unparseable engine output is infra failure
                raise RuntimeError(
                    f"could not parse marouter netload {netload!r}: {exc}\n  cmd: "
                    f"{' '.join(cmd)}"
                ) from exc
            if n_intervals != 1:
                raise RuntimeError(
                    f"marouter netload has {n_intervals} intervals (expected 1); the "
                    "single-interval reader would drop mass -- refusing (adr-027)."
                )
            if n_matched == 0:
                raise RuntimeError(
                    "marouter wrote a netload with no edge data while demand is "
                    f"positive (engine failure, not an infeasible flow):\n  cmd: "
                    f"{' '.join(cmd)}"
                )
        finally:
            if not keep:
                shutil.rmtree(workdir, ignore_errors=True)
                self.last_workdir = None

        coords = BudgetCoords(
            iterations=int(max_iters),  # the CONFIGURED cap: marouter hides the executed count
            sp_calls=0,  # marouter exposes no shortest-path count (disclosed, not hidden)
            wall_ms=1000.0 * (time.perf_counter() - start),
        )
        trace.record(
            flows,
            coords,
            # Disclosure (P6): marouter does not report the number of outer
            # iterations it actually ran, so `iterations` above is the cap, not
            # the executed count -- flagged rather than silently over-reported.
            engine_iterations_executed_unknown=1.0,
            # Mapping-floor provenance (the certified gap includes these).
            mapping_forced_intercept=float(build.forced_intercept),
            mapping_parasitic_slope=float(build.parasitic_slope),
            flow_scale=float(build.flow_scale),
        )
        return self._bundle(trace, rng)

    def _bundle(self, trace: Trace, rng: RngBundle) -> ResultBundle:
        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info={**rng.describe(), "engine": _engine_version()},
        )
