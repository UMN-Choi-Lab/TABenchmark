"""DTALite ``assignment()`` static traffic assignment as a benchmark adapter (Zhou & Taylor 2014).

The PyPI ``DTALite`` wheel exposes a static **Frank-Wolfe** user-equilibrium
solver its own source calls *TAPLite* (a Bar-Gera ``FW.zip``-derived link-based FW
loop on BPR costs). This adapter compiles a fixed-demand scenario into the engine's
GMNS CSVs, runs ``DTALite.assignment()`` in a throwaway subprocess, reads the emitted
per-link flows back into repo-native units, and lets the harness certify the
equilibrium gap under the scenario's DECLARED BPR costs (P1).

**What the certified row means (adr-029).** Unlike ``sumo-marouter`` (whose linear
class law forced a mapping floor), DTALite's per-link ``vdf_fftt/vdf_alpha/vdf_beta``
is the repo BPR ``t = fft*(1 + b*(v/cap)^power)`` *exactly* -- Sioux Falls power-4 maps
with no cost-model approximation. So the certified gap is the engine's own
**Frank-Wolfe convergence** under an exactly-represented cost law, not a mapping error.
The catch is honest and measured: the engine's Armijo line search collapses to step
0 within a few iterations, so the certified relative gap freezes at a floor (Braess
~1.2e-2, Sioux Falls ~5.0e-3) far above a converged white-box solver's ~1e-16 -- the
row is the wheel-engine-as-shipped, and the ceiling is its line-search stall, not the
mapping. The citation anchors the DTALite *software lineage* (tool-paper discipline,
the lopez2018 precedent): ``assignment()`` is static FW on BPR, NOT the 2014 paper's
mesoscopic DUE machinery, which is the separate ``simulation()`` entry -- a named
non-goal here, now shipped on the OBSERVATIONAL track as ``dtalite-simulation``
(adr-040, the third EDOC-1 row), which closes the honest-sourcing loop this docstring
opened: the queue-DNL content the citation names lives in that row's EDOC producer, not
in this static-assignment adapter.

**Honest traps this adapter refuses / neutralizes (measured on 0.8.1):**

* the engine's ``ExitMessage`` does ``getchar()`` then ``exit()`` in-process, so a bad
  input would HANG (open stdin) or KILL the host -- the mandatory subprocess wrapper
  runs the child with ``stdin=DEVNULL`` and a single wall deadline.
* a second ``assignment()`` call in one process doubles the flows (residual global
  state) -- one subprocess is one solve, ever.
* almost no bad input crashes (missing files/columns, dropped links, zone!=node ->
  rc 0 with zero/garbage flows), so ``returncode == 0`` is NOT trusted: success
  requires the read-back below to match every repo link.
* the ``lanes`` field enters the congestion ratio as ``lanes^2`` -- lanes is ALWAYS
  written as 1 with ``capacity`` = the total link capacity, so the engine's cost is
  the textbook BPR.
* tolls / generalized-cost fixed terms have a ``toll`` hook but unvalidated
  time-conversion semantics -- any nonzero ``fixed_cost`` is REFUSED, not silently run.

``DTALite`` is an optional extra (``pip install tabench[dtalite]``); this module never
imports it at module scope (the package prints a banner and ctypes-loads an OpenMP
engine into the host on import), and is guarded in ``models/__init__.py`` so the
numpy/scipy core stays dependency-free.
"""

from __future__ import annotations

import csv
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import sys
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

# Probe availability WITHOUT importing the package: `import DTALite` prints a version
# banner to stdout and ctypes-loads the engine .so + libgomp into the host process
# (adr-029). find_spec touches neither. Absent -> ModuleNotFoundError(name="DTALite"),
# which models/__init__.py swallows by exact name (the sumo guard shape).
if importlib.util.find_spec("DTALite") is None:  # pragma: no cover - core-install leg
    raise ModuleNotFoundError("No module named 'DTALite'", name="DTALite")

__all__ = ["DTALiteTapModel"]

_ENGINE_MODULE = "DTALite"
_DEFAULT_ITERATIONS = 100  # number_of_iterations when the budget bounds only wall/None
# The demand period is fixed to one hour (07:00-08:00): the engine's VDF divides the
# link flow by the period length, so a 1 h period makes I = V and the ratio V/cap the
# textbook v/c (measured, adr-029).
_PERIOD_START_HOUR = 7
_PERIOD_END_HOUR = 8
# Read-back tolerances on the engine's echoed VDF parameters. The engine stores
# capacity as float32 and writes the link_performance columns to 4 decimals (measured:
# capacity 25900.20064 -> echoed 25900.2012, rel ~3e-8; fft=1e-6 -> echoed 0.0), so an
# exact echo is impossible; these are ~30x above that noise floor. The echo check has a
# KNOWN blind spot below atol=1e-3 absolute (a doctored/ignored sub-1e-3 fftt is accepted
# -- adr-029 review); the runtime A2 cost-match below is the LIVE gate that closes it.
_READBACK_RTOL = 1e-3
_READBACK_ATOL = 1e-3
# Runtime A2 cost-match: the engine's own travel_time must equal the repo BPR at the
# emitted flows on EVERY link (the identity map's payoff). Measured legit max relative
# error is ~2e-5 on the anchors and ~7.6e-4 on Barcelona's power-16.83 links (the
# engine's 4-decimal travel_time column); 1e-3 clears them yet catches a cost-law drift
# -- the capacity clamp below (rel ~0.93) and an ignored fftt both fail it (adr-029).
_A2_RTOL = 1e-3
_A2_ATOL = 1e-3
# The engine clamps capacity at fmax(0.1, cap) in the COST law ONLY (TAPLite.cpp; the
# Beckmann integral is left unclamped -- internally inconsistent below 0.1), so a
# capacity in (1e-4, 0.1) equilibrates under a DIFFERENT law while passing the read-back
# (measured A2 rel 0.93). Refuse below the clamp threshold (the engine's other fmax
# floors -- lanes 0.01, period 0.001, plf 0.0001 -- are all neutralized by our constants
# lanes=1 / period=1 / plf=1). A link with capacity <= 1e-4 is also silently DROPPED.
_MIN_CAPACITY = 0.1
# Per-origin mass gate (the ~1e7 BIG-M ceiling): an origin zone's emitted outflow must
# cover its routable demand within this loose tolerance (through traffic only ADDS, so a
# deficit is always an engine drop -- measured deficit on clean inputs is 0; a BIG-M
# zeroing is a 100% deficit, far above the 4-decimal rounding floor ~1e-4).
_MASS_RTOL = 1e-2
_MASS_ATOL = 1e-2

#: Scenario task fields DTALite's static ``assignment()`` cannot represent in a form
#: the repo certificate can score: refused loudly (ValueError naming the field), never
#: run with a silently wrong model (adr-029). ``multiclass`` is the natural future
#: extension (the engine does native multiclass via ``mode_type.csv``), refused in
#: sprint 1 pending a per-class-flow certificate.
REFUSED_TASK_FIELDS = (
    "sue_theta",
    "elastic_demand",
    "combined_demand",
    "br_epsilon",
    "side_capacities",
    "link_interaction",
    "multiclass",
)

# link_performance.csv column names the read-back consumes (lowercase, case-sensitive).
_LP_FILE = "link_performance.csv"
_SUMMARY_FILE = "summary_log_file.txt"


def _engine_version() -> str:
    """Installed ``DTALite`` version, recorded as manifest provenance.

    Read from package metadata, never by importing the package (which would pull the
    banner + OpenMP engine into the host). The wheel bundles a compiled ``.so`` whose
    FW/VDF behavior can shift under the ``>=0.8`` floor, so CI pins ``==0.8.1`` and the
    running version is recorded here (adr-029)."""
    try:
        return importlib.metadata.version(_ENGINE_MODULE)
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - guarded above
        return "unknown"


@register_model
class DTALiteTapModel(TrafficAssignmentModel):
    """DTALite static Frank-Wolfe UE assignment (Zhou & Taylor 2014; ``heuristic``).

    A registered, CI-validated external-engine adapter. It reuses the ``heuristic``
    paradigm (the ``aon``/``sumo-marouter`` precedent): the Evaluator branches on the
    scenario's task fields, not the paradigm, so a heuristic-paradigm model earns the
    certified ``relative_gap`` on static scenarios exactly like ``aon``. Deterministic
    (the FW loop has no RNG -- byte-identical ``link_performance.csv`` across reruns at
    ``number_of_processors=1``); ``seedable=False`` (the engine exposes no seed to pin,
    unlike marouter's command line -- the RngBundle root seed still lands in the
    manifest as provenance). ``provides_gap=False``: the harness recomputes the gap;
    the engine's self-reported gap is recorded as provenance only (it uses a different
    ``(TSTT-SPTT)/SPTT`` normalization, measured, so it is NOT the repo gap).
    """

    name = "dtalite-tap"
    capabilities = Capabilities(
        paradigm="heuristic",
        deterministic=True,
        provides_gap=False,
        seedable=False,
    )
    factors = {
        "keep_files": FactorSpec(
            default=False, kind="bool",
            doc="Keep the generated DTALite working directory for debugging instead of "
            "deleting it (path stored on the model as ``last_workdir``).",
        ),
    }

    def __init__(self, **factor_overrides: object) -> None:
        super().__init__(**factor_overrides)
        self.last_command: list[str] = []  # for provenance / test inspection
        self.last_workdir: str | None = None

    def _refuse_unrepresentable(self, scenario: Scenario) -> None:
        """Raise ValueError naming the first field that makes the instance
        non-representable in DTALite's static ``assignment()`` (adr-029)."""
        for field in REFUSED_TASK_FIELDS:
            if getattr(scenario, field, None) is not None:
                raise ValueError(
                    f"dtalite-tap accepts only fixed-demand deterministic-UE scenarios; "
                    f"scenario '{scenario.name}' sets '{field}', which the engine's "
                    "static assignment() cannot represent in a certifiable form "
                    "(adr-029)."
                )
        # Generalized-cost fixed terms: the engine HAS a `toll` column, but its
        # toll/vot time-conversion semantics are unvalidated, and a negative toll KILLS
        # the host process. Refuse now, naming the field (the possible lift is recorded
        # in adr-029); never silently drop the cost the way marouter would.
        if np.any(np.asarray(scenario.network.fixed_cost, dtype=np.float64) != 0.0):
            raise ValueError(
                f"dtalite-tap cannot represent generalized-cost fixed terms; scenario "
                f"'{scenario.name}' has a nonzero fixed cost (toll_weight*toll + "
                "distance_weight*length); the engine's toll/vot conversion is "
                "unvalidated -- set toll_weight and distance_weight to 0 (adr-029)."
            )
        cap = np.asarray(scenario.network.capacity, dtype=np.float64)
        if np.any(cap < _MIN_CAPACITY):
            raise ValueError(
                f"dtalite-tap cannot represent scenario '{scenario.name}': a link "
                f"capacity < {_MIN_CAPACITY} is CLAMPED by the engine at fmax(0.1, cap) "
                "in the cost law only (measured), so the link would equilibrate under a "
                "different BPR while passing the read-back; a capacity <= 1e-4 is also "
                "silently dropped -- refuse rather than certify a wrong cost law (adr-029)."
            )

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        start = time.perf_counter()
        self._refuse_unrepresentable(scenario)

        # Budget mapping (P6). The engine hides its shortest-path (Dijkstra) count, so
        # an sp_calls-ONLY budget cannot bound the run and is refused up front rather
        # than silently running unbounded (the adr-027/028 pattern). iterations ->
        # number_of_iterations (floored at 1 -- a 0 makes the engine switch to
        # accessibility mode and emit an EMPTY link_performance, the marouter
        # iterations=0 lesson); wall_seconds -> one subprocess deadline;
        # target_relative_gap ignored (the engine has no gap-target setting -- it runs
        # a fixed count -- so it is disclosed, not honored).
        if budget.iterations is None and budget.wall_seconds is None:
            raise ValueError(
                "dtalite-tap cannot honor an sp_calls-only budget (the engine exposes "
                "no shortest-path count); constrain iterations or wall_seconds so the "
                "run is bounded (adr-029)."
            )
        # number_of_iterations = N runs N-1 Frank-Wolfe line-search iterations after an
        # initial all-or-nothing load, and labels the output row iteration_no=N
        # (measured off-by-one, adr-029). N=1 emits the pure AON (the honest near-AON
        # row); N=0 is accessibility mode (empty), so 1 is the floor.
        if budget.iterations is not None:
            n_iterations = max(1, int(budget.iterations))
        else:
            n_iterations = _DEFAULT_ITERATIONS
        deadline = start + budget.wall_seconds if budget.wall_seconds is not None else None

        # Zero routable demand: the exact equilibrium is the zero flow. Short-circuit
        # before writing any file or spawning the engine (the marouter precedent);
        # `last_command` stays empty (no engine run).
        od = np.asarray(scenario.demand.matrix, dtype=np.float64)
        if not np.any(od[~np.eye(od.shape[0], dtype=bool)] > 0):
            flows = np.zeros(scenario.network.n_links)
            coords = BudgetCoords(
                iterations=0, sp_calls=0, wall_ms=1000.0 * (time.perf_counter() - start)
            )
            # Emit BOTH self-report keys so the schema matches the engine-run path (the
            # zero-demand exact-equilibrium ran no engine, hence executed 0).
            trace.record(
                flows, coords,
                engine_configured_iterations=float(n_iterations),
                engine_iterations_executed=0.0,
            )
            return self._bundle(trace, rng)

        keep = bool(self.factor_values["keep_files"])
        workdir = tempfile.mkdtemp(
            prefix="tabench-dtalite-keep-" if keep else "tabench-dtalite-"
        )
        self.last_workdir = workdir
        try:
            # Phase 1 (in-host): write the GMNS CSVs. Counts against the SAME wall
            # deadline as the engine run (no phase is unbounded; the adr-027 review
            # MAJOR). Zero-demand is already short-circuited above.
            _write_gmns(scenario, workdir, n_iterations)

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
                # solution: RAISE, never launder into feasible=0.
                raise RuntimeError(
                    "DTALite exceeded the wall_seconds budget and was killed:\n  "
                    f"cmd: {' '.join(cmd)}"
                ) from exc
            except OSError as exc:
                raise RuntimeError(
                    f"DTALite could not be executed ({exc}):\n  cmd: {' '.join(cmd)}"
                ) from exc

            lp_path = os.path.join(workdir, _LP_FILE)
            # NEVER trust returncode alone: the engine exits 0 on missing files, dropped
            # links, zone!=node, garbage bytes (adr-029). A nonzero exit / timeout /
            # missing-or-empty output IS an engine failure (RuntimeError with tails);
            # success additionally REQUIRES the read-back to match every repo link.
            if proc.returncode != 0:
                raise RuntimeError(
                    f"DTALite failed (exit {proc.returncode}):\n  cmd: {' '.join(cmd)}\n"
                    f"  stdout tail: {proc.stdout[-600:]}\n"
                    f"  stderr tail: {proc.stderr[-600:]}"
                )
            if not os.path.exists(lp_path) or os.path.getsize(lp_path) == 0:
                raise RuntimeError(
                    "DTALite produced no link_performance.csv (or an empty one) while "
                    f"demand is positive:\n  cmd: {' '.join(cmd)}\n"
                    f"  stdout tail: {proc.stdout[-600:]}\n"
                    f"  stderr tail: {proc.stderr[-600:]}"
                )
            # The wall deadline must cover the read-back/parse phase too, not just the
            # engine subprocess (adr-029 review): check it once here (the engine may have
            # consumed nearly the whole budget) and again after the read-back (a large
            # network's parse is not free) -- otherwise a slow parse silently overruns.
            _check_deadline(deadline, cmd, "after the engine run, before read-back")
            try:
                flows = _parse_and_readback(lp_path, scenario.network)
            except _ReadBackError as exc:
                raise RuntimeError(
                    f"DTALite output read-back failed ({exc}); the engine did not parse "
                    f"the GMNS inputs as declared:\n  cmd: {' '.join(cmd)}\n"
                    f"  stdout tail: {proc.stdout[-400:]}\n"
                    f"  stderr tail: {proc.stderr[-400:]}"
                ) from exc
            _check_deadline(deadline, cmd, "during the read-back/parse phase")
            # Per-origin mass gate: the engine's ~1e7 BIG-M cost ceiling silently ZEROES
            # an OD whose congested cost exceeds it (measured single link, demand 1e4 past
            # cap=100 pow=4 -> 0.0), emitting a well-formed all-zero row that passes the
            # read-back and would otherwise launder to feasible=0. Each origin zone's
            # emitted outflow must cover its routable demand (through traffic only ADDS,
            # so a deficit is always an engine drop, never an infeasible solution).
            deficits = _origin_mass_deficits(flows, scenario.network, od)
            if deficits:
                raise RuntimeError(
                    "DTALite dropped demand from origin zone(s) [(zone, demand, emitted "
                    f"outflow)]: {deficits[:5]} -- the ~1e7 BIG-M cost ceiling zeroed an OD "
                    f"whose congested cost exceeded it:\n  cmd: {' '.join(cmd)}\n"
                    f"  stdout tail: {proc.stdout[-400:]}\n  stderr tail: {proc.stderr[-400:]}"
                )
            executed, engine_gap = _parse_summary(os.path.join(workdir, _SUMMARY_FILE))
        finally:
            if not keep:
                shutil.rmtree(workdir, ignore_errors=True)
                self.last_workdir = None

        coords = BudgetCoords(
            iterations=int(executed),  # engine-reported executed FW iterations (adr-029)
            sp_calls=0,  # the engine exposes no shortest-path count (disclosed, not hidden)
            wall_ms=1000.0 * (time.perf_counter() - start),
        )
        self_report = {
            # The configured setting and the executed FW-loop count (off by one: the
            # engine runs number_of_iterations - 1 line searches after the initial AON).
            "engine_configured_iterations": float(n_iterations),
            "engine_iterations_executed": float(executed),
        }
        if engine_gap is not None:
            # The engine's own gap = (TSTT - SPTT)/SPTT in percent (measured); the repo
            # RG normalizes by TSTT, so this is NOT the certified gap -- provenance only.
            self_report["engine_relative_gap"] = float(engine_gap)
        trace.record(flows, coords, **self_report)
        return self._bundle(trace, rng)

    def _bundle(self, trace: Trace, rng: RngBundle) -> ResultBundle:
        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info={**rng.describe(), "engine": _engine_version()},
        )


# --------------------------------------------------------------------------------------
# GMNS compilation and read-back (near-identity mapping; adr-029)
# --------------------------------------------------------------------------------------


class _ReadBackError(Exception):
    """The engine's echoed VDF parameters / link set do not match the declared GMNS."""


def _write_gmns(scenario: Scenario, workdir: str, n_iterations: int) -> None:
    """Write node/link/demand/settings CSVs for one static ``assignment()`` solve.

    The compile map is the IDENTITY on ``(free_flow_time, b, power, capacity)``: the
    engine's per-link VDF ``t = vdf_fftt*(1 + vdf_alpha*(V/(lanes*cap*period*plf))^
    vdf_beta)`` reduces to the repo BPR when ``lanes=1``, ``period=1 h``, ``plf=1`` and
    the VDF columns carry the repo values verbatim (measured, adr-029). Column names are
    lowercase (the engine is case-sensitive -- an uppercased column is silently ignored)
    and floats are written at full ``repr`` precision.
    """
    net = scenario.network

    # node.csv: zones are nodes 1..n_zones with zone_id == node_id (the hard engine
    # check); other nodes get zone_id 0. Coordinates are display-only (flows are
    # byte-identical without them, measured) -- scatter them so junction geometry is
    # non-degenerate.
    with open(os.path.join(workdir, "node.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["node_id", "zone_id", "x_coord", "y_coord"])
        for nid in range(1, net.n_nodes + 1):
            zid = nid if nid <= net.n_zones else 0
            w.writerow([nid, zid, float(nid), float((nid * 7) % 13)])

    # link.csv: rows MUST be written grouped/sorted by (from_node_id, to_node_id) -- the
    # engine builds its adjacency from contiguous FirstLinkFrom/LastLinkFrom ranges and
    # an ungrouped file silently corrupts routing (measured CRITICAL: permuted Braess
    # certifies wrong flows, permuted Sioux Falls sends the FW loop into an INFINITE
    # loop; adr-029). The read-back matches by node pair, so the emitted order is
    # transparent to it. lanes ALWAYS 1 with capacity = total link capacity (the lanes^2
    # trap); vdf_fftt overrides length/free_speed (harmless display dummies); toll = 0.
    order = sorted(
        range(net.n_links),
        key=lambda i: (int(net.init_node[i]), int(net.term_node[i])),
    )
    with open(os.path.join(workdir, "link.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["from_node_id", "to_node_id", "link_id", "lanes", "capacity", "free_speed",
             "length", "vdf_fftt", "vdf_alpha", "vdf_beta", "vdf_plf", "toll"]
        )
        for i in order:
            w.writerow([
                int(net.init_node[i]), int(net.term_node[i]), i + 1, 1,
                repr(float(net.capacity[i])), 60.0, 0.0,
                repr(float(net.free_flow_time[i])), repr(float(net.b[i])),
                repr(float(net.power[i])), 1, 0,
            ])

    # demand.csv: one row per positive off-diagonal OD cell (period total = veh/hour).
    od = np.asarray(scenario.demand.matrix, dtype=np.float64)
    n = od.shape[0]
    with open(os.path.join(workdir, "demand.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["o_zone_id", "d_zone_id", "volume"])
        for o in range(n):
            for d in range(n):
                if o != d and od[o, d] > 0:
                    w.writerow([o + 1, d + 1, repr(float(od[o, d]))])

    # settings.csv: single demand period 07:00-08:00 (1 h), determinism knobs, lean
    # assignment-only run (no route/vehicle/log output, ODME off). first_through_node_id
    # carries the scenario's TNTP first_thru_node verbatim.
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
            int(net.first_thru_node), 0, 0, 0, 0, 0, 0,
        ])


def _cell_float(row: dict, field: str, n_rows: int) -> float:
    """Parse one CSV cell to float, mapping a malformed/missing value to ``_ReadBackError``
    (never a raw ValueError/TypeError that would escape solve() and collide with the
    ValueError refusal channel; adr-029 review)."""
    raw = row.get(field)
    try:
        return float(raw)
    except (ValueError, TypeError) as exc:
        raise _ReadBackError(f"link row {n_rows}: unparseable {field}={raw!r}") from exc


def _parse_and_readback(lp_path: str, network) -> np.ndarray:
    """Parse ``link_performance.csv`` and verify the engine parsed the GMNS as declared.

    The engine echoes ``vdf_fftt/vdf_alpha/vdf_beta/vdf_plf/link_capacity`` per link -- a
    free compile-check channel (adr-027 discipline). This requires: every repo link
    matched exactly once by ``(from_node_id, to_node_id)``, the total row count equal to
    the link count (no phantom/dropped links), each echoed VDF parameter equal to the
    declared value within the engine's float32/4-decimal precision, a finite volume, and
    -- the LIVE gate (adr-029 review) -- the A2 cost-match: the engine's own
    ``travel_time`` equal to ``network.link_cost(flows)`` on EVERY link, which closes the
    echo check's sub-atol blind spot and catches a clamped/ignored cost law. Every cell
    conversion and the file read are wrapped so a corrupt-but-parseable engine output
    raises ``_ReadBackError`` (the caller maps it to the contract's RuntimeError), never a
    raw exception. Emitted ``volume``/``travel_time`` are 1e-4-quantized by the engine (a
    sub-5e-5 demand emits zero and certifies rg=0 within tolerance). Returns repo-native
    link flows from the ``volume`` column.
    """
    idx = {
        (int(network.init_node[i]), int(network.term_node[i])): i
        for i in range(network.n_links)
    }
    flows = np.full(network.n_links, np.nan)
    times = np.full(network.n_links, np.nan)
    seen = np.zeros(network.n_links, dtype=bool)
    n_rows = 0
    try:
        fh = open(lp_path, newline="", encoding="utf-8")
    except OSError as exc:  # pragma: no cover - the caller already checked existence
        raise _ReadBackError(f"cannot open link_performance.csv: {exc}") from exc
    try:
        reader = csv.DictReader(fh)
        required = {"from_node_id", "to_node_id", "volume", "travel_time", "vdf_fftt",
                    "vdf_alpha", "vdf_beta", "vdf_plf", "link_capacity"}
        try:
            fieldnames = set(reader.fieldnames or [])
        except (UnicodeDecodeError, csv.Error) as exc:
            raise _ReadBackError(f"undecodable link_performance.csv header: {exc}") from exc
        missing = required - fieldnames
        if missing:
            raise _ReadBackError(f"link_performance.csv missing columns {sorted(missing)}")
        try:
            rows = list(reader)
        except (UnicodeDecodeError, csv.Error) as exc:
            raise _ReadBackError(f"undecodable link_performance.csv body: {exc}") from exc
    finally:
        fh.close()
    for row in rows:
        raw_from = (row.get("from_node_id") or "").strip()
        if raw_from in ("", "0"):
            continue  # skip the engine's zero-id filler rows (dropped/empty links)
        n_rows += 1
        from_id = _cell_float(row, "from_node_id", n_rows)
        to_id = _cell_float(row, "to_node_id", n_rows)
        if not (np.isfinite(from_id) and np.isfinite(to_id)):
            raise _ReadBackError(f"link row {n_rows}: non-finite node id ({from_id}, {to_id})")
        key = (int(from_id), int(to_id))
        i = idx.get(key)
        if i is None:
            raise _ReadBackError(
                f"engine emitted link {key} absent from the scenario (phantom link)"
            )
        if seen[i]:
            raise _ReadBackError(f"engine emitted link {key} more than once")
        seen[i] = True
        volume = _cell_float(row, "volume", n_rows)
        if not np.isfinite(volume):
            raise _ReadBackError(f"link {key} has a non-finite volume {volume!r}")
        flows[i] = volume
        times[i] = _cell_float(row, "travel_time", n_rows)
        _check_echo(key, "vdf_fftt", _cell_float(row, "vdf_fftt", n_rows),
                    float(network.free_flow_time[i]))
        _check_echo(key, "vdf_alpha", _cell_float(row, "vdf_alpha", n_rows), float(network.b[i]))
        _check_echo(key, "vdf_beta", _cell_float(row, "vdf_beta", n_rows), float(network.power[i]))
        _check_echo(key, "vdf_plf", _cell_float(row, "vdf_plf", n_rows), 1.0)
        _check_echo(key, "link_capacity", _cell_float(row, "link_capacity", n_rows),
                    float(network.capacity[i]))
    if n_rows != network.n_links or not seen.all():
        missing_links = [k for k, i in idx.items() if not seen[i]]
        raise _ReadBackError(
            f"engine emitted {n_rows} links; scenario has {network.n_links} "
            f"(unmatched repo links: {missing_links[:8]})"
        )
    # A2 runtime cost-match: engine travel_time == repo BPR at the emitted flows on every
    # link (the identity map's payoff). This is the LIVE gate that closes the echo check's
    # sub-atol blind spot (a doctored/ignored fftt) AND catches the fmax(0.1,cap) capacity
    # clamp (measured rel ~0.93); a legit anchor's max rel is ~2e-5 (Barcelona ~7.6e-4).
    if not np.all(np.isfinite(times)):
        raise _ReadBackError("engine emitted a non-finite travel_time")
    repo_cost = network.link_cost(flows)
    a2 = np.abs(times - repo_cost)
    tol = _A2_RTOL * np.abs(repo_cost) + _A2_ATOL
    bad = np.nonzero(a2 > tol)[0]
    if bad.size:
        j = int(bad[np.argmax(a2[bad] - tol[bad])])
        key = (int(network.init_node[j]), int(network.term_node[j]))
        rel = a2[j] / max(abs(float(repo_cost[j])), 1e-30)
        raise _ReadBackError(
            f"A2 cost-match failed on link {key}: engine travel_time {times[j]!r} vs repo "
            f"BPR {float(repo_cost[j])!r} at flow {flows[j]!r} (rel {rel:.3g} > {_A2_RTOL}); "
            "the engine solved a different cost law (e.g. a clamped capacity)"
        )
    return flows


def _check_echo(key: tuple[int, int], name: str, echoed: float, declared: float) -> None:
    """Verify one echoed VDF parameter equals the declared value within the engine's
    float32/4-decimal precision (adr-029); raise ``_ReadBackError`` on a gross mismatch
    (an ignored column falling back to a default, or the lanes^2 capacity trap). NOTE the
    known sub-atol blind spot -- the runtime A2 cost-match above is the live backstop."""
    if not np.isfinite(echoed) or not np.isclose(
        echoed, declared, rtol=_READBACK_RTOL, atol=_READBACK_ATOL
    ):
        raise _ReadBackError(
            f"link {key}: engine echoed {name}={echoed!r} but declared {declared!r} "
            f"(rtol={_READBACK_RTOL}, atol={_READBACK_ATOL})"
        )


def _check_deadline(deadline: float | None, cmd: list[str], phase: str) -> None:
    """Raise the contract's RuntimeError if the wall deadline has passed. Threaded through
    every post-subprocess phase so a slow read-back/parse cannot silently overrun the
    ``wall_seconds`` budget the ADR promises to enforce end-to-end (adr-029 review)."""
    if deadline is not None and time.perf_counter() > deadline:
        raise RuntimeError(
            f"wall_seconds budget exhausted {phase}:\n  cmd: {' '.join(cmd)}"
        )


def _origin_mass_deficits(
    flows: np.ndarray, network, od: np.ndarray
) -> list[tuple[int, float, float]]:
    """Origin zones whose emitted outflow fails to cover their routable demand -- the
    ~1e7 BIG-M ceiling silently zeroes such an OD (adr-029 review). Through traffic only
    ADDS to a zone's outflow, so ``outflow(zone) >= productions(zone)`` always holds for a
    real solution; a shortfall past the loose tolerance is an engine drop, not a solution.
    Returns ``[(zone_id, productions, emitted_outflow), ...]`` (empty when all covered)."""
    v = np.asarray(flows, dtype=np.float64)
    outflow = np.bincount(network.init_node - 1, weights=v, minlength=network.n_nodes)
    off = od - np.diag(np.diag(od))
    productions = off.sum(axis=1)  # per zone (rows 0..n_zones-1 are zones 1..n_zones)
    deficits: list[tuple[int, float, float]] = []
    for z in range(network.n_zones):
        prod = float(productions[z])
        if prod <= 0:
            continue
        out = float(outflow[z])
        if out < prod * (1.0 - _MASS_RTOL) - _MASS_ATOL:
            deficits.append((z + 1, prod, out))
    return deficits


def _parse_summary(summary_path: str) -> tuple[int, float | None]:
    """Return ``(executed_fw_iterations, engine_relative_gap)`` from
    ``summary_log_file.txt``.

    The per-iteration trace lines read ``iter No = k, ... gap = g %``; the executed FW
    count is the last ``k`` (0 when only the initial AON ran, i.e. number_of_iterations
    = 1) and the self-reported gap is the last ``g/100``. Both fields are parsed into
    locals and committed together, so a format-drift line missing one field never leaves a
    TORN pair (executed from line k with a stale gap from line k-1; adr-029 review). The
    self-report is PROVENANCE ONLY: it uses a ``(TSTT-SPTT)/max(0.1, SPTT)`` normalization
    (not the repo RG) that on converged anchors equals ``RG/(1-RG)`` but can be NEGATIVE
    or frozen on a stalled instance -- it is never gated on. Missing/short file -> (0,
    None), never an error (the flows are already validated by the read-back)."""
    executed = 0
    engine_gap: float | None = None
    if not os.path.exists(summary_path):
        return executed, engine_gap
    with open(summary_path) as fh:
        for line in fh:
            if not line.lstrip().startswith("iter No ="):
                continue
            try:
                ex = int(line.split("iter No =")[1].split(",")[0])
                g = float(line.split("gap =")[1].split("%")[0]) / 100.0
            except (IndexError, ValueError):  # pragma: no cover - defensive
                continue  # torn/format-drift line: keep the last COMPLETE pair
            executed, engine_gap = ex, g
    return executed, engine_gap
