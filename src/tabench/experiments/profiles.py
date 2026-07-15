"""SimOpt-style progress curves and solvability profiles (docs/design/adr-032).

Post-hoc reporting arithmetic over already-certified experiment rows: the
diagnostics half of the SimOpt design (Eckman, Henderson & Shashaani 2023,
"Diagnostic Tools for Evaluating and Comparing Simulation-Optimization
Algorithms", INFORMS J. on Computing 35(2):350-367) that TABenchmark's P5/P6
text already promises. The experiment half — macroreplications, the fixed
stream schema, hardware-free budget coordinates, per-checkpoint certified
progress, terminal bootstrap — is shipped by ``runner.py``/``bootstrap.py``;
this module consumes their output and never touches a solver, certifier, or the
runner (adr-032).

Everything here is a deterministic pure function of the certified CSV rows plus
the run manifest, so there is no new trust surface and no new certifier: the
values are pinned by the closed-form tests in ``tests/test_profiles.py`` and the
provenance by the ``profiles.json`` artifact schema.

Deliberate deviations from SimOpt are disclosed in adr-032 (D1-D8). The two that
shape the code:

* the curve **y-axis is the certified ranking metric** (relative gap / SUE
  fixed-point residual / SO relative gap), already an absolute scale-free
  optimality measure with a true zero, not SimOpt's ``(f-f*)/(f(x0)-f*)`` (D1);
* the **x-axis is a hardware-free work coordinate** (``sp_calls`` default,
  ``iterations`` accepted; ``wall_ms`` descriptive only, never in a ranked
  profile, P6/D2).

SimOpt semantics kept faithfully: strict-``<`` crossing (D3), censored entries
staying in every cdf denominator (D4), last recommendation carried to the budget
end and no fictitious pre-first-checkpoint solve (D5), the flat-zero curve for a
non-finite quantile, and union-mesh curve aggregation.
"""

from __future__ import annotations

import csv
import json
import math
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from statistics import quantiles
from typing import Any

import numpy as np

import tabench

from .bootstrap import bootstrap_curve_band

__all__ = [
    "StepCurve",
    "Run",
    "load_run",
    "progress_curves",
    "solve_times",
    "cdf_solvability",
    "quantile_solvability",
    "diff_profile",
    "data_profile",
    "mean_of_curves",
    "quantile_of_curves",
    "difference_of_curves",
    "bootstrap_progress_band",
    "write_profiles",
    "read_profiles",
    "run_provenance",
    "ARTIFACT_SCHEMA",
    "RANKED_AXES",
    "QUANTILE_METHODS",
]

INF = float("inf")
RANKED_AXES = ("iterations", "sp_calls")
_DESCRIPTIVE_AXES = ("wall_ms",)
ARTIFACT_SCHEMA = "tabench-profiles-v1"
# Quantile conventions (adr-032 D9): SimOpt's exclusive-interpolated estimator is
# the default (parity with statistics.quantiles(n=100)[int(beta*99)]); the
# censoring-robust type-1 lower inverted-cdf is an opt-in that never interpolates a
# censored +inf.
QUANTILE_METHODS = ("simopt", "censoring-robust")
# Columns unique to the T2 estimation CSV schema; profiles are an assignment-track
# reporting layer and refuse these until the column mapping ships (adr-032 D6).
_T2_COLUMNS = ("estimator", "od_feasible")


# --------------------------------------------------------------------------- I/O


def _to_float(value: Any) -> float:
    """Parse a CSV/in-memory cell to float; blank or unparseable -> NaN.

    Certified CSVs write floats as strings and censored cells as ``""``; the
    in-memory ``ExperimentResult.rows`` mixes floats and ``""``. Both collapse
    to the same typed value here so the round-trip (in-memory vs ``load_run`` of
    the written pair) produces identical curves (adr-032 D8, round-trip test).
    """
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


@dataclass
class Run:
    """One grid run: certified rows plus the manifest that carries the budget.

    Rows alone do not carry the total budget or the scenario hash, so profiles
    always pair them with the manifest (in memory or the on-disk ``{stem}.csv`` +
    ``{stem}.manifest.json`` pair, ``load_run``).
    """

    rows: list[dict[str, Any]]
    manifest: dict[str, Any]

    @classmethod
    def from_result(cls, result: Any) -> Run:
        """Wrap an in-memory :class:`ExperimentResult` (or any rows+manifest)."""
        return cls(list(result.rows), dict(result.manifest))


def load_run(csv_path: str | Path) -> Run:
    """Load a run from the runner's ``{stem}.csv`` + ``{stem}.manifest.json`` pair.

    The typed parse lives here (``csv.DictReader`` yields strings); the manifest
    supplies the total budget and scenario hash the rows lack.
    """
    csv_path = Path(csv_path)
    if csv_path.suffix != ".csv":
        raise ValueError(f"expected a .csv run file, got {csv_path.name!r}")
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or ())
        _reject_t2_schema(fields, source=csv_path.name)
        rows = list(reader)
    manifest_path = csv_path.with_name(csv_path.name[: -len(".csv")] + ".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest {manifest_path.name!r} not found next to {csv_path.name!r}; "
            "profiles need the manifest for the total budget and scenario hash"
        )
    manifest = json.loads(manifest_path.read_text())
    return Run(rows, manifest)


def _as_run(obj: Any) -> Run:
    if isinstance(obj, Run):
        return obj
    if hasattr(obj, "rows") and hasattr(obj, "manifest"):
        return Run(list(obj.rows), dict(obj.manifest))
    raise TypeError(
        "expected a Run, an ExperimentResult, or a rows+manifest object; "
        f"got {type(obj).__name__}"
    )


def _reject_t2_schema(columns: set[str], source: str) -> None:
    """Refuse a T2 estimation run with a clear, named limitation (adr-032 D6).

    The estimation tracks (``run_estimation_experiment`` / its dynamic sibling)
    key rows on ``estimator`` and censor on ``od_feasible``, not ``model`` /
    ``feasible``, and rank on ``heldout_count_rmse``. Profiles are an
    assignment-track reporting layer; T2 profiles need a column mapping and
    ``od_feasible`` censoring that are a scoped follow-up, so a T2 pair raises here
    rather than dying deep inside with ``KeyError('model')``.
    """
    if "model" not in columns and (set(_T2_COLUMNS) & columns):
        raise ValueError(
            f"{source} is a T2 estimation run (columns include "
            f"{sorted(set(_T2_COLUMNS) & columns)}, no 'model' column); "
            "experiments.profiles reports the assignment track only — T2 progress "
            "profiles (estimator/heldout_count_rmse axis + od_feasible censoring) "
            "are a scoped follow-up, not yet shipped (adr-032 D6)"
        )


def _require_assignment_schema(run: Run) -> None:
    """In-memory guard mirroring :func:`_reject_t2_schema` (adr-032 D6)."""
    columns: set[str] = set()
    for row in run.rows:
        columns |= set(row)
        break
    _reject_t2_schema(columns, source="this run")


def _require_congruent_model_sets(run_list: list[Run]) -> None:
    """Refuse a cross-scenario profile with incongruent model sets (adr-032 D10).

    SimOpt's solvability/data profiles assume the full cross design — every solver
    is run on every problem. A model absent from a scenario would otherwise shrink
    only its own per-model denominator, so a model that skipped the hard scenarios
    could top the profile a model run everywhere honestly trails. The per-run model
    sets must be identical; otherwise raise, naming the offending models and the
    scenarios they are missing from.
    """
    for run in run_list:
        _require_assignment_schema(run)
    if len(run_list) < 2:
        return
    per_run = [(str(r.manifest.get("scenario", f"run{i}")), _model_names(r))
               for i, r in enumerate(run_list)]
    union = set().union(*(names for _, names in per_run))
    incongruent: dict[str, list[str]] = {}
    for scenario, names in per_run:
        for missing in union - names:
            incongruent.setdefault(missing, []).append(scenario)
    if incongruent:
        detail = "; ".join(
            f"{m!r} missing from {sorted(sc)}" for m, sc in sorted(incongruent.items())
        )
        raise ValueError(
            "cross-scenario profiles require the full cross design (every model on "
            f"every scenario, adr-032 D10); incongruent model sets: {detail}"
        )


def _model_names(run: Run) -> set[str]:
    return {str(row["model"]) for row in run.rows}


# ----------------------------------------------------------------- step curves

_INF_TOKEN = "Infinity"
_NEG_INF_TOKEN = "-Infinity"


def _encode_y(value: float) -> float | str:
    """Encode a curve value for strict JSON: +inf -> "Infinity"; NaN is refused."""
    if math.isnan(value):
        raise ValueError(
            "NaN in a profile curve is a bug (censored is +inf, never NaN); "
            "refusing to serialize (adr-032 M3)"
        )
    if value == INF:
        return _INF_TOKEN
    if value == -INF:
        return _NEG_INF_TOKEN
    return value


def _decode_y(value: Any) -> float:
    """Decode a curve value from strict JSON (the "Infinity" tokens back to floats)."""
    if value == _INF_TOKEN:
        return INF
    if value == _NEG_INF_TOKEN:
        return -INF
    f = float(value)
    if math.isnan(f):
        raise ValueError("NaN in a profile artifact is not permitted (adr-032 M3)")
    return f


@dataclass(frozen=True)
class StepCurve:
    """An immutable right-continuous step function on a strictly increasing mesh.

    ``lookup(t)`` holds ``y[i]`` from knot ``x[i]`` until the next knot (SimOpt
    ``Curve`` semantics); below the first knot the value is ``+inf`` — a model
    that has not yet emitted a checkpoint is unsolved, never fictitiously solved
    at ``t=0`` (adr-032 D5). ``crossing_time`` is the first knot whose value is
    strictly below the threshold (D3); ``area`` is the left-endpoint step AUC.
    """

    x: tuple[float, ...]
    y: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.x) != len(self.y):
            raise ValueError(f"x and y differ in length: {len(self.x)} vs {len(self.y)}")
        if not all(math.isfinite(v) for v in self.x):
            raise ValueError("StepCurve x (work coordinate) must be finite: no NaN/inf knots")
        if any(math.isnan(v) for v in self.y):
            raise ValueError("StepCurve y must not be NaN (censored is +inf, never NaN)")
        if any(b <= a for a, b in zip(self.x, self.x[1:], strict=False)):
            raise ValueError("StepCurve x must be strictly increasing")

    def lookup(self, t: float) -> float:
        """Value of the step function at ``t`` (``+inf`` before the first knot)."""
        if not self.x or t < self.x[0]:
            return INF
        return self.y[bisect_right(self.x, t) - 1]

    def crossing_time(self, threshold: float, strict: bool = True) -> float:
        """First knot ``x[i]`` with ``y[i] < threshold`` (``<=`` if not strict).

        SimOpt's ``compute_crossing_time`` uses strict ``<`` (D3); the Moré-Wild
        convergence test uses ``<=`` (its "satisfies the test", D6). Returns
        ``+inf`` when never crossed (censored).
        """
        for xi, yi in zip(self.x, self.y, strict=True):
            if (yi < threshold) if strict else (yi <= threshold):
                return xi
        return INF

    def area(self) -> float:
        """Left-endpoint step area under the curve, ``sum(y_i * (x_{i+1}-x_i))``."""
        spans = zip(self.y, self.x, self.x[1:], strict=False)
        return float(sum(yi * (b - a) for yi, a, b in spans))

    def to_json(self) -> dict[str, list[Any]]:
        # x is finite by construction and coerced to float so the artifact is
        # byte-stable regardless of int/float construction; a censored y (+inf)
        # becomes the JSON string "Infinity" so the artifact is strict RFC-8259 (no
        # bare Infinity token) and round-trips exactly; a NaN refuses (adr-032 D8/M3).
        return {"x": [float(v) for v in self.x], "y": [_encode_y(v) for v in self.y]}

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> StepCurve:
        return cls(
            tuple(float(v) for v in obj["x"]),
            tuple(_decode_y(v) for v in obj["y"]),
        )


def _step_from_points(
    points: list[tuple[float, float, float]], carry_to: float | None
) -> StepCurve:
    """Build a step curve from ``(x, tiebreak, y)`` points.

    ``points`` need not be sorted; when two checkpoints share an ``x`` (work
    coordinate) the one with the larger ``tiebreak`` (the ``iterations`` column,
    i.e. the newer recommendation) wins — a deterministic rule independent of CSV
    row order (adr-032 D5). ``carry_to`` appends a terminal knot holding the final
    value to the budget end when it lies strictly beyond the last knot.
    """
    knots: dict[float, float] = {}
    for x, _tiebreak, y in sorted(points, key=lambda p: (p[0], p[1])):
        knots[x] = y
    xs = sorted(knots)
    ys = [knots[x] for x in xs]
    if carry_to is not None and xs and carry_to > xs[-1]:
        xs.append(carry_to)
        ys.append(ys[-1])
    return StepCurve(tuple(xs), tuple(ys))


# --------------------------------------------------------- manifest / axis reads


def _default_metric(manifest: dict[str, Any]) -> str:
    """The track's ranking column, from the manifest (adr-032 D1).

    SUE scenarios rank on the fixed-point residual; a grid whose models are **all**
    system-optimum ranks on the SO relative gap; everything else on the UE relative
    gap. A single ``static_so`` model in a mixed grid must NOT flip the default for
    the UE solvers beside it (they have no ``so_relative_gap``, and it would score
    their transient crossings as SO convergence) — mixed grids keep ``relative_gap``
    and the SO column is requested explicitly.
    """
    if manifest.get("scenario_sue_theta") is not None:
        return "sue_fixed_point_residual"
    models = manifest.get("models", {})
    paradigms = [spec.get("capabilities", {}).get("paradigm") for spec in models.values()]
    if paradigms and all(p == "static_so" for p in paradigms):
        return "so_relative_gap"
    return "relative_gap"


def _budget_end(run: Run, axis: str) -> float:
    """Total budget on ``axis``: the manifest value, else the realized envelope.

    The manifest budget need not constrain the chosen axis (e.g. an
    ``iterations``-only budget leaves ``sp_calls=None``), so the realized maximum
    axis value across every checkpoint is the fallback normalizer (SimOpt's
    ``problem.factors['budget']`` is likewise the realized total).
    """
    budget = (run.manifest.get("budget") or {}).get(axis)
    if budget is not None and float(budget) > 0:
        return float(budget)
    observed = [_to_float(r.get(axis)) for r in run.rows]
    top = max((v for v in observed if math.isfinite(v)), default=0.0)
    if top <= 0:
        raise ValueError(
            f"axis {axis!r} has no positive budget in the manifest and no positive "
            "observed value; cannot place a curve on it"
        )
    return top


def _validate_axis(run: Run, axis: str, ranked: bool, metric: str) -> None:
    """Guard axis abuse (adr-032 D2).

    A ranked profile refuses ``wall_ms`` outright (P6). Any profile refuses an
    axis that is zero at every checkpoint of a model that nonetheless produced a
    genuine (feasible, finite-metric) checkpoint — a learned model discloses
    ``sp_calls=0`` while doing real work, so ranking it on ``sp_calls`` would be
    a degenerate curve, not a fast one. An already-censored model (all
    checkpoints infeasible) is exempt: it is ``+inf`` regardless of the axis. The
    censoring is judged against the **caller's** ``metric``, not the default — an
    alternate metric can make an otherwise-censored model finite (and vice versa),
    so the guard must see the same metric the curve will (adr-032 D2).
    """
    if axis not in RANKED_AXES and axis not in _DESCRIPTIVE_AXES:
        raise ValueError(
            f"unknown axis {axis!r}; choose from {RANKED_AXES + _DESCRIPTIVE_AXES}"
        )
    if ranked and axis in _DESCRIPTIVE_AXES:
        raise ValueError(
            f"axis {axis!r} is wall-clock; P6 forbids ranking on it — "
            "use sp_calls or iterations (wall_ms is descriptive only)"
        )
    if axis not in RANKED_AXES:
        return
    per_model: dict[str, list[dict[str, Any]]] = {}
    for row in run.rows:
        # A blank/non-finite work coordinate is a data error, reported clearly here
        # rather than misread by the zero-axis guard below as "zero work" (M8e).
        if not math.isfinite(_to_float(row.get(axis))):
            raise ValueError(
                f"blank/non-finite {axis!r} cell for model {row.get('model')!r}; a work "
                "coordinate must be present and finite on every checkpoint (adr-032 M8e)"
            )
        per_model.setdefault(row["model"], []).append(row)
    degenerate = []
    for model, rows in per_model.items():
        has_finite = any(not _is_censored(r, metric) for r in rows)
        max_axis = max((_to_float(r.get(axis)) for r in rows), default=0.0)
        if has_finite and not (max_axis > 0):
            degenerate.append(model)
    if degenerate:
        raise ValueError(
            f"axis {axis!r} is 0 at every checkpoint for {sorted(degenerate)} despite a "
            "certified checkpoint; that axis does not measure their work (adr-032 D2) — "
            "rank on iterations instead"
        )


def _is_censored(row: dict[str, Any], metric: str) -> bool:
    """A checkpoint is censored when infeasible or its metric is not finite (D4)."""
    if _to_float(row.get("feasible")) == 0.0:
        return True
    return not math.isfinite(_to_float(row.get(metric)))


# ----------------------------------------------------------------- progress data


def progress_curves(
    run: Any,
    metric: str | None = None,
    axis: str = "sp_calls",
) -> dict[tuple[str, int], StepCurve]:
    """Per-(model, macrorep) progress curve: certified metric vs work coordinate.

    ``metric`` defaults to the track's ranking column read from the manifest
    (D1). Infeasible or non-finite-metric checkpoints become ``+inf`` (censored,
    D4); the final value is carried to the budget end (D5). ``axis`` is a
    hardware-free coordinate (``sp_calls`` default; ``wall_ms`` allowed here for
    descriptive curves but refused by every ranked profile, D2).
    """
    run = _as_run(run)
    _require_assignment_schema(run)
    metric = metric or _default_metric(run.manifest)
    _require_metric_column(run, metric)
    _validate_axis(run, axis, ranked=False, metric=metric)
    carry_to = _budget_end(run, axis)
    grouped: dict[tuple[str, int], list[tuple[float, float, float]]] = {}
    for row in run.rows:
        if "model" not in row or "macrorep" not in row:
            raise ValueError(
                "row is missing the 'model'/'macrorep' columns "
                "(adr-032: profiles read the assignment-track CSV schema)"
            )
        key = (str(row["model"]), int(_to_float(row["macrorep"])))
        x = _to_float(row.get(axis))
        if not math.isfinite(x):
            raise ValueError(
                f"blank/non-finite {axis!r} cell for model {row['model']!r}; a work "
                "coordinate must be present and finite on every checkpoint (adr-032 M8e)"
            )
        it = _to_float(row.get("iterations"))
        tiebreak = it if math.isfinite(it) else 0.0
        y = INF if _is_censored(row, metric) else _to_float(row.get(metric))
        grouped.setdefault(key, []).append((x, tiebreak, y))
    return {key: _step_from_points(pts, carry_to) for key, pts in grouped.items()}


def _require_metric_column(run: Run, metric: str) -> None:
    """Refuse an unknown metric name up front instead of silent flat-zero curves.

    A metric that is not a column in the run would make every checkpoint read as
    ``None`` (censored) and produce an all-flat-zero profile — an unhelpful lie for
    a typo. The column may still be entirely blank (e.g. ``so_relative_gap`` on a
    UE run); that is honest censoring, distinct from a name that does not exist.
    """
    columns: set[str] = set()
    for row in run.rows:
        columns |= set(row)
    if run.rows and metric not in columns:
        raise ValueError(
            f"unknown metric {metric!r}: not a column in this run. "
            f"Available columns: {sorted(columns)}"
        )


def solve_times(
    curves: dict[tuple[str, int], StepCurve], alpha: float, strict: bool = True
) -> dict[tuple[str, int], float]:
    """α-solve time per curve: first work coordinate with metric < α (``+inf`` if none).

    Reported in the axis's own work units (P6), inf-censored (D4).
    """
    return {key: curve.crossing_time(alpha, strict=strict) for key, curve in curves.items()}


# --------------------------------------------------------- curve algebra (SimOpt)


def _union_mesh(curves: list[StepCurve]) -> tuple[float, ...]:
    return tuple(sorted({x for c in curves for x in c.x}))


def mean_of_curves(curves: list[StepCurve]) -> StepCurve:
    """Union-mesh mean curve (SimOpt ``mean_of_curves``).

    Evaluated on the union of every input's knots; a censored (``+inf``) value
    propagates to the mean, so a mean progress curve over macroreps that include
    a never-solved run is honestly ``+inf`` there.
    """
    curves = list(curves)
    if not curves:
        raise ValueError("mean_of_curves needs at least one curve")
    mesh = _union_mesh(curves)
    ys = tuple(float(np.mean([c.lookup(t) for c in curves])) for t in mesh)
    return StepCurve(mesh, ys)


def _simopt_quantile(data: list[float], beta: float) -> float:
    """SimOpt's β-quantile: ``statistics.quantiles(data, n=100)[int(beta*99)]``.

    The exclusive-interpolated estimator SimOpt actually uses (``curve_utils``).
    A single point is its own quantile (``statistics.quantiles`` needs >=2). An
    interpolation window touching ``+inf`` yields ``+inf``, never ``NaN`` — the
    honest censored value (adr-032 D9/M3); the caller maps that to the flat-zero
    solvability curve or a censored progress value.
    """
    vals = list(data)
    if len(vals) == 1:
        return float(vals[0])
    qs = quantiles(vals, n=100)
    idx = min(max(int(beta * 99), 0), len(qs) - 1)
    q = qs[idx]
    return INF if math.isnan(q) else float(q)


def _pointwise_quantile(values: list[float], beta: float, method: str) -> float:
    """One mesh point's β-quantile under the requested convention (adr-032 D9)."""
    if method == "simopt":
        return _simopt_quantile(values, beta)
    if method == "censoring-robust":
        ordered = sorted(values)
        k = min(max(int(math.ceil(beta * len(ordered))), 1), len(ordered))
        return float(ordered[k - 1])
    raise ValueError(f"unknown quantile method {method!r}; choose from {QUANTILE_METHODS}")


def quantile_of_curves(curves: list[StepCurve], beta: float, method: str = "simopt") -> StepCurve:
    """Union-mesh pointwise β-quantile curve (SimOpt ``quantile_of_curves``, D9).

    ``method="simopt"`` (default) is parity with SimOpt's exclusive-interpolated
    ``statistics.quantiles(n=100)[int(beta*99)]``, made inf-aware so a censored
    ``+inf`` never poisons the result to ``NaN``. ``method="censoring-robust"`` is
    the type-1 lower inverted-cdf that never interpolates a censored ``+inf``.
    """
    curves = list(curves)
    if not curves:
        raise ValueError("quantile_of_curves needs at least one curve")
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta!r}")
    mesh = _union_mesh(curves)
    ys = tuple(_pointwise_quantile([c.lookup(t) for c in curves], beta, method) for t in mesh)
    return StepCurve(mesh, ys)


def _safe_diff(a: float, b: float) -> float:
    """Difference honest at infinities: both-censored -> 0, one-censored -> ±inf.

    ``inf - inf`` (or ``-inf - -inf``) is defined as ``0.0`` — two curves both
    censored at ``t`` differ by nothing — rather than the numpy ``NaN`` that would
    read as a spurious never-crossed value in the artifact (adr-032 M3).
    """
    if (a == INF and b == INF) or (a == -INF and b == -INF):
        return 0.0
    return a - b


def difference_of_curves(curve: StepCurve, reference: StepCurve) -> StepCurve:
    """Pointwise ``curve - reference`` on the union mesh (SimOpt difference profile)."""
    mesh = _union_mesh([curve, reference])
    ys = tuple(_safe_diff(curve.lookup(t), reference.lookup(t)) for t in mesh)
    return StepCurve(mesh, ys)


def diff_profile(profile: dict[str, StepCurve], reference_model: str) -> dict[str, StepCurve]:
    """Difference profiles vs a reference model; the reference maps to a zero curve."""
    if reference_model not in profile:
        raise ValueError(
            f"reference model {reference_model!r} not in profile keys {sorted(profile)}"
        )
    ref = profile[reference_model]
    return {model: difference_of_curves(curve, ref) for model, curve in profile.items()}


# ------------------------------------------------------------- solvability cdfs


def _cdf_from_times(times: list[float], domain_max: float) -> StepCurve:
    """cdf of solve times over ``[0, domain_max]``; censored inf stay in the denominator.

    ``cdf(t) = #{finite solve time <= t} / N`` with ``N`` the full count including
    the ``+inf`` (never-solved) entries — the SimOpt convention that censored
    runs never inflate the profile (adr-032 D4).
    """
    n = len(times)
    if n == 0:
        raise ValueError("_cdf_from_times needs at least one solve time")
    finite = sorted(t for t in times if math.isfinite(t))
    knot_xs = sorted({0.0, domain_max, *finite})
    xs, ys = [], []
    for t in knot_xs:
        if t > domain_max:
            continue
        xs.append(t)
        ys.append(bisect_right(finite, t) / n)
    return StepCurve(tuple(xs), tuple(ys))


def _quantile_jump(
    times: list[float], beta: float, domain_max: float, method: str = "simopt"
) -> StepCurve:
    """β-quantile solve time as a 0->1 jump curve; a non-finite quantile -> flat zero.

    ``method="simopt"`` (default) is exact parity with SimOpt's
    ``quantile_cross_jump``: ``statistics.quantiles(times, n=100)[int(beta*99)]``
    (exclusive-interpolated), and an infinite/NaN quantile — which happens whenever
    the interpolation window reaches a censored ``+inf`` — yields the flat-zero
    (unsolvable) curve. ``method="censoring-robust"`` is the type-1 lower
    inverted-cdf ``sorted[ceil(beta*n)-1]`` that never interpolates a censored
    ``+inf`` (adr-032 D9). The jump is clamped to ``domain_max``.
    """
    n = len(times)
    if n == 0:
        raise ValueError("_quantile_jump needs at least one solve time")
    if method == "simopt":
        q = _simopt_quantile(times, beta)
    elif method == "censoring-robust":
        ordered = sorted(times)
        k = min(max(int(math.ceil(beta * n)), 1), n)
        q = ordered[k - 1]
    else:
        raise ValueError(f"unknown quantile method {method!r}; choose from {QUANTILE_METHODS}")
    if not math.isfinite(q):
        return StepCurve((0.0, domain_max), (0.0, 0.0))
    q = min(q, domain_max)  # clamp to the domain (overshoot is censored upstream)
    if q <= 0.0:
        return StepCurve((0.0, domain_max), (1.0, 1.0))
    if q >= domain_max:
        return StepCurve((0.0, q), (0.0, 1.0))
    return StepCurve((0.0, q, domain_max), (0.0, 1.0, 1.0))


def _per_model_solve_times(
    run: Run, alpha: float, axis: str, metric: str | None, strict: bool
) -> tuple[dict[str, list[float]], float]:
    """Normalized (fraction-of-budget) solve times per model, for one run.

    A crossing beyond the realized budget envelope is censored (``+inf``), so the
    cdf and the quantile jump treat an overshoot identically — never a
    normalized-time > 1 that one profile counts and the other drops (adr-032 M7).
    """
    metric = metric or _default_metric(run.manifest)
    curves = progress_curves(run, metric=metric, axis=axis)
    norm = _budget_end(run, axis)
    raw = solve_times(curves, alpha, strict=strict)
    by_model: dict[str, list[float]] = {}
    for (model, _macrorep), t in raw.items():
        value = t / norm if (math.isfinite(t) and t <= norm) else INF
        by_model.setdefault(model, []).append(value)
    return by_model, norm


def cdf_solvability(
    runs: Any,
    alpha: float,
    axis: str = "sp_calls",
    metric: str | None = None,
    strict: bool = True,
) -> dict[str, StepCurve]:
    """cdf-solvability profile per model over normalized budget fraction ``[0, 1]``.

    Per scenario: the cdf of α-solve times over macroreps (censored in the
    denominator, D4). Across scenarios: the union-mesh mean of the per-scenario
    cdf curves (SimOpt ``mean_of_curves``). ``axis`` must be a ranked coordinate
    (P6).
    """
    run_list = _run_list(runs)
    _require_congruent_model_sets(run_list)
    for run in run_list:
        _validate_axis(run, axis, ranked=True, metric=metric or _default_metric(run.manifest))
    per_scenario: dict[str, list[StepCurve]] = {}
    for run in run_list:
        by_model, _ = _per_model_solve_times(run, alpha, axis, metric, strict)
        for model, times in by_model.items():
            per_scenario.setdefault(model, []).append(_cdf_from_times(times, 1.0))
    return {model: mean_of_curves(curves) for model, curves in per_scenario.items()}


def quantile_solvability(
    runs: Any,
    alpha: float,
    beta: float,
    axis: str = "sp_calls",
    metric: str | None = None,
    strict: bool = True,
    quantile_method: str = "simopt",
) -> dict[str, StepCurve]:
    """β-quantile solvability profile per model over normalized budget fraction.

    Per scenario the β-quantile solve time as a 0->1 jump (flat zero when the
    quantile is censored); across scenarios the union-mesh mean.
    ``quantile_method`` selects SimOpt's exclusive-interpolated estimator (default)
    or the censoring-robust type-1 variant (adr-032 D9).
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta!r}")
    run_list = _run_list(runs)
    _require_congruent_model_sets(run_list)
    for run in run_list:
        _validate_axis(run, axis, ranked=True, metric=metric or _default_metric(run.manifest))
    per_scenario: dict[str, list[StepCurve]] = {}
    for run in run_list:
        by_model, _ = _per_model_solve_times(run, alpha, axis, metric, strict)
        for model, times in by_model.items():
            per_scenario.setdefault(model, []).append(
                _quantile_jump(times, beta, 1.0, method=quantile_method)
            )
    return {model: mean_of_curves(curves) for model, curves in per_scenario.items()}


# --------------------------------------------------------- Moré-Wild data profiles


def data_profile(
    runs: Any,
    tau: float,
    axis: str = "sp_calls",
    work_unit: float | dict[str, float] = 1.0,
    metric: str | None = None,
) -> dict[str, StepCurve]:
    """Moré-Wild data profile per model (Moré & Wild 2009, adr-032 D6).

    ``d_s(kappa)`` is the fraction of problems — here ``(scenario, macrorep)``
    pairs, pooled across ``runs`` — that solver ``s`` drives to the convergence
    test ``certified metric <= tau`` within ``kappa`` work units, where one work
    unit is an all-or-nothing pass (the TA analog of Moré-Wild's ``n_p+1``
    simplex-gradient budget). ``work_unit`` is that pass's cost in axis units per
    scenario (``n_origins`` with positive demand); it is a scenario property not
    carried by the certified row schema, so it is supplied by the caller and
    defaults to ``1`` (kappa = raw work). Censored problems stay in the
    denominator (D4).
    """
    run_list = _run_list(runs)
    _require_congruent_model_sets(run_list)
    for run in run_list:
        _validate_axis(run, axis, ranked=True, metric=metric or _default_metric(run.manifest))
    kappas: dict[str, list[float]] = {}
    for run in run_list:
        scenario = str(run.manifest.get("scenario", ""))
        if isinstance(work_unit, dict):
            if scenario not in work_unit:
                raise ValueError(
                    f"work_unit has no entry for scenario {scenario!r} (keys "
                    f"{sorted(work_unit)}); a silent fallback would mix raw sp_calls with "
                    "per-pass units in one profile — supply every scenario's AON-pass unit"
                )
            unit = float(work_unit[scenario])
        else:
            unit = float(work_unit)
        if unit <= 0:
            raise ValueError(f"work_unit for scenario {scenario!r} must be positive, got {unit}")
        curves = progress_curves(run, metric=metric, axis=axis)
        for (model, _macrorep), curve in curves.items():
            work = curve.crossing_time(tau, strict=False)
            kappas.setdefault(model, []).append(work / unit if math.isfinite(work) else INF)
    domain = max(
        (k for ks in kappas.values() for k in ks if math.isfinite(k)),
        default=1.0,
    )
    return {model: _cdf_from_times(ks, domain) for model, ks in kappas.items()}


# ----------------------------------------------------------- functional bootstrap


def bootstrap_progress_band(
    curves: list[StepCurve],
    root_seed: int,
    b: int = 10000,
    level: float = 0.95,
) -> tuple[StepCurve, StepCurve]:
    """Functional percentile bootstrap band of the mean curve over macroreps (D7).

    Resamples the macrorep curves with replacement on the reserved
    ``SOURCE_BOOTSTRAP`` stream (extending ``bootstrap.py``'s discipline: one
    level, macroreps only — the certificate has no post-replication noise to
    resample, D7), takes the pointwise central ``level`` percentiles of the
    resampled mean curve on the fixed union mesh, and returns ``(lo, hi)`` step
    curves. Byte-reproducible in ``(curves, root_seed, b, level)``; identical
    macroreps give a zero-width band.
    """
    curves = list(curves)
    if len(curves) < 2:
        raise ValueError(
            "bootstrap_progress_band needs >= 2 macrorep curves (a single curve has "
            "no sampling spread; mirrors bootstrap_ci's values.size > 1 gate, adr-032 M8a)"
        )
    mesh = _union_mesh(curves)
    matrix = np.array([[c.lookup(t) for t in mesh] for c in curves], dtype=float)
    lo, hi = bootstrap_curve_band(matrix, root_seed=root_seed, b=b, level=level)
    return StepCurve(mesh, tuple(map(float, lo))), StepCurve(mesh, tuple(map(float, hi)))


# ------------------------------------------------------------- the profiles.json


def _run_list(runs: Any) -> list[Run]:
    if isinstance(runs, (list, tuple)):
        return [_as_run(r) for r in runs]
    return [_as_run(runs)]


def run_provenance(runs: Any) -> list[dict[str, Any]]:
    """Per-run provenance block for the artifact: scenario hash, budget, seed."""
    provenance = []
    for run in _run_list(runs):
        m = run.manifest
        provenance.append(
            {
                "scenario": m.get("scenario"),
                "scenario_hash": m.get("scenario_hash"),
                "scenario_family": m.get("scenario_family"),
                "budget": m.get("budget"),
                "seed": m.get("seed"),
                "macroreps": m.get("macroreps"),
                "git_commit": (m.get("environment") or {}).get("git_commit"),
            }
        )
    return provenance


def write_profiles(
    out_path: str | Path,
    profiles: dict[str, dict[str, StepCurve]],
    protocol: dict[str, Any],
    provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write the certified ``profiles.json`` artifact (adr-032 D8).

    The artifact is profile curves keyed ``{kind: {model: curve}}`` plus the
    protocol constants (metric, axis, α/τ/β, crossing rule) and full provenance
    (scenario hashes, manifest budgets, seeds, tabench version). Plots are
    rendering, never the artifact. A censored ``+inf`` curve value is written as
    the JSON **string** ``"Infinity"`` (not a bare token), so the artifact is
    strict RFC-8259 that any conformant consumer (jq, a browser) can parse; the
    ``read_profiles`` reader restores it. ``json.dumps(allow_nan=False)`` is the
    belt-and-suspenders: any ``NaN``/bare-``Infinity`` that escaped the curve
    encoders raises rather than emitting a non-conformant token (adr-032 D8/M3/M8b).
    """
    doc = {
        "schema": ARTIFACT_SCHEMA,
        "tabench_version": tabench.__version__,
        "protocol": protocol,
        "provenance": provenance,
        "profiles": {
            kind: {model: curve.to_json() for model, curve in by_model.items()}
            for kind, by_model in profiles.items()
        },
    }
    Path(out_path).write_text(json.dumps(doc, indent=2, allow_nan=False))
    return doc


def read_profiles(path: str | Path) -> tuple[dict[str, Any], dict[str, dict[str, StepCurve]]]:
    """Load a ``profiles.json`` artifact back to (document, {kind: {model: curve}})."""
    doc = json.loads(Path(path).read_text())
    profiles = {
        kind: {model: StepCurve.from_json(cj) for model, cj in by_model.items()}
        for kind, by_model in doc["profiles"].items()
    }
    return doc, profiles
