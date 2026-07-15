# ADR-032: simopt-profiles — SimOpt-style progress curves and solvability profiles

**Status:** accepted (implemented)
**Date:** 2026-07-15
**Deciders:** non-solver roadmap — redeeming the P5/P6 progress-curve / solvability-profile promise
**File:** `docs/design/adr-032-simopt-profiles.md`

## Context

SimOpt (Eckman, Henderson & Shashaani 2023, *SimOpt: A testbed for
simulation-optimization experiments*, INFORMS J. on Computing 35(2):495-508,
`eckman2023simopt`, tier-1 metric/protocol in `docs/references.json`) is one of
the three named design sources in `docs/ARCHITECTURE.md`. Its diagnostics —
progress curves, α-solve-time solvability profiles, Moré-Wild data profiles — are
defined not in the testbed paper but in its companion, **Eckman, Henderson &
Shashaani (2023), "Diagnostic Tools for Evaluating and Comparing
Simulation-Optimization Algorithms", INFORMS J. on Computing 35(2):350-367, DOI
[10.1287/ijoc.2022.1261](https://doi.org/10.1287/ijoc.2022.1261)** and, for data
profiles, **Moré & Wild (2009), "Benchmarking Derivative-Free Optimization
Algorithms", SIAM J. Optim. 20(1):172-191, DOI
[10.1137/080724083](https://doi.org/10.1137/080724083)**. Neither is in the
verified canon (which holds only the testbed paper, `eckman2023simopt`); this ADR
cites both freely in text — ADRs cite non-canon works — and does **not** grow the
canon (it stays at 246 references).

The architecture already **promises** these deliverables in normative language:
P5 (lines 98-114) says the deterministic track is "reported as progress curves
and Moré-Wild-style data profiles", the stochastic track carries "mean/quantile
progress curves and α-solve-time solvability profiles", and no-certificate models
appear "as censored entries in solvability profiles"; the v0.x roadmap lists
"progress-curve/solvability-profile plotting" as still open. This is the last
unredeemed piece of that text.

The experiment half of SimOpt already ships and was verified live against the
working tree: macroreplications with the fixed `(macrorep, source, replication)`
stream schema (P8, `core/rng.py`), hardware-free budget coordinates on every
checkpoint (`BudgetCoords`, P6), one certified CSV row per checkpoint (the
post-replication-equivalent P1 certificate — deterministic metrics recomputed
exactly, the probit residual certified through a pinned common Monte-Carlo sample,
adr-003), row-level censoring of infeasible flows, and the terminal percentile
bootstrap across macroreps (`experiments/bootstrap.py`). What was missing is the
*diagnostics* half: curve objects, α-solve times, solvability/data profiles,
difference profiles, functional bootstrap bands, and a profile artifact. That
missing piece is **pure post-hoc arithmetic over already-certified rows**, so it
needs no new certifier and no change to any solver, certifier, or the runner.

## Decision

Ship a new module `src/tabench/experiments/profiles.py` — at the same altitude as
`experiments/bootstrap.py` (aggregation over certified outputs), **not** in
`metrics/` (those are P1 certifiers over `(scenario, flows)`; profiles consume
already-certified rows). It reads either an in-memory `ExperimentResult` or the
runner's on-disk `{stem}.csv` + `{stem}.manifest.json` pair (`load_run`, which
owns the typed parse and the total-budget/scenario-hash the rows lack). Curves are
immutable `StepCurve` objects; profiles are plain data; the certified artifact is
`profiles.json`. No file in `models/`, `metrics/`, `core/`, or `runner.py`
changes; the runner keeps writing the same CSV+manifest, so the golden Braess hash
`cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d` is byte-identical
(re-asserted in `tests/test_profiles.py`).

Ten protocol decisions are **deliberate, disclosed deviations** from a literal
SimOpt port:

**D1 — the curve y-axis is the certified ranking metric, not SimOpt's
`(f-f*)/(f(x0)-f*)`.** SimOpt normalizes because raw objectives are incomparable
across problems and `f` is noisy; TABenchmark's certified relative gap (UE), SUE
fixed-point residual, or SO relative gap is *already* an absolute, scale-free,
harness-computed optimality measure with a true zero. Adopting it makes α-solve
mean "certified gap < α" — the repo's existing convergence-target language
(Boyce et al. 2004, `Budget.target_relative_gap`) — and avoids estimating
`f(x0)`/`f(x*)` entirely. The metric defaults to the track's ranking column read
from the manifest: `sue_fixed_point_residual` when `sue_theta` is set;
`so_relative_gap` **only when every model in the grid is system-optimum** — a
single `static_so` model in a mixed grid must NOT flip the UE solvers beside it
onto a column they do not populate (that would score their transient crossings as
SO convergence); mixed grids keep `relative_gap` and the SO column is requested
explicitly. A SimOpt-faithful Beckmann-normalized variant stays possible later (the
`beckmann_objective` column + best-known anchors exist); it is not v1 scope.

**D2 — the budget axis is a declared hardware-free coordinate** (P6): default
`sp_calls`, `iterations` accepted; `wall_ms` is allowed for descriptive progress
curves but **refused by every ranked profile** (cdf/quantile solvability, data
profiles). The α-solve time is reported in the axis's own work units, not hidden
behind a budget fraction. An axis that is 0 at every checkpoint of a model that
nonetheless produced a genuine (feasible, finite-metric) checkpoint is **refused**
(a learned model / `spsa-sumo` discloses `sp_calls=0` while doing real work, so
ranking it on `sp_calls` would be a degenerate curve, not a fast one) — but an
already-censored model at `sp_calls=0` is exempt, since it is `+inf` regardless of
the axis. The zero-axis guard is judged against the **caller's** metric, not the
default: an alternate metric can make an otherwise-censored model finite (and vice
versa), so the guard must see the same metric the curve will. `wall_ms` reaches
`progress_curves`/`solve_times` for descriptive curves and raw α-solve times only
— every ranked profile refuses it (P6: wall-clock is recorded, never ranked).
When the manifest budget does not constrain the chosen axis (an `iterations`-only
budget leaves `sp_calls=None`), the normalizer is the realized budget envelope (the
maximum observed axis value). This **differs** from SimOpt, whose
`problem.factors['budget']` is a *declared* factor fixed up front: the realized-max
fallback means adding a longer-running model to a grid rescales the normalized
solvability x-axis, so cdf/quantile profiles are comparable only within a fixed
grid composition — the raw-work-unit α-solve times (`solve_times`) are not
rescaled and stay directly comparable.

**D3 — strict `<` crossing.** `StepCurve.crossing_time(α)` is the first knot whose
value is strictly below α (SimOpt `Curve.compute_crossing_time` parity), `+inf`
when never crossed. Pinned by a knot-exact test.

**D4 — censoring stays in the denominator.** A checkpoint with `feasible=0` or a
non-finite metric becomes `y=+inf` (censored); a never-solved run has an `+inf`
solve time; **censored entries stay in every cdf denominator** so they never
inflate a profile (SimOpt parity; P5's "censored entries in solvability profiles"
made literal). A black box that emits infeasible flows is thus a first-class
censored row, never an error and never a leaderboard-topper.

**D5 — step semantics.** The last recommendation is carried to the budget end
(`curve_to_full_curve`); before a model's first checkpoint the curve is `+inf`
(censored), so there is no fictitious `t=0` solve. An early-converged trace (bfw
reaches its target in 3 checkpoints on Braess) carries its final value forward to
the realized budget end.

**D6 — the data-profile work unit is one all-or-nothing pass.** Moré-Wild's
`d_s(κ)` is the fraction of problems solved within `κ` work units, where one unit
is `n_p+1` function evaluations (a simplex gradient). The TA analog is one AON
pass = `n_origins` shortest-path trees, so κ = `sp_calls / n_origins` (Braess 1,
Sioux Falls 24). Because `n_origins` is a scenario property **not carried by the
certified row schema** (and the runner is unchanged), it is supplied by the caller
(default 1 ⇒ κ = raw work); a `work_unit` dict with a scenario key missing is
**refused**, never silently defaulted to 1 (that would mix raw `sp_calls` with
per-pass units in one profile). The convergence test is the certified metric
`<= τ` (a gap-based MW variant; MW's `f`-based test `f(x) <= f_L + τ(f(x0)-f_L)`
needs a common `x0`, which non-iterative paradigms do not share — disclosed). Each
`(scenario, macrorep)` pair is one equally-weighted problem: a deterministic model
contributes a single rep per scenario while a stochastic model contributes M, so a
mixed deterministic/stochastic grid weights the stochastic model's scenarios more
heavily in the pooled data profile — disclosed, not corrected (collapsing macroreps
to a per-scenario summary would discard the sampling spread the stochastic track
exists to show).

**D7 — one-level functional bootstrap** (macroreps only). SimOpt's second
resampling level draws post-replications; here the deterministic certificate has
zero estimation noise and the probit certificate is a *pinned protocol constant*
(adr-003: a fixed `r_cert` sample, CRN across everything in the run), so there is
no post-replication noise to resample. `bootstrap_curve_band` extends
`bootstrap.py`'s discipline — percentile, never parametric (P5), on the reserved
`SOURCE_BOOTSTRAP` stream, byte-reproducible from `root_seed` alone (P8).

**D8 — the certified artifact is `profiles.json`.** It carries the profile curves,
the protocol constants (metric, axis, α/τ/β, crossing rule, censoring rule, mesh),
and full provenance (scenario hashes, manifest budget blocks, seeds, tabench
version, git commit). Plots are rendering, never the artifact. A censored `+inf`
curve value is written as the JSON **string** `"Infinity"` (not a bare `Infinity`
token), so the artifact is strict RFC-8259 that any conformant consumer (jq, a
browser) parses; the reader restores it. A `NaN` never reaches the file — the curve
encoder refuses it and `json.dumps(allow_nan=False)` is the belt-and-suspenders.
The P1 story: profiles are deterministic pure functions of already-certified rows,
so there is no new trust surface and therefore no new certifier — correctness is
pinned by the closed-form tests below, provenance by the artifact schema.

**D9 — the β-quantile is SimOpt's exact estimator, made inf-aware.** The default
(`quantile_method="simopt"`) is byte-parity with SimOpt's `quantile_cross_jump` /
`quantile_of_curves`: `statistics.quantiles(values, n=100)[int(β·99)]`
(exclusive-interpolated). Because that estimator interpolates, a window that
reaches a censored `+inf` returns a non-finite quantile — which for a solvability
jump is the **flat-zero** (unsolvable) curve and for a progress-curve quantile is a
censored `+inf`; it is made inf-aware so it never poisons to `NaN`. Note this means
`{0.2, 0.5, ∞}` at β=0.5 is flat-zero (the median lands in the censored tail under
the exclusive interpolation), not a jump at 0.5 — the earlier draft's "parity"
claim while shipping the type-1 estimator was false and always flattered the solver.
The type-1 lower inverted-cdf `sorted[⌈β·n⌉−1]`, which never interpolates a censored
`+inf`, is retained as an opt-in `quantile_method="censoring-robust"` — the honest
alternative when interpolating into the censored tail is undesirable.

**D10 — solvability and data profiles require the full cross design.** SimOpt's
profiles assume every solver is run on every problem. A model absent from a
scenario would shrink only its own per-model denominator, so a solver that skipped
the hard scenarios could top the profile a solver run everywhere honestly trails.
`cdf_solvability` / `quantile_solvability` / `data_profile` therefore **refuse**
(naming the offending models and the scenarios they are missing from) when the
per-run model sets are incongruent; a single run is trivially congruent.

## Closed-form anchors

Every profile value is derivable by hand and recomputed in `tests/test_profiles.py`
(no BLAS-sensitive assertions):

- **α-solve regression (integration).** Braess `msa`/`fw`/`bfw` under
  `Budget(iterations=50)`, α=1e-4, on `sp_calls`: solve times **{msa: 5, fw: 24,
  bfw: 4}** (measured directly from the certified rows). All three share the AON
  start (iteration 1, `sp_calls=2`, gap 1.9118e-1, P7). `msa`'s crossing at
  `sp_calls=5` is a genuine *first* crossing even though its trace ends
  unconverged (6.5e-3) — the strict-`<` first-crossing semantics, not sustained
  convergence. `bfw`'s 3-checkpoint trace (`sp_calls` 2,3,4) carries forward to the
  realized budget end 51 (D5).
- **Strict-`<` at a knot (D3).** On `x=(1,2,3)`, `y=(0.5,0.2,0.1)`:
  `crossing_time(0.2)` strict = 3 (the knot valued exactly 0.2 does not count),
  non-strict = 2.
- **Censored denominator (D4).** cdf of solve times `{0.2, 0.5, ∞}`: terminal
  value **2/3**, the `∞` counted in the denominator.
- **Quantile parity (D9), checked against `statistics.quantiles`.** Under the
  default SimOpt estimator: `{0.2, 0.5, ∞}` at β=0.5 ⇒ **flat-zero** (the exclusive
  interpolation reaches the censored tail); `{0.2, 0.5, ∞, ∞}` at β=0.5 ⇒ flat-zero;
  `{0.1, 0.2, 0.3, ∞}` at β=0.5 ⇒ jump at **0.25**; `{0.2, 0.5, 0.9, ∞}` at β=0.5 ⇒
  jump at **0.7**. The `censoring-robust` opt-in gives jump at 0.5 / 0.2 for the
  first and third — the divergence the shipped "parity" claim had hidden.
- **Inf-honesty (M3).** `quantile_of_curves` and the bootstrap band over curves
  that carry `+inf` yield `+inf`, never `NaN`; a both-censored difference is the
  zero curve, a one-censored difference is `±inf`; a `NaN` reaching `write_profiles`
  raises.
- **Full-cross-design refusal (D10).** Two scenarios where model `B` is run only on
  the first ⇒ `cdf_solvability` raises naming `B` missing from the second (else `B`,
  tested on one easy scenario, would top the honest `A` tested on both).
- **Overshoot censoring (M7).** With a manifest `sp_calls` budget 10 and a crossing
  at `sp_calls=11`, the cdf and the quantile jump BOTH read it as censored (terminal
  0), never one counting and the other dropping it.
- **Moré-Wild staircase (D6).** Two scenarios × two solvers with solve work
  `A=(2,6)`, `B=(4,∞)`, unit 1: `d_A(2)=1/2`, `d_A(6)=1`, `d_B(4)=1/2`, terminal
  `d_B=1/2` (the never-solved problem stays in the denominator).
- **Difference profile.** A solver differenced against itself is the identically
  zero curve.
- **Bootstrap band.** Byte-deterministic in `root_seed`; identical macroreps give
  a zero-width band; the band brackets the mean curve; `M < 2` refuses.
- **Artifact round-trip.** In-memory rows and `load_run` of the written CSV+manifest
  pair yield identical profiles; the artifact is strict RFC-8259 (censored `+inf`
  as the string `"Infinity"`, no bare token) and carries the protocol constants and
  the scenario hash. A T2 estimation CSV raises a clear, named limitation.

## Alternatives considered

- **Put profiles in `metrics/`:** rejected — `metrics/*` are P1 certifiers over
  `(scenario, flows)`; profiles consume already-certified rows and add no trust
  surface. `experiments/` (the bootstrap altitude) is the honest home.
- **SimOpt-faithful Beckmann-normalized progress curves:** the `(f-f*)/(f(x0)-f*)`
  normalization needs estimated `f(x0)`/`f(x*)`; the certified gap is already the
  scale-free measure with a true zero (D1). A Beckmann variant is a clean later
  addition, not v1 scope.
- **A two-level bootstrap resampling post-replications:** the certificate has no
  post-replication noise to resample (D7); the single macrorep level is the whole
  sampling story here.
- **Normalizing every solve time to a budget fraction:** rejected as the primary
  report — α-solve times are reported in hardware-free work units (P6/D2); the
  budget-fraction normalization is used only inside the cross-scenario cdf mean,
  where different scenarios have different budgets.
- **A `tabench profiles <run.csv>...` CLI subcommand writing `profiles.json`:**
  deferred to a follow-up; the demo exercises the module end-to-end and the
  artifact writer is public.
- **Profiling the estimation (T2/T2d) track:** a **named follow-up, not free.**
  The T2 CSVs key rows on `estimator` (not `model`) and censor on `od_feasible`
  (not `feasible`), and rank on `heldout_count_rmse` — so a T2 pair needs a column
  mapping and `od_feasible` censoring, not just the shared budget-coordinate shape.
  Until that ships, `load_run` and the profile functions detect the T2 schema and
  raise a clear, named limitation rather than dying with `KeyError('model')`.

## Consequences

The benchmark redeems the last P5/P6 promise: certified progress curves,
α-solve-time cdf/quantile solvability profiles, Moré-Wild data profiles,
difference profiles, and functional bootstrap bands, all as a `profiles.json`
artifact — with zero changes to any solver, certifier, or the runner, no new
runtime dependency (numpy/scipy core; the demo's matplotlib is import-guarded),
and no external data or spend (P9 untouched). The golden Braess hash is
byte-identical. Follow-ups: the `tabench profiles` CLI, an estimation-track demo,
and the Beckmann-normalized progress-curve variant.

## Adversarial review

Three independent lenses (semantics-fidelity vs the fetched SimOpt master
source, closed-form hand-derivations + real-data behavior, numerics/API/
integration), each executing repros; every finding CONFIRMED and fixed with a
per-finding regression (streak: 20/20 sprints with at least one material
defect; 48 tests after the fixes, from 31). For pure reporting code the
false-certify analog is STATISTICAL DISHONESTY, and the review found four
material ways this module could flatter models:

**MAJORs, all fixed + pinned:** (a) — found by ALL THREE lenses — missing
(model, scenario) cells silently shrank per-model denominators, so a model
that skipped a hard scenario topped the profile (measured: honest 0.5 vs
selective 1.0, no warning; SimOpt's full cross design cannot express the
state) → cross-scenario profiles now REFUSE incongruent model sets (D10);
(b) the β-quantile estimator was NOT SimOpt's despite explicit parity claims,
diverging always in the flattering direction (type-1 lower inverted-CDF vs
`statistics.quantiles(n=100)[int(β·99)]` exclusive-interpolated with the
NaN→flat-zero censored branch — the draft's own closed-form anchor
contradicted measured SimOpt) → SimOpt-exact is now the default, pinned
in-test against `statistics.quantiles` itself, with the censoring-robust
type-1 variant as an explicit opt-in (D9); (c) censored macroreps
NaN-poisoned quantile curves and bootstrap bands (np's lerp computes
inf−inf) where the honest value is +inf → inf-aware percentiles throughout,
and a NaN reaching `write_profiles` raises; (d) the degenerate-axis guard
checked the DEFAULT metric rather than the REQUESTED one, so a zero-work
model topped a ranked profile on an alternate metric — the exact flattery D2
promises to refuse → the metric is threaded through the guard, both
directions pinned.

**MEDIUMs/MINORs, fixed:** one `static_so` model in a mixed grid flipped the
default metric for every model (now: all-static_so only); T2 estimation runs
crashed with a bare KeyError under a "generalizes for free" claim (now a
clear named refusal; T2 profiles are a named follow-up); cdf and quantile
disagreed on crossings past the realized budget envelope (both now censor);
M=1 bootstrap "confidence" bands refused; `profiles.json` was not RFC-8259
under censoring (+inf now the string `"Infinity"`, `allow_nan=False`, schema
`tabench-profiles-v1`); `work_unit` key misses, duplicate-x row-order
dependence, NaN x-knots, blank axis cells, and unknown metric names all
refuse loudly instead of degrading silently; the ADR's D-decisions were
corrected where the code or SimOpt source contradicted them.

**Survived (highlights):** the strict-< first-crossing rule verified
identical to the fetched `Curve.compute_crossing_time`; the re-crossing
honesty of first-crossing solve times demonstrated on a REAL oscillating
trace (braess msa: 3.6e-9 at sp=5 → 0.104 at sp=6 → terminal 6.5e-3) and
disclosed; the cdf estimator arithmetic-equivalent to SimOpt's
bisect-over-sorted-crossings; the braess anchor {msa:5, fw:24, bfw:4}
reproduced through the new code path against the scoping dossier's
measurement; left-endpoint step AUC, union-mesh means, carry-to-budget-end
and the +inf-before-first-checkpoint convention all as disclosed;
byte-determinism of the bootstrap band in root_seed; additive-only changes
to `bootstrap.py` and the `__init__` exports; the golden Braess hash
byte-identical.

## Sourcing

The SimOpt diagnostics semantics were measured against the library source
(`github.com/simopt-admin/simopt`, master: `simopt/curve.py`,
`simopt/curve_utils.py`, `simopt/experiment/post_normalize.py`,
`simopt/experiment/single.py`, `simopt/plots/solvability_profile.py`) and the two
definition papers above (Diagnostic Tools, DOI 10.1287/ijoc.2022.1261;
Moré & Wild 2009, DOI 10.1137/080724083). The strict-`<` crossing, the
censored-in-denominator cdf (`bisect_right(sorted_crossings, t) / n_curves`), the
flat-zero quantile, the union-mesh `mean_of_curves`, and the left-endpoint step
AUC are direct ports of that source, pinned by the closed-form anchors. The eight
deviations (D1-D8) are TABenchmark's, disclosed here; no SimOpt number is
reproduced (their curves are over their own simulation-optimization problems).
