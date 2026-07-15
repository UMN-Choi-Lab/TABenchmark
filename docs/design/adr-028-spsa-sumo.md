# ADR-028 — spsa-sumo: SPSA calibration against a production simulator, as one more estimator row

**Status:** accepted (shipped in v0.2)
**File:** `docs/design/adr-028-spsa-sumo.md`

## Context — the T2 thesis, extended one sentence

T2 (ADR-002) shipped a demand-free estimation contract: anything that emits an
OD matrix — a 1980 balancing loop, a GLS solve, a neural inverse — is scored by
the identical **pinned-bfw certifier**, which recomputes count-fit and OD-fit
from the emitted OD through *its own* reference assignment (P1, ADR-002 Decision
2). The runner even hard-refuses any non-`bfw` certificate pin
(`runner.py`), and "which inner solver an estimator uses" is already a declared
factor paid from that estimator's own budget (`base.py`).

Balakrishna, Ben-Akiva & Koutsopoulos (2007) — *Offline calibration of dynamic
traffic assignment: simultaneous demand-and-supply estimation* (TRR 2003:50–58,
canon `balakrishna2007offline`) — established the **black-box
simulator-in-the-loop** calibration paradigm: optimize a non-differentiable,
possibly stochastic DTA simulator's inputs by SPSA (Spall 1992), two simulator
runs per iteration. The shipped `spsa` row already implements Spall's SPSA, but
its "black box" is the repo's own in-process MSA/AON oracle. This ADR makes the
loop **real**: the inner assignment oracle is the shipped `sumo-marouter`
adapter (ADR-027) — a subprocess production engine with its own hardcoded linear
cost law, real refusal surfaces, and byte-deterministic output. The T2 thesis
gains one clause: *…or a production simulator calibration loop.* That is a **row,
not a new question**.

## Decision 1 — an estimator, not a task family; the certifier is untouched

`spsa-sumo` is a `SumoSPSAEstimator(SPSAEstimator)` registered in
`ESTIMATOR_REGISTRY` behind a guarded import (Decision 4). It reuses **every**
piece of `spsa` — the log-space demand parametrization, Spall gains, the
per-component clip, best-iterate tracking, sparse pinned-certificate checkpoints
— and overrides only the oracle hook `_assign_obs` (build a private inner
`Scenario(network=task.network, demand=od_from_pairs(...))`, solve it with
`SumoMarouterModel`, return the sensor flows) and `_sp_cost_per_eval` (Decision
3), plus an `estimate` prelude for the up-front refusals and the wall deadline.

A **sumo-pinned certifier was considered and REJECTED.** Scoring estimators
through one simulator's mapped law would destroy the shared cross-estimator
scale ADR-002 exists to protect (every other estimator is scored through `bfw`),
and it is needed for nothing: the certifier recomputes fit from the emitted OD
regardless of how that OD was produced. A scratchpad pilot confirmed **zero
changes** to the task, runner, or certifier — the subclass ran through the
UNCHANGED `run_estimation_experiment` + `ODCertifier` and certified
`od_feasible=1` on two-route. That is the point of the row.

## Decision 2 — DEMAND-ONLY; the joint contribution is honestly not shipped

Scope is **demand only**, on the unchanged `EstimationTask`. Balakrishna et
al.'s **joint demand+supply** estimation — their title contribution — is **NOT
shipped**. Option (b) joint calibration changes what *truth* is (the task
network stops being ground truth), needs a new emitted artifact (supply
parameters), a new certificate surface (score capacities through *what* pinned
map? *what* identifiability report?), truth-side supply-perturbed instance
generation, and would immediately also want W-SPSA and within-day dynamics — an
entire **task-family ADR of its own**, not a rider. The formulation research
verified joint recovery end-to-end (prior (4.5,1.1) → (6.04,0.80), 121 marouter
runs) *and* found two load-bearing negatives that would have to ship as
executable caveats of that future family — see Decision 6. This ADR defers all
of it.

The name `spsa-sumo` follows repo convention (methods are named: `spsa`, `gls`,
`od-congested`, `od-kalman`). The **"DTA" in the paper's title is its setting**,
not this row's: `marouter` is a *static* macroscopic assignment, so this is the
H=1 special case of the thesis's time-dependent formulation (Eq. 3.9 holds the
supply parameters constant; here there are none). The within-day analogue needs
the `duaIterate` dynamic adapter ADR-027 explicitly deferred (a different
equilibrium concept) plus `DynamicEstimationTask` coupling. **Counts only**, not
speeds: `LinkCounts` is the observation level (`marouter`'s netload does carry a
traveltime attribute, so a speeds channel — the thesis's SD(c)→SD(cs) step — is
honest future work needing a new observation level). This row ships the
published counts-only variant.

## Decision 3 — the fabricated-`sp_calls` trap and the budget contract

`marouter` exposes no shortest-path (Dijkstra) count, so `spsa-sumo` **discloses
`sp_calls = 0`** rather than fabricating it from a meaningless `k_inner`. The
inherited loop charges `2 * k_inner` per iteration at two sites (`spsa.py`); a
`_sp_cost_per_eval()` **hook** (parent returns `k_inner`, subclass returns `0`)
lets the loop be reused verbatim without over-reporting — the *only* change to
`spsa.py` (+4 lines, behavior-identical for the parent). Correspondingly, an
`sp_calls`-**only** budget cannot bound the loop and is **refused up front**
(mirror of `sumo_marouter.py`, the inverted ADR-025 wall_seconds lesson); the
loop is bounded by the `iters` factor and by any `iterations`/`wall_seconds`
budget. The CLI T2 dispatch (`cli.py`) previously built `Budget(sp_calls=…)`
only; it now passes the card's `iterations`/`wall_seconds` through when present
(absent → `None` → byte-identical budget for the classical rows).

**One wall deadline is threaded across ALL 2I+1 inner solves.** An engine
`RuntimeError` (timeout/crash) inside any evaluation **aborts** `estimate()` by
the crash-vs-censor discipline — infrastructure failure is never laundered into
`feasible=0`. A tight `wall_seconds` therefore kills a single mid-loop solve
with the engine command in the message (ADR-027's wall-kill), not the loop with
an opaque error.

## Decision 4 — the FIRST guarded estimator

`eclipse-sumo` is an optional extra. `estimation/__init__.py` wraps `from
.spsa_sumo import SumoSPSAEstimator` in the exact `models/__init__.py` guard —
`try/except ModuleNotFoundError`, re-raise unless `exc.name == "sumo"`,
conditional `__all__` append. Registration is decorator-time, so on a core
install the `@register_estimator` never runs and `ESTIMATOR_REGISTRY` / `tabench
list` simply lack the name (`tabench` still imports; `tabench run --models
spsa-sumo` errors cleanly with "Unknown estimator …; see `tabench list`"). This
is the estimator-side twin of the ADR-027 model guard; the 731-test numpy suite
runs without the wheel as the live regression.

## Decision 5 — the P1 honesty-diff REFRAMED as a measured bias (not dishonesty)

The certifier reports the estimator's self-reported `obs_count_rmse`
(`self_obs_count_rmse`) alongside its own recomputed `obs_mean_count_rmse` — the
ADR-002 Decision-2 honesty diff. For a white-box estimator scored through its own
math, these agree in the mean reduction. For `spsa-sumo` **they are EXPECTED to
differ at the mapping-floor scale**: SPSA equilibrates the demand under
`marouter`'s *mapped/SUE* law (`logit_theta = 200` default, calibrated on the
asymmetric two-route anchor per ADR-027 — **never re-tuned on Braess**, the
theta-tuning trap), while the certificate re-assigns the emitted OD under the
DECLARED BPR via `bfw`. The diff is therefore the **measured
simulator-in-the-loop bias** — the T2 transport of ADR-027's simulator-to-
benchmark gap. It is **not a hard bound**: measured `|self - certified|` is
~7e-4 count-RMSE on the clean two-route anchor and ~2e-3 under poisson counts —
the **same order** as the ADR-027 mapping floor (itself a relative gap, ~1.7e-4
Braess / ~5.4e-4 two-route), quoted here only as a scale, not a ceiling on an
absolute count-RMSE difference. It is meaningful **only when no box projection
engaged**: a binding box makes the emitted and evaluated points coincide by
construction (Decision 6), so the diff stays at the mapping-floor scale rather
than blowing up. It is provenance, **not** an estimator-dishonesty signal, and no
gate fires on it (verified: none added). The loss-landscape corollary: the
count-misfit minimizer under `marouter`'s law is slightly offset from the
`bfw`-certified optimum, so anchors pin only **improves-on-prior** directional
bounds at pinned seeds, never tight cross-platform decimals (the BLAS lesson).

**Anchor fragility (honestly disclosed).** The shipped recovery anchor is pinned
on **clean counts** (`noise="none"`) at a fixed seed, where SPSA drives count and
demand RMSE ~5× below the stale prior. Under **poisson** counts (3 periods) the
same seed neither meets that count-fit bound (~0.7× improvement, not 0.19×) nor
improves demand RMSE (it can degrade vs the prior — measured up to ~11× worse at
adverse seeds): small-sample count noise plus the mapping bias, a fragility
**shared by any count-matching estimator** on this 1-pair instance, not a
`spsa-sumo` defect. A `poisson`-noise negative-control test pins that the noisy
variant stays a certifiable-but-not-improving row so it can never be quietly
promoted into the ranked clean anchor.

## Decision 6 — the box constraint, and the corner-plateau caveat (scoped)

The thesis imposes box constraints (Eqs. 3.5–3.6) enforced by projection **twice
per iteration**, and `spsa-sumo` implements **both**. The box is carried as
`demand_lo_frac` / `demand_hi_frac` factors (`[lo_frac, hi_frac] × prior`
elementwise; default lower `0×` — inert, log space already gives `g > 0` — and
upper `100×`, wide so demand-only recovery is unclamped). Projection is exposed to
the parent loop through two hooks that are **identity in `SPSAEstimator`** (so the
`spsa` row is byte-for-byte unchanged, verified with the reviewers'
parent-identity harness across scenarios × budgets × seeds × factors × macroreps,
zero drift) and overridden in `SumoSPSAEstimator`:

- **`_project(g)` — step 5, evaluation-time.** Applied to each perturbed
  candidate `g_plus`/`g_minus` **in the parent loop, BEFORE `loss()` and before
  best-iterate tracking**, so the evaluated, tracked and emitted demand are one
  array. The emitted best-iterate is therefore in-box **by construction** and its
  self-reported `obs_count_rmse` describes the same point the certifier scores
  (P1 emitted == evaluated). `_assign_obs` no longer clips.
- **`_project_log(u)` — step 8, iterate-time.** Applied to the log-demand iterate
  **after the update step**, clamping it to `[log lo, log hi]` so the iterate
  cannot drift far outside the box and freeze on a flat corner where both
  perturbations clamp to one boundary (`ghat == 0` from the deterministic oracle).

This replaces the first draft's evaluation-time-only clip inside `_assign_obs`,
which the three-lens review falsified: the parent tracked the **unclipped**
candidate as best while its loss was measured at the clipped point, so a binding
box emitted an **out-of-box** OD (2.67 vs 3.49) whose self-report described a
different point (honesty diff ~0.43, ~800× the mapping floor) **and** froze the
iterate out-of-box for most iterations. Both are gone: the box-binding regression
pins, on a seed where the box binds (ceiling below truth), that the emitted demand
is elementwise `≤ hi_frac × prior` and the honesty diff collapses to the
mapping-floor scale — and it FAILS under a clip-removal mutation (verified before
shipping).

The dossier's **corner-plateau negative finding is a SUPPLY phenomenon**: without
bounds, log-space SPSA on the *capacity scale* collapses into the bypass-
saturated corner (`s < 0.675`) where `marouter` flows are constant `(3,3,0,3,3)`
and the count loss is flat at `0.2354` (zero gradient from a deterministic
oracle). It is therefore a caveat of the **deferred joint task family**, not of
this demand-only row — the two disjoint routes of the shipped two-route anchor
have no such saturation corner. For demand-only, the shipped safeguards are
log-space positivity + the tightened `step_clip` default `0.5` (the plateau-
escape value, vs `spsa`'s `1.0`) + the two-sided box projection above; the
executable regression pins that when the box binds (an adversarially far prior +
tight ceiling) the emitted demand respects the box and the self-report stays
consistent with it. When the joint family ships, the box + the `z3` prior term
become load-bearing on the *supply* axis and the supply corner-plateau ships as
its own regression.

## Wall math (measured on this box, sumo 1.27.1 wheel)

- Per inner solve: two-route **0.21–0.23 s**, Braess 0.64–0.73 s (netconvert
  compile dominates). SPSA = **2I+1** adapter solves.
- CI (two-route only, never Braess, never wall-time asserts): the recovery
  anchor (20 iters, +prior baseline) ≈ 9 s; the bit-repro pair, the poisson
  negative control, and the box-binding regression add the rest — the whole
  `test_spsa_sumo.py` runs in ~55 s uncontended (projecting ~2 min on a 2–3×
  GitHub runner), inside the sumo job's 2–4 min budget. Anchors stay on two-route
  at ≤ 30 iters.
- Compile-once caching would give a measured 5–7× but is **deferred**: the OD
  window and flow-scale are *demand-dependent* even though the compiled net is
  not — exactly the silent-corruption surface behind the ADR-027 CRITICAL.

## Honest sourcing

- **Primary (attributed, PDF unread):** Balakrishna, Ben-Akiva & Koutsopoulos
  (2007), TRR 2003:50–58, DOI 10.3141/2003-07 — canon-verified via Crossref. Its
  contents are established from Balakrishna's **MIT PhD thesis read in full**
  (DSpace 1721.1/35120; the TRR paper condenses thesis chs. 3+5 — same authors,
  same formulation, same LA South Park case study per the TRID abstract) and
  cross-verified word-for-word against **Lu's open MIT W-SPSA thesis** ch. 3
  (DSpace 1721.1/88395), the same attributed-unread convention as
  `implicit-ue-nn` / `het-gnn`.
- **NOT shipped and stated as such:** the joint demand+supply contribution
  (Decision 2); within-day DTA (`marouter` is static; the `duaIterate` analogue
  is deferred); speeds (counts only); the W-SPSA weighting matrix (Lu et al. 2015
  TR-C, attributed unread); any online/rolling-horizon calibration; a sumo-pinned
  certificate; a netconvert compile cache; **gradient smoothing** (the thesis
  option of averaging `ghat` over multiple `Delta` draws / past iterations — the
  single-draw `ghat` is shipped, exactly like the `spsa` row).
- Spall (1992) is attributed through the existing `spsa` row, unchanged. The
  `marouter` engine provenance (PTV-Validate / VISUM-Cologne vdf lineage) is
  ADR-027's, unchanged.

## Consequences

- One new registered estimator (guarded), `src/tabench/estimation/spsa_sumo.py`
  (~250 lines). `spsa.py` gains three identity hooks (`_sp_cost_per_eval`,
  `_project`, `_project_log`) wired into the reused loop — behavior-identical for
  the parent (verified byte-for-byte with the reviewers' parent-identity harness).
  `cli.py` passes T2 `iterations`/`wall_seconds` through. No new paradigm token,
  certificate column, scenario field, `Evaluator`/`ODCertifier` branch, or hash
  change — the golden Braess hash `cf00f411…` is re-asserted byte-identical.
- `tests/test_spsa_sumo.py` (13 tests, module `importorskip('sumo')`, two-route
  only): registry+capabilities, golden hash, the `sp_calls`-only refusal and the
  `sp_calls=0` disclosure, the delegated power/fixed-cost envelope refusals, a
  pinned-seed clean-count recovery anchor with a loose improves-on-prior bound
  certified through the unchanged certifier, a **poisson negative control** (the
  noisy variant certifies but does NOT improve — the disclosed fragility), the
  **box-binding regression** (emitted `≤ hi_frac × prior`, self-report consistent,
  and it fails under a clip-removal mutation), bit-reproducibility + macrorep
  divergence, the wall-kill `RuntimeError`, and sparse checkpointing. The CI
  `sumo` job's pytest line becomes the explicit two-file list (the torch-job
  precedent).
- Scenario-domain narrowness: the row runs only on `power == 1`, toll-free,
  fixed-demand UE instances — **the adapter's documented envelope IS the
  estimator's**, refused up front by delegation to the adapter's own
  `_refuse_unrepresentable`.

## Adversarial review

Three independent lenses (soundness, formulation, numerics), each executing
Python/pytest/marouter; every finding CONFIRMED by a runnable repro and fixed
with a per-finding regression (streak: 17/17 sprints with at least one material
defect; 13 spsa-sumo tests after the fixes, from 12).

**MAJOR (all three lenses converged): the first draft's evaluation-time-only
box clip broke P1 — emitted ≠ evaluated.** The draft clipped inside
`_assign_obs`, invisible to the parent loop: `loss()` was measured at the
CLIPPED point while best-iterate tracking stored the RAW candidate, so a
binding box emitted an **out-of-box** OD carrying another point's loss.
Reviewers measured emitted `2.60` vs box ceiling `2.0` (and `8.10` vs `3.82`
on a second config), a self-report describing a different point than the one
certified (honesty diff ~0.43 count-RMSE, ~800× the mapping-floor scale the
docstring claimed bounded it), and — the mechanism twist — the box itself
CREATED the frozen corner plateau the dossier warned about: with both
perturbations clamping to one boundary the deterministic oracle returns
`ghat == 0`, and the un-projected iterate sat out-of-box for 22 of 25
iterations. The shipped guard test could not fail (its box never bound; its
"projection" assertion was a factors-dict tautology). FIXED structurally with
the thesis's own two projections surfaced to the parent loop as hooks that are
**identity in `SPSAEstimator`**: `_project(g)` (step 5) applied to every
candidate BEFORE `loss()` and best-iterate tracking, and `_project_log(u)`
(step 8) applied to the iterate after the update; `_assign_obs` no longer
clips (Decision 6). Verified post-fix: the reviewers' own repro now emits
exactly AT the ceiling with the self-report matching the emitted point to
1e-9; the parent-identity harness shows **zero drift** for the `spsa` row
across scenarios × budgets × seeds × factor sets × macroreps (full traces,
coords, self-reports, certified rows); and the new box-BINDING regression
(seed 14, ceiling below truth) fails under a clip-removal mutation
(mutation-verified: emitted 3.487 > box 2.669 with the clip removed).

**MINORs, all fixed:** the docstring/ADR claim that the honesty diff is
"bounded by the mapping floor" was false and unit-incommensurate (an absolute
count-RMSE compared against a relative gap) — reworded everywhere as a
MEASURED bias (~7e-4 clean / ~2e-3 poisson) of the same ORDER as the floor,
never a bound, meaningful only when no projection engaged; the recovery
anchor's noise fragility was undisclosed (under poisson counts the same seed
can degrade demand RMSE vs the prior, measured up to ~11× at adverse seeds) —
now disclosed in Decision 5 + VALIDATION and pinned by a poisson
negative-control test that keeps the noisy variant certifiable-but-not-
improving; the bit-reproducibility test ran a redundant third experiment
(trimmed to two, one of which carries both macroreps); stale doc counts and
wall claims (test count, ~130 → ~250 lines, "well under a minute" → measured
~55 s) corrected and **gradient smoothing** added to the NOT-shipped list; the
adapter's tempdir-hygiene test false-failed under concurrent local sessions
sharing `/tmp` (glob window caught a sibling session's workdirs) — now runs
against a private monkeypatched tempdir.

**Survived (highlights):** the parent `spsa` row byte-identical through the
hook refactor (the reviewers' identity harness, zero drift — re-run
independently after the fix batch); ZERO certificate/task/runner changes (the
pilot's point: the unchanged pinned-bfw certifier scores the emitted OD
regardless of which simulator produced it) with the golden Braess hash
re-asserted; `sp_calls` honesty (disclosed 0, never fabricated; the
sp_calls-only budget refused up front) under attack; the single wall deadline
threading all 2I+1 inner solves (a mid-loop kill raises with the engine
command, never `feasible=0`); the guarded registry on a simulated core
install (`import tabench` works, the name is simply absent); the CLI T2 card
path with the iterations passthrough; bit-reproducibility across processes
with macrorep divergence; and the clean-count recovery anchor with measured
ratio ~0.19 against its loose 0.6 improves-on-prior bound.
