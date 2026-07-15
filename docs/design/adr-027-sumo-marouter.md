# ADR-027 — SUMO marouter: the first external-simulator adapter, and the simulator-to-benchmark model gap

**Status:** accepted (shipped in v0.2)
**File:** `docs/design/adr-027-sumo-marouter.md`

## Context — Phase 4 opens, and a stale assumption is overturned

Phases 1–3 shipped white-box solvers, day-to-day dynamics, DTA/DNL primitives,
estimators, and three learned models. Phase 4 is the first **external
simulator**: Lopez et al. (2018) "Microscopic Traffic Simulation using SUMO"
(IEEE ITSC, canon `lopez2018microscopic`, tier 1), the tool paper for the SUMO
package whose macroscopic assignment tool `marouter` this adapter wraps.

`TASKS.md` recorded that this row "cannot validate in CI." **That assumption is
now false and this ADR corrects it.** `eclipse-sumo` is pip-installable as a
binary wheel (`eclipse-sumo==1.27.1`, manylinux, ~150 MB) that ships the real
`marouter`/`netconvert` ELF binaries *inside the package* at
`sumo.SUMO_HOME/bin`. So the adapter can be a fully CI-validated benchmark model
behind an optional extra — exactly the `torch`-extra precedent from adr-025 — and
the whole pilot below was verified by running SUMO 1.27.1 on this box.

This is **act four** of the accuracy-vs-certificate story (act 1 ridge: accuracy
≠ certificate; act 2 implicit-ue-nn: feasibility ≠ equilibrium quality; act 3
het-gnn: the feasibility-mechanism gradient). Act 4: a **production simulator's
macroscopic assignment**, run to its own convergence under its own hardcoded cost
law, emits perfectly demand-feasible flows (audit passes, mass exact) whose
certified relative gap under the scenario's DECLARED BPR costs is real but small.
The row is the first number for the **simulator-to-benchmark model gap** that
practitioners implicitly set to zero when calibrating against SUMO.

## Decision 1 — dependency mechanics (the adr-025 torch precedent)

- **Extra:** `sumo = ["eclipse-sumo>=1.27"]` in `[project.optional-dependencies]`,
  a **floor not a pin**. A hard `==` in library metadata conflicts with user
  environments and blocks fixes. The het-gnn unpinned-dep lesson is solved where
  it belongs — **in the CI workflow**, which installs `eclipse-sumo==1.27.1`
  before `pip install -e ".[dev,sumo]"` (already-satisfied → no re-resolve), so
  the hardcoded-vdf mapping-floor anchors are pinned to the tested engine.
- **Guard:** `models/adapters/__init__.py` and `models/__init__.py` wrap
  `from .sumo_marouter import SumoMarouterModel` in
  `try/except ModuleNotFoundError`, re-raising unless `exc.name == "sumo"`. On a
  core install the model is simply not registered, so `MODEL_REGISTRY` /
  `tabench list` lack it and the register-model invariant holds. The sumo-free
  matrix legs are the live regression that `import tabench` works without it.
- **Binary discovery is `sumo.SUMO_HOME`-only.** Never PATH / `shutil.which` /
  the ambient `SUMO_HOME` env var: this very box ships a stale
  `SUMO_HOME=/opt/sumo-1.12` beside the 1.27.1 binaries — the exact discovery
  hazard. Every subprocess gets absolute wheel-binary paths and
  `env={**os.environ, "SUMO_HOME": sumo.SUMO_HOME}`.
- **CI: a FOURTH job `sumo`**, a clone of the torch-job shape (py3.12, pip cache
  keyed `pip-sumo-<hash(pyproject)>`, data cache + prefetch + `TABENCH_REQUIRE_DATA=1`,
  pinned install, then only `pytest tests/test_sumo_marouter.py`). **Not folded
  into the torch job** (rejected): folding couples a ~150 MB sumo download onto
  the torch cache key and makes either dependency's breakage kill the other's
  signal. The job runs in ~2–4 min (marouter solves the anchors in < 0.5 s).

## Decision 2 — the model contract

`SumoMarouterModel` (`sumo-marouter`), **registered** with class-level
capabilities, in `src/tabench/models/adapters/sumo_marouter.py`.

- **Paradigm: reuse `"heuristic"`** (the `aon` precedent). The Evaluator branches
  on scenario task fields, not paradigm, so a heuristic-paradigm model earns the
  certified `relative_gap` on static scenarios exactly like `aon`. A new
  `external` token is a core `PARADIGMS` change buying zero harness behavior;
  external-ness is provenance (the manifest records the engine version). **No
  `capabilities.py` change.**
- **`deterministic=True, provides_gap=False, seedable=True`.** marouter's
  SUE/logit/MSA path uses **no RNG** — netload is byte-identical across seeds AND
  reruns — so the seed is drawn from the `RngBundle` and pinned on the command
  line for provenance, and `--routing-threads 1` is unconditional (byte-
  determinism verified only single-threaded). Caveat: the anchors were measured on
  the `manylinux_2_28` x86_64 wheel (glibc ≥ 2.28, what ubuntu-latest CI selects);
  the `manylinux2014` build for older glibc was not measured.
- **Budget (P6).** `iterations → --max-iterations` (floored at **1** — a 0 cap
  makes marouter emit the all-zero flow gawron/lohse are refused for). `wall_seconds`
  is threaded as a **single deadline through BOTH the netconvert compile phase and
  the marouter run** (the review MAJOR: the compile phase was silently unbounded,
  and combined with a lane explosion the overrun was unbounded — a tiny wall now
  raises promptly in either phase). `sp_calls` is unmappable (marouter exposes no
  Dijkstra count): `coords.sp_calls = 0`, and an **sp_calls-ONLY budget raises
  `ValueError` up front** — never silently unbounded (the adr-025 `wall_seconds`
  MAJOR, inverted). ONE checkpoint. `coords.iterations` records the configured cap
  with a `self_report` disclosure `engine_iterations_executed_unknown=1` — marouter
  does not report the executed count (`marouter -v` was checked once: no iteration
  line), so this is an honest over-report flagged, not hidden. `target_relative_gap`
  is ignored (single shot; `--tolerance` is SUE stability under ITS costs, not the
  repo gap).
- **Error surfaces, three distinct.** (a) missing `sumo` → absent from registry;
  (b) netconvert / marouter nonzero exit, timeout, or unparseable netload →
  **`RuntimeError`** with command + stderr tail — an engine crash is an
  infrastructure failure and must NEVER be laundered into `feasible=0`; (c)
  emitted-but-bad flows → the normal harness censor path, no adapter logic. Zero
  off-diagonal demand → in-adapter exact-zero short-circuit before any
  subprocess.
- **Hygiene.** Each solve runs in a `tempfile` working dir with prefix
  `tabench-sumo-` (cleanup guaranteed on raise via `finally`); the `keep_files`
  factor keeps it for debugging (`last_workdir`). Never writes to CWD/repo.

## Decision 3 — the compile map and the representability floors (the central honest content)

marouter does **not** take a user BPR function. Its congestion law is **hardcoded
per road class** (`src/marouter/ROMAAssignments.cpp`: `getCapacity` +
`capacityConstraintFunction`, "based on the definitions in PTV-Validate and in
the VISUM-Cologne network"); with `--capacities.default` every edge uses one
default class whose latency is **linear in flow**, measured on 1.27.1:

    t(f) = t0 * (1 + K * f / C),   t0 = length / speed   (free-flow seconds)
    speed > 26 m/s : C = 1400 * numLanes,  K = 2
    speed <=  5 m/s : C =  200 * numLanes,  K = 6

A repo link with **linear** latency `t(v) = A + B v` (`power == 1` only; `A=fft`,
`B=fft·b/cap`) compiles, at time scale `tau` (s/native-cost) and flow scale `s`
(veh/h per native flow), to a SUMO edge solving

    t0    = A·tau                        -> length   = A·tau·speed
    slope = t0·K/C = tau·B/s             -> numLanes = A·K·s / (B·cap_per_lane)

`s = s_base · m`, where `s_base` is the smallest scale making **every
representable link's lane count an exact integer** (so those links match the repo
BPR to machine precision — the A2 anchor) and `m` an integer scaling the forced
intercepts below tolerance (integrality is preserved for any `m`).

**THREE representability bounds are unavoidable** (slope/intercept coupled through
`t0`; lane counts and edge lengths physically bounded). Each is validated BEFORE
any file is written and refused loudly — never silently clamped or capped (the
review CRITICAL/MAJORs):

- **Zero-intercept links** (`A ≈ 0`, `B > 0`): the true intercept is 0 but every
  edge has `t0 = tau·B·(C/K)/s > 0` (`speed ≤ 5` band, `C/K = 200·lanes/6`). Forced
  abstract intercept **`eps = B·(200·lanes/6)/s`** must stay ≤ `_INTERCEPT_FLOOR_TOL`
  AND its edge length `5·tau·eps` must clear netconvert's silent **0.1 m minimum-
  length clamp** — so the eps-edge lane count is chosen to satisfy both, refusing
  when it cannot. (The clamp was the review CRITICAL: at default factors a sub-0.1 m
  eps-edge compiled to 0.1 m, silently 3× the intercept AND slope, scoring wrong
  flows with false mapping-floor provenance.)
- **Zero-slope links** (`B ≈ 0`, `A > 0`): every edge has `slope > 0`; the parasitic
  abstract slope **`A·K·s/C`** is minimized with many lanes, bounded ≤
  `_PARASITIC_SLOPE_TOL` (default 0.004 → 1000 lanes on the two-route legs) —
  **refused** (not silently capped at `_MAX_LANES`) when the cap cannot reach it.
- **Lane quantization**: `numLanes` must be an exact positive integer ≤ `_MAX_LANES`.
  The flow scale is rationalized with a **bounded denominator** (`10**6`, not the
  `10**12` that chased binary-double noise) and the coefficient reproduced to
  relative `1e-9`; a generic decimal parameter (whose `s`/lane count explodes) is
  refused up front — a documented capability limit, not an hours-long netconvert
  hang. `s = s_base · m` (`s_base` the lcm making every representable link integral;
  `m` bounds the forced intercepts and honors `min_lanes`).

Two silent-failure hazards of the engine are caught rather than trusted: (1) the
**compile read-back** reparses `net.net.xml` and verifies every lane's
`length`/`speed`/`numLanes` equals the declared value, raising `RuntimeError` on
mismatch (the 0.1 m clamp backstop, and any other geometry rewrite); (2) marouter
reverts an edge to **free-flow when a path's cumulative time exceeds the OD window**
(a hardcoded 1 h by default — it collapsed Braess to AON at `demand=360`), so the
`$OR;D2` window is **sized from a worst-path bound** under the mapped law, the trip
count is **scaled by the window so the flow RATE (and thus the equilibrium) is
window-invariant** (recovered by `flow_scale = s·window`), a `No interval matches`
stderr is treated as `RuntimeError`, and the netload is required to carry exactly
one interval (multi-interval would silently drop mass).

Other mapping pitfalls, all hit and neutralized: netconvert rounds declared lengths
to `--precision` (default 2 → 0.8% slope error on sub-metre edges; **use 6**);
geometry-based connection pruning silently drops turns on an abstract net (it
collapsed Braess to a single path) → the converter emits an **explicit connection
file** allowing every in→out movement (minus U-turns), geometry-independent;
`--weights.minor-penalty` defaults to 1.5 s (**zeroed**); TAZ source/sink on the
zone's **boundary edges** conserves flow exactly; netload is read at `--precision 9`;
flows are read from the `entered` attribute (exact macroscopic doubles), **never**
the integerized route file; `theta_sumo = theta_repo/tau` (per second); **tolls /
generalized-cost fixed terms** (`toll_weight·toll + distance_weight·length`) have no
marouter hook and are **refused** (they would be silently dropped).

Because the certificate recomputes every metric repo-side from `v = entered/flow_scale`
in native units, `tau` and `s` divide out of **mass** exactly (link flows are native)
— but `tau` sets the compiled edge lengths (too small → the 0.1 m clamp) and the path
times vs the OD window (too large → free-flow revert), so its factor bounds are the
validated envelope `[0.2, 30]`, backstopped by the read-back and window sizing (P1/P9).

## Measured anchors (all certified by the harness under the true BPR, P1)

- **A1 Braess** (SUE, logit, `theta=200`, `beta=gamma=0`, `paths=4`,
  `tolerance=1e-7`): flows `(3.998, 2.002, 1.996, 2.002, 3.998)` vs oracle
  `(4,2,2,2,4)`; certified `relative_gap ≈ 1.74e-4`, `node_balance ~7e-14`,
  `feasible=1`. This **equals the analytic mapping floor** (the exact UE of the
  eps-perturbed costs; the pilot measured 1.727e-4 at `s=14000`) — marouter
  contributes essentially zero extra error. A converged `bfw` certifies `~1e-16`
  (the white-box solver wins the convergence axis, as expected).
- **A2 cost-matched anchor (the hardest risk).** marouter's internal `traveltime`
  equals the repo BPR cost at the emitted flows on **representable** links to
  ≤ `1e-6` (measured `~2e-10` Braess, `~4e-10` two-route). This SEPARATES the
  mapping floor (the eps/parasitic perturbations on the non-representable links)
  from any solver error — without it the certified gap is not interpretable. The
  row is not shipped without A2.
- **A3 two-route UE-approx** (asymmetric; deterministic UE `f_A = 2.5` at `D=4`):
  `feasible=1`, `relative_gap ≈ 5.4e-4 < 1e-3`; `bfw` orders better. `theta` is
  **calibrated on THIS anchor, never Braess** — Braess's UE is the equal logit
  split at any theta (the theta-tuning trap), so its gap is theta-insensitive
  while the asymmetric two-route needs `theta ≳ 50` to approach the deterministic
  UE. `theta=200` gives comfortable margin with no exp saturation (stable through
  `theta=1000`).
- **Negative controls (honest rows).** `assignment_method="incremental"` (the
  marouter DEFAULT, a non-equilibrium loading) certifies `relative_gap ≈ 0.07`
  on Braess — feasible but far from equilibrium. `"UE"` raises (marouter silently
  falls back to SUE). `gawron`/`lohse` are refused (they emit all-zero flows the
  harness would censor).
- **Determinism.** Byte-identical netload across same-seed reruns AND across
  seeds (no RNG in the SUE path).

## What the certified row MEANS

"Industrially converged" is **not** "certified equilibrium." marouter
equilibrates its OWN model; the certified gap under the scenario's declared BPR
conflates (a) marouter's SUE truncation and (b) the mapping floor. On a linear
scenario the adapter drives (b) to the measured representability floor and A2
proves the matched links contribute nothing, so the small residual is the honest
simulator-to-benchmark gap, not "SUMO is bad at UE." Per the adr-025 lesson: the
headline always names WHOSE equilibrium and WHICH axis — `bfw`-beats-it is pinned
on the convergence axis only.

## Alternatives considered

- **`duarouter` one-shot** — free-flow all-or-nothing, adds nothing over the
  repo's `aon`. Rejected.
- **`duaIterate.py` microscopic DUA** — a genuinely different equilibrium concept
  (congestion emerges from car-following, not BPR), so it cannot be certified
  against the repo's static BPR costs. It belongs to a future DTA/DNL ladder, not
  this static row. (It would also need the stale `SUMO_HOME` fixed; harmless for
  marouter/netconvert.)
- **Folding into the torch CI job** — rejected (Decision 1): couples two heavy
  independent downloads and cross-contaminates their signals.
- **A new `external` paradigm token / an Evaluator branch / a scenario field** —
  none built: the `heuristic` paradigm + the existing certificate already score
  the row honestly; the golden Braess hash is re-asserted byte-identical.
- **MATSim / DTALite / TraCI / libsumo / a Docker adapter** — out of scope (Java/
  C++ with no wheels, or online-control APIs); the eclipse-sumo wheel obsoletes
  the Docker route for SUMO.

## Honest sourcing

- **`lopez2018microscopic` is a TOOL paper** (IEEE ITSC, DOI
  `10.1109/ITSC.2018.8569938`, canon-verified, PDF attributed unread). The
  citation anchors the SUMO suite (marouter/netconvert ship in the package it
  describes); the benchmark row validates the **ADAPTER's fidelity** (mapping
  floor + harness-certified gap), **never the paper's numerics** — it contains no
  marouter assignment numbers to validate against.
- **The vdf lineage is PTV-Validate / VISUM-Cologne**, per the SUMO source
  comment in `ROMAAssignments.cpp` — NOT the ITSC paper. The route-choice
  c-logit commonality is Cascetta-style; the default methods are Gawron/Lohse.
  These are named from SUMO option names and code structure; the original PTV /
  Gawron documents were attributed unread.
- The capacity-restraint law and capacity rules were established **empirically**
  here (fit + probes on the installed 1.27.1 binary), agreeing with the read
  source to < 1e-9; the ADR rests on the measured anchors, and the CI pins
  `eclipse-sumo==1.27.1` as the tripwire since the tables are hardcoded upstream.
- **Adapter-refused traps, documented:** marouter's `--assignment-method UE`
  silently degrades to SUE (warning only); the DEFAULT method is non-equilibrium
  `incremental`. An adapter that trusted defaults would emit a `RG ~1e-1` flow
  that still passes feasibility and get mislabeled as an equilibrium solver — so
  the adapter requests `SUE` explicitly, refuses `UE`, and restricts route choice
  to `logit`.

## Consequences

- **New:** `SumoMarouterModel` (`sumo-marouter`, registered when sumo present);
  `src/tabench/models/adapters/_sumo_io.py` (converter) + `sumo_marouter.py`; the
  `sumo` optional extra; one CI job; `tests/test_sumo_marouter.py` (30 tests,
  including the review regressions: compile read-back, the eps-edge min-length
  clamp, the high-demand OD-window sizing, toll refusal, lane-explosion refusal,
  the compile-phase wall budget, the `time_scale` envelope, the parasitic-slope
  cap, the `iterations=0` floor, and the empty/multi-interval netload guards).
  No new certificate, scenario field, Evaluator branch, or paradigm; no change to
  `capabilities.py`, `gaps.py`, or any hash.
- **Unchanged:** the Evaluator, the fairness gate, every hash (the golden Braess
  content hash is re-asserted byte-identical in the new test file), and the
  numpy-only core (`import tabench` and the full 731-test suite pass without the
  wheel).
- **Follow-ups:** the microscopic `duaIterate` DTA row (a different equilibrium
  concept, DTA/DNL track); MATSim / DTALite adapters (each its own ADR);
  large-network lane-quantization scaling (1000 lanes verified; the upper bound
  on extreme-capacity-ratio nets is untested).

## Adversarial review

Three independent lenses (soundness, formulation, numerics), each executing
Python/pytest/marouter; every finding CONFIRMED by a runnable repro and fixed
with a per-finding regression (streak: 16/16 sprints with at least one material
defect; 30 sumo tests after the fixes, from 19).

**CRITICAL (formulation): netconvert's silent 0.1 m minimum-length clamp
corrupted zero-intercept eps-edges.** The compile map's smallest edges could
fall below netconvert's undocumented length floor, which clamps them WITHOUT
error — scoring wrong flows while the trace's mapping-floor provenance claimed
otherwise. The shipped anchors passed only because their lengths happened to
clear the clamp. FIXED structurally: a **compile read-back** reparses the
compiled `net.net.xml` and verifies every edge's length/lanes/speed against the
declared values (mismatch = RuntimeError, the crash discipline), and
zero-intercept lane counts are chosen so eps-edges clear the clamp while the
intercept stays within the floor tolerance — refusing when both cannot hold.

**MAJORs, all fixed + pinned:** (a) the hardcoded 1-hour OD window made
marouter silently revert edges to free-flow when path times crossed 3600 s —
an AON collapse that fired at the shipped Braess demand dial of 360; the
window is now sized from a worst-path AON bound under the mapped law, the
'No interval matches' stderr is treated as an engine failure, and the netload
parse verifies exactly one interval (multiple would silently drop mass);
(b) tolled scenarios ran under a silently WRONG cost model (the compile map
dropped `fixed_cost`; a tolled two-route scored gap 0.293 with false
provenance) — any nonzero `fixed_cost` now refuses loudly; (c) the exact-
rational flow scale exploded on ordinary decimal coefficients (2-decimal
capacities → 1e5..1e6-lane edges → netconvert hangs measured at >690 s on
49/50 realistic fuzz scenarios) — rationalization is now denominator-bounded
with a reproduction check, and ANY link whose quantized lane count exceeds the
documented cap refuses before a subprocess spawns; (d) the `wall_seconds`
budget never covered the COMPILE phase (a 3 s budget ran 23 s) — one deadline
now threads through both the netconvert and marouter subprocesses.

**MINORs/NOTEs, fixed:** `time_scale` FactorSpec bounds admitted values that
silently broke the mapping (bounds narrowed to the validated envelope; the
false 'divides out of the certificate' doc corrected — tau divides out of
mass, not the certified gap; the read-back now catches residual corruption as
a raise); the `_MAX_LANES` cap silently violated the parasitic-slope claim
(now a refusal naming the link and achievable floor); `Budget(iterations=0)`
produced a censored all-zero row (floored at 1); a well-formed but edge-less
exit-0 netload was laundered into the censor path (now raises when zero
matching edges coexist with positive demand); missing binaries surfaced as
bare `FileNotFoundError` (wrapped into the contract's diagnostic
RuntimeError); the CI engine pin gained `sumo-data==1.27.1` (it floated
independently); a test helper leaked its workdir on failure paths.

**Survived (highlights):** the vdf tables re-fit by measurement against the
installed 1.27.1 across speed bands and lane counts (exact match to the
adapter's tables); both anchor floors independently re-derived (Braess
eps-perturbed exact UE → RG 1.736e-4 vs the adapter's 1.739e-4; two-route
mapped fixed point matched to 5.6e-8); the cost-matched anchor (~1e-10 on
representable links); command-injection immunity (list argv, scenario fields
never reach the command line); env hardening against a booby-trapped ambient
`SUMO_HOME`; temp hygiene across normal/refusal/raise/timeout paths with no
zombie processes; byte-determinism across seeds, tempdirs, and processes;
fresh-venv simulations of the core install (neither torch nor sumo pulled)
and of the CI install order (no re-resolution); actionlint-clean workflow;
no exact-decimal pin on any marouter output (version-robustness audit).
