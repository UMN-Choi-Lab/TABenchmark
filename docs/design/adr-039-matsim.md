# ADR-039 — `matsim`: the second EDOC-1 row, the first agent-based / first stochastic-track external engine

**Status:** accepted (shipped)
**File:** `docs/design/adr-039-matsim.md`

## Context — the stochastic flagship ADR-036 named

[ADR-036](adr-036-external-dynamic-observational-certificate.md) is the design
authority for **EDOC-1**; its first row shipped as `sumo-duaiterate`
([ADR-037](adr-037-sumo-duaiterate.md)) together with the shared substrate. This
ADR ships the **second row**: MATSim 2025.0 (Horni, Nagel & Axhausen 2016,
`horni2016multiagent` — already in the canon; this row adds **no** new canon
entry), superseding the [ADR-030](adr-030-external-dta-simulators-deferred.md)
deferral exactly as ADR-036 prescribed. The ADR-030 blocker stands in its own
terms — QSim has no static latency function, so the A2 cost-matched anchor stays
impossible *in kind* — and A2's role is played by the pinned engine under **G1
replay fidelity** (`lastIteration = firstIteration` on the emitted plans is a
zero-replanning replay; replanning fires only *between* iterations).

Two firsts land with the row, both named by ADR-036:

* the **stochastic track** (R5): `global.randomSeed` is outcome-bearing
  (ChangeExpBeta selection + qsim insertion/merge order), so the row is scored as
  **P8 macroreps over a pinned seed list** with a bootstrap CI on the mean —
  realized as the new engine-free substrate module `tabench.edoc.macrorep`;
* the **MATSim canonicalizer** (R10): same `tabench-edoc-canon-v1;` version, as
  `canon.py` committed ("the MATSim/DTALite canonicalizers land with their rows
  under the same version").

Every number below was **measured on this box on 2026-07-17** with the SHIPPED
estimator (occupancy-aware field + walk-enumeration TD-SP + per-first-edge
origin-wait profiles) on the pinned toolchain; the engine-free substrate
additions are exercised in `tests/test_edoc.py`, the row (engine-free + gated
halves) in `tests/test_matsim_edoc.py` (the matsim CI leg).

## The corrected R10 record (a pilot-record correction; ADR-036 is not edited)

ADR-036 recorded the same-timestamp event-tie permutation as occurring "between
replay and original". Re-verification for this row shows the causal story is
different — **the permutation is a MULTITHREADING artifact**:

* at `numberOfThreads=8` (global + qsim + eventsManager), two identical-seed runs
  differ in `output_events.xml.gz` — **104/1400 event lines permuted within equal
  timestamps, multiset identical** (plans + links stay byte-identical);
* at the pinned `numberOfThreads=1`, the replay's raw (decompressed) event stream
  is **byte-identical** to the certified run — even with forced same-second
  departure ties, reversed input person order, and a different replay seed;
* the pilot's "framework decides" `eventsManager` default plausibly chose
  parallel on the pilot box — the observation was real, its mechanism was not.

Consequences shipped here: **(a)** `numberOfThreads=1` is pinned in `global`,
`qsim` AND `eventsManager`, all inside the hashed `semantic_config`; **(b)** the
G1 certificate hash is the **R10-canonicalized stream hash** (same-timestamp
`<event .../>` runs stable-sorted by full line bytes, post-gzip-decompress) —
invariant across thread counts / replay seeds / input order while any CONTENT
change still moves it; **(c)** raw-byte identity at threads=1 is a **stricter
engine-gated bonus test** (`test_g1_replay_deterministic_and_raw_byte_stable`),
never the gate. The tie-sort ships as `canon.canonicalize_matsim` under the
UNCHANGED canon version.

## The hash surface — the twin-run byte census (the S2 23/184 analogue)

Two identical-seed certified runs (seed 42, iterations 0..10) in **different**
temp trees, all 69 output files compared: **66/69 raw-byte-identical**; the three
divergent files — `logfile.log`, `logfileWarningsErrors.log`, `stopwatch.csv` —
carry wall-clock text and stay different after canonicalization. The G1 surface
is therefore an explicit **ALLOWLIST** (`canon._MATSIM_HASHED_BASENAMES`):

| artifact | on the G1 hash surface? | why |
|---|---|---|
| `output_events.xml.gz` | YES (tie-sorted) | the experienced record |
| `output_plans.xml.gz` | YES | the emitted plans echo |
| `output_network.xml.gz` | YES | the compiled dynamics (read-back too) |
| `output_config.xml` | YES | census-stable (the certifier writes purely RELATIVE paths, so the echo is byte-identical across working dirs) — hashing it pins the engine's own record of `firstIteration == lastIteration` (pair N6) |
| logs / `stopwatch.csv` | no | wall-clock text (the 3 measured divergents) |
| `scorestats.csv`, `modestats.csv`, `modeChoiceCoverage*`, `output_links.csv.gz`, everything else | no | provenance; R10 hashes simulation state only |

`output_links.csv.gz` is additionally **never a flow source**: measured (ADR-036
pilot, re-confirmed), its `vol_car` equals left-link counts, reading the arrival
link as **zero** despite 100 entered-link events. The shipped parser derives
flows from **entered-link + vehicle-enters-traffic** events (in) and **left-link
+ vehicle-leaves-traffic** events (out) — the entered-link-only rule of ADR-036
undercounts the *departure* link (which has no entered-link events), the measured
correction this row records.

**Measured event semantics** the parsers pin: the departure link is entered at
its downstream end (`vehicle enters traffic`, discharge ≈ 1 s, so the family
declares `fftt(home) = 1`); the arrival link is traversed in full
(`entered link → arrival` = length/freespeed); `depart_delay = vehicle-enters-
traffic − departure` (the departDelay analogue, forgery pair 1); the on-network
experienced time decomposes EXACTLY into the per-link (enter→leave) spans, which
is why the resolution delta below is small.

## Replay seed-dependence — the pair-N2 record (ruling 5)

Measured ON THE SHIPPED FAMILY: replaying one emission's plans under a
**different** pinned seed (7 vs 42) reproduces the certified event stream
**raw-byte identically** (all 100 per-agent (dep, arr, route, experienced)
tuples equal). **The zero-replanning replay map is seed-INDEPENDENT.** (The
full G1 canon hash still differs across seeds — `output_config.xml` on the
surface echoes `randomSeed` — so the claim is about the SIMULATION STATE, as
the pinned test states.) Forgery pair 5 / N2
consequence: cross-macrorep artifact reuse **collapses to legal pair-11
optimization** — a plan set that certifies under every pinned seed's (identical)
replay map is genuinely seed-robust, and its macrorep mean is that emission's own
honest score. The per-seed EMISSIONS remain seed-dependent (the co-evolution
differs by seed — five distinct converged `RG_D1` values below), so the
stochastic track still measures real seed variance. Pinned by
`test_replay_is_seed_independent_the_n2_record`, which instructs a future
maintainer to update THIS record if the assertion ever flips.

## The row (mirrors ADR-037 section-for-section; deviations named)

* **EDOC producer, not a `TrafficAssignmentModel`** — `MatsimAdapter`
  (`tabench.models.adapters.matsim_edoc`) emits plans `P` (the final iterate's
  selected plans under `scenario.seed`) and derives `X` + the field from its OWN
  pinned replay of `P` (the ADR-037 artifact-contract clarification). NOT in
  `MODEL_REGISTRY`; re-exported **unconditionally** (a Java engine needs no
  guarded python import — engine absence is the runtime `matsim_available()`
  probe / a G0 RAISE). Class attrs `track="edoc-stochastic"`, `seedable=True`.
* **Addressing (F8):** the engine ONLY via `TABENCH_MATSIM_HOME` (jar or the
  `matsim-2025.0/` release layout, whose `libs/` the jar manifest requires) and
  java ONLY via absolute `$TABENCH_JAVA_HOME/bin/java` (fallback `JAVA_HOME`) —
  never PATH. The probe is side-effect free (no JVM started).
* **G0 pins:** `engine_version = "matsim-2025.0;jar-md5=…;jdk-major=…"` read from
  the addressed artifacts at certify time (jar md5 measured
  `fd4217f965221d4c4f35fed107d73d2f`); the **full JDK build**
  (`Temurin-21.0.11+10`) is a family-declared constant embedded in the hashed
  `semantic_config` with a G0 RAISE against `java -version` (the
  strictly-stronger reading ADR-036's G0 licenses: a JDK patch drift is an
  uncontrolled G1 censor surface, and hashing the *declared* constant keeps the
  hash box-stable).
* **The certifier writes EVERY config** (pair N6 closed structurally): network,
  plans and config are written from hashed fields; the replay derives
  `firstIteration == lastIteration` from the ONE constant `_REPLAY_ITERATION`
  (self-asserted); the strategy gate refuses time/mode-mutating strategies
  eagerly (G2); `removeStuckVehicles=false` so gridlock censors as G3
  incompletion; a **read-back** of `output_network.xml.gz` RAISES on any
  length/freespeed/capacity/permlanes drift. Scoring `modeParams` are
  re-declared for car+pt+walk (the measured 2025.0 parameterset-replaces-defaults
  NPE) and the car network gets deterministic sink→source return links (the
  measured TripRouter strong-connectivity abort).
* **Subprocess discipline** (S2 verbatim): `Popen(start_new_session=True)` +
  `killpg(SIGKILL)` (F2 — an orphaned JVM idles at multi-hundred-MB);
  `stdin=DEVNULL`; `-Xmx1g` bounds the child (R7); locale pins
  `-Duser.language=en -Duser.country=US` (java number formatting must not float
  with the host locale; kept, inside the hashed `semantic_config`);
  `PlanReplayFailure` ONLY from the plan-replay java step; rc never trusted;
  `_intersect_replay_deadline` makes the hashed `replay_deadline_s` bound every
  certifier replay (F3); `mkdtemp` + `rmtree` in `finally`.
* **R3 disclosure (not a weakening):** MATSim has **no standalone pinned router
  artifact** (QSim routing is in-process), so ADR-036 R3's cross-check clause —
  conditional on "where a pinned engine router exists" — has nothing to bind to.
  The substrate TD-SP is **normative-only** on this row, and the row substitutes
  a **harness self-cross-check**: every driven cost is re-derived by an
  independently written field composition and compared to
  `tdsp.evaluate_route` under `r3_tolerance_s` (infra RAISE on disagreement) —
  a field-arithmetic regression guard, disclosed here.
* **The engine-clock offset** `t0 = 3600 s` (writer adds, parsers subtract) keeps
  scenario time on `[0, horizon)` while dodging midnight-boundary special cases;
  a writer constant inside the hashed `semantic_config`.

### Substrate additions (hash migration disclosed)

* **`EdocScenario.seed_list`** (tuple of ≥ 5 distinct ints, the instance's own
  seed among them; empty = deterministic track): a hashed field, appended
  length-framed (`seeds:n;` + int64 bytes). **This migrates every existing
  EdocScenario digest**, including the shipped `sumo-duaiterate` reference's —
  permitted because the edoc instance hashes are unpublished (ADR-036) and no
  golden EDOC digest is pinned anywhere (verified by grep); the S2 row did not
  anticipate the move, so it is disclosed here. The golden static Braess
  `cf00f411…` is domain-separated and byte-untouched (re-asserted in both test
  files). The mechanical hash-coverage test gains a tuple-of-int mutation branch
  and its fixture a nonempty 5-seed list (closing the empty-tuple IndexError).
* **`tabench.edoc.macrorep`** (NEW): `per_seed_scenarios` (harness-derived
  macrorep instances — pair N1), `certify_macroreps` (any censored macrorep
  censors the row; infra RAISES un-laundered; CI via the house
  `experiments.bootstrap.bootstrap_ci` on `SOURCE_BOOTSTRAP` at
  `root_seed = seed_list[0]` — byte-reproducible, report-only, pair N4), and
  `MacrorepResult`.
* **`canon`**: `canonicalize_matsim` + `is_hashed_matsim_artifact` +
  `hash_matsim_artifacts`; `hash_artifacts` gains a `surface` parameter whose
  default IS the SUMO surface — every pre-S3 SUMO digest is byte-identical,
  regression-pinned with a frozen digest in `test_edoc`.

### Parent rulings ratified at this row (R5's silences, closed)

* **Row floor (ruling 2):** `floor_gap(row) = mean(per-seed floor_gaps)` —
  consistent with the mean score's estimand.
* **Row censoring (ruling 3):** ANY censored macrorep censors the WHOLE row
  (`feasible=0`, NaN mean/CI; per-seed diagnostics preserved). Subset means
  re-open seed shopping (pairs 4/N3). Infra exceptions RAISE (R6).
* **Separation (ruling 4):** mean-vs-mean over the SAME pinned seed list — and
  computed on **floor-DISPLAYED values** (`max(mean, floor_gap)` each side),
  grounded in ADR-036's own rule that a sub-floor value is *displayed at the
  floor*. This is what makes the gate non-vacuous on the self-certifying
  shared-edge topology: **measured, the shared-drop family scores `RG_D1 = 0`
  exactly on BOTH anchors** (driven == BR on the frozen field — the shared queue
  cancels), so a raw ratio degenerates to `0/0 → ∞` and would vacuously pass;
  displayed values separate 1.00x and **refuse** it (forgery pair 12 closed for
  the stochastic track). The reference family is unaffected (both anchors ranked).

### The family and the R4 re-derivation (shipped estimator, 2026-07-17)

`build_matsim_diamond_scenario`: `O0 →home→ O`, route A `a1,a2` (fftt 90+90) vs
route B `b1,b2` (100+100), `D →work→ D2`; the 1-lane drop on `a2` (MATSim's
outflow queue sits ON the drop edge — unlike SUMO meso's upstream storage, the
cost signal lands exactly on the route-distinguishing edge); lanes
(4,3,1,2,2,4) × `capPerLane = 600 veh/h` (the explicit engine capacity dial,
hashed via lanes + the semantic_config constant); 100 agents at 2 s spacing
(1800 veh/h vs the 600 veh/h drop), `departure_quantum = 1.0 s`;
`seed_list = (42, 7, 123, 2024, 31337)` — **exactly 5** (the R5 floor IS the R7
bound). Reference instance hash
`8156c2eb03bd59f38d1860f1eda11b48aff9c4fb974938f9978096bafebc0cf2` (moved once
pre-commit by fix F2 below — the scoring/router constants joining the hashed
`semantic_config`; the written engine config is byte-identical, so every anchor
VALUE was re-derived unchanged and the pre-fix hash `e41f7ebf…` never shipped).

| dial | value | how derived (all shipped-estimator measurements) |
|---|---|---|
| Δ / `n_intervals` | **20 s / 90** (1800 s horizon) | Δ-scan of the field-vs-experienced delta: control-state delta 26.5 s @ Δ=60 → 14.4 @ 30 → **9.5 @ 20** → 7.3 @ 15; Δ=20 keeps both anchors under the floor with margin and the row ranked |
| `floor_seconds` | **15.0** | > the 9.48 s control delta (anchors must stay feasible); converged deltas 1.4–2.5 s; `floor_gap(row)` = **0.0676** < the 0.0861 mean → the row is **ranked**, not sub-floor |
| **negative control** | AON control mean `RG_D1` **0.45913** (identical across all 5 seeds — iteration 0 has no route choice randomness) vs converged mean **0.08609** | **5.33×** displayed-value separation ≥ the declared **5.0**; deterministic on the pinned toolchain (same-seed byte determinism), so the margin cannot flake |
| per-seed converged `RG_D1` | {42: 0.0852, 7: 0.0957, 123: 0.0642, 2024: 0.0702, 31337: 0.1151} | real macrorep variance; row mean **0.086091**, 95% bootstrap CI **[0.070815, 0.102219]** |
| `backlog_bound` | **60.0** | measured max insertion backlog **0 s** on every anchor/seed (home at 2400 veh/h never binds); the bound catches gross insertion failure only |
| `replay_deadline_s` | **60.0** | the R6 form: 30 s fixed JVM-startup allowance + 10 × the measured whole-run replay wall (2.53 s → 3 s) — never a bare per-iteration multiple |
| `walk_bound` | 4 (shared variant: 5) | driven routes are 4-edge walks → in the TD-SP universe, `c_br ≤ c_drv` by construction |

**Walls (this box):** emit control 5.2 s / converged 7.1 s; certify (G1 double +
scoring) 5.2 s; single replay 2.53 s; the **5-seed `certify_row` 74.2 s** —
inside ADR-036 R7's 71–124 s projection; `negative_control_separation` 112.5 s
(2 states × 5 macroreps); the full 31-test row file **337.6 s**. The leg exceeds
the S2 sumo file's 26 s because every engine call pays a ~1.5–2 s JVM startup
floor (~50 JVM launches across the gate + row + refusal + probes) — the
**scored** row budget (the macrorep set) is what R7 bounds, and it holds.

### Forgery pairs (new N1–N6 + the S2 ports)

**N1 seed-list tampering** — hashed `seed_list`; harness-derived per-seed
instances; wrong-seed emission RAISES (macrorep + evaluator G0).
**N2 cross-macrorep artifact reuse** — measured seed-INDEPENDENT replay ⇒
collapses to legal pair-11 (record above).
**N3 subset/mean forgery** — the mean exists only inside `certify_macroreps`
over the full pinned list; one censored macrorep censors the row.
**N4 bootstrap-CI forgery** — harness-side, reserved-stream, byte-reproducible,
report-only.
**N5 event-tie canonicalization bypass** — tie-sort on both sides; content still
moves the hash; idempotent (pinned engine-free).
**N6 replay-config forgery** — the certifier writes every config; first==last
from ONE constant (self-asserted + engine-free pin); the config echo is ON the
hash surface; engine-gated probe shows a patched `lastIteration=1` "replay"
diverges from the honest replay on the control state.
**Ports:** pairs 1 (backlog census from departure→enters-traffic), 2 (G1 per
seed; doctored-X censors, engine-gated), 3 (full-network TD-SP), 4
(strengthened by the pinned list), 5 (see N2), 6 (jar-md5 + JDK G0), 7
(raw-field delta gate), 8 (occupancy witness exact from event spans), 10
(exact-departure G2 on the 1 s grid), 11 (legal), 12 (construction refusals +
the displayed-value separation gate). Pair 9's MATSim realization IS the new
canonicalizer (N5 + decompress + allowlist).

### CI, tutorial, docs

* **CI:** a NEW seventh `matsim` job — `actions/setup-java` is NOT used: it can
  pin `21.0.11` but not the exact `+10` build, so the job downloads the JDK from
  the hard Adoptium asset URL (sha256
  `4b2220e232a97997b436ca6ab15cbf70171ecff52958a46159dfa5a8c44ca4de`) and
  `matsim-2025.0-release.zip` from the `2025.0` release tag (the guessable
  `matsim-2025.0` tag 404s), **md5 `c65f35eafabea2456b818875f08048ca` verified
  BEFORE unzip** (supply-chain gate); both cached keyed on the checksums;
  toolchain paths exported via `$GITHUB_ENV` (absolute, the F8 rule).
* **Tutorial:** `tutorials/11-external/05-matsim.ipynb`
  (`{track: external, unit: matsim, requires_extra: matsim, covers: []}`,
  committed stripped, polite guard cell) + the UNCONDITIONAL `_track_manifest`
  entry landing atomically with it (`_ALLOWLIST` stays EMPTY). `requires_extra:
  "matsim"` names an engine gate with **no pyproject extra** behind it (an empty
  marker extra would be dishonest metadata for a Java-only engine); the
  tutorials/docs probe maps extend to a **callable** probe
  (`matsim_available`), with the pre-S3 find_spec behavior for
  torch/sumo/dtalite kept byte-for-byte (regression-pinned). The notebook
  execution per-CELL cap in `tests/test_tutorials.py` rises 120 s → 300 s: the
  negative-control cell is one ~112 s call (2 states × 5 macroreps × JVMs) and
  120 s would flake on a slower runner — still a hang-stop, not a budget (the
  docs build allows 600 s).
* **Docs:** the ROADMAP Horni row flips deferred → shipped-as-`matsim` with the
  R8 non-comparability sentence; `docs/ARCHITECTURE.md`'s external-engines
  paragraph now names the shipped EDOC track + this row. MODELS.md gains no card
  (EDOC rows are not the `MODEL_REGISTRY` surface). Hand-annotated;
  `tools/generate_references.py` never run; **no new canon entry**
  (`horni2016multiagent` pre-exists).

## Adversarial review — S3 finding record

A 3-lens review (certificate-soundness / honesty-antiforgery / infra-canon; all
findings EXECUTED against the live pinned toolchain, ~80 asserted probes, 10/10
defense mutants killed) ran against the uncommitted row. Clean areas included
the macrorep arithmetic and CI report-only status, the displayed-value
separation basis (floor-gaming fails closed in both directions), the R10
canonicalizer under 220 adversarial shuffles, an independent bit-exact RG_D1
re-derivation, the R6 typed-exception map, and the CI supply-chain pins. The
fix batch, deduped (three fixes deliberately touch the committed S2 sumo
adapter — inherited defects, fixed here with their own pins; no published hash
moves):

| # | Sev | Finding (executed evidence) | Fix | Pin |
|---|---|---|---|---|
| F1 | MAJOR | R6 censor-launder: a caller wall tighter than the hashed `replay_deadline_s` killed the replay JVM mid-startup and was typed `PlanReplayFailure` → `feasible=0` for a certifier-side budget exhaustion (executed: `deadline=now+1.5` on a real feasible bundle → censored; in `certify_row` a starved 5th seed would censor the whole row blaming the model). Same pattern shipped in S2; MATSim's ~2.8 s JVM startup widened the window ~100× | `_intersect_replay_deadline` returns `(deadline, clipped_by_caller)`; the replay passes `censor_on_timeout=not clipped` (an engine CRASH still censors regardless) — mirrored in `sumo_duaiterate.py` | stub-sleeper `test_replay_timeout_typing_scenario_deadline_censors_caller_clip_raises` in BOTH test files (caller-clipped → `RuntimeError`; scenario-deadline → `PlanReplayFailure`) |
| F2 | MAJOR | hash-discipline gap: the certifier-written scoring constants (mode marginal utilities, activity typicalDurations) and `routingAlgorithmType` are outcome-bearing (executed: `-6.0 → -6000.0` changed 13/100 selected routes, +609 s total experienced time, with `content_hash` byte-unchanged) but were not in the hashed `semantic_config` | hoisted into module constants (`_ROUTING_ALGORITHM`, `_ACTIVITY_TYPICAL_DURATIONS`, `_MODE_MARGINAL_UTILITY_HR`) folded into `_semantic_config()`; the written config is byte-identical, so the engine outputs and every anchor value are unchanged — only the instance hash moved (`e41f7ebf… → 8156c2eb…`, permitted pre-publication under the same unpublished-digest clause as `seed_list`, and re-derived-confirmed) | `test_semantic_config_carries_every_pinned_constant` extended: presence + a 7-constant mutation loop |
| F3 | MINOR | F10 vetting bound the family NAME string: after a legitimate diamond vetting, the self-certifying shared-edge topology relabeled `family='matsim-diamond'` sailed past `certify_emitted` (executed). Inherited from S2 | vetting keyed on a TOPOLOGY digest (`_topology_digest`: edges/lanes/fftt/OD sub-hash; runtime state only, no instance-hash movement) in BOTH adapters | `test_certify_emitted_vetting_is_topology_keyed_engine_free` (matsim, incl. the relabel probe) + the reworked S2 vetting branch |
| F4 | NOTE | `PlanReplayFailure` could escape `certify_emitted`'s third (R3 self-check) replay uncaught — fail-closed but mistyped (the censor-signal type behaving as an infra raise) | the third replay catches `PlanReplayFailure` and re-raises `RuntimeError` with context. DEFERRED deliberately: reusing G1's replay for the self-check (saves ~2.5 s × 5/row) needs an `EdocEvaluator` interface change — a future batch | typed by the wrap itself (reachable only via F1-style starvation) |
| F5 | MINOR | box-global temp-dir hygiene globs: a concurrent engine session's live workdirs flaked the S2 hygiene test (executed, clean 8 s later) | every `mkdtemp` prefix embeds `os.getpid()` (both adapters); both hygiene tests snapshot-diff only their own pid's prefix and document the concurrency assumption | the pid-scoped hygiene tests themselves |
| F6 | NOTE | stale docs-job comment (excluded set said torch/sumo/dtalite) | comment now names matsim | — |
| F7 | NOTE | the actions/cache key was a hand-maintained literal coupled to the checksum pins by convention only (a pin bump that forgets the key gets a stale hit and skips re-verification); curl without `--fail` | both checksums moved to job-level `env` referenced by the cache key AND the verify lines; `curl -fsSL` | actionlint + the CI-sim executed by the review |
| — | MINOR (recorded, no code) | N2 substrate boundary: `certify_macroreps` itself cannot detect a hand-crafted emit reusing one bundle across macroreps (zero-width CI, best-of-N ~25% lower mean, executed) — NOT reachable via `certify_row`/`negative_control_separation`, which hardwire the per-seed engine emit; legal-by-design per the pair-11 collapse | the harness-controlled-emit CONTRACT is now stated in `certify_macroreps`'s docstring; a future stochastic row with a model-controlled emit AND a seed-independent replay must add a per-seed-distinctness guard | the docstring sentence + this record |

## Consequences

* **New code + one CI job + ONE disclosed hash migration** (the `seed_list`
  block; unpublished digests only). `RG_D1(matsim)` lives on the dynamic-external
  leaderboard table — "frozen-field BR gap (RG_D1)", mean ± bootstrap CI — and is
  **never** compared to static Wardrop `relative_gap` (R8) nor, numerically, to
  `sumo-duaiterate`'s deterministic-track single-seed value (different engines,
  different families, same certificate).
* **Named follow-ups:** the `dtalite-simulation` row (S4, deterministic track,
  R9 plans construction) and the BO4Mob stage-2 D2 certificate (R11) — both
  reuse this substrate; the macrorep harness is ready for any future
  stochastic-track engine.

## Sources (the honest ledger)

* **MEASURED THIS SPRINT (Temurin-21.0.11+10 + matsim-2025.0 on this box,
  2026-07-17; fresh scratch trees, the research toolchain reused read-only):**
  the corrected R10 threads mechanism (re-verified from the S3 research record's
  executed runs; the shipped family's replay raw-byte identity at threads=1);
  the 66/69 twin-run census + the `output_config.xml` relative-path stability;
  the replay seed-independence; the Δ-scan (26.5/14.4/9.5/7.3 s), the anchors,
  per-seed gaps, CI, separation 5.33×, the shared-edge `RG_D1 = 0` self-
  certification and its displayed-value refusal; all walls; the jar md5; the
  2025.0 config-writer gotchas (scoring parameterset NPE, strong-connectivity
  abort) — reproduced from the research record and re-exercised by the shipped
  writers.
* **ATTRIBUTED UNREAD (canon tool book, software lineage — tool-paper
  discipline):** Horni, Nagel & Axhausen (2016), *The Multi-Agent Transport
  Simulation MATSim*, Ubiquity Press (`horni2016multiagent`, tier 1) — anchors
  the engine's lineage; the row validates ADAPTER + engine fidelity, never the
  book's numerics (the ADR-027/029/037 posture).
* **Design authority + repo precedents read in full:** ADR-036 (the certificate),
  ADR-037 + `sumo_duaiterate.py`/`_sumo_io.py` (the row precedent mirrored
  section-for-section), ADR-030 (the superseded deferral), the S3 research
  dossier (toolchain re-verification + design). No `references.bib` edit and no
  `tools/generate_references.py` run were made.
