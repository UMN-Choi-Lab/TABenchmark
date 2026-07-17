# ADR-040 — `dtalite-simulation`: the third EDOC-1 row, the first deterministic-track external engine, closing the adr-029 honest-sourcing loop

**Status:** accepted (shipped)
**File:** `docs/design/adr-040-dtalite-simulation.md`

## Context — the third row ADR-036 named, and the loop ADR-029 left open

[ADR-036](adr-036-external-dynamic-observational-certificate.md) is the design
authority for **EDOC-1**; its first row shipped as `sumo-duaiterate`
([ADR-037](adr-037-sumo-duaiterate.md)) and its second as `matsim`
([ADR-039](adr-039-matsim.md), the stochastic flagship + the shared macrorep
substrate). This ADR ships the **third row**: DTALite 0.8.1's `simulation()` —
the Zhou & Taylor (2014) mesoscopic queue-DNL simulator (`zhou2014dtalite`,
already in the canon; this row adds **no** new canon entry). It is the DTALite
wheel's **other** entry point: [ADR-029](adr-029-dtalite-tap.md) shipped
`assignment()` (a static Frank-Wolfe on an exactly-mapped BPR, certified against
the DECLARED cost law) and named `simulation()` "a named non-goal here." This row
closes that honest-sourcing loop — the 2014 paper's actual mesoscopic content
lands as an **EDOC producer**, on the observational track, and the adr-029
docstring + the ROADMAP row now point to it.

Two firsts land with the row:

* the **deterministic track** (ADR-036 R5's complement): `simulation()`'s only
  RNG is an LCG re-seeded `101 + time_step` at every simulation step
  (TAPLite.cpp:5171-5174 + 2066-2068; `a=17364, c=0, M=65521`), so the engine
  consumes **no seed** — `seedable=False`, `seed_list=()`, **no macroreps**. The
  score is a single `RG_D1`. `scenario.seed` stays hashed but is engine-INERT;
  `per_seed_scenarios` refuses the empty list, so macrorep misuse is structurally
  impossible. This is stronger than `sumo-duaiterate`'s below-floor-seed-spread
  justification: there is no seed to spread over.
* the **DTALite canonicalizer** (R10): same `tabench-edoc-canon-v1;` version, as
  `canon.py` already promised ("the DTALite canonicalizer lands with its row, S4").

Every number below was **measured on this box on 2026-07-17** with the SHIPPED
estimator (occupancy-aware field + walk-enumeration TD-SP + per-first-edge
origin-wait profiles) on the installed `DTALite==0.8.1` wheel; the engine-free
substrate additions are exercised in `tests/test_edoc.py`, the row (engine-free +
gated halves) in `tests/test_dtalite_simulation.py` (the `dtalite` CI leg, now
shared with `dtalite-tap`).

## Provenance disclosure (ruling 9 — the adr-029 posture, carried)

The wheel-to-source linkage is **UNVERIFIED and disclosed exactly as adr-029
discloses it**: the wheel ships **no C++ source** (`RECORD` lists only `DTALite.py`,
`__init__.py`, and four binaries; the Python wrapper `DTALite.py` calls exactly two
ctypes entry points — `DTA_AssignmentAPI` and `DTA_SimulationAPI` — out of the 78
symbols `nm -D` exports, whose demangled mangled-C++ names match the vendored TAPLite
functions and so CORROBORATE, but do not prove, the trace-target linkage). The G0 pins
are therefore the ARTIFACTS:

* `DTALite==0.8.1` (the CI-pinned wheel version, read via `importlib.metadata`);
* the Linux engine `.so` md5 **`e179ed662ae2984ba09e5230ed4151b9`** (the byte the
  engine runs; dll/x86-dylib/arm-dylib differ and are out of certified scope).

The **trace target** — where every cited TAPLite.cpp line was verified — is
`github.com/itsfangtang/DTALite_release` commit
**`df8a09346a592ffe3b1b8ea3a7ba8b14f0aa150e`** (tarball md5
`31ed19ef065c11cf5185aac4097151d8`). That the 0.8.1 `.so` was *built from* that
commit is **not provable** from the wheel (symbols + format strings are consistent
only). Byte-determinism is pinned on **linux-x86_64 ONLY** (CI runs
`ubuntu-latest`); other platforms are out of certified scope. Cited lines, all
verified against the trace target:

| line | fact | used by |
|---|---|---|
| 2238 | `route_volume = 0` — `assignment()`'s `vehicle.csv` is DEAD code | R9 (plans from `route_assignment.csv`, never the dead file) |
| 4950 | `number_of_seconds_per_interval = 6` | the 6 s dynamics grid (`_SIM_STEP_S`) |
| 4979 / 5301 | `capacity_per_time_step = Link_Capacity/3600`; `entrance_queue.size() < capacity_per_time_step` | the ~600 veh/h admission law (the engine's cost law) |
| 5116 / 5123-5127 | `calculateOutflowCapacity` x10 boost, window `total_intervals - t <= 2*60*6 = 720` (a units bug) | the boost census (`_BOOST_WINDOW_INTERVALS = 720`) |
| 5171-5174 + 2066-2068 | LCG re-seeded `101 + time_step` (`a=17364, c=0, M=65521`) | `seedable=False`, the deterministic track |
| 5378 | main loop bounded by `total_intervals` | the head-block is bounded, NOT an infinite loop (ruling 1) |
| 5387 | early exit `t >= 600 && total_completed == total_loaded` | the lull-drop hazard (ruling 2) |

## The corrected/extended pilot record (ADR-036 is not edited)

ADR-036's `simulation()` section was written from a pilot; three of its records are
refined here from the from-scratch reconstruction and this sprint's shipped-family
measurements. **ADR-036 itself is not edited** (the adr-039 R10 precedent — a row
ADR corrects the pilot record in place):

* **Head-block is a bounded FAST drop, not an infinite loop (ruling 1).** ADR-036
  lines 258/348 say a pre-period departure "loops forever at rc=0" head-blocking
  `loadNewAgents`. Measured: the `simulate()` loop is structurally bounded by
  `total_intervals` (5378), so a pre-period departure terminates in ~0.03 s at
  rc=0 with the pre-period agent AND **every later same-first-link agent silently
  dropped** (all-filler `07:00:00` chains). On the **shipped family** this FAST
  census variant is what reproduces: a doctored `v0` at engine-minute 410 corrupts
  the whole **781-agent route-A cohort** (`test_headblock_hazard_...`), wall <
  `replay_deadline_s`. The pilot's infinite-loop phrasing likely migrated from
  adr-029's `assignment()` permuted-input loop. **Both defenses ship regardless**:
  the R6 `replay_deadline_s` (its defense if the loop variant ever occurs) AND the
  per-agent completion census (its defense for the measured fast variant).
* **The lull-drop hazard, unnamed in ADR-036 (ruling 2).** `simulate()` breaks at
  the first interval `t >= 600` (60 min into the period) where
  `total_completed == total_loaded` (5387), silently DROPPING any still-pending
  departure at rc=0. The family constructor **refuses** departure profiles that
  allow such an all-completed instant before a later departure — `_assert_no_lull_drop`,
  an eager `ValueError`, bounding earliest completions by the free-flow shortest
  path — and the G3 completion census is the certify-time backstop. This is coherent
  with the whole-hours/inside-one-day period gate (`_engine_period_hours`, eager):
  the horizon must be a whole number of engine hours ending ≤ hour 24, and the
  departure profile must keep someone provably in flight past 60 min.
* **The OMP hazard is a CORRECTNESS pin, demonstrated (understated in ADR-036's
  favor).** ADR-036 argued the `#pragma omp parallel for` over a shared `std::deque`
  is unsafe. Executed on an 8-link congested net: **OMP=4 gives 6 divergent
  `trajectory.csv` md5s + one SIGSEGV in 10 runs** (plus 2347 torn
  `simulation_debug.csv` lines from a shared ofstream race), while OMP=1 is
  byte-stable; default OMP on this 192-core box is **144 s vs 0.012 s** (~10⁴×).
  `OMP_NUM_THREADS=1` is therefore a **G0 correctness pin, not hygiene** — set on
  every child, hashed in `semantic_config`, and the child override beats a hostile
  parent (`test_hostile_parent_omp_is_overridden`, env `OMP_NUM_THREADS=64` → same
  canon hash). We do **not** assert a single racing pair — the empirical
  divergence+crash suffices for the pin.
* **Wall boundary made precise (ruling 8).** ADR-036's "0.04-0.05 s" is the
  SUBPROCESS wall (python startup + banner + engine ≈ 0.045 s); the engine-only
  in-child wall is **0.012-0.015 s**. This box measured a single subprocess replay
  (write GMNS + `vehicle.csv` + spawn + engine + parse) at **~0.08 s**.
* **The ~600 veh/h admission law is a STAIRCASE, not a flat band (scope note).**
  `entrance_queue.size() < capacity_per_time_step = capacity/3600` admits
  `ceil(capacity/3600)`-ish agents per 6 s interval as an integer-boundary staircase:
  the queue-empty case admits **1 agent per interval (≈600 veh/h) for capacity in
  (0, 3600]**, **2 per interval (≈1200 veh/h) for (3600, 7200]**, stepping up to
  ~6000 veh/h at cap 36000. The reference's first-link capacities — **600** (1-lane
  bottleneck) and **1200** (2-lane) — both sit in the FLAT first step (≤ 3600), so
  both admit 1/interval ≈ 600 veh/h and the congestion physics (and therefore
  `RG_D1`/`floor_gap`/separation) are exactly as measured. The instance defines its own
  admission as the engine's cost law.
* **MSA is hash-derived, not numpy-RNG (ruling 3).** The pilot's converged
  `RG_D1 = 0.039738` was numpy-2.2.6 `default_rng(20260716).choice`-specific. The
  shipped MSA switching picks are **hash-derived** (`_msa_pick(instance_hash, k,
  improvers, n_move)` — no cross-version `Generator.choice` dependence), so the
  shipped converged anchor is re-derived: **`0.050309`** (it differs from the
  pilot's 0.039738 — expected and correct).

## The fftt-column probe (ruling 6 — decisive)

The pilot wrote `length`/`free_speed`/`vdf_fftt` consistently and could not
distinguish which column `simulation()` reads. **One-perturbation probe, executed
this sprint** (O→M→D, one agent, baseline fftt 5 min each link):

| perturbation | measured exits (`departure_times` chain) | conclusion |
|---|---|---|
| baseline (fftt 5,5) | `07:05:00;07:10:00` | 5 min / link |
| ONLY `vdf_fftt` link1 → 15 | `07:15:00;07:20:00` | traversal **MOVED** |
| ONLY `length` link1 → 15 (fftt 5) | `07:05:00;07:10:00` | traversal **UNCHANGED** |

`simulation()` reads **`vdf_fftt`** (minutes), NOT `length`/`free_speed`. The writer
sets `vdf_fftt` = the hashed fftt AND keeps `length = vdf_fftt` miles at
`free_speed = 60` mph, so the geometry-derived fftt agrees by construction (either
read path yields the hashed fftt — robustness against an upstream read-path change).

## The hash surface — the twin-run byte census, allowlist-only (R10)

The engine is **byte-deterministic at OMP=1** (`test_g1_replay_bit_deterministic`:
twin replays in different dirs emit byte-identical `trajectory.csv`), so the
canonicalizer is the **IDENTITY plus the gzip-decompress rule** — idempotent and
content-sensitive by construction. The R10 founding-spec "positional-parse the
DTALite trajectory" necessity is a **PARSER rule** realized in
`_parse_trajectory` (a header/row-shape RAISE — the canon-version-bump trigger for
upstream format drift), **not** the hash canonicalizer.

Twin-run census: **14/14 `simulation()` emissions byte-identical** on this box
across two temp trees — but `sim_info.csv` carries a `timestamp` header column and
the debug/summary logs are latent wall-clock surfaces. The G1 hash surface is
therefore an explicit **ALLOWLIST** of exactly **`{trajectory.csv}`**
(`canon._DTALITE_HASHED_BASENAMES`) — the one artifact the score consumes. The
pre-S4 SUMO/MATSim digests are byte-untouched (the `surface=` regression in
`test_edoc.py` re-run; both engine-gated canon digests re-confirmed).

The trajectory positional-parse facts the parser pins (all EXECUTED):

* header = **13 names**, every data row = **12 fields** — the emitted row drops
  `travel_time`, so positional field[7] = `current_link_seq_no` (1 = completed
  2-link route, 0 = truncated/stuck);
* `loaded_status` (field[3]) is **DEAD** — reads 0 for completed, truncated, AND
  never-loaded alike (never read; pair D2);
* unvisited links carry literal **`07:00:00`** (= `_T0`) filler chains; a
  period-end-truncated agent shows real first-link times then filler; a pre-period /
  head-blocked agent shows all-filler;
* raw second-fields carry per-agent sub-interval residuals (396/1000 rows), so the
  parser does **NOT** assert 6 s alignment of raw fields (the header-shape RAISE is
  the drift guard); the scheduled departure is the **verbatim-echoed**
  `departure_time` column;
* the completion census is **chain-consistency-keyed** (`current_link_seq_no ==
  len(links) − 1` AND a full non-filler monotone chain), NOT time-keyed — a
  LEGITIMATE `07:00:00` first entry (an agent scheduled at t=0) is not read as
  filler (pair D1).

## The row (mirrors ADR-037/039 section-for-section; deviations named)

* **EDOC producer, not a `TrafficAssignmentModel`** — `DTALiteSimulationAdapter`
  (`tabench.models.adapters.dtalite_simulation`) emits plans `P` (the R9 step-0 FW
  split, then `iterations` MSA `1/(k+2)` best-response blends) and derives `X` + the
  field from its OWN pinned replay of `P` (the adr-037 artifact contract). NOT in
  `MODEL_REGISTRY`; re-exported **UNCONDITIONALLY** (a named deviation from
  `dtalite_tap`'s module-scope `find_spec` guard, mirroring `matsim_edoc`'s
  rationale — the module never imports `DTALite` in-host, so its engine-free test
  half runs on the core matrix legs and engine absence surfaces as the runtime G0
  read). Class attrs `track="edoc-deterministic"`, `seedable=False`. It deliberately
  does **not** extend `dtalite_tap.py` (different track, compile map, outputs, and
  failure typing; only ~15 lines of subprocess boilerplate overlap, and its shape
  differs anyway). `dtalite_tap.py`'s own guard is untouched.
* **G0 pins:** `engine_version` read at certify time via `importlib.metadata`
  (never an `import` — the wheel prints a banner and ctypes-loads the OpenMP `.so`);
  `assert_engine_pin` against the instance pin; the CI `==0.8.1` pin. The seed pin
  is inert-but-hashed (deterministic track).
* **The certifier writes EVERY engine input** (pair N6 closed structurally):
  node/link/demand/settings from hashed fields, **links sorted by
  `(from_node_id, to_node_id)`** (the adr-029 CRITICAL — same GMNS reader), engine
  `lanes = 1` always with `capacity = edge_lanes × 600` (the adr-029 lanes² trap,
  live for the R9 step). `vehicle.csv` is written from the parsed plans dict,
  **SORTED ASCENDING by `(departure_time, agent)`** — the D5 mandatory gate
  (measured: an unsorted file silently filler-corrupts the later-departing agent at
  rc=0; `test_d5_unsorted_vehicle_csv...`). Because the certifier regenerates
  `vehicle.csv`, no model-controlled byte order ever reaches the engine. There is
  **no config echo** to hash (unlike MATSim's `output_config.xml`) — mode selection
  IS the subprocess command (`_SIM_CMD` / `_ASSIGN_CMD`, one constant each, the N6
  one-constant discipline), so the N6 defense here is purely structural.
* **Subprocess discipline** (S2 verbatim, S3 fixes from birth): `Popen(
  start_new_session=True)` + `killpg(SIGKILL)` (F2 — a torn/empty `trajectory.csv`
  from a mid-run kill is never parsed); `stdin=DEVNULL` (`ExitMessage` =
  `getchar()`); one subprocess per engine call (a second in-process call doubles
  state); the child env ALWAYS pins `OMP_NUM_THREADS=1`; `_intersect_replay_deadline`
  makes the hashed `replay_deadline_s` bound every certifier replay with the **F1
  caller-clip/scenario-deadline censor split** (a caller wall tighter than the
  hashed deadline RAISES `RuntimeError` as certifier-side budget exhaustion; only a
  SCENARIO-deadline expiry raises `PlanReplayFailure`); **pid-scoped `mkdtemp`
  prefixes** (F5); rc never trusted (every caller re-reads its artifact — success is
  the parse + census, the adr-029 doctrine).
* **R3 disclosure (not a weakening).** DTALite has **no standalone router artifact**
  — `simulation()` never routes, `assignment()` never simulates — so ADR-036 R3's
  engine-router cross-check clause has nothing to bind to. The substrate TD-SP is
  **normative-only** on this row, and the row substitutes a **harness
  self-cross-check** (`_field_selfcheck`, the adr-039 shape): every driven cost is
  re-derived by an independently written field composition and compared to
  `tdsp.evaluate_route` under `r3_tolerance_s` — a field-arithmetic regression
  guard, infra RAISE, never a censor (measured `r3_max_s = 0.000 s`). The third
  (R3) replay's `PlanReplayFailure` is re-wrapped to `RuntimeError` (F4 typing).
* **R9 plans construction (rulings 7).** `_run_assignment_for_routes` runs
  `assignment()` on its OWN 1 h demand period (the adr-029 identity condition — the
  VDF divides volume by the period hours, so 1 h makes `v/c` the textbook ratio; the
  6 h simulation horizon would starve the FW of congestion and collapse the split to
  AON) and parses `route_assignment.csv` by NAME (drift RAISES; the 0.8.1 header
  carries a trailing comma). `assignment()`'s own `vehicle.csv` is **never read**
  (header-only dead code, 2238). `_integerize_route_volumes` largest-remainder rounds
  the FW `volume` shares onto each OD's agent count (`|count − share·N| ≤ 1`) and
  fractional-position-interleaves the routes across the departure profile. The
  reference exercises a **genuine multi-route split — 781.25/218.75 → 781/219** (the
  CI anchor tripwire pins it, `test_r9_split_tripwire...`; the adr-029
  `link_performance.csv` precedent).
* **The MSA loop — ROW-LOCAL.** ADR-036 names no substrate MSA deliverable (its
  named substrate pieces all exist), and hoisting waits for a second router-less
  engine. `emit()` = P0 from R9 → per iterate k: pinned replay → in-loop boost-census
  RAISE → substrate field + origin-wait builders → per-agent BR via the certifier's
  own composition → per-OD blend moving `round(frac·n)` improvers (hash-picked) onto
  BR routes at `1/(k+2)`. Emitting the certifier's BR is pair-11 legal; the measured
  one-step overshoot (the reference's step-0 0.372 would over-swing) is exactly why
  the model is the smoothed blend, not the one-step BR (R12: realized-BR is Tier-B,
  never scored). `iterations=0` emits the step-0 FW split as-is — the negative
  control.

## The family and the R4 re-derivation (shipped estimator, 2026-07-17)

`build_dtalite_corridor_scenario` / `reference_scenario`: a **two-route corridor**
`O →a1→ MA →a2→ D` (fftt 2×300 s, 1-lane 600 veh/h bottleneck) vs `O →b1→ MB →b2→ D`
(2×420 s, 2-lane 1200 veh/h nominal — capped at ~600 by the admission law); 1000
agents at 2-per-6-s-slot over 50 min (1200 veh/h aggregate vs ~600 per-route), so
the step-0 FW piles the transfer queue onto route A and separates from the converged
blend. Reference content hash
**`f3bf543b1fc21e08dba9c5078d0114caea5ac26fcff9211c6e61ac6b9362d7c7`** (topology
digest `f93e00ed…`). BURSTY profiles are refused by shape (`agents_per_slot > 2` →
`ValueError`: the transfer-entrance service is non-FIFO under burst arrivals —
measured 10-per-slot parks ~2 agents/slot until the stream ends, a delta no
interval-mean field represents; at ≤ 2/instant the queue is FIFO, delta ~2 s).

| dial | value | how derived (all shipped-estimator measurements) |
|---|---|---|
| Δ / `n_intervals` | **6 s / 3600** (21600 s = 6 h, period 7→13 h) | the engine dynamics grid IS 6 s (4950); the horizon must be whole engine hours (`_engine_period_hours` eager gate) |
| `departure_quantum` | **6.0 s** (≤ 2 per grid instant) | the 6 s grid; a coarser multiple is unnecessary — departures are already whole-6-s slots and the FIFO-≤2 shape gate binds |
| **negative control** | step-0 FW split `RG_D1` **0.372244** vs MSA-converged (16 iters) **0.050309** | **7.399147×** floor-displayed separation ≥ the declared **5.0**; both anchors floor-displayed (`max(rg, floor_gap)`), deterministic on the pinned toolchain (same-byte determinism), so the margin cannot flake |
| `floor_seconds` | **10.0** | Δ-scan of the field-vs-experienced delta: the converged reference measures **delta = 1.75 s** (≪ 10); floor 10 keeps the anchor feasible with wide margin AND leaves the row RANKED — `rg_d1 0.050309 > floor_gap 0.011840`, `sub_floor = 0` |
| `backlog_bound` | **60.0** | measured max insertion backlog **0 s** (origin insertion is NOT admission-gated — 976 veh/h inserted onto a 600-cap link with zero origin wait), so `depart_delay ≈ 0`; `mean_backlog −0.4 s` is the sub-second grid-print residual (in [−1, 0]); the bound catches gross insertion failure only |
| `replay_deadline_s` | **30.0** | the R6 form: a fixed python-child startup allowance + a multiple of the measured **whole-run** replay wall (~0.08 s), NEVER a bare per-iteration multiple; 30 s is ~375× the single-replay wall |
| `walk_bound` | **2** | driven routes are 2-edge walks → in the TD-SP universe, `c_br ≤ c_drv` by construction |

**RANKED, not manufactured (ruling 4).** The Δ/floor were derived FIRST, then the
converged reference measured: it lands **above** the floor (`RG_D1 0.050309 >
floor_gap 0.011840`), so it ships **RANKED** honestly — no family tuning was applied
to force it. Had it landed sub-floor, it would ship displayed-at-floor; the
separation gate uses the floor-displayed basis on **both** sides regardless (the
adr-039 ruling-4 basis, adopted from birth because the 0/0 shared-topology vacuous
pass exists on the deterministic track too — `shared_bottleneck_scenario` is the
refusal demonstration).

**Vetting-scope disclosure.** The separation gate is TOPOLOGY-digest-keyed (the S3 F3
anti-forgery choice — vetting a forgeable family STRING is the defect it fixes), and
that digest EXCLUDES `agent_depart` (and `dt`/`n_intervals`), so a same-topology
instance with a DIFFERENT departure profile borrows the reference's vetting. The
≥ `separation_factor`× guarantee therefore holds for the **vetted** departure profile;
`sub_floor` is the per-instance vacuity backstop that still fires on a borrowed profile
(a review-constructed borrowed profile that separates only 4.85× lands `sub_floor=1`,
i.e. NOT ranked — every borrowed profile that DID rank also genuinely separated > 5×).
Folding `agent_depart` + `dt`/`n_intervals` into the vetting digest is a possible
future strengthening, **deliberately deferred**: folding it now would perturb the
boost-variant test's same-topology-digest assertion (`n_intervals` is not a topology
field, which is exactly what lets the 1 h boost variant reuse the reference's vetting).

**Reference certify readout:** `feasible=1`, `rg_d1=0.050309`, `floor_gap=0.011840`,
`sub_floor=0`, `delta=1.753 s`, `max_backlog=0 s`, `br_coverage=1.0`, `n_improvers=523`,
`tstt=844614 s`, `r3_max_s=0.000 s`.

**Walls (this box):** vet (2 emit + 4 certify replays) **2.1 s**; emit (R9 + 16 MSA
replays + final) **1.6 s**; certify (2 G1 + 1 R3 replay) **0.25 s**; single
subprocess replay **~0.08 s**; engine-only in-child **0.012-0.015 s**. The whole row
file runs well inside the shared `dtalite` CI job's minutes.

**The boost census is boost-CLEAN on the reference.** Onset = 21600 − 720×6 =
**17280 s (4.8 h)**; the family exits ~10 min in, hours before onset, so no crossing.
The **1 h-horizon variant** (`n_intervals=600`) has onset = 3600 − 4320 = **−720 s <
0**: the entire horizon is inside the boost window, so the instance is
boost-DEGENERATE — every agent is boost-exposed and `boost_crossing_n = n_agents`
(see the three-layer gate).

## Forgery pairs

**Ports of the twelve** (adr-036) with DTALite realizations: **P1** unreached-demand
hiding — ports WITH FORCE (pre-period silent drop + head-block + period-end
truncation, all rc=0): defended by the departure-window gate + the G3 census from
`current_link_seq_no`/non-filler chains (never `loaded_status`) + the boost census +
the lull-drop gate. **P2** self-report substitution — G1 replays `trajectory.csv`
byte-exactly (the strongest of the three engines; the printed gap/summary is
provenance). **P3** plan-set impoverishment — full-network substrate TD-SP; the
0.372 step-0 anchor IS this defense measured. **P4/N1-N4** seed forgeries — VACUOUS
on the deterministic track (no seed; `seed_list=()`); pinned by the
`per_seed_scenarios` empty-list refusal. **P5** cost-averaging — an averaged
trajectory is not the fixed replay output; byte-exact G1 censors it. **P6**
version drift — G0 `importlib.metadata` + `assert_engine_pin` + the CI `==0.8.1`
pin. **P7** aggregation-window gaming — hashed Δ + the raw-field delta floor gate.
**P8** poison-the-alternative — occupancy witness EXACT from trajectory link spans
(the matsim event-span precedent). **P9** volatile-byte hash games — the
realization IS the new canonicalizer + `{trajectory.csv}` allowlist + the parser's
header-shape RAISE. **P10** departure-time gaming — G2 exact departures on the 6 s
grid (verbatim schedule echo makes the write-back exact when on-grid). **P11**
emit-the-certifier's-BR — legal; the MSA blend is its principled use. **P12**
vacuous/degenerate instances — construction refusals (`_assert_no_lull_drop`, the
whole-hours period gate, the burst-shape gate, the boost census) + the
floor-displayed separation refusal of the shared-edge control.

**New DTALite-specific pairs** (named for the review): **D1** trajectory-filler
exploitation (read `07:00:00` filler as real samples) — the chain-consistency filler
rule; **D2** dead-column games (claim state via `loaded_status`) — never read; **D3**
boost-window laundering (finish congested agents inside the last 72 min) — the
three-layer gate below; **D4** OMP nondeterminism as a hash-dodge — the always-on
child `OMP_NUM_THREADS=1` + the G1 double + X-equals-replay collapse to P2; **D5**
`vehicle.csv` order games (an unsorted file silently corrupts later agents at rc=0) —
the certifier writes it sorted, no model byte order reaches the engine.

**The three-layer boost gate (D3 / pair 12), three correctly-typed layers:**

1. **in-loop RAISE** — any MSA iterate whose replay census crosses the onset raises
   `ValueError` in `emit()` (the model's own field would be boost-contaminated); this
   also realizes pair-12's "every constructor-side run" clause.
2. **certify-time CENSOR** — `certify_emitted` runs the boost census on
   `emitted.experienced` (G1 has bound X to the replay tuple-for-tuple, so the census
   on X IS the census on the replay, **zero extra engine calls**) and returns
   `feasible=0`, `boost_crossing_n`, `boost_onset_s` **whenever crossings exist —
   even if a downstream substrate gate also censored** (a too-short horizon also
   truncates late agents at G3; the boost window is the instance-design ROOT cause,
   so its diagnostic must reach the metrics dict). A boost-clean emission passes
   through unchanged.
3. **constructor ValueError** — `negative_control_separation` censors on a boost
   crossing via the same certify-time arm, so a boost-degenerate topology RAISES.

`_boost_crossings` semantics: for onset **> 0** (a partial window) only COMPLETED
agents whose exit is at/after the onset are contaminated; for onset **≤ 0** the
entire horizon is boost-covered, so the census is the whole emission (`n_agents`) —
a topology-stable count. On the 1 h variant this is exactly **1000**
(`test_boost_window_censor_live`).

## CI, tutorial, docs

* **CI:** the existing `dtalite` job is EXTENDED (not cloned) — same 1.3 MB wheel,
  same `DTALite==0.8.1` pin, same failure domain (adr-029's per-dependency isolation
  is already served; a sixth job would isolate nothing). The test step adds
  `tests/test_dtalite_simulation.py` (its engine-gated half un-skips here; its
  engine-free half already ran on the matrix legs); the notebook step becomes
  `-k "dtalite-tap or dtalite-simulation"`; the job comment names both rows.
  `actionlint` clean.
* **Tutorial:** `tutorials/11-external/06-dtalite-simulation.ipynb`
  (`{track: external, unit: dtalite-simulation, requires_extra: dtalite, covers: []}`,
  committed stripped, `find_spec` guard cell). `requires_extra: "dtalite"` already
  has a `find_spec("DTALite")` probe in both maps — **zero probe-map changes** (the
  named contrast with matsim's callable probe). The `_track_manifest["dtalite-simulation"]`
  entry lands UNCONDITIONALLY and atomically with the notebook (the matsim pattern —
  the module imports everywhere); `_ALLOWLIST` stays EMPTY. Headless execution ~8.5 s
  (well inside the 300 s per-cell cap).
* **Docs:** the ROADMAP `zhou2014dtalite` row gains the shipped-as-`dtalite-simulation`
  extension + the R8 non-comparability sentence (the adr-036-promised annotation,
  previously outstanding); `docs/ARCHITECTURE.md`'s external-engines paragraph names
  the third EDOC row; `dtalite_tap.py`'s "a named non-goal here" docstring gains the
  pointer noting `simulation()` now closes the honest-sourcing loop. MODELS.md
  unchanged (EDOC rows are not the `MODEL_REGISTRY` surface); **no new canon entry**
  (`zhou2014dtalite` pre-exists); `tools/generate_references.py` never run.

## Substrate reuse (no hash migration this row)

The row plugs into the EDOC substrate purely as the injected `ReplayRunner` —
`EdocEvaluator`, `field`, `tdsp`, `replay`, `scenario`, `macrorep` are UNCHANGED.
`canon.py` gains `canonicalize_dtalite` + `is_hashed_dtalite_artifact` +
`hash_dtalite_artifacts` under the UNCHANGED version, on the existing `surface=`
plumbing (S3), so every pre-S4 SUMO/MATSim digest is byte-identical. **No new
`EdocScenario` field, so NO hash migration** (the `seed_list` migration was S3's; S4
adds none). The golden static Braess `cf00f411…` is domain-separated and
byte-untouched (re-asserted).

## Adversarial review — S4 finding record

A 3-lens review (1 Opus certificate-soundness + 2 Sonnet lenses, run against the
uncommitted row) found **NO certificate-soundness code defect**. The Opus lens
independently re-derived `RG_D1` from the raw `trajectory.csv` bytes and matched the
harness **bit-exact** (`0.05030897743426841`), and confirmed every gate fires: the
three-layer boost census, the R6 caller-clip/scenario-deadline censor split, the
head-block + lull-drop defenses, hash discipline over every outcome-bearing constant,
the D5 sorted-`vehicle.csv` gate, the G0 pins, the OMP=1 determinism, and the canon
allowlist. Two design points were ratified as reviewed-and-correct:

* **The onset ≤ 0 boost-census count = `n_agents`.** The 1 h variant reports
  `boost_crossing_n = 1000` while only 769 agents completed (231 truncate at the
  period end): the shipped semantics count the whole emission as boost-exposed when the
  horizon is fully boost-covered (a scenario property, version-stable at `n_agents`),
  and the 231 truncations are separately caught at G3. This replaced the dead
  implementer's unverified test guess of 1000 with a measurement-backed definition that
  yields exactly 1000.
* **`certify_emitted` runs the boost census UNCONDITIONALLY** (not only on
  `feasible==1.0`), so `boost_crossing_n` surfaces even when a substrate gate already
  censored — safe because G1 independently censors a doctored X (the boost diagnostic on
  a doctored X is harmless; the emission is already rejected).

The fix batch, all EXECUTED against the live pinned toolchain:

| # | Sev | Finding | Fix | Pin |
|---|---|---|---|---|
| F1 | MAJOR | the DTALite canon additions (`canonicalize_dtalite`, `is_hashed_dtalite_artifact`, `hash_dtalite_artifacts`) had ZERO direct tests — this scoring-integrity anti-forgery code (P2/P9) ran only on the engine-gated `dtalite` CI leg, and adr-040's "exercised in `tests/test_edoc.py`" claim (below) was FALSE | a dedicated engine-free canon block in `tests/test_edoc.py` mirroring the SUMO/MATSim blocks: allowlist surface (bare + path forms), identity/idempotence/decompress-contract, byte-sensitivity incl. the dead `loaded_status` column, and non-surfaced-key invariance | 4 new tests (`test_edoc.py` 54 → 58); the SUMO/MATSim digest pins re-run green |
| F2 | MAJOR (doc) | the ~600 veh/h admission was described as a flat band "[600, 7200)" — it is an integer-boundary STAIRCASE (`cap/3600`): flat 1/interval only for cap ≤ 3600, 2/interval for (3600, 7200] | rewrote the paragraph as the staircase; the reference caps 600 + 1200 both sit in the flat first step (≤ 3600), so every certified number is unaffected (the Opus lens reproduced them bit-exact) | the corrected prose + the unchanged anchors |
| F3 | MINOR | the `t>=600` early-exit was cited as TAPLite.cpp:5386 (a blank line); it is 5387 | fixed 5386 → 5387 in the adapter module docstring, `_assert_no_lull_drop`'s docstring, its runtime `ValueError` text, and the adr-040 table + prose | the corrected citations |
| F4 | MINOR | "`nm -D` exports exactly `DTA_AssignmentAPI` and `DTA_SimulationAPI`" overstated — `nm -D` exports 78 T-symbols | reworded: the Python wrapper CALLS exactly two ctypes entry points out of 78 symbols, whose demangled names corroborate (not prove) the trace-target linkage | the corrected provenance prose |
| F5 | NOTE | the topology-digest vetting excludes `agent_depart`, so a borrowed departure profile reuses the reference's vetting (the Opus lens confirmed `sub_floor` is the working per-instance backstop — a 4.85× borrowed profile lands `sub_floor=1`, not ranked) | a vetting-scope disclosure sentence + a deliberately-deferred strengthening note (folding `agent_depart`/`dt`/`n_intervals` in would perturb the boost-variant same-digest assertion) | the disclosure prose |

## Sources (the honest ledger)

* **MEASURED THIS SPRINT (DTALite 0.8.1 on this box, 2026-07-17; the shipped
  estimator on fresh scratch trees):** the fftt-column probe; the R4 anchors
  (control 0.372244, converged 0.050309, 7.399147× separation, delta 1.753 s); the
  RANKED-vs-floor outcome; the head-block fast-census variant on the 781-agent
  cohort; the lull-drop gate; the D5 sorted-`vehicle.csv` gate; the twin-run byte
  census + allowlist; the R9 781.25/218.75 split; all walls; the reference content
  hash. The `.so` md5, the TAPLite.cpp line citations, the OMP=4 divergence/SIGSEGV,
  the 600 veh/h band, and the boost demonstration reproduce the research record.
* **ATTRIBUTED (canon tool-paper, software lineage):** Zhou & Taylor (2014), *DTALite*
  (`zhou2014dtalite`, pre-exists) — anchors the engine's lineage; the row validates
  ADAPTER + engine fidelity, never the paper's numerics (the adr-027/029/037/039
  posture).
* **Design authority + repo precedents read in full:** ADR-036 (the certificate),
  ADR-037 + ADR-039 (the row precedents mirrored section-for-section), ADR-029 +
  `dtalite_tap.py` (the sibling engine, honest-sourcing loop closed here), ADR-030
  (the superseded deferral), the S4 research dossier (from-scratch anchor
  reconstruction + design). No `references.bib` edit and no
  `tools/generate_references.py` run were made.
