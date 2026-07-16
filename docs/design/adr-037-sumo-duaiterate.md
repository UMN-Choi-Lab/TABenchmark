# ADR-037 — `sumo-duaiterate`: the first EDOC-1 row, and the shipped external-dynamic substrate

**Status:** accepted (shipped)
**File:** `docs/design/adr-037-sumo-duaiterate.md`

## Context — EDOC-1's first shipping row

[ADR-036](adr-036-external-dynamic-observational-certificate.md) is the design
authority for **EDOC-1**, the external-dynamic-engine observational certificate: a
frozen-field best-response gap `RG_D1` with gates G0–G4, a versioned
canonicalization surface (`tabench-edoc-canon-v1;`, R10), and a certifier-owned
multi-OD time-dependent shortest-path harness. It shipped **no code** — like
ADR-030, it is a documentation-only record whose two-round adversarial review
(1 CRITICAL + 11 MAJORs, on a *document*, before any code existed) fixed the design
up front.

This ADR ships the **first row** and, with it, the **shared substrate** ADR-036
specified: the certifier (`tabench.metrics.edoc_gaps`), the canonicalizer
(`tabench.edoc.canon`), the occupancy-aware frozen field + per-first-edge
origin-wait profiles (`tabench.edoc.field`), the non-FIFO-sound TD-SP
(`tabench.edoc.tdsp`), the frozen `EdocScenario` instance (`tabench.edoc.scenario`),
and the replay-harness types (`tabench.edoc.replay`). The row itself is
`sumo-duaiterate` — SUMO's dynamic user-assignment driver `duaIterate.py` on the
shipped `eclipse-sumo==1.27.1` wheel — resolving the ADR-027 `duaIterate` deferral.
`lopez2018microscopic` anchors the software lineage (the tool-paper discipline of
ADR-027/029; the row validates the ADAPTER + engine fidelity, never the ITSC
paper's numerics). Every number below was **measured on this box on 2026-07-16** by
running the pinned wheel; the engine-free substrate is exercised in
`tests/test_edoc.py` and the live engine in `tests/test_sumo_duaiterate.py` (the
sumo CI leg).

## Decision — the row

### `sumo-duaiterate` is an EDOC *producer*, not a `TrafficAssignmentModel`

Unlike `sumo-marouter` (a static macroscopic SUE whose flows are certified against
the scenario's *declared* BPR — ADR-027), `duaIterate` is a **dynamic mesoscopic**
assignment with **no declared cost law** to certify against. Per ADR-036 the engine
IS the instance: the certifier re-derives every scored number by re-running the
pinned engine on the model's emitted plans (G1). So the adapter
(`tabench.models.adapters.sumo_duaiterate.SumoDuaIterateAdapter`) emits the ADR-036
artifact contract — plans `P`, the door-to-door experienced record `X`, and
provenance — for `EdocEvaluator` to certify; it is **not** registered in
`MODEL_REGISTRY` (that registry is the static gap-certified track). All ADR-027
subprocess discipline is inherited: binaries **and** the `tools/assign/duaIterate.py`
driver are addressed only through `sumo.SUMO_HOME` (never the stale
`/opt/sumo-1.12`); `stdin=DEVNULL`; a **single wall deadline** threads netconvert +
every duaIterate iteration + the replay(s) + the R3 duarouter cross-check; a return
code is never trusted (every step re-reads the artifact it wrote); an infra failure
RAISES `RuntimeError` and is never laundered into `feasible=0`; every run is a
`mkdtemp` tree removed in `finally`.

### The instance → SUMO compile is deterministic and fully hashed (substrate extension)

The abstract `EdocScenario` graph carried only `edge_fftt` — not enough to compile a
SUMO net deterministically or to **hash the network's dynamics** (ADR-036 hashes
"the network + demand artifacts" and places capacity drops *inside*
route-distinguishing edges). Two per-edge/scalar fields were added, both hashed:

* **`edge_lanes`** (per-edge `int64`): the ENGINE-side **capacity dial**. SUMO meso
  flow capacity is a function of lanes + freespeed + the pinned meso config —
  **measured ~1584 veh/h/lane at 13.89 m/s** (a saturating single-lane flood: steady
  `entered` ≈ 132 veh / 300 s). A capacity drop is therefore *fewer lanes* on a
  route-distinguishing edge; no separate veh/h field exists because capacity is
  emergent from lanes + speed + config, which is how meso actually works.
* **`canon_speed_mps`** (scalar, default 13.89): the canonical freespeed. Edge
  `length = fftt · canon_speed_mps`, so free-flow time = `length / speed = fftt`
  **exactly**; `numLanes = edge_lanes`, `speed = canon_speed_mps`. Node coordinates
  are a deterministic 2-D layout carrying **no** cost (explicit lengths override
  geometry; meso junction control is off), so the whole net is a function of
  already-hashed fields. A **compile read-back** re-parses `net.net.xml` and RAISES
  if any edge's `numLanes`/`length` did not survive (netconvert's silent 0.1 m
  min-length clamp — the `_sumo_io` hazard).

A third hashed scalar, **`r3_tolerance_s`** (default 15 s), pins the R3 cross-check
agreement bound (parity with `replay_deadline_s`). `semantic_config` is **derived
from the pinned meso option set** so a drift in those options moves the instance hash
(it cannot silently change dynamics under a frozen hash). The mechanical
hash-coverage test (mutate every field → the digest must move; ADR-024) covers the
three new dials; the golden Braess hash `cf00f411…` is byte-untouched (the static
scenario class is domain-separated).

### `X` is produced by the adapter's own pinned replay (never scraped from duaIterate)

Measured on 1.27.1: a bare pinned `sumo` meso replay is **bit-deterministic** (twin
runs byte-identical → the G1 determinism double passes), but it does **not** reproduce
duaIterate's last-iteration tripinfo to the second (152 vs 153 s per agent —
duaIterate's internal `sumo` call uses different options). So scraping `X` from
duaIterate's internal tripinfo would G1-censor every honest run. Resolution: the
adapter emits `P` = the duaIterate final chosen-routes file and derives `X` + the
experienced-cost field from **its own** pinned meso replay of `P`; the certifier's G1
re-runs the *identical* pinned replay (recompiling the net from the hashed fields, so
the emit-time and both certifier replays agree byte-for-byte) and reproduces `X` by
construction. Trips are emitted **junction-to-junction** (`fromJunction`/`toJunction`
+ `duarouter--junction-taz`) so the routes duaIterate chooses are **pure scenario-edge
walks** — no src/sink edges that would break G2 route validity. Meso **teleport is
disabled** (`--time-to-teleport -1`): a gridlocked instance runs to the declared
horizon and shows as backlog / incomplete (censored by G3), rather than silently
teleporting demand away (the ADR-036 head-block-loss hazard); the wall deadline is the
hard stop against a true hang.

**Artifact-contract clarification (refines ADR-036; ADR-036 itself is unchanged).**
This row makes ADR-036's emitted-artifact contract precise: **`X` and the frozen
field are DEFINED by the pinned replay config (a hashed instance field) applied to
`P`; the solver-internal experienced records are provenance, never gated.** The
"pinned replay config" is the pinned meso option set + Δ + seed, all inside the
instance hash (`semantic_config` is *derived* from `_MESO_OPTS`, and Δ/seed are their
own hashed fields), so the map `P ↦ (X, field)` is frozen with the instance. The
justification is measured: duaIterate's last-iteration internal tripinfo differs from
the pinned replay by **152 vs 153 s** per agent (its internal `sumo` call uses
different options), so gating on the solver's self-reported experienced record would
false-censor every honest run. This is the *stronger* reading of "the engine is the
instance," not a weakening: the certified claim is precisely "`P` is best-response
stable under the **pinned replay map**," and the solver that produced `P` (duaIterate
— an MSA-style loop — or anything else) is exactly as model-blind as every other
solver in the benchmark. There is **no self-certification loop**: G1 *independently*
re-runs the pinned replay (recompiling the net from the hashed fields), so a doctored
`X` still diverges from the replay and censors.

### Deliverables (all shipped this sprint)

* **The certifier substrate** (G0–G4 + `RG_D1`), **canonicalizer**, **field builder**,
  **TD-SP**, **`EdocScenario`**, **replay types** — engine-free, tested in
  `tests/test_edoc.py` (**35 tests** after the S2 fix batch's engine-free pins). The
  G0 split adds a **runner engine-pin**: the
  ReplayRunner asserts the installed `eclipse-sumo` version (`importlib.metadata`)
  equals the instance pin **before** replaying (`assert_engine_pin`) — a mismatch
  RAISES `ValueError` (a config error, never a censor). The **walk-enumeration TD-SP**
  is the **reference implementation** of ADR-036's pinned universe ("walks under a
  hashed length bound"); its exhaustive minimum is trivially sound. *Upgrade path:* a
  later row needing a faster label-correcting time-expanded search MUST prove
  equivalence against this reference on the same universe before substituting it.
* **The `sumo-duaiterate` adapter** — compile → duaIterate → pinned replay → the R3
  duarouter cross-check.
* **The row certification entry point** (`certify_emitted`, added in the S2 fix
  batch): the G0–G4 + `RG_D1` substrate certificate (`EdocEvaluator`) PLUS the
  mandatory R3 cross-check (F4) PLUS the separation-vetting assertion (F10), all under
  one wall deadline. `EdocEvaluator` stays engine-agnostic (it cannot run `duarouter`
  nor know a family's engine-run separation status), so the row couples both here —
  the realization of ADR-036 R3's "mandatory cross-check on that engine's rows."
* **The R3 duarouter cross-check** (`duarouter_recost_crosscheck`, **wired into
  `certify_emitted`**): re-cost the driven
  plans with the pinned `duarouter` on the replay's frozen dump and compare, per agent,
  to the substrate's *driven* field cost on the same field (no routing, no origin wait
  on either side — like-for-like). A disagreement beyond `r3_tolerance_s` RAISES: the
  field the certifier scores on must be the field the engine's own weight reader sees.
* **The scenario family** — `build_diamond_scenario` (the parametric O→N1→D / O→N2→D
  diamond with the 1-lane drop on `a2`), `reference_scenario` (the pinned instance),
  `shared_bottleneck_scenario` (the refusal demonstration), and
  `negative_control_separation` (the gate).

### Family constants — re-derived with the SHIPPED estimator (R4)

Per ADR-036 R4, the family floor and separation factor were re-derived with the
shipped estimator (occupancy-aware field + non-FIFO walk-enumeration TD-SP +
per-first-edge origin-wait profile), **not** the pilot estimator. The pinned
`reference_scenario` (2026-07-16, eclipse-sumo 1.27.1, seed 42):

| dial | value | how derived |
|---|---|---|
| topology | diamond, route A (`a1,a2`) fftt 140 s < route B (`b1,b2`) 150 s | A is uniquely free-flow-shorter → AON piles onto the `a2` bottleneck |
| `edge_lanes` | `(2, 1, 2, 2)` | `a2` = the 1-lane route-distinguishing drop |
| `canon_speed_mps` | 13.89 | length = fftt·speed (≈ 972 m / 1042 m) |
| demand | 720 agents, quantum 2.0 s (≈ 1800 veh/h, ~1.14× the 1584 veh/h/lane capacity) | AON saturates the bottleneck; converged splits |
| Δ / `n_intervals` | 300 s / 16 (4800 s horizon) | Δ=300 aggregation; horizon covers completion |
| **separation** | **AON `RG_D1` ≈ 0.139 vs converged ≈ 0.021 → ~6.5×** | declared `separation_factor` **5.0** (margin) |
| **floor** | field-vs-experienced delta ≈ **8.8 s** (converged) | declared `floor_seconds` **15.0** — ~3× **tighter** than the pilot's 25.6 s Δ=300 floor (the occupancy-aware + non-FIFO estimator is less noisy) |
| **R3** | duarouter re-cost vs substrate field cost: max ≈ **5.0 s**, mean ≈ 1.3 s | declared `r3_tolerance_s` **15.0** |
| backlog | max ≈ 0 s at the reference operating point | `backlog_bound` 600 s (catches gross insertion failure only) |

### Track decision — deterministic, single seed (R5/deliverable 6)

Cross-seed converged `RG_D1` measured at seeds {42, 7, 123, 2024} = {0.0215, 0.0232,
0.0274, 0.0211} → **range 0.0063**, well below the resolution `floor_gap ≈ 0.094`
(15 s / mean cost ≈ 160 s). The seed variation is **below the resolution floor**, so
the row is on the **deterministic track with a single pinned seed (42)**, disclosed —
not the R5 stochastic macrorep-list track (that ships with `matsim`, the stochastic
flagship). Timings: emit (compile + duaIterate 18 iters + replay) ≈ 3–4 s; certify
(G1 double + scoring) ≈ 0.5 s; the full sumo-gated test file (**17 tests** after the
S2 fix batch's engine-gated pins, incl. the separation gate's 4 emits + the refusal)
≈ 26 s — within the sumo leg's 2–4 min shape.

## Consequences

* **New code, one CI-leg extension, no hash change to any existing instance.** The
  substrate + adapter + `tests/test_sumo_duaiterate.py` are additive; the sumo CI job
  gains the one file; the golden Braess hash is byte-identical (re-asserted in both the
  substrate and row tests). `eclipse-sumo` stays an optional extra behind the guarded
  import in `models/__init__.py`; the numpy/scipy core (and the EDOC substrate's own
  tests) run without it.
* **R8 leaderboard non-comparability.** `sumo-duaiterate` scores `RG_D1` — the
  **frozen-field best-response gap**, NOT the Wardrop `relative_gap` of the static
  rows. It lives on a **separate leaderboard table** and its number is never compared
  to `sumo-marouter`'s ~1.7e-4 mapping floor or any static `relative_gap` (ADR-036's
  headline: "the frozen-field gap is a different, honestly-labeled quantity on its own
  scale").
* **ROADMAP.** The `lopez2018microscopic` row annotation extends with a
  `sumo-duaiterate` (adr-036/037) unblock pointer + the R8 non-comparability sentence
  (it carries no deferral note to remove). MODELS.md gains no card — the EDOC track is
  not `generate_models.py`'s `MODEL_REGISTRY` surface, and no card exists to annotate
  (hand-annotated like every shipped flip; `tools/generate_references.py` is never run).
* **Named follow-ups (each its own sprint):** the `matsim` row (the P8 macrorep +
  bootstrap-CI stochastic track), the `dtalite-simulation` row (the R9 `vehicle.csv`
  construction + ~600 veh/h inflow law), and the BO4Mob stage-2 D2 certificate — all
  reuse this substrate. The **row's tutorial notebook**
  (`tutorials/11-external/02-sumo-duaiterate.ipynb`, ADR-035 schema
  `{track:"external", unit:"sumo-duaiterate", requires_extra:"sumo"}`) lands in this
  row's commit, once the `11-external` track's `01-sumo-marouter.ipynb` co-exists (the
  within-folder numbering gate is contiguous-from-01).
* **EDOC producers join the tutorials coverage manifest in the same commit that
  registers them.** Because an EDOC row is not in `MODEL_REGISTRY` (the registry-driven
  coverage gate would miss it), the row binds to the tutorials gate explicitly: a
  **guarded** `_track_manifest` entry (`"sumo-duaiterate"` → the adapter's public symbol,
  imported behind the same `ModuleNotFoundError`/`sumo` guard as the adapter itself, so it
  is enforced on the sumo CI leg and invisible on the core-only legs) lands **atomically**
  with the notebook — never via the shrink-only `_ALLOWLIST` (which stays untouched: a new
  unit shipping notebook + manifest in the same commit as the unit IS that invariant
  working).

## Adversarial review — S2 finding record

Two review rounds ran against this row before it lands. **Round 1** (infra-numerics
lens, on the uncommitted code) surfaced the CRITICAL crash-vs-censor launder plus four
MAJORs; **round 2** (soundness-forgery + adr-fidelity lenses, re-run fresh) converged on
the SAME CRITICAL from three independent probes and added the R3-wiring, G2-tolerance,
and separation-vetting findings. All were repro-confirmed against the pinned
eclipse-sumo 1.27.1 wheel and each carries a regression pin that fails under
fix-removal. **No hashed instance constant changed** (the reference/golden hashes are
byte-untouched — the fixes touch certifier *behavior* and *code* constants, not the
instance definition). The batch, deduped:

| # | Sev | Finding | Fix | Pin |
|---|---|---|---|---|
| F1 | CRIT | certifier-side infra faults (deadline pre-exhaustion, missing binary, netconvert/compile failure, read-back mismatch) laundered into `feasible=0` (violates R6) | `PlanReplayFailure` (in `edoc.replay`) is raised ONLY by the pinned `sumo` plan-replay subprocess crashing/timing out; `EdocEvaluator.certify` catches ONLY that; every other `RuntimeError`/`OSError` propagates as the R6 infra RAISE | `test_certifier_censors_only_plan_replay_failure_not_infra` (engine-free) + `test_missing_binary_is_infra_raise_not_censor` |
| F2 | MAJOR | wall-kill orphaned the `sumo` grandchild (`subprocess.run` reaps only the direct child) | `_run` uses `Popen(start_new_session=True)` + `os.killpg(SIGKILL)` + reap on `TimeoutExpired` | `test_wall_kill_reaps_process_group` |
| F3 | MAJOR | hashed `replay_deadline_s` was validated + hashed but never read (certifier replays ran unbounded) | `_intersect_replay_deadline` derives the per-call deadline from the scenario field (tighter caller wall wins), applied in `pinned_meso_replay` + `duarouter_recost_crosscheck` | `test_replay_deadline_s_is_enforced` (0.001 s → infra RAISE, not censor) |
| F4 | MAJOR | the R3 duarouter cross-check was never invoked on any certification path | wired into the new `certify_emitted` row entry point under the shared deadline | `test_certify_emitted_wires_r3_and_requires_separation_vetting` (poison duarouter → fires; forced disagreement → RAISE) |
| F5 | MAJOR | no departure-window construction gate (negative + beyond-horizon departures constructed) | `EdocScenario.__post_init__` refuses departures outside `[0, dt*n_intervals)`; clearing-headroom stays a certify-time G3 concern (engine-dependent) | `test_scenario_construction_gates_raise_valueerror` (3 new cases) + boundary accept |
| F6 | MAJOR | the netconvert 0.1 m clamp read-back was dead code (0.5 m absolute tolerance unreachable) | `compile_net` refuses any edge with `fftt*canon_speed_mps` below `0.1 m × 1.05`; read-back uses relative `1e-3` (the `_sumo_io` precedent) | `test_compile_refuses_subclamp_edge_and_readback_is_relative` |
| F7 | MINOR | G2 departure tolerance was `0.5*departure_quantum` (de-peaking freedom) | tightened to exact-within-`self.tol` (honest emissions carry the trip-table departure exactly) | `test_g2_half_quantum_departure_shift_censors` (engine-free) |
| F8 | MINOR | ambient `SUMO_BINARY`/`DUAROUTER_BINARY`/`NETCONVERT_BINARY` bypassed the wheel-only rule inside `duaIterate.py` (`sumolib.checkBinary` reads them first) | `sumo_env` pins those keys to the wheel's absolute binaries | `test_sumo_env_pins_binaries_over_poisoned_ambient` |
| F9 | MINOR | (a) unparseable read-backs raised raw `ParseError`; (b) temp-dir hygiene glob was mostly blind; (c) `tdsp` docstring claimed a nonexistent construction refusal; (d) count drift | (a) wrap parse/gzip failures to the contract `RuntimeError`; (b) snapshot-diff `tabench-edoc-*`; (c) amend the docstring to the certify-time guard truth (NO DFS pre-count added); (d) counts now 35 engine-free / 17 sumo-gated | `test_read_backs_wrap_parse_errors_as_runtime_error`, rewritten `test_temp_dir_hygiene`, `test_tdsp_docstring_states_certify_time_guard_not_construction_refusal`, this record |
| F10 | NOTE | the negative-control separation refusal was procedural (a separate callable `certify` did not depend on) | `negative_control_separation` marks the FAMILY separation-vetted; `certify_emitted` asserts it before certifying | un-vetted branch of `test_certify_emitted_wires_r3_and_requires_separation_vetting` + `test_negative_control_separates` vetting assertion |

**F1 censor-scope boundary (ratified).** `PlanReplayFailure` is the ONE censor
signal and it fires ONLY on the pinned `sumo` plan-replay subprocess crashing
(nonzero rc) or timing out (the wall hit *during* the replay step). A **missing or
garbage plan-run artifact after `rc=0`** is deliberately an infrastructure
`RuntimeError` RAISE, **not** a censor — because this row disables meso teleport
(`--time-to-teleport -1`), so an invalid or gridlocked emitted plan runs to `--end`
and yields a `tripinfo` whose *missing agents* are censored by G1/G3, never a
missing/garbage file. A file that vanishes or is unparseable after a `rc=0` replay
is therefore an engine/disk anomaly (certifier-side infrastructure), not an invalid
emission. This is the boundary that reconciles R6's first arm (engine crash while
replaying model plans → censor) with its second arm (certifier crash on valid inputs
→ RAISE) for this engine, and it is what lets F1 (three RAISE pins) and F9(a)
(garbage → contract `RuntimeError` RAISE) hold simultaneously. Consistent with
ADR-036 R6 — no certificate semantics changed.

## Sources (the honest ledger)

The row's numbers are the ADR author's **own executed measurements**, not pilot
provenance; the ADR-036 pilot anchors (converged `RG_D1` 0.01526, AON 0.17203, 11×,
floor 25.6 s on the pilot's `net3` with the pilot estimator) are **not comparable in
kind** — a different net and the pilot estimator — and are superseded for the *row* by
the re-derived shipped-estimator constants above.

* **MEASURED THIS SPRINT (eclipse-sumo 1.27.1 wheel, this box, 2026-07-16; fresh
  scratch dirs, never mutating any pilot tree):** the meso ~1584 veh/h/lane capacity
  (single-lane saturating flood); the reference converged/AON/separation/floor/R3/
  cross-seed constants tabulated above; the replay bit-determinism (twin-run
  byte-identical canonical hash) and its non-identity to duaIterate's internal
  tripinfo; the `fromJunction`/`toJunction` + `--junction-taz` route-choice mechanic;
  the compile read-back.
* **ATTRIBUTED UNREAD (canon tool paper, software lineage — tool-paper discipline):**
  Lopez et al. (2018), *Microscopic Traffic Simulation using SUMO*, IEEE ITSC
  (`lopez2018microscopic`, tier 1) — anchors the `duaIterate`/`sumo`/`duarouter`
  toolchain's lineage, per the ADR-027 precedent; Gawron (1998) IJMPC
  (`gawron1998iterative`) — `duaIterate`'s default route-choice dynamics.
* **Design authority + repo precedents read in full:**
  `docs/design/adr-036-external-dynamic-observational-certificate.md` (the certificate
  this row realizes) and `adr-027-sumo-marouter.md` (the subprocess-discipline
  precedent, `_sumo_io` hazards); `src/tabench/models/adapters/_sumo_io.py`
  (`sumo_binary`/`sumo_env`); the shipped EDOC substrate modules. No `references.bib`
  edit and no `tools/generate_references.py` run were made.
