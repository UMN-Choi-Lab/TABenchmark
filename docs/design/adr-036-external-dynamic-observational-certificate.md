# ADR-036: edoc-1 — the external-dynamic-engine observational certificate

**Status:** accepted (design authority — unblocks three deferred rows; no code ships this sprint)
**Date:** 2026-07-16
**Deciders:** external-engine track — the certificate ADR that [ADR-030](adr-030-external-dta-simulators-deferred.md) names as the single unblocker for three rows: MATSim (ADR-030's own deferral) plus the `duaIterate` and `simulation()` follow-ups [ADR-027](adr-027-sumo-marouter.md) and [ADR-029](adr-029-dtalite-tap.md) defer to the DTA/DNL track
**File:** `docs/design/adr-036-external-dynamic-observational-certificate.md`

## Context — the blocker ADR-030 named, and three pilots that measured the way through

ADR-030 deferred three adapter rows — MATSim, DynaMIT, DYNASMART — on a **measured** record,
and named the one ADR that would unblock a *different* trio: MATSim plus the ADR-027/029
named follow-ups `duaIterate` and `simulation()` (DynaMIT/DYNASMART stay blocked upstream —
see Consequences). Its blocker, verbatim: *"The blocker is formulation — A2 is
impossible in kind, not degree. ... MATSim's QSim has no static latency function: each
link's traversal time is `length/freespeed` — a constant — whenever the flow capacity does
not bind ... No `(freespeed, flowCapacity, storageCapacity)` choice produces
`t = fft·(1 + b·(v/c)^p)` on any interior flow range."* And its remedy, verbatim: *"One
ADR: the external-dynamic-engine observational certificate ... It unblocks three queued
rows at once — MATSim, SUMO `duaIterate`, DTALite `simulation()`."* `docs/ARCHITECTURE.md`
records the same gap: external dynamic engines are *"otherwise scored on the observational
track (which for external dynamic engines does not exist yet ... the mandatory cost-matched
anchor is impossible in kind until that certificate ADR ships)."* This is that ADR.

The deferral was correct and stays correct in its own terms: there is no static latency
function to match, so ADR-027/029's shipping bar — the A2 cost-matched anchor, *"the row is
not shipped without A2"* — cannot be met **in kind**. This ADR does not weaken that bar; it
supplies the object that plays A2's role when no cost law exists to match: **the pinned
engine itself**. The certified claim shifts accordingly — not "certified Wardrop UE under
the declared BPR" but "the literature-standard simulation-DTA experienced-time gap w.r.t.
the engine's own frozen realized field," a different, honestly-labeled quantity on its own
leaderboard scale.

Three pilots ran the candidate engines fresh on this box (2026-07-16) and measured every
load-bearing quantity the certificate needs. **SUMO `duaIterate`** (eclipse-sumo 1.27.1
wheel): a 20-iteration meso run emits per-iteration experienced edge-cost dumps, plan/
alternative files, and tripinfo+vehroute; one pinned `duarouter` best-response call on the
frozen final dump recomputes a relative gap on experienced costs model-blind, with a working
negative control (AON `RG_D1` 0.17203 vs converged 0.01526 — the 11x separation ADR-030 said
the dynamic track lacked). **MATSim** (2025.0 release on Temurin 21): `lastIteration =
firstIteration` on the emitted plans is a zero-replanning replay that reproduces all 100
per-agent (departure, arrival, route) tuples exactly, and `global.randomSeed` gives raw-
gzip-byte determinism even with replanning — the first stochastic-track external engine.
**DTALite `simulation()`** (0.8.1): a deterministic sub-second queue-DNL map from an
explicit plan artifact (`vehicle.csv`) to an experienced-time artifact (`trajectory.csv`),
bit-identical across re-runs, whose gap the harness recomputes from `trajectory.csv` alone.
Each pilot also measured the hazards that make the naive design wrong — insertion-backlog
invisibility, self-report substitution, plan-set impoverishment, aggregation-window noise,
head-block infinite loops at `rc=0` — and those hazards *are* the certificate's gates.

## Decision — EDOC-1, the external-dynamic-engine observational certificate

**EDOC-1** scores **D1** (the frozen-field best-response gap) with a **gating slice of D3**
(conservation/self-consistency) and non-gating D3 diagnostics; **D2** (held-out real counts)
applies only where a real sensor panel exists and is never part of this score (its own ADR,
ruling R11). The engine is part of the instance on the **hashed-loading-kernel** precedent of
ADR-031 (`TDTAScenario` hashes the loading operator; the certificate is defined w.r.t. one
kernel) and the **pinned-engine observational** posture of ADR-034's stage 2 (re-run the
pinned engine at emitted inputs, score on its own scale). Putting engine identity + version
**inside the instance hash** is a NEW move relative to ADR-027/029, where the engine's cost
law was matched against the *declared* BPR and its version was manifest provenance + a CI pin
(marouter's A2 anchor existed precisely to SEPARATE the engine's hardcoded law from the
certificate). Here no declared cost law exists to certify against, so the certifier instead
re-derives every scored number by re-running that pinned engine on the model's emitted plans
— the engine is the matched object, not a provenance note.

### The instance

A content hash under a new domain prefix **`"tabench-edoc-scenario-v1;"`**, length-framing
every array (the newell-3det / ADR-031 defense-in-depth, done while the edoc hashes are
unpublished), over: the network + demand artifacts; the **engine identity + EXACT version
pin** (pip wheel version for `eclipse-sumo`/`DTALite`; the release-zip md5 + JDK major for
MATSim); the pinned **seed** (or the pinned macrorep seed **list** on the stochastic track,
R5); the semantic engine config; the cost-field aggregation interval **Δ**; the backlog
bound; the negative-control separation factor; the **resolution floor and replay hard-deadline
constant**; and — the same criterion, **every certifier-side constant that changes a censor /
floor / score outcome** — the **R2 field-semantics + field-completion-rule selections**, the
**origin-wait convention**, and the **option-B BR walk universe + length bound**. Each bears on
the scored outcome (censor / `<= floor` / ranked eligibility), so a post-publication edit to any
of them must mint a new instance hash — the same hash-everything discipline this domain invokes.
Domain-separated, so no existing static/
dnl/dta/tdta hash moves — the golden Braess `cf00f411…` is byte-untouched (re-asserted in
the first shipping row's tests, per every prior ADR).

### Emitted artifacts (mandatory)

A model emits, and only emits:

* **Plans `P`** — per-agent route + departure time (SUMO `vehroute`/`.rou.alt.xml.gz`;
  MATSim `output_plans.xml.gz`; DTALite `vehicle.csv`).
* **Experienced record `X`** — per-agent **door-to-door experienced time including
  pre-insertion delay** (SUMO `tripinfo` `duration + departDelay`; MATSim events; DTALite
  `trajectory.csv`), plus the per-interval experienced link-cost field the run consumed
  (`dump_Δ.xml.gz` etc.).
* **Provenance** — engine configs, version string, seed. Self-reported convergence metrics
  (`duaIterate` relStdDev, engine gap prints) are provenance **only, NEVER gated on** — the
  ADR-029 posture, earned again by measurement (`duaIterate`'s relStdDev was 29x smaller than
  the recomputed gap and read exactly 0.00000 while stuck at pure AON).

### Gates (censor; only wrong shapes / config errors raise)

* **G0 — pin.** The installed-engine version is read at certify time (`importlib.metadata` /
  jar md5 / `sumo` version string), the instance hash and seed are checked; a mismatch is a
  **configuration error and RAISES eagerly** (the ADR-020 eager-config discipline). The pin
  extends to the **model-side execution environment** the engine's byte-determinism requires
  — `OMP_NUM_THREADS=1` for DTALite (its `#pragma omp parallel for` over shared `std::deque`
  is a *correctness* requirement, not just determinism hygiene) and the certifier's **full**
  JDK version (not only the major) for MATSim, since G1 demands canonicalized byte equality
  and a JDK minor/patch drift between the model's and certifier's toolchains is otherwise an
  uncontrolled censor surface. These sit in the instance provenance; a mismatch RAISES.
* **G1 — replay fidelity (the A2 ANALOGUE).** With no static latency function there is no
  cost-matched anchor, so the matched object becomes **the engine itself**: the certifier
  re-runs the pinned engine in **zero-replanning replay** on the emitted plans (one pinned
  `sumo` meso run; MATSim `lastIteration = firstIteration` — measured: replanning fires only
  *between* iterations, so this is a pure replay; DTALite `simulation()` on `vehicle.csv`),
  and the replayed per-agent (departure, arrival, route, experienced-time) tuples must equal
  `X` **exactly** under the pinned canonicalization (R10). The replay runs **TWICE** and the
  two canonicalized outputs must be identical (the determinism gate). Any divergence from `X`
  censors as non-replayable / self-report substitution.
* **G2 — plan-demand match, two-sided.** The agent set, per-agent OD endpoints, and per-OD
  agent *count* match the instance trip table **EXACTLY** — a bijection, no tolerance;
  per-agent departure times match **exactly up to the instance's declared engine-grid
  quantization**. **Fixed departures give the model ZERO timing freedom** (the ADR-031
  demand-match pattern; it is the strongest gate here). ADR-031's `eps_od = tol·max(1, D_od)`
  is inherited as *ancestry, not identity*: it applies ONLY where the trip table is
  real-valued and must be integerized (the R9 DTALite `route_assignment.csv` construction),
  bounding per-OD vehicle-*count* mismatch — never departure-time slack (a timing tolerance
  would reopen the very door this gate closes). A construction-time rule (eager G0-style
  RAISE) restricts the pinned semantic config to **route/selection-only replanning
  strategies**: a departure-time- or mode-mutating strategy (e.g. MATSim
  `TimeAllocationMutator`) is a config error, because its faithful final iterate would break
  the exact-departure bijection.
* **G3 — two-sided delivery.** A per-agent completion census from the replay (arrivals ==
  departures, no period-end truncation, no pre-period head-block loss), **`departDelay`
  included in every cost**, and max/mean insertion backlog under the scenario-declared bound
  (the Primer's vehicles-waiting-to-enter warning, made a gate). Violations censor.
* **G4 — conservation / self-consistency (gating D3).** Per-link entered/left conservation,
  monotone cumulatives, per-agent link-time chains (`arrival_{k+1} >= departure_k`) — the
  `dnl_gaps` C0/C1 shapes run on the replay output.

### The scored metric — RG_D1, the frozen-field best-response gap

From the G1 replay the harness builds the **frozen experienced cost field `Ĉ`** (per-link,
per-interval experienced traversal times, entry-time-resolved at the pinned Δ, R2). For each
agent `i` departing at `t_i`:

* `c_drv(i)` = the **origin-wait profile** for the driven route's first link at `t_i` (a
  per-first-link interval-mean of the replay's own `departDelay` samples, built with the same
  construction and occupancy-aware completion as the link field — the frozen field `Ĉ` cannot
  itself carry off-network insertion delay) **plus** the time-dependent evaluation of the
  **driven** route on `Ĉ` from the agent's first-link entry time;
* `c_br(i)` = the min over the **FULL-NETWORK** route universe of [the origin-wait profile for
  **that route's** first link at `t_i` **plus** the same field evaluation] — a
  **certifier-owned time-dependent shortest path** (R3) starting at the agent's origin, so the
  best response is charged the **alternative** first link's own measured wait.

Both sides compose the *same* frozen profiles (origin-wait + link field), so the driven route is
in the min set and `c_drv(i) >= c_br(i)` per agent **by construction** (the `tdta_gaps`
property, ADR-031). **This DEFAULT scores entrance-choice (insertion-queue) disequilibrium** — a
state where every agent could skip a measured 186 s queue via a cheaper first link no longer
scores 0. A family MAY instead declare the **agent-symmetric** convention (the same agent's
`departDelay_i` on both sides, which cancels in the numerator); it then MUST disclose
**entrance-choice-disequilibrium blindness** in its scope section and its R8 non-comparability
sentence. The origin-wait convention is a **hashed instance field** (the instance section). The
score is

    RG_D1 = Σ_i (c_drv − c_br) / Σ_i c_drv       (TSTT-normalized — ruling R1)

D1 scores the **PREDICTED frozen-cost gap ONLY** — never the realized re-simulation of the
BR plan (ruling R12; the measured overshoots are exactly why: SUMO +9.3%, MATSim naive
one-step −0.252, DTALite 0.394 → 0.513).

**Disclosure (pilot vs shipped estimator).** The pilot anchors quoted below are
*estimator-approximate* to this composition, in two disclosed ways. (i) Each used the
**agent-symmetric** origin-wait convention (the same agent's wait on both sides, cancelling in
the numerator) — now the *optional* convention, not the shipped **default** (the per-first-link
origin-wait profile), so the pilot anchors are **blind to entrance-choice disequilibrium** by
construction; the shipped default charges the alternative first link's own measured wait and is
re-derived on the first row. (ii) Each used a simplified
BR universe/profile: SUMO a pinned `duarouter` on the Δ=300 dump; MATSim experienced driven
cost vs a nearest-neighbor field capped `min(c_br, c_exp)` (the by-construction inequality
does *not* hold in that estimator — the cap is the tell); DTALite an observed-two-route
nearest-neighbor estimator (the exact observational shape forgery pair 3 disqualifies). The
separation factor and floor anchors (R4) inherit this approximation and are **re-derived with
the shipped substrate on the first row before family constants freeze** — the posture R2
already takes for Δ.

### The negative-control anchor and the resolution floor

* **Negative-control anchor (per shipped instance, the ADR-027 incremental-control analogue).**
  The pinned AON / step-0 state must score `RG_D1` at least the declared **separation factor**
  above the reference state (measured separations, **estimator-approximate** — re-derived on
  the first row, see the composition disclosure: 0.17203 vs 0.01526 = **11x SUMO**; 0.393839
  vs ~0.04 **DTALite**). An instance whose control cannot separate — e.g. the meso shared-edge
  queue-storage topology where the cost signal cancels — is **REFUSED at scenario
  construction**, never certified (ruling R4).
* **Resolution-floor gate.** `delta = mean| Ĉ-evaluated driven cost − experienced cost from
  X |`, **always computed on the RAW interval-mean field** (it measures aggregation fidelity,
  not monotonization distance — so option-A monotonization, which deflates `Ĉ` vs experienced
  by a measured mean 189.8 s on a non-FIFO field, never self-fires this gate), must stay under
  the scenario-declared floor (measured **25.6 s mean / 80 s max at Δ=300 ≈ 7.9% of a ~325 s
  trip**). A run whose `delta` exceeds the floor is **CENSORED** (field-unfaithful — this also
  closes intra-interval oscillation gaming); run-specific `delta` only ever **censors**, it
  **never** reclassifies a row to sub-floor. **Sub-floor classification is by the FAMILY floor
  expressed in gap units:** the floor is declared in seconds, and the seconds→gap conversion is
  stated once here as `floor_gap = floor_seconds / (Σ c_drv / N)` (the mean per-agent driven
  cost); a row with `RG_D1 < floor_gap` is reported **`<= floor`, not ranked**, displayed at the
  floor value, tied with other sub-floor rows, and ordered relative to ranked rows **by that
  displayed value like any row** (no categorical placement). Because every *ranked* row scores
  `>= floor_gap` by definition, the ordering is total and no true-worse row can outrank a
  true-better one — closing the inversion a run-specific threshold would open.

### Tier-B (report-never-gate)

The per-departure-interval `RG_D1` profile (the Primer's mandate, flow-weighted, bin = the
pinned Δ, R1) + per-OD `RG_D1`, `n_improvers`, the observational experienced-cost gap from
`X` alone (coverage-limited — vacuously 0 at AON, which is *why* it is not the score), the
realized-BR overshoot (**only when the row enables the optional realized-BR run**, R12), max
queue, the backlog series (which bounds the entrance-choice channel when the agent-symmetric
origin-wait convention is used), the `delta` distribution, and the **BR-path field-coverage
fraction** (the share of the certifier's TD-SP path-link-intervals actually loaded vs filled by
completion — R2), a **disclosure by default**; a family MAY promote the coverage fraction to a
censor gate at construction, in which case it is hashed and listed under **Gates**, not here.

### Seed semantics and the stochastic track

The seed sits **inside the instance hash**. Deterministic engines (DTALite: an LCG keyed by
time step — `seedable=False`) ride the **deterministic track** with no macroreps (disclosed).
`duaIterate` pins `duarouter --seed` / `sumo --seed` as provenance — but it is *seed-dependent*,
and since it ships first (before the MATSim stochastic track), **its track is decided at row
time**: the **row-shipping sprint** must measure a cross-seed `RG_D1` spread on the reference
solver state (in gap units per the floor's declared seconds→gap conversion, after the R4 floor
re-derivation and before family constants freeze) and either (a) justify single-pinned-seed
**deterministic-track** treatment when the spread is below the family floor, disclosing the
spread as the score's seed-conditionality, or (b) fall back to the R5 pinned-seed-list
stochastic shape. A single-seed leaderboard number with
unquantified seed-lottery variance is forbidden. MATSim
(`global.randomSeed`, raw-gzip-byte deterministic) is the first stochastic-track external
engine: **P8 macroreps over a pinned seed list (≥5, ruling R5)** + bootstrap CIs; single-
iteration / single-seed readouts are **forbidden** (measured `RG` noise 0.0012–0.0979 across
it.14..19). Each macrorep emits and certifies **its own** final iterate under its own seed
(G1 per seed); the score is the mean `RG_D1` with a reported bootstrap CI.

### Wall, the crash-vs-censor map, and rc-is-never-trusted

The **certifier** pays the re-run wall — measured marginals 0.092 + 0.131 s (SUMO D1),
~10.6 s per MATSim seed (2× replay + BR; **~71–124 s** for the ≥5-seed macrorep set — R7),
0.04–0.05 s (DTALite). Every certifier engine call gets `stdin=DEVNULL` + a **hard subprocess
deadline** (scenario-declared; the default is a **fixed interpreter/JVM-startup allowance
plus a multiple of the measured whole-run replay wall** — not a bare multiple of the
per-iteration wall, which fails against MATSim's own numbers: 10× the 0.39 s mean per-iteration
wall is 3.9 s vs a measured 3.42 s replay atop ~2.8 s of fixed JVM startup that does not scale,
so a naive multiplier censors an honest replay; ruling R6), and `rc` is **never trusted** —
success is DEFINED as G1 matching (the ADR-029 doctrine; the engines exit 0 on missing/garbage
inputs).

**Crash-vs-censor map (ruling R6).** An engine crash/timeout while replaying **model-emitted
plans** = **CENSOR with diagnostics** — an unexecutable / head-blocking plan is an invalid
emission (DTALite's `loadNewAgents` head-block loops forever at `rc=0` on a pre-period
departure, measured, the canonical case). A crash in the certifier's **OWN** BR / field
computation over valid inputs = **RuntimeError infrastructure RAISE**, never laundered into
`feasible=0`.

### Headline discipline and the leaderboard (rulings R1, R8)

`RG_D1` is a **monotone transform** of the literature-standard simulation-DTA gap w.r.t. the
**frozen realized field** — TSTT-normalized here (R1), with the Primer's SPTT-normalized form
recoverable as `g ↦ g/(1−g)` (Chiu et al. `chiu2011dynamic`, the experienced-time relative gap
over used routes; the Lu–Mahmassani–Zhou lineage `lu2009equivalent`; the Peeta–Mahmassani
experienced-time conditions `peeta1995system`). It is **NOT** a Wardrop-under-
declared-BPR certificate and **NOT** comparable to the static bfw-certified RG scale.
Dynamic-external rows therefore get their **OWN leaderboard table**; the metric column is
named **"frozen-field BR gap (RG_D1)"** with an explicit non-comparability sentence vs static
Wardrop RG, echoed in the ROADMAP/MODELS rows. The re-simulated deviation would change costs
— the `tdta_gaps` Tier-B caveat (ADR-031), disclosed, never scored.

### The twelve rulings (parent decisions; stated, not relitigated)

**R1 — Normalization.** `RG_D1 = Σ(c_drv − c_br) / Σ c_drv`, TSTT-denominator (the house
static-RG shape; all pilot numbers are already in it). *Rationale:* one convention must be
pinned for the dynamic external track, and the TSTT denominator keeps the whole external
family — static (ADR-027/029) and dynamic — on one arithmetic shape. The DTA Primer's SPTT
convention and the shipped `tdue_gap`'s `TC_min` denominator are **disclosed** with the
monotone equivalence `g ↦ g/(1−g)`; the per-departure-interval profile the Primer mandates is
**Tier-B**, flow-weighted, bin = the pinned Δ.

**R2 — Frozen field.** Entry-time-resolved evaluation on interval-mean experienced profiles
at the pinned Δ (a hashed scenario field), with **waiting-not-allowed** TD-SP labels (matching
the engine routers). **The field's temporal semantics are pinned to close a false-certify
hole:** on a non-FIFO frozen field (a queue clearing across a Δ boundary makes an interval-mean
cost drop faster than real time — an overtake), a naive FIFO-assuming label-*setting* Dijkstra
prunes the boundary-crossing path and returns a too-high `c_br`, which **deflates** `RG_D1` (a
false-certify, invisible to G1–G4 — measured on a constructed Δ=300 field: label-setting
`c_br` 380 s where label-correcting finds 115 s). Each scenario family therefore pins
**EXACTLY ONE** field semantics — either **monotonized + label-setting** (FIFO-consistent
lower-envelope arrival-time smoothing of the interval means) OR **raw + label-correcting /
time-expanded** no-wait TD-SP over an explicit universe (simple paths, or walks under a hashed
length bound — under non-FIFO no-wait semantics the optimum need not be simple, so 'min over
routes' must be pinned). **The two are NOT scored-outcome-equivalent** (measured on the
constructed field: `c_br` 455 / 405 / 400 across option-B-simple-paths / option-B-walks /
option-A on one emission — disclosed), so the choice, and option B's universe + length bound,
are **hashed instance fields**; the `delta` resolution-floor gate is defined on the **raw**
field regardless (above), because it scores aggregation fidelity, not monotonization distance.
**Field completion is occupancy-aware and normative** (moved here from the forgery analysis):
never-loaded links stay at free flow (optimistic *for* the deviation — the
plan-set-impoverishment defense); an interior gap of a **loaded** link carries forward the last
congested cost **ONLY where the replay shows nonzero occupancy / a standing entrance queue**
during that gap (all three engines' artifacts reconstruct occupancy), and a **zero-occupancy**
interior gap falls back to free flow (deviation-optimistic, pair-3-consistent). A *blind*
carry-forward would let two bracketing congested samples **poison every interior interval** of
an alternative route — the burst-poisoning attack (forgery pair 8), deflating `RG_D1` ~100× at
the cost of a handful of agents; occupancy-awareness is the detection that closes it. Δ must be
chosen **per scenario family** so the measured field-vs-experienced `delta` floor is below the
family's declared resolution. *Rationale:* the RULE ships here; the numeric Δ default ships with
each row, justified by measurement (300 s is documented too coarse — the 7.9% floor).
**Non-FIFO and poisoned-alternative regression instances** are named substrate deliverables
(the pilots' 4–6-link nets never produced link costs near Δ nor the burst topology, so both
regimes are untested by the record). This closes interval-aggregation gaming, intra-interval
oscillation, the non-FIFO deflation hole, and the carry-forward poison.

**R3 — BR oracle.** The certifier-owned time-dependent shortest path is **NORMATIVE
everywhere**; where a pinned engine router exists (`duarouter` on the frozen dump with
`--weights.expand --skip-new-routes --keep-all-routes --routing-threads 1`) it serves as a
mandatory **CROSS-CHECK** on that engine's rows — disagreement beyond the declared tolerance
is an **infrastructure RAISE, never a censor**. *Rationale:* owning the SP is strictly
stronger than ADR-031's declared-path-universe disclosure (the certifier supplies the
counterfactual, so hidden cheap paths score against the plan); the cross-check keeps a pinned
production router honest against the harness implementation. **The harness multi-OD TD-SP
module is a named deliverable of the FIRST shipping row** (the repo's DNL ladder has the
ingredients — pilot C's named remaining piece).

**R4 — Floors / factors.** The `delta` floor and the negative-control separation factor are
declared **PER SCENARIO FAMILY at construction** (eager gates). *Rationale:* the ADR sets the
mechanism + the **estimator-approximate** anchors (**11x SUMO, ~10x DTALite** — each measured
under its pilot's simplified estimator, not the R2/R3 composition; separation-factor **default
≥5**); the anchors are **re-derived with the shipped substrate on the first row before the
family constants freeze** (extending R2's Δ posture to the separation factor and floor),
because the resolvable floor is engine- and net-specific (the 6 s DTALite quantization floor
differs from the 300 s SUMO meso aggregation floor).

**R5 — MATSim stochastic shape.** P8 macroreps over a pinned seed **list (≥5)**; each
macrorep emits and certifies **its own** final iterate under its own seed (G1 per seed); the
score is the mean `RG_D1` with a **bootstrap CI** reported. Last-k within-run averaging is
**Tier-B diagnostic only**. *Rationale:* mixing states across iterations would blur the
certified-artifact identity (G1 must replay a single, fixed emission); a model never chooses
its seed, and a single-iteration readout is noise-dominated (RG 0.0012–0.0979).

**R6 — Replay timeout.** Replaying model-emitted plans that hang/crash = **CENSOR with
diagnostics** (the DTALite head-block infinite loop at `rc=0` is the canonical case); default
wall = a **fixed interpreter/JVM-startup allowance plus a multiple of the measured whole-run
replay wall**, on the mean not the min, family-declarable. (NOT a bare 10× the per-iteration
wall — that censors MATSim's own honest 3.42 s replay: 10× the 0.39 s mean per-iteration wall
= 3.9 s barely clears it, 10× the 0.34 s min already fails, and ~2.8 s of the replay is fixed
JVM startup that does not scale with per-iteration wall.) Failures in the certifier's OWN
BR/field computation = **RuntimeError RAISE**. *Rationale:* an
unexecutable plan is an invalid emission (the model's fault → censor); a certifier crash on
valid inputs is infrastructure (never laundered into `feasible=0`).

**R7 — Scale.** The per-row CI budget follows the ADR-027 **2–4 min job shape**; HPC-scale
instances are **refused at construction** (the `Bo4MobHpcOnlyError` precedent, ADR-034);
big-net certifier walls are measured per row. The stochastic-track marginal is
`n_seeds × (2×replay + BR) + toolchain` — for MATSim at the R5 floor of 5 seeds,
`5 × (2×3.42 + 3.71) + 17.8 ≈ 71 s` (≈ 124 s if each macrorep's certified run is also
regenerated in CI), NOT the ~10.6 s single-seed figure; the 2–4 min budget holds at exactly
5 seeds, so **each family must bound `n_seeds`** at construction or it silently busts R7.
*Rationale:* the certifier pays the re-run wall and MATSim's 550–772 MB JVM baseline is the CI
sizing constraint; DTALite big-net scaling is unmeasured beyond 4 links, so each row measures
its own wall before shipping.

**R8 — Leaderboard.** Dynamic-external rows get their **OWN table**; the metric column is
named **"frozen-field BR gap (RG_D1)"** with an explicit non-comparability sentence vs static
Wardrop RG; ROADMAP/MODELS rows say it. *Rationale:* the headline discipline of ADR-025/027/
029/034 — the score names whose equilibrium and which axis, and this is not the static
bfw scale (the ADR-034 stage-2 separate-table posture).

**R9 — DTALite plans.** The adapter constructs `vehicle.csv` from `route_assignment.csv`
volumes (the engine emission is **dead code** — `TAPLite.cpp:2238` `route_volume = 0`); the
integerization rule + its quantization bound are pinned in that row's ADR; `assignment()`-
seeded plans are **adapter plumbing, NOT a model row**. *Rationale:* `assignment()`'s
`vehicle.csv` is always header-only, so plans must be built from the FW split; the integer-
agent quantization is adapter policy whose noise bound must be disclosed like any mapping
floor.

**R10 — Canonicalization.** A **versioned, hashed harness module**, domain prefix
**`"tabench-edoc-canon-v1;"`** (trailing semicolon, the house form), length-framed per
ADR-031. Its founding spec is the **four measured necessities**: strip the SUMO `generated on`
timestamp comment + `summary` duration attr; sort MATSim same-timestamp event ties; hash the
**decompressed** payload; positional-parse the DTALite trajectory (13-header / 12-field rows).
The spec also **defines an explicit hash surface**: the four strips do NOT make the whole
emitted tree byte-identical (measured: **23/184** SUMO files — `*.sumo.log`, `driver.out`,
`dua.log`, `stdout.log` — still differ on wall-clock text after stripping), so **only
simulation-state artifacts are hashed** (tripinfo, vehroute, dumps, `.rou`/`.rou.alt`,
configs; events; trajectory) and engine/driver logs are provenance, **never** on the G1 hash
surface — else the determinism double would over-censor every honest run. Upstream format
drift **bumps the version** = new instance hashes, disclosed. *Rationale:* every one of the four is a measured
false-hash surface from a pilot; a hashed, versioned module makes the canonicalization itself
auditable and its drift explicit.

**R11 — D2 boundary.** The BO4Mob stage-2 held-out-count certificate gets **its own ADR** but
**SHALL reuse this ADR's replay + canonicalization substrate**; this ADR defines that
substrate interface (the ADR-034 co-design). *Rationale:* D2 (real-sensor NRMSE on held-out
dates) is a different scored object with no true OD and no declared BPR, but it re-runs the
pinned engine at emitted inputs exactly as G1 does — shared plumbing, separate score and
scale.

**R12 — Realized-BR.** The realized one-step re-simulation is **never part of the SCORE or any
gate**; it ships as an **optional per-row Tier-B diagnostic** run (budgeted in R7 when a row
enables it), because it is overshoot-noisy by construction — measured **+9.3% SUMO, 0.394 →
0.513 DTALite** — so a re-simulated full deviation changes the field it was measured against
(everyone deviates simultaneously) and is not a gap bound. **No small-fraction-reroute variant**
either (an unprincipled dial). The overshoot numbers appear in the ADR as the **REASON D1
scores the predicted frozen-field gap**. *Rationale:* a fractional-reroute knob would be a
tunable the benchmark cannot defend; the optional realized run is a probe (forgery pair 8),
never a score, so the certifier's default path does not double engine calls.

## Forgery analysis — twelve attack/defense pairs

Named so an adversarial review attacks them; each carries a tag and a measured defense.

1. **Unreached-demand hiding** — *[GATE].* Insertion backlog (`departDelay`) is off-network
   waiting invisible to every edge dump (measured mean **186 s / max 414 s** on the pathology
   nets); DTALite silently drops pre-period departures AND head-blocks later agents on the
   same first link, and truncates in-flight agents at period end — all at `rc=0`. **Defense:**
   G3 two-sided delivery — a per-agent completion census from the pinned replay (arrivals ==
   departures; DTALite completion inferred from `current_link_seq_no`, never the dead
   `loaded_status` flag), `departDelay` in every scored cost, a scenario-declared backlog
   bound, and a construction gate quantizing departures into `[start, end)`.

2. **Self-report substitution** — *[GATE].* Emit doctored `tripinfo`/`trajectory`/experienced
   costs, or lean on the engine's own convergence flag (`duaIterate` relStdDev measured **29x
   smaller** than the recomputed gap and **exactly 0.00000 while stuck at pure AON**;
   DTALite's printed gap uses a different normalization and can be negative/frozen).
   **Defense:** G1 replay fidelity — the pinned engine re-run reproduces every per-agent tuple
   exactly under the pinned canonicalization (doctored artifacts diverge byte-canonically:
   MATSim sorted-stream md5 identical on both sides; DTALite trajectory md5-pinned; SUMO
   bit-identical modulo timestamp text). Engine self-reports are provenance-only.

3. **Plan-set impoverishment** — *[STRUCTURAL + GATE].* Emit only plans the best-response step
   cannot improve because alternatives were never loaded (a deliberately tiny route universe);
   pure observational scoring gives AON `RG=0` by construction (measured: only **6/10** depart
   bins observe both routes even at the fixed point). **Defense:** the BR route universe is the
   FULL network via the certifier-owned TD-SP (R3), so unloaded links enter the frozen field at
   free flow — optimistic *for* the deviation — and hidden cheap paths score AGAINST the plan
   (measured: the AON / step-0 states score **0.17203** (SUMO, pure AON — all 500 on one route)
   **/ 0.393839** (DTALite, the step-0 781/219 FW proportional split), not 0). Strictly stronger
   than ADR-031's
   declared-path-universe disclosure. The residual shared-edge-cancellation case is closed by
   the construction-time negative-control separation gate.

4. **Seed shopping** — *[GATE].* Run many seeds, emit the luckiest state. **Defense:** the
   seed is inside the instance content hash; the emission must replay under **the** pinned seed
   or G1 censors. Stochastic-track rows (MATSim) are scored as P8 macroreps over a PINNED seed
   list with bootstrap CIs (R5) — a model never chooses its seed.

5. **Cost-averaging games** — *[GATE].* Emit seed-averaged or smoothed artifacts to flatter
   the gap. **Defense:** an averaged artifact is not any fixed-seed replay output, so G1
   censors it; each macrorep's artifacts must replay under their own pinned seed.

6. **Engine-version drift** — *[GATE + RAISE].* Solve under a newer/better engine build, claim
   the pinned version. **Defense:** version is read from the INSTALLED engine at certify time
   and sits in the instance hash (G0: mismatch RAISES as a config error); G1's replay under the
   pinned build diverges byte-canonically from another build's artifacts. CI pins the tested
   engine exactly (the ADR-027/029 floor-not-pin + CI-pin split).

7. **Aggregation-window gaming** — *[GATE + DISCLOSED-SCOPE].* Shape within-interval congestion
   so the interval-averaged field under-reports the driven plan's cost (measured field-vs-
   experienced divergence **25.6 s mean / 80 s max at Δ=300, ~7.9%**). **Defense:** same-basis
   scoring (driven AND best-response both on the same frozen field, driven route in the min
   set) keeps the gap `>= 0` by construction; the `delta` field-vs-experienced gate censors
   when the field no longer represents experienced costs; Δ is a pinned scenario field required
   much smaller than the congestion timescale, and gaps below the **family** floor (in gap units,
   per the floor-gate conversion) are reported `<= floor`, not ranked.

8. **Poison-the-alternative** — *[STRUCTURAL + DISCLOSED-SCOPE].* Route some of the instance's
   own agents onto alternative routes to inflate their frozen-field costs so the best response
   stays home — including the **burst-poisoning** variant: reroute only the first- and
   last-departing agents onto an alternative and congest it inside *their own* intervals, so a
   blind carry-forward completion would paint every interior interval congested (measured:
   `RG_D1` 0.4024 → 0.0040, a ~100× deflation, at the cost of a handful of agents). **Defense:**
   G2's demand-match bijection forbids phantom demand, so poison agents are real agents whose
   experienced cost lands inside `Σ c_drv` — the attack raises the score it tries to lower unless
   it actually congests the alternative, i.e. approaches a frozen-field equilibrium, which is
   exactly the certified object; and R2's **occupancy-aware** field completion refuses to carry a
   congested cost across an interior gap the replay shows **empty** (zero-occupancy → free flow),
   so the burst cannot paint the intervals between its two samples. The remaining wedge
   (frozen-field equilibrium vs re-simulated DUE) is the literature-standard caveat, disclosed in
   the headline and probed by the **optional** realized-BR Tier-B diagnostic (R12), never gated.

9. **Hash games on volatile bytes** — *[GATE].* SUMO's `generated on` timestamp comment in
   every XML, summary wall-clock duration attrs, MATSim same-timestamp event-order permutation
   between replay and original, gzip metadata, DTALite's 13-header/12-field trajectory rows.
   **Defense:** the pinned canonicalization spec (versioned, hashed harness module,
   `tabench-edoc-canon-v1;`, R10) — strip timestamp comments and duration attrs, sort same-
   timestamp events, hash decompressed payloads, positional trajectory parse — all four
   measured necessities from the pilots.

10. **Departure-time gaming** — *[GATE].* Shift departures to dodge congestion while claiming
    the instance's demand (the ADR-031 timing-freedom door). **Defense:** G2 requires per-agent
    departure times equal to the instance trip table **exactly, up to the declared engine-grid
    quantization** — NOT at `eps_od`, which is a vehicle-*count* tolerance for the R9
    integerization only (a timing tolerance would reopen this door); fixed departures give zero
    timing freedom, the ADR-031 demand-match pattern.

11. **Emit the certifier's own best-response output as the plan** — *[NO DEFENSE NEEDED].* It
    is legal optimization, and self-defeating: the BR plan re-scored under ITS OWN replay field
    shows the measured simultaneous-deviation overshoot (**DTALite RG 0.393839 → 0.512861;
    SUMO +9.3%**), so gaming the metric means actually solving the fixed point.

12. **Vacuous or degenerate instances** — *[RAISE at construction].* Zero/sub-floor demand,
    unclearable horizons, the DTALite clearance-boost overlap, shared-edge-queue topologies
    where every state self-certifies. **Defense:** RAISE at scenario construction (the ADR-020/
    031 eager-config discipline): sub-floor demand, unclearable horizon, boost-window overlap
    (the DTALite **x10 discharge boost fires whenever `total_intervals − t <= 720` intervals**
    — a units bug, `2·60·6` where 2 h *should* be 1200 six-second intervals, so the boost is ON
    for the last **72 min of ANY horizon**; a `≥2 h period` does NOT exclude it, and the
    constructor must instead refuse unless the completion census shows **every agent exits
    before `total_intervals − 720`** (horizon ≥ clearing time + 72 min) on **every
    constructor-side run — reference, negative control, and any anchor-derivation iterates** —
    **re-asserted at certify time on the replay**, where a replay whose census crosses
    `total_intervals − 720` is a **CENSOR** with diagnostics (an invalid emission, R6), never a
    construction RAISE), and a non-separating AON negative control are the constructor refusals
    — config errors, never certified rows.

## Per-engine unblock record

The **first shipping row** carries the shared deliverables: the **EDOC certifier substrate**
(G0–G4 + the RG_D1 scorer), the **canonicalization module** (`tabench-edoc-canon-v1;`, R10),
and the **general multi-OD TD-SP harness module** (R3, with the non-FIFO and poisoned-alternative
regression instances).
SUMO `duaIterate` is the natural first
row — it has the lowest certifier wall and a pinned `duarouter` that supplies the R3 cross-
check for the harness TD-SP — with MATSim as the stochastic-track flagship.

### SUMO `duaIterate` — UNBLOCKED (resolves the ADR-027 deferral)

Measured on the shipped `eclipse-sumo 1.27.1` wheel (`SUMO_HOME` via `sumo.SUMO_HOME` only,
never the stale `/opt/sumo-1.12`). Per-iteration artifacts are complete (`tripinfo` +
`vehroute` + `dump_Δ` + `.rou`/`.rou.alt` + pinned configs); bit-deterministic under pinned
seeds modulo timestamp text — **all simulation-state payloads** (tripinfo, vehroute, dumps,
`.rou`/`.rou.alt`, configs) byte-identical after the two strips (the `generated on` comment +
the `summary` duration attr); the **23/184** `*.sumo.log` / `driver.out` / `dua.log` /
`stdout.log` files carry wall-clock text and are **excluded from the hash surface** (R10).
Full 20-iteration run **6.73 s / 6.44 s (0.30–0.34
s per iteration)**. D1 is one pinned `duarouter` call on the frozen final dump + optional one
pinned `sumo` replay: measured **0.092 s + 0.131 s**. **Converged `RG_D1` = (157005.87 −
154609.65)/157005.87 = 0.01526** (126/500 strict improvers, BR split 248/252) vs the step-0
AON control **`RG_D1` = (174381.01 − 144382.31)/174381.01 = 0.17203** (285/500 improvers) —
the **11x** attributable negative control ADR-030 said the dynamic track lacked. (These
anchors use the pilot estimator — pinned `duarouter` on the Δ=300 dump, origin wait excluded
on both sides — and are re-derived under the R2/R3 composition on the first row.) The
observational gap from `X` alone (Tier-B) is **RG_global 0.03887, RG_binned 0.04699 with only
6/10 depart bins observing both routes** — coverage-limited, which is why it is not the score.
The cost basis MUST be `duration + departDelay` (the engine's own convergence metric excludes
`departDelay` and "converges" while hiding door-to-door cost: measured hidden backlog mean
**186 s / max 414 s**). The Δ=300 aggregation noise floor (**mean 25.6 s / max 80 s ≈ 7.9%**)
must shrink or be disclosed as the gap resolution. The realized-BR run overshoots (**meanCost
353.87 s vs 323.88 s, +9.3%, realized RG 0.159**) — Tier-B, never the score (R12). The
scenario family must place capacity drops **inside** route-distinguishing edges (meso stores
bottleneck queues on the upstream edge; the shared-edge topology is refused by the negative-
control separation gate). **Row deliverables:** the `sumo-duaiterate` model + the shared
substrate/canon/TD-SP; the `duarouter` R3 cross-check; the departDelay backlog gate.

### MATSim — UNBLOCKED (supersedes the ADR-030 formulation blocker)

Two pilot amendments are baked into the certificate. **(1)** Replay = `lastIteration =
firstIteration` on `output_plans.xml.gz` — MATSim replans only *between* iterations, so this
is the zero-replanning experienced-cost recompute (measured: reproduces all 100 per-agent
(dep, arr, route) tuples exactly; **sorted-stream md5 `a8d8e6f60497e468b8908f2d9a73d61a`
identical on both sides**). **(2)** `RG_D1` is evaluated harness-side against the FIXED replay
field (**measured 0.0916 at the it.19 state**), never the executed one-step (naive gap
**−0.252**: `ReRoute 1.0` moved 100/100 agents on its coarse 15-min-bin field and overshot).
The comparator sensitivity on that ONE state — **route-average RG 0.0045 vs harness fixed-
field BR gap 0.0916 vs naive engine one-step −0.252** (C0 248.0 s → C1 310.5 s) — is exactly
why R1/R2 pin one definition. (The 0.0916 is the pilot estimator — experienced driven cost vs
a nearest-neighbor field, capped `min(c_br, c_exp)` precisely because `c_drv >= c_br` does not
hold there — re-derived under the R2/R3 composition on the first row.) First stochastic-track
external engine: `global.randomSeed`
gives raw-gzip-byte determinism even with replanning (events, links, AND plans), so **P8
macroreps over a pinned seed list + bootstrap CIs** are executable (R5); single-iteration
readouts are noise (route-average RG **0.0012–0.0979 across it.14..19**). Canonicalize event
ties (same-timestamp order permutes between replay and original) before any hash; derive flows
from entered-link events (`output_links.csv` undercounts the arrival link). **Version pin:
`matsim-2025.0-release.zip` md5 `c65f35eafabea2456b818875f08048ca` on Temurin 21.0.11+10.**
Walls: certified run **10.62 s / 772 MB**, replay **3.42 s / 312 MB**, BR run **3.71 s**,
fresh toolchain **17.8 s**, **~10.6 s per seed** (2× replay + BR); the ≥5-seed macrorep
marginal is **~71–124 s** (R7), and the JVM RSS baseline is the CI sizing
constraint. **Row deliverables:** the `matsim` stochastic-track row (the first agent-
based, first stochastic-track external engine); the P8 macrorep + bootstrap-CI harness for
this certificate; the event-tie canonicalization (R10).

### DTALite `simulation()` — UNBLOCKED (resolves the ADR-029 deferral)

`simulation()` 0.8.1 is a deterministic (LCG keyed by time step — `seedable=False`,
deterministic track only, no macroreps, disclosed), version-pinnable, **sub-second (0.04–0.05
s)** plan-to-experienced-time map: `vehicle.csv` in, `trajectory.csv` out, bit-identical
(**md5 `655b1dde23c6d1e3fb94779b4ffec81e` re-verified across three dirs**, also at OMP=4 on
the 4-link net — but `OMP_NUM_THREADS=1` stays a **correctness** requirement, not just
determinism hygiene: raw `#pragma omp parallel for` over shared `std::deque`). G1 replay is
exact; **`RG_D1` from artifacts alone = (22654.000 − 13731.967)/22654.000 = 0.393839** (706/
1000 would-switch agents, 1000/1000 completed); the harness MSA `1/(k+2)` loop drives it to
**~0.0397 in 12 pinned re-runs** (plateau 0.03–0.05 = the 6 s quantization floor), so the dial
is meaningful. (This gap uses the pilot's observed-two-route nearest-neighbor estimator — not
the R2/R3 interval-mean field + TD-SP — and is re-derived under the shipped composition on the
first row.) The engine has **no router** (`simulation()` never routes; `assignment()` never
simulates), so the BR is certifier-owned TD-SP + one pinned re-simulation — which *strengthens*
P1. Mandatory adapter gates, all from measured `rc=0` silent failures: departure-window
`[start, end)` + per-agent completion census; positional trajectory parse (13-col header /
12-field rows; `loaded_status` dead; `07:00:00` filler on unvisited links); **plans constructed
from `route_assignment.csv` since `assignment()`'s `vehicle.csv` emission is dead code
(`TAPLite.cpp:2238` `route_volume = 0`, R9)**; an **agent-level boost-clean horizon** (every
agent must exit before `total_intervals − 720`; the x10 discharge-boost `<=720`-interval units
bug means a bare `≥2 h period` does not suffice — see below); links sorted by `(from, to)`. The
**~600 veh/h effective-inflow law** (`entrance_queue.size() < capacity/3600` admission; measured
**36.5 min mean vs 14 free-flow** on a nominal-1200 link) is **documented as the engine's cost
law inside the instance definition** — no A2 cost-match exists, consistent with ADR-030. The
unsmoothed one-step BR overshoots as theory predicts (**TSTT 22654.000 → 33365.933, RG 0.393839
→ 0.512861**) — Tier-B, not the score (R12).

**Boost-clean re-measurement (this review-revision sprint, verified on disk).** The pilot's
anchors were measured on a 7→9 h (2 h) horizon whose last 72 min (from 7:48) run under the x10
discharge boost — 463/1000 agents are in flight then, 37.3% of veh-min fall in the window.
Re-running the identical 781/219 step-0 split on a **7→13 h horizon** (boost window 11:48
onward; every agent — base, one-step BR, and all 12 MSA iterates — exiting by ≤ 8:47 h, so
**provably boost-clean**) reproduces **every anchor byte-for-byte**: `trajectory.csv` md5
`655b1dde…` identical to the 2 h run, `RG_D1` 0.393839, TSTT 22654.000, 706 would-switch, the
one-step overshoot 22654.000 → 33365.933 (`RG_D1` 0.512861), and MSA final 0.039738. The boost
fired but changed nothing here because it inflates only the **discharge** `base_capacity`
(`calculateOutflowCapacity`, `TAPLite.cpp:5127`), while the binding constraint on this net is
the **un-boosted entrance admission** `entrance_queue.size() < capacity_per_time_step =
Link_Capacity/3600` (`TAPLite.cpp:5301`/`4979`) — the ~600 veh/h inflow ceiling. So the
anchors stand; the gate is still required, because on a net where **discharge** binds the boost
WOULD move experienced times — hence the agent-level completion census above (which CENSORs an
emission whose certify-time replay crosses `total_intervals − 720` as invalid per R6, and at
construction must pass on every constructor-side run — as this re-measurement censused across the
base, one-step BR, and all 12 MSA iterates), not a bare period length. **Row deliverables:** the `dtalite-simulation` deterministic-track row; the
`vehicle.csv`-from-`route_assignment.csv` construction with its integerization bound (R9); the
positional-parse + departure-window + completion-census adapter gates.

## Consequences

* **This ADR is the design authority for the EDOC certifier substrate, the canonicalization
  module, and the TD-SP harness — which ship with the first row, not in this sprint.** No code,
  no dependency, no CI job, no hash change lands here; it is a documentation-only record, like
  ADR-030.
* **Supersedes ADR-030's MATSim formulation blocker** (A2 impossible in kind) **and resolves
  the ADR-027 `duaIterate` and ADR-029 `simulation()` named-follow-up deferrals** — the three
  rows ADR-030 said one ADR would unblock at once. A2's shipping bar is not weakened: its role
  is played by the pinned engine under **G1 replay fidelity**, and the certified claim is
  relabeled (frozen-field BR gap, not Wardrop RG).
* **ADR-030 forecast the C0–C8 DNL ladder as the natural scoring substrate;** EDOC-1 narrows
  that: it realizes C0–C8 as the **gating D3 slice** (G4 runs the `dnl_gaps` C0/C1 conservation
  shapes on the replay) and moves the *score* to `RG_D1`, because conservation alone cannot
  separate a non-equilibrium AON state from equilibrium — the measured 11x negative control is
  a frozen-field BR gap, not a conservation residual. The narrowing is disclosed here so a
  reader diffing the two ADRs finds it named.
* **ADR-030's DynaMIT artifact-block and DYNASMART license-block are UNCHANGED.** DynaMIT
  (`benakiva2001network`) has zero public artifact; DYNASMART (Jayakrishnan et al.,
  `jayakrishnan1994evaluation`) is FHWA/McTrans-licensed with no CI-installable binary. Those
  need upstream changes no repo ADR can supply; the Peeta & Mahmassani (1995) white-box row
  (`peeta1995system`, shipped as ADR-031) is unrelated and already live.
* **Named follow-up rows and deliverables** (each its own adversarial-review sprint):
  * **First shipping row — `sumo-duaiterate`** (SUMO `duaIterate`) + the **shared substrate**
    (G0–G4 + RG_D1 scorer), the **canonicalization module** (`tabench-edoc-canon-v1;`), and the
    **multi-OD TD-SP harness** with the `duarouter` R3 cross-check and the non-FIFO +
    poisoned-alternative regressions.
  * **Second row — `matsim`**, the first agent-based / first stochastic-track external engine:
    the P8 macrorep + bootstrap-CI harness for this certificate, on the pinned `matsim-2025.0` /
    Temurin 21 build.
  * **Third row — `dtalite-simulation`** (DTALite `simulation()`, deterministic track): the
    `vehicle.csv`-construction rule (R9) + the positional-parse / departure-window / completion-
    census adapter gates + the documented ~600 veh/h inflow law.
  * **Then — the BO4Mob stage-2 D2 held-out-count observational certificate** (`ryu2025bo4mob`):
    its **own** ADR (R11), reusing this ADR's replay + canonicalization substrate — the ADR-034
    co-design (ADR-034: *"the honest-oracle surface arrives with the stage-2 certificate ADR"*;
    *"Do not start it until the certificate design is written."*), now satisfied for the
    substrate interface.
* **ROADMAP/MODELS (to reality):** the ROADMAP Horni et al. (2016) `horni2016multiagent` row
  replaces its adr-030 deferral note with a pointer here; the **shipped** `lopez2018microscopic`
  (marouter) and `zhou2014dtalite` (tap) rows **extend** their annotations with the `duaIterate`
  / `simulation()` unblock pointer + the R8 non-comparability sentence ("frozen-field BR gap,
  not Wardrop RG; separate leaderboard table") — they carry **no** deferral note to remove; and
  MODELS.md gains cards only when the rows ship, via `tools/generate_models.py` — there is
  nothing to annotate there now (no MATSim/duaIterate/simulation() card exists). Hand-annotated
  like every shipped flip — `tools/generate_references.py` is **never** run.

## Sources (the honest ledger)

Reproduced from the theory synthesis; the read-in-full vs attributed-unread distinctions are
preserved and not upgraded, and the ADR author's own repo reads are listed separately.

* **READ (open, fetched + text-extracted this session):** Chiu, Bottom, Mahut, Paz,
  Balakrishna, Waller & Hicks (2011), *Dynamic Traffic Assignment: A Primer*, TRB
  Transportation Research Circular E-C153 (`chiu2011dynamic`) — the experienced-time relative
  gap over used routes normalized by total shortest-path times; the mandate that the gap "must
  be calculated for (and reported by) each departure time interval"; the MSA critique that
  flow-change stability is "an algorithmic construct unrelated to the equilibrium requirement of
  minimizing experienced travel time"; vehicles-waiting-to-enter as a warning measure (grounds
  the backlog gate).
* **READ (open, fetched this session):** SUMO documentation, "Dynamic User Assignment"
  (`sumo.dlr.de/docs/Demand/Dynamic_User_Assignment.html`) — `duaIterate` convergence is the
  relative standard deviation of average travel times (`--max-convergence-deviation` /
  `--convergence-iterations` / `--convergence-steps`); the docs themselves caution convergence
  "is hard to predict and results may continue to vary even after 1000 iterations"; no
  experienced-time gap is defined by the tool.
* **ATTRIBUTED UNREAD (paywalled; abstract + bibliographic records read):** Lu, Mahmassani &
  Zhou (2009), "Equivalent gap function-based reformulation and solution algorithm for the
  dynamic user equilibrium problem," *Transportation Research Part B* 43(3):345–364,
  doi:10.1016/j.trb.2008.07.005 (`lu2009equivalent`) — the gap-function DUE lineage D1 descends
  from; already the named canon successor in ADR-031.
* **HOUSE PRIMARY, read in full for ADR-031 (not re-read here):** Peeta & Mahmassani (1995),
  *Annals of Operations Research* 60:81–113, doi:10.1007/BF02031941 (`peeta1995system`) —
  experienced-time TD-UE conditions; equilibrium on instantaneous times "conceptually
  meaningless" — the conceptual basis of scoring on experienced costs.
* **ATTRIBUTED UNREAD (canon tool/method papers):** Horni, Nagel & Axhausen (2016) *The Multi-
  Agent Transport Simulation MATSim* (`horni2016multiagent`); Lopez et al. (2018) IEEE ITSC
  (`lopez2018microscopic`); Zhou & Taylor (2014) Cogent Engineering (`zhou2014dtalite`); Gawron
  (1998) IJMPC (`gawron1998iterative`, `duaIterate`'s default route-choice dynamics); Sbayti, Lu
  & Mahmassani (2007) TRR (`sbayti2007efficient`).
* **Repo precedents read in full (theory synthesis + this ADR author):** `docs/design/adr-020,
  adr-027, adr-029, adr-030, adr-031, adr-034`; `src/tabench/metrics/gaps.py`;
  `src/tabench/metrics/tdta_gaps.py`; the `dnl_gaps.py` C0–C8 gate headers; `docs/ARCHITECTURE.md`
  (the observational-track paragraph this certificate fills). The ADR author additionally read
  `docs/ROADMAP.md` (the deferred external-engine rows) and the pilot-C vendored `TAPLite.cpp`
  (the boost / entrance-admission / dead-code line numbers, re-checked this revision sprint).
* **Pilot numbers:** from Pilots A (SUMO `duaIterate`), B (MATSim), C (DTALite `simulation()`),
  whose artifact trees are on disk under `scratchpad/extdyn/{fable-duaiterate,pilotB,pilotC}`.
  **Re-verified on disk by the ADR author this sprint:** Pilot C's gap recompute (`analyze.py`
  re-run: **RG 0.393839, 706/1000 would-switch, 1000/1000 completed**) and determinism
  (`md5sum`: **655b1dde…** identical across `sim1`/`det1`/`det2`); Pilot A's step-19 gaps from
  `gap_runC1.json` (**rg_global 0.03887385796059968, rg_binned 0.04699429081763454, split
  344/156, 6/10 bins**). **In the review-revision sprint** the DTALite anchors were
  additionally re-measured **boost-clean** by re-invoking the pinned 0.8.1 engine on a fresh
  7→13 h horizon (a NEW scratch dir, never mutating the pilot tree; same `stdin=DEVNULL` /
  `OMP_NUM_THREADS=1` / timeout discipline): base `RG_D1` 0.393839 / TSTT 22654.000 / 706
  would-switch, one-step overshoot 33365.933 (`RG_D1` 0.512861), MSA final 0.039738 — all
  byte-identical (`trajectory.csv` md5 `655b1dde…`) to the pilot's 2 h run and provably
  boost-clean (every agent exits ≤ 8:47 h vs boost onset 11:48). Apart from that disclosed
  re-measurement, no engine binaries were re-invoked to author this document.
