# ADR-043 — `bo4mob-joint-estimation`: the joint (demand, supply) estimand is **under-identified** from counts+speeds — a measured deferral

**Status:** accepted (**measured deferral** — the scored joint row is **DEFERRED** on an executed
negative result; **ADR-ONLY**, no code, no estimator, no certifier, no registry/task change)
**Date:** 2026-07-17
**Deciders:** the `task-family-joint` sprint (S7) — the joint-calibration follow-up that
[ADR-041](adr-041-bo4mob-estimation.md) Decision 8 (*"A joint demand+supply certificate is a
separate row, not a rider here"*) and [ADR-028](adr-028-spsa-sumo.md) Decision 2 (*"an entire
task-family ADR of its own"*) each named.
**File:** `docs/design/adr-043-bo4mob-joint-estimation.md`

This is a **measured-deferral / negative-result ADR** in the
[ADR-030](adr-030-external-dta-simulators-deferred.md) house style: it defers the scored row on
an **executed record**, not on an unmeasured "too hard." This sprint set out to design **and
validate** a separated-evidence joint certificate; an adversarial document review found — and
this author **independently reproduced on the adversary harness** — that the design's central
**identification** claim is false. The honest, valuable contribution of this ADR is therefore
the **measured negative** (the joint estimand is under-identified from the available evidence)
plus a **named path forward**, exactly the ADR-030 contribution on the program's hardest design
problem. Two pilots and the confirmatory reverse-launder run executed on this box (2026-07-17)
against the pinned BO4Mob commit and `eclipse-sumo==1.27.1`; artifact trees under
`scratchpad/s7congested/` and `scratchpad/s7adversary/` (including the reproduction script
`verify_reverse.py`). Every load-bearing number below was executed; attributed-unread material is
marked as such.

---

## Context — two deferrals this would have composed, and the lineage it inherits

Balakrishna, Ben-Akiva & Koutsopoulos (2007), *Offline calibration of dynamic traffic
assignment: simultaneous demand-and-supply estimation* (TRR 2003:50–58, canon
`balakrishna2007offline`, tier 1, **already shipped** — no new bib/canon entry) is the paper the
T2 estimation track has cited **twice** without shipping its title contribution:

* **`spsa-sumo` ([ADR-028](adr-028-spsa-sumo.md))** implemented Balakrishna's black-box
  SPSA-over-a-simulator paradigm but scoped **demand-only** on the static `marouter`/`bfw` track
  (Decision 2). Its Decision 6 measured a load-bearing negative on the supply axis (unconstrained
  log-space supply search collapses into a saturated, zero-gradient corner) and named it *"a
  caveat of the deferred joint task family"* — this family.
* **`bo4mob-estimation` ([ADR-041](adr-041-bo4mob-estimation.md))** shipped the **D2**
  held-out-**count** certificate for BO4Mob's OD estimand on the pinned-engine observational
  track (no declared BPR, no true OD, equilibrium never claimed). Its Decision 8 scoped the joint
  estimand out as *"a separate row, not a rider here."*

BO4Mob is the natural home for the joint estimand (Balakrishna's own setting is a DTA simulator;
BO4Mob's mesoscopic SUMO run is a genuine DTA engine, unlike `marouter`'s static H=1 case). This
ADR is the **BO4Mob-side** attempt; the `marouter`-side joint extension stays out of scope.

**Honest sourcing (Balakrishna 2007 — inherited, not re-derived).** This ADR does **not**
re-derive the paper; it inherits ADR-028's already-adversarially-reviewed sourcing chain verbatim
(primary attributed/PDF-unread, established from Balakrishna's MIT PhD thesis, DSpace
1721.1/35120, cross-verified against Lu's open W-SPSA thesis ch. 3, DSpace 1721.1/88395).
Re-fetching either thesis would manufacture a second, potentially inconsistent attribution —
deliberately not done. What this ADR adds is the **new empirical grounding below**, executed
against the live BO4Mob artifacts.

---

## Executed grounding — the three measurements (all on `3junction`, the certificate's 18 GT sensors, seed 0)

**Scope honesty (read first).** All numbers below are on the single congested instance
`3junction`, scored on **its 18 real PeMS GT sensors** (the same set the shipped D2 certificate
scores — `bo4mob_nrmse(GT_CSV, ·)`, not the 128 raw sim edges), at **seed 0**, at the demand
perturbations named. The confirmatory second-instance and `od∈{0.8,1.5}` supply runs were
*planned* (`phase2_confirmatory.py`) but **not preserved**, so **nothing here is claimed to
generalize across instances or seeds** (this is itself a reason the row defers — the negative is
already decisive on one instance; a positive would have needed the breadth). The adversary's
"truth" is the honest-OD **simulated** field (a simulated-vs-simulated stress test isolating
*"can one estimand hide an error in the other?"*); the shipped certificate scores against **real
PeMS**, and the sim→real transfer is a **hypothesis, not a measured margin** (Decision 7).

### Measurement 1 — physical separation FAILS under congestion (`3junction`, 18 GT sensors)

The design sketch's separability rested on one pilot (`1ramp`, uncongested): a `--meso-tauff`
sweep left counts byte-identical while moving speeds, read as "supply moves speed, not counts."
That null **does not generalize.** On the congested `3junction`, **every** global meso supply
knob moves the sensor **counts** (sim-vs-honest-baseline, 18 GT sensors):

| supply knob (single, at bound) | count NRMSE vs baseline | speed NRMSE vs baseline | count rel-L2 ‖Δc‖/‖c‖ |
|---|---|---|---|
| `--meso-taufj 0.3` | 0.0247 | 0.0338 | 0.0199 |
| `--meso-taujj 0.4` | 0.0480 | 0.0833 | 0.0388 |
| `--meso-jam-threshold 0.8` | 0.1253 | 0.1463 | 0.1012 |
| `--meso-taufj 3.0` | 0.1631 | 0.1873 | 0.1317 |
| `--meso-taujf 0.5` | 0.1707 | 0.1012 | 0.1379 |
| `--meso-tauff 0.3` | 0.2916 | 0.3273 | 0.2356 |
| `--meso-taujj 4.0` | 0.3685 | 0.2644 | 0.2977 |
| `--meso-tauff 3.0` | 0.5855 | 0.1937 | 0.4730 |
| `--meso-taujf 5.0` | 0.7057 | 0.2415 | 0.5702 |
| `--meso-jam-threshold 0.05` | 0.7252 | 0.3371 | 0.5860 |

The sketch's proposed "safe" knob `tauff` is a **severe** count-launderer under congestion
(rel-L2 **0.47** at its high bound — not the single worst; `taujf` 0.57 and `jam-threshold` 0.59
move counts more). Symmetrically, a **demand** error alone moves **speeds** materially: at
default supply, `od_scale=0.8` moves speed a mean of **2.74 mph** (max 24.92); `od_scale=1.25`
a mean of **2.10 mph** (max 13.69) — on the same order as the supply knobs' own speed effects.
**Both observables are moved by both estimands.** (This finding is correct, and it is now the
*root* of the negative, not a footnote.)

### Measurement 2 — the single-supply-knob frontier is **one 1-D slice**, not the estimand's worst case

Holding a **+25 % wrong demand** (`od_scale=1.25`) fixed and letting an adversary tune **only
`tauff`** against the honest-OD truth (coarse grid + Nelder-Mead polish, 18 GT sensors):

| operating point on the OD=1.25 slice | `tauff` | count NRMSE | speed NRMSE | max(·,·) |
|---|---|---|---|---|
| default supply (`tauff=1.13`) | 1.13 | 0.2007 | 0.0963 | 0.2007 |
| adversary min-count | 1.457 | 0.1310 | 0.1598 | 0.1598 |
| adversary min-speed | 0.890 | 0.1721 | 0.0755 | 0.1721 |
| adversary knee (min-of-max) | 0.881 | 0.1548 | 0.0981 | 0.1548 |

Along **this one slice** — demand frozen at +25 %, supply free — the adversary cannot drive both
channels below `max ≈ 0.155`. The **error corrected below**: an earlier draft of this ADR read
that 0.155 as the estimand's *"irreducible laundering-resistance margin,"* a *"complete in-scope
worst case,"* a *"guarantee, not a sample,"* and called the single-knob estimand **"fully
validated."** **All of that is RETRACTED.** The estimator emits **`(OD, supply)`**, so the
adversary's move set is the **two-dimensional** `(OD, supply)` space; Measurement 2 searched only
**one axis** at **one** frozen wrong-demand level. The 0.155 is the residual along a single 1-D
slice, **not** the minimum over the estimator's move set. Measurement 3 finds a far lower point
on a *different* slice.

### Measurement 3 — the reverse/joint launder (the decisive negative — independently reproduced)

The symmetric attack: **fix a wrong supply** and **tune demand**. Fixing `tauff = 0.90`
(≈ 20 % below the 1.13 default) and sweeping the OD scale against the same honest-OD truth
(`verify_reverse.py`, 18 GT sensors, seed 0):

| wrong supply `tauff=0.90`, tuned demand | count NRMSE | speed NRMSE | max(·,·) |
|---|---|---|---|
| **apples-to-apples control** (`od=1.0`, **default** supply) | 0.0000 | 0.0000 | **0.0000** |
| `od_scale=0.90` | 0.0967 | 0.0508 | 0.0967 |
| `od_scale=0.93` | 0.0506 | 0.1017 | 0.1017 |
| **`od_scale=0.95`** | **0.0522** | **0.0492** | **0.0522** |
| `od_scale=0.97` | 0.0391 | 0.0659 | 0.0659 |
| `od_scale=1.00` | 0.0306 | 0.0754 | 0.0754 |

A **substantially wrong supply** (`tauff` ~20 % off) paired with a modestly wrong demand
(`od=0.95`) produces `max(count, speed) = 0.0522` — **roughly a third of** the 0.155 the earlier
draft advertised as the certificate's protective margin, and far inside any plausible
`{max < 0.155}` acceptance region. (The document review corroborated with additional operating
points `od∈{0.90, 0.70, 1.50}` in the same **0.0508–0.0986** band; the exact `0.0522` figure is
this author's independent reproduction.) The `0.0000` control confirms the harness is exact at
the honest `(true OD, true supply)` point, so the `0.0522` is a real low-max point at a **wrong**
`(demand, supply)`, not harness noise.

**The conclusion (Decision 2): the `{max < 0.155}` sub-level set is a wide two-dimensional
valley containing substantially-wrong-supply points.** The joint `(demand, supply)` estimand is
**under-identified** from the counts+speeds evidence: a wrong supply hides through demand
(Measurement 3), and a wrong demand partially hides through supply (Measurement 2). Both are 1-D
slices of the same 2-D loss surface, and the surface has a **low-max ridge** a joint estimator
can sit anywhere along.

### Measurement 4 — the two supporting numbers that survive (unchanged)

* **Multi-knob erodes even the slice.** An exploratory 2-knob `(tauff, jam-threshold)` search
  (85 evals = 64 grid + 21 Nelder-Mead, OD `1.25×`) reached `max = 0.14676` at
  `(tauff=1.466, jam-threshold=0.269)` — below the single-knob 0.1548 slice value, before the
  reverse launder is even considered. More supply freedom, more headroom.
* **Determinism holds.** Swept, non-default supply vectors are bit-deterministic across seeds at
  `OMP_NUM_THREADS=1`: `--meso-tauff 3.0` and `--meso-taujf 5.0` each produce **byte-identical
  parsed counts *and* speeds** across two independent tempdirs at seed 0. (The raw `edge_data.xml`
  differs only in SUMO's wall-clock `generated on` timestamp comment — the ADR-036 R10
  canonicalization surface; the **scored** arrays are identical.) ADR-034's determinism claim
  extends to swept meso queue-timing parameters — this survives and would carry to any future row.

---

## Decision 1 — physical channel separation is REFUTED (kept, now central)

Under congestion, no meso supply knob is a pure speed dial and no demand error is a pure count
error — both observables are moved by both estimands (Measurement 1). Any joint-certificate
design that leans on "supply→speed, demand→counts" physical orthogonality is therefore unsound
**on the congested instances that are the interesting ones**. The sketch's `1ramp` null (supply
moves speed, demand barely moves speed over a 10× range) was a special case of "nothing moves
because nothing is congested." This finding is correct and is the **premise** of Decision 2.

## Decision 2 — THE CORE NEGATIVE: the joint estimand is under-identified from counts+speeds

The estimator's emission is the pair `(OD vector, supply vector)`; the adversary's move set is
the full 2-D `(OD, supply)` space. Measurements 2 and 3 are two 1-D slices of that space's
`max(count_nrmse, speed_nrmse)` surface, and they disagree by 3× (0.155 vs 0.052) — because the
surface has a **wide low-max valley** through wrong-`(demand, supply)` points. A wrong supply of
~20 % and a wrong demand of ~5 % jointly fit both channels to ≈ 0.05. **There is no separated
evidence here that pins supply independent of demand.** The following are therefore **RETRACTED**
from any version of this design: *"irreducible laundering-resistance margin," "complete in-scope
worst case," "guarantee, not a sample," "the single-knob estimand is fully validated,"* and any
claim that separated evidence *defeats* supply-parameter laundering. What is true is narrower and
negative: separated scoring makes the trade **visible on two axes**, but the trade can be made
**cheap** because both axes move together.

## Decision 3 — "report both channels, never combine" is NECESSARY but NOT SUFFICIENT

Reporting the two NRMSE channels side by side, never collapsed into a joint scalar or gate,
remains the **right** reporting discipline (a combined gate would itself be a laundering surface
in reverse, and the ADR-036 R8 non-comparability posture still applies to any future joint
table). But it is **not sufficient** for identification: because **both** channels are moved by
**both** estimands (Measurement 1), a **low pair does not imply correct demand OR correct
supply** (Measurement 3 exhibits a low pair at a wrong pair). In particular:

* `heldout_speed_nrmse` is **NOT** "separated supply evidence." It is a **jointly-observed
  speed-fit column whose causal attribution to supply is not identified**: a good speed fit can
  come from a correct supply *or* from a compensating demand error, and the two are
  indistinguishable in the pair. Do not label it, or ship it, as a supply signal.
* **The attribution caveat is symmetric** (this corrects the earlier one-directional disclosure,
  which warned only that *"a worse speed fit is not unambiguous evidence supply is wrong"*): a
  **better** speed fit is **not** evidence that supply is **right**, and a low count fit is not
  evidence demand is right. Neither direction of either channel carries identified attribution to
  a single estimand.

## Decision 4 — VERDICT: DEFER the scored joint row (not an authorized follow-up)

The scored `bo4mob-joint` row is **DEFERRED**, not authorized. Shipping a leaderboard whose
`(heldout_nrmse, heldout_speed_nrmse)` pair can be driven low by a substantially-wrong supply
would publish a certificate that certifies the wrong thing — the exact failure a T2 certificate
exists to prevent. No amount of the design's own machinery (composition, two-channel scoring,
box constraints) closes the identification gap, because the gap is in the **evidence**, not the
code. The deferral is the correct decision on the measured record.

## Decision 5 — the named unblocker (concrete, and UNMEASURED)

The entanglement that breaks identifiability is **congestion-driven**: on uncongested `1ramp`,
supply moved speed while a 10× demand range barely moved speed (~0.5 %; the sketch's G5). This
points to a concrete, promising unblocker: **supply evidence taken where demand does not leak
into speed.** Candidates, in order of appeal:

1. **Off-peak / uncongested-window speeds.** A harmonic-mean-speed panel from an
   uncongested hour (or the uncongested sub-intervals of the within-day speed profile), where the
   speed field is dominated by the free-flow cost law and is (near-)insensitive to demand scale —
   so the speed channel identifies **supply** largely independent of demand.
2. **A free-flow-speed / cost-law measurement** that constrains the supply parameters directly
   (e.g. the queue-model headway law fit on low-occupancy intervals).
3. **A supply-known controlled sub-instance** (a scenario where the true supply parameters are
   declared, so the supply channel has a truth to be scored against, and only demand is
   estimated) — a construction the benchmark can build, unlike real PeMS.

**Each is UNMEASURED and requires its own pilot before any row ships** (the ADR-030
named-unblocker discipline): none of the three has been demonstrated to identify supply
independent of demand on a BO4Mob instance this sprint. This ADR names the path; it does not
claim to have walked it.

## Decision 6 — the design SUBSTRATE is preserved, but is insufficient ALONE

The structural design work is **not** wasted — it is what an unblocked row would build on, and it
is preserved as such (necessary, not sufficient):

* **Composition, not extension.** A `Bo4MobJointEstimationTask` that **wraps an unmodified**
  `Bo4MobEstimationTask` (folding `demand_task.content_hash()` verbatim), plus a new ABC and a
  fourth registry, keeps the load-bearing **type gate** (a demand-only estimator is structurally
  unable to receive a joint task) and keeps a demand-only estimator from ever reaching a
  `speed_dataset` (the P7 leak-by-convenience concern). This is the right shape for the family
  *if* the identification problem is solved by better evidence.
* **The two-channel certifier structure** (one meso run, two metric dicts, each a pure function
  of only its own column, never combined; `supply_feasible` mirroring `od_feasible` — shape /
  finite / in-box only, never fit quality; the ADR-036 R8 own-table non-comparability) remains
  the right **reporting** substrate (Decision 3), and is wall-neutral over the D2 certifier.
* **Mandatory `supply_bounds` + two-sided box projection from day one** remains mandatory (the
  ADR-028 Decision 6 corner-plateau finding is real and pre-paid here), and is orthogonal to the
  identification problem — a real requirement of any supply search regardless.
* **`heldout_speed_nrmse` as a faithful port of BO4Mob's dead upstream `eval_measure='speed'`
  path** (m/s→mph `*2.23694`, arithmetic mean over `interval_nVehContrib>0` sub-intervals, sensor
  window) remains an honest transform with honest provenance (real, upstream, never invoked in any
  shipped config; even upstream it fits *demand* to speed, never supply) — but per Decision 3 it
  ships, if ever, as a **jointly-observed speed-fit column**, never as separated supply evidence.

**None of this identifies supply.** The substrate is the scaffolding; the missing beam is the
evidence in Decision 5.

## Decision 7 — sim→real transfer is a HYPOTHESIS, not the shipped margin

Every number here is **simulated-vs-simulated** on `3junction`. The shipped certificate would
score against **real PeMS**, and ADR-034 already measured non-trivial **engine drift** (the meso
`edgeData` schema change between SUMO 1.12 and 1.27.1) plus real-sensor **noise**. Both plausibly
**shrink** any protective margin — i.e. laundering is likely **easier**, not harder, on real
data. The reverse-launder negative is therefore, if anything, **understated** by the sim
measurement; but no positive margin measured in sim may be presented as the real certificate's
property without its own real-data measurement.

## Decision 8 — measurement scope, stated honestly (no generalization)

The decisive measurements are: Measurement 2 at `od=1.25`, Measurement 3 at `tauff=0.90` /
`od∈[0.90,1.00]`, both on `3junction` at **seed 0**. The confirmatory `od∈{0.8,1.5}` supply runs
and any second-instance replication were planned (`phase2_confirmatory.py`) but **not
preserved**. The **negative** is decisive on one instance at one seed (an existence proof — a
single low-max wrong-`(demand,supply)` point refutes identification, and it is reproduced); a
**positive** identification claim would have required the breadth that was not run, which is a
further reason the row defers rather than ships. Nothing here is generalized across instances or
seeds.

## Consequences

* **This ADR is a documentation-only record** (like ADR-030): no code, no dependency, no CI job,
  no hash change. The golden Braess hash `cf00f411…` is untouched.
* **It records a measured negative on the program's hardest design problem** — the joint
  `(demand, supply)` estimand is under-identified from BO4Mob's counts+speeds — and defers the
  scored row on that record, naming a concrete unblocker (Decision 5). This is the ADR-030
  contribution: a hard, honest, measured deferral plus a path forward, not an unmeasured "too
  hard."
* **No new canon/bib entry** — `balakrishna2007offline` (ADR-028) and `ryu2025bo4mob` (ADR-034)
  cover this use; the Balakrishna attribution is **inherited, not re-derived**.
* **What survives for a future sprint** (the preserved substrate, Decision 6): the composition /
  type-gate task shape, the two-channel non-combining certifier structure, mandatory
  `supply_bounds` + two-sided projection, the dead-speed-path port (relabeled a jointly-observed
  speed-fit column), and the determinism result — all **conditional on** the Decision-5 evidence
  unblocker being measured first.
* **Named follow-up (its own pilot + adversarial-review sprint):** measure a Decision-5 unblocker
  — an uncongested/off-peak speed panel (or a free-flow-speed measurement, or a supply-known
  controlled sub-instance) — and demonstrate that it identifies supply independent of demand on a
  BO4Mob instance. Only if that succeeds does the scored joint row become designable.
* **Dual-benchmark honesty (ADR-034, carried):** BO4Mob is the lab's own benchmark; nothing here
  reproduces BO4Mob's numbers; any future joint table would carry the R8 non-comparability +
  forbidden-clause disclosure.

## Sourcing (honest ledger)

* **Read in full, this session (repo):** `adr-041`, `adr-028`, `adr-036`, `adr-030`;
  `estimation/bo4mob_base.py`, `metrics/estimation_bo4mob.py`, `data/bo4mob.py`.
* **Attributed unread, inherited verbatim from ADR-028 (NOT re-derived):** Balakrishna,
  Ben-Akiva & Koutsopoulos (2007), TRR 2003:50–58, DOI 10.3141/2003-07 (`balakrishna2007offline`)
  — via Balakrishna's MIT PhD thesis (DSpace 1721.1/35120) and Lu's W-SPSA thesis ch. 3 (DSpace
  1721.1/88395). **No second attribution manufactured here.**
* **Executed this session (not read about):** Measurement 1 (congested coupling, `3junction`, 18
  GT sensors), Measurement 2 (single-supply-knob frontier, `od=1.25`), Measurement 3 (the
  reverse/joint launder, `verify_reverse.py`, independently reproduced: control `max=0.0000`,
  reverse best `max=0.0522` at `tauff=0.90 / od=0.95`), Measurement 4 (2-knob min `0.14676`;
  determinism). All against the pinned BO4Mob commit and `eclipse-sumo==1.27.1`; artifacts under
  `scratchpad/s7congested/` and `scratchpad/s7adversary/`, re-verified from the raw eval logs.
* **Dual-benchmark honesty:** BO4Mob is the lab's own benchmark; the paper's numbers are never
  claimed reproduced; any future joint surface carries the R8 non-comparability + forbidden-clause
  disclosure.
