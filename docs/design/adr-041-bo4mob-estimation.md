# ADR-041 — `bo4mob-estimation`: the BO4Mob held-out-count OD-estimation family (stage 2), a D2 observational T2 certificate

**Status:** accepted (shipped)
**Date:** 2026-07-17
**Deciders:** the `bo4mob_estimation` stage-2 follow-up named in [ADR-034](adr-034-bo4mob-scenarios.md)
**File:** `docs/design/adr-041-bo4mob-estimation.md`

## Context — the stage-2 follow-up ADR-034 named

[ADR-034](adr-034-bo4mob-scenarios.md) shipped the BO4Mob San Jose freeway
instances as **scenarios/data only** (stage 1): a P9 commit-pinned fetcher for the
four small instances, the pandas-free count/NRMSE transforms, and a guarded smoke
test that runs od2trips + mesoscopic SUMO end-to-end on `1ramp`. It explicitly
deferred "the `bo4mob_estimation` T2 family with a pinned-engine held-out-date
observational certificate" to a **named stage-2 follow-up**. This ADR ships it.

BO4Mob (Ryu, Kwon, Choi, Deshwal, Kang & Osorio 2025, arXiv:2510.18824; canon
`ryu2025bo4mob`, already shipped with ADR-034 — **no new canon entry**) has **no
ground-truth OD**: truth is the real Caltrans PeMS panel (14 dates × 3 hour
windows). A T2 estimator emits a continuous OD vector over a fixed set of active
`(fromTaz, toTaz)` pairs; the score is the NRMSE of mesoscopic-SUMO link counts vs
the real sensors. This is an **observational** certificate (a **D2** in the
ADR-036 vocabulary): the harness re-runs the pinned engine on the emitted OD and
compares to real data. **Equilibrium is never claimed** — there is no true OD, no
declared BPR network, and no `bfw` pin.

## Decision 1 — a NEW T2 sibling family, not a guarded estimator on the existing machinery

ADR-028's `spsa-sumo` added an **estimator** to the EXISTING static-T2 machinery
with zero task/runner/certifier changes. That ideal does **not** transfer here.
Three concrete blockers, each confirmed by reading the live code:

1. `EstimationTask.network: Network` (`estimation/base.py:60`) requires a declared
   BPR graph; a BO4Mob instance is a mesoscopic SUMO net with **no** BPR.
2. `ODCertifier` (`metrics/estimation.py:108-115`) hard-raises unless
   `certificate['assignment']=='bfw'` and reads `scenario.demand.matrix` as truth
   — BO4Mob has neither a `bfw`-certifiable network nor a true OD.
3. `run_estimation_experiment` needs a full `Scenario` (`runner.py`), and ADR-034's
   own shipped test `test_bo4mob_keys_are_not_load_scenario_scenarios` pins that a
   bo4mob key can **never** resolve to a `Scenario`.

So, exactly as ADR-023 did for the within-day dynamic track, the family gets its
own task type, ABC, and registry (`estimation/bo4mob_base.py`:
`Bo4MobEstimationTask` frozen dataclass, `Bo4MobODEstimator` ABC,
`BO4MOB_ESTIMATOR_REGISTRY`, `register_bo4mob_estimator`, `Bo4MobPriorBaseline`)
and its own certifier (`metrics/estimation_bo4mob.py`: `Bo4MobODCertifier`). This
is the **third** T2 sibling (static, dynamic, BO4Mob) — the honest type gate that
keeps the CLI from running a static/dynamic estimator on a BO4Mob task or vice
versa (ADR-002 Decision 1 rationale, ADR-023).

**What reuses verbatim** (the genuine minimal surface, honoring ADR-028's spirit):
`ODTrace`/`ODState`/`ODResultBundle` (shape-agnostic containers already
precedented for a non-`(Z,Z)` estimand by `DynamicEstimationTask`);
`fill_single_od`/`fix_routes_single`/`local_edgedata_additional`/`edgedata_counts`/
`bo4mob_nrmse` (`data/bo4mob.py`, plus a new `fill_od_from_vector` /
`bo4mob_pairs` / `bo4mob_prior_vector`); `assert_engine_pin` (`edoc/replay.py:89`,
the tiny G0 helper); and the ADR-027/029 subprocess discipline verbatim.

**The estimator/certifier guard asymmetry.** `spsa-sumo` guards its ESTIMATOR
import (its inner oracle is sumo). Here the do-nothing prior baseline imports **no**
sumo and registers **unconditionally**; the sumo dependency is the CERTIFIER's
(every `certify` runs od2trips+meso), so the "install `tabench[sumo]`" error lives
at the runner/CLI boundary, checked at run time — not at decorator-time
registration.

## Decision 2 — NO EDOC-1 reuse: a category error boundary (ADR-036 R11 honored at the right layer)

EDOC-1's D1 substrate (ADR-036/037) validates a model's OWN per-agent self-report
(the `EdocScenario` per-agent trip table + `EdocEvaluator`/G1 replay-fidelity
bijection). BO4Mob emits **only an aggregate OD vector**; od2trips (`--spread.uniform`)
and the fixed `routes_single.csv` generate all routing/departure detail
deterministically, so there is **no per-agent self-report to validate and no G1
replay-fidelity gate**. ADR-036 R11's "shared substrate" is honored at the
**subprocess-discipline + engine-pin** layer ONLY (`assert_engine_pin` + the
adr-027/029 wrapper). Importing `EdocScenario`/`EdocEvaluator`/`RG_D1` here would
be a **structural category error**; this ADR records the boundary so a later sprint
does not over-apply R11 to the scored object.

## Decision 3 — the held-out split: one fixed OD, same-hour different-DATE dates (framing b)

The estimator emits **one** fixed OD vector from the TRAIN anchor; the certifier
re-simulates it **once** and scores the resulting counts against a pinned HELD-OUT
panel of same-hour-window, different-DATE real counts. The ranking column is
`heldout_nrmse` — the **MEAN** of BO4Mob's per-date count NRMSE (deliberately not
`heldout_count_rmse`, to avoid implying comparability to the static T2 raw-RMSE
scale). Per-date OD emission (framing a) is **rejected**: it has a free-riding hole
(copy the nearest train date's OD) the harness cannot structurally forbid; framing
b closes it by construction and is O(1) engine wall regardless of how many held-out
dates are scored.

**The hour window is held FIXED** at the anchor hour `06-07`. PeMS counts vary
enormously across the day (the anchor `06-07` totals ~2121 vs a `17-18` window
~7700), so a single fixed OD only represents one hour's demand; held-out probes
same-hour, different-DATE generalisation only.

**The concrete split.** TRAIN = the anchor date `221008` at `06-07` (kept in TRAIN
for stage-1 continuity — the obs/in-sample fit). HELD-OUT = the 13 consecutive
dates `221009`–`221021` at `06-07` (`BO4MOB_HELDOUT_DATES`). Both dials are public
and hashed via `heldout_digest`. The 13-date panel is pinned for **all four** small
instances (52 new commit-pinned SHA-256 + byte-size entries, `BO4MOB_HELDOUT`),
fetched on demand by `fetch_bo4mob_heldout` on a SEPARATE registry never in the
CI-prefetched `REGISTRY`. `5fullRegion` stays HPC-only (no panel, refuses to fetch).

**Pilot (executed on this box, eclipse-sumo 1.27.1).** The prior-baseline OD's
cross-DATE NRMSE stability on `1ramp` over the 13 held-out `06-07` dates:

| statistic | value |
|---|---|
| in-sample obs NRMSE (anchor 221008) | 2.432471221214843 |
| held-out per-date NRMSE, min | 0.563911 |
| held-out per-date NRMSE, max | 3.297711 |
| held-out per-date NRMSE, mean (= `heldout_nrmse`) | 1.697988 |
| held-out per-date NRMSE, stdev | 0.787832 |

The wide single-day spread is precisely the argument for **aggregating over all 13
dates**: the mean-over-13 is stable by construction, materially below the anchor
(1.698 ≠ 2.432), non-vacuous, and improvable. The prior OD `[2092, 609, 386]`
**over-scales** — its simulated sensor counts `[2092, 2701, 2478]` sit far above the
held-out ground-truth mean `[613, 1193, 1146]` — so the held-out mean is lowered by
**reducing** demand toward the held-out sensor counts, not raising it (a measured
scaling sweep: prior×0.25 → 0.411, ×0.5 → 0.454, ×1.0 → 1.698, ×1.5 → 3.040, ×2.0 →
4.388; the empirical minimum is near prior×0.25 ≈ 0.41, the fit-train optimum ≈ 0.303,
and the oracle floor ≈ 0.267). Pinning fewer dates would let one idiosyncratic day
dominate. Wall-neutral: framing b runs meso once regardless of date count.

## Decision 4 — held-out leakage is structural (P7), not conventional

Only the sha256 `heldout_digest` may appear in `Bo4MobEstimationTask`. The held-out
CSV bytes and held-out date/hour identifiers are constructed ONLY inside the
certifier's closure (`Bo4MobODCertifier.heldout_sensors`) and are never reachable
from the task. `test_task_does_not_leak_heldout_dates_or_counts` walks the task's
fields/repr/dataset and asserts none of `BO4MOB_HELDOUT_DATES` and no held-out count
value is present (the TRAIN anchor `221008` IS allowed — it is in TRAIN). The
harness manifest (written after scoring) DOES record the full held-out design for
provenance — that is not the leakage surface; the task is.

**Complete panel pin (F3).** `heldout_digest` folds BOTH the held-out (date, hour)
DESIGN and the sanctioned DATA identity — the commit-pinned `(sha256, size)` of each
held-out CSV from `BO4MOB_HELDOUT[instance_key]` — so the task `content_hash`
**completely pins the scored held-out panel** (design + the sanctioned data
checksums), not merely the dates. It hashes the sanctioned checksums, NEVER the raw
held-out bytes, so P7 is preserved and the digest is now instance-specific. The
recomputed 1ramp values: `heldout_digest =
b8cc933a50bbde923a3232d7df1c155e7bcfc122ba1f705665adddbdc3470547`, task
`content_hash = a0b7f1872abe2eeabd6c473b621b659fdc8f0872194501b02753e7b50020eb04`
(the row is unpublished, so this hash move is permitted; no golden bo4mob task hash
exists, and the golden Braess hash is untouched).

## Decision 5 — the OD-window fill is a load-bearing anti-laundering control

`fill_od_from_vector` MUST inherit the exact `od_end_time` interval-rewrite
(ADR-034 Decision 3). The one real laundering vector, found and fixed once in
stage 1, is leaving the template interval `end=3600` on `1ramp` (`od_end_time=3300`):
that releases ~5% of demand past the OD window and mis-scores the NRMSE. The vector
fill is byte-identical to `fill_single_od` on the prior vector, so the prior
baseline's certified obs NRMSE reproduces the stage-1 faithful pipeline **exactly**;
a duplicated regression pins the pair live:

| od_end_time | 1ramp prior obs NRMSE |
|---|---|
| 3300 (correct, shipped) | 2.432471221214843 |
| 3600 (the pre-fix demand-leak bug) | 2.3147038842862218 |

Both are bit-stable (pilot-reproduced).

## Decision 6 — engine pin, and crash-vs-censor

The exact `eclipse-sumo` version (`1.27.1`, a CONSTANT — `BO4MOB_ENGINE_VERSION`)
sits in the task `content_hash` and is checked at `certify()` via `assert_engine_pin`
(reused verbatim), which **RAISES** `ValueError` on a mismatch — never silently
scoring under a drifted engine. An engine crash / timeout / read-back failure in the
certifier's OWN od2trips+meso pipeline is **infrastructure** → `RuntimeError` that
PROPAGATES, never `od_feasible=0` (ADR-036 R6, carried here). `od_feasible=0` is
reserved for a well-formed OD that fails the certificate's own validity gates
(shape / finite / non-negativity). A **zero OD is not censored** — a legitimate,
terrible estimate that short-circuits the engine (adr-027 zero-demand fast path)
and certifies with catastrophic-but-finite NRMSE. `rc` is never trusted: success is
DEFINED by the read-back of the produced artifact. All of these are pinned by
**executed** tests (an injected fake runner for the engine-free control flow; a
poisoned net.xml and an impossible deadline for the live crash), not by analogy.

## Decision 7 — identifiability is provenance-only (no rank test applies)

Hazelton's linear-identifiability rank test needs a declared assignment/proportion
matrix that BO4Mob lacks. The task/manifest carry a **light provenance-only**
diagnostic (`n_active_pairs`, `n_train_sensors`, `sensor_pair_coverage`) and a
`rank_test_applicable=False` flag — never a gate. Overbuilding a rank test here
would be dishonest about what BO4Mob supports.

## Decision 8 — demand-only scope

Demand-only, exactly as ADR-028 Decision 2: `DEFAULT_VEHTYPE` (IDM, `speedDev=0`)
is preserved attribute-exact by `local_edgedata_additional`; **no supply dial is
exposed to the estimator**. A joint demand+supply certificate is a separate row, not
a rider here.

## Wall budget + CI test scope (measured on this box)

Certify re-runs od2trips+meso **once** (held-out dates are cheap numpy comparisons
against the same simulated output). Measured per-instance certify wall:

| instance | certify wall | CI scope |
|---|---|---|
| 1ramp | 0.45 s | full estimator + certify (liveness/correctness anchor) |
| 2corridor | 9.4 s | single-certify liveness |
| 3junction | 13.6 s | single-certify liveness |
| 4smallRegion | ~129 s (meso alone ~126 s) | REGISTERED, opt-in behind `TABENCH_RUN_SLOW_BO4MOB` |

CI (the existing `sumo` job) runs 1ramp + 2corridor + 3junction (~24 s of meso,
inside the 2-4 min sumo-job budget) via `tests/test_bo4mob_estimation.py`, plus the
notebook (1ramp only, ~0.4 s). `4smallRegion` is registered but gated so CI never
pays its ~129 s. The engine-free half of `test_bo4mob_estimation.py` runs on the
sumo-free matrix legs.

## Dual-benchmark honesty (ADR-034, carried to every stage-2 surface)

BO4Mob is the lab's OWN benchmark; every new surface (`Bo4MobEstimationTask`,
`Bo4MobODCertifier`, the runner manifest `notes`, the CLI caption, the notebook)
declares the affiliation openly, hosts the instances as **scenarios/tasks/
certificates only — never validation of TABench methods**, and does **not**
reproduce BO4Mob's published numbers. A T2 estimator ranking table is exactly the
artifact that could accidentally read as "TABench methods beat BO4Mob's own
strategies," so the extended **forbidden clause 3** is pinned on every surface:
this D2 held-out NRMSE is NOT comparable to the static/dynamic T2
`heldout_count_rmse` scale, and does NOT reproduce BO4Mob's own SPSA/BO leaderboard
rankings. A grep tripwire (`test_no_forbidden_comparability_claims_in_new_modules`)
fails on any affirmative "reproduces/validated on/beats BO4Mob" phrase.

## Consequences

* **New:** `estimation/bo4mob_base.py`, `metrics/estimation_bo4mob.py`,
  `tests/test_bo4mob_estimation.py`, `tutorials/03-estimation/08-bo4mob-estimation.ipynb`.
* **Extended:** `data/bo4mob.py` (`fill_od_from_vector`, `bo4mob_pairs`,
  `bo4mob_prior_vector`, `BO4MOB_HELDOUT`, `fetch_bo4mob_heldout`, the engine-pin
  constants, a shared `_fetch_checked`); `experiments/runner.py`
  (`run_bo4mob_estimation_experiment`); `cli.py` (`t2_bo4mob_estimation` card,
  dispatched before `load_scenario`); the `sumo` CI job (test file + the notebook
  `-k`); `tests/test_tutorials.py` (a sumo-guarded manifest entry; `_ALLOWLIST`
  stays empty); ROADMAP/VALIDATION/ARCHITECTURE.
* **Unchanged:** the golden Braess hash `cf00f411…` (re-asserted byte-identical);
  every existing scenario/EDOC/estimation content hash; `docs/model-specs.json`
  (an estimator is not a model); no new pip dependency; no new CI job.
* **No new canon entry** — `ryu2025bo4mob` (shipped with ADR-034) covers the
  estimation use.
* **Open items for a later sprint:** a black-box `spsa-bo4mob` estimator (its
  estimator import IS sumo-guarded, unlike the prior baseline) and, with it, the
  fairness-gate lineage check keyed off the instance key (deferred here — the prior
  baseline has empty `trained_on`, and `run_bo4mob_estimation_experiment` raises
  `NotImplementedError` for any estimator that declares lineage); a joint
  demand+supply certificate; per-hour-window config variation if other hour windows
  ever ship.
