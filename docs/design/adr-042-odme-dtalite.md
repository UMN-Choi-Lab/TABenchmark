# ADR-042 — odme-dtalite: DTALite's static ODME as one more guarded T2 estimator row

**Status:** accepted (shipped in v0.2)
**File:** `docs/design/adr-042-odme-dtalite.md`

## Context — the T2 thesis, and a second engine-in-the-loop estimator

T2 (ADR-002) ships a demand-free estimation contract: anything that emits an OD
matrix is scored by the identical **pinned-bfw certifier** (`metrics/estimation.py`),
which recomputes count-fit and OD-fit from the emitted OD through *its own* reference
assignment (P1, ADR-002 Decision 2). `spsa-sumo` (ADR-028) was the first *guarded*
estimator riding a production engine, and — crucially — it did so with **zero task,
runner, or certifier changes** (the adr-028 ideal): the estimator emits, the unchanged
certifier scores.

DTALite 0.8.1 (Zhou & Taylor 2014, canon `zhou2014dtalite` — the tool paper whose title
names "fast model evaluation and *calibration*") hides an Origin-Destination Matrix
Estimation routine (`performODME`) **inside the same static `assignment()` entry** the
`dtalite-tap` T1 adapter (ADR-029) already wraps: it runs in `TAPLite.cpp`'s
`AssignmentAPI()` right after the Frank-Wolfe loop when `settings.csv` carries
`odme_mode=1`. There is NO separate ODME API. It is a gradient-descent OD calibrator —
given a seed OD, a target OD, and sensor link counts (`obs_volume`), it re-weights the
route-flows the base assignment discovered so the modeled link volumes approach the
counts. This ADR ships it as `odme-dtalite`, the second engine-in-the-loop T2 estimator,
under the same adr-028 zero-certifier-change ideal.

## Decision 1 — a GUARDED STATIC estimator on the UNCHANGED certifier

`odme-dtalite` is a `DtaliteODMEEstimator(ODEstimator)` registered in
`ESTIMATOR_REGISTRY` behind a guarded import (Decision 6). It emits an OD matrix
through `ODResultBundle.final.od_matrix` — the SAME channel `spsa`/`gls`/`spsa-sumo`
use — and the EXISTING pinned-bfw `ODCertifier` scores it unchanged. **No
`estimation/base.py`, `metrics/estimation.py` (`ODCertifier`), `experiments/runner.py`,
or `EstimationTask` code is touched.** The scope-probe confirmed this is possible:
ODME lives on the *static* TAPLite path (never `SimulationAPI`/`trajectory.csv`), so the
OD it emits is certifiable through the exact same declared-BPR pinned-bfw pin every
static T2 row uses — a **guarded static estimator**, not a D2 observational certificate
(which would only have been forced if ODME needed a re-simulated held-out count).

A **DTALite-pinned certifier was considered and REJECTED**, for the same reason
adr-028 rejected a sumo-pinned one: scoring estimators against the very engine one of
them runs in the loop is a conflict of interest, and the whole T2 point is that the
certifier is model-blind. Keeping the pinned-bfw certifier makes the standard
self-vs-certified honesty diff MEASURE the engine-in-the-loop bias (Decision 5).

## Decision 2 — `route_output=1` is REQUIRED (the non-obvious, measured pin)

This is the load-bearing finding of the sprint, and it is **not** in the design dossier
— it was caught by the mandatory measure-first pilot (below). ODME does not read the
Frank-Wolfe `MainVolume` when it computes each observed link's *modeled* volume; it
reconstructs that volume from the stored **route/path history**. With DTALite's lean
`route_output=0` (the `dtalite-tap` T1 default), that history is never populated, so the
reconstruction collapses to ~0 on every sensor. ODME then sees a fake, enormous deficit
on every observed link and inflates every OD cell to its box ceiling.

Measured on the Sioux Falls marquee anchor (below), with `route_output=0`:
the emitted OD total demand is **+40 %** over truth, `od_rmse` ~2× WORSE than the prior,
and the certified `obs_count_rmse` ~7× WORSE — a *degenerate* estimator that would have
silently shipped had the pilot not measured the certified before/after. Flipping to
`route_output=1` restores a faithful reconstruction (the engine's internal predicted
volume matches the observed count to ~1.0 ratio) and turns the row non-degenerate
(Decision 3). `settings.csv` therefore pins `route_output=1` AND `odme_mode=1`; a
structural test fails if either reverts. `route_output=1` is part of the pinned
estimator identity.

## Decision 3 — the marquee anchor is Sioux Falls scale (the tol=1 floor), measured

The pilot (ruling: *pilot-and-decide the anchor BEFORE the full build*) proved a
non-degenerate recovery exists. **Marquee anchor:** `siouxfalls` UE (24 zones / 76 links
/ 528 OD pairs, BPR power-4), clean counts, sensors random coverage 0.5 (38 links),
held-out coverage 0.2, stale prior `cv=0.3`, seed 7. Through the UNCHANGED pinned-bfw
certifier:

| metric | prior baseline | odme-dtalite | ratio |
|---|---|---|---|
| `obs_count_rmse` | 994.8 | 365.3 | **0.37×** |
| `heldout_count_rmse` (ranking) | 816.5 | 556.6 | **0.68×** |
| `od_rmse` | 267.4 | 264.5 | 0.99× |
| `total_demand_error` | 0.014 | −0.002 | — |

`od_feasible = 1`, `certificate_converged = 1`, `certificate_gap < 1e-6`, and the ODME
gradient descent ran **69 iterations** (never a "Convergence reached after 0 iterations"
no-op). This is a real "estimator beats prior" row: ODME measurably improves BOTH the
observed count fit it calibrates AND the ranking held-out count fit. `od_rmse` barely
moves — ODME is a count-matcher and Sioux Falls OD is not linearly identifiable from
these sensors (`od_identifiable = 0`, reported, honest).

**Why Sioux Falls, not Braess.** DTALite's ODME convergence tolerance is a HARDCODED
*absolute* gradient-norm floor `tol=1` (Decision 4). On Braess-scale demand (~6), a
realistic count mismatch never clears the floor and ODME no-ops with "0 iterations"
(measured). The marquee anchor MUST be at a demand magnitude where realistic deviations
clear the fixed floor — Sioux Falls (OD mean ~683, min 100) does; Braess does not. This
is the same A4 scale choice `dtalite-tap` made.

## Decision 4 — the disclosed engine envelope (part of the pinned 0.8.1 identity)

Three hardcoded, non-configurable engine behaviors bound what this row can do. They are
disclosed as HONEST scope boundaries — reproducibility-good, like marouter's hardcoded
linear vdf (ADR-027) — not defects:

1. **A hardcoded `[0.5·min(seed,target), 1.5·max(seed,target)]` box** baked into the
   line search, no settings-file exposure. With `demand_target = prior`, the estimator
   can only recover a truth cell within ~2× of the prior. Measured: with a large count
   mismatch the emitted cell lands at EXACTLY `1.5×` and sits there for the rest of the
   descent (a hard wall, not a soft penalty). `demand_target_frac` (default 1.0) is a
   documented **one-sided** dial: `demand_target.csv = prior · frac`, so `>1` raises the
   upper bound, `<1` lowers the lower bound (never both), at the cost of biasing the
   (also hardcoded) OD-regularization pull toward the inflated target. It is a hashed
   estimator-identity factor.
2. **Hardcoded penalty weights** (`w_link=0.1`, `w_od=0.01`, `w_vmt=1e-6`) and the
   ABSOLUTE `tol=1` convergence floor (Decision 3). Only `odme_mode`/`odme_vmt` are
   settings-configurable; the rest are C++ literals, along with the fixed 400-iteration
   descent. `budget.iterations` therefore maps to the base FW `number_of_iterations`
   only — the ODME descent itself is not budget-controllable, which is disclosed, not
   hidden.
3. **A route-history fidelity caveat.** ODME re-weights routes the prior BFW/AON
   assignment already loaded; it cannot invent flow on an unloaded route. `route_output=1`
   (Decision 2) is what makes the captured routes' reconstruction faithful; a genuinely
   unloaded observed link stays a blind spot.

**Scope:** single-mode "auto" demand-only, matching `dtalite-tap`'s sprint-1 scope.
Native multiclass ODME (`mode_type.csv`) is a named deferred follow-up. The scenario
envelope IS the T1 adapter's — refusals (nonzero fixed cost / toll, sub-0.1 capacity
clamp, the SUE-family task fields) are delegated to
`DTALiteTapModel._refuse_unrepresentable` verbatim (the spsa-sumo precedent). Note
power≠1 is representable here (DTALite's VDF is the repo BPR exactly), unlike the
marouter-backed `spsa-sumo` which refused power-4.

## Decision 5 — `link_performance.csv` is corrupt under ODME; source the OD from `od_performance.csv` ONLY

Measured: under `odme_mode=1` a lossy post-ODME "final synchronization" reconstructs the
link volumes from the route-history representation rather than the Frank-Wolfe
`MainVolume`, so they no longer conserve OD demand. The illustrative signature — a link
reporting `volume=0` while its own `travel_time` implies a nonzero flow, and total
reconstructed link flow not matching total OD demand even at 0 gradient steps — was
measured on a **partially-loaded** two-route toy net (the design-probe `designB`
instance), and reproduces generally under the reconstruction-starved `route_output=0`
regime. It does NOT reproduce uniformly on the SHIPPED `route_output=1` + fully-loaded
marquee (on Sioux Falls 0/76 links show the `volume=0`/`travel_time>0` inversion — every
link is loaded). The
substantive invariant is config-independent and is what the row relies on:
`link_performance.csv` is **structurally never read** (no `open()` touches it; the emitted
OD is byte-identical after deleting the file, pinned by a test), so whatever corruption
it carries is unreachable by the certifier regardless of scenario or settings. The OD
estimate is therefore sourced **exclusively** from `od_performance.csv`'s `volume` column
(`= MDODflow`), which is unaffected. And the `dtalite-tap`
`odme_mode=0` read-back gates (the A2 cost-match / per-origin mass-gate / echo-check) are
NEVER reused on an ODME run — they are validated only for `odme_mode=0`. The read-back
transports `dtalite-tap`'s "every repo link matched exactly once" discipline to the
OD-pair domain: every prior-support pair present (completeness), a pair's route rows
consistent, no off-support/phantom pair, every volume finite and non-negative — else
`RuntimeError` (never trust `returncode` alone; the engine exits 0 on garbage).

## Decision 6 — the anti-laundering property, and the honesty diff as a measured bias

The **anti-laundering property is sacred and holds by construction.**
`ODCertifier.certify()` (read in full) accepts only a raw `(Z,Z)` numpy OD and re-runs
its OWN pinned bfw solve — there is no DTALite import, no engine branch anywhere in that
file. The ONLY channel from `odme-dtalite` to the harness is
`ODResultBundle.final.od_matrix`. So even though DTALite's own `link_performance.csv` is
corrupt (Decision 5), that corruption is *structurally unreachable* by the certifier. A
buggy or adversarial DTALite run can at worst emit a *bad* OD that certifies honestly as
a no-improvement row — it CANNOT forge a good certificate.

This is pinned from both sides by tests (Decision 7). Measured on the marquee: DTALite's
OWN self-reported count fit (from its stalled-assignment, box-clamped predicted volumes)
is `obs_count_rmse ≈ 8`, while the pinned-bfw certifier scores the emitted OD at
`≈ 365`. The certifier ignored the rosy self-report entirely. That large self-vs-certified
diff MEASURES the engine-in-the-loop bias (the `spsa-sumo` reframing: DTALite descends
its own stalled law under a hardcoded box; the certificate re-assigns under the declared
BPR), reported as provenance, never a bound or a score.

## Decision 7 — the FIRST guarded ODME estimator (guard shape, subprocess discipline)

Guarded byte-parallel to the `spsa-sumo` block (and the T1 model guard): a module-top
`find_spec("DTALite")` probe raises `ModuleNotFoundError(name="DTALite")`, swallowed by
exact name in `estimation/__init__.py` — the exact CAPITAL `DTALite` name (the wheel's
module name, ADR-029), never imported in-host (the wheel prints a banner and ctypes-loads
an OpenMP engine on import). Subprocess discipline is the adr-029 contract verbatim:
subprocess-only (`[sys.executable, "-c", "import DTALite; DTALite.assignment()"]`),
`stdin=DEVNULL`, `OMP_NUM_THREADS=1` (a correctness pin — the assignment/ODME path was
measured race-free, bit-identical across 21/21 runs at OMP∈{1,4}, but the pin is kept as
defense-in-depth), tempdir-per-run with `finally` cleanup, one wall deadline threaded
across write → subprocess → parse, `returncode` never trusted (success = read-back), and
an engine crash/timeout RAISES `RuntimeError` (never a false estimate). An `sp_calls`-only
budget is refused up front (the engine hides its Dijkstra count — the adr-027/028/029
pattern).

## Determinism / wall math (measured on this box, DTALite 0.8.1 wheel)

The assignment+ODME path is byte-deterministic under `OMP_NUM_THREADS=1`: the marquee's
`od_performance.csv` is md5-identical across 3 reruns, so the estimator is
`deterministic=True`, `seedable=False` (no engine seed to pin — the RngBundle root seed
lands in the bundle as provenance). The ODME subprocess wall at the marquee scale is
~0.4–2.0 s (base `number_of_iterations` 20→400); the row rides the EXISTING `dtalite` CI
job (same 0.8.1 pin), adding `tests/test_odme_dtalite.py` to that job's explicit file
list and `odme-dtalite` to its notebook `-k` filter. No new CI job.

## Honest sourcing

`zhou2014dtalite` anchors the DTALite **software lineage** (tool-paper discipline, the
lopez2018/adr-029 precedent): its title names calibration, of which ODME is an instance,
so no new bib entry is needed. The row validates the ADAPTER + engine fidelity, never the
paper's numerics. The engine version + the ODME settings (`odme_mode=1`, `route_output=1`,
the hardcoded weights/box/tol it implies) are part of the estimator's pinned identity.
This is the same wheel and `.so` (md5 `e179ed66…`, S4-verified) as `dtalite-tap` and
`dtalite-simulation` — three rows, three engine entries, one pinned wheel.

## Consequences

* One more T2 leaderboard row, on the SAME pinned-bfw certifier and the SAME separate-no
  table as every static estimator (directly comparable — unlike the EDOC observational
  rows). Additive; the golden Braess content hash `cf00f411…` is byte-identical.
* A new, disclosed pattern: a guarded estimator whose engine has a hardcoded
  representability box + convergence floor, honestly bounded rather than tuned. The
  marquee is chosen to clear the floor; a small-demand anchor no-ops (disclosed).
* The measure-first pilot earned its keep: the `route_output=0` degeneration would have
  shipped a silently-broken always-inflating estimator. `route_output=1` is now pinned
  and tested.

## Adversarial review — measured hazards, all disclosed, none blocking

* **`route_output=0` degeneration (Decision 2)** — the sprint's headline finding, caught
  by the pilot, pinned by a structural settings test.
* **link_performance corruption (Decision 5)** — never read; a test deletes it post-run
  and re-parses `od_performance.csv` to byte-identical OD.
* **the `[0.5,1.5]×` box + `tol=1` floor (Decisions 3,4)** — disclosed envelope; the
  marquee clears the floor with 69 iterations, and `demand_target_frac` is a documented
  one-sided dial.
* **anti-laundering (Decision 6)** — pinned from both sides: the certifier ignores
  DTALite's rosy self-report (measured 8 vs certified 365), and a poisoned OD certifies
  feasible-but-worse.
