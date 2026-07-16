# ADR-034: bo4mob-scenarios — the BO4Mob San Jose freeway OD-estimation instances (stage 1)

**Status:** accepted (implemented)
**Date:** 2026-07-16
**Deciders:** non-solver roadmap — the `bo4mob-scenario` queue item (ROADMAP.md:110)
**File:** `docs/design/adr-034-bo4mob-scenarios.md`

## Context

`BO4Mob` (Ryu, Kwon, Choi, Deshwal, Kang & Osorio, 2025, arXiv:2510.18824,
**NeurIPS 2025 Datasets & Benchmarks**; canon `ryu2025bo4mob`, tier 1) poses five
San Jose freeway networks as high-dimensional black-box **OD-estimation** problems:
minimise the NRMSE between mesoscopic-SUMO link counts and **real Caltrans PeMS**
sensor data over a continuous OD vector (3 → 10,100 OD pairs). It is an *inverse*
problem with a stochastic, non-differentiable, expensive objective and — the fact
that shapes everything below — **no ground-truth OD**: truth is the real sensor
panel (14 dates × 3 hour windows = 210 CSVs in-repo).

**The dual-benchmark sensitivity.** BO4Mob is **the lab's own benchmark**
(`github.com/UMN-Choi-Lab/BO4Mob`, MIT). Integrating a benchmark the lab authored
into a benchmark the lab authors is a standing honesty hazard, and the contract
that governs it (below) is the central content of this ADR, not a footnote. The
canon already carries the cross-benchmark pointer (`ryu2025bo4mob`, MODELS.md
roadmap card); this ADR adds the **instances**.

**This is stage 1 only: data availability + pipeline liveness.** It ships the P9
data/scenario family (the four small instances, checksummed download-on-demand)
and a guarded smoke test proving the mesoscopic-SUMO evaluation pipeline runs
end-to-end under the shipped `eclipse-sumo` wheel. It ships **no task family, no
certificate, no estimator** — those are a **named stage-2 follow-up** with its own
future ADR: a `bo4mob_estimation` T2 task with a pinned-engine held-out-date
**observational** certificate (the honest certifier shape given there is no true OD
and no declared BPR network — see "Stage 2" below).

## The dual-benchmark honesty contract (the load-bearing section)

The lab owns both benchmarks. The only honest claim shape:

- **ALLOWED:** "TABenchmark hosts BO4Mob's instances as scenarios/data (same-lab
  provenance disclosed in `notes` and `bo4mob_citation`); any future results on
  them are scored by TABench's own certificate discipline."
- **FORBIDDEN, and guarded against here:**
  1. "TABench methods validated on the external BO4Mob benchmark" — BO4Mob is
     **not external**, and its instances are **scenarios, never validation** of
     TABench methods. Stage 1 ships **no method scored on BO4Mob at all**; the
     smoke test is a *liveness* check of the engine pipeline, explicitly **not** an
     oracle (VALIDATION.md tier 5, below everything).
  2. Any claim of **reproducing BO4Mob's published numbers.** The shipped
     `eclipse-sumo` wheel is **1.27.1**; BO4Mob's paper ran SUMO **1.12**. A
     **measured schema drift** (Decision 4) makes the simulated *values*
     non-reproducible here — the *instances* transfer, the *numbers* do not — so
     paper numbers are **never claimed**, in prose or in a pin.
  3. Using TABench runs as "independent replication" of BO4Mob's published strategy
     rankings (SPSA/SAASBO/TuRBO), or vice versa. TABench does **not** re-host
     BO4Mob's BO leaderboard; those comparisons remain BO4Mob's.
  4. Tuning TABench estimators on BO4Mob anchors and reporting it as cross-benchmark
     generalisation.

The disclosure ships **in-artifact**, not only in docs: every registry entry's
`notes` and `bo4mob_citation()` carry the affiliation, the MIT license, the PeMS
provenance, the "scenarios/data only" scope, and the "does not reproduce BO4Mob's
published numbers" clause. Tests pin all four strings.

## Decision 1 — P9 checksummed fetcher family; a separate registry

Data are fetched, never vendored (P9). `BO4MOB_REGISTRY` (`data/bo4mob.py`) holds
the five instances; `fetch_bo4mob` fetches each file from a **commit-pinned** raw
URL (`UMN-Choi-Lab/BO4Mob@ef571e68`), verifies it against a pinned **per-file
SHA-256** on every load, caches it under `~/.cache/tabench/bo4mob/<key>/`, and
evicts + `ChecksumError`s on any mismatch — the TNTP fetcher mechanics, extended
with the xu2024 `.part` hygiene (`try/finally`-unlinked on a mid-download failure).
The body is **streamed with a cap** at each file's pinned **byte size** (+4 KB
slack): an oversized upstream body raises `Bo4MobUpstreamError` mid-stream, before
it is materialised (a hostile 256 MB body no longer fills RSS/disk ahead of the
checksum — the fresh adr-033 standard, restored after review). The `.part` carries
a **per-process suffix** (`.part.<pid>`) so concurrent cold-start fetches never
collide; `fetcher.py` / `xu2024.py` share the latent single-`.part` race (a
documented follow-up, deliberately **not** touched this sprint).

The four small instances (`1ramp`, `2corridor`, `3junction`, `4smallRegion`) each
ship a **single-evaluation bundle**: `net.xml`, `taz.xml`, `od.xml`,
`additional.xml`, `routes_single.csv`, the `od_for_single_run` OD vector, the
`config` JSON, and the **one** canonical sensor CSV (`221008 06-07`). Bundle sizes
measured at the pin: 1ramp **27,327 B**, 2corridor 146,814 B, 3junction 361,055 B,
4smallRegion 643,697 B — **1,178,893 B total**, all download-on-demand. The full
14-date × 3-hour PeMS panel and the `routes_multiple` stochastic route table are
available upstream but **not** pinned in stage 1; stage 2's held-out-date
certificate pins them. (`routes_single` is used precisely because it is
deterministic — one route per OD pair — so the meso run is seed-stable;
`routes_multiple` draws routes with `np.random.choice` and is out of scope.)

The registry is **deliberately separate** from the CI-prefetched TNTP `REGISTRY`
(the xu2024 lesson): a core CI run never fetches any BO4Mob file on its own. Only
the sumo job prefetches the 1ramp bundle (the smoke test's input, < 30 KB). The
registry keys are the **bare instance names** (`1ramp`, `2corridor`, `3junction`,
`4smallRegion`, `5fullRegion`), and — unlike xu2024 — they are **not**
`load_scenario` scenarios: a BO4Mob instance is a mesoscopic-SUMO net with **no
BPR network and no true OD**, so no `Scenario` (Network/Demand/ReferenceSolution)
can be built. That is not a gap; it is the honest statement that these are data +
an engine, not equilibrium instances. A test pins that `load_scenario` raises on
both a bare key (`1ramp`) and a `bo4mob-`-prefixed key.

**`5fullRegion` is registered metadata-only and refuses to fetch.** At 10,100 OD
pairs, 74 MB, and ~11 h/eval it is never a CI input, never a default, never an
anchor. The chosen mechanism is an explicit `hpc_only=True` flag whose fetch raises
`Bo4MobHpcOnlyError` with a clear message — a **named refusal**, not a silent
exclusion (an HPC run fetches it deliberately by other means). This is cleaner than
excluding it: the instance's existence and cost stay visible in the registry and
its metadata (dimension, sensor count) is recorded, while the refusal makes the
"never in CI / never a laptop" boundary executable.

## Decision 2 — the evaluation convention lives in the module; the engine run lives in the test

`data/bo4mob.py` ships the **pure, pandas-free (numpy/stdlib)** transforms of
BO4Mob's own pipeline: `fill_single_od` (fill the count=0 `od.xml` template from
the single-run OD CSV **and rewrite the interval end to the config's
`od_end_time`** — load-bearing, Decision 3), `fix_routes_single` (BO4Mob's
`update_trip_routes` for `routes_per_od='single'` — rewrite each trip's from/to to
the route table's start/last edge, sort by departure, `departLane='best'`),
`edgedata_counts` (the `arrived + left` count convention, `link_flow_analysis.py`),
`bo4mob_nrmse` (the count NRMSE `√(n·Σ(gt−sim)²)/Σgt`, sim filled 0 for absent
sensors), and `local_edgedata_additional` (redirect the meso `edgeData` output to a
per-run workdir file while keeping the `DEFAULT_VEHTYPE` IDM/`speedDev=0`
**attribute-exact** — semantically identical; ElementTree normalises serialisation
whitespace/quoting, so this is not a byte-level claim). The benchmark
core stays pandas-free (BO4Mob's own pipeline uses pandas; this is a faithful
numpy/stdlib reimplementation, validated against the upstream result to the digit —
Decision 3).

The **subprocess orchestration** (the `od2trips` and mesoscopic `sumo` invocations)
lives in the guarded smoke test, exactly as the sumo-marouter adapter keeps its
`_run_marouter` helper in `test_sumo_marouter.py` (adr-027). This keeps the data
layer free of `subprocess`/`sumo` — the module imports numpy/stdlib only and is
importable on a core install — while the engine run is exercised behind the sumo
extra. Binary discovery is `sumo.SUMO_HOME`-only (never PATH / the ambient stale
`SUMO_HOME=/opt/sumo-1.12` this box ships), each subprocess gets `stdin=DEVNULL`
and a workdir-local output, and **one wall deadline threads both subprocesses**
(the adr-027/028 wrapper discipline); a nonzero exit or timeout raises loudly (a
liveness smoke must fail, not censor). Stage 2's estimator will reuse the module's
pure transforms behind a proper guarded adapter.

## Decision 3 — pipeline liveness: the measured 1ramp run (provenance, pinned loose)

Reproduced end-to-end on this box (`eclipse-sumo==1.27.1` wheel), at the pinned
commit, through the pandas-free transforms above:

- **NRMSE = 2.432471221214843** at the shipped `od_for_single_run` OD, date
  `221008`, hour `06-07`, over the 3 GT sensors — **byte-identical across seeds
  0/1/2** (max−min = 0.0). 1ramp is uncongested and `speedDev=0`, so the meso run
  is deterministic; the paper treats the objective as stochastic in general, so
  **no cross-seed byte-identity is claimed beyond 1ramp** (P8 macroreps stay
  mandatory for any future scored row). 3,087 trips kept, 10 edges emitted, ~0.41 s
  per pipeline run (paper reports 0.80 ± 0.29 s for 1ramp).

This is the value **after** the `od_end_time` fill fix (below). The first draft
kept the shipped `od.xml` template's interval `end=3600` instead of rewriting it to
1ramp's `od_end_time=3300` (BO4Mob's own `create_od_tazrelation_xml` **always**
rewrites it), which released ~5% of demand after the OD window — vehicles that
never reach the sensors — and returned **2.314704**, a value that mis-attributed a
TABench od-fill bug to engine drift. The corrected transform reproduces the
upstream-faithful pipeline to the digit (cross-checked against BO4Mob's own pandas
`update_trip_routes` + `parse_link_flow_xml_to_pandas` on the identical
`edge_data.xml`); the mismatch is **1ramp-only** — the other three small instances
have `od_end_time == 3600 ==` the template end, so the rewrite is a no-op there
(measured deltas exactly 0.0).

The smoke test pins these **loose**: the NRMSE in a wide band **1.5 < x < 3.5**
(the measured 2.4325 with generous headroom — a version-robust ceiling, never an
exact decimal, since the meso values shift with the engine build), seed-stability
as `max−min < 1e-3` (measured 0.0, the tol survives cross-platform float drift),
and the pipeline emitting per-edge counts with trips kept. A regression pins that
`fill_single_od` rewrites the interval end to `od_end_time` regardless of the
template end. The **2.432471221214843** is recorded here as *provenance only* — it
is the value under 1.27.1, explicitly **not** BO4Mob's published number under 1.12.

## Decision 4 — the measured SUMO 1.12 → 1.27.1 schema drift (documented, tested)

Under the shipped 1.27.1 wheel, the mesoscopic `edgeData` output carries **no
`nVehContrib` attribute** — a naive parser keying on it silently reads 0.0 for
every edge (a silent-corruption surface). BO4Mob's own count convention is
`arrived + left`, whose attributes still exist under 1.27.1, so the **evaluation
convention transfers unchanged** while the simulated **values** do not match the
1.12 paper values. `edgedata_counts` therefore uses `arrived + left` (with a code
comment naming the drift), and `edgedata_has_nvehcontrib` is a witness the smoke
test asserts **False** — pinning the drift as an executable fact, not a claim.
(The GT CSV column is named `interval_nVehContrib`; that is the PeMS-side count and
is unrelated to the SUMO-side attribute drift — `bo4mob_nrmse` documents this so
the two `nVehContrib` names are not conflated.)

## Stage 2 (named follow-up — NOT shipped here)

A `bo4mob_estimation` T2 task family, its own ADR + adversarial review. The
ADR-002 pinned-bfw certifier **cannot apply**: there is no declared BPR network
(meso SUMO) and **no true OD** (so `od_rmse` / OD-fit columns and the linear
identifiability report are undefined). The honest certificate is a **pinned-engine
observational** one: re-run the pinned, seeded engine at the emitted OD and
recompute NRMSE on **held-out dates/hours/sensors** (the 14×3 panel gives natural
splits; `siouxfalls-t2` already ranks on held-out count fit), macroreplicated (P8).
It carries its **own** non-comparable scale (disclosed as not comparable to the
bfw-certified families — a separate leaderboard table), because no shared-scale
alternative exists (ADR-028 rejected a sumo-pinned certifier only where `bfw`
exists; here it does not). First estimators would be `prior` (the Improvement%
anchor) + an SPSA over the BO4Mob pipeline reusing ADR-028's `SPSAEstimator` hooks;
BoTorch baselines stay **out** (TABench does not re-host BO4Mob's BO leaderboard).
This is the estimation-side sibling of adr-030's deferred observational-certificate
ADR and should be co-designed with it. **Do not start it until the certificate
design is written.**

## Consequences

- **New:** `src/tabench/data/bo4mob.py` (the five-instance registry with a per-file
  SHA-256 **and byte size** pinned at `ef571e68`; the commit-pinned checksummed
  fetcher — streamed with a size cap, per-process `.part`, named HPC-only refusal;
  the pandas-free BO4Mob evaluation transforms; the citation + in-artifact
  provenance line); `tests/test_bo4mob.py` (14 tests — 13 core registry +
  fetcher-hardening + the `fill_single_od` od_end_time regression + the oversized-
  body refusal, 1 sumo-gated pipeline-liveness smoke). The six symbols
  `BO4MOB_REGISTRY`, `BO4MOB_SMOKE`, `Bo4MobSpec`, `Bo4MobHpcOnlyError`,
  `fetch_bo4mob`, `bo4mob_citation` are exported from the data layer
  (`Bo4MobUpstreamError` is available on the module).
- **CI:** the existing `sumo` job gains the 1ramp prefetch (< 30 KB) and
  `tests/test_bo4mob.py` in its pytest list; **no sixth job**. The data cache key
  hashes `bo4mob.py` (the xu2024 lesson) across all jobs. The registry + fetcher
  tests run in the core matrix (no sumo, no network — the fetcher-hardening tests
  mock `urlopen`); the smoke test `importorskip('sumo')`-skips there.
- **Unchanged:** no new dependency (numpy/stdlib core; the smoke uses the existing
  `sumo` extra — the wheel already ships `od2trips` and the meso-capable `sumo`
  binary); the Evaluator, the fairness gate, the TNTP/xu2024 fetchers, `REGISTRY`,
  `load_scenario`, and **every content hash** including the golden Braess hash
  `cf00f411…` (re-asserted in `test_bo4mob.py`). `docs/model-specs.json` is
  untouched — a data family is not a solver (the TNTP / xu2024 precedent), so
  `tools/generate_models.py` is not run.
- **Gaps deliberately left (stage 2):** the `bo4mob_estimation` task + its
  pinned-engine held-out-date observational certificate + estimators; the full
  14×3 PeMS panel and `routes_multiple`; 4smallRegion's pipeline run (only 1ramp is
  the CI liveness anchor); the four small instances beyond 1ramp are fetchable +
  checksummed (data availability) but their engine pipeline is exercised in
  stage 2; 5fullRegion at HPC scale.

## Adversarial review

Three independent lenses (fetcher/P9 integrity, pipeline fidelity, infra), each
executing repros against the uncommitted tree and the real upstream artifact.

**MATERIAL (pipeline fidelity): the `od.xml` interval end was not rewritten to
`od_end_time`.** The first draft's `fill_single_od` kept the shipped template's
`end=3600`; BO4Mob's own `create_od_tazrelation_xml` **always** rewrites it to the
config `od_end_time`. On 1ramp (`od_end_time=3300`) this released ~5% of demand
after the OD window, so the recorded provenance NRMSE **2.314704** mis-attributed a
TABench od-fill bug to engine drift; the BO4Mob-faithful value under 1.27.1 is
**2.432471221214843** (bit-identical across seeds 0/1/2). The reviewer's variant
matrix isolated it: `departSpeed` has zero effect, the `od_end_time` fill is the
whole delta, and the shipped numpy transforms match BO4Mob's own pandas pipeline to
the digit once fixed. The mismatch is **1ramp-only** (the other three small
instances have `od_end_time == 3600 ==` the template end, deltas exactly 0.0).
FIXED in `fill_single_od` (interval end := `od_end_time`), re-recorded here + in
VALIDATION + ROADMAP, and pinned by an offline regression (template-end-agnostic,
mutation-verifiable) plus the corrected smoke band.

**Hygiene/precision, all fixed:** (a) the fetcher buffered `resp.read()` unbounded
(a hostile 256 MB body drove +256 MB RSS / cache disk before the checksum evicted
it — a regression vs the fresh adr-033 standard) → per-file **byte sizes** pinned
in the registry and a **capped stream** that raises `Bo4MobUpstreamError` before an
oversized body materialises; (b) the `.part`-hygiene test was vacuous (its mock
raised at `urlopen()`, before any `.part` existed, so it passed under a
finally-removal mutant) → the mock now fails **mid-body** and the test is
mutation-verified; (c) a concurrent-fetch `.part` race (loud flakes at ~25–37%
under 8-way cold start) → **per-process `.part.<pid>`** suffix, with a note that
`fetcher.py`/`xu2024.py` share the latent pattern (a follow-up, untouched here);
(d) doc precision — the cache path (`bo4mob/<key>/`, not `bo4mob-*`), the bare
registry keys (not `bo4mob-<inst>`), and `DEFAULT_VEHTYPE` "attribute-exact" (not
byte-exact, since ElementTree normalises serialisation); (e) per-seed `edge_data`
filenames in the smoke loop so a future engine exiting 0 without writing cannot
re-read a stale file.

**Repo-hygiene (pre-existing, surfaced this sprint):** `tools/generate_references.py`'s
internal `SHIPPED` dict had drifted far behind the hand-flipped ROADMAP `[x]` rows,
so one accidental `python tools/generate_references.py` would have **unchecked every
shipped row** whose bibkey was absent. Fixed defensively **without running the
generator**: every currently-shipped bibkey was added to `SHIPPED` (verified by
`ast.literal_eval` back-parse against the live ROADMAP `[x]` rows), and the ROADMAP
footer now states the actual practice (rows are hand-maintained on shipping; the
generator carries `SHIPPED` forward and is only for adding new canon entries).

Stage 1 still asserts **no oracle** (no best-known-flow claim, no certified gap):
the honest-oracle surface arrives with the stage-2 certificate ADR.
