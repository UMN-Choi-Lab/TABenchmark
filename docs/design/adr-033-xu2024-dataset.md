# ADR-033: xu2024-dataset — the Xu et al. (2024) 20-US-city cross-domain axis

**Status:** accepted (implemented)
**Date:** 2026-07-15
**Deciders:** non-solver roadmap — redeeming the adr-006 cross-domain-axis promise
**File:** `docs/design/adr-033-xu2024-dataset.md`

## Context

Every scored instance in TABenchmark so far is either a hand-built analytic anchor
or one of the four donated TNTP networks (Sioux Falls, Anaheim, Barcelona,
Winnipeg). adr-006 (learned-model certification) named the missing piece
explicitly: "The Xu et al. 2024 real-city dataset, disjoint from TNTP and CC-BY,
is a natural future *cross-domain* test set; it is referenced, not vendored."
This ADR redeems that promise — a benchmark that claims to test whether a method
(learned or classical) *generalizes* needs instances off the standard four.

**The artifact.** Xu, Zheng, Hu, Feng & Ma (2024), *A unified dataset for the
city-scale traffic assignment model in 20 U.S. cities*, Scientific Data 11:325,
DOI [10.1038/s41597-024-03149-8](https://doi.org/10.1038/s41597-024-03149-8)
(`xu2024unified`, canon reference 276). The data live on figshare
(`10.6084/m9.figshare.24235696`, v4) as one **276 MB** zip (file id 48908890,
whole-zip md5 `3f7632e00599588abecbcfc488f862b2`), **CC BY 4.0**. Each of the 20
cities ships an OSM-derived network, an OD demand matrix, and computed equilibria
from two solvers (TransCAD and AequilibraE). The integration target is the
per-city **AequilibraE trio**: `network.csv` (link table + per-city-calibrated
BPR α/β), `od_demand.aem` (the OD matrix), and `assignment_result.csv` (the
published UE link flows).

**The discovery (the load-bearing finding).** The dataset's published AequilibraE
runs injected the OD demand at node ids `1..Z` — which are **not** the tract
centroids the dataset intends. This is not an inference: it is visible in the
authors' own published `AequilibraE_assignment.py`, which calls
`g.prepare_graph(np.arange(zones)+1)`. Measured consequence: with centroids taken
as `1..Z`, the shipped AequilibraE flows conserve machine-exactly (max node
residual ~1e-11); with the intended tract nodes (`10000000+k`) conservation fails
wildly, and those nodes sit a median **6.4 km** from the corresponding tract. The
dataset's **TransCAD** side used the *correct* centroids (and a different
connector speed), so the two shipped solver results are **different instances**.
Any "cross-solver agreement" claim on the shipped artifacts would therefore be
false, and this ADR never makes one.

## Decision 1 — Ship Variant A (AS-PUBLISHED), defer Variant B

We ship the **as-published AequilibraE instances**: node ids `1..Z` are the zone
centroids, exactly as the published run treated them. These are self-consistent
and machine-verifiable — the published flows conserve and (approximately)
equilibrate on this graph — and need only the numpy/stdlib parsing of the trio.
`first_thru_node=1`: the published run allowed through-centroid flow and the
low-id "centroids" are real intersections, not pure zone stubs.

The **corrected tract-centroid variant** (Variant B) — demand at the true tract
centroids, cross-checkable against the TransCAD reference — is **deferred**. It
needs DBF/SHP/TransCAD-convention parsing, its reference is only ~1e-2 accurate,
and it carries the connector-speed divergence; it is named future work, not part
of this sprint.

## Decision 2 — The wrong-centroid defect is a documented known-defect (P2)

The defect is recorded prominently, in the same honest spirit as the TNTP unit
quirks: in the module docstring (`data/xu2024.py`), in each scenario card's
`known_defect` field, and in every scenario's `ReferenceSolution.note`. The
shipped instances are labelled AS-PUBLISHED throughout. The one claim the data
*cannot* support — cross-solver TransCAD↔AequilibraE agreement — is stated as
off-limits everywhere it might be tempting.

## Decision 3 — P9 integration by HTTP-range extraction, on a separate registry

Data are fetched, never vendored (P9). The primary fetch path opens the figshare
zip over HTTP **byte ranges** (`_HttpRangeFile` + `zipfile`): the central
directory is read from the zip's 276 MB tail, and only the three needed members
are inflated — a single-city load transfers ~1–4 MB, **never** the whole archive
(measured: Honolulu's trio downloads + builds in ~12 s over ~12 range requests).
Each extracted file is verified against a pinned SHA-256; a documented fallback
(whole-zip download + md5 verify + extract) covers the case where a server
refuses ranges. The version-immutable figshare file id keeps the URL alive across
future deposit versions.

The 17 cities live in a **separate** `XU2024_REGISTRY`, deliberately **not** in
the TNTP `REGISTRY` that CI prefetches (`.github/workflows/ci.yml` runs
`[fetch(s) for s in REGISTRY.values()]`). So a CI run never pulls 276 MB on its
own: the two rungs' trios are fetched lazily only when their tests run, and the
other 15 cities are never fetched in CI at all. `load_scenario("xu2024-<city>")`
routes to the builder; the family key `xu2024-<city>` gives the P7 fairness gate
the cross-domain axis for free.

## Decision 4 — Honest validation: the published flows are a LOOSE reference

The published AequilibraE flows are **not** a best-known oracle like the TNTP
`*_flow.tntp` solutions (those certify at AEC ~1e-15). Their own recomputed
relative gap is ~1e-3 — roughly 11 orders looser — because the paper solved to a
gap target of 0.001. They are used only as a **provenance cross-check**, never as
a regression oracle. The real validation is four measured facts:

1. **Builder invariants on all 20 cities** (the audit below): unique links,
   `fft > 0`, `cap > 0`, BPR α,β ≥ 0, node-id ranges, `link_id = 1..L`, the
   AEM header size identity, and the `1..Z`-centroid convention.
2. **Published-flow conservation**, recomputed from the flows: max node-balance
   residual **≤ 3e-10** (worst: New York ~2e-10; Honolulu 1.1e-11) across the 17
   shipped cities. The prose bound is deliberately loose of the measurement —
   the residual is summation-order sensitive (two review lenses measured New
   York at 2.00e-10 and 2.04e-10), so a zero-headroom "≤ 2.0e-10" pin would be
   fragile; the test asserts the far looser `< 1e-9`.
3. **The repo's own bfw certifying a tighter gap on the identical instance** —
   genuine cross-*implementation* agreement (AequilibraE bfw ↔ tabench bfw).
   On Honolulu, `BiconjugateFrankWolfeModel` reaches relative gap **1.07e-4** in
   400 iterations (~100 s, measured 104.5 s on the dev box), with **correlation
   0.99992** to the published flows and Σ|Δv|/Σv = 0.96%. (The CI anchor uses a
   small budget and checks
   feasibility + correlation > 0.99; the tight number is this in-sprint
   measurement, not a CI pin.)
4. **Correlation with the published flows** as provenance (Honolulu 0.99992).

## Decision 5 — The 20-city audit: 17 ship, 3 excluded and named

The builder + invariants + conservation battery was run on **all 20** cities
in-sprint. **17 pass and ship.** **3 are excluded and named**: Washington,
Pittsburgh, Phoenix. Their failure is itself further evidence of the
wrong-centroid defect — a handful of centroid ids in `1..Z` are **absent** from
the network as low-id nodes (they exist only as their `10000000+k` tract ids) yet
still carry demand (Washington 2 of 179; Pittsburgh 5 of 149; Phoenix 4 of 378),
so no valid `Network` can be built on the `1..Z` convention. They are candidates
for the deferred Variant B. BPR α/β vary by city (per-city grid-search
calibration, paper Table 5) but are uniform within each city; published gaps span
7.6e-4–1.37e-3.

| # | City | Zones | Nodes | Links | BPR α | BPR β | Demand | Published rgap | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 01 | San Francisco | 194 | 4986 | 18002 | 0.5 | 1.8 | 168,828 | 1.27e-03 | **ship** |
| 02 | Seattle | 139 | 6891 | 27361 | 0.6 | 3.0 | 133,900 | 9.18e-04 | **ship** |
| 03 | Portland | 157 | 8245 | 31939 | 0.5 | 1.2 | 136,471 | 8.97e-04 | **ship** |
| 04 | Las Vegas | 175 | 7823 | 28831 | 0.5 | 1.3 | 76,532 | 1.02e-03 | **ship** |
| 05 | Chicago | 819 | 14434 | 54469 | 0.5 | 1.2 | 468,616 | 1.21e-03 | **ship** |
| 06 | New Orleans | 185 | 7217 | 24073 | 0.6 | 1.8 | 59,875 | 7.78e-04 | **ship** |
| 07 | Austin | 199 | 10717 | 40158 | 0.5 | 1.5 | 202,867 | 1.03e-03 | **ship** |
| 08 | Minneapolis | 130 | 4004 | 15363 | 0.15 | 1.8 | 60,167 | 7.58e-04 | **ship** |
| 09 | Dallas | 328 | 21389 | 77818 | 0.6 | 1.3 | 200,434 | 1.04e-03 | **ship** |
| 10 | Milwaukee | 234 | 8521 | 30747 | 0.5 | 1.5 | 98,220 | 9.37e-04 | **ship** |
| 11 | New York | 2005 | 28626 | 99408 | 0.25 | 1.5 | 1,782,020 | 9.47e-04 | **ship** |
| 12 | Washington | 179 | 6136 | 23573 | 0.5 | 1.5 | 123,084 | — | **excluded** |
| 13 | Boston | 191 | 5542 | 20487 | 0.25 | 2.0 | 119,746 | 1.37e-03 | **ship** |
| 14 | Philadelphia | 389 | 10410 | 38641 | 0.5 | 1.2 | 233,554 | 9.49e-04 | **ship** |
| 15 | Pittsburgh | 149 | 3532 | 13662 | 0.5 | 2.0 | 48,436 | — | **excluded** |
| 16 | Miami | 108 | 4121 | 15108 | 0.5 | 1.5 | 39,412 | 8.92e-04 | **ship** |
| 17 | Atlanta | 141 | 5207 | 20243 | 0.2 | 1.5 | 61,069 | 8.10e-04 | **ship** |
| 18 | Phoenix | 378 | 15324 | 58070 | 0.15 | 1.2 | 284,808 | — | **excluded** |
| 19 | Denver | 175 | 9205 | 34724 | 0.5 | 1.5 | 120,434 | 1.00e-03 | **ship** |
| 20 | Honolulu | 117 | 2982 | 11205 | 0.5 | 1.5 | 107,515 | 1.04e-03 | **ship** |

## Rungs

**Honolulu** (rung 5, 11,205 links) and **San Francisco** (rung 6, 18,002 links)
are the CI-sized rungs (`scenarios/5honolulu-xu2024.yaml`,
`scenarios/6sanfrancisco-xu2024.yaml`): small enough to fetch (~1 MB trio) and
certify quickly. The other 15 cities build via the identical path but are
local-only by download/solve cost (New York is ~10× Honolulu). New York, at 2,005
zones and 99,408 links, is the largest instance in the entire benchmark.

## Consequences

- **New:** `src/tabench/data/xu2024.py` (registry of 17 cities; HTTP-range
  fetcher — primary ranged path with a narrowed whole-zip fallback that fires
  only on a transport error, and a pre-transfer size guard against the pinned v4
  artifact on both paths; AEM/network/assignment parsers with drift guards on the
  1..Z zone index and the fully-directed convention; scenario builder; citation);
  two scenario cards; `tests/test_xu2024.py` (20 tests). `load_scenario` routes
  `xu2024-<city>`; `xu2024_scenario`, `XU2024_REGISTRY`, `XU2024_RUNGS`,
  `Xu2024UpstreamError` are exported from the package root / data layer.
- **Unchanged:** no new dependency (numpy/stdlib only — no pandas); the Evaluator,
  the fairness gate, the CI prefetch, and **every content hash**, including the
  golden Braess hash `cf00f411…` (re-asserted in `test_xu2024.py`). The `xu2024`
  data node in `docs/model-specs.json` stays `shipped:false`/`not-a-solver`,
  matching the TNTP (`stabler2016transportation`) precedent — a dataset is not a
  solver.
- **Cache-eviction skip semantics (inherited from the TNTP fetcher):** a checksum
  mismatch evicts the cached file and raises `ChecksumError`; a *subsequent*
  offline load in the same session then reports "data unavailable" and skips (via
  the house `load_or_skip`) — that skip reflects the eviction, not a missing
  initial download. CI closes this window with `TABENCH_REQUIRE_DATA=1`, under
  which any data failure is a hard error, never a skip.
- **Gaps deliberately left:** the corrected tract-centroid **Variant B** (with a
  TransCAD cross-check) and its three currently-excluded cities; the 15 non-rung
  cities are available but not CI-run. A learned-model cross-domain train/test
  split (TNTP → xu2024) is now *possible* but is future work, not shipped here.

## Adversarial review

Three independent lenses (fetcher/P9 integrity, builder correctness + honest-
validation claims, numerics/infra/integration), each executing repros against the
uncommitted tree and the REAL figshare artifact. **The first sprint of the review
program with zero confirmed material defects** (21 sprints reviewed, 20 with at
least one confirmed material defect): every corruption, tamper, truncation, and
upstream-v5-swap attack ended in a loud `ChecksumError` with nothing unverified
left in the cache — the measure-first, integrate-second dossier (which had already
solved Honolulu and found the upstream defect before any integration code existed)
is the plausible cause, and the pattern is worth keeping.

**Hygiene/precision findings, all fixed:** the "≤ 2.0e-10" conservation prose
bound had zero headroom (one lens recomputed New York at 2.037e-10 while another
measured exactly 2.0e-10 — summation-order sensitivity; the BLAS lesson in prose
form; restated with headroom); a failed whole-zip fallback stranded an unbounded
`.part` file in the cache (now `try/finally`-unlinked); the whole-zip fallback
triggered on ANY ranged-path exception including permanent structural errors,
contradicting the ci.yml "never the 276 MB zip" comment (now transport-only, with
`ZIP_SIZE` as a live pre-transfer guard raising `Xu2024UpstreamError` on an
upstream artifact change BEFORE any bulk transfer); adr-006 still called the set
"future work" (touched up); the `.aem` zone index and `direction` column are now
asserted at parse time (verified true on all 17 cities — upstream drift becomes
loud, not silent); wall-time claims harmonized; the checksum-eviction offline-skip
semantics disclosed.

**Attacked and survived (highlights):** per-file sha256 checked on EVERY load,
post-decompress (byte flips in all three file kinds → loud, evicted, re-fetch
heals); a tampered zip member with a VALID CRC still refused; truncated ranged
responses and mixed 206/200 servers never silently parse; a v5 upstream re-upload
cannot poison either fetch path; the CI byte-budget claim measured TRUE against
real figshare (both rungs = 1,027,033 bytes in 24 range reads + 2 probes — 0.37%
of the 276 MB zip); warm-cache tests fully offline, cold-cache skips clean,
`TABENCH_REQUIRE_DATA=1` turns skips loud; CLI byte-parity for every existing
scenario (all 12 existing content hashes identical, `run`/`fetch` outputs
byte-identical against HEAD); two cold builds byte-identical (hash
`1baa0c04…` stable across processes and caches); the excluded-city analysis
reproduced exactly (the missing low-id centroids still carry demand — further
evidence of the wrong-centroid defect); the Honolulu bfw anchor reproduced to
every printed digit (rgap 1.074e-4 @ 400 iters, corr 0.999919); no cross-solver
claim vs TransCAD anywhere (repo-wide sweep — every mention is a disclaimer);
`tools/generate_models.py` reproduces the shipped MODELS.md byte-identically;
the golden Braess hash untouched.
