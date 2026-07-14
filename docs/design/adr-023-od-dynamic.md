# ADR-023: od-dynamic — Cascetta, Inaudi & Marquis (1993) within-day dynamic OD estimation

**Status:** accepted (implemented)
**Date:** 2026-07-14
**Deciders:** T2 estimation track — adding the within-day time-sliced OD estimator
**File:** `docs/design/adr-023-od-dynamic.md`

## Context

The shipped T2 estimators (`prior`, `vzw-entropy`, `gls`, `spiess`, `spsa`,
`od-congested`, `od-kalman`) all estimate a **single static** OD matrix. Two of
them touch a time axis and neither treats it as the estimand: `gls` collapses its
count periods to a mean (`counts.mean(axis=0)`) because the periods are repeated
draws of *one* stationary flow (time = replication); `od-kalman` (ADR-012) keeps
the day-to-day axis but only as **noise** structure (a VAR(1) covariance) around a
static OD. Cascetta, Inaudi & Marquis (1993, *Transportation Science*
27(4):363-373, `cascetta1993dynamic`, tier-1/white-box in `docs/references.json`)
make the time axis the **signal**: the estimand is a sequence of within-day
time-slice OD matrices `d_h` (departures in slice `h = 1..H`), recovered from
time-sliced link counts `c_{t,a}` (interval `t = 1..T`) linked by a **lagged**
assignment map — a slice-`h` trip crosses a link during a *later* interval
`t >= h` set by the travel time to that link.

The paper's contribution is a taxonomy of estimators and, within it, a
**simultaneous vs sequential** GLS pair:

* **simultaneous** — one GLS over the full stacked (block-lower-banded) system,
  all slices jointly; statistically efficient;
* **sequential** — slice by slice, earlier estimates fixed and subtracted, no
  covariance propagation; online-capable, cheaper, provably less efficient
  because it discards the information later counts carry about earlier slices.

The assignment map is **exogenous and demand-independent** — computed from a
path-choice model and known/measured link travel times, valid on uncongested
networks or congested networks with known link costs. Congestion feedback `M(d)`
is explicitly **out of scope** for the 1993 paper (that is Cascetta & Postorino
2001, `cascetta2001fixed`); so is any Kalman covariance recursion (the
Ashok & Ben-Akiva 2000 state-space successor line). The static assignment core is
sufficient: no dynamic network loading is needed, which dissolves the
`TASKS.md` "(may depend on Phase 2 dynamics)" caveat — the scenario carries the
exogenous frozen map exactly as the paper assumes.

## Decision

Ship od-dynamic as an **estimation-track-local parallel unit** (the ADR-012
pattern): a new task type, ABC, registry, data level, certifier, and runner, all
additive. No edit to `core/scenario.py`, `estimation/base.py`, or
`metrics/estimation.py`; the golden Braess content hash
`cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d` is
byte-identical (re-asserted in `tests/test_od_dynamic.py`, alongside a pinned
static-T2 task hash).

1. **Frozen exogenous lag map** (`estimation/_dynamic_map.py`,
   `lagged_assignment_tensor`). A deterministic, RNG-free
   **free-flow two-interval-split** tensor `M` of shape `(L+1, n_links, n_pairs)`:
   per pair, the free-flow shortest path; per on-path link `a`, the entry offset
   `tau_a` = sum of upstream free-flow times; with `tau_a = q*Delta + r`
   (`0 <= r < Delta`, `Delta` the slice length), uniform within-slice departures
   split the crossings exactly two ways — fraction `1 - r/Delta` at lag `q` and
   `r/Delta` at lag `q+1`. Predicted counts `c_t = sum_l M[l] @ d_{t-l}` (zero
   for `t < h` by causality). This is a **time-invariant** lag form
   (`M_{h,t} = M[t-h]`, one crossing profile reused per slice) — a documented
   restriction of the paper's general time-varying `m^a_ij(t,h)`, and a
   memory-safe representation; it is hashed under `map_recipe = frozen_freeflow_v1`
   so a time-varying variant is a clean bump.

2. **Task type + contract** (`estimation/dynamic_base.py`).
   `DynamicEstimationTask` (frozen, no true profile) carries a `(H, Z, Z)`
   `prior_profile`, the `DynamicLinkCounts` dataset, the exogenous
   sensor-restricted `lag_tensor`, the active `pairs`, and the identifiability
   report; its `content_hash` is domain-prefixed `tabench-t2d-task-v1;` over the
   scenario hash, the slicing dials (`H`, `T`, slice length, map recipe id), the
   prior bytes, sensor links, count bytes, lag-tensor bytes, certificate, held-out
   digest, and seed. `DynamicODEstimator` mirrors `ODEstimator`;
   `DYNAMIC_ESTIMATOR_REGISTRY` / `register_dynamic_estimator` are **separate**
   from the static registry — the type gate that stops the CLI running a dynamic
   estimator on a static task (ADR-002 Decision 1 rationale). `ODTrace` /
   `ODState` / `ODResultBundle` are reused verbatim (they copy any ndarray, so an
   `(H, Z, Z)` checkpoint records exactly as a `(Z, Z)` one).

3. **Data level** (`observe/levels.py`, `DynamicLinkCounts`). Additive: takes the
   expected interval-crossing counts `(T, n_links)` and draws per-`(day, interval,
   sensor)` Poisson counts `(n_days, T, S)` (or exact repeats for `noise='none'`).
   The count axis `t` is **never** collapsed by the certifier.

4. **Estimators** (`estimation/cascetta1993.py`). `od-dynamic-sim`
   (`SimultaneousDynamicGLSEstimator`): one whitened nonnegative GLS
   (`scipy.optimize.lsq_linear`, `bounds=(0, inf)`, the `gls` pattern) over the
   stacked block-lower-banded system with `W = diag((cv_prior*prior)^2 + eps)` per
   `(slice, pair)` and `V = diag(max(cbar, 1)/n_days)` per `(interval, sensor)`.
   `od-dynamic-seq` (`SequentialDynamicGLSEstimator`): for each slice in order,
   estimate it from its **earliest observed interval**, subtracting the frozen
   contributions of already-estimated earlier slices; later slices are not
   subtracted and the slice's own later crossings are discarded (the two
   information losses that make it less efficient). A slice never observed within
   the horizon keeps its prior. Both are single-shot solves with `sp_calls = 0`
   (the map is exogenous — no inner assignment, no outer fixed point), so the T2
   sp-call budget is inert on this track (documented, not a fake charge); the
   GLS core acts on a **general** block map, so it also covers a time-varying
   stacked map (the A2b pure-math pin).

5. **Certificate (P1)** (`metrics/estimation_dynamic.py`, `DynamicODCertifier`).
   Model-blind and **exact** — the map is linear, so there is no pinned assignment
   to run. The certifier **regenerates the full-network lag tensor from the hashed
   recipe** (never the estimator's payload — map forgery is impossible) and scores
   the emitted `(H, Z, Z)` profile: `obs_count_rmse` over every `(day, interval,
   obs sensor)` (never collapsed over `t`), `obs_mean_count_rmse` (the P1
   honesty-diff target), the `oracle_*` floors at the planted truth, and — the
   **ranking** column — `heldout_count_rmse` on a disjoint held-out sensor set
   (plus `heldout_flow_rmse`). OD-fit columns (`od_rmse`, `od_nrmse`, signed
   `total_demand_error`, and the new descriptive `profile_rmse` over normalized
   slice totals) are always reported, ranking nothing, flagged `od_identifiable=0`
   when the identifiability report is negative. Censoring matches `ODCertifier`:
   wrong shape raises; non-finite / sub-tolerance-negative censored; a zero
   profile is not censored.

6. **Identifiability** (`experiments/runner.py`, `dynamic_identifiability_report`).
   **Exact** (not linearized as static T2 must be — the benchmark generates counts
   through the same linear map): the stacked observed map `A` (`(T*S, H*P)`) is
   built and its rank, Hazelton distinct-nonzero-column condition, number of
   horizon-**truncated** slices (all-zero column blocks — the genuinely new edge),
   and number of **confounded** columns (duplicate classes, including cross-slice
   temporal confounds) are reported. `linear_identifiable` (full column rank) gates
   `od_identifiable`.

7. **Runner + CLI + cards.** `run_dynamic_estimation_experiment` parallels
   `run_estimation_experiment` (same reserved RNG streams, sensor-placement
   substream, obs/held-out disjointness guard, held-out digest, stochastic-implies-
   reps semantics); the truth profile is `d*_h = rho_h * scenario.demand.matrix`
   from a hashed per-slice profile dial. The CLI dispatches the `t2_dynamic_
   estimation` card token and grows a "Dynamic estimators" `list` section; cards
   `scenarios/0braess-t2d.yaml` and `0tworoute-t2d.yaml` ship (small `H`; Sioux
   Falls is skipped — its `H*n_pairs` stacked system is large and the small rungs
   carry the anchors).

## Analytic anchors

All recomputed in-test as closed forms (`tests/test_od_dynamic.py`, house style:
no trusted digits; the rationals below are the recomputed values).

- **A1 — integer-lag reduction.** `tau = Delta` exactly (`M0=0, M1=p`): each slice
  couples to exactly one interval, the stacked GLS block-diagonalizes, and BOTH
  estimators reduce per slice to the static gls closed form
  `d_h = (z_h + p c_{h+1})/(1 + p^2)` (assert `1e-12`; sim == seq).
- **A2 — fractional-lag (the load-bearing anchor).** `tau = Delta/2`
  (`M0 = M1 = 1/2`), `H=2`, `T=3`, truth `(4,6)`, prior `(3,3)`, `V=W=I`:
  simultaneous `(128/35, 142/35)`, sequential `(16/5, 94/25)`; the simultaneous
  estimate **strictly dominates** the sequential componentwise vs truth — the
  paper's information-loss claim as a deterministic hand computation.
- **A2b — pure-math pin (time-varying stacked map).** `M = [[.5,0],[.5,1]]`,
  `z=(10,10)`, `y=(7,14)`: simultaneous `(116/11, 103/11)`, sequential
  `(10.8, 9.3)`, and the no-prior square case `xhat = M^{-1} y = (14,7)`. The
  sequential-minus-simultaneous error-covariance gap is exactly **rank-1 PSD**
  (eigenvalues `{0, 17/220}`) — a **test-local derivation** of the plug-in
  scheme's efficiency loss, NOT attributed to the paper, with no shipped
  covariance API.
- **A3 — mean-collapse distinctness.** Truths `(4,6)` and `(6,4)` give identical
  interval **means** but distinct per-interval counts; `od-dynamic-sim` separates
  them, a mean-collapsing estimator provably cannot — the executable "not a gls
  rename" witness.
- **A4 — temporal confounding (the new false-accept surface).** Two origins, lags
  0 and 1, identical stacked columns: `od_identifiable = 0`,
  `n_confounded_columns > 0`, and a count-invariant cross-slice shift certifies
  IDENTICAL obs **and** held-out counts — because held-out sensors *share* the lag
  structure (unlike static T2, where held-out links broke the Braess `D=2/D=6`
  tie). Defense: rank/`sigma_min` confound flags gate `od_identifiable`; OD
  columns never rank. A third count-invariant family — demand on OFF-SUPPORT
  cells, which have no lag column at all — is censored outright (review MAJOR,
  see below).
- **A5 — horizon truncation.** Last slice unobservable (`T = H` with an integer
  lag): `n_truncated_slices = 1`; demand dumped there leaves the count-fit (obs
  and held-out) unchanged while `od_rmse` / `total_demand_error` move — the
  executable warning that OD columns are descriptive under `od_identifiable = 0`.

## Alternatives considered

- **Extend `gls` / `EstimationTask` to carry a time axis:** rejected — it would
  collapse the estimand back to a static matrix and bloat the static contract.
  A parallel task/registry keeps both contracts honest (the ADR-012 precedent).
- **Congestion-endogenous map `M(d)` (bilevel dynamic ODME):** out of scope for
  the 1993 paper; belongs with Balakrishna (2007) / a DTA-calibration track.
- **A Kalman/state-space sequential recursion with covariance propagation:**
  belongs to the Ashok & Ben-Akiva (2000) successor line; the read sources present
  the 1993 sequential estimator as a plug-in *without* covariance propagation, so
  that is what ships. The rank-1 efficiency-gap covariance (A2b) is our derivation
  of the plug-in scheme's behavior, clearly labeled and test-local.
- **A per-(pair, slice) map from the DNL core:** the DNL loader is aggregate
  single-commodity (ADR-010 defers multi-commodity per-destination emission), so a
  certifiable per-pair dynamic map does not exist there yet; the exogenous
  free-flow map is the paper-faithful realization.
- **Average-flow estimators / non-GLS variants** from the paper's taxonomy:
  excluded from the certified scope (only the simultaneous/sequential GLS pair was
  recovered in equation form).

## Consequences

The benchmark gains its **first within-day time-sliced** OD estimator — the third
leg of the T2 temporal triangle, genuinely distinct from `gls` (time =
replication) and `od-kalman` (time = day-to-day noise). Certification is exact
linear algebra (cheaper than static T2's pinned bfw). All changes are additive;
the golden Braess hash is provably preserved; no output contract changed. The new
false-accept surface — held-out sensors sharing the lag structure — is met by the
exact identifiability report and descriptive-only OD columns (A4/A5). Follow-ups:
a time-varying lag tensor (`v2` recipe), the average-flow estimator variants, and
a congestion-endogenous map.

## Adversarial review

Three independent lenses (soundness, formulation, numerics), each executing
code; every finding below is CONFIRMED by a runnable repro and regression-pinned
in `test_od_dynamic.py` (streak: 12/12 sprints with at least one material
defect caught).

**MAJOR (formulation): off-support demand was count-invisible.** The certifier
extracted only the active-pair cells of the emitted `(H, Z, Z)` profile, so
demand dumped on any *other* off-diagonal cell (e.g. the unused reverse
direction) contributed to NO harness-recomputed count — obs, held-out, and the
ranking `heldout_count_rmse` were byte-identical to the truth's while
`od_identifiable = 1` was asserted (the identifiability report is built over
active-pair columns only). A count-invariant family *beyond* the documented
A4/A5 surfaces. FIXED: profiles carrying off-support mass beyond tolerance
(support = active pairs + diagonal) are censored — the dynamic analogue of the
static track, where the pinned assignment loads the FULL emitted matrix and
unroutable demand raises. Not rank-exploitable (ties, never beats), hence MAJOR
not CRITICAL.

**MAJOR (numerics): `prior_var_floor = 0` hung the harness.** The factor's
declared bounds allowed `0.0`; with any zero prior cell the whitened prior row
became infinite and `lsq_linear` hung indefinitely (or raised `LinAlgError`).
FIXED: bounds now `(1e-12, 1e12)` on both estimators; the smallest allowed
floor solves a zero-prior-cell instance cleanly.

**MINORs, all fixed + pinned:** (a) the negativity censor's tolerance scaled
with `max|q|` *including the diagonal*, so a huge intrazonal cell let a
genuinely negative active cell escape censoring — the scale is now off-diagonal
only, and the identical inherited convention in the static `ODCertifier` was
fixed for parity (pinned in `test_estimation.py`); (b) any `estimation.map`
string was silently accepted while the v1 map was built anyway, letting a bogus
recipe id flow into the certificate/manifest/hash — now validated against
`MAP_RECIPE`; (c) the lag offsets `tau` accumulated the free-flow *generalized
cost*, so tolls (money, not minutes) shifted crossing intervals on tolled
networks — `tau` now accumulates pure `free_flow_time` while path *choice*
keeps the generalized cost; (d) `content_hash` omitted `payload["pairs"]`, so
hand-built tasks with estimands in different OD cells hashed equal — hashed
now, domain bumped to `tabench-t2d-task-v2;`; (e) the identifiability gate was
a default-tolerance float SVD rank documented as "exact" — it now reports
`sigma_min` explicitly and asserts `linear_identifiable` only at full nominal
rank AND `sigma_min > 1e-6` (documented practical floor); (f) `n_slices: 0`
crashed deep inside scipy and an empty held-out set silently NaN'd the ranking
column — both are clean `ValueError`s now. NOTE: the sequential docstring's
"later slices are not subtracted" loss was provably vacuous on time-invariant
tensors — reworded (the real losses: later crossings discarded; frozen
carryover without covariance propagation).

**Survived (highlights):** no information-free ranking exploit on
`heldout_count_rmse` (overfitting obs sensors *raises* the held-out RMSE;
confound/truncation shifts only tie); 1500 random time-invariant + 800
time-varying instances against two independent from-scratch GLS
implementations — zero disagreements, zero KKT violations; the pinned
"earliest observed interval" sequential variant matches the interval-indexed
reading on both shipped cards and is a defensible member of the paper's
sequential class; the two-interval split confirmed by a 2M-departure
simulation; every anchor rational re-derived in sympy; map forgery impossible
(certifier regenerates from `(scenario, slice_length, n_lags)` only); golden
Braess + static-T2 hashes byte-identical; both cards run end-to-end through
the CLI with sane rankings; no BLAS-sensitive assertions.

## Sourcing

The **primary is paywalled** (INFORMS; Unpaywall status CLOSED). Its abstract was
read verbatim at the INFORMS landing page
(https://pubsonline.informs.org/doi/10.1287/trsc.27.4.363) — the simultaneous vs
sequential taxonomy, the GLS test on the Italian Brescia-Padua motorway with
"true" OD flows, and the "no a priori information" result. The paper's own symbols,
equation numbers, and section structure are **unverified**, so no equation number
of the 1993 paper is cited here. The formulation was **cross-verified against open
sources read in full**:

- Djukic, T. (2014). *Dynamic OD demand estimation and prediction for dynamic
  traffic management.* PhD thesis, TU Delft (TRAIL) — measurement eq. 2.3/2.7, the
  assignment decomposition eq. 2.8 (credited to Cascetta et al. 1993), the
  simultaneous GLS eqs. 2.10-2.11 and sequential GLS eqs. 2.12-2.13, and the
  exogenous-assignment/known-cost regime.
  https://repository.tudelft.nl/file/File_38167790-381b-43ae-91a1-44464be30a36
- Peterson, A. (2007). *The Origin-Destination Matrix Estimation Problem —
  Analysis and Computations.* Linköping University licentiate thesis (DiVA) — the
  two-objective time-dependent form (eq. 16) and the Cascetta-1993 assignment-
  fraction decomposition (eq. 17).
  http://www.diva-portal.org/smash/get/diva2:23558/FULLTEXT01.pdf
- Castiglione, M. et al. (2021). *Assignment Matrix Free Algorithms for On-line
  Estimation of Dynamic Origin-Destination Matrices.* Frontiers in Future
  Transportation, doi:10.3389/ffutr.2021.640570 — the sequential/rolling-horizon
  formulation fixing the ODs of previous time slices, and the computational-
  complexity contrast.
- Cipriani, E. et al. (2014). *Effectiveness of link and path information on
  simultaneous adjustment of dynamic O-D demand matrix.* European Transport
  Research Review 6:139-148, doi:10.1007/s12544-013-0115-z — an independent
  statement of the simultaneous joint-slice GLS objective (citing Cascetta et al.
  1993).

The **sequential covariance analysis** (A2b's rank-1 PSD efficiency gap) is **our**
derivation of the plug-in scheme's exact linear-Gaussian behavior, clearly labeled
in-test and not attributed to the paper. The **time-invariant lag tensor** is a
documented restriction of the paper's general time-varying fractions. **No
Brescia-Padua experiment detail** (slice duration, counter/slice counts, error
metrics, how "true" OD was obtained) is asserted. The **average-flow estimator
variants** are excluded from the certified scope. No number from the primary is
reproduced — every anchor is a hand-derived closed form.
