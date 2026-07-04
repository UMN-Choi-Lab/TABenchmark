# ADR-002 — T2 estimation track: OD demand from link counts, and the pinned-assignment certificate

**Status:** accepted (implemented in v1)
**File:** `docs/design/adr-002-t2-estimation-certificate.md`

## Context

ARCHITECTURE §2 promises **T2 Estimation/calibration**: recover demand from a `DataLevel`
output, with point metrics reported separately from distributional metrics. The Hazelton
(2015) axis — observation processes as first-class, identifiability reported per sensor
configuration, out-of-sample judgment — is what differentiates this benchmark from every
solver bake-off; T2 is where that axis carries a leaderboard. What exists today:
`observe/levels.py` (`FullOD`, `LinkCounts(sensor_links, n_periods, noise)` with per-period
Poisson counts, `random_sensor_mask`, and `distinct_nonzero_columns` — Hazelton Prop. 1),
reserved RNG sources (`SOURCE_OBSERVATION`, `SOURCE_EVALUATION`), the frozen `Scenario`
with `content_hash()`, and the P1 `Evaluator` for flows. Nothing consumes `LinkCounts` yet.

Two hard questions, in ADR-001's spirit:

1. **The contract.** An estimator must see the network, a prior OD, and the observed
   dataset — and must be *structurally unable* to see `scenario.demand`. P7 says fairness
   is enforced by the harness, never by convention, so "please don't read `.demand`" is
   not a design.
2. **The certificate (P1 analog).** For T1 the harness certifies emitted *flows* with one
   AON pass. For T2 the emitted artifact is an **OD matrix**; its quality is only defined
   *through an assignment map*. Whose assignment? If each estimator's own, the metric is
   gameable (assign badly, fit counts trivially) and models don't share a scale. The
   harness must therefore run a **pinned reference assignment** on the emitted OD and score
   the resulting flows — the T2 analogue of "the harness runs AON at your costs".
   And per Hazelton, count-fit must be judged **out of sample**: a wrong OD can fit the
   observed sensors perfectly whenever the sensor set under-determines demand (on Sioux
   Falls, 76 links vs 528 positive OD pairs — *always* under-determined at the mean-count
   level), so the certificate carries held-out sensors and never collapses count-fit and
   OD-fit into one number.

## Decision 1 — Contract: `ODEstimator` ABC over a demand-free `EstimationTask`

```python
# src/tabench/estimation/base.py
@dataclass(frozen=True)
class EstimationTask:
    """Everything an estimator may see. Contains NO true demand, by construction."""
    name: str
    network: Network              # declared cost functions -> white-box assignment allowed
    prior: Demand                 # target/seed matrix (StalePriorOD projection, Decision 5)
    dataset: Dataset              # LinkCounts payload: counts (n_periods, n_obs), sensor_links
    identifiability: Mapping[str, Any]   # public report (Decision 4)
    scenario_hash: str            # provenance pin (SHA-256; not invertible)
    heldout_digest: str           # SHA-256 over sorted held-out links + ho n_periods
                                  # (the *design*, not the locations -> no P7 leak)
    def content_hash(self) -> str: ...   # SHA-256 over scenario_hash + prior + sensor
                                         # links + observed counts + dataset dials +
                                         # certificate pin + heldout_digest + seed

class ODEstimator(ABC):
    name: ClassVar[str]
    capabilities: ClassVar[Capabilities]   # paradigm="estimation" (added to PARADIGMS),
                                           # inputs_required={"link_counts","prior_od"},
                                           # outputs={"od_estimate"}
    factors: ClassVar[dict[str, FactorSpec]] = {}
    # __init_subclass__/__init__ factor resolution identical to TrafficAssignmentModel
    @abstractmethod
    def estimate(self, task: EstimationTask, budget: Budget,
                 rng: RngBundle, trace: ODTrace) -> ODResultBundle: ...
```

`ODTrace`/`ODState` mirror `Trace`/`FlowState` with `od_matrix` in place of `link_flows`
(defensive copy, `coords: BudgetCoords`, `self_report` provenance-only) — a 30-line
parallel in `estimation/base.py`, mirroring how ADR-001 added `StochEngine` beside
`PathEngine` rather than generalizing either. `ESTIMATOR_REGISTRY` + `register_estimator`
mirror `MODEL_REGISTRY`. A black-box estimator plugs in through `CallableEstimator`
(`fn(task, generator) -> od_matrix`, instance-level capabilities with `trained_on`),
mirroring `CallableModel`.

**Alternatives rejected.**
- *(b) `solve_estimation(...)` on `TrafficAssignmentModel`:* bloats the one-method
  contract (P4) — every T1 solver would carry an unimplemented second abstract method or
  a stub; `Trace` would have to hold two artifact types; `run_experiment` would branch on
  task type inside the loop. The two tasks share `Capabilities`, `FactorSpec`, `Budget`,
  `RngBundle`, and the registry *pattern*, which is the right amount of sharing.
- *Passing the full `Scenario` and trusting estimators not to read `.demand`:*
  unenforceable, violates P7. Also breaks the fairness audit story for subprocess/neural
  adapters, which serialize whatever they receive.
- *Cloning the scenario with `demand=prior`:* conflates prior with truth, silently changes
  `content_hash`, and makes the prior look load-bearing to feasibility audits. Rejected.
- *A generic `Task` layer now:* ADR-001 already deliberated this and noted "a v1 Task
  layer can construct/point to the hashed scenario". `EstimationTask` **is** that layer,
  scoped to T2 only; T1 stays exactly as is.

The estimator receives the `Network` *with its declared cost functions*, so classical
estimators run their own inner assignments (white-box, P4); which inner solver and inner
gap they use are declared **factors** of the estimator, paid for out of its own budget —
only the *certificate* assignment is pinned (Decision 2). The harness applies
`assert_fair_evaluation(est.capabilities, scenario)` against the true scenario before
building the task (P7 lineage gate unchanged).

## Decision 2 — Certificate: pinned reference assignment + (count-fit, OD-fit) pair

For every emitted checkpoint `Q̂` that passes censoring, the harness
(`metrics/estimation.py::ODCertifier`, model-blind, mirroring `Evaluator`):

```
v̂ = UE(Q̂)      # PINNED: bfw, cold start, target_relative_gap=1e-6,
                 # max_iterations=5000, line_search_xtol=1e-12 (factor defaults)
```

**Why this pin.** `bfw` is the repo's fastest certified UE solver (Winnipeg RG 1e-4 in 57
iterations vs fw's 161); UE link flows are unique under strictly increasing BPR costs, so
the *map* Q→v is well-defined — the pin only fixes the finite-budget approximation and
tie-breaking bytes. `1e-6` sits two orders below Boyce, Ralevic-Dekic & Bar-Gera's (2004)
1e-4 stable-link-flows threshold, so certificate noise is far below any count-fit
difference the leaderboard could care about; the *achieved* gap is recorded per
certificate (`certificate_gap`) with `certificate_converged` flagged 0 if the cap bound
first — recorded, never silently absorbed. Cold start is pinned: warm-starting across
checkpoints would change bytes and break bit-reproducibility (P8) for a speed win we
don't need on the sprint grid. The pin `(model, target gap, cap)` is **part of the task
definition**: it appears in the scenario card's `estimation.certificate` block, feeds
`EstimationTask.content_hash()`, and lands in the manifest. SUE-side estimation (pin =
Dial load at `sue_theta`) is deferred; this ADR covers UE tasks.

**Scored columns (the pair, never collapsed):**
- `obs_count_rmse` = `sqrt(mean over (period t, link a in S_obs) of (v̂_a − c_{t,a})²)` —
  fit to the very data the estimator saw. Poisson noise gives an irreducible floor
  (≈ `sqrt(mean flow)` per period), so the harness also reports
  `oracle_obs_count_rmse` — the same metric for `UE(Q_true)` against the same counts —
  as the achievable floor and the BO4Mob "Improvement% vs common baseline" anchor.
- `heldout_count_rmse` — same formula on a **disjoint held-out sensor set** `S_ho` with
  counts the harness draws from truth on the reserved `SOURCE_EVALUATION` stream
  (that stream's documented purpose). Never shown to the estimator. **This is the ranking
  column on every T2 task** — Hazelton's out-of-sample principle made mechanical.
  `oracle_heldout_count_rmse` reported alongside. A noise-free
  `heldout_flow_rmse` (v̂ vs true flows on `S_ho`) is a descriptive column.
- `od_rmse` = RMSE over **off-diagonal** cells of `(Q̂ − Q_true)`; `od_nrmse`
  (normalized by mean positive true off-diagonal entry); `total_demand_error` =
  `(ΣQ̂ − ΣQ_true)/ΣQ_true`, signed, off-diagonal sums (intrazonal demand never enters
  the network — same convention as `node_balance_residual`). OD columns are always
  *reported*, but rank nothing; on tasks whose identifiability report is negative
  (Decision 4) they are additionally flagged descriptive-only. Count-fit and OD-fit form
  a pair/tuple in every table and plot — the ARCHITECTURE "fit-vs-gap is a pair" rule
  transported to T2. A single scalar would let an estimator trade unidentifiable OD error
  for count fit invisibly, which is precisely the failure mode Hazelton documents.

**Censoring (mirrors `Evaluator.evaluate`):** wrong shape `(n_zones, n_zones)` → raise
`ValueError` (programming error in the wrapper); non-finite entries → censored; negative
entries below `−1e-9·max|Q̂|` → censored, else clipped to 0; positive demand between
zones with no connecting path → the pinned assignment's `RuntimeError` is caught →
censored. Censored checkpoints get `od_feasible = 0` and NaN metrics — never a score,
never a crash of the surrounding experiment. A **zero matrix is not censored**: unlike T1
(where zero flows would fake a perfect gap), a zero OD is a legitimate, terrible estimate
— it certifies with catastrophic count-fit, which is the honest outcome.

**Budget accounting.** The estimator's budget is expressed in **assignment-equivalents =
`sp_calls`** (existing convention: one batched AON sweep = one Dial load = 1). Inner
assignment runs charge their true `sp_calls` to the estimator's `BudgetCoords`;
`iterations` counts outer estimation iterations. Certification cost (a full pinned bfw
run per checkpoint, roughly 10²–10³ sp-call-equivalents on ladder rungs) is harness-side
and uncharged, per the ADR-001 precedent — but it is *not* per-iteration cheap, so
estimators are documented to emit O(10–20) checkpoints; the runner certifies every
emitted checkpoint (no silent thinning — thinning would corrupt progress curves).
Self-reported `obs_count_rmse` in `self_report` (estimators report fit to the
period-**mean** counts) is diffed against the harness `obs_mean_count_rmse` column —
the pinned-assignment fit to that same period-mean — as the honesty check (P1); diffing
against the per-period `obs_count_rmse` would flag every honest estimator by the
irreducible Poisson within-period floor. Estimators that carry an optional
prior-deviation penalty (e.g. `spsa`'s `prior_weight`) self-report the count-fit term
only, never the penalty, so the label stays an honest count RMSE.

## Decision 3 — Shipped estimators

All four classical estimators consume an **assignment-proportion matrix**
`P ∈ R^(n_links × n_od)` with `p^a_ij` = fraction of demand (i,j) using link a. Pinned
extraction (`estimation/_proportions.py`): run the inner assignment as **MSA over AON
trees**, accumulating `P = (1/K) Σ_k P_k` where `P_k` is the sparse 0/1 per-OD tree
incidence at iteration-k costs (per origin, one predecessor-tree walk — no path
enumeration). Because MSA's flows are exactly the same average, **`v = P·g` holds to
machine precision by construction**, and equilibrium route ties (at UE, all used routes
tie by definition — verified on Braess D=6, all three routes at 92) are averaged through
the MSA trajectory instead of being tie-broken to one arbitrary tree. Single-tree
proportions at equilibrium costs are demonstrably fragile (on Braess at prior D=4 the
misfit gradient even changes *sign* with the tie-break). Congested-case coupling is the
standard outer fixed point: assign current ĝ → extract P → estimate → repeat (Cascetta &
Postorino 2001). `K_inner` and the outer count are estimator factors, paid in sp_calls.

1. **`vzw-entropy` — Van Zuylen & Willumsen (1980).** Maximum-entropy matrix reproducing
   counts: Lagrangian form `T_ij = t_ij · Π_{a∈Â} X_a^{p^a_ij}` (prior `t` = task prior).
   Algorithm: cyclic multiplicative balancing — for each observed link a with modeled
   flow `v_a(T) = Σ p^a_ij T_ij > 0`, set `T_ij ← T_ij · (ĉ_a / v_a(T))^{p^a_ij}`.
   Counts entering as `ĉ_a` = period mean. The exponent `p^a_ij ≤ 1` damps each pass, so
   the balancing is a contraction: with consistent counts it converges geometrically to
   the count-reproducing matrix; with mutually inconsistent counts it converges to a
   compromise whose residual settles above tolerance rather than oscillating. Deterministic.
   Safeguards: skip sensors with `v_a = 0`; iterate to budget; flag
   `counts_consistent=False` keyed on that **converged** residual. Zero prior cells stay
   zero. Cost: outer_iters × K_inner sp_calls (proportions), balancing itself is free.
2. **`gls` — Cascetta (1984).** `ĝ = argmin_{g≥0} (g−ĝ_pr)ᵀW⁻¹(g−ĝ_pr) + (c̄−Pg)ᵀV⁻¹(c̄−Pg)`
   with `W = diag((cv_prior·ĝ_pr,ij)² + ε)` (prior quality from the task card) and
   `V = diag(max(c̄_a,1)/n_periods)` (Poisson variance of the period mean). Solved as
   bounded least squares on the whitened stacked system (`scipy.optimize.lsq_linear`,
   bounds `[0, ∞)`) — the nonnegativity Cascetta notes but drops for the closed form.
   Deterministic; unique for any sensor set because `W⁻¹ ≻ 0` — **the estimator that
   stays well-posed under non-identifiability**, which is why it is the default baseline.
   Safeguard: the outer fixed point keeps the **best self-obs-RMSE iterate** (seeded from
   the prior), so under noisy counts that push the fixed point across a regime boundary it
   can never return an outer iterate it measures as worse than its own starting point (same
   rationale as vzw's noisy-counts non-convergence). Cost: outer_iters × K_inner sp_calls.
3. **`spiess` — Spiess (1990, CRT-693).** Bilevel descent on
   `Z(g) = ½ Σ_{a∈Â} (v_a(g) − ĉ_a)²` under locally-constant proportions:
   `∂Z/∂g_ij = Σ_a p^a_ij (v_a − ĉ_a)` (misfit accumulated along used paths — computed
   from the same sparse P, one pass); multiplicative update
   `g_ij ← g_ij · (1 − λ ∇_ij)`, which preserves nonnegativity and never creates trips
   for zero-prior pairs (Spiess's stated design feature — the prior's *structure* is the
   regularizer). Step: with `w_a = Σ_ij p^a_ij g_ij ∇_ij`, the linearized-optimal
   `λ* = Σ_a w_a(v_a−ĉ_a) / Σ_a w_a²`, capped at `1/max ∇_ij` for feasibility. Z is
   nonconvex through the equilibrium map, so the Armijo safeguard is **retrospective**:
   after each outer re-assignment, if Z under the fresh proportions rose above the last
   accepted iterate's, the step overshot the equilibrium map — revert to that iterate and
   halve a persistent damping factor (≤ `max_halvings` times) before re-stepping, at no
   extra assignment (still 1 inner assignment, K_inner sp_calls, per outer iteration).
   The descent also keeps the **best self-obs-RMSE iterate** (seeded from the prior), so it
   never returns something it measures as worse than its start. Deterministic.
4. **`spsa` — Spall (1992).** Black-box calibration baseline; the only estimator that
   never sees P — it treats assignment as an oracle, exactly how a simulator or a neural
   surrogate would be calibrated (Lu et al. 2015 for OD-SPSA practice). Parametrize
   `u = log g` on the prior's positive support (scale-free, positivity by construction).
   Per iteration: Rademacher `Δ` from `rng.generator(source=0)`; evaluate
   `L(exp(u ± c_k Δ))` — **two inner assignments** — with
   `L = Σ_{a∈Â}(v_a − c̄_a)²/|Â| (+ optional prior-deviation weight, factor, default 0)`;
   `ĝrad = (L⁺−L⁻)/(2c_k) · Δ` (Δ_i ∈ {±1} so Δ⁻¹=Δ); `u ← u − a_k ĝrad`.
   Gains `a_k = a/(k+1+A)^0.602`, `c_k = c/(k+1)^0.101` (Spall's practical exponents; the
   1992 paper gives the asymptotic conditions), `A` ≈ 10% of budgeted iterations, `a`
   calibrated from a target initial step (factors with defaults). Seeded, not
   deterministic (`deterministic=False, seedable=True`) → macroreplicated. Safeguards:
   clip `|a_k ĝrad|` per component (blow-up guard); best-iterate tracking.
   Cost: 2 × K_inner sp_calls per iteration.
5. **`prior` — identity baseline.** Emits the prior unchanged (iterations=1, sp_calls=0).
   Every leaderboard needs the do-nothing anchor (BO4Mob Improvement% convention).

## Decision 4 — Identifiability: what gets reported per (sensor set, task)

The existing `observe.distinct_nonzero_columns` is Hazelton's Prop. 1 condition on a
(monitored links × routes) incidence — exact for *route-flow* means from repeated counts
in the linear (fixed-routes) model. T2 operationalizes it at the OD level: the harness
builds the **truth-side proportion matrix** `P*` (Decision-3 extraction at the true
equilibrium — harness-only; the estimator never sees it) and reports, per task:

- `hazelton_condition` = `distinct_nonzero_columns(P*[S_obs, active_pairs])` — no zero
  columns (every active pair touches a sensor) and no duplicated columns (no two pairs
  indistinguishable). Necessary for identifiability; exact in Hazelton's linear setting.
- `linear_identifiable` = `rank(P*[S_obs, active]) == n_active` — mean-count
  identifiability under fixed proportions. Computed densely only when `n_zones ≤ 100`
  (Sioux Falls: 76×528 is trivial; big rungs get `hazelton_condition` + a
  `rank_not_computed` flag).
- Diagnostics: `n_active_pairs`, `n_unseen_pairs` (zero columns), `n_confounded_pairs`
  (members of duplicate-column classes).

The report is computed once per task, stored in `Dataset.meta`, echoed in
`EstimationTask.identifiability` (it is *public* — sensor-design quality is knowable in
practice and per ARCHITECTURE P3(iii) deliberately-violating configurations are included
and flagged), written to the manifest, and drives scoring: when `linear_identifiable`
is false, the certificate still computes `od_*` columns but flags `od_identifiable=0`,
and the task ranks on `heldout_count_rmse` (+ the estimator's declared
regularization-fit, i.e. prior-deviation, as a descriptive column) — estimators are
never ranked on recovering an OD the data provably does not determine.

**Documented caveat (with a concrete counterexample):** the report is a **linearized /
local** statement. The true demand→flow map is nonlinear through congestion; on Braess
with the single sensor {3→4}, `hazelton_condition` is true (the one active pair loads
the bypass with proportion 1/3 at truth), yet demand is globally non-identifiable:
`v_{3→4}(D) = D` for `D ≤ 40/11`, then `(80−9D)/13` — non-monotone, so `D=2` and `D=6`
both produce the exact count 2 (verified with bfw at RG 1e-10). The certificate's
held-out sensors are the mechanical defense: link 1→3 carries 2 vs 4 under the two
demands. This example ships as both a test and the ADR's warning against reading the
flag as a global guarantee.

## Decision 5 — Prior generation, RNG schema, and macroreplication

- **`StalePriorOD` data level** (the P3 table row, now implemented):
  `prior_ij = truth_ij · Gamma(shape=1/cv², scale=cv²)` i.i.d. per positive cell
  (mean 1, coefficient of variation `cv`; zero cells stay zero — surveys know which
  pairs exist; the support leak is stated on the card). `cv` is a task dial, hashed.
- **New reserved stream `SOURCE_PRIOR = 1_000_003`** in `core/rng.py` (prior draws must
  be independent of count draws); observed counts use `SOURCE_OBSERVATION`
  (replication=0), held-out counts use `SOURCE_EVALUATION` — its documented purpose.
  Estimator-internal randomness uses model source ids (0…), as `CallableModel` does.
- **Macroreplication semantics differ from T1:** the dataset itself is stochastic, so
  `reps = macroreps` even for deterministic estimators when `noise != "none"` — each
  macrorep draws fresh counts + prior + held-out counts and re-runs everything;
  deterministic estimators on `noise="none"` collapse to 1 rep. Bootstrap CIs over
  macroreps on `SOURCE_BOOTSTRAP` (existing schema, P8).

## Decision 6 — Analytic anchors (all numbers machine-verified against repo code)

1. **Braess, full sensors, noiseless** (truth D=6, UE flows (4,2,2,2,4), link order
   1→3,1→4,3→4,3→2,4→2): the only routable pair is 1→2 (node 2 has no outgoing links)
   and v(D) is injective at full coverage — identifiability report positive. `gls`,
   `spiess`, and `vzw-entropy` from a prior in the **global basin** (D=5.5) recover
   `|ĝ−6| < 1e-3` (the prior must lie in the convergence basin). Prior D=4 sits below the
   frozen-proportion barrier near 40/11≈3.636: its linearized gradient points *down*
   toward the spurious stationary point D=10/3 (bypass-saturated, outer links 1→4 and
   3→2 carry no flow), even though the true equilibrium-map objective decreases
   monotonically toward D=6. So the classical local methods do **not** recover D=6 from
   D=4 — this ships as the executable caveat
   `test_braess_prior4_safeguard_refuses_dominated_trap`, where the best-self-obs-RMSE
   safeguard keeps the estimate at its (better-fitting) D=4 start rather than the
   dominated 10/3 trap.
2. **Braess, sensor {3→4} only, noiseless count = 2:** both Q̂=2 (flows (2,0,2,0,2)) and
   Q̂=6 certify `obs_count_rmse ≈ 0` (emitted via `CallableEstimator` constants); the
   held-out sensor 1→3 discriminates: `heldout_flow_rmse` 2 vs 0. `hazelton_condition`
   true yet globally non-identifiable — the Decision-4 caveat as an executable test.
3. **Two-route UE anchor** (`two_route_scenario(sue_theta=None)`, truth D=4, UE flows
   (2.5, 2.5, 1.5, 1.5), P-proportions p_A=0.625, p_B=0.375): `vzw-entropy` from uniform
   prior t=1 with sensor {link 0}: the damped exponent form gives **one pass = (c/p)^p =
   4^0.625 ≈ 2.3784** (recomputed in-test), converging geometrically to the fixed point
   T = c/p = **4.0** over passes (assert to 1e-9). With inconsistent counts (c₀=3, c₂=1)
   the two damped single-link updates no longer oscillate — they converge to the
   log-space compromise fixed point ≈ **3.5992** (recomputed from its closed form) whose
   residual settles above tolerance, so `counts_consistent=False`.
4. **GLS closed form** (two-route, sensor {0}, ĝ_pr=3, W=1, V=1, p=0.625, c̄=2.5):
   `g* = (3 + 0.625·2.5)/(1 + 0.625²) = 3.2808988764…`; code must match to 1e-10; the
   test recomputes the closed form rather than trusting digits (house style).
5. **SPSA seeded smoke** (two-route, full sensors, noiseless, 200 sp-call budget):
   `|ĝ−4| < 0.2` at the pinned seed; same (root_seed, macrorep) → byte-identical trace;
   different macrorep differs (P8 regression).
6. **Sioux Falls under-determination:** 76 links, 528 positive off-diagonal pairs
   (verified from the TNTP table, total demand 360600) → `rank(P*) ≤ 76 < 528`, so
   `linear_identifiable=False` at *any* coverage — the flagship "OD-fit is descriptive,
   rank by held-out counts" task, and the reason the pair rule exists.

## Decision 7 — Integration

```
src/tabench/estimation/          # new vertical slice (contract + algorithms)
    __init__.py  base.py         # ODEstimator, EstimationTask, ODTrace/ODState/ODResultBundle,
                                 # ESTIMATOR_REGISTRY, register_estimator, CallableEstimator, prior
    _proportions.py              # MSA-averaged per-OD tree proportions (uses models/_paths)
    entropy.py  gls.py  spiess.py  spsa.py
src/tabench/metrics/estimation.py   # ODCertifier (pinned bfw + pair metrics + censoring)
src/tabench/experiments/runner.py   # + run_estimation_experiment (parallel to run_experiment)
src/tabench/observe/levels.py       # + StalePriorOD; LinkCounts unchanged
src/tabench/core/rng.py             # + SOURCE_PRIOR
src/tabench/core/capabilities.py    # + "estimation" in PARADIGMS
```

Layering stays acyclic: `metrics → models/estimation → core`; `estimation` imports
`models._paths`/`frank_wolfe` and `observe`, never `experiments`. (`models/estimators/`
was rejected: a different ABC under `models/` would invite registering estimators in
`MODEL_REGISTRY`; a sibling package keeps the CLI listing and the contract honest.)

**Scenario card (T2 block; every field task-defining and hashed):**

```yaml
scenario: siouxfalls
tasks: [t1_equilibrium, t2_estimation]
estimation:
  data_level: link_counts
  sensors: {kind: random, coverage: 0.3}      # or {kind: explicit, links: [0, 2, 4]}
  n_periods: 15
  noise: poisson
  heldout: {kind: random, coverage: 0.1, n_periods: 5}   # disjoint from sensors
  prior: {kind: stale, cv: 0.30}
  certificate: {assignment: bfw, target_relative_gap: 1.0e-6, max_iterations: 5000}
  identifiability: reported   # harness computes; cards may pin the expected flag
budgets: {sp_calls: 2000}
```

CLI: `tabench run --config <card>` dispatches on `t2_estimation` in `tasks:` to the
estimation runner; `--models prior,gls,vzw-entropy,spiess,spsa` resolves against
`ESTIMATOR_REGISTRY`; an explicit `--models` list is always taken literally (an explicit
T1 name like `aon` on a T2 card errors cleanly, exit 2), while omitting it falls back to
the per-track default; `tabench list` grows an "Estimators" section. New CSV fields:
`task_hash, od_feasible, obs_count_rmse, obs_mean_count_rmse, oracle_obs_count_rmse,
heldout_count_rmse, oracle_heldout_count_rmse, heldout_flow_rmse, od_rmse, od_nrmse,
total_demand_error, od_identifiable, certificate_gap, certificate_converged,
self_obs_count_rmse`.
Manifest gains the estimation block, the certificate pin, the identifiability report,
and `SOURCE_PRIOR` under reserved sources.

**This sprint:** everything above + T2 cards for braess/tworoute/siouxfalls + the
Decision-6 tests. **Deferred:** distributional T2 metrics (held-out count log-likelihood,
CRPS, coverage) until a distribution-emitting estimator (Hazelton-style sampler, v1)
lands — the CSV schema reserves the columns; SUE-pinned certificates; `Trajectories`
level; logit (Dial-STOCH) multipath proportions as an alternative P; computational-graph
estimators (Wu et al. 2018; Ma, Pi & Qian 2020; Patwary et al. 2023) via `WhiteBoxMixin`;
sensor-placement tasks (Yang & Zhou 1998); Antoniou et al. (2016)-style cross-platform
protocol comparison.

## Consequences

- Anything that emits an OD matrix — a 1980 balancing loop or a 2026 neural inverse —
  receives the same externally certified (held-out count-fit, OD-fit) pair from one
  pinned assignment. P1 extends to T2 with no new trust assumptions.
- Non-identifiable tasks are first-class and flagged, never dropped; estimators are
  never ranked on recovering what the data cannot determine (P3, Hazelton).
- Certification cost is one pinned bfw run per checkpoint — materially heavier than T1's
  single AON; controlled by sparse checkpointing conventions, not thinning.
- The MSA-averaged proportion pin trades canonical single-tree fragility for a
  consistency identity (`v = P g`), at the cost of one documented deviation from
  the papers' unstated "given" proportions — every shipped estimator uses the same
  helper, so comparisons remain internally fair.
- `Scenario` is untouched: no hash changes, no new fields; the T2 instance pin lives in
  `EstimationTask.content_hash()`. Golden Braess hash `cf00f411…` is unaffected.

## References

- Van Zuylen, H.J. & Willumsen, L.G. (1980). The most likely trip matrix estimated from
  traffic counts. *Transportation Research Part B* 14(3), 281–293.
- Cascetta, E. (1984). Estimation of trip matrices from traffic counts and survey data:
  A generalized least squares estimator. *Transportation Research Part B* 18(4–5), 289–299.
- Spiess, H. (1990). A gradient approach for the O-D matrix adjustment problem.
  Publication CRT-693, Centre de Recherche sur les Transports, Université de Montréal.
- Spall, J.C. (1992). Multivariate stochastic approximation using a simultaneous
  perturbation gradient approximation. *IEEE Trans. Automatic Control* 37(3), 332–341.
- Hazelton, M.L. (2015). Network tomography for integer-valued traffic.
  *Annals of Applied Statistics* 9(1), 474–506.
- Cascetta, E. & Nguyen, S. (1988). A unified framework for estimating or updating
  origin/destination matrices from traffic counts. *Transportation Research Part B* 22(6), 437–455.
- Cascetta, E. & Postorino, M.N. (2001). Fixed point approaches to the estimation of O/D
  matrices from traffic counts on congested networks. *Transportation Science* 35(2), 134–147.
- Yang, H., Sasaki, T., Iida, Y. & Asakura, Y. (1992). Estimation of origin-destination
  matrices from link traffic counts on congested networks. *Transportation Research Part B* 26(6), 417–434.
- Yang, H. & Zhou, J. (1998). Optimal traffic counting locations for origin-destination
  matrix estimation. *Transportation Research Part B* 32(2), 109–126.
- Lu, L., Xu, Y., Antoniou, C. & Ben-Akiva, M. (2015). An enhanced SPSA algorithm for
  the calibration of DTA models. *Transportation Research Part C* 51, 149–166.
- Antoniou, C. et al. (2016). Towards a generic benchmarking platform for O-D flows
  estimation/updating algorithms. *Transportation Research Part C* 66, 79–98.
- Boyce, D., Ralevic-Dekic, B. & Bar-Gera, H. (2004). Convergence of traffic assignments:
  How much is enough? *J. Transportation Engineering* 130(1), 49–55.
