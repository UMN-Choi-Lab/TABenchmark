# TABenchmark Architecture

**A shared benchmark for 50 years of traffic assignment models.**

This document is the normative design reference. It synthesizes three independently
developed design studies (a SimOpt-style scenario×model matrix, an observability-first
statistical design, and a contract/plugin-first design — see `docs/design/`) informed by
three key references:

- **SimOpt** (Eckman, Henderson & Shashaani): problem–solver testbed machinery —
  macro/post-replication experiments, fixed random-number stream schemas, progress curves
  and solvability profiles.
- **BO4Mob** (Ryu, Kwon, Choi, Deshwal, Kang & Osorio, NeurIPS 2025 D&B): the lab's
  house conventions — scenario ladders, declarative configs, strategy registries, seeded
  subprocess adapters, budget tables, datasheets.
- **Hazelton (2015), AOAS 9(1):474–506**: statistical inference of route flows from link
  counts — identifiability conditions, why observation processes must be first-class,
  and how to score probabilistic outputs.

---

## 1. Design principles

### P1 — The certificate principle
**The harness — never the model — computes every scored metric.** For static tasks with
known link cost functions, the relative equilibrium gap is a property of
`(link_flows, scenario)`, not of the algorithm: given emitted flows, the harness runs
all-or-nothing shortest paths at the flow-induced costs and computes the gap externally.
This single decision is what lets a 1975 Frank–Wolfe implementation and a 2025 GNN
surrogate share one leaderboard: *anything that emits feasible link flows gets an
externally certified gap.* Model self-reports are recorded as provenance and diffed
against harness values as an honesty check, but never scored. This is SimOpt's
"post-replicate, never trust in-run estimates" principle transported to the
deterministic setting.

### P2 — Scenarios are data, not code
A `Scenario` is a frozen, content-hashed, declarative object: network arrays + demand +
cost configuration + units metadata + known-defect notes. `scenario_hash =
SHA-256(canonical serialization)` pins every result to the exact instance evaluated; a
silently edited network can never masquerade as the benchmark scenario. Scenarios carry
**no executable code**, following BO4Mob's data-directory-plus-JSON-config convention.

### P3 — Observation processes are first-class (the data-levels axis)
Models from 1956–2026 disagree less about networks than about *what data they see*.
Ground truth is stored once at the finest granularity available (equilibrium link flows
from a reference solver at tight gap; for synthetic estimation scenarios, realized
integer route flows per Hazelton). Every data level is a seeded, versioned projection:

| Data level | Parameters (dials) |
|---|---|
| `FullOD` | — |
| `NoisyOD` | coefficient of variation |
| `LinkCounts` | sensor subset (mask), noise model, number of observation periods N |
| `DayToDayCounts` | sensor subset, N periods, `population_scale`, day-to-day persistence `rho` (Davis–Nihan large-population VAR(1) count series, ADR-012) |
| `Trajectories` | penetration rate, sampling noise |
| `StalePriorOD` | prior age/quality (Gamma pseudo-counts) |

Per Hazelton's Proposition 1, mean flow parameters are identifiable from a *sequence* of
link-count vectors when the monitored incidence matrix has distinct nonzero columns —
but not from a single snapshot (feasible polytopes exceed 10¹⁴ points on toy networks).
Therefore: (i) per-period counts are distributed, never day-averages; (ii) every
(network, sensor set) configuration reports its identifiability condition; (iii)
configurations that deliberately violate identifiability are included and flagged;
(iv) distributional outputs are scored against distributions (coverage, likelihood),
never against a fictional unique inversion.

### P4 — One minimal model contract, capabilities-declared
Everything a model must do is one method plus one declaration:

```python
class TrafficAssignmentModel(ABC):
    capabilities: Capabilities            # declared, machine-checkable
    factors: dict[str, FactorSpec]        # typed hyperparameters with defaults

    def solve(self, scenario: Scenario, budget: Budget,
              rng: RngBundle, trace: Trace) -> ResultBundle: ...
```

`Capabilities` declares paradigm (`static_ue | static_ue_elastic | static_ue_combined |
static_br_ue | static_sc_ue | static_ue_vi | static_ue_multiclass | static_so | sue | dta | day_to_day |
learned | heuristic | estimation`), determinism, required inputs
(`od_matrix`, `link_counts`, …),
emitted outputs (`link_flows`, `class_link_flows`, `path_flows`, `flow_distribution`, `od_estimate`),
`provides_gap`, `seedable`, and `trained_on` lineage (learned models). The harness
auto-filters the scenario×model×task matrix by compatibility — a counts-only task
refuses a model that requires full OD; a gradient-requiring solver never meets a
subprocess black box.

White-box status means the model's internals match the scenario's *declared* cost
functions (`Network.link_cost` / `link_cost_integral`), which is what makes external
gap certification possible for everyone; a future `WhiteBoxMixin` will additionally
expose Jacobians for derivative-based solvers. Black-box models plug in through adapters:
`CallableAdapter` (any Python callable, e.g. a torch model), `SubprocessAdapter`
(external engines — DTALite, MATSim, SUMO — via file I/O and explicit `--seed`
passthrough, BO4Mob's `sumo_runner` pattern), and later `DockerAdapter` (digest-pinned
engine images).

### P5 — Two evaluation tracks, honestly separated
- **Deterministic track** (convex UE/SO programs): one macroreplication, no
  post-replications; metric = certified relative gap versus budget, reported as
  progress curves and Moré–Wild-style data profiles; regression against best-known
  objectives.
- **Stochastic track** (simulation-based SUE such as probit, day-to-day, simulators, ML
  surrogates): M macroreplications with a fixed random-stream schema, N independent
  post-evaluations for unbiased estimates at recommended solutions, bootstrap confidence
  intervals (never parametric), mean/quantile progress curves and α-solve-time
  solvability profiles. (Logit SUE via Dial's closed-form loading is deterministic —
  "stochastic" there refers to traveler perception — and runs on the deterministic
  track; see docs/design/adr-001.)

Models that cannot certify equilibrium are **first-class, not excluded**: they appear
with externally computed gap-at-budget where feasible-flow output permits, as censored
entries in solvability profiles, and in a separate "no certificate" leaderboard column.
No imputed zeros; no single ranking across tracks. The progress curves, α-solve-time
solvability profiles, Moré–Wild data profiles, and functional bootstrap bands are shipped
as `tabench.experiments.profiles` (pure post-hoc arithmetic over the certified rows, with
censored entries kept in every solvability denominator; [ADR-032](design/adr-032-simopt-profiles.md)).

### P6 — Budgets count work, not hardware
Primary budget coordinates are hardware-free: iterations, shortest-path calls, scenario
(engine) evaluations. Every checkpoint records *all* coordinates plus wall-clock, so any
curve can be re-sliced post hoc. Wall-clock is recorded, never ranked on directly;
normalization by a per-machine calibration constant (timing a standard all-or-nothing
pass) is planned for the stochastic track. Training cost of learned models is reported
as a separate amortized column.

### P7 — Fairness is enforced by the harness, not by convention
- Learned models must register a **training-data card**; the harness refuses evaluation
  when `trained_on` lineage intersects the evaluation scenario's family (scenario
  hashes + a held-out demand/topology perturbation generator mean there is no fixed
  test set to memorize).
- A **feasibility audit** (flow conservation, OD totals) runs before any external gap is
  granted; infeasible flows are flagged, never silently scored.
- Every run emits `manifest.json`: scenario hash, package and engine versions, seeds and
  stream offsets, environment, git commit. The leaderboard rejects incomplete manifests.
- Warm starts are declared factors; all models receive identical budgets and, where
  applicable, identical shared initial candidates (BO4Mob convention).

### P8 — Reproducible randomness by construction
A single counter-based RNG root with a fixed spawn-key schema
`(macroreplication m, declared randomness source i, replication r)` — the SimOpt
MRG32k3a stream/substream design realized with NumPy `SeedSequence`/`Philox` spawn keys.
Common random numbers are switchable per layer (across models on a scenario: default on;
across macroreps: default off). Dedicated streams exist for observation generation,
post-evaluation, and bootstrap. Stochastic models declare their number of randomness
sources; unseedable external engines must declare `seedable=False` and are
macroreplicated more heavily, and are labeled as such.

### P9 — Data are fetched, never vendored
TNTP networks (github.com/bstabler/TransportationNetworks) are donated for academic
research without an OSI license. TABenchmark therefore ships a checksummed
download-on-demand fetcher with a local cache and auto-generated citation strings, and
never commits network data. Units are per-network metadata (Sioux Falls times are 0.01 h
and demand is 0.1× daily; Anaheim lengths are feet; …), never global assumptions. Known
data defects (Austin duplicate links, Chicago-Regional ramp coding, Chicago-Sketch
under-congestion) live in a machine-readable defect registry. The six published
best-known UE solutions (Sioux Falls, Barcelona, Winnipeg, Chicago-Sketch,
Chicago-Regional, Anaheim) are regression oracles with recorded provenance. Because
upstream-quoted objective values use varying unit/scaling conventions (the upstream
README quotes Sioux Falls as 42.31335287107440 while the Beckmann objective in native
TNTP units — what `tabench` computes — is **4231335.28710744**), TABenchmark never
hardcodes quoted objectives: the oracle objective is always *recomputed from the
best-known flows* with the package's own Beckmann implementation.

The same discipline governs the **cross-domain axis** (Xu et al. 2024, 17 real US-city
instances; adr-033): only the per-city AequilibraE trio is fetched, by HTTP byte-range
extraction of exactly the needed members of a 276 MB CC-BY figshare zip (never the whole
archive), on a registry deliberately separate from the CI-prefetched TNTP one. Its
wrong-centroid defect (demand injected at node ids `1..Z`, not the tract centroids) is a
first-class known-defect entry, its 3 unbuildable cities are named exclusions, and its
published flows are labelled a *loose* reference (own gap ~1e-3), never a best-known
oracle — the honest tier separation P9 is built to keep.

The **BO4Mob scenario family** (Ryu et al. 2025, adr-034) is the same P9 discipline applied
to the lab's OWN NeurIPS-2025 benchmark of San Jose freeway OD-estimation instances (`data/bo4mob.py`,
a separate commit-pinned per-file-SHA-256 registry never in the CI-prefetched `REGISTRY`):
stage 1 ships data availability + a guarded mesoscopic-SUMO pipeline-liveness smoke, under a
dual-benchmark honesty contract that hosts the instances as *scenarios/data only* — never as
validation of TABench methods, and never claiming the paper's numbers (a measured SUMO
1.12→1.27.1 `edgeData` drift makes them non-reproducible here). BO4Mob keys are NOT
`load_scenario` scenarios (a meso net with no BPR network and no true OD is data, not a
`Scenario`); `5fullRegion` (74 MB, ~11 h/eval) is metadata-only and refuses to fetch.

---

## 2. Object model

```
Scenario  = Network + Demand + CostConfig (+ ReferenceSolution?)   [frozen, hashed]
DataLevel = observation process: (GroundTruth, rng) -> Dataset      [seeded projection]
Task      = Scenario × DataLevel × objective + metric set           [T1 | T2 | T3]
Model     = TrafficAssignmentModel (Capabilities, FactorSpecs, solve())
Adapter   = CallableAdapter | SubprocessAdapter | DockerAdapter     [same ABC]
Trace     = checkpoint stream: (BudgetCoords, link_flows, self_report)
Evaluator = harness-side scoring: (Scenario, FlowState) -> Metrics  [model-blind]
Experiment= (Task × Model) grid runner: macroreps -> records -> profiles
```

### Tasks
- **T1 Equilibrium** (full specification): compute UE / SO / SUE(θ). Scored by certified
  relative gap (and SUE fixed-point residual for SUE tasks), Beckmann objective, link-flow
  error vs best-known, all versus budget.
- **T2 Estimation / calibration** (partial observations): recover demand and/or flows from
  a `DataLevel` output. Point metrics (RMSE/NRMSE vs truth) reported **separately from**
  distributional metrics (held-out count log-likelihood, CRPS, empirical coverage of 95%
  intervals) — overdispersion misspecification corrupts calibration before point accuracy
  (Hazelton).
- **T3 Prediction under intervention**: fit on pre-intervention data; predict flows after
  a capacity cut / demand shift / link closure applied to the truth generator. The arena
  where a classical model's structural prior competes fairly with a learned model's
  flexibility.

### Metric definitions (single source of truth, `tabench.metrics`)
- `TSTT(v) = Σ_a v_a · t_a(v_a)`; `SPTT(v) = Σ_a y_a · t_a(v_a)` with `y` the
  all-or-nothing assignment at costs `t(v)`.
- **Relative gap** `RG(v) = (TSTT − SPTT) / TSTT` (documented choice; the literature has
  several conventions — see Boyce, Ralevic-Dekic & Bar-Gera 2004).
- **Average excess cost** `AEC(v) = (TSTT − SPTT) / total demand` (the convention used by
  the TransportationNetworks best-known solutions).
- **Beckmann objective** `B(v) = Σ_a ∫₀^{v_a} t_a(s) ds`, closed-form for BPR.
- **SUE fixed-point residual** (scenarios with `sue_theta` set):
  `‖v − L(t(v), θ)‖₁ / total demand`, with `L` the pinned Dial-STOCH loading map —
  misallocated link-traversals per traveler, zero exactly at the logit SUE. Fisk's
  objective cannot serve here (its entropy term needs path flows, which link flows do
  not determine). On SUE tasks this ranks; UE columns remain descriptive. See
  docs/design/adr-001.
- Flow errors vs oracle: RMSE, NRMSE (normalized by mean oracle flow).
- Distributional: log-likelihood of held-out per-period counts (MC-estimated where
  exact evaluation is infeasible), CRPS, 95%-interval empirical coverage.
- Fit-vs-gap is reported as a **pair/curve, never collapsed to one scalar**.
- **Censoring:** flows that fail the demand-aware feasibility audit (non-finite or
  negative flows, unconserved intersections, zone flows inconsistent with OD
  productions/attractions, or negative excess cost) receive `feasible = 0` and NaN gap
  metrics — never a score, and never a crash of the surrounding experiment.

---

## 3. Repository layout

```
TABenchmark/
├── src/tabench/
│   ├── core/              # scenario.py (incl. content hashing), capabilities.py,
│   │                      # factors.py, budget.py, results.py, rng.py
│   ├── data/              # tntp.py (defensive parser), fetcher.py (checksums+citations),
│   │                      # registry.py (networks + units metadata + defects), builtin.py
│   ├── models/            # base.py, aon.py, msa.py, frank_wolfe.py (FW/CFW/BFW),
│   │   │                  # gradient_projection.py, oba.py (Bar-Gera 2002 origin-based),
│   │   │                  # algb.py (Dial 2006 bush), tapas.py (Bar-Gera 2010 PAS),
│   │   │                  # _bush.py (shared bush machinery), so.py (marginal-cost SO),
│   │   │                  # elastic.py (elastic-demand FW, Gartner excess-demand transform),
│   │   │                  # evans.py (Evans 1976 combined distribution+assignment),
│   │   │                  # dtd_swap.py (Smith 1984 route-swap day-to-day dynamics),
│   │   │                  # dtd_swap_sue.py (Smith-Watling 2016 logit-SUE route-swap day-to-day dynamics),
│   │   │                  # dtd_link.py (He-Guo-Liu 2010 link-based day-to-day dynamics),
│   │   │                  # dtd_friesz.py (Friesz 1994 route-space projected-gradient day-to-day PDS),
│   │   │                  # dtd_horowitz.py (Horowitz 1984 cost-smoothing day-to-day logit-SUE),
│   │   │                  # dtd_stochastic.py (Cascetta 1989 finite-population stochastic-process day-to-day),
│   │   │                  # dtd_unifying.py (Cantarella-Cascetta 1995 unifying cost-learning + choice-inertia day-to-day, UE/SUE mode gate),
│   │   │                  # br_ue.py (Mahmassani-Chang 1987 boundedly-rational UE),
│   │   │                  # sc_tap.py (Larsson-Patriksson 1995 side-constrained UE),
│   │   │                  # learned.py (first learned/black-box surrogate),
│   │   │                  # implicit_ue.py (Liu et al. 2023 implicit-NN UE; first torch model, optional [torch] extra),
│   │   │                  # het_gnn.py (Liu & Meidani 2024 heterogeneous-GNN UE; second torch model, same [torch] extra),
│   │   │                  # sue_logit.py, sue_probit.py,
│   │   │                  # _paths.py, _stoch.py (Dial map), _probit.py (MC map)
│   │   └── adapters/      # callable_adapter.py; sumo_marouter.py + _sumo_io.py
│   │   │                  #   (SUMO marouter external simulator, first Phase-4 adapter,
│   │   │                  #   optional [sumo] extra; adr-027); dtalite_tap.py
│   │   │                  #   (DTALite static FW assignment, second external engine,
│   │   │                  #   identity BPR map, optional [dtalite] extra; adr-029)
│   ├── observe/           # data levels + identifiability checks
│   ├── estimation/        # T2 OD-estimation track (base, entropy/gls/spiess/spsa,
│   │                      # yang1992/dn_kalman; spsa_sumo.py — spsa-sumo, the first
│   │                      # GUARDED estimator: SPSA in the marouter loop, optional [sumo]
│   │                      # extra, adr-028; within-day dynamic: dynamic_base.py,
│   │                      # _dynamic_map.py, cascetta1993.py — od-dynamic-sim/seq, ADR-023)
│   ├── dnl/               # Phase-2 dynamic-network-loading foundation (ADR-010): grid.py,
│   │                      # fd.py (fundamental diagram), demand.py, scenario.py (DynamicScenario
│   │                      # + domain-separated hash), link.py + node.py (S/R interfaces),
│   │                      # loader.py, output.py (DNLOutput P1 artifact), _reference.py
│   │                      # (test-only point queue), ctm.py (Daganzo CTM LinkModel, ADR-015),
│   │                      # ltm.py (Yperman LTM LinkModel, ADR-016),
│   │                      # node.py TampereNode (generic merge/diverge, ADR-017),
│   │                      # godunov.py + fd.GreenshieldsFD (general-FD Godunov, ADR-018),
│   │                      # builtin.py — shared by ctm/ltm/godunov (newell/ is a
│   │                      # separate state-estimation module, ADR-024)
│   ├── transit/           # Transit optimal strategies (ADR-014): network.py (TransitNetwork
│   │                      # directed multigraph + TransitScenario, domain-separated hash),
│   │                      # strategy.py (Spiess & Florian 1989 two-pass solver), builtin.py
│   ├── bottleneck/        # Departure-time equilibria: Vickrey (1969) single bottleneck
│   │                      # (ADR-019: scenario.py, solve.py — closed-form UE/SO + emitted
│   │                      # BottleneckSchedule) and Friesz et al. (1993) SRDC dynamic user
│   │                      # equilibrium (ADR-022: due.py — DUEScenario on parallel Vickrey
│   │                      # routes, closed form + emitted DUEProfile), builtin.py
│   ├── newell/            # Three-detector interior reconstruction — the first
│   │                      # traffic-state-estimation task (ADR-024): scenario.py
│   │                      # (ThreeDetectorScenario, domain-separated hash + truth
│   │                      # recipe regenerated via LTM), observe.py (seeded detector
│   │                      # projection), solve.py (newell-min / newell-min-isotonic
│   │                      # + emitted ThreeDetectorField), builtin.py
│   ├── dta/               # Analytical DTA: Merchant & Nemhauser (1978) exit-function
│   │                      # SO-DTA (ADR-020: scenario.py, solve.py — canonical
│   │                      # Carey-relaxed LP + emitted DTATrajectory w/ duals) and
│   │                      # Ziliaskopoulos (2000) LP SO-DTA on CTM cells (ADR-021:
│   │                      # cells.py — CellSODTAScenario w/ finite storage/spillback,
│   │                      # cell LP + CellTrajectory w/ duals), builtin.py
│   ├── tdta/              # Time-dependent SO/UE route choice — Peeta & Mahmassani
│   │                      # (1995), the first iterative simulation-based TD route
│   │                      # equilibrium (ADR-031): scenario.py (TDTAScenario — enumerated
│   │                      # per-OD path set + kernel, interior-diverge-free, domain-
│   │                      # separated hash + SO cell-LP derivation), loader.py (per-path
│   │                      # first-link injection over the dnl S/R loop + grid extension),
│   │                      # artifact.py (TDPathFlows, decisions only), solve.py (MSA
│   │                      # reference solvers, non-certified), builtin.py
│   ├── metrics/           # gaps.py, flows.py, so.py, estimation.py,
│   │                      # estimation_dynamic.py (bfw-free exact within-day certifier,
│   │                      # ADR-023), dnl_gaps.py (DNL P1
│   │                      # certificates C0–C8, ADR-010), transit_gaps.py (TransitEvaluator,
│   │                      # ADR-014), bottleneck_gaps.py (BottleneckEvaluator, ADR-019),
│   │                      # dta_gaps.py (SODTAEvaluator ADR-020 + CellSODTAEvaluator
│   │                      # ADR-021), due_gaps.py (DUEEvaluator, ADR-022),
│   │                      # newell_gaps.py (ThreeDetectorEvaluator, ADR-024),
│   │                      # tdta_gaps.py (TDTAEvaluator — TD-UE route-swap residual +
│   │                      # TD-SO LP-bound gap, ADR-031)
│   │                      # (planned: distributional.py)
│   ├── experiments/       # runner.py incl. manifests, bootstrap.py, profiles.py (adr-032)
│   └── cli.py             # tabench fetch | list | run (planned: validate)
├── scenarios/             # declarative YAML scenario cards (ladder: 0braess, 1siouxfalls, …)
├── demos/                 # demo_quickstart.py (planned ladder: scenario/model/experiment)
├── tools/                 # generate_references.py (regenerates REFERENCES.md/ROADMAP.md)
├── tests/                 # unit + analytic (Braess) + regression (Sioux Falls oracle)
└── docs/                  # this file, REFERENCES.md, references.bib, ROADMAP.md, design/
```
(`conformance/` — the contract test suite behind a future `tabench validate` — is
planned for v0.x.)

Scenario ladder (BO4Mob convention — strictly increasing scale):
`0braess` (hand-checkable analytic UE) / `0tworoute-sue` (analytic logit-SUE) →
`1siouxfalls` (24z/76l) → `2anaheim` (38z/914l) → `3barcelona` (110z/2522l) →
`4winnipeg` (147z/2836l) → planned: `chicagosketch` (needs the fixed-cost-positive
link relaxation), `chicagoregional`, … (with per-scenario budget tables).

---

## 4. How a 1975 solver and a 2025 surrogate share one harness

**Frank–Wolfe (LeBlanc et al. 1975):** `Capabilities(paradigm="static_ue",
deterministic=True, provides_gap=True, inputs={"od_matrix"})`, `WhiteBoxMixin`. `solve()`
iterates all-or-nothing + exact line search on the Beckmann objective, emitting flows to
`Trace` each iteration. Deterministic track, M=1. Its harness-computed gap must match its
self-report to numerical precision — a built-in honesty regression test.

**GNN surrogate (2020s):** `CallableAdapter` around a trained network;
`Capabilities(paradigm="learned", provides_gap=False, seedable=True,
trained_on=("tntp-small-v1",))`. `solve()` is one forward pass (budget: 1 evaluation).
The harness audits feasibility, computes the same certified relative gap and flow errors
as for Frank–Wolfe, blocks evaluation on scenarios in its training lineage, and
macroreplicates over training seeds for distributional scoring. The leaderboard
legitimately shows "gap 1e-14 in 200 SP-call-equivalents" next to "gap 3e-2 in 1
evaluation" — that contrast *is* the scientific output.

**External engines — SUMO + DTALite shipped; the MATSim/DynaMIT/DYNASMART adapters
deferred on a measured record
([ADR-030](design/adr-030-external-dta-simulators-deferred.md)):** the pattern is write
inputs, shell out (with an explicit seed where the engine takes one), parse outputs —
same ABC, same trace, same certification where static costs permit; otherwise scored on
the observational track (which for external dynamic engines does not exist yet — MATSim
runs headless on this box but its queue model has no static latency function, so the
mandatory cost-matched anchor is impossible in kind until that certificate ADR ships;
DynaMIT has no public artifact; DYNASMART is license-blocked).
The first is `sumo-marouter` (SUMO's macroscopic `marouter`, Lopez et al. 2018,
[ADR-027](design/adr-027-sumo-marouter.md)): the `eclipse-sumo` wheel ships the
binaries inside the package (addressed via `sumo.SUMO_HOME`), so it is a registered,
CI-validated model behind the optional `[sumo]` extra — no `DockerAdapter` needed.
Because marouter's cost law is a *hardcoded* linear-in-flow class function (not a user
BPR), a `power=1` scenario is compiled to a SUMO network matching the BPR to machine
precision on representable links, with two documented representability floors; the
certified gap under the *declared* costs is the honest simulator-to-benchmark model gap.
The second is `dtalite-tap` (the PyPI `DTALite` wheel's static Frank–Wolfe `assignment()`,
Zhou & Taylor 2014, [ADR-029](design/adr-029-dtalite-tap.md)): unlike marouter its
per-link BPR VDF maps the repo cost *exactly* (the compile map is the identity), so BPR
`power=4` encodes directly — Sioux Falls becomes the marquee, the first external engine on
the power-4 ladder — and the certified gap is the engine's own Armijo line-search stall,
not a mapping floor (a converged `bfw` beats it by orders of magnitude). The guard uses
`find_spec` (never `import DTALite`, which prints a banner and ctypes-loads an OpenMP
engine into the host) and the subprocess wrapper is mandatory, not just hygienic — the
engine's error handler does `getchar()` + `exit()` in-process, so a bad input would hang
or kill the host. That same `sumo-marouter` adapter is reused as the inner oracle of the
first **guarded T2 estimator**,
`spsa-sumo` ([ADR-028](design/adr-028-spsa-sumo.md)): Balakrishna et al.'s (2007) SPSA
calibration run against the real `marouter` loop (demand-only) and certified through the
UNCHANGED pinned-bfw certifier — a production simulator calibration loop is just another
estimator row, so the T2 self-vs-certified honesty diff now measures the
simulator-in-the-loop bias rather than estimator dishonesty.

---

## 5. Implementation roadmap

Tiers are driven by the verified reference canon (`docs/REFERENCES.md`, 246 references:
63 tier-1, 134 tier-2, 49 tier-3):

- **v0 (this repo, now):** core abstractions; TNTP fetcher+parser with units metadata;
  Braess (builtin, analytic) + Sioux Falls scenarios; AON, MSA, Frank–Wolfe;
  callable black-box adapter; certified gap/AEC/Beckmann metrics; LinkCounts/FullOD
  observation levels with identifiability check; experiment runner with manifests; tests
  incl. analytic Braess UE and the Sioux Falls best-known-objective regression.
- **v0.x (this release):** conjugate/bi-conjugate FW (shipped); Dial's STOCH and logit
  SUE via MSA with the fixed-point certificate (shipped, docs/design/adr-001); Anaheim,
  Barcelona, Winnipeg scenario rungs with best-known oracles (shipped); convergence
  target protocol per Boyce et al. 2004 (shipped); SimOpt-style progress curves and
  solvability/data profiles (shipped: `experiments.profiles`, docs/design/adr-032). Still
  open in v0.x: gradient projection (path-based), `tabench validate` conformance suite,
  entry-point plugin registry.
- **v1 (in progress):** path-based gradient projection (shipped: `gp`, the first
  solver reaching certified gaps below 1e-8 within ~100 iterations — a regime the
  FW family needs thousands to cross); Dial's Algorithm B bush solver (shipped:
  `algb`, certified below 1e-8 on Sioux Falls in ~20 iterations — 1e-10 by ~18
  on the reference build, where the FW family needs hundreds); Bar-Gera's TAPAS
  paired-alternative-segment solver (shipped: `tapas`, sharing algb's bush
  machinery via `_bush.py`, and the first solver to drive *route* flows to the
  proportional/entropy-consistent solution — its `proportionality_residual`
  diagnostic falls ~5 orders of magnitude vs pure UE at identical link flows;
  [ADR-004](design/adr-004-proportionality-certificate.md)); system optimum +
  certified SO gap + price of anarchy + first-best tolls (shipped: `so-bfw`,
  `metrics.so`); probit SUE via MC-MSA with a pinned Monte-Carlo fixed-point
  certificate (shipped: `sue-probit-msa`, [ADR-003](design/adr-003-probit-sue-mc-certificate.md);
  the first stochastic-track model — macroreplication + bootstrap CIs); the T2
  OD-estimation track (shipped: `estimation/`, [ADR-002](design/adr-002-t2-estimation-certificate.md));
  elastic (variable) demand UE via the Gartner excess-demand transform with a P1-pure
  demand-recomputing certificate (shipped: `fw-elastic`, paradigm `static_ue_elastic`,
  [ADR-005](design/adr-005-elastic-demand.md); Florian & Nguyen 1974 / Gartner 1980 /
  Sheffi 1985); combined trip-distribution + assignment with a fully endogenous OD matrix —
  only the trip-end margins are fixed and the OD flows are distributed by a doubly-constrained
  gravity model at the equilibrium costs, certified by recomputing that gravity demand from
  the flows (shipped: `evans`, paradigm `static_ue_combined`,
  [ADR-007](design/adr-007-combined-distribution-assignment.md); Evans 1976, a reuse of the
  elastic demand-recomputing machinery); the first **day-to-day** model — Smith's (1984)
  proportional route-swap dynamical system, modeling the disequilibrium adjustment toward the
  UE fixed point with a Beckmann Lyapunov function that decreases monotonically each day
  (shipped: `dtd-swap`, paradigm `day_to_day`; certified by the standard UE gap, with the
  Smith & Wisten 1995 step bound preventing the raw swap's limit-cycle), and its link-based
  companion — He, Guo & Liu's (2010) day-to-day defined directly on the aggregate *link*-flow
  vector (adjusted toward the frozen-cost proximal target projected onto the feasible link
  polytope), which reaches the identical certified UE via the same monotone Beckmann descent
  (shipped: `dtd-link`, paradigm `day_to_day`); a **boundedly-rational**
  equilibrium — an indifference-band relaxation of Wardrop where used routes need only lie
  within a band `ε` of the shortest, so the equilibrium is a *set* and the emitted flow sits at
  the band edge (shipped: `br-ue`, paradigm `static_br_ue`,
  [ADR-008](design/adr-008-boundedly-rational-ue.md); Mahmassani & Chang 1987, with a
  necessary-not-sufficient `AEC ≤ ε` link-flow certificate honestly bounded); **side-constrained**
  UE under hard link capacities `v_a ≤ u_a` via an augmented Lagrangian whose multipliers are the
  queueing tolls at binding links, certified by exact link-visible capacity feasibility (shipped:
  `sc-tap`, paradigm `static_sc_ue`, [ADR-009](design/adr-009-side-constrained-ue.md); Larsson &
  Patriksson 1995, reducing exactly to UE when nothing binds); the first **learned** (black-box) model certified by the same P1 harness —
  a per-link surrogate trained on a synthetic family and gated off the disjoint TNTP test
  set by `trained_on` (shipped: `learned-surrogate`, paradigm `learned`,
  [ADR-006](design/adr-006-learned-model-certification.md); Rahman & Hasan 2023 line, with
  the Xu et al. 2024 dataset now shipped as the cross-domain axis
  ([ADR-033](design/adr-033-xu2024-dataset.md)) — link-flow accuracy is shown *not*
  to imply certification), and the first **torch** model behind the optional `[torch]` extra —
  an implicit-NN UE whose output is demand-feasible *by construction* (v = Δᵀh over
  column-generated route sets), so it clears the audit the ridge is censored by yet still
  certifies a gap a converged solver beats (shipped: `implicit-ue-nn`, paradigm `learned`,
  [ADR-025](design/adr-025-implicit-ue-nn.md); Liu et al. 2023, a lean variant — feasibility
  is architectural, equilibrium quality is not), and a second **torch** model on the same
  extra — a heterogeneous GNN whose conservation is only a *soft* loss, so its paper-faithful
  raw emission is censored and a flagged repo-extension route-decode recovers feasibility
  (shipped: `het-gnn`, paradigm `learned`, [ADR-026](design/adr-026-het-gnn.md); Liu & Meidani
  2024, a lean variant — the third act of the feasibility-mechanism gradient: no conservation →
  soft conservation + decode → conservation by construction). Still open: a *scored* route-flow proportionality certificate (ADR-004
  proposes it; the diagnostic ships now);
  distribution-emitting T2 estimators (Hazelton-style samplers) and
  **computational-graph estimators** — assignment/estimation expressed as a layered
  differentiable graph solved by forward-backward passes (Wu, Guo, Xian & Zhou 2018;
  Ma & Qian 2018; Ma, Pi & Qian 2020) or iterative backpropagation through the solver
  (Patwary et al. 2023). This is the natural generalization of the planned
  `WhiteBoxMixin`: a model that exposes gradients of its outputs w.r.t. demand/cost
  parameters unlocks gradient-based T2 baselines and end-to-end learned pipelines,
  while still being certified purely from its emitted flows (P1).
  Frozen v1.0 core grid (3 networks × 4 data levels × T1/T2) with budget tables.
- **v2:** DTA — the **dnl-core dynamic-network-loading foundation has landed**
  ([ADR-010](design/adr-010-dnl-core.md): generic sending/receiving link/node interfaces,
  `DynamicScenario` + domain-separated hash, DNL P1 certificates C0–C7); CTM/LTM/Newell/
  Godunov/node-model build on it as `LinkModel`/`NodeModel` subclasses. Still ahead:
  analytical DUE, subprocess adapters for DTALite/MATSim/SUMO; T3 intervention suite; Docker
  images per engine; public leaderboard.

## 6. Top design risks and mitigations

1. **Metric gaming / self-report drift** → all metrics harness-computed (P1); self-report
   vs harness discrepancy flagged; evaluator regression-tested against oracles.
2. **Cross-vintage unfairness** (C engines vs Python; amortized training) → hardware-free
   budget coordinates (P6); segmented leaderboards; training cost amortized column;
   T3 as the equalizer.
3. **ML train/test contamination** → training-data cards + lineage gates + perturbation
   generator (P7).
4. **Data/licensing rot** → no vendoring, checksummed fetcher, per-network metadata and
   defects registry, import-time validation against published objectives (P9).
5. **Scope explosion** (networks × levels × tasks × models) → frozen versioned core
   grids; everything else behind the registry as contrib; per-scenario budget tables cap
   compute (BO4Mob policy).
