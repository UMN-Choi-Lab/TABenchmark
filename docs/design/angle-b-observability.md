# TABenchmark: An Observability-First Benchmark for 50 Years of Traffic Assignment Models

# TABenchmark — Angle B: Observability-First Design

**Thesis.** Traffic assignment models from 1956–2026 disagree less about networks than about *what data they see*. TABenchmark makes the observation process a first-class object: every benchmark instance = (network, demand, **DataLevel**, task, budget). Ground truth is stored once at the finest granularity (realized integer route flows, per Hazelton AOAS805); every observation level is a seeded, versioned projection of it.

## 1. Core abstractions (SimOpt-inspired, deterministic track added)

```python
class TAModel(ABC):                     # ~ SimOpt Model
    factors: dict[str, Factor]          # typed: scalar/vector/matrix; decision vs setting
    responses: tuple[str, ...]          # e.g. ("link_flows","route_flows","gap","cost")
    capabilities: Capabilities          # white_box, provides_gap, stochastic,
                                        # needs_paths, n_rng_sources, supports_seed
    def evaluate(self, factors, rng: RNGBundle) -> Responses: ...

class Scenario:                         # network + demand + truth, NO code
    network: Network                    # links, BPR params, first_thru_node, A-matrix
    demand: ODMatrix
    truth: GroundTruth                  # route flows x*, link flows, UE objective, gap

class DataLevel(ABC):                   # observation process — the novel axis
    def observe(self, truth, rng) -> Dataset: ...
# concrete: FullOD, NoisyOD(cv), LinkCounts(sensor_set, noise, n_periods),
#           Trajectories(penetration, gps_noise), StalePriorOD(age_scale)

class Problem(ABC):                     # ~ SimOpt Problem: Scenario × DataLevel × Task
    model_cls: type[TAModel]
    decision_factors: tuple[str, ...]
    def objective(self, responses, dataset) -> float | Distribution: ...
    compatibility: ProblemTags          # deterministic?, gradient?, gap-checkable?

class Solver(ABC):
    def solve(self, problem, budget: Budget, rng) -> Trajectory  # recommended
                                                                 # solutions vs t∈[0,1]

REGISTRY = {"models": {...}, "solvers": {...}, "datalevels": {...}}  # BO4Mob pattern
```

`RNGBundle` wraps MRG32k3a with SimOpt's fixed (stream, substream, subsubstream) = (macrorep, randomness-source, replication) schema, plus dedicated Observation/Post-processing/Bootstrap/Overhead streams. CRN across solutions and solvers is switchable; models declare `n_rng_sources` so external engines synchronize.

**Two tracks.** Deterministic (convex UE/SO): M=1, no postreplications; metric = exact relative gap / Beckmann duality gap vs iterations *and* function evaluations. Stochastic (SUE, simulators, ML): full M macroreps + N unbiased postreplications, never trusting in-run estimates.

## 2. Scenario & data model

- **Network ladder** (BO4Mob convention): `0braess` (hand-checkable), `1siouxfalls`, `2anaheim`, `3eastmass`, `4barcelona`, `5chicagosketch`, `6chicagoregional`. Each = data directory + declarative JSON/YAML config (units, toll/distance weights, per-scenario budgets, known-defect notes like Chicago-Sketch's under-congestion). **No vendoring of TNTP** (academic-use-only license): a fetcher downloads from bstabler/TransportationNetworks with SHA256 checksums, cached store, auto-generated citation strings. Defensive TNTP parser (`;` terminators, `~` comments, FIRST-THRU-NODE centroid semantics enforced in all shortest paths, per-network unit metadata — SiouxFalls 0.01-h times documented, never assumed).
- **Ground truth**: reference UE flows at relative gap ≤1e-12 (validated against the 6 published best-known solutions, e.g. SiouxFalls objective 42.31335287107440), plus *realized* integer route flows sampled from a documented generative model — link counts, trajectories, and OD samples all derive from the same x, so levels are mutually consistent.
- **DataLevel dials**: sensor coverage %, *specific sensor set* (not just fraction — we compute and report the Hazelton identifiability condition: distinct nonzero columns of A restricted to monitored links, and total unimodularity), noise model, number of observation periods N (per-period counts distributed, never day-averages — dependence across days carries information), trajectory penetration rate, stale-prior age. Configurations that deliberately *violate* identifiability are included and flagged.
- Every dataset artifact ships its A-matrix, monitored-link mask, per-period counts, priors, generation seed, and datasheet entry (Gebru-style DATASHEET.md; filtering protocols written up as appendices, BO4Mob-style).

## 3. Task & evaluation protocol

- **T1 Equilibrium** (full spec): score = relative gap vs budget (function evaluations, not wall clock), link-flow L2/NRMSE vs best-known flows. Moré–Wild data profiles + SimOpt solvability profiles.
- **T2 Estimation/calibration** (partial observations): point metrics (route/OD/link NRMSE vs true x) **separately from** distributional metrics — held-out-count log-likelihood (stochastic-EM/MCMC estimated where exact is infeasible), CRPS, 95%-interval empirical coverage over replicated ground truths. Rationale (AOAS805): overdispersion misspecification corrupts calibration before point accuracy; single-snapshot counts admit ~10^14-point feasible polytopes, so we score against distributions, never unique inversions. Shared initial candidates per run; Improvement% vs common baseline (BO4Mob).
- **T3 Prediction under intervention**: capacity cut / demand shift / link closure applied to truth generator; models fitted on pre-intervention data predict post-intervention flows. This is the fair arena where a 1975 solver's structural prior competes with a 2025 surrogate's flexibility.
- **Honest non-convergence**: solvers that cannot certify equilibrium report `gap=None`; leaderboards show a certified column (gap-checkable) and an observational column (fit metrics only) — no silent mixing. Feasibility-aware metrics report the count-matching-error vs equilibrium-gap trade-off as a curve, not a scalar.
- **Reproducibility**: every run folder name encodes all factors (BO4Mob convention); manifest.json records package versions, data hashes, RNG offsets; error bars via bootstrap on dedicated stream; Docker image + CI; per-scenario budget/repetition tables published up front.

## 4. Frank-Wolfe (1975) and a GNN surrogate (2025), same harness

**Frank-Wolfe**: a `Solver` on T1 with `capabilities(white_box=True, provides_gap=True, deterministic=True)`. Its all-or-nothing subproblem uses the library's centroid-aware shortest path; each iteration emits (flows, Beckmann objective, relative gap); the harness records the gap trajectory directly — deterministic track, M=1.

**GNN surrogate**: a `TAModel` adapter whose `evaluate()` maps (network tensors, OD) → predicted link flows; `provides_gap=False, stochastic=True` (declared RNG source = weight init/dropout seed). On T1 it is scored by *post-hoc* gap: the harness computes the relative gap of its predicted flows using the known BPR functions — the same certificate FW reports, computed externally. On T2/T3 it trains on DataLevel outputs via a `fit(dataset)` hook and is postreplicated like any stochastic model. Compatibility metadata prevents nonsensical pairings (e.g., a gradient-requiring solver on a subprocess black box); external engines (SUMO/MATSim/binaries) use the same contract through seeded subprocess adapters with file-based I/O, exactly BO4Mob's `sumo_runner` pattern.

## 5. Repo layout

```
tabenchmark/            # pip-installable (pyproject.toml — fixes BO4Mob gap)
  core/     model.py problem.py solver.py factors.py rng.py budget.py
  data/     fetchers/ (tntp, gmns) parsers/ scenarios/*.yaml datalevels/
  models/   fw.py msa.py b_algo.py sue_probit.py adapters/{subprocess,ml}
  tasks/    t1_equilibrium.py t2_estimation.py t3_intervention.py
  metrics/  gaps.py pointwise.py distributional.py identifiability.py
  experiments/ runner.py postprocess.py plots.py   # curves/profiles/bootstrap
demos/      demo_model.py demo_problem.py demo_experiment.py  # SimOpt ladder
tests/      unit + regression vs published UE objectives
docs/       DATASHEET.md CONTRIBUTING.md protocols/  # data-gen appendices
docker/  .github/workflows/   # build+test CI, image push
```

## 6. Top-5 design risks & mitigations

1. **Data licensing** (TNTP is academic-only, no OSI license) → never redistribute; checksum-verified download-on-demand, citation auto-generation, CI uses a synthetic mirror network.
2. **Unknown x\* for normalization on hard/stochastic instances** → prefer gap-certified problems; else best-found proxy with an explicit flag, and bootstrap CIs that acknowledge proxy uncertainty (SimOpt caveat, adopted).
3. **Unfair cross-vintage comparison** (ML models amortize training; FW pays per instance) → budgets counted in target-model function evaluations; training cost reported as a separate amortized column; T3 intervention generalization as the equalizer.
4. **Latent-flow samplers silently failing** in statistical baselines (Tebaldi–West 0% ESS pathology) → ship Hazelton's adaptive-partition sampler as the validated default, check total unimodularity of A per configuration, ESS diagnostics gate results.
5. **Scope explosion** (networks × data levels × tasks × models) → freeze a v1.0 core grid (3 networks × 4 data levels × T1/T2), everything else behind the registry as `contrib/`; per-scenario budget tables cap compute like BO4Mob's reduced-settings-openly-flagged policy.

