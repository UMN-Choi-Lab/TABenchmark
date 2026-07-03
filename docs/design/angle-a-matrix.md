# TABenchmark Angle A: A Scenario × Model Cross-Evaluation Matrix with SimOpt-Style Experiment Machinery

# TABenchmark — Angle A: The Scenario × Model Matrix

Organize the benchmark as a cross-evaluation matrix **Scenario × Model × Task**, adapting SimOpt's Model/Problem/Solver split: a *Scenario* is the problem instance (network + demand + cost config + observation config), an *AssignmentModel* is the method under test (1956 incremental loading through 2025 GNN surrogates), and the *harness* — never the model — computes all metrics.

## Core abstractions

```python
class Scenario:                      # frozen, content-hashed, declarative (no code)
    network: Network                 # capacity, fft, bpr_b, bpr_power, toll, first_thru_node
    demand: DemandMatrix
    cost_config: CostConfig          # per-network toll/distance weights (not in .tntp!)
    ref: ReferenceSolution | None    # best-known flows + objective + provenance/hash

class AssignmentModel(ABC):
    capabilities: Caps   # deterministic, seedable, iterative, emits_path_flows,
                         # self_reports_gap, external_engine, needs_training
    factors: dict[str, FactorSpec]   # typed hyperparameters with defaults (SimOpt-style)
    def setup(self, scenario: Scenario, rngs: RNGBundle) -> None: ...
    def run(self, budget: Budget) -> Iterator[FlowState]: ...   # yield at checkpoints

@dataclass(frozen=True)
class FlowState:
    link_flows: np.ndarray
    path_flows: PathFlows | None
    coords: BudgetCoords             # iters, model_wall_ms, sp_calls, engine_evals
    self_report: dict                # model's own gap/objective — recorded, never trusted

class Evaluator:                     # harness-side, model-blind
    def score(self, s: Scenario, st: FlowState) -> Metrics:
        # AON shortest paths at costs(st.link_flows) -> relative gap, AEC,
        # Beckmann objective, RMSE vs s.ref, TSTT; SUE-gap for SUE tasks
```

The load-bearing decision: **every convergence metric is recomputed by the Evaluator from emitted link flows** (relative gap = 1 − AON-cost/current-cost, AEC, Beckmann). This is SimOpt's "postreplicate, never trust in-run estimates" principle applied to the deterministic setting, and it is what lets a convex-programming solver and an opaque simulator share one results schema: anything that emits link flows gets an externally verified gap. Self-reports are logged and diffed against harness values as an honesty check.

**Randomness.** One MRG32k3a generator, SimOpt's fixed (stream, substream, subsubstream) = (macrorep *m*, declared randomness-source *i*, replication *r*) schema, identical stream universe for all (scenario, model) pairs so results are order-independent. Each stochastic model declares its sources of randomness (probit draws, simulator seed, NN init); CRN is switchable across models-on-a-scenario (default on) and across macroreps (default off). Dedicated streams for evaluation, post-evaluation, and bootstrap. External engines must accept a seed passthrough (`--seed` à la BO4Mob's SUMO adapter) or be declared `seedable=False`, which forces larger *M*.

**Budgets.** Three budget classes — `B-iter(K)`, `B-time(seconds)`, `B-eval(sp_calls or engine_evals)` — but every checkpoint records *all four* coordinates, so any curve can be re-sliced post hoc. Wall-clock is normalized by a per-machine calibration constant (time a standard AON pass on SiouxFalls) and the clock stops during harness evaluation, so checkpointing cadence (geometric in budget) doesn't distort timing.

## Scenario & data model

Scenario = data directory + JSON config, no code (BO4Mob convention). Tiered ladder: `0braess` (hand-checkable), `1siouxfalls`, `2anaheim/easternmass`, `3barcelona/winnipeg/chicagosketch`, `4chicagoregional/…`. TNTP data is **downloaded on demand** with per-file checksums and auto-generated citations (repo terms forbid vendoring); a defensive parser handles `first_thru_node` semantics, `~` comments, and trailing `;`. Per-network metadata modules encode unit conventions (SiouxFalls 0.01-h times), generalized-cost weights, and known defects (Austin duplicate links, Chicago-Regional ramp errors) machine-readably. The six published best-known UE solutions (SiouxFalls objective 42.31335287107440, etc.) are regression oracles stored as `ReferenceSolution` with provenance. Observation-level factors (count coverage, noise, periods *N*, trajectories) are first-class scenario factors, reserved as dials for the estimation-task track; sensor sets carry the distinct-nonzero-columns identifiability flag.

## Task & evaluation protocol

Tasks bind a scenario to a target and metric set: **TASK-UE** (relative gap + link-flow RMSE vs best-known), **TASK-SUE(θ)** (SUE gap / fixed-point residual), **TASK-PREDICT** (amortized flow prediction, scored identically to UE at zero iteration budget), with estimation tasks (OD-from-counts) schema-compatible later. Compatibility metadata (deterministic? provides paths? needs training?) auto-filters the matrix, SimOpt-style.

Protocol per (scenario, model, task):
1. **Run stage:** *M* macroreps (*M*=1 on the deterministic track — no wasted replication machinery); each yields a checkpoint sequence.
2. **Post stage:** Evaluator scores every checkpoint; metrics that are themselves stochastic (simulator TSTT) get *N* post-evaluations on the dedicated stream.
3. **Aggregation:** progress curves (log relative gap vs each budget coordinate, mean/quantile bands over macroreps); α-solve-time solvability curves for α ∈ {1e-2, 1e-4, 1e-6} (CDF of first budget where gap ≤ α); Moré–Wild data/performance profiles aggregated over the scenario tier; AUC scatter; all CIs by bootstrapping macrorep outputs (dedicated stream), never parametric.
4. **Non-convergent models are first-class:** models that cannot reach equilibrium (DTA simulators, surrogates) are reported as gap-at-budget points and appear in solvability profiles as censored (never-solve) entries — honest, not excluded. Results land in one parquet schema: `(scenario_hash, model_id, model_version, task, m, coords, metrics{...}, self_report{...}, env{python, package versions, hardware, calibration_constant})`.

## Frank-Wolfe (1975) and a GNN surrogate (2025) in one harness

**Frank-Wolfe:** `capabilities = {deterministic, iterative, self_reports_gap}`. `run()` yields a `FlowState` per iteration with `sp_calls` incremented; harness gap should match its self-report to numerical precision (a built-in correctness test). M=1; its curve is the classical gap-vs-iteration profile, now also available vs normalized time and SP calls.

**GNN surrogate:** `capabilities = {seedable, needs_training, emits_path_flows=False}`. It registers a **training-data card** declaring exactly which scenario hashes and demand perturbations it trained on; the harness enforces disjointness from evaluation scenarios (frozen published splits + a demand-perturbation generator for held-out instances). `run()` emits one checkpoint (or a few, if it does iterative refinement); `coords` records inference wall-time and zero SP calls. The Evaluator computes its relative gap and flow RMSE **exactly as for FW** — so the headline comparison "GNN reaches gap 3e-3 in 0.1 normalized seconds; FW needs 40 iterations to match" is native output, and its inability to reach 1e-6 shows up honestly in solvability profiles. Seeds over weight init/training give *M* macroreps capturing training variance.

External binaries (SUMO, MATSim, TAP-B) plug in via subprocess adapters exposing the same `setup/run` contract, file-based I/O, and seed passthrough.

## Repo layout

```
tabenchmark/
  core/          # factors, rng (MRG32k3a bundle), budget, flowstate, capabilities
  scenarios/     # tntp parser, fetcher(+checksums), registry, network_notes/, configs/*.json
  models/        # base.py, fw.py, msa.py, bfw.py, algorithm_b.py, sue/*, 
                 # adapters/{subprocess_base, sumo, matsim}, surrogates/
  evaluation/    # gap.py, metrics.py, profiles.py, bootstrap.py
  experiments/   # runner.py (parallel macroreps), grid.py, postprocess.py
  cli/           # evaluate_once.py, run_benchmark.py   (BO4Mob's two-mode pattern)
demos/           # laddered: demo_scenario, demo_model, demo_experiment
tests/  docs/    # DATASHEET.md, extension guides, per-network defect notes
```

Proper `pyproject.toml` pip package with pinned deps + Docker + CI (fixing BO4Mob's gaps); deterministic run-folder naming encoding all factors.

## Top-5 design risks & mitigations

1. **Metric gaming / self-report drift.** Models could tune to their own gap formula. → Harness recomputes everything from flows; self-report vs harness discrepancies flagged in report cards; reference-solution regression tests on evaluator code.
2. **Unfair cross-vintage timing** (C engines vs Python, 1975 vs 2025 hardware). → Multi-axis budgets with SP-call and iteration counts as hardware-free primaries; calibration-normalized wall-clock as secondary; pinned Docker environment recorded per run.
3. **Black boxes without checkpoints or seeds.** → Capability metadata degrades gracefully: final-state-only scoring, gap-at-budget, censored solvability entries; unseedable engines require larger *M* and are labeled as such.
4. **Data/reference rot** (TNTP unit chaos, wrong best-knowns, license limits). → No vendoring; checksummed fetcher; per-network metadata + defect notes; validate parsers against the six published objectives at import time.
5. **ML train/test contamination.** → Mandatory training-data cards, scenario content-hashing, frozen splits, harness-side disjointness enforcement, and held-out demand-perturbed scenarios generated from dedicated RNG streams.

