# TABenchmark Design Proposal — Contract/Plugin-First Extensibility (Angle C)

# TABenchmark: Contract-First Design Proposal (Angle C)

## Core abstractions

Everything hangs on one minimal contract plus declared capabilities. The harness — not the model — computes all scored metrics.

```python
@dataclass(frozen=True)
class Capabilities:
    paradigm: Literal["static_ue","static_so","sue","dta","simulation","learned"]
    determinism: Literal["deterministic","stochastic"]
    inputs_required: frozenset[str]     # {"od_matrix"} | {"link_counts"} | ...
    outputs: frozenset[str]             # {"link_flows","path_flows","flow_dist","od_estimate"}
    provides_gap: bool                  # can self-certify equilibrium gap
    seedable: bool
    trained_on: tuple[str, ...]         # scenario-family lineage IDs (learned models)

class TrafficAssignmentModel(ABC):
    capabilities: ClassVar[Capabilities]
    factors: ClassVar[dict[str, FactorSpec]]   # SimOpt-style: defaults, type, bounds

    @abstractmethod
    def solve(self, scenario: Scenario, budget: Budget,
              rng: RngBundle, callback: Trace | None = None) -> ResultBundle: ...

class WhiteBoxMixin:                    # optional richer surface
    def link_cost(self, v): ...
    def link_cost_jacobian(self, v): ...
    def objective(self, v): ...         # e.g. Beckmann; enables duality-gap checks
```

`Budget` counts **scenario evaluations / iterations**, never wall clock (wall clock is recorded, not ranked). `Trace` receives `(budget_fraction, solution)` pairs, feeding progress curves. `ResultBundle` = link flows (+ optional path flows / distributions / OD estimate), the trace, self-reported gap, seeds consumed, and engine versions. Self-reported numbers are provenance only — the harness recomputes every metric.

**Key soundness move:** for static tasks with known cost functions, relative gap = (TSTT − SPTT)/SPTT is a property of `(link_flows, scenario)`. The harness computes it *externally* for any model whose output passes a demand-feasibility audit (flow conservation, OD totals). So a black-box GNN still receives a certified gap; models producing infeasible flows are flagged, never silently scored.

**Adapters** subclass the same ABC: `SubprocessAdapter` / `DockerAdapter` (DTALite, SUMO, MATSim, arbitrary binaries — BO4Mob's seeded-subprocess pattern: write inputs, shell out with explicit `--seed`, parse outputs) and `CallableAdapter` (torch/GNN callables). Unseedable engines must declare `seedable=False` and get macroreplicated instead.

**Registry:** models register via Python entry points (`[project.entry-points."tabench.models"]`), so third parties `pip install tabench-mymodel` and `tabench run --model mymodel` discovers it — BO4Mob's string registry, made pip-native.

**Task layer (SimOpt's Model/Problem split):** a `Task` wraps a Scenario + ObservationSpec + objective (solve-UE, OD-estimation-from-counts, calibration-from-trajectories). Task↔model compatibility is enforced automatically from `Capabilities` (e.g., a counts-only task refuses a model requiring full OD), so one network spawns many problems cheaply.

## Scenario & data model

A **Scenario** is an immutable, content-addressed spec, not code: canonical network arrays (defensively parsed TNTP/GMNS, `first_thru_node` centroid semantics honored in every shortest path), demand, cost parameters *including per-network toll/distance weights and units metadata* (SiouxFalls's 0.01-hour times documented, not guessed), and a machine-readable known-defects registry (Austin duplicate links, Chicago-Regional ramp errors). `scenario_hash = SHA256(canonical serialization)`; leaderboard entries pin the hash, so a silently edited network can never masquerade as the benchmark scenario.

**Data are fetched on demand, never vendored** (TNTP terms are academic-only, non-OSI): cached downloads with per-file checksums and auto-generated citation strings. Best-known UE solutions (SiouxFalls 42.31335287…, Barcelona, Winnipeg, Chicago-Sketch/Regional, Anaheim) become regression oracles.

**Observation levels derive from stored ground truth**, Hazelton-style: synthetic scenarios store realized integer *route* flows `x`; every data level is a projection — full OD, per-period link counts `y=Aₘx` (number of periods N and monitored-link mask are dials; per-period counts distributed, never time-averages), noisy counts, trajectory samples, stale-prior OD (Gamma pseudo-counts). Each (network, sensor-set) config reports the identifiability condition — distinct nonzero columns of the masked incidence matrix — and the suite deliberately includes violating configs. Incidence matrices, masks, and priors ship as first-class artifacts.

Scenario ladder: `0braess` (hand-checkable), `1siouxfalls`, `2anaheim`/`2eastmass`, `3barcelona`/`3winnipeg`/`3chicagosketch`, `4chicagoregional`+, each with a declarative YAML budget table (BO4Mob Table-3 style).

## Task & evaluation protocol

**Two tracks.** *Deterministic:* M=1, no postreplications; certified relative gap vs budget as Moré–Wild data profiles; convergence regression against best-known objectives. *Stochastic* (SUE, simulators, ML): full SimOpt machinery — M macroreps, N independent postreplications for unbiased objective estimates at recommended solutions, normalization against shared x₀ and x*/best-known, MRG32k3a with the fixed (macrorep, randomness-source, replication) stream schema, switchable CRN layers, bootstrap CIs, mean/quantile progress curves and α-solve-time solvability profiles.

A single `metrics/` module defines each metric once: relative gap, link-flow NRMSE vs oracle, OD/route-flow error, and for distributional outputs empirical coverage of 95% intervals (point accuracy and uncertainty calibration scored *separately* — overdispersion misspecification corrupts the latter first). Feasibility-aware reporting: count-matching error and equilibrium gap are reported as a pair, never collapsed to one scalar.

**Honest non-equilibrium handling:** models without a gap (DTA heuristics, microsimulators, surrogates) appear in a "no equilibrium certificate" leaderboard column with externally computed gap where feasible-flow output permits; otherwise scored only on observable-fit metrics. No imputed zeros, no shared single ranking across tracks.

**Fairness gates the harness enforces:** refuse any run where `trained_on` intersects the evaluation scenario's family lineage; identical budgets in evaluations; warm starts are declared factors; every run folder (deterministic BO4Mob-style naming) carries `manifest.json` (scenario hash, package+engine versions, seeds, env, git commit) and the leaderboard rejects incomplete manifests. **CI conformance suite for contributions:** solve `0braess` to the known answer, budget respected, seeded-twice determinism, declared outputs actually returned, adapter smoke test in Docker.

## 1975 Frank–Wolfe and a 2025 GNN, side by side

**Frank–Wolfe:** ~200-line NumPy/SciPy class with `WhiteBoxMixin`; `Capabilities(paradigm="static_ue", deterministic, provides_gap=True, inputs={"od_matrix"})`. `solve()` iterates AON + line search, emitting flows per iteration to `Trace`. Deterministic track: gap-vs-iteration profile, SiouxFalls objective as a CI regression test.

**GNN surrogate:** `CallableAdapter` around a torch model; `Capabilities(paradigm="learned", provides_gap=False, trained_on=("tntp-small-v1",))`. `solve()` is one forward pass (budget = 1 evaluation) returning link flows; the harness audits feasibility, computes the external gap and NRMSE, and the fairness gate blocks evaluation on its training families. Optionally macroreplicated over seeds for distributional scoring. Both yield `ResultBundle`s consumed by identical profile machinery — the leaderboard legitimately shows "gap 1e-14 in 200 evaluations" vs "gap 3e-2 in 1 evaluation," which *is* the scientific comparison.

## Repo layout

```
tabench/            # installable package (pyproject.toml — fixes BO4Mob gap)
  core/             # model.py task.py scenario.py factors.py results.py rng.py budget.py
  models/           # fw.py msa.py path_based.py sue_logit.py ...
  adapters/         # subprocess.py docker.py callable.py; dtalite/ sumo/ matsim/
  data/             # fetchers (tntp.py, gmns.py), checksums, citations, defects.yaml
  observe/          # count/trajectory/noise samplers, identifiability checks
  metrics/  experiments/  cli.py
scenarios/*.yaml    # hashed declarative specs + budget tables
conformance/        # plugin contract test suite (`tabench validate`)
demos/              # demo_model.py demo_task.py demo_experiment.py (laddered)
docs/               # DATASHEET.md, extending.md, per-network unit notes
tests/  Dockerfile  .github/workflows/   # per-adapter pinned engine images
```

## Top-5 design risks & mitigations

1. **External-engine drift** (SUMO/DTALite versions change results) → per-adapter Docker images pinned by digest; engine version in every manifest; CI smoke tests; unpinned runs flagged non-leaderboard.
2. **Cross-vintage budget incomparability** (an FW iteration ≠ a simulation ≠ a forward pass) → budget = scenario evaluations with wall clock reported separately; multi-column leaderboard segmented by track/certificate, never one global ranking.
3. **Learned-model test leakage** → `trained_on` lineage declarations enforced by the harness, required training-data manifests, plus a held-out *generator* (demand/topology perturbations of base scenarios) so there is no fixed test set to memorize.
4. **Uncomputable certificates for black boxes** → feasibility audit before external gap; explicit "no certificate" labeling; pair-reported fit-vs-gap metrics rather than imputation.
5. **Contract friction deterring contributors** → one abstract method + one declaration; cookiecutter template, laddered demos, and `tabench validate` giving actionable conformance errors before any PR review.

