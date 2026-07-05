# TABenchmark

**A shared benchmark for 50 years of traffic assignment models.**

Traffic assignment research spans from Wardrop's principles (1952) and Beckmann's convex
program (1956) through Frank–Wolfe, bush-based solvers, stochastic and elastic-demand user
equilibrium, and dynamic traffic assignment, to today's GNN surrogates and end-to-end
learned models — yet there has never been a shared testbed on which all of them can be
compared under identical networks, demand, data availability, and budgets. TABenchmark is
that testbed:

- **One harness, every vintage.** A minimal model contract plus capability declarations
  lets a compact Frank–Wolfe, a bush-based solver, and a trained surrogate run in the same
  experiment matrix — a white-box solver, a subprocess-wrapped simulator, and a neural
  model are all just objects that emit link flows.
- **Certified, not self-reported.** The harness recomputes every scored metric from the
  emitted link flows — the equilibrium relative gap is a property of the flows, so even a
  black box receives an externally certified gap (and a demand-feasibility audit first).
  Model self-reports are kept as provenance and diffed as an honesty check, never scored.
- **Data levels are first-class.** Full OD tables, noisy stale-prior OD, and per-period
  link counts on sensor subsets: every observation process is a seeded projection of
  stored ground truth, with identifiability conditions reported per configuration.
- **Fair across the decades.** Hardware-free budgets (iterations / shortest-path calls;
  wall-clock recorded but never the ranking axis), training-lineage gates against test
  contamination, content-hashed scenarios, and full run manifests.

Built on the verified canon of **246 references** spanning 12 model families
([docs/REFERENCES.md](docs/REFERENCES.md)), the design synthesizes
[SimOpt](https://github.com/simopt-admin/simopt)'s testbed machinery,
[BO4Mob](https://github.com/UMN-Choi-Lab/BO4Mob)'s benchmark conventions, and
Hazelton's (AOAS 2015) statistical treatment of network flow observability —
see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quickstart

```bash
pip install -e .
python demos/demo_quickstart.py         # built-in Braess scenario, no download
```

```
model             certified rel. gap  feasible
----------------------------------------------
aon                        1.912e-01         1
msa                        1.577e-03         1
fw                         7.188e-14         1
cfw                       -2.060e-16         1
bfw                       -2.060e-16         1
toy-surrogate                    nan         0   <- black box failed the demand
                                                    audit: censored, not scored
```

Run a downloaded instance (data fetched on demand, checksummed) — the certified solver
ladder, an elastic-demand equilibrium, or a learned surrogate, all through one CLI:

```bash
tabench run --scenario siouxfalls --models fw,cfw,bfw,algb,tapas --iterations 300 --out results/
tabench run --scenario winnipeg   --models bfw --iterations 500 --target-gap 1e-4 --out results/
tabench run --scenario elastic-tworoute --models fw-elastic --out results/
tabench list                            # scenarios, models, and T2 estimators
```

Wrap **your** model in three lines — the harness scores it identically:

```python
from tabench import CallableModel, Budget, load_scenario, run_experiment

model = CallableModel(fn=my_gnn_predict, name="my-gnn", trained_on=("tntp-small-v1",))
result = run_experiment(load_scenario("siouxfalls"), [model], Budget(iterations=1))
```

(If `trained_on` intersects the evaluation scenario's lineage, the harness refuses to
run it — that's the point.)

## The model roster

Sixteen assignment models and five OD-estimation baselines ship today, spanning white-box
solvers, a stochastic track, elastic and combined-distribution demand, a day-to-day dynamical
system, and the first learned model — every one certified from its emitted flows by the
identical P1 harness. For what each model does
*differently* from its predecessors and the lineage of the whole family, see the
**[model compendium](docs/MODELS.md)** and its [evolution graph](docs/model-evolution.svg).

| Paradigm | Models |
|---|---|
| Baselines | `aon` all-or-nothing · `msa` method of successive averages |
| Link-based UE | `fw` Frank–Wolfe · `cfw` / `bfw` conjugate & bi-conjugate FW (Mitradjieva & Lindberg 2013) |
| Path / bush-based UE | `gp` gradient projection (Jayakrishnan et al. 1994) · `oba` origin-based assignment (Bar-Gera 2002) · `algb` Algorithm B (Dial 2006) · `tapas` paired alternative segments (Bar-Gera 2010) |
| Stochastic UE | `sue-msa` logit via Dial-STOCH + MSA (Fisk 1980) · `sue-probit-msa` probit via Monte-Carlo MSA (Sheffi & Powell 1982) |
| System optimum | `so-bfw` marginal-cost UE — certified SO gap, price of anarchy, first-best tolls |
| Elastic demand | `fw-elastic` variable-demand UE (Florian & Nguyen 1974 via the Gartner excess-demand transform) |
| Combined distribution + assignment | `evans` fully endogenous OD from fixed trip-end margins via a doubly-constrained gravity, in one convex program (Evans 1976) |
| Day-to-day dynamics | `dtd-swap` Smith's (1984) proportional route-swap dynamical system — models the disequilibrium adjustment toward the UE, with a Beckmann Lyapunov function that decreases monotonically day-to-day |
| Learned (black box) | `learned-surrogate` a ridge volume/capacity surrogate — the ML-wrapper demonstrator |
| OD estimation (T2) | `vzw-entropy` · `gls` (Cascetta 1984) · `spiess` gradient (Spiess 1990) · `spsa` (Spall 1992) · `prior` stale-prior baseline |

## What ships

| Component | Contents |
|---|---|
| Core | `Scenario` (frozen, content-hashed; optional SUE dispersion `θ`; optional `ElasticDemand`), `Capabilities`, `Budget` (incl. Boyce-style convergence target), `Trace`, spawn-key RNG schema |
| Data | Defensive TNTP parser, commit-pinned checksummed fetcher, per-network units metadata; scenario ladder Braess → Sioux Falls → Anaheim → Barcelona → Winnipeg, plus built-in analytic anchors (two-route logit-SUE, two-route probit, elastic two-route) |
| Models | 16 assignment models across the roster above: baselines, link-based UE, path/bush-based UE (gradient projection, origin-based, Algorithm B, TAPAS), stochastic UE, system optimum, elastic demand, combined distribution+assignment, a day-to-day route-swap dynamical system, and a learned surrogate — mixing freely in one experiment matrix via the `CallableModel` adapter |
| Metrics | Certified relative gap / average excess cost / Beckmann objective; certified SO gap + price of anarchy + first-best tolls (Yang & Huang 1998; Roughgarden & Tardos 2002); SUE fixed-point residual — closed-form for logit ([ADR-001](docs/design/adr-001-logit-sue-dial-certificate.md)), pinned-Monte-Carlo with noise floor for probit ([ADR-003](docs/design/adr-003-probit-sue-mc-certificate.md)); route-flow proportionality diagnostic for TAPAS ([ADR-004](docs/design/adr-004-proportionality-certificate.md)); demand-recomputing gap for elastic demand ([ADR-005](docs/design/adr-005-elastic-demand.md)); a gravity-recomputing gap for combined distribution+assignment ([ADR-007](docs/design/adr-007-combined-distribution-assignment.md)); a demand-aware feasibility audit and flow RMSE vs. best-known throughout |
| Observe | `FullOD`, `LinkCounts` (sensor mask × periods × noise), `StalePriorOD`, Hazelton identifiability check |
| Estimation (T2) | OD estimation from link counts under a pinned-assignment certificate ([ADR-002](docs/design/adr-002-t2-estimation-certificate.md)): VZW entropy, Cascetta GLS, Spiess gradient, SPSA, and a stale-prior baseline; held-out sensors rank, identifiability reported per task |
| Learned models | The wrapper, certificate, and lineage gate a learned model plugs into ([ADR-006](docs/design/adr-006-learned-model-certification.md)) — trained on a synthetic family, evaluated on disjoint TNTP networks |
| Experiments | Grid runner (T1) + estimation runner (T2), CSV results, full provenance manifests |
| Validation | Per-model provenance report ([docs/VALIDATION.md](docs/VALIDATION.md)) tying every solver to an independent oracle: published best-known flows, cross-solver agreement, and exact analytic anchors |
| Tests | 238 tests: analytic Braess UE/SO and two-route logit/probit/elastic oracles, a symmetric-bipartite combined distribution+assignment oracle (with the certificate's aggregate-multicommodity limitation pinned transparently), the day-to-day route-swap convergence + Beckmann-Lyapunov monotonicity; best-known-solution regressions on Sioux Falls, Anaheim, Barcelona, Winnipeg; cross-family solver agreement; conjugacy-identity and golden-hash regressions |

The certified solver ladder on Winnipeg (147 zones, 2,836 links; iterations to
self-monitored relative gap 1e-4, then the externally certified gap at a fixed
100-iteration budget):

| model | iters to RG 1e-4 | certified gap @ 100 iters |
|---|---|---|
| aon | – | 3.2e-01 |
| msa | – | 1.4e-03 |
| fw  | 161 | 2.2e-04 |
| cfw | 70 | 5.6e-05 |
| bfw | 57 | 2.8e-05 |

The staged roadmap toward the full canon (DTA, dynamic network loading, day-to-day
dynamics, engine adapters, and the T3 intervention track) is in
[docs/ROADMAP.md](docs/ROADMAP.md).

## Why it is trustworthy

TABenchmark rests on nine design principles ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md));
four carry the most weight:

- **P1 — the certificate principle.** The harness, never the model, computes every scored
  metric. The relative gap is a property of `(link_flows, scenario)`, so a 1975 Frank–Wolfe
  and a 2025 GNN share one leaderboard, and a black box cannot self-report its way to the top.
- **P2 — scenarios are data.** Every `Scenario` is frozen and content-hashed; a silently
  edited network cannot masquerade as the benchmark instance.
- **P7 — fairness is enforced by the harness.** Training-lineage gates block evaluating a
  model on a scenario it was trained on; budgets count work, not wall-clock.
- **P9 — data are fetched, never vendored.** Networks are downloaded on demand, checksummed
  at fetch, and regression-tested against published best-known objectives — no dataset is
  redistributed.

Each model is additionally validated against an **independent oracle**, not just its own
gap ([docs/VALIDATION.md](docs/VALIDATION.md)): the UE solvers converge to the published
best-known link flows and agree with one another across algorithm families (link-based,
path-based, bush-based, PAS-based); the Sioux Falls best-known flows reproduce the
Transportation Networks published optimal Beckmann objective (`42.31335287107440`) to full
precision; and the Braess network pins the exact UE (route cost 92), the system optimum
(route cost 83), and the price of anarchy (`≈ 1.108`).

## Data licensing

Benchmark networks come from
[TransportationNetworks](https://github.com/bstabler/TransportationNetworks), donated
for academic research. TABenchmark **never redistributes** these files: they are fetched
on demand from a commit-pinned URL, verified against SHA-256 checksums, cached locally,
and cited (`tabench fetch <scenario>` prints the citation). The package code is MIT.

## Citing

If you use TABenchmark, please cite this repository (see `CITATION.cff`) and the
original references of any model or dataset you evaluate — all 246 entries in
[docs/REFERENCES.md](docs/REFERENCES.md) carry verified BibTeX in
[docs/references.bib](docs/references.bib).

## Contributing

New models, networks, observation levels, and reference implementations from the
community are the purpose of this repository — see
[CONTRIBUTING.md](CONTRIBUTING.md) for the model contract and conformance checklist.

---

Maintained by the [UMN Choi Lab](https://choi-seongjin.umn.edu/) ·
University of Minnesota Twin Cities
