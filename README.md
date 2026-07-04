# TABenchmark

**A shared benchmark for 50 years of traffic assignment models.**

Traffic assignment research spans from Wardrop's principles (1952) and Beckmann's convex
program (1956) through Frank–Wolfe, bush-based solvers, stochastic user equilibrium, and
dynamic traffic assignment, to today's GNN surrogates and end-to-end learned models —
yet there has never been a shared testbed on which all of them can be compared under
identical networks, demand, data availability, and budgets. TABenchmark is that testbed:

- **One harness, every vintage.** A minimal model contract plus capability declarations
  lets a 200-line Frank–Wolfe, a subprocess-wrapped simulator, and a trained neural
  surrogate run in the same experiment matrix.
- **Certified, not self-reported.** The harness recomputes every scored metric from
  emitted link flows — the equilibrium relative gap is a property of the flows, so even
  a black box receives an externally certified gap (and a feasibility audit first).
- **Data levels are first-class.** Full OD tables, noisy OD, per-period link counts on
  sensor subsets, sampled trajectories: every observation process is a seeded projection
  of stored ground truth, with identifiability conditions reported per configuration.
- **Fair across 70 years.** Hardware-free budgets (iterations / shortest-path calls /
  evaluations), training-lineage gates against test contamination, content-hashed
  scenarios, and full run manifests.

Built on the verified canon of **172 references** spanning 12 model families
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

Run the classic Sioux Falls instance (data downloaded on demand, checksummed):

```bash
tabench run --scenario siouxfalls --models fw,cfw,bfw --iterations 300 --out results/
tabench run --scenario winnipeg --models bfw --iterations 500 --target-gap 1e-4 --out results/
tabench list                            # available scenarios and models
```

Wrap **your** model in three lines — the harness scores it identically:

```python
from tabench import CallableModel, Budget, load_scenario, run_experiment

model = CallableModel(fn=my_gnn_predict, name="my-gnn", trained_on=("tntp-small-v1",))
result = run_experiment(load_scenario("siouxfalls"), [model], Budget(iterations=1))
```

(If `trained_on` intersects the evaluation scenario's lineage, the harness refuses to
run it — that's the point.)

## What ships in v0.x

| Component | Contents |
|---|---|
| Core | `Scenario` (frozen, content-hashed, optional SUE θ), `Capabilities`, `Budget` (incl. Boyce-style convergence target), `Trace`, spawn-key RNG schema |
| Data | Defensive TNTP parser, commit-pinned checksummed fetcher, per-network units metadata; scenario ladder Braess → Sioux Falls → Anaheim → Barcelona → Winnipeg (+ analytic two-route SUE anchor) |
| Models | All-or-nothing, MSA, Frank–Wolfe, conjugate & bi-conjugate FW (Mitradjieva & Lindberg 2013), logit SUE via Dial-STOCH + MSA, black-box `CallableModel` adapter |
| Metrics | Certified relative gap / average excess cost / Beckmann objective, SUE fixed-point residual ([ADR-001](docs/design/adr-001-logit-sue-dial-certificate.md)), feasibility audit, flow RMSE vs best-known |
| Observe | `FullOD`, `LinkCounts` (sensor mask × periods × noise), Hazelton identifiability check |
| Experiments | Grid runner, CSV results, full provenance manifests |
| Tests | Analytic Braess UE + two-route logit-SUE oracles; best-known-solution regressions on Sioux Falls, Anaheim, Barcelona, Winnipeg; conjugacy-identity and golden-hash regressions |

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

The staged roadmap toward the full canon (bush-based solvers, probit SUE, DTA,
day-to-day, estimation tasks, engine adapters) is in [docs/ROADMAP.md](docs/ROADMAP.md).

## Data licensing

Benchmark networks come from
[TransportationNetworks](https://github.com/bstabler/TransportationNetworks), donated
for academic research. TABenchmark **never redistributes** these files: they are fetched
on demand from a commit-pinned URL, verified against SHA-256 checksums, cached locally,
and cited (`tabench fetch <scenario>` prints the citation). The package code is MIT.

## Citing

If you use TABenchmark, please cite this repository (see `CITATION.cff`) and the
original references of any model or dataset you evaluate — every entry in
[docs/REFERENCES.md](docs/REFERENCES.md) carries verified BibTeX in
[docs/references.bib](docs/references.bib).

## Contributing

New models, networks, observation levels, and reference implementations from the
community are the purpose of this repository — see
[CONTRIBUTING.md](CONTRIBUTING.md) for the model contract and conformance checklist.

---

Maintained by the [UMN Choi Lab](https://choi-seongjin.umn.edu/) ·
University of Minnesota Twin Cities
