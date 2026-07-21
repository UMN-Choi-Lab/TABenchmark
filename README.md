# TABenchmark

**A shared benchmark for 50 years of traffic assignment models.** One harness runs
everything from Frank–Wolfe (1975) and bush-based equilibrium solvers to external
simulators and learned GNN surrogates on identical networks, demand, data availability,
and budgets. The harness — never the model — recomputes every scored metric from the
link flows a model emits, so a black box cannot self-report its way onto the leaderboard.
Full documentation and executed tutorials: **<https://tabenchmark.readthedocs.io>**.

## Install

```bash
pip install -e .        # core: numpy + scipy + pyyaml only, Python >= 3.10
```

Optional extras (the core imports and runs without any of them; models behind a missing
extra simply do not register):

| Extra | Adds | Note |
|---|---|---|
| `torch` | learned models `implicit-ue-nn`, `het-gnn` | for CPU-only, install torch from the CPU wheel index first |
| `sumo` | adapter `sumo-marouter`, guarded estimator `spsa-sumo`, EDOC row `sumo-duaiterate` | `eclipse-sumo` wheel bundles the binaries (`sumo-duaiterate` is an EDOC row, not in `tabench list`) |
| `dtalite` | external engine `dtalite-tap` | PyPI `DTALite` wheel |
| `viz` | house visualizer `tabench.viz` (matplotlib) | never imported by the core |
| `tutorials` | execute the notebooks (`nbclient`, `ipykernel`) | drives `TABENCH_RUN_TUTORIALS=1` |
| `docs` | build the readthedocs.io site (Sphinx + MyST) | needs `viz` too: `pip install -e ".[docs,viz]"`, then `python docs/build_site.py` |
| `dev` | pytest + ruff | `pytest -q` runs the full suite |

## Quickstart (60 seconds)

```bash
python demos/demo_quickstart.py   # built-in Braess scenario, no download
```

```text
Scenario: braess (hash cf00f411cdccec88)
model             certified rel. gap  feasible
----------------------------------------------
aon                        1.912e-01         1
msa                        1.577e-03         1
fw                         7.188e-14         1
cfw                       -2.060e-16         1
bfw                       -2.060e-16         1
toy-surrogate                    nan         0
```

`toy-surrogate`'s flows fail the demand-feasibility audit, so its gap is censored to
`nan` rather than scored — a black box can neither crash the run nor top the leaderboard.

Wrap your own model in three lines — the harness certifies it identically:

```python
from tabench import CallableModel, Budget, load_scenario, run_experiment

model = CallableModel(fn=my_predict, name="my-gnn", trained_on=("tntp-small-v1",))
result = run_experiment(load_scenario("braess"), [model], Budget(iterations=1))
```

`my_predict(scenario, rng)` returns link flows; the harness recomputes the relative gap
from them, censors it (`nan`) if the flows fail the demand-feasibility audit, and refuses
to run at all if `trained_on` intersects the scenario's training lineage. CLI equivalent:

```bash
tabench run --scenario siouxfalls --models fw,cfw,bfw --iterations 300 --out results/
tabench list      # all scenarios, models, and estimators registered in this install
```

## Repository map

```text
TABenchmark/
├── src/tabench/           # the package (core is numpy/scipy-only)
│   ├── core/              # Scenario (frozen, content-hashed), Capabilities, Budget, Trace, RNG
│   ├── data/              # TNTP parser; checksummed on-demand fetchers (-> ~/.cache/tabench)
│   ├── models/            # road assignment models: AON/MSA/FW/CFW/BFW, bush & path UE, SUE,
│   │   │                  #   SO, elastic, combined, day-to-day, BR/side-constrained/VI,
│   │   │                  #   multiclass, learned surrogates
│   │   └── adapters/      # CallableModel wrapper; external engines (sumo-marouter, dtalite-tap)
│   ├── observe/           # observation processes: FullOD, LinkCounts, StalePriorOD + identifiability
│   ├── estimation/        # T2 OD estimation: entropy, GLS, Spiess, SPSA, Yang'92, DN-Kalman,
│   │                      #   within-day dynamic (Cascetta'93), guarded spsa-sumo
│   ├── metrics/           # the certifiers — every scored metric recomputed here from emitted flows
│   ├── experiments/       # T1 grid + T2 estimation runners, provenance manifests, profiles
│   ├── dnl/               # dynamic network loading: CTM, LTM, Godunov, Tampère node model
│   ├── dta/  tdta/  bottleneck/  # analytical & time-dependent DTA: Merchant–Nemhauser,
│   │                      #   Ziliaskopoulos LP, Vickrey, VI-DUE, Peeta–Mahmassani
│   ├── transit/           # Spiess–Florian optimal strategies
│   ├── newell/            # three-detector traffic-state estimation
│   ├── edoc/              # external-dynamic-engine observational certificate (adr-036/037)
│   ├── viz.py             # plotting helpers ([viz] extra)
│   └── cli.py             # tabench fetch | list | run
├── scenarios/             # declarative YAML scenario cards (ladder 0braess → 4winnipeg, + xu2024;
│                          #   bo4mob is fetch-and-cite, registered separately — not a YAML card)
├── tutorials/             # 58 numbered notebooks, simple → complex: one executed, certified
│                          #   notebook per unit (<NN>-track/<MM>-unit.ipynb; committed with
│                          #   outputs, bar the Java-gated matsim page)
├── demos/                 # demo_quickstart.py, demo_profiles.py
├── tests/                 # 1000+ tests: analytic anchors, published-oracle regressions (pytest -q)
├── tools/                 # doc generators (model compendium, references, evolution graph)
├── docs/                  # architecture, model compendium, validation, roadmap, references, 37 ADRs
└── CITATION.cff, LICENSE  # MIT
```

Orientation facts (for humans and agents):

- Every scored metric is computed in `src/tabench/metrics/`, never by a model; model
  self-reports are provenance only.
- Scenarios are frozen and content-hashed; benchmark networks are fetched on demand,
  SHA-256-checksummed, cached under `~/.cache/tabench`, and never redistributed.
- Tests: `pytest -q` (numpy core needs no extra). CI runs a py3.10/3.12 core matrix
  plus dedicated `torch`, `sumo`, `dtalite`, and `docs` (site build, warnings-as-errors) jobs.
- Optional-extra models register only when their extra is installed; `import tabench`
  stays numpy/scipy-only either way.

## Models, leaderboard, validation

The full roster — road assignment across five decades, day-to-day dynamics, a transit
optimal-strategy model, static and within-day-dynamic OD estimators, and DNL/DTA/state-
estimation tracks — ships today, each certified by the same harness and validated against
an independent oracle (analytic anchors, published best-known flows, cross-solver agreement).

- What each model does and its lineage: [docs/MODELS.md](docs/MODELS.md) ·
  [evolution graph](docs/model-evolution.svg)
- Certified results and per-model oracles: [docs/VALIDATION.md](docs/VALIDATION.md)
- Staged plan toward the full canon: [docs/ROADMAP.md](docs/ROADMAP.md)

## Documentation

The docs site (<https://tabenchmark.readthedocs.io>) carries the tutorials rendered and
**executed at build** (bar the engine-gated torch/sumo/dtalite pages, which render
un-executed), the design principles, the CLI reference, leaderboards, and data licensing.
Canonical sources in-repo: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (design principles
P1–P9 and the object model), [docs/REFERENCES.md](docs/REFERENCES.md) (verified reference
canon, BibTeX in [docs/references.bib](docs/references.bib)), and 37 architecture decision
records in [docs/design/](https://github.com/UMN-Choi-Lab/TABenchmark/tree/main/docs/design)
(also in the site sidebar). The one PI-only step to connect the repo to
readthedocs.io is [docs/RTD_SETUP.md](docs/RTD_SETUP.md).

## Data licensing

Benchmark networks are **never redistributed**: they are fetched on demand from
commit-pinned URLs, checksum-verified, and cited (`tabench fetch <scenario>` prints the
citation). Sources: [TransportationNetworks](https://github.com/bstabler/TransportationNetworks)
(academic use), Xu et al. 2024 20-US-cities (CC BY 4.0, adr-033), and the lab's
[BO4Mob](https://github.com/UMN-Choi-Lab/BO4Mob) instances (MIT, adr-034). Package code is MIT.

## Citing

Cite the repository via [CITATION.cff](CITATION.cff), plus the original references of any
model or dataset you evaluate — all carry verified BibTeX in
[docs/references.bib](docs/references.bib).

## Contributing

New models, networks, and observation levels are the purpose of this repository — the
model contract and conformance checklist are in [CONTRIBUTING.md](CONTRIBUTING.md).

---

Maintained by the [UMN Choi Lab](https://choi-seongjin.umn.edu/) ·
University of Minnesota Twin Cities · MIT License
