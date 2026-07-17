# TABenchmark

A shared benchmark for 50 years of traffic assignment models. One harness runs
everything from Frank–Wolfe (1975) and bush-based equilibrium solvers to external
simulators and learned GNN surrogates on identical networks, demand, data
availability, and budgets — and the harness, never the model, recomputes every
scored metric from the link flows a model emits.

This site renders the project documentation and the tutorial notebooks, **executed at
build** (except the engine-gated torch/sumo/dtalite pages, which render un-executed).
Start with the [README](README) for orientation and the repository map,
walk the [tutorials](tutorials/README) in order (simple → complex), and consult the
[architecture](docs/ARCHITECTURE) and [model compendium](docs/MODELS) for depth.

```{toctree}
:maxdepth: 1
:caption: Overview

README
docs/ARCHITECTURE
docs/MODELS
docs/VALIDATION
docs/ROADMAP
docs/REFERENCES
CONTRIBUTING
```

```{toctree}
:maxdepth: 1
:caption: Tutorials
:glob:

tutorials/README
tutorials/*/*
```

```{toctree}
:maxdepth: 2
:caption: API reference

api/index
```

```{toctree}
:maxdepth: 1
:caption: Design records (ADRs)
:glob:

docs/design/*
```
