# TABenchmark tutorials

One notebook per benchmark unit — every registered model and estimator, plus each
parallel-track solver — showing how to run it through the **public** `tabench` API and,
above all, how the harness *certifies* the result.

**Follow them in order.** Notebooks are numbered **simple → complex** (PI directive,
2026-07-16): `tutorials/<NN>-<track>/<MM>-<unit>.ipynb`. The numeric prefixes make the
GitHub directory listing itself read as the learning path — start at `01-static/01-aon`
and walk up the vintage ladder. Track folders mirror the `src/tabench` parallel modules,
so a track's notebooks and its code land in the same review.

## The honesty rule

**A notebook never claims a number it does not compute in that cell.** Every scored
quantity is recomputed live by the track's P1 evaluator from the outputs the model
emitted, in the cell where it is claimed; model self-reports are shown only as
provenance and diffed against the certificate as an honesty check. Analytic anchors are
recomputed, never quoted. Plots draw only quantities certified in the notebook — through
the house visualizer `tabench.viz` where the artifact is road link flows / an OD matrix;
non-road tracks (cumulative diagrams, occupancy series, `(x,t)` fields, transit
multigraphs) plot the certified artifact directly with plain matplotlib, stating the
reason in the Visualize cell's markdown (adr-035 Decision C).

## Running them

```bash
pip install -e '.[dev,viz,tutorials]'                # nbclient/nbformat/ipykernel + matplotlib
jupyter lab tutorials/01-static/05-bfw.ipynb         # interactive
jupyter nbconvert --to notebook --execute tutorials/01-static/05-bfw.ipynb --stdout  # headless
```

Notebooks are committed **stripped** (no cell outputs, no execution counts) so they can
never drift out of sync with the code — the gate is *execution success plus the
in-notebook certified asserts*, never output identity, once execution is wired into CI.
**As shipped at C2**, every CI leg enforces existence + stripping + metadata + numbering
(`tests/test_tutorials.py` runs unconditionally, no env flag needed for those checks);
the notebook-EXECUTION test (`test_notebook_executes`, gated on `TABENCH_RUN_TUTORIALS=1`)
is currently **collected but skipped on every leg** — no workflow sets the flag or
installs the `tutorials` extra yet. That wiring (Python 3.12 leg:
`pip install -e '.[dev,viz,tutorials]'` + `TABENCH_RUN_TUTORIALS=1`; torch / sumo /
dtalite legs: `-k` filters on their extra-gated notebooks) lands in the
immediately-following CI commit — until then, run
`TABENCH_RUN_TUTORIALS=1 pytest tests/test_tutorials.py -k executes` locally to get the
execution guarantee this section describes.

## Enforcement

`tests/test_tutorials.py` makes "a new model ships a tutorial" mechanical: every key of
`MODEL_REGISTRY ∪ ESTIMATOR_REGISTRY ∪ DYNAMIC_ESTIMATOR_REGISTRY` and every unit in the
import-anchored 11-unit parallel-track manifest must map to a notebook (or a same-ADR
`covers` sibling), notebooks must be stripped, and `metadata.tabench` must match the
folder. A registered unit with no notebook turns the suite red. Guarded units register
only where their extra is installed, so enforcement is automatically environment-correct.

The coverage gate matches a unit `u` at `tutorials/*/[0-9][0-9]-{u}.ipynb`, and
`metadata.tabench` carries the **bare** names — `unit` = the registry key, `track` = the
folder name *sans* its numeric prefix — so a renumber never touches notebook content.
`test_notebook_numbering_is_ordered` binds the ordering: each track folder carries its
canonical number (unique across folders), and within a folder the file numbers are unique
and contiguous from `01`, so the learning path can never silently drift or collide.

Each notebook carries `metadata.tabench = {track, unit, requires_extra, covers}`; the CI
executor and the coverage gate key off that, not off filenames.

### Same-ADR `covers` folds

Mechanical folds — `metadata.tabench.covers` mirrors `tests/test_tutorials.py::_COVERS`
exactly (test-pinned, see Enforcement above):

| notebook | also certifies | why |
|---|---|---|
| `03-estimation/01-gls` | `prior` | ADR-002 static T2: the stale prior is the baseline the GLS estimator improves on |
| `03-estimation/07-od-dynamic` | `od-dynamic-sim`, `od-dynamic-seq`, `prior-profile` | ADR-023 within-day dynamic estimation — one notebook, three estimator variants |

`08-tdta/01-pm-td` is NOT a registry fold (no `pm-td-ue`/`pm-td-so` enforced units exist,
`metadata.tabench.covers == []`) — it is one notebook exercising both of `solve_td_ue`'s
and `solve_td_so`'s objectives over the single ADR-031 path-marginal task.

## Index (the learning path, in order)

`extra` blank = core install. Runtimes are wall-clock on the author's box (kernel start
~2–3 s dominates the small ones); treat them as order-of-magnitude, re-measured in CI.
Rows marked *(batch NN)* are planned in that internal batch and enforced via the
shrinking allowlist until they ship; their within-track numbers are assigned here so
forward-links and later batches stay consistent.

### `01-static/` — road static assignment (T1), the model-evolution ladder
| # | unit | extra | runtime | status |
|---|---|---|---|---|
| 01 | `aon` | | 3.9 s | shipped |
| 02 | `msa` | | 4.0 s | shipped |
| 03 | `fw` | | 4.0 s | shipped |
| 04 | `cfw` | | 3.5 s | shipped |
| 05 | `bfw` | | 3.9 s | shipped |
| 06 | `gp` | | 3.9 s | shipped |
| 07 | `oba` | | 3.8 s | shipped |
| 08 | `algb` | | 3.8 s | shipped |
| 09 | `tapas` | | 3.6 s | shipped |
| 10 | `sue-msa` | | 4.2 s | shipped |
| 11 | `sue-probit-msa` | | 4.3 s | shipped |
| 12 | `so-bfw` | | 3.7 s | shipped |
| 13 | `fw-elastic` | | 3.9 s | shipped |
| 14 | `evans` | | 3.9 s | shipped |
| 15 | `br-ue` | | 3.6 s | shipped |
| 16 | `sc-tap` | | 3.5 s | shipped |
| 17 | `vi-asym` | | 3.9 s | shipped |
| 18 | `multiclass` | | 3.6 s | shipped |
| 19 | `learned-surrogate` | | 3.6 s | shipped |

### `02-day-to-day/` — day-to-day dynamics
| # | unit | extra | runtime | status |
|---|---|---|---|---|
| 01 | `dtd-swap` | | 6.6 s | shipped |
| 02 | `dtd-swap-sue` | | 5.5 s | shipped |
| 03 | `dtd-link` | | 5.1 s | shipped |
| 04 | `dtd-friesz` | | 5.2 s | shipped |
| 05 | `dtd-horowitz` | | 5.7 s | shipped |
| 06 | `dtd-stochastic` | | 7.1 s | shipped |
| 07 | `dtd-unifying` | | 5.9 s | shipped |

### `03-estimation/` — T2 OD estimation
| # | unit | extra | runtime | status |
|---|---|---|---|---|
| 01 | `gls` (covers `prior`) | | 7.7 s | shipped |
| 02 | `spiess` | | 12.5 s | shipped |
| 03 | `vzw-entropy` | | 7.3 s | shipped |
| 04 | `od-congested` | | 10.3 s | shipped |
| 05 | `spsa` | | 12.3 s | shipped |
| 06 | `od-kalman` | | 7.1 s | shipped |
| 07 | `od-dynamic` (covers `od-dynamic-sim`, `od-dynamic-seq`, `prior-profile`) | | 5.3 s | shipped |

### `04-transit/`
| # | unit | extra | runtime | status |
|---|---|---|---|---|
| 01 | `transit-strategy` | | 5.2 s | shipped |

### `05-dnl/` — dynamic network loading
| # | unit | extra | runtime | status |
|---|---|---|---|---|
| 01 | `ctm` | | 5.5 s | shipped |
| 02 | `ltm` | | 4.8 s | shipped |
| 03 | `godunov` | | 5.3 s | shipped |
| 04 | `node-model` | | 5.0 s | shipped |

### `06-bottleneck/` · `07-dta/` · `08-tdta/` — analytic dynamics
| folder / # | units | extra | runtime | status |
|---|---|---|---|---|
| `06-bottleneck` 01 | `vickrey` | | 5.0 s | shipped |
| `06-bottleneck` 02 | `vi-due` | | 5.2 s | shipped |
| `07-dta` 01 | `merchant-nemhauser` | | 5.5 s | shipped |
| `07-dta` 02 | `lp-so-dta` | | 5.3 s | shipped |
| `08-tdta` 01 | `pm-td` | | 37.5 s | shipped |

### `09-newell/`
| # | unit | extra | runtime | status |
|---|---|---|---|---|
| 01 | `newell-3det` | | 5.4 s | shipped |

### `10-learned/` — optional-extra learned models
| # | unit | extra | status |
|---|---|---|---|
| 01–02 | `implicit-ue-nn`, `het-gnn` | torch | *(batch 11)* |

### `11-external/` — external-engine adapters
| # | unit | extra | status |
|---|---|---|---|
| 01 | `sumo-marouter` | sumo | *(batch 12)* |
| 02 | `sumo-duaiterate` | sumo | *(S2 / extdyn-impl)* |
| 03 | `dtalite-tap` | dtalite | *(batch 12)* |
| 04 | `spsa-sumo` | sumo | *(batch 12)* |

### `12-data/` · `13-experiments/` — tours (outside the enforced unit set)
| folder / # | notebook | extra | status |
|---|---|---|---|
| `12-data` 01–02 | `xu2024`, `bo4mob` | network download | *(batch 13; never default-prefetch)* |
| `13-experiments` 01 | `profiles` | | *(batch 13)* |
