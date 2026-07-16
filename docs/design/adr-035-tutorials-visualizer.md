# ADR-035: tutorials-visualizer — the house visualizer and the per-unit tutorial notebooks

**Status:** accepted (implemented: `tabench.viz` + `demo_quickstart --viz` + the tutorials
infrastructure, enforcement, and 44 notebooks across static, day-to-day, estimation, transit,
dnl, bottleneck, dta, tdta, and newell — every core track; the extras (`10-learned`,
`11-external`) and `12-data`/`13-experiments` tours remain, under this same design)
**Date:** 2026-07-16
**Deciders:** S0 sprint — the visual layer plus the "every model ships a tutorial" deliverable
**File:** `docs/design/adr-035-tutorials-visualizer.md`

## Context

Two directives (2026-07-16) came due together. First, the benchmark needed a good visualizer —
a `--viz` option on the quickstart that, for each model plus an in-run ground truth, shows the
OD demand and per-link flow counts (turning the P1 certificate story from a printed table into
a picture; `docs/ARCHITECTURE.md`'s demos line had so far promised only a planned demo ladder).
Second, the PI's standing rule "whenever new models arrive, a tutorial must ship" needed to be
made mechanical — a registered model without a runnable, honest tutorial should turn the suite
red, not rely on reviewer memory. This ADR records the design of both halves and the fix-batch
findings that hardened them, because both are new public surfaces (`import tabench.viz`, the
`tutorials/` tree) that future contributors extend.

The governing constraint is the same one every other ADR obeys: the numpy/scipy core must import
without any optional dependency, and no scored quantity may be claimed without being recomputed
by the harness. The visualizer and the notebooks are both **downstream** of the certificate —
they render and teach it, they never redefine it.

## Decision A — `tabench.viz`, an optional visual layer

* **Optional extra, never in the core.** matplotlib ships behind `pip install tabench[viz]`
  (`matplotlib>=3.8`, floor only), joining the `[torch]`/`[sumo]`/`[dtalite]` extras. `tabench.viz`
  is **never** imported by `tabench`'s top-level `__init__`, so `import tabench` stays
  numpy/scipy-only; the module guards its matplotlib import with the house pattern (swallow only a
  missing-matplotlib `ModuleNotFoundError`, re-raise any other) and every plotting entry point
  raises a clear install-hinted error when the extra is absent. Pure geometry (`node_positions`)
  works without matplotlib.
* **Deterministic layout resolution.** `node_positions(network, pos=None)` resolves in a fixed
  order — explicit `pos` > cached TNTP node-coordinate file (only when the registry key equals the
  network name AND the file supplies *exactly* the network's node set) > a hand layout for the
  built-ins (Braess is the canonical diamond) > a deterministic layered-BFS fallback (x = hop
  distance from the zone nodes). Identical inputs give byte-identical positions; there is no
  randomness anywhere. The cache path degrades to the fallback — never crashes — on a corrupt,
  non-UTF-8, NaN-coordinate, or wrong-size cached file.
* **The display-length lesson (M1).** A picture that silently drops most of a network is the
  plotting analog of a false certificate. A fixed point-based arrow shrink erased every link
  shorter than the shrink budget — on real Sioux Falls WGS84 coordinates, 70 of 76 links vanished.
  The fix trims each arrow in **data space** to the middle ~68% of its link (always visible and
  correctly directed at any display scale), and scales node markers/fonts *down* and the figure
  *up* with node count so a 24-node Sioux Falls or a 100-node fallback stays legible while Braess
  is unchanged.
* **The span-floor root cause (R1/R2).** A render review found Sioux Falls still crammed into the
  canvas centre with outlying nodes (1, 2, 13) reading as disconnected. Root cause: `_span` floored
  the data extent at `1.0` *unconditionally*, but Sioux Falls's WGS84 extent is ~0.06–0.11, so every
  span-relative quantity — axis margin, reverse-link perpendicular offset, label offset — was
  inflated ~9× (the offset even exceeded the median link length, so paired arrows pointed sideways).
  The floor now applies **only** to a fully degenerate (all-coincident) layout; the true extent is
  used otherwise. Braess (extent 2) is untouched. On top of the root-cause fix: **(R1)** each arrow's
  per-end trim is capped at `_INSET_CAP_C ·` (node-marker radius in data units) so genuinely long
  links reach their nodes while short links keep the proportional 16 % trim. `c` is calibrated
  **empirically, not to the "small c ~ 1.5–2.5" first guess**: in the house framing the marker is
  small enough that Braess's longest link has an uncapped trim of ~3.57 marker-radii, so any smaller
  `c` would shorten Braess's arrows and change "the render that is right"; `c = 4.0` sits just above
  it — the built-ins keep byte-identical proportional trim, and only long links (Sioux Falls's
  outlying-node links, ~6.9 marker-radii uncapped) are pulled in to ~4.9. **(R2)** the figure is
  sized to the **data aspect ratio** (`_figsize`), since with `aspect='equal'` a figure whose shape
  differs from the data's leaves the network in a tall/wide empty band; the axis margin is tightened
  to `0.10·span`. Each fix is regression-pinned (`test_r1_*`, `test_r2_figure_matches_data_aspect`).
* **Library-style figures.** Public calls build `matplotlib.figure.Figure` objects (not
  `plt.subplots`), so they never grow the global pyplot `Gcf` registry — 25 sequential calls leave
  `plt.get_fignums()` empty and a mid-render exception cannot leak an uncloseable figure. The
  notebooks therefore `display(fig)` explicitly rather than `plt.show()`.
* **House style, applied locally.** Light surface `#fcfcfb`, text `#0b0b0b`/`#52514e`, grid
  `#e5e4e0`, series `#2a78d6` then `#1baf7a` extended tastefully, recessive spines, no dual axes.
  Applied through an `rc_context` at figure creation plus explicit per-artist colours — never a
  global rcParams mutation, so a user's own session is untouched. Public API:
  `plot_network_flows`, `plot_od_demand`, `plot_flow_scatter`, `compare_models`.

## Decision B — `demo_quickstart --viz`

The demo grows a `--viz` flag (default OFF; `--viz-out DIR`). With the flag OFF, stdout is
byte-identical to before (regression-pinned). With it ON, the demo solves a **ground truth in the
same run** — a generous-budget `bfw`, labelled by its OWN certified relative gap from the same
`run_experiment` machinery, never a pasted analytic number — and writes three PNGs (OD heatmap,
per-model + GT link-flow panels, model-vs-GT scatter) through `tabench.viz`. Agg is forced before
pyplot loads (it is a script, not a notebook); a missing matplotlib fails LOUDLY with the install
hint (crash-vs-censor discipline applied to UX). The scatter is the P1 story made visual — and it
tells it honestly: the CONVERGED solvers cluster on `y = x` while the capacity-blind `aon`
baseline sits farthest off it, *exactly* as its ~1.9e-1 certified gap says (off-diagonal is the
gap, not censorship; `aon` is feasible and honestly scored).

## Decision C — the tutorials architecture

* **Layout — numbered simple→complex.** `tutorials/<NN>-<track>/<MM>-<unit>.ipynb`
  (PI directive 2026-07-16): numeric prefixes on both folders and files so the GitHub
  directory listing reads as the learning path (`01-static` → `13-experiments`; within
  `01-static`, the MODELS.md vintage ladder `01-aon` … `19-learned-surrogate`). Folders
  otherwise mirror the `src/tabench` parallel modules, plus `tutorials/README.md`. Each
  notebook carries `metadata.tabench = {track, unit, requires_extra, covers}` with the **bare**
  registry names (`unit`, and `track` = folder *sans* prefix), so a renumber never edits notebook
  content; the coverage gate globs `tutorials/*/[0-9][0-9]-{unit}.ipynb` and a numbering-integrity
  test binds each folder's canonical number (unique) and within-folder contiguity from `01`.
* **Stripped outputs, not committed outputs.** Notebooks ship with outputs cleared
  (`outputs == []`, `execution_count is None`). Committed executed outputs read well on GitHub but
  drift silently when code changes; stripped notebooks can never lie. The gate is therefore
  *execution success plus the in-notebook certified asserts*, never output identity — CI
  re-executes from a cleared state. Rendering for the docs site is a later, additive concern
  (CI artifacts / HTML), never a weaker gate.
* **The template.** title + what/why + bibkey (verified canon only) + ADR → the honesty rule
  verbatim ("A notebook never claims a number it does not compute in that cell") → an extra-guard
  setup with `%matplotlib inline` (never `matplotlib.use("Agg")` in-kernel — it suppresses inline
  capture) → load the scenario, printing `content_hash()[:16]` (P2) → solve via the public API,
  self-reports labelled "provenance only" → **certify in-cell** through the track's P1 evaluator
  with mandatory asserts (feasibility gate, metric bound, self-report ≈ certified for white boxes,
  analytic anchors RECOMPUTED never quoted) → visualize → takeaways. Core notebooks run in ≤ ~60 s
  (the static track measured 3.5–4.3 s each), seeds pinned, no timestamps in stdout.
* **The viz rule.** Notebooks plot via `tabench.viz` where the track's artifacts are road link
  flows / OD matrices; non-road tracks (cumulative diagrams, occupancy series, (x,t) fields,
  transit multigraphs) use plain matplotlib, with the reason stated in the Visualize cell's
  markdown.
* **Executor.** `nbclient` driven from a parametrized pytest test (`kernel_name="python3"`,
  `timeout=120`), gated on `TABENCH_RUN_TUTORIALS=1` so laptops skip by design once CI hard-runs
  — the `TABENCH_REQUIRE_DATA` discipline of `tests/conftest.py`. `nbmake` was rejected (glob-
  collected, so it cannot *demand* a missing notebook); `jupyter nbconvert --execute` stays the
  human CLI.
* **CI wiring (C3, not C2).** The `tutorials` extra
  (`nbclient>=0.10, nbformat>=5.10, ipykernel>=6.29, matplotlib>=3.8`) and the core-job install of
  `.[dev,viz,tutorials]` with `TABENCH_RUN_TUTORIALS=1` on the 3.12 leg ship in the commit
  immediately following the tutorials commit (C2); the extras jobs (torch/sumo/dtalite) append
  `test_tutorials.py` with `-k` filters there too. No new CI job. **At C2, `ci.yml` is untouched**
  (it carries a concurrent workstream's uncommitted edits, so C2 does not stage the whole file):
  `test_notebook_executes` is collected but skipped on every leg until C3 lands, while
  existence/stripping/metadata/numbering enforcement (the rest of `test_tutorials.py`) runs
  unconditionally on every leg from C2 onward. These install-line edits were ALSO deliberately
  deferred earlier while the viz commit (C1) was under review, to keep that review's
  pip-install/actionlint repros stable — the C3 gap is a second, independent deferral for the
  commit-ordering reason above, not a repeat of the C1 one.

## Decision D — the enforcement test (`tests/test_tutorials.py`)

Coverage is a gate over `MODEL_REGISTRY ∪ ESTIMATOR_REGISTRY ∪ DYNAMIC_ESTIMATOR_REGISTRY` plus an
**import-anchored 11-unit manifest** for the parallel tracks (which carry no registry — each entry
imports its solver/class, so a rename breaks the test at import). Same-ADR `covers` folds let one
notebook certify sibling units (`gls` covers `prior`; `od-dynamic` covers the three dynamic
estimators). A **shrinking allowlist** — the single obvious module-level constant, from which
batches may only ever REMOVE — carries not-yet-written units and reaches empty when the last batch
lands, at which point the gate is fully strict; a companion test fails if an allowlisted unit
already has a notebook (drift) or is not a real unit (typo). Guarded units register only where
their extra is installed, so enumeration is automatically environment-correct.

DNL open-endedness is closed by a `LinkModel`/`NodeModel` subclass walk that must be a subset of
the manifest. `__subclasses__()` has **two** leakage paths, both closed: (1) the private reference
link model registers only when a dnl test imports it lazily — closed by importing every `dnl`
submodule up front (deterministic set) plus waiving the reference/boundary classes; (2) ad-hoc
test-LOCAL subclasses defined in other test functions linger until garbage collection — closed by
filtering the walked set to `cls.__module__.startswith("tabench.")`. Both are pinned (a scratch
subclass defined inside the test, not gc'd, must not trip the gate). The execution gate probes
each extra with `importlib.util.find_spec` — never `import DTALite`, which prints a banner and
ctypes-loads the engine (ADR-029), and whose module name is `DTALite`, not the extra name.

## Fix-batch finding record (S0a three-lens review)

The visualizer shipped, then a three-lens review (visual-honesty / api-infra / demo-contract) ran
and every finding was reproduced and pinned:

* **M1 (visual-honesty)** — Sioux Falls silently dropped 70/76 links to over-shrunk arrows. Fixed
  by data-space arrow trimming + node/figure scaling; pinned by rendering the cached Sioux Falls
  scenario and asserting every link's arrow has nonzero display length.
* **M2 (api-infra)** — a corrupt / non-UTF-8 / NaN / superset cached node file crashed or
  mis-placed the network. Fixed by catching `UnicodeDecodeError`, skipping non-finite rows, and
  requiring the cached node set to equal the network's exactly; pinned per case (each degrades to
  the fallback).
* **M3 (demo-contract)** — the P1-story print falsely said "the certified solvers cluster on the
  line" while `aon` (certified, feasible) was the farthest off-diagonal series. Reworded to name
  the converged solvers and describe `aon` honestly; pinned on the demo stdout.
* Minors (all pinned): flow-array length validation in `compare_models`/`plot_flow_scatter`;
  library-style figures (no pyplot leak); scatter axis including the reference minimum; OD
  finite-masked colour scale under a NaN cell; a visible (not dead) reference-panel background;
  `--viz-out` without `--viz` warning on stderr while stdout stays byte-identical; clear
  `ValueError`s for a missing-node `pos` and an empty `compare_models`; separable scatter markers
  for coincident series.

## Consequences

* One visual style across the demo and every tutorial; a matplotlib-free core preserved (the
  torch-free CI legs remain the live regression, now joined by a matplotlib-blocked import test).
* "A new model ships a tutorial" is mechanical: a registered unit without a notebook turns the
  suite red, in the environment where that unit registers. The tutorial notebook becomes a
  standing per-sprint deliverable (pipeline step 7).
* The golden Braess hash `cf00f411…` is byte-untouched (no scenario/network/metric code changed);
  the visualizer and notebooks are pure downstream consumers of the existing certificate.
* Cost: `tabench.viz` is ~600 lines of matplotlib to maintain, and each future model adds one
  notebook to author and execute. The review record above is the standing checklist for both.
