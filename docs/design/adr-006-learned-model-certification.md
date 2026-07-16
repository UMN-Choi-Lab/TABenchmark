# ADR-006 — Learned (black-box) models: certified by P1, gated by lineage

**Status:** accepted (shipped in v1)
**File:** `docs/design/adr-006-learned-model-certification.md`

## Context

Machine-learning models for traffic assignment — GNNs mapping OD demand + network
to link flows, trained on solver equilibria (Rahman & Hasan 2023; Liu & Meidani 2024;
the Xu et al. 2024 20-city dataset) — are an active line. That literature grades a learned
model on **link-flow error** (MAE / MAPE / correlation) against the solver it imitates.
TABenchmark already has everything needed to grade one *the same way it grades Frank-Wolfe*,
and this ADR records how a learned model plugs in and what the benchmark's certification
adds over the literature's metrics.

## Decision 1 — No new machinery: reuse the wrapper, the gate, and the certificate

The infrastructure a learned model needs already exists and is exercised by tests, so v1
ships the first genuine learned model **without new plumbing**:

- **Wrapper**: a learned model is just a `TrafficAssignmentModel` that emits `link_flows`
  (via `CallableModel` for a closure, or a class like `LearnedSurrogateModel`). It mixes
  freely with white-box solvers in one `run_experiment` grid, keyed by `name`.
- **Certificate (P1)**: the harness recomputes the equilibrium gap **and the
  demand-feasibility audit** from the emitted flows in the exact same `Evaluator` path —
  no learned-vs-classical branch. Approximate/garbage flows are *censored*
  (`feasible=0`, NaN gaps), never crash the run.
- **Fairness gate (P7)**: `Capabilities.trained_on` + `assert_fair_evaluation`, already
  enforced at the top of the per-model loop in `run_experiment`, refuses to score a
  learned model on any scenario whose family (or content hash) is in its training lineage.

## Decision 2 — The reference model is an honest, dependency-free stand-in

`LearnedSurrogateModel` (`learned-surrogate`, paradigm `learned`, deterministic) is a
per-link **ridge regression** predicting each link's equilibrium volume/capacity ratio
from two smooth, bounded transforms of its free-flow all-or-nothing loading, fitted on
solver equilibria of a synthetic network family. It is **not** a GNN and does not pretend to
be — a torch-based graph model (Rahman-Hasan / Liu-Meidani architectures) is the natural
extension, kept out of the core so the benchmark stays numpy/scipy-only. Its purpose is to
exercise the full contract (the `learned` paradigm, the `trained_on` gate, identical
certification), not to win. Being a per-link predictor it does not enforce flow conservation,
which is exactly why it makes the point below. Like every UE solver it needs each
positive-demand OD pair reachable (its feature step raises otherwise); the one-time offline
training cost is reported as `training_sp_calls`/`training_wall_ms` provenance rather than
hidden, and per-solve `wall_ms` measures inference only.

## Decision 3 — The train/test split: synthetic → TNTP (stricter than the field)

The surrogate trains on a **synthetic** random-network family (`trained_on =
"synthetic-net"`) and is evaluated on the **disjoint TNTP** scenarios. There is no shared
network identity, so the fairness gate has real teeth and the comparison is leakage-free —
stricter than the ML-TA norm, which usually trains and tests on the *same* topology with
only demand varied. (The Xu et al. 2024 real-city dataset, disjoint from TNTP and CC-BY, is
now shipped as the *cross-domain* axis — integrated download-on-demand, not vendored; adr-033.)

## What this buys — the methodological point

The ML-TA literature reports link-flow error/correlation; it almost never recomputes the
equilibrium gap of the predicted flows. TABenchmark does, and the two questions diverge.
Across the four TNTP test networks the surrogate's correlation with the best-known
equilibrium ranges widely — **0.63 (Sioux Falls), 0.87 (Barcelona), 0.93 (Winnipeg), 0.99
(Anaheim)** — yet it is censored **`feasible=0` on every one of them**, because a per-link
predictor does not route the actual demand (its node-balance residual is orders of magnitude
above tolerance). Even its best case (Anaheim, correlation 0.99, ~12% *demand-weighted*
MAPE — the unweighted per-link MAPE is far higher) does not certify. Link-flow accuracy is
not a certificate, and certifying a learned model's flows the same way as a solver's —
recomputing the gap and the conservation audit from `link_flows` alone (P1) — is the
contribution. A conservation-aware learned model would clear the audit and then be scored on
its (expected non-trivial) equilibrium gap; that, and a real GNN, are the follow-ups.

## Consequences

- **New:** `LearnedSurrogateModel` (registered, CLI-reachable); a synthetic random-network
  generator (training data only, `family="synthetic-net"`). No new certificate, no scenario
  schema change, no new dependency.
- **Unchanged:** the Evaluator, the fairness gate, every hash, and all 189 prior tests
  (197 total with the 8 new).
- **Gaps deliberately left:** `inputs_required`/`outputs` remain declarative (not enforced);
  a learned model that only sees link counts is not yet *structurally* fenced the way T2 is.
  A torch GNN and conservation-aware learned outputs are future work; the Xu 2024
  cross-domain set is now shipped (adr-033).
