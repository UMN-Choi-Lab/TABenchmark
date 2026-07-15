# ADR-026 — Heterogeneous-GNN traffic assignment: the third learned model, feasibility as a decode

**Status:** accepted (shipped in v0.2)
**File:** `docs/design/adr-026-het-gnn.md`

## Context — act three of the ADR-006 argument, a feasibility-mechanism gradient

ADR-006 shipped `learned-surrogate` (a per-link ridge regressor) to make *act
one*: **link-flow accuracy is not an equilibrium certificate.** A per-link
predictor emits a flow nobody routed, so it is censored `feasible=0` on every
TNTP net. ADR-025 shipped `implicit-ue-nn` for *act two*: **demand feasibility is
not equilibrium quality.** Liu et al. (2023) bake the equilibrium into the
architecture — the output *is* a fixed point of a learned cost, demand-feasible
by construction — and the harness then recomputes its gap under the true costs.

`het-gnn` (Liu & Meidani 2024) is *act three*, and it completes a **gradient of
feasibility mechanisms** across the three learned models:

| model | conservation | raw emission | feasible score |
|---|---|---|---|
| `learned-surrogate` | none | censored | — |
| `het-gnn` | **soft loss** (`w_c=0.05`) | **censored** | **by an explicit decode** |
| `implicit-ue-nn` | architectural | feasible | by construction |

The decisive machine-verified finding (anchor A2) is that the paper's own
conservation metric `L~_c` **is** the L1/D form of the harness's censoring
statistic `metrics.gaps.node_balance_residual`: the paper trains toward exactly
the quantity the audit thresholds, and its best reported values (`L~_c` 0.02–0.19
of demand) still imply max-node residuals 3–5 orders of magnitude above the
`1e-6·D` feasibility tolerance. So the paper-faithful raw emission is censored
`feasible=0` with certainty, trained or untrained. That censored row is the
honest headline; the decode is what puts a certified gap on the leaderboard.

## Decision 1 — the shipped unit emits TWO checkpoints (option (c))

`solve` records two harness-certified checkpoints (`runner.py` certifies every
checkpoint from the emitted flows, P1 — one CSV row each, nothing self-attested):

1. **Raw (paper-faithful).** `f~ = cap · relu(alpha~)` at `sp_calls=0,
   iterations=0` — the GNN forward needs no shortest path, a genuinely new budget
   point below the ridge's 1. The audit censors it `feasible=0` with its
   recomputed node-balance residual. **This row IS the paper's model.**
2. **Decoded (a repo EXTENSION — NOT in the paper).** The raw flows are projected
   onto the demand-feasible route polytope: `min_h ||Δᵀh − v_raw||²` s.t.
   `h ≥ 0`, `Σ_{r∈od} h_r = D_od`, by projected gradient with per-OD scaled-
   simplex projection over the SAME `implicit_ue._build_routes` column-generated
   route sets. The emission `v = Δᵀh` is demand-feasible by the identical
   `v = Δᵀh` mechanism `implicit-ue-nn` uses, so the two models' certified-gap
   comparison isolates "GNN flow regression + projection" vs "learned-cost fixed
   point" at matched shortest-path budget. `sp_calls = n_cg`, `iterations` = the
   executed decode steps.

**Why (c) and not (a) raw-only or (b) decode-only.** (a) alone adds a third
censored row that duplicates the ridge's leaderboard signal — its only new
content (a lower node-balance residual) lives in an unranked column. (b) alone
hides the paper-faithful behaviour and *overstates* the paper: its output is NOT
feasible (conservation is a soft `w_c=0.05` penalty — verified). (c) costs
nothing (the checkpoint machinery exists), keeps the paper-faithful censored row
visible in the same CSV, and completes the feasibility-mechanism gradient above.

The decode is a **flagged repo extension** wherever it could be mistaken for the
paper's method: the module docstring, this ADR, and `model-specs.json`. The
projected-gradient step is fixed from the Lipschitz bound of `ΔΔᵀ` (power
iteration on the small `ΔᵀΔ`) with **adaptive damping** (step halved whenever the
objective rises — the repo's recurring fixed-point defect, adr-025 review), and
the residual is measured at the emitted iterate. The projection is exact demand
feasibility by construction (node balance ~1e-11·D, machine precision) regardless
of convergence.

## Decision 2 — the LEAN VARIANT is a size-agnostic node-kernel (the equivariance dichotomy)

The paper's node feature is `x_u ∈ R^(|V|+2)` — that node's entire OD-demand row
plus two geographic coordinates. Two machine-verified facts make this the crux of
the lean variant:

- **It is NOT node-permutation equivariant** (anchor: verified max output change
  **21.5** under a consistent relabeling; the identical architecture with
  intrinsic features changes by **0.0**). The dense preprocessing layer mixes the
  node-indexed demand-row axis, so the emitted flows depend on node ordering.
- **Its dimension is `|V|`**, which is exactly why the paper needs *transfer
  learning* or dummy-node *homogenized training* (paper §5.2.2) to change graph
  size — the benchmark's disjoint train (8–14-node synthetic) / test (24–416-node
  TNTP) split is precisely the paper's hardest case.

The lean variant replaces it with the intrinsic per-node
`[production_u, attraction_u, out_degree_u, in_degree_u]` (demand terms
normalized by total demand, degrees by link count; coordinates dropped —
`Network` has no geometry). This is **exactly permutation equivariant** (anchor
A4, measured `1.1e-16 < 1e-8`) and size-agnostic, so ONE trained model runs on
every graph size with no dummy-node padding or retraining. It is the node-level
analogue of `implicit-ue-nn`'s per-link "kernel strategy": **link-kernel there,
node-kernel here — both size-agnostic by different routes**, and that contrast is
the headline of the two-model learned story.

The rest is the lean floor (adr-025 Decision 1, no torch-geometric/DGL):
hand-rolled heterogeneous message passing with plain-torch `index_add_` /
`scatter_reduce`, the per-tail-node attention softmax being a **segment softmax
keyed on the node index** (the `implicit_ue._segment_softmax_load` pattern
transplanted from OD groups to nodes); stacked V-Encoders (virtual OD edges,
adaptive weight `β_e = FFN([x_u‖x_v])` from the endpoints since virtual edges
carry no features — Eq 7-8) then R-Encoders (real links, whose weight here is a
LEARNED affine scalar of the real edge features — a **flagged generalization**
of Eq 9, whose weight is the FIXED sum of the normalized edge features, with
the feature set extended from the paper's `[fft, cap]` to `[fft, cap, b,
power]` — adr-026 review); the edge head `alpha~_a = MLP([o_u‖o_v‖e_a])`
(Eq 10); single head, hidden 16, float64 (the paper's 8 heads / 3-layer
preprocessing / LayerNorm towers are not built). Trained by plain backprop (no
implicit layer → no hypergradient machinery — simpler than adr-025). The budget
point `sp_calls=0` was verified accepted by `BudgetCoords` and un-divided-by
anywhere downstream (already used across the estimation modules).

## Anchors

- **A1 untrained forward (verified).** An untrained HetGNN on the builtin Sioux
  Falls scenario emits finite flow/capacity ratios of shape `(|E_r|,)`.
- **A2 raw censoring + decoded feasibility (verified).** The raw emission is
  `feasible=0` with a harness-recomputed node-balance residual orders above
  tolerance (the paper-faithful honest row); the decoded emission is `feasible=1`
  with a real certified gap. Both rows appear in one CSV.
- **A4 permutation equivariance (verified, `1.1e-16 < 1e-8`).** Under a
  zones-among-zones / non-zones-among-non-zones relabeling (TNTP convention) with
  edge order fixed, the raw ratio per edge is invariant to float precision — the
  size-agnostic featurization's defining property.
- **Decode is a projection (verified).** Fed a representable demand-feasible flow
  (the Braess analytic UE `(4,2,2,2,4)`, and any `Δᵀh0`), the decode returns it
  (objective → 0, `feasible=1`, gap ~0). At random untrained weights every decoded
  emission passes the audit (node balance ~1e-11·D — mirror of implicit-ue A4).
- **Decode converges, no limit cycle (verified).** The projected-gradient L2
  OBJECTIVE is monotone non-increasing; the emitted L-inf residual is near-
  stationary but NOT monotone (wiggles <1% after convergence — the review
  caught the earlier 'more steps never raise the residual' claim passing only
  by an accidental step-pair choice); feasibility is exact regardless of the
  target's infeasibility.
- **In-sample training works (verified).** Adam on the composite loss reduces the
  in-family flow loss below the untrained head's (282 vs 653).
- **Conservation ablation, in-family (verified).** Training WITH the soft
  conservation loss (`w_c=0.05`) yields a lower in-family aggregate node-balance
  residual than `w_c=0` (6.4 vs 11.7, re-measured on the shipped code — the
  review caught a stale 7.9-vs-11.6 pair from a pre-ship revision) — the paper's contribution (2) made
  measurable. Scoped IN-FAMILY (where the loss is optimized); its held-out
  transfer does NOT hold (measured backwards on Sioux Falls: 296 vs 120) — the
  identifiability caveat again, pinned honestly rather than forced.
- **Held-out directions (verified, honest — every axis NAMED and MEASURED).** On
  disjoint Sioux Falls, at the pinned `epochs=100, lr=0.01` (same-platform, loose
  margins):
  - *flow-error axis (wmape):* the RAW emission (0.560) transfers WORSE than the
    ridge (0.282) — the GNN's raw ratios are poor in magnitude, but it is censored
    either way; the DECODED emission (0.168) is BETTER than the ridge — projecting
    onto the demand-feasible polytope recovers accuracy;
  - *certified-gap axis at matched route sets:* the same-platform measurement at
    the pinned `epochs=100` is `implicit-ue-nn` 0.168 < `het-gnn` 0.259 — but the
    review PROVED this ordering is NOT a CI invariant: it inverts along het-gnn's
    own training trajectory by ~epoch 130 (0.238@110, 0.188@120, 0.135@130) and
    flips outright under a 1e-6 weight perturbation, while the CI torch job
    installs an UNPINNED torch. The ordering is therefore recorded HERE as
    provenance (same-platform, pinned epochs), and the test asserts only the
    stable structure (both feasible, gaps in a sane band). Any retune of
    `_TRAIN_EPOCHS`/`_TRAIN_LR` must re-measure this paragraph;
  - *wall/convergence axis:* a converged bfw certifies an orders-better gap
    (`3.2e-6` vs 0.259).

## The identifiability caveat (the honest finding, inherited)

As in adr-025, training matches equilibrium *flows*, not the mechanism, so a GNN
fit on 8–14-node synthetic graphs need not transfer to TNTP congestion regimes.
We therefore scope claims to **in-family loss reduction** and the **in-family
conservation ablation**, and do NOT claim the trained model beats an untrained one
on the held-out *gap* — forcing that by cherry-picking a seed would be exactly the
dishonesty the benchmark exists to expose. The conservation ablation not
transferring to held-out (measured) is itself a result, not a bug.

## Honest sourcing

The canon paper — Liu, T. & Meidani, H. (2024), *End-to-end heterogeneous graph
neural networks for traffic assignment*, **Transportation Research Part C**
165:104695, DOI `10.1016/j.trc.2024.104695` — is **paywalled and was attributed
unread**. The formulation was recovered and cross-verified with **zero
discrepancies** from the authors' own open sources, read in full:

- **arXiv:2310.13193v3** — the preprint carrying the identical title (read via
  LaTeXML HTML; math recovered from the LaTeX alt-text).
- **NSF PAR accepted manuscript** — `par.nsf.gov/servlets/purl/10572434` (20 pp,
  NSF award CMMI-1752302). Every checked fact agrees between the two: architecture
  (3-layer preprocessing → embed 32; two V-Encoders + two R-Encoders; 8 heads;
  hidden 64), loss weights `(1.0, 0.005, 0.05)`, `lr=1e-3` batch 128, the
  `U(0.5,1.5)` per-OD / `U(0.2–1.0)` capacity scalings, "OD normalized to 100",
  Table 1 sizes/demands, Table 2 numbers, timings, the transfer + homogenized
  strategies, PyTorch+DGL, 5-fold CV.

The shipped unit is explicitly a **lean variant** (single head, hidden 16, size-
agnostic node-kernel features, plain backprop, the flagged feasible decode).
**Underdetermined details** the reimplementation had to call, flagged here and in
the docstring: whether the β-FFN and update-FFN are per-head or shared; the
nonstandard `x + LayerNorm(FFN(z))` update order; whether Eq-8 aggregation over
`N_o(u)` is outgoing-only (taken literally). **Documented deviations:** (i) labels
are `BiconjugateFrankWolfeModel` equilibria at the repo's certified-gap criterion,
not the paper's successive-flow-change Frank-Wolfe criterion (which bounds flow
movement, not the gap); (ii) `L_f`/`L_c` are normalized by total demand for
cross-scenario scale stability — the paper's "OD normalized to 100" is
underspecified, and its Table 1 demand totals (Sioux Falls 188,960) do not match
the canonical TNTP/repo value (360,600), so the paper's MAE magnitudes cannot be
matched exactly, only relative orderings. **Do NOT conflate** with the companion
papers arXiv:2408.04131 (dynamic HetGSeq sequel, tier-2) or arXiv:2501.09117
(multi-class multi-view follow-up) — separate canon items, out of scope.

## Alternatives considered

- **Raw-only (a) / decode-only (b).** Rejected (Decision 1): (a) duplicates the
  ridge's censored signal; (b) hides the paper-faithful behaviour and overstates
  the paper's (non-feasible) output.
- **The paper's `|V|+2` OD-row featurization + homogenized dummy-node padding /
  transfer learning.** Rejected: not permutation equivariant and size-locked;
  replaced by the size-agnostic node-kernel (Decision 2).
- **`torch-geometric` / DGL.** Rejected (adr-025 Decision 1): a compiled,
  version-locked dependency has no place in a public benchmark's install story —
  the single `torch` extra already covers both Phase-3 models.
- **Multi-head attention, 3-layer preprocessing, LayerNorm towers, 8 heads.** Not
  built — the lean floor (1 head, hidden 16, 2 V + 2 R layers) already exercises
  typed heterogeneous passing and adaptive virtual-edge weights.
- **Committed weight blobs / a weight fetcher.** Against the from-seed
  reproducibility story; train-at-solve-time under a fixed internal seed instead.
- **A decode on/off factor.** Dead config: the raw→decoded pipeline is fixed and
  both checkpoints always emit.
- **Sparse `Δ` for the decode.** The dense route-incidence cost note from adr-025
  (~1.75 GB on Barcelona) applies verbatim — same documented future-work line.

## Consequences

- **New:** `HetGNNModel` (`het-gnn`, registered when torch present); the CI torch
  job's test step becomes the explicit two-file list
  `pytest tests/test_implicit_ue.py tests/test_het_gnn.py`; `tests/test_het_gnn.py`
  (19 tests). No new certificate, scenario field, or Evaluator branch; no change
  to `gaps.py`, `learned.py` (imported from), or `implicit_ue.py` (`_build_routes`
  / `_RouteSet` imported); no torch in the core matrix; no second optional extra.
- **Unchanged:** the Evaluator, the fairness gate, every hash (the golden Braess
  content hash is re-asserted byte-identical in the new test file), and the
  numpy-only core (`import tabench` and the full 731-test suite pass without
  torch). This closes Phase 3 — `het-gnn` is the last torch model, so the CI list
  is closed (no glob machinery).
- **Follow-ups:** the Xu et al. (2024) cross-domain test set; sparse-`Δ` decode
  for larger ladders; training-seed variance as a macrorep experiment (deferred).

## Adversarial review

Three independent lenses (soundness, formulation, numerics), each executing
code; every finding CONFIRMED by a runnable repro and fixed (streak: 15/15
sprints with at least one material defect).

**MAJOR (soundness + numerics, converging): the Axis-2 direction pin was a CI
flake in waiting.** The strict `het-gnn gap > implicit-ue gap` assertion sat 30
epochs from inversion on het-gnn's own training trajectory (inverts by ~epoch
130, not the self-flagged ≥150), flipped outright under a 1e-6 weight
perturbation, and the CI torch job installs an UNPINNED torch — a
kernel/reduction-order change integrated over 100 epochs is exactly a
perturbation of that scale. The two sibling directional pins in the same test
(decoded wmape < ridge; raw wmape > ridge) survived the identical perturbation
sweep with zero flips — the fragility was specific to this one inequality.
FIXED by demoting the ordering to provenance (recorded above with its
same-platform/pinned-epochs scope) and asserting only the stable structure.

**MAJOR (formulation): a silent deviation misattributed to paper Eq 9.** The
R-Encoder edge weight is a LEARNED affine scalar of the edge features, but the
paper's Eq 9 uses the FIXED sum of the normalized features (and the feature
set was extended from `[fft, cap]` to `[fft, cap, b, power]`). A strict
generalization, harmless to scoring — but the "paper Eq 9" attribution was
false as written. FIXED: flagged in the docstring, here, and model-specs.

**MAJOR (formulation): a stale measured anchor.** The in-family conservation-
ablation pair quoted 7.9 vs 11.6; the shipped code reproduces 6.4 vs 11.7
(direction intact). Corrected above.

**MAJOR (numerics): the decode-monotonicity claim was accidentally true.** The
emitted L-inf residual is NOT monotone in steps — the old assert passed only by
the accidental (200, 600) step-pair choice. FIXED: the test now pins the L2
projection OBJECTIVE non-increasing (what projected gradient actually
descends) with the L-inf comparison loosened to a documented <1% near-
stationary wiggle bound.

**NOTEs:** the zero-demand "short-circuit before any torch work" comment
overstated (the cached one-time training still runs so its cost is reported —
reworded in code and test); a dead per-virtual-edge demand feature was built
and shipped but never consumed (removed; demand reaches the model through node
features, which is the paper-consistent path); the pyproject torch-extra
comment now names both models; this section documents the flip boundary the
old draft dangled.

**Survived (highlights):** two-checkpoint certification is airtight (the
runner recomputes every checkpoint, self-reports never reach the CSV, no
best-checkpoint selection anywhere); 124-scenario decode fuzz — every decoded
emission feasible, zero crashes, including 5 ms wall budgets; cross-model
cache isolation byte-identical across processes AND training orders (het-gnn
first vs implicit-ue first); all adr-025 state-restoration lessons verified
executed (RNG/threads/det-flag/default-dtype restored bit-exact after a full
cold solve); the fairness gate blocks all 12 training instances (6 base + 6
augmented) under renamed families; permutation equivariance at 1.1e-16 over 40
zone-respecting permutations while the paper's featurization fails at 21.5;
the simplex projection matches an independent implementation; Anaheim scales
(6.6 s, +65 MB); torch 2.4.1 floor passes all 19 tests; the conservation
ablation's in-family direction is robust (44-50% margin under perturbations)
while its held-out reversal is pinned honestly.
