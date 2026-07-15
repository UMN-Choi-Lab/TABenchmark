# ADR-025 — Implicit-NN user equilibrium: the first torch model, feasibility as architecture

**Status:** accepted (shipped in v0.2)
**File:** `docs/design/adr-025-implicit-ue-nn.md`

## Context — act two of the ADR-006 argument

ADR-006 shipped the first learned model (`learned-surrogate`, a per-link ridge
regressor) to make one point: **link-flow accuracy is not an equilibrium
certificate.** The surrogate correlates 0.63–0.99 with the best-known flows yet
is censored `feasible=0` on every TNTP net, because a per-link predictor emits a
flow nobody routed (its node-balance residual is orders above tolerance). That is
*act one*: accuracy ≠ certificate.

`implicit-ue-nn` is *act two*: **demand feasibility is not equilibrium quality.**
Liu et al. (2023) bake the equilibrium condition into the architecture — the
output *is* a fixed point of a learned cost — so the emitted flow is demand-
feasible by construction and clears the audit the ridge is censored by. The
benchmark then recomputes the equilibrium gap of that feasible flow under the
*true* costs (P1). The comparison axis must be NAMED (adr-025 review MAJOR —
the first draft matched on no axis at all): a CONVERGED classical solver
(matched-or-less wall-clock) certifies an orders-better gap, while at matched
SHORTEST-PATH-CALL budget the direction REVERSES — six Dijkstra sweeps plus
cheap fixed-point iterations certify better than six sweeps' worth of
Frank-Wolfe AON iterations (measured 0.168 vs 0.223 on Sioux Falls).
Feasibility is architectural; equilibrium quality is still paid for in
routing work — and on the repo's own hardware-free sp_calls axis the learned
layer buys it cheaper. The suite pins BOTH directions with their axes named.

This is also the benchmark's first model with an optional heavy dependency
(torch), so the ADR records the dependency mechanics as much as the model.

## Decision 1 — torch is an optional extra, never a core dependency

The numpy/scipy-only core is a stated design feature (ADR-006: learned/torch
models are "kept out of the core so the benchmark stays numpy/scipy-only"). So:

- `pyproject.toml` gains `torch = ["torch>=2.4"]` under
  `[project.optional-dependencies]`; the core deps (`numpy>=1.24`, `scipy>=1.10`,
  `pyyaml`) are untouched. `torch>=2.4` is numpy-2 safe with cp310/cp312 wheels;
  no upper pin.
- `src/tabench/models/__init__.py` wraps `from .implicit_ue import
  ImplicitUENNModel` in `try/except ModuleNotFoundError`, re-raising unless
  `exc.name == 'torch'` (so a real bug in the module is never swallowed).
  `@register_model` runs only when torch is present, so on a core install
  `MODEL_REGISTRY` and `tabench list` simply lack the model and the
  register-model invariant (every registered model is instantiable) holds. The
  two torch-free CI matrix legs are the live regression that `import tabench`
  works without torch.
- **CI** keeps the existing py3.10+3.12 matrix exactly as-is and adds one job
  (`torch`, py3.12 only) that installs the CPU-only build
  (`pip install torch --index-url https://download.pytorch.org/whl/cpu`, whose
  `+cpu` local version satisfies `torch>=2.4` so the editable install does not
  re-resolve to the CUDA wheel), caches `~/.cache/pip` keyed on the pyproject
  hash, reuses the data cache with `TABENCH_REQUIRE_DATA=1`, and runs **only**
  `tests/test_implicit_ue.py`.
- **Caveat, documented for users:** pip extras cannot pin an index, so
  `pip install tabench[torch]` from PyPI gives Linux users the multi-GB CUDA
  build by default. The README documents the CPU-only incantation prominently.
- **Extra-creep is killed now:** the single `torch` extra covers *both* Phase-3
  models. `het-gnn` (Liu & Meidani 2024) will **hand-roll heterogeneous message
  passing with plain-torch scatter ops (`index_add_`)** — `torch-geometric` is a
  compiled, version-locked dependency that must never enter a public benchmark's
  install story. Decided here so it never becomes a second extra.

## Decision 2 — the shipped unit is a LEAN VARIANT (flagged, like `dtd-stochastic`)

The model plugs into the existing contract with zero new plumbing (ADR-006
Decision 1 stands: same wrapper, same certificate, same fairness gate). It is a
lean instantiation of Liu et al.'s method, flagged exactly the way
`dtd-swap-sue`/`dtd-stochastic` flag their filter variants:

- **Learned cost head** — a small MLP emits a per-link, flow-*monotone* cost
  correction added to the true BPR latency. Monotonicity in flow is
  architectural: `relu(gain) · softplus(mlp(static)) · (v/cap)`, nonnegative and
  increasing in flow for *every* parameter value — so the smoothed equilibrium
  stays unique: the logit fixed point is the optimum of a strictly convex
  entropy program when route costs are nondecreasing (Fisk 1980; the review
  corrected an earlier misuse of Dafermos-1988 VI uniqueness here, and
  uniqueness does NOT by itself make the *iteration* convergent — see the
  adaptive-damping finding below) — and
  identically zero at zeroed parameters (so a zeroed head reduces the layer to a
  plain logit loading at the true costs — the θ=0 anchor). The MLP reads only the
  standardized per-link features `[fft, cap, b, power]` — network-size-agnostic,
  the paper's "kernel strategy" — with the flow channel entering only through the
  guaranteed-nonnegative slope.
- **Implicit layer** — over PathEngine column-generated per-OD route sets, the
  layer solves the logit route-choice fixed point
  `h* = D_od · softmax_od(−β·c_θ(Δᵀh*))` and emits `v = Δᵀh*`. Each OD's route
  flows sum to its demand, so **node balance is exact by construction** and the
  harness feasibility audit always passes. The forward pass is a gently damped
  iteration (constant step at a modest β), which converges without the
  overshoot/limit-cycling an aggressive step or near-all-or-nothing β produces on
  congested power-4 nets (the repo's recurring fixed-point defect); the layer's
  own residual `max|g(h)−h|` is emitted as a descriptive truncation trace.
- **Hypergradient** — training uses the implicit-differentiation (IMD/adjoint)
  hypergradient `dL/dθ = (dg/dθ)ᵀ(I−dg/dh)⁻ᵀ dL/dh`, with the adjoint solved
  exactly as a small dense linear system (the Braess/synthetic route sets are
  tiny; a Neumann series would diverge where the logit map's Jacobian is stiff).
  torch autograd supplies the Jacobian columns and the final VJP, so no unrolled
  forward graph is stored — the paper's IFT gradient, not a truncated unroll.
- **Budget semantics (P6)** — inference `sp_calls` are the real Dijkstra sweeps
  (column-generation rounds, >1 unlike the ridge's 1); `iterations` are the
  fixed-point steps; the one-time offline training cost is reported as
  `training_sp_calls`/`training_wall_ms` provenance, never scored. No `.pt`
  blobs are committed (the repo's first binary artifact, stale-vs-code risk,
  against the everything-from-seed reproducibility story): training runs at solve
  time on the tiny synthetic fixture (< 60 s CPU) under a fixed internal seed and
  a module-level cache — the `learned.py` precedent. If training ever outgrows CI,
  the documented escape hatch is a checksummed weight artifact through the
  existing data registry/fetcher pipeline (exactly like TNTP); not built now.

## Anchors

- **A1 identity (verified).** With the cost head pinned to the true BPR latency
  (zeroed correction), the layer's logit fixed point on the builtin Braess net is
  the analytic UE — link flows `(4,2,2,2,4)`, common used-route time 92, certified
  gap ~0. (The Braess UE is an equal-cost point, so the logit split is uniform for
  every β — the identity holds regardless of temperature.)
- **A2 IMD hypergradient (verified).** The adjoint hypergradient matches central
  finite differences of the full solve on Braess to < 1e-5 relative error at a
  well-conditioned cost head (near-machine at eps 1e-4). This is the
  implicit-function-theorem gradient the paper differentiates.
- **A4 feasibility by construction (verified).** At *random untrained* θ every
  emission passes the demand-feasibility audit (`feasible=1`, node-balance ~0) —
  the architectural property the censored ridge lacks.
- **θ=0 reduction (verified).** With the correction zeroed, the two-route layer
  equals the analytic binary-logit split at the true costs and the layer's β
  (recomputed via brentq) — the NN learns a correction, not a rename.
- **Held-out direction (verified, honest — see the caveat below).** On disjoint
  Sioux Falls the trained model is `feasible=1` with a finite positive certified
  gap; it clears the audit the ridge is censored by (feasible 1 vs 0) with no
  worse flow error; a CONVERGED bfw certifies a strictly better gap; and at
  MATCHED sp_calls the direction reverses in the NN's favor (pinned as a
  result, not hidden). Directions pinned with axes named, margins loose.
- **In-sample training works (verified).** IMD hypergradient descent reduces the
  synthetic-family flow loss below the untrained (zeroed) head's.
- **Fixed-point stability (fuzzed + review-hardened).** The review CONFIRMED the
  original constant-damping iteration LIMIT-CYCLED on the congested power-4 net
  while the test asserted only by-construction properties. The iteration now
  uses adaptive damping (step halved whenever the residual rises, floored at
  1e-3), the residual is measured AT the emitted iterate, and the test pins a
  real demand-relative residual bound (< 1e-6; measured ~5e-13).

## The identifiability caveat (the honest finding, a weak spot for review)

Training matches equilibrium *flows*, not cost *parameters*: the equilibrium is
identified, θ is not. So the learned cost can fit the synthetic family yet induce
a **larger** certified gap than the untrained plain-logit baseline on a held-out
TNTP net in a different congestion regime. The pilot confirms this: unregularized
training halves the synthetic-family loss but *degrades* the Sioux Falls certified
gap (≈0.08 plain-logit → ≈0.28 overfit). We therefore (a) regularize the
correction (weight decay) so it stays small and transferable, and (b) **do not**
claim the trained model beats an untrained/random-init head on the held-out
*gap* — that claim is false for this lean variant, and forcing it by cherry-
picking a seed would be exactly the dishonesty the benchmark exists to expose.
What we *do* pin is: feasibility is architectural (feasible 1 vs the ridge's 0),
the gap is finite and honestly positive, bfw beats it, and training genuinely
reduces the in-family loss. This accuracy-vs-certificate identifiability gap is a
*result*, not a bug — it is precisely the phenomenon `implicit-ue-nn` is here to
make measurable.

## Alternatives considered

- **Full IFT/DEQ machinery (Anderson acceleration, Broyden adjoint).** Overkill at
  benchmark scale; the dense adjoint solve is exact on these route sets. Flagged
  as the variant boundary, not built.
- **Truncated unrolled backprop (ITD).** Simpler but stores the forward graph and
  is a biased gradient at small unroll depth; the exact dense adjoint is cleaner
  and is what A2 verifies.
- **A torch-native shortest path.** Slow, unfaithful, and pointless — the P1
  certifier recomputes the gap from emitted flows regardless, so the PathEngine
  (numpy/scipy) does the combinatorial work and torch does only the smooth,
  differentiable work (the hybrid oracle).
- **Committed weight blobs / a weight fetcher.** Against the fetched-never-vendored
  ethos and the from-seed reproducibility story; train-at-solve-time instead.
- **`torch-geometric` for het-gnn.** Rejected now (Decision 1): a compiled,
  version-locked dep has no place in a public benchmark's install story.

## Honest sourcing

The canon paper — Liu Z., Yin Y., Bai F., Grimm D.K. (2023), *End-to-end learning
of user equilibrium with implicit neural networks*, **Transportation Research
Part C** 150:104085, DOI `10.1016/j.trc.2023.104085` — is **paywalled and was
attributed unread**; there is no arXiv version, and both SSRN preprints (abstract ids
4198835, 4908029) are bot-blocked. The mechanism was recovered and cross-verified
from the authors' **own open sources, read in full**:

- Liu Z., Yin Y. (2024), *A Unified Framework for End-to-End Learning of User
  Equilibrium*, hEART 2024 paper 0470 —
  `https://transp-or.epfl.ch/heart/2024/abstracts/hEART_2024_paper_0470.pdf`
- The authors' TRB poster for the exact canon paper (TRBAM-23-02639) —
  `https://zhichen6.github.io/images/e2e_part1_poster.pdf`
- The authors' poster for the TS-2025 sequel —
  `https://zhichen6.github.io/images/ts_e2e_2024_poster.pdf`

The shipped unit is explicitly a **lean variant** — a logit route-choice
fixed-point layer with a monotone MLP cost head and an exact dense IMD adjoint —
of their IFT/DEQ method; 2023-paper-specific architectural details (the exact
decoupled-projection operator, the Weight/Attribute-Net layer sizes) remain
unconfirmed. This is flagged in `model-specs.json` and the module docstring the
way `dtd-stochastic` flags its filter variant.

## Consequences

- **New:** `ImplicitUENNModel` (`implicit-ue-nn`, registered when torch present);
  the `torch` optional extra; one CI job; `tests/test_implicit_ue.py`. No new
  certificate, scenario field, or Evaluator branch; no change to `learned.py`
  (the training family is imported from it); no torch in the core matrix.
- **Unchanged:** the Evaluator, the fairness gate, every hash (the golden Braess
  content hash is re-asserted byte-identical in the new test file), and the
  numpy-only core (`import tabench` and the full 731-test suite pass without
  torch).
- **Follow-ups:** `het-gnn` (hand-rolled message passing, same extra, same
  training family); the Xu et al. (2024) cross-domain test set; training-seed
  variance as a macrorep experiment (deferred — v1 uses one fixed internal seed).

## Review

Three independent lenses (soundness, formulation, numerics), each executing
code; every finding CONFIRMED by a runnable repro and regression-pinned in
`test_implicit_ue.py` (streak: 14/14 sprints with at least one material defect).

**MAJOR — zero-demand crash.** All-zero or diagonal-only OD matrices crashed
`_build_routes` on an empty route set (`(0,) @ (n_links,)`), killing a whole
`run_experiment` grid, while every classical solver handles the same input.
FIXED: `solve` short-circuits the no-routable-demand case with the exact zero
emission (feasible, gap 0).

**MAJOR — `wall_seconds` silently ignored.** A wall-only budget was consumed
37× over (1877 ms at a 50 ms budget on Sioux Falls) while every classical
solver checks `budget.exhausted` per iteration. FIXED: the deadline is enforced
between column-generation rounds and inside the fixed-point loop; truncation
never breaks feasibility (structural).

**MAJOR — the headline named no axis, and on the sp_calls axis it REVERSES.**
The "bfw certifies a strictly better gap at matched budget" claim was pinned by
a comparison matched on *no* budget axis; at matched `sp_calls = 6` the NN's
certificate (0.168) actually beats bfw's (0.223) — six Dijkstra sweeps plus
cheap fixed-point iterations outperform six sweeps' worth of Frank-Wolfe AON
iterations, exactly the isolating comparison `frank_wolfe.py` itself names.
FIXED: both directions are now pinned with their axes named (converged bfw wins
the wall/convergence axis by orders; the NN wins the matched-sp axis) — a real
result reported, not an embarrassment hidden.

**MAJOR — the fixed point limit-cycled on the congested power-4 net** while the
stability test asserted only by-construction properties, and the docstring
claimed convergence with a misapplied Dafermos-1988 uniqueness argument. FIXED:
adaptive damping (step halved whenever the residual rises, floored at 1e-3)
converges the reviewer's cycling instance to ~5e-13 demand-relative residual in
~41 steps; the residual is measured at the emitted iterate (never a stale
mid-loop value); the test pins a real residual bound; the uniqueness citation is
now Fisk (1980) with the iteration/uniqueness distinction stated.

**MINORs, fixed + pinned:** the first cold-cache solve permanently reseeded the
process-global torch RNG (training now saves/restores `torch.get_rng_state()`,
completing the thread-count/deterministic-flag restoration set);
`coords.iterations` recorded the cap rather than the ~100 steps actually
executed (29× P6 over-report — the executed count is recorded now). NOTEs: the
"audit always passes" docstring claim scoped to fixed-demand tasks
(elastic/combined censoring reflects demand consistency, as for every
fixed-demand solver); ADR DOI markup and the "read UNREAD" typo fixed; the
dense route-incidence matrix costs ~1.75 GB on Barcelona — a
`scipy.sparse`/incremental build is noted as future work if bigger ladders
land.

**Survived (highlights):** A2 IMD hypergradient vs central finite differences
at 25 *random* heads (max rel err 2e-7); architectural monotonicity
brute-forced over 200 heads at weight scales 1e-2..1e3; feasibility by
construction under 110-scenario fuzz including the limit-cycling regime and
maximal truncation (`sp_calls=1, iterations=1`); the fairness gate blocks both
the family name and renamed exact training instances; training-cost smearing
absent (cold/warm solve wall ratio 1.15 while training took ~7 s, reported
separately); byte-determinism across four processes; the torch-free matrix leg
simulated in a fresh venv (no torch pulled, imports clean, test file skips);
all 14 tests pass under torch 2.4.1+cpu (the declared floor) and 2.12;
actionlint-clean CI; golden Braess hash byte-identical.
