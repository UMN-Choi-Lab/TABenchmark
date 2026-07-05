# ADR-007 — Combined trip distribution + assignment (Evans 1976): a P1-pure certificate

**Status:** accepted (shipped in v1)
**File:** `docs/design/adr-007-combined-distribution-assignment.md`

## Context

Every fixed-demand task fixes the whole OD matrix; the elastic task (ADR-005) makes each
pair's *total* demand a pointwise function of its own cost. **Combined trip distribution +
assignment** (Evans 1976) goes one step further: only the trip-end margins are fixed — the
productions `O_i` and attractions `D_j` — and the *entire* OD matrix `d_ij` is endogenous,
distributed across pairs by a doubly-constrained **gravity** model at the equilibrium travel
costs. Distribution and assignment, historically two separately-iterated models, become one
convex program (Evans 1976; Sheffi, *Urban Transportation Networks* 1985 ch. 6; Boyles,
Lownes & Unnikrishnan, *Transportation Network Analysis* §6):

```
min_{x,d}  Σ_a ∫_0^{x_a} t_a(w) dw  +  (1/β) Σ_ij d_ij (ln d_ij − 1)
s.t.  Σ_j d_ij = O_i,   Σ_i d_ij = D_j,   d ≥ 0,   x = assign(d)
```

Its stationarity conditions give **both** the doubly-constrained gravity
`d_ij = A_i B_j exp(−β u_ij)` (`u_ij` = equilibrium OD cost, `A/B` = Furness balancing
factors) **and** Wardrop route equilibrium of `d`. This is a genuinely new problem class,
and — like elastic demand — it collides with the harness's fixed-OD assumptions. ADR-005
anticipated it ("Deferred: a combined distribution–assignment task (Evans 1976) that reuses
this machinery"); this ADR is that reuse.

## Sourcing (dual-verified against open textbooks)

The primary — **Evans (1976)**, *Derivation and analysis of some models for combining trip
distribution and assignment*, Transportation Research 10(1), DOI
`10.1016/0041-1647(76)90100-3` (canon key `evans1976derivation`, tier 1) — is paywalled; its
abstract and the "Evans algorithm" convergence result are attributed unread. The **formulas
implemented here** (the combined objective, the partial-linearization algorithm, the
doubly-constrained gravity subproblem, the Furness/IPF balancing) were recovered from two
primary-accessible textbooks read in full and cross-checked formula-for-formula: **Sheffi
(1985) ch. 6** and **Boyles et al. TNA §6**. Wilson's entropy-maximising gravity model is the
second intellectual parent (via the `(1/β) Σ d(ln d − 1)` term) but has no bibkey in the
canon and is noted, not cited. Canon keys already present: `evans1976derivation`,
`beckmann1956studies`, `florian1974method`, `sheffi1985urban`.

## Decision 1 — Represent the fixed margins as content-hashed scenario data

`CombinedDemand(productions, attractions, beta)` (`core/scenario.py`) carries the fixed
trip-end margins `O_i`, `D_j` and the single shared gravity dispersion `β`. It validates
`Σ O = Σ D` (doubly-constrained feasibility) and `β > 0`, and exposes the reusable
`gravity(od_cost)` — the doubly-constrained Furness / iterative-proportional-fitting
recursion over the interzonal support (`i ≠ j`, `O_i > 0`, `D_j > 0`), made deterministic
(fixed tolerance/iteration cap, exact-row final rescale) so the solver and the harness
recompute byte-identical demand from the same costs.

`Scenario.combined_demand` is optional and **hashed only when set**, appended after the SUE
and elastic fields, so every prior scenario keeps its byte-identical hash — the golden Braess
hash `cf00f411…` is asserted preserved (`test_golden_braess_hash_preserved`). Combined is
mutually exclusive with the SUE fields and `elastic_demand` (each makes the OD demand
non-fixed in an incompatible way). The ordinary `Demand.matrix` carries the **free-flow
gravity** distribution — the deterministic uncongested-equilibrium OD matrix, a meaningful
reference with the right margins and full support.

## Decision 2 — Solve with Evans' partial-linearization Frank-Wolfe

`evans` (`EvansCombinedModel`, paradigm `static_ue_combined`) linearizes only the assignment
(Beckmann) term and keeps the entropy term exact. At iterate `(x, d)` the subproblem

```
min_y  Σ_ij u_ij y_ij + (1/β) Σ_ij y_ij(ln y_ij − 1)   s.t. margins
```

has the closed-form solution `y = gravity(O, D, β, u)` (the doubly-constrained Furness
balancing); its all-or-nothing assignment `w` gives the descent direction, and an **exact
Brent line search** on the combined objective — root of
`g(α) = t(x + α dx)·dx + (1/β) Σ ln(d + α dd)·dd`, nondecreasing on `[0,1]` — sets the step.
`x` is kept a feasible assignment of `d` throughout (`x₀ = AON(gravity at free-flow)`, and
every update advances `x` and `d` by the same step to convex combinations), so the
route-equilibrium gap of `d` stays `≥ 0`. Only **real link flows** are emitted; the auxiliary
gravity `y` is both the FW subproblem solution *and* the demand the harness recomputes.

## Decision 3 — A P1-pure certificate: recompute the demand, then score

The harness knows `O, D, β` (content-hashed), so from the emitted link flows `v` it recomputes
everything (`metrics/gaps.py`, gated on `scenario.combined_demand`), the elastic recipe with
the gravity in place of `D(u)`:

```
t   = link_cost(v)
u   = od_cost_matrix(t)          # per-OD shortest-path cost over the gravity support
d*  = gravity(u)                 # doubly-constrained gravity demand at those costs
relative_gap          = (v·t − Σ u·d*) / (v·t)        # route equilibrium given d*
node_balance_residual = ‖ balance(v) − div(d*) ‖_∞    # demand-consistency + conservation
realized_demand       = Σ d*  (= Σ O_i)               # scored quantity
```

Both are pure functions of `(v, scenario)` — no self-report is trusted (P1) — and the model's
self-reported `relative_gap` is *defined* to equal the scored gap, so the honesty diff is
`~0`. The skim is driven by the gravity **support** (the fixed margins), not the reference
matrix's nonzeros, so a reference entry that underflowed to zero can never desync the solver
from the certificate.

### The feasibility gate — and an honest bound on soundness (adversarial-review finding)

As for elastic (ADR-005), there is no given matrix, so through-node conservation alone is not
a sufficient gate; the gate is `node_balance(v, d*)` (a phantom flow routing zero OD demand
disagrees with `d* > 0` and is censored — regression-tested), plus the negative-excess guard
(`SPTT > TSTT` censored). One property is **specific to combined demand** and must be stated
plainly: the doubly-constrained gravity **always** reproduces the margins `O, D`, so
`node_balance(v, d*)` only certifies that `v` routes the correct *margins* — it carries **no**
information about the OD-pair *distribution* (unlike elastic, where `d* = D(u)` has cost-varying
margins). The only distributional teeth is therefore the negative-excess guard.

**This is NOT a full soundness guarantee, and we do not claim one.** An adversarial review
constructed the exact counterexample: on a *cost-degenerate* instance where every
margin-feasible flow induces the same uniform OD-cost skim `u`, all such flows satisfy
`Σ u·d* = u·T = v·t`, so a wrong-distribution flow (e.g. "dump all trips on the cheapest
links") certifies with `feasible = 1` and `relative_gap = 0` even though it is not the
equilibrium. We verified this is **exactly the aggregate-vs-per-OD limitation the whole harness
already documents** (single-commodity node balance is necessary, not sufficient, for
multi-commodity feasibility): the **fixed-demand** certificate admits the *identical* false
positive on the same network — it is inherited, not introduced by Evans. A fully sound
link-flow-only certificate is impossible here, because distinct margin-feasible distributions
that induce the same skim are indistinguishable at the link level; closing it needs per-OD /
multi-commodity information the "real link flows only" contract withholds (a full
multi-commodity feasibility check is future harness-wide work). We handle it three ways rather
than hide it:

1. **The scored `relative_gap` remains a valid *necessary* condition** — a genuine combined
   equilibrium always passes — and stays P1-pure.
2. **The analytic anchor is deliberately made non-degenerate.** Its near/far link intercepts
   are spaced (1 vs 3) so `c_near(s) − c_far(T − s) = 0.2 s − 3 < 0` for every feasible split;
   the margin-feasible flows form a one-parameter family `v(s) = (s, T−s, T−s, s)` and
   `gap(s) ∝ (s − s*)·(c_near − c_far)` has a **unique** zero at the equilibrium `s* ≈ 6.92`
   (β = 0.5), every other `s` being censored (negative excess) or strictly gapped. So the
   anchor itself does **not** exhibit the limitation, and "dump on the cheapest links" is
   censored on it (regression-tested, `test_anchor_admits_only_the_true_equilibrium`).
3. **The limitation is pinned transparently** by `test_aggregate_multicommodity_limitation`,
   which constructs the degenerate instance, asserts the false positive, and asserts the
   fixed-demand certificate shares it. On any scenario carrying a reference (like the anchor)
   the `flow_rmse_vs_reference` column exposes such flows independently of the gap.

### Convergence and scope

The solver early-stops on `max(route-gap, distribution-gap)`, both non-negative, so
`target_relative_gap` never trips on the transiently-negative combined gap a naive stop would.
The doubly-constrained gravity requires `Σ O = Σ D`; a disconnected instance raises during the
shortest-path step and the certificate censors it. Per-OD dispersion parameters, a
singly-constrained (production-constrained / destination-choice-logit) variant, and
conjugate/bush acceleration of the inner assignment are future extensions.

## Consequences

- **New:** `CombinedDemand`; `Scenario.combined_demand`; paradigm `static_ue_combined`;
  the `evans` model; the combined branch of `Evaluator` (scored `relative_gap`,
  `node_balance_residual`, `realized_demand`); `evans_symmetric_scenario` analytic anchor
  (symmetric bipartite → binary logit split, a scalar fixed point recomputed with brentq —
  flows `(p, q, q, p)`, `p ≈ 6.92` at `β = 0.5`, not trusted; intercepts spaced so the anchor
  is degeneracy-free); `tabench run --scenario evans`.
- **Unchanged:** every prior scenario hash (golden Braess preserved); the fixed-demand / SO /
  SUE / elastic certificate paths; all prior models and tests (201 prior pass, +19 new = 220).
- **Deferred:** per-OD dispersion; singly-constrained / logit destination-choice variant;
  conjugate/bush inner solver; large-network scaling of the `~zones²`-support gravity.
