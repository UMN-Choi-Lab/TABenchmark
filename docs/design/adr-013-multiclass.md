# ADR-013: multiclass — Dafermos (1972) multiclass-user equilibrium

**Status:** accepted (implemented)
**Date:** 2026-07-09
**Deciders:** static-extensions track — the multiclass half of the "multiclass / VI" roadmap item
**File:** `docs/design/adr-013-multiclass.md`

## Context

`vi-asym` (adr-011) shipped the *single-class* asymmetric variational inequality
`t(v) = t_BPR(v) + C v`. adr-011 explicitly deferred the **multiclass** case
(Dafermos 1972) for one reason: certifying a *per-class* Wardrop condition needs
*per-class* link flows, but `FlowState.link_flows` / `Trace` / `Evaluator.evaluate`
were all hardwired to a single aggregate `(n_links,)` vector — a harness-contract
change, not a self-contained model.

Dafermos (1972, *Transportation Science* 6(1):73-87, `dafermos1972traffic`)
generalizes UE to `K` user classes that share the network but perceive
class-specific, mutually-coupled costs. Each class `i` routes its own demand
`g^i` to a Wardrop equilibrium in *its* cost `t^i`, but every class's cost
depends on the joint flow, so the equilibria are simultaneous. Stacking the
class-indexed flows `V = (v^1, ..., v^K)` this is a block-structured single-class
asymmetric VI

    find V* in K = K_1 x ... x K_K  s.t.  <T(V*), V - V*> >= 0  for all V in K,

i.e. exactly `vi-asym` promoted one level up (`K = 1` recovers it) — with two
genuinely new ingredients: the feasible set is a **product** of per-class demand
polytopes (routing is per class), and the emitted flow gains a **class axis**.
Dafermos (1972)'s specific contribution is the model *and* the symmetry
condition under which it reduces to a single convex program (the integrable
case); the genuinely-asymmetric VI with no equivalent optimization is Smith
(1979) / Dafermos (1980).

## Decision

1. **Cost model (linear class interaction).**
   `t_a^i(V) = t_a^BPR(v_a) + sum_j M_ij v_a^j`, where `v_a = sum_j v_a^j` is the
   total link flow and `M = scenario.multiclass.interaction` is a `(K, K)` matrix
   applied per link. `M` is what makes the per-class split well-defined: with
   `M = 0` every class sees the identical cost `t_a^BPR(v_a)` and the split is
   arbitrary (the model degenerates to single-class UE on the summed demand).
   A **symmetric** `M` is the integrable case (the equilibrium minimizes a convex
   multiclass-Beckmann potential; Dafermos 1972); an **asymmetric** `M` is a
   genuine VI with no equivalent optimization (Smith 1979; Dafermos 1980). This
   is the block-diagonal generalization of `vi-asym`'s `C`.

2. **Schema — `MulticlassDemand` + `Scenario.multiclass`.** A new frozen
   dataclass (`core/scenario.py`) holds `matrices` (`(K, n_zones, n_zones)`,
   `K >= 2` per-class OD) and `interaction` (`(K, K)`). It is referenced by a new
   optional `Scenario.multiclass` field, content-hashed only when present
   (appended last, after `link_interaction`), so every scenario without it — the
   golden Braess instance included — hashes byte-identically
   (`cf00f411…` re-asserted in `tests/test_multiclass.py`). It is mutually
   exclusive with the other six optional task fields, and the aggregate `demand`
   is validated to equal the class sum so every single-class consumer of `demand`
   (feasibility scale, fairness gate, aggregate loading) stays consistent.

3. **Output-contract change (the crux adr-011 deferred), done additively.** An
   OPTIONAL `class_link_flows: (K, n_links) | None` is added to `FlowState` and
   `Trace.record`, defaulting to `None`. Every single-class model leaves it
   `None`, so their emissions, the aggregate `link_flows` contract, and the CSV
   schema are byte-identical to before this field existed (the full suite is the
   guard). It is a **first-class emitted object**, not a `self_report` entry — so
   the harness *recomputes* the per-class VI residual from it (P1), never trusts
   it. When present, `link_flows` is the class sum. This is the additive form of
   exactly the reshape adr-011 flagged as the "real" fix: no existing signature
   changes, no existing model touched.

4. **Solver `multiclass`** (`models/multiclass.py`, `MulticlassModel`, paradigm
   `static_ue_multiclass`). Multiclass diagonalization (nonlinear Gauss-Seidel):
   sweep the classes; for class `i` freeze the other classes' flows, which makes
   class `i`'s cost separable in its own flow — an ordinary single-class UE that
   Frank-Wolfe (exact Brent line search on the diagonalized cost) solves — then
   relax `v^i <- v^i + step (v^i_inner - v^i)` and use each class's updated flow
   immediately. It reuses `vi-asym`'s inner FW verbatim, once per class. `K = 1`
   would collapse to `vi-asym`. Convergence follows the Dafermos (1982) /
   Florian & Spiess (1982) diagonal-dominance condition, not monotonicity alone;
   a coupling that drives a cost non-positive stops the sweep and emits a flow
   the certificate censors (never a false accept).

5. **Certificate (P1).** The harness recomputes, from the emitted `V`, the
   class-summed VI residual — the product feasible set makes the VI gap decompose
   into per-class all-or-nothing minima:

       gap = (sum_i <t^i(V), v^i> - sum_i min_{y^i in K_i} <t^i(V), y^i>) / sum_i <t^i(V), v^i>,

   the ordinary relative gap summed over classes at the coupled cost (0 iff `V`
   solves the multiclass VI). Feasibility is a **per-class** conservation audit
   (each class routes its own demand). `beckmann_objective` is NaN (no potential
   for an asymmetric `M`). No new scored CSV key: the residual reuses
   `relative_gap` / `average_excess_cost`. A multiclass scenario emitted without
   per-class flows is censored (an aggregate flow cannot certify a per-class
   equilibrium).

6. **Analytic anchors** (`multiclass_two_route_scenario`, both recomputed in the
   tests). Diamond routes 1->3->2 / 1->4->2, two classes (cars `g=4`, trucks
   `g=2`), zero BPR slopes so congestion is purely the per-link `M`. Because a
   2-link route applies the per-link coupling on both legs, the route-level
   coupling is `2 M`, and the equilibrium route-A flows solve
   `4 M [p - g_cars/2, q - g_trucks/2]^T = [a2, a2]^T`, i.e.
   `[p, q] = [g_cars/2, g_trucks/2] + (a2/4) M^{-1} [1,1]^T`:
   - **symmetric** `M = [[0.5, 0.25], [0.25, 0.5]]` → cars `(2.5, 1.5)`, trucks
     `(1.5, 0.5)`, aggregate `(4, 4, 2, 2)`, class costs equalized 3.25 / 2.75 —
     the integrable / multiclass-Beckmann check;
   - **asymmetric** `M = [[0.5, 0.5], [0, 0.5]]` (trucks slow cars, not vice
     versa) → cars `(2, 2)`, trucks `(1.75, 0.25)`, aggregate `(3.75, 3.75, 2.25,
     2.25)` — a genuine-VI flow no Beckmann/FW solver reaches; classes route
     differently (the multiclass signature).
   All values are exact multiples of 0.125.

## Alternatives considered

- **Breaking reshape of `link_flows` to `(K, n_links)` everywhere:** rejected —
  it would touch all 25 single-class models and the CSV schema. The optional
  `class_link_flows` field is the additive equivalent (this ADR is the
  realization of adr-011's deferred item, done without a breaking change).
- **Per-class flows in `self_report`:** rejected — the certificate would then be
  recomputing a scored metric from a trusted self-report, violating P1. Per-class
  flows must be a first-class emitted object to keep the residual harness-pure.
- **Nonlinear / flow-dependent PCE coupling (Riente de Andrade et al. 2017):**
  deferred — the linear `M` is the minimal faithful coupling that gives a
  closed-form anchor and reuses `vi-asym`'s affine machinery; a flow-dependent
  interaction is a separately-scoped follow-up.
- **A dedicated sequential multiclass solver instead of reusing vi-asym's FW:**
  the per-class diagonalized subproblem *is* an ordinary UE, so the established
  FW+brentq inner loop is reused rather than reimplemented.

## Consequences

The benchmark gains its first **multiclass-user** equilibrium model and its first
per-class-flow output, with a sound, harness-recomputed VI certificate and
hand-derived integrable + genuine-VI anchors. The `class_link_flows` extension is
additive: every prior model and the golden Braess hash are provably unchanged.
Follow-ups: flow-dependent-PCE coupling, per-link (rather than uniform) `M`, and
reporting the per-class residual breakdown alongside the class sum.

## Sourcing

Dafermos (1972, *Transportation Science* 6(1):73-87, `dafermos1972traffic`, DOI
`10.1287/trsc.6.1.73`, Crossref-verified) is the multiclass model and the
symmetry/integrability condition; Smith (1979, `smith1979existence`) and Dafermos
(1980, `dafermos1980traffic`) are the asymmetric-VI characterization; the
diagonalization / relaxation convergence is Dafermos (1982) / Florian & Spiess
(1982). All three primaries were already in the verified reference canon. The
block-VI reading and both analytic anchors are hand-derived here (cross-verified
against an independent multiclass diagonalization); no number from any paper is
reproduced.
