# ADR-011: vi-asym — asymmetric variational-inequality UE (non-separable costs)

**Status:** accepted (implemented)
**Date:** 2026-07-07
**Deciders:** feasibility scoping of the Phase-1 `multiclass`/VI item → lowest-infra slice
**File:** `docs/design/adr-011-asymmetric-vi.md`

## Context

`TASKS.md` lists "**multiclass / VI** — Dafermos (1972) multiclass + Dafermos
(1980) / Smith (1979) VI formulation & stability. Add multiclass demand support +
a VI-residual metric." Full multiclass demand is *core-invasive*: certifying a
per-class Wardrop condition needs per-class link flows, which would change the
`FlowState` / `Trace` / `Evaluator.evaluate()` output contract shared by every
shipped model (aggregate flow cannot certify a per-class equilibrium). That is a
harness-contract change, not a self-contained model, and is deferred.

The **VI formulation itself** (Dafermos 1980 / Smith 1979) — the mathematical
content that "an equilibrium need not minimize any potential" — can be shipped
over the *existing single-class* setting with no output-shape change, by
introducing **non-separable link costs**. That is this ADR.

## Decision

1. **Model = affine non-separable cost VI.** A new optional `Scenario`
   field `link_interaction: np.ndarray | None` carries an interaction operator
   `C` of shape `(n_links, n_links)`; the link cost becomes
   `t(v) = t_BPR(v) + C v`. When `C` is **asymmetric** (`C != C^T`) the Jacobian
   `nabla t = diag(t_BPR') + C` is non-symmetric, so **no Beckmann potential
   exists** and the equilibrium is defined only by the variational inequality
   `<t(v*), v - v*> >= 0` for all demand-feasible `v` (Smith 1979 / Dafermos
   1980). The field follows the five shipped optional-field precedents exactly
   (`sue_theta`, `elastic_demand`, `combined_demand`, `br_epsilon`,
   `side_capacities`): own validation (shape, finiteness), mutual exclusivity with
   all five, and **conditional, order-appended `content_hash` inclusion** — so the
   golden Braess hash `cf00f411…` is byte-identical (re-asserted in
   `tests/test_vi_asym.py`).

2. **Solver = Dafermos diagonalization** (`models/vi_asym.py`, `name="vi-asym"`,
   paradigm `static_ue_vi`). Freeze the interaction `offset = C v` at the current
   iterate — which makes the cost separable — solve the resulting ordinary UE by
   Frank-Wolfe (exact Brent line search on the diagonalized Beckmann objective),
   re-freeze at the new flow, repeat; an outer relaxation `v <- v + step (v_inner
   - v)` damps oscillation on strong interactions. The fixed point solves the VI.
   When `C = 0` the outer loop is a no-op and the model reduces **exactly** to
   Frank-Wolfe UE (regression-tested against the shipped `bfw`). Structurally this
   mirrors the shipped `sc-tap` augmented-Lagrangian wrapper (an outer loop around
   the existing single-class FW machinery on a modified cost).

3. **Certificate (P1) = the ordinary relative gap at the asymmetric cost.** The
   scored quantity is the normalized VI residual `(<t(v),v> - min_{y in K}
   <t(v),y>) / <t(v),v>`, which is **identical in form to the shipped
   `relative_gap`** — a VI gap needs no potential, so `metrics/gaps.py` reuses the
   existing TSTT/SPTT/relative-gap machinery verbatim and only swaps the cost map
   to `t_BPR(v) + C v` (one gated branch, mirroring `_side_capacities`). The
   residual is 0 **iff** `v` solves the VI (necessary *and* sufficient), and it is
   fully harness-recomputed from the emitted flows (never a self-report).
   `beckmann_objective` is reported `NaN` (no potential exists); the fixed-demand
   feasibility/conservation audit is unchanged. A flow whose interaction drives an
   augmented cost non-positive is censored (shortest paths need positive costs).

4. **Analytic anchor** (`data/builtin.py::vi_two_route_scenario`). Two disjoint
   2-link routes with an asymmetric coupling between the congestible legs
   (`C[1,3]=c13 != c31=C[3,1]`). Hand-derived closed form
   `f_A* = (1 + (1-c13) D) / (2 - c13 - c31)` (= `6/1.3 = 4.6154` at
   `D=10, c13=0.5, c31=0.2`), both route costs `8.3077`. This differs from the
   plain-UE split `(D+1)/2 = 5.5` **and** from the symmetrized-interaction Beckmann
   split `7.5/1.3 = 5.769` — so the asymmetry is load-bearing and the equilibrium
   is one no potential-minimizing (Beckmann/FW/gradient-projection) solver reaches.

## Alternatives considered

- **Full multiclass demand (Dafermos 1972):** rejected for now — needs the
  per-class output-shape core-contract change; deferred to its own sprint.
- **MSA on the asymmetric-cost AON map** instead of diagonalization: valid for
  strictly monotone VI but slower; diagonalization reuses the FW machinery and is
  the canonical Dafermos algorithm.
- **A new `vi_relative_gap` scored key:** rejected as redundant — the VI residual
  *is* `relative_gap` at the asymmetric cost; adding a synonym would fork the
  leaderboard column for no gain.

## Consequences

A genuinely non-integrable (asymmetric-Jacobian) equilibrium is now benchmarkable
with a sound, harness-recomputed, necessary-and-sufficient VI residual — content
no shipped separable-cost solver can reach. All changes are additive; the golden
Braess hash is provably preserved; no `FlowState`/`Trace`/`Evaluator` signature
changed. Full multiclass demand and `transit-strategy` remain separately-scoped
larger sprints. **Monotonicity vs convergence are distinct.** Strict monotonicity
(`nabla t` PD-symmetric-part; Dafermos 1980) guarantees the VI *solution* exists
and is unique, but NOT that the *diagonalization algorithm* converges to it — that
needs the stronger contraction/diagonal-dominance condition of Dafermos (1982)
plus augmented costs staying positive along the iteration. Positive,
diagonally-dominant `C` (the shipped anchor) converges; a competitive/skew `C`
with negative off-diagonals can drive an augmented cost non-positive from the
route-concentrated free-flow start, at which point the solver stops and emits a
flow the certificate CENSORS (feasible=0) — never a false accept, but not a
solution. Neither is enforced at construction (both depend on the flow); the
always-reported VI residual makes non-convergence visible rather than hiding it.

## Sourcing

Dafermos (1980, *Transportation Science* 14(1):42-54, `dafermos1980traffic`) is
the VI formulation; Smith (1979, *Transportation Research Part B* 13(4):295-304,
`smith1979existence`) is the equivalent existence/uniqueness characterization
(itself a `not-a-solver` grounding reference for the FW family's convergence, now
also grounding this VI model); the diagonalization algorithm is Dafermos (1982).
Both primaries attributed unread; the VI condition, monotonicity uniqueness, and
diagonalization are cross-verified from the open Boyles et al. TNA non-separable
-cost chapter. The anchor numbers are hand-derived here, not quoted.
