# ADR-009 — Side-constrained UE: hard link capacities with a link-visible feasibility certificate

**Status:** accepted (shipped in v1)
**File:** `docs/design/adr-009-side-constrained-ue.md`

## Context

Ordinary UE lets a link's flow grow without bound (the BPR cost just rises). **Side-constrained
traffic assignment** (Larsson & Patriksson 1995) adds **hard link-capacity constraints**
`v_a <= u_a` — physical throughput limits — to the Beckmann program:

```
min_x  sum_a integral_0^{v_a} t_a(w) dw   s.t.  demand feasibility,  v_a <= u_a  for all a.
```

Its KKT conditions are a Wardrop equilibrium on the capacity-**augmented** cost

```
c_a(v) = t_a(v_a) + beta_a,   beta_a >= 0,   beta_a (u_a - v_a) = 0,
```

where `beta_a` is a multiplier that is zero off the binding set and, where a capacity binds, is
the **queueing delay / congestion toll** that stops travelers piling onto the physically-cheap
but full link (Larsson & Patriksson 1999: "the Lagrange multipliers … are the link tolls the
travellers are willing to pay … the delays in steady-state link queues"). When no capacity
binds, SC-TAP is literally the unconstrained program, so it reduces **exactly** to plain UE.

## Sourcing

Larsson & Patriksson (1995, *TR-B* 29(6):433-455) is paywalled and **attributed unread**; the
augmented-cost equilibrium, the multiplier-as-toll reading, and the augmented-Lagrangian form
are cross-verified from the 1999 companion, Nie-Zhang-Lee (2004), and standard
augmented-Lagrangian theory (Bertsekas 1982). The analytic anchor numbers are **derived here**,
hand-checked, and not quoted from the primary.

## Decision 1 — Per-link capacities as content-hashed scenario data

`Scenario.side_capacities: np.ndarray | None` (`core/scenario.py`) carries the hard caps `u_a`
(length `n_links`). Validated finite and `> 0`; **content-hashed only when set** (appended last,
golden Braess hash preserved); **mutually exclusive** with `sue_theta` / `elastic_demand` /
`combined_demand` / `br_epsilon`. A change in any `u_a` is a different benchmark instance.

## Decision 2 — Method of multipliers wrapping Frank-Wolfe

`sc-tap` (`SideConstrainedModel`, paradigm `static_sc_ue`) solves it by the augmented-Lagrangian
method of multipliers. The inner problem, for fixed `(beta, rho)`, is an ordinary UE with the
modified — still non-decreasing — link cost `t~_a(v) = t_a(v_a) + max{0, beta_a + rho(v_a-u_a)}`,
solved by Frank-Wolfe with an exact Brent line search on the augmented objective. The outer loop
updates `beta_a <- max{0, beta_a + rho(v_a - u_a)}` and grows `rho` when the worst violation
stops shrinking. At the fixed point the constraints hold exactly and the recovered `beta_a` is
the true multiplier (unlike a fixed large-penalty solve, which is only feasible as `rho ->
infinity`).

**Robustness (pre-emptive fuzz).** An **infeasible** instance — a capacity below a cut link's
*forced* flow, where no SC solution exists — would otherwise drive `beta`/`rho` to overflow (a
crash). We cap `rho <= 1e10` and `beta <= 1e8` (both far above any real queueing toll for the
benchmark's cost scales) and break on any non-finite cost or shortest-path failure, so an
infeasible instance stops gracefully with the constraint reported violated rather than crashing.

## Decision 3 — A link-visible capacity-feasibility certificate

The SC-specific scored quantity is **capacity feasibility**, link-visible and checked **per
link** to a tight *relative* tolerance (unlike the multipliers, which are duals). A hard cap is
a per-link quantity, so the tolerance is relative to each link's own capacity — scaling it by
total demand (as the demand-feasibility audit does) would let a fixed absolute overload certify
on a high-demand network (an adversarial-review finding, corrected before this commit):

```
max_capacity_violation = max_a (v_a - u_a)+                # absolute, for diagnosis
rel_overload           = max_a (v_a - u_a)+ / u_a          # per-link relative
sc_capacity_feasible   = 1.0  iff  rel_overload <= feasibility_tol
```

The **raw-cost relative gap stays positive** at a correct SC equilibrium (binding links carry
flow that would prefer to grow), so it is reported for provenance but is **not** the acceptance
criterion — the acceptance criterion is capacity feasibility. The recovered `beta_a` and the
augmented-cost gap are model self-reports; a fully harness-recomputed augmented-cost equilibrium
gap (recovering `beta` as shadow prices on the binding set by a small convex program) is a
documented enhancement. This is honest: the scored certificate certifies *feasibility*, not the
full equilibrium; equilibrium is validated by the analytic anchor and the no-binding UE
reduction.

## Consequences

- **New:** `Scenario.side_capacities`; paradigm `static_sc_ue`; the `sc-tap` model;
  `sc_capacity_feasible` + `max_capacity_violation` scored metrics; `sc_two_route_scenario`
  anchor (no-binding → exact UE `(5.5,5.5,4.5,4.5)`; cap 4 → `(4,4,6,6)`, recovered `beta = 3 =
  1 + D - 2*cap`); `tabench run --scenario sc-tworoute`.
- **Validation:** the anchor + monotone-tightening sweep; the exact no-binding UE reduction
  (matches the shipped FW solver); a fuzz confirming **zero crashes on solvable instances** and,
  by a min-cut classification, that sc-tap converges to capacity-feasibility on **100% of
  feasible instances** and only reports infeasibility on genuinely-infeasible ones.
- **Unchanged:** every prior scenario hash (golden Braess preserved); all other certificate
  paths; all prior models and tests.
- **Deferred:** the harness-recomputed augmented-cost equilibrium gap (shadow-price recovery);
  per-class capacities; explicit infeasibility *reporting* (beyond `sc_capacity_feasible = 0`).
