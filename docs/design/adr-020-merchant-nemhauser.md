# ADR-020: merchant-nemhauser — Merchant & Nemhauser (1978) exit-function SO-DTA

**Status:** accepted (implemented)
**Date:** 2026-07-10
**Deciders:** analytical-DTA track — the first network DTA model
**File:** `docs/design/adr-020-merchant-nemhauser.md`

## Context

Merchant & Nemhauser (1978) is the first dynamic traffic assignment model of any
kind: a discrete-time, single-destination, SYSTEM-OPTIMAL program that routes
time-varying node demands `d_j(t)` through links whose per-period outflow is
governed by exit functions `g_a(x_a)` of the start-of-period occupancy — congestion
endogenized two decades before the CTM. It is a network-wide *optimization*
paradigm (the decision is the whole inflow/routing plan), distinct from the repo's
route-choice equilibria, the `dnl/` loading kernels, and the `bottleneck/`
departure-time equilibrium — so it gets its own parallel module `src/tabench/dta/`.

## Decision

1. **A parallel module `src/tabench/dta/`** touching no road/DNL/transit/bottleneck
   code (golden Braess hash `cf00f411…` re-asserted in the tests).
   `SODTAScenario` (`scenario.py`) is frozen and content-hashed under a
   `"tabench-dta-scenario-v1;"` domain prefix: nodes, one absorbing destination,
   links, per-link piecewise-linear concave exit functions (pointwise min of
   affine pieces; a mandatory `(slope <= 1, intercept 0)` piece enforces the
   standing M-N assumptions `g(0) = 0` and `g(x) <= x`; slopes `>= 0` keep only
   the nondecreasing branch, exactly as M-N assumed), a `(T, n_nodes)` demand
   table, and per-link cost weights `w_a > 0`.

2. **The canonical program is the Carey (1987) relaxation** (`solve.py`):
   explicit exit variables with `e_a(t) <= g_a(x_a(t))` instead of M-N's equality
   `e = g(x)`. The equality form is nonconvex (Carey 1992); the relaxation is the
   standard convex reading, under which piecewise-linear `g` and linear costs make
   the whole program an LP — one `e - s*x <= c` row per affine piece. Slack in the
   bound is deliberate *holding back* (ramp metering), legitimate and sometimes
   strictly optimal SO behaviour (see anchor B). Time semantics (cross-verified,
   see Sourcing): `x_a(t)` is the start-of-period occupancy; inflow during `t`
   joins the state at `t+1` (cannot exit during `t`); exits of period `t` feed
   downstream inflows of period `t` (same-period hand-off); the destination
   absorbs. Two benchmark conventions close the program: the network starts empty
   (`x(0) = 0`), and **terminal clearance `x(T) = 0`** — without it "total cost"
   is ill-posed (flow can strand more cheaply than delivering); an infeasible LP
   therefore *means* the horizon is too short. `solve_so_dta` solves the LP
   (HiGHS) and emits a `DTATrajectory` — inflows/exits/occupancies plus the LP
   dual vectors in the documented canonical row order.

3. **P1 certificate** (`metrics/dta_gaps.py`, `SODTAEvaluator`). From
   `(scenario, emitted trajectory)` alone the harness recomputes every gate:
   conservation `x(t+1) = x(t) + u(t) - e(t)`, per-period node balance
   `sum u_out = d + sum e_in`, the exit bound `e <= g(x)` with `g` re-evaluated
   from the scenario's pieces, empty start, nonnegativity, and terminal clearance
   (stranded flow is CENSORED). Scored: recomputed `total_cost` and
   `so_optimality_gap = (total_cost - Z*)/max(1, |Z*|)` against a harness-resolved
   canonical-LP optimum `Z*` (the `TransitEvaluator` LP-optimum pattern). If the
   trajectory carries a dual certificate the harness ADDITIONALLY verifies global
   optimality by pure arithmetic — `y_ub <= 0`, reduced costs
   `c - A_eq'y_eq - A_ub'y_ub >= 0`, and the duality gap against the RECOMPUTED
   primal cost (`dual_gap`, `dual_infeasibility`; weak duality makes a passing
   certificate a proof, and a forged one is reported, never believed).
   `exit_slack_max` reports holding-back as a non-gating Tier-B diagnostic.

## Analytic anchors (machine-verified — `test_dta_mn.py`)

- **Anchor A, `mn_parallel_scenario` (capacity metering):** 6 vehicles at `t=0`
  choose between a fast capacitated link (`g = min(x, 2)`, 1 period) and a slow
  uncapacitated 2-link route (2 periods); T=5, unit weights. SO = **10**
  vehicle-periods by an aggregate earliest-arrival bound (`E(0)=0`, `E(1)<=2`,
  `E(tau)<=6` gives cost `>= 30-24 = 10`; achieved by any split sending 2–4
  vehicles down the fast route). Every optimum exits the fast link at its
  capacity bound in period 1. The all-on-fast plan costs 12 and certifies
  `gap = 0.2`.
- **Anchor B, `mn_metering_scenario` (holding back strictly optimal):** series
  `O -A-> M -B-> D`, `g_A = min(x,2)` at weight 1, `g_B = min(x,1)` at weight 2,
  4 vehicles, T=7. Relaxed SO = **18**, and EVERY optimum meters A at rate 1
  while `g_A(x_A(1)) = 2` — strict slack in the exit bound; the naive M-N
  equality dynamics are decision-free here and cost **22**. (The strict gap
  needs the unequal weights: with uniform weights this instance ties at 14 —
  consistent with the literature's account of holding back.) Both anchors carry
  exact LP-duality certificates (verified to zero residual in the tests, and to
  exact rational arithmetic during research).
- Censoring: wrong hash, teleported vehicles, node imbalance, exit-bound
  violations, stranded flow, and negative flows are all censored; forged dual
  certificates are reported with large `dual_gap`/`dual_infeasibility` while the
  untouched primal still certifies.

## Alternatives considered

- **Implementing the M-N equality form with a local NLP solver:** rejected — the
  equality form is nonconvex (Carey 1992), a local KKT point carries no global
  certificate, and the benchmark's P1 discipline wants machine-checkable
  optimality. The Carey relaxation is the field's standard convex reading and
  admits the LP-duality proof.
- **Reusing the `dnl/` stack:** rejected — exit-function dynamics (outflow a
  function of total occupancy, uncapacitated inflow, no spillback) are a
  different physics from the S/R kinematic-wave interface; forcing them into
  `LinkModel` would misrepresent both. The CTM connection is historical, not
  structural: with cells sized to one-period free-flow, M-N's dynamics restricted
  to the sending branch coincide with an optimization CTM (Carey & Watling 2012),
  which is exactly the `lp-so-dta` (Ziliaskopoulos 2000) sprint — a separate
  model.
- **Terminal clearance as a certifier gate only (not an LP constraint):**
  rejected — then the reference `Z*` could strand flow (cheaper than delivering
  near the horizon), and a delivering trajectory would be measured against an
  incomparable bound.

## Adversarial review

Three independent attack lenses, each executing code against the module. What
survived: formulation fidelity (the canonical LP matrices equal hand-written
ones on tiny instances; 40/40 differential agreement with an independently
written LP; time semantics exact — no instantaneous traversal, k-link chains
need exactly k periods), both anchors' strongest claims re-proven independently
(SO=10 with `E(1)=2` forced in EVERY optimum; SO=18 with strict holding back in
EVERY optimum via optimal-face probing; equality form 22; uniform-weight tie
14), and the dual-certificate machinery (forgery is mathematically blocked —
the maximum dual objective over the evaluator's exact feasibility set equals
`Z*`, verified by solving that LP; HiGHS marginals pass at ~1e-15 over 25
random instances with zero false-censors; NaN/inf/wrong-shape forgeries all
rejected). What broke, all CONFIRMED with repros and fixed + regression-pinned:

- **CRITICAL false-accept:** the per-cell tolerance `eps = tol*max(1, demand.sum())`
  let ~eps-sized residuals accumulate over the `T*(links+nodes)` cells into a
  material teleport — a cheating trajectory certified `feasible=1` with
  `so_optimality_gap = -9.9e-4` while ~496 of 1e6 vehicles vanished. Fixed with
  two-scale gates (per-cell residuals at the local `tol*max(1, demand.max())`
  scale AND each violation family's absolute sum capped by the aggregate mass
  budget `tol*max(1, demand.sum())`), a delivery gate, and the weak-duality
  backstop: any cost undercutting the harness's own `Z*` beyond `tol` is
  CENSORED (no feasible plan can beat `Z*`, so undercutting proves infeasibility).
- **MAJOR:** negative-occupancy cost credit — shifting an honest optimum's
  occupancies down by `0.99*eps` kept `feasible=1` while scoring below `Z*` on
  the shipped anchor. Fixed: cost is computed on occupancies clamped at zero
  (plus the aggregate/undercut gates above).
- **MAJOR:** `certify()` raised on a gate-passing trajectory whenever the
  scenario's LP was unsolvable, violating the only-wrong-shapes-raise contract.
  Fixed: `Z*` resolves eagerly in `__init__`, so an unclearable scenario is a
  configuration error at construction, never a crash on model output.
- **MINOR:** the validator admitted scenarios that are infeasible for EVERY
  horizon (binding intercept-0 exit slope < 1 decays geometrically and never
  reaches `x(T)=0`). Fixed: the binding intercept-0 piece must have slope
  exactly 1 (`g(x)=x` near empty — Carey & Watling's coordinated
  discretization; longer free-flow times are link chains).
- **MINOR:** the "frozen" scenario's arrays were mutable in place, silently
  desyncing the content hash from an evaluator's cached `Z*`. Fixed: arrays are
  copied in and frozen read-only.
- **MINOR:** `max(1, |Z*|)` normalization reported absolute (not relative) gaps
  on small-optimum instances, hiding a 22%-suboptimal plan behind a 4e-3 gap.
  Fixed: both gaps normalize by `Z*` itself (strictly positive by validation).

Known scalability note (accepted, not fixed): `canonical_lp` assembles dense
constraint matrices — fine at anchor scale (~KB) but O((3LT)^2) memory; move to
`scipy.sparse` before shipping large generated families.

## Consequences

The benchmark gains the founding network-DTA model and the `dta/` module the
remaining analytical-DTA sprints (Friesz VI-DUE; Ziliaskopoulos LP SO-DTA on CTM)
will build alongside. All changes are additive (a new module + a new certifier +
tests), so the 605-test suite, every existing hash, and the golden Braess content
hash are byte-untouched.

## Sourcing

Merchant & Nemhauser (1978a) "A Model and an Algorithm for the Dynamic Traffic
Assignment Problems," *Transportation Science* 12(3):183–199,
doi:10.1287/trsc.12.3.183, and (1978b) "Optimality Conditions for a Dynamic
Traffic Assignment Model," same issue 200–207, doi:10.1287/trsc.12.3.200 — both
paywalled, attributed (abstracts read; no page-precise content reproduced). The
formulation, time-indexing semantics, exit-function assumptions, relaxation, and
nonconvexity landscape are cross-verified from >= 2 of these OPEN sources (read):
Carey & Watling (2012) (Program S restatement + M-N/CTM App. A), Carey &
McCartney (2004) (discrete dynamics + exit-function branches), the
Friesz/Kwon/Bernstein handbook chapter (dynamics + node balance), Lafortune et
al. (1993) §2 (uncapacitated links, relaxation reading), and Peeta &
Ziliaskopoulos (2001) §2.1 (relaxation tightness, holding back). Carey (1987)
*Oper. Res.* 35(1):58–69, doi:10.1287/opre.35.1.58 (convex relaxation) and Carey
(1992) *Transp. Res. B* 26(2):127–133, doi:10.1016/0191-2615(92)90003-F
(nonconvexity) are in the verified canon. Both anchors were derived from scratch
for this benchmark (no canonical M-N instance exists) and verified by LP +
exact rational duality certificates.
