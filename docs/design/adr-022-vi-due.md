# ADR-022: vi-due — Friesz et al. (1993) VI dynamic user equilibrium

**Status:** accepted (implemented)
**Date:** 2026-07-10
**Deciders:** analytical-DTA track — the dynamic USER equilibrium (closes the track)
**File:** `docs/design/adr-022-vi-due.md`

## Context

Friesz, Bernstein, Smith, Tobin & Wie (1993) cast the simultaneous
route-AND-departure-time (SRDC) dynamic user equilibrium as a variational
inequality over path departure-rate profiles: find `h*` in the volume-feasible
set (`sum_p int h_p = Q` per OD) with `<Psi(h*), h - h*> >= 0`, where
`Psi_p(t, h)` is the *effective delay* (traversal plus schedule-delay penalty).
At a DUE every USED (route, departure-time) pair carries equal minimal
effective delay and unused pairs cost at least as much — the field's standard
DUE definition, first stated in this paper. The VI is loading-agnostic; the
1993 paper's own link-delay model is FIFO-safe only for affine delays, so the
computable-DUE literature instantiates the same VI with the *generalized
Vickrey* point-queue loading (Han, Friesz & Yao 2013 — the standing existence
theory). That is exactly the loading this repo already ships (`bottleneck/`,
ADR-019), which makes the faithful, certifiable benchmark unit: **the Friesz
VI on parallel Vickrey-bottleneck routes** — one route with `f = 0` reduces
literally to the shipped `vickrey` model, and the route axis adds the
simultaneous-choice content the single-bottleneck model lacks.

## Decision

1. **Extend the `bottleneck/` module** (`due.py`) — the machinery *is* the
   departure-time-equilibrium machinery. `DUEScenario` (frozen, read-only,
   hashed under `"tabench-due-scenario-v1;"`) is the Vickrey scalars plus
   per-route `(f_r, s_r)` arrays. The analytic structure is exposed on the
   scenario: with `delta = beta*gamma/(beta+gamma)` each used route runs its
   own Vickrey equilibrium at the common cost level `C`, so
   `N_r = s_r (C - alpha f_r)/delta` and

       C = (delta*N + alpha * sum_U s_r f_r) / sum_U s_r

   with the used set `U` found by a greedy sweep in increasing-`f` order
   (a route is used iff `alpha f_r < C`). `due_closed_form` emits a
   `DUEProfile` — per-route cumulative departure curves on one shared grid
   (the multi-route `BottleneckSchedule`).

2. **P1 certificate** (`metrics/due_gaps.py`, `DUEEvaluator`). From
   `(scenario, emitted profile)` alone: censor gates (hash, finiteness, grid
   monotonicity, zero starts, nondecreasing rows, TOTAL-volume conservation —
   the per-route split is deliberately ungated: the DUE chooses it); per route,
   shift by `f_r`, extend the grid until the queue provably clears, simulate
   the deterministic point queue, and score PER TRAVELER by level inversion of
   both curves (the ADR-019 lesson, unchanged). The route axis introduces a
   genuinely new failure mode the single-route certifier cannot see: an
   all-on-one-route profile equalizes its own used costs while the idle route
   is strictly cheaper. The reference minimum therefore scans the
   **marginal-insertion cost of every route** (used or not) — an infinitesimal
   deviator joins behind `R_r(t)` travelers, exiting at
   `max(t + f_r, D_r^{-1}(R_r(t)))` — over the profile's grid UNION a dense
   harness-chosen sweep of the analytic window UNION `{t* - f_r}` (a profile
   cannot hide the cheap region by its choice of grid). Score:

       due_gap = (max used per-traveler cost - min marginal cost anywhere) / C

   `0` iff the discretized Friesz DUE conditions hold across BOTH choice
   dimensions; positive with the incentive to deviate in time OR route.
   Tier-B: recomputed totals (cost, expected cost, max queue, travel delay).

## Analytic anchors (machine-verified — `test_due.py`)

Worked instance `friesz_two_route_scenario` (`N=6000`, routes `(f=0.2,
s=3000)` and `(f=0.7, s=1500)`, `alpha=1, beta=0.5, gamma=2, t*=9`):

- `C = 0.9`; split `(5250, 750)`; queue costs `(0.7, 0.2)`; windows
  `[7.4, 9.15]` and `[7.9, 8.4]`; total cost `C*N = 5400`; max queue 2100.
  The closed form certifies `due_gap = 0`.
- **Both-used threshold:** routes 2 is used iff
  `N > alpha*s_1*(f_2 - f_1)/delta = 3750`; at `N = 3000` only route 1 runs
  (`C = 0.6 < alpha f_2 = 0.7`) and still certifies.
- **Equal free-flow times:** the split is capacity-proportional,
  `C = delta*N/(s_1+s_2)`.
- **Single-route `f = 0` reduction:** identical `C*` to the shipped `vickrey`
  model, the emitted curve matches `ue_closed_form` pointwise, and the DUE
  profile certifies `equilibrium_gap ~ 1e-13` under the ADR-019
  `BottleneckEvaluator` — a cross-certifier consistency pin.
- **False equilibria rejected:** all-on-one-route scores exactly
  `(1.0 - 0.7)/0.9 = 1/3` (the marginal-insertion scan); the ADR-019
  burst-dump and an SO-style metered profile score large positive gaps;
  non-conserving/non-monotone/wrong-hash profiles are censored.

During research the closed form was additionally verified by an independent
complementarity-bisection solve (scalar monotone fixed point on `C` — machine
precision agreement) and by discrete-event queue simulation; naive fixed-point
dynamics (Smith swap, raw projection) were confirmed to CYCLE on this operator
(it is non-monotone — the known instability of departure-time adjustment), and
a projection iterate's ergodic average converges in aggregates while its
profile is NOT an equilibrium (repo-certified gap 0.48) — exactly why the P1
certifier, not the solver's claim, is the arbiter.

## Alternatives considered

- **A general-network VI-DUE with grid-based fixed-point dynamics:** rejected
  for this sprint — the effective-delay operator is non-monotone and the naive
  dynamics demonstrably cycle; shipping a solver whose convergence cannot be
  certified would put the benchmark's weight on luck. The parallel-bottleneck
  class has the exact closed form, full existence theory under GVM loading,
  and machine-checkable certificates; a general-network extension belongs with
  a route-and-time column-generation sprint of its own.
- **The 1993 link-delay-model loading:** rejected — FIFO-safe only for affine
  delays, uncapacitated exits, and superseded by the GVM instantiation in the
  computable-DUE literature; the VI itself is loading-agnostic, so the GVM
  choice is faithful to the formulation.
- **Gating the per-route split:** rejected — the DUE *chooses* the split;
  identical routes make it non-unique. Only total volume is conservation.

## Adversarial review

Three independent attack lenses were launched; in round 1 two of them
(soundness, formulation) died on an infrastructure session limit and ONLY the
numerics lens completed — so the two dead lenses were re-run as a fresh round
2 against the round-1-fixed code. Full coverage was reached across the two
rounds; every finding below is CONFIRMED by an executed repro and every fix is
regression-pinned in `test_due.py`.

**Round 1 (numerics lens).** (a) MAJOR: the certifier extended the served
curve past the horizon with a 65-point *interpolated* linspace, bending the
queue-clearing kink — the same piecewise-linear plan scored up to 25.6×
differently depending on where its emitted grid ended. Fixed by
reconstructing the point-queue served curve EXACTLY: within each segment the
arrival rate is constant, so the queue either serves at `s_r` or tracks the
arrival curve, with at most one interior queue-empties kink — computed
analytically and inserted, plus the exact post-horizon clearing chord.
(b) MINOR: the greedy used-set broke at `k = 1` when `alpha*f_1` was so large
that the queue term rounded away (`C = inf`); the cheapest route is now
always used. (c) MINOR: `N_r = s(C - alpha*f)/delta` cancelled
catastrophically at large `f` (the closed form missed its own conservation
gate); replaced by the difference form
`Cq_r = (delta*N + alpha*sum s_i(f_i - f_r))/sum s_i` plus exact
renormalization. (d) MINOR: degeneracy floors `alpha - beta >= 1e-9*alpha`,
`gamma <= 1e9*beta` added as documented domain bounds. NOTEs: the per-step
monotonicity gate was replaced by a running-max total-retraction gate (the
DTA eps-accumulation family). Survived: greedy vs `2^R` brute force on 12k
draws; closed-form exactness ~2.3e-10 across grid densities; cross-certifier
agreement with the adr-019 evaluator.

**Round 2 (soundness + formulation lenses, on the fixed code).** Both lenses
independently converged on the same CRITICAL: the marginal-insertion
reference scan swept `linspace(min(span_lo, t[0]), max(span_hi, t[-1]),
4001)` — a fixed COUNT over a hull the solver's own grid stretches — and the
true cheap-insertion times (the queue-drain dip, where waiting for a residual
queue to vanish is cheap) are kinks that were never candidates. Executed
repros: a truncated-isocost two-route profile in which EVERY traveler pays
exactly 2.0 while deviating into the hidden drain window costs 1.6 (a 20%
improvement) certified `due_gap = 0.0` once one flat pad point at `t = 1e6`
diluted the sweep; a single-ramp profile with true gap 0.765 certified
−0.0001. FIXED by making `min_ref` EXACT: the marginal-insertion cost is
piecewise linear in the insertion time with enumerable kinks — the profile's
grid, the pullbacks `A^{-1}` of every served-curve kink level, the pullback
of level `S(t*)`, and the queue-vanishing zeros of `A(t) - S(t + f_r)` — all
enumerated per route, with the dense sweep demoted to belt-and-braces over
the FIXED analytic window. The repros now score 0.2027 and 0.7653, invariant
under pads to `1e7`. Also confirmed and fixed: MAJOR — scenarios with
`N <= tol` skipped every route and certified arbitrary conserving garbage at
`due_gap = -inf` (now censored explicitly); MAJOR — in-domain instances with
`(beta+gamma)*ulp(t*)` comparable to `C` score the honest closed form at gap
up to 1.0 because no float64 profile can resolve the equilibrium (98/1100
fuzz draws pre-fix; now rejected at construction by a conditioning gate
`(beta+gamma)*eps_mach*(|t*|+window) <= 1e-9*C`); MINOR — the level sampling
excluded the first/last traveler where the cost supremum is often attained
(boundary levels `eps`/`n_r - eps` now sampled); NOTEs — Tier-B totals now
use a grid-size-independent level count plus the switch/`t*`-crossing levels.
Survived round 2: greedy vs exact `2^R` KKT enumeration on 6000 draws (zero
structural disagreements, near-ties at 1e-13, thresholds at ±1e-15); an
independent event-driven point-queue simulator matching the reconstructed
served curves on 120 profiles to <1e-9; all 15 numeric anchors re-derived
independently; gate-boundary float-dirt probes; the round-1 fixes themselves.
Post-fix, the reviewers' 2200-draw property fuzz re-run: every accepted draw
certifies its closed form at worst `|due_gap| = 8.4e-9` (pre-fix: up to 1.0),
zero false-censors, zero negative gaps, zero crashes.

## Consequences

The analytical-DTA track is complete: departure-time UE (`vickrey`), SO-DTA in
both classical formulations (`merchant-nemhauser`, `lp-so-dta`), and the
dynamic USER equilibrium (`vi-due`) — Wardrop's two principles, dynamic, under
one P1 pattern. All changes are additive; the 652-test suite and the golden
Braess hash are byte-untouched.

## Sourcing

Friesz, Bernstein, Smith, Tobin & Wie (1993) "A Variational Inequality
Formulation of the Dynamic Network User Equilibrium Problem," *Operations
Research* 41(1):179–191, doi:10.1287/opre.41.1.179 — paywalled, attributed.
The VI statement, SRDC-DUE conditions, effective-delay operator, and the
link-delay-model caveats are cross-verified from open sources read in full:
Han, Friesz & Yao (arXiv:1211.0898 — the GVM instantiation and existence
theory; also arXiv:1211.4621), the Friesz–Kwon–Bernstein handbook chapter
(nested exit-time composition, LDM history), and Peeta & Ziliaskopoulos
(2001) §2 (the 1993 attribution of the DUE conditions). The parallel-route
closed form is standard bottleneck-theory material (Arnott–de Palma–Lindsey
lineage, attributed), derived from scratch here from the isocost conditions
and machine-verified by LP-free bisection + discrete-event simulation; it is
deliberately NOT attributed to the 1993 paper, which is a
formulation/existence paper with no closed form. No page-precise quotes
reproduced.
