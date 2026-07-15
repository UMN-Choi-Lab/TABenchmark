# ADR-031: pm-td-ue / pm-td-so — Peeta & Mahmassani (1995) time-dependent SO/UE

**Status:** accepted (implemented)
**Date:** 2026-07-15
**Deciders:** analytical-DTA track — the first *iterative, simulation-based* time-dependent route-choice equilibrium in the benchmark
**File:** `docs/design/adr-031-peeta-mahmassani.md`

## Context

Peeta & Mahmassani (1995), "System optimal and user equilibrium time-dependent
traffic assignment in congested networks" (*Annals of Operations Research*
60:81–113), is THE methodology paper of the simulation-based DTA school — the
DYNASMART-P lineage. It defines two problems on a general multi-origin/
multi-destination network with a *given* time-dependent OD table and FIXED
departure times, the decision being route choice only (the per-departure-interval
path split `r_ijk^tau`):

* **TD-UE** (Eq. 5): a temporal extension of Wardrop's first principle on
  EXPERIENCED path travel times — for every `(i, j, tau)`, used paths equalize
  their experienced time and unused paths cost at least as much. Equilibrium on
  time-dependent *instantaneous* times is, in the paper's own words,
  "conceptually meaningless."
* **TD-SO** (Eqs. 3/15): minimize total system travel time; the optimality
  conditions equalize the time-dependent path MARGINAL travel time (Eq. 6 =
  experienced time plus the aggregate externality), whose computable form uses
  the local link marginal `t_ta = T_ta + x_ta ∂T_ta/∂x_ta` (Eq. 18).

Both are solved by ONE method of successive averages (MSA, step `1/(l+1)`) around
a single mesoscopic loading (DYNASMART used strictly as a simulator), differing
ONLY in the path cost the all-or-nothing direction minimizes — average
experienced time (UE) vs path marginal time (SO). The program is nonlinear,
nonconvex, and the loading map `F` (constraint (3g)) has "no known analytical
functions"; the paper claims no convergence guarantee, only empirical
convergence, and no gap or duality object is ever computed.

Against the shipped rows this is a genuinely new object. `merchant-nemhauser`
(ADR-020) and `lp-so-dta` (ADR-021) are single-destination, link/cell-based **SO**
LPs where holding is legal and LP duality certifies optimality — precisely the
"exit-function" world P&M's §2.3.3 rejects (holding "not acceptable socially nor
realistic operationally"; simulation "implicitly precludes" it). `vickrey`
(ADR-019) is departure-time-only, no routes; `vi-due` (ADR-022) is
*variable*-departure-time SRDC-DUE on parallel Vickrey point queues with a closed
form and no iteration. Nothing shipped does route choice ON a dynamic network
loading, MSA on a DNL, multi-destination DTA, or experienced-time equilibrium on
a general network. pm-1995 is the first.

The DYNASMART binary is FHWA/McTrans-licensed with no public artifact, so the
adapter path is deferred (ADR-030); the white-box sibling — the paper's
architecture (simulation defines `F`) instantiated on the repo's OWN CTM/LTM
loading — is the row that ships. The paper's own §3.2/§2.3 architecture names
exactly what `F` must provide (discrete-time loading producing experienced path
times, link/node conservation, FIFO in the per-interval average sense, no
artificial holding, congestion-endogenous times, and a two-grid structure); the
repo's `dnl/` CTM/LTM + `TampereNode` satisfy all of them by construction. This
is a *formulation-vs-loading* substitution, stated exactly as ADR-022 declared
the GVM loading for the 1993 VI: faithful to the paper's architecture, different
physics (the repo's kinematic-wave loading, not DYNASMART's modified-Greenshields
vehicle packets).

## Decision

Ship a parallel module `src/tabench/tdta/` (like `dnl/`, `dta/`, `bottleneck/`),
touching no existing hashed class, certifier, or evaluator.

1. **Artifact `TDPathFlows` (decisions only).** `departures[p, k]` = vehicles on
   declared path `p` departing in step `k`, plus `scenario_hash`. Nothing else:
   cumulative link curves, experienced times, and TSTT are all consequences the
   harness recomputes by running its OWN loading of the emitted departures (the
   ADR-022 lesson, sharpened — there are no emitted curves to be inconsistent
   with, so ADR-010's C8 aggregate-observability attack class is structurally
   absent).

2. **Scenario `TDTAScenario` (new frozen type, hash domain
   `"tabench-tdta-scenario-v1;"`).** Composes the DNL primitives read-only
   (`Network`, `LinkDynamics`, `DynamicDemand`, `TimeGrid`) and adds the
   enumerated per-OD path set (the decision universe, P2 data) and the loading
   `kernel` (`ctm`/`ltm`, hashed — the certificate is defined w.r.t. one loading
   operator). Domain-separated, so no existing dnl/dta/static hash can move
   (golden Braess `cf00f411…` re-asserted byte-identical).

   **v1 topology restriction — the decidability guarantee.** The union of path
   links has NO effective interior diverge: at every interior node each incoming
   link is followed by exactly one outgoing link across all paths (one-hot turn
   rows), so paths branch only at their origins and may only merge downstream.
   Per-commodity experienced times are then EXACTLY decidable from aggregate link
   curves (the origin branch is model-controlled and observed; a merge preserves
   attribution because each in-link's outflow is observed separately, and FIFO
   does the rest). This designs out ADR-010's C8 multi-in undecidability rather
   than tolerating it. A consequence used throughout: each path owns a PRIVATE
   first link, so per-path origin injection needs no multi-out origin split —
   ADR-010's deferred `OriginNode` placeholder is sidestepped, not reused.
   General interior diverges (per-commodity emission + time-varying turns,
   ADR-010 R7/R8) are the named v2 with a domain-string bump.

3. **Loader `PathLoader`.** A thin runner over the DNL S/R loop whose only new
   code is per-path first-link injection; interior merges/series reuse the
   shipped `TampereNode` with the one-hot turns the path set implies. The
   used-link graph is acyclic with positive capacities, so every conserving
   emission clears in finite time; the runner runs on an EXTENDED horizon (the
   original grid plus a clearing pad, zero new departures beyond `K`) so no
   traveler's experienced time is truncated — the DNL analogue of `due_gaps`'s
   post-horizon clearing chord. The aggregate emission is a full-width
   `DNLOutput`, so the shipped `dnl_gaps` C0–C8 certificate is a free correctness
   oracle on it (pinned on the corridor, merge, and wedge anchors).

4. **Certificate `TDTAEvaluator` (`metrics/tdta_gaps.py`), a pure function of
   `(TDTAScenario, TDPathFlows)`.**

   * **Gates (censor, not raise; only wrong shapes raise).** Hash mismatch;
     non-finite; and — at a **PER-OD** tolerance `eps_od = tol·max(1, D_od)`
     (three-lens review MAJOR: a single global `eps` scaled by the LARGEST OD let
     a tiny OD shift its whole demand or retract a real vehicle beside a huge OD)
     — negative departures at two scales per OD (a negative departure is a
     cumulative retraction, so this doubles as the retraction gate) and the
     **demand-match** gate: fixed departure times give the model ZERO timing
     freedom, so per OD and per grid edge the cumulative emitted departures must
     equal `DynamicDemand.cumulative` (per-edge AND aggregate mass, both scaled by
     that OD's own total). This closes every departure-time-gaming door at the
     gate, not the score. After loading, a two-sided delivery gate censors any
     emission that does not clear (assigned flow left in-network after the
     extended horizon).

   * **TD-UE score `tdue_gap`.** The discrete experienced-time route-swap
     residual (relative average excess cost, mirroring the static RG the
     leaderboard speaks): `(TC_used − TC_min) / TC_min`, where `TC_used` is the
     total experienced travel time the emission incurs and `TC_min` is the total
     it would incur if every traveler took their OD's cheapest available path at
     their own departure time — both by FIFO level composition on the harness's
     realized curves (origin-queue wait included: cost starts at the emitted
     departure time). The used cost and the reference minimum are the SAME
     marginal-insertion composition, so the used path is in the min set and the
     residual is `>= 0` by construction, exactly `0` iff no traveler can lower
     their experienced time by swapping routes — the discrete Wardrop conditions
     (5a)/(5b). The reference minimum scans EVERY declared path (used or not) at
     every traveler's departure time, so hiding the cheap path cannot dilute it.
     Tier-B: the per-traveler max-form `tdue_gap_max` (the worst single-traveler
     relative excess), the recomputed TSTT, and the max queue. The max-form is
     resolved at the LEVEL (count) domain — the max is over ACTUAL travelers, so
     a departure-time sweep would spuriously score hypothetical travelers inside
     departure plateaus — with the level set seeded by the pullback kinks of the
     composed cost (each free-flow entry crossing a grid edge, mapped to the
     departing traveler's level) plus a bisection zoom on the queue-clearing
     corners, so the reported peak matches a dense level reference to ~1e-6 and is
     never UNDER-reported (three-lens review MAJOR: an under-resolved sweep was
     model-flattering; the headline `tdue_gap` integral already agreed to ~1e-6).

   * **TSTT convention (three-lens review MAJOR).** The reported `tstt` — the
     system time incl. origin queue, and the SO comparison's primal — is the
     AVAILABILITY-based occupancy area `dt·Σ_k (D(t_{k+1}) − A(t_k))` with `D` the
     total cumulative emitted departures and `A` the arrivals. This (i) counts
     each vehicle only from when it is generated, not from `t=0` (the earlier
     `V − A` form charged the late-departing tail pre-departure waiting: a demand
     spread over several steps scored a spurious 30 where the true experienced
     total was 12), and (ii) is `>= 0` by construction (arrivals never exceed
     departures-so-far), so a sub-budget over-emission can no longer forge a
     negative TSTT that undercuts `Z*`. On a first-interval burst it coincides
     with the LP's initial-occupancy convention (corridor stays 33 = Z*), which is
     the convention the LP lower-bounds (the per-traveler EXPERIENCED total — 27
     on the corridor, strictly below the interval-counting Z* — is reported
     separately as `total_experienced_time`; it is NOT the SO-comparison quantity,
     because the LP does not lower-bound it).

   * **TD-SO score `so_bound_gap`.** `(TSTT − Z*) / Z*` where `Z*` is the
     `lp-so-dta` LP optimum on the CTM-cell instance DERIVED from the same grid
     (`derive_cell_scenario`: each cell → LP cell with `Q = q_max·dt`,
     `N = kappa·vf·dt`, `delta = w/vf`; each origin an inf-storage source with
     its total demand as initial occupancy; the single destination a sink),
     resolved eagerly at construction (HiGHS). The derivation requires the demand
     to be a burst that fits in ONE grid step (`breakpoints[1] <= dt`, review
     MAJOR): a demand spread over several steps loads gradually and avoids the
     queue the burst-as-initial-occupancy LP charges, so its `Z*` would be a
     spurious positive bound — such an instance reports `so_bound_gap = NaN`
     instead. The LP relaxes the Godunov flux to its four linear bounds, so its
     optimum PROVABLY lower-bounds every strict-CTM loading in this convention:
     `so_bound_gap >= −tol` always, and an undercut is CENSORED (the ADR-020/021
     weak-duality-undercut discipline — by weak duality no conforming loading can
     beat `Z*`, so an undercut is a proof of infeasibility, not a warning).
     Because no duality object exists for the nonconvex simulation program, the
     row **cannot** certify SO-optimality; it scores a bound gap and each SO
     anchor pins its own attainability (the LP is a *controlled* CTM, so the bound
     may be unattainable by any path-flow loading at interior merges —
     report-never-gate). A multi-destination (UE-only) instance simply reports
     `so_bound_gap = NaN`; the bound is never faked.

   UE and SO are the SAME evaluator (the paper runs the same machinery in two
   modes); both metric families are reported for any emission, so an SO
   emission's positive `tdue_gap` and a UE emission's positive `so_bound_gap` are
   the paper's own SO≠UE headline, machine-verified.

5. **Reference MSA solvers (`solve.py`, NON-certified).** Both are the paper's
   MSA with step `1/(l+1)`, sharing every component except the AON path cost —
   average experienced time (UE, §4.3) vs time-dependent path marginal time (SO,
   §4.1, local link marginals via the 3-point quadratic fit of Fig. 3, sampled at
   the traveler's link-EXIT interval so the externality is read from the queue it
   sits through, not the empty cell it enters). The enumerated path set collapses
   the paper's Ziliaskopoulos–Mahmassani TDSP + column generation to evaluating
   each declared path and AON-ing onto the minimum. The certifier — not the
   solver's claim — is the arbiter (the vi-due lesson): emit the best-certified
   iterate.

   **Model names.** Two registered names `pm-td-ue` / `pm-td-so` sharing one
   scenario/loader/evaluator (the `od-dynamic-sim`/`-seq` precedent), for the two
   tasks the one paper defines. One canon bibkey (`peeta1995system`), one row.

### Model-name and stopping-rule deviations (disclosed)

* **Stopping rule.** The paper stops on a solution-STABILITY count
  `N(eps) <= Omega` (Eq., §4.2) — it never measures a gap. The benchmark instead
  reports the certified `tdue_gap`/`so_bound_gap` of the best iterate on the
  deterministic track (P5): a simulation-based MSA provably has no convergence
  guarantee (the paper says so), so the certifier arbitrates and no convergence
  is claimed. This is STRICTER than the primary's own stopping rule.
* **Marginal estimator.** The SO marginal uses the paper's approximate LOCAL link
  marginal (the 3-point quadratic `∂T/∂x` fit), so an "equal-marginal" check is a
  diagnostic, not an optimality proof (unlike ADR-020/021 duality). The SO
  certificate therefore scores recomputed TSTT against the LP bound, NEVER
  solver-reported marginals (the F6 family below).

## Anchors (derived from scratch; the paper's numerics are irreproducible)

All bounds calibrated to MEASURED values on this box.

* **`pm_corridor` (cross-model pin, anchor C).** The ADR-021 `zil_corridor` as a
  single-path TDTAScenario. `derive_cell_scenario` reproduces the zil cells
  byte-for-byte (same content hash), so `Z* = 33`; the single path forces the
  split (`tdue_gap = 0`) and the per-path loader's TSTT equals the LP optimum
  `33` EXACTLY through the new code path — re-pinning the DynamicScenario↔cell-LP
  correspondence via `tdta`.
* **`pm_diamond` (exact TD-UE, anchor A).** Two byte-identical routes; by symmetry
  the exact TD-UE is the 50/50 split with equal experienced times, certifying
  `tdue_gap = 0` and `tdue_gap_max = 0` EXACTLY (TSTT 14 = `Z*`). The all-on-one
  control queues one bottleneck while the identical twin sits idle: the reference
  minimum (scanning every declared path) scores the hand gap `0.75` (and
  `so_bound_gap = 2/7`). Loaded under both kernels the scores agree to machine
  precision (the ADR-016 ltm==ctm pin).
* **`pm_wedge` (SO≠UE, anchor B — the paper's headline).** A fast capacitated
  route (`Q=1` bottleneck, free-flow 2) vs a slower uncongested route (free-flow
  3, ample capacity), 6 vehicles. `Z* = 23`. An explicit SO split (1 fast, 5
  slow) attains the bound EXACTLY (`so_bound_gap = 0`, TSTT 23) and is NOT a UE
  (`tdue_gap > 0`); a UE split (3/3) has TSTT 24 (`so_bound_gap = 1/23`); selfish
  all-on-fast is worst at TSTT 33 (`so_bound_gap = 10/23`). SO TSTT (23) < UE
  TSTT (24) — SO strictly beats UE, executable. **UE-label caveat (review
  MINOR):** with a first-interval burst the wedge has an intrinsic positive
  per-traveler gap floor at `dt=1`, so the per-interval MSA fixed point (the 3/3
  split) is NOT the per-traveler certificate minimizer — its certified `tdue_gap`
  is exactly `1/11 > 0` and a nearby split scores strictly lower. A
  certificate-zero UE exists only where per-traveler costs can equalize (the
  symmetric diamond); the SO<UE headline survives regardless (that is what the
  wedge pins), and the "UE (3/3)" label is a per-interval convention, disclosed.
* **`pm_merge` (attribution, anchor D).** Two origins (one two-route) feed a
  shared `Q=1` bottleneck via a `TampereNode` merge; per-commodity experienced
  times stay decidable (each in-link observed + FIFO). A balanced split certifies
  clean and attains the bound; the loader's aggregate emission passes C0–C8.

Measured runtimes: each anchor certification < 1 s; the MSA solvers (25–40
iterations, full loading + certification per iteration) ≈ 3–5 s each. The full
`test_tdta.py` runs in ≈ 26 s — comfortably CI-sized (the max-form's dense-
reference regression dominates). A P&M-scale 168-link replica is NOT CI-sized
(~8–10 min); CI anchors stay at diamond/corridor scale, the same boundary posture
as the dense-LP notes in ADR-020/021. The clearing pad is HARD-CAPPED at 20000
extension steps (drain estimated at the slowest USED-link capacity, review
MAJOR): a bounded censor-not-crash safety that could in principle bite a
legitimately enormous instance.

## Boundary (what the row must NOT claim)

Verified against the shipped solvers' code, not just the ADRs:

* Not the first DUE *concept* (Friesz conditions ship in `vi-due`), not the first
  SO-DTA (`merchant-nemhauser`, `lp-so-dta`), not the first MSA (`models/msa.py`,
  static), not the first experienced-time certificate (`due_gaps`
  marginal-insertion is experienced-time). The novel object is the *algorithmic
  architecture*: simulator-in-the-loop iterative TD route equilibrium,
  multi-origin/multi-destination, experienced-time.
* Single user class only. Multiple user classes (MUCTDTA) are the paper's own
  future work (Peeta thesis / Mahmassani et al.); `multiclass` (ADR-013) is
  static. Do not build a class axis on the 1995 paper's authority.
* Fixed departure times. Variable departure choice belongs to `vi-due` (SRDC);
  fixed departures is also what makes the demand-match gate — the strongest gate
  here — possible.
* NOT rolling-horizon. The paper explicitly RESERVES "dynamic"/real-time for its
  *TR-C* 3(1) sibling ([14]) and treats THIS problem, with full a-priori OD, as
  the *time-dependent* (TDTA) problem. The canon `references.json` blurb's
  "rolling-horizon" descriptor is a misattribution (corrected in the
  `model-specs.json` blurb only; `references.json` untouched pending a docs
  sprint).
* Leave headroom for the canon's successors: `lu2009equivalent` (gap-function DUE
  + path swapping) and `sbayti2007efficient` (MSA efficiency variants). This row
  ships vanilla MSA + AON on the enumerated path set only.
* The loading is a DECLARED substitution: P&M's `F` is DYNASMART; the white-box
  row evaluates the same constraint role with the repo's CTM/LTM + `TampereNode`.
  Faithful architecture, different physics — stated the way ADR-022 declared the
  GVM loading for the 1993 VI.
* **The certificate scores relative to the DECLARED path universe (review
  MINOR).** `tdue_gap` and `so_bound_gap` (and `Z*` itself) are defined over the
  enumerated `paths`, exactly like `vi-due`'s declared route list — an all-on-A
  plan on a network that also contains an idle byte-identical route B certifies
  gap 0 if B is not declared. Restricted choice sets are legitimate scenario
  design, so this is NOT a constructor gate; `declared_paths_omitting_shortest()`
  is a non-gating completeness diagnostic (flags an OD whose declared set omits a
  strictly-faster free-flow path), and the builtin anchors assert it is empty.
* **Construction-time config gates (review MINORs).** Demand may not extend past
  the grid horizon (no columns for the tail), and total demand at/below the
  float64 conditioning floor (`1e-6`) cannot resolve an equilibrium — both raise
  at `TDTAScenario` construction (the ADR-020/021 eager-config discipline), never
  silently censor. The content hash length-frames every array (the newell-3det
  lesson, defense-in-depth while the tdta hashes are unpublished).

### Experienced-vs-instantaneous false-accept families (F1–F6)

Named so an adversarial review attacks them; each has a gate or a pinned answer:

* **F1 — instantaneous certifier × experienced target (the dangerous
  direction):** never built. There is no instantaneous machinery anywhere in the
  row (or the repo); `DNLOutput.travel_time` and the composition here are
  experienced FIFO times. Architectural absence.
* **F2 — experienced certifier × instantaneous-DUO emission:** an instantaneous
  fixed point scores a genuinely positive experienced gap (correct); the row is
  LABELED experienced-TD-UE so no instantaneous solver is marketed as "the same
  equilibrium." The demand-match gate is the concrete guard against
  timing-based instantaneous gaming.
* **F3 — horizon truncation:** experienced cost is undefined for vehicles that
  never exit. Handled by grid extension to analytic clearing (nothing strands on
  the acyclic positive-capacity used-graph); an emission unresolved at the pad is
  CENSORED (pinned by a zero-pad regression on a short-horizon corridor).
* **F4 — interval aggregation / burst dump:** scored PER TRAVELER by level
  composition, never per-interval means; and with fixed departure times the
  demand-match gate makes a within/across-interval burst dump impossible at the
  gate (pinned).
* **F5 — hide the cheap path from the reference minimum:** the minimum scans
  EVERY declared path at every departure time; the grid is scenario-fixed, so no
  emitted-grid dilution exists (the ADR-022 round-2 CRITICAL cannot recur here).
  Pinned by the diamond all-on-one control (gap 0.75 because the idle twin is in
  the min set).
* **F6 — SO-marginal forgery:** P&M's marginals are estimator-defined (quadratic
  fit), not recomputable model-blind. The SO certificate scores recomputed TSTT
  against the harness LP bound (gap-to-bound `>= 0`, tightness not promised) and
  NEVER reads solver-reported marginals. Pinned by the wedge (SO attains the
  bound; all-fast has a positive bound gap).

## Consequences

The analytical-DTA track gains its first ITERATIVE, simulation-based row:
fixed-departure route-choice TD-UE on the repo's own kinematic-wave loading, plus
its SO twin scored against the `lp-so-dta` bound. All changes are additive; no
existing certifier, evaluator, or hashed class is touched, and the golden Braess
hash `cf00f411…` is byte-identical. The row is honest about what it cannot do
(no SO-optimality certificate for a nonconvex simulation program; the
experienced-time gap is w.r.t. frozen realized times — a re-simulated deviation
would differ, a Tier-B caveat, not the score).

## Sourcing

Peeta, S. & Mahmassani, H.S. (1995) "System optimal and user equilibrium
time-dependent traffic assignment in congested networks," *Annals of Operations
Research* 60:81–113, doi:10.1007/BF02031941. **The primary is READ IN FULL (all
33 pages) — a first for a DTA row** — via Peeta's own Georgia Tech author-page
scan (the open channel; Unpaywall/Semantic Scholar list the DOI as closed). The
TD-UE/TD-SO definitions, the SO optimality conditions (Eqs. 6/15/18), the MSA +
marginal-penalty TDSP algorithm (§4, Figs. 1/4), and the load-bearing-vs-
DYNASMART-incidental split (§3.2) are taken directly from the paper. The
DYNASMART speed–density family (used only to confirm the incidental physics is
replaceable) is cross-verified from the open Mahmassani-school Northwestern/FHWA
weather report. The paper's 50-node experiment numbers (Tables 2–3) are
transcribed but the row is **irreproducible from the paper alone** — the node OD
matrix, per-interval generation seeds, DYNASMART parameter file, and the
"averaging techniques" of the marginal estimator are not published, and the
adapter path is licensing-deferred (ADR-030) — so no anchor is a table match;
every anchor is derived from scratch on the repo's own loading and machine-
verified, and the certificate is an INVARIANT certification (SO≤UE, the LP bound,
the experienced-time route-swap residual, the ltm==ctm and lp-so-dta cross-model
pins), not a numeric replication. No page-precise quotes reproduced.

## Review

A three-lens adversarial review of this sprint (certificate-soundness,
formulation-fidelity, numerics/hashing — each executing repros) CONFIRMED four
MAJORs (two independently found by multiple lenses — the convergence
strengthened both), four MINORs, and one defense-in-depth item — all fixed and
regression-pinned (`tests/test_tdta.py`, 38 tests from 27; streak: 19/19
sprints with at least one material defect), with the SO<UE headline and every
anchor surviving.

* **MAJOR — TSTT/SO-undercut family (one root: the `V − A` TSTT + one-sided
  gates).** A sub-budget over-emission in the last step made the V-based TSTT go
  negative on trailing steps and certify `so_bound_gap < −tol` at `feasible=1`
  (on the wedge AND the forced corridor); a demand SPREAD over several steps got
  the burst LP's `Z*` (a spurious positive bound on a congestion-free loading)
  and the TSTT charged pre-departure waiting (30 vs the true 12). Fixed: the
  availability-based TSTT (`Σ D(t_{k+1}) − A(t_k)`, `>= 0`, corridor still 33 =
  Z*, no pre-departure charge); the undercut branch is now a hard CENSOR (ADR-020
  weak-duality discipline); `derive_cell_scenario` requires `breakpoints[1] <= dt`
  (spread → `so_bound_gap = NaN`); and a two-sided delivery gate. The true
  experienced total is reported as `total_experienced_time`.
* **MAJOR (all three lenses) — clearing pad.** The drain estimate read only SINK
  capacities, false-censoring honest finite-clearing plans behind an interior
  bottleneck. Fixed: drain at the slowest USED-link capacity; the 20000-step hard
  cap documented as a bounded censor-not-crash safety.
* **MAJOR — global gate scale.** A global `eps ∝ tol·V_max` let a tiny OD shift
  or retract a real vehicle beside a huge OD. Fixed: per-OD `eps_od = tol·D_od`
  on the demand-match, negative-departure, and retraction gates.
* **MAJOR — `tdue_gap_max` under-reported (model-flattering, scored).** The
  coarse level sweep missed the peak, which sits at a pullback kink of the
  composed cost. Fixed: a level-domain scan seeded with the free-flow pullback
  kinks plus a bisection zoom on the queue-clearing corners — now matches a dense
  level reference to ~1e-6 and is conservative (never under-reports); the headline
  `tdue_gap` integral already agreed to ~1e-6.
* **MINORs.** Under-declared path sets score against the declared universe
  (disclosed; non-gating `declared_paths_omitting_shortest()` helper, anchors
  assert empty). Demand past the grid horizon and degenerate (sub-floor) demand
  now raise at construction. The wedge "UE (3/3)" label is a per-interval
  convention (its certified `tdue_gap = 1/11 > 0`; the per-traveler minimizer is
  elsewhere) — disclosed; the SO<UE headline is unaffected.
* **Defense-in-depth.** `TDTAScenario.content_hash` now length-frames every array
  (the newell-3det lesson), done while the tdta hashes are still unpublished; the
  golden Braess hash is a different domain and is byte-unchanged.

Every reviewer repro under `scratchpad/pmrev/` was rerun and flips to the safe
behavior; the max-form now matches a 300k-level reference to ~1e-6 with the
correct `tc_min/V` normalizer.

**Survived (highlights):** every anchor value independently re-derived from
scratch and matched exactly (corridor 33 = Z*; diamond 14, gaps 0 / 0.75 / 1.5
/ 2/7; wedge Z* = 23, 1/23, 10/23; merge 25 = Z*); the FIFO level composition
verified against an independent event-level simulator (exact on aligned grids,
O(dt)-convergent off them, per-halving ratios measured); per-commodity merge
attribution matched by a from-scratch Tampère implementation to ≤1.3e-6; the
interior-diverge refusal unfoolable (turns derive from the combinatorial path
set, never emitted numbers); the disclosed exit-interval marginal-sampling
deviation confirmed sound with the entry-interval cold-start trap REPRODUCED
(SO-MSA pinned at the worst plan for 30 iterations under the control); SO<UE
robust to 2× grid refinement; Dossier B's remaining attack-table entries
structurally unforgeable (decisions-only artifact); LP-side relaxations
(holding, fractional splits, early release) all bound-safe; hash
byte-migration structurally closed (and length-framed anyway); paper fidelity
verified line-by-line against the read-in-full primary (MSA 1/(l+1),
marginal = T + x·∂T/∂x with the 3-point quadratic fit, penalties-vs-movement
split per §4.2.4); C0–C8 loading oracle held on adversarial instances; 90
neighbor DTA/DNL/DUE tests and the golden Braess hash untouched.
