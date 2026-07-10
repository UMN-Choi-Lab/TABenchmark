# ADR-021: lp-so-dta — Ziliaskopoulos (2000) LP single-destination SO-DTA on CTM cells

**Status:** accepted (implemented)
**Date:** 2026-07-10
**Deciders:** analytical-DTA track — cell-level SO-DTA with finite storage
**File:** `docs/design/adr-021-lp-so-dta.md`

## Context

Ziliaskopoulos (2000) casts single-destination system-optimal DTA as a linear
program over cell-transmission dynamics: the CTM's Godunov flux EQUALITY
`y = min(x_i, Q_i, Q_j, delta_j (N_j - x_j))` is relaxed to its four linear
`<=` constraints while conservation stays an equality. Against the shipped
Merchant-Nemhauser model (ADR-020) the physics is new — **finite storage and
spillback** (exit functions with uncapacitated inflow cannot represent either),
inherited from the CTM the repo already ships as `dnl/ctm.py`. The historical
line closes: M-N restricted to the CTM's sending branch at coordinated
discretization *is* an optimization CTM (Carey & Watling 2012), and this model
is that program with the full supply side.

## Decision

1. **Same parallel module** — `src/tabench/dta/cells.py` beside the M-N files,
   touching no existing code (golden Braess hash re-asserted).
   `CellSODTAScenario` is frozen, read-only, and content-hashed under a new
   domain prefix `"tabench-dta-cell-scenario-v1;"`: cells with per-interval
   capacity `Q_i` (inf allowed), storage `N_i` (inf allowed), `delta_i = w/v in
   (0, 1]` (the LP analogue of the CTM's `w <= vf` gate — with the spillback
   row it gives `x <= N` by induction), one absorbing sink, sources with no
   predecessors, a `(T, cells)` demand table plus an initial occupancy.

2. **The canonical LP** (`cell_canonical_lp`): variables `x_i(t)`, `y_c(t)`;
   equalities = initial condition, conservation (demand as exogenous RHS; the
   sink absorbs), and terminal clearance `x_i(T) = 0` for non-sink cells — the
   ADR-020 benchmark convention layered on the primary (which has no terminal
   condition; without one "total cost" rewards stranding). Inequalities are the
   four CTM families in the uniform AGGREGATE form (per-cell summed sending and
   receiving), which coincides with the per-cell-type constraint lists on
   Ziliaskopoulos's network class and is the safe superset in general; rows
   with infinite bounds are omitted. Objective: `sum_{t=0}^{T-1} sum_{i != sink}
   x_i(t)` — total system travel time; source queues are costed, the sink is
   not. Merge priorities and diverge turning fractions are deliberately ABSENT:
   the SO program chooses them (a *controlled* CTM). `solve_cell_so_dta` (HiGHS)
   emits a `CellTrajectory` with the LP dual certificate in the documented
   canonical row order.

3. **P1 certificate** (`metrics/dta_gaps.py`, `CellSODTAEvaluator`) inherits
   the ADR-020 adversarial-review hardening wholesale: two-scale gates (local
   per-cell `eps` + aggregate mass budget on each violation family), the
   `x <= N` envelope, delivery into the sink, clamped-occupancy cost, an
   eagerly harness-resolved `Z*` (unclearable horizon = construction-time
   error), the weak-duality undercut censor, and pure-arithmetic dual
   verification. New here: `holding_max`, a Tier-B diagnostic reporting the
   largest headroom a queued connector leaves unused — LP "traffic holding" is
   legitimate optimal-face behaviour for a single destination (a non-holding
   optimum always exists: the earliest-arrival-flow property, Zheng & Chiu
   2011, which is also why the LP relaxation is tight), never an error.

## Analytic anchors (machine-verified — `test_dta_zil.py`)

- **`zil_diverge_spillback_scenario`:** 6 vehicles at source R, a diverge at A
  between a short route through the one-vehicle bottleneck cell B (`Q=1, N=1`)
  and a longer C→D route (`Q=2`). SO = **26** veh-intervals by the spillback
  **pair lemma** `y_BS(s) + y_BS(s+1) <= 1` (storage + backward wave, not just
  capacity) plus earliest-arrival bounds; cell B is jam-full at `t=2` in EVERY
  optimum (the storage row is tight, dual price −1), and relaxing `N_B` to 2
  drops the optimum to **25** — the finite-storage constraint is worth exactly
  one vehicle-interval. A strict-CTM rollout attains 26 (the LP bound is tight
  and realizable); the all-long-route plan costs 30 and certifies `gap = 4/26`;
  a plan holding one vehicle at the source below ALL four bounds still attains
  26 and certifies with `holding_max = 1` — holding on the optimal face,
  exactly as the theory says.
- **`zil_corridor_scenario`:** a control-free corridor whose LP optimum (33)
  equals the repo's own `CTMLink`/`NetworkLoader` strict-CTM loading **exactly**
  — cross-model consistency between the analytical-DTA track and the DNL stack.
- Censoring: teleports, spillback violations (inflow above `delta (N - x)` that
  capacity alone would allow), stranded flow, negative flows, wrong hashes, and
  ADR-020's shadow-shift undercut are all censored; forged dual certificates
  report large `dual_gap`/`dual_infeasibility` while the untouched primal still
  certifies.

## Alternatives considered

- **Building on `dnl/DynamicScenario` instead of a cell scenario:** rejected —
  the LP's data are per-cell `(Q, N, delta)` with sources/sinks and free
  merge/diverge controls; deriving them from link-level `LinkDynamics` + node
  models would couple the modules and misstate the model (the LP has no node
  model — that absence is canonical). The corridor cross-check test pins the
  correspondence where it exists.
- **Per-cell-type constraint lists (Ziliaskopoulos's exact typography):**
  the uniform aggregate encoding is used instead — identical on his network
  class (no cell both merges and diverges), well-defined and conservative in
  general, and one code path instead of five.
- **Adding merge priorities / FIFO diverge rows:** rejected — the SO program
  *chooses* the controls; adding them would change the model.

## Adversarial review

Three independent attack lenses, each executing code. What survived: the LP
matrices equal hand-written ones coefficient-for-coefficient (including the
`delta < 1` receiving-space rows and demand RHS); the anchor was re-proven by
hand AND by an independently written per-connector LP (26; `N_B=2` → 25;
`N_B=1.25` → 25.75, i.e. `dJ*/dN = -1` exactly; the storage row tight on the
ENTIRE optimal face); the aggregate encoding was shown *identical* to
Ziliaskopoulos's per-cell-type lists even at an off-class merge+diverge cell
(optimum 34 = 34, support functions equal to 1e-14) and strictly tighter than
a naive per-connector reading (28, CTM-impossible); 5,584 strict-CTM rollouts
across 300 random DAGs never beat the LP bound (zero `LP > CTM` violations);
dual forgery is blocked mathematically (the dual LP's maximum equals `Z*`) and
empirically (20,000-draw search); the reference solver's duals were never
false-censored (worst residual ~9e-16 over 70+ instances, jitter, and 1e±6
scalings). What broke, all CONFIRMED with repros and fixed + regression-pinned:

- **CRITICAL false-accept:** the initial-condition gate had a per-cell `eps`
  (scaled by `x0.max`) but NO aggregate budget — the one gate that missed the
  ADR-020 two-scale fix. With a 1e6-vehicle source setting `eps = 1.0`, a
  cheater deleted whole trickle-source vehicles at `t=0`, conjured
  replacements at ghost cells beside the sink (delivery nets out; conservation
  uses the CLAIMED `x[0]`), buried the savings as holding to dodge the
  undercut censor, and certified `feasible=1` while hiding **495x tol** of
  optimality gap. Fixed: the initial-condition family now carries the same
  absolute-sum budget as every other family (plus an aggregate negativity
  budget for depth).
- **MAJOR false-censor:** demand pulses into finite-storage sources put the LP
  and the certifier in disagreement — the receiving-space rows exist only for
  connector-fed cells, so the LP (correctly, per its own rules) overfills the
  source while the certifier's `x <= N` envelope censors EVERY mass-conserving
  trajectory, including the solver's own optimum (18/30 such fuzz scenarios).
  Fixed at the root: validation now requires infinite storage on
  demand-loaded cells, exactly as the docstring always claimed (`x0 <= N`
  loading stays legal — occupancy is non-increasing at a source).
- **MINOR scale-blind dual reporting:** a `+1e-12` sign violation on a
  large-`b` spillback row moved the "certified" bound by 56x tol while
  `dual_infeasibility` read 1e-12. Fixed in BOTH evaluators (ADR-020's
  inherits it): `y_ub` is clipped at 0 before the bound is computed — the
  clipped vector is sign-feasible by construction, the bound is conservative,
  and the raw violation is still reported.
- Two docstring clarifications from NOTE findings: the demand-entry costing
  convention (injection interval uncosted; `x0` costed from `t=0`) and the
  "safe superset" phrasing (superset of controlled-CTM trajectories; tighter
  than per-connector encodings).

Known scalability note (accepted): dense LP assembly — ~1.5 GB at 50 cells x
100 steps, and the evaluator re-solves at construction; fine at anchor scale,
move to `scipy.sparse` before larger families (same boundary as ADR-020).

## Consequences

The analytical-DTA track now spans both classical SO-DTA formulations — exit
functions (M-N 1978) and cell transmission (Ziliaskopoulos 2000) — under one
P1 pattern (canonical LP + eager harness `Z*` + pure-arithmetic dual
certificates). All changes are additive; the 629-test suite and the golden
Braess hash are byte-untouched. Dense LP assembly remains the known
scalability boundary (ADR-020); move to `scipy.sparse` before large families.

## Sourcing

Ziliaskopoulos (2000) "A Linear Programming Model for the Single Destination
System Optimum Dynamic Traffic Assignment Problem," *Transportation Science*
34(1):37–49, doi:10.1287/trsc.34.1.37.12281 — paywalled, attributed. The LP
(variables, objective incl. source queues, the four relaxed CTM families with
per-cell-type aggregation, source/sink conventions) is cross-verified from
open restatements read in full: arXiv 1708.03759 (Eqs (1), (4)–(9)), arXiv
2112.14389 (Eqs (6)–(9), (13)–(15) — the Beard–Ziliaskopoulos generalized
aggregate form), and Peeta & Ziliaskopoulos (2001) §2 (relaxation and holding
semantics). CTM correspondence and holding realizability: Daganzo (1994/1995)
via the repo's own `dnl/ctm.py` (ADR-015); Zheng & Chiu (2011,
doi:10.1016/j.trb.2011.03.001, attributed) for the earliest-arrival-flow
tightness/non-holding result; Como–Lovisari–Savla (arXiv 1509.06189, read) for
controlled realizability. Both anchors were derived from scratch (no canonical
instance exists) and verified by LP + zero-residual duality certificates.
