# ADR-010: dnl-core — generic supply/demand dynamic-network-loading foundation

**Status:** accepted (implemented in v1)
**Deciders:** dnl-core design panel (3 competing designs, judged) → synthesizer
**Date:** 2026-07-07
**File:** `docs/design/adr-010-dnl-core.md`

## Context

`TASKS.md` Phase 2 opens with `dnl-core`: the time-expanded / cell infrastructure
that five later sprints must share without rework — **ctm** (Daganzo 1994/95),
**ltm** (Yperman 2007), **newell-kw** (Newell 1993), **godunov** (Lebacque 1996),
and **node-model** (Tampère et al. 2011). The foundation has to anticipate all
five link/node paradigms, emit a P1-certifiable artifact, and stay strictly
additive — the static machinery is frozen and the golden Braess content hash
`cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d` must remain
byte-identical.

## Decision

1. **Generic sending/receiving (S/R) architecture.** Every link model exposes
   `sending(k)` / `receiving(k)` in vehicles-per-step (`dnl/link.py`); node models
   allocate transfer flows from `(S, R, turning fractions, capacities)` under the
   Tampère first-order node axioms (`dnl/node.py`); time-indexed per-link cumulative
   curves `n_in` / `n_out` at grid edges are the canonical emitted output
   (`dnl/output.py::DNLOutput`). CTM, LTM, Newell and Godunov all become `LinkModel`
   subclasses with no interface change — Lebacque demand/supply on the
   `FundamentalDiagram` ABC (`dnl/fd.py`), whose `envelope_params()` triangular
   majorant keeps every certificate a *necessary condition* for any concave FD.

2. **New, domain-separated scenario construct.** `DynamicScenario` (`dnl/scenario.py`)
   reuses the static `Network` topology **read-only** and adds `LinkDynamics` +
   piecewise-constant `DynamicDemand` + `TimeGrid` + optional `TurningFractions`. It
   hashes under a `"tabench-dnl-scenario-v1;"` domain prefix over exactly the scored
   content (topology, KW physics, demand, grid, turns) — the static BPR fields are
   ignored and unhashed (P2: hash exactly what is scored). `core/scenario.py` is never
   edited, so no static hash can move (re-asserted in the dnl tests).

3. **P1 certification in `metrics/dnl_gaps.py`.** From `(DynamicScenario, DNLOutput)`
   alone the harness recomputes, censoring on any gating failure
   (`dnl_feasible = 0`, scored quantities NaN, residual columns populated —
   mirroring `metrics/gaps.py`):
   - **C0** validity (finite, zero-start, monotone, matching grid, scenario-hash),
   - **C1** conservation (per interior node + per step, **per-origin coupling**,
     global vehicle identity),
   - **C2** capacity (both boundary fluxes ≤ `q_max · dt`),
   - **C3** storage bounds (nonnegative; ≤ `κ·L` on finite-jam links),
   - **C4** free-flow causality (grid-edge-relaxed Newell upper envelope),
   - **C6** FIFO / travel-time consistency (level-matched curve inversion),
   - **C7** demand coupling (release ≤ cumulative demand).
   **C5** (backward-wave envelope) is a **non-gating Tier-B** residual with
   two-level flags: standard CTM at CFL = 1 under spillback provably violates the
   sharp bound via numerical hole diffusion, so gating it would falsely censor a
   correct convergent scheme; the raw residual is always reported so no threshold
   hides a real violation. Scored TSTT / delay / unserved / completed / in-network
   are recomputed from counts, never trusted.

4. **Sanctioned operating point = CFL = 1 on cell-aligned grids.** The grid-edge
   relaxations (C4, and C6's travel-time face) are *exact* when `L/(vf·dt)` is an
   integer and conservative by at most one step otherwise. Gating conformance is
   promised only at CFL = 1; the ctm/godunov sprints must ship congested-instance
   conformance regressions before leaderboard use. Unaligned (CFL < 1) schemes are
   answerable to the always-reported raw residuals, not the gate.

5. **Test-only reference link.** `dnl/_reference.py::PointQueueLink` (Vickrey-lineage
   point queue; **never registered as a benchmark model**) exercises every interface
   without pre-empting the ctm sprint. Two analytic anchors are hand-derived AND
   machine-verified: `single_link_dynamic_scenario` (free-flow translation,
   TSTT = 5.0, zero delay) and `bottleneck_dynamic_scenario` (2-link corridor queue
   build/dissipate).

## Alternatives considered

- **Cumulative-curve-native banks with path demand** (design 1): rejected on
  minimality — path demand departs from the repo's OD world and pre-loads FIFO
  machinery the five sprints don't need. Its **Tier-B semantics for the
  backward-wave bound was adopted** (Decision 3).
- **Time-expanded vectorized engine with CSR commodity bookkeeping** (design 2):
  rejected on foundation weight, but its **envelope majorant, trapezoidal `q_cap`
  FD, node-capacity signature and PQ-spillback fixture were adopted**.
- **Extending `core.Scenario` with optional dnl fields** (the static house pattern):
  rejected — it edits a frozen file and drags BPR bytes into DNL identity.

## Known limitations & deferred hardening

Recorded honestly from the post-implementation adversarial review (2026-07-07):

- **C6 off CFL = 1 (deferred, latent).** C6 is exact and gating on aligned grids.
  Off them the inverse interpolation carries an O(`dt`) time-quantization error the
  current gate does not correct, so it can (a) false-**censor** correct emissions by
  up to one step, and (b) because it samples entry-curve levels only, miss a sub-step
  violation confined to an exit-curve level between two entry edges that C4's slack
  admits (false-**accept**). Both are out of the CFL = 1 promise; the raw
  `fifo_residual` is always reported. The sound joint fix — sample the **union** of
  both curves' edge levels **and** relax by the principled per-level interpolation
  bound (the two directions are coupled; a flat `dt` relaxation reopens the
  false-accept) — is a certificate-numeric change reserved for the adversarial DNL
  review. No shipped scenario is unaligned, so the gap is latent.
- **Turn-fraction fidelity (deferred, latent).** `scenario.turns` is hashed content
  but not yet read by any certificate, so a diverge violating a mandated split is not
  yet censored. The split is exactly recoverable at a 1-in diverge and
  underdetermined at a multi-in node; a gating check must also settle the
  congested-diverge turn-conservation convention. Reserved additive gating extension.
  No shipped scenario has a diverge.
- **Certificate test coverage.** The review found four surviving mutations
  (tolerance magnitude, C2 inflow-side, C1 origin-coupling, C6 plateau convention).
  Regression tests pinning each were added in the hardening commit and this scope was
  documented in the module docstring.
- **Open questions carried for the implementing sprints:** origins-first supply
  convention (R1), per-link Python objects at corridor scale (R3),
  `OriginNode` multi-out split placeholder (R4), Budget/Trace integration deferred to
  the DTA-runner sprint (R6), multi-commodity per-destination emission as a strict
  additive superset (R7), time-varying turning fractions as a v2 hash bump (R8).

## Consequences

The five downstream sprints implement `LinkModel` / `NodeModel` subclasses only. All
DNL work is additive; static golden hashes are provably untouched (re-asserted in
`tests/`). Certificates are recomputable, censor-first, and honest about what
aggregate single-commodity counts can and cannot falsify (within-link overtaking is
not observable; spillback fidelity is reported, not gated; CFL < 1 and turn fidelity
are scoped, not silently certified).

## Sourcing

Paywalled primaries are **attributed unread**; equations are taken from open
restatements and cross-verified; no DOIs or page-precise quotes.

- Daganzo (1994) *TR-B* 28(4) + (1995) Part II *TR-B* 29(2) — CTM cell demand/supply
  mins and the `(vf, w, kj, q_max)` trapezoidal form, cross-verified from open
  restatements (Yperman thesis; mirrored lecture notes). Unread.
- Yperman (2007) *The Link Transmission Model for Dynamic Network Loading*, PhD
  thesis, KU Leuven — **open**; primary source for LTM sending/receiving in
  cumulative form and per-step vehicle-count conventions.
- Newell (1993) "A simplified theory of kinematic waves" Parts I–III, *TR-B* 27(4) —
  the two cumulative-curve envelopes (C4/C5) restated openly. Unread.
- Lebacque (1996) "The Godunov scheme and what it means for first order traffic flow
  models" (ISTTT 13) — demand/supply Δ/Σ formalism; open restatements circulate.
- Tampère, Corthout, Cattrysse, Immers (2011) "A generic class of first order node
  models" *TR-B* 45(1) — **paywalled, unread**; requirements N1–N6 restated from
  Yperman (open) and Lebacque & Khoshyaran (2005, invariance). The node-model sprint
  re-verifies its requirement list against whatever open restatement it implements.
- Vickrey (1969) — point-queue lineage for the test-only reference link; attribution
  only.
