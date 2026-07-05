# ADR-004 — Route-flow proportionality: a diagnostic now, a scored certificate proposed

**Status:** proposed (diagnostic shipped in v1; scored certificate awaits PI decision)
**File:** `docs/design/adr-004-proportionality-certificate.md`

## Context

TAPAS (`tapas`, Bar-Gera 2010) is the first solver whose *point* is a property of route
flows, not link flows. Every UE solver so far is certified on `relative_gap`, which is a
function of link flows alone — and link flows at UE are **unique**. TAPAS converges to the
*proportional* (entropy-consistent) route-flow solution: for every pair of alternative
segments shared by several origins, each origin splits its flow between the two segments
in the **same ratio** (Boyles/Lownes/Unnikrishnan, *Transportation Network Analysis* v1.0,
Theorem 5.4 p. 127; condition eq. 6.94 p. 226; the "condition of proportionality" of
Bar-Gera & Boyce 1999, Bar-Gera 2006, Bar-Gera, Boyce & Nie 2012).

The problem: **proportionality is invisible to link flows.** For a fixed UE link-flow
vector `v*`, the set of route/segment-flow decompositions that produce it is in general a
polyhedron, not a point (Dafermos 1980). Two solvers — or the same solver under different
tie-breaks — can emit byte-identical `v*` while their per-origin segment splits differ
arbitrarily. So the harness **cannot** recompute a proportionality score from `(link_flows,
scenario)` the way P1 requires of every scored metric. The naive fix that works for the UE
gap and for ADR-001/003's SUE residuals — reload `v` at its own induced cost via
all-or-nothing — is not merely biased here, it is *wrong*: an AON reload picks one
arbitrary shortest-path tree per origin, i.e. exactly the tie-broken decomposition that can
show maximal proportionality violation even when `v` is exactly at UE.

This ADR records (a) what v1 ships — a **reported diagnostic**, provenance-only — and (b) a
proposed **scored certificate** whose GO/NO-GO is the maintainer's, because it requires a
genuine P1 trust-boundary change that should not be minted in a solver PR.

## What v1 ships (no decision required) — the proportionality diagnostic

`TapasModel` self-reports, on every checkpoint, from its own PAS bookkeeping:

```
pi_p                = (Σ_{r∈O_p} g^r(σ1)) / (Σ_{r∈O_p} [g^r(σ1)+g^r(σ2)])     # PAS p aggregate share
proportionality_residual = ( Σ_p Σ_{r∈O_p} | g^r(σ1) − w^r · pi_p | ) / D      # eq. 6.94, L1 / total demand
pas_proportionality_max  = max_{p,r} | g^r(σ1)/w^r − pi_p |                     # worst single-origin deviation
```

with `w^r = g^r(σ1)+g^r(σ2)` and `g^r(σ)` the origin-`r` through-flow on segment `σ`. These
are `self_report` scalars — **provenance only, never scored** — exactly like every other
model self-report (P1). The normalization (`L1 / D`, intensive per-traveler) matches the UE
relative gap and the SUE residuals, so the number is cross-scenario comparable.

The diagnostic is meaningful because TAPAS actually drives it down: with the eq. 6.100
proportionality adjustment on, the residual on Sioux Falls falls from ~1.4e-2 (pure UE,
`prop_rounds=0`) to ~7e-8 (`prop_rounds=5`) at the same link flows — the route flows become
proportional. Turned off, TAPAS is a UE solver with arbitrary route flows and the diagnostic
stays large. That contrast is the honest evidence a scored certificate would formalize.

## The scored certificate — proposed, PI to decide

### Decision 1 (proposed) — harness-owned vs model-attested

The central asymmetry vs ADR-003 must be stated up front. ADR-003's auxiliary structure
(the MC perturbation matrix `E`) is **model-blind**: the harness draws it itself from a
documented RNG stream, independent of which solver produced `v`, so pinning it costs no
trust. Proportionality's auxiliary structure **cannot** be model-blind — it *is* the model's
internal route/segment decomposition, and there is no canonical, algorithm-independent
distribution to draw it from. So a scored proportionality certificate must accept
**model-attested, harness-audited** data, not harness-pinned data:

- **Harness-owned (no trust extension):** PAS identification. A PAS (diverge node, merge
  node, two cost-tied positive-flow segments at `t(v)`) is a pure graph object of
  `(v, network)` and can be found by a harness-side label search, reusing `_bush._scan`'s
  min/max DAG labels or `PathEngine`'s shortest-path trees. And the residual formula above,
  given the segment flows, is a pure harness computation.
- **Model-attested (the trust extension):** the origin-disaggregated flow decomposition
  `X[o, a]` (flow origin `o` contributes to link `a`). The harness's only new trust is
  auditing `v = Σ_o X[o, :]` to machine precision — the same "exact resync" invariant
  `algb`/`tapas` already enforce internally every iteration. An `X` that fails the audit
  gets **no** proportionality score (NaN), exactly as flows failing the demand-feasibility
  audit get `feasible=0` today.

### Decision 2 (proposed) — `FlowState` schema

`FlowState` today is `link_flows: np.ndarray` + `self_report: dict[str, float]` (scalars
only). A scored certificate is therefore **architecturally blocked**, not just a policy
choice: it needs a new array-valued field, a **sparse** origin-disaggregated structure
(dense `n_origins × n_links` is infeasible at regional scale — Chicago is ~1,790 origins ×
~39,000 links). This ADR proposes the field but does not add it; only bush/origin solvers
(`algb`, `tapas`) can emit it natively (they already carry `_BushState.x`), and a
certification-cost accounting convention (à la ADR-003's "`R_cert` sp-equivalents per
checkpoint") must be pinned before it is buildable.

### Decision 3 (proposed) — scope and naming

PAS-level (first-order) proportionality is **necessary but not sufficient** for the full
maximum-entropy UE route flow (Bar-Gera 2006; the higher-order "general proportionality
condition", Borchers et al. 2015). A certificate scoped to PAS-level proportionality
certifies Bar-Gera's originally-proposed, practically-testable condition — not full MEUE
optimality. So the eventual scored column must be named **`pas_proportionality_residual`**,
never `proportionality_residual` unqualified or `meue_residual`, and its docs must state the
limitation. (The v1 self-report keeps the shorter `proportionality_residual` name because,
as provenance, it makes no ranking claim.)

### Decision 4 (proposed) — anti-gaming

Unlike ADR-003's harness-drawn `E` (ungameable by construction), a model here supplies its
own audit-passing `X`. The conservation audit (`v = Σ_o X`) rules out inconsistent `X` but
cannot verify that an audit-passing `X` is the *causal* decomposition the algorithm computed
rather than one constructed post hoc to look proportional. This is a strictly harder
anti-gaming surface and is the core of why the GO/NO-GO is the maintainer's call, not a
mechanical follow-on.

## Recommendation

Ship the diagnostic (done). Defer the scored certificate to an explicit maintainer decision
on Decisions 1–4 — in particular whether **model-attested, harness-audited** data is an
acceptable P1 extension at all. If accepted, the free next step is to wire `algb` to emit
the same diagnostic from its already-maintained `_BushState.x`, giving cross-model (algb vs
tapas) evidence before any scored column is minted.

## Sourcing (honesty note)

Consistent with `algb.py`/`tapas.py`: the primary texts Bar-Gera (2006, *Transportation
Science* 40(3):269–286) and Bar-Gera, Boyce & Nie (2012, *TR-B* 46(3):440–462) are
paywalled and were **not** read directly. The condition of proportionality, eq. 6.94, and
the eq. 6.100 restoring shift are taken from the open Boyles/Lownes/Unnikrishnan textbook
(§5.2.2, §6.5.3) — the same source cited for Algorithm B — and cross-verified against
Aungsuyanon, Boyce & Ran (2013) and Li, Wang, Feng, Xie & Nie (2024, arXiv:2401.08013, a
paper co-authored by Y. Nie of the 2012 paper). The necessary-but-not-sufficient caveat
(Bar-Gera 2006) and the higher-order condition (Borchers et al. 2015) are known here only
via those secondary restatements; a maintainer spot-check of the paywalled PDFs is
recommended before any scored certificate is finalized.

## References

- Bar-Gera, H. (2010). Traffic assignment by paired alternative segments.
  *Transportation Research Part B* 44(8–9), 1022–1046.
- Bar-Gera, H. (2006). Primal method for determining the most likely route flows in large
  road networks. *Transportation Science* 40(3), 269–286.
- Bar-Gera, H., Boyce, D. & Nie, Y.M. (2012). User-equilibrium route flows and the condition
  of proportionality. *Transportation Research Part B* 46(3), 440–462.
- Boyles, S.D., Lownes, N.E. & Unnikrishnan, A. (2025). *Transportation Network Analysis*,
  Vol. I, v1.0. §5.2.2, §6.5.3. (open textbook)
- Dafermos, S. (1980). Traffic equilibrium and variational inequalities.
  *Transportation Science* 14(1), 42–54.
