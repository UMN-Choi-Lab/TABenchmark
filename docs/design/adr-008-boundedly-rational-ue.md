# ADR-008 — Boundedly-rational user equilibrium: a band-relaxed UE with a necessary link-flow certificate

**Status:** accepted (shipped in v1)
**File:** `docs/design/adr-008-boundedly-rational-ue.md`

## Context

Every equilibrium so far is *perfectly rational*: at Wardrop UE every used route is exactly
the minimum cost. **Boundedly-rational UE** (Mahmassani & Chang 1987) imports Herbert Simon's
bounded rationality: a traveler does not switch routes to save less than an **indifference
band** `ε` (in native cost units). A flow is an **ε-BRUE** iff for every OD pair and every
route,

```
h_π > 0   ⟹   c_π ≤ κ_rs + ε
```

— used routes lie within an absolute band `ε` above the OD minimum `κ_rs`. This relaxes
Wardrop's *equality* to a one-sided band, with two consequences that this ADR must handle:

1. **The equilibrium is a SET, not a point.** For `ε > 0` the conditions are inequalities, so
   the acceptable flows form a region; the link flows are **not unique** and there is **no
   Beckmann convex program** (Boyles, Lownes & Unnikrishnan, *Transportation Network Analysis*
   ch. 5; the rigorous static set formulation is Di, Liu, Pang & Ban 2013, *TR-B* 57:300–313).
   `ε → 0` recovers Wardrop UE; `ε → ∞` admits *any* feasible flow.
2. **The band is a per-route condition, but the harness sees only link flows.**

## Sourcing

Concept: **Mahmassani & Chang (1987)**, *Transportation Science* 21(2):89–99 — paywalled, a
behavioural (departure-time + route-search) paper, **attributed unread**. The static ε-BRUE
condition, non-uniqueness, and the ε-monotone acceptable set are from **Di–Liu–Pang–Ban
(2013)** and cross-verified in **Boyles TNA ch. 5** (used-path band condition; "no convex
program; link-flow uniqueness not guaranteed"). The absolute band + VI form were cross-checked
against open restatements. No numeric result from the primary is claimed.

## Decision 1 — Represent the band as content-hashed scenario data

`Scenario.br_epsilon: float | None` (`core/scenario.py`) carries the band width. Validated `> 0`
and finite; **mutually exclusive** with `sue_theta` / `elastic_demand` / `combined_demand`
(BR-UE is a deterministic fixed-demand route equilibrium; those make demand non-fixed);
**content-hashed only when set**, appended last, so every existing scenario keeps its
byte-identical hash (golden Braess `cf00f411…` asserted preserved). Two scenarios differing
only in `ε` are different benchmark instances, and the model cannot choose its own `ε`. The
band unit is network-specific (scenario cards state it, P9).

## Decision 2 — Find an acceptable flow at the BAND EDGE (not an early-stopped UE)

`br-ue` (`BoundedlyRationalUEModel`, paradigm `static_br_ue`) reuses the `gp` gradient-projection
skeleton (per-OD path sets, column generation, exact route→link resync) with a **band-thresholded
Newton shift**: from a **pinned free-flow all-or-nothing** start, for each OD pair shift flow off
every route whose cost exceeds the OD minimum by more than the band, onto the cheapest ("basic")
route, by a projected Newton step sized to reduce that route's excess to *exactly* `ε` — not to
zero. Routes already inside the band are never touched (the incentive is boundedly-rational). It
stops at a **rest point** where no used route on any OD exceeds its OD minimum by more than `ε` —
ε-BRUE by construction.

**This is a genuine BR-UE, not a renamed early-stopped UE.** A UE solver stopped at `gap ≤ ε`
truncates the trajectory inside an ε-ball of the *unique* UE point (used-route excess `≈ 0`). The
band-thresholded shift instead stops at the **band edge** (used-route excess `≈ ε`), generally
far from UE, and stable under bounded-rational perturbation. On the two-route anchor the emitted
split is `f_A* + ε/2` (excess exactly `ε`), *not* `f_A*` — the distinctness gate, regression-tested.
Because the shift is a Newton step (not a proportional route swap, which drains a low-flow
out-of-band route only at a rate `∝` its vanishing flow — verified to need ~30× more iterations
and to leave the band violated on congested nets within budget), convergence is fast (one step
per OD on a linear network) and the band is satisfied on congested multi-OD networks (0 band
violations over a 107-net power-4 fuzz). Path-dependence/hysteresis is real (different starts →
different edges); the pinned free-flow-AON start makes it deterministic.

## Decision 3 — A necessary link-flow certificate, honest about its one-sidedness

The identity `TSTT − SPTT = Σ_rs Σ_π h_π (c_π − κ_rs) = D · AEC` makes the average excess cost
the **demand-weighted mean per-traveler excess**. So from link flows the harness scores

```
br_acceptable = 1.0  iff  AEC ≤ ε        (metrics/gaps.py, gated on scenario.br_epsilon)
```

**Necessary:** a true ε-BRUE has every used route within `ε` of `κ`, so mean excess `≤ ε` and
`AEC ≤ ε` — every BR-UE flow passes.

**Not sufficient, and we do not pretend otherwise.** Because `AEC` *averages* excess weighted by
flow, a flow can concentrate a little traffic on a route far outside the band and still average
under it: 1% of an OD's travelers at excess `50ε` and 99% at excess `0` give a per-route
violation of `50ε` yet `AEC = 0.5ε ≤ ε`. This is the **same aggregate-vs-disaggregate limitation
the node-balance audit documents** (and elastic/combined inherit): link flows never carry the
per-route information a fully sufficient check needs (route flows are not emitted). It is
**pinned transparently** by `test_certificate_is_necessary_not_sufficient` (the false accept)
and its complement (a grossly out-of-band flow *is* rejected). The **two-route anchor is the
exception** where per-route band membership *is* link-visible (`|c_A − c_B| ≤ ε` directly), so
the anchor validates tightly; a sufficient longest-positive-flow-path certificate for general
acyclic-positive-flow instances is possible and left as future work.

## Consequences

- **New:** `Scenario.br_epsilon`; paradigm `static_br_ue`; the `br-ue` model; `br_acceptable`
  scored flag; `br_two_route_scenario` anchor (`f_A* = (D+1)/2 = 5.5`, band edge `f_A* + ε/2`,
  half-width `ε/(b_A+b_B)`; at `D=10, ε=1` flows `(6,6,4,4)`, `AEC = 0.6 ≤ 1`, band edge
  regression-tested against the UE's `5.5`); `tabench run --scenario br-tworoute`.
- **Unchanged:** every prior scenario hash (golden Braess preserved); all other certificate
  paths; all prior models and tests.
- **Honest limitation:** the scored `br_acceptable` is necessary, not sufficient (documented,
  pinned) — the one-sidedness is inherent to a link-flow-only certificate for a per-route
  condition.
- **Deferred:** the sufficient longest-positive-flow-path certificate (when the positive-flow
  subnetwork is acyclic); per-OD / relative bands; the full path-dependent acceptable-set
  characterisation (Di–Liu–Ban).
