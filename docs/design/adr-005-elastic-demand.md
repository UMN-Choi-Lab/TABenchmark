# ADR-005 — Elastic (variable) demand: a new problem class with a P1-pure certificate

**Status:** accepted (shipped in v1)
**File:** `docs/design/adr-005-elastic-demand.md`

## Context

Every task so far fixes the OD demand matrix. **Elastic-demand user equilibrium**
(Florian & Nguyen 1974) makes demand endogenous: the trips between OD pair `rs` are a
strictly decreasing function of that pair's equilibrium travel cost, `d_rs = D_rs(u_rs)`.
The equilibrium couples two conditions (Boyles/Lownes/Unnikrishnan, *Transportation
Network Analysis* v1.0 §9.1, eqs (9.8)–(9.14); Sheffi, *Urban Transportation Networks*
1985 ch. 6):

1. **Route equilibrium** given demand `d` — the ordinary Wardrop condition.
2. **Demand consistency** — `d_rs = D_rs(u_rs)` with `u_rs` the equilibrium min OD cost.

The convex program is the Beckmann objective plus an inverse-demand term (Sheffi [6.1a],
Boyles (9.16)):

```
min_{x,d}  Σ_a ∫_0^{x_a} t_a(w) dw  −  Σ_rs ∫_0^{d_rs} D_rs^{-1}(w) dw
```

This is a genuinely new problem class, and it collides with two harness assumptions that
this ADR resolves.

## Sourcing (dual-verified, one attribution corrected)

Formulas were recovered from two primary-accessible textbooks read in full — **Sheffi
(1985) ch. 6** and **Boyles et al. TNA §9.1** — which agree formula-for-formula. Two
things are deliberately *not* attributed to the paywalled primary:

- The **excess-demand / dummy-arc transformation** used by the solver is **Gartner (1980)**
  (Boyles §9.1.2, verbatim: "developed by Gartner circa 1980"), *not* Florian & Nguyen.
- **Florian & Nguyen (1974)** is cited as the seminal *computational* paper for the
  problem; its own method is Generalized Benders Decomposition (abstract verified; body
  paywalled/unread), which is **not** what `fw-elastic` implements.

Canon keys: `florian1974method`, `gartner1980optimal`, `sheffi1985urban`,
`beckmann1956studies` — all already in `docs/references.json`.

## Decision 1 — Represent the demand law as content-hashed scenario data

`ElasticDemand(form, param)` (`core/scenario.py`) carries the decay law; the reference
demand `d0_rs = D_rs(0)` is the ordinary `Demand.matrix` (already hashed). Two forms:

| form | `D(u)` | `D^{-1}(d)` | excess-arc cost `W(e)=D^{-1}(d0−e)` |
|------|--------|-------------|-------------------------------------|
| `linear`      | `d0·max(0, 1−u/param)` | `param·(1−d/d0)` | `param·e/d0` — bounded, singularity-free |
| `exponential` | `d0·exp(−param·u)`     | `ln(d0/d)/param` | `−ln(1−e/d0)/param` — cited default, log-singular as `d→0` (floored) |

`Scenario.elastic_demand` is optional and **hashed only when set**, appended after the SUE
fields, so every fixed-demand scenario keeps its byte-identical hash — the golden Braess
hash `cf00f411…` is asserted preserved (`test_golden_braess_hash_preserved`). Elastic and
SUE are mutually exclusive (deterministic-endogenous-demand vs stochastic-fixed-demand).

## Decision 2 — Solve as fixed-demand UE on the Gartner excess-demand network

`fw-elastic` (`ElasticDemandFWModel`, paradigm `static_ue_elastic`) adds one direct `r→s`
"excess-demand" arc per OD pair with the increasing cost `W(e)=D_rs^{-1}(d0−e)`, fixes the
augmented demand at `d0`, and runs Frank & Wolfe with an exact augmented-Beckmann line
search. At equilibrium every used `r→s` path — real or dummy — has the common cost `u_rs`,
so the dummy flow `e_rs = d0 − D_rs(u_rs)` is the unmet demand and the realized demand is
exactly `D_rs(u_rs)`. The solver starts feasible at `e=0` (all demand on the real network),
which sidesteps the exponential form's `D^{-1}(0)` singularity. **Only real link flows are
emitted** — the dummy arcs are an internal device the certificate never sees. FW's slow
tail is inherited (and documented); conjugate/bush variants on the augmented network are a
future extension.

## Decision 3 — A P1-pure certificate: recompute the demand, then score

The harness knows `D` (content-hashed), so from the emitted **real** link flows `v` it
recomputes everything (`metrics/gaps.py`, gated on `scenario.elastic_demand`):

```
t   = link_cost(v)
u   = od_cost_matrix(t)          # per-OD shortest-path cost (new PathEngine skim)
d*  = D(u)                       # demand-consistent demand
relative_gap          = (v·t − Σ u·d*) / (v·t)         # route equilibrium given d*
node_balance_residual = ‖ balance(v) − div(d*) ‖_∞     # demand-consistency + conservation
realized_demand       = Σ d*                            # scored elastic quantity
```

Both `relative_gap` and `node_balance_residual` are `0` **iff** `v` is the elastic UE, and
both are pure functions of `(v, scenario)` — no self-report is trusted (P1). This is the
"direct-solver" gap of Boyles §9.1.5 (relative gap + total-misplaced-flow), computed by the
harness. It is consistent whether or not the model used dummy arcs: at equilibrium
`D(u(v)) = d0 − e = realized`, so the reconstruction recovers the model's own realized
demand exactly (proven by hand on the analytic anchor).

### The feasibility gate is demand-consistency (a stricter, principled choice)

For fixed demand, `node_balance` is an exact invariant (all-or-nothing always routes the
matrix). For elastic demand there is no given matrix, and through-node conservation alone is
**not** a sufficient gate: a flow circulating a through-node cycle — or, on an all-zone
network like Sioux Falls where there are *no* through nodes, **any** flow — carries no OD
traffic yet conserves everywhere, and would certify a phantom elastic UE with a tuned
`relative_gap ≈ 0` (this exact exploit was found in adversarial review). So the gate must
tie `v` to the demand it routes, and the only quantity that does is `node_balance(v, d*)`.
It is therefore the **feasibility gate**, exactly as for fixed demand.

The consequence — unique to elastic demand — is that an off-equilibrium flow is *not*
feasible: the real flows route `D(u(v))` only at equilibrium, so `node_balance(v, d*)` is a
convergence quantity and a checkpoint certifies only once the solver has (nearly) converged.
This is correct: unlike fixed demand there is no given demand for an intermediate flow to be
feasible-but-suboptimal *against* — off equilibrium there is no well-defined routed demand
at all. A fixed-demand solver run on an elastic task is thus censored (it routes the
reference demand, not `D(u(v))`); zero and phantom flows are censored (their zone divergences
disagree with `d* = D(u) > 0`). The known aggregate-vs-per-OD limitation of the fixed-demand
audit (necessary, not sufficient for multi-commodity feasibility) is inherited unchanged.

### Convergence and scope

`fw-elastic` self-reports the **real-route** relative gap (identical to the scored
`relative_gap`) and early-stops on `max(real, augmented)` gap, so `target_relative_gap` is
comparable to a fixed-demand solver's and never trips on the transiently-negative real gap a
naive stop would. Because the augmented network carries one excess-demand arc per OD pair
(≈ `zones²`), FW's tail is slower than fixed-demand FW: it certifies tightly on small
networks and converges (more slowly) on large ones — conjugate/bush variants on the
augmented network, or a demand-diagonalization reusing the fast fixed-demand solvers, are
future work. Every positive-demand OD pair must be reachable in the real network (a
disconnected instance, which the formulation could absorb as fully unmet demand, is out of
scope for v1 and raises during the shortest-path step; the certificate censors it rather
than crashing). Intrazonal (diagonal) reference demand never enters the network and is
excluded from `realized_demand`.

## Consequences

- **New:** `ElasticDemand`; `Scenario.elastic_demand`; paradigm `static_ue_elastic`;
  `PathEngine.od_cost_matrix`; `fw-elastic` model; `realized_demand` (scored) +
  `self_realized_demand` (provenance) CSV columns; `elastic_two_route_scenario` analytic
  anchor (`u=5, f_A=3, f_B=2`, realized 5, flows `(3,3,2,2)` — recomputed with brentq in
  tests, not trusted).
- **Unchanged:** every fixed-demand scenario hash (golden Braess preserved); the
  fixed-demand / SO / SUE certificate paths; all prior models and tests (187 pass).
- **Deferred:** per-OD demand parameters; conjugate/bush solvers on the augmented network;
  a combined distribution–assignment task (Evans 1976) that reuses this machinery.
