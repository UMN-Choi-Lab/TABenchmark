# ADR-003 — Probit SUE: Monte Carlo fixed-point certificate with a pinned evaluation stream

**Status:** accepted (implemented in v1)
**File:** `docs/design/adr-003-probit-sue-mc-certificate.md`

## Context

ADR-001 certifies logit SUE through Dial-STOCH's *closed-form* loading map: the harness
recomputes `L(t(v), θ)` deterministically and scores the fixed-point residual
`‖v − L(t(v))‖₁ / D`. Probit SUE (Daganzo & Sheffi 1977) has **no closed-form loading
map** — probit route-choice probabilities are high-dimensional normal orthant integrals,
and the practical loading is Monte Carlo simulation (Sheffi & Powell 1982). Every
analytic alternative fails at scale (see Rejected alternatives). So the T1 certificate
question returns in a harder form: how does a harness certify a fixed point of a map it
can only *sample*?

The solver this ADR certifies is `sue-probit-msa`, the repo's **first non-deterministic
model** — it activates the stochastic track (P5/P8) that has been designed but idle:
`Capabilities(paradigm="sue", deterministic=False, seedable=True, provides_gap=False)`,
macroreplication in `run_experiment` (already routed by `deterministic=False`; zero
runner changes), and bootstrap aggregation.

## Decision 1 — The pinned MC certificate

For a scenario with `sue_family="probit"`, the Evaluator computes from emitted flows `v`
(after the feasibility audit, unchanged):

```
t     = network.link_cost(v)
v̂     = L_cert(t) = (1/R_cert) · Σ_{i=1..R_cert} AON( max(t + E_i, 1e-9) )
E     = sqrt(β · t0)[None, :] * Z          # (R_cert, n_links), Z iid N(0,1)
```

`E` is drawn **once per Evaluator** from `RngBundle(root_seed, macrorep=0)
.generator(SOURCE_EVALUATION)` — the reserved stream's documented purpose. Pinning `E`
is legal *precisely because* the pinned variance is flow-independent (free-flow form,
Decision 3), so `E` never depends on `t(v)`.

Scored columns:

- `sue_fixed_point_residual` = `‖v − v̂‖₁ / D` — the ranking column (same name and
  semantics as ADR-001; the map is MC where Dial's was closed-form).
- `sue_residual_se` — jackknife standard error over the `R_cert` samples.
- `sue_residual_floor` = `Σ_a sqrt(2·s_a²/(π·R_cert)) / D` with `s_a²` the
  across-sample link-flow variance — the CLT estimate of the expected residual when `v`
  *is* the fixed point (the certificate's own noise floor; positive bias O(1/√R_cert)).

**Significance rule (printed with every probit-SUE table):** residual differences below
`max(sue_residual_floor, 2·sue_residual_se)` are **ties**. On tworoute at
`R_cert=6400` the jackknife SE dominates the floor (threshold ≈ 4.7e-2), so the rule
must take the max, not quote the floor alone. This is also the anti-gaming clause:
`root_seed` is public in the manifest, so a solver could in principle target the fixed
point of the *finite-R_cert pinned map* and certify below floor — sub-threshold
residuals therefore carry no ranking information by protocol, and targeting the pinned
map buys nothing the rule doesn't already discount.

Why pinning (rather than fresh draws per certification):

1. **P1 restored exactly** — the certificate is a pure deterministic function of
   `(link_flows, scenario, root_seed)`, byte-reproducible across machines (verified:
   repeated certification returns identical tuples).
2. **Common random numbers** across every model, macrorep, and checkpoint certified in
   one experiment — the SimOpt "CRN across solutions" design — so *comparisons* carry
   far less noise than the marginal floor suggests.
3. The honest residue of MC error is **quantified and printed** (se + floor), never
   hidden.

`R_cert` is certificate *protocol*, not instance data (ADR-001/002 precedent): it lives
in the scenario card's `sue:` block, the manifest, and a CSV column; it is not hashed
into the scenario. Measured pins: tworoute `R_cert=6400`, siouxfalls `R_cert=2000`
(~1.5e-2 floor, 3–4 s/checkpoint). Certification costs `R_cert` sp-equivalents per
checkpoint — estimators/solvers are documented to emit O(10–20) checkpoints (ADR-002
precedent, never silent thinning).

**Honesty check change:** ADR-001's byte-equality regression does not transfer — the
solver samples its own stream at its own `R`, and (Decision 3) its `R=1` self-reported
direction norm does not decay to zero even at the fixed point (measured ≈1.5 on tworoute
near `v*`, vs a floor ≈0.04). So there is *no* probit analogue of the logit honesty
regression: `self_sue_residual` is recorded as **provenance only** (which stream, which
`R`, what the solver saw), never diffed against the certified residual, because the two
quantities measure different things (a running descent-direction norm vs a fixed-point
residual under the pinned map). The certificate's own `sue_residual_se` and
`sue_residual_floor` — printed on every row — are the integrity mechanism: they bound how
much of the certified residual is MC noise, and the significance rule discounts anything
below `max(floor, 2·se)`. The self-report cannot inflate a ranking (only the certified
column ranks) and cannot be gamed into looking converged, so no cross-check is needed.

## Decision 2 — `sue_family` scenario dial and hashing

`Scenario` gains `sue_family: str = "logit"` (validated in `{"logit", "probit"}`;
unrelated to `Scenario.family`, which is *data lineage* for the `trained_on` gate).
`sue_theta` is reused as the family's dispersion dial — logit: θ in 1/(native cost
unit); probit: β = perception variance per unit **free-flow** time, native cost units
(cards must state units, P9). Validation: `sue_family == "probit"` requires
`sue_theta is not None`.

`content_hash()` appends `f"sue_family={self.sue_family};"` **only when
`sue_family != "logit"`**, after the existing conditional `sue_theta` block. Verified
consequences: (a) every existing scenario hashes byte-identically — golden Braess hash
`cf00f411…` recomputed unchanged; (b) probit and logit tasks on the same
network/demand/θ can never collide; (c) solvers dispatch truthfully — `sue-msa` raises
on probit scenarios, `sue-probit-msa` raises on logit ones.

## Decision 3 — Solver pins (`sue-probit-msa`), sources per formula

- **Variance spec (pinned): `var(T_a) = β·t0_a`** (free-flow). Sheffi & Powell (1982)'s
  equivalent unconstrained program requires a flow-independent perception-error
  distribution; Sheffi (1985) eq. [12.57] uses exactly this form for the probit SUE
  example. Catalogued disagreements: Sheffi ch. 11 eq. [11.12] (pure loading, no
  equilibrium) uses `β·t_a` "for simplicity"; CiudadSim/ScicosLab implements *standard
  deviation* ∝ current time; oyama's ngev-mte uses `sqrt(θ·c)` with current cost. The
  free-flow form is the only one consistent with the Sheffi–Powell equilibrium theory —
  and it is what makes the certificate's pinned `E` legal.
- **Draws per iteration (default factor `R=1`).** Sheffi (1985, pp. 333–334) reports
  fig. 12.10 (one draw per iteration) "displays the best convergence pattern" and
  recommends the minimum computational effort per iteration (paraphrase; page-cited).
  Replicated here: equal 300-sp-call budget on Sioux Falls gives certified residual
  8.0e-2 at `R=1, k=300` vs 1.38e-1 at `R=10, k=30`. `R` stays a declared factor so the
  trade-off is exhibitable.
- **Step size (pinned): `α_k = 1/k`** plain MSA (Sheffi ch. 12; Powell & Sheffi 1982
  Blum conditions — already SHIPPED for the logit solver). No Polyak averaging: with
  `1/k` steps the iterate *is* the running average of sampled loads.
- **Negative sampled times:** truncate at `max(·, 1e-9)` (PathEngine requires strictly
  positive costs; Sheffi p. 300 sanctions truncation). Truncation bias is measurable and
  β-dependent: at the tworoute card dial β=0.1 the flow bias is below MC resolution
  (measured −8e-5 ± 3e-4, consistent with zero); at β=0.5 it is +1.25e-2 flow —
  documented hazard, not an anchor.
- **Initialization:** load at free-flow costs `t(0)` (Sheffi ch. 12 Step 0), mirroring
  `sue_logit.py`.
- **`provides_gap=False`:** the R=1 self-reported direction norm `‖y_k − v_k‖₁/D` does
  **not** decay (measured O(1) at k=5000) — the model must not early-stop on
  `budget.target_met`; budget axes only. Self-report carries the raw direction norm
  plus a Sheffi eq. [12.52]-style moving average (m=3), provenance-only.
- **Checkpoint indexing:** `trace.record` fires on `v_k` *before* the k-th update; a
  k-iteration run's final certified state is `v_k`, not `v_{k+1}`.

## Decision 4 — Macroreps and bootstrap aggregation

Macroreps are independent **solver** trajectories: `RngBundle(root_seed, macrorep=m)`,
solver-internal draws on model source 0 with `replication=k` per outer iteration. The
certificate uses `SOURCE_EVALUATION` under `macrorep=0` so **all macroreps share one
pinned map** (comparable numbers). Measured: macrorep spread of the certified residual
at k=100 on Sioux Falls is 0.023 — dominating the certificate SE of 0.004 — so
macroreplication, not a tighter certificate, is what CIs need (recommend M=10 tworoute,
M=5 siouxfalls).

What was missing: `run_experiment` writes per-macrorep rows and lists
`SOURCE_BOOTSTRAP` in the manifest but never draws from it. Ship
`experiments/bootstrap.py::bootstrap_ci(values, root_seed, B=10000)` — percentile
(never parametric, P5) CIs drawn on `SOURCE_BOOTSTRAP`, applied to the final certified
residual across macroreps.

## Analytic anchors (recomputed in-test, never trusted digits)

Two-route (`two_route_scenario`, D=4, links (1,1,1,0.5) free-flow): routes are disjoint
2-link chains, so perceived route costs are **independent normals** and probit has a
closed form through the binormal difference:

```
P(A) = Φ( (c_B(D−f_A) − c_A(f_A)) / sqrt(3.5·β) ),  c_A(f)=2+f, c_B(g)=1.5+2g
f_A* : f_A = D·P(A)   (brentq in-test)
```

β=0.1 → `f_A* = 2.4443574168` (UE 2.5; logit θ=0.5 anchor 2.2990959494). Free-flow
first iterate `f_A(1) = 4·Φ(−0.5/√0.35) = 0.7960494390`. Certified residual at the
exact analytic `v*`: consistent with pure MC noise (6.9e-3 at R_cert=1600 vs floor
3.9e-2). Seeded reproducibility: same `(root_seed, macrorep)` byte-identical; different
macrorep differs. Sioux Falls smoke: β=0.5 (native 0.01 h units), R=1, k=30 in ~0.1 s.

## Consequences

- The certificate degrades gracefully from closed-form (logit) to sampled (probit)
  without changing its meaning: one pinned, model-blind map per task; residual +
  uncertainty; ranking ties below the noise threshold.
- The stochastic track (P5/P8) is now exercised end to end: macroreps, spawn-key
  streams, bootstrap CIs — machinery every future simulator/DTA adapter will reuse.
- MC certification cost (R_cert AON sweeps per checkpoint) is the T1-stochastic
  analogue of ADR-002's pinned bfw run — controlled by checkpoint conventions.
- Golden Braess hash `cf00f411…` unchanged; logit tasks hash as before.

## Rejected alternatives

- **Sheffi–Powell objective Z(v) as the certificate:** needs satisfaction terms
  `S_rs = E[min_k C_k]` per OD — path-level MC on top of link MC, no zero-at-equilibrium
  residual semantics. Rejected.
- **Clark's approximation as a deterministic analytic L:** requires path enumeration;
  Sheffi p. 300: inaccurate beyond 10–20 alternatives. Rejected.
- **Certify probit emissions against the logit-Dial map:** certifies the wrong fixed
  point (tworoute: logit θ=0.5 → 2.299 vs probit β=0.1 → 2.444). Rejected outright.
- **Fresh draws per certification (no pinning):** unbiased but breaks P8
  byte-reproducibility and destroys CRN across models. Rejected.
- **Adaptive R_cert (sequential stopping on SE):** unequal treatment across models,
  non-pinned bytes. Rejected — pin R_cert, record se/floor.

## References

- Daganzo, C.F. & Sheffi, Y. (1977). On stochastic models of traffic assignment.
  *Transportation Science* 11(3), 253–274.
- Sheffi, Y. & Powell, W.B. (1982). An algorithm for the equilibrium assignment problem
  with random link times. *Networks* 12(2), 191–207.
- Powell, W.B. & Sheffi, Y. (1982). The convergence of equilibrium algorithms with
  predetermined step sizes. *Transportation Science* 16(1), 45–55.
- Sheffi, Y. (1985). *Urban Transportation Networks*. Prentice-Hall. Ch. 11–12.
