# ADR-001 — Logit SUE: Dial's STOCH loading, MSA-SUE, and the fixed-point certificate

**Status:** accepted (implemented in v0.x)
**File:** `docs/design/adr-001-logit-sue-dial-certificate.md`

## Context

The roadmap (ARCHITECTURE §5, v0.x) calls for logit stochastic user equilibrium as the
first route-choice component, and T1 already promises "SUE fixed-point residual for SUE
tasks" (ARCHITECTURE §2). The hard question is not the solver — MSA with a stochastic
loader is textbook (Sheffi 1985, ch. 12) — but the **certificate**: P1 demands that the
harness recompute every scored metric from `(link_flows, scenario)` alone, for black
boxes included. For UE this is the externally computed relative gap. For SUE there is no
analogue via an objective: Fisk's (1980) equivalent convex program is

```
min  Σ_a ∫₀^{v_a} t_a(s) ds  +  (1/θ) Σ_rs Σ_p f_p^rs ln f_p^rs
```

whose entropy term is a function of **path** flows `f_p`, and a link-flow vector does not
determine its path-flow decomposition (nor, under Dial, even the admissible path set,
which moves with costs). So the Fisk objective cannot be evaluated from emitted link
flows and cannot be the certificate. What *can* be evaluated from link flows is the
defining fixed-point condition of SUE itself: `v = L(t(v), θ)`, where `L` is a pinned,
deterministic stochastic-loading map. This ADR pins that map (Dial's STOCH), the solver
(MSA), the certificate (an L1 fixed-point residual in AEC-like per-traveler units), and
where `θ` lives (an optional `Scenario` field that leaves every existing content hash
unchanged).

Terminology note: "stochastic" in SUE refers to traveler perception, not algorithmic
randomness. Dial's logit loading is a closed-form deterministic function of `(t, θ)` —
no RNG, no Monte Carlo — so this model runs on the *deterministic* track (M=1). P5's
"stochastic track: SUE" wording should be refined to "simulation-based SUE (e.g. probit,
v1)"; logit-SUE-via-Dial belongs with FW and MSA.

## Decision 1 — The loading map: Dial's STOCH, double-pass, origin-based efficiency

`L(t, θ)` is Dial's (1971) STOCH algorithm in the **double-pass form with the
origin-based efficient-link criterion `r(i) < r(j)`**, exactly as presented in Sheffi
(1985, §11.2 — the canonical free reference implementers actually use).

Dial's paper defines a "reasonable" (efficient) path with two labels — every link must
lead *away from the origin* (`r(i) < r(j)`, `r` = shortest time from the origin) *and
toward the destination* (`s(i) > s(j)`, `s` = shortest time to the destination) — but
his recommended double-pass algorithm, and Sheffi's textbook presentation of it, retain
only the origin criterion. We follow that choice because:

1. **Reproducibility.** Sheffi §11.2 is the presentation of record; matching it means a
   third party implementing "Dial per Sheffi" reproduces our certificate bit-for-bit in
   exact arithmetic. The two-label variant makes the efficient set destination-dependent,
   and implementations disagree on details.
2. **Complexity.** Origin-based efficiency lets one forward+backward sweep per origin
   serve *all* destinations (like AON): `O(|origins| · (m + n log n))`. The two-label
   criterion needs per-OD-pair sweeps: `O(|zones|²)` label trees.
3. **Task-definition honesty.** Whatever criterion we pick *defines* the benchmark's SUE
   task; models and harness share the same map, so the choice must be pinned, not
   optimal. We pin the canonical one and record it in the scenario card
   (`sue.loading: dial-stoch`).

### Exact algorithm (per origin `o` with a positive demand row)

All computation is on the **PathEngine expanded graph** (`models/_paths.py`): centroids
below `first_thru_node` keep their index in the tail role and get an arc-less shadow
index in the head role, so no efficient path can traverse a centroid — the distance
labels `r(·)` automatically honor centroid restrictions. Costs `t = Network.link_cost(v)`
are generalized (BPR time + toll/distance fixed cost), strictly positive and finite
(guarded exactly as in `PathEngine._graph`).

1. **Labels.** Dijkstra from `o`'s tail index: `r(i)` for every expanded node.
2. **Efficient links.** `E_o = { a=(i→j) : r(i), r(j) finite and r(i) < r(j) }`, strict
   float comparison, no tolerance (ties excluded; see Decision 3).
3. **Link likelihood** (Sheffi eq. 11.7-style):
   `L_a = exp(x_a)` with `x_a = θ·(r(j) − r(i) − t_a)` for `a ∈ E_o`, else `L_a = 0`.
   By shortest-path optimality `r(j) ≤ r(i) + t_a`, so `x_a ≤ 0` and `L_a ≤ 1`:
   likelihoods can never overflow, and `x_a = 0` exactly on shortest-path links.
4. **Forward pass (node weights, log domain), ascending `r`.** Sheffi's recursion is
   `W(o)=1`, `W(j) = Σ_{(i→j)∈E_o} L_ij · W(i)`. `W(j)` sums `exp(θ(r(j) − C_P))` over
   efficient paths `P: o→j` (telescoping product), and the number of efficient paths can
   be exponential — `W` can overflow float64 on large networks at small θ. We therefore
   compute `b(j) = log W(j)` directly: `b(o)=0`,
   `b(j) = logsumexp_{(i→j)∈E_o} ( x_ij + b(i) )` (max-shifted). This is algebraically
   identical to Sheffi's recursion and overflow-free. Invariant: every reachable node's
   shortest path consists of efficient links with `x=0`, so `W(j) ≥ 1`, `b(j) ≥ 0` —
   no zero-weight node, no 0/0 in the backward pass, ever (guaranteed by the Network
   validation `free_flow_time > 0`).
5. **Backward pass (assignment), descending `r`.** Initialize node volume
   `V(head_index(d)) = q_od` for each destination `d ≠ o` with `q_od > 0` (intrazonal
   demand skipped; unreachable positive-demand destination raises `RuntimeError`,
   mirroring `PathEngine.all_or_nothing`). Sweep nodes in descending `r`; at node `j`
   with `V(j) > 0`, split over incoming efficient links with fractions
   `φ_ij = exp(x_ij + b(i) − b(j))` — each exponent ≤ 0 by definition of `b(j)`, so
   `φ ∈ [0,1]` with `Σφ = 1` in exact arithmetic; renormalize by the float sum so link
   flows conserve demand to machine precision — then `flow_ij += V(j)·φ_ij`,
   `V(i) += V(j)·φ_ij`.

The resulting path shares are exactly the logit model over Dial-efficient paths:
`Pr(P) = exp(−θ C_P) / Σ_{P'∈E_o} exp(−θ C_{P'})`.

**Processing order and ties.** Nodes with equal `r` cannot be joined by an efficient
link (strict inequality), so any order among equal labels is correct; a stable argsort
on `r` (as in PathEngine) keeps runs bit-deterministic. Underflow of `exp(x)` for deep
detours silently rounds negligible path shares to zero — benign, since shortest-path
terms are `exp(0)=1` and never underflow.

**θ units.** θ multiplies generalized-cost differences, so its unit is 1/(native cost
unit) and is **network-specific** (P9): Sioux Falls times are 0.01-hour units, so
`θ = 1.0` there means 100 h⁻¹. Scenario cards must state the unit next to the value.

**θ → ∞ limit.** Shares concentrate on minimum-cost efficient paths, split *evenly*
across exact ties (unlike AON's arbitrary single-tree tie-breaking) — documented, and
irrelevant off ties.

## Decision 2 — The reference solver: MSA-SUE (`sue-msa`)

`models/sue_logit.py::DialSUEModel`, registered as `sue-msa`, mirrors `msa.py`:

- `v_1 = L(t(0), θ)` (stochastic loading at free-flow costs); then for `k = 1, 2, …`:
  `y_k = L(t(v_k), θ)`; checkpoint `v_k` with self-report
  `sue_fixed_point_residual = ‖y_k − v_k‖₁ / D`; step `v_{k+1} = v_k + (y_k − v_k)/k`
  (the repo's MSA indexing; identical to `v_{k+1} = v_k + (1/(k+1))(L(t(v_k)) − v_k)`
  up to an index shift). Powell & Sheffi (1982) prove convergence for predetermined
  Blum step sizes (`Σα_k = ∞`, `Σα_k² < ∞`) given a continuous loading map; Dial's
  cost-dependent efficient sets make `L` only piecewise continuous, so the guarantee is
  local (efficient set constant near the fixed point) — a documented caveat, not a
  blocker (Decision 3).
- θ is read from `scenario.sue_theta`; `None` raises `ValueError` — θ is task data, not
  a model factor, so no solver may tune it (P7).
- Budget convention: one full Dial load = 1 `sp_call` (the same one-batched-sweep unit
  as one `all_or_nothing` in FW/MSA; the Dijkstra sweep dominates both).
- `Capabilities(paradigm="sue", deterministic=True, provides_gap=True, seedable=True)` —
  `"sue"` is already in `PARADIGMS`; `deterministic=True` routes it to M=1 in the runner.
- The self-reported residual at checkpoint `v_k` is byte-comparable to the harness value
  (both are `‖L(t(v_k),θ) − v_k‖₁/D`), giving the FW-style honesty regression for free.

**Why the Fisk objective is not reported:** it needs path flows (entropy term) that link
flows do not determine; under Dial the path set additionally moves with costs. Stated in
the module docstring so nobody "fixes" this later.

## Decision 3 — The certificate: SUE fixed-point residual (extends P1)

For a scenario with `sue_theta = θ`, the harness computes, from emitted flows `v` that
pass the existing demand-aware feasibility audit:

```
t = network.link_cost(v)                    # same costs as the UE certification
v̂ = L(t, θ)                                 # ONE Dial load, deterministic
sue_fixed_point_residual = ‖v − v̂‖₁ / D     # D = total demand
```

**Norm and normalization.** Both `v` (audited) and `v̂` (constructed) route the same OD
totals, so `v − v̂` is a pure redistribution of flow among routes; the L1 norm measures
the total redistributed volume in vehicle·link units. Dividing by total demand `D`
yields *misallocated link-traversals per traveler* — the same intensive, per-traveler
convention as AEC (= excess time per traveler), zero exactly at equilibrium, and
comparable across ladder rungs (Braess `D=6` vs Chicago `D≈10⁶`). L2 has no flow
interpretation and is dominated by high-volume links; L∞ ignores diffuse misallocation;
normalizing by `‖v‖₁` instead of `D` would conflate residual with average trip length.

**Discontinuity caveat.** `t ↦ L(t, θ)` is discontinuous exactly where an efficient set
changes — a label tie `r(i) = r(j)`, a measure-zero event in cost space but reachable at
symmetric networks. If a scenario's fixed point sits *on* a tie, the residual need not
approach 0 and MSA can oscillate between the two branch maps. Within a fixed efficient
set, `L` is smooth and the residual is locally Lipschitz
(`≈ ‖(I − ∇L·∇t)(v − v*)‖₁/D` near `v*`). Consequences: the strict-inequality rule is
pinned (no tolerance parameter to disagree about); SUE scenario cards must verify —
analytically or empirically — that the fixed point is tie-free, recording exceptions in
the known-defects style (P9).

**Expected magnitudes.** At an exact analytic fixed point the residual is pure float
noise (cost evaluation + Dijkstra + logsumexp + brentq root tolerance): observed
`< 1e-12` on the two-route scenario; tests assert `< 1e-8`. For MSA-after-k the harness
residual at checkpoint `v_k` *is* the MSA direction norm; measured on the two-route
network: `2.8e-3` (k=10), `6.2e-6` (k=100), `1.1e-7` (k=500); on real networks expect
markedly slower, roughly `O(1/k)`-like decay (Powell & Sheffi prove no rate).

**Why this is P1-valid for black boxes.** The residual is a function of
`(link_flows, scenario, θ)` only — no path flows, no iterates, no model internals — and
costs one deterministic Dial load per checkpoint (≈ one AON; certification cost for SUE
scenarios roughly doubles). Censoring precedes certification: flows failing the demand
audit get `feasible = 0` and a NaN residual, so an all-zero emission is censored, never
scored. UE metrics (relative gap, AEC, Beckmann) remain **descriptive columns** on SUE
tasks — they are strictly positive at SUE by design — and the SUE task ranks by the
residual alone. Fit-vs-gap stays a pair, never a scalar (ARCHITECTURE §2).

## Decision 4 — Where θ lives: `Scenario.sue_theta`, hash-append-when-set

```python
@dataclass(frozen=True)
class Scenario:
    ...
    reference: ReferenceSolution | None = None
    family: str = field(default="")
    sue_theta: float | None = None      # NEW: logit dispersion, 1/(native cost unit)
```

`__post_init__` gains: `sue_theta` must be `None` or finite and `> 0`.
`content_hash()` gains, **after** the existing array loop and before `hexdigest()`:

```python
if self.sue_theta is not None:
    h.update(f"sue_theta={float(self.sue_theta)!r};".encode())
```

Because the update is conditional and appended last, the byte stream fed to SHA-256 when
`sue_theta is None` is identical to today's — every existing hash (Braess
`cf00f411cdccec88…`, Sioux Falls pins in manifests, `trained_on` lineage tokens) is
preserved, enforced by a new golden-hash regression test. When θ is set the hash
changes, which P2 *requires*: θ parametrizes a scored metric, so two scenarios differing
only in θ are different benchmark instances and must never share a hash. `!r` on the
float matches the existing `tw=…!r;dw=…!r` convention (exact round-trip repr).

**Alternatives rejected.** *Task object:* the right home eventually, but no `Task` class
exists in code (tasks are YAML strings), and θ must be content-hashed *today*; the
Scenario field is forward-compatible — a v1 Task layer can construct/point to the hashed
scenario. *Evaluator argument:* violates P2 — two runs with different θ would share a
`scenario_hash`, a silently different problem masquerading as the same instance, and
manifests would not pin θ. *Model factor:* worst — models tuning the task definition
violates P7 outright.

## Decision 5 — Analytic anchor: the two-route scenario

`Network` validation forbids parallel links, so the classic two-parallel-links example
is realized as two disjoint 2-link routes. `data/builtin.py::two_route_scenario(demand=4.0,
sue_theta=0.5)`: nodes 1 (origin zone), 2 (destination zone), 3, 4; `first_thru_node=1`;
links, in order, with linear costs via the existing `bpr_linear` device:

| link | cost | role |
|---|---|---|
| 1→3 | `t = 1` (constant) | route A first leg |
| 3→2 | `t = 1 + f` | route A, `c_A(f_A) = 2 + f_A` |
| 1→4 | `t = 1` (constant) | route B first leg |
| 4→2 | `t = 0.5 + 2f` | route B, `c_B(f_B) = 1.5 + 2 f_B` |

Since `r(3) = r(4) = 1 < min(c_A, c_B) ∈ [1.5, ∞)` at every nonnegative flow, both
routes are Dial-efficient at *all* costs encountered, so `L` reduces exactly to a binary
logit and the SUE fixed point to the scalar equation
`f_A = D / (1 + exp(θ·(c_A(f_A) − c_B(D − f_A))))`, solvable by `brentq` inside the test
(no hard-coded truth digits). Key verified numbers (θ=0.5, D=4): free-flow Dial load
`(1.7512939965, …)`; fixed point `f_A* = 2.2990959494`; UE `f_A = 2.5`; θ-ladder
`|f_A*(θ) − 2.5| = 0.2009, 0.0724, 0.0318, 0.0034` for θ = 0.5, 2, 5, 50; certificate
`≈ 1.165` after shifting 0.5 units A→B. Full test list in the test-plan section below.

## Decision 6 — Integration

The loader lives in **`src/tabench/models/_stoch.py`** as a `StochEngine` class
(mirroring `_paths.py::PathEngine`), because both the SUE model and the Evaluator need
it and `metrics/gaps.py` *already* imports `..models._paths.PathEngine` — the precedent
direction is `metrics → models → core`, acyclic. `_stoch.py` imports only
`core.scenario` (+ scipy) and reuses PathEngine's expansion. A top-level shared
`routing/` package (moving `_paths.py` too) is the cleaner long-term layering but a
larger refactor; deferred, noted for v1.

Touch points: `Evaluator` gains the residual column when `scenario.sue_theta` is set;
runner CSV gains `sue_fixed_point_residual` and `self_sue_residual`;
`models/sue_logit.py` + registry export; `scenarios/0tworoute-sue.yaml` card with an
explicit `sue: {theta, theta_units, loading}` block; ARCHITECTURE metric-definitions
bullet + P5 wording refinement; README model/metric rows. Exact diffs in the
implementation spec accompanying this ADR.

## Consequences

- Anything emitting feasible link flows on a SUE scenario — a 1971 STOCH heir or a 2025
  GNN — receives an externally certified SUE residual, in per-traveler units, from one
  deterministic Dial load. P1 extends to SUE with no new trust assumptions.
- The efficient-set discontinuity is accepted (canonical fidelity over smoothness) and
  managed by scenario-card vetting rather than by a hidden tie tolerance.
- Existing content hashes, manifests, and lineage tokens are untouched; a golden-hash
  test makes that a regression guarantee.
- Evaluator cost on SUE scenarios roughly doubles per checkpoint (AON + Dial load);
  acceptable, with a shared-Dijkstra optimization available later.

## References

- Dial, R.B. (1971). A probabilistic multipath traffic assignment model which obviates
  path enumeration. *Transportation Research* 5(2), 83–111.
- Sheffi, Y. (1985). *Urban Transportation Networks*, ch. 11 (STOCH), ch. 12 (MSA-SUE).
  Free PDF at sheffi.mit.edu.
- Fisk, C. (1980). Some developments in equilibrium traffic assignment.
  *Transportation Research Part B* 14(3), 243–255.
- Powell, W.B. & Sheffi, Y. (1982). The convergence of equilibrium algorithms with
  predetermined step sizes. *Transportation Science* 16(1), 45–55.
- Daganzo, C.F. & Sheffi, Y. (1977). On stochastic models of traffic assignment.
  *Transportation Science* 11(3), 253–274.
