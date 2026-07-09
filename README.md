# TABenchmark

**A shared benchmark for 50 years of traffic assignment models.**

Traffic assignment research spans from Wardrop's principles (1952) and Beckmann's convex
program (1956) through Frank–Wolfe, bush-based solvers, stochastic and elastic-demand user
equilibrium, and dynamic traffic assignment, to today's GNN surrogates and end-to-end
learned models — yet there has never been a shared testbed on which all of them can be
compared under identical networks, demand, data availability, and budgets. TABenchmark is
that testbed:

- **One harness, every vintage.** A minimal model contract plus capability declarations
  lets a compact Frank–Wolfe, a bush-based solver, and a trained surrogate run in the same
  experiment matrix — a white-box solver, a subprocess-wrapped simulator, and a neural
  model are all just objects that emit link flows.
- **Certified, not self-reported.** The harness recomputes every scored metric from the
  emitted link flows — the equilibrium relative gap is a property of the flows, so even a
  black box receives an externally certified gap (and a demand-feasibility audit first).
  Model self-reports are kept as provenance and diffed as an honesty check, never scored.
- **Data levels are first-class.** Full OD tables, noisy stale-prior OD, and per-period
  link counts on sensor subsets: every observation process is a seeded projection of
  stored ground truth, with identifiability conditions reported per configuration.
- **Fair across the decades.** Hardware-free budgets (iterations / shortest-path calls;
  wall-clock recorded but never the ranking axis), training-lineage gates against test
  contamination, content-hashed scenarios, and full run manifests.

Built on the verified canon of **246 references** spanning 12 model families
([docs/REFERENCES.md](docs/REFERENCES.md)), the design synthesizes
[SimOpt](https://github.com/simopt-admin/simopt)'s testbed machinery,
[BO4Mob](https://github.com/UMN-Choi-Lab/BO4Mob)'s benchmark conventions, and
Hazelton's (AOAS 2015) statistical treatment of network flow observability —
see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quickstart

```bash
pip install -e .
python demos/demo_quickstart.py         # built-in Braess scenario, no download
```

```
model             certified rel. gap  feasible
----------------------------------------------
aon                        1.912e-01         1
msa                        1.577e-03         1
fw                         7.188e-14         1
cfw                       -2.060e-16         1
bfw                       -2.060e-16         1
toy-surrogate                    nan         0   <- black box failed the demand
                                                    audit: censored, not scored
```

Run a downloaded instance (data fetched on demand, checksummed) — the certified solver
ladder, an elastic-demand equilibrium, or a learned surrogate, all through one CLI:

```bash
tabench run --scenario siouxfalls --models fw,cfw,bfw,algb,tapas --iterations 300 --out results/
tabench run --scenario winnipeg   --models bfw --iterations 500 --target-gap 1e-4 --out results/
tabench run --scenario elastic-tworoute --models fw-elastic --out results/
tabench list                            # scenarios, models, and T2 estimators
```

Wrap **your** model in three lines — the harness scores it identically:

```python
from tabench import CallableModel, Budget, load_scenario, run_experiment

model = CallableModel(fn=my_gnn_predict, name="my-gnn", trained_on=("tntp-small-v1",))
result = run_experiment(load_scenario("siouxfalls"), [model], Budget(iterations=1))
```

(If `trained_on` intersects the evaluation scenario's lineage, the harness refuses to
run it — that's the point.)

## The model roster

Twenty-six road assignment models, a transit optimal-strategy model (Spiess & Florian 1989), and seven OD-estimation baselines ship today, spanning white-box
solvers, a stochastic track, elastic and combined-distribution demand, day-to-day dynamical
systems (route-swap, link-based, route-space projected-gradient, cost-smoothing SUE, a finite-population stochastic process, and the Cantarella–Cascetta unifying process), a boundedly-rational equilibrium, a capacity-constrained equilibrium, a non-separable-cost asymmetric-VI equilibrium, a multiclass-user equilibrium, and the first learned model — every one certified
from its emitted flows by the identical P1 harness. For what each model does
*differently* from its predecessors and the lineage of the whole family, see the
**[model compendium](docs/MODELS.md)** and its [evolution graph](docs/model-evolution.svg).

| Paradigm | Models |
|---|---|
| Baselines | `aon` all-or-nothing · `msa` method of successive averages |
| Link-based UE | `fw` Frank–Wolfe · `cfw` / `bfw` conjugate & bi-conjugate FW (Mitradjieva & Lindberg 2013) |
| Path / bush-based UE | `gp` gradient projection (Jayakrishnan et al. 1994) · `oba` origin-based assignment (Bar-Gera 2002) · `algb` Algorithm B (Dial 2006) · `tapas` paired alternative segments (Bar-Gera 2010) |
| Stochastic UE | `sue-msa` logit via Dial-STOCH + MSA (Fisk 1980) · `sue-probit-msa` probit via Monte-Carlo MSA (Sheffi & Powell 1982) |
| System optimum | `so-bfw` marginal-cost UE — certified SO gap, price of anarchy, first-best tolls |
| Elastic demand | `fw-elastic` variable-demand UE (Florian & Nguyen 1974 via the Gartner excess-demand transform) |
| Combined distribution + assignment | `evans` fully endogenous OD from fixed trip-end margins via a doubly-constrained gravity, in one convex program (Evans 1976) |
| Day-to-day dynamics | `dtd-swap` Smith's (1984) proportional route-swap dynamical system — models the disequilibrium adjustment toward the UE, with a Beckmann Lyapunov function that decreases monotonically day-to-day · `dtd-friesz` Friesz et al.'s (1994) route-based *projected dynamical system* `ḣ = P_K(h, −c(h))` — the state is per-OD route flows moved along the projection of the negative route-cost vector onto the demand simplex, i.e. projected gradient descent on Beckmann in *route* space (Jacobi: the whole route-flow vector is projected against today's frozen costs at once, via an exact Euclidean simplex projection that conserves each OD's demand every day); reaches the identical UE as `dtd-swap`/`dtd-link` by the same monotone Beckmann descent, distinguishing the route-space projection paradigm from Smith's swap and He-Guo-Liu's link projection · `dtd-link` He, Guo & Liu's (2010) link-based day-to-day — the state is the aggregate *link*-flow vector (not per-OD route flows), adjusted toward the frozen-cost proximal target projected onto the feasible link polytope, reaching the identical UE via the same monotone Beckmann descent · `dtd-swap-sue` Smith & Watling's (2016) logit-SUE sibling of `dtd-swap` — the same route-swap dynamics driven by the Fisk-generalized cost `c_k + (1/θ) ln h_k`, so the rest point is the logit stochastic user equilibrium (not deterministic UE), with Fisk's SUE objective as the monotone day-to-day Lyapunov function · `dtd-horowitz` Horowitz's (1984) cost-smoothing day-to-day SUE — travelers carry a perceived *link*-cost vector exponentially smoothed toward the experienced costs `p ← (1−w)p + w·t(v)` and logit-load at it, reaching the same logit SUE as `sue-msa`; but uniquely among the day-to-day models NO damping is added, so above a task-dependent stability threshold `w* ≈ 0.81` the process settles into a period-2 limit cycle instead of converging — that instability is the phenomenon the model exists to exhibit · `dtd-stochastic` Cascetta's (1989) finite-population *stochastic process* — the benchmark's first genuinely stochastic day-to-day model: each day a finite population of travelers draws routes by multinomial sampling from the Dial-STOCH logit fractions at `dtd-horowitz`'s smoothed perceived costs, so `{p}` is a Markov chain whose "equilibrium" is a stationary *distribution*, not a fixed point; daily flows keep a persistent `O(1/√N)` variability while the emitted burnt-in time average converges (ergodic theorem) to the stationary mean ≈ logit SUE (Davis & Nihan 1993 large-population limit), certified by the same ADR-001 residual — which honestly floors at O(finite-population bias + sampling SE), not at solver precision · `dtd-unifying` Cantarella & Cascetta's (1995) *unifying* process — one two-equation link-space process (exponential cost-learning filter `p ← (1−w)p + w·t(v)` plus choice-inertia update `v ← v + αₙ(ChoiceLoad(p) − v)`, only a fraction `αₙ` of travelers reconsidering each day) whose choice map is gated per scenario: all-or-nothing best response on deterministic tasks (fixed point = Wardrop UE) and the Dial-STOCH logit load on SUE tasks (fixed point = logit SUE); at `α=1` it reduces exactly to `dtd-horowitz` and at `w=1, α=1` to `msa`, and its joint `(α, w)` flip boundary `(2−w)(2−α) = αw\|φ′\|` exhibits C&C's headline: *either* cost memory or choice inertia stabilizes the process that limit-cycles with neither |
| Boundedly-rational UE | `br-ue` an indifference-band relaxation of Wardrop (Mahmassani & Chang 1987) — used routes need only lie within a band `ε` of the shortest, so equilibrium is a *set* and the emitted flow sits at the band edge, not the UE point |
| Side-constrained UE | `sc-tap` UE under hard link capacities `v_a ≤ u_a` (Larsson & Patriksson 1995) — an augmented-Lagrangian whose multipliers are the queueing tolls at binding links; reduces exactly to UE when nothing binds |
| Asymmetric-VI UE | `vi-asym` non-separable cost `t(v) = t_BPR(v) + Cv` with an asymmetric interaction `C ≠ Cᵀ`, solved by Dafermos diagonalization — a variational-inequality equilibrium that minimizes no potential, so no Beckmann/Frank–Wolfe solver reaches it (Dafermos 1980; Smith 1979) |
| Learned (black box) | `learned-surrogate` a ridge volume/capacity surrogate — the ML-wrapper demonstrator |
| OD estimation (T2) | `vzw-entropy` · `gls` (Cascetta 1984) · `spiess` gradient (Spiess 1990) · `spsa` (Spall 1992) · `od-congested` bilevel ODME on congested networks (Yang et al. 1992) · `od-kalman` Davis–Nihan (1993) linear-Gaussian estimation from a *time series* of counts — whitens by the DN cross-link covariance and an AR(1) effective-sample-size (`τ`) correction, the first estimator to use the counts' covariance and day-to-day autocorrelation rather than their mean · `prior` stale-prior baseline |

## What ships

| Component | Contents |
|---|---|
| Core | `Scenario` (frozen, content-hashed; optional SUE dispersion `θ`; optional `ElasticDemand`), `Capabilities`, `Budget` (incl. Boyce-style convergence target), `Trace`, spawn-key RNG schema |
| Data | Defensive TNTP parser, commit-pinned checksummed fetcher, per-network units metadata; scenario ladder Braess → Sioux Falls → Anaheim → Barcelona → Winnipeg, plus built-in analytic anchors (two-route logit-SUE, two-route probit, elastic two-route) |
| Models | 26 assignment models across the roster above: baselines, link-based UE, path/bush-based UE (gradient projection, origin-based, Algorithm B, TAPAS), stochastic UE, system optimum, elastic demand, combined distribution+assignment, day-to-day dynamical systems (route-swap toward UE, its logit-SUE variant, link-based, route-space projected-gradient, Horowitz cost-smoothing SUE, Cascetta's finite-population stochastic process, and the Cantarella–Cascetta unifying cost-learning + choice-inertia process), a boundedly-rational band equilibrium, a side-constrained capacitated equilibrium, an asymmetric-VI non-separable-cost equilibrium, a multiclass-user equilibrium (Dafermos 1972: per-class demand + class-coupled costs, certified by a class-summed VI residual), and a learned surrogate — mixing freely in one experiment matrix via the `CallableModel` adapter; **plus**, in its own parallel module, a **transit** optimal-strategy model (Spiess & Florian 1989: uncongested frequency-based hyperpaths on a directed transit multigraph, certified by an LP optimality gap) |
| Metrics | Certified relative gap / average excess cost / Beckmann objective; certified SO gap + price of anarchy + first-best tolls (Yang & Huang 1998; Roughgarden & Tardos 2002); SUE fixed-point residual — closed-form for logit ([ADR-001](docs/design/adr-001-logit-sue-dial-certificate.md)), pinned-Monte-Carlo with noise floor for probit ([ADR-003](docs/design/adr-003-probit-sue-mc-certificate.md)); route-flow proportionality diagnostic for TAPAS ([ADR-004](docs/design/adr-004-proportionality-certificate.md)); demand-recomputing gap for elastic demand ([ADR-005](docs/design/adr-005-elastic-demand.md)); a gravity-recomputing gap for combined distribution+assignment ([ADR-007](docs/design/adr-007-combined-distribution-assignment.md)); a necessary indifference-band acceptability check for boundedly-rational UE ([ADR-008](docs/design/adr-008-boundedly-rational-ue.md)); an exact link-capacity feasibility check for side-constrained UE ([ADR-009](docs/design/adr-009-side-constrained-ue.md)); a demand-aware feasibility audit and flow RMSE vs. best-known throughout |
| Observe | `FullOD`, `LinkCounts` (sensor mask × periods × noise), `StalePriorOD`, Hazelton identifiability check |
| Estimation (T2) | OD estimation from link counts under a pinned-assignment certificate ([ADR-002](docs/design/adr-002-t2-estimation-certificate.md)): VZW entropy, Cascetta GLS, Spiess gradient, SPSA, the Yang et al. (1992) bilevel congested-network estimator (`od-congested`), the Davis–Nihan (1993) linear-Gaussian time-series estimator (`od-kalman`, [ADR-012](docs/design/adr-012-dn-kalman.md); a new `DayToDayCounts` observation level emits the DN large-population VAR(1) count series it consumes), and a stale-prior baseline; held-out sensors rank, identifiability reported per task |
| Learned models | The wrapper, certificate, and lineage gate a learned model plugs into ([ADR-006](docs/design/adr-006-learned-model-certification.md)) — trained on a synthetic family, evaluated on disjoint TNTP networks |
| Experiments | Grid runner (T1) + estimation runner (T2), CSV results, full provenance manifests |
| Validation | Per-model provenance report ([docs/VALIDATION.md](docs/VALIDATION.md)) tying every solver to an independent oracle: published best-known flows, cross-solver agreement, and exact analytic anchors |
| Tests | 570 tests: analytic Braess UE/SO and two-route logit/probit/elastic oracles, a symmetric-bipartite combined distribution+assignment oracle (with the certificate's aggregate-multicommodity limitation pinned transparently), the day-to-day route-swap convergence + Beckmann-Lyapunov monotonicity, its logit-SUE sibling reaching the analytic logit fixed point with a monotone Fisk-Lyapunov objective (with its two honestly-scoped hard-network limits pinned: the now-exposed efficient-set enumeration cap and the never-pruned-stale-route exact rest point that plateaus O(1) above sue-msa), the link-based day-to-day invariance (link flows stay in the OD polytope every day) reaching the same certified UE as the route-swap dynamics — plus a regression pinning its live-cost Gauss-Seidel inner solve, which must recompute the proximal path costs before each shift or the default config stalls short of UE on overlapping high-curvature instances, the route-space projected-gradient (Friesz) day-to-day reaching that identical certified UE with a hand-checked projection-step direction, an exact demand-conserving simplex projection, and an Armijo-backtracking monotone-descent regression, the Horowitz cost-smoothing day-to-day SUE converging to the analytic logit split below its stability threshold and settling into a period-2 limit cycle above it, the Cascetta stochastic-process day-to-day (sampled-load unbiasedness at CLT tolerance, per-day demand feasibility, persistent daily variability shrinking O(1/√N) in population, seeded byte-reproducibility with independent macroreps, the burn-in soft handoff, and its task-dependent stability + certificate-amplification adversarial regressions), the Cantarella–Cascetta unifying day-to-day process recovering *both* anchor limits (the Wardrop UE and the analytic logit-SUE split) with exact reductions to `dtd-horowitz` at `α=1` and to `msa` at `w=1, α=1` pinned to float precision, plus its re-derived joint `(α, w)` flip boundary — no inertia limit-cycles, either form of inertia converges, bracketing Horowitz's `w* ≈ 0.81` at `α=1`, the boundedly-rational two-route band edge (and its necessary-not-sufficient certificate); the dynamic-network-loading core primitives (fundamental diagrams, link/node-model axioms, time-varying demand, loader, and DNL certificates) — including the gated-review C6/C8 certificate hardening (C6 free-flow reformulated as the per-vehicle travel-time restatement of the C4 envelope, a redundant belt-and-suspenders cross-check honestly scoped gating-complete only at CFL=1; C8 turning-fraction fidelity now gating every diverge's split `d_in[out_j] = Σᵢ frac[i,j]·d_out[in_i]` — the review caught that the 1-in-only draft silently certified a gross multi-in merge-diverge violation); the Daganzo (1994) cell transmission model (`ctm`: the first DNL `LinkModel`, Godunov cell dynamics at CFL=1 — free-flow translation `n_out(t)=n_in(t−L/vf)` bit-exact, a queue-spillback anchor reproducing the Rankine-Hugoniot shock speed `−0.5` with exact bottleneck boundary curves and the congested density `κ−q_B/w=3.5`, backward-wave diffusion correctly left non-gating Tier-B); the Yperman (2007) link transmission model (`ltm`: the second DNL `LinkModel`, the stateless Newell-Daganzo cumulative-curve method — free-flow bit-exact, reproducing CTM's symmetric-bottleneck curves byte-for-byte, an asymmetric `w<vf` spillback with RH speed `−0.25`/shock-to-`x=0`-at-`t=18`, and — its concrete advantage over CTM — running on a non-cell-aligned grid `L/vf=1.5` where CTM raises; both primaries open and read, with the Yperman signed-`w` vs `wave_speed>0` sign convention pinned); the Yang (1992) bilevel OD-estimation-on-congested-networks estimator (closed-form θ-weighted anchor, the prior↔count θ spectrum, and congested-fixed-point recovery of the equilibrium-consistent truth); the Dafermos (1980)/Smith (1979) asymmetric-VI UE with non-separable costs (hand-derived asymmetric anchor whose equilibrium differs from BOTH the plain-UE and the symmetrized-Beckmann split — a flow no potential-minimizing solver reaches — with the VI residual harness-recomputed at the asymmetric cost and reducing exactly to Frank-Wolfe UE when the interaction vanishes); the Davis–Nihan (1993) linear-Gaussian time-series OD estimator (`od-kalman`: the two-route DN cross-link covariance closed form `(D²/N)·p_A·p_B` with same-route/cross-route sign structure, the single-sensor DN-GLS closed form, the AR(1) `τ = (1+ρ)/(1−ρ)` effective-sample-size factor recovered from the count series, UE-centered day-to-day counts whose fluctuation shrinks O(1/population), and planted-demand recovery — distinct from `gls` in both the off-diagonal covariance whitening and the temporal correction); the Dafermos (1972) multiclass-user equilibrium (`multiclass`: hand-derived symmetric/integrable and asymmetric/genuine-VI two-class anchors recovered exactly — cars and trucks routing distinctly — with the class-summed VI residual harness-recomputed from the emitted per-class flows, a per-class conservation audit, and censoring of an aggregate-only flow; the additive `class_link_flows` contract leaving the golden Braess hash byte-identical); the Spiess & Florian (1989) transit optimal strategies (`transit-strategy`: the two common-lines anchors recovered exactly — both lines attractive → 24 min with a 2:1 frequency split, and the attractiveness threshold excluding a slow line → 21 min all on one line — a multi-leg interchange, a deterministic-walk-dominates case, the primal-dual identity `Z_emitted = Z*`, and a multi-destination shared-node scenario plus a near-zero-frequency parasite-arc guard (both adversarial-review CRITICALs, fixed to a per-destination LP-minimal-wait `w_i = maxₐ vₐ/fₐ` certificate that keeps the optimality gap provably ≥ 0), in a parallel module that leaves the road suite and golden hash byte-untouched); best-known-solution regressions on Sioux Falls, Anaheim, Barcelona, Winnipeg; cross-family solver agreement; conjugacy-identity and golden-hash regressions |

The certified solver ladder on Winnipeg (147 zones, 2,836 links; iterations to
self-monitored relative gap 1e-4, then the externally certified gap at a fixed
100-iteration budget):

| model | iters to RG 1e-4 | certified gap @ 100 iters |
|---|---|---|
| aon | – | 3.2e-01 |
| msa | – | 1.4e-03 |
| fw  | 161 | 2.2e-04 |
| cfw | 70 | 5.6e-05 |
| bfw | 57 | 2.8e-05 |

The staged roadmap toward the full canon (DTA, dynamic network loading, day-to-day
dynamics, engine adapters, and the T3 intervention track) is in
[docs/ROADMAP.md](docs/ROADMAP.md).

## Why it is trustworthy

TABenchmark rests on nine design principles ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md));
four carry the most weight:

- **P1 — the certificate principle.** The harness, never the model, computes every scored
  metric. The relative gap is a property of `(link_flows, scenario)`, so a 1975 Frank–Wolfe
  and a 2025 GNN share one leaderboard, and a black box cannot self-report its way to the top.
- **P2 — scenarios are data.** Every `Scenario` is frozen and content-hashed; a silently
  edited network cannot masquerade as the benchmark instance.
- **P7 — fairness is enforced by the harness.** Training-lineage gates block evaluating a
  model on a scenario it was trained on; budgets count work, not wall-clock.
- **P9 — data are fetched, never vendored.** Networks are downloaded on demand, checksummed
  at fetch, and regression-tested against published best-known objectives — no dataset is
  redistributed.

Each model is additionally validated against an **independent oracle**, not just its own
gap ([docs/VALIDATION.md](docs/VALIDATION.md)): the UE solvers converge to the published
best-known link flows and agree with one another across algorithm families (link-based,
path-based, bush-based, PAS-based); the Sioux Falls best-known flows reproduce the
Transportation Networks published optimal Beckmann objective (`42.31335287107440`) to full
precision; and the Braess network pins the exact UE (route cost 92), the system optimum
(route cost 83), and the price of anarchy (`≈ 1.108`).

## Data licensing

Benchmark networks come from
[TransportationNetworks](https://github.com/bstabler/TransportationNetworks), donated
for academic research. TABenchmark **never redistributes** these files: they are fetched
on demand from a commit-pinned URL, verified against SHA-256 checksums, cached locally,
and cited (`tabench fetch <scenario>` prints the citation). The package code is MIT.

## Citing

If you use TABenchmark, please cite this repository (see `CITATION.cff`) and the
original references of any model or dataset you evaluate — all 246 entries in
[docs/REFERENCES.md](docs/REFERENCES.md) carry verified BibTeX in
[docs/references.bib](docs/references.bib).

## Contributing

New models, networks, observation levels, and reference implementations from the
community are the purpose of this repository — see
[CONTRIBUTING.md](CONTRIBUTING.md) for the model contract and conformance checklist.

---

Maintained by the [UMN Choi Lab](https://choi-seongjin.umn.edu/) ·
University of Minnesota Twin Cities
