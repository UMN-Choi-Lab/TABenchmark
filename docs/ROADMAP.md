# Implementation Roadmap

The tier-1 canon (63 must-implement references from
[REFERENCES.md](REFERENCES.md)), staged by version. Checked items ship in the
current release (v0 also ships MSA and all-or-nothing as baselines).

Version staging: **v0** core harness + link-based solvers -> **v0.x** accelerated
FW, logit SUE, Anaheim/Barcelona/Winnipeg rungs (this release; plugin registry and
profiles still open) -> **v1** bush-based solvers, SUE variants, static extensions,
T2 estimation track -> **v2** DTA, network loading, engine adapters, day-to-day,
T3 interventions.

## Foundations — v0-v1 (formulations underpin metrics)

- [x] Wardrop (1952) — *Some Theoretical Aspects of Road Traffic Research* (metric/protocol) — **shipped in v0** (equilibrium conditions in the certified gap)
- [x] Beckmann et al. (1956) — *Studies in the Economics of Transportation* (white-box solver) — **shipped in v0** (Beckmann objective in metrics)
- [x] Bureau of Public Roads (1964) — *Traffic Assignment Manual* (network-loading component) — **shipped in v0** (BPR link performance function (Network.link_cost))
- [x] Dafermos (1972) — *The Traffic Assignment Problem for Multiclass-User Transportation Networks* (white-box solver) — **shipped** as `multiclass` (adr-013)
- [ ] Smith (1979) — *The Existence, Uniqueness and Stability of Traffic Equilibria* (metric/protocol)
- [x] Dafermos (1980) — *Traffic Equilibrium and Variational Inequalities* (white-box solver) — **shipped** as `vi-asym`

## Link-based UE algorithms — v0 (FW, MSA) / v0.x (CFW, BFW)

- [x] Frank & Wolfe (1956) — *An Algorithm for Quadratic Programming* (white-box solver) — **shipped in v0** (Frank-Wolfe solver)
- [x] LeBlanc et al. (1975) — *An Efficient Approach to Solving the Road Network Equilibrium Traffic Assignment Problem* (white-box solver) — **shipped in v0** (Frank-Wolfe solver)
- [x] Boyce et al. (2004) — *Convergence of Traffic Assignments: How Much Is Enough?* (metric/protocol) — **shipped in v0.x** (convergence target protocol (Budget.target_relative_gap))
- [x] Mitradjieva & Lindberg (2013) — *The Stiff Is Moving—Conjugate Direction Frank-Wolfe Methods with Applications to Traffic Assignment* (white-box solver) — **shipped in v0.x** (conjugate and bi-conjugate FW solvers)

## Path/bush-based UE algorithms — v1

- [x] Jayakrishnan et al. (1994) — *A faster path-based algorithm for traffic assignment* (white-box solver) — **shipped in v1** (path-based gradient projection solver (gp))
- [x] Bar-Gera (2002) — *Origin-based algorithm for the traffic assignment problem* (white-box solver) — **shipped** as `oba` (origin-based M/D-label proportion solver)
- [x] Dial (2006) — *A path-based user-equilibrium traffic assignment algorithm that obviates path storage and enumeration* (white-box solver) — **shipped in v1** (Algorithm B bush-based solver (algb))
- [x] Bar-Gera (2010) — *Traffic assignment by paired alternative segments* (white-box solver) — **shipped in v1** (TAPAS paired-alternative-segment solver (tapas) + proportionality diagnostic (ADR-004))

## Stochastic UE & route choice — v0.x (Dial, logit-SUE MSA) / v1 (probit)

- [x] Dial (1971) — *A probabilistic multipath traffic assignment model which obviates path enumeration* (network-loading component) — **shipped in v0.x** (STOCH loading map (models/_stoch.py))
- [x] Daganzo & Sheffi (1977) — *On stochastic models of traffic assignment* (white-box solver) — **shipped in v1** (SUE definition underlying the probit task)
- [x] Fisk (1980) — *Some developments in equilibrium traffic assignment* (white-box solver) — **shipped in v0.x** (logit SUE task (fixed-point certificate, ADR-001))
- [x] Powell & Sheffi (1982) — *The convergence of equilibrium algorithms with predetermined step sizes* (white-box solver) — **shipped in v0.x** (MSA-SUE solver step sizes)
- [x] Sheffi & Powell (1982) — *An algorithm for the equilibrium assignment problem with random link times* (white-box solver) — **shipped in v1** (probit SUE solver (sue-probit-msa) + MC certificate (ADR-003))

## System optimum & pricing — v1

- [x] Yang & Huang (1998) — *Principle of Marginal-Cost Pricing: How Does It Work in a General Road Network?* (white-box solver) — **shipped in v1** (first-best marginal-cost tolls (metrics.so))
- [x] Roughgarden & Tardos (2002) — *How Bad Is Selfish Routing?* (metric/protocol) — **shipped in v1** (price-of-anarchy protocol + certified SO gap)

## Static extensions — v1

- [x] Florian & Nguyen (1974) — *A Method for Computing Network Equilibrium with Elastic Demands* (white-box solver) — **shipped in v1** (elastic (variable) demand UE task/problem (ADR-005))
- [x] Evans (1976) — *Derivation and analysis of some models for combining trip distribution and assignment* (white-box solver) — **shipped** as `evans` (ADR-007)
- [x] Mahmassani & Chang (1987) — *On Boundedly Rational User Equilibrium in Transportation Systems* (route-choice component) — **shipped** as `br-ue` (indifference-band relaxation, ADR-008)
- [x] Spiess & Florian (1989) — *Optimal strategies: A new assignment model for transit networks* (white-box solver) — **shipped** as `transit-strategy` (adr-014, parallel `transit/` module)
- [x] Larsson & Patriksson (1995) — *An augmented Lagrangean dual algorithm for link capacity side constrained traffic assignment problems* (white-box solver) — **shipped** as `sc-tap` (ADR-009)

## Analytical DTA — v2

- [x] Vickrey (1969) — *Congestion Theory and Transport Investment* (white-box solver) — **shipped** as `vickrey` (adr-019, the first departure-time equilibrium — a parallel `bottleneck/` module with a closed-form UE/SO and a P1 certifier that recomputes the point queue + generalized costs from the emitted departure curve; `equilibrium_gap=0` for the UE, PoA=2)
- [x] Merchant & Nemhauser (1978) — *A Model and an Algorithm for the Dynamic Traffic Assignment Problems* (white-box solver) — **shipped** as `merchant-nemhauser` (adr-020, the first network DTA model — a parallel `dta/` module: exit-function scenario, Carey(1987)-relaxed canonical LP with terminal clearance, and a P1 certifier that recomputes conservation/node-balance/exit-bounds/cost, resolves the LP optimum harness-side, and arithmetically verifies emitted LP-duality certificates; two hand-derived anchors incl. one where holding back is strictly optimal)
- [x] Friesz et al. (1993) — *A Variational Inequality Formulation of the Dynamic Network User Equilibrium Problem* (white-box solver) — **shipped** as `vi-due` (adr-022, the simultaneous route-and-departure-time DUE closing the analytical-DTA track — the Friesz VI instantiated with generalized-Vickrey point-queue loading on parallel routes: exact closed form C=(δN+αΣs·f)/Σs with greedy used-set, single-route f=0 reduction certified by the adr-019 BottleneckEvaluator at 1e-13, and a P1 `DUEEvaluator` whose marginal-insertion reference scan catches the all-on-one-route false equilibrium the single-route certifier cannot see)
- [x] Ziliaskopoulos (2000) — *A Linear Programming Model for the Single Destination System Optimum Dynamic Traffic Assignment Problem* (white-box solver) — **shipped** as `lp-so-dta` (adr-021, cell-level SO-DTA with finite storage/spillback — the CTM min-flux relaxed to four linear families over `CellSODTAScenario`; hand-derived diverge/spillback anchor J*=26 via the storage pair lemma, holding-on-the-optimal-face demonstrated, and the corridor LP == the repo's own `CTMLink` loading exactly; P1 `CellSODTAEvaluator` with the adr-020 hardening + LP-duality certificates)

## Dynamic network loading — v2

- [x] Newell (1993) — *A simplified theory of kinematic waves in highway traffic, part I: General theory* (network-loading component) — **shipped** as `newell-3det` (adr-024, the interior minimum-principle three-detector reconstruction — the benchmark's FIRST traffic-state-estimation task: given noisy/partial boundary detector curves, reconstruct the interior cumulative field `N(x,t) = min(N_up(t−x/vf), N_dn(t−(L−x)/w) + κ(L−x))` at a fixed hashed query grid, scored against the harness-regenerated closed-form min. Newell's LOADING content — the minimum principle at the link ENDS — already shipped as `ltm` (adr-016), so this ships the unshipped INTERIOR content, NOT a third `LinkModel`; the oracle and certifier are the paper's minimum principle, the observation dials are the repo's own P3 conventions (the paper contains no estimation numerics), and the clean level is an oracle row (ranking lives on the noisy levels where the naive/isotonic pair discriminates))
- [x] Daganzo (1994) — *The cell transmission model: A dynamic representation of highway traffic consistent with the hydrodynamic theory* (network-loading component) — **shipped** as `ctm` (adr-015, the first DNL `LinkModel` on the dnl-core; Godunov cell dynamics at CFL=1, free-flow translation bit-exact + RH shock/spillback anchors)
- [ ] Daganzo (1995) — *The cell transmission model, part II: Network traffic* (network-loading component)
- [x] Lebacque (1996) — *The Godunov scheme and what it means for first order traffic flow models* (network-loading component) — **shipped** as `godunov` (adr-018, `GodunovLink` + `GreenshieldsFD`; the general-FD Godunov scheme on the first non-triangular FD — the first rarefaction physics in the benchmark, CTM being its triangular special case)
- [x] Yperman (2007) — *The Link Transmission Model for dynamic network loading* (network-loading component) — **shipped** as `ltm` (adr-016, the second DNL `LinkModel`; stateless Newell-Daganzo cumulative-curve method, matches CTM byte-for-byte on aligned grids + runs on non-cell-aligned grids CTM rejects)
- [x] Tampère et al. (2011) — *A generic class of first order node models for dynamic macroscopic simulation of traffic flows* (network-loading component) — **shipped** as `node-model` (adr-017, `TampereNode`; the general merge/diverge solver — oriented-capacity-proportional with FIFO, satisfies node axioms N1–N6 — that unlocks network loading for `ctm`/`ltm`)

## Simulation-based DTA & software — v2 (adapters)

- [ ] Jayakrishnan et al. (1994) — *An evaluation tool for advanced traffic information and management systems in urban networks* (black-box wrapper)
- [ ] Peeta & Mahmassani (1995) — *System optimal and user equilibrium time-dependent traffic assignment in congested networks* (white-box solver)
- [ ] Ben-Akiva et al. (2001) — *Network State Estimation and Prediction for Real-Time Traffic Management* (black-box wrapper)
- [ ] Zhou & Taylor (2014) — *DTALite: A queue-based mesoscopic traffic simulator for fast model evaluation and calibration* (black-box wrapper)
- [ ] Horni et al. (2016) — *The Multi-Agent Transport Simulation MATSim* (black-box wrapper)
- [ ] Lopez et al. (2018) — *Microscopic Traffic Simulation using SUMO* (black-box wrapper)

## Day-to-day dynamics — v2

- [x] Horowitz (1984) — *The stability of stochastic equilibrium in a two-link transportation network* (white-box solver) — **shipped** as `dtd-horowitz` (the perceived-cost-*state* day-to-day model: travelers carry a perceived link-cost vector exponentially smoothed toward the experienced costs `p ← (1−w)p + w·t(v)` and logit-load at it via the pinned Dial-STOCH map, reaching the same logit-SUE fixed point as `sue-msa`/`dtd-swap-sue`; certified by the existing logit-SUE fixed-point residual (ADR-001, no new scenario field); uniquely among the day-to-day models NO damping is added, so above the task-dependent stability threshold `w* ≈ 0.81` the process settles into a period-2 limit cycle instead of converging — the very (in)stability Horowitz set out to study)
- [x] Smith (1984) — *The stability of a dynamic model of traffic assignment — an application of a method of Lyapunov* (white-box solver) — **shipped** as `dtd-swap` (first day-to-day model)
- [x] Cascetta (1989) — *A stochastic process approach to the analysis of temporal dynamics in transportation networks* (white-box solver) — **shipped** as `dtd-stochastic` (the benchmark's first genuinely *stochastic* day-to-day model: a finite-population Markov chain — each day `N_od = max(1, round(population_scale·d_od))` travelers per OD pair draw routes by multinomial sampling from the Dial-STOCH logit fractions at the exponentially smoothed perceived costs `p ← (1−w)p + w·t(v)`, driven by the *realized* daily flow, so "equilibrium" is the chain's stationary distribution, not a fixed point; daily flows keep a persistent `O(1/√N)` variability while the emitted burnt-in time average converges (ergodic theorem) to the stationary mean ≈ logit SUE (Davis & Nihan 1993 large-population limit); certified by the existing logit-SUE fixed-point residual (ADR-001, no new scenario field), which honestly floors at O(finite-population bias + sampling SE); `deterministic=False` routes it onto the existing macrorep stochastic track; the exponential filter is the canonical Cantarella & Cascetta (1995) special case of Cascetta's original m-day moving-average filter — a flagged, documented variant)
- [x] Friesz et al. (1994) — *Day-to-day dynamic network disequilibria and idealized traveler information systems* (white-box solver) — **shipped** as `dtd-friesz` (route-flow-state day-to-day: the state is per-OD route flows evolved by the projected dynamical system `ḣ = P_K(h, −c(h))`, discretized by the Bertsekas & Gafni (1982) projection step `h_{k+1} = P_K(h_k − α c(h_k))`; because `∂Z/∂h_p = c_p` exactly for the Beckmann objective `Z`, this is projected gradient descent on Beckmann in *route* space, projecting the whole route-flow vector against today's frozen costs at once (Jacobi) via an exact Euclidean simplex projection that conserves each OD's demand every day; reaches the identical certified UE as the route-swap `dtd-swap` and the link-based `dtd-link` via the same monotone Beckmann descent, certified by the standard UE relative gap — no new scenario field)
- [x] Cantarella & Cascetta (1995) — *Dynamic processes and equilibrium in transportation networks: towards a unifying theory* (white-box solver) — **shipped** as `dtd-unifying` (the unifying-theory node realized as a per-scenario mode gate: one two-equation process — exponential cost-learning filter `p ← (1−w)p + w·t(v)` plus choice update `v ← v + αₙ(ChoiceLoad(p) − v)` where a fraction `αₙ` of travelers reconsiders at the forecast costs — whose choice map is the all-or-nothing best response on deterministic scenarios (fixed point = Wardrop UE, annealed `α/n` step, standard relative-gap certificate) and the pinned Dial-STOCH logit load on SUE scenarios (fixed point = logit SUE, constant `α`, existing ADR-001 residual — no new scenario field); exact reductions regression-tested to float precision (stochastic `α=1` ≡ `dtd-horowitz`, deterministic `w=1, α=1` ≡ `msa`), and the re-derived joint `(α, w)` flip boundary `(2−w)(2−α) = αw|φ′|` confirms C&C's headline on the anchor: `(1,1)` period-2 limit-cycles while *either* form of inertia — cost memory or choice inertia — restores convergence, reducing at `α=1` to `dtd-horowitz`'s documented `w* ≈ 0.81`)
- [x] He et al. (2010) — *A link-based day-to-day traffic assignment model* (white-box solver) — **shipped** as `dtd-link` (link-flow-state day-to-day: the state is the aggregate link-flow vector, adjusted toward the frozen-cost proximal target projected onto the feasible link polytope; reaches the identical certified UE as the route-swap `dtd-swap` via the same monotone Beckmann descent)
- [x] Smith & Watling (2016) — *A route-swapping dynamical system and Lyapunov function for stochastic user equilibrium* (white-box solver) — **shipped** as `dtd-swap-sue` (logit-SUE sibling of `dtd-swap`: the same proportional route-swap day-to-day dynamics driven by the Fisk-generalized cost `c_k + (1/θ) ln h_k`, so the rest point is the logit stochastic user equilibrium (Fisk 1980), not deterministic UE; certified by the existing logit-SUE fixed-point residual (ADR-001, no new scenario field) with Fisk's SUE convex objective as the monotone day-to-day Lyapunov function)

## ML-based traffic assignment — v1 (baseline wrappers)

- [x] Liu et al. (2023) — *End-to-end learning of user equilibrium with implicit neural networks* (black-box wrapper) — **shipped** as `implicit-ue-nn`, the first **torch** model (optional `[torch]` extra, ADR-025). A lean variant (the TR-C primary is paywalled/unread; formulation cross-verified from the authors' open hEART 2024 paper + two posters): a flow-monotone MLP cost head inside a differentiable logit route-choice fixed-point layer over PathEngine column-generated route sets, trained by an exact IMD/adjoint hypergradient on the synthetic-net family against bfw reference equilibria. Its emission `v = Δᵀh` is demand-feasible **by construction**, so it clears the audit the ridge surrogate is censored by (act two of ADR-006: feasibility is architectural, equilibrium quality is not) — bfw still certifies a better gap at matched budget, and the held-out identifiability caveat is documented honestly. Anchors: A1 Braess identity `(4,2,2,2,4)`/route time 92, A2 IMD hypergradient vs central FD `< 1e-5`, A4 feasible=1 at random θ.
- [x] Rahman & Hasan (2023) — *Data-Driven Traffic Assignment: A Novel Approach for Learning Traffic Flow Patterns Using Graph Convolutional Neural Network* (black-box wrapper) — **shipped in v1** (learned-model wrapper + a ridge reference surrogate (not the GCN itself) certified by P1 (learned-surrogate, ADR-006))
- [ ] Liu & Meidani (2024) — *End-to-end heterogeneous graph neural networks for traffic assignment* (black-box wrapper)
- [ ] Xu et al. (2024) — *A unified dataset for the city-scale traffic assignment model in 20 U.S. cities* (data/scenario)

## Data, estimation & benchmarking — v1 (T2 estimation track)

- [x] Van Zuylen & Willumsen (1980) — *The most likely trip matrix estimated from traffic counts* (white-box solver) — **shipped in v1** (T2 entropy estimator (vzw-entropy, ADR-002))
- [x] Cascetta (1984) — *Estimation of trip matrices from traffic counts and survey data: A generalized least squares estimator* (white-box solver) — **shipped in v1** (T2 GLS estimator (gls, ADR-002))
- [x] Spiess (1990) — *A gradient approach for the O-D matrix adjustment problem* (white-box solver) — **shipped in v1** (T2 gradient OD adjustment (spiess, ADR-002))
- [x] Spall (1992) — *Multivariate stochastic approximation using a simultaneous perturbation gradient approximation* (white-box solver) — **shipped in v1** (T2 SPSA calibration baseline (spsa, ADR-002))
- [x] Yang et al. (1992) — *Estimation of origin-destination matrices from link traffic counts on congested networks* (white-box solver) — **shipped** as `od-congested`
- [x] Cascetta et al. (1993) — *Dynamic Estimators of Origin-Destination Matrices Using Traffic Counts* (white-box solver) — **shipped** as `od-dynamic-sim` / `od-dynamic-seq` (adr-023, the within-day time-sliced OD estimator — the third leg of the T2 temporal triangle, distinct from `gls` (time = replication) and `od-kalman` (time = day-to-day noise): the estimand is the `(H, Z, Z)` departure-slice profile, recovered from time-sliced link counts linked by a frozen **exogenous** free-flow two-interval-split lag map `M[l]` (congestion feedback out of scope, `cascetta2001fixed`), via the paper's SIMULTANEOUS (all slices jointly, efficient) and SEQUENTIAL (slice-by-slice, earlier estimates frozen, online-capable but provably less efficient) GLS pair; a bfw-free EXACT linear certifier regenerates the full-network map from the hashed recipe and scores per-interval obs/held-out count RMSE (ranking = `heldout_count_rmse`) with descriptive OD/profile columns and an exact stacked-map identifiability report whose new edges are horizon truncation and cross-slice temporal confounding — the latter a genuinely new false-accept surface because held-out sensors share the lag structure; five hand-derived anchors including the fractional-lag instance where simultaneous strictly dominates sequential `(128/35, 142/35)` vs `(16/5, 94/25)` against truth `(4,6)` and a mean-collapse witness proving distinctness from `gls`; additive, golden Braess hash `cf00f411…` byte-identical)
- [ ] Balakrishna et al. (2007) — *Offline calibration of dynamic traffic assignment: Simultaneous demand-and-supply estimation* (black-box wrapper)
- [x] Stabler et al. (2016) — *Transportation Networks for Research* (data/scenario) — **shipped in v0.x** (checksummed TNTP fetcher + 4 registered networks)
- [ ] Eckman et al. (2023) — *SimOpt: A testbed for simulation-optimization experiments* (metric/protocol)
- [ ] Ryu et al. (2025) — *BO4Mob: Bayesian Optimization Benchmarks for High-Dimensional Urban Mobility Problem* (data/scenario)

---
*Generated from the verified canon `references.json` by
`tools/generate_references.py`; regenerate rather than hand-edit.*
