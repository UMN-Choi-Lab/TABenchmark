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
- [ ] Dafermos (1972) — *The Traffic Assignment Problem for Multiclass-User Transportation Networks* (white-box solver)
- [ ] Smith (1979) — *The Existence, Uniqueness and Stability of Traffic Equilibria* (metric/protocol)
- [ ] Dafermos (1980) — *Traffic Equilibrium and Variational Inequalities* (white-box solver)

## Link-based UE algorithms — v0 (FW, MSA) / v0.x (CFW, BFW)

- [x] Frank & Wolfe (1956) — *An Algorithm for Quadratic Programming* (white-box solver) — **shipped in v0** (Frank-Wolfe solver)
- [x] LeBlanc et al. (1975) — *An Efficient Approach to Solving the Road Network Equilibrium Traffic Assignment Problem* (white-box solver) — **shipped in v0** (Frank-Wolfe solver)
- [x] Boyce et al. (2004) — *Convergence of Traffic Assignments: How Much Is Enough?* (metric/protocol) — **shipped in v0.x** (convergence target protocol (Budget.target_relative_gap))
- [x] Mitradjieva & Lindberg (2013) — *The Stiff Is Moving—Conjugate Direction Frank-Wolfe Methods with Applications to Traffic Assignment* (white-box solver) — **shipped in v0.x** (conjugate and bi-conjugate FW solvers)

## Path/bush-based UE algorithms — v1

- [x] Jayakrishnan et al. (1994) — *A faster path-based algorithm for traffic assignment* (white-box solver) — **shipped in v1** (path-based gradient projection solver (gp))
- [ ] Bar-Gera (2002) — *Origin-based algorithm for the traffic assignment problem* (white-box solver)
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
- [ ] Evans (1976) — *Derivation and analysis of some models for combining trip distribution and assignment* (white-box solver)
- [ ] Mahmassani & Chang (1987) — *On Boundedly Rational User Equilibrium in Transportation Systems* (route-choice component)
- [ ] Spiess & Florian (1989) — *Optimal strategies: A new assignment model for transit networks* (white-box solver)
- [ ] Larsson & Patriksson (1995) — *An augmented Lagrangean dual algorithm for link capacity side constrained traffic assignment problems* (white-box solver)

## Analytical DTA — v2

- [ ] Vickrey (1969) — *Congestion Theory and Transport Investment* (white-box solver)
- [ ] Merchant & Nemhauser (1978) — *A Model and an Algorithm for the Dynamic Traffic Assignment Problems* (white-box solver)
- [ ] Friesz et al. (1993) — *A Variational Inequality Formulation of the Dynamic Network User Equilibrium Problem* (white-box solver)
- [ ] Ziliaskopoulos (2000) — *A Linear Programming Model for the Single Destination System Optimum Dynamic Traffic Assignment Problem* (white-box solver)

## Dynamic network loading — v2

- [ ] Newell (1993) — *A simplified theory of kinematic waves in highway traffic, part I: General theory* (network-loading component)
- [ ] Daganzo (1994) — *The cell transmission model: A dynamic representation of highway traffic consistent with the hydrodynamic theory* (network-loading component)
- [ ] Daganzo (1995) — *The cell transmission model, part II: Network traffic* (network-loading component)
- [ ] Lebacque (1996) — *The Godunov scheme and what it means for first order traffic flow models* (network-loading component)
- [ ] Yperman (2007) — *The Link Transmission Model for dynamic network loading* (network-loading component)
- [ ] Tampère et al. (2011) — *A generic class of first order node models for dynamic macroscopic simulation of traffic flows* (network-loading component)

## Simulation-based DTA & software — v2 (adapters)

- [ ] Jayakrishnan et al. (1994) — *An evaluation tool for advanced traffic information and management systems in urban networks* (black-box wrapper)
- [ ] Peeta & Mahmassani (1995) — *System optimal and user equilibrium time-dependent traffic assignment in congested networks* (white-box solver)
- [ ] Ben-Akiva et al. (2001) — *Network State Estimation and Prediction for Real-Time Traffic Management* (black-box wrapper)
- [ ] Zhou & Taylor (2014) — *DTALite: A queue-based mesoscopic traffic simulator for fast model evaluation and calibration* (black-box wrapper)
- [ ] Horni et al. (2016) — *The Multi-Agent Transport Simulation MATSim* (black-box wrapper)
- [ ] Lopez et al. (2018) — *Microscopic Traffic Simulation using SUMO* (black-box wrapper)

## Day-to-day dynamics — v2

- [ ] Horowitz (1984) — *The stability of stochastic equilibrium in a two-link transportation network* (white-box solver)
- [ ] Smith (1984) — *The stability of a dynamic model of traffic assignment — an application of a method of Lyapunov* (white-box solver)
- [ ] Cascetta (1989) — *A stochastic process approach to the analysis of temporal dynamics in transportation networks* (white-box solver)
- [ ] Friesz et al. (1994) — *Day-to-day dynamic network disequilibria and idealized traveler information systems* (white-box solver)
- [ ] Cantarella & Cascetta (1995) — *Dynamic processes and equilibrium in transportation networks: towards a unifying theory* (white-box solver)
- [ ] He et al. (2010) — *A link-based day-to-day traffic assignment model* (white-box solver)
- [ ] Smith & Watling (2016) — *A route-swapping dynamical system and Lyapunov function for stochastic user equilibrium* (white-box solver)

## ML-based traffic assignment — v1 (baseline wrappers)

- [ ] Liu et al. (2023) — *End-to-end learning of user equilibrium with implicit neural networks* (black-box wrapper)
- [ ] Rahman & Hasan (2023) — *Data-Driven Traffic Assignment: A Novel Approach for Learning Traffic Flow Patterns Using Graph Convolutional Neural Network* (black-box wrapper)
- [ ] Liu & Meidani (2024) — *End-to-end heterogeneous graph neural networks for traffic assignment* (black-box wrapper)
- [ ] Xu et al. (2024) — *A unified dataset for the city-scale traffic assignment model in 20 U.S. cities* (data/scenario)

## Data, estimation & benchmarking — v1 (T2 estimation track)

- [x] Van Zuylen & Willumsen (1980) — *The most likely trip matrix estimated from traffic counts* (white-box solver) — **shipped in v1** (T2 entropy estimator (vzw-entropy, ADR-002))
- [x] Cascetta (1984) — *Estimation of trip matrices from traffic counts and survey data: A generalized least squares estimator* (white-box solver) — **shipped in v1** (T2 GLS estimator (gls, ADR-002))
- [x] Spiess (1990) — *A gradient approach for the O-D matrix adjustment problem* (white-box solver) — **shipped in v1** (T2 gradient OD adjustment (spiess, ADR-002))
- [x] Spall (1992) — *Multivariate stochastic approximation using a simultaneous perturbation gradient approximation* (white-box solver) — **shipped in v1** (T2 SPSA calibration baseline (spsa, ADR-002))
- [ ] Yang et al. (1992) — *Estimation of origin-destination matrices from link traffic counts on congested networks* (white-box solver)
- [ ] Cascetta et al. (1993) — *Dynamic Estimators of Origin-Destination Matrices Using Traffic Counts* (white-box solver)
- [ ] Balakrishna et al. (2007) — *Offline calibration of dynamic traffic assignment: Simultaneous demand-and-supply estimation* (black-box wrapper)
- [x] Stabler et al. (2016) — *Transportation Networks for Research* (data/scenario) — **shipped in v0.x** (checksummed TNTP fetcher + 4 registered networks)
- [ ] Eckman et al. (2023) — *SimOpt: A testbed for simulation-optimization experiments* (metric/protocol)
- [ ] Ryu et al. (2025) — *BO4Mob: Bayesian Optimization Benchmarks for High-Dimensional Urban Mobility Problem* (data/scenario)

---
*Generated from the verified canon `references.json` by
`tools/generate_references.py`; regenerate rather than hand-edit.*
