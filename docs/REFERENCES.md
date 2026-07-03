# The TABenchmark Reference Canon

The models, algorithms, data practices, and protocols that a complete traffic
assignment benchmark must cover — compiled family-by-family across ~50 years of
transportation research and **verified reference-by-reference** against Crossref /
Semantic Scholar (via refcheck). BibTeX for every entry is in
[`references.bib`](references.bib); the machine-readable canon is
[`references.json`](references.json).

**Tiers** — 1: must implement in the benchmark core; 2: should implement as a
variant/extension; 3: background, survey, or book (cite, don't implement).
**Verified** — ✓: metadata verified against publication databases; *book*:
hand-checked @book entry; *partial*: verified after correcting the originally
compiled metadata (corrections are recorded in the entry's `contribution` field
in `references.json`).

**172 references** — 57 tier-1, 80 tier-2, 35 tier-3.

## Foundations of Static Equilibrium Assignment

| Reference | Title | Venue | Tier | Role | Verified |
|---|---|---|---|---|---|
| Wardrop (1952) | [Some Theoretical Aspects of Road Traffic Research](https://doi.org/10.1680/ipeds.1952.11259) | Proceedings of the Institution of Civil Engineers, Part II | 1 | metric/protocol | ✓ |
| Beckmann et al. (1956) | Studies in the Economics of Transportation | Yale University Press | 1 | white-box solver | book |
| Bureau of Public Roads (1964) | Traffic Assignment Manual | U.S. Department of Commerce, Urban Planning Division | 1 | network loading | book |
| Dafermos (1972) | [The Traffic Assignment Problem for Multiclass-User Transportation Networks](https://doi.org/10.1287/trsc.6.1.73) | Transportation Science | 1 | white-box solver | ✓ |
| Smith (1979) | [The Existence, Uniqueness and Stability of Traffic Equilibria](https://doi.org/10.1016/0191-2615(79)90022-5) | Transportation Research Part B: Methodological | 1 | metric/protocol | ✓ |
| Dafermos (1980) | [Traffic Equilibrium and Variational Inequalities](https://doi.org/10.1287/trsc.14.1.42) | Transportation Science | 1 | white-box solver | ✓ |
| Braess (1968) | [Über ein Paradoxon aus der Verkehrsplanung](https://doi.org/10.1007/BF01918335) | Unternehmensforschung | 2 | data/scenario | ✓ |
| Dafermos & Sparrow (1969) | [The Traffic Assignment Problem for a General Network](https://doi.org/10.6028/jres.073b.010) | Journal of Research of the National Bureau of Standards, Series B | 2 | white-box solver | ✓ |
| Aashtiani & Magnanti (1981) | [Equilibria on a Congested Transportation Network](https://doi.org/10.1137/0602024) | SIAM Journal on Algebraic and Discrete Methods | 2 | white-box solver | ✓ |
| Hearn (1982) | [The Gap Function of a Convex Program](https://doi.org/10.1016/0167-6377(82)90049-9) | Operations Research Letters | 2 | metric/protocol | ✓ |
| Rossi et al. (1989) | [Entropy Model for Consistent Impact-Fee Assessment](https://doi.org/10.1061/(ASCE)0733-9488(1989)115:2(51)) | Journal of Urban Planning and Development | 2 | route choice | ✓ |
| Bar-Gera (2006) | [Primal Method for Determining the Most Likely Route Flows in Large Road Networks](https://doi.org/10.1287/trsc.1050.0142) | Transportation Science | 2 | route choice | ✓ |
| Sheffi (1985) | Urban Transportation Networks: Equilibrium Analysis with Mathematical Programming Methods | Prentice-Hall | 3 | white-box solver | book |
| Patriksson (1994) | The Traffic Assignment Problem: Models and Methods | VSP (republished by Dover, 2015) | 3 | survey/context | book |
| Boyce et al. (2005) | [A Retrospective on Beckmann, McGuire and Winsten's Studies in the Economics of Transportation](https://doi.org/10.1111/j.1435-5957.2005.00005.x) | Papers in Regional Science | 3 | survey/context | ✓ |

## Link-Based UE Algorithms

| Reference | Title | Venue | Tier | Role | Verified |
|---|---|---|---|---|---|
| Frank & Wolfe (1956) | [An Algorithm for Quadratic Programming](https://doi.org/10.1002/nav.3800030109) | Naval Research Logistics Quarterly | 1 | white-box solver | ✓ |
| LeBlanc et al. (1975) | [An Efficient Approach to Solving the Road Network Equilibrium Traffic Assignment Problem](https://doi.org/10.1016/0041-1647(75)90030-1) | Transportation Research | 1 | white-box solver | ✓ |
| Boyce et al. (2004) | [Convergence of Traffic Assignments: How Much Is Enough?](https://doi.org/10.1061/(ASCE)0733-947X(2004)130:1(49)) | Journal of Transportation Engineering | 1 | metric/protocol | ✓ |
| Mitradjieva & Lindberg (2013) | [The Stiff Is Moving—Conjugate Direction Frank-Wolfe Methods with Applications to Traffic Assignment](https://doi.org/10.1287/trsc.1120.0409) | Transportation Science | 1 | white-box solver | ✓ |
| Fukushima (1984) | [A Modified Frank-Wolfe Algorithm for Solving the Traffic Assignment Problem](https://doi.org/10.1016/0191-2615(84)90029-8) | Transportation Research Part B: Methodological | 2 | white-box solver | ✓ |
| Lawphongpanich & Hearn (1984) | [Simplicial Decomposition of the Asymmetric Traffic Assignment Problem](https://doi.org/10.1016/0191-2615(84)90026-2) | Transportation Research Part B: Methodological | 2 | white-box solver | ✓ |
| LeBlanc et al. (1985) | [Improved Efficiency of the Frank-Wolfe Algorithm for Convex Network Programs](https://doi.org/10.1287/trsc.19.4.445) | Transportation Science | 2 | white-box solver | ✓ |
| Florian et al. (1987) | [An Efficient Implementation of the "PARTAN" Variant of the Linear Approximation Method for the Network Equilibrium Problem](https://doi.org/10.1002/net.3230170307) | Networks | 2 | white-box solver | ✓ |
| Hearn et al. (1987) | [Restricted Simplicial Decomposition: Computation and Extensions](https://doi.org/10.1007/BFb0121181) | Mathematical Programming Study | 2 | white-box solver | ✓ |
| Perederieieva et al. (2015) | [A Framework for and Empirical Study of Algorithms for Traffic Assignment](https://doi.org/10.1016/j.cor.2014.08.024) | Computers & Operations Research | 3 | metric/protocol | ✓ |

## Path-Based and Bush/Origin-Based UE Algorithms

| Reference | Title | Venue | Tier | Role | Verified |
|---|---|---|---|---|---|
| Jayakrishnan et al. (1994) | A faster path-based algorithm for traffic assignment | Transportation Research Record 1443 | 1 | white-box solver | ✓ (manual) |
| Bar-Gera (2002) | [Origin-based algorithm for the traffic assignment problem](https://doi.org/10.1287/trsc.36.4.398.549) | Transportation Science 36(4) | 1 | white-box solver | ✓ |
| Dial (2006) | [A path-based user-equilibrium traffic assignment algorithm that obviates path storage and enumeration](https://doi.org/10.1016/j.trb.2006.02.008) | Transportation Research Part B: Methodological 40(10) | 1 | white-box solver | ✓ |
| Bar-Gera (2010) | [Traffic assignment by paired alternative segments](https://doi.org/10.1016/j.trb.2009.11.004) | Transportation Research Part B: Methodological 44(8-9) | 1 | white-box solver | ✓ |
| Larsson & Patriksson (1992) | [Simplicial decomposition with disaggregated representation for the traffic assignment problem](https://doi.org/10.1287/trsc.26.1.4) | Transportation Science 26(1) | 2 | white-box solver | ✓ |
| Florian et al. (2009) | [A New Look at Projected Gradient Method for Equilibrium Assignment](https://doi.org/10.3141/2090-02) | Transportation Research Record 2090 | 2 | white-box solver | ✓ |
| Nie (2010) | [A class of bush-based algorithms for the traffic assignment problem](https://doi.org/10.1016/j.trb.2009.06.005) | Transportation Research Part B: Methodological 44(1) | 2 | white-box solver | ✓ |
| Gentile (2014) | [Local User Cost Equilibrium: a bush-based algorithm for traffic assignment](https://doi.org/10.1080/18128602.2012.691911) | Transportmetrica A: Transport Science 10(1) | 2 | white-box solver | ✓ |
| Xie & Xie (2016) | [New insights and improvements of using paired alternative segments for traffic assignment](https://doi.org/10.1016/j.trb.2016.08.009) | Transportation Research Part B: Methodological 93 | 2 | white-box solver | ✓ |
| Xie et al. (2018) | [A Greedy Path-Based Algorithm for Traffic Assignment](https://doi.org/10.1177/0361198118774236) | Transportation Research Record 2672(48) | 2 | white-box solver | ✓ |
| Bertsekas & Gafni (1982) | [Projection methods for variational inequalities with application to the traffic assignment problem](https://doi.org/10.1007/BFb0120965) | Mathematical Programming Study 17 | 3 | survey/context | ✓ |
| Xie & Xie (2015) | [Origin-Based Algorithms for Traffic Assignment: Algorithmic Structure, Complexity Analysis, and Convergence Performance](https://doi.org/10.3141/2498-06) | Transportation Research Record 2498 | 3 | metric/protocol | ✓ |
| Boyles et al. (2023) | Transportation Network Analysis, Volume I: Static and Dynamic Traffic Assignment | Open-access textbook (self-published, sboyles.github.io) | 3 | survey/context | book |

## Stochastic User Equilibrium and Route Choice

| Reference | Title | Venue | Tier | Role | Verified |
|---|---|---|---|---|---|
| Dial (1971) | [A probabilistic multipath traffic assignment model which obviates path enumeration](https://doi.org/10.1016/0041-1647(71)90012-8) | Transportation Research | 1 | network loading | ✓ |
| Daganzo & Sheffi (1977) | [On stochastic models of traffic assignment](https://doi.org/10.1287/trsc.11.3.253) | Transportation Science | 1 | white-box solver | ✓ |
| Fisk (1980) | [Some developments in equilibrium traffic assignment](https://doi.org/10.1016/0191-2615(80)90004-1) | Transportation Research Part B: Methodological | 1 | white-box solver | ✓ |
| Powell & Sheffi (1982) | [The convergence of equilibrium algorithms with predetermined step sizes](https://doi.org/10.1287/trsc.16.1.45) | Transportation Science | 1 | white-box solver | ✓ |
| Bell (1995) | [Alternatives to Dial's logit assignment algorithm](https://doi.org/10.1016/0191-2615(95)00005-X) | Transportation Research Part B: Methodological | 2 | network loading | ✓ |
| Cascetta et al. (1996) | A modified logit route choice model overcoming path overlapping problems: specification and some calibration results for interurban networks | Proceedings of the 13th International Symposium on Transportation and Traffic Theory (ISTTT), Lyon (Pergamon) | 2 | route choice | book |
| Maher (1998) | [Algorithms for logit-based stochastic user equilibrium assignment](https://doi.org/10.1016/S0191-2615(98)00015-0) | Transportation Research Part B: Methodological | 2 | white-box solver | ✓ |
| Vovsha & Bekhor (1998) | [Link-Nested Logit Model of Route Choice: Overcoming Route Overlapping Problem](https://doi.org/10.3141/1645-17) | Transportation Research Record | 2 | route choice | ✓ |
| Ben-Akiva & Bierlaire (1999) | [Discrete choice methods and their applications to short term travel decisions](https://doi.org/10.1007/978-1-4615-5203-1_2) | Handbook of Transportation Science (Kluwer Academic Publishers) | 2 | route choice | ✓ |
| Bekhor & Toledo (2005) | [Investigating path-based solution algorithms to the stochastic user equilibrium problem](https://doi.org/10.1016/S0191-2615(04)00049-9) | Transportation Research Part B: Methodological | 2 | white-box solver | ✓ |
| Bekhor et al. (2006) | [Evaluation of choice set generation algorithms for route choice models](https://doi.org/10.1007/s10479-006-0009-8) | Annals of Operations Research | 2 | route choice | ✓ |
| Fosgerau et al. (2013) | [A link based network route choice model with unrestricted choice set](https://doi.org/10.1016/j.trb.2013.07.012) | Transportation Research Part B: Methodological | 2 | route choice | ✓ |
| Kitthamkesorn & Chen (2013) | [A path-size weibit stochastic user equilibrium model](https://doi.org/10.1016/j.trb.2013.06.001) | Transportation Research Part B: Methodological | 2 | white-box solver | ✓ |
| Rasmussen et al. (2015) | [Stochastic user equilibrium with equilibrated choice sets: Part II – Solving the restricted SUE for the logit family](https://doi.org/10.1016/j.trb.2015.03.009) | Transportation Research Part B: Methodological | 2 | white-box solver | ✓ |
| Prashker & Bekhor (2004) | [Route choice models used in the stochastic user equilibrium problem: a review](https://doi.org/10.1080/0144164042000181707) | Transport Reviews | 3 | survey/context | ✓ |

## System Optimum, Congestion Pricing, and Efficiency

| Reference | Title | Venue | Tier | Role | Verified |
|---|---|---|---|---|---|
| Yang & Huang (1998) | [Principle of Marginal-Cost Pricing: How Does It Work in a General Road Network?](https://doi.org/10.1016/S0965-8564(97)00018-9) | Transportation Research Part A: Policy and Practice | 1 | white-box solver | ✓ |
| Roughgarden & Tardos (2002) | [How Bad Is Selfish Routing?](https://doi.org/10.1145/506147.506153) | Journal of the ACM | 1 | metric/protocol | ✓ |
| Verhoef et al. (1996) | [Second-Best Congestion Pricing: The Case of an Untolled Alternative](https://doi.org/10.1006/juec.1996.0033) | Journal of Urban Economics | 2 | data/scenario | ✓ |
| Hearn & Ramana (1998) | [Solving Congestion Toll Pricing Models](https://doi.org/10.1007/978-1-4615-5757-9_6) | Equilibrium and Advanced Transportation Modelling (Marcotte & Nguyen, eds.), Kluwer | 2 | white-box solver | ✓ |
| Verhoef (2002) | [Second-Best Congestion Pricing in General Networks: Heuristic Algorithms for Finding Second-Best Optimal Toll Levels and Toll Points](https://doi.org/10.1016/S0191-2615(01)00025-X) | Transportation Research Part B: Methodological | 2 | white-box solver | ✓ |
| Correa et al. (2004) | [Selfish Routing in Capacitated Networks](https://doi.org/10.1287/moor.1040.0098) | Mathematics of Operations Research | 2 | metric/protocol | ✓ |
| Lawphongpanich & Hearn (2004) | [An MPEC Approach to Second-Best Toll Pricing](https://doi.org/10.1007/s10107-004-0536-5) | Mathematical Programming, Series B | 2 | white-box solver | ✓ |
| Youn et al. (2008) | [Price of Anarchy in Transportation Networks: Efficiency and Optimality Control](https://doi.org/10.1103/PhysRevLett.101.128701) | Physical Review Letters | 2 | metric/protocol | ✓ |
| O'Hare et al. (2016) | [Mechanisms That Govern How the Price of Anarchy Varies with Travel Demand](https://doi.org/10.1016/j.trb.2015.12.005) | Transportation Research Part B: Methodological | 2 | metric/protocol | ✓ |
| Pigou (1920) | The Economics of Welfare | Macmillan (London) | 3 | survey/context | book |
| Knight (1924) | [Some Fallacies in the Interpretation of Social Cost](https://doi.org/10.2307/1884592) | Quarterly Journal of Economics | 3 | survey/context | ✓ |
| Walters (1961) | [The Theory and Measurement of Private and Social Cost of Highway Congestion](https://doi.org/10.2307/1911814) | Econometrica | 3 | survey/context | ✓ |
| Braess et al. (2005) | [On a Paradox of Traffic Planning](https://doi.org/10.1287/trsc.1050.0127) | Transportation Science | 3 | data/scenario | ✓ |
| Roughgarden (2005) | Selfish Routing and the Price of Anarchy | MIT Press | 3 | survey/context | book |
| Yang & Huang (2005) | [Mathematical and Economic Theory of Road Pricing](https://doi.org/10.1108/9780080456713) | Elsevier | 3 | survey/context | ✓ |

## Static Assignment Extensions

| Reference | Title | Venue | Tier | Role | Verified |
|---|---|---|---|---|---|
| Florian & Nguyen (1974) | [A Method for Computing Network Equilibrium with Elastic Demands](https://doi.org/10.1287/trsc.8.4.321) | Transportation Science | 1 | white-box solver | ✓ |
| Evans (1976) | [Derivation and analysis of some models for combining trip distribution and assignment](https://doi.org/10.1016/0041-1647(76)90100-3) | Transportation Research | 1 | white-box solver | ✓ |
| Mahmassani & Chang (1987) | [On Boundedly Rational User Equilibrium in Transportation Systems](https://doi.org/10.1287/trsc.21.2.89) | Transportation Science | 1 | route choice | ✓ |
| Larsson & Patriksson (1995) | [An augmented Lagrangean dual algorithm for link capacity side constrained traffic assignment problems](https://doi.org/10.1016/0191-2615(95)00016-7) | Transportation Research Part B: Methodological | 1 | white-box solver | ✓ |
| Gartner (1980) | [Optimal Traffic Assignment with Elastic Demands: A Review Part I. Analysis Framework](https://doi.org/10.1287/trsc.14.2.174) | Transportation Science | 2 | white-box solver | ✓ |
| Florian & Spiess (1982) | [The convergence of diagonalization algorithms for asymmetric network equilibrium problems](https://doi.org/10.1016/0191-2615(82)90007-8) | Transportation Research Part B: Methodological | 2 | white-box solver | ✓ |
| Larsson & Patriksson (1999) | [Side constrained traffic equilibrium models — analysis, computation and applications](https://doi.org/10.1016/S0191-2615(98)00024-1) | Transportation Research Part B: Methodological | 2 | white-box solver | ✓ |
| Boyce & Bar-Gera (2004) | [Multiclass Combined Models for Urban Travel Forecasting](https://doi.org/10.1023/B:NETS.0000015659.39216.83) | Networks and Spatial Economics | 2 | white-box solver | ✓ |
| Lou et al. (2010) | [Robust congestion pricing under boundedly rational user equilibrium](https://doi.org/10.1016/j.trb.2009.06.004) | Transportation Research Part B: Methodological | 2 | white-box solver | ✓ |
| Di et al. (2013) | [Boundedly rational user equilibria (BRUE): Mathematical formulation and solution sets](https://doi.org/10.1016/j.trb.2013.06.008) | Transportation Research Part B: Methodological | 2 | white-box solver | ✓ |
| Nagurney (1993) | [Network Economics: A Variational Inequality Approach](https://doi.org/10.1007/978-94-011-2178-1) | Kluwer Academic Publishers | 3 | survey/context | ✓ |
| Boyce (2007) | [Forecasting Travel on Congested Urban Transportation Networks: Review and Prospects for Network Equilibrium Models](https://doi.org/10.1007/s11067-006-9009-0) | Networks and Spatial Economics | 3 | survey/context | ✓ |
| Di & Liu (2016) | [Boundedly rational route choice behavior: A review of models and methodologies](https://doi.org/10.1016/j.trb.2016.01.002) | Transportation Research Part B: Methodological | 3 | survey/context | ✓ |

## Analytical Dynamic Traffic Assignment

| Reference | Title | Venue | Tier | Role | Verified |
|---|---|---|---|---|---|
| Vickrey (1969) | Congestion Theory and Transport Investment | American Economic Review | 1 | white-box solver | ✓ (manual) |
| Merchant & Nemhauser (1978) | [A Model and an Algorithm for the Dynamic Traffic Assignment Problems](https://doi.org/10.1287/trsc.12.3.183) | Transportation Science | 1 | white-box solver | ✓ |
| Friesz et al. (1993) | [A Variational Inequality Formulation of the Dynamic Network User Equilibrium Problem](https://doi.org/10.1287/opre.41.1.179) | Operations Research | 1 | white-box solver | ✓ |
| Ziliaskopoulos (2000) | [A Linear Programming Model for the Single Destination System Optimum Dynamic Traffic Assignment Problem](https://doi.org/10.1287/trsc.34.1.37.12281) | Transportation Science | 1 | white-box solver | ✓ |
| Merchant & Nemhauser (1978) | [Optimality Conditions for a Dynamic Traffic Assignment Model](https://doi.org/10.1287/trsc.12.3.200) | Transportation Science | 2 | white-box solver | ✓ |
| Carey (1987) | [Optimal Time-Varying Flows on Congested Networks](https://doi.org/10.1287/opre.35.1.58) | Operations Research | 2 | white-box solver | ✓ |
| Arnott et al. (1990) | [Economics of a Bottleneck](https://doi.org/10.1016/0094-1190(90)90028-L) | Journal of Urban Economics | 2 | white-box solver | ✓ |
| Ran et al. (1993) | [A New Class of Instantaneous Dynamic User-Optimal Traffic Assignment Models](https://doi.org/10.1287/opre.41.1.192) | Operations Research | 2 | white-box solver | ✓ |
| Lo & Szeto (2002) | [A Cell-Based Variational Inequality Formulation of the Dynamic User Optimal Assignment Problem](https://doi.org/10.1016/S0191-2615(01)00011-X) | Transportation Research Part B: Methodological | 2 | white-box solver | ✓ |
| Nie (2011) | [A Cell-Based Merchant-Nemhauser Model for the System Optimum Dynamic Traffic Assignment Problem](https://doi.org/10.1016/j.trb.2010.07.001) | Transportation Research Part B: Methodological | 2 | white-box solver | ✓ |
| Ran & Boyce (1996) | [Modeling Dynamic Transportation Networks: An Intelligent Transportation System Oriented Approach](https://doi.org/10.1007/978-3-642-80230-0) | Springer | 3 | survey/context | book |
| Zhu & Marcotte (2000) | [On the Existence of Solutions to the Dynamic User Equilibrium Problem](https://doi.org/10.1287/trsc.34.4.402.12322) | Transportation Science | 3 | survey/context | ✓ |
| Peeta & Ziliaskopoulos (2001) | [Foundations of Dynamic Traffic Assignment: The Past, the Present and the Future](https://doi.org/10.1023/A:1012827724856) | Networks and Spatial Economics | 3 | survey/context | ✓ |
| Han et al. (2013) | [Existence of Simultaneous Route and Departure Choice Dynamic User Equilibrium](https://doi.org/10.1016/j.trb.2013.01.009) | Transportation Research Part B: Methodological | 3 | survey/context | ✓ |
| Wang et al. (2018) | [Dynamic Traffic Assignment: A Review of the Methodological Advances for Environmentally Sustainable Road Transportation Applications](https://doi.org/10.1016/j.trb.2018.03.011) | Transportation Research Part B: Methodological | 3 | survey/context | ✓ |

## Dynamic Network Loading Models

| Reference | Title | Venue | Tier | Role | Verified |
|---|---|---|---|---|---|
| Newell (1993) | [A simplified theory of kinematic waves in highway traffic, part I: General theory](https://doi.org/10.1016/0191-2615(93)90038-C) | Transportation Research Part B: Methodological | 1 | network loading | ✓ |
| Daganzo (1994) | [The cell transmission model: A dynamic representation of highway traffic consistent with the hydrodynamic theory](https://doi.org/10.1016/0191-2615(94)90002-7) | Transportation Research Part B: Methodological | 1 | network loading | ✓ |
| Daganzo (1995) | [The cell transmission model, part II: Network traffic](https://doi.org/10.1016/0191-2615(94)00022-R) | Transportation Research Part B: Methodological | 1 | network loading | ✓ |
| Yperman (2007) | The Link Transmission Model for dynamic network loading | PhD thesis, Katholieke Universiteit Leuven | 1 | network loading | book |
| Tampère et al. (2011) | [A generic class of first order node models for dynamic macroscopic simulation of traffic flows](https://doi.org/10.1016/j.trb.2010.06.004) | Transportation Research Part B: Methodological | 1 | network loading | ✓ |
| Gentile (2010) | [The General Link Transmission Model for Dynamic Network Loading and a Comparison with the DUE Algorithm](https://doi.org/10.4337/9781781000809.00015) | New Developments in Transport Planning: Advances in Dynamic Traffic Assignment (Edward Elgar) | 2 | network loading | ✓ |
| Flötteröd & Rohde (2011) | [Operational macroscopic modeling of complex urban road intersections](https://doi.org/10.1016/j.trb.2011.04.001) | Transportation Research Part B: Methodological | 2 | network loading | ✓ |
| Ban et al. (2012) | [Continuous-time point-queue models in dynamic network loading](https://doi.org/10.1016/j.trb.2011.11.004) | Transportation Research Part B: Methodological | 2 | network loading | ✓ |
| Ma et al. (2014) | [Continuous-time dynamic system optimum for single-destination traffic networks with queue spillbacks](https://doi.org/10.1016/j.trb.2014.06.003) | Transportation Research Part B: Methodological | 2 | network loading | ✓ |
| Lighthill & Whitham (1955) | [On kinematic waves. II. A theory of traffic flow on long crowded roads](https://doi.org/10.1098/rspa.1955.0089) | Proceedings of the Royal Society of London. Series A | 3 | survey/context | ✓ |
| Richards (1956) | [Shock waves on the highway](https://doi.org/10.1287/opre.4.1.42) | Operations Research | 3 | survey/context | ✓ |
| Nie & Zhang (2005) | [A comparative study of some macroscopic link models used in dynamic traffic assignment](https://doi.org/10.1007/s11067-005-6663-6) | Networks and Spatial Economics | 3 | metric/protocol | ✓ |
| van Wageningen-Kessels et al. (2015) | [Genealogy of traffic flow models](https://doi.org/10.1007/s13676-014-0045-5) | EURO Journal on Transportation and Logistics | 3 | survey/context | ✓ |
| Jin (2021) | Introduction to Network Traffic Flow Theory: Principles, Concepts, Models, and Methods | Elsevier | 3 | survey/context | book |

## Simulation-Based DTA and Software Systems

| Reference | Title | Venue | Tier | Role | Verified |
|---|---|---|---|---|---|
| Jayakrishnan et al. (1994) | [An evaluation tool for advanced traffic information and management systems in urban networks](https://doi.org/10.1016/0968-090X(94)90005-1) | Transportation Research Part C | 1 | black-box wrapper | ✓ |
| Ben-Akiva et al. (2001) | [Network State Estimation and Prediction for Real-Time Traffic Management](https://doi.org/10.1023/A:1012883811652) | Networks and Spatial Economics | 1 | black-box wrapper | partial |
| Zhou & Taylor (2014) | [DTALite: A queue-based mesoscopic traffic simulator for fast model evaluation and calibration](https://doi.org/10.1080/23311916.2014.961345) | Cogent Engineering | 1 | black-box wrapper | ✓ |
| Horni et al. (2016) | [The Multi-Agent Transport Simulation MATSim](https://doi.org/10.5334/baw) | Ubiquity Press | 1 | black-box wrapper | ✓ |
| Lopez et al. (2018) | [Microscopic Traffic Simulation using SUMO](https://doi.org/10.1109/ITSC.2018.8569938) | IEEE Intelligent Transportation Systems Conference (ITSC) | 1 | black-box wrapper | ✓ |
| Ziliaskopoulos & Waller (2000) | [An Internet-based geographic information system that integrates data, models and users for transportation applications](https://doi.org/10.1016/S0968-090X(00)00027-9) | Transportation Research Part C | 2 | black-box wrapper | ✓ |
| Mahmassani (2001) | [Dynamic network traffic assignment and simulation methodology for advanced system management applications](https://doi.org/10.1023/A:1012831808926) | Networks and Spatial Economics | 2 | black-box wrapper | ✓ |
| Nagel & Rickert (2001) | [Parallel implementation of the TRANSIMS micro-simulation](https://doi.org/10.1016/S0167-8191(01)00106-5) | Parallel Computing | 2 | black-box wrapper | ✓ |
| Sbayti et al. (2007) | [Efficient Implementation of Method of Successive Averages in Simulation-Based Dynamic Traffic Assignment Models for Large-Scale Network Applications](https://doi.org/10.3141/2029-03) | Transportation Research Record | 2 | route choice | ✓ |
| Casas et al. (2010) | [Traffic Simulation with Aimsun](https://doi.org/10.1007/978-1-4419-6142-6_5) | Fundamentals of Traffic Simulation (Springer, ed. Barceló) | 2 | black-box wrapper | ✓ |
| Osorio & Bierlaire (2013) | [A simulation-based optimization framework for urban transportation problems](https://doi.org/10.1287/opre.2013.1226) | Operations Research | 2 | metric/protocol | ✓ |
| Lu et al. (2015) | [An enhanced SPSA algorithm for the calibration of Dynamic Traffic Assignment models](https://doi.org/10.1016/j.trc.2014.11.006) | Transportation Research Part C | 2 | metric/protocol | ✓ |
| Antoniou et al. (2016) | [Towards a generic benchmarking platform for origin-destination flows estimation/updating algorithms: Design, demonstration and validation](https://doi.org/10.1016/j.trc.2015.08.009) | Transportation Research Part C | 2 | metric/protocol | ✓ |
| Auld et al. (2016) | [POLARIS: Agent-based modeling framework development and implementation for integrated travel demand and network and operations simulations](https://doi.org/10.1016/j.trc.2015.07.017) | Transportation Research Part C | 2 | black-box wrapper | ✓ |
| Barceló (2010) | [Fundamentals of Traffic Simulation](https://doi.org/10.1007/978-1-4419-6142-6) | Springer (International Series in Operations Research & Management Science) | 3 | survey/context | book |
| Chiu et al. (2011) | [Dynamic Traffic Assignment: A Primer](https://doi.org/10.17226/22872) | Transportation Research Circular E-C153, Transportation Research Board | 3 | survey/context | ✓ |

## Day-to-Day Dynamics and Learning Processes

| Reference | Title | Venue | Tier | Role | Verified |
|---|---|---|---|---|---|
| Horowitz (1984) | [The stability of stochastic equilibrium in a two-link transportation network](https://doi.org/10.1016/0191-2615(84)90003-1) | Transportation Research Part B: Methodological | 1 | white-box solver | ✓ |
| Smith (1984) | [The stability of a dynamic model of traffic assignment — an application of a method of Lyapunov](https://doi.org/10.1287/trsc.18.3.245) | Transportation Science | 1 | white-box solver | ✓ |
| Cascetta (1989) | [A stochastic process approach to the analysis of temporal dynamics in transportation networks](https://doi.org/10.1016/0191-2615(89)90019-2) | Transportation Research Part B: Methodological | 1 | white-box solver | ✓ |
| Friesz et al. (1994) | [Day-to-day dynamic network disequilibria and idealized traveler information systems](https://doi.org/10.1287/opre.42.6.1120) | Operations Research | 1 | white-box solver | ✓ |
| Cantarella & Cascetta (1995) | [Dynamic processes and equilibrium in transportation networks: towards a unifying theory](https://doi.org/10.1287/trsc.29.4.305) | Transportation Science | 1 | white-box solver | ✓ |
| He et al. (2010) | [A link-based day-to-day traffic assignment model](https://doi.org/10.1016/j.trb.2009.10.001) | Transportation Research Part B: Methodological | 1 | white-box solver | ✓ |
| Smith & Watling (2016) | [A route-swapping dynamical system and Lyapunov function for stochastic user equilibrium](https://doi.org/10.1016/j.trb.2015.12.015) | Transportation Research Part B: Methodological | 1 | white-box solver | ✓ |
| Davis & Nihan (1993) | [Large population approximations of a general stochastic traffic assignment model](https://doi.org/10.1287/opre.41.1.169) | Operations Research | 2 | white-box solver | ✓ |
| Zhang & Nagurney (1996) | [On the local and global stability of a travel route choice adjustment process](https://doi.org/10.1016/0191-2615(95)00034-8) | Transportation Research Part B: Methodological | 2 | white-box solver | ✓ |
| Watling (1999) | [Stability of the stochastic equilibrium assignment problem: a dynamical systems approach](https://doi.org/10.1016/s0191-2615(98)00033-2) | Transportation Research Part B: Methodological | 2 | white-box solver | ✓ |
| Hazelton & Watling (2004) | [Computation of equilibrium distributions of Markov traffic-assignment models](https://doi.org/10.1287/trsc.1030.0052) | Transportation Science | 2 | metric/protocol | ✓ |
| Bie & Lo (2010) | [Stability and attraction domains of traffic equilibria in a day-to-day dynamical system formulation](https://doi.org/10.1016/j.trb.2009.06.007) | Transportation Research Part B: Methodological | 2 | white-box solver | ✓ |
| Cantarella & Watling (2016) | [Modelling road traffic assignment as a day-to-day dynamic, deterministic process: a unified approach to discrete- and continuous-time models](https://doi.org/10.1007/s13676-014-0073-1) | EURO Journal on Transportation and Logistics | 2 | white-box solver | ✓ |
| Watling & Hazelton (2003) | [The dynamics and equilibria of day-to-day assignment models](https://doi.org/10.1023/a:1025398302560) | Networks and Spatial Economics | 3 | survey/context | ✓ |
| Cantarella et al. (2019) | Dynamics and Stochasticity in Transportation Systems: Tools for Transportation Network Modelling | Elsevier | 3 | survey/context | book |

## Machine-Learning-Based Traffic Assignment

| Reference | Title | Venue | Tier | Role | Verified |
|---|---|---|---|---|---|
| Liu et al. (2023) | [End-to-end learning of user equilibrium with implicit neural networks](https://doi.org/10.1016/j.trc.2023.104085) | Transportation Research Part C: Emerging Technologies | 1 | black-box wrapper | ✓ |
| Rahman & Hasan (2023) | [Data-Driven Traffic Assignment: A Novel Approach for Learning Traffic Flow Patterns Using Graph Convolutional Neural Network](https://doi.org/10.1007/s42421-023-00073-y) | Data Science for Transportation (Springer) | 1 | black-box wrapper | ✓ |
| Liu & Meidani (2024) | [End-to-end heterogeneous graph neural networks for traffic assignment](https://doi.org/10.1016/j.trc.2024.104695) | Transportation Research Part C: Emerging Technologies | 1 | black-box wrapper | ✓ |
| Xu et al. (2024) | [A unified dataset for the city-scale traffic assignment model in 20 U.S. cities](https://doi.org/10.1038/s41597-024-03149-8) | Scientific Data (Nature) | 1 | data/scenario | ✓ |
| Agrawal et al. (2019) | Differentiable Convex Optimization Layers | NeurIPS | 2 | white-box solver | ✓ |
| Shou et al. (2022) | [Multi-agent reinforcement learning for Markov routing games: A new modeling paradigm for dynamic traffic assignment](https://doi.org/10.1016/j.trc.2022.103560) | Transportation Research Part C: Emerging Technologies | 2 | route choice | ✓ |
| Liu & Meidani (2024) | [Heterogeneous Graph Sequence Neural Networks for Dynamic Traffic Assignment](https://doi.org/10.48550/arXiv.2408.04131) | arXiv (2408.04131) | 2 | black-box wrapper | ✓ |
| Jungel et al. (2025) | WardropNet: Traffic Flow Predictions via Equilibrium-Augmented Learning | ICLR | 2 | black-box wrapper | ✓ |
| Lassen et al. (2025) | [Learning traffic flows: Graph Neural Networks for Metamodelling Traffic Assignment](https://doi.org/10.1109/mt-its68460.2025.11223524) | MT-ITS 2025 (IEEE); arXiv:2505.11230 | 2 | black-box wrapper | ✓ |
| Liu & Yin (2025) | [End-to-End Learning of User Equilibrium: Expressivity, Generalization, and Optimization](https://doi.org/10.1287/trsc.2023.0489) | Transportation Science | 2 | metric/protocol | ✓ |
| Wang et al. (2025) | [Scalable and reliable multi-agent reinforcement learning for traffic assignment](https://doi.org/10.1109/itsc60802.2025.11423512) | IEEE ITSC 2025; arXiv:2506.17029 | 2 | route choice | partial |
| Ameli et al. (2026) | [From Optimization to Prediction: Transformer-Based Path-Flow Estimation to the Traffic Assignment Problem](https://doi.org/10.1016/j.trc.2026.105808) | Transportation Research Part C: Emerging Technologies | 2 | black-box wrapper | partial |
| Amos & Kolter (2017) | OptNet: Differentiable Optimization as a Layer in Neural Networks | ICML | 3 | survey/context | ✓ |
| Bai et al. (2019) | Deep Equilibrium Models | NeurIPS | 3 | survey/context | ✓ |
| Xue et al. (2025) | [Data Science in Transportation Networks with Graph Neural Networks: A Review and Outlook](https://doi.org/10.1007/s42421-025-00124-6) | Data Science for Transportation (Springer) | 3 | survey/context | partial |

## Data, OD Estimation, Calibration, and Benchmarking Practice

| Reference | Title | Venue | Tier | Role | Verified |
|---|---|---|---|---|---|
| Van Zuylen & Willumsen (1980) | [The most likely trip matrix estimated from traffic counts](https://doi.org/10.1016/0191-2615(80)90008-9) | Transportation Research Part B: Methodological | 1 | white-box solver | ✓ |
| Cascetta (1984) | [Estimation of trip matrices from traffic counts and survey data: A generalized least squares estimator](https://doi.org/10.1016/0191-2615(84)90012-2) | Transportation Research Part B: Methodological | 1 | white-box solver | ✓ |
| Spiess (1990) | A gradient approach for the O-D matrix adjustment problem | Publication CRT-693, Centre de Recherche sur les Transports, Universite de Montreal | 1 | white-box solver | book |
| Yang et al. (1992) | [Estimation of origin-destination matrices from link traffic counts on congested networks](https://doi.org/10.1016/0191-2615(92)90008-K) | Transportation Research Part B: Methodological | 1 | white-box solver | ✓ |
| Balakrishna et al. (2007) | [Offline calibration of dynamic traffic assignment: Simultaneous demand-and-supply estimation](https://doi.org/10.3141/2003-07) | Transportation Research Record | 1 | black-box wrapper | ✓ |
| Stabler et al. (2016) | Transportation Networks for Research | GitHub repository | 1 | data/scenario | book |
| Eckman et al. (2023) | [SimOpt: A testbed for simulation-optimization experiments](https://doi.org/10.1287/ijoc.2023.1273) | INFORMS Journal on Computing | 1 | metric/protocol | ✓ |
| Ryu et al. (2025) | BO4Mob: Bayesian Optimization Benchmarks for High-Dimensional Urban Mobility Problem | arXiv:2510.18824 (NeurIPS D&B submission) | 1 | data/scenario | ✓ |
| Bell (1991) | [The estimation of origin-destination matrices by constrained generalised least squares](https://doi.org/10.1016/0191-2615(91)90010-G) | Transportation Research Part B: Methodological | 2 | white-box solver | ✓ |
| Vardi (1996) | [Network tomography: Estimating source-destination traffic intensities from link data](https://doi.org/10.1080/01621459.1996.10476697) | Journal of the American Statistical Association | 2 | white-box solver | ✓ |
| Tebaldi & West (1998) | [Bayesian inference on network traffic using link count data](https://doi.org/10.1080/01621459.1998.10473707) | Journal of the American Statistical Association | 2 | white-box solver | ✓ |
| Yang & Zhou (1998) | [Optimal traffic counting locations for origin-destination matrix estimation](https://doi.org/10.1016/S0191-2615(97)00016-7) | Transportation Research Part B: Methodological | 2 | metric/protocol | ✓ |
| Hazelton (2015) | [Network tomography for integer-valued traffic](https://doi.org/10.1214/15-AOAS805) | Annals of Applied Statistics | 2 | white-box solver | ✓ |
| Osorio (2019) | [High-dimensional offline origin-destination (OD) demand calibration for stochastic traffic simulators of large-scale road networks](https://doi.org/10.1016/j.trb.2019.01.005) | Transportation Research Part B: Methodological | 2 | black-box wrapper | ✓ |
| Camargo (2024) | AequilibraE: Transportation modeling in Python | Open-source software | 2 | black-box wrapper | book |
| Cascetta & Nguyen (1988) | [A unified framework for estimating or updating origin/destination matrices from traffic counts](https://doi.org/10.1016/0191-2615(88)90024-0) | Transportation Research Part B: Methodological | 3 | survey/context | ✓ |

---

*Compiled by a 12-family literature sweep with per-reference verification;
0 references failed verification. Cross-family duplicates were merged (a
reference appears in the family where it is most load-bearing;
`references.json` records all its families). Generated by
`tools/generate_references.py` — edit the JSON, not this file.*
