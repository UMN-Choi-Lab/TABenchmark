# Numerical Validation

Every solver in TABenchmark is validated against an **independent oracle**, because
the harness's own gap is not a substitute for checking the *numbers*. This document
records, per model, its source, what that source reports, and how the shipped
implementation is verified. The executable form of this table is
`tests/test_validation.py` (plus the per-model test files); the numbers below are
recomputed there, not trusted from here.

## The oracle hierarchy

In priority order — most models fall back through these because most original papers
report results on private/non-standard instances or 1970s-hardware timings, not
reproducible link flows:

1. **Best-known link flows** (Transportation Networks project, `bstabler/TransportationNetworks`,
   pinned commit `d1639b4e`). UE **link** flows are unique (Beckmann convexity), so every
   correct UE solver must converge to the same published vector. These `*_flow.tntp` files
   are the de-facto community oracle, each solved to a tiny average excess cost (Sioux Falls
   `3.9e-15`, Chicago-Sketch `2.1e-13`, Chicago-Regional `8.3e-12`). **Philadelphia has no
   published flow file — it is not a validation oracle.**
2. **Cross-solver agreement.** Different algorithm families — link-based Frank-Wolfe,
   path-based gradient projection, bush-based Algorithm B, PAS-based TAPAS — must converge
   to the *same* link flows. Verified: on Sioux Falls at gap `1e-9`, `gp`/`algb`/`tapas`
   agree to `max pairwise |v_i − v_j| ≈ 3e-3` (`test_validation.py`).
3. **Analytic small-network anchors** (built in, no download): Braess UE, Braess SO,
   two-route logit/probit SUE, elastic two-route. Exact and hand-checkable.

## Per-model validation

| model | source | what the source reports | how verified here |
|---|---|---|---|
| `fw` | Frank & Wolfe 1956; LeBlanc, Morlok & Pierskalla 1975 | LeBlanc **introduced Sioux Falls** (24-node); only a 1970s-hardware timing, no reproducible flows | Braess UE `(4,2,2,2,4)`; Sioux Falls best-known flows + objective (below) |
| `cfw` / `bfw` | Mitradjieva & Lindberg 2013 (*Transp. Sci.* 47(2), **full text read**) | Table 3 iterations-to-`1e-4`: Sioux Falls **BFW 124 / CFW 357 / FW 1869**; headline "FW ≈ 10× the conjugate variants" | **the ranking** is reproduced (in-repo `bfw 102 / cfw 219 / fw 1053`; `fw > 2× cfw,bfw`). Absolute counts differ (their gap is objective-bound `(UBD−LBD)/LBD`, ours is AEC; line-search/tie-break differ) — a **ranking/order-of-magnitude** pin, not an exact match |
| `gp` | Jayakrishnan et al. 1994 (TRR 1443, **full text read**) | "10–15 GP iterations vs 300–2000 FW"; but on random grids + a **private COMEST Anaheim matrix** (not TNTP demand) | not reproducible; best-known flows + cross-solver agreement. Qualitative "path-based reaches accuracy in far fewer iterations than link-based FW" holds in-repo |
| `algb` | Dial 2006 (*TR-B* 40(10), **paywalled/unread**; formulas cross-verified vs Boyles TNA §6.4, Nie 2010, TAP-B/TAsK) | Chicago-region ~40k arcs; "precision unreachable by FW" — no clean table | best-known flows; reaches tighter gaps than FW at equal budget; cross-solver agreement |
| `tapas` | Bar-Gera 2010 (*TR-B* 44(8-9), **paywalled/unread**; cross-verified vs Boyles TNA §6.5.3, TAsK, iTAPAS) | five networks; tight UE gap **and** entropy-maximizing proportional route flows | best-known flows; cross-solver agreement; proportionality on an analytic PAS (`test_tapas.py`) |
| `so-bfw` | Beckmann, McGuire & Winsten 1956; Yang & Huang 1998 | system optimum = UE of the marginal-cost network | **Braess SO**: flows `(3,3,0,3,3)`, TSTT `498` (route cost 83), vs UE TSTT `552` (route cost 92) ⇒ **Price of Anarchy `552/498 ≈ 1.108`** (exact, `test_validation.py`) |
| `sue-msa` | Dial 1971; Fisk 1980; Powell & Sheffi 1982 | logit SUE convex program + MSA convergence; no cross-implementation-stable flow (θ- and overlap-dependent) | analytic 2-route logit fixed point `f_A = 2.2990959494` at θ=0.5 (recomputed by `brentq`, `test_sue.py`) |
| `sue-probit-msa` | Sheffi & Powell 1982; Daganzo & Sheffi 1977 | probit SUE — **no closed form**; algorithm + MC loading | analytic/MC 2-route probit `f_A ≈ 2.444` at β=0.1, certified by the ADR-003 MC residual (`test_probit.py`) |
| `fw-elastic` | Florian & Nguyen 1974 (**paywalled/unread**); Gartner 1980; Sheffi 1985 ch.6 / Boyles §9.1 | elastic-demand equilibrium method | analytic elastic 2-route: `u=5, f_A=3, f_B=2`, realized demand `5`, flows `(3,3,2,2)` (recomputed by `brentq`, `test_elastic.py`) |
| `learned-surrogate` | Rahman & Hasan 2023 (learned-TA line) | link-flow MAE/correlation (the literature's metric) | not a certification target; the harness recomputes the gap — corr `0.63–0.99` across TNTP yet `feasible=0` on all (ADR-006, `test_learned.py`) |
| `vi-asym` | Dafermos 1980 (*Transp. Sci.* 14(1), **paywalled/unread**); Smith 1979 (*TR-B* 13(4)); cross-verified vs Boyles TNA non-separable-cost chapter | an asymmetric-Jacobian equilibrium minimizes no Beckmann potential, so there is no reproducible standard flow — validated by a hand-derived analytic VI anchor | 2-route asymmetric anchor `f_A* = (1+(1−c₁₃)D)/(2−c₁₃−c₃₁) = 6/1.3 ≈ 4.6154` (D=10, c₁₃=0.5, c₃₁=0.2), a flow distinct from BOTH the plain-UE split `5.5` **and** the symmetrized-Beckmann split `≈5.769` — one no potential solver reaches; VI residual harness-recomputed at the asymmetric cost, reducing exactly to FW UE when `C=0` (ADR-011, `test_vi_asym.py`) |
| `multiclass` | Dafermos 1972 (*Transp. Sci.* 6(1), exact Crossref); cross-verified vs Boyles TNA non-separable-cost chapter + an independent multiclass diagonalization | a multiclass equilibrium is per-class (each class Wardrop-optimal in its coupled cost); asymmetric interaction has no Beckmann potential — validated by hand-derived two-class anchors | two 2-class two-route anchors `[p,q] = [g_cars/2, g_trucks/2] + (a2/4)·M⁻¹[1,1]`: **symmetric/integrable** `M=[[.5,.25],[.25,.5]]` → cars `(2.5,1.5)`/trucks `(1.5,0.5)`, aggregate `(4,4,2,2)`; **asymmetric/genuine-VI** `M=[[.5,.5],[0,.5]]` → cars `(2,2)`/trucks `(1.75,0.25)`, aggregate `(3.75,3.75,2.25,2.25)` — a flow no Beckmann/FW solver reaches, classes routing distinctly; class-summed VI residual harness-recomputed from the emitted per-class flows, per-class conservation audited, aggregate-only flow censored (ADR-013, `test_multiclass.py`) |
| `transit-strategy` | Spiess & Florian 1989 (*Transp. Res. B* 23(2), exact Crossref); common-lines formula hand-derived + internally consistency-checked | uncongested frequency-based optimal-strategy assignment is a convex LP with a closed-form common-lines cost — validated by hand-derived anchors, no external oracle needed | two common-lines anchors: **both attractive** `(f=1/6,t=21),(1/12,18)` → `C* = (1+1/6·21+1/12·18)/(1/4) = 24` min, split 2:1; **threshold** `(1/6,15),(1/12,40)` → line 2 excluded (`40 ≥ 21`), `C* = 21`, all on line 1; a multi-leg interchange (`u = 39`), a deterministic-walk-dominates case, the primal-dual identity `Z_emitted = Z*` exact, a multi-destination shared-node scenario and a near-zero-frequency parasite arc (both adversarial-review CRITICALs, fixed to a per-destination, LP-minimal-wait `w_i = maxₐ vₐ/fₐ` certificate that keeps the gap provably ≥ 0; a non-proportional split is scored as suboptimal, only non-conserving flows are censored); the optimality gap `(Z_emitted − Z*)/Z*` harness-recomputed against the independently-solved LP optimum, in a parallel `transit/` module that leaves the road hashes byte-untouched (ADR-014, `test_transit.py`) |
| T2 estimators (`vzw-entropy`, `gls`, `spiess`, `spsa`, `od-congested`, `od-kalman`) | Van Zuylen & Willumsen 1980; Cascetta 1984; Spiess 1990; Spall 1992; Yang et al. 1992; Davis & Nihan 1993 (**read**, JSTOR stable/171951) | OD estimation is underdetermined — no reproducible standard flow | planted-matrix recovery on synthetic counts + held-out sensor fit; `od-congested` adds a closed-form θ-weighted anchor (GLS with scalar variances `1/θ`, `1/(1−θ)`), prior↔count θ-limit consistency, and congested-fixed-point recovery of the equilibrium-consistent truth; `od-kalman` adds the Davis–Nihan large-population Gaussian limit — a two-route DN cross-link covariance closed form `Var(link) = (D²/N)·p_A·p_B` (same-route `+`, cross-route `−` correlation), the single-sensor DN-GLS closed form `g* = (g_pr/w² + p·c/s²)/(1/w² + p²/s²)`, and the AR(1) effective-sample-size factor `τ = (1+ρ)/(1−ρ)` recovered from the count series (`test_estimation.py`, `test_dn_kalman.py`, ADR-002/ADR-012) |

## The one external objective pin

The Sioux Falls best-known flows reproduce the **published optimal Beckmann objective**
`42.31335287107440` (Transportation Networks README) to full precision, up to the repo's
`1e5` unit factor (free-flow times in `0.01 h`): the recomputed objective is
`4231335.28710744`, i.e. `42.31335287107440 × 1e5` with **relative error `0`**. This
validates both the flows *and* the objective against an external published number, and is
pinned in `test_validation.py`.

## Honesty notes

- The **only** paper whose numerics are reproducible instance-for-instance is Mitradjieva &
  Lindberg 2013, and even there the gap definition differs, so it is a ranking pin. Every
  other UE paper uses non-TNTP/private instances or hardware-bound timings; the best-known
  flow oracle + cross-solver agreement is the correct, honest fallback.
- `algb`, `tapas`, `fw-elastic`, and `vi-asym` cite **paywalled primaries that were not read**;
  their formulas were cross-verified against open secondary sources (Boyles TNA, TAsK, iTAPAS,
  Sheffi 1985) — see each model's module docstring and the relevant ADR.
- Iteration counts are **BLAS-sensitive** (they vary across platforms and Python builds), so
  only *rankings* and *order-of-magnitude* ratios are pinned, never exact counts.
