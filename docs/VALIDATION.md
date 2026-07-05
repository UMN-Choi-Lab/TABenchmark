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
| T2 estimators (`vzw-entropy`, `gls`, `spiess`, `spsa`) | Van Zuylen & Willumsen 1980; Cascetta 1984; Spiess 1990; Spall 1992 | OD estimation is underdetermined — no reproducible standard flow | planted-matrix recovery on synthetic counts + held-out sensor fit (`test_estimation.py`, ADR-002) |

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
- `algb`, `tapas`, and `fw-elastic` cite **paywalled primaries that were not read**; their
  formulas were cross-verified against open secondary sources (Boyles TNA, TAsK, iTAPAS,
  Sheffi 1985) — see each model's module docstring and the relevant ADR.
- Iteration counts are **BLAS-sensitive** (they vary across platforms and Python builds), so
  only *rankings* and *order-of-magnitude* ratios are pinned, never exact counts.
