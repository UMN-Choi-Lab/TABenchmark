# ADR-038 — Cumulative-logit day-to-day dynamics: boundedly-rational logit choice with an exact-Wardrop-UE limit

**Status:** accepted (shipped in v2)
**File:** `docs/design/adr-038-dtd-cumlog.md`

## Context

Every logit-choice day-to-day model shipped so far rests at **stochastic** user equilibrium:
`dtd-horowitz` (perceived-cost smoothing + logit → the Dial-STOCH fixed point), `dtd-stochastic`
(Cascetta's finite-population Markov chain → a stationary distribution ≈ SUE), and `dtd-swap-sue`
(Smith–Watling swap on the Fisk cost → the path-flow logit SUE). Every day-to-day model whose
limit **is** deterministic Wardrop UE — `dtd-swap`, `dtd-friesz`, `dtd-link`, `dtd-unifying`'s
deterministic branch — reaches it with a **perfectly rational** adjustment direction (a proportional
swap, a projected gradient, a link projection, an all-or-nothing best response). The shipped matrix
(choice map × limit point) therefore has an **empty cell**: a *boundedly-rational logit* choice map
whose limit is *exact deterministic Wardrop UE at a finite* `r`.

Li, Wang & Nie (2024) fill it. Their **cumulative logit (CumLog)** model keeps the classical
day-to-day skeleton — a per-OD route-**valuation** vector `s` mapped to choice probabilities by the
logit model `p_k = exp(−r s_k) / Σ_{k'} exp(−r s_{k'})` (Eq. 2) — but makes one change: the
experienced route cost `c(p) = Lᵀ u(v)` is **accumulated** into the valuation,

```
s_t = s_{t-1} + η_t · c(p_{t-1})        (Eq. 6, CumLog)
```

rather than **averaged** as in the classical successive-average (SA) scheme

```
s_t = (1 − η_t) · s_{t-1} + η_t · c(p_{t-1})    (Eq. 4, → SUE; Horowitz 1984).
```

The travelers are boundedly rational every day — they assign strictly positive probability to
acceptable suboptimal routes — yet the global limit is exact WE, at a **finite** `r`, with **no**
equilibrium-concept relaxation and **no** indifference band. On the WE support the accumulated
valuation *differences* converge to finite nonzero constants (so equal-cost routes carry unequal
probabilities — resolving Harsanyi's instability, the gap being `−(log p*_k − log p*_{k'})/r`),
while routes no WE strategy uses have valuations that diverge to `+∞`, so CumLog eliminates non-WE
routes even at finite `r` (a logit model on *averaged* costs cannot, unless `r → ∞`; its limit is
SUE).

This is a distinct axis of bounded rationality from `br-ue` (ADR-008 / Mahmassani–Chang 1987).
`br-ue` relaxes the **equilibrium concept** to an ε-indifference-band *set* (a new `br_epsilon`
scenario field, a necessary-only AEC certificate); CumLog keeps point-set WE as the limit and relaxes
the **adjustment process** (imperfect choices along the path). Nie's thesis is the converse of
ADR-008's: WE itself is defensible under bounded rationality. The two are **process-level** vs
**concept-level** bounded rationality — complementary bookends, with no scenario field and no
certificate interaction between them.

## Sourcing

Li, Wang & Nie (2024), *Transportation Science* 58(5):973–994, DOI 10.1287/trsc.2023.0132, was
**READ IN FULL** from the published PDF: the model (Eqs. 2, 6), Assumptions 1–2, Theorem 1 with its
KL-divergence / dual-averaging proof (Lemmas 1–2, Props. 2–3, Remarks 2–4), and all four experiment
sets (Sec. 6) are used verbatim. This is the **strongest sourcing in the day-to-day family** — no
`attributed unread` caveat is needed at all, unlike every other dtd primary (Smith 1984, Horowitz
1984, Cascetta 1989, Cantarella–Cascetta 1995, Friesz 1994, He 2010, Smith–Watling 2016).

Open anchors for citing the equations/theorem/numerics: **arXiv:2304.02500** (v2, 2024-02-06,
econ.TH / cs.GT) and the NSF PAGES accepted manuscript (par.nsf.gov/servlets/purl/10537406). A
pre-ship diligence diff confirmed arXiv v2 states Assumptions 5.1–5.2 and Theorem 5.4 with content
**mathematically identical** to the journal's Assumptions 1–2 and Theorem 1 (same `u` twice-C¹ on
`X`; same PSD-symmetric-parts condition on `∇u` and `(∇u)²`; same conditions (i) `η_t → 0`,
`Σ η_t = ∞` and (ii) `η_t = η < 1/(2rL)`) — the **only** difference is the numbering scheme
(section-based `5.x` on arXiv vs sequential in the journal). There is therefore **no wording drift**
and no NSF-PAGES fallback caveat: the journal primary (read in full) is cited with arXiv v2 as the
open anchor. Canon: `li2024wardrop` (`docs/references.json` / `references.bib`, tier-1, day-to-day;
`primaryClass = {econ.TH}` per the verified arXiv classification).

## Decision 1 — The state, the day map, and the accumulate-vs-average rule

`dtd-cumlog` (`CumLogDTDModel`, paradigm `day_to_day`, deterministic) carries a per-OD route-valuation
vector `s` over `gp`/`dtd-swap`-style column-generated working route sets. Each day it (a) loads the
full OD demand by the logit map `f_k = d_w · softmax(−r s)_k` — demand-feasible every day, node
balance ~ 0, so the emitted link flow is what the harness certifies; (b) prices one batched Dijkstra
(supplying both the new column and `SPTT`); (c) accumulates the experienced route costs,
`s ← s + η_t c(p)`. Valuations are stored **min-normalized per OD** (`s ← s − min(s)`, paper
Variant 1 / Remark 2 — mathematically identical because the logit map depends only on valuation
*differences*, and it keeps the used-route floats finite while dropped-route valuations still
diverge). A newly generated route enters at the per-OD benchmark valuation `min(s)`, so a freshly
cheapest route is immediately competitive.

The **`accumulate` factor** (default `True`) is the one-line switch between Eq. 6 (accumulation, WE
limit) and Eq. 4 (averaging, SUE limit). **`accumulate=False` is a comparison/regression knob for the
Remark 3 contrast, NEVER a shipped SUE mode: `dtd-horowitz` remains the benchmark's logit-SUE
day-to-day row.** The averaging branch does not reach deterministic UE, so its UE relative gap stays
strictly positive.

Two schedule factors follow Theorem 1. `"harmonic"` (default, `η_t = eta0/(t+1)`) satisfies
`η_t → 0`, `Σ η_t = ∞`, so it converges to WE for **any** `r` (condition (i)) — the robust default,
where the SA model's WE coupling is a knife-edge (Sec. 6.2: perturbing either the `η` or `r` exponent
by 0.01 destroys SA convergence). `"constant"` uses `eta0` literally; Theorem 1(ii) gives the
**sufficient** condition `eta0 < 1/(2 r L)`, whose `L` is the demand-scaled Lipschitz constant of the
route-cost map (asymmetric case `L = max_w d_w · H · ||Λ||²`) — **not computed here**, and the paper
never claims divergence above it. The provenance column `eta_heuristic_scale = 1/(2 r · max_a t'_a(v))`
reports only the house step-scale heuristic (`max_a t'_a` — the `dtd-link`/`dtd-friesz`
step-normalization precedent), a **step-scale reference, not a bound in either direction** (see the
Consequences): a genuinely too-large constant step diverges and that divergence is preserved, not
damped, but the heuristic does not locate where. `init_valuation_scale` (default 0.0 = the paper's
deterministic `s0 = 0`) seeds entering routes with `N(0, scale)` valuations from the seeded RNG,
exposing the paper's Sec. 6.4 random-`s0` experiment.

## Decision 2 — No certificate change: the standard UE relative gap

The rest point is deterministic Wardrop UE (Prop. 1's VI `⟨c(p*), p − p*⟩ ≥ 0`), so the scored
quantity is the **existing** UE relative gap `(TSTT − SPTT)/TSTT` recomputed by the harness from the
emitted link flows — identical to `dtd-swap`/`dtd-friesz`/`dtd-link`/`dtd-unifying` (det.), and to the
paper's own convergence measure (its Eq. 11 is the same normalized relative gap against the AON
minimizer). **No new certificate and no new scenario field.** `r`, `eta0`, `eta_schedule`,
`accumulate`, and `init_valuation_scale` are model factors like `dtd-swap`'s `swap_rate`, so the
golden Braess content hash is byte-identical (regression-pinned). The model self-reports the same
relative gap the harness recomputes, so the P1 honesty check passes to float precision. SUE / elastic
/ combined / BR scenarios are refused — the limit is WE regardless of `r`, and in particular
`scenario.sue_theta` (task data) must NEVER be mapped to the exploitation parameter `r`.

## Decision 3 — Headline validation: the accumulation-vs-averaging distinctness gate

Remark 3's central claim — that Eq. 6 converges to WE while the one-line-different Eq. 4 converges to
SUE — is made an **executable fact on a single instance with identical machinery and the same `r`**.
On the two-route anchor (as a deterministic UE task) at `r = 1`:

* `accumulate=True` drives the certified UE relative gap below `1e-6` with `f_A → 2.5` (exact Wardrop
  UE);
* `accumulate=False` rests at `f_A → 2.373888`, matching the analytic **binary-logit SUE** (the brentq
  root of `f_A = D/(1 + exp(r(c_A − c_B)))` at dispersion `r`) to **six digits**, with a UE relative
  gap `> 0.01`.

The only difference is the update rule; the categorically different limit (WE vs SUE) is the paper's
thesis. Supporting validations: exact Braess UE `[4,2,2,2,4]` and the paper's three-parallel-link
example `(2,1,0)`; harmonic r-independence (`r ∈ {1,10,40}` all reach the Braess UE); the constant
schedule converges (small step) or diverges (large step), with the `eta_heuristic_scale` shown to be a
step-scale reference rather than a bound (below); s0-independence (different finite `s0` reach the SAME
unique WE link flow on Braess through DIFFERENT strategy entropies on Sioux Falls — this row's
extension of the paper's 3N4L entropy histogram, Fig. 10, whose Sioux Falls half of Sec. 6.4 reports
used-route counts, Fig. 11); and the valuation-divergence signature (used-route spread stabilizes
finite, dropped-route valuation max diverges).

## Consequences

- **New:** the `dtd-cumlog` model and `tests/test_dtd_cumlog.py`; `tutorials/02-day-to-day/08-dtd-cumlog.ipynb`;
  the `li2024wardrop` canon entry. **Unchanged:** every scenario hash (golden Braess preserved), the
  certificate, and all prior models/tests.
- **Disclosed consequence — the divergence signature is column-generation-scoped.** The paper
  enumerates the *full* route set up front, so a never-shortest non-WE route (e.g. route 3 of the
  three-parallel example, or a Sioux Falls route that is never the strict minimum) has an explicit
  diverging valuation. Under this benchmark's column generation, such a route is simply **never
  generated** (only once-shortest routes enter the working set) — the model still reaches the correct
  WE by *excluding* it, but its valuation is not tracked. The dropped-route-divergence half of the
  signature is therefore observable only where transient shortest-path switching sheds a *generated*
  route (Sioux Falls, where the active-route count descends to the WE support, Fig. 11), NOT on tiny
  fixed networks where every generated route is a WE route. This is the same working-set-vs-full-set
  distinction `dtd-swap-sue` documents (its efficient-set enumeration, ADR-001 lineage); the tests
  scope the signature to Sioux Falls accordingly. The used-route half (finite valuation differences)
  is exhibited everywhere.
- **`eta_heuristic_scale` is a step-scale reference, not a bound in either direction.** Theorem 1(ii)'s
  `eta0 < 1/(2rL)` is *sufficient only*, with `L` the demand-scaled route-cost Lipschitz constant the
  model does not compute; the reported heuristic uses `max_a t'_a` and is therefore flow-independent
  for linear costs. Two executed counterexamples pin the honesty (regression-tested): on demand-6
  Braess a constant `eta0 = 0.08` **above** the heuristic (0.05) still converges to machine precision
  (`~6e-16`), and on `braess_scenario(demand=60)` a constant `eta0 = 0.02` **below** the same 0.05
  heuristic diverges (tail gap `~0.93`). True stability is flow-dependent; the heuristic is a house
  step-scale reference (the `dtd-link`/`dtd-friesz` precedent), never a boundary claim.
- **Harmonic is asymptotic, not unconditionally fast (Theorem 1(i)).** When route costs are large the
  logit saturates while the accumulated valuation differences slowly build up, and the time to
  de-saturate scales like `r · eta0 · cost-scale` **even on tiny networks** (a fuzz found random BPR
  instances still far from UE after 4000 days, and high-cost 4-link nets that stall for 300k days even
  with the full up-front route set) — a transient of the paper's dynamics, not non-convergence, and the
  certified gap stays honest throughout. On high-cost instances pick `eta0 ~ 1/cost-scale` (or the
  constant schedule with a small step).
- **Slow on large networks (acknowledged in the primary).** CumLog uses no higher-order information
  (only route costs), so Sioux Falls convergence is slow; the scaling test uses a constant `η = 1`
  step (the paper's Sec. 4.3 / Sec. 6.1 setting) at `r = 0.25` (a smaller exploitation parameter than
  the paper's `r = 2.5` Sioux Falls runs, de-saturating faster under column generation) and a loose
  Beckmann tolerance (scaling demonstration, not a tight terminal gap).
- **Deferred:** an efficient WE-route-generation scheme (the primary's own stated missing component)
  and the full-route-set divergence signature on small networks.
