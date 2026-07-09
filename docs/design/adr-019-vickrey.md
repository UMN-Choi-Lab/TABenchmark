# ADR-019: vickrey — Vickrey (1969) single-bottleneck departure-time equilibrium

**Status:** accepted (implemented)
**Date:** 2026-07-09
**Deciders:** analytical-DTA track — the first departure-time-equilibrium model
**File:** `docs/design/adr-019-vickrey.md`

## Context

The analytical-DTA track opens with Vickrey's (1969) single-bottleneck model — the
canonical **departure-time** equilibrium. `N` travelers each choose a departure
time to trade point-queue delay at a bottleneck of capacity `s` against schedule
delay relative to a desired arrival `t*` (early penalty `beta`, late `gamma`,
travel-time value `alpha`, with `0 < beta < alpha`). This is a different paradigm
from the repo's route-choice and link-loading models — there is no network topology,
just six scalars and a departure-rate decision — so it gets its own parallel module,
exactly as `transit/` and `dnl/` did (ADR-014/010).

## Decision

1. **A parallel module `src/tabench/bottleneck/`** touching no road/DNL/transit
   code (the golden Braess hash `cf00f411…` re-asserted in the tests).
   `BottleneckScenario` (`scenario.py`) is the six scalars, frozen and
   content-hashed under a `"tabench-bottleneck-scenario-v1;"` domain prefix.

2. **Closed-form UE and SO** (`solve.py`), each emitting a `BottleneckSchedule` —
   a cumulative departure curve `R(t)` on a time grid, the P1-certifiable artifact
   (analogue of `FlowState`/`TransitStrategy`). The UE queue-builds at
   `r_early = s·alpha/(alpha-beta)` on `[t1, t_n]` then queue-dissipates at
   `r_late = s·alpha/(alpha+gamma)` on `[t_n, t2]`, with `t1 = t* - C*/beta`,
   `t2 = t* + C*/gamma`, `t_n = t* - C*/alpha`, and `C* = beta·gamma/(beta+gamma)·N/s`.
   The SO meters departures uniformly at `s` over the same window (no queue).

3. **P1 certificate** (`metrics/bottleneck_gaps.py`, `BottleneckEvaluator`). From
   `(scenario, emitted R(t))` alone — never the solver's `r_early`/`t1`/`C*`
   provenance — the harness simulates the deterministic point queue
   `n_{k+1} = max(0, n_k + dR_k - s·dt)`, recomputes each used departure time's
   generalized cost `c(t) = alpha·T + beta·[t*-(t+T)]+ + gamma·[(t+T)-t*]+`
   (`T = n/s`), and scores `equilibrium_gap = (max c - min c)/C*` over used
   departure times — `0` iff no traveler can improve by shifting (a user
   equilibrium), positive otherwise. Feasibility gates on conservation
   (`R` ends at `N`), monotonicity, and the scenario hash. Total/expected cost,
   max queue, and total travel delay are recomputed and reported.

## Analytic anchors (machine-verified — `test_bottleneck.py`)

Worked instance `N=6000, s=3000, alpha=1, beta=0.5, gamma=2, t*=9`:
- `C* = 0.8`; window `t1=7.4, t2=9.4` (width `N/s = 2`); peak `t_n=8.2`;
  `r_early=6000, r_late=1000`; max queue `2400`; UE total `4800`, mean `0.8`.
- **The UE certifies `equilibrium_gap = 0`** (every used departure time yields
  exactly `C*`), while the **SO certifies a positive gap** (uniform metering
  spreads schedule delay, so it is *not* a departure-time equilibrium) with no
  queue and total `2400`.
- **Price of anarchy `= 2` for any `beta, gamma`** (`UE_total / SO_total = 2`,
  a general bottleneck result, regression-fuzzed over random penalties — not just
  the symmetric case).
- A schedule perturbed off the equilibrium curve certifies a positive gap;
  non-conserving / non-monotone / wrong-hash schedules are censored.

## Alternatives considered

- **Reusing the DNL point queue (`dnl/_reference.py::PointQueueLink`):** rejected —
  that is a *numerical loading* kernel driven by a grid loop; Vickrey is an
  analytic departure-time *equilibrium* over a decision (the departure rate), with
  a schedule-delay objective the DNL machinery has no notion of. The certifier's
  own tiny point-queue recomputation keeps the module self-contained.
- **Trusting the solver's `C*`/rates:** rejected (P1) — the certifier recomputes
  the queue and costs from the emitted curve, so it certifies *any* departure-time
  schedule (a future heuristic/learned solver included), not just the closed form.

## Adversarial review

The review confirmed the closed form (`c(t) ≡ C*` re-derived algebraically for any
`alpha,beta,gamma`), the PoA-2 result at extremes, the feasibility gates, hash, and
isolation — and caught **two real certifier bugs**, both fixed:

- **CRITICAL false-accept:** the first certifier sampled each step's cost at the
  *start-of-step* queue, so a "burst dump" schedule (all mass at the window
  boundaries) certified `gap ≈ 0` despite a total cost 2.1× the true UE — the
  intra-step congestion was never seen. Fixed by scoring **per traveler**: invert
  *both* the arrival curve `A(t)=R(t)` and the bottleneck-served curve
  `D(t) = min(A, D_prev + s·dt)` at each count level to get each traveler's actual
  departure and exit times (the level-based approach the DNL/transit certifiers use).
- **False-censor:** a fixed `eps = tol·N` "used-step" mask censored legitimate
  schedules on very fine grids / tiny `N`. Removed — level sampling needs no such
  threshold.

Both are regression-pinned (`test_burst_dump_is_not_a_false_equilibrium`,
`test_fine_grid_and_small_n_not_false_censored`).

## Consequences

The benchmark gains its first departure-time equilibrium and the entry point of the
analytical-DTA track (Merchant–Nemhauser, Friesz VI-DUE, Ziliaskopoulos LP to
follow). All changes are additive (a new module + a new certifier + tests), so the
592-test suite, every road/DNL/transit hash, and the golden Braess content hash are
byte-untouched.

## Sourcing

Vickrey (1969) "Congestion Theory and Transport Investment," *AER P&P* 59(2):251-260
(no DOI; JSTOR 1823678) — attributed. The bottleneck equilibrium closed form is
standard textbook; taken from Boyles *Transportation Network Analysis* Ch. 10
(open, the repo's own `tna_ch10_tdsp.md` grounding) and independently re-derived
from the first-order equal-cost conditions and machine-verified against a
discrete-event queue simulation. Arnott–de Palma–Lindsey (1990, *JUE* 27(1)) is the
standard general-`alpha,beta,gamma` reference (attributed, not full-text-read). No
DOIs or page-precise quotes reproduced.
