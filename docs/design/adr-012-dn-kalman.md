# ADR-012: od-kalman — Davis & Nihan (1993) linear-Gaussian OD estimation from a time series of link counts

**Status:** accepted (implemented)
**Date:** 2026-07-08
**Deciders:** T2 estimation track — adding a time-series / covariance-aware baseline
**File:** `docs/design/adr-012-dn-kalman.md`

## Context

The six shipped T2 estimators (`prior`, `vzw-entropy`, `gls`, `spiess`, `spsa`,
`od-congested`) all collapse the observed link-count series to a single **mean**
vector (`counts.mean(axis=0)`) and fit that. They discard the two things Davis &
Nihan (1993, *Operations Research* 41(1):169-178, `davis1993large`) show carry
demand information: the **cross-link covariance** of the counts and their
**day-to-day autocorrelation**.

Davis & Nihan prove that as the traveler population grows, a general stochastic
assignment process converges to a deterministic mean (the SUE/UE loading, Prop 2)
plus a stationary linear-Gaussian VAR(1)/VARMA(1,1) fluctuation whose covariance
is their eq. (8) (Prop 3). Their stated application (§4) is exactly an estimator:
the VAR process *"admits a state-space description, which can then be combined
with the discrete-time Kalman filter to produce an algorithm for computing the
likelihood function of a time-series of observed link volume counts [...] permits
estimation of the underlying travel demands via prediction-error minimization."*

**A DN estimator cannot be built on the existing observation level.** `LinkCounts`
emits IID *per-link* Poisson counts (diagonal covariance) or exact repeats — no
cross-link and no temporal correlation. A DN estimator whitening a diagonal
covariance **is `gls`** (a rename the repo forbids; cf. the `od-congested`
"not-a-gls-rename" tests). The DN covariance only bites when the counts carry the
multinomial route structure and day-to-day persistence.

## Decision

1. **New observation level `DayToDayCounts`** (`observe/levels.py`, ADR-012)
   generating a benchmark realization of the DN Gaussian limit as a stationary
   VAR(1):
   `x(t) = x_UE + e(t)`, `e(t) = rho·e(t-1) + a(t)`, `a(t) ~ N(0, (1-rho^2) Q)`,
   so `e(t)` has stationary covariance `Q` and the series is **centered on the UE
   loading** `x_UE = P g`. `Q` is the route-exact DN multinomial covariance
   `Q = sum_j (g_j^2/N_j) Delta_j^T (diag p_j - p_j p_j^T) Delta_j`
   (`observe/_dn_process.py`), extracted from the equilibrium MSA/AON route sets
   so its link marginal equals the proportion column and the mean is exactly
   `x_UE`. `N_j = max(1, round(population_scale·g_j))`; the fluctuation vanishes as
   `1/population_scale` (Prop 2 SLLN) and reduces to `noise='none'` in the limit.
   `rho in [0,1)` is the day-to-day persistence dial; `rho=0` gives IID
   (cross-link-correlated only) counts.

2. **UE-centering (the load-bearing modeling choice).** The DN VARMA temporal
   structure is a *logit/SUE* phenomenon (the smooth `q_j(g)` feedback), but the
   T2 certifier pins **deterministic UE (bfw)** and the runner rejects SUE
   instances. Centering the VAR(1) on the UE loading with the DN innovation
   covariance keeps the **pinned-UE certificate valid with no SUE-certifier
   infra**, while preserving the two DN signals the estimator needs (spatial `Q`
   off-diagonals + temporal `rho`). It is honestly a *linearized Model-B process
   around UE* (Prop 3 form), not a re-simulation of the full logit adjustment.

3. **Estimator `od-kalman`** (`estimation/dn_kalman.py`,
   `DavisNihanKalmanEstimator`, paradigm `estimation`). For a **static** OD
   observed through a stationary Gaussian process the prediction-error / Kalman
   estimate is the GLS whitened by the covariance of the count **time-mean** — the
   steady-state Kalman gain for a constant state reduces to this batch BLUE. Two
   non-classical ingredients (`dn_gls_solve`):
   - **spatial:** whiten by the full (non-diagonal) sample covariance of the
     series, not `gls`'s diagonal Poisson `V`;
   - **temporal:** inflate the count-mean covariance by the AR(1) integrated
     autocorrelation `tau = (1+rho_hat)/(1-rho_hat)` (`ar1_tau`) — the
     effective-sample-size correction; `T` correlated observations carry the
     information of `T/tau` independent ones. `tau=1` recovers the naive mean.
   The solve runs the same congested outer fixed point as `gls`/`od-congested`
   (Cascetta & Postorino 2001) with the best-self-obs-RMSE safeguard (ADR-002
   Decision 3).

4. **Certificate (P1) unchanged.** The harness recomputes `od_feasible`,
   `od_rmse`, `obs_count_rmse` from the emitted OD matrix through the pinned bfw
   assignment (`metrics.estimation`); `self_report` is provenance only. No new
   scored key, no scenario-field or `FlowState`/`Trace`/`Evaluator` change — the
   golden Braess content hash `cf00f411…` is byte-identical (re-asserted in
   `tests/test_dn_kalman.py`).

5. **Analytic anchors** (`tests/test_dn_kalman.py`, all recomputed):
   - two-route DN covariance closed form `Var(link 0) = (D^2/N) p_A p_B`, with
     same-route `+` and cross-route `−` correlation (exact on link-disjoint
     routes);
   - single-sensor `dn_gls` closed form
     `g* = (g_pr/w^2 + p·c/s^2)/(1/w^2 + p^2/s^2)`;
   - `tau` recovers `(1+rho)/(1-rho)` on a DN series and `=1` when `rho=0`;
   - recovery of the planted `D` under full sensors at the finite-population floor.

## Alternatives considered

- **Ship on the existing IID Poisson counts (no new observation level):**
  rejected — the DN covariance is then diagonal and the estimator degenerates to
  `gls`; it would be a rename.
- **Full VARMA(1,1) marginal + a real logit/SUE adjustment process:** rejected for
  now — its mean is the SUE, which needs an SUE certifier (the runner rejects SUE
  T2 instances). The VAR(1)-around-UE slice is the minimal faithful realization
  that keeps the pinned-UE certificate; it also makes the pure-AR(1) form correct
  (no MA term to identify), sidestepping the VARMA MA-cancellation subtlety
  (p.175) — the `tau=(1+rho)/(1-rho)` constant is the *asymptotic* integrated
  autocorrelation, applied as `sigma_stat·tau/T` (a conservative finite-`T`
  effective-sample-size approximation). A full VARMA(1,1) + SUE-certified variant
  is a separately-scoped follow-up.
- **A sequential Kalman-filter recursion over the periods:** the batch
  DN-whitened GLS *is* the steady-state Kalman/BLUE for a static OD state and
  returns the same estimate; a sequential filter is more code for an identical
  result, so the established outer-loop/GLS machinery is reused.

## Consequences

The benchmark gains its **first time-series / covariance-aware** OD estimator: one
that exploits the cross-link and day-to-day structure of the counts rather than
their mean alone, with a sound, harness-recomputed certificate and hand-derived
anchors. All changes are additive; the golden Braess hash is provably preserved;
no output contract changed. Follow-ups: the full VARMA(1,1) + SUE-certified
variant, and reporting the DN posterior covariance of the estimate (not just the
point OD).

## Sourcing

Davis & Nihan (1993, *Operations Research* 41(1):169-178, `davis1993large`, JSTOR
stable/171951) is the primary and was **read** (the eq. (8) covariance recursion,
the VAR(1) logit form on p.175, the Prop 2 SUE mean, and the §4 Kalman /
prediction-error program). The route-level innovation covariance
`sum_j n_j [diag q_j - q_j q_j^T]` is COV[x|s] on p.171, written route-exactly;
it was cross-checked against an independent open implementation of the same eq.
(8) covariance. No number from the paper is reproduced — every anchor is a
hand-derived closed form.
