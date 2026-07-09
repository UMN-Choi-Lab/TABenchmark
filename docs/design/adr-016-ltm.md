# ADR-016: ltm — Yperman (2007) link transmission model

**Status:** accepted (implemented)
**Date:** 2026-07-09
**Deciders:** DNL-models track — the second DNL link model (the Newell-Daganzo method)
**File:** `docs/design/adr-016-ltm.md`

## Context

The Link Transmission Model (Yperman 2007) is the second DNL link sprint after
`ctm` (adr-015). It is the Newell-Daganzo cumulative-curve method: instead of
discretising the link interior into cells, it evaluates Newell's two shifted
cumulative curves directly at the link's two ends, trading interior state for a
need to remember the boundary cumulative curves back `L/vf` and `L/w` in time —
which the `LinkModel` base already retains in full.

## Decision

1. **`LTMLink(LinkModel)`** in `src/tabench/dnl/ltm.py`, additive on the frozen
   sending/receiving interface. It is **stateless** beyond the base cumulative
   curves `n_in` (upstream) / `n_out` (downstream): no cells, `_advance_state`
   is the inherited no-op, and it carries no turning logic (node models handle
   junctions). It requires a **finite jam density** (the `kappa·L` receiving
   term) — the mirror image of the point-queue reference's `kappa = inf`.

2. **Newell-Daganzo sending/receiving** (Yperman eq. 4.31/4.35 ≡ Boyles eq.
   9.65/9.67):
   - `sending(k)   = min(N_up(t_{k+1} - L/vf) - N_dn(t_k), q_max·dt)` — byte-
     identical to the point queue's;
   - `receiving(k) = min(N_dn(t_{k+1} - L/w) + kappa·L - N_up(t_k), q_max·dt)` —
     the `kappa·L` storage term is exactly what turns the point queue's
     unconstrained receiving into a finite backward wave.
   The shifted terms use the base class's exact linear `interp_curve`; `n_out[k]`
   / `n_in[k]` are read at grid edges directly. `assert_wave_resolved`
   (`dt <= min(L/vf, L/w)`, already enforced at scenario construction) is both
   the stability and the causality guarantee — the look-ahead never reads a
   future value.

3. **No CFL=1 cell alignment (the LTM advantage).** LTM has no cells, so — unlike
   CTM — a link length need not be an integer multiple of `vf·dt`; LTM runs on
   any wave-resolved grid, including coarser / non-cell-aligned ones CTMLink
   rejects at construction. This grid flexibility is LTM's concrete, testable
   edge over CTM (anchor d).

## On numerical diffusion (honest scope)

Boyles §9.5.4 states the Newell-Daganzo values are *exact*, with no
backward-shock spreading, "which does happen in the cell transmission model."
That advantage is real in principle (LTM never discretises the interior), but on
the small single-shock anchors here **LTM and CTM agree to machine precision** —
CTM's `O((w/vf)^n_cells)` spreading stays below the certificate tolerance at that
scale, so the harness's backward-wave residual (C5) is `~0` for *both*. The
sprint therefore does not assert an "LTM exact / CTM diffuse" gap it cannot
demonstrate; the demonstrable distinction is grid flexibility (anchor d).

## Analytic anchors (hand-derived from the read primaries, machine-verified — `test_dnl_ltm.py`)

- **(a) Free-flow translation:** `L=4`, `vf=w=1`, `kappa=4`, `cap=2`, inflow `1.0`.
  `n_out(t)=n_in(t-4)` exactly, `TSTT=16`, zero delay — bit-identical to CTM (a).
- **(b) Symmetric bottleneck (CTM cross-check):** the `L=4` link feeds a `0.5`
  bottleneck at inflow `1.5`. LTM reproduces CTM's curves **byte-for-byte** —
  `n_in=1.5t`, `n_out=max(0,0.5(t-4))`, storage `3.5·4=14`, RH shock speed `-0.5`.
- **(c) Asymmetric wave (`w<vf`):** `vf=2, w=1, kappa=3, cap=2`, `L=4`, inflow
  `1.0`, `0.5` bottleneck. RH speed `s=(1-0.5)/(0.5-2.5)=-0.25`, shock reaches
  `x=0` at `t=18`; `n_out=max(0,0.5(t-2))`, storage `k_B·L=2.5·4=10` (verified via
  the Yperman receiving recursion).
- **(d) Grid flexibility:** `L=3, vf=2, dt=1` gives `L/vf=1.5` (non-integer) —
  `CTMLink` raises, LTM free-flow-translates by the `1.5` lag exactly.

## Alternatives considered

- **Reusing `PointQueueLink` with a finite-`kappa` FD:** rejected — the point
  queue's `receiving = q_max·dt` (unbounded) has no backward wave; LTM's whole
  content is the finite `kappa·L` receiving. But the shared `sending` confirms the
  "LTM = point queue + finite receiving" framing.
- **Cell-based like CTM:** rejected — the interior-free formulation is LTM's
  point (exactness + grid flexibility), and it reuses the base cumulative curves
  with no new state.

## Consequences

The benchmark gains its second DNL link model and a diffusion-free, grid-flexible
alternative to CTM, sharing the same interface, certifier, node models, and loader.
All changes are additive (a new module + tests + exports), so the 562-test suite,
every road/DNL hash, and the golden Braess content hash are byte-untouched.

## Sourcing

Both primaries are **open and were read**: Yperman (2007) PhD thesis (KU Leuven,
`mech.kuleuven.be` mirror) §4.6 eq. 4.31/4.35, and Boyles/Lownes/Unnikrishnan
*Transportation Network Analysis* Vol. I (2025) §9.5.2 eq. 9.65/9.67 (worked
example Table 9.6). **Sign-convention note:** Yperman typesets the backward term
`+L/w` because his `w` is the *signed* (negative) backward-wave velocity; this
repo (and Boyles) use `wave_speed > 0` as a magnitude, so the equivalent form is
`- L/w` — which this module uses, verified against Boyles' `R(10)=5` example.
