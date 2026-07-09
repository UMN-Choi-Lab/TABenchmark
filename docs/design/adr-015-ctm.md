# ADR-015: ctm — Daganzo (1994/1995) cell transmission model link

**Status:** accepted (implemented)
**Date:** 2026-07-09
**Deciders:** DNL-models track — the first shipped DNL link model, on the adr-010 core
**File:** `docs/design/adr-015-ctm.md`

## Context

`dnl-core` (adr-010) built the generic sending/receiving foundation "five later
sprints must share without rework" and named `ctm` (Daganzo 1994/1995) first. The
Cell Transmission Model is a finite-cell Godunov discretisation of the LWR
kinematic-wave PDE: the road is cut into equal cells, and the flux between adjacent
cells is `min(upstream sending, downstream receiving)`, which produces shocks,
queues and spillback automatically. It is a **link loading model**, not an
equilibrium principle — the inner map a DTA equilibrium (a later sprint) iterates.

## Decision

1. **`CTMLink(LinkModel)`** in `src/tabench/dnl/ctm.py`, a subclass on the frozen
   `sending`/`receiving`/`_advance_state` interface — no change to `link.py`,
   `loader.py`, `node.py`, or the certifier. At the sanctioned **CFL = 1** operating
   point the cell length is `dx = vf·dt`, so a link of length `L` is `n = L/dx`
   equal cells (a free-flow vehicle crosses exactly one cell per step). Internal
   state is the per-cell occupancy `occ[i]` (vehicles), all zero at `t = 0`.

2. **Lebacque demand/supply flux the FD already exposes.** `fd.demand_at(k)` = Δ
   and `fd.supply_at(k)` = Σ, so the CTM needs no new FD algebra:
   - `sending(k)` = last cell's demand `demand_at(k_last)·dt`, capped by `occ[-1]`;
   - `receiving(k)` = first cell's supply `supply_at(k_first)·dt`;
   - interior Godunov flux `y_i = min(demand_at(k_i), supply_at(k_{i+1}))·dt`
     (`i = 0..n-2`), then the conservation update
     `occ[:-1] -= y; occ[1:] += y; occ[0] += inflow; occ[-1] -= outflow`,
   where `inflow`/`outflow` are the node-allocated boundary transfers. A single
   cell (`L = vf·dt`) reduces to `occ[0] += inflow - outflow`.

3. **No turning logic (division of responsibility).** A `LinkModel` sees only two
   aggregate scalars (`inflow` into cell 0, `outflow` out of the last cell); the
   node models allocate transfers. So the CTM **link** is single-corridor cell
   dynamics only — Daganzo (1995) Part II's *network* merge/diverge extension is
   the node-model sprint's (`daganzo1995cell` stays unshipped until then), exactly
   the adr-010 division.

4. **Finite jam, cell alignment, and the backward-wave CFL enforced.** CTM models
   bounded storage, so `kappa = inf` (point queue) raises — the unbounded point
   queue stays the test-only reference. A non-cell-aligned `L` (not an integer
   multiple of `vf·dt`) raises rather than silently loading off the certificates'
   gating point. And at CFL = 1 the standard stability requirement
   `dt <= dx/max(vf, w)` reduces to **`w <= vf`** (the backward wave must be
   resolved per cell too); `w > vf` — a legal `TriangularFD` — would let the
   congested-branch flux `supply_at(k)·dt = w·(kappa-k)·dt` overfill a cell past
   `kappa·dx`, so it raises. This per-cell condition is invisible to the
   scenario-level `assert_wave_resolved`, which uses the whole-link length `L`
   (`dt <= L/w`), so the guard lives in `CTMLink` — a defect the adversarial
   review caught (66% of `w > vf` configs otherwise produced C3-censored,
   physically impossible output).

## Exactness and the Tier-B backward wave

At CFL = 1 the **free-flow** branch is linear advection at Courant number 1 — the
classical zero-numerical-diffusion case — so `n_out(t) = n_in(t - L/vf)` is
**bit-exact** (anchor a). The **congested** branch runs at Courant number `w/vf < 1`
(the grid is tuned to the faster forward speed), so a backward shock spreads by
O(one cell): standard CTM is provably diffusive for backward waves whenever
`w < vf` (Daganzo's open 1999 ISTTT restatement; Boyles TNA §10.5). This is expected
scheme physics, and is exactly why the harness demotes the backward-wave envelope
(C5) to a **non-gating Tier-B residual** — a correct CTM shows a bounded,
resolution-shrinking `kw_backward_residual` under spillback, not a censoring bug.
The gating certificates (C1 conservation, C2 capacity, C3 storage, C4 causality,
C6 FIFO) all hold.

## Analytic anchors (hand-derived, machine-verified — `test_dnl_ctm.py`)

FD `vf = w = 1, kappa = 4` ⇒ `capacity = vf·w·kappa/(vf+w) = 2`, `k_c = 2`; `dt = 1`.

- **(a) Free-flow translation:** `L = 4`, inflow `1.0 < 2`. `n_out(t) = n_in(t-4)`
  exactly, `TSTT = (L/vf)·D = 4·4 = 16`, zero delay.
- **(b) Queue spillback:** the `L = 4` link feeds a `0.5` bottleneck at inflow `1.5`.
  Rankine–Hugoniot shock speed `s = (q_A-q_B)/(k_A-k_B) = (1.5-0.5)/(1.5-3.5) = -0.5`,
  starting at `x = 4, t = 4` and reaching `x = 0` (full spillback) at `t = 12`;
  exact bottleneck boundary curves `n_in = 1.5t`, `n_out = max(0, 0.5(t-4))`; storage
  at `t = 12` `= k_B·L = 3.5·4 = 14`.
- **(c) Congested density:** every fully-queued cell settles at `k_B = kappa - q_B/w
  = 3.5`, the root of `supply_at(k) = q_B` — the entropy state Lebacque's supply
  selects by construction.

## Alternatives considered

- **Density (not occupancy) state:** equivalent; occupancy keeps the update pure
  counts (no `dx` bookkeeping in the conservation step), matching the node models'
  vehicle-count convention.
- **Trapezoidal-FD rarefaction anchor:** deferred — a strictly triangular FD
  discharges as two clean shocks, not a smooth fan, so the three anchors above
  already cover free-flow, shock, and steady-congested without it.
- **Registering CTM in a road model registry:** rejected — a `LinkModel` is selected
  as the loader's `link_factory`, exercised through `NetworkLoader`, not the road
  CLI/runner (same standalone treatment the DNL core set).

## Consequences

The benchmark gains its first dynamic-loading model and the template the ltm /
newell-kw / godunov sprints follow (`LinkModel` subclasses, same interface). All
changes are additive (a new module + new tests + exports), so the 552-test suite,
every road/DNL hash, and the golden Braess content hash are byte-untouched.

## Sourcing

Daganzo (1994) *TR-B* 28(4):269-287 and (1995) Part II *TR-B* 29(2) are **paywalled,
attributed unread**. The `min(sending, receiving)` recipe, the `(vf, w, kappa)`
triangular FD, and all three anchors were cross-verified from open sources — Boyles
*Transportation Network Analysis* §10.5 and Daganzo's own open ISTTT (1999)
restatement of the identical base scheme — and hand-derived here from LWR /
Rankine–Hugoniot theory. No DOIs or page-precise quotes are reproduced from the
paywalled primaries.
