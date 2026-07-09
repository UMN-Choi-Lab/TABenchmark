# ADR-018: godunov — Lebacque (1996) Godunov scheme + the first non-triangular FD

**Status:** accepted (implemented)
**Date:** 2026-07-09
**Deciders:** DNL-models track — the general-FD Godunov scheme (first rarefaction physics)
**File:** `docs/design/adr-018-godunov.md`

## Context

`ctm` (adr-015) is the Godunov scheme for the **triangular** fundamental diagram.
Lebacque (1996) frames the Godunov scheme for **any** first-order (LWR) model via
the demand/supply (`Delta`/`Sigma`) flux — exactly the `fd.demand_at`/`supply_at`
interface the dnl-core already exposes. The genuinely new content this sprint adds
over `ctm` is a **smooth, non-triangular concave FD** and the **rarefaction fan**
physics it produces — a triangular FD produces only shocks and contact
discontinuities, so no benchmark test has ever exercised a rarefaction.

## Decision

1. **`GreenshieldsFD`** in `fd.py` — the first smooth, strictly concave FD,
   `Q(k) = vf·k·(1 - k/kappa)` (`k_c = kappa/2`, `q_max = vf·kappa/4`, `vf = Q'(0)`,
   `w = |Q'(kappa)| = vf`). Its Lebacque demand `Q(min(k, k_c))` and supply
   `Q(max(k, k_c))` give the exact Godunov flux `min(demand, supply)` for a concave
   FD, and its inherited `envelope_params` triangular majorant `(vf, vf, kappa)` is
   sound: `Q(k) <= min(vf·k, vf·(kappa - k))` with equality only at the endpoints.

2. **`GodunovLink(CTMLink)`** in `godunov.py` — reuses the verified CTMLink cell
   update **unchanged** (the scheme is identical; only the FD differs), substituting
   a `GreenshieldsFD` built from the link's `(vf = free_speed, kappa = jam_density)`.
   The transonic Godunov flux is entropy-correct by construction: at an interface
   `k_L > k_c > k_R` (a rarefaction spanning the sonic point) `min(demand(k_L),
   supply(k_R)) = q_max`, the maximum-flux sonic value.

3. **Certifier unchanged — the triangular majorant is sound for the concave FD.**
   The certifier reads `dynamics.fd(a).envelope_params()` and `dynamics.capacity`
   (a triangular majorant + capacity), NOT `GodunovLink`'s FD; because Greenshields
   is majorized by `(vf, vf, kappa)` and its capacity is `vf·kappa/4`, a Greenshields
   loading certifies with no certificate change (its parabolic free-flow speed
   `vf(1 - k/kappa) < vf` is *slower* than the Newell `vf` envelope, so C4 holds a
   fortiori). `GodunovLink` **guards** that the `LinkDynamics` is Greenshields-
   consistent (`wave_speed == free_speed`, `capacity == vf·kappa/4`) so the
   certifier never gates against a mismatched capacity.

## Analytic anchors (machine-verified — `test_dnl_godunov.py`)

- **GreenshieldsFD**: derived quantities, the parabolic `flow`/`demand`/`supply`
  branch values, majorant soundness, validation.
- **Transonic Godunov flux**: `min(demand(3.5), supply(0.5)) = q_max = 2` (the
  entropy-correct rarefaction value, not a shock value).
- **Loader + certification**: a Greenshields loading through `NetworkLoader` +
  `DNLEvaluator` certifies (`dnl_feasible = 1`), demonstrating the triangular
  majorant is sound for the smooth FD.
- **Rarefaction convergence** (the distinguishing anchor): a dam-break Riemann
  problem (jam left, empty right) stepped by the Godunov flux converges to the
  analytic self-similar fan `k(x,t) = (kappa/2)(1 - (x-x0)/(vf·t))`; the L1 error
  shrinks monotonically as the cells are refined (`0.128 → 0.041` over `20 → 160`
  cells — first-order Godunov, slowed by the sonic-point sqrt singularity).

## Alternatives considered

- **A distinct GodunovLink cell update:** rejected — the cell update is identical
  to CTM's; only the FD differs. `GodunovLink` subclasses `CTMLink`, reusing the
  reviewed (and w>vf-hardened) code, and ships the new FD + rarefaction validation.
- **Extending `LinkDynamics` with an `fd_kind` field:** deferred — `GodunovLink`
  building the Greenshields FD from `(vf, kappa)` (with a consistency guard) needs
  no change to the frozen `LinkDynamics`/scenario hash; a general FD registry can
  come when a second non-triangular FD ships.
- **An asymmetric (`w < vf`) smooth FD:** deferred — Greenshields (`w = vf`) already
  exercises rarefactions; an asymmetric smooth FD is a follow-up if a `w < vf`
  rarefaction anchor is wanted.

## Consequences

The benchmark gains its first non-triangular fundamental diagram and its first
rarefaction physics, on the shipped cell scheme + certifier with no certificate
change. All changes are additive (a new FD + a new link + tests + exports), so the
582-test suite, every road/DNL hash, and the golden Braess content hash are
byte-untouched.

## Sourcing

Lebacque (1996, ISTTT 13) "The Godunov scheme and what it means for first order
traffic flow models" — the demand/supply Godunov formalism; open restatements
circulate, attributed. Greenshields (1935) for the parabolic FD; Godunov (1959) /
LeVeque *Finite Volume Methods for Hyperbolic Problems* for the exact-Riemann-solver
flux and rarefaction theory. No DOIs or page-precise quotes reproduced.
