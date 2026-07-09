# ADR-014: transit-strategy — Spiess & Florian (1989) optimal-strategy transit assignment

**Status:** accepted (implemented)
**Date:** 2026-07-09
**Deciders:** static-extensions track — the last Phase-1 static model (transit)
**File:** `docs/design/adr-014-transit-strategy.md`

## Context

The remaining Phase-1 model, Spiess & Florian (1989, *Transportation Research
Part B* 23(2):83-102, `spiess1989optimal`), assigns transit demand by **optimal
strategies**: at each stop a passenger designates a set of *attractive* lines and
boards the first-arriving one, minimizing the **expected** total travel time
(in-vehicle time + expected wait). Under random arrivals the wait at a stop whose
attractive lines have combined frequency `F` is `1/F`, and the passenger boards
line `a` with probability `f_a/F`. This is a convex LP (globally optimal), solved
in two label-setting/loading passes — not an iterative equilibrium, and
**uncongested / frequency-based** (costs are flow-independent).

It cannot be a road model. The road `Network` (`core/scenario.py`) forbids
parallel links (unique `(init_node, term_node)`) and its costs are flow-dependent
BPR — but the whole point of the "common lines" problem is **several lines
serving the same ordered stop pair**, and transit costs here do not depend on
volume. ADR-010 already rejected "extend `core.Scenario` with optional fields"
for the *less* invasive DNL case; transit is further still (it cannot even reuse
`Network` read-only for topology, given the parallel-arc requirement).

## Decision

1. **A parallel module `src/tabench/transit/`** (mirroring `src/tabench/dnl/`),
   touching no road code, so no static golden hash can move (the Braess hash
   `cf00f411…` is re-asserted in `tests/test_transit.py`). It has its own frozen,
   content-hashed scenario, its own solver, and its own certifier.

2. **A directed multigraph as the network** (`transit/network.py`,
   `TransitNetwork`): arrays `tail`/`head`/`time`/`freq`, one entry per arc, with
   **parallel arcs allowed**. A finite `freq` is a boardable line (expected wait
   `1/freq`); `freq = inf` is a deterministic arc (walk / transfer / in-vehicle
   continuation, no wait). This directly encodes the common-lines problem without
   the split-node convention. `TransitDemand` carries `(origin, destination,
   volume)` triples (0-based node ids); `TransitScenario` wraps them with a
   domain-separated `content_hash()` (`"tabench-transit-scenario-v1;"` prefix).
   The emitted solution is `TransitStrategy` (arc volumes + per-destination labels
   + per-pair costs) — the transit analogue of `FlowState`.

3. **Two-pass solver** (`transit/strategy.py`, `optimal_strategy`,
   `OptimalStrategyModel`, name `transit-strategy`). Per destination: (a)
   label-setting over arcs in nondecreasing onward cost `u[head]+time`, adding arc
   `a=(i,j)` to node `i`'s attractive set iff its onward cost is strictly below
   `u_i`, updating `u_i ← (f_i u_i + f_a(u_j+c_a))/(f_i+f_a)`, `f_i ← f_i+f_a`,
   with the expected wait seeded as the `+1` on the FIRST attractive line
   (`u_i ← 1/f_a + (u_j+c_a)`); a deterministic arc that beats the current cost
   closes the node. (b) Loading nodes farthest-first, splitting by frequency share
   `v_a = (f_a/f_i)V_i`. It is standalone (its scenario is a `TransitScenario`,
   not the road `Scenario`), so it is NOT in the road `MODEL_REGISTRY` — the same
   parallel-module choice the DNL core made.

4. **Certificate (P1)** (`metrics/transit_gaps.py`, `TransitEvaluator`). The
   harness recomputes the LP optimum `Z*` independently from the scenario (as the
   road certifier recomputes an all-or-nothing bound), and scores the emitted
   primal cost `Z_emitted` — recomputed from the emitted arc volumes as
   `sum_a c_a v_a + sum_i w_i` — via `optimality_gap = (Z_emitted − Z*)/Z*` (≥ 0,
   0 iff optimal), the transit analogue of the relative gap. The wait is per
   commodity: it is per-(node, destination), so the harness certifies the
   PER-DESTINATION arc-volume decomposition (`TransitStrategy.dest_arc_volumes`;
   a single-destination scenario may pass the summed volumes) with per-commodity
   conservation, and at each node uses the LP-minimal feasible wait
   `w_i = max_a v_a/f_a` (every out-arc must satisfy `f_a w_i >= v_a`) — NOT
   `V_i/F_i` over a tolerance-thresholded subset, which would let a sub-tolerance
   sliver on a near-zero-frequency arc dodge its wait and drive the gap negative.
   With `w_i = max_a v_a/f_a` the gap is provably ≥ 0, and a non-proportional
   split is feasible-but-suboptimal (a larger `w_i` raises the gap) rather than
   censored. Certification gates on demand feasibility alone (nonnegative,
   conserving each destination's demand); the model's self-reported labels/costs
   are never trusted. **[Both a multi-destination wait-aggregation bug and the
   near-zero-frequency negative-gap hole were caught by adversarial review and
   fixed to this per-destination, max-wait form; regression-pinned in
   `tests/test_transit.py`.]**

5. **Analytic anchors** (`transit/builtin.py`, both recomputed by the closed form
   in the tests). The classic common-lines example, two parallel lines to the
   sink:
   - `common_lines_scenario`: line 0 `(f=1/6, t=21)`, line 1 `(f=1/12, t=18)` →
     combined `F=1/4`, optimal `C* = (1 + 1/6·21 + 1/12·18)/(1/4) = 6/(1/4) = 24`
     min (wait 4 + ride 20), demand split 2:1 (`v0 = 2D/3`, `v1 = D/3`);
   - `common_lines_unattractive_scenario`: line 0 `(1/6, 15)` alone gives 21 min,
     line 1 `(1/12, 40)` has onward cost `40 ≥ 21` so it is excluded — all demand
     on line 0, `C* = 21`.
   Both satisfy the primal-dual identity `Z_emitted = Z*` exactly.

## Alternatives considered

- **Optional field on the road `Scenario`** (the static house pattern used by
  `evans`/`br-ue`/`sc-tap`/`vi-asym`/`multiclass`): rejected — parallel arcs
  break `Network`'s unique-link invariant and BPR costs are the wrong model; a
  transit hash would also have to live in the road hash space. Same reasoning as
  ADR-010 for DNL, only stronger.
- **Split-node encoding** (separate line-nodes per stop, so all arcs are unique):
  rejected as unnecessary — the multigraph with per-arc `(time, freq)` models the
  common-lines problem directly and more legibly; deterministic in-vehicle
  continuations are just `freq = inf` arcs.
- **Registering it in the road `MODEL_REGISTRY`**: rejected — `TrafficAssignmentModel.solve`
  is typed to the road `Scenario` and the runner constructs road scenarios; the
  DNL core set the precedent that a different scenario type gets its own
  module/solver/certifier, exercised directly (not through the road CLI/runner).

## Consequences

The benchmark gains its first **transit** model and its first uncongested,
frequency-based, hyperpath assignment — a genuinely different domain from road
UE, with a sound harness-recomputed optimality certificate and hand-derived
common-lines anchors. All changes are additive (a new module + new files only),
so every road/DNL hash and the 517-test road suite are untouched. Follow-ups: the
congested transit extension (De Cea & Fernández 1993, `decea1993transit`, already
tier-2 in-canon), transfer/boarding penalties, and CLI/experiment-matrix
integration for the transit scenario type.

## Sourcing

Spiess & Florian (1989, *Transportation Research Part B* 23(2):83-102,
`spiess1989optimal`, DOI `10.1016/0191-2615(89)90034-9`, Crossref-verified) is
the primary; it was already in the verified reference canon. The two-pass
algorithm, the common-lines expected-cost formula, and both analytic anchors are
hand-derived here (the expected wait `1/F`, the frequency-share split, and the
attractiveness threshold `c_l < C`), checked for internal consistency
(`Z_emitted = Z*`); no number from the paper is reproduced. It must not be
confused with Spiess (1990) `spiess1990gradient`, the OD-adjustment gradient
method already shipped as the T2 estimator `spiess`.
