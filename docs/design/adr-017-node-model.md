# ADR-017: node-model — Tampère et al. (2011) generic first-order node model

**Status:** accepted (implemented)
**Date:** 2026-07-09
**Deciders:** DNL-models track — the general merge/diverge node (unlocks network loading)
**File:** `docs/design/adr-017-node-model.md`

## Context

The dnl-core (adr-010) shipped the node axioms N1–N6, `assert_node_axioms`, and
the trivially-axiom-satisfying `SeriesNode` / `OriginNode` / `DestinationNode`,
but deliberately deferred the *general* merge/diverge solver — the `NetworkLoader`
raised for any interior node that was not 1-in-1-out. Without it, the DNL link
models (`ctm`, `ltm`) could only load single corridors. This sprint implements the
general node model, which is what Daganzo (1995) Part II's *network* CTM needs and
what makes the loader usable on real junctions.

## Decision

1. **`TampereNode(NodeModel)`** in `src/tabench/dnl/node.py`, on the frozen
   `transfer(s, r, turns, caps)` interface — no new state, no signature change.
   `caps` (`q_max_i·dt`, already passed by the loader) are the priority weights.

2. **Oriented-capacity-proportional distribution with FIFO** (the "equal priority
   movements" algorithm). Each active movement `(i, j)` (`turns[i,j] > 0`,
   `s[i] > 0`) flows at rate `alpha[i,j] = caps[i]·turns[i,j]`; advance every
   active movement by the largest common step `theta` before some incoming link
   exhausts its `s` budget or some outgoing link saturates its `r`; a saturated
   outgoing link then removes **every** movement of each competing approach that
   uses it (FIFO — a blocked turn holds its whole approach back in proportion);
   repeat until no movement is active. It terminates in at most `n_in` binding
   rounds. Because every movement of row `i` accrues `turns[i,j]·(caps[i]·Σθ)`,
   the row is turn-proportional by construction (**N4 exact**), and it reduces to
   `min(s, r)` at a series node, capacity-proportional priority at a merge, and
   the FIFO hold-back at a diverge.

3. **Loader default.** `NetworkLoader` now instantiates `TampereNode` for every
   interior merge/diverge node instead of raising (an explicit `node_models`
   entry still overrides it). No other loader change — it already assembles
   exactly `(sending[ins], receiving[outs], turns, caps)` per interior node.

## Analytic anchors (exact fractions, machine-verified — `test_dnl_tampere_node.py`)

- **Merge 2→1:** `s=[1,1]`, `r=1` → `caps=[1,1]` gives `[0.5,0.5]`;
  `caps=[2,1]` gives `[2/3,1/3]` (capacity-proportional).
- **Diverge 1→2 FIFO:** `s=[2]`, `turns=[[0.6,0.4]]`, out-link 2 supplies `0.4` →
  `phi=min(1e6/1.2, 0.4/0.8, 1)=0.5`, so `q=[0.6,0.4]` (one of two vehicles held
  back on *both* movements — the whole approach throttled).
- **2×2, out-link A binding:** `s=[10,10]`, `caps=[6,8]`,
  `turns=[[0.7,0.3],[0.4,0.6]]`, `r=[5,100]` → `q=[[105/37,45/37],[80/37,120/37]]`,
  out-link A saturated at 5, rows exactly turn-proportional.
- **N6 invariance:** inflating a non-binding sending flow (a FIFO-blocked diverge
  approach `s: 2→200`; a receiving-limited merge approach `s: 5→500`) leaves `q`
  bit-identical.
- **End-to-end:** the loader now certifies merge and diverge `DynamicScenario`s
  (C1 conservation clean, C8 turn fidelity `~0`, capacity-proportional bottleneck
  sharing) that previously raised.

## Alternatives considered

- **Deferring to an explicit `node_models` arg (status quo):** rejected — the
  point of this sprint is that the loader handles junctions by default; the
  override path is retained for research.
- **Daganzo (1995) specific merge/diverge rules:** the Tampère generic model
  subsumes them (Daganzo's are special cases), so `TampereNode` covers the network
  CTM/LTM case with one algorithm; `daganzo1995cell` is left as its historical
  special-case reference.

## Adversarial review

An adversarial review (300k+ fuzzed `transfer` calls + end-to-end loader runs)
confirmed termination (proven + 200k trials), N6 invariance, FIFO correctness,
determinism, and clean loader loading (a 6-link diamond diverge→merge under CTM
and LTM, never censored). It caught one real defect: a **global** `s.sum()`-scaled
tolerance let one huge approach's float dust swallow a tiny co-incident approach's
entire sending flow (an N5 violation at ~1e12 capacity ratios; lost flow bounded
by machine-dust, hard to trigger loudly through the loader). Fixed to **per-element
tolerances** relative to each row's `caps[i]` and each finite column's `r[j]`;
regression-pinned (`test_tiny_approach_not_dropped_at_extreme_capacity_ratio` +
a 2000-case extreme-ratio fuzz). The benign divide-by-zero warning the review noted
was also removed (masked division).

## Consequences

The benchmark gains network loading — `ctm`/`ltm` on arbitrary merge/diverge
networks — via one axiom-satisfying node model. All changes are additive (a new
class + loader default + tests + exports); the one loader test that asserted the
old raise is updated to assert the new default. The 570-test suite, every road/DNL
hash, and the golden Braess content hash are byte-untouched.

## Sourcing

Tampère, Corthout, Cattrysse & Immers (2011) *TR-B* 45(1):289–309 is **paywalled,
attributed unread**. The algorithm is restated from two open, read sources:
Boyles/Lownes/Unnikrishnan *Transportation Network Analysis* Vol. I §9.6.2 ("equal
priority movements"; §9.7 notes credit the desiderata and node models to Tampère
et al. 2011) and Yperman (2007) PhD thesis (KU Leuven) Ch. 5 (eq. 5.6–5.12).
Flötteröd & Rohde (2011) and Corthout et al. (2012, non-unique intersection flows)
are attributed via Boyles' secondary citation only, not independently read.
