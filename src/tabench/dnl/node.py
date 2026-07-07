"""Node models for dynamic network loading (dnl-core, adr-010).

Layer-2 component; imports numpy only (no grid/fd/demand coupling): a node
model is pure count algebra, allocating transfer flows ``q[i, j]`` (vehicles
this step) from sending flows S, receiving flows R, turning fractions, and
per-incoming-link capacities. Node models never see ``dt``: every quantity is
a vehicle COUNT per step, so min-allocations and conservation sums need no
unit bookkeeping.

Node axioms N1-N6 — the shared contract every :class:`NodeModel` must satisfy.
Sourcing honesty: Tampere, Corthout, Cattrysse & Immers (2011), "A generic
class of first order node models", Transp. Res. B 45(1), is PAYWALLED —
attributed unread; the requirement list is restated from open literature
(Yperman's 2007 LTM thesis, open; Lebacque & Khoshyaran 2005 invariance,
restated in open follow-ups). Daganzo (1995, Part II) diverge FIFO likewise
restated. The node-model sprint re-verifies its requirement list against
whatever open restatement it implements from.

* **N1 nonnegativity:** ``q >= 0``.
* **N2 demand respect:** ``q.sum(axis=1) <= s`` elementwise.
* **N3 supply respect:** ``q.sum(axis=0) <= r`` elementwise.
* **N4 conservation of turning fractions (CTF):**
  ``q[i, :] == turns[i, :] * q[i, :].sum()`` (FIFO across destinations at a
  diverge; Daganzo 1995 restated).
* **N5 local flow maximization:** ``q[i, :].sum() < s_i`` implies some ``j``
  with ``turns[i, j] > 0`` and ``q[:, j].sum() >= r_j`` — holding flow back
  is legal only when an eligible outgoing link is supply-saturated.
* **N6 invariance principle** (Lebacque & Khoshyaran 2005, restated): the
  solution must not change when a constrained sending flow (or supply) is
  inflated. NOT checkable from a single ``(s, r, q)`` triple — documented
  here, covered by the behavioral test pattern
  ``transfer(s, r, t, c) == transfer(s', r, t, c)`` with ``s'`` inflated on
  constrained inputs. The node-model sprint must satisfy it.

:func:`assert_node_axioms` checks N1-N5 on one allocation. The shipped nodes
(Series/Origin/Destination) are the trivially-axiom-satisfying minimum;
general merge/diverge solvers are the node-model sprint (the loader refuses
to guess at real junctions).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

__all__ = [
    "NodeTopology",
    "NodeModel",
    "assert_node_axioms",
    "SeriesNode",
    "OriginNode",
    "DestinationNode",
]


@dataclass(frozen=True)
class NodeTopology:
    """A node's incident links, indexed in scenario link order.

    ``in_links``/``out_links`` are ASCENDING-sorted link indices — the
    deterministic ordering that :class:`~tabench.dnl.demand.TurningFractions`
    matrices (rows = incoming, columns = outgoing) are aligned with, and that
    the loader processes nodes in. Empty arrays are allowed (origin zones
    have no network in-links; destination zones no out-links).
    """

    node_id: int
    in_links: np.ndarray  # (n_in,) int64 link indices, ascending
    out_links: np.ndarray  # (n_out,) int64 link indices, ascending

    def __post_init__(self) -> None:
        if isinstance(self.node_id, bool) or not isinstance(self.node_id, (int, np.integer)):
            raise ValueError(f"NodeTopology node_id must be an int, got {self.node_id!r}")
        object.__setattr__(self, "node_id", int(self.node_id))
        for name in ("in_links", "out_links"):
            arr = np.asarray(getattr(self, name))
            if arr.ndim != 1:
                raise ValueError(f"NodeTopology {name} must be 1-D, got shape {arr.shape}")
            if arr.size and not np.issubdtype(arr.dtype, np.integer):
                raise ValueError(
                    f"NodeTopology {name} must be integer link indices, got dtype {arr.dtype}"
                )
            arr = np.ascontiguousarray(arr, dtype=np.int64)
            if arr.size and np.any(arr < 0):
                raise ValueError(f"NodeTopology {name} must be nonnegative link indices")
            if arr.size > 1 and np.any(np.diff(arr) <= 0):
                raise ValueError(
                    f"NodeTopology {name} must be strictly ascending "
                    "(deterministic order; turn matrices align with it)"
                )
            object.__setattr__(self, name, arr)

    @property
    def n_in(self) -> int:
        return int(self.in_links.shape[0])

    @property
    def n_out(self) -> int:
        return int(self.out_links.shape[0])


class NodeModel(ABC):
    """Allocates transfer flows ``q[i, j]`` (vehicles this step) from S/R/turns/caps."""

    @abstractmethod
    def transfer(
        self,
        s: np.ndarray,
        r: np.ndarray,
        turns: np.ndarray,
        caps: np.ndarray,
    ) -> np.ndarray:
        """Allocate transfer counts for one step.

        Parameters: ``s`` — ``(n_in,)`` sending flows of incoming links
        [veh/step]; ``r`` — ``(n_out,)`` receiving flows of outgoing links
        [veh/step], ``+inf`` allowed (destination absorption); ``turns`` —
        ``(n_in, n_out)`` row-stochastic turning fractions aligned with the
        node's ascending link orderings; ``caps`` — ``(n_in,)``
        ``q_max_i * dt`` of incoming links [veh/step], the priority weights
        for Tampere-style capacity-proportional merges. Shipped nodes ignore
        ``caps``; the node-model sprint requires it — it is carried NOW so
        the frozen contract never changes.

        Returns a fresh ``(n_in, n_out)`` float64 array ``q`` satisfying
        axioms N1-N5 (N6 behaviorally); inputs are never mutated.
        """


def _coerce_transfer_inputs(
    s: np.ndarray, r: np.ndarray, turns: np.ndarray, caps: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Coerce/validate one transfer call's inputs (programming-error checks)."""
    s = np.ascontiguousarray(s, dtype=np.float64)
    r = np.ascontiguousarray(r, dtype=np.float64)
    turns = np.ascontiguousarray(turns, dtype=np.float64)
    caps = np.ascontiguousarray(caps, dtype=np.float64)
    if s.ndim != 1 or s.shape[0] < 1:
        raise ValueError(f"transfer: s must be a nonempty (n_in,) array, got shape {s.shape}")
    if r.ndim != 1 or r.shape[0] < 1:
        raise ValueError(f"transfer: r must be a nonempty (n_out,) array, got shape {r.shape}")
    if turns.shape != (s.shape[0], r.shape[0]):
        raise ValueError(
            f"transfer: turns must have shape {(s.shape[0], r.shape[0])}, got {turns.shape}"
        )
    if caps.shape != s.shape:
        raise ValueError(f"transfer: caps must have shape {s.shape}, got {caps.shape}")
    if np.isnan(s).any() or np.isnan(r).any() or np.isnan(turns).any() or np.isnan(caps).any():
        raise ValueError("transfer: inputs must not contain NaN")
    if np.any(s < 0) or np.any(r < 0):
        raise ValueError("transfer: sending/receiving flows must be nonnegative")
    return s, r, turns, caps


def assert_node_axioms(
    q: np.ndarray,
    s: np.ndarray,
    r: np.ndarray,
    turns: np.ndarray,
    *,
    eps: float,
) -> None:
    """Check node axioms N1-N5 on one allocation; raise ``ValueError`` naming
    the violated axiom.

    N6 (invariance) is NOT checkable from a single ``(s, r, q)`` triple — see
    the module docstring for the behavioral test pattern. ``r`` entries may
    be ``+inf`` (destination absorption); an infinite-supply column can never
    be saturated, so holding back eligible flow toward it violates N5.
    ``eps`` is an absolute count tolerance (callers scale it; the evaluator
    uses ``tol * max(1, V)``-style scales).
    """
    eps = float(eps)
    if not (np.isfinite(eps) and eps >= 0):
        raise ValueError(f"assert_node_axioms: eps must be finite and >= 0, got {eps!r}")
    q = np.ascontiguousarray(q, dtype=np.float64)
    s = np.ascontiguousarray(s, dtype=np.float64)
    r = np.ascontiguousarray(r, dtype=np.float64)
    turns = np.ascontiguousarray(turns, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError(f"assert_node_axioms: q must be 2-D (n_in, n_out), got shape {q.shape}")
    if s.shape != (q.shape[0],) or r.shape != (q.shape[1],) or turns.shape != q.shape:
        raise ValueError(
            f"assert_node_axioms: shape mismatch — q {q.shape}, s {s.shape}, r {r.shape}, "
            f"turns {turns.shape}"
        )
    if not np.isfinite(q).all():
        raise ValueError("assert_node_axioms: q must be finite")
    if np.isnan(s).any() or np.isnan(r).any() or not np.isfinite(turns).all():
        raise ValueError("assert_node_axioms: s/r must not be NaN and turns must be finite")

    if np.any(q < -eps):
        raise ValueError(
            f"node axiom N1 (nonnegativity) violated: min q = {float(q.min())!r} < -{eps!r}"
        )
    row_sum = q.sum(axis=1)
    excess = row_sum - s
    if np.any(excess > eps):
        i = int(np.argmax(excess))
        raise ValueError(
            f"node axiom N2 (demand respect) violated at in-link {i}: "
            f"sum_j q[{i}, j] = {float(row_sum[i])!r} > s = {float(s[i])!r} + {eps!r}"
        )
    col_sum = q.sum(axis=0)
    excess = col_sum - r
    if np.any(excess > eps):
        j = int(np.argmax(excess))
        raise ValueError(
            f"node axiom N3 (supply respect) violated at out-link {j}: "
            f"sum_i q[i, {j}] = {float(col_sum[j])!r} > r = {float(r[j])!r} + {eps!r}"
        )
    ctf_resid = np.abs(q - turns * row_sum[:, None])
    if np.any(ctf_resid > eps):
        i, j = np.unravel_index(int(np.argmax(ctf_resid)), ctf_resid.shape)
        raise ValueError(
            f"node axiom N4 (conservation of turning fractions) violated at ({i}, {j}): "
            f"|q - turns * row_sum| = {float(ctf_resid[i, j])!r} > {eps!r}"
        )
    undersent = row_sum < s - eps
    if np.any(undersent):
        saturated = col_sum >= r - eps
        for i in np.nonzero(undersent)[0]:
            if not np.any((turns[i] > 0.0) & saturated):
                raise ValueError(
                    f"node axiom N5 (local flow maximization) violated at in-link {int(i)}: "
                    f"sum_j q = {float(row_sum[i])!r} < s = {float(s[i])!r} - {eps!r} "
                    "but no eligible outgoing link is supply-saturated"
                )


class SeriesNode(NodeModel):
    """1 in, 1 out: ``q = [[min(s0, r0)]]`` — the unique N1-N6 solution for 1x1.

    Uniqueness: N2/N3 force ``q <= min(s, r)`` and N5 forbids anything
    smaller (an undersent row requires a saturated column). ``turns`` must be
    ``(1, 1)``; its value is not consulted (a 1x1 row-stochastic matrix is
    identically 1). ``caps`` is ignored (documented: no merge competition at
    a series node).
    """

    def transfer(
        self, s: np.ndarray, r: np.ndarray, turns: np.ndarray, caps: np.ndarray
    ) -> np.ndarray:
        s, r, turns, caps = _coerce_transfer_inputs(s, r, turns, caps)
        if s.shape[0] != 1 or r.shape[0] != 1:
            raise ValueError(
                f"SeriesNode is 1-in/1-out, got n_in = {s.shape[0]}, n_out = {r.shape[0]}"
            )
        return np.array([[min(float(s[0]), float(r[0]))]], dtype=np.float64)


class OriginNode(NodeModel):
    """Origin injection node (0 network in-links).

    The loader feeds a synthetic ``s = [waiting]`` (the origin's vertical
    queue content, exact piecewise-linear cumulative demand minus releases)
    and the origin's outgoing split as ``turns`` (shape ``(1, n_out)``), with
    ``caps = [inf]``.

    ``n_out == 1`` (all v1 anchors): ``q = [[min(waiting, r0)]]``.
    ``n_out >= 2``: fixed-split allocation
    ``q[0, j] = min(waiting * split_j, r_j)`` — satisfies N1-N3 and is a
    documented PLACEHOLDER policy: it does NOT re-normalize across blocked
    splits, so N4/N5 hold per column, not jointly, when some split is
    supply-blocked. The node-model sprint replaces/extends origin policies;
    nothing downstream bakes against this. The general formula reduces
    exactly to the single-out rule at ``n_out == 1`` (``split_0 = 1``).
    ``caps`` is ignored (an origin queue has no physical capacity in v1).
    """

    def transfer(
        self, s: np.ndarray, r: np.ndarray, turns: np.ndarray, caps: np.ndarray
    ) -> np.ndarray:
        s, r, turns, caps = _coerce_transfer_inputs(s, r, turns, caps)
        if s.shape[0] != 1:
            raise ValueError(
                f"OriginNode takes exactly one synthetic waiting entry, got n_in = {s.shape[0]}"
            )
        return np.minimum(s[0] * turns[0], r).reshape(1, -1)


class DestinationNode(NodeModel):
    """Destination absorption node (0 network out-links): absorbs everything.

    The loader treats head-of-link-at-destination supply as ``r = +inf`` (its
    step loop absorbs sending flows directly, without a transfer call — the
    class is shipped so the node contract is total). ``r`` is therefore
    IGNORED here (the +inf convention), as is ``caps``. Returns
    ``q = s[:, None] * turns``, so ``q[i, :].sum() == s_i`` for row-stochastic
    ``turns`` — every sending vehicle is absorbed.
    """

    def transfer(
        self, s: np.ndarray, r: np.ndarray, turns: np.ndarray, caps: np.ndarray
    ) -> np.ndarray:
        s, r, turns, caps = _coerce_transfer_inputs(s, r, turns, caps)
        return s[:, None] * turns
