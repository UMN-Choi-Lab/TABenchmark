"""The canonical Merchant-Nemhauser LP, its reference solver, and the emitted
trajectory artifact (docs/design/adr-020-merchant-nemhauser.md).

The benchmark's canonical program is the Carey (1987) relaxation of M-N 1978
with piecewise-linear concave exit functions and linear costs — an LP:

    min   sum_{t=1..T} sum_a w_a * x_a(t)
    s.t.  x_a(t+1) = x_a(t) + u_a(t) - e_a(t)              (conservation)
          sum_{a out of j} u_a(t) = d_j(t)
              + sum_{b into j} e_b(t)   for j != destination (node balance)
          x_a(T) = 0                                        (terminal clearance)
          0 <= e_a(t) <= g_a(x_a(t))  encoded one row per affine piece
          x, u, e >= 0,   x_a(0) = 0 (empty network, eliminated)

with x_a(t) the start-of-period-t occupancy, u_a(t) the inflow DURING period t
(joins the state at t+1, so it cannot exit during t), e_a(t) the exit DURING
period t, and same-period node hand-off (exits of period t feed downstream
inflows of period t). Relaxing M-N's exit EQUALITY ``e = g(x)`` to ``<=``
(outflow as a control bounded by the exit function) is what makes the program
convex — Carey (1987); the M-N equality form is nonconvex (Carey 1992) and slack
in the bound is deliberate "holding back" (ramp metering), which can be strictly
optimal (see ``builtin.mn_metering_scenario``). Terminal clearance is the
benchmark convention that makes total cost well-posed (no stranded flow);
an infeasible LP means the horizon ``T`` is too short for the demand.

``DTATrajectory`` is the emitted, P1-certifiable artifact. ``duals`` optionally
carries an LP dual certificate ``{"eq": y_eq, "ub": y_ub}`` in THIS module's
canonical row order; the certifier (``metrics/dta_gaps.py``) verifies it by
pure arithmetic (weak duality + zero gap = global optimality) and never trusts
it — a wrong certificate is reported, not believed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from .scenario import SODTAScenario

__all__ = ["DTATrajectory", "canonical_lp", "solve_so_dta"]


@dataclass(frozen=True)
class DTATrajectory:
    """Emitted M-N plan: ``inflows``/``exits`` are ``(T, n_links)`` per-period
    flows, ``occupancies`` the ``(T+1, n_links)`` start-of-period states
    (row 0 = the empty initial network). ``duals`` is the optional LP dual
    certificate; ``provenance`` carries solver self-reports (never certified).
    """

    scenario_hash: str
    inflows: np.ndarray  # (T, L) float64
    exits: np.ndarray  # (T, L)
    occupancies: np.ndarray  # (T+1, L)
    duals: dict[str, np.ndarray] | None = None
    provenance: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("inflows", "exits", "occupancies"):
            arr = np.ascontiguousarray(getattr(self, name), dtype=np.float64)
            object.__setattr__(self, name, arr)
        if self.inflows.ndim != 2 or self.inflows.shape != self.exits.shape:
            raise ValueError("DTATrajectory inflows/exits must be (T, n_links)")
        t, n_links = self.inflows.shape
        if self.occupancies.shape != (t + 1, n_links):
            raise ValueError("DTATrajectory occupancies must be (T+1, n_links)")
        if self.duals is not None:
            duals = {
                k: np.ascontiguousarray(v, dtype=np.float64) for k, v in self.duals.items()
            }
            if set(duals) != {"eq", "ub"}:
                raise ValueError('DTATrajectory duals must have exactly keys "eq" and "ub"')
            object.__setattr__(self, "duals", duals)


def canonical_lp(
    scenario: SODTAScenario,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(c, A_eq, b_eq, A_ub, b_ub)`` for the canonical M-N LP.

    Variable layout (all >= 0): ``x_a(t)`` for ``t=1..T`` at ``a*T + (t-1)``;
    ``u_a(t)`` for ``t=0..T-1`` at ``L*T + a*T + t``; ``e_a(t)`` at
    ``2*L*T + a*T + t``. Equality-row order: conservation ``(a, t)`` in
    lexicographic ``a``-major order, then node balance ``(j, t)`` over
    non-destination nodes ``j`` ascending, then terminal clearance ``x_a(T) = 0``
    per link. Inequality rows: one per ``(a, t, piece)`` in that nesting. This
    fixed ordering is part of the artifact contract — an emitted dual
    certificate refers to it.
    """
    n_l, n_t = scenario.n_links, scenario.n_periods
    tail, head = scenario.link_tail, scenario.link_head
    nvar = 3 * n_l * n_t

    def ix(a: int, t: int) -> int:  # x_a(t), t = 1..T
        return a * n_t + (t - 1)

    def iu(a: int, t: int) -> int:  # u_a(t), t = 0..T-1
        return n_l * n_t + a * n_t + t

    def ie(a: int, t: int) -> int:  # e_a(t), t = 0..T-1
        return 2 * n_l * n_t + a * n_t + t

    c = np.zeros(nvar)
    for a in range(n_l):
        c[ix(a, 1) : ix(a, n_t) + 1] = scenario.cost_weights[a]

    eq_rows: list[np.ndarray] = []
    b_eq: list[float] = []
    # conservation: x_a(t+1) - x_a(t) - u_a(t) + e_a(t) = 0 (x_a(0) = 0 eliminated)
    for a in range(n_l):
        for t in range(n_t):
            row = np.zeros(nvar)
            row[ix(a, t + 1)] = 1.0
            if t >= 1:
                row[ix(a, t)] = -1.0
            row[iu(a, t)] = -1.0
            row[ie(a, t)] = 1.0
            eq_rows.append(row)
            b_eq.append(0.0)
    # node balance at every non-destination node, every period
    for j in range(scenario.n_nodes):
        if j == scenario.destination:
            continue
        out_links = np.nonzero(tail == j)[0]
        in_links = np.nonzero(head == j)[0]
        for t in range(n_t):
            row = np.zeros(nvar)
            for a in out_links:
                row[iu(int(a), t)] = 1.0
            for b in in_links:
                row[ie(int(b), t)] = -1.0
            eq_rows.append(row)
            b_eq.append(float(scenario.demand[t, j]))
    # terminal clearance x_a(T) = 0
    for a in range(n_l):
        row = np.zeros(nvar)
        row[ix(a, n_t)] = 1.0
        eq_rows.append(row)
        b_eq.append(0.0)

    ub_rows: list[np.ndarray] = []
    b_ub: list[float] = []
    # Carey relaxation, one row per affine piece: e_a(t) - s*x_a(t) <= c
    for a in range(n_l):
        for t in range(n_t):
            for slope, icpt in scenario.exit_pieces[a]:
                row = np.zeros(nvar)
                row[ie(a, t)] = 1.0
                if t >= 1:
                    row[ix(a, t)] = -slope
                ub_rows.append(row)
                b_ub.append(icpt)

    return c, np.array(eq_rows), np.array(b_eq), np.array(ub_rows), np.array(b_ub)


def solve_so_dta(scenario: SODTAScenario) -> DTATrajectory:
    """Solve the canonical M-N LP (HiGHS) and emit the optimal trajectory with
    its dual certificate. Raises ``ValueError`` if the LP is infeasible — the
    horizon is too short to clear the demand."""
    c, a_eq, b_eq, a_ub, b_ub = canonical_lp(scenario)
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, method="highs")
    if res.status != 0:
        raise ValueError(
            f"SO-DTA LP failed for '{scenario.name}' (status {res.status}: {res.message}); "
            "an infeasible LP means the horizon T cannot clear the demand"
        )
    n_l, n_t = scenario.n_links, scenario.n_periods
    x = res.x[: n_l * n_t].reshape(n_l, n_t).T  # (T, L): x(1..T)
    u = res.x[n_l * n_t : 2 * n_l * n_t].reshape(n_l, n_t).T
    e = res.x[2 * n_l * n_t :].reshape(n_l, n_t).T
    occ = np.vstack([np.zeros((1, n_l)), x])  # x(0) = 0
    return DTATrajectory(
        scenario_hash=scenario.content_hash(),
        inflows=u,
        exits=e,
        occupancies=occ,
        duals={"eq": res.eqlin.marginals, "ub": res.ineqlin.marginals},
        provenance={"objective": float(res.fun)},
    )
