"""Ziliaskopoulos (2000) single-destination LP SO-DTA on CTM cells
(docs/design/adr-021-lp-so-dta.md).

The system-optimal DTA as a linear program over cell-transmission dynamics:
cells ``i`` carry occupancies ``x_i(t)`` (vehicles at the start of interval
``t``), connectors ``(i, j)`` carry flows ``y_ij(t)``, and the CTM's Godunov
flux EQUALITY ``y = min(x_i, Q_i, Q_j, delta_j (N_j - x_j))`` is RELAXED to its
four linear ``<=`` constraints — conservation stays an equality. The LP's
feasible set therefore strictly contains every CTM trajectory (its optimum
lower-bounds any CTM-realizable total system travel time), and slack in all
four bounds at once is LP "traffic holding" — for a single destination a
non-holding optimum always exists (the earliest-arrival-flow property, Zheng &
Chiu 2011), so the bound is tight; holding shows up only on the optimal face.
Merge priorities and diverge turning fractions are deliberately ABSENT: the SO
program chooses them, so the correspondence is to a *controlled* CTM.

The canonical program (uniform aggregate form — per-cell SUM sending and
receiving constraints, which coincides with the per-cell-type constraint lists
on Ziliaskopoulos's network class and remains a safe superset of every
controlled-CTM trajectory in general — it is in fact the TIGHTER encoding:
per-connector rows alone would be a looser relaxation at cells that merge or
diverge):

    min   sum_{t=0}^{T-1} sum_{i != sink} x_i(t)          (veh-intervals)
    s.t.  x_i(0) = x0_i                                    (initial condition)
          x_i(t+1) = x_i(t) + d_i(t) + sum_in y - sum_out y   (conservation)
          sum_{j} y_ij(t) <= x_i(t)                        (sending occupancy)
          sum_{j} y_ij(t) <= Q_i          [Q_i finite]     (sending capacity)
          sum_{k} y_ki(t) <= Q_i          [Q_i finite]     (receiving capacity)
          sum_{k} y_ki(t) <= delta_i (N_i - x_i(t))  [N_i finite]  (spillback)
          x_i(T) = 0 for i != sink                         (terminal clearance)
          x, y >= 0.

Sources have no predecessors, and cells receiving exogenous demand must have
infinite storage (the origin queue is unbounded; validation enforces this —
demand bypasses the receiving-space rows, so a finite-N source would let the
LP overfill where the certifier censors). Queue waiting is costed: demand
injected during interval ``t`` joins the source occupancy at ``t+1`` and is
costed from ``t+1`` on (the injection interval itself is uncosted, per the
conservation convention), while ``initial_occupancy`` is costed from ``t=0``.
The single sink absorbs (no outflow variables) and is not costed.
``delta = w/v <= 1`` plus the spillback row give ``x <= N`` by induction (for
connector-fed cells) — the LP analogue of the CTM's ``w <= vf`` CFL gate.
Terminal clearance is the repo's adr-020 benchmark convention (the primary has
no terminal condition; without one, "total cost" rewards stranding), so an
infeasible LP means the horizon cannot clear the demand. This differs from the
Merchant-Nemhauser model (``scenario.py``) in the physics: finite storage and
spillback, which exit functions with uncapacitated inflow cannot represent.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

__all__ = [
    "CellSODTAScenario",
    "CellTrajectory",
    "cell_canonical_lp",
    "solve_cell_so_dta",
]


def _as_f64(x) -> np.ndarray:
    return np.array(x, dtype=np.float64, order="C")


def _as_i64(x) -> np.ndarray:
    return np.array(x, dtype=np.int64, order="C")


@dataclass(frozen=True)
class CellSODTAScenario:
    """Frozen, content-hashed Ziliaskopoulos cell instance (P2).

    Cells are ``0..n_cells-1``; connector ``c`` runs ``conn_tail[c] ->
    conn_head[c]``. ``capacity[i] = Q_i`` (vehicles per interval, ``inf``
    allowed), ``storage[i] = N_i`` (vehicles, ``inf`` allowed), ``delta[i] =
    w_i/v_i in (0, 1]``. ``sink`` is the single absorbing destination.
    ``demand[t, i]`` (vehicles entering source ``i`` during interval ``t``)
    sets the horizon ``T``; ``initial_occupancy[i]`` is the ``t = 0`` state
    (defaults to an empty network). ``family`` is P7 lineage (unhashed)."""

    name: str
    n_cells: int
    sink: int
    conn_tail: np.ndarray  # (n_conns,) int64
    conn_head: np.ndarray  # (n_conns,)
    capacity: np.ndarray  # (n_cells,) float64, Q_i > 0, inf allowed
    storage: np.ndarray  # (n_cells,) float64, N_i > 0, inf allowed
    delta: np.ndarray  # (n_cells,) float64 in (0, 1]
    demand: np.ndarray  # (T, n_cells) float64 >= 0
    initial_occupancy: np.ndarray | None = None  # (n_cells,) float64 >= 0
    family: str = field(default="")

    def __post_init__(self) -> None:
        tail = _as_i64(self.conn_tail)
        head = _as_i64(self.conn_head)
        cap = _as_f64(self.capacity)
        sto = _as_f64(self.storage)
        dlt = _as_f64(self.delta)
        dem = _as_f64(self.demand)
        x0 = (
            np.zeros(self.n_cells)
            if self.initial_occupancy is None
            else _as_f64(self.initial_occupancy)
        )
        for attr, val in (
            ("conn_tail", tail),
            ("conn_head", head),
            ("capacity", cap),
            ("storage", sto),
            ("delta", dlt),
            ("demand", dem),
            ("initial_occupancy", x0),
        ):
            object.__setattr__(self, attr, val)
        if not self.family:
            object.__setattr__(self, "family", self.name)

        name = self.name
        n_c = self.n_cells
        if n_c < 2:
            raise ValueError(f"CellSODTAScenario '{name}': need >= 2 cells")
        if not (0 <= self.sink < n_c):
            raise ValueError(f"CellSODTAScenario '{name}': sink out of range")
        n_e = tail.size
        if head.size != n_e or n_e == 0:
            raise ValueError(f"CellSODTAScenario '{name}': need >= 1 connector")
        ends = np.concatenate([tail, head])
        if ends.min() < 0 or ends.max() >= n_c:
            raise ValueError(f"CellSODTAScenario '{name}': connector endpoints out of range")
        if np.any(tail == head):
            raise ValueError(f"CellSODTAScenario '{name}': self-loop connectors not allowed")
        if np.any(tail == self.sink):
            raise ValueError(
                f"CellSODTAScenario '{name}': connectors out of the sink are not "
                "allowed (the sink is absorbing)"
            )
        pairs = set(zip(tail.tolist(), head.tolist(), strict=True))
        if len(pairs) != n_e:
            raise ValueError(f"CellSODTAScenario '{name}': duplicate connectors")
        for label, arr in (("capacity", cap), ("storage", sto)):
            if np.any(np.isnan(arr)) or np.any(arr <= 0.0):
                raise ValueError(
                    f"CellSODTAScenario '{name}': {label} must be > 0 (inf allowed)"
                )
        if not np.all(np.isfinite(dlt)) or np.any(dlt <= 0.0) or np.any(dlt > 1.0):
            raise ValueError(
                f"CellSODTAScenario '{name}': delta = w/v must lie in (0, 1] — the "
                "no-overfill induction (the LP analogue of the CTM w <= vf gate) "
                "needs it"
            )
        if dem.ndim != 2 or dem.shape[1] != n_c or dem.shape[0] < 1:
            raise ValueError(f"CellSODTAScenario '{name}': demand must be (T, n_cells)")
        if not np.all(np.isfinite(dem)) or np.any(dem < 0.0):
            raise ValueError(f"CellSODTAScenario '{name}': demand must be finite and >= 0")
        if x0.shape != (n_c,) or not np.all(np.isfinite(x0)) or np.any(x0 < 0.0):
            raise ValueError(
                f"CellSODTAScenario '{name}': initial_occupancy must be (n_cells,), "
                "finite and >= 0"
            )
        if np.any(x0 > sto):
            raise ValueError(f"CellSODTAScenario '{name}': initial_occupancy exceeds storage")
        if x0[self.sink] != 0.0 or np.any(dem[:, self.sink] != 0.0):
            raise ValueError(
                f"CellSODTAScenario '{name}': the sink takes no demand or initial "
                "occupancy"
            )
        has_in = np.zeros(n_c, dtype=bool)
        has_in[head] = True
        loaded = (dem.sum(axis=0) + x0) > 0.0
        if np.any(loaded & has_in):
            raise ValueError(
                f"CellSODTAScenario '{name}': source cells (with demand or initial "
                "occupancy) must have no incoming connectors"
            )
        # Exogenous demand bypasses the receiving-space rows (they exist only
        # for cells with predecessors), so a pulse above N would overfill a
        # finite-storage source INSIDE the LP while the certifier's x <= N
        # envelope censors every mass-conserving trajectory — the adversarial
        # review caught the LP and the certifier disagreeing about the feasible
        # set. The origin queue is unbounded, exactly as documented.
        if np.any((dem.sum(axis=0) > 0.0) & np.isfinite(sto)):
            raise ValueError(
                f"CellSODTAScenario '{name}': cells receiving exogenous demand must "
                "have infinite storage (the origin queue is unbounded; a demand "
                "pulse above N would overfill the cell outside the LP's receiving "
                "rows). Cells loaded only via initial_occupancy <= N are fine."
            )
        if float(dem.sum() + x0.sum()) <= 0.0:
            raise ValueError(f"CellSODTAScenario '{name}': total demand must be > 0")
        # every loaded cell must reach the sink
        reaches = np.zeros(n_c, dtype=bool)
        reaches[self.sink] = True
        into: dict[int, list[int]] = {}
        for c in range(n_e):
            into.setdefault(int(head[c]), []).append(int(tail[c]))
        frontier = [self.sink]
        while frontier:
            j = frontier.pop()
            for i in into.get(j, ()):
                if not reaches[i]:
                    reaches[i] = True
                    frontier.append(i)
        bad = np.nonzero(loaded)[0]
        bad = bad[~reaches[bad]]
        if bad.size:
            raise ValueError(
                f"CellSODTAScenario '{name}': loaded cells {bad.tolist()} cannot "
                "reach the sink"
            )
        for arr in (tail, head, cap, sto, dlt, dem, x0):
            arr.flags.writeable = False

    @property
    def n_conns(self) -> int:
        return self.conn_tail.size

    @property
    def n_periods(self) -> int:
        return self.demand.shape[0]

    def content_hash(self) -> str:
        """SHA-256 over the canonical serialization, domain-separated
        (``"tabench-dta-cell-scenario-v1;"`` prefix); ``inf`` hashes as its
        IEEE-754 pattern."""
        h = hashlib.sha256()
        h.update(b"tabench-dta-cell-scenario-v1;")
        h.update(f"cells={self.n_cells};sink={self.sink};".encode())
        for label, arr in (
            ("tail", self.conn_tail),
            ("head", self.conn_head),
            ("Q", self.capacity),
            ("N", self.storage),
            ("delta", self.delta),
            ("demand", self.demand),
            ("x0", self.initial_occupancy),
        ):
            h.update(label.encode())
            h.update(_as_f64(arr).tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class CellTrajectory:
    """Emitted plan: ``occupancies`` is ``(T+1, n_cells)`` start-of-interval
    states (row 0 = the initial condition), ``flows`` is ``(T, n_conns)``
    connector flows. ``duals`` is the optional LP dual certificate in
    :func:`cell_canonical_lp`'s row order; ``provenance`` is never certified."""

    scenario_hash: str
    occupancies: np.ndarray  # (T+1, n_cells) float64
    flows: np.ndarray  # (T, n_conns)
    duals: dict[str, np.ndarray] | None = None
    provenance: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for attr in ("occupancies", "flows"):
            arr = np.ascontiguousarray(getattr(self, attr), dtype=np.float64)
            object.__setattr__(self, attr, arr)
        if self.occupancies.ndim != 2 or self.flows.ndim != 2:
            raise ValueError("CellTrajectory occupancies/flows must be 2-D")
        if self.occupancies.shape[0] != self.flows.shape[0] + 1:
            raise ValueError("CellTrajectory occupancies must have T+1 rows to flows' T")
        if self.duals is not None:
            duals = {
                k: np.ascontiguousarray(v, dtype=np.float64) for k, v in self.duals.items()
            }
            if set(duals) != {"eq", "ub"}:
                raise ValueError('CellTrajectory duals must have exactly keys "eq" and "ub"')
            object.__setattr__(self, "duals", duals)


def cell_canonical_lp(
    scenario: CellSODTAScenario,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(c, A_eq, b_eq, A_ub, b_ub)`` for the canonical cell LP.

    Variable layout (all >= 0): ``x_i(t)`` for ``t = 0..T`` at ``i*(T+1) + t``;
    ``y_c(t)`` for ``t = 0..T-1`` at ``n_cells*(T+1) + c*T + t``. Equality-row
    order: initial condition per cell ascending, conservation ``(i, t)``
    cell-major, terminal clearance per non-sink cell ascending. Inequality-row
    order: for ``t = 0..T-1``, for each cell ascending — sending-occupancy
    (cells with successors), sending-capacity (finite ``Q``, successors),
    receiving-capacity (finite ``Q``, predecessors), receiving-space (finite
    ``N``, predecessors). This fixed ordering is part of the artifact contract
    — an emitted dual certificate refers to it."""
    n_c, n_e, n_t = scenario.n_cells, scenario.n_conns, scenario.n_periods
    tail, head = scenario.conn_tail, scenario.conn_head
    nvar = n_c * (n_t + 1) + n_e * n_t

    def ix(i: int, t: int) -> int:  # x_i(t), t = 0..T
        return i * (n_t + 1) + t

    def iy(c: int, t: int) -> int:  # y_c(t), t = 0..T-1
        return n_c * (n_t + 1) + c * n_t + t

    outs = [np.nonzero(tail == i)[0] for i in range(n_c)]
    ins = [np.nonzero(head == i)[0] for i in range(n_c)]

    c_vec = np.zeros(nvar)
    for i in range(n_c):
        if i != scenario.sink:
            c_vec[ix(i, 0) : ix(i, n_t - 1) + 1] = 1.0  # x_i(0..T-1)

    eq_rows: list[np.ndarray] = []
    b_eq: list[float] = []
    for i in range(n_c):  # initial condition
        row = np.zeros(nvar)
        row[ix(i, 0)] = 1.0
        eq_rows.append(row)
        b_eq.append(float(scenario.initial_occupancy[i]))
    for i in range(n_c):  # conservation
        for t in range(n_t):
            row = np.zeros(nvar)
            row[ix(i, t + 1)] = 1.0
            row[ix(i, t)] = -1.0
            for c in ins[i]:
                row[iy(int(c), t)] = -1.0
            for c in outs[i]:
                row[iy(int(c), t)] = 1.0
            eq_rows.append(row)
            b_eq.append(float(scenario.demand[t, i]))
    for i in range(n_c):  # terminal clearance
        if i != scenario.sink:
            row = np.zeros(nvar)
            row[ix(i, n_t)] = 1.0
            eq_rows.append(row)
            b_eq.append(0.0)

    ub_rows: list[np.ndarray] = []
    b_ub: list[float] = []
    for t in range(n_t):
        for i in range(n_c):
            if outs[i].size:
                row = np.zeros(nvar)  # sum_out y <= x_i(t)
                for c in outs[i]:
                    row[iy(int(c), t)] = 1.0
                row[ix(i, t)] = -1.0
                ub_rows.append(row)
                b_ub.append(0.0)
                if np.isfinite(scenario.capacity[i]):  # sum_out y <= Q_i
                    row = np.zeros(nvar)
                    for c in outs[i]:
                        row[iy(int(c), t)] = 1.0
                    ub_rows.append(row)
                    b_ub.append(float(scenario.capacity[i]))
            if ins[i].size:
                if np.isfinite(scenario.capacity[i]):  # sum_in y <= Q_i
                    row = np.zeros(nvar)
                    for c in ins[i]:
                        row[iy(int(c), t)] = 1.0
                    ub_rows.append(row)
                    b_ub.append(float(scenario.capacity[i]))
                if np.isfinite(scenario.storage[i]):  # sum_in y <= delta (N - x)
                    row = np.zeros(nvar)
                    for c in ins[i]:
                        row[iy(int(c), t)] = 1.0
                    row[ix(i, t)] = float(scenario.delta[i])
                    ub_rows.append(row)
                    b_ub.append(float(scenario.delta[i] * scenario.storage[i]))

    return c_vec, np.array(eq_rows), np.array(b_eq), np.array(ub_rows), np.array(b_ub)


def solve_cell_so_dta(scenario: CellSODTAScenario) -> CellTrajectory:
    """Solve the canonical cell LP (HiGHS) and emit the optimal trajectory with
    its dual certificate. Raises ``ValueError`` if the LP is infeasible — the
    horizon cannot clear the demand."""
    c, a_eq, b_eq, a_ub, b_ub = cell_canonical_lp(scenario)
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, method="highs")
    if res.status != 0:
        raise ValueError(
            f"cell SO-DTA LP failed for '{scenario.name}' (status {res.status}: "
            f"{res.message}); an infeasible LP means the horizon T cannot clear "
            "the demand"
        )
    n_c, n_e, n_t = scenario.n_cells, scenario.n_conns, scenario.n_periods
    occ = res.x[: n_c * (n_t + 1)].reshape(n_c, n_t + 1).T  # (T+1, C)
    flows = res.x[n_c * (n_t + 1) :].reshape(n_e, n_t).T  # (T, E)
    return CellTrajectory(
        scenario_hash=scenario.content_hash(),
        occupancies=occ,
        flows=flows,
        duals={"eq": res.eqlin.marginals, "ub": res.ineqlin.marginals},
        provenance={"objective": float(res.fun)},
    )
