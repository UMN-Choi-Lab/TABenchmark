"""Implicit-neural-network user equilibrium (Liu et al. 2023), a lean torch variant.

Liu, Yin, Bai & Grimm (2023, *Transportation Research Part C* 150:104085) learn
user equilibrium *end to end*: a neural network parameterizes the travelers'
link/route cost function and Wardrop UE is imposed as a parametrized variational
inequality embedded as an **implicit layer** — the forward pass iterates a
closed-form equilibrating fixed point and the backward pass differentiates
through the equilibrium condition itself (implicit function theorem), so the
equilibrium is *architectural*, not a supervised loss term. Training matches the
layer's equilibrium *flows* to observed flows; because the output is always an
equilibrium of *some* learned cost, it is demand-feasible by construction — the
property the shipped per-link ridge surrogate (``learned-surrogate``) throws
away and is censored for.

This module ships a **lean variant** of that method, flagged the way
``dtd-stochastic`` flags its filter variant. The primary TR-C article and both
SSRN preprints are paywalled/bot-blocked and were attributed unread; the
mechanism was recovered and cross-verified from the authors' own open sources
(the hEART 2024 unified-framework paper and two author posters — see
docs/design/adr-025). Concretely:

* **Learned cost head** — a small MLP produces a per-link, flow-*monotone* cost
  correction added to the true BPR latency. Monotonicity in flow is
  architectural (a ``softplus`` nonnegative slope on the dimensionless flow
  channel ``v/cap``, gated by a nonnegative output gain), so the smoothed
  equilibrium stays unique — the logit fixed point is the optimum of a strictly
  convex entropy program when route costs are nondecreasing (Fisk 1980).
  Uniqueness does NOT make the *iteration* convergent by itself: a constant
  step can limit-cycle on strongly congested power-4 networks (adr-025 review),
  which is why the solver uses adaptive damping (step halved whenever the
  residual rises) and reports the residual measured at the emitted iterate.
  The MLP reads only the per-link dimensionless features ``[fft, cap, b, power]``
  (standardized across links, so it is network-size-agnostic and transfers
  across topologies — the paper's "kernel strategy"); the flow channel enters
  only through the guaranteed-nonnegative slope. With the parameters zeroed the
  gain is zero and the correction vanishes, recovering a plain logit loading at
  the true costs.
* **Implicit layer** — over PathEngine column-generated per-OD route sets, the
  layer solves the logit route-choice fixed point
  ``h* = D_od · softmax_od(-beta · c_theta(v(h*)))`` with ``v(h) = Delta^T h`` by
  a damped iteration. Every emission is ``v = Delta^T h*`` with each OD's route
  flows summing to its demand, so node balance is exact **by construction** and
  the FIXED-demand feasibility audit (metrics/gaps.py, P1/P7) always passes; on
  endogenous-demand tasks (elastic/combined) censoring reflects demand
  consistency, exactly as for every fixed-demand solver (adr-025 review).
* **Hypergradient** — the training gradient is the implicit-differentiation
  (IMD/adjoint) hypergradient ``dL/dtheta = (dg/dtheta)^T nu`` with the adjoint
  ``nu = (I - dg/dh)^{-T} dL/dh`` solved exactly as a small dense linear system;
  torch autograd supplies the Jacobian columns and the final vector-Jacobian
  product, so no unrolled forward graph is stored (a dense solve on the small
  Braess/synthetic route sets is robust where a Neumann series would diverge on a
  stiff logit map). It matches central finite differences of the full solve
  (anchor A2).

Trained offline on the synthetic-net family reused from ``learned.py`` against
:class:`~tabench.models.frank_wolfe.BiconjugateFrankWolfeModel` reference
equilibria under the *true* BPR costs, evaluated on the disjoint TNTP scenarios
under the ``trained_on`` fairness gate. The harness recomputes the equilibrium
gap under the true costs from the emitted flows (P1); that certified gap isolates
the learned-cost error and is expected to lose to a converged solver at matched
budget — the honest headline (docs/design/adr-025). The one-time training budget
is reported as ``training_sp_calls``/``training_wall_ms`` provenance (P6), never a
score. No weights are committed: training runs at solve time on the tiny fixture
under a fixed internal seed and a module-level cache (the ``learned.py``
precedent).

torch is an optional dependency (``pip install tabench[torch]``); this module is
import-guarded in ``models/__init__.py`` so the numpy/scipy core stays torch-free.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from ..core.budget import Budget, BudgetCoords
from ..core.capabilities import Capabilities
from ..core.results import ResultBundle, Trace
from ..core.rng import RngBundle
from ..core.scenario import Demand, Network, Scenario
from ._paths import PathEngine
from .base import TrafficAssignmentModel, register_model
from .frank_wolfe import BiconjugateFrankWolfeModel
from .learned import _TRAINING_SCENARIOS, TRAINING_FAMILY

__all__ = ["ImplicitUENNModel", "TRAINING_FAMILY"]

# --- fixed-point / layer hyperparameters (named so the CI wall-time budget is a
#     visible design commitment, not an accident of tuning) -------------------
_LOGIT_BETA = 0.1  # route-choice inverse temperature (1/native cost unit)
_N_CG = 6  # column-generation rounds = Dijkstra sweeps counted as sp_calls
_N_FP_ITER = 3000  # damped logit fixed-point iterations (the `iterations` coord)
_FP_DAMPING = 0.1  # initial step; halved adaptively whenever the residual rises
_FP_OMEGA_MIN = 1e-3  # damping floor so progress never stalls entirely
_FP_TOL = 1e-11  # early-stop once the fixed-point residual falls below this
_HIDDEN = 8  # cost-head MLP hidden width
_DTYPE = torch.float64  # anchors need float64 (FD match to 1e-5)

# --- training hyperparameters (training wall-time budget < 60 s CPU) ----------
_TRAIN_SEED = 20230  # FIXED internal seed (seedable=False; not the harness RNG)
_TRAIN_EPOCHS = 60
_TRAIN_LR = 0.05
# Weight decay keeps the learned cost correction small and transferable: the
# synthetic family and the TNTP test nets sit in different congestion regimes,
# so an unregularized fit overfits the training family and degrades the held-out
# certified gap (the identifiability caveat, docs/design/adr-025).
_TRAIN_WEIGHT_DECAY = 0.1
_REF_BUDGET = Budget(iterations=200, target_relative_gap=1e-7)


class _CostHead(nn.Module):
    """Per-link flow-monotone BPR cost correction (the learned component).

    The correction is ``relu(gain) * softplus(mlp(static_features)) * (v/cap)``:
    nonnegative and increasing in flow for *every* parameter value (architectural
    monotonicity), and identically zero when all parameters are zero (so a zeroed
    head reduces the layer to a plain logit loading at the true BPR costs).
    ``static_features`` are the standardized per-link ``[fft, cap, b, power]`` —
    network-size-agnostic, so one trained head transfers across topologies.
    """

    def __init__(self, hidden: int = _HIDDEN) -> None:
        super().__init__()
        self.l1 = nn.Linear(4, hidden, dtype=_DTYPE)
        self.l2 = nn.Linear(hidden, 1, dtype=_DTYPE)
        # Nonnegative output gain, initialized positive so relu is live at init;
        # zeroing it (or all params) removes the correction entirely.
        self.gain = nn.Parameter(torch.ones((), dtype=_DTYPE))

    def slope(self, static: torch.Tensor) -> torch.Tensor:
        return self.l2(torch.tanh(self.l1(static))).squeeze(-1)

    def correction(self, ratio: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.gain) * nn.functional.softplus(self.slope(static)) * ratio


def _torch_network(network: Network) -> dict[str, torch.Tensor]:
    """BPR parameters + standardized static features as float64 CPU tensors."""
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=_DTYPE)  # noqa: E731
    static_raw = np.column_stack(
        [network.free_flow_time, network.capacity, network.b, network.power]
    )
    mu = static_raw.mean(axis=0)
    sigma = static_raw.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    return {
        "fft": t(network.free_flow_time),
        "cap": t(network.capacity),
        "b": t(network.b),
        "power": t(network.power),
        "fixed": t(network.fixed_cost),
        "static": t((static_raw - mu) / sigma),
    }


def _link_cost_torch(head: _CostHead, v: torch.Tensor, net: dict) -> torch.Tensor:
    ratio = v / net["cap"]
    bpr = net["fft"] * (1.0 + net["b"] * ratio ** net["power"]) + net["fixed"]
    return bpr + head.correction(ratio, net["static"])


class _RouteSet:
    """Column-generated per-OD route sets as a dense path-link incidence Delta.

    ``delta`` is ``(n_routes, n_links)`` with ``delta[r, a] = 1`` iff link ``a``
    lies on route ``r``; ``od_index`` maps each route to its OD group; ``demand``
    is the per-group OD demand. Built at the *true* BPR costs (theta-independent),
    so the same set is used for training and inference and is fully deterministic.
    """

    def __init__(
        self, delta: torch.Tensor, od_index: torch.Tensor, demand: torch.Tensor, n_cg: int
    ) -> None:
        self.delta = delta
        self.od_index = od_index
        self.demand = demand
        self.n_groups = int(demand.shape[0])
        self.sp_calls = n_cg


def _numpy_logit_load(
    delta: np.ndarray, od_index: np.ndarray, demand: np.ndarray, network: Network
) -> np.ndarray:
    """Plain logit loading at the TRUE BPR costs over a fixed route set (numpy).

    Used only to diversify routes between column-generation rounds; the learned
    correction never enters here."""
    n_routes = delta.shape[0]
    h = np.zeros(n_routes)
    for g in range(len(demand)):
        mask = od_index == g
        h[mask] = demand[g] / max(1, int(mask.sum()))
    for _ in range(60):
        v = delta.T @ h
        c_route = delta @ network.link_cost(v)
        h_new = np.empty_like(h)
        for g in range(len(demand)):
            mask = od_index == g
            z = -_LOGIT_BETA * c_route[mask]
            z -= z.max()
            e = np.exp(z)
            h_new[mask] = demand[g] * e / e.sum()
        if np.abs(h_new - h).max() < 1e-12:
            h = h_new
            break
        h = 0.5 * h + 0.5 * h_new
    return h


def _build_routes(
    network: Network,
    demand: Demand,
    engine: PathEngine,
    n_cg: int,
    deadline: float | None = None,
) -> _RouteSet:
    """``n_cg`` rounds of column generation at the true costs (each a Dijkstra sweep).

    Callers guarantee at least one routable (off-diagonal, positive) OD pair —
    ``solve`` short-circuits the no-demand case with the exact zero emission
    (adr-025 review). ``deadline`` stops further rounds once the ``wall_seconds``
    budget is spent (at least one round always runs)."""
    od = demand.matrix
    keys = [
        (int(o), int(d))
        for o in range(od.shape[0])
        for d in range(od.shape[1])
        if o != d and od[o, d] > 0
    ]
    group_of = {k: g for g, k in enumerate(keys)}
    routes: dict[tuple[int, int], list[tuple[int, ...]]] = {k: [] for k in keys}
    v = np.zeros(network.n_links)
    n_done = 0
    for round_idx in range(n_cg):
        if deadline is not None and round_idx > 0 and time.perf_counter() >= deadline:
            break
        paths, _ = engine.shortest_paths(network.link_cost(v), demand)
        n_done += 1
        for key, links in paths.items():
            tup = tuple(int(a) for a in links)
            if tup not in routes[key]:
                routes[key].append(tup)
        rows, od_index = [], []
        for key in keys:
            for links in routes[key]:
                row = np.zeros(network.n_links)
                row[list(links)] = 1.0
                rows.append(row)
                od_index.append(group_of[key])
        delta_np = np.asarray(rows)
        idx_np = np.asarray(od_index, dtype=np.int64)
        demand_np = np.asarray([od[o, d] for (o, d) in keys], dtype=np.float64)
        v = delta_np.T @ _numpy_logit_load(delta_np, idx_np, demand_np, network)
    return _RouteSet(
        torch.as_tensor(delta_np, dtype=_DTYPE),
        torch.as_tensor(idx_np, dtype=torch.int64),
        torch.as_tensor(demand_np, dtype=_DTYPE),
        n_done,  # sweeps actually run (deadline can stop early)
    )


def _segment_softmax_load(z: torch.Tensor, rs: _RouteSet) -> torch.Tensor:
    """``D_od * softmax_od(z)`` over the per-OD route groups (demand-conserving)."""
    gmax = torch.full((rs.n_groups,), float("-inf"), dtype=_DTYPE)
    gmax = gmax.scatter_reduce(0, rs.od_index, z, reduce="amax", include_self=False)
    e = torch.exp(z - gmax[rs.od_index])
    denom = torch.zeros(rs.n_groups, dtype=_DTYPE).index_add(0, rs.od_index, e)
    return rs.demand[rs.od_index] * e / denom[rs.od_index]


def _layer_map(head: _CostHead, h: torch.Tensor, rs: _RouteSet, net: dict) -> torch.Tensor:
    """One application of the logit route-choice fixed-point operator g(theta, h)."""
    v = rs.delta.t() @ h
    c_route = rs.delta @ _link_cost_torch(head, v, net)
    return _segment_softmax_load(-_LOGIT_BETA * c_route, rs)


def _solve_fixed_point(
    head: _CostHead,
    rs: _RouteSet,
    net: dict,
    n_iter: int,
    deadline: float | None = None,
) -> tuple[torch.Tensor, float, int]:
    """Adaptively damped iteration of the logit layer map to equilibrium (no grad).

    The step ``h <- h + omega (g(h) - h)`` starts at ``_FP_DAMPING`` and HALVES
    whenever the residual increases (floored at ``_FP_OMEGA_MIN``) — a constant
    step limit-cycles on strongly congested power-4 networks (the repo's
    recurring fixed-point defect, confirmed by the adr-025 review); the adaptive
    halving damps any cycle monotonically. The step scheme is immaterial to the
    hypergradient, which differentiates the fixed-point equation, not the
    iteration. ``deadline`` (a ``time.perf_counter()`` instant) enforces the
    ``wall_seconds`` budget axis mid-loop. Returns the iterate, the residual
    ``max|g(h) - h|`` measured AT the returned iterate (one extra map
    application — never a stale mid-loop value), and the number of steps
    actually executed (the honest P6 ``iterations`` coordinate). Whatever the
    residual, the emission stays demand-feasible by construction."""
    with torch.no_grad():
        counts = torch.zeros(rs.n_groups, dtype=_DTYPE).index_add(
            0, rs.od_index, torch.ones_like(rs.od_index, dtype=_DTYPE)
        )
        h = rs.demand[rs.od_index] / counts[rs.od_index]  # uniform per-OD start
        omega = _FP_DAMPING
        prev_res = float("inf")
        steps = 0
        for _ in range(n_iter):
            y = _layer_map(head, h, rs, net)
            residual = float((y - h).abs().max())
            if residual < _FP_TOL:
                break
            if residual > prev_res:
                omega = max(0.5 * omega, _FP_OMEGA_MIN)
            prev_res = residual
            h = h + omega * (y - h)
            steps += 1
            if deadline is not None and time.perf_counter() >= deadline:
                break
        # residual of the RETURNED iterate (the value the trace reports)
        residual = float((_layer_map(head, h, rs, net) - h).abs().max())
    return h, residual, steps


def _hypergradient(
    head: _CostHead, rs: _RouteSet, net: dict, v_obs: torch.Tensor, scale: float
) -> tuple[float, list[torch.Tensor]]:
    """IMD/adjoint hypergradient of ``0.5||Delta^T h*(theta) - v_obs||^2 / scale``.

    Solves the fixed point without grad, then obtains the adjoint
    ``nu = (I - dg/dh)^{-T} dL/dh`` by an EXACT dense linear solve, using autograd
    VJPs of a single differentiable application of the layer map at the (detached)
    fixed point for the Jacobian columns, and returns ``dL/dtheta = (dg/dtheta)^T
    nu`` per parameter. No unrolled forward graph is stored (the paper's
    implicit-function-theorem gradient)."""
    params = list(head.parameters())
    h_star, _, _ = _solve_fixed_point(head, rs, net, _N_FP_ITER)
    v = rs.delta.t() @ h_star
    loss = 0.5 * ((v - v_obs) ** 2).sum() / scale
    dL_dh = (rs.delta @ (v - v_obs)) / scale  # dL/dh at the fixed point

    h0 = h_star.detach().clone().requires_grad_(True)
    g0 = _layer_map(head, h0, rs, net)  # differentiable one-step map at h*
    # Adjoint nu = (I - dg/dh^T)^{-1} dL/dh, solved EXACTLY as a dense linear
    # system. The route sets here (Braess + the synthetic training family) are
    # small, so this is cheap and — unlike a Neumann/Richardson series — robust
    # to the logit map's Jacobian having a large spectral radius on congested
    # power-4 nets (where the series would diverge). Columns of dg/dh^T come from
    # autograd VJPs; this is the implicit-function-theorem gradient (no unrolled
    # graph), matching central finite differences to ~1e-9 at a conditioned point.
    n = h_star.shape[0]
    eye = torch.eye(n, dtype=_DTYPE)
    jt_cols = [
        torch.autograd.grad(g0, h0, grad_outputs=eye[i], retain_graph=True)[0] for i in range(n)
    ]
    jt = torch.stack(jt_cols, dim=1)  # jt[:, i] = dg/dh^T e_i  ->  jt = dg/dh^T
    nu = torch.linalg.solve(eye - jt, dL_dh)
    grads = torch.autograd.grad(g0, params, grad_outputs=nu, retain_graph=False)
    return float(loss), [g.detach() for g in grads]


#: cached (_CostHead, {"sp_calls", "wall_ms"}) — the one-time offline training cost.
_TRAINED: tuple[_CostHead, dict[str, float]] | None = None


def _train() -> tuple[_CostHead, dict[str, float]]:
    """Fit the cost head on BFW equilibria of the synthetic family (cached).

    Deterministic: fixed internal seed, single-threaded, deterministic torch
    algorithms, deterministic bi-conjugate FW. Returns the trained head and the
    one-time offline training budget (summed shortest-path calls and wall time),
    reported as provenance rather than hidden (docs/ARCHITECTURE.md P6)."""
    global _TRAINED
    if _TRAINED is not None:
        return _TRAINED

    prev_threads = torch.get_num_threads()
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_rng = torch.get_rng_state()  # never clobber the process-global default
    torch.use_deterministic_algorithms(True)  # generator (adr-025 review MINOR)
    torch.set_num_threads(1)
    torch.manual_seed(_TRAIN_SEED)
    try:
        solver = BiconjugateFrankWolfeModel()
        cases, sp_calls = [], 0
        start = time.perf_counter()
        for scenario in _TRAINING_SCENARIOS:
            engine = PathEngine(scenario.network)
            trace = Trace()
            solver.solve(scenario, _REF_BUDGET, RngBundle(0), trace)
            sp_calls += trace.final.coords.sp_calls
            rs = _build_routes(scenario.network, scenario.demand, engine, _N_CG)
            sp_calls += rs.sp_calls
            net = _torch_network(scenario.network)
            v_obs = torch.as_tensor(trace.final.link_flows, dtype=_DTYPE)
            scale = max(1.0, float(scenario.demand.total))
            cases.append((rs, net, v_obs, scale))

        head = _CostHead()
        optimizer = torch.optim.Adam(
            head.parameters(), lr=_TRAIN_LR, weight_decay=_TRAIN_WEIGHT_DECAY
        )
        for _ in range(_TRAIN_EPOCHS):
            optimizer.zero_grad()
            for rs, net, v_obs, scale in cases:
                _, grads = _hypergradient(head, rs, net, v_obs, scale)
                for p, g in zip(head.parameters(), grads, strict=True):
                    p.grad = g if p.grad is None else p.grad + g
            optimizer.step()
        head.eval()
        stats = {"sp_calls": float(sp_calls), "wall_ms": 1000.0 * (time.perf_counter() - start)}
        _TRAINED = (head, stats)
    finally:
        torch.use_deterministic_algorithms(prev_det)
        torch.set_num_threads(prev_threads)
        torch.set_rng_state(prev_rng)
    return _TRAINED


@register_model
class ImplicitUENNModel(TrafficAssignmentModel):
    """Implicit-NN user equilibrium (Liu et al. 2023, lean variant; paradigm ``learned``).

    Emits ``v = Delta^T h*`` from the logit route-choice fixed point under a learned
    flow-monotone cost head, over PathEngine column-generated route sets. Demand-
    feasible by construction (unlike ``learned-surrogate``), so it clears the
    harness feasibility audit and earns a real certified gap under the true costs.
    Deterministic (fixed internal training seed); refused on ``synthetic-net``
    scenarios by the ``trained_on`` fairness gate."""

    name = "implicit-ue-nn"
    capabilities = Capabilities(
        paradigm="learned",
        deterministic=True,
        provides_gap=False,
        seedable=False,
        trained_on=(TRAINING_FAMILY,) + tuple(s.content_hash() for s in _TRAINING_SCENARIOS),
    )

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        head, train_cost = _train()  # cached; its cost is reported, not timed here
        start = time.perf_counter()  # measure INFERENCE only (deterministic)
        # Budget respect (P6): the sp_calls axis caps column-generation rounds
        # (each a Dijkstra sweep); the iterations axis caps fixed-point steps;
        # wall_seconds is enforced as a mid-loop deadline (adr-025 review: it
        # was silently ignored while every classical solver checks it).
        n_cg = _N_CG if budget.sp_calls is None else max(1, min(_N_CG, budget.sp_calls))
        n_fp = (
            _N_FP_ITER
            if budget.iterations is None
            else max(1, min(_N_FP_ITER, budget.iterations))
        )
        deadline = None if budget.wall_seconds is None else start + budget.wall_seconds
        od = scenario.demand.matrix
        if not np.any(od[~np.eye(od.shape[0], dtype=bool)] > 0):
            # no routable (off-diagonal) demand: the zero flow is the exact
            # equilibrium; the empty route set used to crash here (adr-025
            # review MAJOR — every classical solver handles this input)
            flows = np.zeros(scenario.network.n_links)
            coords = BudgetCoords(
                iterations=0, sp_calls=1, wall_ms=1000.0 * (time.perf_counter() - start)
            )
            trace.record(
                flows,
                coords,
                fixed_point_residual=0.0,
                training_sp_calls=train_cost["sp_calls"],
                training_wall_ms=train_cost["wall_ms"],
            )
            return ResultBundle(
                model_name=self.name,
                final=trace.final,
                trace=trace,
                factors=dict(self.factor_values),
                seed_info=rng.describe(),
            )
        engine = PathEngine(scenario.network)
        rs = _build_routes(scenario.network, scenario.demand, engine, n_cg, deadline)
        net = _torch_network(scenario.network)
        h_star, residual, steps = _solve_fixed_point(head, rs, net, n_fp, deadline)
        flows = (rs.delta.t() @ h_star).detach().numpy()
        coords = BudgetCoords(
            iterations=steps,  # steps actually executed, not the cap (P6 honesty)
            sp_calls=rs.sp_calls,  # real Dijkstra sweeps (> 1, unlike the ridge surrogate)
            wall_ms=1000.0 * (time.perf_counter() - start),
        )
        trace.record(
            flows,
            coords,
            fixed_point_residual=residual,  # layer truncation, descriptive (P6)
            # One-time offline training budget, reported as provenance (P6).
            training_sp_calls=train_cost["sp_calls"],
            training_wall_ms=train_cost["wall_ms"],
        )
        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
