"""Heterogeneous-GNN traffic assignment (Liu & Meidani 2024), a lean torch variant.

Liu, T. & Meidani, H. (2024, *Transportation Research Part C* 165:104695, DOI
``10.1016/j.trc.2024.104695``) learn user equilibrium as a **supervised surrogate
over a heterogeneous graph**: real road links and virtual OD links are two edge
types over one node set, type-specific attention passes messages, and an edge
MLP reads the final node embeddings to predict each real link's flow/capacity
ratio ``alpha~_a`` (link flow ``f~_a = cap_a * alpha~_a``). Unlike ``implicit-ue-nn``
the equilibrium is NOT architectural: flow conservation enters ONLY as a soft
aggregate node-balance penalty (loss weight ``w_c = 0.05``), so the raw emission
is a per-link regression that routes no demand exactly. This is *act three* of
the learned-model story (docs/design/adr-026): ``learned-surrogate`` has no
conservation and is censored; ``het-gnn`` has soft conservation and is censored
raw but recovers feasibility by an explicit decode; ``implicit-ue-nn`` has
conservation by construction.

This module ships a **lean variant** of that method, flagged the way
``dtd-stochastic`` / ``implicit-ue-nn`` flag their variants. The primary TR-C
article is paywalled and was attributed unread; the formulation was recovered
and cross-verified with zero discrepancies from the authors' own open sources
read in full (arXiv:2310.13193v3 and the NSF PAR accepted manuscript — see
docs/design/adr-026). Concretely:

* **Size-agnostic node features** — the lean substitution for the paper's node
  feature ``x_u in R^(|V|+2)`` (that node's whole OD-demand row plus two
  geographic coordinates). That featurization is NOT node-permutation
  equivariant (the dense preprocessing mixes the node-indexed demand-row axis;
  machine-verified max output change 21.5 under relabeling) and its dimension is
  ``|V|``, which is exactly why the paper needs transfer learning or dummy-node
  "homogenized training" to change graph size (paper Sec 5.2.2). We replace it
  with the intrinsic per-node ``[production_u, attraction_u, out_degree_u,
  in_degree_u]`` (demand terms normalized by total demand, degrees by link
  count) — exactly equivariant and size-agnostic, so ONE trained model runs on
  the 8–14-node synthetic training graphs and the 24–416-node TNTP test graphs
  under the repo's disjoint split. Coordinates are dropped (``Network`` has no
  geometry). This is the node-level analogue of ``implicit-ue-nn``'s per-link
  "kernel strategy": link-kernel there, node-kernel here, both size-agnostic by
  different routes.
* **Heterogeneous message passing** — hand-rolled plain-torch scatter ops
  (``index_add_``/``scatter_reduce``; NO torch-geometric/DGL — decided in
  adr-025). Stacked ``V-Encoder`` layers (attention over virtual OD edges, whose
  adaptive scalar weight ``beta_e = FFN([x_u || x_v])`` is learned from the
  endpoints because virtual edges carry no features — paper Eq 7-8) then
  ``R-Encoder`` layers (attention over real edges, whose weight here is a
  LEARNED affine scalar of the real edge features — a flagged generalization of
  paper Eq 9, whose weight is the FIXED sum of the normalized edge features; the
  feature set is also extended from the paper's ``[fft, cap]`` to
  ``[fft, cap, b, power]`` — adr-026 review). The per-source-node attention
  softmax is a **segment softmax keyed on the tail-node index** — the exact
  ``implicit_ue._segment_softmax_load`` pattern transplanted from OD groups to
  node groups. Single head, small hidden width, float64 (the lean floor that
  still exercises typed heterogeneous passing and adaptive virtual-edge weights;
  the paper's 8 heads / 3-layer preprocessing / LayerNorm towers are not built).
* **Edge head** — ``alpha~_a = MLP([o_u || o_v || e_a])`` from the final node
  embeddings ``o`` and the real edge features ``e_a`` (paper Eq 10);
  ``f~_a = cap_a * alpha~_a``, clipped ``>= 0`` at emission (I/O hygiene).
* **Loss** — the paper's composite ``L = w_a L_a + w_f L_f + w_c L_c`` with
  weights ``(1.0, 0.005, 0.05)`` (Eqs 11-14): mean-absolute ratio error, mean-
  absolute flow error, and the aggregate node-balance residual ``L_c =
  sum_i |sum_in f~ - sum_out f~ - (attr_i - prod_i)|``. That residual vector is
  *identical* to the harness ``metrics.gaps.node_balance_residual`` (the audit
  thresholds its L-inf; the paper reports its L1/D) — so the paper trains toward
  exactly the statistic the harness censors on, and its own best reported values
  land 3–5 orders of magnitude above the ``1e-6`` feasibility tolerance. ``L_f``
  and ``L_c`` are normalized by total demand for cross-scenario scale stability
  (the paper's "OD normalized to 100" is underspecified — a documented
  deviation). Trained by plain backprop (no implicit layer → no hypergradient
  machinery — simpler than adr-025).

**Feasible decode (a REPO EXTENSION — NOT in the paper; flagged here, in adr-026,
and in model-specs.json).** The paper has no feasible decoder; its emission is
the censored raw flow. To place ``het-gnn`` on the same certified-gap axis as the
other models, ``solve`` records a SECOND checkpoint: the raw flows are projected
onto the demand-feasible route polytope by projected gradient of
``||Delta^T h - v_raw||^2`` subject to ``h >= 0`` and ``sum_{r in od} h_r = D_od``
(per-OD scaled-simplex projection) over the SAME ``implicit_ue._build_routes``
column-generated route sets. The emission ``v = Delta^T h`` is demand-feasible by
the identical ``v = Delta^T h`` mechanism ``implicit-ue-nn`` uses, so the two
models' certified-gap comparison isolates "GNN flow regression + projection" vs
"learned-cost fixed point" at matched shortest-path budget. The projected-
gradient step is fixed from the Lipschitz bound of ``Delta Delta^T`` with adaptive
damping (step halved whenever the objective rises — the repo's recurring
fixed-point defect, adr-025 review), and the residual is measured at the emitted
iterate.

``solve`` therefore emits TWO checkpoints, both certified by the harness (P1):
(i) the paper-faithful **raw** emission at ``sp_calls=0, iterations=0`` (the GNN
forward needs no shortest path — a genuinely new budget point below the ridge's
1), which the audit censors ``feasible=0`` with its recomputed node-balance
residual, and (ii) the **decoded** feasible flow at ``sp_calls=n_cg`` with a real
certified gap. Nothing is self-attested; the raw censored row IS the paper's
model, kept visible in the same CSV.

Trained offline on the ``learned.py`` synthetic-net family plus a small fixed-seed
per-OD demand-rescaling augmentation (the paper's Eq 15 recipe), against
:class:`~tabench.models.frank_wolfe.BiconjugateFrankWolfeModel` reference
equilibria at a certified-gap criterion (a documented deviation from the paper's
successive-flow-change Frank-Wolfe labels). Evaluated on the disjoint TNTP
scenarios under the ``trained_on`` fairness gate; every trained-on instance's
content hash (base + augmentations) is declared. No weights are committed:
training runs at solve time under a fixed internal seed and a module-level cache
(< 60 s CPU — the ``learned.py`` / adr-025 precedent).

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
from .implicit_ue import _build_routes, _RouteSet
from .learned import _TRAINING_SCENARIOS, TRAINING_FAMILY

__all__ = ["HetGNNModel", "TRAINING_FAMILY"]

# --- architecture hyperparameters (named so the CI wall-time budget is a visible
#     design commitment, not an accident of tuning). Single head, small width,
#     float64 — the lean floor (adr-026). ----------------------------------------
_DIM = 16  # node embedding width (paper: 32, 8 heads — here 1 head, 16)
_N_V_LAYERS = 2  # V-Encoder (virtual OD edge) attention layers
_N_R_LAYERS = 2  # R-Encoder (real link) attention layers
_FFN_HIDDEN = 16  # update-FFN hidden width
_HEAD_HIDDEN = 16  # edge-head MLP hidden width
_N_EDGE_FEAT = 4  # standardized real-edge features [fft, cap, b, power]
_N_NODE_FEAT = 4  # size-agnostic node features [prod, attr, out_deg, in_deg]
_DTYPE = torch.float64  # anchors (permutation equivariance) need float64

# --- loss weights (paper Eqs 11-14). L_f and L_c normalized by total demand. ----
_W_ALPHA = 1.0
_W_FLOW = 0.005
_W_CONS = 0.05

# --- training hyperparameters (training wall-time budget < 60 s CPU) ------------
_TRAIN_SEED = 20240  # FIXED internal seed (seedable=False; not the harness RNG)
_TRAIN_EPOCHS = 100
_TRAIN_LR = 0.01  # lr >= 0.02 destabilizes the composite loss (measured, adr-026)
_TRAIN_WEIGHT_DECAY = 1e-4  # keeps the fit transferable (the identifiability caveat)
_N_AUGMENT = 1  # fixed-seed per-OD demand-rescaling copies per base scenario
_AUGMENT_SEED = 4270  # seed for the U(0.5, 1.5) per-OD scalings (paper Eq 15)
_REF_BUDGET = Budget(iterations=200, target_relative_gap=1e-7)

# --- feasible-decode hyperparameters (repo extension) ---------------------------
_N_CG = 6  # column-generation rounds = Dijkstra sweeps counted as sp_calls
_N_DECODE = 500  # projected-gradient steps cap (the `iterations` coord)
_DECODE_OMEGA_MIN = 1e-6  # damping floor (relative to the Lipschitz step)
_DECODE_TOL = 1e-14  # early-stop once the projection objective falls below this


def _het_graph(network: Network, demand: Demand) -> dict[str, torch.Tensor]:
    """Heterogeneous graph tensors for one scenario (CPU float64).

    Node features are the SIZE-AGNOSTIC intrinsic substitution for the paper's
    OD-row featurization: ``[production_u, attraction_u, out_degree_u,
    in_degree_u]`` with the demand terms normalized by total demand and the
    degrees by link count. Each is a per-node intrinsic quantity, so the whole
    graph is exactly permutation equivariant (anchor A4). Real-edge features are
    the standardized ``[fft, cap, b, power]`` (the ``implicit_ue`` convention);
    virtual edges are one directed edge per off-diagonal positive OD pair.
    """
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=_DTYPE)  # noqa: E731
    n = network.n_nodes
    init0 = network.init_node - 1  # 0-based tail node ids
    term0 = network.term_node - 1  # 0-based head node ids

    od = np.asarray(demand.matrix, dtype=np.float64)
    off = od - np.diag(np.diag(od))
    total = float(off.sum())
    scale = total if total > 0 else 1.0
    prod = np.zeros(n)
    attr = np.zeros(n)
    nz = network.n_zones
    prod[:nz] = off.sum(axis=1) / scale  # trips originating at each zone
    attr[:nz] = off.sum(axis=0) / scale  # trips destined for each zone

    out_deg = np.bincount(init0, minlength=n).astype(np.float64) / max(1, network.n_links)
    in_deg = np.bincount(term0, minlength=n).astype(np.float64) / max(1, network.n_links)
    node_x = np.column_stack([prod, attr, out_deg, in_deg])

    static_raw = np.column_stack(
        [network.free_flow_time, network.capacity, network.b, network.power]
    )
    mu = static_raw.mean(axis=0)
    sigma = static_raw.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    edge_feat = (static_raw - mu) / sigma

    # virtual edges: one per off-diagonal positive OD pair (tail=origin node,
    # head=destination node). Virtual edges carry NO features, per the paper —
    # their adaptive weight beta_e is learned from the endpoint embeddings
    # (Eq 7-8); demand reaches the model through the node features (adr-026
    # review removed a dead per-edge demand feature that was never consumed).
    v_rows, v_cols = np.nonzero(off > 0)
    v_init = v_rows.astype(np.int64)
    v_term = v_cols.astype(np.int64)

    return {
        "n_nodes": n,
        "node_x": t(node_x),
        "r_init": torch.as_tensor(init0, dtype=torch.int64),
        "r_term": torch.as_tensor(term0, dtype=torch.int64),
        "r_feat": t(edge_feat),
        "v_init": torch.as_tensor(v_init, dtype=torch.int64),
        "v_term": torch.as_tensor(v_term, dtype=torch.int64),
        "cap": t(network.capacity),
    }


def _node_segment_softmax(
    logits: torch.Tensor, tail: torch.Tensor, n_nodes: int
) -> torch.Tensor:
    """Per-tail-node softmax over outgoing edges (the ``_segment_softmax_load``
    pattern, keyed on node index instead of OD group). Returns a per-edge
    attention weight; every node with an outgoing edge is normalized to sum 1."""
    gmax = torch.full((n_nodes,), float("-inf"), dtype=_DTYPE)
    gmax = gmax.scatter_reduce(0, tail, logits, reduce="amax", include_self=False)
    e = torch.exp(logits - gmax[tail])
    denom = torch.zeros(n_nodes, dtype=_DTYPE).index_add(0, tail, e)
    return e / denom[tail].clamp(min=1e-300)


class _EncoderLayer(nn.Module):
    """One V- or R-Encoder attention layer (paper Eqs 7-9), single head.

    Aggregation is at the tail node ``u`` over its OUTGOING edges (paper's
    ``N_o(u)``), weighted by the per-tail-node segment softmax of
    ``(q_u . k_v / sqrt(d)) * beta_e``. For a virtual layer the scalar edge
    weight ``beta_e = FFN([x_u || x_v])`` is learned from the endpoints (virtual
    edges have no features); for a real layer it is a learned scalar of the real
    edge features. Update is the paper's residual ``x_u + LayerNorm(FFN(z_u))``
    (the nonstandard order is the paper's — an underdetermined detail, adr-026).
    """

    def __init__(self, dim: int, virtual: bool, edge_feat_dim: int = 0) -> None:
        super().__init__()
        self.dim = dim
        self.virtual = virtual
        self.qkv = nn.Linear(dim, 3 * dim, bias=False, dtype=_DTYPE)
        if virtual:
            self.beta = nn.Sequential(
                nn.Linear(2 * dim, _FFN_HIDDEN, dtype=_DTYPE),
                nn.Tanh(),
                nn.Linear(_FFN_HIDDEN, 1, dtype=_DTYPE),
            )
        else:
            self.beta = nn.Linear(edge_feat_dim, 1, dtype=_DTYPE)
        self.upd = nn.Sequential(
            nn.Linear(dim, _FFN_HIDDEN, dtype=_DTYPE),
            nn.Tanh(),
            nn.Linear(_FFN_HIDDEN, dim, dtype=_DTYPE),
        )
        self.norm = nn.LayerNorm(dim, dtype=_DTYPE)

    def forward(
        self,
        x: torch.Tensor,
        tail: torch.Tensor,
        head: torch.Tensor,
        n_nodes: int,
        edge_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qkv = self.qkv(x).view(-1, 3, self.dim)
        q, k, val = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        logits = (q[tail] * k[head]).sum(-1) / np.sqrt(self.dim)
        if self.virtual:
            b = self.beta(torch.cat([x[tail], x[head]], dim=-1)).squeeze(-1)
        else:
            b = self.beta(edge_feat).squeeze(-1)
        attn = _node_segment_softmax(logits * b, tail, n_nodes)
        z = torch.zeros_like(x).index_add(0, tail, attn.unsqueeze(-1) * val[head])
        return x + self.norm(self.upd(z))


class _HetGNN(nn.Module):
    """Preprocess -> V-Encoders -> R-Encoders -> edge ratio head (paper Sec 4)."""

    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Linear(_N_NODE_FEAT, _DIM, dtype=_DTYPE)
        self.v_layers = nn.ModuleList(
            _EncoderLayer(_DIM, virtual=True) for _ in range(_N_V_LAYERS)
        )
        self.r_layers = nn.ModuleList(
            _EncoderLayer(_DIM, virtual=False, edge_feat_dim=_N_EDGE_FEAT)
            for _ in range(_N_R_LAYERS)
        )
        self.head = nn.Sequential(
            nn.Linear(2 * _DIM + _N_EDGE_FEAT, _HEAD_HIDDEN, dtype=_DTYPE),
            nn.Tanh(),
            nn.Linear(_HEAD_HIDDEN, 1, dtype=_DTYPE),
        )

    def forward(self, g: dict[str, torch.Tensor]) -> torch.Tensor:
        n = g["n_nodes"]
        h = self.pre(g["node_x"])
        for layer in self.v_layers:
            h = layer(h, g["v_init"], g["v_term"], n)
        for layer in self.r_layers:
            h = layer(h, g["r_init"], g["r_term"], n, edge_feat=g["r_feat"])
        u, v = g["r_init"], g["r_term"]
        alpha = self.head(torch.cat([h[u], h[v], g["r_feat"]], dim=-1)).squeeze(-1)
        return alpha  # raw flow/capacity ratio per real link (paper Eq 10)


def _conservation_residual(
    flows: torch.Tensor, g: dict[str, torch.Tensor], expected: torch.Tensor
) -> torch.Tensor:
    """L1 aggregate node-balance residual of ``flows`` (paper Eq 13; identical to
    ``metrics.gaps.node_balance_residual`` up to the L1-vs-Linf reduction)."""
    n = g["n_nodes"]
    inflow = torch.zeros(n, dtype=_DTYPE).index_add(0, g["r_term"], flows)
    outflow = torch.zeros(n, dtype=_DTYPE).index_add(0, g["r_init"], flows)
    return (inflow - outflow - expected).abs().sum()


def _expected_balance(network: Network, demand: Demand) -> torch.Tensor:
    """Per-node ``attractions_i - productions_i`` (zone nodes only), as float64."""
    od = np.asarray(demand.matrix, dtype=np.float64)
    off = od - np.diag(np.diag(od))
    expected = np.zeros(network.n_nodes)
    nz = network.n_zones
    expected[:nz] = off.sum(axis=0)[:nz] - off.sum(axis=1)[:nz]
    return torch.as_tensor(expected, dtype=_DTYPE)


def _augmented_scenario(base: Scenario, rng: np.random.Generator, tag: int) -> Scenario:
    """A per-OD demand-rescaled copy of ``base`` (paper Eq 15: each positive
    off-diagonal entry scaled by ``U(0.5, 1.5)``); same network, same family."""
    od = np.asarray(base.demand.matrix, dtype=np.float64).copy()
    off = ~np.eye(od.shape[0], dtype=bool)
    positive = off & (od > 0)
    od[positive] *= rng.uniform(0.5, 1.5, size=int(positive.sum()))
    return Scenario(
        name=f"{base.name}-aug{tag}",
        network=base.network,
        demand=Demand(matrix=od),
        family=TRAINING_FAMILY,
    )


#: base + augmented training scenarios, built once at import (network construction
#: + deterministic augmentation, no solving). Their content hashes are declared in
#: ``trained_on`` so the fairness gate refuses every training instance by hash.
def _all_training_scenarios() -> tuple[Scenario, ...]:
    rng = np.random.default_rng(_AUGMENT_SEED)
    augmented = []
    for base in _TRAINING_SCENARIOS:
        for j in range(_N_AUGMENT):
            augmented.append(_augmented_scenario(base, rng, j))
    return tuple(_TRAINING_SCENARIOS) + tuple(augmented)


_TRAINING_INSTANCES = _all_training_scenarios()


def _training_cases() -> list[dict]:
    """BFW reference equilibria + graph tensors for every training instance.

    Labels are ``BiconjugateFrankWolfeModel`` equilibria at the repo's certified-
    gap criterion (a documented deviation from the paper's successive-flow-change
    Frank-Wolfe labels). Returns the per-scenario tensors the fit consumes plus
    the summed training shortest-path budget (reported as provenance, never
    scored)."""
    solver = BiconjugateFrankWolfeModel()
    cases: list[dict] = []
    for scenario in _TRAINING_INSTANCES:
        trace = Trace()
        solver.solve(scenario, _REF_BUDGET, RngBundle(0), trace)
        g = _het_graph(scenario.network, scenario.demand)
        cap = g["cap"]
        v_obs = torch.as_tensor(trace.final.link_flows, dtype=_DTYPE)
        cases.append(
            {
                "g": g,
                "cap": cap,
                "v_obs": v_obs,
                "alpha_obs": v_obs / cap,
                "expected": _expected_balance(scenario.network, scenario.demand),
                "scale": max(1.0, float(scenario.demand.total)),
                "sp_calls": int(trace.final.coords.sp_calls),
            }
        )
    return cases


def _fit(cases: list[dict], w_cons: float = _W_CONS) -> _HetGNN:
    """Fit a HetGNN on the training cases by plain Adam backprop (fixed seed).

    The caller is responsible for the torch global-state save/restore; this
    routine only seeds the parameter init/optimizer. ``w_cons`` is exposed so the
    conservation ablation (adr-026) can fit a ``w_c = 0`` head on the same cases."""
    torch.manual_seed(_TRAIN_SEED)
    model = _HetGNN()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=_TRAIN_LR, weight_decay=_TRAIN_WEIGHT_DECAY
    )
    for _ in range(_TRAIN_EPOCHS):
        optimizer.zero_grad()
        loss = torch.zeros((), dtype=_DTYPE)
        for c in cases:
            alpha = model(c["g"])
            f_pred = alpha * c["cap"]
            l_alpha = (alpha - c["alpha_obs"]).abs().mean()
            l_flow = (f_pred - c["v_obs"]).abs().mean() / c["scale"]
            l_cons = _conservation_residual(f_pred, c["g"], c["expected"]) / c["scale"]
            loss = loss + _W_ALPHA * l_alpha + _W_FLOW * l_flow + w_cons * l_cons
        loss.backward()
        optimizer.step()
    model.eval()
    return model


#: cached (_HetGNN, {"sp_calls", "wall_ms"}) — the one-time offline training cost.
_TRAINED: tuple[_HetGNN, dict[str, float]] | None = None


def _train() -> tuple[_HetGNN, dict[str, float]]:
    """Fit the default (``w_c = 0.05``) head on the synthetic family (cached).

    Deterministic: fixed internal seed, single-threaded, deterministic torch
    algorithms, deterministic bi-conjugate FW. Saves/restores the full torch
    global state (threads, deterministic flag, RNG) so the first cold-cache solve
    never perturbs the process (the complete adr-025 review lesson set)."""
    global _TRAINED
    if _TRAINED is not None:
        return _TRAINED

    prev_threads = torch.get_num_threads()
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_rng = torch.get_rng_state()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        start = time.perf_counter()
        cases = _training_cases()
        sp_calls = sum(c["sp_calls"] for c in cases)
        model = _fit(cases)
        stats = {"sp_calls": float(sp_calls), "wall_ms": 1000.0 * (time.perf_counter() - start)}
        _TRAINED = (model, stats)
    finally:
        torch.use_deterministic_algorithms(prev_det)
        torch.set_num_threads(prev_threads)
        torch.set_rng_state(prev_rng)
    return _TRAINED


def _spectral_step(delta: torch.Tensor) -> float:
    """Lipschitz step ``1 / lambda_max(Delta Delta^T)`` for the projected-gradient
    decode, via deterministic power iteration on ``Delta^T Delta`` (an
    ``(n_links, n_links)`` matrix — small even when the route count is large)."""
    gram = delta.t() @ delta
    b = torch.ones(gram.shape[0], dtype=_DTYPE)
    b = b / b.norm()
    lam = 1.0
    for _ in range(60):
        b = gram @ b
        norm = b.norm()
        if float(norm) <= 0.0:
            return 1.0
        b = b / norm
        lam = float(b @ (gram @ b))
    return 1.0 / lam if lam > 0.0 else 1.0


def _project_route_simplex(
    y: torch.Tensor,
    od_index: torch.Tensor,
    slot: torch.Tensor,
    counts: torch.Tensor,
    demand: torch.Tensor,
    max_k: int,
) -> torch.Tensor:
    """Euclidean projection of ``y`` onto the product of per-OD scaled simplices
    ``{h_od >= 0, sum h_od = D_od}`` (vectorized Duchi et al. 2008 over a padded
    ``(n_groups, max_k)`` matrix — masked slots sink to ``-inf`` and take no
    mass). Guarantees each OD routes exactly its demand, so ``Delta^T h`` is
    demand-feasible by construction regardless of the decode's convergence."""
    n_groups = int(demand.shape[0])
    pad = torch.full((n_groups, max_k), float("-inf"), dtype=_DTYPE)
    pad[od_index, slot] = y
    ys, _ = torch.sort(pad, dim=1, descending=True)  # invalid (-inf) sort last
    css = torch.cumsum(torch.where(torch.isfinite(ys), ys, torch.zeros_like(ys)), dim=1)
    j = torch.arange(1, max_k + 1, dtype=_DTYPE)
    theta_cand = (css - demand.unsqueeze(1)) / j
    cond = ys > theta_cand  # invalid slots are -inf -> False; rho >= 1 since D > 0
    rho = cond.sum(dim=1)
    theta = (css.gather(1, (rho - 1).unsqueeze(1)).squeeze(1) - demand) / rho
    projected = torch.clamp(pad - theta.unsqueeze(1), min=0.0)  # -inf slots -> 0
    return projected[od_index, slot]


def _decode(
    rs: _RouteSet,
    v_raw: torch.Tensor,
    n_steps: int,
    deadline: float | None = None,
) -> tuple[torch.Tensor, float, int]:
    """Projected-gradient projection of ``v_raw`` onto the feasible route polytope.

    Minimizes ``0.5||Delta^T h - v_raw||^2`` over ``h`` in the per-OD scaled-
    simplex product. The step is fixed at ``1 / lambda_max(Delta Delta^T)`` and
    HALVED whenever the objective rises (adaptive damping, floored) — the same
    guard ``implicit-ue-nn``'s fixed point uses against the repo's recurring
    limit-cycle defect. Returns ``(h, residual, steps)`` with the residual
    ``||Delta^T h - v_raw||_inf`` measured AT the returned iterate and ``steps``
    the count actually executed (the honest P6 ``iterations`` coordinate)."""
    delta = rs.delta
    od_index = rs.od_index
    counts = torch.bincount(od_index, minlength=rs.n_groups).to(_DTYPE)
    group_start = torch.cat(
        [torch.zeros(1, dtype=torch.int64), torch.cumsum(counts.to(torch.int64), 0)[:-1]]
    )
    slot = torch.arange(od_index.shape[0], dtype=torch.int64) - group_start[od_index]
    max_k = int(counts.max()) if counts.numel() else 1

    step = _spectral_step(delta)
    omega = step
    h = rs.demand[od_index] / counts[od_index]  # uniform per-OD start
    prev_obj = float("inf")
    steps = 0
    for _ in range(n_steps):
        v = delta.t() @ h
        obj = float(0.5 * ((v - v_raw) ** 2).sum())
        if obj < _DECODE_TOL:
            break
        if obj > prev_obj:
            omega = max(0.5 * omega, _DECODE_OMEGA_MIN * step)
        prev_obj = obj
        grad = delta @ (v - v_raw)
        h = _project_route_simplex(
            h - omega * grad, od_index, slot, counts, rs.demand, max_k
        )
        steps += 1
        if deadline is not None and time.perf_counter() >= deadline:
            break
    residual = float((delta.t() @ h - v_raw).abs().max())
    return h, residual, steps


@register_model
class HetGNNModel(TrafficAssignmentModel):
    """Heterogeneous-GNN UE surrogate (Liu & Meidani 2024, lean variant; ``learned``).

    Emits TWO checkpoints per solve: (i) the paper-faithful RAW flow/capacity
    prediction ``f~ = cap * relu(alpha~)`` at ``sp_calls=0`` — censored
    ``feasible=0`` by the harness demand-feasibility audit, because conservation
    is only a soft loss term; and (ii) a route-based feasible DECODE (a repo
    extension — NOT in the paper) that projects the raw flows onto the demand-
    feasible route polytope, earning a real certified gap. Deterministic (fixed
    internal training seed); refused on ``synthetic-net`` scenarios (and every
    trained-on instance's hash) by the ``trained_on`` fairness gate."""

    name = "het-gnn"
    capabilities = Capabilities(
        paradigm="learned",
        deterministic=True,
        provides_gap=False,
        seedable=False,
        trained_on=(TRAINING_FAMILY,)
        + tuple(s.content_hash() for s in _TRAINING_INSTANCES),
    )

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        model, train_cost = _train()  # cached; its cost is reported, not timed here
        start = time.perf_counter()  # measure INFERENCE only (deterministic)
        n_cg = _N_CG if budget.sp_calls is None else max(1, min(_N_CG, budget.sp_calls))
        n_steps = (
            _N_DECODE
            if budget.iterations is None
            else max(1, min(_N_DECODE, budget.iterations))
        )
        deadline = None if budget.wall_seconds is None else start + budget.wall_seconds

        od = scenario.demand.matrix
        if not np.any(od[~np.eye(od.shape[0], dtype=bool)] > 0):
            # No routable (off-diagonal) demand: the zero flow is the exact
            # equilibrium, and the raw GNN emission would be a phantom censored
            # flow. Short-circuit before any PER-SCENARIO torch work (the
            # cached one-time training above still runs so its cost is
            # reported as provenance — adr-026 review corrected the wording).
            flows = np.zeros(scenario.network.n_links)
            coords = BudgetCoords(
                iterations=0, sp_calls=0, wall_ms=1000.0 * (time.perf_counter() - start)
            )
            trace.record(
                flows,
                coords,
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

        # (i) paper-faithful RAW emission: no shortest path -> sp_calls=0. The
        #     runner certifies every checkpoint (P1), so this row appears censored
        #     with its harness-recomputed node-balance residual (nothing
        #     self-attested — the honest paper-model row).
        g = _het_graph(scenario.network, scenario.demand)
        with torch.no_grad():
            alpha = model(g)
        v_raw = torch.clamp(alpha, min=0.0) * g["cap"]
        trace.record(
            v_raw.numpy(),
            BudgetCoords(iterations=0, sp_calls=0, wall_ms=1000.0 * (time.perf_counter() - start)),
            training_sp_calls=train_cost["sp_calls"],
            training_wall_ms=train_cost["wall_ms"],
        )

        # (ii) route-based feasible DECODE (repo extension): project the raw flows
        #      onto the per-OD scaled-simplex product over column-generated routes.
        engine = PathEngine(scenario.network)
        rs = _build_routes(scenario.network, scenario.demand, engine, n_cg, deadline)
        h, residual, steps = _decode(rs, v_raw, n_steps, deadline)
        v_dec = (rs.delta.t() @ h).detach().numpy()
        trace.record(
            v_dec,
            BudgetCoords(
                iterations=steps,  # executed decode steps, not the cap (P6 honesty)
                sp_calls=rs.sp_calls,  # real Dijkstra sweeps
                wall_ms=1000.0 * (time.perf_counter() - start),
            ),
            decode_residual=residual,  # projection truncation, descriptive (P6)
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
