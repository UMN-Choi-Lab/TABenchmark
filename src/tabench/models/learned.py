"""A lean reference *learned* traffic-assignment surrogate (the first black-box).

Machine-learning models for traffic assignment (GNNs mapping OD demand + network
to link flows, trained on solver equilibria) are an active line — Rahman & Hasan
(2023) graph-convolutional, Liu & Meidani (2024) heterogeneous GNN, and the Xu
et al. (2024) 20-city dataset. That literature grades a learned model on
*link-flow error* (MAE/MAPE) against the solver it imitates. TABenchmark grades
it the same way it grades Frank-Wolfe: the harness recomputes the **equilibrium
gap** from the emitted link flows (P1). Those are different questions — a
surrogate can match flows to a few percent yet sit far from equilibrium — and
exposing that gap is the point of wrapping a learned model here.

``LearnedSurrogateModel`` is a deliberately small, dependency-free stand-in: a
per-link ridge regression that predicts each link's equilibrium volume/capacity
ratio from its free-flow all-or-nothing loading and BPR shape, fitted on
solver equilibria of a *synthetic* network family (``trained_on`` =
``"synthetic-net"``). It is not a GNN — a torch-based graph model is the natural
extension — but it exercises the full contract a learned model plugs into: the
per-instance ``learned`` paradigm, the ``trained_on`` fairness gate (it is
refused on any ``synthetic-net`` scenario), and identical certification. Being a
per-link predictor it does not enforce flow conservation, so the harness's
demand-feasibility audit is what tells the honest story about its output.

Trained on synthetic networks and evaluated on the (disjoint) TNTP scenarios —
the clean train/test split the ML-TA literature usually skips — with no shared
network identity, so the fairness gate has real teeth.
"""

from __future__ import annotations

import time

import numpy as np

from ..core.budget import Budget, BudgetCoords
from ..core.capabilities import Capabilities
from ..core.results import ResultBundle, Trace
from ..core.rng import RngBundle
from ..core.scenario import Demand, Network, Scenario
from ._paths import PathEngine
from .base import TrafficAssignmentModel, register_model
from .frank_wolfe import BiconjugateFrankWolfeModel

__all__ = ["LearnedSurrogateModel", "TRAINING_FAMILY"]

#: family name shared by every synthetic training scenario; the ``trained_on``
#: lineage the fairness gate matches against.
TRAINING_FAMILY = "synthetic-net"

#: (seed, n_nodes, n_zones, extra_edges) for the synthetic training networks.
_TRAINING_SPECS = (
    (1, 8, 3, 4),
    (2, 10, 4, 6),
    (3, 12, 4, 8),
    (4, 9, 3, 5),
    (5, 14, 5, 10),
    (6, 11, 4, 7),
)

_RIDGE_LAMBDA = 1e-2


def _random_network_scenario(
    seed: int, n_nodes: int, n_zones: int, extra_edges: int
) -> Scenario:
    """A small connected random BPR network with random OD demand.

    A random spanning tree over all nodes plus ``extra_edges`` random chords,
    every undirected edge made bidirectional (so the graph is strongly connected
    and every OD pair is reachable). Zones are nodes ``1..n_zones`` (TNTP
    convention); ``family="synthetic-net"``."""
    rng = np.random.default_rng(seed)
    undirected: set[tuple[int, int]] = set()
    for child in range(2, n_nodes + 1):  # spanning tree: connect to an earlier node
        parent = int(rng.integers(1, child))
        undirected.add((parent, child))
    attempts = 0
    while len(undirected) < (n_nodes - 1) + extra_edges and attempts < 100 * extra_edges:
        u = int(rng.integers(1, n_nodes + 1))
        v = int(rng.integers(1, n_nodes + 1))
        attempts += 1
        if u != v:
            undirected.add((min(u, v), max(u, v)))
    init, term = [], []
    for u, v in sorted(undirected):
        init += [u, v]
        term += [v, u]  # bidirectional
    n_links = len(init)
    network = Network(
        name=f"synthetic-{seed}",
        n_nodes=n_nodes,
        n_zones=n_zones,
        first_thru_node=1,
        init_node=np.array(init, dtype=np.int64),
        term_node=np.array(term, dtype=np.int64),
        capacity=rng.uniform(20.0, 100.0, n_links),
        length=np.zeros(n_links),
        free_flow_time=rng.uniform(1.0, 10.0, n_links),
        b=np.full(n_links, 0.15),
        power=np.full(n_links, 4.0),
        toll=np.zeros(n_links),
        link_type=np.ones(n_links, dtype=np.int64),
    )
    od = np.zeros((n_zones, n_zones))
    for o in range(n_zones):
        for d in range(n_zones):
            if o != d and rng.random() < 0.6:
                od[o, d] = float(rng.uniform(5.0, 50.0))
    if od.sum() == 0.0:  # guarantee at least one OD pair
        od[0, n_zones - 1] = 20.0
    return Scenario(
        name=f"synthetic-{seed}",
        network=network,
        demand=Demand(matrix=od),
        family=TRAINING_FAMILY,
    )


#: the synthetic training scenarios, built once at import (no solving here — just
#: network construction). Reused by ``_train`` and by ``trained_on`` so the
#: fairness gate declares the exact training instances' content hashes, not only
#: the coarse family name.
_TRAINING_SCENARIOS = tuple(_random_network_scenario(*spec) for spec in _TRAINING_SPECS)


def _features(scenario: Scenario, engine: PathEngine) -> np.ndarray:
    """Per-link, dimensionless, network-size-agnostic feature matrix.

    The free-flow all-or-nothing loading (one Dijkstra sweep) is the dominant
    predictor of the equilibrium flow. Only two smooth, bounded transforms of
    the free-flow volume/capacity ratio are used — deliberately robust:
    higher-order polynomial features (``vc0**2``, ``b*vc0**power``) fit the
    small synthetic training networks better but extrapolate badly to the far
    larger TNTP test networks, flipping the sign of the correlation on the
    biggest ones. Returns ``(n_links, n_feat)``."""
    net = scenario.network
    y0, _ = engine.all_or_nothing(net.link_cost(np.zeros(net.n_links)), scenario.demand)
    vc0 = y0 / net.capacity  # free-flow volume/capacity ratio
    return np.column_stack([vc0, np.log1p(vc0)])


class _Ridge:
    """Standardized ridge regression with an unregularized intercept."""

    def __init__(self, x: np.ndarray, y: np.ndarray, lam: float) -> None:
        self.mu = x.mean(axis=0)
        self.sigma = x.std(axis=0)
        self.sigma[self.sigma == 0.0] = 1.0
        xs = np.column_stack([np.ones(len(x)), (x - self.mu) / self.sigma])
        n_feat = xs.shape[1]
        reg = _RIDGE_LAMBDA * np.eye(n_feat)
        reg[0, 0] = 0.0  # do not regularize the intercept
        self.w = np.linalg.solve(xs.T @ xs + reg, xs.T @ y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        xs = np.column_stack([np.ones(len(x)), (x - self.mu) / self.sigma])
        return xs @ self.w


#: cached (ridge, {"sp_calls", "wall_ms"}) — the one-time offline training cost.
_TRAINED: tuple[_Ridge, dict[str, float]] | None = None


def _train() -> tuple[_Ridge, dict[str, float]]:
    """Fit the surrogate on solver equilibria of the synthetic family (cached).

    Deterministic: fixed seeds, deterministic bi-conjugate FW, closed-form ridge.
    Also returns the one-time offline training cost (summed shortest-path calls
    and wall time across the training solves) so a learned model reports its
    training budget rather than hiding it (docs/ARCHITECTURE.md P6).
    """
    global _TRAINED
    if _TRAINED is not None:
        return _TRAINED
    solver = BiconjugateFrankWolfeModel()
    feats, targets = [], []
    sp_calls = 0
    start = time.perf_counter()
    for scenario in _TRAINING_SCENARIOS:
        engine = PathEngine(scenario.network)
        trace = Trace()
        solver.solve(
            scenario, Budget(iterations=200, target_relative_gap=1e-7), RngBundle(0), trace
        )
        sp_calls += trace.final.coords.sp_calls
        feats.append(_features(scenario, engine))
        targets.append(trace.final.link_flows / scenario.network.capacity)  # equilibrium V/C
    ridge = _Ridge(np.vstack(feats), np.concatenate(targets), _RIDGE_LAMBDA)
    stats = {"sp_calls": float(sp_calls), "wall_ms": 1000.0 * (time.perf_counter() - start)}
    _TRAINED = (ridge, stats)
    return _TRAINED


@register_model
class LearnedSurrogateModel(TrafficAssignmentModel):
    """Per-link learned regression surrogate (paradigm ``learned``).

    Predicts each link's equilibrium volume/capacity ratio from its free-flow
    loading and BPR shape, fitted offline on synthetic solver equilibria, then
    emits ``predicted_v/c * capacity`` (clipped nonnegative) as its single
    checkpoint. Deterministic. The harness recomputes the equilibrium gap and
    demand-feasibility from these flows exactly as for any solver — a per-link
    predictor conserves flow only approximately, so it is typically censored or
    carries a large certified gap even when its link-flow RMSE is small (that
    contrast is the point). Refused on ``synthetic-net`` scenarios by the
    ``trained_on`` fairness gate.
    """

    name = "learned-surrogate"
    capabilities = Capabilities(
        paradigm="learned",
        deterministic=True,
        provides_gap=False,
        seedable=False,
        # Declare BOTH the family and the exact training instances' content
        # hashes (the rename-robust form the fairness gate recommends), so the
        # gate refuses this model even on a training network republished under a
        # different family name.
        trained_on=(TRAINING_FAMILY,) + tuple(s.content_hash() for s in _TRAINING_SCENARIOS),
    )

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        model, train_cost = _train()  # cached; its cost is reported, not timed here
        start = time.perf_counter()  # measure INFERENCE only (deterministic, cache-independent)
        engine = PathEngine(scenario.network)
        vc_pred = model.predict(_features(scenario, engine))
        flows = np.maximum(vc_pred, 0.0) * scenario.network.capacity
        coords = BudgetCoords(
            iterations=1,
            sp_calls=1,  # one free-flow all-or-nothing to build the features
            wall_ms=1000.0 * (time.perf_counter() - start),
        )
        trace.record(
            flows,
            coords,
            predicted_mean_vc=float(np.mean(vc_pred)),
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
