"""Tests for the first torch model — ``implicit-ue-nn`` (Liu et al. 2023, lean variant).

Torch is an OPTIONAL extra; this whole file is skipped on a core install
(``pytest.importorskip('torch')``), and the 731-test numpy suite runs without it
(the torch-free CI matrix legs are the live regression for that). What these
tests pin: the analytic anchors verified in the research prototype (A1 identity,
A2 IMD/adjoint hypergradient vs finite differences), the property that makes this
model the answer to ``learned-surrogate`` (demand feasibility BY CONSTRUCTION,
A4), the honest held-out story (feasible with a real certified gap that a
converged solver still beats — the accuracy-vs-certificate point, never hidden),
the ``trained_on`` fairness gate, budget/provenance accounting, and the
fixed-point's stability on a congested power-4 net.

Cross-platform note: only IN-PROCESS byte determinism is asserted; every
held-out claim is a PROPERTY bound (feasible==1, finite gap, loose ceilings,
directional comparisons), never a pinned trained-output byte or tight decimal.
"""

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import load_or_skip  # noqa: E402

from tabench import (  # noqa: E402
    BiconjugateFrankWolfeModel,
    Budget,
    Evaluator,
    LearnedSurrogateModel,
    RngBundle,
    Trace,
    braess_scenario,
    run_experiment,
    two_route_scenario,
)
from tabench.core.capabilities import ContaminationError  # noqa: E402
from tabench.models import implicit_ue as M  # noqa: E402
from tabench.models._paths import PathEngine  # noqa: E402
from tabench.models.base import MODEL_REGISTRY  # noqa: E402
from tabench.models.implicit_ue import ImplicitUENNModel  # noqa: E402
from tabench.models.learned import TRAINING_FAMILY, _random_network_scenario  # noqa: E402

# The golden Braess content hash: this additive torch model must leave it — and
# thus the whole scored instance canon — byte-identical (HARD RULE).
BRAESS_GOLDEN_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"


def _zeroed_head() -> "M._CostHead":
    """Cost head with all parameters zeroed: the correction vanishes (gain=0), so
    the layer reduces to a plain logit loading at the TRUE BPR costs."""
    head = M._CostHead()
    for p in head.parameters():
        torch.nn.init.zeros_(p)
    return head


def _solve_layer(scenario, head, n_iter=M._N_FP_ITER):
    engine = PathEngine(scenario.network)
    rs = M._build_routes(scenario.network, scenario.demand, engine, M._N_CG)
    net = M._torch_network(scenario.network)
    h, residual, _steps = M._solve_fixed_point(head, rs, net, n_iter)
    return (rs.delta.t() @ h).detach().numpy(), rs, net, residual


def _solve(scenario, model, **budget):
    trace = Trace()
    model.solve(scenario, Budget(**budget), RngBundle(0), trace)
    return trace.final


# ------------------------------------------------------------- wrapper contract
def test_registered_and_capabilities():
    assert "implicit-ue-nn" in MODEL_REGISTRY
    caps = ImplicitUENNModel.capabilities
    assert caps.paradigm == "learned"
    assert caps.deterministic is True
    assert caps.seedable is False
    assert caps.provides_gap is False
    # Declares the training family AND the training instances' content hashes.
    assert TRAINING_FAMILY in caps.trained_on
    assert len(caps.trained_on) > 1
    # The additive torch model leaves the scored-instance canon untouched.
    assert braess_scenario().content_hash() == BRAESS_GOLDEN_HASH


# ------------------------------------------------------------- analytic anchors
def test_a1_identity_anchor_braess():
    """A1: with the cost head pinned to the true BPR latency (zeroed correction),
    the layer's logit fixed point on Braess is the analytic UE — link flows
    (4,2,2,2,4), a common used-route time of 92, and a certified gap ~0. (The
    Braess UE is an equal-cost point, so the logit split is uniform for any beta.)"""
    sc = braess_scenario()
    v, rs, _, residual = _solve_layer(sc, _zeroed_head())
    assert residual < 1e-8
    np.testing.assert_allclose(v, [4.0, 2.0, 2.0, 2.0, 4.0], atol=1e-5)
    route_times = rs.delta.numpy() @ sc.network.link_cost(v)
    np.testing.assert_allclose(route_times, 92.0, atol=1e-4)
    metrics = Evaluator(sc).evaluate(v)
    assert metrics["feasible"] == 1.0
    assert abs(metrics["relative_gap"]) < 1e-6


def test_a2_imd_hypergradient_matches_finite_differences():
    """A2: the implicit-differentiation (IMD/adjoint) hypergradient equals central
    finite differences of the full solve on Braess, at a well-conditioned cost
    head — the implicit-function-theorem gradient the paper differentiates."""
    sc = braess_scenario()
    engine = PathEngine(sc.network)
    rs = M._build_routes(sc.network, sc.demand, engine, M._N_CG)
    net = M._torch_network(sc.network)
    v_obs = torch.as_tensor([4.0, 2.0, 2.0, 2.0, 4.0], dtype=M._DTYPE)
    scale = float(sc.demand.total)

    # A deterministic, mild, guaranteed-interior cost head (a random init can be
    # softmax-saturated and stiff, where BOTH the solve and FD are unreliable).
    head = M._CostHead()
    with torch.no_grad():
        head.l1.weight.copy_(
            torch.linspace(-0.2, 0.2, head.l1.weight.numel(), dtype=M._DTYPE).reshape_as(
                head.l1.weight
            )
        )
        head.l1.bias.zero_()
        head.l2.weight.fill_(0.15)
        head.l2.bias.zero_()
        head.gain.fill_(1.0)

    _, grads = M._hypergradient(head, rs, net, v_obs, scale)

    def full_loss() -> float:
        h, _, _ = M._solve_fixed_point(head, rs, net, M._N_FP_ITER)
        v = rs.delta.t() @ h
        return 0.5 * ((v - v_obs) ** 2).sum().item() / scale

    eps = 1e-4
    max_rel = 0.0
    for param, grad in zip(head.parameters(), grads, strict=True):
        flat = param.detach().reshape(-1)
        gflat = grad.reshape(-1)
        for i in range(flat.numel()):
            orig = flat[i].item()
            with torch.no_grad():
                flat[i] = orig + eps
            lp = full_loss()
            with torch.no_grad():
                flat[i] = orig - eps
            lm = full_loss()
            with torch.no_grad():
                flat[i] = orig
            fd = (lp - lm) / (2 * eps)
            max_rel = max(max_rel, abs(gflat[i].item() - fd) / (abs(fd) + 1e-8))
    assert max_rel < 1e-5


def test_zero_head_reduces_to_logit_loading():
    """theta=0: with the correction zeroed the two-route layer equals the analytic
    binary-logit split at the true costs and the layer's beta — the NN learns a
    correction, not a rename of plain logit loading."""
    from scipy.optimize import brentq

    sc = two_route_scenario()
    v, _, _, _ = _solve_layer(sc, _zeroed_head())
    demand = sc.demand.total

    # Route A = 1->3->2 (c_A = 2 + f_A); route B = 1->4->2 (c_B = 1.5 + 2 f_B).
    def split(f_a: float) -> float:
        c_a = 2.0 + f_a
        c_b = 1.5 + 2.0 * (demand - f_a)
        return f_a - demand / (1.0 + np.exp(M._LOGIT_BETA * (c_a - c_b)))

    f_a = brentq(split, 0.0, demand)
    np.testing.assert_allclose(v[:2], f_a, atol=1e-5)


def test_a4_feasible_by_construction_at_random_theta():
    """A4: at RANDOM untrained theta every emission is v = Delta^T h with each OD's
    route flows summing to its demand, so node balance is exact and the harness
    demand-feasibility audit passes (feasible==1) — the architectural property
    the censored per-link ridge surrogate lacks."""
    from tabench.metrics.gaps import node_balance_residual

    sc = braess_scenario()
    torch.manual_seed(4)
    head = M._CostHead()  # random init, never trained
    v, _, _, _ = _solve_layer(sc, head)
    assert np.all(np.isfinite(v)) and np.all(v >= 0.0)
    assert node_balance_residual(sc, v) < 1e-9
    assert Evaluator(sc).evaluate(v)["feasible"] == 1.0
    # Contrast on the same scenario: the ridge surrogate emits an unrouted flow.
    ridge_v = _solve(sc, LearnedSurrogateModel(), iterations=1).link_flows
    assert node_balance_residual(sc, ridge_v) > node_balance_residual(sc, v)


def test_deterministic_in_process():
    """Two solves are byte-identical in-process (mirrors test_learned; determinism
    is asserted same-platform only — no cross-platform byte claims)."""
    a = _solve(braess_scenario(), ImplicitUENNModel(), iterations=M._N_FP_ITER).link_flows
    b = _solve(braess_scenario(), ImplicitUENNModel(), iterations=M._N_FP_ITER).link_flows
    np.testing.assert_array_equal(a, b)


# ------------------------------------------------------ training + held-out story
def test_training_reduces_training_family_loss():
    """The IMD hypergradient descent genuinely learns: on a synthetic training
    scenario the trained head's flow loss is below the untrained (zeroed) head's.
    This is the in-sample 'trained beats untrained' pin; the held-out certified
    gap is a separate, honestly-scoped question (see the held-out test / adr-025)."""
    cases = []
    for scenario in M._TRAINING_SCENARIOS:
        engine = PathEngine(scenario.network)
        trace = Trace()
        BiconjugateFrankWolfeModel().solve(scenario, M._REF_BUDGET, RngBundle(0), trace)
        cases.append(
            (
                M._build_routes(scenario.network, scenario.demand, engine, M._N_CG),
                M._torch_network(scenario.network),
                torch.as_tensor(trace.final.link_flows, dtype=M._DTYPE),
                max(1.0, float(scenario.demand.total)),
            )
        )

    def family_loss(head) -> float:
        total = 0.0
        for rs, net, v_obs, scale in cases:
            h, _, _ = M._solve_fixed_point(head, rs, net, M._N_FP_ITER)
            total += float(((rs.delta.t() @ h - v_obs) ** 2).sum()) / scale
        return total

    trained, _ = M._train()
    assert family_loss(trained) < family_loss(_zeroed_head())


def test_heldout_direction_and_honest_headline():
    """A6 (honest): on a disjoint TNTP scenario the trained model is demand-
    feasible with a FINITE POSITIVE certified gap, it clears the audit the ridge
    surrogate is censored by (feasible 1 vs 0) with no worse flow error, AND a
    CONVERGED solver still certifies a strictly better gap (the wall-clock /
    convergence axis), while at MATCHED SHORTEST-PATH-CALL budget the direction
    REVERSES — the accuracy-vs-certificate / feasibility-vs-equilibrium point,
    every direction pinned with its axis named, never hidden (adr-025 review:
    the old headline named no axis). Directions only; margins are loose."""
    sc = load_or_skip("siouxfalls")
    assert sc.family != TRAINING_FAMILY
    oracle = sc.reference.link_flows
    wmape = lambda v: float(np.abs(v - oracle).sum() / np.abs(oracle).sum())  # noqa: E731

    v_impl = _solve(sc, ImplicitUENNModel(), iterations=M._N_FP_ITER).link_flows
    m_impl = Evaluator(sc).evaluate(v_impl)
    assert m_impl["feasible"] == 1.0  # feasible BY CONSTRUCTION
    assert np.isfinite(m_impl["relative_gap"]) and 0.0 < m_impl["relative_gap"] < 1.0

    v_ridge = _solve(sc, LearnedSurrogateModel(), iterations=1).link_flows
    m_ridge = Evaluator(sc).evaluate(v_ridge)
    assert m_ridge["feasible"] == 0.0  # censored: a per-link regressor routes nobody
    assert wmape(v_impl) < wmape(v_ridge)  # ... and its flows are no worse

    # Headline direction 1 — a CONVERGED bfw (matched-or-less wall-clock)
    # certifies an orders-better gap:
    v_bfw = _solve(sc, BiconjugateFrankWolfeModel(), iterations=300, target_relative_gap=1e-6)
    m_bfw = Evaluator(sc).evaluate(v_bfw.link_flows)
    assert m_bfw["relative_gap"] < m_impl["relative_gap"]  # converged bfw wins

    # Headline direction 2 — at MATCHED sp_calls the direction REVERSES: six
    # Dijkstra sweeps + cheap fixed-point iterations buy a better certificate
    # than six sweeps' worth of Frank-Wolfe AON iterations (adr-025 review
    # measured 0.168 vs 0.223). A real result, pinned rather than hidden.
    v_bfw_sp = _solve(sc, BiconjugateFrankWolfeModel(), sp_calls=M._N_CG)
    m_bfw_sp = Evaluator(sc).evaluate(v_bfw_sp.link_flows)
    assert m_impl["relative_gap"] < m_bfw_sp["relative_gap"]  # NN wins the sp axis


# --------------------------------------------------------------- fairness gate
def test_fairness_gate_blocks_training_family():
    """Declares trained_on=('synthetic-net', ...); evaluating on a synthetic-net
    scenario is refused (train/test contamination), mirroring learned-surrogate."""
    train_scenario = _random_network_scenario(1, 8, 3, 4)
    assert train_scenario.family == TRAINING_FAMILY
    with pytest.raises(ContaminationError):
        run_experiment(train_scenario, [ImplicitUENNModel()], Budget(iterations=1), seed=0)


# ---------------------------------------------------- budget / provenance (P6)
def test_sp_calls_and_budget_accounting():
    """Emitted sp_calls are the real Dijkstra sweeps (column-generation rounds,
    > 1 unlike the ridge surrogate's 1), and the sp_calls/iterations axes cap the
    layer's work without ever exceeding the budget."""
    sc = braess_scenario()
    full = _solve(sc, ImplicitUENNModel(), iterations=M._N_FP_ITER)
    assert full.coords.sp_calls == M._N_CG > 1
    capped = _solve(sc, ImplicitUENNModel(), sp_calls=3, iterations=50)
    assert capped.coords.sp_calls == 3 <= 3
    assert capped.coords.iterations <= 50


def test_training_provenance_keys_present():
    """The one-time offline training budget is reported as provenance, not hidden
    and not scored (P6, learned-surrogate precedent)."""
    final = _solve(braess_scenario(), ImplicitUENNModel(), iterations=M._N_FP_ITER)
    assert "training_sp_calls" in final.self_report
    assert "training_wall_ms" in final.self_report
    assert final.self_report["training_sp_calls"] > 0
    assert "fixed_point_residual" in final.self_report


def test_training_wall_time_budget():
    """Training is a design commitment to stay well under a minute of CPU; a real
    (uncached) train must complete inside the budget so the torch CI job's long
    pole cannot silently grow."""
    M._TRAINED = None  # force a real training run, not the module cache
    start = time.perf_counter()
    M._train()
    assert time.perf_counter() - start < 60.0


# ------------------------------------------------- fixed-point robustness (fuzz)
def test_fixed_point_stable_on_congested_power4():
    """The layer must not overshoot/limit-cycle on a strongly congested power-4
    network (the repo's recurring fixed-point defect). Feasibility is structural,
    so under random untrained heads every emission stays finite, nonnegative, and
    demand-feasible with a finite gap — the property bound, not a tight value."""
    from tabench import Demand, Network, Scenario

    # Two disjoint 2-link routes, power-4 BPR, demand far above capacity.
    net = Network(
        name="congested-p4",
        n_nodes=4,
        n_zones=2,
        first_thru_node=1,
        init_node=np.array([1, 3, 1, 4], dtype=np.int64),
        term_node=np.array([3, 2, 4, 2], dtype=np.int64),
        capacity=np.array([1.0, 1.0, 1.5, 1.5]),
        length=np.zeros(4),
        free_flow_time=np.array([1.0, 2.0, 1.0, 3.0]),
        b=np.full(4, 0.15),
        power=np.full(4, 4.0),
        toll=np.zeros(4),
        link_type=np.ones(4, dtype=np.int64),
    )
    od = np.zeros((2, 2))
    od[0, 1] = 20.0
    sc = Scenario(name="congested-p4", network=net, demand=Demand(od), family="fuzz-p4")
    for seed in range(5):
        torch.manual_seed(seed)
        head = M._CostHead()
        v, _, _, residual = _solve_layer(sc, head)
        assert np.all(np.isfinite(v)) and np.all(v >= -1e-9)
        # adr-025 review MAJOR: constant damping LIMIT-CYCLED here and this
        # test could not see it (it asserted only by-construction properties).
        # Adaptive damping converges; pin the REAL residual bound.
        assert residual < 1e-6 * 20.0  # demand-relative fixed-point residual
        metrics = Evaluator(sc).evaluate(v)
        assert metrics["feasible"] == 1.0
        assert np.isfinite(metrics["relative_gap"])


# ------------------------------------------------- adr-025 review regressions
def test_zero_demand_scenarios_emit_zero_flows():
    """Review MAJOR: all-zero (and diagonal-only) OD matrices crashed
    _build_routes with an empty route set, killing a whole run_experiment grid
    while every classical solver handled the same input. The zero flow is the
    exact equilibrium; it must be emitted, feasible, gap 0."""
    from tabench import Demand, Scenario

    base = braess_scenario()
    for od in (np.zeros((2, 2)), np.diag([5.0, 3.0])):
        sc = Scenario(name="z", network=base.network, demand=Demand(od), family="fuzz")
        final = _solve(sc, ImplicitUENNModel(), iterations=10)
        assert np.array_equal(final.link_flows, np.zeros(base.network.n_links))
        metrics = Evaluator(sc).evaluate(final.link_flows)
        assert metrics["feasible"] == 1.0
        assert metrics["relative_gap"] == pytest.approx(0.0, abs=1e-12)


def test_wall_seconds_budget_is_respected():
    """Review MAJOR: a wall-only budget was silently ignored (37x overrun on
    siouxfalls) while every classical solver checks it. The deadline now stops
    both column generation and the fixed-point loop; allow generous slack for
    the non-interruptible tail (one sweep + one map application)."""
    sc = load_or_skip("siouxfalls")
    final = _solve(sc, ImplicitUENNModel(), wall_seconds=0.05)
    assert final.coords.wall_ms < 1000.0  # was ~1877ms pre-fix at 50ms budget
    metrics = Evaluator(sc).evaluate(final.link_flows)
    assert metrics["feasible"] == 1.0  # truncation never breaks feasibility


def test_iterations_coordinate_reports_executed_steps():
    """Review MINOR: coords.iterations recorded the CAP (3000) although the
    early stop finished in ~100 steps on Braess — corrupting P6 work-normalized
    comparisons. The executed count is recorded now."""
    final = _solve(braess_scenario(), ImplicitUENNModel(), iterations=3000)
    assert 0 < final.coords.iterations < 1000  # actual steps, far below the cap


def test_training_restores_global_torch_rng():
    """Review MINOR: the first (cold-cache) solve permanently reseeded the
    process-global torch RNG. Training now saves/restores the RNG state."""
    M._TRAINED = None  # force a cold-cache training run
    try:
        torch.manual_seed(777)
        before = torch.get_rng_state().clone()
        _solve(braess_scenario(), ImplicitUENNModel(), iterations=50)
        assert torch.equal(torch.get_rng_state(), before)
    finally:
        pass  # cache repopulated by the training run itself


def test_mixes_with_classical_solvers_in_one_grid():
    """A learned torch model and a classical solver score through the identical
    harness in one grid; the run survives a black box regardless of its output."""
    result = run_experiment(
        braess_scenario(),
        [BiconjugateFrankWolfeModel(), ImplicitUENNModel()],
        Budget(iterations=M._N_FP_ITER, target_relative_gap=1e-8),
        seed=0,
    )
    last = {row["model"]: row for row in result.rows}
    assert "bfw" in last and "implicit-ue-nn" in last
    assert last["bfw"]["feasible"] == 1.0
    assert last["implicit-ue-nn"]["feasible"] == 1.0  # feasible by construction
