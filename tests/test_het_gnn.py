"""Tests for the second (and last Phase-3) torch model — ``het-gnn``.

Liu & Meidani (2024) heterogeneous-GNN UE surrogate, shipped as a lean variant
(docs/design/adr-026). Torch is an OPTIONAL extra; this whole file is skipped on
a core install (``pytest.importorskip('torch')``), and the 731-test numpy suite
runs without it (the torch-free CI matrix legs are the live regression for that).

What these tests pin: the wrapper contract + the golden Braess hash (this additive
model must leave the scored-instance canon byte-identical); the paper-faithful RAW
emission being censored by the demand-feasibility audit (soft conservation is not
a constraint) AND the repo-extension DECODE being feasible with a real certified
gap; the size-agnostic featurization's exact permutation equivariance (A4); the
decode as a genuine projection (feasible by construction, monotone, no limit
cycle); the honest held-out story with every cross-model axis NAMED and MEASURED
(never a pre-committed flattering direction — adr-025's lesson); the ``trained_on``
fairness gate over the augmented instances; and the adr-025 review regressions
(zero-demand short-circuit, wall deadline, executed-step honesty, torch global-
state restoration, training < 60 s).

Cross-platform note: only IN-PROCESS byte determinism is asserted; every held-out
claim is a PROPERTY / directional bound with loose margins, never a pinned trained
byte or tight decimal (the BLAS-sensitivity lesson).
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
)
from tabench.core.capabilities import ContaminationError  # noqa: E402
from tabench.core.scenario import Demand, Network, Scenario  # noqa: E402
from tabench.metrics.gaps import node_balance_residual  # noqa: E402
from tabench.models import het_gnn as M  # noqa: E402
from tabench.models._paths import PathEngine  # noqa: E402
from tabench.models.base import MODEL_REGISTRY  # noqa: E402
from tabench.models.het_gnn import HetGNNModel  # noqa: E402
from tabench.models.implicit_ue import ImplicitUENNModel  # noqa: E402
from tabench.models.learned import TRAINING_FAMILY, _random_network_scenario  # noqa: E402

# The golden Braess content hash: this additive torch model must leave it — and
# thus the whole scored instance canon — byte-identical (HARD RULE).
BRAESS_GOLDEN_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"


def _solve(scenario, model, **budget) -> Trace:
    trace = Trace()
    model.solve(scenario, Budget(**budget), RngBundle(0), trace)
    return trace


def _permute_scenario(base: Scenario, perm: np.ndarray) -> Scenario:
    """Relabel node ``u`` -> ``perm[u]`` (0-based) keeping edge ORDER fixed.

    ``perm`` must map zones (0..n_zones-1) among themselves and non-zones among
    themselves (TNTP convention: zones are nodes 1..n_zones). Real-edge order is
    preserved, so edge ``a`` in both scenarios is the same road segment with
    relabeled endpoints — a correct GNN must emit the identical ``alpha_a``."""
    net = base.network
    nz = net.n_zones
    od = base.demand.matrix
    od_p = np.zeros_like(od)
    for i in range(nz):
        for j in range(nz):
            od_p[perm[i], perm[j]] = od[i, j]
    net_p = Network(
        name="perm",
        n_nodes=net.n_nodes,
        n_zones=nz,
        first_thru_node=net.first_thru_node,
        init_node=perm[net.init_node - 1] + 1,
        term_node=perm[net.term_node - 1] + 1,
        capacity=net.capacity,
        length=net.length,
        free_flow_time=net.free_flow_time,
        b=net.b,
        power=net.power,
        toll=net.toll,
        link_type=net.link_type,
    )
    return Scenario(name="perm", network=net_p, demand=Demand(od_p), family="fuzz-perm")


# ------------------------------------------------------------- wrapper contract
def test_registered_and_capabilities():
    assert "het-gnn" in MODEL_REGISTRY
    caps = HetGNNModel.capabilities
    assert caps.paradigm == "learned"
    assert caps.deterministic is True
    assert caps.seedable is False
    assert caps.provides_gap is False
    # Declares the family AND every training instance's content hash, including
    # the demand-rescaling augmentations (paper Eq 15).
    assert TRAINING_FAMILY in caps.trained_on
    assert len(caps.trained_on) == 1 + len(M._TRAINING_INSTANCES)
    assert len(M._TRAINING_INSTANCES) > len(M._TRAINING_SCENARIOS)  # augmentations added
    # The additive torch model leaves the scored-instance canon untouched.
    assert braess_scenario().content_hash() == BRAESS_GOLDEN_HASH


# ------------------------------------------------------------- analytic anchors
def test_a1_untrained_forward_sanity():
    """A1: an untrained HetGNN forward on the builtin Sioux Falls scenario emits
    finite flow/capacity ratios of shape ``(|E_r|,)`` (paper Eq 10 head)."""
    sc = load_or_skip("siouxfalls")
    g = M._het_graph(sc.network, sc.demand)
    torch.manual_seed(0)
    model = M._HetGNN()
    with torch.no_grad():
        alpha = model(g)
    assert alpha.shape == (sc.network.n_links,)
    assert torch.all(torch.isfinite(alpha))


def test_a2_raw_censored_decoded_feasible():
    """A2: the paper-faithful RAW emission (sp_calls=0) is censored feasible=0 by
    the harness-recomputed node-balance residual — soft conservation is not a
    constraint — while the repo-extension DECODE is feasible=1 with a real
    certified gap. Both checkpoints appear, nothing self-attested."""
    sc = load_or_skip("siouxfalls")
    trace = _solve(sc, HetGNNModel(), iterations=M._N_DECODE)
    assert len(trace) == 2
    # The two checkpoints are uniquely identified by their coords: the raw GNN
    # forward needs no shortest path (sp_calls=0), the decode runs n_cg sweeps.
    raw, dec = trace.checkpoints[0], trace.checkpoints[1]

    assert raw.coords.sp_calls == 0 and raw.coords.iterations == 0  # new budget point
    m_raw = Evaluator(sc).evaluate(raw.link_flows)
    assert m_raw["feasible"] == 0.0
    # harness-recomputed residual, orders above the 1e-6*D tolerance (the paper's
    # own best L~_c lands 3-5 orders above tolerance — certain censoring).
    assert node_balance_residual(sc, raw.link_flows) > 1e-6 * sc.demand.total

    assert dec.coords.sp_calls == M._N_CG
    m_dec = Evaluator(sc).evaluate(dec.link_flows)
    assert m_dec["feasible"] == 1.0
    assert np.isfinite(m_dec["relative_gap"]) and 0.0 < m_dec["relative_gap"] < 1.0
    # provenance (P6): training budget reported, decode residual descriptive.
    assert dec.self_report["training_sp_calls"] > 0
    assert "training_wall_ms" in dec.self_report
    assert "decode_residual" in dec.self_report


def test_a4_permutation_equivariance():
    """A4: the size-agnostic node featurization is EXACTLY permutation equivariant
    — under a zones-among-zones / non-zones-among-non-zones relabeling (TNTP
    convention) with edge order fixed, the raw ratio per edge is invariant to
    float precision. This is what lets one trained model transfer across graph
    sizes; the paper's |V|-row featurization fails it (verified max diff 21.5)."""
    base = _random_network_scenario(2, 10, 4, 6)
    n, nz = base.network.n_nodes, base.network.n_zones
    rng = np.random.default_rng(0)
    perm = np.arange(n)
    perm[:nz] = rng.permutation(nz)  # zones among zones
    perm[nz:] = nz + rng.permutation(n - nz)  # non-zones among non-zones
    sc_p = _permute_scenario(base, perm)

    torch.manual_seed(7)
    model = M._HetGNN()  # equivariance is architectural, holds at any fixed weights
    with torch.no_grad():
        a = model(M._het_graph(base.network, base.demand))
        a_p = model(M._het_graph(sc_p.network, sc_p.demand))
    assert float((a - a_p).abs().max()) < 1e-8


def test_decode_identity_returns_representable_flow():
    """The decode is a PROJECTION, not a solver: fed a demand-feasible flow that
    lies in the span of the column-generated routes it returns it. On Braess the
    analytic UE (4,2,2,2,4) is representable, and so is any Delta^T h0."""
    sc = braess_scenario()
    engine = PathEngine(sc.network)
    rs = M._build_routes(sc.network, sc.demand, engine, M._N_CG)

    v_ue = torch.as_tensor([4.0, 2.0, 2.0, 2.0, 4.0], dtype=M._DTYPE)
    h, residual, _ = M._decode(rs, v_ue, M._N_DECODE)
    v_back = (rs.delta.t() @ h).numpy()
    assert residual < 1e-6
    np.testing.assert_allclose(v_back, v_ue.numpy(), atol=1e-5)
    metrics = Evaluator(sc).evaluate(v_back)
    assert metrics["feasible"] == 1.0
    assert abs(metrics["relative_gap"]) < 1e-6

    # A non-uniform representable target (h0 != the uniform start): the iteration
    # must still drive the projection objective to ~0.
    h0 = torch.as_tensor(np.array([1.0, 2.0, 3.0]), dtype=M._DTYPE)  # sums to demand 6
    assert rs.delta.shape[0] == 3 and rs.n_groups == 1
    v_target = rs.delta.t() @ h0
    _, residual2, steps2 = M._decode(rs, v_target, M._N_DECODE)
    assert steps2 > 0  # it actually moved from the uniform start
    assert residual2 < 1e-4


def test_decode_feasible_by_construction_at_random_weights():
    """At RANDOM untrained weights every DECODED emission passes the demand-
    feasibility audit (node balance exact by the v = Delta^T h mechanism) — the
    architectural property the censored per-link raw emission lacks (mirror of
    implicit-ue A4)."""
    sc = braess_scenario()
    for seed in range(4):
        torch.manual_seed(seed)
        model = M._HetGNN()  # random init, never trained
        g = M._het_graph(sc.network, sc.demand)
        with torch.no_grad():
            v_raw = torch.clamp(model(g), min=0.0) * g["cap"]
        engine = PathEngine(sc.network)
        rs = M._build_routes(sc.network, sc.demand, engine, M._N_CG)
        h, _, _ = M._decode(rs, v_raw, M._N_DECODE)
        v_dec = (rs.delta.t() @ h).numpy()
        assert np.all(np.isfinite(v_dec)) and np.all(v_dec >= -1e-9)
        assert node_balance_residual(sc, v_dec) < 1e-9
        assert Evaluator(sc).evaluate(v_dec)["feasible"] == 1.0


def test_decode_converges_no_limit_cycle():
    """The projected-gradient decode converges without overshoot / limit cycle
    (the repo's recurring fixed-point defect): the L2 projection OBJECTIVE is
    monotone non-increasing in steps, the emitted L-inf residual is near-
    stationary (wiggles <1% after convergence — not monotone), and the decoded
    flow is demand-feasible to machine precision regardless of the projection
    target's infeasibility."""
    sc = load_or_skip("siouxfalls")
    model, _ = M._train()
    g = M._het_graph(sc.network, sc.demand)
    with torch.no_grad():
        v_raw = torch.clamp(model(g), min=0.0) * g["cap"]
    rs = M._build_routes(sc.network, sc.demand, PathEngine(sc.network), M._N_CG)

    h_short, resid_short, _ = M._decode(rs, v_raw, 200)
    h_long, resid_long, _ = M._decode(rs, v_raw, 600)
    # Monotone descent holds for the L2 PROJECTION OBJECTIVE (what projected
    # gradient actually descends); the emitted L-inf residual is near-
    # stationary but NOT monotone — it wiggles <1% after convergence (adr-026
    # review: the old strict L-inf assert passed only by the accidental choice
    # of the (200, 600) step pair).
    def objective(h):
        return float(0.5 * ((rs.delta.t() @ h - v_raw) ** 2).sum())

    assert objective(h_long) <= objective(h_short) * (1.0 + 1e-12)
    assert resid_long <= 1.01 * resid_short  # near-stationary wiggle bound
    assert abs(resid_short - resid_long) / resid_short < 0.05
    # Demand feasibility is exact by construction (node balance ~ machine eps*D).
    v_dec = (rs.delta.t() @ h_long).numpy()
    assert node_balance_residual(sc, v_dec) < 1e-6 * sc.demand.total
    assert Evaluator(sc).evaluate(v_dec)["feasible"] == 1.0


# ------------------------------------------------------ training + held-out story
def test_training_reduces_in_family_flow_loss():
    """Adam on the composite loss genuinely learns: the trained head's in-family
    flow loss is below the untrained head's. This is the in-sample 'trained beats
    untrained' pin; the held-out certified gap is a separate, honestly-scoped
    question (the identifiability caveat, adr-026)."""
    cases = M._training_cases()

    def family_loss(model) -> float:
        total = 0.0
        with torch.no_grad():
            for c in cases:
                total += float(((model(c["g"]) * c["cap"] - c["v_obs"]) ** 2).sum()) / c["scale"]
        return total

    trained, _ = M._train()
    torch.manual_seed(M._TRAIN_SEED)
    untrained = M._HetGNN()
    assert family_loss(trained) < family_loss(untrained)


def test_conservation_loss_reduces_in_family_residual():
    """The paper's contribution (2) made measurable: training WITH the soft
    conservation loss (w_c > 0) yields a lower in-family aggregate node-balance
    residual than training without it (w_c = 0). Scoped IN-FAMILY — where the loss
    is optimized; its held-out transfer is subject to the identifiability caveat
    (measured to NOT transfer on Sioux Falls — adr-026)."""
    cases = M._training_cases()

    def in_family_residual(model) -> float:
        total = 0.0
        with torch.no_grad():
            for c in cases:
                f = model(c["g"]) * c["cap"]
                total += float(M._conservation_residual(f, c["g"], c["expected"])) / c["scale"]
        return total

    prev_threads = torch.get_num_threads()
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_rng = torch.get_rng_state()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        head_wc = M._fit(cases, w_cons=M._W_CONS)
        head_0 = M._fit(cases, w_cons=0.0)
    finally:
        torch.use_deterministic_algorithms(prev_det)
        torch.set_num_threads(prev_threads)
        torch.set_rng_state(prev_rng)
    assert in_family_residual(head_wc) < in_family_residual(head_0)


def test_heldout_directions_and_honest_headline():
    """The honest held-out story on a disjoint TNTP scenario, every axis NAMED and
    MEASURED (never a pre-committed flattering direction — adr-025):

    * the paper-faithful RAW emission is censored (feasible 0), and its flow error
      is WORSE than the ridge surrogate's — the GNN's raw ratios transfer poorly in
      magnitude (but it is censored either way);
    * the repo-extension DECODE is feasible with a finite positive certified gap,
      and its flow error is BETTER than the ridge's — projecting onto the demand-
      feasible polytope recovers accuracy the raw emission lacks;
    * on the certified-gap axis at matched route sets, ``implicit-ue-nn``'s learned-
      cost fixed point beats ``het-gnn``'s GNN-regression + projection (the two
      feasibility mechanisms, isolated);
    * a CONVERGED solver certifies an orders-better gap (the wall/convergence axis).

    Directions only; margins loose; same-platform.
    """
    sc = load_or_skip("siouxfalls")
    assert sc.family != TRAINING_FAMILY
    oracle = sc.reference.link_flows
    wmape = lambda v: float(np.abs(v - oracle).sum() / np.abs(oracle).sum())  # noqa: E731

    trace = _solve(sc, HetGNNModel(), iterations=M._N_DECODE)
    raw, dec = trace.checkpoints[0], trace.checkpoints[1]
    m_raw = Evaluator(sc).evaluate(raw.link_flows)
    m_dec = Evaluator(sc).evaluate(dec.link_flows)
    assert m_raw["feasible"] == 0.0
    assert m_dec["feasible"] == 1.0
    assert np.isfinite(m_dec["relative_gap"]) and 0.0 < m_dec["relative_gap"] < 1.0

    v_ridge = _solve(sc, LearnedSurrogateModel(), iterations=1).final.link_flows
    m_ridge = Evaluator(sc).evaluate(v_ridge)
    assert m_ridge["feasible"] == 0.0  # a per-link regressor routes nobody either
    # Axis 1 — flow error (wmape): raw transfers WORSE than the ridge, decoded BETTER.
    assert wmape(raw.link_flows) > wmape(v_ridge)
    assert wmape(dec.link_flows) < wmape(v_ridge)

    # Axis 2 — certified gap at matched route sets. The measured same-platform
    # ordering at the pinned _TRAIN_EPOCHS=100 is implicit-ue 0.168 < het-gnn
    # 0.259, but the adr-026 review PROVED this direction is NOT a CI
    # invariant: it inverts by ~epoch 130 of het-gnn's own training trajectory
    # and flips under a 1e-6 weight perturbation while the CI torch job
    # installs an unpinned torch. The ordering is therefore recorded as
    # provenance in adr-026, and the test asserts only the STABLE structure:
    # both models feasible with finite gaps in a sane band (the wmape and bfw
    # directions above/below survived the same perturbation sweep; this one
    # did not).
    v_impl = _solve(sc, ImplicitUENNModel(), iterations=3000).final.link_flows
    m_impl = Evaluator(sc).evaluate(v_impl)
    assert m_impl["feasible"] == 1.0
    assert 0.02 < m_impl["relative_gap"] < 0.6
    assert 0.02 < m_dec["relative_gap"] < 0.6

    # Axis 3 — a CONVERGED bfw certifies an orders-better gap.
    v_bfw = _solve(sc, BiconjugateFrankWolfeModel(), iterations=300, target_relative_gap=1e-6)
    m_bfw = Evaluator(sc).evaluate(v_bfw.final.link_flows)
    assert m_bfw["relative_gap"] < 1e-4 < m_dec["relative_gap"]


# --------------------------------------------------------------- fairness gate
def test_fairness_gate_blocks_family_and_augmented():
    """Declares trained_on = ('synthetic-net', base hashes..., augmented hashes...);
    evaluating on a training-family scenario OR on a demand-rescaled augmentation
    is refused (train/test contamination), mirroring learned-surrogate."""
    train_scenario = _random_network_scenario(1, 8, 3, 4)
    assert train_scenario.family == TRAINING_FAMILY
    with pytest.raises(ContaminationError):
        run_experiment(train_scenario, [HetGNNModel()], Budget(iterations=1), seed=0)
    # The augmentations are declared by hash, so the gate blocks them even though
    # they are not in _TRAINING_SCENARIOS.
    aug = M._TRAINING_INSTANCES[len(M._TRAINING_SCENARIOS)]
    assert aug.content_hash() in HetGNNModel.capabilities.trained_on


# ---------------------------------------------------- budget / provenance (P6)
def test_sp_calls_and_budget_accounting():
    """The raw checkpoint sits at sp_calls=0 (a new budget point below the ridge's
    1 — the GNN forward needs no shortest path); the decoded checkpoint's sp_calls
    are the real column-generation Dijkstra sweeps, capped by the budget."""
    sc = braess_scenario()
    full = _solve(sc, HetGNNModel(), iterations=M._N_DECODE)
    assert full.checkpoints[0].coords.sp_calls == 0
    assert full.checkpoints[1].coords.sp_calls == M._N_CG > 1
    capped = _solve(sc, HetGNNModel(), sp_calls=3, iterations=40)
    assert capped.checkpoints[1].coords.sp_calls == 3
    assert capped.checkpoints[1].coords.iterations <= 40


def test_training_wall_time_budget_and_no_smearing():
    """Training is a design commitment to stay well under a minute of CPU (so the
    torch CI job's long pole cannot silently grow), and its one-time cost is
    reported as provenance — never smeared into the inference wall: the cold-cache
    solve's inference wall stays close to a warm solve's (training happens before
    the inference clock starts)."""
    M._TRAINED = None
    start = time.perf_counter()
    M._train()
    assert time.perf_counter() - start < 60.0

    M._TRAINED = None
    cold = _solve(braess_scenario(), HetGNNModel(), iterations=M._N_DECODE).final
    warm = _solve(braess_scenario(), HetGNNModel(), iterations=M._N_DECODE).final
    assert cold.self_report["training_wall_ms"] > 0
    # Inference wall excludes training (measured after the cache lookup), so cold
    # and warm agree up to noise despite the cold run having just trained.
    assert cold.coords.wall_ms < 5.0 * warm.coords.wall_ms + 50.0


def test_iterations_coordinate_reports_executed_steps():
    """coords.iterations is the executed decode-step count, not the cap: a tight
    wall budget truncates the loop and the recorded count reflects the truncation
    (never the full _N_DECODE cap)."""
    sc = load_or_skip("siouxfalls")
    full = _solve(sc, HetGNNModel(), iterations=M._N_DECODE)
    truncated = _solve(sc, HetGNNModel(), iterations=M._N_DECODE, wall_seconds=0.02)
    assert truncated.checkpoints[1].coords.iterations < full.checkpoints[1].coords.iterations


# ------------------------------------------------- adr-025 review regressions
def test_zero_demand_scenarios_emit_zero_flows():
    """All-zero and diagonal-only OD matrices short-circuit — before any
    PER-SCENARIO torch work; the cached one-time training still runs and is
    reported as provenance — to the exact zero equilibrium (the empty route set
    would otherwise crash _build_routes, and the raw GNN emission would be a
    phantom censored flow)."""
    base = braess_scenario()
    for od in (np.zeros((2, 2)), np.diag([5.0, 3.0])):
        sc = Scenario(name="z", network=base.network, demand=Demand(od), family="fuzz")
        final = _solve(sc, HetGNNModel(), iterations=10).final
        assert np.array_equal(final.link_flows, np.zeros(base.network.n_links))
        metrics = Evaluator(sc).evaluate(final.link_flows)
        assert metrics["feasible"] == 1.0
        assert metrics["relative_gap"] == pytest.approx(0.0, abs=1e-12)


def test_wall_seconds_budget_is_respected():
    """A wall-only budget stops both column generation and the decode loop; the
    non-interruptible tail (one sweep + the raw forward) gets generous slack, and
    truncation never breaks the decoded flow's feasibility (structural)."""
    sc = load_or_skip("siouxfalls")
    final = _solve(sc, HetGNNModel(), wall_seconds=0.05).final
    assert final.coords.wall_ms < 1000.0
    assert Evaluator(sc).evaluate(final.link_flows)["feasible"] == 1.0


def test_training_restores_global_torch_state():
    """The first (cold-cache) solve must not perturb the process-global torch RNG,
    thread count, or deterministic-algorithms flag (the complete adr-025 review
    restoration set)."""
    M._TRAINED = None
    torch.manual_seed(999)
    before_rng = torch.get_rng_state().clone()
    before_threads = torch.get_num_threads()
    before_det = torch.are_deterministic_algorithms_enabled()
    _solve(braess_scenario(), HetGNNModel(), iterations=50)
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert torch.get_num_threads() == before_threads
    assert torch.are_deterministic_algorithms_enabled() == before_det


def test_deterministic_in_process():
    """Two solves are byte-identical in-process (determinism asserted same-platform
    only — no cross-platform byte claims)."""
    a = _solve(braess_scenario(), HetGNNModel(), iterations=M._N_DECODE).final.link_flows
    b = _solve(braess_scenario(), HetGNNModel(), iterations=M._N_DECODE).final.link_flows
    np.testing.assert_array_equal(a, b)


def test_mixes_with_classical_solvers_in_one_grid():
    """A learned torch model with TWO checkpoints (raw + decoded) and a classical
    solver score through the identical harness in one grid: both het-gnn rows
    appear (the censored paper-faithful raw and the feasible decoded), and the run
    survives regardless of any black box's output."""
    result = run_experiment(
        braess_scenario(),
        [BiconjugateFrankWolfeModel(), HetGNNModel()],
        Budget(iterations=M._N_DECODE, target_relative_gap=1e-8),
        seed=0,
    )
    het_rows = [r for r in result.rows if r["model"] == "het-gnn"]
    assert len(het_rows) == 2  # raw (censored) + decoded (feasible) both certified
    assert {r["feasible"] for r in het_rows} == {0.0, 1.0}
    bfw_rows = [r for r in result.rows if r["model"] == "bfw"]
    assert bfw_rows and bfw_rows[-1]["feasible"] == 1.0
