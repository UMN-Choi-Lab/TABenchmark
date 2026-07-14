"""Tests for od-dynamic: Cascetta, Inaudi & Marquis (1993) within-day dynamic OD
estimation (ADR-023).

The paper estimates a sequence of time-slice OD matrices ``d_h`` from time-sliced
link counts linked by an exogenous lagged assignment map, via a SIMULTANEOUS GLS
(all slices jointly, efficient) and a SEQUENTIAL GLS (slice by slice, earlier
estimates frozen, online-capable but provably less efficient). Every anchor is
recomputed in-test as a closed form (house style: no trusted digits); the exact
rationals in the comments are the recomputed values, not inputs.
"""

import dataclasses
from fractions import Fraction

import numpy as np
import pytest

from tabench import (
    Budget,
    Demand,
    RngBundle,
    braess_scenario,
    two_route_scenario,
)
from tabench.core.rng import SOURCE_OBSERVATION
from tabench.estimation import (
    DYNAMIC_ESTIMATOR_REGISTRY,
    ESTIMATOR_REGISTRY,
    DynamicEstimationTask,
    DynamicPriorBaseline,
    GLSEstimator,
    ODTrace,
    SequentialDynamicGLSEstimator,
    SimultaneousDynamicGLSEstimator,
    dynamic_gls_sequential,
    dynamic_gls_simultaneous,
    lagged_assignment_tensor,
    predict_interval_counts,
    stacked_tensor_map,
    tensor_blocks,
)
from tabench.estimation._proportions import active_pairs
from tabench.estimation.base import EstimationTask
from tabench.experiments.runner import (
    dynamic_identifiability_report,
    run_dynamic_estimation_experiment,
)
from tabench.metrics.estimation_dynamic import DynamicODCertifier
from tabench.observe.levels import DynamicLinkCounts, LinkCounts

BRAESS_GOLDEN_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"
# A deterministic static-T2 task hash: adding DynamicLinkCounts must not perturb
# the static LinkCounts / StalePriorOD / EstimationTask hashing (byte-identical).
STATIC_T2_TASK_HASH = "8350e5145a234f08a981e371c59381168a6a94e311e3ebfc20a305bff14f2781"


def _tensor(m0: float, m1: float) -> np.ndarray:
    """Single-pair single-sensor lag tensor with lag-0/lag-1 fractions (m0, m1)."""
    m = np.zeros((2, 1, 1), dtype=np.float64)
    m[0, 0, 0] = m0
    m[1, 0, 0] = m1
    return m


def _sim(m, counts, prior, w_prior, v_count, n_intervals):
    n_slices, n_pairs = prior.shape
    a = stacked_tensor_map(m, n_slices, n_intervals)
    return dynamic_gls_simultaneous(
        a, counts.reshape(-1), prior.reshape(-1), w_prior.reshape(-1), v_count.reshape(-1)
    ).reshape(n_slices, n_pairs)


def _seq(m, counts, prior, w_prior, v_count, n_intervals):
    blocks = tensor_blocks(m, prior.shape[0], n_intervals)
    return dynamic_gls_sequential(blocks, counts, prior, w_prior, v_count, n_intervals)


# ----------------------------------------------------------------- Anchor A1


def test_a1_integer_lag_reduces_to_static_gls_sim_equals_seq():
    """A1: with tau = Delta exactly (M0=0, M1=p), each slice couples to exactly one
    interval, so the stacked GLS block-diagonalizes and BOTH estimators reduce per
    slice to the static gls closed form d_h = (z_h + p c_{h+1})/(1 + p^2)."""
    p = 1.0
    m = _tensor(0.0, p)
    n_slices, n_intervals = 3, 4  # T = H + 1 so the last slice is observed at lag 1
    truth = np.array([[4.0], [6.0], [5.0]])
    counts = predict_interval_counts(m, truth, n_intervals)  # (0, 4, 6, 5)
    assert np.allclose(counts.reshape(-1), [0.0, 4.0, 6.0, 5.0])
    prior = np.full((n_slices, 1), 3.0)
    w, v = np.ones((n_slices, 1)), np.ones((n_intervals, 1))
    sim = _sim(m, counts, prior, w, v, n_intervals)
    seq = _seq(m, counts, prior, w, v, n_intervals)
    closed = np.array(
        [[(prior[h, 0] + p * counts[h + 1, 0]) / (1.0 + p * p)] for h in range(n_slices)]
    )
    assert np.allclose(sim, closed, atol=1e-12)
    assert np.allclose(seq, closed, atol=1e-12)
    assert np.allclose(sim, seq, atol=1e-12)


# ----------------------------------------------------------------- Anchor A2


def test_a2_fractional_lag_simultaneous_strictly_dominates_sequential():
    """A2 (THE anchor): tau = Delta/2 (M0 = M1 = 1/2), H=2, T=3, truth (4,6),
    prior (3,3), V=W=I. The simultaneous solve lets c_2, c_3 revise slice 1 through
    the lag entry, so it strictly dominates the sequential (which freezes slice 1
    from c_1 alone). Exact rationals recomputed in-test with fractions."""
    m = _tensor(0.5, 0.5)
    n_intervals = 3
    truth = np.array([[4.0], [6.0]])
    counts = predict_interval_counts(m, truth, n_intervals)
    assert np.allclose(counts.reshape(-1), [2.0, 5.0, 3.0])
    prior = np.array([[3.0], [3.0]])
    w, v = np.ones((2, 1)), np.ones((3, 1))
    sim = _sim(m, counts, prior, w, v, n_intervals).reshape(-1)
    seq = _seq(m, counts, prior, w, v, n_intervals).reshape(-1)

    # Exact simultaneous closed form: (I + A'A) d = z + A'c, solved over the rationals.
    half = Fraction(1, 2)
    a_rows = [[half, Fraction(0)], [half, half], [Fraction(0), half]]
    ztc = [Fraction(3), Fraction(3)]  # prior
    for t in range(3):
        for h in range(2):
            ztc[h] += a_rows[t][h] * Fraction(int(counts[t, 0]))
    ata = [[sum(a_rows[t][i] * a_rows[t][j] for t in range(3)) for j in range(2)] for i in range(2)]
    hmat = [[ata[i][j] + (1 if i == j else 0) for j in range(2)] for i in range(2)]
    det = hmat[0][0] * hmat[1][1] - hmat[0][1] * hmat[1][0]
    sim_exact = [
        (hmat[1][1] * ztc[0] - hmat[0][1] * ztc[1]) / det,
        (hmat[0][0] * ztc[1] - hmat[1][0] * ztc[0]) / det,
    ]
    assert sim_exact == [Fraction(128, 35), Fraction(142, 35)]
    assert np.allclose(sim, [float(sim_exact[0]), float(sim_exact[1])], atol=1e-12)

    # Exact sequential: d_1 from c_1 (lag 0), then d_2 from c_2 minus frozen d_1 (lag 1).
    d1 = (Fraction(3) + half * Fraction(2)) / (1 + half * half)
    resid2 = Fraction(5) - half * d1
    d2 = (Fraction(3) + half * resid2) / (1 + half * half)
    assert [d1, d2] == [Fraction(16, 5), Fraction(94, 25)]
    assert np.allclose(seq, [float(d1), float(d2)], atol=1e-12)

    # Simultaneous strictly dominates the sequential componentwise vs truth.
    assert np.all(np.abs(sim - truth.reshape(-1)) < np.abs(seq - truth.reshape(-1)))


# ----------------------------------------------------------------- Anchor A2b


def test_a2b_pure_math_pin_and_rank1_efficiency_gap():
    """A2b (pure-math pin, time-VARYING stacked map M=[[.5,0],[.5,1]], z=(10,10),
    y=(7,14)): simultaneous (116/11, 103/11), sequential (10.8, 9.3), and the
    no-prior square case A^-1 y = (14, 7). The sequential-minus-simultaneous error
    covariance gap is exactly rank-1 PSD (eigenvalues 0 and 17/220) -- a
    test-local derivation of the plug-in scheme's efficiency loss, NOT attributed
    to the paper and with no shipped covariance API."""
    blocks = [{0: np.array([[0.5]])}, {0: np.array([[0.5]]), 1: np.array([[1.0]])}]
    a = np.array([[0.5, 0.0], [0.5, 1.0]])
    prior = np.array([[10.0], [10.0]])
    counts = np.array([[7.0], [14.0]])
    w, v = np.ones((2, 1)), np.ones((2, 1))
    sim = dynamic_gls_simultaneous(
        a, counts.reshape(-1), prior.reshape(-1), w.reshape(-1), v.reshape(-1)
    )
    seq = dynamic_gls_sequential(blocks, counts, prior, w, v, 2).reshape(-1)

    # Exact simultaneous (I + A'A) d = z + A'c over the rationals.
    F = Fraction
    am = [[F(1, 2), F(0)], [F(1, 2), F(1)]]
    y = [F(7), F(14)]
    z = [F(10), F(10)]
    ata = [[sum(am[t][i] * am[t][j] for t in range(2)) for j in range(2)] for i in range(2)]
    hmat = [[ata[i][j] + (1 if i == j else 0) for j in range(2)] for i in range(2)]
    rhs = [z[i] + sum(am[t][i] * y[t] for t in range(2)) for i in range(2)]
    det = hmat[0][0] * hmat[1][1] - hmat[0][1] * hmat[1][0]
    sim_exact = [
        (hmat[1][1] * rhs[0] - hmat[0][1] * rhs[1]) / det,
        (hmat[0][0] * rhs[1] - hmat[1][0] * rhs[0]) / det,
    ]
    assert sim_exact == [Fraction(116, 11), Fraction(103, 11)]
    assert np.allclose(sim, [float(x) for x in sim_exact], atol=1e-11)

    x0 = (F(10) + F(1, 2) * F(7)) / (1 + F(1, 4))  # 10.8
    x1 = (F(10) + (F(14) - F(1, 2) * x0)) / 2  # 9.3
    assert (float(x0), float(x1)) == (10.8, 9.3)
    assert np.allclose(seq, [float(x0), float(x1)], atol=1e-11)

    # No-prior identifiable square case: xhat = A^-1 y = (14, 7).
    assert np.allclose(np.linalg.solve(a, [7.0, 14.0]), [14.0, 7.0])

    # Efficiency gap: derive both linear estimators' error covariances (V=W=I).
    mmat = np.array([[0.5, 0.0], [0.5, 1.0]])
    hs = np.eye(2) + mmat.T @ mmat
    sig_sim = np.linalg.inv(hs)
    ksim_y = np.linalg.solve(hs, mmat.T)
    sig_sim2 = (ksim_y @ mmat - np.eye(2)) @ (ksim_y @ mmat - np.eye(2)).T + ksim_y @ ksim_y.T
    assert np.allclose(sig_sim, sig_sim2, atol=1e-12)  # posterior cov == error cov
    kz = np.zeros((2, 2))
    ky = np.zeros((2, 2))
    kz[0, 0], ky[0, 0] = 1.0 / 1.25, 0.5 / 1.25  # x0 = (z0 + .5 y0)/1.25
    kz[1, 1], ky[1, 1] = 0.5, 0.5  # x1 = (z1 + (y1 - .5 x0))/2, split below
    kz[1] += -0.25 * kz[0]
    ky[1] += -0.25 * ky[0]
    assert np.allclose(kz + ky @ mmat, np.eye(2), atol=1e-12)  # unbiased at the mean
    sig_seq = (ky @ mmat - np.eye(2)) @ (ky @ mmat - np.eye(2)).T + ky @ ky.T
    gap = sig_seq - sig_sim
    evals = np.sort(np.linalg.eigvalsh(gap))
    assert evals[0] == pytest.approx(0.0, abs=1e-12)  # rank 1
    assert evals[1] > 1e-9 and evals.min() > -1e-12  # PSD, one positive eigenvalue
    assert evals[1] == pytest.approx(17.0 / 220.0, abs=1e-12)
    assert np.trace(sig_seq) > np.trace(sig_sim)  # sequential strictly less efficient


# ----------------------------------------------------------------- Anchor A3


def test_a3_mean_collapse_distinctness():
    """A3: truths (4,6) and (6,4) have IDENTICAL interval means but DISTINCT
    per-interval counts; od-dynamic-sim separates them, while any estimator that
    collapses counts over t (as every static T2 estimator does) provably cannot."""
    m = _tensor(0.5, 0.5)
    ta = np.array([[4.0], [6.0]])
    tb = np.array([[6.0], [4.0]])
    ca = predict_interval_counts(m, ta, 3).reshape(-1)
    cb = predict_interval_counts(m, tb, 3).reshape(-1)
    assert ca.mean() == pytest.approx(cb.mean())  # same mean (the collapse target)
    assert not np.allclose(ca, cb)  # distinct per-interval counts
    prior = np.array([[5.0], [5.0]])
    w, v = np.ones((2, 1)), np.ones((3, 1))
    est_a = _sim(m, ca.reshape(3, 1), prior, w, v, 3).reshape(-1)
    est_b = _sim(m, cb.reshape(3, 1), prior, w, v, 3).reshape(-1)
    assert not np.allclose(est_a, est_b)  # sim uses the time axis, so it separates
    # Each estimate leans toward its own truth's ordering.
    assert (est_a[1] > est_a[0]) and (est_b[0] > est_b[1])


# ----------------------------------------------------------------- Anchor A4


def _confounded_map():
    """Full map with a cross-slice temporal confound on BOTH obs and held-out
    sensors: pair A at lag 0, pair B at lag 1, identical fractions on links 0, 1."""
    m = np.zeros((2, 2, 2), dtype=np.float64)  # (L+1=2, n_links=2, n_pairs=2)
    m[0, :, 0] = 1.0  # pair A (col 0) crosses both links at lag 0
    m[1, :, 1] = 1.0  # pair B (col 1) crosses both links at lag 1
    return m


def test_a4_temporal_confounding_flags_and_count_invariant_shift():
    """A4: pair A at lag 0 and pair B at lag 1 give identical stacked columns for
    (slice h, B) and (slice h+1, A), so od_identifiable=0 and n_confounded_columns
    > 0. A count-invariant shift between the confounded columns leaves EVERY
    sensor's counts unchanged -- obs AND held-out, because held-out shares the lag
    structure (the genuinely new false-accept surface vs static T2)."""
    full = _confounded_map()
    pairs = [(0, 2), (1, 2)]
    n_slices, n_intervals = 3, 4
    rep = dynamic_identifiability_report(full, np.array([0]), pairs, n_slices, n_intervals)
    assert rep["n_confounded_columns"] > 0
    assert rep["linear_identifiable"] is False
    assert rep["hazelton_condition"] is False

    truth1 = np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])
    truth2 = truth1.copy()
    truth2[0, 1] -= 1.0  # (slice 0, pair B) -> interval 1 at lag 1
    truth2[1, 0] += 1.0  # (slice 1, pair A) -> interval 1 at lag 0 : cancels exactly
    obs_map = full[:, [0], :]
    ho_map = full[:, [1], :]
    assert np.allclose(
        predict_interval_counts(obs_map, truth1, n_intervals),
        predict_interval_counts(obs_map, truth2, n_intervals),
    )
    assert np.allclose(
        predict_interval_counts(ho_map, truth1, n_intervals),
        predict_interval_counts(ho_map, truth2, n_intervals),
    )
    assert not np.allclose(truth1, truth2)  # a genuinely different profile


# ----------------------------------------------------------------- Anchor A5


def test_a5_horizon_truncation_dumped_demand_is_count_invariant():
    """A5: with an integer-lag map (M0=0, M1=1) and T = H, the last slice's only
    interval falls outside the horizon -- its stacked column block is all-zero, so
    n_truncated_slices=1. Demand dumped into that slice leaves EVERY count (obs and
    held-out) unchanged while od_rmse / total_demand_error move -- the executable
    warning that OD columns never rank."""
    full = np.zeros((2, 2, 1), dtype=np.float64)  # (L+1=2, n_links=2, n_pairs=1)
    full[1, :, 0] = 1.0  # single pair crosses both links only at lag 1
    pairs = [(0, 1)]
    n_slices = n_intervals = 3
    rep = dynamic_identifiability_report(full, np.array([0]), pairs, n_slices, n_intervals)
    assert rep["n_truncated_slices"] == 1
    assert rep["linear_identifiable"] is False

    truth = np.array([[3.0], [4.0], [5.0]])
    dumped = truth.copy()
    dumped[2, 0] += 100.0  # dump into the unobservable last slice
    obs_map = full[:, [0], :]
    ho_map = full[:, [1], :]
    assert np.allclose(
        predict_interval_counts(obs_map, truth, n_intervals),
        predict_interval_counts(obs_map, dumped, n_intervals),
    )
    assert np.allclose(
        predict_interval_counts(ho_map, truth, n_intervals),
        predict_interval_counts(ho_map, dumped, n_intervals),
    )
    assert not np.allclose(truth, dumped)  # OD moved, counts did not


# --------------------------------------------------------- registry + hashes


def test_registry_separation():
    """Dynamic estimators live in their own registry, never in the static one, and
    vice versa -- the type gate that stops the CLI running a dynamic estimator on a
    static task (ADR-023)."""
    assert "od-dynamic-sim" in DYNAMIC_ESTIMATOR_REGISTRY
    assert "od-dynamic-seq" in DYNAMIC_ESTIMATOR_REGISTRY
    assert "prior-profile" in DYNAMIC_ESTIMATOR_REGISTRY
    for name in DYNAMIC_ESTIMATOR_REGISTRY:
        assert name not in ESTIMATOR_REGISTRY
    for name in ESTIMATOR_REGISTRY:
        assert name not in DYNAMIC_ESTIMATOR_REGISTRY
    caps = SimultaneousDynamicGLSEstimator().capabilities
    assert caps.paradigm == "estimation"
    assert caps.deterministic is True
    assert caps.inputs_required == frozenset({"dynamic_link_counts", "prior_od_profile"})
    assert caps.outputs == frozenset({"od_profile_estimate"})


def test_golden_braess_hash_and_static_task_hash_preserved():
    """od-dynamic is additive: no scenario field and no static-T2 hashing changed,
    so the golden Braess content hash and a pinned static task hash are byte-identical."""
    assert braess_scenario().content_hash() == BRAESS_GOLDEN_HASH
    sc = braess_scenario(6.0)
    ds = LinkCounts(np.array([1, 2, 3]), n_periods=10, noise="none").observe(
        sc, np.array([4.0, 2.0, 2.0, 2.0, 4.0]), RngBundle(0).generator(SOURCE_OBSERVATION)
    )
    task = EstimationTask(
        "t", sc.network, Demand(sc.demand.matrix.copy()), ds, {}, sc.content_hash(), seed=0
    )
    assert task.content_hash() == STATIC_T2_TASK_HASH


# ---------------------------------------------------------- map determinism


def _build_dyn_task(sc, slice_length, n_slices, n_days) -> DynamicEstimationTask:
    """Build a DynamicEstimationTask exactly as the runner does (noise='none')."""
    pairs = active_pairs(sc.demand.matrix)
    full = lagged_assignment_tensor(sc.network, pairs, slice_length)
    n_lags = full.shape[0] - 1
    n_intervals = n_slices + n_lags
    rho = np.linspace(0.6, 1.4, n_slices)
    truth = rho[:, None, None] * sc.demand.matrix[None, :, :]
    truth_pairs = np.array(
        [[truth[h, i, j] for (i, j) in pairs] for h in range(n_slices)]
    ).reshape(n_slices, len(pairs))
    obs_sensors = np.array([3])
    expected = predict_interval_counts(full, truth_pairs, n_intervals)
    obs_ds = DynamicLinkCounts(obs_sensors, n_days, "none").observe(
        sc, expected, RngBundle(0).generator(SOURCE_OBSERVATION)
    )
    dataset = dataclasses.replace(
        obs_ds,
        payload={**obs_ds.payload, "lag_tensor": full[:, obs_sensors, :],
                 "pairs": np.asarray(pairs, dtype=np.int64)},
        meta={**obs_ds.meta, "n_slices": n_slices, "slice_length": slice_length,
              "n_lags": n_lags, "map_recipe": "frozen_freeflow_v1"},
    )
    return DynamicEstimationTask(
        "t", sc.network, truth, dataset, {}, sc.content_hash(), seed=0
    )


def test_map_determinism_and_task_hash_sensitivity():
    """The frozen free-flow map is deterministic (RNG-free), and the task content
    hash is sensitive to the slicing dials (slice length, n_slices, n_days)."""
    sc = two_route_scenario(sue_theta=None)
    pairs = active_pairs(sc.demand.matrix)
    m1 = lagged_assignment_tensor(sc.network, pairs, 2.0)
    m2 = lagged_assignment_tensor(sc.network, pairs, 2.0)
    assert np.array_equal(m1, m2)
    assert not np.array_equal(
        m1, lagged_assignment_tensor(sc.network, pairs, 3.0)[: m1.shape[0]]
    )

    h0 = _build_dyn_task(sc, 2.0, 3, 6).content_hash()
    assert h0 == _build_dyn_task(sc, 2.0, 3, 6).content_hash()  # deterministic
    assert len(h0) == 64  # SHA-256 hex digest
    assert h0 != _build_dyn_task(sc, 1.0, 3, 6).content_hash()  # slice_length
    assert h0 != _build_dyn_task(sc, 2.0, 4, 6).content_hash()  # n_slices
    assert h0 != _build_dyn_task(sc, 2.0, 3, 9).content_hash()  # n_days


# -------------------------------------------------------------- certifier


def _braess_certifier(noise="none", cv=0.0):
    sc = braess_scenario(6.0)
    pairs = active_pairs(sc.demand.matrix)
    slice_length, n_slices = 7.0, 3
    full = lagged_assignment_tensor(sc.network, pairs, slice_length)
    n_lags = full.shape[0] - 1
    n_intervals = n_slices + n_lags
    rho = np.array([0.6, 1.0, 1.4])
    truth = rho[:, None, None] * sc.demand.matrix[None, :, :]
    truth_pairs = np.array(
        [[truth[h, i, j] for (i, j) in pairs] for h in range(n_slices)]
    ).reshape(n_slices, len(pairs))
    obs_sensors, ho_sensors = np.array([2, 4]), np.array([0])
    exp = predict_interval_counts(full, truth_pairs, n_intervals)
    obs_counts = exp[:, obs_sensors][None, :, :]
    ho_counts = exp[:, ho_sensors][None, :, :]
    ident = dynamic_identifiability_report(full, obs_sensors, pairs, n_slices, n_intervals)
    cert = DynamicODCertifier(
        sc, obs_sensors, ho_sensors, obs_counts, ho_counts, truth,
        slice_length, n_lags, n_intervals, ident,
    )
    return cert, truth


def test_certifier_censoring_conventions():
    """Censoring mirrors ODCertifier: wrong shape raises; non-finite and
    sub-tolerance negatives are censored; a zero profile is NOT censored."""
    cert, truth = _braess_certifier()
    # Truth certifies at the oracle floor (zero count RMSE here, noise='none').
    good = cert.certify(truth)
    assert good["od_feasible"] == 1.0
    assert good["obs_count_rmse"] == pytest.approx(0.0, abs=1e-9)
    assert good["od_rmse"] == pytest.approx(0.0, abs=1e-9)
    # Wrong shape -> raise.
    with pytest.raises(ValueError):
        cert.certify(truth[:, :, 0])
    # Non-finite -> censored.
    bad = truth.copy()
    bad[0, 0, 1] = np.nan
    assert cert.certify(bad)["od_feasible"] == 0.0
    # Large negative -> censored.
    neg = truth.copy()
    neg[0, 0, 1] = -5.0
    assert cert.certify(neg)["od_feasible"] == 0.0
    # Zero profile -> NOT censored (a legitimate, terrible estimate).
    zero = cert.certify(np.zeros_like(truth))
    assert zero["od_feasible"] == 1.0
    assert zero["obs_count_rmse"] > 0.0


# ------------------------------------------------- adr-023 review regressions


def test_offsupport_demand_is_censored():
    """Review MAJOR: demand emitted on a cell that is neither an active pair nor
    the diagonal has NO lag column, so it was invisible to EVERY count column —
    obs, held-out, and the ranking heldout_count_rmse — while od_identifiable
    claimed 1. Off-support mass beyond tolerance is now censored (the dynamic
    analogue of the static track loading the FULL emitted matrix)."""
    cert, truth = _braess_certifier()
    hostile = truth.copy()
    # find an off-diagonal cell that is NOT an active pair
    sc_pairs = {tuple(p) for p in cert.pairs}
    z = truth.shape[1]
    cell = next(
        (i, j) for i in range(z) for j in range(z) if i != j and (i, j) not in sc_pairs
    )
    hostile[:, cell[0], cell[1]] += 1e6
    bad = cert.certify(hostile)
    assert bad["od_feasible"] == 0.0
    assert np.isnan(bad["heldout_count_rmse"])
    # sub-tolerance off-support dirt is NOT censored (tolerance semantics)
    dirt = truth.copy()
    dirt[0, cell[0], cell[1]] += 1e-12
    assert cert.certify(dirt)["od_feasible"] == 1.0
    # diagonal (intrazonal) mass remains allowed pass-through
    diag = truth.copy()
    diag[0, 0, 0] += 1e9
    assert cert.certify(diag)["od_feasible"] == 1.0


def test_negativity_gate_immune_to_diagonal_scale():
    """Review MINOR: a huge intrazonal diagonal cell inflated the negativity
    tolerance (scale = |q|.max()) so a genuinely negative active cell escaped
    censoring. The scale is now off-diagonal only — in BOTH certifiers."""
    cert, truth = _braess_certifier()
    neg = truth.copy()
    neg[0, 0, 1] = -500.0
    neg[0, 0, 0] = 1e12  # diagonal mass must not buy negativity tolerance
    assert cert.certify(neg)["od_feasible"] == 0.0


def test_prior_var_floor_must_be_positive():
    """Review MAJOR: prior_var_floor=0.0 with a zero prior cell made a whitened
    row infinite and HUNG lsq_linear. The factor bound now excludes 0, and the
    lower bound solves a zero-prior-cell instance without hanging."""
    with pytest.raises(ValueError):
        SimultaneousDynamicGLSEstimator(prior_var_floor=0.0)
    with pytest.raises(ValueError):
        SequentialDynamicGLSEstimator(prior_var_floor=0.0)
    # at the smallest allowed floor, a zero prior cell still solves cleanly
    m = _tensor(0.5, 0.5)
    prior = np.array([[0.0], [3.0]])  # zero prior cell
    counts = predict_interval_counts(m, np.array([[4.0], [6.0]]), 3)
    w = (0.3 * prior) ** 2 + 1e-12
    v = np.ones((3, 1))
    x = _sim(m, counts, prior, w, v, 3)
    assert np.all(np.isfinite(x))
    assert x[0, 0] == pytest.approx(0.0, abs=1e-6)  # pinned at the zero prior


def test_bogus_map_recipe_is_rejected():
    """Review MINOR: any estimation.map string was silently accepted and flowed
    into the certificate/manifest/hash while the v1 map was built anyway."""
    sc = two_route_scenario(sue_theta=None)
    with pytest.raises(ValueError, match="recipe"):
        run_dynamic_estimation_experiment(
            sc, [DynamicPriorBaseline()], Budget(sp_calls=100),
            estimation=_card(map="evil_v9"),
        )


def test_lag_offsets_use_pure_time_not_tolls():
    """Review MINOR: tau accumulated the free-flow GENERALIZED cost, so a toll
    (money, not minutes) shifted which interval a trip crossed a counter in.
    Adding a toll that leaves route choice unchanged must not move the tensor."""
    sc = two_route_scenario(sue_theta=None)
    pairs = active_pairs(sc.demand.matrix)
    base = lagged_assignment_tensor(sc.network, pairs, 2.0)
    tolled_net = dataclasses.replace(
        sc.network,
        toll=np.full(sc.network.n_links, 0.25),
        name=sc.network.name + "-tolled",
    )
    tolled = lagged_assignment_tensor(tolled_net, pairs, 2.0)
    assert np.array_equal(base, tolled)


def test_task_hash_covers_pairs():
    """Review MINOR: payload['pairs'] was unhashed, so hand-built tasks whose
    estimands live in different OD cells hashed identically (now v2 domain)."""
    sc = two_route_scenario(sue_theta=None)
    task = _build_dyn_task(sc, 2.0, 3, 6)
    swapped = dataclasses.replace(
        task.dataset,
        payload={**task.dataset.payload,
                 "pairs": task.dataset.payload["pairs"][:, ::-1].copy()},
    )
    other = dataclasses.replace(task, dataset=swapped)
    assert task.content_hash() != other.content_hash()


def test_degenerate_cards_fail_cleanly():
    """Review MINOR + NOTE: n_slices=0 crashed deep inside scipy; an empty
    held-out set silently NaN'd the ranking column. Both are clean errors now."""
    sc = two_route_scenario(sue_theta=None)
    with pytest.raises(ValueError, match="n_slices"):
        run_dynamic_estimation_experiment(
            sc, [DynamicPriorBaseline()], Budget(sp_calls=100),
            estimation=_card(n_slices=0),
        )
    with pytest.raises(ValueError, match="held-out"):
        run_dynamic_estimation_experiment(
            sc, [DynamicPriorBaseline()], Budget(sp_calls=100),
            estimation=_card(heldout={"kind": "explicit", "links": []}),
        )


def test_identifiability_reports_sigma_min_and_gates_near_singular():
    """Review MINOR: the rank gate was a float SVD sold as 'exact'. The report
    now carries sigma_min, and near-singular full-nominal-rank maps (OD shifts
    count-invisible below any realistic noise) do NOT assert identifiability."""
    full = np.zeros((1, 2, 2))
    full[0] = np.array([[1.0, 1.0 + 1e-9], [1.0, 1.0]])  # nearly identical columns
    rep = dynamic_identifiability_report(full, np.array([0, 1]), [(0, 2), (1, 2)], 1, 1)
    assert "sigma_min" in rep
    assert rep["sigma_min"] < 1e-6
    assert rep["linear_identifiable"] is False
    # a well-conditioned map still certifies identifiable with sigma_min reported
    good = np.zeros((1, 2, 2))
    good[0] = np.array([[1.0, 0.0], [0.0, 1.0]])
    rep2 = dynamic_identifiability_report(good, np.array([0, 1]), [(0, 2), (1, 2)], 1, 1)
    assert rep2["linear_identifiable"] is True
    assert rep2["sigma_min"] == pytest.approx(1.0)


# -------------------------------------------------------------- end to end


def _card(**over):
    cfg = {
        "sensors": {"kind": "explicit", "links": [3]},
        "heldout": {"kind": "explicit", "links": [2]},
        "n_slices": 3, "slice_length": 2.0, "n_days": 12, "noise": "poisson",
        "prior": {"kind": "stale", "cv": 0.3},
    }
    cfg.update(over)
    return cfg


def test_end_to_end_prior_baseline_and_ranking():
    """The prior-profile baseline runs through the exact certificate, and on the
    two-route card the simultaneous estimator recovers a lower descriptive od_rmse
    than the sequential one (the efficiency ordering), both feasible and ranked by
    the harness-recomputed heldout_count_rmse."""
    sc = two_route_scenario(sue_theta=None)
    res = run_dynamic_estimation_experiment(
        sc,
        [DynamicPriorBaseline(), SimultaneousDynamicGLSEstimator(),
         SequentialDynamicGLSEstimator()],
        Budget(sp_calls=2000), seed=1, macroreps=4, estimation=_card(),
    )
    finals = {}
    for row in res.rows:
        if row["macrorep"] == 0:
            finals[row["estimator"]] = row
    for name in ("prior-profile", "od-dynamic-sim", "od-dynamic-seq"):
        assert finals[name]["od_feasible"] == 1.0
        assert np.isfinite(float(finals[name]["heldout_count_rmse"]))
    assert res.manifest["identifiability"]["linear_identifiable"] is True
    assert finals["od-dynamic-sim"]["od_rmse"] < finals["od-dynamic-seq"]["od_rmse"]
    assert finals["od-dynamic-sim"]["od_rmse"] < finals["prior-profile"]["od_rmse"]


def test_seeded_byte_reproducibility():
    """Same (seed) -> identical certified rows; a different seed draws fresh counts,
    so the task hash and the rows differ (P8 regression)."""
    sc = two_route_scenario(sue_theta=None)
    a = run_dynamic_estimation_experiment(
        sc, [SimultaneousDynamicGLSEstimator()], Budget(sp_calls=2000),
        seed=3, macroreps=2, estimation=_card(),
    )
    b = run_dynamic_estimation_experiment(
        sc, [SimultaneousDynamicGLSEstimator()], Budget(sp_calls=2000),
        seed=3, macroreps=2, estimation=_card(),
    )
    keys = ("obs_count_rmse", "heldout_count_rmse", "od_rmse", "profile_rmse")
    for ra, rb in zip(a.rows, b.rows, strict=True):
        for k in keys:
            assert ra[k] == pytest.approx(rb[k], abs=0.0, rel=0.0) or (
                np.isnan(ra[k]) and np.isnan(rb[k])
            )
    c = run_dynamic_estimation_experiment(
        sc, [SimultaneousDynamicGLSEstimator()], Budget(sp_calls=2000),
        seed=7, macroreps=2, estimation=_card(),
    )
    assert a.rows[0]["obs_count_rmse"] != c.rows[0]["obs_count_rmse"]


def test_rejects_sue_instance():
    """The dynamic runner rejects an SUE instance (the exogenous free-flow map is
    defined on the deterministic instance, ADR-023)."""
    sc = two_route_scenario()  # default sue_theta=0.5
    with pytest.raises(ValueError, match="SUE instance"):
        run_dynamic_estimation_experiment(
            sc, [DynamicPriorBaseline()], Budget(sp_calls=100), estimation=_card()
        )


def test_empty_prior_profile_emits_prior():
    """A dynamic estimator with an all-zero prior profile (no active pairs) emits
    the prior instead of crashing the least-squares solve (mirrors gls)."""
    sc = two_route_scenario(sue_theta=None)
    pairs = active_pairs(sc.demand.matrix)
    full = lagged_assignment_tensor(sc.network, pairs, 2.0)
    n_lags = full.shape[0] - 1
    n_slices, n_intervals = 3, 3 + n_lags
    zero_profile = np.zeros((n_slices, sc.network.n_zones, sc.network.n_zones))
    obs_sensors = np.array([3])
    expected = np.zeros((n_intervals, sc.network.n_links))
    obs_ds = DynamicLinkCounts(obs_sensors, 5, "none").observe(
        sc, expected, RngBundle(0).generator(SOURCE_OBSERVATION)
    )
    dataset = dataclasses.replace(
        obs_ds,
        payload={**obs_ds.payload, "lag_tensor": full[:, obs_sensors, :],
                 "pairs": np.asarray(pairs, dtype=np.int64)},
        meta={**obs_ds.meta, "n_slices": n_slices, "slice_length": 2.0,
              "n_lags": n_lags, "map_recipe": "frozen_freeflow_v1"},
    )
    task = DynamicEstimationTask(
        "t", sc.network, zero_profile, dataset, {}, sc.content_hash(), seed=0
    )
    trace = ODTrace()
    SimultaneousDynamicGLSEstimator().estimate(task, Budget(sp_calls=100), RngBundle(0), trace)
    assert np.array_equal(trace.final.od_matrix, zero_profile)


def test_gls_static_estimator_unrelated_to_dynamic_registry():
    """A sanity guard: the static gls estimator is untouched and still registered."""
    assert "gls" in ESTIMATOR_REGISTRY
    assert GLSEstimator().capabilities.inputs_required == frozenset({"link_counts", "prior_od"})
