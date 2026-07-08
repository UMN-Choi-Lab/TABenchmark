"""Tests for od-kalman: Davis & Nihan (1993) linear-Gaussian OD estimation from a
time series of link counts (ADR-012).

Davis & Nihan prove the large-population link-count process is a stationary
linear-Gaussian VAR/VARMA around the equilibrium loading (Prop 2 mean + Prop 3
covariance). ``DayToDayCounts`` realizes that limit as a VAR(1) centered on the
UE flow; ``od-kalman`` recovers the OD by GLS whitened by the DN count-mean
covariance (the cross-link spatial structure + an AR(1) effective-sample-size
correction ``tau``). All anchor numbers are recomputed in-test as closed forms
(house style: no trusted digits).
"""

import numpy as np
import pytest

from tabench import (
    Budget,
    Demand,
    Network,
    RngBundle,
    Scenario,
    Trace,
    braess_scenario,
    two_route_scenario,
)
from tabench.core.rng import SOURCE_OBSERVATION
from tabench.estimation import (
    ESTIMATOR_REGISTRY,
    DavisNihanKalmanEstimator,
    GLSEstimator,
    ODTrace,
    ar1_tau,
    dn_gls_solve,
    gls_solve,
)
from tabench.estimation.base import EstimationTask
from tabench.experiments.runner import run_estimation_experiment
from tabench.models.frank_wolfe import BiconjugateFrankWolfeModel
from tabench.observe._dn_process import dn_spatial_covariance
from tabench.observe.levels import DayToDayCounts

BRAESS_GOLDEN_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"


def _ue_flows(scenario: Scenario) -> np.ndarray:
    trace = Trace()
    BiconjugateFrankWolfeModel().solve(
        scenario, Budget(iterations=5000, target_relative_gap=1e-10), RngBundle(0), trace
    )
    return trace.final.link_flows


def _single_pair_prior(demand: float) -> np.ndarray:
    m = np.zeros((2, 2))
    m[0, 1] = demand
    return m


def _hub2_scenario() -> Scenario:
    """A 2-OD-pair network (zones 1,2,3 via hub node 4) for the multi-pair paths."""
    net = Network(
        name="hub2", n_nodes=4, n_zones=3, first_thru_node=4,
        init_node=np.array([1, 4, 4], dtype=np.int64),
        term_node=np.array([4, 2, 3], dtype=np.int64),
        capacity=np.array([4.0, 3.0, 3.0]), length=np.zeros(3),
        free_flow_time=np.array([1.0, 1.0, 1.0]), b=np.array([0.1, 0.1, 0.1]),
        power=np.ones(3), toll=np.zeros(3), link_type=np.ones(3, dtype=np.int64),
    )
    od = np.zeros((3, 3))
    od[0, 1], od[0, 2] = 3.0, 2.0
    return Scenario("hub2", net, Demand(od), family="hub2")


def _dn_task(scenario, truth, sensors, prior_d, n_periods, scale, rho, seed=0):
    ds = DayToDayCounts(np.asarray(sensors), n_periods, scale, rho, k_inner=80).observe(
        scenario, truth, RngBundle(seed).generator(SOURCE_OBSERVATION)
    )
    return EstimationTask(
        name="t",
        network=scenario.network,
        prior=Demand(_single_pair_prior(prior_d)),
        dataset=ds,
        identifiability={},
        scenario_hash=scenario.content_hash(),
        seed=seed,
    )


# ---------------------------------------------------------------- hash preserved


def test_golden_braess_hash_preserved():
    """The DN observation level and od-kalman are additive: no scenario field
    changed, so the golden Braess content hash is byte-identical."""
    assert braess_scenario().content_hash() == BRAESS_GOLDEN_HASH


# ---------------------------------------------------- DN covariance closed form


def test_dn_spatial_covariance_two_route_closed_form():
    """On the two-route anchor the DN link-count covariance is the exact
    multinomial form: routes A={0,1}, B={2,3}, so Var(link 0) = (D^2/N) p_A p_B,
    links on the same route are +correlated and A-vs-B are -correlated."""
    sc = two_route_scenario(sue_theta=None)
    truth = _ue_flows(sc)
    demand = float(sc.demand.matrix[0, 1])
    p_a = truth[0] / demand  # recomputed route-A proportion
    p_b = 1.0 - p_a
    n_trav = np.array([round(50.0 * demand)])
    q = dn_spatial_covariance(
        sc.network, sc.demand, np.array([demand]), n_trav, 80, pairs=[(0, 1)]
    )
    var0 = (demand**2 / n_trav[0]) * p_a * p_b  # closed form, no trusted digits
    assert q[0, 0] == pytest.approx(var0, abs=1e-9)
    assert q[0, 1] == pytest.approx(var0, abs=1e-9)  # same route -> +correlated
    assert q[1, 1] == pytest.approx(var0, abs=1e-9)
    assert q[0, 2] == pytest.approx(-var0, abs=1e-9)  # A vs B -> -correlated
    assert np.allclose(q, q.T)  # symmetric
    # PSD (a covariance) and singular by flow conservation (rank 1 here).
    evals = np.linalg.eigvalsh(q)
    assert evals.min() > -1e-9
    assert np.linalg.matrix_rank(q, tol=1e-9) == 1


def test_dn_gls_scalar_closed_form():
    """Single pair, single sensor: g* = (g_pr/w^2 + p*c/s^2)/(1/w^2 + p^2/s^2)."""
    p, c, g_pr, s2, w2 = 0.625, 2.5, 3.0, 0.5, 4.0
    expected = (g_pr / w2 + p * c / s2) / (1.0 / w2 + p * p / s2)
    got = dn_gls_solve(
        np.array([[p]]), np.array([c]), np.array([g_pr]), np.array([[s2]]), np.array([w2])
    )
    assert got[0] == pytest.approx(expected, abs=1e-10)


def test_dn_gls_offdiagonal_whitening_differs_from_diagonal():
    """The cross-link (off-diagonal) DN whitening is load-bearing: two sensors
    whose counts are correlated are NOT the two independent measurements gls
    (diagonal V) assumes, so a non-diagonal Sigma changes the estimate. This is
    the spatial sense in which od-kalman is not a gls rename."""
    p_obs = np.array([[0.6], [0.4]])  # two sensors, one pair
    counts = np.array([3.0, 2.0])
    prior = np.array([4.0])
    w_var = np.array([9.0])
    var = np.array([1.0, 1.0])
    diagonal = dn_gls_solve(p_obs, counts, prior, np.diag(var), w_var)
    correlated = dn_gls_solve(
        p_obs, counts, prior, np.array([[1.0, 0.8], [0.8, 1.0]]), w_var
    )
    # Diagonal DN == gls with the same per-sensor variances (sanity anchor).
    via_gls = gls_solve(p_obs, counts, prior, w_var, var)
    assert diagonal[0] == pytest.approx(via_gls[0], abs=1e-8)
    # Off-diagonal correlation moves the estimate materially.
    assert abs(correlated[0] - diagonal[0]) > 0.05


def test_singular_covariance_no_overflow():
    """Adversarial-review Major #1: an exactly singular count-mean covariance
    (reachable via a `cov_ridge=0` override on a single-route sub-network where Q
    is rank-deficient) must not drive the DN whitening to ~1e155 and emit a
    finite-but-corrupted OD the `isfinite` certificate would not catch. The
    absolute floor in `_inv_sqrt_psd` keeps every whitening entry bounded, and the
    end-to-end estimate stays finite even with the ridge disabled."""
    from tabench.estimation.dn_kalman import _inv_sqrt_psd

    whitening = _inv_sqrt_psd(np.zeros((3, 3)))
    assert np.all(np.isfinite(whitening))
    assert np.abs(whitening).max() < 1e9  # bounded, NOT ~1e155
    g = dn_gls_solve(
        np.array([[0.6], [0.4]]), np.array([3.0, 2.0]), np.array([4.0]),
        np.zeros((2, 2)), np.array([9.0]),
    )
    assert np.all(np.isfinite(g)) and g[0] >= 0.0
    sc = two_route_scenario(sue_theta=None)
    truth = _ue_flows(sc)
    task = _dn_task(sc, truth, np.arange(4), 3.0, 100, 50.0, 0.5)
    trace = ODTrace()
    DavisNihanKalmanEstimator(k_inner=100, outer_iters=30, cov_ridge=0.0).estimate(
        task, Budget(sp_calls=10**9, iterations=100), RngBundle(0), trace
    )
    assert np.all(np.isfinite(trace.final.od_matrix))


def test_dn_spatial_covariance_rejects_pair_subset():
    """Adversarial-review #2: a strict subset of the active OD pairs would assign
    an equilibrium that understates congestion and silently corrupt Q, so it is
    rejected (the shipped caller always passes the full active set)."""
    sc = _hub2_scenario()  # two active pairs (0,1), (0,2)
    with pytest.raises(ValueError, match="subset"):
        dn_spatial_covariance(
            sc.network, sc.demand, np.array([3.0]), np.array([180]), 80, pairs=[(0, 1)]
        )


# ---------------------------------------------------------- temporal correction


def test_ar1_tau_recovers_persistence_and_is_one_for_iid():
    """tau tracks the AR(1) variance-inflation (1+rho)/(1-rho) on a DN series and
    collapses to 1 for an IID (rho=0) series -- the effective-sample-size knob
    that distinguishes od-kalman from every mean-collapsing estimator."""
    sc = two_route_scenario(sue_theta=None)
    truth = _ue_flows(sc)
    for rho in (0.0, 0.6):
        ds = DayToDayCounts(np.array([0]), 4000, 50.0, rho, k_inner=80).observe(
            sc, truth, RngBundle(0).generator(SOURCE_OBSERVATION)
        )
        tau, rho_hat = ar1_tau(ds.payload["counts"])
        target = (1.0 + rho) / (1.0 - rho)
        assert tau == pytest.approx(target, rel=0.15)
        if rho == 0.0:
            assert tau == pytest.approx(1.0, abs=0.15)


def test_ar1_tau_robust_to_anticorrelated_sensors():
    """tau must not collapse when the monitored links are on competing routes
    whose fluctuations are anti-correlated: a mean-pooled lag-1 estimate would
    cancel to tau~1 and silently drop the temporal correction. Sensors on link 0
    (route A) and link 2 (route B) are perfectly anti-correlated in the DN
    covariance; the trace-ratio ``ar1_tau`` still recovers rho."""
    sc = two_route_scenario(sue_theta=None)
    truth = _ue_flows(sc)
    rho = 0.7
    ds = DayToDayCounts(np.array([0, 2]), 4000, 50.0, rho, 80).observe(
        sc, truth, RngBundle(0).generator(SOURCE_OBSERVATION)
    )
    counts = ds.payload["counts"]
    assert np.corrcoef(counts[:, 0], counts[:, 1])[0, 1] < -0.9  # anti-correlated
    _tau, rho_hat = ar1_tau(counts)
    assert rho_hat == pytest.approx(rho, rel=0.15)  # not collapsed toward 0


# ---------------------------------------------------- DN process properties


def test_day_to_day_centered_on_ue_and_shrinks_with_population():
    """DayToDayCounts is centered on the UE loading (harness-consistent mean) and
    its fluctuation covariance vanishes as 1/population_scale (Prop 2 SLLN)."""
    sc = two_route_scenario(sue_theta=None)
    truth = _ue_flows(sc)
    small = DayToDayCounts(np.array([0]), 3000, 10.0, 0.0, 80).observe(
        sc, truth, RngBundle(0).generator(SOURCE_OBSERVATION)
    )
    large = DayToDayCounts(np.array([0]), 3000, 1000.0, 0.0, 80).observe(
        sc, truth, RngBundle(0).generator(SOURCE_OBSERVATION)
    )
    assert small.payload["counts"].mean() == pytest.approx(truth[0], abs=0.05)
    assert large.payload["counts"].mean() == pytest.approx(truth[0], abs=0.02)
    # ~100x population -> ~100x smaller variance (SLLN), so std ratio ~ 10x.
    ratio = small.payload["counts"].std() / large.payload["counts"].std()
    assert ratio == pytest.approx(10.0, rel=0.4)


# ------------------------------------------------------------------- recovery


def test_two_route_recovery_from_dn_series():
    """Under full sensors on a DN count series, od-kalman recovers the planted
    demand D=4 to the finite-population sampling floor."""
    sc = two_route_scenario(sue_theta=None)
    truth = _ue_flows(sc)
    task = _dn_task(sc, truth, np.arange(4), prior_d=3.0, n_periods=400, scale=50.0, rho=0.5)
    trace = ODTrace()
    DavisNihanKalmanEstimator(k_inner=120, outer_iters=60).estimate(
        task, Budget(sp_calls=10**9, iterations=200), RngBundle(0), trace
    )
    assert abs(trace.final.od_matrix[0, 1] - 4.0) < 0.1


def test_distinct_from_gls_on_correlated_series():
    """od-kalman and gls allocate the prior<->count trade-off differently on a DN
    series -- od-kalman uses the true (tight) multinomial covariance and the tau
    autocorrelation inflation, gls assumes IID Poisson counts, which overstates
    the noise of a near-deterministic multinomial split. Under a tight prior
    (cv=0.05, so the count variance actually competes with the prior) the two
    estimates differ materially, and od-kalman -- trusting the accurate counts
    more -- recovers closer to the planted D=4 while gls under-corrects toward the
    off prior. Not a gls rename at the estimator level."""
    sc = two_route_scenario(sue_theta=None)
    truth = _ue_flows(sc)
    task = _dn_task(sc, truth, np.array([0]), prior_d=3.0, n_periods=40, scale=50.0, rho=0.7)
    budget = Budget(sp_calls=10**9, iterations=200)
    k, g = ODTrace(), ODTrace()
    DavisNihanKalmanEstimator(k_inner=120, outer_iters=60, cv_prior=0.05).estimate(
        task, budget, RngBundle(0), k
    )
    GLSEstimator(k_inner=120, outer_iters=60, cv_prior=0.05).estimate(
        task, budget, RngBundle(0), g
    )
    kalman, gls = k.final.od_matrix[0, 1], g.final.od_matrix[0, 1]
    assert abs(kalman - gls) > 0.2  # the covariance model is load-bearing
    assert abs(kalman - 4.0) < abs(gls - 4.0)  # DN variance recovers better here


# ------------------------------------------------------------ registry + hash


def test_registered_as_od_kalman():
    assert "od-kalman" in ESTIMATOR_REGISTRY
    assert ESTIMATOR_REGISTRY["od-kalman"] is DavisNihanKalmanEstimator
    caps = DavisNihanKalmanEstimator().capabilities
    assert caps.paradigm == "estimation"
    assert caps.deterministic is True
    assert caps.inputs_required == frozenset({"link_counts", "prior_od"})
    assert caps.outputs == frozenset({"od_estimate"})


def test_content_hash_distinguishes_rho():
    """The DN dials (rho, population_scale) enter dataset.meta, so two instances
    differing only in persistence are different benchmark tasks and must not
    collide in the EstimationTask content hash."""
    sc = two_route_scenario(sue_theta=None)
    truth = _ue_flows(sc)
    t_lo = _dn_task(sc, truth, np.array([0]), 3.0, 50, 50.0, 0.3)
    t_hi = _dn_task(sc, truth, np.array([0]), 3.0, 50, 50.0, 0.9)
    assert t_lo.content_hash() == t_lo.content_hash()  # deterministic
    assert t_lo.content_hash() != t_hi.content_hash()  # rho is pinned


def test_empty_prior_emits_prior_unchanged():
    """An all-zero prior (no active pairs) emits the prior instead of crashing the
    least-squares solve (mirrors gls/od-congested)."""
    sc = two_route_scenario(sue_theta=None)
    truth = _ue_flows(sc)
    ds = DayToDayCounts(np.array([0]), 20, 50.0, 0.5, 80).observe(
        sc, truth, RngBundle(0).generator(SOURCE_OBSERVATION)
    )
    task = EstimationTask(
        "t", sc.network, Demand(np.zeros((2, 2))), ds, {}, sc.content_hash(), seed=0
    )
    trace = ODTrace()
    DavisNihanKalmanEstimator().estimate(task, Budget(sp_calls=500), RngBundle(0), trace)
    assert np.array_equal(trace.final.od_matrix, np.zeros((2, 2)))


# ---------------------------------------------------------------- end to end


def test_end_to_end_day_to_day():
    """od-kalman runs through the pinned P1 certificate on a day_to_day task: the
    SCORE (od_feasible, od_rmse) is harness-recomputed from the emitted OD matrix,
    never a self-report. With a cv=0 prior (=truth) it certifies feasible and
    recovers under full sensors; the self-report is finite provenance."""
    sc = braess_scenario(6.0)
    cfg = {
        "sensors": {"kind": "explicit", "links": [0, 1, 2, 3, 4]},
        "heldout": {"kind": "explicit", "links": []},
        "n_periods": 60,
        "noise": "day_to_day",
        "population_scale": 80.0,
        "rho": 0.5,
        "prior": {"kind": "stale", "cv": 0.0},
        "identifiability_k_inner": 40,
    }
    result = run_estimation_experiment(
        sc, [DavisNihanKalmanEstimator(k_inner=60, outer_iters=15)],
        Budget(sp_calls=5000), seed=0, macroreps=1, estimation=cfg,
    )
    row = [r for r in result.rows if r["estimator"] == "od-kalman"][-1]
    assert row["od_feasible"] == 1.0
    assert row["od_rmse"] < 0.5  # recovers near the planted truth at the DN floor
    assert np.isfinite(float(row["self_obs_count_rmse"]))
    assert result.manifest["estimation"]["noise"] == "day_to_day"
    assert result.manifest["estimation"]["rho"] == 0.5
