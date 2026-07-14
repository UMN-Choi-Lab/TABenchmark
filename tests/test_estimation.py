"""Tests for the T2 estimation track (ADR-002).

The Decision-6 analytic anchors are recomputed as closed forms in-test (house
style: no trusted digits). One anchor is deliberately reframed: the ADR
narrative claims ``gls``/``spiess`` recover Braess ``D=6`` from prior ``D=4``,
but the count-misfit objective has a spurious local minimum at ``D=10/3`` that
captures the prior-``D=4`` basin (verify_t2.py only checked the closed forms,
never ran the estimators for this claim). We therefore test recovery from a
prior in the global basin, test the convex two-route recovery, AND assert the
local-minimum trap as an executable fact.
"""

import numpy as np
import pytest
from conftest import load_or_skip

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
    CallableEstimator,
    EstimationTask,
    GLSEstimator,
    ODTrace,
    PriorBaseline,
    SpiessEstimator,
    SPSAEstimator,
    VZWEntropyEstimator,
    Yang1992Estimator,
    gls_solve,
    vzw_balance,
    yang_solve,
)
from tabench.experiments.runner import identifiability_report, run_estimation_experiment
from tabench.metrics.estimation import ODCertifier
from tabench.models.frank_wolfe import BiconjugateFrankWolfeModel
from tabench.observe.levels import LinkCounts, StalePriorOD

# Braess link order 1->3, 1->4, 3->4, 3->2, 4->2; two-route 1->3, 3->2, 1->4, 4->2.
BRAESS_TRUTH = np.array([4.0, 2.0, 2.0, 2.0, 4.0])
TWOROUTE_TRUTH = np.array([2.5, 2.5, 1.5, 1.5])


def _ue_flows(scenario: Scenario) -> np.ndarray:
    trace = Trace()
    BiconjugateFrankWolfeModel().solve(
        scenario, Budget(iterations=5000, target_relative_gap=1e-10), RngBundle(0), trace
    )
    return trace.final.link_flows


def _task(scenario, truth, sensors, prior_matrix, n_periods=1, noise="none"):
    ds = LinkCounts(np.asarray(sensors), n_periods, noise).observe(
        scenario, truth, RngBundle(0).generator(SOURCE_OBSERVATION)
    )
    return EstimationTask(
        name="t",
        network=scenario.network,
        prior=Demand(np.asarray(prior_matrix, dtype=np.float64)),
        dataset=ds,
        identifiability={},
        scenario_hash=scenario.content_hash(),
        seed=0,
    )


def _single_pair_prior(demand: float) -> np.ndarray:
    m = np.zeros((2, 2))
    m[0, 1] = demand
    return m


def _hub_two_pair_scenario() -> tuple[Scenario, np.ndarray]:
    """A 2-OD-pair network: zones 1,2,3 route to/from the hub node 4."""
    net = Network(
        name="hub2",
        n_nodes=4,
        n_zones=3,
        first_thru_node=4,
        init_node=np.array([1, 4, 4], dtype=np.int64),
        term_node=np.array([4, 2, 3], dtype=np.int64),
        capacity=np.array([4.0, 3.0, 3.0]),
        length=np.zeros(3),
        free_flow_time=np.array([1.0, 1.0, 1.0]),
        b=np.array([0.1, 0.1, 0.1]),
        power=np.ones(3),
        toll=np.zeros(3),
        link_type=np.ones(3, dtype=np.int64),
    )
    od = np.zeros((3, 3))
    od[0, 1] = 3.0
    od[0, 2] = 2.0
    scenario = Scenario("hub2", net, Demand(od), family="hub2")
    return scenario, _ue_flows(scenario)


# ------------------------------------------------------------------ VZW anchors


def test_vzw_single_sensor_one_pass_exact():
    """One damped pass gives T = (c/p)^p; the fixed point c/p = 4.0 is reached
    geometrically over many passes (Van Zuylen-Willumsen exponent form)."""
    p_a = TWOROUTE_TRUTH[0] / 4.0  # route-A proportion = 2.5 / 4 = 0.625, recomputed
    count = TWOROUTE_TRUTH[0]  # noiseless count on link 0 = 2.5
    best_g, trajectory, consistent = vzw_balance(
        np.array([1.0]), np.array([[p_a]]), np.array([count]), n_passes=1
    )
    # Exponent form: prior 1 -> (count / (p*1))^p = (c/p)^p, NOT c/p in one touch.
    assert best_g[0] == pytest.approx((count / p_a) ** p_a, abs=1e-12)
    assert consistent is False  # one pass has not yet reached the fixed point
    assert len(trajectory) == 2  # prior + one update
    # Geometric convergence to the exact single-pair fixed point c/p = 4.0.
    conv_g, _, conv_consistent = vzw_balance(
        np.array([1.0]), np.array([[p_a]]), np.array([count]), n_passes=60
    )
    assert conv_g[0] == pytest.approx(count / p_a, abs=1e-9)
    assert conv_g[0] == pytest.approx(4.0, abs=1e-9)
    assert conv_consistent is True


def test_vzw_inconsistent_counts_converge_damped():
    """Two mutually inconsistent sensors on one pair -> damped convergence to a
    compromise above tolerance (the exponent form does not oscillate)."""
    p_a = TWOROUTE_TRUTH[0] / 4.0  # 0.625
    p_b = TWOROUTE_TRUTH[2] / 4.0  # 0.375
    c_a, c_b = 3.0, 1.0
    best_g, _, consistent = vzw_balance(
        np.array([1.0]), np.array([[p_a], [p_b]]), np.array([c_a, c_b]), n_passes=400
    )
    # Sequential log-space fixed point of the two damped single-link updates
    # x <- (1-p)*x + p*log(c/p), applied for sensor a then b (recomputed):
    t_a, t_b = np.log(c_a / p_a), np.log(c_b / p_b)
    x_star = ((1 - p_b) * p_a * t_a + p_b * t_b) / (1 - (1 - p_a) * (1 - p_b))
    assert best_g[0] == pytest.approx(np.exp(x_star), abs=1e-6)
    assert consistent is False  # mutually inconsistent -> residual above tol


# ------------------------------------------------------------------ GLS anchor


def test_gls_scalar_closed_form():
    """Single pair, single sensor, W=V=1: g* = (g_pr + p*c)/(1 + p^2)."""
    p, c, g_pr = 0.625, 2.5, 3.0
    expected = (g_pr + p * c) / (1.0 + p * p)  # recompute, no trusted digits
    got = gls_solve(
        np.array([[p]]), np.array([c]), np.array([g_pr]), np.array([1.0]), np.array([1.0])
    )
    assert got[0] == pytest.approx(expected, abs=1e-10)


# --------------------------------------------------------------- recovery tests


def test_two_route_convex_recovery():
    """On the convex two-route network, all deterministic estimators recover D=4."""
    sc = two_route_scenario(sue_theta=None)
    truth = _ue_flows(sc)
    prior = _single_pair_prior(3.0)
    budget = Budget(sp_calls=10**9, iterations=200)
    task = _task(sc, truth, np.arange(4), prior)
    for est in (
        SpiessEstimator(k_inner=120, outer_iters=80),
        VZWEntropyEstimator(k_inner=120, outer_iters=80),
        GLSEstimator(k_inner=120, outer_iters=80, cv_prior=50.0),
    ):
        trace = ODTrace()
        est.estimate(task, budget, RngBundle(0), trace)
        assert abs(trace.final.od_matrix[0, 1] - 4.0) < 1e-3


def test_braess_full_sensor_recovery_from_global_basin():
    """gls + spiess + vzw-entropy recover Braess D=6 under full noiseless sensors.

    The prior is D=5.5 (inside the global basin); prior D=4 sits in the spurious
    local minimum's basin (see test_braess_prior4_local_minimum).
    """
    sc = braess_scenario(6.0)
    prior = _single_pair_prior(5.5)
    budget = Budget(sp_calls=10**9, iterations=200)
    task = _task(sc, BRAESS_TRUTH, np.arange(5), prior)
    for est in (
        SpiessEstimator(k_inner=120, outer_iters=80),
        GLSEstimator(k_inner=120, outer_iters=80, cv_prior=50.0),
        VZWEntropyEstimator(k_inner=120, outer_iters=80),
    ):
        trace = ODTrace()
        est.estimate(task, budget, RngBundle(0), trace)
        assert abs(trace.final.od_matrix[0, 1] - 6.0) < 1e-3


def test_braess_prior4_safeguard_refuses_dominated_trap():
    """Executable ADR caveat + safeguard: from prior D=4 the count-misfit basin
    does not lead to D=6, and spiess's retrospective Armijo + best-self-obs-RMSE
    safeguard (ADR-002 Decision 3) refuse to descend into the dominated D=10/3
    trap. That trap is a frozen-proportion local minimum (bypass-saturated
    regime, outer links 1->4 and 3->2 carry no flow) whose self obs-RMSE exceeds
    the prior's, so the estimate stays near its own D=4 start instead.
    """
    sc = braess_scenario(6.0)
    task = _task(sc, BRAESS_TRUTH, np.arange(5), _single_pair_prior(4.0))
    trace = ODTrace()
    SpiessEstimator(k_inner=120, outer_iters=120).estimate(
        task, Budget(sp_calls=10**9, iterations=120), RngBundle(0), trace
    )
    g = trace.final.od_matrix[0, 1]
    assert abs(g - 6.0) > 1.0  # emphatically not the global optimum
    assert abs(g - 10.0 / 3.0) > 0.25  # and NOT the dominated 10/3 trap either
    # The safeguard held: the returned self obs-RMSE is below the trap's, where
    # the bypass carries all demand (flows (10/3, 0, 10/3, 0, 10/3)).
    trap_flows = np.array([10.0 / 3.0, 0.0, 10.0 / 3.0, 0.0, 10.0 / 3.0])
    trap_resid = float(np.sqrt(np.mean((trap_flows - BRAESS_TRUTH) ** 2)))
    assert trace.final.self_report["obs_count_rmse"] < trap_resid


# ----------------------------------------------- certificate + held-out anchor


def _braess_certifier(obs, heldout, oracle, identifiable=False):
    sc = braess_scenario(6.0)
    obs = np.asarray(obs)
    heldout = np.asarray(heldout)
    obs_counts = oracle[obs][None, :] if obs.size else np.zeros((1, 0))
    ho_counts = oracle[heldout][None, :] if heldout.size else np.zeros((1, 0))
    return ODCertifier(
        sc, obs, heldout, obs_counts, ho_counts, oracle,
        {"linear_identifiable": identifiable},
    )


def test_braess_single_sensor_counterexample():
    """Sensor {3->4} cannot tell D=2 from D=6; held-out {1->3} does (Decision 4).

    Both demands produce link-2 flow 2, so obs_count_rmse ~ 0 for each, but the
    held-out link 1->3 carries 2 vs 4 -> heldout metrics discriminate.
    """
    oracle = BRAESS_TRUTH  # UE(D=6)
    cert = _braess_certifier(obs=[2], heldout=[0], oracle=oracle)
    m2 = cert.certify(_single_pair_prior(2.0))
    m6 = cert.certify(_single_pair_prior(6.0))
    assert m2["obs_count_rmse"] < 1e-3 and m6["obs_count_rmse"] < 1e-3
    # D=6 UE(link 1->3) = 4 matches the held-out truth; D=2 gives 2.
    assert m6["heldout_flow_rmse"] == pytest.approx(0.0, abs=1e-3)
    assert m2["heldout_flow_rmse"] == pytest.approx(2.0, abs=1e-3)
    assert m6["heldout_count_rmse"] < m2["heldout_count_rmse"]


# --------------------------------------------------------------- SPSA anchors


def test_spsa_seeded_smoke():
    """Two-route full sensors, 200 sp-call budget: |g-4| < 0.2, byte-reproducible."""
    sc = two_route_scenario(sue_theta=None)
    truth = _ue_flows(sc)
    task = _task(sc, truth, np.arange(4), _single_pair_prior(3.0))
    budget = Budget(sp_calls=200, iterations=10**6)
    t1 = ODTrace()
    SPSAEstimator().estimate(task, budget, RngBundle(0, macrorep=0), t1)
    assert abs(t1.final.od_matrix[0, 1] - 4.0) < 0.2
    t2 = ODTrace()
    SPSAEstimator().estimate(task, budget, RngBundle(0, macrorep=0), t2)
    assert np.array_equal(t1.final.od_matrix, t2.final.od_matrix)  # byte-identical


def test_spsa_macrorep_differs():
    """Distinct macroreps draw distinct perturbations (P8 regression, 2+ pairs).

    SPSA is invariant to the Rademacher sign in 1-D, so this needs >=2 OD pairs.
    """
    sc, truth = _hub_two_pair_scenario()
    prior = np.zeros((3, 3))
    prior[0, 1] = 2.0
    prior[0, 2] = 3.0
    ds = LinkCounts(np.array([1, 2]), 1, "none").observe(
        sc, truth, RngBundle(0).generator(SOURCE_OBSERVATION)
    )
    task = EstimationTask("t", sc.network, Demand(prior), ds, {}, sc.content_hash(), seed=0)
    budget = Budget(sp_calls=400, iterations=10**6)
    t0 = ODTrace()
    SPSAEstimator().estimate(task, budget, RngBundle(0, macrorep=0), t0)
    t0b = ODTrace()
    SPSAEstimator().estimate(task, budget, RngBundle(0, macrorep=0), t0b)
    t1 = ODTrace()
    SPSAEstimator().estimate(task, budget, RngBundle(0, macrorep=1), t1)
    assert np.array_equal(t0.final.od_matrix, t0b.final.od_matrix)
    assert not np.array_equal(t0.final.od_matrix, t1.final.od_matrix)


# ---------------------------------------------------------------- censoring


def test_censoring_and_zero_not_censored():
    oracle = BRAESS_TRUTH
    cert = _braess_certifier(obs=[2], heldout=[0], oracle=oracle)
    # Wrong shape is a wrapper programming error -> raises.
    with pytest.raises(ValueError):
        cert.certify(np.zeros((3, 3)))
    # Non-finite and (materially) negative entries are censored.
    nf = _single_pair_prior(6.0)
    nf[0, 1] = np.inf
    assert cert.certify(nf)["od_feasible"] == 0.0
    neg = _single_pair_prior(6.0)
    neg[0, 1] = -5.0
    assert cert.certify(neg)["od_feasible"] == 0.0
    # A zero matrix is a legitimate (terrible) estimate, NOT censored.
    zero = cert.certify(np.zeros((2, 2)))
    assert zero["od_feasible"] == 1.0
    assert zero["obs_count_rmse"] > 1.0  # catastrophic but honest count-fit


def test_tiny_negative_clipped_not_censored():
    oracle = BRAESS_TRUTH
    cert = _braess_certifier(obs=[2], heldout=[0], oracle=oracle)
    q = _single_pair_prior(6.0)
    q[0, 1] = 6.0
    q[1, 0] = -1e-13  # below 1e-9 * max|Q| -> clipped to 0, not censored
    assert cert.certify(q)["od_feasible"] == 1.0


def test_negativity_gate_immune_to_diagonal_scale():
    """adr-023 review parity fix: a huge intrazonal diagonal cell must not
    inflate the negativity tolerance under which a genuinely negative
    inter-zonal cell escapes censoring (scale is off-diagonal only)."""
    oracle = BRAESS_TRUTH
    cert = _braess_certifier(obs=[2], heldout=[0], oracle=oracle)
    q = _single_pair_prior(6.0)
    q[1, 0] = -5.0
    q[0, 0] = 1e12  # diagonal mass must not buy negativity tolerance
    assert cert.certify(q)["od_feasible"] == 0.0


# ---------------------------------------------------------------- identifiability


def test_sioux_falls_underdetermined():
    """76 links vs 528 positive OD pairs -> linear_identifiable is False (Decision 6)."""
    scenario = load_or_skip("siouxfalls")
    n_links = scenario.network.n_links
    report = identifiability_report(
        scenario.network, scenario.demand, np.arange(n_links), k_inner=20
    )
    assert report["n_active_pairs"] > n_links  # 528 > 76
    assert report["linear_identifiable"] is False
    assert report["rank"] <= n_links


# ---------------------------------------------------------------- honesty (P1)


def test_self_report_obs_rmse_matches_certificate():
    """The estimator's self-reported obs_count_rmse tracks the pinned certificate."""
    sc = braess_scenario(6.0)
    prior = _single_pair_prior(6.0)
    task = _task(sc, BRAESS_TRUTH, np.array([1, 2, 3]), prior)
    trace = ODTrace()
    GLSEstimator(k_inner=120, outer_iters=30, cv_prior=0.3).estimate(
        task, Budget(sp_calls=10**9, iterations=30), RngBundle(0), trace
    )
    oracle = BRAESS_TRUTH
    cert = ODCertifier(
        sc, np.array([1, 2, 3]), np.array([0, 4]),
        oracle[[1, 2, 3]][None, :], oracle[[0, 4]][None, :], oracle,
        {"linear_identifiable": True},
    )
    final = trace.final
    certified = cert.certify(final.od_matrix)["obs_count_rmse"]
    self_reported = final.self_report["obs_count_rmse"]
    # Self-report (MSA proportions) vs pinned bfw differ only by the inner
    # assignment gap; the honesty diff must stay small.
    assert abs(self_reported - certified) < 0.1


def test_self_report_honesty_diff_targets_mean_under_noise():
    """P1 honesty diff must target obs_mean_count_rmse, not per-period counts.

    Under Poisson counts with n_periods>1 the estimator self-reports fit to the
    period-mean, while the per-period obs_count_rmse carries an irreducible noise
    floor (harness^2 = mean-fit^2 + within-period var). Diffing the self-report
    against the per-period column would flag every honest estimator; against
    obs_mean_count_rmse it is small (ADR-002 Decision 2, finding 11).
    """
    sc = braess_scenario(6.0)
    cfg = {
        "sensors": {"kind": "explicit", "links": [1, 2, 3]},
        "heldout": {"kind": "explicit", "links": [0, 4]},
        "n_periods": 10,
        "noise": "poisson",
        "prior": {"kind": "stale", "cv": 0.0},  # prior = truth: isolate the noise floor
    }
    result = run_estimation_experiment(
        sc, [GLSEstimator(cv_prior=0.3)], Budget(sp_calls=2000), seed=1,
        macroreps=1, estimation=cfg,
    )
    row = [r for r in result.rows if r["estimator"] == "gls"][-1]
    self_rmse = float(row["self_obs_count_rmse"])
    obs_mean = row["obs_mean_count_rmse"]
    obs_per_period = row["obs_count_rmse"]
    # Self-report tracks the fit-to-mean column (MSA-vs-bfw slack only) ...
    assert abs(self_rmse - obs_mean) < 0.1
    # ... and is far from the per-period column, which the Poisson floor inflates.
    assert obs_per_period - obs_mean > 0.3
    assert abs(self_rmse - obs_mean) < abs(self_rmse - obs_per_period)


# ---------------------------------------------------------------- stale prior


def test_stale_prior_zero_cells_and_mean():
    sc = braess_scenario(6.0)
    truth = sc.demand.matrix
    ds = StalePriorOD(cv=0.3).observe(sc, BRAESS_TRUTH, RngBundle(0).generator(0))
    prior = ds.payload["prior_od"]
    assert prior[truth == 0.0].sum() == 0.0  # zero cells stay zero
    assert prior[0, 1] > 0.0
    # cv=0 returns the truth exactly.
    exact = StalePriorOD(cv=0.0).observe(sc, BRAESS_TRUTH, RngBundle(0).generator(0))
    assert np.array_equal(exact.payload["prior_od"], truth)


# ---------------------------------------------------------------- registry + hash


def test_registry_and_content_hash():
    for name in ("prior", "gls", "vzw-entropy", "spiess", "spsa"):
        assert name in ESTIMATOR_REGISTRY
    assert ESTIMATOR_REGISTRY["prior"] is PriorBaseline
    sc = braess_scenario(6.0)
    task_a = _task(sc, BRAESS_TRUTH, np.array([2]), _single_pair_prior(4.0))
    task_b = _task(sc, BRAESS_TRUTH, np.array([2]), _single_pair_prior(5.0))
    assert task_a.content_hash() == task_a.content_hash()  # deterministic
    assert task_a.content_hash() != task_b.content_hash()  # prior bytes hashed


def test_prior_baseline_emits_prior_unchanged():
    sc = braess_scenario(6.0)
    prior = _single_pair_prior(4.2)
    task = _task(sc, BRAESS_TRUTH, np.array([2]), prior)
    trace = ODTrace()
    bundle = PriorBaseline().estimate(task, Budget(sp_calls=10), RngBundle(0), trace)
    assert np.array_equal(trace.final.od_matrix, prior)
    assert trace.final.coords.sp_calls == 0
    assert bundle.estimator_name == "prior"


# ---------------------------------------------------------------- end to end


def test_run_estimation_experiment_end_to_end():
    sc = braess_scenario(6.0)
    estimators = [PriorBaseline(), GLSEstimator(k_inner=40, outer_iters=8)]
    cfg = {
        "sensors": {"kind": "explicit", "links": [0, 1, 2, 3, 4]},
        "heldout": {"kind": "explicit", "links": []},
        "n_periods": 1,
        "noise": "none",
        "prior": {"kind": "stale", "cv": 0.0},
        "identifiability_k_inner": 30,
    }
    result = run_estimation_experiment(
        sc, estimators, Budget(sp_calls=2000), seed=0, macroreps=1, estimation=cfg
    )
    assert result.rows
    required = {
        "task_hash", "od_feasible", "obs_count_rmse", "oracle_obs_count_rmse",
        "heldout_count_rmse", "od_rmse", "od_nrmse", "total_demand_error",
        "od_identifiable", "certificate_gap", "certificate_converged",
        "self_obs_count_rmse",
    }
    assert required <= set(result.rows[0])
    assert result.manifest["identifiability"]["linear_identifiable"] is True
    # cv=0 prior is the truth; gls's OD error is negligible under full sensors.
    last_gls = [r for r in result.rows if r["estimator"] == "gls"][-1]
    assert last_gls["od_feasible"] == 1.0
    assert last_gls["od_rmse"] < 1e-2


def test_braess_t2_card_safeguard_no_collapse():
    """Regression (ADR-002 Decision 3 safeguards): on the 0braess-t2 card config
    (seed 0, macrorep 0) gls and spiess must not collapse below the prior into
    the bypass-saturated regime. Before the best-self-obs-RMSE outer-loop
    safeguard they returned ghat~2 (heldout ~3.6/3.7, od_rmse ~2.8/3.0 -- far
    worse than the do-nothing prior); the safeguard returns the good early
    iterate (ghat~5, od_rmse ~0.7/0.9 at the oracle Poisson floor).

    On this macrorep the prior draw is noise-lucky -- its heldout (2.277) sits
    *below* the oracle Poisson floor (2.345), so 'heldout <= prior' is not
    attainable by any honest estimator. We require instead that gls/spiess land
    at the oracle floor and recover a better OD than doing nothing.
    """
    sc = braess_scenario(6.0)
    cfg = {
        "sensors": {"kind": "explicit", "links": [1, 2, 3]},
        "heldout": {"kind": "explicit", "links": [0, 4]},
        "n_periods": 10,
        "noise": "poisson",
        "prior": {"kind": "stale", "cv": 0.30},
    }
    ests = [PriorBaseline(), GLSEstimator(), SpiessEstimator()]
    result = run_estimation_experiment(
        sc, ests, Budget(sp_calls=2000), seed=0, macroreps=1, estimation=cfg
    )
    last = {r["estimator"]: r for r in result.rows if r["macrorep"] == 0}
    prior = last["prior"]
    for name in ("gls", "spiess"):
        row = last[name]
        # At the oracle Poisson floor, not the collapsed ghat~2 (heldout ~3.6).
        assert row["heldout_count_rmse"] <= row["oracle_heldout_count_rmse"] + 0.1
        # And a better OD than doing nothing (the collapse gave od_rmse ~2.8).
        assert row["od_rmse"] < prior["od_rmse"]


def test_overlapping_sensors_heldout_raises():
    """Explicit sensor/held-out sets that overlap are rejected at construction
    (P7 harness-enforced disjointness; ADR-002 heldout_count_rmse contract)."""
    sc = braess_scenario(6.0)
    cfg = {
        "sensors": {"kind": "explicit", "links": [0, 1, 2]},
        "heldout": {"kind": "explicit", "links": [2, 3]},  # overlap on link 2
        "n_periods": 1, "noise": "none", "prior": {"kind": "stale", "cv": 0.0},
    }
    with pytest.raises(ValueError, match="disjoint"):
        run_estimation_experiment(sc, [PriorBaseline()], Budget(sp_calls=50), estimation=cfg)


def test_unsupported_certificate_assignment_raises():
    """Only bfw is a supported certificate pin this sprint; another label is
    rejected rather than silently run as bfw (ADR-002 Decision 2)."""
    sc = braess_scenario(6.0)
    cfg = {
        "sensors": {"kind": "explicit", "links": [1, 2, 3]},
        "heldout": {"kind": "explicit", "links": [0, 4]},
        "n_periods": 1, "noise": "none", "prior": {"kind": "stale", "cv": 0.0},
        "certificate": {"assignment": "fw"},
    }
    with pytest.raises(ValueError, match="assignment"):
        run_estimation_experiment(sc, [PriorBaseline()], Budget(sp_calls=50), estimation=cfg)


def test_sue_scenario_rejected_by_t2_runner():
    """T2 certifies against the pinned UE assignment only; an SUE instance is
    rejected by the public runner, not just the CLI (ADR-002 Decision 2)."""
    sc = two_route_scenario()  # default sue_theta=0.5
    assert sc.sue_theta is not None
    cfg = {
        "sensors": {"kind": "explicit", "links": [0]},
        "heldout": {"kind": "explicit", "links": [2]},
        "n_periods": 1, "noise": "none", "prior": {"kind": "stale", "cv": 0.0},
    }
    with pytest.raises(ValueError, match="SUE"):
        run_estimation_experiment(sc, [PriorBaseline()], Budget(sp_calls=50), estimation=cfg)


def test_task_hash_distinguishes_sensors_and_heldout():
    """content_hash pins the estimation instance: equal-size sensor placements
    and held-out designs no longer collide (ADR-002 Decision 1)."""
    sc = braess_scenario(6.0)

    def task_hash(sensors, heldout):
        cfg = {
            "sensors": {"kind": "explicit", "links": sensors},
            "heldout": {"kind": "explicit", "links": heldout},
            "n_periods": 1, "noise": "none", "prior": {"kind": "stale", "cv": 0.0},
        }
        res = run_estimation_experiment(
            sc, [PriorBaseline()], Budget(sp_calls=50), estimation=cfg
        )
        return res.rows[0]["task_hash"]

    base = task_hash([1, 2, 3], [4])
    assert base != task_hash([1, 2], [4])  # sensor placement is hashed
    assert base != task_hash([1, 2, 3], [0])  # held-out design is hashed


def test_callable_estimator_wraps_constant():
    sc = braess_scenario(6.0)
    task = _task(sc, BRAESS_TRUTH, np.array([2]), _single_pair_prior(6.0))
    est = CallableEstimator(lambda t, rng: _single_pair_prior(2.0), name="const2")
    trace = ODTrace()
    est.estimate(task, Budget(sp_calls=10), RngBundle(0), trace)
    assert trace.final.od_matrix[0, 1] == 2.0
    assert est.capabilities.paradigm == "learned"
    assert est.capabilities.outputs == frozenset({"od_estimate"})


# ------------------------------------------------- od-congested (Yang et al. 1992)


def test_yang_scalar_closed_form_and_gls_relationship():
    """Single pair, single sensor: Yang's theta-weighted QP has the closed form
    g* = (theta*g_pr + (1-theta)*p*c) / (theta + (1-theta)*p^2). At theta=0.5 it
    coincides with gls's identity-covariance anchor (g_pr+p*c)/(1+p^2), and in
    general yang_solve IS gls_solve with the scalar variances 1/theta, 1/(1-theta)
    -- the exact sense in which od-congested is the deterministic-trade-off case
    of generalized least squares (all recomputed, no trusted digits)."""
    p, c, g_pr = 0.625, 2.5, 3.0
    for theta in (0.2, 0.5, 0.8):
        expected = (theta * g_pr + (1 - theta) * p * c) / (theta + (1 - theta) * p * p)
        got = yang_solve(np.array([[p]]), np.array([c]), np.array([g_pr]), theta)
        assert got[0] == pytest.approx(expected, abs=1e-10)
        via_gls = gls_solve(
            np.array([[p]]), np.array([c]), np.array([g_pr]),
            np.array([1.0 / theta]), np.array([1.0 / (1.0 - theta)]),
        )
        assert got[0] == pytest.approx(via_gls[0], abs=1e-9)
    half = yang_solve(np.array([[p]]), np.array([c]), np.array([g_pr]), 0.5)[0]
    assert half == pytest.approx((g_pr + p * c) / (1.0 + p * p), abs=1e-10)


def test_yang_theta_limits_are_prior_and_count_consistent():
    """The single trade-off theta spans the whole prior<->count spectrum (the
    distinctive knob vs gls's covariances / spiess's misfit-only): theta->1
    recovers the prior; theta->0 fits the count exactly (c/p)."""
    p, c, g_pr = 0.5, 3.0, 1.0
    near_prior = yang_solve(np.array([[p]]), np.array([c]), np.array([g_pr]), 1 - 1e-8)[0]
    near_count = yang_solve(np.array([[p]]), np.array([c]), np.array([g_pr]), 1e-8)[0]
    assert near_prior == pytest.approx(g_pr, abs=1e-4)
    assert near_count == pytest.approx(c / p, abs=1e-4)  # 6.0


def test_yang_differs_from_gls_covariance_weighting():
    """od-congested weights every prior cell / count residual uniformly by theta;
    gls weights per cell (prior CV -> W). Given a count the prior does NOT already
    satisfy, gls loads most of the adjustment onto the uncertain (large-prior)
    cell, while od-congested splits it uniformly -- so the two allocate the fit
    differently and od-congested is not a gls rename."""
    p_obs = np.array([[1.0, 1.0]])          # one sensor sees both pairs
    c = np.array([14.0])                     # prior sum (2+8=10) does NOT match it
    g_pr = np.array([2.0, 8.0])             # priors of very different magnitude
    yang = yang_solve(p_obs, c, g_pr, 0.5)  # uniform weighting: near-even split
    w_var = (0.5 * g_pr) ** 2 + 1e-6        # gls: large-prior cell far more uncertain
    gls = gls_solve(p_obs, c, g_pr, w_var, np.array([1.0]))
    assert np.abs(yang - gls).max() > 0.5
    # gls moves the uncertain cell 1 much more than od-congested's uniform split.
    assert (gls[1] - g_pr[1]) > (yang[1] - g_pr[1]) + 0.5


def test_yang_congested_recovery_and_prior_limit():
    """The bilevel outer fixed point recovers the equilibrium-consistent truth as
    theta->0 (count-trusting) on the convex two-route (D=4) and Braess (D=6, from
    a global-basin prior) networks; at theta->1 it instead holds at the prior."""
    budget = Budget(sp_calls=10**9, iterations=200)
    for scenario_fn, truth, prior_d, target in (
        (lambda: two_route_scenario(sue_theta=None), TWOROUTE_TRUTH, 3.0, 4.0),
        (lambda: braess_scenario(6.0), BRAESS_TRUTH, 5.5, 6.0),
    ):
        sc = scenario_fn()
        task = _task(sc, truth, np.arange(len(truth)), _single_pair_prior(prior_d))
        trace = ODTrace()
        Yang1992Estimator(k_inner=120, outer_iters=80, theta=1e-3).estimate(
            task, budget, RngBundle(0), trace
        )
        assert abs(trace.final.od_matrix[0, 1] - target) < 1e-2
        trace_prior = ODTrace()
        Yang1992Estimator(k_inner=120, outer_iters=30, theta=1 - 1e-6).estimate(
            task, budget, RngBundle(0), trace_prior
        )
        assert abs(trace_prior.final.od_matrix[0, 1] - prior_d) < 1e-2


def test_yang_registered_as_od_congested():
    assert "od-congested" in ESTIMATOR_REGISTRY
    assert ESTIMATOR_REGISTRY["od-congested"] is Yang1992Estimator
    caps = Yang1992Estimator().capabilities
    assert caps.paradigm == "estimation"
    assert caps.inputs_required == frozenset({"link_counts", "prior_od"})
    assert caps.outputs == frozenset({"od_estimate"})


def test_yang_certifies_honestly_end_to_end():
    """od-congested runs through the pinned P1 certificate with no model-specific
    trust: the SCORE (od_feasible, od_rmse) is recomputed by the harness's
    ODCertifier from the emitted OD matrix, never from a self-report. With a cv=0
    prior (=truth) it certifies od_feasible=1 and recovers under full sensors."""
    sc = braess_scenario(6.0)
    cfg = {
        "sensors": {"kind": "explicit", "links": [0, 1, 2, 3, 4]},
        "heldout": {"kind": "explicit", "links": []},
        "n_periods": 1,
        "noise": "none",
        "prior": {"kind": "stale", "cv": 0.0},
        "identifiability_k_inner": 30,
    }
    result = run_estimation_experiment(
        sc, [Yang1992Estimator(k_inner=40, outer_iters=10, theta=1e-3)],
        Budget(sp_calls=2000), seed=0, macroreps=1, estimation=cfg,
    )
    row = [r for r in result.rows if r["estimator"] == "od-congested"][-1]
    assert row["od_feasible"] == 1.0
    # Near-exact under a cv=0 prior; the ~0.3%-of-D residual is the MSA-vs-bfw
    # inner-assignment slack (tight recovery is pinned by the recovery test). The
    # self-report is provenance only -- the certified obs_count_rmse column is the
    # harness recomputation, and it is what the leaderboard scores.
    assert row["od_rmse"] < 5e-2
    assert np.isfinite(float(row["self_obs_count_rmse"]))


def test_yang_safeguard_returns_best_not_dominated_last_iterate():
    """Confirmed adversarial-review Major: the best-self-obs-RMSE safeguard
    (docstring + ADR-002 Decision 3, 'never returns a strictly dominated last
    iterate') must be PINNED. On Braess (prior D=3, theta=0.2, single sensor {0})
    the it=1 iterate (g=3.8, obs-RMSE 0.20) dominates every later outer iterate,
    so the emitted OD is that early iterate, not the dominated last one. Deleting
    the re-record block ships a worse OD and fails this test."""
    sc = braess_scenario(6.0)
    task = _task(sc, BRAESS_TRUTH, np.array([0]), _single_pair_prior(3.0))
    trace = ODTrace()
    Yang1992Estimator(k_inner=120, outer_iters=40, theta=0.2).estimate(
        task, Budget(sp_calls=10**9, iterations=200), RngBundle(0), trace
    )
    resids = [s.self_report["obs_count_rmse"] for s in trace]
    final_resid = trace.final.self_report["obs_count_rmse"]
    # The emitted final is the best iterate seen ...
    assert final_resid == pytest.approx(min(resids), abs=1e-12)
    # ... the safeguard was load-bearing (a later iterate was strictly worse) ...
    assert max(resids) > final_resid + 0.02
    # ... and the recovered value is the dominating early iterate, not a late one.
    assert trace.final.od_matrix[0, 1] == pytest.approx(3.8, abs=1e-3)


def test_yang_shipped_estimator_distinct_from_gls():
    """The SHIPPED Yang1992Estimator (uniform theta) and GLSEstimator (per-cell
    covariance W) allocate an under-identified fit differently -- od-congested is
    not a gls rename even at the estimator level, not only in the solve primitive.
    On the 2-pair hub with a single shared sensor and an off, magnitude-swapped
    prior, the two disagree materially on the pair split."""
    sc, truth = _hub_two_pair_scenario()
    prior = np.zeros((3, 3))
    prior[0, 1], prior[0, 2] = 2.0, 4.0  # off truth (3, 2) + heterogeneous
    task = _task(sc, truth, np.array([0]), prior)  # one shared sensor: under-identified
    budget = Budget(sp_calls=10**9, iterations=80)
    y, g = ODTrace(), ODTrace()
    Yang1992Estimator(k_inner=120, outer_iters=40, theta=0.5).estimate(
        task, budget, RngBundle(0), y
    )
    GLSEstimator(k_inner=120, outer_iters=40, cv_prior=0.3).estimate(
        task, budget, RngBundle(0), g
    )
    assert np.abs(y.final.od_matrix - g.final.od_matrix).max() > 0.1


def test_yang_respects_budget_and_handles_empty_prior():
    """Coverage for two reachable branches: (1) the sp_calls budget breaks the
    outer loop early (iterations far below outer_iters), and (2) an all-zero prior
    (no active pairs) emits the prior unchanged instead of crashing the QP solve."""
    sc = braess_scenario(6.0)
    task = _task(sc, BRAESS_TRUTH, np.array([1, 2, 3]), _single_pair_prior(4.0))
    trace = ODTrace()
    Yang1992Estimator(k_inner=50, outer_iters=100).estimate(
        task, Budget(sp_calls=120), RngBundle(0), trace
    )
    assert trace.final.coords.iterations < 100  # broke early on the budget
    assert trace.final.coords.sp_calls < 100 * 50  # nowhere near the full run
    empty = _task(sc, BRAESS_TRUTH, np.array([1, 2, 3]), np.zeros((2, 2)))
    trace2 = ODTrace()
    Yang1992Estimator().estimate(empty, Budget(sp_calls=500), RngBundle(0), trace2)
    assert np.array_equal(trace2.final.od_matrix, np.zeros((2, 2)))
