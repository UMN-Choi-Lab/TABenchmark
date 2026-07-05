"""Tests for the probit-SUE task: Monte Carlo loading, MSA-SUE, and the pinned
fixed-point certificate (docs/design/adr-003).

The two-route network has disjoint 2-link routes, so perceived route costs are
independent normals and probit reduces to a scalar fixed point solved here with
brentq — the tests never trust pre-computed digits. Every closed form used as
an anchor is recomputed in-test; the leaderboard-facing certificate columns are
checked against the ADR's own significance rule
``max(sue_residual_floor, 2 * sue_residual_se)``, not against a bare floor.
"""

import dataclasses
import math

import numpy as np
import pytest
from conftest import load_or_skip
from scipy.optimize import brentq
from scipy.stats import norm

from tabench import (
    Budget,
    DialSUEModel,
    Evaluator,
    RngBundle,
    SueProbitMsaModel,
    Trace,
    bootstrap_ci,
    braess_scenario,
    run_experiment,
    two_route_scenario,
)
from tabench.models._probit import ProbitEngine

# Golden hash of the Braess scenario (a logit-default scenario): the sue_family
# append must leave it, and every other logit scenario, byte-identical.
BRAESS_GOLDEN_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"

BETA = 0.1  # the two-route card dial (native abstract time units)


def _fixed_point_route_a(beta: float, demand: float = 4.0) -> float:
    """Root of ``f = D * Phi((c_B(D - f) - c_A(f)) / sqrt(3.5 beta))``.

    Disjoint routes make the perceived route costs independent normals with
    difference variance ``(1 + 1 + 1 + 0.5) beta = 3.5 beta``; ``c_A(f) = 2 + f``
    and ``c_B(g) = 1.5 + 2 g`` are the two-route linear latencies.
    """
    scale = math.sqrt(3.5 * beta)

    def residual(f_a: float) -> float:
        c_a = 2.0 + f_a
        c_b = 1.5 + 2.0 * (demand - f_a)
        return f_a - demand * norm.cdf((c_b - c_a) / scale)

    return brentq(residual, 1e-12, demand - 1e-12, xtol=1e-14, rtol=8.9e-16)


@pytest.fixture(scope="module")
def scenario():
    return two_route_scenario(sue_theta=BETA, sue_family="probit")


# ------------------------------------------------------------- analytic anchors


def test_two_route_fixed_point_matches_known_digit():
    """The closed-form probit split recomputes to the ADR's anchor digits."""
    f_a = _fixed_point_route_a(BETA)
    assert f_a == pytest.approx(2.4443574168, abs=1e-9)
    # UE (beta -> 0) is 2.5, and probit sits below it just like logit's 2.299.
    assert 2.29 < f_a < 2.5


def test_free_flow_first_iterate(scenario):
    """The free-flow loading expectation is 4*Phi(-0.5/sqrt(0.35))."""
    analytic = 4.0 * norm.cdf(-0.5 / math.sqrt(0.35))
    assert analytic == pytest.approx(0.7960494390, abs=1e-9)

    net = scenario.network
    engine = ProbitEngine(net)
    n = 20000
    gen = RngBundle(0).generator(source=0, replication=0)
    mean = engine.load(net.link_cost(np.zeros(net.n_links)), scenario.demand, BETA, gen, n)
    # CLT sd of the mean route-A flow at n draws: D sqrt(p(1-p)/n).
    p = analytic / 4.0
    sd = 4.0 * math.sqrt(p * (1.0 - p) / n)
    assert abs(mean[0] - analytic) < 4.0 * sd


def test_msa_converges_to_analytic_fixed_point(scenario):
    """MSA reaches the probit fixed point within its terminal sampling sd."""
    f_a = _fixed_point_route_a(BETA)
    trace = Trace()
    k = 5000
    SueProbitMsaModel().solve(scenario, Budget(iterations=k), RngBundle(0), trace)
    final_a = trace.final.link_flows[0]
    # Terminal iterate sd of a 1/k-averaged Bernoulli(p) load over k draws.
    p = f_a / 4.0
    sd = 4.0 * math.sqrt(p * (1.0 - p) / k)
    assert abs(final_a - f_a) < 4.0 * sd
    # The final certified state is v_k (recorded BEFORE the k-th update).
    assert trace.final.coords.iterations == k
    assert len(trace) == k


# ---------------------------------------------------------------- certificate


def test_certified_residual_at_fixed_point_within_floor(scenario):
    """At the exact analytic v*, the certified residual is pure MC noise."""
    f_a = _fixed_point_route_a(BETA)
    v_star = np.array([f_a, f_a, 4.0 - f_a, 4.0 - f_a])
    metrics = Evaluator(scenario, root_seed=0, r_cert=1600).evaluate(v_star)
    assert metrics["feasible"] == 1.0
    residual = metrics["sue_fixed_point_residual"]
    se = metrics["sue_residual_se"]
    floor = metrics["sue_residual_floor"]
    # At v* the residual is a half-normal draw whose mean IS the floor, so it
    # is O(floor), not reliably below it: across seeds `residual < floor` is a
    # ~50/50 coin flip (max ratio ~3.4). Assert it stays within the MC noise
    # band — far below the >> floor residual a perturbed flow certifies (the
    # test below) — rather than relying on the pinned seed's low draw.
    assert se > 0.0
    assert residual < 5.0 * floor
    assert floor == pytest.approx(0.03894, abs=2e-3)


def test_probit_scenario_carries_no_logit_reference():
    """A probit instance must not inherit the binary-logit analytic oracle:
    the two equilibria differ (logit f_A=2.299 vs probit f_A=2.444 at beta=0.1),
    so scoring flow_rmse against the logit flows would be silently wrong."""
    logit = two_route_scenario(sue_theta=0.5, sue_family="logit")
    probit = two_route_scenario(sue_theta=0.5, sue_family="probit")
    assert logit.reference is not None  # logit at (theta=0.5, D=4) still has one
    assert probit.reference is None


def test_evaluator_rejects_r_cert_below_two(scenario):
    """The jackknife SE and CLT floor are undefined below r_cert=2; a feasible
    row must never emit the NaN that elsewhere signals censoring (ADR-003)."""
    for bad in (0, 1):
        with pytest.raises(ValueError, match="r_cert >= 2"):
            Evaluator(scenario, root_seed=0, r_cert=bad)
    # The default and any r_cert >= 2 construct fine.
    assert Evaluator(scenario, root_seed=0, r_cert=2) is not None


def test_certificate_pinned_stream_byte_reproducible(scenario):
    """Pinning E on SOURCE_EVALUATION restores P1: identical bytes on replay."""
    f_a = _fixed_point_route_a(BETA)
    v = np.array([f_a, f_a, 4.0 - f_a, 4.0 - f_a])
    a = Evaluator(scenario, root_seed=0, r_cert=1600).evaluate(v)
    b = Evaluator(scenario, root_seed=0, r_cert=1600).evaluate(v)
    for key in ("sue_fixed_point_residual", "sue_residual_se", "sue_residual_floor"):
        assert a[key] == b[key]


def test_macrorep_bootstrap_ci_in_manifest():
    """The stochastic track wires bootstrap_ci into the run_experiment manifest
    (ADR-003 Decision 4): a percentile CI of the final certified residual across
    macroreps, not just an unused library helper. A single macrorep carries none.
    """
    sc = two_route_scenario(sue_theta=BETA, sue_family="probit")
    result = run_experiment(
        sc, [SueProbitMsaModel()], Budget(iterations=8), seed=0, macroreps=4, r_cert=64
    )
    ci = result.manifest["bootstrap"]["sue-probit-msa"]
    assert ci["metric"] == "sue_fixed_point_residual"
    assert ci["n_macroreps"] == 4
    assert ci["lo"] <= ci["point"] <= ci["hi"]

    single = run_experiment(
        sc, [SueProbitMsaModel()], Budget(iterations=8), seed=0, macroreps=1, r_cert=64
    )
    assert single.manifest["bootstrap"] == {}


def test_certificate_positive_for_perturbed_flows(scenario):
    """A flow half a unit off the fixed point certifies well above the floor."""
    f_a = _fixed_point_route_a(BETA) - 0.5
    metrics = Evaluator(scenario, root_seed=0, r_cert=1600).evaluate(
        np.array([f_a, f_a, 4.0 - f_a, 4.0 - f_a])
    )
    assert metrics["feasible"] == 1.0
    assert metrics["sue_fixed_point_residual"] > max(
        metrics["sue_residual_floor"], 2.0 * metrics["sue_residual_se"]
    )


def test_evaluator_censors_infeasible_flows(scenario):
    """Zero flows are censored: feasible=0 and all three SUE columns NaN."""
    metrics = Evaluator(scenario, root_seed=0, r_cert=64).evaluate(np.zeros(4))
    assert metrics["feasible"] == 0.0
    assert math.isnan(metrics["sue_fixed_point_residual"])
    assert math.isnan(metrics["sue_residual_se"])
    assert math.isnan(metrics["sue_residual_floor"])


# ---------------------------------------------------------------- reproducibility


def test_seeded_reproducibility_and_macrorep_independence(scenario):
    """Same (root_seed, macrorep) is byte-identical; a new macrorep differs."""

    def solve(macrorep: int) -> np.ndarray:
        trace = Trace()
        SueProbitMsaModel().solve(
            scenario, Budget(iterations=50), RngBundle(0, macrorep=macrorep), trace
        )
        return trace.final.link_flows

    assert np.array_equal(solve(0), solve(0))
    assert not np.array_equal(solve(0), solve(1))


# ---------------------------------------------------------------------- hashing


def test_golden_and_logit_hashes_preserved():
    """The sue_family append leaves every logit scenario byte-identical."""
    assert braess_scenario().content_hash() == BRAESS_GOLDEN_HASH
    logit = two_route_scenario(sue_theta=BETA)  # default family="logit"
    explicit = two_route_scenario(sue_theta=BETA, sue_family="logit")
    # An explicit default appends nothing, matching the implicit default.
    assert logit.content_hash() == explicit.content_hash()


def test_probit_and_logit_hash_differently_at_same_theta():
    logit = two_route_scenario(sue_theta=BETA).content_hash()
    probit = two_route_scenario(sue_theta=BETA, sue_family="probit").content_hash()
    assert logit != probit


def test_invalid_sue_family_rejected():
    with pytest.raises(ValueError, match="sue_family"):
        two_route_scenario(sue_theta=BETA, sue_family="weibit")
    with pytest.raises(ValueError, match="sue_family='probit' requires"):
        dataclasses.replace(braess_scenario(), sue_family="probit")


# ------------------------------------------------------------- solver dispatch


def test_solvers_reject_the_wrong_family(scenario):
    """sue-msa raises on a probit task; sue-probit-msa raises on a logit task."""
    with pytest.raises(ValueError, match="sue_family"):
        DialSUEModel().solve(scenario, Budget(iterations=3), RngBundle(0), Trace())
    logit = two_route_scenario(sue_theta=0.5)
    with pytest.raises(ValueError, match="sue_family"):
        SueProbitMsaModel().solve(logit, Budget(iterations=3), RngBundle(0), Trace())


def test_probit_solver_requires_sue_scenario():
    with pytest.raises(ValueError, match="sue_theta"):
        SueProbitMsaModel().solve(
            braess_scenario(), Budget(iterations=3), RngBundle(0), Trace()
        )


# ----------------------------------------------------------------- bootstrap CI


def test_bootstrap_ci_deterministic_and_covers_the_mean():
    """Percentile CI is reproducible from root_seed and covers a known mean."""
    sample = np.random.default_rng(20260705).normal(0.3, 1.0, size=200)
    first = bootstrap_ci(sample, root_seed=0)
    second = bootstrap_ci(sample, root_seed=0)
    assert first == second
    # The 95% CI of the mean brackets the true mean of this large sample.
    assert first.lo < 0.3 < first.hi
    assert first.lo < first.point < first.hi
    # A different resampling seed gives a (slightly) different interval.
    assert bootstrap_ci(sample, root_seed=1) != first


def test_bootstrap_ci_rejects_empty_and_bad_level():
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([]), root_seed=0)
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([1.0, 2.0]), root_seed=0, level=1.5)


# ------------------------------------------------------------- Sioux Falls smoke


def test_siouxfalls_probit_macroreps_route():
    """Sioux Falls probit smoke: macroreps produce distinct certified rows."""
    base = load_or_skip("siouxfalls")
    scenario = dataclasses.replace(
        base, sue_theta=0.5, sue_family="probit", reference=None
    )
    macroreps = 3
    result = run_experiment(
        scenario,
        [SueProbitMsaModel()],
        Budget(iterations=30),
        macroreps=macroreps,
        r_cert=40,
    )
    macros = sorted({row["macrorep"] for row in result.rows})
    assert macros == [0, 1, 2]
    # 30 checkpoints per macrorep, one row each.
    assert len(result.rows) == macroreps * 30
    for row in result.rows:
        if row["feasible"] == 1.0:
            assert math.isfinite(row["sue_residual_se"])
            assert math.isfinite(row["sue_residual_floor"])
    # Independent solver trajectories: many-route network, so flows diverge.
    f0 = result.bundles[("sue-probit-msa", "m0")].final.link_flows
    f1 = result.bundles[("sue-probit-msa", "m1")].final.link_flows
    assert not np.array_equal(f0, f1)
