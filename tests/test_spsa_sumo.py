"""Tests for the first guarded T2 estimator -- ``spsa-sumo`` (Balakrishna 2007).

``eclipse-sumo`` is an OPTIONAL extra; this whole file is skipped on a core
install (``pytest.importorskip('sumo')``), and the 731-test numpy suite runs
without it -- the live regression that ``import tabench`` works and the estimator
is simply ABSENT from ``ESTIMATOR_REGISTRY`` when the wheel is missing (the
sumo-free CI/matrix legs).

All anchors run on the ASYMMETRIC two-route UE instance (~0.22 s/inner solve;
never Braess in CI, never wall-time asserts -- GitHub runners are slower than the
dev box). What these pin: the registry/capabilities/golden-hash invariants; the
sp_calls-only budget refusal and the sp_calls=0 disclosure (the fabricated-
sp_calls trap); the adapter-delegated envelope refusals (power!=1 / nonzero fixed
cost -- the estimator's envelope IS the adapter's); a planted-truth recovery
anchor at a pinned seed with a LOOSE improves-on-prior bound (never tight
cross-platform decimals -- the BLAS lesson); a poisson negative-control pinning
that the clean-count anchor's improvement does NOT survive noise (an honest
disclosure, not a defect); bit-reproducibility given (root_seed, macrorep) and
macrorep divergence; certified ``od_feasible=1`` through the UNCHANGED pinned-bfw
certifier (the pilot's point: zero certificate changes); the wall-kill
RuntimeError (crash discipline, never feasible=0); the box-BINDING regression
(the three-lens review's confirmed P1 fix -- emitted == evaluated == in-box, and
it fails under a clip-removal mutation); and sparse pinned-certificate
checkpointing. See docs/design/adr-028-spsa-sumo.md.
"""

import numpy as np
import pytest

pytest.importorskip("sumo")

from tabench import Budget, Demand, Network, Scenario, two_route_scenario  # noqa: E402
from tabench.estimation import ESTIMATOR_REGISTRY, SumoSPSAEstimator  # noqa: E402
from tabench.estimation.base import PriorBaseline  # noqa: E402
from tabench.experiments.runner import run_estimation_experiment  # noqa: E402

try:
    from tabench import braess_scenario  # noqa: E402
except ImportError:  # pragma: no cover
    from tabench.data.builtin import braess_scenario  # noqa: E402

# The golden Braess content hash: this additive estimator must leave it -- and
# thus the whole scored instance canon -- byte-identical (HARD RULE).
BRAESS_GOLDEN_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"

# The recovery anchor: clean counts (noise='none') + a stale prior (cv=0.3) on
# the two-route UE instance, sensors on each route's split leg, held out on each
# route's second leg. Pinned seed 13 (measured: prior obs RMSE ~0.35, demand
# RMSE ~0.47; spsa-sumo drives both ~5x down through the marouter loop). Loose
# ceilings only.
_RECOVERY_EST = {
    "sensors": {"kind": "explicit", "links": [0, 2]},
    "heldout": {"kind": "explicit", "links": [1, 3]},
    "noise": "none",
    "n_periods": 3,
    "prior": {"kind": "stale", "cv": 0.3},
}
_RECOVERY_SEED = 13


def _two_route_like(fft, b, power, *, demand=4.0, toll=None, toll_weight=0.0):
    """A 2-route net (links 1->3, 3->2, 1->4, 4->2) with tunable cost columns.

    Used to build the delegated-refusal instances (power != 1 / nonzero fixed
    cost) the adapter -- and hence the estimator -- refuses."""
    init = np.array([1, 3, 1, 4], dtype=np.int64)
    term = np.array([3, 2, 4, 2], dtype=np.int64)
    network = Network(
        name="tr", n_nodes=4, n_zones=2, first_thru_node=1, init_node=init, term_node=term,
        capacity=np.ones(4), length=np.zeros(4), free_flow_time=np.asarray(fft, float),
        b=np.asarray(b, float), power=np.asarray(power, float),
        toll=np.zeros(4) if toll is None else np.asarray(toll, float),
        link_type=np.ones(4, dtype=np.int64), toll_weight=toll_weight,
    )
    od = np.zeros((2, 2))
    od[0, 1] = demand
    return Scenario(name="tr", network=network, demand=Demand(od), family="test-tr")


def _final_rows(result):
    """The last (final-checkpoint) certified row per estimator name."""
    last = {}
    for row in result.rows:
        last[row["estimator"]] = row
    return last


# --- registry / capabilities / golden hash -----------------------------------
def test_registered_and_capabilities():
    assert "spsa-sumo" in ESTIMATOR_REGISTRY
    caps = SumoSPSAEstimator.capabilities
    assert caps.paradigm == "estimation"
    assert caps.deterministic is False  # seeded, macroreplicated
    assert caps.seedable is True
    assert caps.provides_gap is False
    assert caps.inputs_required == frozenset({"link_counts", "prior_od"})
    assert caps.outputs == frozenset({"od_estimate"})


def test_golden_braess_hash_unchanged():
    # The new estimator is additive: it must not perturb the scored canon.
    assert braess_scenario().content_hash() == BRAESS_GOLDEN_HASH


# --- budget refusals / disclosure --------------------------------------------
def test_sp_calls_only_budget_refused():
    # marouter exposes no SP count, so an sp_calls-only budget cannot bound the
    # loop; it is refused up front rather than silently ignored (adr-028).
    scen = two_route_scenario(sue_theta=None)
    with pytest.raises(ValueError, match="sp_calls-only"):
        run_estimation_experiment(
            scen, [SumoSPSAEstimator(iters=3)], Budget(sp_calls=500),
            seed=0, macroreps=1, estimation=_RECOVERY_EST,
        )


def test_sp_calls_disclosed_as_zero():
    scen = two_route_scenario(sue_theta=None)
    res = run_estimation_experiment(
        scen, [SumoSPSAEstimator(iters=5)], Budget(iterations=1000),
        seed=_RECOVERY_SEED, macroreps=1, estimation=_RECOVERY_EST,
    )
    # Every emitted checkpoint discloses sp_calls == 0 (never fabricated from a
    # meaningless k_inner), on the coords AND in the certified CSV row.
    bundle = res.bundles[("spsa-sumo", "m0")]
    assert len(bundle.trace) >= 1
    assert all(state.coords.sp_calls == 0 for state in bundle.trace)
    assert all(row["sp_calls"] == 0 for row in res.rows if row["estimator"] == "spsa-sumo")


# --- adapter-delegated envelope refusals -------------------------------------
def test_delegated_power_refusal():
    # power != 1 is UNREPRESENTABLE in marouter's linear vdf -> the adapter (and
    # thus the estimator, by construction) refuses at estimate() start.
    scen = _two_route_like([1.0, 1.0, 1.0, 0.5], [0.0, 1.0, 0.0, 2.0], [1, 4, 1, 1])
    with pytest.raises(ValueError, match="power"):
        run_estimation_experiment(
            scen, [SumoSPSAEstimator(iters=3)], Budget(iterations=10),
            seed=0, macroreps=1, estimation=_RECOVERY_EST,
        )


def test_delegated_fixed_cost_refusal():
    # A nonzero generalized-cost fixed term (toll_weight*toll) enters the
    # certified cost but has no marouter hook -> refused, not silently dropped.
    scen = _two_route_like(
        [1.0, 1.0, 1.0, 0.5], [0.0, 1.0, 0.0, 2.0], [1, 1, 1, 1],
        toll=[0.0, 2.0, 0.0, 0.0], toll_weight=1.0,
    )
    with pytest.raises(ValueError, match="fixed cost"):
        run_estimation_experiment(
            scen, [SumoSPSAEstimator(iters=3)], Budget(iterations=10),
            seed=0, macroreps=1, estimation=_RECOVERY_EST,
        )


# --- recovery anchor (loose improves-on-prior) + certified feasibility --------
def test_recovery_improves_on_prior_and_certifies():
    scen = two_route_scenario(sue_theta=None)
    res = run_estimation_experiment(
        scen, [PriorBaseline(), SumoSPSAEstimator(iters=20)], Budget(iterations=1000),
        seed=_RECOVERY_SEED, macroreps=1, estimation=_RECOVERY_EST,
    )
    last = _final_rows(res)
    prior, spsa = last["prior"], last["spsa-sumo"]
    # Certified through the UNCHANGED pinned-bfw certifier (the pilot's point:
    # zero certificate changes); a converged, feasible OD.
    assert spsa["od_feasible"] == 1.0
    assert spsa["certificate_converged"] == 1.0
    assert abs(float(spsa["certificate_gap"])) < 1e-6
    # LOOSE improves-on-prior: SPSA at least halves both the observed count
    # misfit it minimizes and the (descriptive) demand RMSE (measured ~5x at
    # this seed). Never a tight cross-platform decimal.
    assert float(spsa["obs_count_rmse"]) < 0.6 * float(prior["obs_count_rmse"])
    assert float(spsa["od_rmse"]) < 0.6 * float(prior["od_rmse"])


def test_poisson_anchor_is_fragile_negative_control():
    # HONEST DISCLOSURE regression (adr-028): the recovery anchor above is pinned
    # on CLEAN counts. Under poisson counts the SAME pinned seed neither meets the
    # clean count-fit bound NOR improves the demand RMSE on the prior -- small-
    # sample count noise (3 periods) plus the mapping bias, a fragility shared by
    # any count-matching estimator on this 1-pair instance. This pins that the
    # noisy variant stays a certifiable-but-not-improving row, so a future edit
    # cannot quietly promote it into the ranked clean anchor.
    res = run_estimation_experiment(
        two_route_scenario(sue_theta=None),
        [PriorBaseline(), SumoSPSAEstimator(iters=20)], Budget(iterations=1000),
        seed=_RECOVERY_SEED, macroreps=1,
        estimation={**_RECOVERY_EST, "noise": "poisson"},
    )
    last = _final_rows(res)
    prior, spsa = last["prior"], last["spsa-sumo"]
    assert spsa["od_feasible"] == 1.0  # still an honest, certifiable row
    # The clean-anchor bounds do NOT hold under noise (measured: obs ratio ~0.71,
    # od ratio ~1.17): count-fit improvement shrinks and demand RMSE degrades.
    assert float(spsa["obs_count_rmse"]) > 0.6 * float(prior["obs_count_rmse"])
    assert float(spsa["od_rmse"]) > float(prior["od_rmse"])


# --- determinism: bit-reproducible given (root_seed, macrorep) ---------------
def test_bit_reproducible_and_macrorep_differs():
    scen = two_route_scenario(sue_theta=None)

    def _run():
        # ONE run with macroreps=2 already returns BOTH m0 and m1 bundles.
        return run_estimation_experiment(
            scen, [SumoSPSAEstimator(iters=6)], Budget(iterations=1000),
            seed=_RECOVERY_SEED, macroreps=2, estimation=_RECOVERY_EST,
        ).bundles

    a = _run()
    b = _run()  # a second identical run, for byte-identity
    m0a = a[("spsa-sumo", "m0")].final.od_matrix
    m1a = a[("spsa-sumo", "m1")].final.od_matrix
    m0b = b[("spsa-sumo", "m0")].final.od_matrix
    # Same (root_seed, macrorep): the SUE oracle is RNG-free and the SPSA
    # Rademacher draws replay -> the emitted OD is byte-identical across runs.
    assert np.array_equal(m0a, m0b)
    # A different macrorep draws an independent SPSA stream (and dataset) ->
    # a genuinely different trace, so the estimator is macroreplicable.
    assert not np.array_equal(m0a, m1a)


# --- wall-kill: crash discipline, never feasible=0 ---------------------------
def test_wall_budget_kill_raises_runtimeerror():
    # A wall_seconds budget exhausted mid-loop is an infrastructure outcome; it
    # RAISES (with the engine command), never launders into a feasible zero-flow
    # solution. The single deadline is threaded across all inner solves.
    scen = two_route_scenario(sue_theta=None)
    with pytest.raises(RuntimeError):
        run_estimation_experiment(
            scen, [SumoSPSAEstimator(iters=8)], Budget(iterations=1000, wall_seconds=1e-6),
            seed=0, macroreps=1, estimation=_RECOVERY_EST,
        )


# --- box projection BINDS: emitted==evaluated==in-box (the review-MAJOR fix) --
def test_box_projection_binds_and_emitted_respects_box():
    # Seed 14 puts the stale prior (cv=0.6) far below truth (4.0) and the 3x box
    # ceiling (~2.67) below truth, so SPSA WANTS to climb past the box and the
    # thesis step-5/8 projection must cap it. Regression for the review MAJOR:
    # the emitted best-iterate must be the EVALUATED in-box point (P1) -- never
    # the raw out-of-box candidate the eval-time-only clip used to emit
    # (reviewers measured emitted 3.49 > box 2.67, self-report describing a
    # different point, honesty diff ~0.43). Discriminating: every assertion below
    # FAILS under a clip-removal mutation of SumoSPSAEstimator._project.
    HI_FRAC = 3.0
    est = SumoSPSAEstimator(iters=12, c=0.8, step_clip=2.0, demand_hi_frac=HI_FRAC)
    res = run_estimation_experiment(
        two_route_scenario(sue_theta=None), [est], Budget(iterations=1000),
        seed=14, macroreps=1,
        estimation={**_RECOVERY_EST, "prior": {"kind": "stale", "cv": 0.6}},
    )
    rows = [r for r in res.rows if r["estimator"] == "spsa-sumo"]
    assert rows and all(r["od_feasible"] == 1.0 for r in rows)
    hi = float(est._hi_vec[0])  # 3 * prior[0,1] on the single active OD pair
    emitted = float(res.bundles[("spsa-sumo", "m0")].final.od_matrix[0, 1])
    # The box BINDS (ceiling below truth): emitted sits AT the ceiling, in-box.
    assert emitted <= hi + 1e-9  # in-box (fails on the raw-candidate P1 bug)
    assert abs(emitted - hi) < 0.05  # the clip actually fired (~2.67, not ~3.49)
    # ... and the self-report describes the EMITTED point, not a clipped-away one:
    # the honesty diff collapses to the mapping-floor scale (was ~0.43 out-of-box).
    last = rows[-1]
    assert abs(float(last["self_obs_count_rmse"]) - float(last["obs_mean_count_rmse"])) < 0.05


# --- sparse pinned-certificate checkpointing ---------------------------------
def test_sparse_checkpointing():
    # Each checkpoint costs a full pinned certificate, so emission is sparse
    # (~15 + final), never one-per-iteration for a larger budget (ADR-002 Dec 2).
    scen = two_route_scenario(sue_theta=None)
    res = run_estimation_experiment(
        scen, [SumoSPSAEstimator(iters=30)], Budget(iterations=1000),
        seed=_RECOVERY_SEED, macroreps=1, estimation=_RECOVERY_EST,
    )
    bundle = res.bundles[("spsa-sumo", "m0")]
    assert 1 <= len(bundle.trace) <= 16


# --- CLI T2 card path: the iterations passthrough (adr-028, cli.py) ----------
def test_cli_card_path_passes_iterations_budget(tmp_path):
    # Without the T2 dispatch passing budgets{iterations,wall_seconds}, a
    # spsa-sumo card would build Budget(sp_calls=...) only and die on the
    # estimator's own sp_calls-only refusal. The card path must succeed.
    import yaml

    from tabench.cli import main

    card = {
        "scenario": "tworoute",
        "tasks": ["t2_estimation"],
        "estimation": {
            "sensors": {"kind": "explicit", "links": [0, 2]},
            "heldout": {"kind": "explicit", "links": [1, 3]},
            "noise": "none", "n_periods": 3, "prior": {"kind": "stale", "cv": 0.3},
        },
        "budgets": {"iterations": 6, "sp_calls": 2000},
    }
    card_path = tmp_path / "spsa_sumo_t2.yaml"
    card_path.write_text(yaml.safe_dump(card))
    out = tmp_path / "out"
    rc = main([
        "run", "--config", str(card_path), "--models", "spsa-sumo",
        "--seed", str(_RECOVERY_SEED), "--out", str(out),
    ])
    assert rc == 0  # 2 would mean the sp_calls-only refusal fired
    assert list(out.glob("*.csv"))
