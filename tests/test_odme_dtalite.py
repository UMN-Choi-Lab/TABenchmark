"""Tests for the DTALite-ODME guarded T2 estimator -- ``odme-dtalite`` (adr-042).

``DTALite`` is an OPTIONAL extra; this whole file is skipped on a core install
(``pytest.importorskip('DTALite')``, EXACT case), and the numpy suite runs without it
(the dtalite-free CI/matrix legs are the live regression that ``import tabench`` still
works and the estimator is simply ABSENT from ``ESTIMATOR_REGISTRY``).

What these pin, all VERSION-ROBUST (properties + loose ceilings on MEASURED values, never
tight cross-platform decimals -- the FW/VDF/ODME behavior lives in a bundled ``.so``):
the registry/capabilities/golden-hash invariants; banner-suppression + core-install guard;
the ``route_output=1`` + ``odme_mode=1`` settings pin (THE non-obvious requirement -- with
``route_output=0`` the ODME reconstruction collapses and inflates the OD +40%, adr-042);
the Sioux Falls marquee recovery through the UNCHANGED pinned-bfw certifier (obs/heldout
count RMSE improve on the prior; ODME runs > 0 gradient iterations); the anti-laundering
property from BOTH sides (the certifier scores the emitted OD only and IGNORES DTALite's
rosy self-report; a poisoned OD certifies feasible-but-worse); the ``od_performance.csv``
completeness/consistency/phantom read-back gates; that ``link_performance.csv`` corruption
never reaches the OD estimate; the delegated toll/capacity refusals + the ``sp_calls``-only
refusal; the wall-kill RuntimeError; byte-determinism; the one-sided ``demand_target_frac``
box dial. See docs/design/adr-042-odme-dtalite.md.
"""

import csv
import os
import shutil

import numpy as np
import pytest

pytest.importorskip("DTALite")  # EXACT case; the extra name 'dtalite' is not the module

from conftest import load_or_skip  # noqa: E402

from tabench import (  # noqa: E402
    BiconjugateFrankWolfeModel,
    Budget,
    Demand,
    Network,
    RngBundle,
    Scenario,
    Trace,
    braess_scenario,
)
from tabench.estimation import ESTIMATOR_REGISTRY, EstimationTask, ODTrace  # noqa: E402
from tabench.estimation import odme_dtalite as od  # noqa: E402
from tabench.estimation._proportions import active_pairs  # noqa: E402
from tabench.estimation.base import PriorBaseline  # noqa: E402
from tabench.estimation.odme_dtalite import DtaliteODMEEstimator  # noqa: E402
from tabench.experiments.runner import (  # noqa: E402
    _SENSOR_PLACEMENT_REPLICATION,
    SOURCE_OBSERVATION,
    SOURCE_PRIOR,
    _draw_sensors,
    identifiability_report,
    run_estimation_experiment,
)
from tabench.metrics.estimation import CERTIFICATE_DEFAULTS, ODCertifier  # noqa: E402
from tabench.observe.levels import LinkCounts, StalePriorOD  # noqa: E402

# The golden Braess content hash: this additive estimator must leave the scored canon
# byte-identical (HARD RULE).
BRAESS_GOLDEN_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"

# THE marquee anchor (measured, adr-042): siouxfalls UE, clean counts, sensors random
# cov=0.5, held-out cov=0.2, stale prior cv=0.3, seed 7. Prior obs RMSE ~995, heldout
# ~816; odme drives obs ~0.37x and the ranking heldout ~0.68x through the UNCHANGED
# pinned-bfw certifier, with the ODME descent running ~69 gradient iterations. Loose
# ceilings only.
_MARQUEE = {
    "sensors": {"kind": "random", "coverage": 0.5},
    "heldout": {"kind": "random", "coverage": 0.2},
    "noise": "none",
    "n_periods": 1,
    "prior": {"kind": "stale", "cv": 0.3},
}
_MARQUEE_SEED = 7


def _final_rows(result):
    return {row["estimator"]: row for row in result.rows}


def _build_sf_task(seed=_MARQUEE_SEED, cov=0.5):
    """Build (scenario, task, oracle_flows, sensors) for the siouxfalls marquee anchor.

    Replicates ``run_estimation_experiment``'s task construction so the estimator-direct
    tests (determinism, link_performance independence) share ONE pinned-bfw oracle solve
    rather than paying it per full run."""
    scen = load_or_skip("siouxfalls")
    net = scen.network
    bfw = BiconjugateFrankWolfeModel(line_search_xtol=1e-12)
    tr = Trace()
    bfw.solve(scen, Budget(iterations=5000, target_relative_gap=1e-6), RngBundle(seed), tr)
    oracle = tr.final.link_flows
    place = RngBundle(root_seed=seed, macrorep=0).generator(
        SOURCE_OBSERVATION, replication=_SENSOR_PLACEMENT_REPLICATION
    )
    sensors = _draw_sensors(net.n_links, {"kind": "random", "coverage": cov}, place)
    rb = RngBundle(root_seed=seed, macrorep=0)
    prior = Demand(
        StalePriorOD(cv=0.3).observe(scen, oracle, rb.generator(SOURCE_PRIOR)).payload["prior_od"]
    )
    obs_ds = LinkCounts(sensors, 1, "none").observe(
        scen, oracle, rb.generator(SOURCE_OBSERVATION)
    )
    ident = identifiability_report(net, scen.demand, sensors, k_inner=40)
    task = EstimationTask(
        name=scen.name, network=net, prior=prior, dataset=obs_ds, identifiability=ident,
        scenario_hash=scen.content_hash(), certificate=CERTIFICATE_DEFAULTS, seed=seed,
    )
    return scen, task, oracle, sensors


def _two_route_ue(fft, b, power, *, demand=4.0, toll=None, toll_weight=0.0, cap=None):
    """A UE (sue_theta=None) 2-route net (links 1->3,3->2,1->4,4->2) with tunable cols."""
    init = np.array([1, 3, 1, 4], dtype=np.int64)
    term = np.array([3, 2, 4, 2], dtype=np.int64)
    network = Network(
        name="tr", n_nodes=4, n_zones=2, first_thru_node=1, init_node=init, term_node=term,
        capacity=np.ones(4) if cap is None else np.asarray(cap, float),
        length=np.zeros(4), free_flow_time=np.asarray(fft, float),
        b=np.asarray(b, float), power=np.asarray(power, float),
        toll=np.zeros(4) if toll is None else np.asarray(toll, float),
        link_type=np.ones(4, dtype=np.int64), toll_weight=toll_weight,
    )
    ok = np.zeros((2, 2))
    ok[0, 1] = demand
    return Scenario(name="tr", network=network, demand=Demand(ok), family="test-tr")


def _write_od_csv(path, rows):
    """Write a minimal od_performance.csv (o_zone_id,d_zone_id,volume) for parser tests."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["route_key", "o_zone_id", "d_zone_id", "volume"])
        for (o, d, v) in rows:
            w.writerow([f"{o}_{d}_auto", o, d, v])


# --- registry / capabilities / golden hash -----------------------------------
def test_registered_and_capabilities():
    assert "odme-dtalite" in ESTIMATOR_REGISTRY
    caps = DtaliteODMEEstimator.capabilities
    assert caps.paradigm == "estimation"
    assert caps.deterministic is True  # byte-identical reruns at OMP_NUM_THREADS=1
    assert caps.seedable is False  # the engine exposes no seed to pin
    assert caps.provides_gap is False
    assert caps.inputs_required == frozenset({"link_counts", "prior_od"})
    assert caps.outputs == frozenset({"od_estimate"})


def test_golden_braess_hash_unchanged():
    # The new estimator is additive: it must not perturb the scored canon.
    assert braess_scenario().content_hash() == BRAESS_GOLDEN_HASH


# --- banner suppression + core-install guard ---------------------------------
def test_import_tabench_silent_and_guard_unregisters_when_blocked():
    """`import tabench` must print NOTHING (a naive module-scope `import DTALite` would
    leak the wheel's banner), and blocking DTALite must unregister ``odme-dtalite`` while
    leaving the numpy core importable -- run in a subprocess with a meta_path blocker."""
    import subprocess
    import sys

    code = (
        "import sys, io, importlib.abc\n"
        "class B(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'DTALite' or name.startswith('DTALite.'):\n"
        "            raise ModuleNotFoundError(name, name='DTALite')\n"
        "        return None\n"
        "sys.meta_path.insert(0, B()); sys.modules.pop('DTALite', None)\n"
        "buf = io.StringIO(); old = sys.stdout; sys.stdout = buf\n"
        "import tabench\n"
        "from tabench.estimation import ESTIMATOR_REGISTRY\n"
        "sys.stdout = old\n"
        "assert 'odme-dtalite' not in ESTIMATOR_REGISTRY, 'registered despite blocked DTALite'\n"
        "assert 'prior' in ESTIMATOR_REGISTRY, 'numpy core failed to register'\n"
        "assert buf.getvalue() == '', 'import tabench polluted stdout: %r' % buf.getvalue()\n"
        "print('GUARD_OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "GUARD_OK" in proc.stdout


# --- route_output=1 + odme_mode=1 settings pin (THE non-obvious requirement) --
def test_settings_pins_route_output_and_odme_mode(tmp_path):
    """The single most load-bearing setting: ``route_output=1`` (so ODME can reconstruct
    modeled link volume from the route history) AND ``odme_mode=1``. With ``route_output=0``
    the reconstruction collapses to ~0 on every sensor and ODME inflates the OD to its box
    ceiling (measured +40% total demand, a degenerate estimate, adr-042). This structural
    pin fails if a future edit reverts route_output to dtalite-tap's lean 0."""
    _, task, _, _ = _build_sf_task()
    od._write_odme_inputs(task, str(tmp_path), n_iterations=50, demand_target_frac=1.0)
    with open(tmp_path / "settings.csv", newline="") as fh:
        row = next(csv.DictReader(fh))
    assert row["route_output"] == "1", "route_output MUST be 1 for the ODME reconstruction"
    assert row["odme_mode"] == "1"
    assert row["number_of_processors"] == "1"  # OMP determinism pin partner


# --- one-sided demand_target_frac box dial -----------------------------------
def test_demand_target_frac_is_one_sided_and_hashed(tmp_path):
    # frac scales demand_target.csv only (the box anchor / OD-reg target); demand.csv (seed)
    # is always the raw prior. frac>1 raises the upper recovery bound one-sidedly (adr-042).
    _, task, _, _ = _build_sf_task()
    prior = task.prior.matrix
    pairs = active_pairs(prior)
    od._write_odme_inputs(task, str(tmp_path), n_iterations=50, demand_target_frac=2.0)
    seed = {(int(r["o_zone_id"]), int(r["d_zone_id"])): float(r["volume"])
            for r in csv.DictReader(open(tmp_path / "demand.csv"))}
    target = {(int(r["o_zone_id"]), int(r["d_zone_id"])): float(r["volume"])
              for r in csv.DictReader(open(tmp_path / "demand_target.csv"))}
    (i, j) = pairs[0]
    assert seed[(i + 1, j + 1)] == pytest.approx(prior[i, j])
    assert target[(i + 1, j + 1)] == pytest.approx(2.0 * prior[i, j])
    # frac is a hashed estimator-identity factor (lands in the bundle factors).
    assert "demand_target_frac" in DtaliteODMEEstimator(demand_target_frac=2.0).factor_values


# --- od_performance.csv read-back gates (engine-free parser) ------------------
def test_readback_completeness_gate(tmp_path):
    prior = np.array([[0.0, 3.0], [5.0, 0.0]])
    pairs = active_pairs(prior)  # (0,1) and (1,0)
    _write_od_csv(tmp_path / "od.csv", [(1, 2, 3.0)])  # omit the (1,0) pair
    with pytest.raises(od._ODMEReadError, match="omitted"):
        od._read_od_estimate(str(tmp_path / "od.csv"), prior, pairs)


def test_readback_consistency_gate(tmp_path):
    prior = np.array([[0.0, 3.0], [5.0, 0.0]])
    pairs = active_pairs(prior)
    # Two route rows for one pair reporting DIFFERENT total volumes -> raise.
    _write_od_csv(tmp_path / "od.csv", [(1, 2, 3.0), (1, 2, 9.0), (2, 1, 5.0)])
    with pytest.raises(od._ODMEReadError, match="inconsistent"):
        od._read_od_estimate(str(tmp_path / "od.csv"), prior, pairs)


def test_readback_phantom_pair_gate(tmp_path):
    # 3-zone prior whose support is (0,1) and (1,0) only; the engine emits an off-support
    # off-diagonal pair (0,2) -> phantom -> raise.
    prior = np.array([[0.0, 3.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    pairs = active_pairs(prior)
    _write_od_csv(tmp_path / "od.csv", [(1, 2, 3.0), (2, 1, 5.0), (1, 3, 7.0)])
    with pytest.raises(od._ODMEReadError, match="phantom|absent"):
        od._read_od_estimate(str(tmp_path / "od.csv"), prior, pairs)


def test_readback_scatters_volume_and_carries_diagonal(tmp_path):
    prior = np.array([[2.0, 3.0], [5.0, 4.0]])  # diagonal 2,4 = intrazonal
    pairs = active_pairs(prior)
    _write_od_csv(tmp_path / "od.csv", [(1, 2, 30.0), (2, 1, 50.0)])
    est = od._read_od_estimate(str(tmp_path / "od.csv"), prior, pairs)
    assert est[0, 1] == 30.0 and est[1, 0] == 50.0  # from od_performance volume
    assert est[0, 0] == 2.0 and est[1, 1] == 4.0  # diagonal carried from prior


# --- link_performance.csv corruption never reaches the OD estimate -----------
def test_link_performance_corruption_never_reaches_od():
    """The OD estimate is sourced ONLY from od_performance.csv (link_performance.csv is
    corrupted under odme_mode=1, adr-042). Run with keep_files, then DELETE
    link_performance.csv and re-parse od_performance.csv -> byte-identical OD."""
    _, task, _, _ = _build_sf_task()
    est = DtaliteODMEEstimator(keep_files=True)
    tr = ODTrace()
    bundle = est.estimate(task, Budget(iterations=100), RngBundle(0), tr)
    workdir = est.last_workdir
    try:
        emitted = bundle.final.od_matrix
        lp = os.path.join(workdir, "link_performance.csv")
        if os.path.exists(lp):
            os.remove(lp)  # corrupt/remove the channel we must NOT depend on
        pairs = active_pairs(task.prior.matrix)
        reparsed = od._read_od_estimate(
            os.path.join(workdir, "od_performance.csv"), task.prior.matrix, pairs
        )
        assert np.array_equal(emitted, reparsed)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# --- MARQUEE: recovery improves on prior + certifies + ODME actually descends -
def test_marquee_recovery_improves_on_prior_and_certifies():
    scen = load_or_skip("siouxfalls")  # BPR power-4, representable by DTALite (not marouter)
    res = run_estimation_experiment(
        scen, [PriorBaseline(), DtaliteODMEEstimator()], Budget(iterations=100),
        seed=_MARQUEE_SEED, macroreps=1, estimation=_MARQUEE,
    )
    last = _final_rows(res)
    prior, odme = last["prior"], last["odme-dtalite"]
    # Certified through the UNCHANGED pinned-bfw certifier (the adr-028 ideal: zero
    # certifier changes); a converged, feasible OD.
    assert odme["od_feasible"] == 1.0
    assert odme["certificate_converged"] == 1.0
    assert abs(float(odme["certificate_gap"])) < 1e-6
    # LOOSE improves-on-prior on BOTH the observed count fit ODME calibrates AND the
    # ranking held-out count fit (measured ~0.37x / ~0.68x). Never a tight decimal.
    assert float(odme["obs_count_rmse"]) < 0.6 * float(prior["obs_count_rmse"])
    assert float(odme["heldout_count_rmse"]) < 0.9 * float(prior["heldout_count_rmse"])
    # The ODME gradient descent actually FIRED (> 0 iterations): the demand magnitude
    # clears the hardcoded tol=1 floor at Sioux-Falls scale (NOT a "0 iterations" no-op).
    bundle = res.bundles[("odme-dtalite", "m0")]
    assert float(bundle.trace.final.self_report["engine_odme_iterations"]) > 0.0


# --- anti-laundering, side 1: the certifier IGNORES DTALite's self-report -----
def test_certifier_ignores_dtalite_self_report():
    """The sacred property (adr-042): the ONLY channel is ODResultBundle.final.od_matrix;
    the certifier re-solves bfw and NEVER reads DTALite's own bookkeeping. DTALite reports a
    rosy count fit (its stalled-assignment, box-clamped predicted volumes ~a few veh RMSE),
    but the pinned-bfw certifier scores the EMITTED OD honestly (hundreds RMSE). The
    certified value -- not the self-report -- is what the leaderboard sees."""
    res = run_estimation_experiment(
        load_or_skip("siouxfalls"), [DtaliteODMEEstimator()], Budget(iterations=100),
        seed=_MARQUEE_SEED, macroreps=1, estimation=_MARQUEE,
    )
    row = _final_rows(res)["odme-dtalite"]
    bundle = res.bundles[("odme-dtalite", "m0")]
    self_fit = float(bundle.trace.final.self_report["obs_count_rmse"])
    certified = float(row["obs_count_rmse"])
    # DTALite's optimistic self-report is FAR below the certified value: the certifier did
    # not trust it (the measured engine-in-the-loop bias, the spsa-sumo reframing).
    assert self_fit < 0.5 * certified
    # The scored CSV column is the certified value, not the self-report.
    assert float(row["self_obs_count_rmse"]) == pytest.approx(self_fit)


# --- anti-laundering, side 2: a poisoned OD certifies feasible-but-worse ------
def test_poisoned_od_certifies_honestly_as_no_improvement():
    """A bad/adversarial OD cannot forge a good certificate: the model-blind pinned-bfw
    ODCertifier scores whatever OD it is given. A 3x-inflated OD certifies feasible=1 with a
    catastrophically WORSE count fit than the truth OD -- the adr-028 negative-control shape,
    proving DTALite cannot launder a bad estimate."""
    scen = _two_route_ue([1.0, 1.0, 1.0, 0.5], [0.0, 1.0, 0.0, 2.0], [1, 1, 1, 1], demand=4.0)
    bfw = BiconjugateFrankWolfeModel(line_search_xtol=1e-12)
    tr = Trace()
    bfw.solve(scen, Budget(iterations=5000, target_relative_gap=1e-6), RngBundle(0), tr)
    oracle = tr.final.link_flows
    sensors = np.array([0, 2], dtype=np.int64)
    heldout = np.array([1, 3], dtype=np.int64)
    counts = oracle[sensors][None, :]
    ho_counts = oracle[heldout][None, :]
    certifier = ODCertifier(
        scen, sensors, heldout, counts, ho_counts, oracle,
        {"linear_identifiable": True}, CERTIFICATE_DEFAULTS,
    )
    good = certifier.certify(scen.demand.matrix)
    poisoned = certifier.certify(3.0 * scen.demand.matrix)
    assert good["od_feasible"] == 1.0 and poisoned["od_feasible"] == 1.0
    assert float(poisoned["obs_count_rmse"]) > 5.0 * float(good["obs_count_rmse"])


# --- byte-determinism (deterministic=True) -----------------------------------
def test_byte_determinism():
    _, task, _, _ = _build_sf_task()

    def _run():
        tr = ODTrace()
        DtaliteODMEEstimator().estimate(task, Budget(iterations=100), RngBundle(0), tr)
        return tr.final.od_matrix

    a, b = _run(), _run()
    assert np.array_equal(a, b)  # md5-identical od_performance.csv across reruns (adr-042)


# --- budget + crash discipline -----------------------------------------------
def test_sp_calls_only_budget_refused():
    scen = load_or_skip("siouxfalls")
    with pytest.raises(ValueError, match="sp_calls-only"):
        run_estimation_experiment(
            scen, [DtaliteODMEEstimator()], Budget(sp_calls=500),
            seed=_MARQUEE_SEED, macroreps=1, estimation=_MARQUEE,
        )


def test_wall_budget_kill_raises_runtimeerror():
    # A wall_seconds budget exhausted mid-run is an infrastructure outcome; it RAISES
    # (with the engine command), never launders into a bad OD (crash discipline).
    scen = load_or_skip("siouxfalls")
    with pytest.raises(RuntimeError):
        run_estimation_experiment(
            scen, [DtaliteODMEEstimator()], Budget(iterations=100, wall_seconds=1e-6),
            seed=_MARQUEE_SEED, macroreps=1, estimation=_MARQUEE,
        )


# --- delegated envelope refusals (the adapter's envelope IS the estimator's) --
def test_delegated_toll_refusal():
    # A nonzero generalized-cost fixed term is refused by the delegated
    # DTALiteTapModel._refuse_unrepresentable (unvalidated vot conversion), not silently run.
    scen = _two_route_ue(
        [1.0, 1.0, 1.0, 0.5], [0.0, 1.0, 0.0, 2.0], [1, 1, 1, 1],
        toll=[0.0, 2.0, 0.0, 0.0], toll_weight=1.0,
    )
    with pytest.raises(ValueError, match="fixed cost|toll"):
        run_estimation_experiment(
            scen, [DtaliteODMEEstimator()], Budget(iterations=10),
            seed=0, macroreps=1, estimation=_MARQUEE,
        )


def test_delegated_capacity_clamp_refusal():
    # A sub-0.1 capacity link is CLAMPED by the engine in the cost law only, so it would
    # equilibrate under a different BPR -- refused up front by the delegated adapter.
    scen = _two_route_ue([1.0, 1.0, 1.0, 0.5], [0.0, 1.0, 0.0, 2.0], [1, 1, 1, 1],
                         cap=[1.0, 0.05, 1.0, 1.0])
    with pytest.raises(ValueError, match="capacity"):
        run_estimation_experiment(
            scen, [DtaliteODMEEstimator()], Budget(iterations=10),
            seed=0, macroreps=1, estimation=_MARQUEE,
        )


def test_power4_is_accepted_not_refused():
    # Unlike spsa-sumo (marouter's linear vdf refused power!=1), DTALite's VDF is the repo
    # BPR exactly, so the power-4 Sioux Falls anchor is representable -- the delegated
    # refusal must NOT fire on it.
    scen = load_or_skip("siouxfalls")
    probe = Scenario(name=scen.name, network=scen.network, demand=Demand(scen.demand.matrix))
    from tabench.models.adapters.dtalite_tap import DTALiteTapModel

    DTALiteTapModel()._refuse_unrepresentable(probe)  # must not raise
