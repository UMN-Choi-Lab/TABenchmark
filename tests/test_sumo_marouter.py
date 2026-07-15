"""Tests for the first Phase-4 external-simulator adapter -- ``sumo-marouter``.

``eclipse-sumo`` is an OPTIONAL extra; this whole file is skipped on a core
install (``pytest.importorskip('sumo')``), and the 731-test numpy suite runs
without it (the sumo-free CI/matrix legs are the live regression for that).

What these tests pin, all VERSION-ROBUST (properties and loose ceilings, never
exact split decimals -- the vdf tables are hardcoded upstream and could shift):
the registry/capabilities/golden-hash invariants; the measured Braess mapping
floor (A1); the MANDATORY cost-matched anchor that separates the mapping floor
from solver error (A2 -- marouter's internal traveltimes equal the repo BPR at
the emitted flows on representable links); the asymmetric two-route UE-approx
direction with bfw strictly better (A3); byte-determinism + the seed reaching the
command line; the honest negative controls (incremental is a much worse gap; UE /
gawron / lohse are refused); the capability-refusal gates (sue/elastic/power-4);
the zero-demand short-circuit; the sp_calls-only budget refusal; the wall-timeout
kill (RuntimeError, never a hang); temp-dir hygiene; multi-OD mass conservation;
and the mapping-floor closed forms. See docs/design/adr-027-sumo-marouter.md.
"""

import glob
import os
import subprocess
import tempfile

import numpy as np
import pytest

pytest.importorskip("sumo")

from conftest import load_or_skip  # noqa: E402

from tabench import (  # noqa: E402
    BiconjugateFrankWolfeModel,
    Budget,
    Demand,
    Evaluator,
    Network,
    RngBundle,
    Scenario,
    Trace,
    braess_scenario,
    two_route_scenario,
)
from tabench.models import MODEL_REGISTRY  # noqa: E402
from tabench.models.adapters import _sumo_io as sio  # noqa: E402
from tabench.models.adapters.sumo_marouter import SumoMarouterModel  # noqa: E402

# The golden Braess content hash: this additive adapter must leave it -- and thus
# the whole scored instance canon -- byte-identical (HARD RULE).
BRAESS_GOLDEN_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"


def _solve(scenario, budget=None, **factors):
    """Solve ``scenario`` through the adapter, returning (flows, metrics, model)."""
    model = SumoMarouterModel(**factors)
    trace = Trace()
    model.solve(scenario, budget or Budget(iterations=50), RngBundle(42), trace)
    metrics = Evaluator(scenario).evaluate(trace.final.link_flows)
    return trace.final.link_flows, metrics, model


def _run_marouter(scenario, theta, paths, method="SUE"):
    """Explicit converter + marouter run returning (flows, traveltimes, build).

    Used by A2 to read marouter's INTERNAL traveltime attribute (seconds =
    time_scale * native cost) alongside the emitted flows."""
    import shutil

    workdir = tempfile.mkdtemp(prefix="tabench-sumo-test-")
    try:
        build = sio.scenario_to_sumo(scenario, workdir, time_scale=1.0, min_lanes=1)
        netload = os.path.join(workdir, "nl.xml")
        cmd = [
            sio.sumo_binary("marouter"),
            "--net-file", build.net_file, "--additional-files", build.taz_file,
            "--od-matrix-files", build.od_file, "--netload-output", netload,
            "--output-file", os.devnull, "--precision", "9",
            "--weights.minor-penalty", "0", "--left-turn-penalty", "0",
            "--capacities.default", "true", "--assignment-method", method,
            "--route-choice-method", "logit", "--logit.beta", "0", "--logit.gamma", "0",
            "--logit.theta", repr(float(theta)), "--paths", str(paths),
            "--max-iterations", "50", "--max-inner-iterations", "5000",
            "--tolerance", "1e-7", "--seed", "42", "--routing-threads", "1",
        ]
        proc = subprocess.run(cmd, env=sio.sumo_env(), capture_output=True, text=True, cwd=workdir)
        assert proc.returncode == 0, proc.stderr[-1000:]
        n = scenario.network.n_links
        flows = sio.parse_netload(netload, n, build.flow_scale)
        tt = sio.parse_netload_attr(netload, n, "traveltime")
        return flows, tt, build
    finally:  # keep the workdir off the runner even on the failure paths CI debugs
        shutil.rmtree(workdir, ignore_errors=True)


def _star_multi_od():
    """A 3-zone star network with linear (power=1) costs and multiple OD pairs."""
    # zones 1,2,3 spoke into hub node 4; all links t(v) = 5 + v (normal links).
    init = np.array([1, 2, 3, 4, 4, 4], dtype=np.int64)
    term = np.array([4, 4, 4, 1, 2, 3], dtype=np.int64)
    n = len(init)
    network = Network(
        name="star", n_nodes=4, n_zones=3, first_thru_node=1,
        init_node=init, term_node=term,
        capacity=np.ones(n), length=np.zeros(n),
        free_flow_time=np.full(n, 5.0), b=np.ones(n), power=np.ones(n),
        toll=np.zeros(n), link_type=np.ones(n, dtype=np.int64),
    )
    od = np.zeros((3, 3))
    od[0, 1] = 3.0  # 1 -> 2
    od[0, 2] = 2.0  # 1 -> 3
    od[2, 1] = 1.5  # 3 -> 2
    return Scenario(name="star", network=network, demand=Demand(od), family="test-star")


def _two_route_like(params, demand=4.0, toll=None, toll_weight=0.0):
    """A 2-route net (links 1->3, 3->2, 1->4, 4->2) with per-link (fft, b, cap)."""
    init = np.array([1, 3, 1, 4], dtype=np.int64)
    term = np.array([3, 2, 4, 2], dtype=np.int64)
    fft = np.array([p[0] for p in params])
    b = np.array([p[1] for p in params])
    cap = np.array([p[2] for p in params])
    network = Network(
        name="tr", n_nodes=4, n_zones=2, first_thru_node=1, init_node=init, term_node=term,
        capacity=cap, length=np.zeros(4), free_flow_time=fft, b=b, power=np.ones(4),
        toll=np.zeros(4) if toll is None else toll, link_type=np.ones(4, dtype=np.int64),
        toll_weight=toll_weight,
    )
    od = np.zeros((2, 2))
    od[0, 1] = demand
    return Scenario(name="tr", network=network, demand=Demand(od))


# --- registry / capabilities / golden hash -----------------------------------
def test_registered_and_capabilities():
    assert "sumo-marouter" in MODEL_REGISTRY
    caps = SumoMarouterModel.capabilities
    assert caps.paradigm == "heuristic"
    assert caps.deterministic is True
    assert caps.provides_gap is False
    assert caps.seedable is True


def test_golden_braess_hash_unchanged():
    assert braess_scenario().content_hash() == BRAESS_GOLDEN_HASH


# --- A1: the measured Braess mapping floor -----------------------------------
def test_braess_certified_mapping_floor():
    flows, metrics, _ = _solve(braess_scenario())
    assert metrics["feasible"] == 1.0
    # Certified gap is the analytic mapping floor (measured 1.727e-4 at s=14000);
    # the adapter's s gives ~1.74e-4. Loose, version-robust ceiling.
    assert metrics["relative_gap"] < 3e-4
    assert metrics["node_balance_residual"] < 1e-6
    # Flows match the analytic oracle (4,2,2,2,4) to the perturbation floor.
    assert np.allclose(flows, [4.0, 2.0, 2.0, 2.0, 4.0], atol=0.01)


# --- A2: MANDATORY cost-matched anchor (mapping floor vs solver error) --------
def test_cost_matched_internal_traveltimes():
    """marouter's internal traveltime must equal the repo BPR cost at the emitted
    flows on REPRESENTABLE links, so the certified gap is interpretable: any
    residual is the mapping floor (the eps/parasitic perturbations on the
    non-representable links), not a cost-model mismatch on the matched links."""
    for scenario, theta, paths in (
        (braess_scenario(), 200.0, 4),
        (two_route_scenario(sue_theta=None), 200.0, 2),
    ):
        flows, tt, build = _run_marouter(scenario, theta, paths)
        rep = build.representable
        assert rep.any()
        repo_cost = scenario.network.link_cost(flows)
        # traveltime is seconds = time_scale * native cost; time_scale == 1 here.
        diff = np.abs(tt[rep] / build.time_scale - repo_cost[rep])
        assert diff.max() < 1e-6, diff


# --- A3: asymmetric two-route UE-approx, bfw strictly better ------------------
def test_two_route_ue_approx_direction():
    scenario = two_route_scenario(sue_theta=None)
    flows, metrics, _ = _solve(scenario)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-3  # UE-approx, calibrated on THIS anchor
    # A converged white-box solver certifies an orders-better gap (the honest
    # headline: feasibility is not equilibrium quality; adr-027).
    bfw = BiconjugateFrankWolfeModel()
    tr = Trace()
    bfw.solve(scenario, Budget(iterations=200, target_relative_gap=1e-10), RngBundle(0), tr)
    bfw_rg = Evaluator(scenario).evaluate(tr.final.link_flows)["relative_gap"]
    assert bfw_rg < metrics["relative_gap"] / 10.0


# --- determinism + seed on the command line ----------------------------------
def test_byte_determinism_and_seed_on_command_line():
    scenario = braess_scenario()
    f1, _, m1 = _solve(scenario)
    f2, _, m2 = _solve(scenario)
    assert np.array_equal(f1, f2)  # byte-identical (marouter SUE path is RNG-free)
    # The seed is drawn from the RngBundle and reaches the command line.
    assert "--seed" in m1.last_command
    seed_value = m1.last_command[m1.last_command.index("--seed") + 1]
    assert int(seed_value) >= 1
    assert "--routing-threads" in m1.last_command  # pinned single-threaded


# --- negative controls -------------------------------------------------------
def test_incremental_is_a_much_worse_gap():
    _, metrics, _ = _solve(braess_scenario(), assignment_method="incremental")
    assert metrics["feasible"] == 1.0  # honestly feasible but NOT an equilibrium
    assert metrics["relative_gap"] > 0.05


def test_ue_method_refused():
    with pytest.raises(ValueError, match="UE"):
        _solve(braess_scenario(), assignment_method="UE")


@pytest.mark.parametrize("choice", ["gawron", "lohse"])
def test_nonlogit_route_choice_refused(choice):
    with pytest.raises(ValueError, match="logit"):
        _solve(braess_scenario(), route_choice=choice)


# --- capability-refusal gates (named field / power) --------------------------
def test_refuse_sue_theta_scenario():
    with pytest.raises(ValueError, match="sue_theta"):
        _solve(two_route_scenario())  # default sue_theta=0.5


def test_refuse_elastic_scenario():
    from tabench import elastic_two_route_scenario

    with pytest.raises(ValueError, match="elastic_demand"):
        _solve(elastic_two_route_scenario())


def test_refuse_power4_siouxfalls():
    scenario = load_or_skip("siouxfalls")  # BPR power == 4
    with pytest.raises(ValueError, match="power"):
        _solve(scenario)


# --- zero-demand short-circuit -----------------------------------------------
def test_zero_demand_short_circuit():
    scenario = braess_scenario(demand=0.0)
    flows, metrics, model = _solve(scenario)
    assert np.array_equal(flows, np.zeros(scenario.network.n_links))
    assert not model.last_command  # never spawned a subprocess


# --- sp_calls-only budget is refused (never silently unbounded) ---------------
def test_sp_calls_only_budget_refused():
    with pytest.raises(ValueError, match="sp_calls"):
        _solve(braess_scenario(), budget=Budget(sp_calls=100))


# --- wall timeout is enforced (RuntimeError, not a hang) ---------------------
def test_wall_timeout_raises():
    with pytest.raises(RuntimeError, match="wall_seconds|budget"):
        _solve(braess_scenario(), budget=Budget(wall_seconds=0.001))


# --- temp-dir hygiene (no leftovers, including on raise) ----------------------
def test_tempdir_cleanup_including_on_raise(tmp_path, monkeypatch):
    # Point tempfile at a PRIVATE dir so the glob below only ever sees THIS
    # test's own adapter workdirs -- a concurrent tabench session churning its
    # own tabench-sumo-* dirs on a shared /tmp otherwise lands inside the
    # before/after window and false-fails this test (observed under parallel
    # local runs; GitHub runners are single-tenant so CI was never affected).
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    pattern = os.path.join(tempfile.gettempdir(), "tabench-sumo-*")
    before = set(glob.glob(pattern))
    _solve(braess_scenario())  # normal path
    with pytest.raises(RuntimeError):
        _solve(braess_scenario(), budget=Budget(wall_seconds=0.001))  # raising path
    leftover = set(glob.glob(pattern)) - before
    # keep_files defaults False, so neither path may leave a working directory.
    keep_prefix = os.path.join(tempfile.gettempdir(), "tabench-sumo-keep-")
    assert not [d for d in leftover if not d.startswith(keep_prefix)]


# --- multi-OD mass conservation ----------------------------------------------
def test_multi_od_mass_conservation():
    scenario = _star_multi_od()
    _, metrics, _ = _solve(scenario)
    assert metrics["feasible"] == 1.0  # per-node demand conservation is exact
    assert metrics["node_balance_residual"] < 1e-6


# --- mapping-floor closed forms ----------------------------------------------
def test_mapping_floor_closed_forms():
    workdir = tempfile.mkdtemp(prefix="tabench-sumo-test-")
    try:
        # Braess: the zero-intercept links (B=10) get a forced intercept
        # eps = B * (200/6) / s; no zero-slope links -> parasitic 0.
        b = sio.scenario_to_sumo(braess_scenario(), workdir, time_scale=1.0, min_lanes=1)
        expected_eps = 10.0 * (sio._ZI_CAP_PER_LANE / sio._ZI_K) / b.flow_scale
        assert abs(b.forced_intercept - expected_eps) < 1e-12
        assert b.parasitic_slope == 0.0
    finally:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)

    workdir = tempfile.mkdtemp(prefix="tabench-sumo-test-")
    try:
        # two-route: the constant first legs (A=1, B=0) get a parasitic slope
        # A*K*s/(1400*lanes); no zero-intercept links -> forced intercept 0.
        b = sio.scenario_to_sumo(
            two_route_scenario(sue_theta=None), workdir, time_scale=1.0, min_lanes=1
        )
        assert b.forced_intercept == 0.0
        assert 0.0 < b.parasitic_slope <= sio._PARASITIC_SLOPE_TOL + 1e-12
    finally:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)


# === adversarial-review regressions (adr-027 review) =========================
def _compiled_edges(net_file):
    """{edge_id: (numLanes, length, speed)} from a compiled net.net.xml."""
    import xml.etree.ElementTree as ET

    out = {}
    for edge in ET.parse(net_file).getroot().findall("edge"):
        eid = edge.get("id", "")
        if eid.startswith(":"):
            continue
        lanes = edge.findall("lane")
        out[eid] = (len(lanes), float(lanes[0].get("length")), float(lanes[0].get("speed")))
    return out


# CRITICAL — netconvert's 0.1 m min-length clamp: read-back + eps-edge lane choice
def test_compile_readback_declared_equals_compiled():
    import shutil

    for scenario in (braess_scenario(), two_route_scenario(sue_theta=None)):
        workdir = tempfile.mkdtemp(prefix="tabench-sumo-test-")
        try:
            # scenario_to_sumo runs the read-back internally; also assert the
            # compiled net matches the declared edge specs to tolerance here.
            build = sio.scenario_to_sumo(scenario, workdir, time_scale=1.0, min_lanes=1)
            s, kind, coeff = sio._flow_scale(scenario.network, 1)
            specs, *_ = sio._edge_specs(scenario.network, s, kind, coeff, 1.0)
            compiled = _compiled_edges(build.net_file)
            for i, spec in enumerate(specs):
                n_lanes, length, speed = compiled[f"e{i}"]
                assert n_lanes == spec.lanes
                assert abs(length - spec.length) <= 1e-3 * max(spec.length, 1e-9)
                # every eps-edge clears the 0.1 m clamp (no silent corruption)
                assert length >= sio._MIN_EDGE_LENGTH_M
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


def test_would_clamp_scenario_compiles_or_refuses_never_silently_corrupts():
    # A zero-intercept link whose lanes=1 length would fall below 0.1 m: the eps-
    # edge lane choice must clear the clamp (read-back passes) or the compile must
    # refuse -- never a silently 3x-scaled cost. This one clears (lanes bumped).
    scenario = _two_route_like(
        [(1e-6, 1.0 / 1e-6, 1.0), (2.0, 0.5, 1.0), (1.0, 0.0, 1.0), (0.5, 4.0, 1.0)]
    )
    flows, metrics, _ = _solve(scenario)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-2  # a real small gap, not a clamped 0.3+


# MAJOR — hardcoded 1-hour OD window: high demand must not collapse to free-flow
def test_high_demand_window_no_aon_collapse():
    # Braess D=360's AON entry time hits the 3600 s window exactly; the sized
    # window keeps the assignment an equilibrium (was RG 0.52 pure AON).
    for demand in (350.0, 360.0, 1000.0):
        _, metrics, _ = _solve(braess_scenario(demand=demand))
        assert metrics["feasible"] == 1.0
        assert metrics["relative_gap"] < 0.05  # not the ~0.52 AON collapse


# MAJOR — tolls / generalized-cost fixed terms are refused, not silently dropped
def test_toll_refused():
    tolled = _two_route_like(
        [(1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 1.0), (0.5, 4.0, 1.0)],
        toll=np.array([0.0, 3.0, 0.0, 0.0]), toll_weight=1.0,
    )
    assert np.any(tolled.network.fixed_cost != 0.0)  # the trap the reviewer set
    with pytest.raises(ValueError, match="fixed cost|toll"):
        _solve(tolled)


# MAJOR — lane explosion: generic decimal parameters refuse FAST, never hang
def test_lane_explosion_refused_fast():
    import time as _time

    explode = _two_route_like(
        [(10.0, 1.0, 4833.61), (2.0, 1.0, 1.0), (1.0, 0.0, 1.0), (0.5, 2.0, 1.0)]
    )
    t0 = _time.perf_counter()
    with pytest.raises(ValueError, match="lane|rational"):
        _solve(explode)
    assert _time.perf_counter() - t0 < 5.0  # refused up front, not a netconvert hang


# MAJOR — wall budget must bound the compile phase too
def test_wall_budget_covers_compile_phase():
    # A tiny wall budget must raise even though the (netconvert) compile phase
    # runs before marouter -- previously it was silently unbounded.
    with pytest.raises(RuntimeError, match="wall_seconds|budget"):
        _solve(braess_scenario(), budget=Budget(wall_seconds=0.001))


# MINOR — time_scale bounds narrowed to the validated envelope
def test_time_scale_out_of_envelope_refused():
    with pytest.raises(ValueError, match="time_scale"):
        SumoMarouterModel(time_scale=1e-6)
    with pytest.raises(ValueError, match="time_scale"):
        SumoMarouterModel(time_scale=1e6)
    SumoMarouterModel(time_scale=1.0)  # in-envelope: constructs fine


# MINOR — zero-slope parasitic slope past _MAX_LANES is refused, not capped
def test_parasitic_slope_cap_refused():
    parasitic = _two_route_like(
        [(1e6, 0.0, 1.0), (2.0, 1.0, 1.0), (1.0, 0.0, 1.0), (0.5, 2.0, 1.0)]
    )
    with pytest.raises(ValueError, match="zero-slope|parasitic|_MAX_LANES"):
        _solve(parasitic)


# MINOR — Budget(iterations=0) is floored to 1 (no all-zero censor row)
def test_iterations_zero_floored_to_one():
    _, metrics, model = _solve(braess_scenario(), budget=Budget(iterations=0))
    assert metrics["feasible"] == 1.0  # not the censored all-zero emission
    assert model.last_command[model.last_command.index("--max-iterations") + 1] == "1"


# MINOR — edge-less netload (engine wrote nothing) raises, not a censor row
def test_netload_helpers_detect_empty_and_multi_interval(tmp_path):
    empty = tmp_path / "empty.xml"
    empty.write_text('<meandata><interval begin="0" end="3600"></interval></meandata>')
    assert sio.netload_matched_edge_count(str(empty), 5) == 0
    assert sio.netload_interval_count(str(empty)) == 1
    multi = tmp_path / "multi.xml"
    multi.write_text(
        '<meandata><interval><edge id="e0" entered="5"/></interval>'
        '<interval><edge id="e0" entered="7"/></interval></meandata>'
    )
    assert sio.netload_interval_count(str(multi)) == 2
    assert sio.netload_matched_edge_count(str(multi), 5) == 1


# NOTE — a missing/undiscoverable binary surfaces as the contract RuntimeError
def test_missing_binary_wrapped_as_runtimeerror(monkeypatch):
    import sumo

    monkeypatch.setattr(sumo, "SUMO_HOME", "/nonexistent-sumo-home-xyz")
    with pytest.raises(RuntimeError, match="could not be executed|netconvert"):
        _solve(braess_scenario(), budget=Budget(iterations=5))
