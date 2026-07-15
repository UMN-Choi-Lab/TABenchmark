"""Tests for the DTALite static-assignment adapter -- ``dtalite-tap`` (adr-029).

``DTALite`` is an OPTIONAL extra; this whole file is skipped on a core install
(``pytest.importorskip('DTALite')``, EXACT case), and the numpy suite runs without it
(the dtalite-free CI/matrix legs are the live regression for that).

What these tests pin, all VERSION-ROBUST (properties and loose ceilings calibrated to
MEASURED values, never tight cross-platform decimals -- the FW/VDF behavior lives in a
bundled ``.so`` that can shift under the ``>=0.8`` floor): the registry/capabilities/
golden-hash invariants; the banner-suppression + core-install guard (``import tabench``
prints NOTHING and the model unregisters when ``DTALite`` is blocked); the Braess
convergence-floor anchor (A1) and the MANDATORY cost-matched anchor separating the exact
BPR representation from FW truncation (A2 -- the engine's ``travel_time`` equals the repo
BPR at the emitted flows on EVERY link, the identity map's payoff); the two-route exact
deterministic UE needing no theta calibration (A3); the Sioux Falls power-4 marquee (A4
-- the first external engine on the power-4 ladder); the honest negative controls
(iterations=1 near-AON is a much worse gap; a converged bfw beats the engine's line-search
floor); byte-determinism; the capability-refusal + sp_calls-only + wall-timeout gates; the
read-back gate; temp-dir hygiene. See docs/design/adr-029-dtalite-tap.md.
"""

import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

pytest.importorskip("DTALite")

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
from tabench.models.adapters import dtalite_tap as dt  # noqa: E402
from tabench.models.adapters.dtalite_tap import DTALiteTapModel  # noqa: E402

# The golden Braess content hash: this additive adapter must leave it -- and thus the
# whole scored instance canon -- byte-identical (HARD RULE).
BRAESS_GOLDEN_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"


def _solve(scenario, budget=None, **factors):
    """Solve ``scenario`` through the adapter, returning (final_state, metrics, model)."""
    model = DTALiteTapModel(**factors)
    trace = Trace()
    model.solve(scenario, budget or Budget(iterations=100), RngBundle(42), trace)
    metrics = Evaluator(scenario).evaluate(trace.final.link_flows)
    return trace.final, metrics, model


def _run_engine_raw(scenario, iterations=100):
    """Explicit GMNS compile + engine run returning (flows, traveltimes) by link index.

    Used by A2 to read the engine's INTERNAL ``travel_time`` column (repo cost units)
    alongside the emitted ``volume``, matched to the repo link order by (from, to)."""
    import shutil

    workdir = tempfile.mkdtemp(prefix="tabench-dtalite-test-")
    try:
        dt._write_gmns(scenario, workdir, iterations)
        cmd = [sys.executable, "-c", "import DTALite; DTALite.assignment()"]
        proc = subprocess.run(
            cmd, cwd=workdir, stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=120, env={**os.environ, "OMP_NUM_THREADS": "1"},
        )
        assert proc.returncode == 0, proc.stderr[-1000:]
        net = scenario.network
        idx = {(int(net.init_node[i]), int(net.term_node[i])): i for i in range(net.n_links)}
        flows = np.full(net.n_links, np.nan)
        times = np.full(net.n_links, np.nan)
        with open(os.path.join(workdir, dt._LP_FILE), newline="") as fh:
            for row in csv.DictReader(fh):
                raw = (row.get("from_node_id") or "").strip()
                if raw in ("", "0"):
                    continue
                key = (int(float(raw)), int(float(row["to_node_id"])))
                if key in idx:
                    flows[idx[key]] = float(row["volume"])
                    times[idx[key]] = float(row["travel_time"])
        return flows, times
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


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
    assert "dtalite-tap" in MODEL_REGISTRY
    caps = DTALiteTapModel.capabilities
    assert caps.paradigm == "heuristic"
    assert caps.deterministic is True
    assert caps.provides_gap is False
    assert caps.seedable is False


def test_golden_braess_hash_unchanged():
    assert braess_scenario().content_hash() == BRAESS_GOLDEN_HASH


# --- banner suppression + core-install guard (unique to this adapter) --------
def test_import_tabench_silent_and_guard_unregisters_when_blocked():
    """`import tabench` must print NOTHING (the DTALite banner would leak on a naive
    module-scope `import DTALite`), and blocking DTALite must unregister the model while
    leaving the numpy core importable -- run in a subprocess with a meta_path blocker."""
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
        "from tabench.models import MODEL_REGISTRY\n"
        "sys.stdout = old\n"
        "assert 'dtalite-tap' not in MODEL_REGISTRY, 'model registered despite blocked DTALite'\n"
        "assert 'aon' in MODEL_REGISTRY, 'numpy core failed to register'\n"
        "assert buf.getvalue() == '', 'import tabench polluted stdout: %r' % buf.getvalue()\n"
        "print('GUARD_OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "GUARD_OK" in proc.stdout


# --- A1: Braess convergence floor --------------------------------------------
def test_braess_certified_convergence_floor():
    final, metrics, _ = _solve(braess_scenario())
    assert metrics["feasible"] == 1.0
    # The engine's Armijo line search stalls to step 0 within a few iterations, so the
    # certified gap freezes at a floor (measured ~1.18e-2). Loose, version-robust
    # ceiling -- the honest engine-as-shipped number, NOT a mapping floor (adr-029).
    assert metrics["relative_gap"] < 5e-2
    assert metrics["node_balance_residual"] < 1e-6
    # Self-report provenance is recorded (executed FW count + the engine's own gap,
    # which uses a DIFFERENT normalization and is never the certified gap).
    assert final.self_report["engine_iterations_executed"] > 0
    assert "engine_relative_gap" in final.self_report


def test_braess_bfw_strictly_better_on_convergence_axis():
    """The honest headline: a converged white-box solver beats the engine's line-search
    floor by orders of magnitude on the convergence axis (adr-029)."""
    _, metrics, _ = _solve(braess_scenario())
    bfw = BiconjugateFrankWolfeModel()
    tr = Trace()
    bfw.solve(braess_scenario(), Budget(iterations=200, target_relative_gap=1e-12),
              RngBundle(0), tr)
    bfw_rg = Evaluator(braess_scenario()).evaluate(tr.final.link_flows)["relative_gap"]
    assert bfw_rg < metrics["relative_gap"] / 10.0


# --- A2: MANDATORY cost-matched anchor (representation vs FW truncation) ------
def test_cost_matched_travel_times():
    """The engine's internal ``travel_time`` must equal the repo BPR cost at the emitted
    flows on EVERY link (the identity map's payoff -- no mapping floor, unlike marouter),
    so the certified gap is pure FW truncation, not a cost-model mismatch. Measured max
    relative error: Braess ~7e-6 (the fft=1e-6/alpha=1e7 sentinel links), Sioux Falls
    power-4 ~2e-5 (the engine's 4-decimal travel_time column); pinned loose."""
    for scenario in (braess_scenario(), load_or_skip("siouxfalls")):
        flows, times = _run_engine_raw(scenario, 100)
        assert np.all(np.isfinite(times))
        repo_cost = scenario.network.link_cost(flows)
        rel = np.abs(times - repo_cost) / np.maximum(np.abs(repo_cost), 1e-30)
        assert rel.max() < 1e-3, rel


# --- A3: two-route exact deterministic UE (no theta calibration) --------------
def test_two_route_exact_deterministic_ue():
    """Unlike marouter (which needed a calibrated logit theta), the engine solves the
    deterministic UE natively and -- on this well-conditioned instance -- exactly."""
    scenario = two_route_scenario(sue_theta=None)
    flows, metrics, _ = _solve(scenario)
    assert metrics["feasible"] == 1.0
    assert metrics["relative_gap"] < 1e-6  # measured 0.0 (exact UE f_A = 2.5)
    assert np.allclose(flows.link_flows, [2.5, 2.5, 1.5, 1.5], atol=1e-3)


# --- A4: Sioux Falls power-4 marquee (marouter-impossible) --------------------
def test_siouxfalls_power4_certified():
    scenario = load_or_skip("siouxfalls")  # BPR power == 4
    final, metrics, _ = _solve(scenario, Budget(iterations=100))
    assert metrics["feasible"] == 1.0
    # The first external engine on the power-4 ladder. Certified RG at the engine's
    # line-search floor (measured ~5.0e-3); loose ceiling.
    assert metrics["relative_gap"] < 5e-2
    # Flow NRMSE vs the best-known UE flows (measured ~1.6%); loose ceiling.
    ref = scenario.reference.link_flows
    nrmse = np.sqrt(np.mean((final.link_flows - ref) ** 2)) / np.mean(ref)
    assert nrmse < 0.1


# --- negative control + monotonicity -----------------------------------------
def test_iterations_one_is_near_aon_worse_gap():
    """iterations=1 runs the pure all-or-nothing load (0 FW line searches): feasible but
    a much worse gap than the converged run -- and RG(1) > RG(100) pins the dial live
    (compared at iteration 1, not an intermediate: the gap freezes from ~iter 40)."""
    _, aon, _ = _solve(braess_scenario(), Budget(iterations=1))
    _, conv, _ = _solve(braess_scenario(), Budget(iterations=100))
    assert aon["feasible"] == 1.0
    assert aon["relative_gap"] > conv["relative_gap"]
    assert aon["relative_gap"] > 0.05  # honestly far from equilibrium


# --- byte-determinism --------------------------------------------------------
def test_byte_determinism():
    f1, _, _ = _solve(braess_scenario())
    f2, _, _ = _solve(braess_scenario())
    assert np.array_equal(f1.link_flows, f2.link_flows)  # FW loop is RNG-free


# --- capability-refusal gates (named field) ----------------------------------
def test_refuse_sue_theta_scenario():
    with pytest.raises(ValueError, match="sue_theta"):
        _solve(two_route_scenario())  # default sue_theta=0.5


def test_refuse_elastic_scenario():
    from tabench import elastic_two_route_scenario

    with pytest.raises(ValueError, match="elastic_demand"):
        _solve(elastic_two_route_scenario())


def test_refuse_toll_scenario():
    tolled = _two_route_like(
        [(1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 1.0), (0.5, 4.0, 1.0)],
        toll=np.array([0.0, 3.0, 0.0, 0.0]), toll_weight=1.0,
    )
    assert np.any(tolled.network.fixed_cost != 0.0)
    with pytest.raises(ValueError, match="fixed cost|toll"):
        _solve(tolled)


# --- sp_calls-only budget is refused (never silently unbounded) ---------------
def test_sp_calls_only_budget_refused():
    with pytest.raises(ValueError, match="sp_calls"):
        _solve(braess_scenario(), budget=Budget(sp_calls=100))


# --- wall timeout is enforced (RuntimeError, not a hang) ----------------------
def test_wall_timeout_raises():
    with pytest.raises(RuntimeError, match="wall_seconds"):
        _solve(braess_scenario(), budget=Budget(wall_seconds=1e-6))


# --- zero-demand short-circuit (no subprocess) -------------------------------
def test_zero_demand_short_circuit():
    scenario = braess_scenario(demand=0.0)
    final, _, model = _solve(scenario)
    assert np.array_equal(final.link_flows, np.zeros(scenario.network.n_links))
    assert not model.last_command  # never spawned a subprocess


# === adversarial-review regressions (adr-029 review) =========================
# link_performance.csv columns the read-back consumes (travel_time is REQUIRED: the
# runtime A2 cost-match reads it).
_LP_HEADER = ["from_node_id", "to_node_id", "volume", "travel_time", "vdf_fftt",
              "vdf_alpha", "vdf_beta", "vdf_plf", "link_capacity"]


def _write_lp(path, net, flows, mutate=None):
    """Write a structurally-valid link_performance.csv (travel_time = the repo BPR at the
    flows so the A2 gate passes, echoes = declared values), then apply ``mutate(rows)``."""
    cost = net.link_cost(np.asarray(flows, dtype=np.float64))
    rows = []
    for i in range(net.n_links):
        rows.append([int(net.init_node[i]), int(net.term_node[i]), float(flows[i]),
                     float(cost[i]), net.free_flow_time[i], net.b[i], net.power[i], 1.0,
                     net.capacity[i]])
    if mutate is not None:
        mutate(rows)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_LP_HEADER)
        w.writerows(rows)


# CRITICAL — ungrouped link.csv corrupts the engine's adjacency: the adapter must ALWAYS
# write links sorted by (from, to), so any input permutation reproduces the same solve.
def test_permuted_links_reproduce_solve():
    import dataclasses

    base = braess_scenario()
    f_base, _, _ = _solve(base)
    net = base.network
    for perm in ([4, 3, 2, 1, 0], [0, 2, 1, 3, 4], [3, 0, 4, 1, 2]):
        p = np.asarray(perm)
        pnet = dataclasses.replace(
            net, init_node=net.init_node[p], term_node=net.term_node[p],
            capacity=net.capacity[p], length=net.length[p],
            free_flow_time=net.free_flow_time[p], b=net.b[p], power=net.power[p],
            toll=net.toll[p], link_type=net.link_type[p],
        )
        psc = Scenario(name="braess-perm", network=pnet, demand=Demand(base.demand.matrix))
        f_perm, m_perm, _ = _solve(psc)
        assert m_perm["feasible"] == 1.0
        # Map the permuted flows back to the base link order and compare (the sort makes
        # the emitted equilibrium permutation-invariant; was RG 0.208 wrong when ungrouped).
        assert np.allclose(f_perm.link_flows, f_base.link_flows[p], atol=1e-6)


# MAJOR — the engine clamps capacity at fmax(0.1, cap) in the cost law only: a capacity in
# (1e-4, 0.1) must be REFUSED (it would equilibrate under a different BPR).
def test_refuse_capacity_clamp():
    clamped = _two_route_like(
        [(1.0, 0.15, 0.05), (1.0, 0.15, 0.05), (10.0, 0.15, 10.0), (10.0, 0.15, 10.0)],
        demand=1.0,
    )
    with pytest.raises(ValueError, match="clamp|capacity"):
        _solve(clamped)


# MAJOR — the ~1e7 BIG-M cost ceiling silently zeroes an OD: the per-origin mass gate must
# RAISE (never launder to feasible=0). Single link cap=100 pow=4, demand past the ceiling.
def test_bigm_demand_drop_raises():
    init = np.array([1], dtype=np.int64)
    term = np.array([2], dtype=np.int64)
    net = Network(
        name="single", n_nodes=2, n_zones=2, first_thru_node=1, init_node=init,
        term_node=term, capacity=np.array([100.0]), length=np.zeros(1),
        free_flow_time=np.ones(1), b=np.array([0.15]), power=np.array([4.0]),
        toll=np.zeros(1), link_type=np.ones(1, dtype=np.int64),
    )
    od = np.zeros((2, 2))
    od[0, 1] = 10000.0
    scenario = Scenario(name="single", network=net, demand=Demand(od))
    with pytest.raises(RuntimeError, match="dropped demand|BIG-M"):
        _solve(scenario, budget=Budget(iterations=5))
    # Just under the ceiling the same network solves cleanly (the gate is not spurious).
    od[0, 1] = 9000.0
    ok = Scenario(name="single", network=net, demand=Demand(od))
    _, metrics, _ = _solve(ok, budget=Budget(iterations=5))
    assert metrics["feasible"] == 1.0


# MAJOR — corrupt-but-parseable engine output must raise the contract RuntimeError, never a
# raw ValueError/TypeError that would collide with the ValueError refusal channel.
def test_readback_rejects_garbage_cells(tmp_path):
    net = braess_scenario().network
    flows = np.full(net.n_links, 1.0)
    cases = {
        "volume": lambda rows: rows[0].__setitem__(2, "abc"),
        "from_node_id": lambda rows: rows[1].__setitem__(0, "xyz"),
        "to_node_id": lambda rows: rows[1].__setitem__(1, "nan"),
        "travel_time": lambda rows: rows[2].__setitem__(3, "abc"),
        "short row": lambda rows: rows[3].__setitem__(slice(None), rows[3][:2]),
    }
    for field, fn in cases.items():
        lp = tmp_path / f"lp_{field.replace(' ', '_')}.csv"
        _write_lp(str(lp), net, flows, mutate=fn)
        with pytest.raises(dt._ReadBackError):
            dt._parse_and_readback(str(lp), net)


# MINOR — the runtime A2 cost-match is the LIVE gate closing the echo check's sub-atol
# blind spot: a well-echoed row whose travel_time does not match the repo BPR is rejected.
def test_readback_a2_cost_match_rejects_wrong_travel_time(tmp_path):
    net = braess_scenario().network
    flows = np.full(net.n_links, 1.0)
    lp = tmp_path / "lp_a2.csv"
    # Double the travel_time on one link (echoes stay declared -> echo check passes).
    _write_lp(
        str(lp), net, flows,
        mutate=lambda rows: rows[0].__setitem__(3, float(rows[0][3]) * 2),
    )
    with pytest.raises(dt._ReadBackError, match="A2 cost-match"):
        dt._parse_and_readback(str(lp), net)


# read-back still rejects a short output (missing link) and an ignored VDF parameter.
def test_readback_rejects_short_output(tmp_path):
    net = braess_scenario().network
    flows = np.full(net.n_links, 1.0)
    lp = tmp_path / "lp_short.csv"
    _write_lp(str(lp), net, flows, mutate=lambda rows: rows.pop())  # drop the last link
    with pytest.raises(dt._ReadBackError):
        dt._parse_and_readback(str(lp), net)


def test_readback_rejects_wrong_echo(tmp_path):
    net = braess_scenario().network
    flows = np.full(net.n_links, 1.0)
    lp = tmp_path / "lp_echo.csv"
    # Corrupt vdf_beta on one link (engine "ignored" power -> default 4).
    _write_lp(str(lp), net, flows, mutate=lambda rows: rows[0].__setitem__(6, 4.0))
    with pytest.raises(dt._ReadBackError, match="vdf_beta"):
        dt._parse_and_readback(str(lp), net)


# MINOR — the wall deadline is enforced through the read-back/parse phase, not just the
# subprocess: a slow parse must RAISE, never silently overrun.
def test_wall_deadline_covers_parse_phase(monkeypatch):
    import time as _time

    real = dt._parse_and_readback

    def slow(lp_path, network):
        _time.sleep(2.5)
        return real(lp_path, network)

    monkeypatch.setattr(dt, "_parse_and_readback", slow)
    with pytest.raises(RuntimeError, match="wall_seconds"):
        _solve(braess_scenario(), budget=Budget(iterations=100, wall_seconds=1.0))


# MINOR — _parse_summary keeps the last COMPLETE (iter, gap) pair under format drift, never
# a torn pair (iter from a later line with a stale gap).
def test_parse_summary_no_torn_pair(tmp_path):
    drift = tmp_path / "summary_log_file.txt"
    drift.write_text("iter No = 3, gap = 2.0 %\niter No = 7, step = 0.0\n")
    assert dt._parse_summary(str(drift)) == (3, 0.02)
    assert dt._parse_summary(str(tmp_path / "nonexistent.txt")) == (0, None)


# --- temp-dir hygiene (no leftovers, including on raise) ----------------------
def test_tempdir_cleanup_including_on_raise(tmp_path, monkeypatch):
    # Point tempfile at a PRIVATE dir so the glob below only sees THIS test's own
    # adapter workdirs -- a concurrent tabench session churning tabench-dtalite-* dirs on
    # a shared /tmp otherwise lands in the before/after window and false-fails (the
    # sumo-test precedent; GitHub runners are single-tenant so CI was never affected).
    import glob

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    pattern = os.path.join(tempfile.gettempdir(), "tabench-dtalite-*")
    before = set(glob.glob(pattern))
    _solve(braess_scenario())  # normal path
    with pytest.raises(RuntimeError):
        _solve(braess_scenario(), budget=Budget(wall_seconds=1e-6))  # raising path
    leftover = set(glob.glob(pattern)) - before
    keep_prefix = os.path.join(tempfile.gettempdir(), "tabench-dtalite-keep-")
    assert not [d for d in leftover if not d.startswith(keep_prefix)]
