"""Tests for the BO4Mob T2 OD-estimation family (adr-041).

Two layers, mirroring tests/test_bo4mob.py:

* **Engine-free** (no sumo, no live network): the task/registry/certifier contract
  — content-hash sensitivity, the prior baseline's unconditional sumo-free
  registration, the certifier's censor/raise control flow via an INJECTED counts
  runner (the edoc ReplayRunner injection precedent), the ``fill_od_from_vector``
  od_end_time regression (the demand-loss laundering vector), the held-out registry
  integrity + HPC refusal, P7 held-out non-leakage, the golden Braess hash, and the
  dual-benchmark honesty disclosures + a forbidden-clause tripwire.
* **Pipeline liveness** (``importorskip('sumo')``, the sumo CI job): 1ramp full
  estimator+certify reproducing the anchor NRMSE exactly and the od_end_time
  regression pair live; 2corridor/3junction single-certify liveness; 4smallRegion
  behind an opt-in slow gate (its meso alone is ~126 s); an executed crash
  simulation raising (never censoring).

CI test scope (adr-041 ruling 8; measured certify walls 1ramp 0.44 s, 2corridor
9.4 s, 3junction 13.6 s, 4smallRegion ~129 s): the sumo job runs 1ramp + 2corridor
+ 3junction (~24 s of meso, inside the 2-4 min sumo-job budget); 4smallRegion is
registered but gated behind TABENCH_RUN_SLOW_BO4MOB so CI never pays its ~129 s.
"""

from __future__ import annotations

import csv
import dataclasses
import inspect
import json
import os
import re

import numpy as np
import pytest

import tabench.data.bo4mob as bo
import tabench.estimation.bo4mob_base as bb
import tabench.metrics.estimation_bo4mob as eb
from tabench import braess_scenario
from tabench.core.budget import Budget
from tabench.core.rng import RngBundle
from tabench.data import REGISTRY
from tabench.data.bo4mob import (
    BO4MOB_ENGINE_VERSION,
    BO4MOB_HELDOUT,
    BO4MOB_HELDOUT_DATES,
    BO4MOB_HELDOUT_HOUR,
    BO4MOB_REGISTRY,
    Bo4MobHpcOnlyError,
    ChecksumError,
    bo4mob_pairs,
    bo4mob_prior_vector,
    fetch_bo4mob,
    fetch_bo4mob_heldout,
    fill_od_from_vector,
    fill_single_od,
)
from tabench.estimation import BO4MOB_ESTIMATOR_REGISTRY, Bo4MobPriorBaseline
from tabench.estimation.base import ODTrace
from tabench.estimation.bo4mob_base import Bo4MobEstimationTask
from tabench.experiments.runner import run_bo4mob_estimation_experiment
from tabench.metrics.estimation_bo4mob import BO4MOB_METRIC_KEYS, Bo4MobODCertifier
from tabench.observe.levels import Dataset

# Golden hash: adding a whole T2 family must not move any existing hash.
GOLDEN_BRAESS_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"

# Bit-stable pilot anchors (adr-041 pilot, eclipse-sumo 1.27.1; measured on this box).
ANCHOR_NRMSE = 2.432471221214843          # 1ramp prior OD, od_end_time=3300 (correct)
UNFIXED_NRMSE = 2.3147038842862218        # the pre-fix bug: interval end left at 3600
PILOT_HELDOUT_MEAN = 1.697988             # mean over the 13 held-out 06-07 dates

SMALL = ("1ramp", "2corridor", "3junction", "4smallRegion")


# --- helpers -------------------------------------------------------------------


def _write_sensor(path, rows):
    """Write a minimal PeMS-schema GT CSV (``link_id, interval_nVehContrib, ...``)."""
    lines = ["link_id,interval_nVehContrib,interval_harmonicMeanSpeed"]
    lines += [f"{lid},{cnt},60.0" for lid, cnt in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def _make_task(**overrides):
    """A synthetic (engine-free) Bo4MobEstimationTask with train date 221008."""
    ds = Dataset(
        kind="bo4mob_train_counts",
        payload={"link_ids": ("L0", "L1", "L2"), "counts": np.array([465.0, 840.0, 816.0])},
        meta={"instance": "1ramp", "date": "221008", "hour": "06-07"},
    )
    base = dict(
        name="1ramp", instance_key="1ramp",
        pairs=(("taz_0", "taz_1"), ("taz_0", "taz_49"), ("taz_49", "taz_1")),
        prior_vector=np.array([2092.0, 609.0, 386.0]), dataset=ds,
        identifiability={"n_active_pairs": 3}, engine={"name": "eclipse-sumo", "version": "1.27.1"},
        certificate={"od_end_time": 3300}, seed=0, heldout_digest="deadbeef",
    )
    base.update(overrides)
    return Bo4MobEstimationTask(**base)


def _make_certifier(tmp_path, runner=None, installed="1.27.1", paths=None):
    """An engine-free certifier over tiny fake train/held-out CSVs."""
    train = _write_sensor(tmp_path / "train.csv", [("L0", 100.0)])
    ho = {
        "221009": _write_sensor(tmp_path / "ho9.csv", [("L0", 50.0)]),
        "221010": _write_sensor(tmp_path / "ho10.csv", [("L0", 200.0)]),
    }
    return Bo4MobODCertifier(
        instance_key="1ramp", pairs=(("a", "b"), ("c", "d"), ("e", "f")),
        train_sensor=train, heldout_sensors=ho, paths=paths,
        od_end_time=3300, sim_end_time=3600.0, sensor_start_time=0.0, sensor_end_time=3600.0,
        engine_version="1.27.1", certificate={"seed": 0},
        installed_version=installed, runner=runner,
    )


def _reachable_numbers(obj, seen=None):
    """Every numeric value reachable from ``obj`` (arrays, scalars, dict/list, dataclass
    fields, and numeric STRINGS), as a set of floats. Used for the P7 count-leak check:
    substring matching is unsafe (a 3-digit held-out count is almost surely a substring
    of the 64-char hex ``heldout_digest``/``content_hash``), so leakage is checked by
    numeric VALUE membership instead."""
    nums: set[float] = set()

    def walk(o):
        if isinstance(o, bool):
            return
        if isinstance(o, np.ndarray):
            for x in o.ravel().tolist():
                nums.add(float(x))
        elif isinstance(o, (int, float)):
            nums.add(float(o))
        elif isinstance(o, str):
            try:
                nums.add(float(o))
            except ValueError:
                pass
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(k)
                walk(v)
        elif isinstance(o, (list, tuple, set)):
            for v in o:
                walk(v)
        elif dataclasses.is_dataclass(o):
            for f in dataclasses.fields(o):
                walk(getattr(o, f.name))

    walk(obj)
    return nums


def _build_real_bo4mob_task(instance):
    """Build the REAL Bo4MobEstimationTask for ``instance`` (mirrors the runner's
    construction) from the fetched bundle — engine-free, so the P7 leakage check runs
    against the actual shipped task, not a synthetic stand-in. Returns (task, cfg)."""
    paths = fetch_bo4mob(BO4MOB_REGISTRY[instance])
    cfg = json.loads(paths["config"].read_text())
    pairs = bo4mob_pairs(paths["od"])
    prior = bo4mob_prior_vector(paths["od"], paths["single_od"])
    ids, counts = [], []
    with open(paths["sensor"], newline="") as f:
        for row in csv.DictReader(f):
            ids.append(row["link_id"])
            counts.append(float(row["interval_nVehContrib"]))
    ds = Dataset(
        kind="bo4mob_train_counts",
        payload={"link_ids": tuple(ids), "counts": np.asarray(counts)},
        meta={"instance": instance, "date": "221008", "hour": "06-07"},
    )
    task = Bo4MobEstimationTask(
        name=instance, instance_key=instance, pairs=pairs, prior_vector=prior, dataset=ds,
        identifiability={
            "n_active_pairs": len(pairs), "n_train_sensors": len(ids),
            "sensor_pair_coverage": len(ids) / len(pairs),
        },
        engine={"name": "eclipse-sumo", "version": "1.27.1"},
        certificate={
            "od_end_time": int(cfg["od_end_time"]), "sim_end_time": float(cfg["sim_end_time"]),
            "sensor_start_time": float(cfg["sensor_start_time"]),
            "sensor_end_time": float(cfg["sensor_end_time"]),
            "wall_deadline_seconds": 300.0, "seed": 0,
        },
        seed=0, heldout_digest="0" * 64,
    )
    return task, cfg, paths


# --- Engine-free: task + registry ----------------------------------------------


def test_content_hash_is_domain_prefixed_and_sensitive():
    t = _make_task()
    h = t.content_hash()
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)
    # Each independent dial must move the hash (adr-023 lesson: different OD cells
    # must never hash equal; the prior/train/heldout/seed/engine each matter).
    reordered = (("taz_0", "taz_49"), ("taz_0", "taz_1"), ("taz_49", "taz_1"))
    assert dataclasses.replace(t, pairs=reordered).content_hash() != h
    assert dataclasses.replace(t, prior_vector=np.array([2093.0, 609.0, 386.0])).content_hash() != h
    assert dataclasses.replace(t, heldout_digest="cafef00d").content_hash() != h
    assert dataclasses.replace(t, seed=1).content_hash() != h
    assert dataclasses.replace(
        t, engine={"name": "eclipse-sumo", "version": "1.28.0"}
    ).content_hash() != h
    # TRAIN count bytes move the hash (a true instance pin, not a shape pin).
    ds2 = dataclasses.replace(
        t.dataset,
        payload={"link_ids": ("L0", "L1", "L2"), "counts": np.array([466.0, 840.0, 816.0])},
    )
    assert dataclasses.replace(t, dataset=ds2).content_hash() != h


def test_content_hash_domain_prefix_distinct_from_siblings():
    # The domain prefix guarantees a BO4Mob task never collides with a static/dynamic
    # one even on coincidentally-equal bytes.
    t = _make_task()
    assert t.content_hash() != _make_task(instance_key="2corridor").content_hash()


def test_prior_baseline_registers_unconditionally_and_imports_no_sumo():
    assert "bo4mob-prior" in BO4MOB_ESTIMATOR_REGISTRY
    assert BO4MOB_ESTIMATOR_REGISTRY["bo4mob-prior"] is Bo4MobPriorBaseline
    # No EXECUTABLE sumo import anywhere in the module (prose mentions are fine): the
    # baseline registers even where the wheel is absent (unlike the spsa-sumo guard).
    src = inspect.getsource(bb)
    assert not re.search(r"(?m)^\s*(import sumo|from sumo)", src)


def test_prior_baseline_emits_prior_vector_unchanged():
    t = _make_task()
    trace = ODTrace()
    bundle = Bo4MobPriorBaseline().estimate(t, Budget(sp_calls=1), RngBundle(0), trace)
    assert bundle.estimator_name == "bo4mob-prior"
    assert np.array_equal(bundle.final.od_matrix, t.prior_vector)
    assert len(trace) == 1


def test_certifier_module_imports_sumo_lazily_only():
    # estimation_bo4mob must import on a core (sumo-free) install: the only sumo import
    # is INSIDE _run_pipeline (indented), never at module top.
    src = inspect.getsource(eb)
    assert not re.search(r"(?m)^(import sumo|from sumo)", src)
    assert re.search(r"(?m)^\s+import sumo", src)  # lazy, indented


# --- Engine-free: certifier control flow (injected runner) ---------------------


def test_certify_scores_via_injected_runner(tmp_path):
    cert = _make_certifier(tmp_path, runner=lambda q: {"L0": 100.0})
    m = cert.certify(np.array([1.0, 2.0, 3.0]))
    assert set(m) == set(BO4MOB_METRIC_KEYS)
    assert m["od_feasible"] == 1.0
    assert m["obs_nrmse"] == pytest.approx(0.0)          # sim 100 == train 100
    # held-out gt 50 -> nrmse 1.0; gt 200 -> nrmse (200-100)/200*sqrt(1)=0.5; mean 0.75
    assert m["heldout_nrmse"] == pytest.approx(0.75)
    assert m["heldout_nrmse_min"] == pytest.approx(0.5)
    assert m["heldout_nrmse_max"] == pytest.approx(1.0)
    assert m["n_heldout_dates"] == 2.0


def test_certify_censors_nonfinite_and_negative(tmp_path):
    cert = _make_certifier(tmp_path, runner=lambda q: {"L0": 100.0})
    bads = (
        np.array([np.inf, 1.0, 1.0]),
        np.array([-5.0, 1.0, 1.0]),
        np.array([np.nan, 1.0, 1.0]),
    )
    for bad in bads:
        m = cert.certify(bad)
        assert m["od_feasible"] == 0.0
        assert np.isnan(m["heldout_nrmse"])
        assert m["n_heldout_dates"] == 2.0  # constant even when censored


def test_certify_wrong_shape_raises(tmp_path):
    cert = _make_certifier(tmp_path, runner=lambda q: {"L0": 100.0})
    with pytest.raises(ValueError, match="shape"):
        cert.certify(np.array([1.0, 2.0]))  # 2 != 3 pairs — a wrapper bug, not a censor


def test_certify_engine_pin_mismatch_raises_never_censors(tmp_path):
    # A drifted engine RAISES (config error), never silently scores (adr-041 ruling 6).
    cert = _make_certifier(tmp_path, runner=lambda q: {"L0": 100.0}, installed="9.9.9")
    with pytest.raises(ValueError, match="engine pin"):
        cert.certify(np.array([1.0, 2.0, 3.0]))


def test_certify_engine_crash_propagates_not_censored(tmp_path):
    # A crash in the certifier's OWN pipeline is infrastructure -> RuntimeError
    # propagates; it is NEVER laundered into od_feasible=0 (adr-041 ruling 7).
    def crashing(_q):
        raise RuntimeError("simulated meso crash")

    cert = _make_certifier(tmp_path, runner=crashing)
    with pytest.raises(RuntimeError, match="simulated meso crash"):
        cert.certify(np.array([1.0, 2.0, 3.0]))


def test_certify_zero_od_short_circuits_finite_not_censored(tmp_path):
    # A zero OD is a legitimate, terrible estimate: no engine call, empty sim ->
    # catastrophic-but-FINITE NRMSE, od_feasible=1 (not censored). runner=None so
    # the zero-demand fast path is exercised without sumo.
    cert = _make_certifier(tmp_path, runner=None, paths=None)
    m = cert.certify(np.zeros(3))
    assert m["od_feasible"] == 1.0
    assert np.isfinite(m["heldout_nrmse"]) and m["heldout_nrmse"] > 0
    # train gt 100, sim 0 -> nrmse sqrt(100^2)/100 = 1.0
    assert m["obs_nrmse"] == pytest.approx(1.0)


def test_certify_no_paths_no_runner_raises_on_positive_od(tmp_path):
    cert = _make_certifier(tmp_path, runner=None, paths=None)
    with pytest.raises(RuntimeError, match="no instance paths and no injected runner"):
        cert.certify(np.array([1.0, 2.0, 3.0]))


# --- Engine-free: fill_od_from_vector regression (the laundering vector) --------


def test_fill_od_from_vector_rewrites_interval_end_to_od_end_time(tmp_path):
    """The load-bearing anti-laundering control (adr-034 Decision 3, adr-041 ruling 5):
    the vector fill MUST rewrite the interval end to od_end_time regardless of the
    template end (keeping 3600 on a 3300-window instance leaks ~5% of demand)."""
    pairs = (("taz_0", "taz_1"),)
    for template_end in ("3600", "9999"):
        template = tmp_path / f"tmpl_{template_end}.xml"
        template.write_text(
            '<data><interval id="DEFAULT_VEHTYPE" begin="0" end="' + template_end + '">'
            '<tazRelation from="taz_0" to="taz_1" count="0"/></interval></data>'
        )
        out = tmp_path / f"filled_{template_end}.xml"
        fill_od_from_vector(template, pairs, np.array([100.0]), out, 3300)
        import xml.etree.ElementTree as ET

        root = ET.parse(out).getroot()
        assert next(root.iter("interval")).get("end") == "3300"  # od_end_time, NOT template end
        assert next(root.iter("tazRelation")).get("count") == "100"


def test_fill_od_from_vector_shape_guard(tmp_path):
    template = tmp_path / "t.xml"
    template.write_text(
        '<data><interval begin="0" end="3600"><tazRelation from="a" to="b" count="0"/>'
        "</interval></data>"
    )
    with pytest.raises(ValueError, match="od_vector shape"):
        fill_od_from_vector(template, (("a", "b"),), np.array([1.0, 2.0]), tmp_path / "o.xml", 3300)


# --- Engine-free: held-out registry integrity + non-leakage --------------------


def test_heldout_registry_integrity():
    assert set(BO4MOB_HELDOUT) == set(SMALL)
    assert BO4MOB_HELDOUT_HOUR == "06-07"
    assert "221008" not in BO4MOB_HELDOUT_DATES  # the anchor stays in TRAIN
    assert len(BO4MOB_HELDOUT_DATES) == 13
    for key in SMALL:
        rows = BO4MOB_HELDOUT[key]
        assert tuple(d for d, _, _ in rows) == BO4MOB_HELDOUT_DATES
        for _date, checksum, size in rows:
            assert len(checksum) == 64 and all(c in "0123456789abcdef" for c in checksum)
            assert isinstance(size, int) and size > 0


def test_heldout_is_separate_from_ci_prefetched_registry():
    # Never auto-pulled: the held-out panel is a separate download-on-demand registry.
    assert not (set(BO4MOB_HELDOUT) & set(REGISTRY))


def test_heldout_fetch_refuses_hpc_and_unknown():
    with pytest.raises(Bo4MobHpcOnlyError, match="HPC-only"):
        fetch_bo4mob_heldout("5fullRegion")
    with pytest.raises(KeyError):
        fetch_bo4mob_heldout("nonsense")


def test_heldout_fetch_checksum_evicts(tmp_path, monkeypatch):
    """A tampered held-out download is caught post-write, evicted, and re-raised."""
    monkeypatch.setenv("TABENCH_CACHE", str(tmp_path))

    class _R:
        def __init__(self):
            self._served = False

        def read(self, size=-1):
            if self._served:
                return b""
            self._served = True
            return b"tampered"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(bo.urllib.request, "urlopen", lambda *a, **k: _R())
    with pytest.raises(ChecksumError, match="checksum mismatch"):
        fetch_bo4mob_heldout("1ramp")
    cached = tmp_path / "bo4mob" / "1ramp" / "heldout"
    assert list(cached.glob("*.part*")) == []


def test_task_does_not_leak_heldout_dates_or_counts():
    """P7 (adr-041 ruling 4): the held-out DATES **and COUNTS** must be STRUCTURALLY
    unreachable from the REAL task — only the sha256 heldout_digest may carry them.

    Checked against the actual shipped task (not a synthetic stand-in), with the DATE
    check excluding the sanctioned digest fields (a purely-numeric date could otherwise
    coincidentally substring a hex digest) and the COUNT check by numeric VALUE
    membership (substring-in-blob is unsafe against the 64-char hex digest/hash)."""
    _fetch_or_skip("1ramp")
    task, cfg, paths = _build_real_bo4mob_task("1ramp")

    # (a) held-out DATE: string-render every field EXCEPT the sanctioned heldout_digest
    # (which may legitimately contain any hex).
    date_blob = "".join(
        repr(getattr(task, f.name)) for f in dataclasses.fields(task)
        if f.name != "heldout_digest"
    ) + repr(task.dataset.payload) + repr(task.dataset.meta)
    leaked_dates = [d for d in BO4MOB_HELDOUT_DATES if d in date_blob]
    assert not leaked_dates, f"held-out DATE leak: {leaked_dates}"
    assert "221008" in date_blob  # the TRAIN anchor date IS allowed (it is in TRAIN)

    # (b) held-out COUNT: no held-out-ONLY count value is numerically reachable from the
    # task. `legit` (the values that ARE supposed to be present) derives its TRAIN counts
    # from SOURCE — the anchor CSV, NOT task.dataset.payload — so a held-out count
    # smuggled into that very field is caught, not self-excused.
    src_train_counts = {
        float(row["interval_nVehContrib"])
        for row in csv.DictReader(open(paths["sensor"], newline=""))
    }
    reachable = _reachable_numbers(task)
    legit = set(task.prior_vector.tolist()) | src_train_counts | {
        float(cfg["od_end_time"]), float(cfg["sim_end_time"]),
        float(cfg["sensor_start_time"]), float(cfg["sensor_end_time"]),
        0.0, 300.0, float(len(task.pairs)), float(len(src_train_counts)),
        len(src_train_counts) / len(task.pairs),
    }
    heldout_counts: set[float] = set()
    for _date, path in fetch_bo4mob_heldout("1ramp").items():
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                heldout_counts.add(float(row["interval_nVehContrib"]))
    leaked_counts = (heldout_counts & reachable) - legit
    assert not leaked_counts, f"held-out COUNT leak: {sorted(leaked_counts)}"


# --- Engine-free: golden hash + dual-benchmark honesty -------------------------


def test_adding_the_family_does_not_move_the_golden_braess_hash():
    assert braess_scenario().content_hash() == GOLDEN_BRAESS_HASH


def test_dual_benchmark_disclosure_present_in_new_modules():
    # Whitespace-normalised so a docstring line-wrap can't hide the disclosure.
    for mod in (bb, eb):
        src = " ".join(inspect.getsource(mod).split())
        assert "lab's OWN benchmark" in src
        assert "validation of TABench methods" in src
    # The runner manifest carries the affiliation + the forbidden-clause-3.
    from tabench.experiments.runner import _BO4MOB_ESTIMATION_NOTES

    notes = " ".join(_BO4MOB_ESTIMATION_NOTES.split())
    assert "UMN Choi Lab" in notes
    assert "never validation of TABench methods" in notes
    assert "SPSA/BO leaderboard" in notes  # extended forbidden clause 3


def test_no_forbidden_comparability_claims_in_new_modules():
    """A tripwire (adr-041): the new surfaces must never AFFIRMATIVELY claim to beat or
    reproduce BO4Mob (the negated disclosures 'does NOT reproduce ...' are the point)."""
    from tabench.experiments.runner import _BO4MOB_ESTIMATION_NOTES

    forbidden = (
        "validated on BO4Mob", "reproduces BO4Mob", "beats BO4Mob",
        "outperforms BO4Mob", "validated against BO4Mob",
    )
    for text in (inspect.getsource(bb), inspect.getsource(eb), _BO4MOB_ESTIMATION_NOTES):
        for phrase in forbidden:
            assert phrase not in text, f"forbidden comparability claim: {phrase!r}"


# --- Pipeline liveness (sumo-gated) --------------------------------------------


def _fetch_or_skip(key):
    try:
        fetch_bo4mob(BO4MOB_REGISTRY[key])
        fetch_bo4mob_heldout(key)
    except ChecksumError:
        raise
    except Exception as exc:  # offline -> skip; TABENCH_REQUIRE_DATA -> hard fail
        if os.environ.get("TABENCH_REQUIRE_DATA"):
            raise
        pytest.skip(f"bo4mob {key} data unavailable: {exc}")


def test_1ramp_full_estimator_and_certify_reproduces_anchor():
    """CI liveness/correctness instance: the prior baseline's certified obs_nrmse
    reproduces the faithful-pipeline anchor EXACTLY, proving fill_od_from_vector
    inherits the od_end_time rewrite; held-out mean lands on the pilot value."""
    pytest.importorskip("sumo")
    _fetch_or_skip("1ramp")
    result = run_bo4mob_estimation_experiment(
        "1ramp", [Bo4MobPriorBaseline()], Budget(sp_calls=1), seed=0
    )
    row = result.rows[0]
    assert row["od_feasible"] == 1.0
    assert row["obs_nrmse"] == ANCHOR_NRMSE
    assert row["heldout_nrmse"] == pytest.approx(PILOT_HELDOUT_MEAN, abs=1e-3)
    assert row["n_heldout_dates"] == 13.0
    assert result.manifest["engine"]["version"] == BO4MOB_ENGINE_VERSION


def test_od_end_time_regression_is_live():
    """The exact demand-loss laundering pair (adr-034 Decision 3), reproduced live:
    the correct od_end_time=3300 gives 2.4325; leaving the template 3600 gives 2.3147."""
    pytest.importorskip("sumo")
    _fetch_or_skip("1ramp")
    import json

    paths = fetch_bo4mob(BO4MOB_REGISTRY["1ramp"])
    ho = fetch_bo4mob_heldout("1ramp")
    cfg = json.loads(paths["config"].read_text())
    pairs = bo4mob_pairs(paths["od"])
    pv = bo4mob_prior_vector(paths["od"], paths["single_od"])

    def certify_with(od_end):
        cert = Bo4MobODCertifier(
            instance_key="1ramp", pairs=pairs, train_sensor=paths["sensor"], heldout_sensors=ho,
            paths=paths, od_end_time=od_end, sim_end_time=float(cfg["sim_end_time"]),
            sensor_start_time=float(cfg["sensor_start_time"]),
            sensor_end_time=float(cfg["sensor_end_time"]), engine_version=BO4MOB_ENGINE_VERSION,
            certificate={"seed": 0},
        )
        return cert.certify(pv)["obs_nrmse"]

    assert certify_with(3300) == ANCHOR_NRMSE       # correct: the shipped fix
    assert certify_with(3600) == UNFIXED_NRMSE      # the pre-fix demand-leak bug


def test_prior_certify_byte_matches_fill_single_od():
    """The prior baseline's vector fill is byte-identical to fill_single_od, so the
    certified obs_nrmse equals the stage-1 faithful pipeline (no re-derivation drift)."""
    pytest.importorskip("sumo")
    _fetch_or_skip("1ramp")
    import filecmp

    paths = fetch_bo4mob(BO4MOB_REGISTRY["1ramp"])
    pairs = bo4mob_pairs(paths["od"])
    pv = bo4mob_prior_vector(paths["od"], paths["single_od"])
    a = paths["sensor"].parent / "_a.xml"
    b = paths["sensor"].parent / "_b.xml"
    fill_single_od(paths["od"], paths["single_od"], a, 3300)
    fill_od_from_vector(paths["od"], pairs, pv, b, 3300)
    assert filecmp.cmp(a, b, shallow=False)
    a.unlink()
    b.unlink()


@pytest.mark.parametrize("key", ["2corridor", "3junction"])
def test_midsize_single_certify_liveness(key):
    """Single-certify liveness for the mid-size instances (measured ~9 s / ~14 s)."""
    pytest.importorskip("sumo")
    _fetch_or_skip(key)
    result = run_bo4mob_estimation_experiment(key, [Bo4MobPriorBaseline()], Budget(sp_calls=1))
    row = result.rows[0]
    assert row["od_feasible"] == 1.0
    assert np.isfinite(row["obs_nrmse"]) and np.isfinite(row["heldout_nrmse"])
    assert row["heldout_nrmse"] > 0


@pytest.mark.skipif(
    not os.environ.get("TABENCH_RUN_SLOW_BO4MOB"),
    reason="4smallRegion meso alone is ~126 s; set TABENCH_RUN_SLOW_BO4MOB=1 to run it",
)
def test_4smallregion_opt_in_slow():
    pytest.importorskip("sumo")
    _fetch_or_skip("4smallRegion")
    result = run_bo4mob_estimation_experiment(
        "4smallRegion", [Bo4MobPriorBaseline()], Budget(sp_calls=1),
        estimation={"wall_deadline_seconds": 400.0},
    )
    assert result.rows[0]["od_feasible"] == 1.0


def test_live_crash_simulation_raises(tmp_path):
    """An executed crash (a poisoned net.xml) RAISES RuntimeError from the certifier's
    own pipeline — never od_feasible=0 (adr-041 ruling 7, executed not by analogy)."""
    pytest.importorskip("sumo")
    _fetch_or_skip("1ramp")
    import json

    paths = dict(fetch_bo4mob(BO4MOB_REGISTRY["1ramp"]))
    ho = fetch_bo4mob_heldout("1ramp")
    cfg = json.loads(paths["config"].read_text())
    pairs = bo4mob_pairs(paths["od"])
    pv = bo4mob_prior_vector(paths["od"], paths["single_od"])
    poisoned = tmp_path / "poisoned_net.xml"
    poisoned.write_text("<net>this is not a valid sumo network</net>")
    paths["net"] = poisoned
    cert = Bo4MobODCertifier(
        instance_key="1ramp", pairs=pairs, train_sensor=paths["sensor"], heldout_sensors=ho,
        paths=paths, od_end_time=int(cfg["od_end_time"]), sim_end_time=float(cfg["sim_end_time"]),
        sensor_start_time=float(cfg["sensor_start_time"]),
        sensor_end_time=float(cfg["sensor_end_time"]), engine_version=BO4MOB_ENGINE_VERSION,
        certificate={"seed": 0, "wall_deadline_seconds": 60.0},
    )
    with pytest.raises(RuntimeError):
        cert.certify(pv)
