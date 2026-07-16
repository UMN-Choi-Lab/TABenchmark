"""Tests for the BO4Mob San Jose scenario family (adr-034).

Two layers:

* **Registry integrity + fetcher hardening** (no engine, no live network): the
  five instances (four fetchable small + one HPC-only that refuses to fetch),
  their pinned checksums, the dual-benchmark honesty disclosures, and the
  guarantee that the family is a *separate* download-on-demand registry — it is
  NOT in the CI-prefetched ``REGISTRY``, and its keys are NOT ``load_scenario``
  scenarios (BO4Mob is a mesoscopic-SUMO net with no BPR network and no true OD,
  so it is data, never a ``Scenario``). Checksum eviction and .part hygiene are
  exercised with a mocked ``urlopen`` (offline).
* **Pipeline liveness** (``importorskip('sumo')``, the sumo CI job): fetch the
  1ramp bundle and run od2trips + mesoscopic SUMO end-to-end via the wheel
  binaries, asserting the pipeline emits per-edge counts, the count NRMSE is
  seed-stable and in a loose measured band, and the 1.27.1 schema drift holds
  (no ``nVehContrib`` — BO4Mob's ``arrived+left`` convention is used).

Stage 1 = data availability + pipeline liveness only; no task, certificate, or
estimator (those are the named stage-2 follow-up). See adr-034.
"""

import os
import time
import xml.etree.ElementTree as ET

import pytest

import tabench.data.bo4mob as bo
from tabench import braess_scenario
from tabench.data import REGISTRY, load_scenario
from tabench.data.bo4mob import (
    BO4MOB_REGISTRY,
    BO4MOB_SMOKE,
    Bo4MobHpcOnlyError,
    ChecksumError,
    bo4mob_citation,
    fetch_bo4mob,
)

# Golden hash: adding a whole scenario family must not move any existing hash.
GOLDEN_BRAESS_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"

SMALL = ("1ramp", "2corridor", "3junction", "4smallRegion")


# --- Registry integrity (no network) -------------------------------------------


def test_five_instances_four_small_one_hpc():
    assert set(BO4MOB_REGISTRY) == set(SMALL) | {"5fullRegion"}
    for key in SMALL:
        assert BO4MOB_REGISTRY[key].hpc_only is False
        assert BO4MOB_REGISTRY[key].files  # fetchable bundle
    hpc = BO4MOB_REGISTRY["5fullRegion"]
    assert hpc.hpc_only is True
    assert hpc.files == {}  # metadata-only, nothing to fetch
    assert hpc.n_od == 10100


def test_smoke_instance_is_the_smallest_registered():
    assert BO4MOB_SMOKE in BO4MOB_REGISTRY
    assert BO4MOB_SMOKE == min(SMALL, key=lambda k: BO4MOB_REGISTRY[k].n_od)


def test_family_is_separate_from_the_ci_prefetched_registry():
    """CI prefetches REGISTRY.values(); BO4Mob must NOT be in it (no auto-pull)."""
    assert not (set(BO4MOB_REGISTRY) & set(REGISTRY))
    for key in BO4MOB_REGISTRY:
        assert f"bo4mob-{key}" not in REGISTRY


def test_bo4mob_keys_are_not_load_scenario_scenarios():
    """BO4Mob is a meso-SUMO net (no BPR network, no true OD): data, never a
    ``Scenario``. ``load_scenario`` must not resolve a bo4mob key."""
    for key in (BO4MOB_SMOKE, f"bo4mob-{BO4MOB_SMOKE}", "5fullRegion"):
        with pytest.raises(KeyError):
            load_scenario(key)


def test_specs_carry_pinned_sha256_and_wellformed_paths():
    for key in SMALL:
        spec = BO4MOB_REGISTRY[key]
        assert spec.key == key
        assert set(spec.files) == {
            "net", "taz", "od", "additional", "routes_single",
            "single_od", "config", "sensor",
        }
        for _role, (repo_path, checksum, size) in spec.files.items():
            assert len(checksum) == 64 and all(c in "0123456789abcdef" for c in checksum)
            assert not repo_path.startswith("/") and ".." not in repo_path
            assert isinstance(size, int) and size > 0  # pinned byte size (size guard)
        assert spec.files["net"][0] == f"network/{spec.netdir}/net.xml"
        assert spec.files["sensor"][0].startswith("sensor_data/221008/")
        assert spec.n_od > 0 and spec.n_sensors_anchor > 0


def test_citation_names_source_license_pems_and_the_dual_benchmark_contract():
    text = bo4mob_citation()
    assert "Ryu" in text and "2510.18824" in text
    assert "MIT" in text
    assert "PeMS" in text
    assert "UMN Choi Lab" in text  # affiliation declared openly
    assert "scenarios/data only" in text
    assert "does not reproduce" in text  # never claims the paper's numbers
    # a per-instance citation names the instance + its dimension
    assert "1ramp" in bo4mob_citation(BO4MOB_REGISTRY["1ramp"])


def test_notes_disclose_affiliation_and_scenarios_only():
    for spec in BO4MOB_REGISTRY.values():
        assert "UMN Choi Lab" in spec.notes
        assert "scenarios/data only" in spec.notes
        assert "does not reproduce" in spec.notes


def test_adding_the_family_does_not_move_the_golden_braess_hash():
    assert braess_scenario().content_hash() == GOLDEN_BRAESS_HASH


# --- Fetcher hardening (mocked urlopen; offline) --------------------------------


def test_hpc_instance_refuses_to_fetch():
    with pytest.raises(Bo4MobHpcOnlyError, match="HPC-only"):
        fetch_bo4mob(BO4MOB_REGISTRY["5fullRegion"])


def _resp(data: bytes):
    """A urlopen stand-in that serves ``data`` once, then EOF (the chunked read)."""

    class _R:
        def __init__(self):
            self._served = False

        def read(self, size=-1):
            if self._served:
                return b""
            self._served = True
            return data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return _R()


def test_checksum_mismatch_evicts_and_raises(tmp_path, monkeypatch):
    """A tampered download is caught post-write, evicted, and re-raised — nothing
    unverified is left in the cache."""
    monkeypatch.setenv("TABENCH_CACHE", str(tmp_path))
    monkeypatch.setattr(bo.urllib.request, "urlopen", lambda *a, **k: _resp(b"tampered"))
    with pytest.raises(ChecksumError, match="checksum mismatch"):
        fetch_bo4mob(BO4MOB_REGISTRY["1ramp"])
    cached = tmp_path / "bo4mob" / "1ramp"
    assert not (cached / "net.xml").exists()
    assert list(cached.glob("*.part*")) == []


def test_download_failure_mid_body_cleans_stray_part(tmp_path, monkeypatch):
    """A failure mid-body (after the .part is opened and partly written) must not
    strand a partial file — the finally-unlink cleans it. Mutation-verified: the
    reviewer's earlier mock raised at urlopen(), before any .part existed, so the
    test passed even under a finally-removal mutant."""
    monkeypatch.setenv("TABENCH_CACHE", str(tmp_path))

    class _MidBodyResp:
        def __init__(self):
            self._served = False

        def read(self, size=-1):
            if not self._served:
                self._served = True
                return b"partial-body-chunk"  # written to the .part, THEN we fail
            raise OSError("connection reset mid-body")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(bo.urllib.request, "urlopen", lambda *a, **k: _MidBodyResp())
    with pytest.raises(OSError):
        fetch_bo4mob(BO4MOB_REGISTRY["1ramp"])
    assert list((tmp_path / "bo4mob" / "1ramp").glob("*.part*")) == []


def test_oversized_body_refused_before_materializing(tmp_path, monkeypatch):
    """An upstream body exceeding the pinned size (+slack) is refused mid-stream,
    before it is materialised — a hostile 256 MB body cannot fill RSS/disk ahead
    of the checksum (lens 1 F1)."""
    monkeypatch.setenv("TABENCH_CACHE", str(tmp_path))

    class _HugeResp:
        def read(self, size=-1):
            return b"x" * (1 << 16)  # endless 64 KB chunks -> exceeds any small cap

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(bo.urllib.request, "urlopen", lambda *a, **k: _HugeResp())
    with pytest.raises(bo.Bo4MobUpstreamError, match="exceeds the pinned size"):
        fetch_bo4mob(BO4MOB_REGISTRY["1ramp"])
    assert list((tmp_path / "bo4mob" / "1ramp").glob("*.part*")) == []


def test_fill_single_od_rewrites_interval_end_to_od_end_time(tmp_path):
    """The material fix (adr-034 Decision 3): fill_single_od must rewrite the
    interval end to od_end_time regardless of the template's end, so 1ramp's
    od_end_time=3300 window is honored (keeping the template's 3600 leaked ~5% of
    demand past the OD window). Template-end-agnostic + still fills the count."""
    single = tmp_path / "od.csv"
    single.write_text("fromTaz,toTaz,flow\ntaz_0,taz_1,100\n")
    for template_end in ("3600", "9999"):
        template = tmp_path / f"tmpl_{template_end}.xml"
        template.write_text(
            '<data><interval id="DEFAULT_VEHTYPE" begin="0" end="' + template_end + '">'
            '<tazRelation from="taz_0" to="taz_1" count="0"/></interval></data>'
        )
        out = tmp_path / f"filled_{template_end}.xml"
        bo.fill_single_od(template, single, out, 3300)
        root = ET.parse(out).getroot()
        interval = next(root.iter("interval"))
        assert interval.get("end") == "3300"  # od_end_time, NOT the template end
        assert next(root.iter("tazRelation")).get("count") == "100"


# --- Pipeline liveness (sumo-gated; the guarded smoke test) ----------------------


def _run_1ramp(paths, work, seed, sumo):
    """od2trips + route-fix + mesoscopic SUMO for one seed, via the wheel binaries.

    One wall deadline threads both subprocesses (stdin=DEVNULL, workdir-local
    output — the adr-027 wrapper discipline). Returns
    (nrmse, n_edges, n_trips, saw_nvehcontrib)."""
    import json
    import subprocess

    cfg = json.loads(paths["config"].read_text())
    sim_end = float(cfg["sim_end_time"])
    od_end = int(cfg["od_end_time"])
    s0, s1 = float(cfg["sensor_start_time"]), float(cfg["sensor_end_time"])
    env = {**os.environ, "SUMO_HOME": sumo.SUMO_HOME}

    def sbin(name):
        return os.path.join(sumo.SUMO_HOME, "bin", name)

    deadline = time.monotonic() + 120.0

    def run(cmd):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("bo4mob smoke exceeded the wall budget")
        try:
            proc = subprocess.run(
                cmd, env=env, cwd=work, stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=remaining,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"{cmd[0]} exceeded the wall budget") from exc
        if proc.returncode != 0:
            raise RuntimeError(f"{cmd[0]} failed (rc={proc.returncode}): {proc.stderr[-800:]}")

    od_filled = work / f"od_filled_{seed}.xml"
    bo.fill_single_od(paths["od"], paths["single_od"], od_filled, od_end)
    trips_before = work / f"trips_before_{seed}.xml"
    run([
        sbin("od2trips"), "--spread.uniform", "--taz-files", str(paths["taz"]),
        "--tazrelation-files", str(od_filled), "-o", str(trips_before),
    ])
    trips_fixed = work / f"trips_fixed_{seed}.xml"
    n_trips = bo.fix_routes_single(trips_before, paths["routes_single"], trips_fixed)
    # Per-seed edge_data name: a future engine exiting 0 without writing must not
    # silently re-read the previous seed's output (lens 2 P3).
    edge_data_name = f"edge_data_{seed}.xml"
    add_local = work / f"additional_local_{seed}.xml"
    bo.local_edgedata_additional(paths["additional"], add_local, edge_data_name)
    run([
        sbin("sumo"), "--mesosim", "true", "--net-file", str(paths["net"]),
        "--routes", str(trips_fixed), "-b", "0", "-e", str(int(sim_end)),
        "--additional-files", str(add_local), "--ignore-route-errors", "true",
        "--xml-validation", "never", "--no-warnings", "--seed", str(seed),
    ])
    edge_data = work / edge_data_name
    counts = bo.edgedata_counts(edge_data, s0, s1)
    saw = bo.edgedata_has_nvehcontrib(edge_data)
    nrmse = bo.bo4mob_nrmse(paths["sensor"], counts)
    return nrmse, len(counts), n_trips, saw


def test_1ramp_pipeline_liveness_and_seed_stability(tmp_path):
    sumo = pytest.importorskip("sumo")
    spec = BO4MOB_REGISTRY[BO4MOB_SMOKE]
    try:
        paths = fetch_bo4mob(spec)
    except ChecksumError:
        raise
    except Exception as exc:  # offline -> skip; TABENCH_REQUIRE_DATA -> hard fail
        if os.environ.get("TABENCH_REQUIRE_DATA"):
            raise
        pytest.skip(f"bo4mob {BO4MOB_SMOKE} data unavailable: {exc}")

    results, edges, trips, drift = [], [], [], []
    for seed in (0, 1, 2):
        nrmse, n_edges, n_trips, saw = _run_1ramp(paths, tmp_path, seed, sumo)
        results.append(nrmse)
        edges.append(n_edges)
        trips.append(n_trips)
        drift.append(saw)

    # The pipeline emitted count files with per-edge rows and kept the trips.
    assert all(n > 0 for n in edges)
    assert all(t > 0 for t in trips)
    # Seed-stable: uncongested 1ramp + speedDev=0 is deterministic under meso
    # (measured byte-identical; a loose tol survives cross-platform float drift).
    assert max(results) - min(results) < 1e-3
    # Loose measured band: NRMSE ~2.4325 on the 1.27.1 wheel with the od_end_time
    # fill (adr-034 Decision 3 provenance); the band survives engine drift.
    assert 1.5 < results[0] < 3.5
    # The measured 1.12 -> 1.27.1 schema drift: no nVehContrib emitted, so
    # BO4Mob's arrived+left convention is what the count uses.
    assert not any(drift)
