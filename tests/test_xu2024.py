"""Tests for the Xu et al. (2024) 20-US-city cross-domain dataset (adr-033).

Two layers:

* **Registry integrity** (no network): the 17 shipped cities, their pinned
  checksums, the named exclusions, and the guarantee that the family is a
  *separate* download-on-demand registry — it is NOT in the CI-prefetched
  ``REGISTRY``, so a CI run never touches the 276 MB figshare zip on its own.
* **Cached-city certification** (network on first run; skipped offline, hard
  failure under ``TABENCH_REQUIRE_DATA`` — the house ``load_or_skip`` gate):
  the two CI-sized rungs (Honolulu, San Francisco) are built via HTTP-range
  extraction of only their AequilibraE trio and certified. The published flows
  are a LOOSE reference (their own gap ~1e-3), never a best-known oracle; the
  repo's own bfw agrees with them (cross-implementation), which is the honest
  claim. Cross-solver agreement against the dataset's TransCAD flows — the
  as-published wrong-centroid defect (adr-033) — is never asserted.

Requires network access on first run (checksummed download-on-demand, P9).
"""

import struct

import numpy as np
import pytest
from conftest import load_or_skip

import tabench.data.xu2024 as xu
from tabench import (
    BiconjugateFrankWolfeModel,
    Budget,
    Evaluator,
    RngBundle,
    Trace,
    braess_scenario,
)
from tabench.data import REGISTRY
from tabench.data.xu2024 import (
    XU2024_EXCLUDED,
    XU2024_REGISTRY,
    XU2024_RUNGS,
    Xu2024UpstreamError,
    xu2024_citation,
)

# Golden hash: adding a whole scenario family must not move any existing hash.
GOLDEN_BRAESS_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"


# --- Registry integrity (no network) -------------------------------------------


def test_seventeen_cities_shipped_three_excluded_and_named():
    assert len(XU2024_REGISTRY) == 17
    # The three excluded cities are named with a reason, disjoint from the registry.
    assert set(XU2024_EXCLUDED) == {"washington", "pittsburgh", "phoenix"}
    assert not (set(XU2024_EXCLUDED) & set(XU2024_REGISTRY))
    for reason in XU2024_EXCLUDED.values():
        assert "centroid" in reason


def test_rungs_are_registered_cities():
    for city in XU2024_RUNGS:
        assert city in XU2024_REGISTRY


def test_family_is_separate_from_the_ci_prefetched_registry():
    """CI prefetches REGISTRY.values(); xu2024 must NOT be in it (no 276 MB pull)."""
    assert not (set(XU2024_REGISTRY) & set(REGISTRY))
    for city in XU2024_REGISTRY:
        assert f"xu2024-{city}" not in REGISTRY


def test_specs_carry_pinned_sha256_and_wellformed_members():
    for city, spec in XU2024_REGISTRY.items():
        assert spec.key == city
        assert set(spec.files) == {"network", "aem", "assignment"}
        for role, (base, checksum) in spec.files.items():
            assert len(checksum) == 64 and all(c in "0123456789abcdef" for c in checksum)
            assert spec.member(role).endswith(f"/03_AequilibraE_results/{base}")
            assert spec.member(role).startswith(spec.dir_name + "/")
        assert spec.n_zones > 0 and spec.n_links > 0
        assert 0.0 <= spec.bpr_alpha and spec.bpr_beta > 0.0
        assert 0.0 < spec.published_relative_gap < 5e-3  # loose published reference


def test_citation_names_source_license_and_doi():
    text = xu2024_citation()
    assert "Xu" in text and "Scientific Data" in text
    assert "10.1038/s41597-024-03149-8" in text
    assert "CC BY 4.0" in text
    assert "figshare" in text


def test_adding_the_family_does_not_move_the_golden_braess_hash():
    assert braess_scenario().content_hash() == GOLDEN_BRAESS_HASH


# --- Cached-city certification (network-backed; offline -> skip) -----------------


@pytest.fixture(scope="module")
def honolulu():
    # Offline -> skip; checksum mismatch or TABENCH_REQUIRE_DATA=1 -> fail.
    return load_or_skip("xu2024-honolulu")


@pytest.fixture(scope="module", params=list(XU2024_RUNGS))
def rung(request):
    return load_or_skip(f"xu2024-{request.param}")


def test_parsed_dimensions_and_bpr_match_the_registry(rung):
    spec = XU2024_REGISTRY[rung.name.removeprefix("xu2024-")]
    net = rung.network
    assert (net.n_zones, net.n_nodes, net.n_links) == (spec.n_zones, spec.n_nodes, spec.n_links)
    # AequilibraE columns a/b map onto BPR b(alpha)/power(beta), per-city uniform.
    assert np.allclose(net.b, spec.bpr_alpha)
    assert np.allclose(net.power, spec.bpr_beta)
    # first_thru_node=1: the as-published run allowed through-centroid flow.
    assert net.first_thru_node == 1
    assert rung.family == rung.name


def test_published_flows_conserve_and_are_a_loose_reference(rung):
    """The published AequilibraE flows certify feasible with a LOOSE gap ~1e-3."""
    spec = XU2024_REGISTRY[rung.name.removeprefix("xu2024-")]
    metrics = Evaluator(rung).evaluate(rung.reference.link_flows)
    assert metrics["feasible"] == 1.0
    # Conservation is machine-exact on the as-published 1..Z-centroid graph.
    assert metrics["node_balance_residual"] < 1e-9
    # A loose published reference (own gap ~1e-3), never a best-known oracle.
    assert 1e-4 < metrics["relative_gap"] < 5e-3
    assert metrics["relative_gap"] == pytest.approx(spec.published_relative_gap, abs=3e-4)


def test_reference_note_discloses_the_defect_and_loose_status(rung):
    note = rung.reference.note.lower()
    assert "not a best-known" in note
    assert "wrong-centroid" in note or "1..z" in note
    assert "transcad" in note  # the un-claimed cross-solver comparison
    assert rung.reference.link_flows.shape == (rung.network.n_links,)
    assert np.all(np.isfinite(rung.reference.link_flows))
    assert np.all(rung.reference.link_flows >= 0)


def test_content_hash_is_stable_across_reloads(honolulu):
    again = load_or_skip("xu2024-honolulu")
    assert honolulu.content_hash() == again.content_hash()


def test_bfw_agrees_with_the_published_flows(honolulu):
    """Cross-implementation agreement: the repo's own bfw converges to the same
    flows as the published AequilibraE run on this identical instance. A tighter
    gap needs more iterations (rgap ~1e-4 at ~400 iters, recorded in adr-033);
    the CI anchor uses a small budget and checks feasibility + correlation."""
    trace = Trace()
    BiconjugateFrankWolfeModel().solve(honolulu, Budget(iterations=30), RngBundle(0), trace)
    flows = trace.final.link_flows
    metrics = Evaluator(honolulu).evaluate(flows)
    assert metrics["feasible"] == 1.0
    # bfw drives the gap well below the all-or-nothing start (measured ~6e-3).
    assert metrics["relative_gap"] < 3e-2
    published = honolulu.reference.link_flows
    assert np.corrcoef(flows, published)[0, 1] > 0.99


# --- Fetcher hardening regressions (no network; adr-033 review fix batch) --------


def test_structural_error_fails_fast_without_whole_zip_fallback(tmp_path, monkeypatch):
    """A wrong member name (KeyError) must NOT escalate to the 276 MB fallback —
    downloading the whole zip cannot conjure a member that is not there."""
    monkeypatch.setenv("TABENCH_CACHE", str(tmp_path))

    def bad_member(spec, roles, timeout):
        raise KeyError("no such member in the zip")

    def no_fallback(*args, **kwargs):
        raise AssertionError("whole-zip fallback must not run for a structural error")

    monkeypatch.setattr(xu, "_extract_ranged", bad_member)
    monkeypatch.setattr(xu, "_extract_whole_zip", no_fallback)
    with pytest.raises(KeyError):
        xu.fetch_city(XU2024_REGISTRY["honolulu"], force=True)


def test_transport_error_triggers_whole_zip_fallback(tmp_path, monkeypatch):
    """A range-refusal / transport error IS the documented fallback trigger."""
    monkeypatch.setenv("TABENCH_CACHE", str(tmp_path))

    class _FallbackInvoked(Exception):
        pass

    def range_refused(spec, roles, timeout):
        raise OSError("server did not honor a Range request")

    def fallback_marker(*args, **kwargs):
        raise _FallbackInvoked()

    monkeypatch.setattr(xu, "_extract_ranged", range_refused)
    monkeypatch.setattr(xu, "_extract_whole_zip", fallback_marker)
    with pytest.raises(_FallbackInvoked):
        xu.fetch_city(XU2024_REGISTRY["honolulu"], force=True)


def test_whole_zip_fallback_cleans_stray_part_on_failure(tmp_path, monkeypatch):
    """A mid-stream failure in the fallback must not strand an unbounded .part."""
    monkeypatch.setenv("TABENCH_CACHE", str(tmp_path))
    (tmp_path / "xu2024").mkdir(parents=True)

    class _StallingResp:
        headers = {"Content-Length": str(xu.ZIP_SIZE)}  # passes the size guard

        def __init__(self):
            self._chunks = iter([b"P" * 4096])

        def read(self, size=-1):
            try:
                return next(self._chunks)
            except StopIteration:
                raise TimeoutError("stalled mid-body") from None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(xu.urllib.request, "urlopen", lambda *a, **k: _StallingResp())
    with pytest.raises(TimeoutError):
        xu._extract_whole_zip(XU2024_REGISTRY["honolulu"], ["network"], timeout=1.0)
    assert list((tmp_path / "xu2024").glob("*.part")) == []


def test_range_probe_rejects_wrong_upstream_size(monkeypatch):
    """The pre-transfer size guard: a probe reporting a non-v4 total must raise
    an upstream-changed error before any member is read."""

    class _WrongSizeResp:
        headers = {"Content-Range": "bytes 0-0/999999"}  # not the pinned v4 size

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(xu.urllib.request, "urlopen", lambda *a, **k: _WrongSizeResp())
    with pytest.raises(Xu2024UpstreamError, match="upstream artifact changed"):
        xu._HttpRangeFile(xu.ZIP_URL, expected_size=xu.ZIP_SIZE)


def test_aem_parser_rejects_non_canonical_zone_index(tmp_path):
    """Drift guard: the .aem zone index must be exactly 1..Z (adr-033 centroids)."""
    zones = 3
    header = bytearray(252)
    struct.pack_into("<I", header, 2, zones * zones)  # cell count
    struct.pack_into("<I", header, 10, zones)  # zone count
    index = np.array([1, 2, 4], dtype="<i8").tobytes()  # NOT 1..Z (3 is missing)
    matrix = np.zeros(zones * zones, dtype="<f8").tobytes()
    path = tmp_path / "bad.aem"
    path.write_bytes(bytes(header) + index + matrix)
    with pytest.raises(ValueError, match="zone index is not"):
        xu._parse_aem(path)


def test_network_parser_rejects_non_directed(tmp_path):
    """Drift guard: this dataset is fully directed (direction == 1 everywhere)."""
    path = tmp_path / "bad_network.csv"
    path.write_text(
        ",a_node,b_node,capacity,free_flow_time,a,b,direction,link_id\n"
        "0,1,2,1400,0.1,0.5,1.5,1,1\n"
        "1,2,1,1400,0.1,0.5,1.5,2,2\n"  # direction=2: not fully directed
    )
    with pytest.raises(ValueError, match="direction != 1"):
        xu._parse_network_csv(path)
