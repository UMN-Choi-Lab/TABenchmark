"""BO4Mob San Jose freeway OD-estimation scenarios (P9; adr-034).

BO4Mob (Ryu, Kwon, Choi, Deshwal, Kang & Osorio, 2025, arXiv:2510.18824,
NeurIPS 2025 Datasets & Benchmarks; canon ``ryu2025bo4mob``) poses five San Jose
freeway networks as high-dimensional black-box OD-estimation problems: minimise
the NRMSE between mesoscopic-SUMO link counts and real Caltrans PeMS sensor data
over a continuous OD vector (3 -> 10,100 OD pairs). There is **no** ground-truth
OD — truth is the real sensor panel (14 dates x 3 hour windows).

**The dual-benchmark honesty contract (the central constraint of adr-034).**
BO4Mob is **the lab's own benchmark** (github.com/UMN-Choi-Lab/BO4Mob, MIT).
This module hosts its instances as *scenarios / data only* — never as validation
of TABenchmark's own methods, and never as a claim to reproduce BO4Mob's
published numbers. The shipped ``eclipse-sumo`` wheel is 1.27.1; BO4Mob's paper
ran SUMO 1.12, and a **measured schema drift** (the mesoscopic ``edgeData`` under
1.27.1 carries no ``nVehContrib`` attribute — BO4Mob's own ``arrived + left``
count convention still holds, so the *instances* transfer but the simulated
*values* do not) makes the paper numbers non-reproducible here anyway. The
affiliation is declared openly (here, in ``notes``, and in ``bo4mob_citation``).

**Scope — stage 1: data availability + pipeline liveness only.** A separate,
checksummed, download-on-demand registry for the four small instances (1ramp,
2corridor, 3junction, 4smallRegion — < 1.2 MB total single-evaluation bundles),
deliberately **not** in the CI-prefetched ``REGISTRY``. ``5fullRegion`` (74 MB,
~11 h/eval) is registered metadata-only and **refuses to fetch** (HPC-only). No
task family, no certificate, no estimator — those are a named stage-2 follow-up
(a pinned-engine held-out-date observational certificate). The pure count/NRMSE
transforms here are BO4Mob's evaluation convention (pandas-free, numpy/stdlib),
reusable by that stage; the actual engine run is exercised by the guarded smoke
test (``tests/test_bo4mob.py``): fetch 1ramp, run od2trips + mesoscopic SUMO via
the wheel binaries, and check the pipeline emits counts with a seed-stable NRMSE.

Data provenance: BO4Mob is MIT-licensed and redistributes public Caltrans PeMS
detector data under its own license appendix + Gebru-style datasheet. This module
fetches from the BO4Mob repo (commit-pinned raw files, per-file SHA-256 AND a
pinned byte size, cached under ``~/.cache/tabench/bo4mob/<key>/``) and credits
both. Never vendored (P9).
"""

from __future__ import annotations

import csv
import os
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fetcher import ChecksumError, _sha256, cache_dir

__all__ = [
    "BO4MOB_COMMIT",
    "BO4MOB_ENGINE_NAME",
    "BO4MOB_ENGINE_VERSION",
    "BO4MOB_HELDOUT",
    "BO4MOB_HELDOUT_DATES",
    "BO4MOB_HELDOUT_HOUR",
    "BO4MOB_REGISTRY",
    "BO4MOB_SMOKE",
    "Bo4MobHpcOnlyError",
    "Bo4MobSpec",
    "Bo4MobUpstreamError",
    "bo4mob_citation",
    "bo4mob_nrmse",
    "bo4mob_pairs",
    "bo4mob_prior_vector",
    "edgedata_counts",
    "edgedata_has_nvehcontrib",
    "fetch_bo4mob",
    "fetch_bo4mob_heldout",
    "fill_od_from_vector",
    "fill_single_od",
    "fix_routes_single",
    "local_edgedata_additional",
]

# The pinned mesoscopic engine the stage-2 D2 certificate re-runs (adr-041). A
# CONSTANT, not "whatever is installed": ``assert_engine_pin`` RAISES if the box's
# ``eclipse-sumo`` differs, so scoring is always under the pinned engine (matches
# the CI-pinned 1.27.1 wheel; the paper's 1.12 numbers are non-reproducible here).
BO4MOB_ENGINE_NAME = "eclipse-sumo"
BO4MOB_ENGINE_VERSION = "1.27.1"

# --- Provenance (commit-pinned raw files; the TNTP fetcher precedent) -----------
BO4MOB_COMMIT = "ef571e6819a6e1eb13388f7c0454d32f665b6ce4"
BO4MOB_BASE = f"https://raw.githubusercontent.com/UMN-Choi-Lab/BO4Mob/{BO4MOB_COMMIT}"
BO4MOB_LICENSE = "MIT"
# The canonical single-evaluation anchor (one date x one hour window). The full
# 14-date x 3-hour PeMS panel is available upstream; stage 2's held-out-date
# certificate pins the rest — stage 1 pins only this anchor per instance.
BO4MOB_ANCHOR_DATE = "221008"
BO4MOB_ANCHOR_HOUR = "06-07"

# Headroom over each file's pinned byte size before the fetcher refuses. The
# checksum is exact, so a well-formed file matches its pinned size exactly; the
# slack only absorbs a benign transfer artifact before the SHA-256 evicts it.
# The cap bounds how much a hostile upstream body can materialise (a 256 MB body
# no longer drives RSS/disk before the checksum runs — lens 1 F1).
_SIZE_SLACK = 4096

# The in-artifact disclosure line (the TNTP ``NetworkSpec.notes`` precedent): the
# dual-benchmark contract travels with the data, not only in the docs.
BO4MOB_ORIGIN = (
    "BO4Mob (Ryu, Kwon, Choi, Deshwal, Kang & Osorio 2025, arXiv:2510.18824, "
    "NeurIPS 2025 Datasets & Benchmarks) is a UMN Choi Lab benchmark, MIT-licensed; "
    "ground truth is public Caltrans PeMS detector data. TABenchmark hosts its "
    "instances as scenarios/data only (adr-034) — never as validation of TABench "
    "methods — and does not reproduce BO4Mob's published numbers (engine drift "
    f"SUMO 1.12 -> 1.27.1). Fetched from BO4Mob @ {BO4MOB_COMMIT[:12]}."
)


class Bo4MobHpcOnlyError(RuntimeError):
    """Fetch refused: an HPC-scale instance (``5fullRegion``, 74 MB, ~11 h/eval).

    Registered metadata-only; it is never fetched in CI or on a laptop. An HPC
    run fetches it deliberately by other means. A named refusal, not a silent
    omission (adr-034)."""


class Bo4MobUpstreamError(RuntimeError):
    """An upstream response exceeded a file's pinned byte size (+slack).

    Raised mid-stream, before an oversized body is fully materialised, so a
    hostile or drifted upstream file fails loudly rather than filling RSS/disk
    ahead of the SHA-256 check (adr-034)."""


@dataclass(frozen=True)
class Bo4MobSpec:
    """Provenance + measured metadata for one BO4Mob instance.

    ``files`` maps role -> (repo path at ``BO4MOB_COMMIT``, sha256 of the bytes,
    pinned byte size). ``n_od`` is the problem dimension (OD pairs, matches paper
    Table 1); ``n_sensors_anchor`` is the measured GT-link count on the canonical
    ``221008 06-07`` anchor (it can vary by date/hour with PeMS coverage). An
    ``hpc_only`` spec carries metadata but no ``files`` and refuses to fetch.
    """

    key: str
    netdir: str
    n_od: int
    n_sensors_anchor: int
    hpc_only: bool
    files: dict[str, tuple[str, str, int]]
    notes: str

    def local_name(self, role: str) -> str:
        """Cache filename for ``role`` (the repo basename; unique per instance)."""
        return Path(self.files[role][0]).name


# --- Checksummed download-on-demand (the TNTP fetcher mechanics) -----------------


def _fetch_checked(
    local: Path, url: str, checksum: str, size: int, force: bool, timeout: float, label: str
) -> None:
    """Download one commit-pinned file to ``local`` and verify it in place.

    Shared by ``fetch_bo4mob`` and ``fetch_bo4mob_heldout`` (same hardening for
    both the single-evaluation bundle and the held-out panel). The body is
    **streamed with a cap** at ``size + _SIZE_SLACK`` — an oversized upstream body
    raises ``Bo4MobUpstreamError`` mid-stream, before it is materialised (lens 1
    F1). The ``.part`` carries a per-process suffix so concurrent cold-start
    fetches never collide, and is cleaned in ``finally`` so a mid-download failure
    strands nothing (xu2024 lesson). The SHA-256 is checked on **every** load; a
    mismatch evicts the file and raises ``ChecksumError``. ``label`` is the
    ``key/name`` (or ``key/heldout/name``) used verbatim in error messages.
    """
    if force or not local.exists():
        tmp = local.with_suffix(local.suffix + f".part.{os.getpid()}")
        cap = size + _SIZE_SLACK
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
                total = 0
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > cap:
                        raise Bo4MobUpstreamError(
                            f"bo4mob/{label}: upstream body exceeds the pinned size "
                            f"{size} B (+{_SIZE_SLACK} slack); refusing before "
                            "materialising an oversized file (adr-034)."
                        )
                    out.write(chunk)
            tmp.replace(local)
        finally:  # a mid-download failure must not strand a .part (xu2024 lesson)
            tmp.unlink(missing_ok=True)
    actual = _sha256(local)
    if actual != checksum:
        local.unlink(missing_ok=True)
        raise ChecksumError(
            f"bo4mob/{label}: checksum mismatch "
            f"(expected {checksum[:12]}…, got {actual[:12]}…). "
            "File removed from cache; re-run to re-download."
        )


def fetch_bo4mob(spec: Bo4MobSpec, force: bool = False, timeout: float = 60.0) -> dict[str, Path]:
    """Ensure an instance's single-evaluation bundle is cached and verified.

    Returns ``role -> local path``. Each file is fetched from a commit-pinned raw
    URL, cached under ``~/.cache/tabench/bo4mob/<key>``, and verified against its
    pinned SHA-256 on **every** load (a mismatch evicts the file and raises
    ``ChecksumError``). The body is **streamed with a cap** at the file's pinned
    byte size (+slack): an oversized upstream body raises ``Bo4MobUpstreamError``
    mid-stream, before it is materialised. The ``.part`` carries a per-process
    suffix so concurrent cold-start fetches never collide (fetcher.py / xu2024.py
    share the latent single-``.part`` race — a documented follow-up, not touched
    here). An ``hpc_only`` instance refuses (``Bo4MobHpcOnlyError``).
    """
    if spec.hpc_only:
        raise Bo4MobHpcOnlyError(
            f"bo4mob {spec.key} is HPC-only (74 MB, ~11 h/eval) and is registered "
            "metadata-only; refusing to fetch (adr-034)."
        )
    target_dir = cache_dir() / "bo4mob" / spec.key
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for role, (repo_path, checksum, size) in spec.files.items():
        local = target_dir / spec.local_name(role)
        _fetch_checked(
            local, f"{BO4MOB_BASE}/{repo_path}", checksum, size, force, timeout,
            f"{spec.key}/{spec.local_name(role)}",
        )
        paths[role] = local
    return paths


def bo4mob_citation(spec: Bo4MobSpec | None = None) -> str:
    """Mandatory attribution for the BO4Mob data source + its PeMS ground truth."""
    base = (
        "Ryu, Kwon, Choi, Deshwal, Kang & Osorio (2025). BO4Mob: Bayesian "
        "Optimization Benchmarks for High-Dimensional Urban Mobility Problem. "
        "arXiv:2510.18824 (NeurIPS 2025 Datasets & Benchmarks). Data + engine: "
        f"github.com/UMN-Choi-Lab/BO4Mob (commit {BO4MOB_COMMIT[:12]}, {BO4MOB_LICENSE}). "
        "Ground truth: Caltrans PeMS detector data (public). BO4Mob is a UMN Choi "
        "Lab benchmark; TABenchmark hosts its instances as scenarios/data only "
        "(adr-034), not as validation of TABench methods, and does not reproduce "
        "BO4Mob's published numbers."
    )
    if spec is not None:
        return f"{base} Instance: {spec.key} ({spec.n_od} OD pairs)."
    return base


# --- BO4Mob evaluation convention (pandas-free; numpy/stdlib only) ---------------
# These are the pure transforms of BO4Mob's own pipeline (od2trips route-fixing +
# the arrived+left count convention + count NRMSE), reusable by the stage-2
# estimation certificate. The engine subprocesses themselves live in the guarded
# smoke test (the sumo-marouter `_run_marouter`-in-test precedent, adr-027).


def fill_single_od(
    od_template: str | Path, single_od: str | Path, out: str | Path, od_end_time: int
) -> None:
    """Fill the ``count=0`` ``od.xml`` template with the single-run OD flows.

    The shipped ``od.xml`` is a zero template; the ``od_for_single_run`` CSV
    (columns ``fromTaz,toTaz,flow``) carries the example OD vector.

    The interval ``end`` is rewritten to ``od_end_time`` (from the instance's
    config), exactly as BO4Mob's own ``create_od_tazrelation_xml`` always does —
    it is **load-bearing**: the shipped template keeps ``end=3600``, but 1ramp's
    OD window is ``od_end_time=3300``, so keeping the template's 3600 releases
    ~5% of demand after the OD window (vehicles that never reach the sensors) and
    silently biases the count NRMSE (adr-034 Decision 3). The other three small
    instances have ``od_end_time == 3600 ==`` the template end, so the rewrite is
    a no-op there (measured delta 0.0). Writes the filled tazRelation XML.
    """
    flows: dict[tuple[str, str], str] = {}
    with open(single_od, newline="") as f:
        for row in csv.DictReader(f):
            flows[(row["fromTaz"], row["toTaz"])] = row["flow"]
    tree = ET.parse(od_template)
    for interval in tree.getroot().iter("interval"):
        interval.set("end", str(od_end_time))
    for rel in tree.getroot().iter("tazRelation"):
        key = (rel.get("from"), rel.get("to"))
        if key in flows:
            rel.set("count", flows[key])
    tree.write(out, encoding="utf-8", xml_declaration=True)


def bo4mob_pairs(od_template: str | Path) -> tuple[tuple[str, str], ...]:
    """The ordered active ``(fromTaz, toTaz)`` OD-cell layout for an instance.

    Read from the shipped ``od.xml`` template's ``tazRelation`` rows **in document
    order** — the fixed, hashed estimand layout a stage-2 OD estimator emits a
    vector over (adr-041). Ordering is load-bearing: two tasks whose estimands
    live in different OD cells must never hash equal (the adr-023 lesson carried
    to the string-keyed BO4Mob pairs), so this order feeds the task content hash
    and ``fill_od_from_vector`` uses the SAME order to place each flow.
    """
    tree = ET.parse(od_template)
    return tuple(
        (rel.get("from"), rel.get("to")) for rel in tree.getroot().iter("tazRelation")
    )


def bo4mob_prior_vector(
    od_template: str | Path, single_od: str | Path
) -> np.ndarray:
    """The prior OD vector (BO4Mob's ``od_for_single_run`` example) over ``bo4mob_pairs``.

    Aligns the ``single_od`` CSV flows to the template's ``tazRelation`` order
    (0.0 for any pair the CSV omits). This is the stage-1 anchor OD promoted to
    the stage-2 prior baseline's ``prior_vector``; filling ``od.xml`` from it
    reproduces ``fill_single_od`` exactly (adr-041 regression).
    """
    flows: dict[tuple[str, str], float] = {}
    with open(single_od, newline="") as f:
        for row in csv.DictReader(f):
            flows[(row["fromTaz"], row["toTaz"])] = float(row["flow"])
    pairs = bo4mob_pairs(od_template)
    return np.asarray([flows.get(p, 0.0) for p in pairs], dtype=np.float64)


def fill_od_from_vector(
    od_template: str | Path,
    pairs: tuple[tuple[str, str], ...],
    od_vector: np.ndarray,
    out: str | Path,
    od_end_time: int,
) -> None:
    """Fill the ``count=0`` ``od.xml`` template from an in-memory OD ``vector``.

    The vector-based analogue of :func:`fill_single_od` for a stage-2 estimator's
    emitted OD (adr-041): ``od_vector[k]`` is the flow on ``pairs[k]``. The
    interval ``end`` is rewritten to ``od_end_time`` **exactly as** ``fill_single_od``
    does — this is **load-bearing** and NOT cosmetic: keeping the template's
    ``end=3600`` on 1ramp (whose ``od_end_time=3300``) leaks ~5% of demand past
    the OD window and silently biases the count NRMSE (adr-034 Decision 3; pilot
    NRMSE 2.314704 unfixed vs 2.432471221214843 fixed). Any vector-fill that omits
    this rewrite re-introduces the demand-loss laundering vector, so it carries a
    duplicated regression test. Counts are written as plain floats stripped of a
    redundant ``.0`` so od2trips sees the same integer flows ``fill_single_od``
    wrote from the CSV.
    """
    vec = np.asarray(od_vector, dtype=np.float64)
    if vec.shape != (len(pairs),):
        raise ValueError(
            f"od_vector shape {vec.shape} != ({len(pairs)},) for the pair layout"
        )
    flows = {p: float(v) for p, v in zip(pairs, vec.tolist(), strict=True)}
    tree = ET.parse(od_template)
    for interval in tree.getroot().iter("interval"):
        interval.set("end", str(od_end_time))
    for rel in tree.getroot().iter("tazRelation"):
        key = (rel.get("from"), rel.get("to"))
        if key in flows:
            value = flows[key]
            rel.set("count", str(int(value)) if value == int(value) else repr(value))
    tree.write(out, encoding="utf-8", xml_declaration=True)


def fix_routes_single(trips_in: str | Path, routes_single: str | Path, out: str | Path) -> int:
    """Rewrite each trip's ``from``/``to`` to the single-route start/last edge.

    BO4Mob's route-fixing step for ``routes_per_od='single'`` (``sumo_runner.py``:
    ``update_trip_routes``). One route per OD pair, so the meso run is
    seed-stable. Trips are sorted by departure time and ``departLane='best'`` is
    set, exactly as upstream. Returns the number of trips kept.
    """
    table: dict[tuple[str, str], tuple[str, str]] = {}
    with open(routes_single, newline="") as f:
        for row in csv.DictReader(f):
            table[(row["fromTaz"], row["toTaz"])] = (row["start_edge"], row["last_edge"])
    trips: list[dict[str, str]] = []
    for _, elem in ET.iterparse(str(trips_in), events=("end",)):
        if elem.tag == "trip":
            trips.append(dict(elem.attrib))
            elem.clear()
    root = ET.Element("routes")
    kept = 0
    for trip in sorted(trips, key=lambda t: float(t["depart"])):
        key = (trip.get("fromTaz"), trip.get("toTaz"))
        if key not in table:
            continue
        start, last = table[key]
        ET.SubElement(
            root,
            "trip",
            {
                "id": trip["id"],
                "depart": trip["depart"],
                "from": start,
                "to": last,
                "type": trip.get("type", "DEFAULT_VEHTYPE"),
                "fromTaz": key[0],
                "toTaz": key[1],
                "departLane": "best",
            },
        )
        kept += 1
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)
    return kept


def local_edgedata_additional(
    additional: str | Path, out: str | Path, edge_data_name: str = "edge_data.xml"
) -> None:
    """Copy the additional file, redirecting ``<edgeData file=...>`` to a local
    ``edge_data_name`` in the working directory.

    The shipped path is ``../../edge_data.xml`` (relative to the network dir);
    redirecting it keeps the meso output inside the working directory. A
    per-run ``edge_data_name`` lets a caller give each run its own output file so
    a future engine that exits 0 without writing cannot silently re-read a stale
    file. The ``DEFAULT_VEHTYPE`` (IDM, ``speedDev=0``) is preserved
    attribute-exact (semantically identical; ElementTree normalises serialisation
    whitespace/quoting) — it is load-bearing for the mesoscopic run.
    """
    tree = ET.parse(additional)
    for ed in tree.getroot().iter("edgeData"):
        ed.set("file", edge_data_name)
    tree.write(out, encoding="utf-8", xml_declaration=True)


def edgedata_counts(
    edge_data: str | Path, sim_start: float, sim_end: float
) -> dict[str, float]:
    """Per-edge ``arrived + left`` summed over intervals within ``[start, end]``.

    BO4Mob's link-flow convention (``src/utils/link_flow_analysis.py``). Under
    SUMO 1.27.1 the mesoscopic ``edgeData`` has NO ``nVehContrib`` attribute
    (the measured 1.12 -> 1.27.1 schema drift, adr-034), but ``arrived`` and
    ``left`` still exist, so this convention transfers unchanged.
    """
    counts: dict[str, float] = {}
    for interval in ET.parse(edge_data).getroot().findall("interval"):
        begin = float(interval.get("begin", 0.0))
        end = float(interval.get("end", 0.0))
        if not (begin >= sim_start and end <= sim_end):
            continue
        for edge in interval.findall("edge"):
            arrived = float(edge.get("arrived", 0.0))
            left = float(edge.get("left", 0.0))
            eid = edge.get("id", "")
            counts[eid] = counts.get(eid, 0.0) + arrived + left
    return counts


def edgedata_has_nvehcontrib(edge_data: str | Path) -> bool:
    """True iff any edge carries an ``nVehContrib`` attribute (the SUMO 1.12
    mesoscopic schema).

    Under the shipped 1.27.1 wheel this is False — the measured schema drift
    adr-034 documents. It is why ``edgedata_counts`` uses ``arrived + left``
    (BO4Mob's own convention) rather than the absent ``nVehContrib``.
    """
    for edge in ET.parse(edge_data).getroot().iter("edge"):
        if edge.get("nVehContrib") is not None:
            return True
    return False


def bo4mob_nrmse(sensor: str | Path, counts: dict[str, float]) -> float:
    """BO4Mob count NRMSE: ``sqrt(n * sum((gt - sim)^2)) / sum(gt)``.

    ``n`` is the GT sensor count; simulated values absent from ``counts`` are
    filled with 0 (``link_flow_analysis.py``: a left merge). The GT CSV carries
    ``interval_nVehContrib`` (its column name is the PeMS-side count, unrelated to
    the SUMO 1.27.1 edgeData drift above).
    """
    ids: list[str] = []
    gt_vals: list[float] = []
    with open(sensor, newline="") as f:
        for row in csv.DictReader(f):
            ids.append(row["link_id"])
            gt_vals.append(float(row["interval_nVehContrib"]))
    gt = np.asarray(gt_vals, dtype=np.float64)
    sim = np.asarray([counts.get(i, 0.0) for i in ids], dtype=np.float64)
    n = gt.shape[0]
    return float(np.sqrt(n * np.sum((gt - sim) ** 2)) / gt.sum())


# --- Registry (measured in-sprint; see docs/design/adr-034 + VALIDATION.md) ------


def _small(
    key: str,
    netdir: str,
    *,
    n_od: int,
    n_sensors: int,
    net: tuple[str, int],
    taz: tuple[str, int],
    od: tuple[str, int],
    additional: tuple[str, int],
    routes_single: tuple[str, int],
    single_od: tuple[str, int],
    config: tuple[str, int],
    sensor: tuple[str, int],
) -> Bo4MobSpec:
    """Terse constructor for a fetchable small instance (the xu2024 ``_spec``).

    Each file argument is ``(sha256, byte_size)`` — both pinned at ``BO4MOB_COMMIT``.
    """
    anchor = f"gt_edge_data_{key}_{BO4MOB_ANCHOR_DATE}_{BO4MOB_ANCHOR_HOUR}.csv"
    files = {
        "net": (f"network/{netdir}/net.xml", *net),
        "taz": (f"network/{netdir}/taz.xml", *taz),
        "od": (f"network/{netdir}/od.xml", *od),
        "additional": (f"network/{netdir}/additional.xml", *additional),
        "routes_single": (f"network/{netdir}/routes_single.csv", *routes_single),
        "single_od": (f"od_for_single_run/od_{key}.csv", *single_od),
        "config": (f"config/sim_setup_{netdir}.json", *config),
        "sensor": (f"sensor_data/{BO4MOB_ANCHOR_DATE}/{anchor}", *sensor),
    }
    return Bo4MobSpec(
        key=key,
        netdir=netdir,
        n_od=n_od,
        n_sensors_anchor=n_sensors,
        hpc_only=False,
        files=files,
        notes=BO4MOB_ORIGIN,
    )


# Four small instances ship as fetchable single-evaluation bundles (dims match
# paper Table 1; sensor counts measured on the 221008 06-07 anchor). 5fullRegion
# (10,100 OD pairs, 74 MB, ~11 h/eval) is metadata-only and refuses to fetch.
BO4MOB_REGISTRY: dict[str, Bo4MobSpec] = {
    spec.key: spec
    for spec in (
        _small(
            "1ramp", "network_1ramp", n_od=3, n_sensors=3,
            net=("ce563b17e68272c56534f4acd4d4001a7b75daedceb4643cd2ec3de7322b01a3", 24923),
            taz=("6ab3ccbedd1b6e36a786a332c290099a240176aaa2b66f76c773f99af06ae296", 793),
            od=("28cdd9a883e01296832d9c15b3236838772078c42cf9ac6a38aae3b70d65e330", 229),
            additional=("119740c3eecb434cd21def8609af25a60ac4a706594fbdde60685af4d57861d8", 351),
            routes_single=("27c469d6b9b144731c382bf41654cffbd325a46bdc23eef511718039db51238f", 367),
            single_od=("5aa6607d7e34cfd712c4addb375248b8c25354057686505923965d5022b70469", 70),
            config=("4b157362730ff073317c1cdf050817777c97c003580ba824cfa1ac1bf30fb224", 468),
            sensor=("2f1cc8759cdddb81ab5b52b1d67409890b87216705921265570fd90329d89ba3", 126),
        ),
        _small(
            "2corridor", "network_2corridor", n_od=21, n_sensors=6,
            net=("72ac5fec2717c06d700513b8dcfc4f3244caac0e9b2cc437f0ddc0944126119e", 136685),
            taz=("6c0ec4dbfe3b9f6952ad4890a90a1f9c8ea2a47494e97c8a402c04381644c0ff", 2242),
            od=("f6343fce4b7a9c51266ba14d1ba7f90b3d5d99096081c84dca0eea9b2622c354", 1157),
            additional=("1d2ea2f81e3e5058ef78a588f4b92b0a64e9bfc397548696bf1bfd248b284ced", 352),
            routes_single=(
                "f6c00bdfc5751b8ab41ef0f5989bdf1f3e0e930a94cf347b036e0417ccb230af", 5284),
            single_od=("32855b5e7b6bc05ec2caecccfd9b5b70c3324777be8600bb58405f1672ae08af", 398),
            config=("9521fec94372aeeb210be520915f71981852e658cd61c2b724c44163d7cae503", 476),
            sensor=("0463c315a0a0fc37ad2d653cc6cfc62c387faa384b53834d49351de546061176", 220),
        ),
        _small(
            "3junction", "network_3junction", n_od=44, n_sensors=18,
            net=("b7d446ba4bdd84da646f1064ad2d63c3f94b3c658065142913e885e041d7b666", 343977),
            taz=("d46bd8ea752e5a4fef2e9439ef1830f75d3bf1d552c60eb887ca5d60ddb516e3", 3399),
            od=("2a2d39edd9710caceab73ae2458a75049166819f6796bddb45bcfc51fdb8398f", 2315),
            additional=("119740c3eecb434cd21def8609af25a60ac4a706594fbdde60685af4d57861d8", 351),
            routes_single=(
                "55db2e057daa41a883292463e619ac686fbcf068ed3ec73dcc45d43666987521", 9200),
            single_od=("0aaf3e86fc35816bb2f80358ee2c55cc21b7291a56b5d60717505462eb6865ab", 783),
            config=("550af8c5633417f13121ae292227ac0bce16620cedb176537e992306cdc8f45c", 477),
            sensor=("5b582485448b96e1c8fd10f6a3be534c15ef3d483ffdec68953f7cb00f7718e7", 553),
        ),
        _small(
            "4smallRegion", "network_4smallRegion", n_od=151, n_sensors=26,
            net=("8698a7bc56c5022c803f2c29545b26ad6eeaf9d2c83acd2305d4a2367608f171", 583484),
            taz=("23a261dfb792776dd23e4309c6b7a44e06fd47305b10e33508b2fe7cbedcb3c4", 6428),
            od=("132524b377d21f2b7aff7a8964d967fb8ce463288399c29c47819943d9eab542", 7826),
            additional=("119740c3eecb434cd21def8609af25a60ac4a706594fbdde60685af4d57861d8", 351),
            routes_single=(
                "b1d45bd049836c520925b78cbc09038c07a884d4e5f207911bb8190a17fc0e71", 41713),
            single_od=("89097d056df77a971b4b62479574bf1f93ba8d74cef9e3a1546d88d0c8c5993e", 2704),
            config=("ac8fc22b852800914683fa30ca17bdbccfe79e8c34ead63428322e7631b28df1", 481),
            sensor=("46c1547ade13e9e8f556d01a654ffcfd68768190f104c92c13db8d9a832458f2", 710),
        ),
        Bo4MobSpec(
            key="5fullRegion",
            netdir="network_5fullRegion",
            n_od=10100,
            n_sensors_anchor=219,  # paper Table 1 (not fetched: metadata-only)
            hpc_only=True,
            files={},
            notes=(
                BO4MOB_ORIGIN + " HPC-only: 74 MB, ~11 h/eval — registered "
                "metadata-only, fetch refused (adr-034)."
            ),
        ),
    )
}

# The single instance whose end-to-end pipeline the guarded CI smoke test runs
# (the smallest; 27 KB single-evaluation bundle, < 0.5 s engine time).
BO4MOB_SMOKE = "1ramp"


# --- Held-out panel (stage 2 observational certificate; adr-041) -----------------
# A stage-2 estimator emits ONE fixed OD from the TRAIN anchor (221008, 06-07);
# the certifier re-simulates it ONCE and scores the resulting link counts against
# each held-out date's real PeMS counts. The ranking column ``heldout_nrmse`` is
# the MEAN of the per-date NRMSE (framing b: one meso run per certify regardless
# of how many held-out dates are scored). This is a SEPARATE, checksummed,
# download-on-demand panel — NEVER in the CI-prefetched ``REGISTRY`` — and (P7)
# the held-out CSV bytes and date strings NEVER enter ``Bo4MobEstimationTask``;
# only their sha256 ``heldout_digest`` does. The hour window is HELD FIXED: PeMS
# counts vary enormously across the day (the anchor 06-07 totals ~2121 vs a
# 17-18 window ~7700), so a single fixed OD only represents one hour's demand and
# held-out probes same-hour, different-DATE generalisation (measured pilot spread
# on 1ramp's prior OD: per-date NRMSE 0.56-3.30, mean 1.70 over the 13 dates —
# non-vacuous and improvable, and the mean-over-13 is stable by construction).
BO4MOB_HELDOUT_HOUR = BO4MOB_ANCHOR_HOUR  # "06-07"
# The 13 consecutive dates AFTER the TRAIN anchor 221008 (which stays in TRAIN
# for stage-1 continuity); held-out is temporally disjoint from train.
BO4MOB_HELDOUT_DATES: tuple[str, ...] = tuple(f"2210{d:02d}" for d in range(9, 22))

# instance key -> ((date, sha256, byte_size), ...) for that instance's held-out
# sensor CSVs at ``BO4MOB_HELDOUT_HOUR`` (commit-pinned at ``BO4MOB_COMMIT``, P9;
# dates align to ``BO4MOB_HELDOUT_DATES``). 5fullRegion stays HPC-only: no panel.
BO4MOB_HELDOUT: dict[str, tuple[tuple[str, str, int], ...]] = {
    "1ramp": (
        ("221009", "3a952307bfffc677cbacdfca401e5d2826bfa8abd7372bdfe6276ca51a24bb4d", 128),
        ("221010", "9ada4679524031049759567fa7c63e3b365d94c676b2b7817b4da2485e7f5569", 126),
        ("221011", "3d1dc688f550f87b41bd2fd8f768ae82ad1a54f6c54cf503aec9c2bad2261b6b", 130),
        ("221012", "3d8218242a06b8cdb2313e22e81937d565814c6e15a2e008098aadea54d410eb", 128),
        ("221013", "ef675690cb246dd0038b4122183526397dc2f21e099fe7f032ed271ee85497b9", 131),
        ("221014", "a7a5e64779b7efb5f102b95ac8e3c91ceb8132abb8a78363a42f4c7c11f7288f", 130),
        ("221015", "e2617f1ea7658fb7ec59ba15a776a0e0a8565d281d9791c44d61cf72012b9ca8", 129),
        ("221016", "82ffb57d9ada27d07a33ae7a4efe5968742e397d15d896fe9e67bc401bbaa318", 127),
        ("221017", "fd13434aa3b63888c316c44ec280b463d993d8f03e470e2a51244620b0ea9110", 127),
        ("221018", "05efdd04caea1b43a568eec61a6482653546a998912ce923bebe6d70f5ff6d03", 128),
        ("221019", "3c64913bc94fab44126c2f89950f6811eee7ba8ea2232b4a82a6e165d6086116", 131),
        ("221020", "1109da1f46c1f053239252a42ec828107a05a2d67d853bbebb07dcc6c1671b31", 131),
        ("221021", "7aed16cc9818664e658cda961d38d95a20e68015122f2a933c3af191d604fa36", 131),
    ),
    "2corridor": (
        ("221009", "d1162b97a82d6e6c62536501c04237c8eaed826c02accd537d9e8fe8f71eb391", 188),
        ("221010", "42014b185a869e08df180e6ad5e68115da325aaf462e55341503f0d53d5581cf", 196),
        ("221011", "1f5a6c1393b84c5f1cc5d31b637e50376ac18944a492dd30a9771e7b30dd1abd", 194),
        ("221012", "2ce9c98ab6bae698caa3f6a6f9808bf551aa9d77bed0d6d42f136773869f4a3a", 196),
        ("221013", "605bb4a3cb26ffdd705db25f0e98bc70b7d62aaf8a3d1b7aa2bb667762cd0b80", 192),
        ("221014", "d1eda01ce34b714ccffc01082accdf2c6a5e3a0db0bb1cd0e725aafa57a96838", 169),
        ("221015", "5603289c6bef674bc4bd510b081071fb79c5731411a086385fe68d7aa1e9e2d2", 170),
        ("221016", "9fc2390dcedb205755c1ee44b4dd1a82cb7d5a5f0e0610fd3601bde1df8a6e9b", 165),
        ("221017", "f464c752799319e6385af95de7a9f9231c98930b024e7bbf37c7694843c871c8", 169),
        ("221018", "7564cb847931025ce9547f2a08c9dfd46cdb52ccb0553348bd7f22042d018790", 154),
        ("221019", "8e1d9be9415ca9cfa8109783ce202e5d4f12a52af76a44b045c15ba006ca1ebe", 170),
        ("221020", "9f67f9804a47217d10241b2da31ae2848a6feb621a2cd56d200fdfb1437ae288", 189),
        ("221021", "1ee8160af1d33b3fb43e2be81c022403fc69bafdec38fa7dcd4e27a28e453194", 196),
    ),
    "3junction": (
        ("221009", "252f0d17f7d248380c6400eb5182b0841775992dd34a12403591318a56c63938", 509),
        ("221010", "1b230ba33d6ebe0d7fc654112c9465c8019d51613941cc1331dde382391ded25", 521),
        ("221011", "d0f6372047032992e0a01ba4c4b1d0efcf8a5fda52e35213b7bf2d102951e3e4", 549),
        ("221012", "f07a48871b2e1a1c8e5358f3b7e5c91501b6ffa9ad46aa1fd2236136e05b7000", 584),
        ("221013", "b91ce8d511be6ea1506a326e3e800cb489741316b8472aceb25be3bbed44b942", 556),
        ("221014", "5199a4511e49cd04c8cbe5b7432bda2b913452a69c5dff9cd0b3bf5364d5ff73", 548),
        ("221015", "b72d222336471fe90d3b959f401c4735ea88db632b4154261fb63c51b859e915", 553),
        ("221016", "d2aec048d0de37690771c69e7b249c1500f93b6e2c80eb61c1aba6f60a2e23de", 512),
        ("221017", "81c05fa1630e44bed827a0f5998b1c73e4d105fb649530d65e9e0cfa528abd78", 550),
        ("221018", "fac8121395be4b4ece1a2bfe7da6ea1ccacf37b33a9d6fe3585bba824a093378", 549),
        ("221019", "2b3f114df15d2edb15a78f675d6abc7f8c4d51f8f09484bb22c223b3647c18a2", 529),
        ("221020", "fd11dbd4c607b4484d7639612365f9c66a85d14d36a33edf2a597e735321d64b", 543),
        ("221021", "ac644983ce7a48727891eedca378d9da2f2ec7794c829f371860964a73eaf218", 595),
    ),
    "4smallRegion": (
        ("221009", "127100ce04d46acb26253575ed1ea8180941b068d99e4864493789b988a3258d", 736),
        ("221010", "c681b692bf319431c7a5fa55efe76729afbe093e0afb283f75a227902742f8f1", 716),
        ("221011", "911d7d31554fcedddde690085c0365455c8bc552bcf23e4f46b27e4aef9a91bb", 749),
        ("221012", "664a14b842d3256c77894cf178fa2aefe7c2eae400e7853e9b00730c89eadb33", 740),
        ("221013", "3a07cb433a04e665d5555d3c9abf2869260b53dcf0a30b56ac7d0f39dc52ee40", 743),
        ("221014", "a460ee07484937711efa8518ec79e728297b88b112f55361c49ddddeb1b28900", 741),
        ("221015", "68795d3ffbdc8b0ab1eebf5570b35552464c29ff6617db26a833f7d3a954ee33", 738),
        ("221016", "63c36150c7dabcd69a0aa9a98e403d2e6d4372392664a7d7753786088198dd7c", 758),
        ("221017", "d0b953d63af52974459f0a9b6d14278654481087405b2ef3657078718c1ac02a", 743),
        ("221018", "6af03c2abd1b8540824b75af109f2393fcc4a005e460808300b01b821164d50a", 749),
        ("221019", "c6fd8b8a38b4aac9f2f88d91eaf5e4d87586ae1f0a2db37d430bc6298bb4d6b7", 722),
        ("221020", "24713782cbd887775513c7028120a3182316532e4ddf5c471883057415dcfadb", 802),
        ("221021", "6e4933539dbd610013abc4c4e9a045ca43678aba61c31f84bbed9e9ef2ec1d53", 805),
    ),
}


def fetch_bo4mob_heldout(
    key: str, force: bool = False, timeout: float = 60.0
) -> dict[str, Path]:
    """Fetch + verify an instance's held-out sensor panel (``date -> local path``).

    The stage-2 held-out counts (``BO4MOB_HELDOUT_HOUR``, ``BO4MOB_HELDOUT_DATES``)
    with the SAME hardening as :func:`fetch_bo4mob` (streamed cap, per-process
    ``.part`` hygiene, per-load SHA-256 eviction), cached under
    ``~/.cache/tabench/bo4mob/<key>/heldout/``. A SEPARATE registry that is NEVER
    CI-prefetched; only the sumo-gated bo4mob_estimation certifier pulls it. An
    HPC-only instance (``5fullRegion``) refuses with the SAME ``Bo4MobHpcOnlyError``
    class — defense-in-depth on top of the single-evaluation fetcher's refusal.
    """
    spec = BO4MOB_REGISTRY.get(key)
    if spec is not None and spec.hpc_only:
        raise Bo4MobHpcOnlyError(
            f"bo4mob {key} is HPC-only and is registered metadata-only; it has no "
            "held-out panel and refuses to fetch (adr-034/041)."
        )
    if key not in BO4MOB_HELDOUT:
        raise KeyError(f"bo4mob {key!r} has no registered held-out panel (adr-041)")
    target_dir = cache_dir() / "bo4mob" / key / "heldout"
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for date, checksum, size in BO4MOB_HELDOUT[key]:
        name = f"gt_edge_data_{key}_{date}_{BO4MOB_HELDOUT_HOUR}.csv"
        local = target_dir / name
        url = f"{BO4MOB_BASE}/sensor_data/{date}/{name}"
        _fetch_checked(local, url, checksum, size, force, timeout, f"{key}/heldout/{name}")
        paths[date] = local
    return paths
