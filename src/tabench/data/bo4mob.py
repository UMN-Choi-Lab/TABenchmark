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
    "BO4MOB_REGISTRY",
    "BO4MOB_SMOKE",
    "Bo4MobHpcOnlyError",
    "Bo4MobSpec",
    "Bo4MobUpstreamError",
    "bo4mob_citation",
    "bo4mob_nrmse",
    "edgedata_counts",
    "edgedata_has_nvehcontrib",
    "fetch_bo4mob",
    "fill_single_od",
    "fix_routes_single",
    "local_edgedata_additional",
]

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
        if force or not local.exists():
            url = f"{BO4MOB_BASE}/{repo_path}"
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
                                f"bo4mob/{spec.key}/{spec.local_name(role)}: upstream body "
                                f"exceeds the pinned size {size} B (+{_SIZE_SLACK} slack); "
                                "refusing before materialising an oversized file (adr-034)."
                            )
                        out.write(chunk)
                tmp.replace(local)
            finally:  # a mid-download failure must not strand a .part (xu2024 lesson)
                tmp.unlink(missing_ok=True)
        actual = _sha256(local)
        if actual != checksum:
            local.unlink(missing_ok=True)
            raise ChecksumError(
                f"bo4mob/{spec.key}/{spec.local_name(role)}: checksum mismatch "
                f"(expected {checksum[:12]}…, got {actual[:12]}…). "
                "File removed from cache; re-run to re-download."
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
