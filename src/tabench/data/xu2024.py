"""Xu et al. (2024) 20-US-city traffic-assignment dataset (P9; adr-033).

A CC-BY, OSM-derived cross-domain scenario family — real city networks with
computed user-equilibrium flows, disjoint from the classic TNTP instances
(the cross-domain axis promised in adr-006). Source: figshare deposit
``10.6084/m9.figshare.24235696`` (v4), one 276 MB zip; paper Xu, Zheng, Hu,
Feng & Ma, *Scientific Data* 11:325 (2024), DOI 10.1038/s41597-024-03149-8.

Like all benchmark data, it is never vendored (P9): only the per-city
AequilibraE trio — ``network.csv`` + ``od_demand.aem`` + ``assignment_result.csv``
— is fetched, by HTTP byte-range extraction of exactly those three zip members
(``fetch_city``), checksum-verified, and cached. The full 276 MB zip is
downloaded only as a documented fallback when the server refuses range requests
(a transport error) — never on a structural or integrity error (a wrong member,
a corrupt directory, a changed upstream artifact, or a checksum mismatch), which
fail fast. Both paths guard the probed/declared size against the pinned v4 size
before any bulk transfer.

KNOWN DEFECT — the shipped "AS-PUBLISHED" instances (adr-033, disclosed per P2).
The dataset's published AequilibraE runs injected the OD demand at node ids
``1..Z`` instead of the intended tract centroids (median 6.4 km off; root cause
is the authors' own ``AequilibraE_assignment.py`` calling
``g.prepare_graph(np.arange(zones)+1)``). The instances built here reproduce
that as-published graph: they are self-consistent — the published flows conserve
machine-exactly and equilibrate on the ``1..Z``-centroid network — but they are a
*different* instance from the dataset's TransCAD side, which used the correct
centroids. **Cross-solver agreement against the published TransCAD flows is
therefore never claimed.** The published AequilibraE flows are a LOOSE reference
(their own relative gap is ~1e-3, ~11 orders looser than a TNTP best-known),
used only as a provenance cross-check, never as a best-known oracle. The
corrected tract-centroid variant is deferred (adr-033, future work).
"""

from __future__ import annotations

import csv
import hashlib
import http.client
import io
import struct
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.scenario import Demand, Network, ReferenceSolution, Scenario
from .fetcher import ChecksumError, _sha256, cache_dir

__all__ = [
    "Xu2024CitySpec",
    "Xu2024UpstreamError",
    "XU2024_REGISTRY",
    "XU2024_RUNGS",
    "fetch_city",
    "xu2024_citation",
    "xu2024_scenario",
]


class Xu2024UpstreamError(RuntimeError):
    """The figshare artifact is not the pinned v4 (size/identity changed).

    Raised *before* any bulk transfer so a re-deposited or moved artifact fails
    loudly rather than being partially fetched. A structural error — never a
    trigger for the whole-zip fallback (downloading it would hit the same
    changed artifact)."""

# --- Provenance (version-immutable figshare file id; pin v4) --------------------
FIGSHARE_DOI = "10.6084/m9.figshare.24235696"
FIGSHARE_VERSION = 4
ZIP_FILE_ID = "48908890"
ZIP_URL = f"https://ndownloader.figshare.com/files/{ZIP_FILE_ID}"
ZIP_MD5 = "3f7632e00599588abecbcfc488f862b2"  # whole-zip, fallback path only
ZIP_SIZE = 276_047_471
LICENSE = "CC BY 4.0"
AEQUILIBRAE_SUBDIR = "03_AequilibraE_results"
_MEMBER_BASENAMES = {
    "network": "network.csv",
    "aem": "od_demand.aem",
    "assignment": "assignment_result.csv",
}


@dataclass(frozen=True)
class Xu2024CitySpec:
    """Provenance + measured metadata for one city's AequilibraE trio.

    ``files`` maps role -> (member basename, sha256 of the extracted bytes). The
    dimensions, BPR parameters, demand total and published relative gap are the
    in-sprint audit measurements (recorded so tests and docs are unit- and
    download-free), not values read off the paper.
    """

    key: str
    dir_name: str
    files: dict[str, tuple[str, str]]
    n_zones: int
    n_nodes: int
    n_links: int
    demand_total: float
    bpr_alpha: float
    bpr_beta: float
    published_relative_gap: float

    def member(self, role: str) -> str:
        """Full zip-member path for ``role`` (forward slashes, as in the zip)."""
        return f"{self.dir_name}/{AEQUILIBRAE_SUBDIR}/{self.files[role][0]}"


# --- Ranged download-on-demand --------------------------------------------------


class _HttpRangeFile(io.RawIOBase):
    """Minimal seekable read-only file over HTTP ``Range`` requests.

    Lets ``zipfile`` read the central directory from the zip's tail and inflate
    only the requested members, so a single-city load transfers ~1-4 MB rather
    than the 276 MB archive. The figshare download URL redirects to S3, which
    honors ``Range`` (probe-verified); the total size is read once from the
    ``Content-Range`` header.
    """

    def __init__(
        self,
        url: str,
        timeout: float = 120.0,
        retries: int = 4,
        expected_size: int | None = None,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self._pos = 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_range = resp.headers.get("Content-Range")
            if not content_range or "/" not in content_range:
                raise OSError("server did not honor a Range request (no Content-Range)")
            self.size = int(content_range.rsplit("/", 1)[1])
        # Pre-transfer identity guard: the probed total must match the pinned v4
        # size before we range-read a single member (adr-033).
        if expected_size is not None and self.size != expected_size:
            raise Xu2024UpstreamError(
                f"upstream artifact changed (expected v4, {expected_size} bytes; "
                f"probe reported {self.size}). Refusing to fetch."
            )

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self.size + offset
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self._pos
        if size == 0 or self._pos >= self.size:
            return b""
        end = min(self._pos + size, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self._pos}-{end}"})
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = resp.read()
                break
            except Exception:  # transient network / S3 hiccup: retry with backoff
                if attempt == self.retries - 1:
                    raise
                time.sleep(2.0 * (attempt + 1))
        self._pos += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def _extract_ranged(spec: Xu2024CitySpec, roles: list[str], timeout: float) -> dict[str, bytes]:
    """Extract the requested members by HTTP range (the primary path)."""
    with zipfile.ZipFile(_HttpRangeFile(ZIP_URL, timeout, expected_size=ZIP_SIZE)) as zf:
        return {role: zf.read(spec.member(role)) for role in roles}


def _extract_whole_zip(spec: Xu2024CitySpec, roles: list[str], timeout: float) -> dict[str, bytes]:
    """Fallback: download the whole 276 MB zip, md5-verify, then extract.

    Used ONLY when the server refuses range requests (never on a structural or
    integrity error — those fail fast in ``fetch_city``). The declared
    ``Content-Length`` is checked against the pinned v4 size before any body is
    read, the whole-zip md5 is pinned, and the ``.part`` is always removed
    (``try/finally``) so a mid-stream failure never strands an unbounded file in
    the cache CI persists.
    """
    target = cache_dir() / "xu2024" / f"_{ZIP_FILE_ID}.zip.part"
    md5 = hashlib.md5()
    try:
        with urllib.request.urlopen(ZIP_URL, timeout=timeout) as resp:
            length = resp.headers.get("Content-Length")
            if length is not None and int(length) != ZIP_SIZE:
                raise Xu2024UpstreamError(
                    f"upstream artifact changed (expected v4, {ZIP_SIZE} bytes; "
                    f"Content-Length {length}). Refusing to download."
                )
            with open(target, "wb") as out:
                for chunk in iter(lambda: resp.read(1 << 20), b""):
                    md5.update(chunk)
                    out.write(chunk)
        if md5.hexdigest() != ZIP_MD5:
            raise ChecksumError(
                f"xu2024 whole-zip md5 mismatch (expected {ZIP_MD5[:12]}…, "
                f"got {md5.hexdigest()[:12]}…); download removed."
            )
        with zipfile.ZipFile(target) as zf:
            return {role: zf.read(spec.member(role)) for role in roles}
    finally:
        target.unlink(missing_ok=True)


def fetch_city(
    spec: Xu2024CitySpec, force: bool = False, timeout: float = 120.0
) -> dict[str, Path]:
    """Ensure a city's AequilibraE trio is cached and checksum-verified.

    Returns ``role -> local path`` (roles ``network``, ``aem``, ``assignment``).
    Missing members are extracted by HTTP range from the figshare zip; every
    extracted file is verified against its pinned SHA-256. The whole-zip fallback
    fires **only** on a range-refusal / transport error — a structural or
    integrity failure (a wrong member name, a corrupt central directory, a
    changed upstream artifact, or a checksum mismatch) is re-raised immediately,
    since downloading the 276 MB zip cannot fix it (adr-033).
    """
    target_dir = cache_dir() / "xu2024" / spec.key
    target_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        role for role, (base, _) in spec.files.items() if force or not (target_dir / base).exists()
    ]
    if missing:
        try:
            extracted = _extract_ranged(spec, missing, timeout)
        except (ChecksumError, Xu2024UpstreamError, KeyError, zipfile.BadZipFile):
            # Structural / integrity: the whole-zip fallback would hit the same
            # wrong member / changed artifact — fail fast, do not download 276 MB.
            raise
        except (OSError, http.client.HTTPException):
            # Range-refusal or transient transport only: the documented fallback.
            extracted = _extract_whole_zip(spec, missing, timeout)
        for role in missing:
            base, _ = spec.files[role]
            local = target_dir / base
            tmp = local.with_suffix(local.suffix + ".part")
            tmp.write_bytes(extracted[role])
            tmp.replace(local)
    paths: dict[str, Path] = {}
    for role, (base, checksum) in spec.files.items():
        local = target_dir / base
        actual = _sha256(local)
        if actual != checksum:
            local.unlink(missing_ok=True)
            raise ChecksumError(
                f"xu2024/{spec.key}/{base}: checksum mismatch "
                f"(expected {checksum[:12]}…, got {actual[:12]}…). "
                "File removed from cache; re-run to re-download."
            )
        paths[role] = local
    return paths


# --- Parsers (numpy/stdlib only; the benchmark core stays pandas-free) ----------


def _parse_network_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Parse an AequilibraE ``network.csv`` into link-table arrays.

    Columns: an unnamed row index, then ``a_node,b_node,capacity,free_flow_time,
    a,b,direction,link_id``, where ``a``/``b`` are the BPR alpha/beta and
    ``direction`` is 1 (fully directed) throughout this dataset.
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    col = {name: i for i, name in enumerate(header)}
    required = ("a_node", "b_node", "capacity", "free_flow_time", "a", "b", "direction", "link_id")
    missing = [c for c in required if c not in col]
    if missing:
        raise ValueError(f"{path}: network.csv missing columns {missing}")

    def column(name: str, dtype) -> np.ndarray:
        return np.array([row[col[name]] for row in rows], dtype=dtype)

    parsed = {
        "a_node": column("a_node", np.int64),
        "b_node": column("b_node", np.int64),
        "capacity": column("capacity", np.float64),
        "free_flow_time": column("free_flow_time", np.float64),
        "alpha": column("a", np.float64),
        "beta": column("b", np.float64),
        "direction": column("direction", np.int64),
        "link_id": column("link_id", np.int64),
    }
    # Drift guard: this dataset is fully directed (direction == 1 on every link);
    # the AS-PUBLISHED builder assumes it (each row is one directed link). A
    # future re-deposit with two-way links would silently mis-map without this.
    if not np.all(parsed["direction"] == 1):
        raise ValueError(
            f"{path}: network.csv has direction != 1 on "
            f"{int((parsed['direction'] != 1).sum())} link(s); the fully-directed "
            "AS-PUBLISHED convention (adr-033) does not hold for this file"
        )
    return parsed


def _parse_aem(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse an AequilibraE ``.aem`` matrix into ``(zone_index, matrix)``.

    Layout (reverse-engineered and file-size-validated): a 252-byte header
    carrying ``uint32`` cell count at offset 2 and ``uint32`` zone count at
    offset 10, then ``int64`` zone ids ``[Z]`` and the ``float64`` matrix
    ``[Z, Z]`` in row-major order.
    """
    raw = Path(path).read_bytes()
    if len(raw) < 252:
        raise ValueError(f"{path}: .aem shorter than its 252-byte header")
    cells = struct.unpack_from("<I", raw, 2)[0]
    zones = struct.unpack_from("<I", raw, 10)[0]
    expected = 252 + 8 * zones + 8 * zones * zones
    if cells != zones * zones or len(raw) != expected:
        raise ValueError(
            f"{path}: .aem header inconsistent (zones={zones}, cells={cells}, "
            f"size={len(raw)}, expected {expected})"
        )
    index = np.frombuffer(raw, dtype="<i8", count=zones, offset=252)
    # Drift guard: the AS-PUBLISHED centroid convention (adr-033) requires the
    # matrix zones to be exactly node ids 1..Z, in order. If a future re-deposit
    # renumbers zones, the demand<->centroid mapping would silently shift.
    if not np.array_equal(index, np.arange(1, zones + 1)):
        raise ValueError(
            f"{path}: .aem zone index is not 1..{zones}; the AS-PUBLISHED "
            "1..Z-centroid convention (adr-033) does not hold for this file"
        )
    matrix = np.frombuffer(
        raw, dtype="<f8", count=zones * zones, offset=252 + 8 * zones
    ).reshape(zones, zones)
    return index.copy(), matrix.copy()


def _parse_assignment_result(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse ``assignment_result.csv`` into ``(link_id, matrix_ab)``.

    ``matrix_ab`` is the published AequilibraE UE flow on each directed link;
    ``matrix_ba`` is blank on this fully-directed dataset (read as 0).
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    col = {name: i for i, name in enumerate(header)}
    for c in ("link_id", "matrix_ab"):
        if c not in col:
            raise ValueError(f"{path}: assignment_result.csv missing column {c!r}")
    link_id = np.array([int(row[col["link_id"]]) for row in rows], dtype=np.int64)
    ab = np.array(
        [float(row[col["matrix_ab"]]) if row[col["matrix_ab"]].strip() else 0.0 for row in rows],
        dtype=np.float64,
    )
    return link_id, ab


# --- Scenario builder -----------------------------------------------------------


def _renumber(nodes: np.ndarray, n_zones: int) -> dict[int, int]:
    """Map raw node ids to 1-based ids with centroids first.

    The as-published AequilibraE graph uses node ids ``1..Z`` as centroids
    (the known defect). TABenchmark requires zones to be nodes ``1..n_zones``,
    so those keep their ids and every other raw id (which includes 0 and the
    ``10000000+k`` tract ids) is mapped to ``Z+1..N`` in sorted order. Raises if
    a centroid id ``1..Z`` is absent from the network (an unconnected zone).
    """
    present = set(int(n) for n in nodes.tolist())
    missing = [k for k in range(1, n_zones + 1) if k not in present]
    if missing:
        raise ValueError(
            f"centroid node ids {missing[:5]} (of 1..{n_zones}) are absent from the "
            "network; the as-published centroid convention does not hold for this city"
        )
    remap = {k: k for k in range(1, n_zones + 1)}
    others = sorted(n for n in present if n > n_zones or n < 1)
    for offset, raw in enumerate(others):
        remap[raw] = n_zones + 1 + offset
    return remap


def _build_scenario(
    spec: Xu2024CitySpec,
    net: dict[str, np.ndarray],
    matrix: np.ndarray,
    link_id: np.ndarray,
    ab: np.ndarray,
) -> Scenario:
    n_zones = matrix.shape[0]
    nodes = np.unique(np.concatenate([net["a_node"], net["b_node"]]))
    remap = _renumber(nodes, n_zones)
    init = np.array([remap[int(x)] for x in net["a_node"].tolist()], dtype=np.int64)
    term = np.array([remap[int(x)] for x in net["b_node"].tolist()], dtype=np.int64)
    n_links = init.shape[0]
    network = Network(
        name=f"xu2024-{spec.key}",
        n_nodes=len(nodes),
        n_zones=n_zones,
        # first_thru_node=1: the as-published run allowed through-centroid flow,
        # and the low-id "centroids" are real intersections, not pure zone stubs.
        first_thru_node=1,
        init_node=init,
        term_node=term,
        capacity=net["capacity"],
        length=np.zeros(n_links),
        free_flow_time=net["free_flow_time"],
        b=net["alpha"],  # AequilibraE column ``a`` is BPR alpha
        power=net["beta"],  # AequilibraE column ``b`` is BPR beta
        toll=np.zeros(n_links),
        link_type=np.ones(n_links, dtype=np.int64),
        units=(
            ("free_flow_time", "minutes"),
            ("capacity", "vehicles/hour"),
            ("demand", "vehicles/hour AM peak (0.6 x the input od.csv)"),
            ("coordinates", "WGS84 lon/lat (not carried; plotting only)"),
        ),
    )
    od = matrix.copy()
    np.fill_diagonal(od, 0.0)  # intrazonal demand never enters the network
    demand = Demand(matrix=od)

    order = np.argsort(link_id)
    if not np.array_equal(link_id[order], np.arange(1, n_links + 1)):
        raise ValueError(f"xu2024-{spec.key}: assignment link_ids are not 1..{n_links}")
    flows = ab[order]
    reference = ReferenceSolution(
        link_flows=flows,
        source=xu2024_citation(spec),
        note=(
            "Published AequilibraE UE link flows (matrix_ab), own relative gap "
            "~1e-3 — a LOOSE published reference, NOT a best-known oracle. "
            "AS-PUBLISHED wrong-centroid instance (demand injected at node ids "
            "1..Z, not the tract centroids; adr-033). Cross-solver agreement "
            "against the dataset's TransCAD flows is not claimed."
        ),
    )
    return Scenario(
        name=f"xu2024-{spec.key}",
        network=network,
        demand=demand,
        reference=reference,
        family=f"xu2024-{spec.key}",
    )


def xu2024_scenario(city: str, force: bool = False) -> Scenario:
    """Load the AS-PUBLISHED xu2024 scenario for ``city`` (downloading if needed)."""
    if city not in XU2024_REGISTRY:
        raise KeyError(f"Unknown xu2024 city {city!r}; available: {sorted(XU2024_REGISTRY)}")
    spec = XU2024_REGISTRY[city]
    paths = fetch_city(spec, force=force)
    net = _parse_network_csv(paths["network"])
    _, matrix = _parse_aem(paths["aem"])
    link_id, ab = _parse_assignment_result(paths["assignment"])
    return _build_scenario(spec, net, matrix, link_id, ab)


def xu2024_citation(spec: Xu2024CitySpec | None = None) -> str:
    """Mandatory CC-BY attribution string for the xu2024 data source."""
    base = (
        "Xu, X., Zheng, Z., Hu, Z., Feng, K. & Ma, W. (2024). A unified dataset for the "
        "city-scale traffic assignment model in 20 U.S. cities. Scientific Data 11:325, "
        "DOI 10.1038/s41597-024-03149-8. Data: figshare "
        f"https://doi.org/{FIGSHARE_DOI} (v{FIGSHARE_VERSION}, file {ZIP_FILE_ID}), "
        f"licensed {LICENSE}."
    )
    if spec is not None:
        return f"{base} City: {spec.dir_name} (03_AequilibraE_results)."
    return base


# --- Registry (measured in-sprint; see docs/design/adr-033 + VALIDATION.md) -----


def _spec(
    key: str,
    dir_name: str,
    *,
    net: str,
    aem: str,
    asg: str,
    zones: int,
    nodes: int,
    links: int,
    demand: float,
    alpha: float,
    beta: float,
    gap: float,
) -> Xu2024CitySpec:
    """Terse constructor for a city spec (17 of them; keeps the table readable)."""
    return Xu2024CitySpec(
        key=key,
        dir_name=dir_name,
        files={
            "network": ("network.csv", net),
            "aem": ("od_demand.aem", aem),
            "assignment": ("assignment_result.csv", asg),
        },
        n_zones=zones,
        n_nodes=nodes,
        n_links=links,
        demand_total=demand,
        bpr_alpha=alpha,
        bpr_beta=beta,
        published_relative_gap=gap,
    )


# 17 of the dataset's 20 cities pass the in-sprint builder audit (adr-033).
# Excluded and named: 3 cities whose as-published 1..Z-centroid graph is
# internally inconsistent — a few centroid ids in 1..Z are absent from the
# network as low-id nodes (they exist only as their 10000000+k tract ids) yet
# still carry demand, so no valid Network can be built on the 1..Z convention.
# Further evidence of the wrong-centroid defect; candidates for the deferred
# corrected-centroid variant (adr-033).
XU2024_EXCLUDED = {
    "washington": "2 of 179 centroids (ids 120, 121) absent as low-id nodes",
    "pittsburgh": "5 of 149 centroids (ids 61-63, 101, 102) absent as low-id nodes",
    "phoenix": "4 of 378 centroids (ids 140-143) absent as low-id nodes",
}

# CI-sized rungs (small enough to fetch + certify in CI); the other 15 cities
# build via the same path but are local-only by download/solve cost.
XU2024_RUNGS = ("honolulu", "sanfrancisco")

XU2024_REGISTRY: dict[str, Xu2024CitySpec] = {
    spec.key: spec
    for spec in (
    _spec(
        "sanfrancisco", "01_San_Francisco",
        net="a1701aaf4ea970a23dd9015a60412ae2ee2a43a70584db98f8de289487555ed5",
        aem="2d7595acb29b30d7dffa0bcb8b3779a93698716d088f895f5c1124cf38b743b6",
        asg="fd7720c74b2af80a8e250492fa93e1b2dd874defd59ea5092a7b805cdd8fab8f",
        zones=194, nodes=4986, links=18002, demand=168828.0, alpha=0.5, beta=1.8, gap=1.265e-03,
    ),
    _spec(
        "seattle", "02_Seattle",
        net="08d1da9dc28a391ca3a97a5f381acc847e7cd5083683d88481d651c03c68639d",
        aem="d3a397f0aa5a4b0bc24426be3f74f2f7ec148c5f233ccb219ed21576a47cbc4b",
        asg="ee11498fba708a4151b82fe7277a23dfa148ec3cc18676188b3c8b630dbae98e",
        zones=139, nodes=6891, links=27361, demand=133899.6, alpha=0.6, beta=3.0, gap=9.179e-04,
    ),
    _spec(
        "portland", "03_Portland",
        net="7e0ea3cce3054b681671d527fbe825a20ec7b999076c7ec1392bb941818802c3",
        aem="b7c5372e877e68e00ad8f38fb2ee4048e4952ad92ece5c372425d5eb6cece0aa",
        asg="27ca385409852f91671f9aedc403d8dd888dcc5d8420dc7b16275032eba985c6",
        zones=157, nodes=8245, links=31939, demand=136470.6, alpha=0.5, beta=1.2, gap=8.968e-04,
    ),
    _spec(
        "lasvegas", "04_Las_Vegas",
        net="360d9eeb0966d5bdbffd006873a5d2b512790bf49d0cb505a02da77047cfbd4d",
        aem="190a2b2663e4e613f53cf2ef0fa9b6ba6018d413bc70f0fea8028ba2363f1a2f",
        asg="79ca33eccd36a974fdc603b57a3321ddeeb1fbd8c8a5f5d4342b3f8de3ce3c4e",
        zones=175, nodes=7823, links=28831, demand=76531.8, alpha=0.5, beta=1.3, gap=1.023e-03,
    ),
    _spec(
        "chicago", "05_Chicago",
        net="b93bbac05914b1972e996d1a8b10226d8bfc8fc411d801f3304a571e695db696",
        aem="7c000cf184c676143e8a5396ed3a8acbd5cf0f28b7a9351a8241a2ce1100c382",
        asg="73a98f1d673f4afa32088f2c139b07a57cb2319ed21061d46851512443737fb0",
        zones=819, nodes=14434, links=54469, demand=468616.2, alpha=0.5, beta=1.2, gap=1.214e-03,
    ),
    _spec(
        "neworleans", "06_New_Orleans",
        net="5319a4a9d4e3709b9de4c1f3d346adbaf73d8e3247d4da7d3a12e20757bee392",
        aem="05836852dc0fe5310a089e5c6f75ad346fbfc945fcb34500309e5c48bf6e92f9",
        asg="fec2a83903cf17ce839709237bedcfa5ee638e331475ed03091358b711281283",
        zones=185, nodes=7217, links=24073, demand=59875.2, alpha=0.6, beta=1.8, gap=7.775e-04,
    ),
    _spec(
        "austin", "07_Austin",
        net="2791a6f6d12928834d383f214a5505d6903f888c7575084085442052f685de2f",
        aem="7040d867ab52c67c7b9c249a344e3b8236c57502840c152f63ffe89941eeda70",
        asg="3642deb0e9ad9b3f5f6a93907cce0332020af2a335a3942b782aa0eebf3fc86d",
        zones=199, nodes=10717, links=40158, demand=202867.2, alpha=0.5, beta=1.5, gap=1.033e-03,
    ),
    _spec(
        "minneapolis", "08_Minneapolis",
        net="479326593ca6f4eb19fdf0099decd49f92cb7ada538517f01bdee5a0006b2647",
        aem="17c750159fdbc8ed056f8ef511b9f99b42893cac5d312744b3a219cc1e4a9c65",
        asg="89d6efc5cea499e921060bc1303939ab3a430c20f890670d9067a6ece8f0812d",
        zones=130, nodes=4004, links=15363, demand=60166.8, alpha=0.15, beta=1.8, gap=7.578e-04,
    ),
    _spec(
        "dallas", "09_Dallas",
        net="f49a52cb9ae735bbd4633211786a0068ec056f884ae0ffff46ee9c29795c7cca",
        aem="ded5438668096b80e5de6db047b4d41899307d3363549070beb9fe3dbed955dd",
        asg="4a0128514813b2f0087a935f3bcc77888b5030df820d4e420223b29a4797056f",
        zones=328, nodes=21389, links=77818, demand=200433.6, alpha=0.6, beta=1.3, gap=1.038e-03,
    ),
    _spec(
        "milwaukee", "10_Milwaukee",
        net="857e25ba2f06b02a127121fff04eb51e950353a14dd2a452333eb380f1cb13f7",
        aem="e59d5d34dabf6f17eba7656ef332a4a290314f8f7bdf0c23fbc5fc27a0e6051b",
        asg="189f714bee9a940e364ae80a7c75c13b97a9b8b8b836453573d391573f29a791",
        zones=234, nodes=8521, links=30747, demand=98220.0, alpha=0.5, beta=1.5, gap=9.367e-04,
    ),
    _spec(
        "newyork", "11_New_York",
        net="08e6d5d094da72c569abaa48bfb7298e7364e3fa5c2a24dc76866c91c41ef03f",
        aem="b5f8280db75ccddacc2bb9e424f0930674568e026e35ba8941321dd37df7765b",
        asg="ba4b20e8ab010cc156509278c79a2782f654bd940873c54ac6d7ebc56705682a",
        zones=2005, nodes=28626, links=99408, demand=1782020.4, alpha=0.25, beta=1.5, gap=9.467e-04,
    ),
    _spec(
        "boston", "13_Boston",
        net="1e6c6708109fd392514631f139f45fe04b73d511d93974bed3b6ccd936549de6",
        aem="329fbac497ae650e0732e5b526fc6ffcd917738b9a2699f1035776708f32b236",
        asg="00570994a7c6306dde4ddb5243b7291856d761fbe88bb9ec5fcc0af4e1c1a975",
        zones=191, nodes=5542, links=20487, demand=119746.2, alpha=0.25, beta=2.0, gap=1.367e-03,
    ),
    _spec(
        "philadelphia", "14_Philadelphia",
        net="8d672a700abec25d0970bbd1e052d830c639558b840864cd1a9ae5d3f3418b3a",
        aem="8c40cff1e05fd04763d1c18f83f3bebd6263a78f4ecb3e5227e24c139fd1259b",
        asg="a03df90f3a2073d1947f166d506db9f06da450b881b97df1761d92c96693e8d0",
        zones=389, nodes=10410, links=38641, demand=233554.2, alpha=0.5, beta=1.2, gap=9.490e-04,
    ),
    _spec(
        "miami", "16_Miami",
        net="cbff7b1c4f9806ee2ea05602c4f05008a95ec09fdfd9024ad71fcf152e6d5c57",
        aem="ef22c4a73d7777ce5b5a88eb3240bd8b0fdd2af5e2bd5f0950d7ce98e7113a7e",
        asg="8c2d60a07fee925271ebb66b596086092d05914f50ad48294eb1f04804e32923",
        zones=108, nodes=4121, links=15108, demand=39412.2, alpha=0.5, beta=1.5, gap=8.921e-04,
    ),
    _spec(
        "atlanta", "17_Atlanta",
        net="8a36de675fbeda31697b436437bf2306e5ac20d24116ec26fb03f3174c952c9a",
        aem="6e4aa0a8d3bb0887dec1a9df7006a3eca5d52d22ec7f09d4ea7aa3866ec738a1",
        asg="52a0717af5919557d7c49728ab4fb0d657879e07c1259b441ed587404ba3e979",
        zones=141, nodes=5207, links=20243, demand=61068.6, alpha=0.2, beta=1.5, gap=8.100e-04,
    ),
    _spec(
        "denver", "19_Denver",
        net="d47e39fb82989cd0a52728b6c840ff127c311e02994f1558ec491cc6e374be4b",
        aem="eedfc44f8ed67e94c20ae1519da3cca1d0112a1a240f34d256dd11672e2edaf8",
        asg="0b1bdf0b1a0dbb1e7d4a9b6fc84fc88bf445722c97c8ba0efeeb2877fbb4ee4b",
        zones=175, nodes=9205, links=34724, demand=120433.8, alpha=0.5, beta=1.5, gap=1.004e-03,
    ),
    _spec(
        "honolulu", "20_Honolulu",
        net="4a5f4080c78bc117808fc5d40df419effeb2a23e89f305af8986245d01d2cf66",
        aem="d1efc73967d43ccf871e5a715368ee7d9e7a3a64560f2bbd63ddf12633711a25",
        asg="12aebeaed18ca8bb6e2170eacf5261f19c2d2fabfc3b941dae67a907ff6a065d",
        zones=117, nodes=2982, links=11205, demand=107515.2, alpha=0.5, beta=1.5, gap=1.036e-03,
    ),
    )
}
