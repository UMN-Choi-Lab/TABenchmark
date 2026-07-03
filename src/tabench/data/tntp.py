"""Defensive parsers for the TNTP formats used by TransportationNetworks.

Format notes (see docs/ARCHITECTURE.md P9 and the repository README):

* ``*_net.tntp``: metadata tags ``<NUMBER OF ZONES>``, ``<NUMBER OF NODES>``,
  ``<FIRST THRU NODE>``, ``<NUMBER OF LINKS>`` until ``<END OF METADATA>``;
  then whitespace-delimited rows terminated by ``;`` with fixed column order
  ``init_node term_node capacity length free_flow_time b power speed toll
  link_type``. Comment lines start with ``~``.
* ``*_trips.tntp``: metadata, then blocks ``Origin <i>`` followed by wrapped
  ``<j> : <flow>;`` entries.
* ``*_flow.tntp`` (best-known solutions): header ``From To Volume Cost`` then
  four whitespace-separated columns, no semicolons.

Units are per-network metadata and are NOT interpreted here.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from ..core.scenario import Network

__all__ = ["parse_net", "parse_trips", "parse_flow", "load_network"]

_NET_COLUMNS = (
    "init_node",
    "term_node",
    "capacity",
    "length",
    "free_flow_time",
    "b",
    "power",
    "speed",
    "toll",
    "link_type",
)


def _clean_lines(text: str) -> list[str]:
    """Strip comments (``~``), blank lines, and surrounding whitespace."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("~"):
            continue
        out.append(line)
    return out


def _split_metadata(lines: list[str]) -> tuple[dict[str, str], list[str]]:
    """Separate ``<TAG> value`` metadata lines from the body."""
    meta: dict[str, str] = {}
    body_start = 0
    for i, line in enumerate(lines):
        if line.startswith("<END OF METADATA>"):
            body_start = i + 1
            break
        if line.startswith("<"):
            tag_end = line.index(">")
            tag = line[1:tag_end].strip().upper()
            meta[tag] = line[tag_end + 1 :].strip()
            body_start = i + 1
    return meta, lines[body_start:]


def parse_net(path: str | Path) -> dict:
    """Parse a ``*_net.tntp`` file into metadata plus link-table arrays."""
    lines = _clean_lines(Path(path).read_text())
    meta, body = _split_metadata(lines)

    required = ("NUMBER OF ZONES", "NUMBER OF NODES", "FIRST THRU NODE", "NUMBER OF LINKS")
    missing = [tag for tag in required if tag not in meta]
    if missing:
        raise ValueError(f"{path}: missing TNTP metadata tags {missing}")

    columns: dict[str, list[float]] = {c: [] for c in _NET_COLUMNS}
    for line in body:
        if line.lower().replace("~", "").strip().startswith("init_node"):
            continue  # some files repeat a header row in the body
        fields = line.rstrip(";").split()
        if len(fields) != len(_NET_COLUMNS):
            raise ValueError(
                f"{path}: expected {len(_NET_COLUMNS)} columns, got {len(fields)}: {line!r}"
            )
        for name, value in zip(_NET_COLUMNS, fields, strict=True):
            columns[name].append(float(value))

    n_links = int(meta["NUMBER OF LINKS"])
    if len(columns["init_node"]) != n_links:
        raise ValueError(
            f"{path}: metadata declares {n_links} links, parsed {len(columns['init_node'])}"
        )

    result: dict = {
        "n_zones": int(meta["NUMBER OF ZONES"]),
        "n_nodes": int(meta["NUMBER OF NODES"]),
        "first_thru_node": int(meta["FIRST THRU NODE"]),
        "n_links": n_links,
    }
    for name in _NET_COLUMNS:
        dtype = np.int64 if name in ("init_node", "term_node", "link_type") else np.float64
        result[name] = np.asarray(columns[name], dtype=dtype)
    return result


def parse_trips(path: str | Path) -> np.ndarray:
    """Parse a ``*_trips.tntp`` file into a dense (n_zones, n_zones) OD matrix."""
    lines = _clean_lines(Path(path).read_text())
    meta, body = _split_metadata(lines)
    n_zones = int(meta["NUMBER OF ZONES"])
    matrix = np.zeros((n_zones, n_zones), dtype=np.float64)

    origin: int | None = None
    for line in body:
        if line.lower().startswith("origin"):
            origin = int(line.split()[1])
            continue
        if origin is None:
            raise ValueError(f"{path}: OD entry before any 'Origin' header: {line!r}")
        for entry in line.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            dest_str, flow_str = entry.split(":")
            dest, flow = int(dest_str), float(flow_str)
            if not (1 <= origin <= n_zones and 1 <= dest <= n_zones):
                raise ValueError(f"{path}: zone pair ({origin},{dest}) outside 1..{n_zones}")
            matrix[origin - 1, dest - 1] = flow

    declared = meta.get("TOTAL OD FLOW")
    if declared is not None:
        total = matrix.sum()
        if not np.isclose(total, float(declared), rtol=1e-6):
            warnings.warn(
                f"{path}: parsed OD total {total:.6f} differs from declared "
                f"<TOTAL OD FLOW> {declared}",
                stacklevel=2,
            )
    return matrix


def parse_flow(path: str | Path) -> dict[str, np.ndarray]:
    """Parse a best-known ``*_flow.tntp`` solution file."""
    lines = _clean_lines(Path(path).read_text())
    init, term, volume, cost = [], [], [], []
    for line in lines:
        fields = line.rstrip(";").split()
        if fields[0].lower() in ("from", "<from"):
            continue  # header row
        if len(fields) != 4:
            raise ValueError(f"{path}: expected 4 columns 'From To Volume Cost': {line!r}")
        init.append(int(fields[0]))
        term.append(int(fields[1]))
        volume.append(float(fields[2]))
        cost.append(float(fields[3]))
    return {
        "init_node": np.asarray(init, dtype=np.int64),
        "term_node": np.asarray(term, dtype=np.int64),
        "volume": np.asarray(volume, dtype=np.float64),
        "cost": np.asarray(cost, dtype=np.float64),
    }


def load_network(
    net_path: str | Path,
    name: str,
    toll_weight: float = 0.0,
    distance_weight: float = 0.0,
    units: tuple[tuple[str, str], ...] = (),
) -> Network:
    """Parse a net file and build a :class:`Network` with per-network metadata."""
    raw = parse_net(net_path)
    return Network(
        name=name,
        n_nodes=raw["n_nodes"],
        n_zones=raw["n_zones"],
        first_thru_node=raw["first_thru_node"],
        init_node=raw["init_node"],
        term_node=raw["term_node"],
        capacity=raw["capacity"],
        length=raw["length"],
        free_flow_time=raw["free_flow_time"],
        b=raw["b"],
        power=raw["power"],
        toll=raw["toll"],
        link_type=raw["link_type"],
        toll_weight=toll_weight,
        distance_weight=distance_weight,
        units=units,
    )


def align_flows_to_network(network: Network, flow_table: dict[str, np.ndarray]) -> np.ndarray:
    """Reorder a parsed flow solution to the network's link order."""
    index = {
        (int(i), int(t)): k
        for k, (i, t) in enumerate(zip(network.init_node, network.term_node, strict=True))
    }
    flows = np.full(network.n_links, np.nan)
    for i, t, v in zip(
        flow_table["init_node"], flow_table["term_node"], flow_table["volume"], strict=True
    ):
        key = (int(i), int(t))
        if key not in index:
            raise ValueError(f"Flow solution contains unknown link {key}")
        flows[index[key]] = v
    if np.isnan(flows).any():
        raise ValueError("Flow solution does not cover every network link")
    return flows
