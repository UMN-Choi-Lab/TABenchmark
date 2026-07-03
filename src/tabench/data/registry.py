"""Registry of benchmark networks: files, checksums, units metadata, defects.

Every entry pins exact file checksums at a fixed upstream commit and records
the per-network unit conventions that the ``.tntp`` files themselves do not
carry (P9).
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["NetworkSpec", "REGISTRY"]


@dataclass(frozen=True)
class NetworkSpec:
    """Provenance and metadata for one downloadable network."""

    key: str
    repo_dir: str
    files: dict[str, tuple[str, str]]  # role -> (filename, sha256)
    toll_weight: float = 0.0
    distance_weight: float = 0.0
    units: tuple[tuple[str, str], ...] = ()
    notes: str = ""
    citation_extra: str = ""
    best_known: dict[str, str] = field(default_factory=dict)


SIOUXFALLS = NetworkSpec(
    key="siouxfalls",
    repo_dir="SiouxFalls",
    files={
        "net": (
            "SiouxFalls_net.tntp",
            "ace99b24cec69c273ff0cf3d6d074110177f0cc0ae24b0c7a9f4f4cb5e27635c",
        ),
        "trips": (
            "SiouxFalls_trips.tntp",
            "56f9566857f3f66730fd5c4232258d7ee3ac2931a476526331afd062f4958de7",
        ),
        "flow": (
            "SiouxFalls_flow.tntp",
            "5d0b83a22ecc3ce79dabb2b2972162b78c5eda571dcb5b3687429d8397654fee",
        ),
        "node": (
            "SiouxFalls_node.tntp",
            "91c0b954dcfc6f6f08f48de1ce9d245261f5188a33aef76acf3ce1fff0809969",
        ),
    },
    toll_weight=0.0,
    distance_weight=0.0,
    units=(
        ("free_flow_time", "0.01 hours (commonly misread as minutes)"),
        ("demand", "0.1 x original daily vehicle trips"),
        ("length", "set equal to free_flow_time; arbitrary"),
        ("coordinates", "WGS84 lon/lat (plotting only)"),
    ),
    notes=(
        "The classic 24-node/76-link test network (LeBlanc, Morlok & Pierskalla "
        "1975). Non-physical units by construction; use only as a benchmark "
        "instance, never for planning conclusions. first_thru_node=1: zones are "
        "regular intersections."
    ),
    citation_extra=(
        "Original source: LeBlanc, Morlok & Pierskalla (1975), Transportation "
        "Research 9(5):309-318."
    ),
    best_known={
        "solution": "SiouxFalls_flow.tntp (best-known UE link flows, AEC ~3.9e-15)",
        "note": (
            "The regression oracle objective is computed from the best-known "
            "flows with this package's Beckmann implementation, so tests are "
            "unit-convention-free."
        ),
    },
)

REGISTRY: dict[str, NetworkSpec] = {
    "siouxfalls": SIOUXFALLS,
}
