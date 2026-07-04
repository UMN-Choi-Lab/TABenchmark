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

ANAHEIM = NetworkSpec(
    key="anaheim",
    repo_dir="Anaheim",
    files={
        "net": (
            "Anaheim_net.tntp",
            "99933b415e9500b13907829c37a43cfa9141714fad5af279081e28e5f9356f9a",
        ),
        "trips": (
            "Anaheim_trips.tntp",
            "906893854cd0db4479c0b5f07678ce5616fa8e42e2b997f918c378309c66a94e",
        ),
        "flow": (
            "Anaheim_flow.tntp",
            "eecd21c2a908b6a1c6729045ea260e96df1de026f05281d341fc281f2552cebe",
        ),
    },
    toll_weight=0.0,
    distance_weight=0.0,
    units=(
        ("free_flow_time", "minutes"),
        ("length", "feet (not part of the cost: distance weight 0)"),
        ("capacity", "vehicles per hour"),
        ("demand", "vehicle trips (total 104694.4)"),
    ),
    notes=(
        "1992 Anaheim: 38 zones, 416 nodes, 914 links; uniform BPR "
        "alpha=0.15, beta=4. first_thru_node=39: zones are centroids that "
        "carry no through traffic (unlike Sioux Falls)."
    ),
    citation_extra="Provided by Jeff Ban and Ray Jayakrishnan; map by Marco Nie.",
    best_known={
        "solution": (
            "Anaheim_flow.tntp (best-known UE link flows, upstream AEC < 1e-15; "
            "recomputed pure-BPR relative gap 5.7e-15)"
        ),
    },
)

BARCELONA = NetworkSpec(
    key="barcelona",
    repo_dir="Barcelona",
    files={
        "net": (
            "Barcelona_net.tntp",
            "74ea13010beca70c641417c38bc900d6d7a2a600f23f18f76e417f7090c69bbd",
        ),
        "trips": (
            "Barcelona_trips.tntp",
            "de485bcc423ff66c8e6601ae718255614d19099c0d0536ffcdb62972e1fcbbe1",
        ),
        "flow": (
            "Barcelona_flow.tntp",
            "cee8df9f7930e52d5779aa6d3c14b25bcede54582c827971b5b5b3fe2f6aa345",
        ),
    },
    toll_weight=0.0,
    distance_weight=0.0,
    units=(
        ("time", "native units (upstream README leaves units undocumented)"),
        ("demand", "native units (total 184679.561)"),
        ("capacity", "set to 1 on all links; the b column encodes B/capacity^power"),
    ),
    notes=(
        "110 zones, 1020 nodes, 2522 links; first_thru_node=111. Mixed BPR "
        "powers: 0 on 565 centroid connectors (constant cost), 2.0 on 19 "
        "links, ~4.1-4.9 on most (602 links at 4.924), 16.83 on 140 links. "
        "Upstream source/units undocumented; treat as an abstract benchmark "
        "instance."
    ),
    best_known={
        "solution": (
            "Barcelona_flow.tntp (upstream AEC 2e-14, optimal Beckmann "
            "1265654.92203176; recomputed pure-BPR relative gap -1.9e-15 and "
            "matching Beckmann to all published digits)"
        ),
    },
)

WINNIPEG = NetworkSpec(
    key="winnipeg",
    repo_dir="Winnipeg",
    files={
        "net": (
            "Winnipeg_net.tntp",
            "b7958f3a25f3d80890b2a4d5c534dc0820d1b4c8e8debb8ddbb5f9eb6f0fb593",
        ),
        "trips": (
            "Winnipeg_trips.tntp",
            "b5b8b08ca486b6213227401695fd8066db98821696513d512ddc4d9220d7397b",
        ),
        "flow": (
            "Winnipeg_flow.tntp",
            "6656dd38647f00879c31eeda533df8f34007e82cbbec03421457dfde2bae81a5",
        ),
    },
    toll_weight=0.0,
    distance_weight=0.0,
    units=(
        ("time", "native units (upstream README leaves units undocumented)"),
        ("demand", "native units (total 64784)"),
        (
            "capacity",
            "arbitrarily 1 on all links; the b column is B/capacity^power "
            "(documented upstream)",
        ),
    ),
    notes=(
        "147 zones, 1052 nodes, 2836 links; first_thru_node=148. Mixed BPR "
        "powers: 0 on 1176 centroid connectors, 3.50-6.87 elsewhere (159 "
        "links above 5.3, max 6.8677). One of several Winnipeg versions in "
        "circulation - this registry pins this one."
    ),
    best_known={
        "solution": (
            "Winnipeg_flow.tntp (upstream AEC 2.8e-15, optimal Beckmann "
            "827911.494629963; recomputed pure-BPR relative gap -2.3e-15, "
            "node balance residual exactly 0)"
        ),
    },
)

REGISTRY: dict[str, NetworkSpec] = {
    "siouxfalls": SIOUXFALLS,
    "anaheim": ANAHEIM,
    "barcelona": BARCELONA,
    "winnipeg": WINNIPEG,
}
