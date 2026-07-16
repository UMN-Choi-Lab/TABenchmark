"""Data layer: TNTP parsing, checksummed fetching, network registry, builtins."""

from pathlib import Path

from ..core.scenario import Demand, ReferenceSolution, Scenario
from .bo4mob import (
    BO4MOB_REGISTRY,
    BO4MOB_SMOKE,
    Bo4MobHpcOnlyError,
    Bo4MobSpec,
    bo4mob_citation,
    fetch_bo4mob,
)
from .builtin import (
    br_two_route_scenario,
    braess_scenario,
    elastic_two_route_scenario,
    evans_symmetric_scenario,
    multiclass_two_route_scenario,
    sc_two_route_scenario,
    two_route_scenario,
    vi_two_route_scenario,
)
from .fetcher import ChecksumError, cache_dir, citation, fetch
from .registry import REGISTRY, NetworkSpec
from .tntp import align_flows_to_network, load_network, parse_flow, parse_net, parse_trips
from .xu2024 import (
    XU2024_REGISTRY,
    XU2024_RUNGS,
    Xu2024CitySpec,
    fetch_city,
    xu2024_citation,
    xu2024_scenario,
)

__all__ = [
    "braess_scenario",
    "two_route_scenario",
    "elastic_two_route_scenario",
    "evans_symmetric_scenario",
    "br_two_route_scenario",
    "sc_two_route_scenario",
    "vi_two_route_scenario",
    "multiclass_two_route_scenario",
    "ChecksumError",
    "cache_dir",
    "citation",
    "fetch",
    "REGISTRY",
    "NetworkSpec",
    "align_flows_to_network",
    "load_network",
    "parse_flow",
    "parse_net",
    "parse_trips",
    "load_scenario",
    "XU2024_REGISTRY",
    "XU2024_RUNGS",
    "Xu2024CitySpec",
    "fetch_city",
    "xu2024_citation",
    "xu2024_scenario",
    "BO4MOB_REGISTRY",
    "BO4MOB_SMOKE",
    "Bo4MobSpec",
    "Bo4MobHpcOnlyError",
    "fetch_bo4mob",
    "bo4mob_citation",
]


def load_scenario(key: str) -> Scenario:
    """Load a benchmark scenario by registry key (downloading data if needed).

    ``braess`` and ``tworoute`` (the logit-SUE anchor) are built in; all
    other keys resolve through the network registry and the checksummed
    fetcher.
    """
    if key == "braess":
        return braess_scenario()
    if key == "tworoute":
        return two_route_scenario()
    if key == "elastic-tworoute":
        return elastic_two_route_scenario()
    if key == "evans":
        return evans_symmetric_scenario()
    if key == "br-tworoute":
        return br_two_route_scenario()
    if key == "sc-tworoute":
        return sc_two_route_scenario()
    if key == "vi-tworoute":
        return vi_two_route_scenario()
    if key == "multiclass":
        return multiclass_two_route_scenario()
    if key.startswith("xu2024-"):
        # Cross-domain axis (adr-033): real US-city instances, a separate
        # download-on-demand registry (never in the CI-prefetched REGISTRY).
        return xu2024_scenario(key[len("xu2024-") :])
    if key not in REGISTRY:
        raise KeyError(
            f"Unknown scenario {key!r}; available: braess, tworoute, "
            "elastic-tworoute, evans, br-tworoute, sc-tworoute, vi-tworoute, "
            f"multiclass, {sorted(REGISTRY)}, "
            f"xu2024-<city> for {sorted(XU2024_REGISTRY)}"
        )
    spec = REGISTRY[key]
    paths: dict[str, Path] = fetch(spec)
    network = load_network(
        paths["net"],
        name=spec.key,
        toll_weight=spec.toll_weight,
        distance_weight=spec.distance_weight,
        units=spec.units,
    )
    demand = Demand(matrix=parse_trips(paths["trips"]))
    reference = None
    if "flow" in paths:
        flows = align_flows_to_network(network, parse_flow(paths["flow"]))
        reference = ReferenceSolution(
            link_flows=flows,
            source=citation(spec),
            note=spec.best_known.get("solution", ""),
        )
    return Scenario(
        name=spec.key,
        network=network,
        demand=demand,
        reference=reference,
        family=f"tntp-{spec.key}",
    )
