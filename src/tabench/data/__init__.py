"""Data layer: TNTP parsing, checksummed fetching, network registry, builtins."""

from pathlib import Path

from ..core.scenario import Demand, ReferenceSolution, Scenario
from .builtin import braess_scenario, elastic_two_route_scenario, two_route_scenario
from .fetcher import ChecksumError, cache_dir, citation, fetch
from .registry import REGISTRY, NetworkSpec
from .tntp import align_flows_to_network, load_network, parse_flow, parse_net, parse_trips

__all__ = [
    "braess_scenario",
    "two_route_scenario",
    "elastic_two_route_scenario",
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
    if key not in REGISTRY:
        raise KeyError(
            f"Unknown scenario {key!r}; available: braess, tworoute, "
            f"elastic-tworoute, {sorted(REGISTRY)}"
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
