"""System-optimum metrics: certified SO gap, price of anarchy, first-best tolls.

All quantities are recomputed from ``(scenario, link_flows)`` (P1). The
marginal social cost used throughout is ``t(v) + v t'(v)``, computed directly
from the network's cost model — no transformed network needed on the
certification side.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..core.scenario import Network, Scenario
from ..models.so import marginal_network

__all__ = ["marginal_costs", "price_of_anarchy", "marginal_cost_tolls", "tolled_network"]


def marginal_costs(network: Network, link_flows: np.ndarray) -> np.ndarray:
    """Marginal social cost t(v) + v t'(v) per link at the given flows.

    Computed via the transformed network's closed form (b -> b(1+p)), which
    is exact everywhere — the naive ``t + v * t'`` product is unreliable for
    0 < p < 1 at subnormal flows, where the derivative clamps to float max.
    """
    return marginal_network(network).link_cost(link_flows)


def price_of_anarchy(
    scenario: Scenario, ue_flows: np.ndarray, so_flows: np.ndarray
) -> float:
    """PoA = TSTT(UE flows) / TSTT(SO flows), recomputed from both flow vectors.

    Roughgarden & Tardos (2002): >= 1 always; <= 4/3 for affine latencies.
    This is an experiment-level report comparing two certified solutions,
    not a per-model metric.
    """
    net = scenario.network
    ue = np.maximum(np.asarray(ue_flows, dtype=np.float64), 0.0)
    so = np.maximum(np.asarray(so_flows, dtype=np.float64), 0.0)
    tstt_ue = float(ue @ net.link_cost(ue))
    tstt_so = float(so @ net.link_cost(so))
    if tstt_so <= 0:
        raise ValueError("TSTT at the SO flows must be positive to define PoA")
    return tstt_ue / tstt_so


def marginal_cost_tolls(network: Network, so_flows: np.ndarray) -> np.ndarray:
    """First-best (Pigouvian) tolls v* t'(v*) at system-optimal flows.

    Charging these tolls makes the system optimum a user equilibrium
    (marginal-cost pricing; Yang & Huang 1998).
    """
    v = np.maximum(np.asarray(so_flows, dtype=np.float64), 0.0)
    return v * network.link_cost_derivative(v)


def tolled_network(network: Network, tolls: np.ndarray) -> Network:
    """The network with the given tolls added to the generalized cost.

    Any pre-existing toll contribution is folded into a weight-1 toll column
    (``fixed_cost_new = fixed_cost_old + tolls`` exactly), so the transform
    composes and works regardless of the base network's toll convention.
    """
    tolls = np.asarray(tolls, dtype=np.float64)
    if tolls.shape != (network.n_links,):
        raise ValueError(f"tolls shape {tolls.shape} != ({network.n_links},)")
    if np.any(tolls < 0) or not np.all(np.isfinite(tolls)):
        raise ValueError("tolls must be finite and nonnegative")
    return dataclasses.replace(
        network,
        name=f"{network.name}+tolled",
        toll=network.toll_weight * network.toll + tolls,
        toll_weight=1.0,
    )
