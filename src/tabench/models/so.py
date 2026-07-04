"""System-optimum assignment via the marginal-cost transformation.

For BPR costs the marginal social cost is again a BPR function:

    t_a(v) + v t'_a(v) = fft_a (1 + b_a (1 + p_a) (v / cap_a)^{p_a}) + fixed_a

(the flow-independent fixed cost appears once, not doubled; p = 0 or b = 0
links are unchanged). The system optimum of a scenario is therefore exactly
the user equilibrium of the marginal-cost network (Beckmann et al. 1956;
Yang & Huang 1998), so the whole verified FW family applies unchanged.

Two identities make the self-reports meaningful without extra work: on the
marginal network, the Beckmann objective equals the ORIGINAL network's TSTT
(``integral of (t(s) + s t'(s)) ds = v t(v)``) — i.e. the true SO objective —
and the self-monitored relative gap is the SO gap.
"""

from __future__ import annotations

import dataclasses

from ..core.budget import Budget
from ..core.capabilities import Capabilities
from ..core.results import ResultBundle, Trace
from ..core.rng import RngBundle
from ..core.scenario import Network, Scenario
from .base import register_model
from .frank_wolfe import BiconjugateFrankWolfeModel

__all__ = ["marginal_network", "SystemOptimumModel"]


def marginal_network(network: Network) -> Network:
    """The network whose UE is the original network's system optimum."""
    return dataclasses.replace(
        network,
        name=f"{network.name}+marginal",
        b=network.b * (1.0 + network.power),
    )


@register_model
class SystemOptimumModel(BiconjugateFrankWolfeModel):
    """System-optimum solver: bi-conjugate FW on the marginal-cost network.

    Emitted flows are for the ORIGINAL scenario (the transformation changes
    costs, not the network structure). Self-reports: ``beckmann`` is the true
    SO objective (original-network TSTT) and ``relative_gap`` is the
    self-monitored SO gap; ``Budget.target_relative_gap`` therefore applies
    to the SO gap.
    """

    name = "so-bfw"
    capabilities = Capabilities(
        paradigm="static_so",
        deterministic=True,
        provides_gap=True,
        seedable=True,
    )

    #: quantities are computed on the marginal network, so label them so:
    #: the transformed gap IS the SO gap and the transformed Beckmann IS the
    #: original network's TSTT (the true SO objective).
    _SELF_REPORT_KEYS = {
        "relative_gap": "so_relative_gap",
        "tstt": "tstt_mc",
        "sptt": "sptt_mc",
        "beckmann": "tstt",
    }

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        marginal = dataclasses.replace(
            scenario,
            network=marginal_network(scenario.network),
            reference=None,  # any UE reference oracle does not apply to SO
        )
        return super().solve(marginal, budget, rng, trace)
