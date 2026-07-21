"""C6 (five-surface consistency review) -- deferred batch B5: pin the honest
input/output capability declarations for every T1 model that enforces an extra
scenario field via a runtime raise, and guard that each declared extra field is
the one its ``solve()`` actually requires (rot guard).

These declarations are pure metadata: only the three T2 estimation manifest
writers read ``Capabilities.inputs_required``/``outputs``; no T1 code path does,
so declaring them changes no behavior -- the point is to make the declaration
honest, not to change what the model does.
"""

from __future__ import annotations

import pytest

import tabench as tb
from tabench.core.budget import Budget
from tabench.core.results import Trace
from tabench.core.rng import RngBundle
from tabench.models.base import MODEL_REGISTRY

_OD = "od_matrix"
_FLOWS = frozenset({"link_flows"})

# model name -> (inputs_required, outputs). Every listed model's solve() raises
# when its extra scenario field is absent; the extra input token is that field's
# name, so a plain UE scenario triggers a raise that names it (the rot guard).
DECLARED: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "sue-msa": (frozenset({_OD, "sue_theta"}), _FLOWS),
    "sue-probit-msa": (frozenset({_OD, "sue_theta"}), _FLOWS),
    "dtd-horowitz": (frozenset({_OD, "sue_theta"}), _FLOWS),
    "dtd-swap-sue": (frozenset({_OD, "sue_theta"}), _FLOWS),
    "dtd-stochastic": (frozenset({_OD, "sue_theta"}), _FLOWS),
    "fw-elastic": (frozenset({_OD, "elastic_demand"}), _FLOWS),
    "evans": (frozenset({_OD, "combined_demand"}), _FLOWS),
    "br-ue": (frozenset({_OD, "br_epsilon"}), _FLOWS),
    "sc-tap": (frozenset({_OD, "side_capacities"}), _FLOWS),
    "vi-asym": (frozenset({_OD, "link_interaction"}), _FLOWS),
    "multiclass": (
        frozenset({_OD, "multiclass"}),
        frozenset({"link_flows", "class_link_flows"}),
    ),
}


@pytest.mark.parametrize("name", sorted(DECLARED))
def test_declared_capabilities_are_pinned(name: str) -> None:
    caps = MODEL_REGISTRY[name].capabilities
    inputs, outputs = DECLARED[name]
    assert caps.inputs_required == inputs
    assert caps.outputs == outputs


@pytest.mark.parametrize("name", sorted(DECLARED))
def test_declared_extra_field_is_actually_required(name: str) -> None:
    """The declared extra input is the field solve() raises about on a plain UE
    scenario, so the declaration cannot silently rot away from the enforcement."""
    extra = DECLARED[name][0] - {_OD}
    assert len(extra) == 1, "each model declares exactly one extra required field"
    token = next(iter(extra))
    model = MODEL_REGISTRY[name]()
    with pytest.raises(ValueError) as exc:
        model.solve(tb.braess_scenario(), Budget(iterations=1), RngBundle(0), Trace())
    # Strip the leading model-name occurrence so the token must appear as the FIELD
    # it names, not merely as the model-name prefix of the message. Without this the
    # multiclass row is vacuous (name == token "multiclass"): it would pass even if
    # the message stopped naming the required field.
    assert token in str(exc.value).split(name, 1)[-1]


def test_only_multiclass_declares_class_link_flows() -> None:
    """Sweep: multiclass is the only registered model emitting class_link_flows,
    so it is the only one that declares it in outputs."""
    claimers = {
        name
        for name, cls in MODEL_REGISTRY.items()
        if "class_link_flows" in cls.capabilities.outputs
    }
    assert claimers == {"multiclass"}
