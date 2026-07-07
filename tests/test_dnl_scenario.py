"""DynamicScenario tests: validation, dnl content hash, static-hash isolation."""

import math

import numpy as np
import pytest

from tabench.core.scenario import Network
from tabench.data import braess_scenario
from tabench.dnl.demand import DynamicDemand, TurningFractions
from tabench.dnl.fd import LinkDynamics
from tabench.dnl.grid import TimeGrid
from tabench.dnl.scenario import DynamicScenario

# The frozen static golden hash (must never move; re-asserted below with the
# dnl package imported) and the dnl golden hash of the inline corridor
# scenario (pinned at implementation time; the builtin scenarios' pins live
# in the runner tests).
GOLDEN_BRAESS_HASH = "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"
GOLDEN_CORRIDOR_DNL_HASH = "ecdea09f1c569e0e775f294b0950ab3dbea4e2982c81d2685ba9a7ade463266e"


def _network(name, n_nodes, n_zones, init, term, **bpr_overrides):
    """Static Network with dummy BPR fields (ignored and unhashed by DNL)."""
    init = np.asarray(init, dtype=np.int64)
    term = np.asarray(term, dtype=np.int64)
    n = len(init)
    fields = dict(
        capacity=np.ones(n),
        length=np.zeros(n),
        free_flow_time=np.ones(n),
        b=np.zeros(n),
        power=np.ones(n),
        toll=np.zeros(n),
    )
    fields.update(bpr_overrides)
    return Network(
        name=name,
        n_nodes=n_nodes,
        n_zones=n_zones,
        first_thru_node=1,
        init_node=init,
        term_node=term,
        link_type=np.ones(n, dtype=np.int64),
        **fields,
    )


def _corridor_scenario(**overrides) -> DynamicScenario:
    """Anchor-2-shaped 2-link point-queue corridor, built inline by hand:
    origin zone 1 -> interior node 3 -> destination zone 2."""
    rates = np.zeros((2, 2, 2))
    rates[0, 0, 1] = 1.5
    rates[1, 0, 1] = 0.5
    fields = dict(
        name="corridor-inline",
        network=_network("corridor", n_nodes=3, n_zones=2, init=[1, 3], term=[3, 2]),
        dynamics=LinkDynamics(
            length=np.array([1.0, 1.0]),
            free_speed=np.array([2.0, 1.0]),
            wave_speed=np.array([math.inf, math.inf]),
            jam_density=np.array([math.inf, math.inf]),
            capacity=np.array([4.0, 1.0]),
        ),
        demand=DynamicDemand(breakpoints=np.array([0.0, 4.0, 12.0]), rates=rates),
        grid=TimeGrid(dt=0.5, n_steps=32),
        turns=None,
    )
    fields.update(overrides)
    return DynamicScenario(**fields)


def _diverge_scenario(
    *,
    dt=0.5,
    n_steps=8,
    breakpoints=(0.0, 1.0, 2.0),
    rate_01=0.6,
    split=(0.6, 0.4),
    length=(1.0, 1.0, 1.0),
    free_speed=(1.0, 1.0, 1.0),
    wave_speed=(1.0, 1.0, 1.0),
    jam_density=(4.0, 4.0, 4.0),
    capacity=(1.0, 1.0, 1.0),
    network=None,
    turns="default",
) -> DynamicScenario:
    """Finite-kappa diverge: origin zone 1 -> node 4 -> destination zones 2, 3.

    Every physics field is independently perturbable so the hash-sensitivity
    battery can flip exactly one at a time.
    """
    if network is None:
        network = _network("diverge", n_nodes=4, n_zones=3, init=[1, 4, 4], term=[4, 2, 3])
    rates = np.zeros((2, 3, 3))
    rates[0, 0, 1] = rate_01
    rates[0, 0, 2] = 0.4
    rates[1, 0, 1] = 0.3
    rates[1, 0, 2] = 0.2
    if turns == "default":
        turns = TurningFractions(frac=((4, np.array([list(split)])),))
    return DynamicScenario(
        name="diverge-inline",
        network=network,
        dynamics=LinkDynamics(
            length=np.asarray(length, dtype=np.float64),
            free_speed=np.asarray(free_speed, dtype=np.float64),
            wave_speed=np.asarray(wave_speed, dtype=np.float64),
            jam_density=np.asarray(jam_density, dtype=np.float64),
            capacity=np.asarray(capacity, dtype=np.float64),
        ),
        demand=DynamicDemand(breakpoints=np.asarray(breakpoints, dtype=np.float64), rates=rates),
        grid=TimeGrid(dt=dt, n_steps=n_steps),
        turns=turns,
    )


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_valid_scenarios_construct_and_family_defaults_to_name():
    corridor = _corridor_scenario()
    assert corridor.family == "corridor-inline"
    assert _corridor_scenario(family="lineage-x").family == "lineage-x"
    assert _diverge_scenario().name == "diverge-inline"


def test_zone_count_mismatch_raises():
    rates = np.zeros((1, 3, 3))
    rates[0, 0, 1] = 1.0
    demand3 = DynamicDemand(breakpoints=np.array([0.0, 1.0]), rates=rates)
    with pytest.raises(ValueError, match="zones"):
        _corridor_scenario(demand=demand3)


def test_dynamics_length_mismatch_raises():
    dyn3 = LinkDynamics(
        length=np.ones(3),
        free_speed=np.ones(3),
        wave_speed=np.full(3, math.inf),
        jam_density=np.full(3, math.inf),
        capacity=np.ones(3),
    )
    with pytest.raises(ValueError, match="links"):
        _corridor_scenario(dynamics=dyn3)


def test_wave_resolution_failure_raises():
    dyn = LinkDynamics(
        length=np.array([1.0, 1.0]),
        free_speed=np.array([4.0, 4.0]),  # L/vf = 0.25 < dt = 0.5
        wave_speed=np.array([math.inf, math.inf]),
        jam_density=np.array([math.inf, math.inf]),
        capacity=np.array([4.0, 1.0]),
    )
    with pytest.raises(ValueError, match="wave-resolved"):
        _corridor_scenario(dynamics=dyn)


def test_missing_turn_entry_raises():
    with pytest.raises(ValueError, match="missing turning fractions"):
        _diverge_scenario(turns=None)


def test_extra_turn_entry_raises():
    # interior node 3 of the corridor is 1-in/1-out: no split to specify
    with pytest.raises(ValueError, match="extra turning-fraction"):
        _corridor_scenario(turns=TurningFractions(frac=((3, np.array([[1.0]])),)))


def test_turns_at_zone_centroid_raise():
    # zone centroids are boundaries in DNL: through traffic is impossible,
    # so a turn split at a zone node is rejected (first_thru_node semantics)
    turns = TurningFractions(
        frac=((1, np.array([[1.0]])), (4, np.array([[0.6, 0.4]])))
    )
    with pytest.raises(ValueError, match="through traffic"):
        _diverge_scenario(turns=turns)


def test_turn_matrix_shape_mismatch_raises():
    turns = TurningFractions(frac=((4, np.array([[0.5, 0.3, 0.2]])),))
    with pytest.raises(ValueError, match="shape"):
        _diverge_scenario(turns=turns)


def test_producing_zone_without_outgoing_link_raises():
    net = _network("no-out", n_nodes=3, n_zones=2, init=[3], term=[2])
    dyn = LinkDynamics(
        length=np.ones(1),
        free_speed=np.ones(1),
        wave_speed=np.array([math.inf]),
        jam_density=np.array([math.inf]),
        capacity=np.ones(1),
    )
    with pytest.raises(ValueError, match="production"):
        _corridor_scenario(network=net, dynamics=dyn)


def test_attracting_zone_without_incoming_link_raises():
    net = _network("no-in", n_nodes=3, n_zones=2, init=[1], term=[3])
    dyn = LinkDynamics(
        length=np.ones(1),
        free_speed=np.ones(1),
        wave_speed=np.array([math.inf]),
        jam_density=np.array([math.inf]),
        capacity=np.ones(1),
    )
    with pytest.raises(ValueError, match="attraction"):
        _corridor_scenario(network=net, dynamics=dyn)


# ---------------------------------------------------------------------------
# content hash
# ---------------------------------------------------------------------------


def test_golden_dnl_hash_of_inline_corridor():
    # Pinned at implementation time: any byte drift in the dnl serialization
    # is a breaking change to the benchmark identity (P2).
    assert _corridor_scenario().content_hash() == GOLDEN_CORRIDOR_DNL_HASH
    # deterministic across independent constructions
    assert _corridor_scenario().content_hash() == _corridor_scenario().content_hash()


def test_dnl_hash_flips_on_every_physics_field():
    base = _diverge_scenario().content_hash()
    perturbed = [
        _diverge_scenario(dt=0.25),
        _diverge_scenario(n_steps=9),
        _diverge_scenario(rate_01=0.7),
        _diverge_scenario(breakpoints=(0.0, 1.5, 2.0)),
        _diverge_scenario(length=(1.25, 1.0, 1.0)),
        _diverge_scenario(free_speed=(1.25, 1.0, 1.0)),
        _diverge_scenario(wave_speed=(1.25, 1.0, 1.0)),
        _diverge_scenario(jam_density=(5.0, 4.0, 4.0)),
        _diverge_scenario(capacity=(1.5, 1.0, 1.0)),
        _diverge_scenario(split=(0.5, 0.5)),
    ]
    hashes = [s.content_hash() for s in perturbed]
    assert base not in hashes
    assert len(set(hashes)) == len(hashes)  # and all distinct from each other


def test_dnl_hash_ignores_static_bpr_fields():
    base = _diverge_scenario().content_hash()
    bpr_net = _network(
        "diverge-bpr",
        n_nodes=4,
        n_zones=3,
        init=[1, 4, 4],
        term=[4, 2, 3],
        capacity=np.full(3, 99.0),
        free_flow_time=np.full(3, 7.0),
        b=np.full(3, 0.15),
        power=np.full(3, 4.0),
        toll=np.full(3, 2.0),
    )
    assert _diverge_scenario(network=bpr_net).content_hash() == base


def test_dnl_hash_ignores_name_and_family():
    assert (
        _corridor_scenario(name="other", family="other-family").content_hash()
        == _corridor_scenario().content_hash()
    )


def test_static_braess_hash_unchanged_with_dnl_imported():
    # tabench.dnl.scenario is imported at the top of this module: the golden
    # static hash must be byte-identical anyway (additive-only guarantee).
    assert braess_scenario().content_hash() == GOLDEN_BRAESS_HASH
