"""``triangular_bottleneck_dynamic_scenario`` / ``greenshields_bottleneck_dynamic_scenario``
(dnl-core additive, tutorials batch): the two anchors 1-2 (``single_link_dynamic_scenario``,
``bottleneck_dynamic_scenario``) omit — both are point-queue (``kappa = inf``), which
``CTMLink``/``LTMLink``/``GodunovLink`` reject at construction. These two finite-jam
corridors are the shared, importable instances the ``05-dnl`` tutorials run on instead of
each hand-rolling a ``Network``/``LinkDynamics``/``DynamicScenario`` triple.

``triangular_bottleneck_dynamic_scenario`` is byte-identical in shape to the private
``_bottleneck_scenario`` anchor in ``tests/test_dnl_ctm.py`` / ``tests/test_dnl_ltm.py``
(same RH-shock anchor: storage 14 at ``t = n_steps = 12``); this file re-derives that
anchor against the PUBLIC scenario rather than re-deriving physics, and additionally pins
both scenarios' content hashes (golden, mirroring ``test_dnl_loader.py``'s
``GOLDEN_SINGLE_LINK_HASH``/``GOLDEN_BOTTLENECK_HASH`` pattern) and confirms neither
existing anchor's hash moved.
"""

import numpy as np
import pytest

import tabench
from tabench.dnl import (
    CTMLink,
    GodunovLink,
    GreenshieldsFD,
    LTMLink,
    NetworkLoader,
    bottleneck_dynamic_scenario,
    greenshields_bottleneck_dynamic_scenario,
    single_link_dynamic_scenario,
    triangular_bottleneck_dynamic_scenario,
)
from tabench.metrics import DNLEvaluator

GOLDEN_TRIANGULAR_HASH = "d4148843c3292e4249fb6ca8585065f71f8e3f4684bfbf7797db1120e0d984d7"
GOLDEN_GREENSHIELDS_HASH = "03afa861280080211b980412b3f0caa117e3dca700ad698787434e3e6457c247"
# The pre-existing point-queue anchors (test_dnl_loader.py) — pinned again here so a
# future edit to this file's helpers cannot silently move them.
GOLDEN_SINGLE_LINK_HASH = "93b258aa6ae6c35264006b3969bb940e90a3ad71158fdd4bcf8e8f6e1ad6a2d7"
GOLDEN_BOTTLENECK_HASH = "ecdea09f1c569e0e775f294b0950ab3dbea4e2982c81d2685ba9a7ade463266e"


def test_public_exports_are_available() -> None:
    assert callable(tabench.triangular_bottleneck_dynamic_scenario)
    assert callable(tabench.greenshields_bottleneck_dynamic_scenario)
    assert tabench.triangular_bottleneck_dynamic_scenario is triangular_bottleneck_dynamic_scenario
    assert (
        tabench.greenshields_bottleneck_dynamic_scenario
        is greenshields_bottleneck_dynamic_scenario
    )


def test_new_scenario_hashes_are_pinned() -> None:
    assert triangular_bottleneck_dynamic_scenario().content_hash() == GOLDEN_TRIANGULAR_HASH
    assert greenshields_bottleneck_dynamic_scenario().content_hash() == GOLDEN_GREENSHIELDS_HASH


def test_existing_point_queue_anchor_hashes_did_not_move() -> None:
    """The additive change must not touch the two shipped point-queue anchors."""
    assert single_link_dynamic_scenario().content_hash() == GOLDEN_SINGLE_LINK_HASH
    assert bottleneck_dynamic_scenario().content_hash() == GOLDEN_BOTTLENECK_HASH


def test_existing_point_queue_anchors_still_reject_finite_link_models() -> None:
    """The motivation for these two new scenarios: CTM/LTM/Godunov cannot load
    the point-queue anchors (kappa = inf) at all — CTM/LTM raise their own
    "finite jam density" guard, GodunovLink's inf leaks into its inner
    GreenshieldsFD construction instead (still a ValueError, different message)."""
    for model in (CTMLink, LTMLink):
        with pytest.raises(ValueError, match="finite jam density"):
            NetworkLoader(bottleneck_dynamic_scenario(), model).run()
    with pytest.raises(ValueError, match="kappa must be finite"):
        NetworkLoader(bottleneck_dynamic_scenario(), GodunovLink).run()


@pytest.mark.parametrize("model", [CTMLink, LTMLink])
def test_triangular_bottleneck_matches_the_rh_shock_anchor(model) -> None:
    """Reproduces the RH-shock closed form from test_dnl_ctm.py/test_dnl_ltm.py's
    private ``_bottleneck_scenario`` (vf=w=1, kappa=4, cap [2, 0.5], rate 1.5)
    against the now-public scenario: n_in[0] = 1.5*edges, n_out[0] = max(0, 0.5*
    (edges-4)), storage = k_B*L = 3.5*4 = 14 at the t=12 horizon."""
    scenario = triangular_bottleneck_dynamic_scenario()
    out = NetworkLoader(scenario, model).run()
    metrics = DNLEvaluator(scenario).evaluate(out)
    edges = scenario.grid.edges

    assert metrics["dnl_feasible"] == 1.0
    assert metrics["conservation_residual"] <= 1e-9
    np.testing.assert_allclose(out.n_in[0], 1.5 * edges, atol=1e-9)
    np.testing.assert_allclose(out.n_out[0], np.maximum(0.0, 0.5 * (edges - 4.0)), atol=1e-9)
    assert out.n_in[0, -1] - out.n_out[0, -1] == pytest.approx(14.0, abs=1e-9)


def test_triangular_bottleneck_ltm_matches_ctm_exactly() -> None:
    scenario = triangular_bottleneck_dynamic_scenario()
    ctm = NetworkLoader(scenario, CTMLink).run()
    ltm = NetworkLoader(scenario, LTMLink).run()
    # Genuinely exact (max diff 0.0, independently re-verified) -- tighten to
    # array_equal so the assert matches the "byte-for-byte" claim the notebook prints,
    # not merely atol=1e-9.
    np.testing.assert_array_equal(ctm.n_in, ltm.n_in)
    np.testing.assert_array_equal(ctm.n_out, ltm.n_out)


def test_greenshields_bottleneck_certifies_under_godunov() -> None:
    scenario = greenshields_bottleneck_dynamic_scenario()
    out = NetworkLoader(scenario, GodunovLink).run()
    metrics = DNLEvaluator(scenario).evaluate(out)
    assert metrics["dnl_feasible"] == 1.0
    assert metrics["conservation_residual"] <= 1e-9
    assert metrics["storage_residual"] <= 1e-9


def test_greenshields_bottleneck_settles_on_the_analytic_parabola_roots() -> None:
    """Distinctive Greenshields physics: the upstream link's near-origin cell
    settles at the free-branch root of Q(k)=1.5 (arrival rate), and its
    near-bottleneck cell settles at the congested-branch root of
    supply_at(k)=0.5 (the bottleneck capacity) — both recomputed here from
    ``GreenshieldsFD`` directly (a brentq root, not a hand quote), matching the
    scheme's actual occupancy to a tight tolerance (no coarse-grid diffusion on
    these two constant-state cells)."""
    from scipy.optimize import brentq

    scenario = greenshields_bottleneck_dynamic_scenario()
    loader = NetworkLoader(scenario, GodunovLink)
    loader.run()
    fd = GreenshieldsFD(vf=1.0, kappa=8.0)  # link 0's Greenshields-consistent FD
    k_free = brentq(lambda k: float(fd.flow_at(np.array([k]))[0]) - 1.5, 0.0, fd.critical_density)
    k_cong = brentq(
        lambda k: float(fd.supply_at(np.array([k]))[0]) - 0.5, fd.critical_density, fd.jam_density
    )
    density = loader.links[0].occupancy / loader.links[0]._dx
    assert density[0] == pytest.approx(k_free, abs=1e-3)
    assert density[-1] == pytest.approx(k_cong, abs=1e-3)


def test_greenshields_scenario_is_also_a_valid_triangular_link_dynamics() -> None:
    """Off GodunovLink, the SAME scenario's LinkDynamics.fd(a) is a plain
    (symmetric) TriangularFD — CTM/LTM load it as such without complaint,
    confirming the Greenshields substitution is purely a GodunovLink-side
    reinterpretation of the same bytes (P2: one instance, two link models)."""
    scenario = greenshields_bottleneck_dynamic_scenario()
    for model in (CTMLink, LTMLink):
        metrics = DNLEvaluator(scenario).evaluate(NetworkLoader(scenario, model).run())
        assert metrics["dnl_feasible"] == 1.0
        assert metrics["conservation_residual"] <= 1e-9


def test_new_scenarios_do_not_move_the_golden_braess_hash() -> None:
    from tabench.data.builtin import braess_scenario

    assert (
        braess_scenario().content_hash()
        == "cf00f411cdccec88019979e8cfbf3d8014ba590688b285a1db667315ac96762d"
    )
