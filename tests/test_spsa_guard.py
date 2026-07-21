"""Guard tests for the base-class SPSA exp-site clamp (``spsa.py``).

The site clamps the log-space exponent at the float64 exp boundary
(``|exponent| <= 709``) before ``np.exp``; the base ``_project`` is the identity,
so an unprojected log-space excursion would otherwise exp to ``+-inf`` and hand a
non-finite demand candidate to the loss / best-iterate tracking. For normal-range
priors (``|log prior| < 709``) the clip never binds and the pinned spsa /
spsa-sumo traces are byte-identical. For subnormal or overflow-scale priors it
deliberately reshapes the candidate -- the protection working -- and the emitted
bytes CAN then differ from unclamped code (a clamped candidate can win
best-iterate tracking).

``test_spsa_clip_binds_on_subnormal_prior_via_real_estimate`` is the KILLABLE
test: it runs the real ``estimate()`` loop on a subnormal-prior braess task so the
clip actually binds, and fails if the clip is deleted. The expression-level test
is a secondary contract pin that does not execute the changed lines.
"""

from __future__ import annotations

import numpy as np

from tabench import braess_scenario
from tabench.core.budget import Budget
from tabench.core.results import Trace
from tabench.core.rng import SOURCE_OBSERVATION, RngBundle
from tabench.core.scenario import Demand
from tabench.estimation import EstimationTask, ODTrace
from tabench.estimation.spsa import SPSAEstimator
from tabench.models.frank_wolfe import BiconjugateFrankWolfeModel
from tabench.observe.levels import LinkCounts


class _RecordingSPSA(SPSAEstimator):
    """SPSA with base semantics intact (``_project`` stays the identity) that also
    records every candidate the loop hands to ``_project`` — so the test inspects
    the REAL exp-site output instead of a re-implemented copy of the clamp."""

    def __init__(self, **factors: object) -> None:
        super().__init__(**factors)
        self.recorded: list[np.ndarray] = []

    def _project(self, g: np.ndarray) -> np.ndarray:
        self.recorded.append(np.asarray(g, dtype=np.float64).copy())
        return g


def _subnormal_prior_braess_task() -> EstimationTask:
    """A real braess estimation task whose prior is SUBNORMAL on the single OD
    pair (0->1): ``u = log(1e-310) ~ -713.8``, so the exp-site clip binds."""
    sc = braess_scenario()
    tr = Trace()
    BiconjugateFrankWolfeModel().solve(
        sc, Budget(iterations=500, target_relative_gap=1e-10), RngBundle(0), tr
    )
    truth = tr.final.link_flows
    sensors = np.arange(sc.network.n_links)
    dataset = LinkCounts(sensors, 1, "none").observe(
        sc, truth, RngBundle(0).generator(SOURCE_OBSERVATION)
    )
    prior = np.zeros((sc.network.n_zones, sc.network.n_zones))
    prior[0, 1] = 1e-310  # subnormal -> log ~ -713.8 < -709
    return EstimationTask(
        name="spsa-clip-subnormal",
        network=sc.network,
        prior=Demand(prior),
        dataset=dataset,
        identifiability={},
        scenario_hash=sc.content_hash(),
        seed=0,
    )


def test_spsa_clip_binds_on_subnormal_prior_via_real_estimate() -> None:
    """Killable end-to-end test that EXECUTES spsa.py's changed lines. Running the
    real ``estimate()`` on the subnormal-prior task drives the exp-site clip to
    bind: with the clip the clamped candidate component is exactly ``exp(-709)``;
    deleting the clip makes it ``exp(~-713.8) ~ 1e-310`` and this assertion fails
    (the confirmed B5-review MAJOR: the old test never called ``estimate()``)."""
    est = _RecordingSPSA()
    est.estimate(
        _subnormal_prior_braess_task(), Budget(iterations=2), RngBundle(0), ODTrace()
    )
    assert est.recorded, "estimate() produced no _project calls"
    clamp_floor = np.exp(-709.0)  # exp of the clamped exponent, exactly
    assert any((rec == clamp_floor).any() for rec in est.recorded), (
        "no recorded candidate equals exp(-709): the exp-site clip did not bind "
        "(without it the subnormal prior yields exp(~-713.8) ~ 1e-310)"
    )


def test_spsa_exp_clamp_expression_contract() -> None:
    """Secondary contract pin (does NOT execute spsa.py): the exp-site expression
    keeps extreme synthetic exponents finite and strictly positive, ceiling at
    ``exp(709)``. Kept alongside the killable test above as a readable statement
    of the clamp's intent."""
    est = SPSAEstimator()  # base class: _project is the identity projection
    u = np.array([720.0, -720.0, 5.0, 1e6, -1e6])
    with np.errstate(over="ignore"):
        assert not np.isfinite(np.exp(u)).all()  # unclamped overflow -- the hazard
    g = est._project(np.exp(np.clip(u, -709.0, 709.0)))
    assert np.isfinite(g).all()
    assert (g > 0.0).all()
    assert g.max() == np.exp(709.0)
