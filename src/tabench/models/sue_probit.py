"""Probit stochastic user equilibrium via MSA with Monte Carlo loading.

The repo's first non-deterministic model (docs/design/adr-003). Probit SUE
(Daganzo & Sheffi 1977) has no closed-form loading map, so this solver runs
plain method of successive averages around the *sampled* loading map
``L_R(t(v))`` (Sheffi & Powell 1982; Sheffi 1985 ch. 12):

    v_{k+1} = v_k + (L_R(t(v_k)) - v_k) / k

With ``1/k`` steps the iterate is exactly the running average of the sampled
loads, so no Polyak averaging is needed (Powell & Sheffi 1982 predetermined
step sizes, already shipped for the logit solver).

``sue_theta`` carries ``beta`` (perception variance per unit free-flow time),
read from the scenario — task data, never a model factor (P7). ``draws_per_
iteration`` (R) is a declared factor: Sheffi (1985 figs 12.8-12.10) finds a
single draw per outer iteration gives the best convergence per unit work, so
R=1 is the default, but the factor stays exposed so the trade-off is
exhibitable.

Non-determinism (``deterministic=False``) routes this model onto the
stochastic track: ``run_experiment`` runs ``macroreps`` independent solver
trajectories, one per ``RngBundle(root_seed, macrorep)`` (P5/P8). ``provides_
gap=False`` because the R=1 self-reported direction norm ``||y_k - v_k||_1/D``
is a single extreme-point estimate that does not decay (it stays O(1) at
k=5000): the model must never early-stop on ``budget.target_met`` — budget
axes only. The self-report carries that raw norm plus a Sheffi eq. [12.52]
moving average as provenance; the harness certifies against its own pinned
Monte Carlo map (adr-003 Decision 1), not against these numbers.
"""

from __future__ import annotations

import time

import numpy as np

from ..core.budget import Budget, BudgetCoords
from ..core.capabilities import Capabilities
from ..core.factors import FactorSpec
from ..core.results import ResultBundle, Trace
from ..core.rng import RngBundle
from ..core.scenario import Scenario
from ._probit import ProbitEngine
from .base import TrafficAssignmentModel, register_model

__all__ = ["SueProbitMsaModel"]

_MOVING_AVERAGE_WINDOW = 3  # Sheffi eq. [12.52] convergence-measure smoothing


@register_model
class SueProbitMsaModel(TrafficAssignmentModel):
    """MSA around Monte Carlo probit loading: v_{k+1} = v_k + (L_R(t(v_k)) - v_k)/k."""

    name = "sue-probit-msa"
    capabilities = Capabilities(
        paradigm="sue",
        deterministic=False,
        provides_gap=False,
        seedable=True,
        # solve() raises without scenario.sue_theta (the probit perception beta).
        inputs_required=frozenset({"od_matrix", "sue_theta"}),
    )
    factors = {
        "draws_per_iteration": FactorSpec(
            default=1,
            kind="int",
            bounds=(1, 64),
            doc="Monte Carlo draws R averaged into each outer loading L_R(t) "
            "(Sheffi 1985 figs 12.8-12.10: R=1 converges best per AON sweep). "
            "One draw's all-or-nothing counts as one sp_call, so an outer "
            "iteration costs R sp_calls.",
        ),
    }

    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        theta = scenario.sue_theta
        if theta is None:
            raise ValueError(
                "sue-probit-msa requires an SUE scenario (scenario.sue_theta is "
                "None); beta is task data, not a model factor"
            )
        if scenario.sue_family != "probit":
            raise ValueError(
                f"sue-probit-msa is the probit-SUE solver but scenario "
                f"'{scenario.name}' declares sue_family={scenario.sue_family!r}; "
                "use sue-msa for the logit-SUE task"
            )
        start = time.perf_counter()
        network = scenario.network
        engine = ProbitEngine(network)
        total = scenario.demand.total
        r = self.factor_values["draws_per_iteration"]

        # Iteration 0: stochastic loading at free-flow costs (Sheffi ch. 12
        # Step 0), mirroring the logit solver.
        v = engine.load(
            network.link_cost(np.zeros(network.n_links)),
            scenario.demand,
            theta,
            rng.generator(source=0, replication=0),
            r,
        )
        sp_calls = r

        recent: list[float] = []
        k = 0
        while True:
            k += 1
            costs = network.link_cost(v)
            y = engine.load(
                costs, scenario.demand, theta, rng.generator(source=0, replication=k), r
            )
            sp_calls += r

            # Provenance only: the raw single-draw direction norm does not decay
            # at R=1, so it is never a stopping signal (provides_gap=False). Its
            # eq.[12.52] moving average is recorded alongside for readability.
            raw = float(np.abs(y - v).sum() / total) if total > 0 else 0.0
            recent.append(raw)
            if len(recent) > _MOVING_AVERAGE_WINDOW:
                recent.pop(0)
            moving_average = float(np.mean(recent))

            coords = BudgetCoords(
                iterations=k,
                sp_calls=sp_calls,
                wall_ms=1000.0 * (time.perf_counter() - start),
            )
            # Checkpoint v_k BEFORE the k-th update (adr-003 Decision 3): a
            # k-iteration run's final certified state is v_k, not v_{k+1}.
            trace.record(
                v,
                coords,
                sue_fixed_point_residual=raw,
                sue_residual_moving_average=moving_average,
            )
            if budget.exhausted(coords):
                break
            v = v + (y - v) / k

        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
