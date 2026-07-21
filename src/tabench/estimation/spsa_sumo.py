"""spsa-sumo: Balakrishna et al. (2007) offline SPSA calibration against a
production traffic simulator (SUMO ``marouter``).

The shipped ``spsa`` estimator calibrates demand by simultaneous perturbation
against the repo's OWN in-process MSA/AON oracle; this subclass makes the loop
REAL -- its inner assignment oracle is the :class:`SumoMarouterModel` adapter
(adr-027), a subprocess production engine with its own hardcoded linear cost
law, real refusal surfaces, and byte-deterministic output. It is the T2
transport of adr-027's simulator-to-benchmark gap: SPSA descends the count
misfit under marouter's mapped law while the certificate scores every emitted OD
through the SAME pinned bfw map (P1, ADR-002 Decision 2), so the standard
self-vs-certified honesty diff now MEASURES the simulator-in-the-loop bias. It is
NOT a hard bound: measured ``|self - certified|`` is ~7e-4 count-RMSE on the
clean two-route anchor and ~2e-3 under poisson counts -- the same ORDER as the
ADR-027 mapping floor (itself a relative gap, ~1.7e-4 Braess / ~5.4e-4
two-route), and meaningful only when no box projection engaged (a binding box
makes the emitted and evaluated points coincide by construction, adr-028).

**Scope (adr-028): DEMAND-ONLY** on the unchanged ``EstimationTask``,
calibrating link counts on ``power == 1`` toll-free fixed-demand UE instances --
the adapter's documented capability envelope IS the estimator's, refused up
front. Balakrishna et al.'s JOINT demand+supply contribution and within-day DTA
setting are NOT shipped (each needs a new task family + certificate surface);
this row implements their SPSA-on-a-black-box-simulator methodology at the H=1
static special case.

**Budget / determinism.** marouter exposes no shortest-path count, so
``sp_calls`` is DISCLOSED as 0 (never fabricated) and an ``sp_calls``-only budget
is refused up front (the inverted adr-025 lesson); the loop is bounded by the
``iters`` factor and by any ``iterations`` / ``wall_seconds`` budget, the latter
threaded as ONE deadline across all 2I+1 inner solves. The engine's SUE path is
RNG-free, so the whole estimate is bit-reproducible from ``(root_seed,
macrorep)`` -- SPSA's Rademacher draws are the only noise (like ``spsa``).

``eclipse-sumo`` is an optional extra (``pip install tabench[sumo]``); this
module imports ``sumo`` (through the adapter) and is guarded in
``estimation/__init__.py`` -- the FIRST guarded estimator, byte-parallel to the
model guards -- so the numpy/scipy core stays dependency-free.
"""

from __future__ import annotations

import time

import numpy as np

from ..core.budget import Budget
from ..core.factors import FactorSpec
from ..core.results import Trace
from ..core.rng import RngBundle
from ..core.scenario import Demand, Scenario
from ..models.adapters.sumo_marouter import SumoMarouterModel, _engine_version
from ._proportions import active_pairs, od_from_pairs
from .base import (
    EstimationTask,
    ODResultBundle,
    ODTrace,
    _estimation_capabilities,
    register_estimator,
)
from .spsa import SPSAEstimator

__all__ = ["SumoSPSAEstimator"]


@register_estimator
class SumoSPSAEstimator(SPSAEstimator):
    """SPSA OD calibration against the SUMO ``marouter`` simulator (Balakrishna 2007).

    Reuses every piece of :class:`SPSAEstimator` -- the log-space demand
    parametrization, Spall gains, per-component clip, best-iterate tracking and
    sparse pinned-certificate checkpoints -- overriding only the oracle hook
    ``_assign_obs`` (whose black box is now marouter, not the in-process MSA) and
    ``_sp_cost_per_eval`` (0: no SP count to charge), plus an ``estimate`` prelude
    for the up-front refusals and the single wall deadline.
    """

    name = "spsa-sumo"
    capabilities = _estimation_capabilities(deterministic=False, seedable=True)
    factors = {
        # --- SPSA gains (same names/semantics as `spsa`, read by the parent loop) ---
        "iters": FactorSpec(
            default=30, kind="int", bounds=(1, 100000),
            doc="SPSA iterations (each costs two inner marouter solves).",
        ),
        "a": FactorSpec(
            default=0.35, kind="float", bounds=(0.0, 1e6),
            doc="Step-gain numerator a in a_k = a/(k+1+A)^0.602.",
        ),
        "c": FactorSpec(
            default=0.15, kind="float", bounds=(1e-9, 1e6),
            doc="Perturbation-gain numerator c in c_k = c/(k+1)^0.101.",
        ),
        "a_stability_frac": FactorSpec(
            default=0.1, kind="float", bounds=(0.0, 10.0),
            doc="Spall's A as a fraction of the iteration budget (step stabilizer).",
        ),
        "step_clip": FactorSpec(
            default=0.5, kind="float", bounds=(1e-6, 1e6),
            doc="Per-component clip on a_k*ghat in log space (blow-up guard). "
            "Balakrishna's percentage-of-magnitude step cap; defaulted to the "
            "plateau-escape value 0.5 (adr-028), tighter than `spsa`'s 1.0.",
        ),
        "prior_weight": FactorSpec(
            default=0.0, kind="float", bounds=(0.0, 1e12),
            doc="GLS z2 weight on squared relative prior deviation. Rescues "
            "plateau escape but biases toward the prior; NEVER enters the scored "
            "count RMSE (the `spsa` P1 convention).",
        ),
        # --- thesis box projection (Eqs. 3.5-3.6, evaluation-time step 5) ---
        "demand_lo_frac": FactorSpec(
            default=0.0, kind="float", bounds=(0.0, 1e6),
            doc="Lower demand box as a fraction of the prior. Log space already "
            "guarantees g > 0, so 0 leaves the lower projection inert.",
        ),
        "demand_hi_frac": FactorSpec(
            default=100.0, kind="float", bounds=(1.0, 1e12),
            doc="Upper demand box as a multiple of the prior -- the thesis "
            "box constraint / overflow guard clamping a runaway log-space "
            "excursion. Wide by default so demand-only recovery is unclamped "
            "(adr-028); tighten it to enforce a prior-anchored feasible region.",
        ),
        # --- inner marouter oracle factors (delegated to the adapter, adr-027) ---
        "inner_iterations": FactorSpec(
            default=50, kind="int", bounds=(1, 100000),
            doc="marouter --max-iterations per inner solve (the adapter budget).",
        ),
        "logit_theta": FactorSpec(
            default=200.0, kind="float", bounds=(0.0, 1e6),
            doc="marouter logit dispersion; the adapter default, calibrated on the "
            "ASYMMETRIC two-route anchor -- NEVER re-tuned on Braess (adr-027).",
        ),
        "paths": FactorSpec(
            default=4, kind="int", bounds=(1, 64),
            doc="marouter --paths: k-shortest paths per OD for the SUE route set.",
        ),
        "time_scale": FactorSpec(
            default=1.0, kind="float", bounds=(0.2, 30.0),
            doc="Adapter tau (seconds per native cost unit); the compile envelope.",
        ),
        "min_lanes": FactorSpec(
            default=1, kind="int", bounds=(1, 1000),
            doc="Adapter minimum quantized lane count (flow-scale resolution).",
        ),
    }

    def _sp_cost_per_eval(self) -> int:
        # marouter exposes no shortest-path count, so sp_calls is DISCLOSED as 0
        # rather than fabricated from a meaningless k_inner (the inverted adr-025
        # lesson). An sp_calls-only budget is refused up front in estimate().
        return 0

    def _project(self, g: np.ndarray) -> np.ndarray:
        # Thesis step-5 evaluation projection: clamp the candidate onto the box
        # [lo_frac, hi_frac] * prior in the PARENT loop -- before loss() and
        # best-iterate tracking -- so the evaluated, tracked and emitted demand
        # are one array (P1 emitted==evaluated, adr-028 Decision 6). Log space
        # already gives g > 0 (the lower clip is normally inert); the upper clip
        # caps a runaway log-space excursion.
        return np.clip(np.asarray(g, dtype=np.float64), self._lo_vec, self._hi_vec)

    def _project_log(self, u: np.ndarray) -> np.ndarray:
        # Thesis step-8 iterate projection: clamp log-demand to [log lo, log hi]
        # after the update so the iterate cannot freeze outside the box on a flat
        # corner (both perturbations clamping to one boundary -> zero SP gradient
        # from the deterministic oracle).
        return np.clip(np.asarray(u, dtype=np.float64), self._log_lo_vec, self._log_hi_vec)

    def _remaining_wall(self) -> float | None:
        """Seconds left on the shared deadline for the next inner solve, or
        ``None`` when no wall budget was set. A non-positive remainder is passed
        through so the adapter refuses it with the engine command in the message
        (its own compile-phase wall check), keeping crash discipline."""
        if self._deadline is None:
            return None
        return self._deadline - time.perf_counter()

    def _assign_obs(self, network, engine, prior_matrix, pairs, sensors, g) -> np.ndarray:
        # `g` is ALREADY box-projected by the parent loop (via `_project` above)
        # before loss evaluation and best-iterate tracking, so the evaluated,
        # tracked and emitted demand are one array -- P1 emitted==evaluated
        # (adr-028 Decision 6). This hook only runs the marouter oracle.
        od = od_from_pairs(prior_matrix, pairs, np.asarray(g, dtype=np.float64))
        inner = Scenario(
            name=f"{self._task_name}-inner", network=network, demand=Demand(matrix=od)
        )
        budget = Budget(
            iterations=self._inner_iterations, wall_seconds=self._remaining_wall()
        )
        tr = Trace()
        # marouter's SUE path is RNG-free -> a constant seed keeps the whole
        # estimate bit-reproducible from (root_seed, macrorep); the only noise is
        # SPSA's own Rademacher draws (adr-028). An engine RuntimeError here
        # (timeout/crash) propagates out of estimate() -- crash discipline.
        self._oracle.solve(inner, budget, RngBundle(0), tr)
        return tr.final.link_flows[np.asarray(sensors, dtype=np.int64)]

    def estimate(
        self, task: EstimationTask, budget: Budget, rng: RngBundle, trace: ODTrace
    ) -> ODResultBundle:
        # sp_calls is unmappable for marouter (no SP count): an sp_calls-only
        # budget cannot bound the run and is refused up front rather than being
        # silently ignored while `iters` runs anyway (mirror of the adapter; the
        # inverted adr-025 lesson). iterations/wall_seconds bound the loop.
        if budget.iterations is None and budget.wall_seconds is None:
            raise ValueError(
                "spsa-sumo cannot honor an sp_calls-only budget (its marouter "
                "oracle exposes no shortest-path count); constrain iterations or "
                "wall_seconds so the calibration loop is bounded (adr-028)."
            )

        # Build the inner oracle once (its factors are the delegated adapter
        # factors) and reuse it across every 2I+1 evaluation.
        self._oracle = SumoMarouterModel(
            assignment_method="SUE",
            route_choice="logit",
            logit_theta=self.factor_values["logit_theta"],
            paths=int(self.factor_values["paths"]),
            time_scale=self.factor_values["time_scale"],
            min_lanes=int(self.factor_values["min_lanes"]),
        )
        self._inner_iterations = int(self.factor_values["inner_iterations"])
        self._task_name = task.name

        # Delegate the adapter's own unrepresentability envelope (power != 1,
        # nonzero fixed cost, SUE-family named fields), surfaced fast on a probe
        # scenario BEFORE any inner solve or checkpoint. The estimator's scenario
        # envelope IS the adapter's, by construction (adr-028).
        probe = Scenario(name=task.name, network=task.network, demand=task.prior)
        self._oracle._refuse_unrepresentable(probe)

        # ONE wall deadline threaded across ALL inner solves (a tight wall then
        # kills a single mid-loop solve with the engine command in the message,
        # not the loop with an opaque error).
        self._deadline = (
            time.perf_counter() + budget.wall_seconds
            if budget.wall_seconds is not None
            else None
        )

        # Box vectors on the prior support (thesis Eqs. 3.5-3.6), computed once.
        prior_matrix = task.prior.matrix
        obs_pairs = active_pairs(prior_matrix)
        g_pr = np.array([prior_matrix[i, j] for (i, j) in obs_pairs], dtype=np.float64)
        self._lo_vec = float(self.factor_values["demand_lo_frac"]) * g_pr
        self._hi_vec = float(self.factor_values["demand_hi_frac"]) * g_pr
        # Log-space box for the step-8 iterate projection. A zero lower fraction
        # (log space already guarantees g > 0) maps to -inf: an inert lower clip.
        pos = self._lo_vec > 0.0
        self._log_lo_vec = np.where(pos, np.log(np.where(pos, self._lo_vec, 1.0)), -np.inf)
        self._log_hi_vec = np.log(self._hi_vec)

        bundle = super().estimate(task, budget, rng, trace)
        # Surface the wrapped marouter engine version into the RECORDED provenance,
        # mirroring odme_dtalite's seed_info engine tag (adr-027): marouter's
        # vdf/capacity tables are hardcoded in the SUMO source and could change
        # between releases, so the running engine identity belongs in the bundle
        # identity, not just the base rng.describe() (never scored -- P1). The whole
        # estimate already ran under this engine, so reading it here is post-hoc.
        bundle.seed_info = {**bundle.seed_info, "engine": _engine_version()}
        return bundle
