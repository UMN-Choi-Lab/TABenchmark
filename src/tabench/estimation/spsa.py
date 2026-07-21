"""spsa: Spall's (1992) simultaneous-perturbation calibration baseline.

The only shipped estimator that never sees ``P`` — it treats assignment as a
black-box oracle, exactly how a microsimulator or a neural surrogate would be
calibrated (Lu et al. 2015 for OD-SPSA practice). The demand is parametrized in
log space on the prior's positive support (``u = log g``: scale-free, positive
by construction). Each iteration draws a Rademacher perturbation ``Delta`` from
the estimator's own RNG source, evaluates the count-misfit loss at
``exp(u +- c_k Delta)`` (two inner assignments), forms the SP gradient estimate
``ghat = (L+ - L-) / (2 c_k) Delta`` (``Delta_i in {+-1}`` so ``Delta^-1 =
Delta``), and steps ``u <- u - a_k ghat``.

Gains follow Spall's practical exponents ``a_k = a / (k+1+A)^0.602`` and
``c_k = c / (k+1)^0.101``. Safeguards: per-component clipping of ``a_k ghat``
(blow-up guard) and best-iterate tracking over every evaluated point. The
self-reported ``obs_count_rmse`` is the square root of the count-fit term only,
never the optional prior-deviation penalty, so it stays an honest count RMSE
even when ``prior_weight > 0`` (P1). Checkpoints are emitted sparsely (every
``max(1, iters // 15)`` iterations plus the final) — each costs a full pinned
certificate. Seeded but not deterministic (``deterministic=False,
seedable=True``) so it is macroreplicated; the same ``(root_seed, macrorep)``
reproduces the trace bit-for-bit. Cost ``2 * k_inner`` shortest-path calls per
iteration.
"""

from __future__ import annotations

import numpy as np

from ..core.budget import Budget, BudgetCoords
from ..core.factors import FactorSpec
from ..core.rng import RngBundle
from ..core.scenario import Demand
from ..models._paths import PathEngine
from ._proportions import active_pairs, od_from_pairs, proportion_matrix
from .base import (
    EstimationTask,
    ODEstimator,
    ODResultBundle,
    ODTrace,
    _estimation_capabilities,
    register_estimator,
)

__all__ = ["SPSAEstimator"]


@register_estimator
class SPSAEstimator(ODEstimator):
    """Black-box OD calibration by simultaneous perturbation (Spall 1992)."""

    name = "spsa"
    capabilities = _estimation_capabilities(deterministic=False, seedable=True)
    factors = {
        "k_inner": FactorSpec(
            default=15, kind="int", bounds=(1, 5000),
            doc="Inner MSA/AON sweeps per assignment-oracle evaluation.",
        ),
        "iters": FactorSpec(
            default=60, kind="int", bounds=(1, 100000),
            doc="SPSA iterations (each costs two inner assignments).",
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
            default=1.0, kind="float", bounds=(1e-6, 1e6),
            doc="Per-component clip on a_k*ghat in log space (blow-up guard).",
        ),
        "prior_weight": FactorSpec(
            default=0.0, kind="float", bounds=(0.0, 1e12),
            doc="Optional weight on squared relative prior deviation in the loss.",
        ),
    }

    def _assign_obs(self, network, engine, prior_matrix, pairs, sensors, g) -> np.ndarray:
        demand_g = Demand(matrix=od_from_pairs(prior_matrix, pairs, g))
        _, _, flows = proportion_matrix(
            network, demand_g, self.factor_values["k_inner"], pairs=pairs, engine=engine
        )
        return flows[sensors]

    def _sp_cost_per_eval(self) -> int:
        """Shortest-path calls charged per oracle evaluation. A simulator-oracle
        subclass whose engine exposes no SP count overrides this to 0 (it then
        DISCLOSES sp_calls=0 rather than fabricating it), so the loop is reused
        without duplicating it or over-reporting (adr-028)."""
        return int(self.factor_values["k_inner"])

    def _project(self, g: np.ndarray) -> np.ndarray:
        """Project a demand candidate onto the feasible box BEFORE it is
        evaluated and tracked. IDENTITY in the base estimator (returns ``g``
        unchanged, byte-for-byte); a bounded subclass overrides it so the
        EVALUATED point and the tracked/emitted best-iterate are the same array
        (thesis Eqs. 3.5-3.6 step 5; P1 emitted==evaluated, adr-028)."""
        return g

    def _project_log(self, u: np.ndarray) -> np.ndarray:
        """Project the log-demand iterate onto the (log) box AFTER the update.
        IDENTITY in the base estimator; a bounded subclass overrides it (thesis
        step 8) so the iterate cannot freeze outside the box on a flat corner
        (both perturbations clamping to one boundary -> zero SP gradient)."""
        return u

    def estimate(
        self, task: EstimationTask, budget: Budget, rng: RngBundle, trace: ODTrace
    ) -> ODResultBundle:
        network = task.network
        engine = PathEngine(network)
        prior_matrix = task.prior.matrix
        pairs = active_pairs(prior_matrix)
        sensors = np.asarray(task.dataset.payload["sensor_links"], dtype=np.int64)
        counts_mean = np.asarray(task.dataset.payload["counts"], dtype=np.float64).mean(axis=0)
        n_obs = max(len(sensors), 1)

        g_pr = np.array([prior_matrix[i, j] for (i, j) in pairs], dtype=np.float64)
        u = np.log(g_pr)
        gen = rng.generator(source=0)

        a = self.factor_values["a"]
        c = self.factor_values["c"]
        iters = int(self.factor_values["iters"])
        a_stab = self.factor_values["a_stability_frac"] * iters
        clip = self.factor_values["step_clip"]
        prior_weight = self.factor_values["prior_weight"]

        def loss(g: np.ndarray) -> tuple[float, float]:
            """Return ``(total, fit)``: the optimization objective and the pure
            count-fit term. They coincide when ``prior_weight == 0``."""
            v_obs = self._assign_obs(network, engine, prior_matrix, pairs, sensors, g)
            fit = float(np.sum((v_obs - counts_mean) ** 2) / n_obs)
            total = fit
            if prior_weight > 0.0:
                total += prior_weight * float(np.sum(((g - g_pr) / g_pr) ** 2))
            return total, fit

        best_g = self._project(g_pr.copy())
        best_loss, best_fit = loss(best_g)
        sp_calls = self._sp_cost_per_eval()
        stride = max(1, iters // 15)
        for k in range(iters):
            a_k = a / (k + 1 + a_stab) ** 0.602
            c_k = c / (k + 1) ** 0.101
            delta = 2.0 * gen.integers(0, 2, size=u.size).astype(np.float64) - 1.0
            # Clamp the log-space exponent at the float64 exp boundary
            # (|exponent| <= 709) before exp: the base-class _project is identity,
            # so an unprojected log-space excursion would otherwise exp to +-inf.
            # For normal-range priors (|log prior| < 709) the clip never binds and
            # the pinned traces are byte-identical (verified). For subnormal or
            # overflow-scale priors it deliberately reshapes the candidate -- the
            # protection working -- and the emitted bytes CAN then differ from
            # unclamped code (a clamped candidate can win best-iterate tracking).
            g_plus = self._project(np.exp(np.clip(u + c_k * delta, -709.0, 709.0)))
            g_minus = self._project(np.exp(np.clip(u - c_k * delta, -709.0, 709.0)))
            l_plus, f_plus = loss(g_plus)
            l_minus, f_minus = loss(g_minus)
            sp_calls += 2 * self._sp_cost_per_eval()
            ghat = (l_plus - l_minus) / (2.0 * c_k) * delta
            step = np.clip(a_k * ghat, -clip, clip)
            u = self._project_log(u - step)
            for cand, cand_loss, cand_fit in (
                (g_plus, l_plus, f_plus),
                (g_minus, l_minus, f_minus),
            ):
                if cand_loss < best_loss:
                    best_loss, best_fit = cand_loss, cand_fit
                    best_g = cand.copy()
            coords = BudgetCoords(iterations=k + 1, sp_calls=sp_calls, wall_ms=0.0)
            done = budget.exhausted(coords)
            if (k + 1) % stride == 0 or (k + 1) == iters or done:
                # Sparse emission (ADR-002 Decision 2): each checkpoint is a full
                # pinned certificate. Self-report the count-fit term only, so the
                # P1 honesty diff stays valid when prior_weight > 0.
                trace.record(
                    od_from_pairs(prior_matrix, pairs, best_g),
                    coords,
                    obs_count_rmse=float(np.sqrt(max(best_fit, 0.0))),
                )
            if done:
                break

        return ODResultBundle(
            estimator_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
