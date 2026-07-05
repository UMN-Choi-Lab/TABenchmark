"""Monte Carlo probit network loading — the sampled SUE loading map.

Probit route choice (Daganzo & Sheffi 1977) has no closed-form loading map:
route-choice probabilities are high-dimensional normal orthant integrals, and
the practical loading is simulation (Sheffi & Powell 1982). This engine draws
perceived link times ``T_a = max(t_a(v) + sqrt(beta * t0_a) * Z, floor)`` with
``Z`` iid ``N(0, 1)`` per link per draw, and averages ``R`` all-or-nothing
assignments at those times:

    L_R(t) = (1/R) sum_i AON(T_i)

an unbiased estimate of the probit loading (Sheffi 1985 p. 327: unbiasedness
suffices for MSA convergence regardless of ``R``).

The perception variance is proportional to **free-flow** time (Sheffi 1985 eq.
[12.57], the Sheffi & Powell equilibrium form) — flow-independent, so the
perturbation matrix never depends on ``t(v)``. That is what lets the harness
certificate pin one perturbation stream across every checkpoint (docs/design/
adr-003, Decision 1). Negative sampled times are truncated at a small positive
floor (Sheffi p. 300 sanctions truncation at zero; the PathEngine requires
strictly positive costs); the truncated map — not the untruncated normal — is
the task definition, with a beta-dependent bias documented in the card.

Both the ``sue-probit-msa`` solver and the harness Evaluator call this engine,
so the sampled map they equilibrate and certify is one implementation. They
differ only in which stream they draw the perturbations from (P8): the solver
draws fresh per outer iteration, the Evaluator pins one matrix per task.
"""

from __future__ import annotations

import numpy as np

from ..core.scenario import Demand, Network
from ._paths import PathEngine

__all__ = ["ProbitEngine", "FLOOR"]

FLOOR = 1e-9  # positive truncation floor (PathEngine requires costs > 0)


class ProbitEngine:
    """Reusable per-network engine for Monte Carlo probit loading."""

    def __init__(self, network: Network) -> None:
        self.network = network
        self._paths = PathEngine(network)

    def perturbations(
        self, beta: float, gen: np.random.Generator, n_draws: int
    ) -> np.ndarray:
        """Perception-error matrix ``E = sqrt(beta * t0) * Z``, shape (n_draws, n_links).

        Flow-independent by construction (free-flow variance), so the same
        matrix perturbs any cost vector — the property the pinned certificate
        relies on. ``beta`` is task data (``scenario.sue_theta``), never a model
        factor.
        """
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError(f"beta must be finite and > 0, got {beta!r}")
        sigma = np.sqrt(beta * self.network.free_flow_time)
        z = gen.standard_normal((n_draws, self.network.n_links))
        return sigma[None, :] * z

    def load_perturbed(
        self,
        costs: np.ndarray,
        demand: Demand,
        perturbations: np.ndarray,
        return_samples: bool = False,
    ):
        """MC average of AON loads at ``max(costs + E_i, FLOOR)`` over the rows of ``E``.

        Pure function of ``(costs, demand, perturbations)``. With
        ``return_samples`` the individual per-draw link flows are also returned
        (the certificate's jackknife and floor need the sample spread).
        """
        costs = np.asarray(costs, dtype=np.float64)
        times = np.maximum(costs[None, :] + perturbations, FLOOR)
        n_draws = times.shape[0]
        acc = np.zeros(self.network.n_links, dtype=np.float64)
        samples = (
            np.empty((n_draws, self.network.n_links), dtype=np.float64)
            if return_samples
            else None
        )
        for i in range(n_draws):
            y, _ = self._paths.all_or_nothing(times[i], demand)
            acc += y
            if return_samples:
                samples[i] = y
        mean = acc / n_draws
        return (mean, samples) if return_samples else mean

    def load(
        self,
        costs: np.ndarray,
        demand: Demand,
        beta: float,
        gen: np.random.Generator,
        n_draws: int,
        return_samples: bool = False,
    ):
        """Draw ``n_draws`` fresh perturbations from ``gen`` and MC-average AON loads.

        The solver path: one call per outer MSA iteration, drawing from that
        iteration's reserved stream so trajectories are byte-reproducible (P8).
        """
        e = self.perturbations(beta, gen, n_draws)
        return self.load_perturbed(costs, demand, e, return_samples=return_samples)
