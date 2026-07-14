"""Friesz et al. (1993) VI dynamic user equilibrium on parallel bottleneck
routes (docs/design/adr-022-vi-due.md).

The simultaneous route-AND-departure-time (SRDC) dynamic user equilibrium:
``N`` travelers choose a route ``r`` (free-flow time ``f_r`` followed by a
Vickrey point-queue bottleneck of capacity ``s_r``) AND a departure time,
trading queue delay against schedule delay around ``t*``. Friesz et al. cast
the equilibrium as a variational inequality over path departure-rate profiles
``h`` on the volume-feasible set ``sum_r int h_r = N``: at a DUE every USED
(route, departure-time) pair carries the same minimal effective delay
``Psi_r(t) = travel + beta*[t* - arr]+ + gamma*[arr - t*]+``, and every unused
pair costs at least as much. The 1993 VI is loading-agnostic; this benchmark
instantiates it with the generalized Vickrey point-queue loading (FIFO by
construction, closed-form, and the loading under which the standing SRDC-DUE
existence theory holds — Han, Friesz & Yao 2013), which reduces EXACTLY to the
shipped ``vickrey`` model when there is one route with ``f = 0``.

Closed form (each used route runs its own Vickrey equilibrium at the common
cost level ``C``): with ``delta = beta*gamma/(beta+gamma)``, a used route's
queue cost is ``Cq_r = C - alpha*f_r`` and it serves ``N_r = s_r*Cq_r/delta``
travelers, so the used set ``U`` (routes with ``alpha*f_r < C``) satisfies

    C = (delta*N + alpha*sum_U s_r*f_r) / sum_U s_r

found by a greedy sweep over routes in increasing ``f_r`` order. Departure
windows per used route: ``t1 = t* - Cq/beta - f``, ``t2 = t* + Cq/gamma - f``,
peak ``t_n = t* - Cq/alpha - f``, with the Vickrey rates
``r_early = s*alpha/(alpha-beta)`` and ``r_late = s*alpha/(alpha+gamma)``.
Total cost is ``C*N``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["DUEScenario", "DUEProfile", "due_closed_form"]


def _as_f64(x) -> np.ndarray:
    return np.array(x, dtype=np.float64, order="C")


@dataclass(frozen=True)
class DUEScenario:
    """Frozen, content-hashed parallel-route SRDC-DUE instance (P2).

    ``route_free_flow[r]`` and ``route_capacity[r]`` describe route ``r``;
    the scalars mirror :class:`~tabench.bottleneck.scenario.BottleneckScenario`
    (which is the single-route ``f = 0`` special case). Documented domain
    bounds: ``alpha - beta >= 1e-9*alpha``, ``gamma <= 1e9*beta``, and the
    conditioning gate ``(beta+gamma)*eps_mach*(|t*|+window) <= 1e-9*C`` —
    beyond them the Vickrey departure rates (or the float64 resolution of the
    equilibrium cost itself) degenerate numerically. ``family`` is P7 lineage
    (provenance only, unhashed)."""

    name: str
    n_travelers: float
    alpha: float
    beta: float
    gamma: float
    t_star: float
    route_free_flow: np.ndarray  # (R,) float64 >= 0
    route_capacity: np.ndarray  # (R,) float64 > 0
    family: str = field(default="")

    def __post_init__(self) -> None:
        for name in ("n_travelers", "alpha", "beta", "gamma", "t_star"):
            object.__setattr__(self, name, float(getattr(self, name)))
        f = _as_f64(self.route_free_flow)
        s = _as_f64(self.route_capacity)
        object.__setattr__(self, "route_free_flow", f)
        object.__setattr__(self, "route_capacity", s)
        if not self.family:
            object.__setattr__(self, "family", self.name)
        name = self.name
        scalars = (self.n_travelers, self.alpha, self.beta, self.gamma, self.t_star)
        if any(not math.isfinite(v) for v in scalars):
            raise ValueError(f"DUEScenario '{name}': all parameters must be finite")
        if self.n_travelers <= 0:
            raise ValueError(f"DUEScenario '{name}': n_travelers must be > 0")
        if not (0.0 < self.beta < self.alpha):
            raise ValueError(
                f"DUEScenario '{name}': need 0 < beta < alpha "
                f"(got beta={self.beta}, alpha={self.alpha})"
            )
        if self.alpha - self.beta < 1e-9 * self.alpha:
            raise ValueError(
                f"DUEScenario '{name}': need alpha - beta >= 1e-9*alpha — the "
                f"early-departure rate s*alpha/(alpha-beta) is degenerate at "
                f"beta ~= alpha (got beta={self.beta}, alpha={self.alpha})"
            )
        if self.gamma <= 0.0:
            raise ValueError(f"DUEScenario '{name}': gamma must be > 0")
        if self.gamma > 1e9 * self.beta:
            raise ValueError(
                f"DUEScenario '{name}': need gamma <= 1e9*beta — the "
                f"schedule-delay ratio is degenerate beyond that (got "
                f"beta={self.beta}, gamma={self.gamma})"
            )
        if f.ndim != 1 or f.shape != s.shape or f.size == 0:
            raise ValueError(
                f"DUEScenario '{name}': route_free_flow/route_capacity must be "
                "equal-length 1-D with >= 1 route"
            )
        if not np.all(np.isfinite(f)) or np.any(f < 0.0):
            raise ValueError(f"DUEScenario '{name}': free-flow times must be finite, >= 0")
        if not np.all(np.isfinite(s)) or np.any(s <= 0.0):
            raise ValueError(f"DUEScenario '{name}': capacities must be finite, > 0")
        for arr in (f, s):
            arr.flags.writeable = False
        # conditioning gate (round-2 review MAJOR): when the schedule-delay
        # slopes amplify one ulp of the departure-time scale past a ~1e-9
        # share of the equilibrium cost, NO float64-representable profile can
        # score near 0 — the best possible solver would be certified as
        # maximally non-equilibrium, so the instance is rejected up front
        c_star = self.equilibrium_structure()[0]
        scale = abs(self.t_star) + c_star / self.beta + c_star / self.gamma + float(f.max()) + 1.0
        if (self.beta + self.gamma) * float(np.finfo(np.float64).eps) * scale > 1e-9 * c_star:
            raise ValueError(
                f"DUEScenario '{name}': ill-conditioned — float64 cannot resolve the "
                f"equilibrium cost C={c_star:.3e} against the schedule-delay slopes "
                f"at this departure-time scale (need "
                f"(beta+gamma)*eps_mach*(|t*|+window) <= 1e-9*C)"
            )

    @property
    def n_routes(self) -> int:
        return self.route_free_flow.size

    def equilibrium_cost(self) -> float:
        """The analytic common effective delay ``C`` (greedy used-set sweep)."""
        c, _, _ = self.equilibrium_structure()
        return c

    def equilibrium_structure(self) -> tuple[float, np.ndarray, np.ndarray]:
        """``(C, used_mask, N_r)``: sweep routes in increasing ``f_r`` order,
        adding the next route while its free-flow cost ``alpha*f`` is below the
        current cost level — the standard parallel-bottleneck greedy."""
        delta = self.beta * self.gamma / (self.beta + self.gamma)
        f, s = self.route_free_flow, self.route_capacity
        order = np.argsort(f, kind="stable")
        c_best = math.inf
        k_used = 1
        for k in range(1, self.n_routes + 1):
            sel = order[:k]
            c_k = (delta * self.n_travelers + self.alpha * float(s[sel] @ f[sel])) / float(
                s[sel].sum()
            )
            # k > 1: the cheapest route is ALWAYS used — analytically
            # c_1 = alpha*f_1 + delta*N/s_1 > alpha*f_1 strictly, but in
            # float64 the queue term can round away entirely (huge alpha*f_1),
            # and breaking at k = 1 would leave c_best = inf
            if k > 1 and c_k <= self.alpha * f[order[k - 1]]:  # route k unused
                break
            c_best, k_used = c_k, k
        used = np.zeros(self.n_routes, dtype=bool)
        sel = order[:k_used]
        used[sel] = True
        # Queue costs via the cancellation-free difference form
        #     Cq_r = (delta*N + alpha*sum_U s_i*(f_i - f_r)) / sum_U s_i
        # (subtracting alpha*f_r from C loses every digit when
        # alpha*f >> delta*N), then renormalize so the split sums to N exactly
        # — the cheapest route's Cq >= delta*N/sum_U s_i > 0, so the sum is
        # strictly positive even in float64.
        s_sel = float(s[sel].sum())
        f_diff = f[sel][None, :] - f[sel][:, None]  # f_diff[i, j] = f_j - f_i
        cq_sel = (delta * self.n_travelers + self.alpha * (f_diff @ s[sel])) / s_sel
        n_r = np.zeros(self.n_routes)
        n_r[sel] = s[sel] * np.maximum(0.0, cq_sel) / delta
        n_r *= self.n_travelers / float(n_r.sum())
        return float(c_best), used, n_r

    def content_hash(self) -> str:
        """SHA-256 over the scored data, domain-separated
        (``"tabench-due-scenario-v1;"`` prefix)."""
        h = hashlib.sha256()
        h.update(b"tabench-due-scenario-v1;")
        for name in ("n_travelers", "alpha", "beta", "gamma", "t_star"):
            h.update(f"{name}={float(getattr(self, name))!r};".encode())
        for label, arr in (("f", self.route_free_flow), ("s", self.route_capacity)):
            h.update(label.encode())
            h.update(_as_f64(arr).tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class DUEProfile:
    """Emitted departure plan: per-route cumulative origin departures
    ``cumulative[r, k]`` on the SHARED strictly-increasing grid ``times[k]``
    (each row nondecreasing from 0; rows sum to ``N`` at the horizon). The
    multi-route analogue of ``BottleneckSchedule``; ``provenance`` carries the
    solver's self-reported ``C``/split/windows and is never certified."""

    scenario_hash: str
    times: np.ndarray  # (K+1,) float64, strictly increasing
    cumulative: np.ndarray  # (R, K+1) float64
    provenance: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "times", np.ascontiguousarray(self.times, dtype=np.float64))
        object.__setattr__(
            self, "cumulative", np.ascontiguousarray(self.cumulative, dtype=np.float64)
        )
        if self.times.ndim != 1 or self.times.shape[0] < 2:
            raise ValueError("DUEProfile needs a 1-D grid with >= 2 edges")
        if self.cumulative.ndim != 2 or self.cumulative.shape[1] != self.times.shape[0]:
            raise ValueError("DUEProfile cumulative must be (n_routes, len(times))")


def due_closed_form(scenario: DUEScenario, n_steps: int = 2000) -> DUEProfile:
    """The closed-form SRDC-DUE: each used route runs its own Vickrey
    queue-build/dissipate schedule at the common cost level ``C``."""
    sc = scenario
    a, b, g = sc.alpha, sc.beta, sc.gamma
    c_star, used, n_r = sc.equilibrium_structure()
    # queue costs recovered from the exact split (Cq_r = delta*N_r/s_r) so the
    # windows are consistent with n_r to the last bit — never C - alpha*f,
    # which cancels catastrophically when alpha*f >> delta*N
    cq = (b * g / (b + g)) * n_r / sc.route_capacity
    t1 = sc.t_star - cq / b - sc.route_free_flow
    t2 = sc.t_star + cq / g - sc.route_free_flow
    t_n = sc.t_star - cq / a - sc.route_free_flow
    r_early = sc.route_capacity * a / (a - b)
    r_late = sc.route_capacity * a / (a + g)
    lo = float(t1[used].min())
    hi = float(t2[used].max())
    pad = 0.05 * (hi - lo) + 1e-9
    kinks = np.concatenate([t1[used], t_n[used], t2[used]])
    times = np.unique(
        np.concatenate([np.linspace(lo - pad, hi + pad, n_steps + 1), kinks])
    )
    cum = np.zeros((sc.n_routes, times.size))
    for r in range(sc.n_routes):
        if not used[r]:
            continue
        ramp = np.where(
            times <= t_n[r],
            r_early[r] * (times - t1[r]),
            r_early[r] * (t_n[r] - t1[r]) + r_late[r] * (times - t_n[r]),
        )
        cum[r] = np.clip(np.where(times <= t1[r], 0.0, ramp), 0.0, n_r[r])
    provenance = {"equilibrium_cost": c_star, "total_cost": c_star * sc.n_travelers}
    for r in range(sc.n_routes):
        provenance[f"n_route_{r}"] = float(n_r[r])
    return DUEProfile(
        scenario_hash=sc.content_hash(),
        times=times,
        cumulative=cum,
        provenance=provenance,
    )
