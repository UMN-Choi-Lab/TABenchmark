"""Bar-Gera's (2002) Origin-Based Assignment (OBA): bush-based user equilibrium.

Like Algorithm B (``algb``) OBA confines each origin's traffic to an acyclic
bush, but the two are genuinely different algorithms and share only the bush
machinery (``_bush._BushMachinery``: initial free-flow trees, the drop/add bush
improvement, and Kahn topological sort). The differences (Boyles, Lownes &
Unnikrishnan, *Transportation Network Analysis* ch. 6, "Bush-Based Algorithms"):

* **State.** Algorithm B stores per-bush *link flows* and shifts flow between the
  single longest- and shortest-used paths into each node. OBA works with
  **approach proportions** ``alpha_hi = x_hi / x_i`` (the fraction of node ``i``'s
  origin-flow arriving via bush link ``(h,i)``, summing to 1 over approaches) and
  rebalances **every** approach at a node toward its least-mean-cost ("basic")
  approach at once.
* **Labels.** Algorithm B uses min/max path-cost labels ``L``/``U``. OBA uses the
  flow-weighted **mean-cost** label ``M`` and an (approximate) **derivative**
  label ``D`` computed in one forward topological pass:

      M_i = sum_h alpha_hi (M_h + t_hi),
      D_i = ( sum_h alpha_hi * sqrt(D_h + t'_hi) )^2 .

  ``M_i`` is the average origin->i travel cost; ``D_i`` approximates its
  derivative w.r.t. origin flow (the closed form is the algebraically-exact
  collapse of Bar-Gera's double sum ``sum_h sum_g alpha_hi alpha_gi
  sqrt(D_hi D_gi)``, a perfect square).
* **Update.** Scanning nodes in reverse topological order, each nonbasic approach
  ``(h,i)`` shifts a Newton step of flow onto the basic approach ``(ĥ,i)``:

      Δx_hi = (M_hi − M_ĥi) / (D_hi + D_ĥi − 2 D_a),   capped at x_hi,

  with ``a`` the divergence node the two approaches share. We use the origin as a
  conservative ``a`` (``D_a = 0``); the proportions are updated locally and the
  link flows rebuilt from ``alpha`` in one reverse pass, avoiding fragile per-shift
  upstream flow propagation. The denominator is floored strictly positive (Nie
  2012, *Transportation Science* 46(1):27-38). **Because ``D`` is only an
  *approximate* second derivative, the raw Newton step overshoots on
  high-curvature (BPR power > 1) objectives and can limit-cycle rather than
  converge** — an adversarial fuzz found instances where the undamped solver
  stalls at a large gap. A ``step_scale`` damping factor (default 0.5) keeps the
  exact Newton direction with a globally stable magnitude; the direction is a
  descent direction (toward the least-mean-cost approach), so the damped iteration
  drives the certified gap to (near-)machine precision. ``step_scale = 1`` recovers
  the raw textbook step, exact only on the linear-cost / quadratic-Beckmann case.

Sourcing. The primary (Bar-Gera, *Transportation Science* 36(4):398-417, 2002) is
paywalled and attributed unread; every load-bearing equation is the Boyles TNA
ch. 6 rendition (read in full), with the derivative-guard cross-checked against
Nie (2012). With the ``step_scale`` damping (above) OBA converges to the unique UE
link flows, so it is validated the same way as the other exact solvers:
cross-family link-flow agreement with ``gp``/``algb``/``tapas`` and a
monotonically shrinking certified relative gap (Boyce, Ralevic-Dekic & Bar-Gera
2004: OBA was the first method to drive TAP to near-machine-precision gaps where
Frank-Wolfe stalls). The damping is load-bearing: without it the approximate
second-derivative label overshoots and can limit-cycle on high-curvature
instances (regression-tested).

Budget accounting (P6), matching ``algb``'s convention: one sp_call per all-origins
sweep. Each outer iteration charges one bush-update scan (when it runs),
``inner_rounds`` M/D shift rounds, and one all-or-nothing for the honest
self-reported gap. As for ``algb`` the sp_calls axis is not comparable across
model families; OBA additionally refreshes the shared link costs after every
origin within a round (Gauss-Seidel), work that shows up in wall-clock but not in
sp_calls -- so read wall-clock, never sp_calls, when comparing OBA's cost to a
link-based solver's. Emitted link flows are rebuilt exactly from the sum of bush
flows before every checkpoint.
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
from ._bush import _BushMachinery
from .base import TrafficAssignmentModel, register_model

__all__ = ["OriginBasedModel"]


@register_model
class OriginBasedModel(_BushMachinery, TrafficAssignmentModel):
    """Bar-Gera (2002) Origin-Based Assignment over per-origin bushes."""

    name = "oba"
    capabilities = Capabilities(
        paradigm="static_ue",
        deterministic=True,
        provides_gap=True,
        seedable=True,
    )
    factors = {
        "inner_rounds": FactorSpec(
            default=8,
            kind="int",
            bounds=(1, 64),
            doc="Gauss-Seidel proportion-shift rounds per outer iteration (one M/D "
            "pass per origin per round; costs refreshed from the shared flow between "
            "origins).",
        ),
        "bush_update_every": FactorSpec(
            default=1,
            kind="int",
            bounds=(1, 8),
            doc="Bush improvement (drop unused / add U-rule shortcut links) every "
            "k-th outer iteration.",
        ),
        "drop_tol": FactorSpec(
            default=1e-13,
            kind="float",
            bounds=(1e-16, 1e-6),
            doc="Bush-flow threshold for dropping links and for donor eligibility of "
            "a nonbasic approach.",
        ),
        "denom_floor": FactorSpec(
            default=1e-10,
            kind="float",
            bounds=(1e-16, 1e-2),
            doc="Strict positive floor on the Newton denominator D_hi + D_ĥi (Nie "
            "2012's guard against the negative/degenerate second-derivative).",
        ),
        "step_scale": FactorSpec(
            default=0.5,
            kind="float",
            bounds=(1e-3, 1.0),
            doc="Damping on the Newton proportion step. The D label is only an "
            "approximate second derivative, so on high-curvature (BPR power > 1) "
            "objectives the undamped step can overshoot and limit-cycle instead of "
            "converging (the pathology Nie 2012 formalizes). Halving the step (the "
            "default) keeps the exact Newton direction but a globally stable "
            "magnitude; 1.0 recovers the raw textbook step (exact and fastest only "
            "on the linear-cost / quadratic-Beckmann case).",
        ),
    }

    # --------------------------------------------------------------- setup
    def _setup(self, scenario: Scenario) -> None:
        self._setup_bush_graph(scenario)  # shared expanded-graph state (_bush)
        self._drop_tol = self.factor_values["drop_tol"]
        self._denom_floor = self.factor_values["denom_floor"]
        self._step_scale = self.factor_values["step_scale"]
        # Out-links per expanded node (mirror of _in_links) for the alpha->flow
        # reconstruction pass.
        out: list[list[int]] = [[] for _ in range(self._n_exp)]
        for k, tail in enumerate(self._tails):
            out[int(tail)].append(k)
        self._out_links = [np.asarray(lst, dtype=np.int64) for lst in out]
        # Per-origin total demand routed (intrazonal excluded), for origin node
        # throughput in the reconstruction.
        self._origin_out = {
            int(o): float(self._od[o].sum() - self._od[o, o]) for o in self._origins
        }

    # --------------------------------------------------------------- shift
    def _shift_pass(
        self, bush, origin_idx: int, t: np.ndarray, dt: np.ndarray
    ) -> bool:
        """One OBA sweep on a fixed bush: forward M/D labels, reverse-topo
        proportion shift toward the basic approach, then rebuild link flows from
        the updated proportions. Reads only ``bush.x`` and the current costs
        ``t``/``dt``; updates ``bush.x`` in place, and the caller resyncs the
        global flow. Returns True if any proportion moved."""
        in_links = self._in_links
        out_links = self._out_links
        tails = self._tails
        heads = self._heads
        topo = bush.topo
        xb = bush.x
        n_exp = self._n_exp

        # Bush-restricted approaches per node (indices into the link array).
        def approaches(node: int) -> np.ndarray:
            ins = in_links[node]
            return ins[bush.in_bush[ins]]

        # Node throughput x_i (frozen), from current inflows; origin = demand out.
        x_node = np.zeros(n_exp)
        for node in topo:
            if int(node) == origin_idx:
                continue
            ins = approaches(int(node))
            x_node[node] = float(xb[ins].sum())
        x_node[origin_idx] = self._origin_out[origin_idx]

        # Frozen approach proportions alpha_hi = x_hi / x_i. A *dead* node
        # (x_i = 0) gets uniform proportions rather than zeros: a shift can route
        # flow onto its out-links this sweep, and with zero proportions the
        # rebuild would emit that flow with no matching inflow (a phantom source).
        # Uniform proportions pull the activated flow in and conserve; the next
        # sweep re-equilibrates the split (Bar-Gera's x_i = 0 free-choice rule).
        alpha = np.zeros(self._n_links)
        for node in topo:
            if int(node) == origin_idx:
                continue
            ins = approaches(int(node))
            if ins.size == 0:
                continue
            xn = x_node[node]
            if xn > 0.0:
                alpha[ins] = xb[ins] / xn
            else:
                alpha[ins] = 1.0 / ins.size

        # Forward pass: mean-cost M and derivative D labels (topo order).
        M = np.zeros(n_exp)
        D = np.zeros(n_exp)
        for node in topo:
            ni = int(node)
            if ni == origin_idx:
                continue
            ins = approaches(ni)
            if ins.size == 0:
                continue
            a = alpha[ins]
            m_app = M[tails[ins]] + t[ins]
            d_app = D[tails[ins]] + dt[ins]
            M[ni] = float((a * m_app).sum())
            root = float((a * np.sqrt(np.maximum(d_app, 0.0))).sum())
            D[ni] = root * root

        # Reverse pass: shift each nonbasic approach's proportion toward the
        # basic (least-mean-cost) approach. Work in alpha; rebuild flows after.
        floor = self._denom_floor
        eps = self._drop_tol
        moved = False
        for node in topo[::-1]:
            ni = int(node)
            if ni == origin_idx:
                continue
            xn = x_node[ni]
            if xn <= 0.0:
                continue
            ins = approaches(ni)
            if ins.size < 2:
                continue
            m_app = M[tails[ins]] + t[ins]
            d_app = D[tails[ins]] + dt[ins]
            bi = int(np.argmin(m_app))
            k_basic = int(ins[bi])
            m_basic = float(m_app[bi])
            d_basic = float(d_app[bi])
            for idx in range(ins.size):
                if idx == bi:
                    continue
                k = int(ins[idx])
                if alpha[k] <= 0.0 or xb[k] <= eps:
                    continue
                excess = float(m_app[idx]) - m_basic
                if excess <= 0.0:
                    continue
                denom = max(float(d_app[idx]) + d_basic, floor)
                d_alpha = min(alpha[k], self._step_scale * excess / (denom * xn))
                if d_alpha <= 0.0:
                    continue
                alpha[k] -= d_alpha
                alpha[k_basic] += d_alpha
                moved = True
            # Renormalize this node's approaches to kill float drift.
            s = float(alpha[ins].sum())
            if s > 0.0:
                alpha[ins] /= s

        if not moved:
            return False

        # Rebuild link flows from the updated proportions. Node throughput in one
        # reverse-topo pass: x_i = term_i + sum_{(i,j) in bush} alpha_ij x_j, where
        # term_i is the origin demand terminating at i (which never leaves i). Then
        # x_hi = alpha_hi x_i. Non-negative and conserved by construction.
        term = np.zeros(n_exp)
        for d in np.nonzero(self._od[origin_idx] > 0)[0]:
            if int(d) == origin_idx:
                continue
            term[self._dest_index(int(d) + 1)] += float(self._od[origin_idx, d])
        xnode = np.zeros(n_exp)
        for node in topo[::-1]:
            ni = int(node)
            passing = 0.0
            for k in out_links[ni]:
                if bush.in_bush[k]:
                    passing += alpha[k] * xnode[heads[k]]
            xnode[ni] = term[ni] + passing
        new_x = np.zeros(self._n_links)
        for node in topo:
            ni = int(node)
            ins = approaches(ni)
            if ins.size:
                new_x[ins] = alpha[ins] * xnode[ni]
        bush.x = new_x
        np.maximum(bush.x, 0.0, out=bush.x)
        return True

    # ---------------------------------------------------------------- solve
    def solve(
        self, scenario: Scenario, budget: Budget, rng: RngBundle, trace: Trace
    ) -> ResultBundle:
        start = time.perf_counter()
        self._setup(scenario)
        network = self._network
        engine = self._engine
        inner_rounds = self.factor_values["inner_rounds"]
        bush_update_every = self.factor_values["bush_update_every"]

        v = np.zeros(self._n_links)
        bushes, v, sp_calls = self._initial_bushes(v)
        t = network.link_cost(v)
        dt = network.link_cost_derivative(v)

        k = 0
        while True:
            k += 1
            rounds = 0
            if (k - 1) % bush_update_every == 0:
                for bush, o in zip(bushes, self._origins, strict=True):
                    self._update_bush(bush, int(o), t)
                rounds += 1
            for _ in range(inner_rounds):
                moved_any = False
                for bush, o in zip(bushes, self._origins, strict=True):
                    if self._shift_pass(bush, int(o), t, dt):
                        moved_any = True
                    # Gauss-Seidel: refresh shared costs after each origin.
                    v = np.zeros(self._n_links)
                    for b in bushes:
                        v += b.x
                    t = network.link_cost(v)
                    dt = network.link_cost_derivative(v)
                rounds += 1
                if not moved_any:
                    break

            sp_calls += rounds
            _, sptt = engine.all_or_nothing(t, scenario.demand)
            sp_calls += 1
            tstt = float(v @ t)
            gap = (tstt - sptt) / tstt if tstt > 0 else 0.0

            coords = BudgetCoords(
                iterations=k,
                sp_calls=sp_calls,
                wall_ms=1000.0 * (time.perf_counter() - start),
            )
            trace.record(
                v,
                coords,
                relative_gap=gap,
                tstt=tstt,
                sptt=sptt,
                beckmann=float(network.link_cost_integral(v).sum()),
                bush_scan_rounds=float(rounds),
            )
            if budget.exhausted(coords) or budget.target_met(gap):
                break

        return ResultBundle(
            model_name=self.name,
            final=trace.final,
            trace=trace,
            factors=dict(self.factor_values),
            seed_info=rng.describe(),
        )
