"""Scenario objects: frozen, content-hashed, declarative problem instances.

Design principle P2 (docs/ARCHITECTURE.md): a scenario is data, never code.
Node ids follow the TNTP convention: 1-based, zones are nodes ``1..n_zones``,
and nodes with id below ``first_thru_node`` are zone centroids that cannot
carry through traffic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Network", "Demand", "ElasticDemand", "ReferenceSolution", "Scenario"]


def _as_f64(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float64)


@dataclass(frozen=True)
class Network:
    """Directed road network with BPR-family link performance functions.

    The generalized link cost (in time units) is

        t_a(v) = fft_a * (1 + b_a * (v / cap_a)^power_a)
                 + toll_weight * toll_a + distance_weight * length_a

    where the toll and distance weights are per-network metadata (they are
    NOT stored in ``.tntp`` files; see docs/ARCHITECTURE.md P9).

    ``units`` records per-network unit conventions (e.g. Sioux Falls free-flow
    times are 0.01 hours) as documentation; costs are always computed in the
    network's native units.
    """

    name: str
    n_nodes: int
    n_zones: int
    first_thru_node: int
    init_node: np.ndarray  # (n_links,) int64, 1-based tail node ids
    term_node: np.ndarray  # (n_links,) int64, 1-based head node ids
    capacity: np.ndarray
    length: np.ndarray
    free_flow_time: np.ndarray
    b: np.ndarray  # BPR "alpha"
    power: np.ndarray  # BPR "beta"
    toll: np.ndarray
    link_type: np.ndarray
    toll_weight: float = 0.0
    distance_weight: float = 0.0
    units: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        n = self.n_links
        for name in ("term_node", "capacity", "length", "free_flow_time", "b", "power", "toll"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"Network '{self.name}': column {name!r} has wrong length")
        pairs = set(zip(self.init_node.tolist(), self.term_node.tolist(), strict=True))
        if len(pairs) != n:
            raise ValueError(
                f"Network '{self.name}' contains parallel (duplicate) links; "
                "TABenchmark v0 requires unique (init_node, term_node) pairs "
                "(see the known-defects registry for affected TNTP networks)."
            )
        if np.any(self.free_flow_time <= 0):
            raise ValueError(
                f"Network '{self.name}': free_flow_time must be strictly positive "
                "(zero-cost links break shortest-path sparsity)."
            )
        if np.any(self.capacity <= 0):
            raise ValueError(f"Network '{self.name}': capacity must be strictly positive")
        if np.any(self.b < 0):
            raise ValueError(
                f"Network '{self.name}': BPR coefficient b (alpha) must be nonnegative "
                "(negative b makes link costs decrease in flow, voiding Beckmann convexity)"
            )
        if np.any(self.power < 0):
            raise ValueError(
                f"Network '{self.name}': BPR exponent power (beta) must be nonnegative"
            )
        if np.any((self.init_node < 1) | (self.init_node > self.n_nodes)) or np.any(
            (self.term_node < 1) | (self.term_node > self.n_nodes)
        ):
            raise ValueError(f"Network '{self.name}': node ids out of range 1..n_nodes")

    @property
    def n_links(self) -> int:
        return len(self.init_node)

    @property
    def fixed_cost(self) -> np.ndarray:
        """Flow-independent generalized-cost component per link."""
        return self.toll_weight * self.toll + self.distance_weight * self.length

    def link_cost(self, link_flows: np.ndarray) -> np.ndarray:
        """Generalized link travel cost t_a(v_a) at the given flows."""
        v = np.maximum(np.asarray(link_flows, dtype=np.float64), 0.0)
        ratio = v / self.capacity
        return self.free_flow_time * (1.0 + self.b * ratio**self.power) + self.fixed_cost

    def link_cost_integral(self, link_flows: np.ndarray) -> np.ndarray:
        """Per-link Beckmann integral: integral of t_a(s) ds from 0 to v_a."""
        v = np.maximum(np.asarray(link_flows, dtype=np.float64), 0.0)
        ratio = v / self.capacity
        variable = self.free_flow_time * (v + self.b * v * ratio**self.power / (self.power + 1.0))
        return variable + self.fixed_cost * v

    def link_cost_derivative(self, link_flows: np.ndarray) -> np.ndarray:
        """Per-link derivative dt_a/dv_a — the diagonal Hessian of the Beckmann
        objective, used by conjugate-direction Frank-Wolfe variants.

        For BPR, t'(v) = fft * b * p * (v/cap)^(p-1) / cap. Edge cases: p = 0
        or b = 0 gives exactly 0 (constant cost); p = 1 gives the constant
        fft*b/cap (numpy 0**0 = 1); 0 < p < 1 has an unbounded derivative at
        v = 0, where H is defined as 0 (it is only a conjugacy scale) and
        0**(negative) is never evaluated.
        """
        v = np.maximum(np.asarray(link_flows, dtype=np.float64), 0.0)
        ratio = v / self.capacity
        exponent = self.power - 1.0
        coeff = self.free_flow_time * self.b * self.power / self.capacity
        # Zero wherever the analytic value is 0 (p or b zero) or defined as 0
        # (v = 0 with p < 1) — without ever evaluating 0**(negative). For
        # 0 < p < 1 at subnormal positive flows the analytic value exceeds
        # float64 range: clamp to the largest finite float (H is only a
        # conjugacy scale) so no inf/RuntimeWarning ever escapes.
        zero = (coeff == 0.0) | ((exponent < 0) & (ratio <= 0.0))
        safe = np.where(zero, 1.0, ratio)
        with np.errstate(over="ignore"):
            h = np.minimum(coeff * safe**exponent, np.finfo(np.float64).max)
        return np.where(zero, 0.0, h)


@dataclass(frozen=True)
class Demand:
    """Origin-destination demand. ``matrix[i, j]`` is flow from zone i+1 to j+1."""

    matrix: np.ndarray  # (n_zones, n_zones) float64

    def __post_init__(self) -> None:
        m = self.matrix
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("Demand matrix must be square (n_zones, n_zones)")
        if np.any(m < 0):
            raise ValueError("Demand must be nonnegative")

    @property
    def n_zones(self) -> int:
        return self.matrix.shape[0]

    @property
    def total(self) -> float:
        return float(self.matrix.sum())


@dataclass(frozen=True)
class ElasticDemand:
    """Elastic (variable) demand law: the demand between an OD pair is a
    strictly decreasing function of that pair's equilibrium travel cost,
    ``d_rs = D_rs(u_rs)``.

    The *reference* demand ``d0_rs = D_rs(0)`` (demand at zero cost — an upper
    bound) is carried by the scenario's ordinary :class:`Demand` matrix; this
    object adds only the decay law and its single shared parameter. Two forms:

    * ``"linear"``      ``D_rs(u) = d0_rs * max(0, 1 - u/param)``   (``param`` = u0)
      inverse on ``[0, d0]``: ``D_rs^{-1}(d) = param * (1 - d/d0)`` — bounded,
      singularity-free (the excess-demand arc cost ``param * e/d0`` is linear).
    * ``"exponential"`` ``D_rs(u) = d0_rs * exp(-param * u)``       (``param`` = beta)
      inverse on ``(0, d0]``: ``D_rs^{-1}(d) = ln(d0/d) / param`` — the
      cited/logit-consistent form, unbounded as ``d -> 0`` (guarded below).

    Objective ``min Sum_a integral t_a - Sum_rs integral D_rs^{-1}`` and the
    excess-demand ("dummy arc") transformation follow Sheffi (1985) ch. 6 and
    Boyles, Lownes & Unnikrishnan, *Transportation Network Analysis* sec. 9.1;
    the transformation itself is due to Gartner (1980), and the seminal
    computational treatment is Florian & Nguyen (1974). See docs/design/adr-005.

    ``param`` is a scalar shared across OD pairs (per-OD curves are a future
    extension). It is task data, content-hashed with the scenario when set.
    """

    form: str
    param: float

    _FORMS = ("linear", "exponential")

    def __post_init__(self) -> None:
        if self.form not in self._FORMS:
            raise ValueError(
                f"ElasticDemand form must be one of {self._FORMS}, got {self.form!r}"
            )
        if not (np.isfinite(self.param) and self.param > 0):
            raise ValueError(
                f"ElasticDemand param must be finite and > 0, got {self.param!r}"
            )

    def realized_demand(self, d0: np.ndarray, cost: np.ndarray) -> np.ndarray:
        """``D_rs(u)`` elementwise: realized demand at OD cost ``cost``, clamped
        to ``[0, d0]``. Both arguments broadcast (typically ``(n_zones,
        n_zones)`` reference-demand and OD shortest-path-cost matrices)."""
        d0 = np.asarray(d0, dtype=np.float64)
        u = np.asarray(cost, dtype=np.float64)
        if self.form == "linear":
            d = d0 * (1.0 - u / self.param)
        else:  # exponential
            d = d0 * np.exp(-self.param * u)
        return np.clip(d, 0.0, d0)

    def inverse_demand(self, d0: np.ndarray, d: np.ndarray) -> np.ndarray:
        """``D_rs^{-1}(d)``: the OD cost at which realized demand equals ``d``
        (``d`` in ``[0, d0]``). Prices the excess-demand arc. For the
        exponential form the ``d -> 0`` log-singularity is floored so the
        transformed solver never evaluates ``ln(inf)``; ``d0 == 0`` pairs
        (no demand) return 0."""
        d0 = np.asarray(d0, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        positive = d0 > 0
        if self.form == "linear":
            return self.param * np.where(positive, 1.0 - d / np.where(positive, d0, 1.0), 0.0)
        floor = np.maximum(d0, 1.0) * 1e-12
        d = np.maximum(d, floor)
        return np.where(positive, np.log(np.where(positive, d0, 1.0) / d) / self.param, 0.0)

    def excess_arc_cost(self, d0: np.ndarray, excess: np.ndarray) -> np.ndarray:
        """Cost of the excess-demand (Gartner dummy) arc carrying unmet flow
        ``excess``: ``W(e) = D_rs^{-1}(d0 - e)`` — increasing in ``e``."""
        d0 = np.asarray(d0, dtype=np.float64)
        return self.inverse_demand(d0, d0 - np.asarray(excess, dtype=np.float64))


@dataclass(frozen=True)
class ReferenceSolution:
    """Best-known solution used as a regression oracle (with provenance)."""

    link_flows: np.ndarray
    source: str
    note: str = ""


@dataclass(frozen=True)
class Scenario:
    """A frozen benchmark instance: network + demand (+ optional oracle).

    ``family`` names the scenario lineage used by the fairness gate (P7):
    a learned model whose ``trained_on`` includes this family is refused
    evaluation on this scenario.

    ``sue_theta`` (optional) makes this an SUE task: the dispersion dial of the
    pinned loading map, in 1/(native cost unit) — so its meaning is
    network-specific (P9; scenario cards state the unit). It is task data,
    never a model factor, and it is content-hashed when set (two scenarios
    differing only in theta are different benchmark instances).

    ``sue_family`` selects the choice-model family of that SUE task —
    ``"logit"`` (Dial-STOCH closed-form loading, docs/design/adr-001) or
    ``"probit"`` (Monte Carlo loading + pinned certificate, adr-003). For
    probit, ``sue_theta`` carries ``beta``, the perception variance per unit
    free-flow time. This is unrelated to ``family``, which is *data lineage*
    for the ``trained_on`` fairness gate (P7); ``sue_family`` names the task's
    equilibrium definition. It is hashed only when non-default so every logit
    scenario keeps the byte-identical hash it had before this field existed.

    ``elastic_demand`` (optional) makes this a variable-demand UE task: the
    :class:`Demand` matrix becomes the *reference* demand ``d0 = D(0)`` and this
    :class:`ElasticDemand` supplies the decay law (docs/design/adr-005). Like
    ``sue_theta`` it is task data content-hashed only when set, so every
    fixed-demand scenario keeps its prior hash. An elastic task is a
    deterministic UE with endogenous demand and so is mutually exclusive with
    the SUE fields.
    """

    name: str
    network: Network
    demand: Demand
    reference: ReferenceSolution | None = None
    family: str = field(default="")
    sue_theta: float | None = None
    sue_family: str = "logit"
    elastic_demand: ElasticDemand | None = None

    def __post_init__(self) -> None:
        if self.demand.n_zones != self.network.n_zones:
            raise ValueError(
                f"Scenario '{self.name}': demand has {self.demand.n_zones} zones, "
                f"network declares {self.network.n_zones}"
            )
        if not self.family:
            object.__setattr__(self, "family", self.name)
        if self.sue_theta is not None and not (
            np.isfinite(self.sue_theta) and self.sue_theta > 0
        ):
            raise ValueError(
                f"Scenario '{self.name}': sue_theta must be finite and > 0, "
                f"got {self.sue_theta!r}"
            )
        if self.sue_family not in ("logit", "probit"):
            raise ValueError(
                f"Scenario '{self.name}': sue_family must be 'logit' or 'probit', "
                f"got {self.sue_family!r}"
            )
        if self.sue_family == "probit" and self.sue_theta is None:
            raise ValueError(
                f"Scenario '{self.name}': sue_family='probit' requires sue_theta "
                "(beta, the perception variance per unit free-flow time)"
            )
        if self.elastic_demand is not None and self.sue_theta is not None:
            raise ValueError(
                f"Scenario '{self.name}': elastic_demand and sue_theta are "
                "mutually exclusive (elastic UE is deterministic with endogenous "
                "demand; SUE is stochastic with fixed demand)"
            )

    def content_hash(self) -> str:
        """SHA-256 over the canonical serialization of all scored content (P2)."""
        h = hashlib.sha256()
        net = self.network
        h.update(
            f"nodes={net.n_nodes};zones={net.n_zones};ftn={net.first_thru_node};"
            f"tw={net.toll_weight!r};dw={net.distance_weight!r};".encode()
        )
        for label, arr in (
            ("init", net.init_node),
            ("term", net.term_node),
            ("cap", net.capacity),
            ("len", net.length),
            ("fft", net.free_flow_time),
            ("b", net.b),
            ("pow", net.power),
            ("toll", net.toll),
            ("od", self.demand.matrix),
        ):
            h.update(label.encode())
            h.update(_as_f64(arr).tobytes())
        # Conditional, appended last: scenarios without a theta hash exactly
        # as before this field existed (pinned by a golden-hash test).
        if self.sue_theta is not None:
            h.update(f"sue_theta={float(self.sue_theta)!r};".encode())
        # Appended after theta and only when non-default, so every logit
        # scenario hashes exactly as before this field existed (golden test):
        # a probit task can never collide with the logit task at the same theta.
        if self.sue_family != "logit":
            h.update(f"sue_family={self.sue_family};".encode())
        # Appended last and only when set: every fixed-demand scenario hashes
        # exactly as before this field existed (golden-hash test). The
        # reference demand d0 is already covered by the ("od", matrix) bytes
        # above; only the decay law and its parameter are new content.
        if self.elastic_demand is not None:
            ed = self.elastic_demand
            h.update(f"elastic_form={ed.form};elastic_param={float(ed.param)!r};".encode())
        return h.hexdigest()
