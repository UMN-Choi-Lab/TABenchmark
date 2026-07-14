"""ThreeDetectorScenario: frozen, content-hashed Newell three-detector instances.

Design: docs/design/adr-024-newell-3det.md. This is the repo's first
traffic-STATE-estimation task. One homogeneous triangular-FD link ``[0, L]`` with
detectors at ``x = 0`` (upstream) and ``x = L`` (downstream); given the OBSERVED
(noisy / partial) boundary cumulative curves, a submission reconstructs the
interior cumulative field ``N(x_i, t_k)`` at a FIXED hashed query grid.

The GROUND-TRUTH boundary curves are never stored — they are regenerated
deterministically from the hashed truth recipe (a piecewise-constant corridor
inflow plus a downstream metering capacity) by running the repo's own
:class:`~tabench.dnl.ltm.LTMLink` on a two-link corridor, exactly as the
adr-023 certifier regenerates its lag tensor. LTM emits the study link's
cumulative inflow (``n_in`` at ``x = 0``) and outflow (``n_out`` at ``x = L``),
which are the two detector curves under a shared vehicle numbering with an
empty-start record (``N0 = 0``). Truth fidelity scope (adr-024 review): the LTM
curves equal the continuous-time LWR solution EXACTLY when the recipe's arrival
and metering events are grid-aligned (all shipped anchors are, verified to
~4e-15); off that subspace the metered onset carries an ``O(meter_cap*dt)``
plateau artifact — the scoring reference is ``M`` (the min principle on THESE
curves), which is identical for every submission, not the continuous-time field.

Newell's loading content (link-end sending/receiving) already shipped as ``ltm``
(adr-016); this unit ships the INTERIOR minimum principle
``N(x,t) = min( N_up(t - x/vf), N_dn(t - (L-x)/w) + kappa*(L-x) )`` (``w > 0``
backward-wave magnitude, so the downstream shift is MINUS ``(L-x)/w``) as an
estimation task, so there is no third ``LinkModel``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from ..core.scenario import Network
from ..dnl import (
    DynamicDemand,
    DynamicScenario,
    LinkDynamics,
    LTMLink,
    NetworkLoader,
    TimeGrid,
)

__all__ = ["ThreeDetectorScenario", "reconstruct_field"]

_EPS_MACH = float(np.finfo(np.float64).eps)


def _as_f64(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))


def _interp_curve(curve: np.ndarray, ts: np.ndarray, dt: float) -> np.ndarray:
    """Vectorized twin of :func:`tabench.dnl.link.interp_curve`: piecewise-linear
    read of a grid-edge cumulative ``curve`` at arbitrary times ``ts``. ``t < 0``
    reads 0 (nothing precedes the empty start); ``t`` beyond the last edge reads
    ``curve[-1]`` (constant continuation). Byte-matches the scalar helper so the
    reference field and every estimator share one audited interpolation."""
    ts = np.asarray(ts, dtype=np.float64)
    out = np.empty(ts.shape, dtype=np.float64)
    x = np.divide(ts, float(dt))
    neg = ts < 0.0
    out[neg] = 0.0
    j = np.floor(np.where(neg, 0.0, x)).astype(np.int64)
    beyond = ~neg & (j >= curve.shape[0] - 1)
    out[beyond] = curve[-1]
    mid = ~neg & ~beyond
    jm = j[mid]
    frac = x[mid] - jm
    out[mid] = curve[jm] + frac * (curve[jm + 1] - curve[jm])
    return out


def reconstruct_field(
    vf: float,
    w: float,
    kappa: float,
    length: float,
    times: np.ndarray,
    n_up: np.ndarray,
    n_dn: np.ndarray,
    x_query: np.ndarray,
    dt: float,
    up_windows: tuple[tuple[float, float], ...] = (),
    dn_windows: tuple[tuple[float, float], ...] = (),
) -> np.ndarray:
    """Newell's interior minimum principle at the fixed query grid ``(x_i, t_k)``.

    ``N(x_i, t_k) = min( N_up(t_k - x_i/vf), N_dn(t_k - (L-x_i)/w) + kappa*(L-x_i) )``
    with the ``w > 0`` MINUS sign on the downstream shift (adr-024). ``up_windows``
    / ``dn_windows`` are unobserved time windows on each detector: when a branch's
    shifted source time lands inside a masked window that branch is dropped, so the
    OTHER branch alone bounds the interior (the A5 observability content). Because
    the true cumulative count is nondecreasing, each position's curve is then
    repaired to the TIGHTEST NONDECREASING upper bound consistent with the retained
    branches — the suffix-minimum (a bound at ``t'`` also bounds every ``t <= t'``),
    which (i) pulls the loose single-branch values inside a window down to the
    tighter bound that reappears after it (the raw min would DIP there — the
    adr-024 review MAJOR: the dip false-censored honest reconstructions on the C3
    retraction gate), and (ii) bridges doubly-masked cells from the next observed
    bound (the raw min was ``+inf`` there — the C0 false-censor). Cells with no
    later bound carry the last observed value forward. On fully observed data the
    repair is the identity (the min of monotone curves is monotone). Returns an
    ``(m, K+1)`` cumulative field. This is a pure function of its curve arguments —
    no truth recipe, no scenario — so an estimator built on it cannot leak the
    ground truth."""
    times = np.asarray(times, dtype=np.float64)
    x_query = np.asarray(x_query, dtype=np.float64)
    m = x_query.shape[0]
    field = np.empty((m, times.shape[0]), dtype=np.float64)
    for i in range(m):
        x = float(x_query[i])
        t_up = times - x / vf
        t_dn = times - (length - x) / w
        free = _interp_curve(n_up, t_up, dt)
        cong = _interp_curve(n_dn, t_dn, dt) + kappa * (length - x)
        if up_windows:
            free = np.where(_in_windows(t_up, up_windows), np.inf, free)
        if dn_windows:
            cong = np.where(_in_windows(t_dn, dn_windows), np.inf, cong)
        row = np.minimum(free, cong)
        if up_windows or dn_windows:
            # tightest nondecreasing upper bound (suffix-min), then carry the
            # last observed bound through any trailing unobserved block
            row = np.minimum.accumulate(row[::-1])[::-1]
            if not np.isfinite(row[-1]):
                finite = np.isfinite(row)
                fill = row[finite][-1] if finite.any() else 0.0
                row = np.where(finite, row, fill)
        field[i] = row
    return field


def _in_windows(ts: np.ndarray, windows: tuple[tuple[float, float], ...]) -> np.ndarray:
    """Boolean mask: which of ``ts`` fall inside any ``[t0, t1]`` masked window."""
    hit = np.zeros(ts.shape, dtype=bool)
    for t0, t1 in windows:
        hit |= (ts >= t0) & (ts <= t1)
    return hit


@dataclass(frozen=True)
class ThreeDetectorScenario:
    """Frozen, content-hashed Newell three-detector instance (P2).

    Study link ``[0, L]``: triangular FD ``(vf, w, kappa)`` with optional capacity
    cap ``capacity`` (``None`` = geometric apex). Truth recipe: a piecewise-constant
    corridor inflow (``inflow_breakpoints`` / ``inflow_rates``, a single OD pair)
    and a downstream metering capacity ``meter_cap`` that shapes the congested
    downstream detector curve. Observation dials degrade the two clean detector
    curves into the model-visible dataset (``observe.py``). ``x_query`` are the
    FIXED query positions in ``[0, L]`` (boundary queries reduce to the detector
    curves); scoring times are the grid edges, so the emission is ``(m, K+1)``.
    ``family`` is P7 lineage (unhashed provenance).
    """

    name: str
    vf: float
    w: float
    kappa: float
    length: float
    meter_cap: float
    inflow_breakpoints: np.ndarray
    inflow_rates: np.ndarray
    grid: TimeGrid
    x_query: np.ndarray
    capacity: float | None = None
    noise: str = "none"
    read_sigma: float = 0.0
    drift: float = 0.0
    up_windows: tuple[tuple[float, float], ...] = ()
    dn_windows: tuple[tuple[float, float], ...] = ()
    n_days: int = 1
    seed: int = 0
    family: str = ""

    def __post_init__(self) -> None:
        for field in ("vf", "w", "kappa", "length", "meter_cap", "read_sigma", "drift"):
            object.__setattr__(self, field, float(getattr(self, field)))
        object.__setattr__(self, "inflow_breakpoints", _as_f64(self.inflow_breakpoints))
        object.__setattr__(self, "inflow_rates", _as_f64(self.inflow_rates))
        object.__setattr__(self, "x_query", _as_f64(self.x_query))
        self.inflow_breakpoints.setflags(write=False)
        self.inflow_rates.setflags(write=False)
        self.x_query.setflags(write=False)
        object.__setattr__(
            self, "up_windows", tuple((float(a), float(b)) for a, b in self.up_windows)
        )
        object.__setattr__(
            self, "dn_windows", tuple((float(a), float(b)) for a, b in self.dn_windows)
        )
        object.__setattr__(self, "n_days", int(self.n_days))
        object.__setattr__(self, "seed", int(self.seed))
        if not self.family:
            object.__setattr__(self, "family", self.name)

        if not (math.isfinite(self.vf) and self.vf > 0):
            raise ValueError(f"ThreeDetectorScenario '{self.name}': vf must be finite and > 0")
        if not (math.isfinite(self.w) and self.w > 0):
            raise ValueError(f"ThreeDetectorScenario '{self.name}': w must be finite and > 0")
        if not (math.isfinite(self.kappa) and self.kappa > 0):
            raise ValueError(f"ThreeDetectorScenario '{self.name}': kappa must be finite and > 0")
        if not (math.isfinite(self.length) and self.length > 0):
            raise ValueError(f"ThreeDetectorScenario '{self.name}': length must be finite and > 0")
        if not (math.isfinite(self.meter_cap) and self.meter_cap > 0):
            raise ValueError(f"ThreeDetectorScenario '{self.name}': meter_cap must be > 0 finite")
        if self.capacity is not None:
            object.__setattr__(self, "capacity", float(self.capacity))
            if not (math.isfinite(self.capacity) and self.capacity > 0):
                raise ValueError(f"ThreeDetectorScenario '{self.name}': capacity must be > 0")
        if self.noise not in ("none", "poisson", "gaussian"):
            raise ValueError(f"ThreeDetectorScenario '{self.name}': unknown noise {self.noise!r}")
        if self.read_sigma < 0 or not math.isfinite(self.read_sigma):
            raise ValueError(f"ThreeDetectorScenario '{self.name}': read_sigma must be >= 0")
        if not math.isfinite(self.drift):
            raise ValueError(f"ThreeDetectorScenario '{self.name}': drift must be finite")
        # degenerate-rankable guard (adr-024 review MAJOR): a byte-clean
        # 'gaussian, sigma=0' card would be RANKED while the min formula scores
        # exactly 0 (the triviality trap), and 'none' with drift would degrade
        # the data while staying unranked — both misclassify the level
        if self.noise == "gaussian" and self.read_sigma == 0.0:
            raise ValueError(
                f"ThreeDetectorScenario '{self.name}': gaussian noise needs read_sigma > 0 "
                "(sigma=0 is a clean level in disguise and must not be ranked)"
            )
        if self.noise == "none" and self.drift != 0.0:
            raise ValueError(
                f"ThreeDetectorScenario '{self.name}': drift requires a noisy level "
                "(noise='none' rows are oracle/identifiability rows, never ranked)"
            )
        if self.n_days < 1:
            raise ValueError(f"ThreeDetectorScenario '{self.name}': n_days must be >= 1")
        if self.x_query.ndim != 1 or self.x_query.size < 1:
            raise ValueError(f"ThreeDetectorScenario '{self.name}': x_query must be 1-D, non-empty")
        if np.any((self.x_query < 0.0) | (self.x_query > self.length)):
            raise ValueError(
                f"ThreeDetectorScenario '{self.name}': x_query must lie in [0, L]"
            )

        # the vehicle scale must be finite: total_demand = inf makes the
        # evaluator's eps_count infinite and silently neuters EVERY censor gate
        # (adr-024 review — it also enabled a hash byte-migration collision)
        v = self.total_demand
        if not math.isfinite(v):
            raise ValueError(
                f"ThreeDetectorScenario '{self.name}': total demand must be finite "
                f"(got {v!r} — an infinite vehicle scale disables all gates)"
            )
        if self.noise == "poisson":
            apex_q = self.vf * self.w * self.kappa / (self.vf + self.w)
            lam_max = float(max(self.inflow_rates.max(initial=0.0), self.meter_cap, apex_q))
            if lam_max * self.grid.dt > 1e15:
                raise ValueError(
                    f"ThreeDetectorScenario '{self.name}': per-step Poisson mean "
                    f"~{lam_max * self.grid.dt:.3g} exceeds the safe sampling bound 1e15"
                )
        # float64 conditioning gate (the vi-due lesson): the congested branch adds
        # kappa*(L-x); its ULP must sit far below the count resolution, else the
        # min-switch is unresolvable and the reconstruction is float-garbage.
        max_store = self.kappa * float(np.max(self.length - self.x_query))
        if max_store * _EPS_MACH > 1e-9 * max(1.0, v):
            raise ValueError(
                f"ThreeDetectorScenario '{self.name}': ill-conditioned storage term "
                f"kappa*(L-x)={max_store!r} against count scale {v!r} "
                "(the congested-branch ULP exceeds the count resolution)"
            )

        # Realizability by construction: the regenerated boundary pair must satisfy
        # the single-link Newell envelopes exactly (it is clean LWR data). Also
        # forces the truth scenario to build (validates the recipe end-to-end).
        times, n_up, n_dn = self.truth_boundary_curves()
        self._assert_envelopes(times, n_up, n_dn)

    # ------------------------------------------------------------------

    @property
    def total_demand(self) -> float:
        """Vehicle scale ``V`` = total corridor inflow over the demand window.
        Overflow to ``inf`` is expected on rejected extreme inputs — the
        constructor's finiteness gate is the consumer of that signal."""
        with np.errstate(over="ignore"):
            return float(np.sum(np.diff(self.inflow_breakpoints) * self.inflow_rates))

    def _truth_scenario(self) -> DynamicScenario:
        """The internal two-link corridor whose LTM run yields the detector curves.

        Link 0 is the study link ``[0, L]``; link 1 reuses the study link's wave
        speeds with a ``meter_cap`` cap and acts as a pure downstream metering
        bottleneck (its head is a destination that absorbs, so it never blocks
        itself). HARNESS-ONLY: this is the ground-truth recipe, never exposed to a
        submission."""
        net = Network(
            name=self.name,
            n_nodes=3,
            n_zones=2,
            first_thru_node=1,
            init_node=np.array([1, 3], dtype=np.int64),
            term_node=np.array([3, 2], dtype=np.int64),
            capacity=np.ones(2),
            length=np.zeros(2),
            free_flow_time=np.ones(2),
            b=np.zeros(2),
            power=np.ones(2),
            toll=np.zeros(2),
            link_type=np.ones(2, dtype=np.int64),
        )
        apex = self.vf * self.w * self.kappa / (self.vf + self.w)
        cap0 = apex if self.capacity is None else min(apex, self.capacity)
        # clip the meter link at the geometric apex: LinkDynamics rejects a
        # capacity above it, and a meter that never binds is physics-identical
        # to one clipped at the apex (receiving is min-capped anyway) — so
        # "meter above inflow" instances stay constructible (adr-024 review)
        dynamics = LinkDynamics(
            length=np.array([self.length, self.length]),
            free_speed=np.array([self.vf, self.vf]),
            wave_speed=np.array([self.w, self.w]),
            jam_density=np.array([self.kappa, self.kappa]),
            capacity=np.array([cap0, min(self.meter_cap, apex)]),
        )
        rates = np.zeros((self.inflow_rates.shape[0], 2, 2))
        rates[:, 0, 1] = self.inflow_rates
        demand = DynamicDemand(breakpoints=self.inflow_breakpoints, rates=rates)
        return DynamicScenario(
            name=self.name, network=net, dynamics=dynamics, demand=demand, grid=self.grid
        )

    def truth_boundary_curves(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Regenerate the CLEAN detector curves ``(times, N_up, N_dn)`` from the
        hashed recipe by running LTM on the corridor. ``N_up`` = study-link
        cumulative inflow at ``x = 0``; ``N_dn`` = cumulative outflow at ``x = L``.
        Recomputed fresh on every call (never a stored array — the adr-023
        certifier discipline)."""
        out = NetworkLoader(self._truth_scenario(), LTMLink).run()
        return self.grid.edges, out.n_in[0].copy(), out.n_out[0].copy()

    def reference_field(self) -> np.ndarray:
        """The oracle interior field ``M(x_i, t_k)`` = Newell's exact min on the
        CLEAN boundary curves at the fixed query grid (``(m, K+1)``)."""
        times, n_up, n_dn = self.truth_boundary_curves()
        return reconstruct_field(
            self.vf, self.w, self.kappa, self.length, times, n_up, n_dn,
            self.x_query, self.grid.dt,
        )

    def _assert_envelopes(self, times: np.ndarray, n_up: np.ndarray, n_dn: np.ndarray) -> None:
        dt = self.grid.dt
        eps = 1e-9 * max(1.0, self.total_demand)
        ff = _interp_curve(n_up, times - self.length / self.vf, dt)
        if float(np.max(n_dn - ff)) > eps:
            raise ValueError(
                f"ThreeDetectorScenario '{self.name}': generated boundary pair "
                "violates the free-flow envelope N_dn(t) <= N_up(t - L/vf)"
            )
        bw = _interp_curve(n_dn, times - self.length / self.w, dt) + self.kappa * self.length
        if float(np.max(n_up - bw)) > eps:
            raise ValueError(
                f"ThreeDetectorScenario '{self.name}': generated boundary pair "
                "violates the backward-wave envelope N_up(t) <= N_dn(t - L/w) + kappa*L"
            )

    # ------------------------------------------------------------------

    def content_hash(self) -> str:
        """SHA-256 over all scored content, domain-separated from every other
        scenario space (``"tabench-newell3d-scenario-v1;"``)."""
        h = hashlib.sha256()
        h.update(b"tabench-newell3d-scenario-v1;")
        cap = "none" if self.capacity is None else repr(float(self.capacity))
        h.update(
            f"vf={self.vf!r};w={self.w!r};kappa={self.kappa!r};L={self.length!r};"
            f"cap={cap};meter={self.meter_cap!r};".encode()
        )
        for label, arr in (
            ("bp", self.inflow_breakpoints),
            ("rate", self.inflow_rates),
            ("xq", self.x_query),
        ):
            # length framing: without it, a float64 whose bytes spell a label
            # lets one array's tail masquerade as the next array's head — a
            # confirmed byte-migration collision (adr-024 review)
            h.update(f"{label}:{arr.size};".encode())
            h.update(_as_f64(arr).tobytes())
        h.update(f"dt={float(self.grid.dt)!r};K={self.grid.n_steps};".encode())
        h.update(
            f"noise={self.noise};sigma={self.read_sigma!r};drift={self.drift!r};"
            f"ndays={self.n_days};seed={self.seed};".encode()
        )
        for label, wins in (("upw", self.up_windows), ("dnw", self.dn_windows)):
            h.update(label.encode())
            for a, b in wins:
                h.update(f"{a!r},{b!r};".encode())
        return h.hexdigest()
