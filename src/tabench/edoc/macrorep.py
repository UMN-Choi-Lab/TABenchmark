"""P8 macroreps + bootstrap CI for the EDOC stochastic track (adr-036 R5, adr-039).

Design: docs/design/adr-036-external-dynamic-observational-certificate.md
(ruling R5) realized by docs/design/adr-039-matsim.md (the first stochastic-track
row). The engine is injected as callables (the ReplayRunner precedent), so this
substrate is engine-free and reusable by every future stochastic row.

The stochastic shape (R5): P8 macroreps over the instance's PINNED ``seed_list``
(>= 5, a hashed field); each macrorep emits and certifies **its own** final
iterate under its own seed (G1 per seed); the row score is the mean ``RG_D1``
with a percentile-bootstrap CI on the reserved ``SOURCE_BOOTSTRAP`` stream
(:func:`tabench.experiments.bootstrap.bootstrap_ci` — house P8, never
reimplemented, byte-reproducible from the pinned list alone). Single-iteration /
single-seed readouts are structurally impossible through this API: the row score
only exists at this harness level.

Row-level semantics (parent rulings, ratified in adr-039):

* **Row floor (ruling 2):** ``floor_gap`` at the row level is the MEAN of the
  per-seed floor_gaps — consistent with the mean score's estimand.
* **Censoring (ruling 3):** ANY censored macrorep censors the WHOLE row
  (``feasible=0``, ``NaN`` mean/CI). A subset mean is a different estimand and
  dropping seeds re-opens seed shopping (forgery pairs 4/N3). Infrastructure
  exceptions from ``emit``/``certify`` still RAISE un-laundered (R6).
* **Seed integrity (forgery pair N1):** per-seed instances are derived HERE from
  the base's hashed list (never read from an emission), and an emission whose
  provenance seed differs from its macrorep's pinned seed RAISES ``ValueError``
  (a config error, never a censor).

The CI is report-only: it never gates and never reclassifies (forgery pair N4 —
there is nothing to launder through it).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np

from .replay import EmittedBundle
from .scenario import EdocScenario

__all__ = ["MacrorepResult", "certify_macroreps", "per_seed_scenarios"]

# Row-level metric keys (NaN'd wholesale when any macrorep censors).
_ROW_SCORED = (
    "rg_d1_mean",
    "rg_d1_ci_lo",
    "rg_d1_ci_hi",
    "floor_gap",
    "sub_floor",
    "mean_backlog_mean",
    "max_backlog_max",
    "delta_mean",
    "delta_max",
)


@dataclasses.dataclass(frozen=True)
class MacrorepResult:
    """The row-level result of a P8 macrorep certification: the per-seed
    certified metrics (diagnostics, one dict per pinned seed) and the row
    ``metrics`` (feasible, mean ``RG_D1`` + bootstrap CI, the row floor, and
    Tier-B backlog/delta aggregates)."""

    per_seed: dict[int, dict[str, float]]
    metrics: dict[str, float]


def per_seed_scenarios(base: EdocScenario) -> tuple[EdocScenario, ...]:
    """Derive the macrorep instances from ``base``: one per pinned seed, each
    pinning ITS OWN ``seed`` (so its content hash moves with the seed and the
    substrate's per-emission G0 ``emitted.seed == sc.seed`` check applies
    unchanged per macrorep) while carrying the full hashed ``seed_list``.

    A deterministic-track instance (empty ``seed_list``) on this stochastic API
    is a config error and RAISES ``ValueError`` (adr-036 R5)."""
    if not base.seed_list:
        raise ValueError(
            f"EdocScenario {base.name!r} has no pinned seed_list — the stochastic "
            "macrorep harness needs the R5 pinned list (>= 5 seeds); deterministic-"
            "track instances are certified per emission, not through this API"
        )
    return tuple(dataclasses.replace(base, seed=s) for s in base.seed_list)


def certify_macroreps(
    base: EdocScenario,
    emit: Callable[[EdocScenario], EmittedBundle],
    certify: Callable[[EdocScenario, EmittedBundle], dict[str, float]],
    *,
    b: int = 10000,
    level: float = 0.95,
) -> MacrorepResult:
    """Emit + certify one macrorep per pinned seed and aggregate to the row.

    ``emit`` produces the model's bundle for one per-seed instance; ``certify``
    is the row's certification path (the row passes its own wrapper so R3 /
    separation vetting run per seed). Exceptions from either PROPAGATE (R6:
    infrastructure is never laundered); a certified ``feasible != 1`` on ANY
    macrorep censors the WHOLE row (ruling 3). On an all-feasible row the score
    is ``mean(rg_d1)`` with the house percentile-bootstrap CI drawn on the
    reserved ``SOURCE_BOOTSTRAP`` stream at ``root_seed = seed_list[0]``
    (deterministic in the pinned list alone — P8), and the row floor is the mean
    of the per-seed floor_gaps (ruling 2).

    CONTRACT (adr-039, the N2 boundary): ``emit`` must be HARNESS-controlled —
    a fresh per-seed engine run keyed on ``sc.seed``, as every shipped row's
    score path wires it — never a model-supplied bundle: when an engine's
    replay map is seed-independent, feeding one cherry-picked bundle to every
    macrorep would pass G1 per seed and void the N2 defense (a zero-width CI
    over a best-of-N emission), which no check here can detect."""
    # function-level import: tabench.experiments pulls estimation -> models ->
    # adapters -> edoc, so a module-level import here would be circular (the
    # sumo_duaiterate EdocEvaluator precedent).
    from ..experiments.bootstrap import bootstrap_ci

    scenarios = per_seed_scenarios(base)
    per_seed: dict[int, dict[str, float]] = {}
    for sc_s in scenarios:
        emitted = emit(sc_s)
        if int(emitted.seed) != int(sc_s.seed):
            raise ValueError(
                f"macrorep seed integrity (pair N1): emission claims seed "
                f"{emitted.seed} but the pinned macrorep instance is seed {sc_s.seed} "
                "— a config error, never a censor"
            )
        per_seed[int(sc_s.seed)] = dict(certify(sc_s, emitted))

    n = len(scenarios)
    metrics: dict[str, float] = {
        "n_seeds": float(n),
        "ci_level": float(level),
    }
    if any(m.get("feasible") != 1.0 for m in per_seed.values()):
        # Ruling 3: one censored macrorep censors the row; subset means re-open
        # seed shopping. Per-seed diagnostics stay available in ``per_seed``.
        metrics["feasible"] = 0.0
        metrics.update(dict.fromkeys(_ROW_SCORED, float("nan")))
        return MacrorepResult(per_seed=per_seed, metrics=metrics)

    rg = np.array([per_seed[s]["rg_d1"] for s in base.seed_list], dtype=np.float64)
    ci = bootstrap_ci(rg, root_seed=int(base.seed_list[0]), b=b, level=level)
    floor_row = float(np.mean([per_seed[s]["floor_gap"] for s in base.seed_list]))
    metrics.update(
        {
            "feasible": 1.0,
            "rg_d1_mean": float(ci.point),
            "rg_d1_ci_lo": float(ci.lo),
            "rg_d1_ci_hi": float(ci.hi),
            "floor_gap": floor_row,
            "sub_floor": 1.0 if float(ci.point) < floor_row else 0.0,
            # Tier-B aggregates (report-never-gate; the per-seed gates already ran)
            "mean_backlog_mean": float(
                np.mean([per_seed[s]["mean_backlog"] for s in base.seed_list])
            ),
            "max_backlog_max": float(
                max(per_seed[s]["max_backlog"] for s in base.seed_list)
            ),
            "delta_mean": float(np.mean([per_seed[s]["delta"] for s in base.seed_list])),
            "delta_max": float(max(per_seed[s]["delta"] for s in base.seed_list)),
        }
    )
    return MacrorepResult(per_seed=per_seed, metrics=metrics)
