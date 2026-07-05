"""Model capability declarations and harness-enforced compatibility (P4, P7)."""

from __future__ import annotations

from dataclasses import dataclass

from .scenario import Scenario

__all__ = ["Capabilities", "ContaminationError", "assert_fair_evaluation"]

PARADIGMS = frozenset(
    {
        "static_ue",
        "static_ue_elastic",
        "static_so",
        "sue",
        "dta",
        "day_to_day",
        "learned",
        "heuristic",
        "estimation",
    }
)


@dataclass(frozen=True)
class Capabilities:
    """What a model declares about itself; the harness trusts nothing else.

    ``provides_gap`` means the model self-certifies an equilibrium gap. The
    harness still recomputes every scored metric externally (P1); self-reports
    are only diffed against harness values as an honesty check.

    ``trained_on`` lists scenario families in the model's training lineage
    (learned models). The fairness gate refuses evaluation on scenarios whose
    family appears here.
    """

    paradigm: str
    deterministic: bool
    provides_gap: bool
    seedable: bool
    inputs_required: frozenset[str] = frozenset({"od_matrix"})
    outputs: frozenset[str] = frozenset({"link_flows"})
    trained_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.paradigm not in PARADIGMS:
            raise ValueError(
                f"Unknown paradigm {self.paradigm!r}; expected one of {sorted(PARADIGMS)}"
            )


class ContaminationError(RuntimeError):
    """Raised when a model's training lineage intersects the evaluation scenario."""


def assert_fair_evaluation(capabilities: Capabilities, scenario: Scenario) -> None:
    """Fairness gate (P7): refuse train/test contamination.

    Declared lineage tokens are matched against both the scenario's family
    name and its content hash. Declaring **content hashes** is robust to
    scenario renaming; declaring only family names is not (a byte-identical
    scenario republished under a new family name will not match). Learned
    models should therefore declare content hashes of their training
    scenarios whenever available, and family names as a coarser fallback.
    """
    lineage = set(capabilities.trained_on)
    if not lineage:
        return
    if scenario.family in lineage or scenario.content_hash() in lineage:
        raise ContaminationError(
            f"Model trained on {sorted(lineage)} may not be evaluated on scenario "
            f"'{scenario.name}' (family '{scenario.family}')."
        )
