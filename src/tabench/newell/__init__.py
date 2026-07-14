"""Newell (1993) three-detector interior reconstruction — the benchmark's first
traffic-state-estimation task (parallel module, adr-024).

Newell's LOADING content (link-end sending/receiving) already ships as ``ltm``
(adr-016); this module ships the INTERIOR minimum principle as a state-estimation
task: given noisy / partial boundary detector curves, reconstruct the interior
cumulative field, scored against the harness-regenerated closed-form min.
"""

from .builtin import (
    newell_free_flow_scenario,
    newell_masked_upstream_scenario,
    newell_noisy_scenario,
    newell_spillback_scenario,
    newell_symmetric_scenario,
)
from .observe import DetectorObservation, observe_detectors
from .scenario import ThreeDetectorScenario, reconstruct_field
from .solve import (
    ThreeDetectorField,
    ThreeDetectorProblem,
    newell_min,
    newell_min_isotonic,
    problem_from_scenario,
)

__all__ = [
    "ThreeDetectorScenario",
    "reconstruct_field",
    "DetectorObservation",
    "observe_detectors",
    "ThreeDetectorProblem",
    "ThreeDetectorField",
    "problem_from_scenario",
    "newell_min",
    "newell_min_isotonic",
    "newell_free_flow_scenario",
    "newell_spillback_scenario",
    "newell_symmetric_scenario",
    "newell_noisy_scenario",
    "newell_masked_upstream_scenario",
]
