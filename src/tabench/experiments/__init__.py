"""Experiment layer: grid runner with certified scoring and manifests."""

from .runner import (
    ExperimentResult,
    identifiability_report,
    run_estimation_experiment,
    run_experiment,
)

__all__ = [
    "ExperimentResult",
    "run_experiment",
    "run_estimation_experiment",
    "identifiability_report",
]
