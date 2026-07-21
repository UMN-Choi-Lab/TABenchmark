"""Shared T2 runner scaffolding helpers + the three deliberate B7 deltas.

B7 hoisted the manifest/writer/sensor-design blocks that were cloned across the
three T2 runners (run_estimation_experiment, run_dynamic_estimation_experiment,
run_bo4mob_estimation_experiment) in experiments/runner.py into module-level
private helpers. The refactor is a byte-for-byte no-op except three deliberate,
tested deltas:
  D1 -- the empty-held-out raise, previously dynamic-only, now fires on the static
        runner too (heldout_count_rmse is the ranking column on both tracks);
  D2 -- the dynamic manifest now records the top-level ``certificate`` block;
  D3 -- the bo4mob manifest now records the ``rng`` provenance block.
These pin each helper's block directly (so a future edit can't silently drift the
manifest) and the three deltas.
"""

import os
import platform
import sys

import numpy
import pytest

import tabench as tb
from tabench.core.budget import Budget
from tabench.core.rng import (
    SOURCE_BOOTSTRAP,
    SOURCE_EVALUATION,
    SOURCE_OBSERVATION,
    SOURCE_PRIOR,
)
from tabench.data.builtin import two_route_scenario
from tabench.estimation import (
    DynamicPriorBaseline,
    PriorBaseline,
)
from tabench.experiments import runner as R
from tabench.experiments.runner import (
    run_dynamic_estimation_experiment,
    run_estimation_experiment,
)


# ------------------------------------------------------------ helper unit pins
def test_budget_manifest_block():
    b = Budget(iterations=10, target_relative_gap=1e-6)
    assert R._budget_manifest(b) == {
        "iterations": 10,
        "sp_calls": None,
        "wall_seconds": None,
        "target_relative_gap": 1e-6,
    }


def test_budget_part_slug_omits_none_axes_and_the_gap():
    # it/sp/ws only (the gap axis is T1's slug, not T2's); None axes are dropped.
    assert R._budget_part(Budget(sp_calls=500)) == "sp500"
    assert R._budget_part(Budget(iterations=3, sp_calls=7)) == "it3-sp7"


def test_rng_manifest_block_carries_all_reserved_sources():
    r = R._rng_manifest(4)
    assert r["root_seed"] == 4
    assert "Philox" in r["schema"]
    assert r["reserved_sources"] == {
        "observation": SOURCE_OBSERVATION,
        "evaluation": SOURCE_EVALUATION,
        "bootstrap": SOURCE_BOOTSTRAP,
        "prior": SOURCE_PRIOR,
    }


def test_environment_manifest_block():
    e = R._environment_manifest()
    assert set(e) == {"python", "platform", "numpy", "scipy", "tabench", "git_commit"}
    assert e["numpy"] == numpy.__version__
    assert e["python"] == sys.version.split()[0]
    assert e["platform"] == platform.platform()


def test_estimators_manifest_block_shape():
    m = R._estimators_manifest([PriorBaseline()])
    assert set(m) == {"prior"}
    assert set(m["prior"]) == {"capabilities", "factors"}
    assert set(m["prior"]["capabilities"]) == {
        "paradigm", "deterministic", "seedable",
        "inputs_required", "outputs", "trained_on",
    }


def test_check_unique_estimator_names_returns_names_and_raises_on_dupes():
    assert R._check_unique_estimator_names([PriorBaseline()]) == ["prior"]
    with pytest.raises(ValueError, match="Duplicate estimator names"):
        R._check_unique_estimator_names([PriorBaseline(), PriorBaseline()])


# --------------------------------------------------------------------- D1
def test_d1_empty_heldout_raises_on_both_t2_runners():
    """D1: an explicit empty held-out set is now rejected on the STATIC runner too
    (it was only guarded on the dynamic runner), with the same message shape. Before
    B7 the static runner silently produced an all-NaN held-out design; both tracks
    rank on heldout_count_rmse, so the defect is the same and now fails fast."""
    with pytest.raises(ValueError, match="held-out sensor set is empty"):
        run_estimation_experiment(
            tb.braess_scenario(6.0), [PriorBaseline()], Budget(sp_calls=100),
            estimation={"sensors": {"kind": "explicit", "links": [1, 2]},
                        "heldout": {"kind": "explicit", "links": []}},
        )
    with pytest.raises(ValueError, match="held-out sensor set is empty"):
        run_dynamic_estimation_experiment(
            two_route_scenario(sue_theta=None), [DynamicPriorBaseline()],
            Budget(sp_calls=100),
            estimation={"sensors": {"kind": "explicit", "links": [3]},
                        "heldout": {"kind": "explicit", "links": []},
                        "n_slices": 3, "slice_length": 2.0},
        )


# --------------------------------------------------------------------- D2
def test_d2_dynamic_manifest_records_certificate():
    """D2: the dynamic manifest now records the top-level ``certificate`` block that
    the runner already builds and passes into the task -- static and bo4mob already
    recorded theirs; the dynamic one was constructed but never surfaced."""
    res = run_dynamic_estimation_experiment(
        two_route_scenario(sue_theta=None), [DynamicPriorBaseline()],
        Budget(sp_calls=100),
        estimation={"sensors": {"kind": "explicit", "links": [3]},
                    "heldout": {"kind": "explicit", "links": [2]},
                    "n_slices": 3, "slice_length": 2.0, "noise": "none",
                    "prior": {"kind": "stale", "cv": 0.0}},
    )
    assert "certificate" in res.manifest
    # equals the certificate the runner threads into the DynamicEstimationTask
    assert res.manifest["certificate"] == {
        "assignment": "exact-linear-map",
        "map_recipe": "frozen_freeflow_v1",
    }


# --------------------------------------------------------------------- D3
def test_d3_bo4mob_manifest_records_rng():
    """D3: the bo4mob manifest now records the shared ``rng`` provenance block (its
    estimators receive RngBundle streams, so the P8 provenance applies) -- static and
    dynamic already recorded it. Engine-gated: skips without the sumo wheel; and, like
    tests/test_bo4mob_estimation.py, skips when the 1ramp bundle is unreachable (offline)
    unless TABENCH_REQUIRE_DATA forces the hard failure CI relies on."""
    pytest.importorskip("sumo")
    from tabench.data.bo4mob import (
        BO4MOB_REGISTRY,
        ChecksumError,
        fetch_bo4mob,
        fetch_bo4mob_heldout,
    )
    from tabench.estimation import Bo4MobPriorBaseline
    from tabench.experiments.runner import run_bo4mob_estimation_experiment

    try:  # the exact offline->skip / TABENCH_REQUIRE_DATA->hard-fail pattern
        fetch_bo4mob(BO4MOB_REGISTRY["1ramp"])
        fetch_bo4mob_heldout("1ramp")
    except ChecksumError:
        raise
    except Exception as exc:  # offline -> skip; TABENCH_REQUIRE_DATA -> hard fail
        if os.environ.get("TABENCH_REQUIRE_DATA"):
            raise
        pytest.skip(f"bo4mob 1ramp data unavailable: {exc}")

    res = run_bo4mob_estimation_experiment(
        "1ramp", [Bo4MobPriorBaseline()], Budget(sp_calls=1), seed=1
    )
    assert "rng" in res.manifest
    assert res.manifest["rng"] == R._rng_manifest(1)
