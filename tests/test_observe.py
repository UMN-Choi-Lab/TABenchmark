"""Observation-level tests: seeded projections and the identifiability check."""

import numpy as np
import pytest

from tabench import RngBundle, braess_scenario
from tabench.core.rng import SOURCE_OBSERVATION
from tabench.observe import (
    FullOD,
    LinkCounts,
    distinct_nonzero_columns,
    random_sensor_mask,
)

TRUTH = np.array([4.0, 2.0, 2.0, 2.0, 4.0])


@pytest.fixture()
def scenario():
    return braess_scenario()


def test_full_od_returns_demand(scenario):
    rng = RngBundle(0).generator(SOURCE_OBSERVATION)
    ds = FullOD().observe(scenario, TRUTH, rng)
    assert np.array_equal(ds.payload["od_matrix"], scenario.demand.matrix)


def test_link_counts_shapes_and_determinism(scenario):
    level = LinkCounts(sensor_links=np.array([0, 2, 4]), n_periods=5, noise="poisson")
    ds1 = level.observe(scenario, TRUTH, RngBundle(7).generator(SOURCE_OBSERVATION))
    ds2 = level.observe(scenario, TRUTH, RngBundle(7).generator(SOURCE_OBSERVATION))
    ds3 = level.observe(scenario, TRUTH, RngBundle(8).generator(SOURCE_OBSERVATION))
    assert ds1.payload["counts"].shape == (5, 3)
    assert np.array_equal(ds1.payload["counts"], ds2.payload["counts"])
    assert not np.array_equal(ds1.payload["counts"], ds3.payload["counts"])
    assert ds1.meta["coverage"] == pytest.approx(3 / 5)


def test_link_counts_noiseless(scenario):
    level = LinkCounts(sensor_links=np.array([1, 3]), n_periods=2, noise="none")
    ds = level.observe(scenario, TRUTH, RngBundle(0).generator(SOURCE_OBSERVATION))
    assert np.array_equal(ds.payload["counts"], np.array([[2.0, 2.0], [2.0, 2.0]]))


def test_random_sensor_mask_reproducible():
    m1 = random_sensor_mask(76, 0.3, RngBundle(1).generator(SOURCE_OBSERVATION))
    m2 = random_sensor_mask(76, 0.3, RngBundle(1).generator(SOURCE_OBSERVATION))
    assert np.array_equal(m1, m2)
    assert len(m1) == round(0.3 * 76)


def test_identifiability_condition():
    # Distinct nonzero columns -> identifiable (Hazelton 2015, Prop. 1).
    good = np.array([[1, 0, 1], [0, 1, 1]])
    assert distinct_nonzero_columns(good)
    # Duplicate columns -> not identifiable.
    dup = np.array([[1, 1], [0, 0]])
    assert not distinct_nonzero_columns(dup)
    # A zero column -> not identifiable.
    zero = np.array([[1, 0], [1, 0]])
    assert not distinct_nonzero_columns(zero)
