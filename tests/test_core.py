"""Core abstraction tests: factors, budgets, RNG schema, hashing, capabilities."""

import numpy as np
import pytest

from tabench import Budget, BudgetCoords, Capabilities, FrankWolfeModel, RngBundle, braess_scenario
from tabench.core.factors import FactorSpec, resolve_factors


def test_budget_requires_a_constraint():
    with pytest.raises(ValueError):
        Budget()


def test_budget_exhaustion_axes():
    b = Budget(iterations=10, wall_seconds=1.0)
    assert not b.exhausted(BudgetCoords(iterations=9, wall_ms=500))
    assert b.exhausted(BudgetCoords(iterations=10, wall_ms=500))
    assert b.exhausted(BudgetCoords(iterations=1, wall_ms=1500))


def test_rng_streams_are_deterministic_and_independent():
    a1 = RngBundle(42, macrorep=0).generator(source=0).random(8)
    a2 = RngBundle(42, macrorep=0).generator(source=0).random(8)
    b = RngBundle(42, macrorep=0).generator(source=1).random(8)
    c = RngBundle(42, macrorep=1).generator(source=0).random(8)
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, b)
    assert not np.array_equal(a1, c)


def test_factor_validation():
    specs = {"tol": FactorSpec(default=1e-6, bounds=(1e-12, 1e-2))}
    assert resolve_factors(specs, {})["tol"] == 1e-6
    assert resolve_factors(specs, {"tol": 1e-4})["tol"] == 1e-4
    with pytest.raises(ValueError):
        resolve_factors(specs, {"unknown": 1})
    with pytest.raises(ValueError):
        resolve_factors(specs, {"tol": 1.0})
    with pytest.raises(ValueError):
        FrankWolfeModel(nonexistent_factor=3)


def test_scenario_hash_is_content_sensitive():
    s1 = braess_scenario()
    s2 = braess_scenario()
    s3 = braess_scenario(demand=7.0)
    assert s1.content_hash() == s2.content_hash()
    assert s1.content_hash() != s3.content_hash()


def test_capabilities_rejects_unknown_paradigm():
    with pytest.raises(ValueError):
        Capabilities(
            paradigm="clairvoyant", deterministic=True, provides_gap=False, seedable=True
        )
