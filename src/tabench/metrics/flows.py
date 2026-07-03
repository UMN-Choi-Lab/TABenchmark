"""Flow-accuracy metrics against a reference (oracle) solution."""

from __future__ import annotations

import numpy as np

__all__ = ["rmse", "nrmse"]


def rmse(link_flows: np.ndarray, oracle: np.ndarray) -> float:
    """Root-mean-square link-flow error."""
    v = np.asarray(link_flows, dtype=np.float64)
    o = np.asarray(oracle, dtype=np.float64)
    if v.shape != o.shape:
        raise ValueError(f"Shape mismatch: {v.shape} vs {o.shape}")
    return float(np.sqrt(np.mean((v - o) ** 2)))


def nrmse(link_flows: np.ndarray, oracle: np.ndarray) -> float:
    """RMSE normalized by the mean oracle flow (BO4Mob convention)."""
    o = np.asarray(oracle, dtype=np.float64)
    mean = float(o.mean())
    if mean <= 0:
        raise ValueError("Oracle flows have nonpositive mean; NRMSE undefined")
    return rmse(link_flows, o) / mean
