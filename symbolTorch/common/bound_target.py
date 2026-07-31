"""Strictly bound SymbolTorch target callables (no nearest-neighbor lookup)."""

from __future__ import annotations

from typing import Callable

import numpy as np


def as_float32_2d(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {a.shape}")
    return a


def as_float32_column(y) -> np.ndarray:
    a = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    return a


def make_bound_target_fn(
    x_expected: np.ndarray,
    y_expected: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Bind (X_train, residual_train) for SymbolTorch.distill.

    Raises if SymbolTorch changes shape, values, or row order.
    Does not deduplicate rows.
    """
    x_ref = as_float32_2d(x_expected)
    y_ref = as_float32_column(y_expected)
    if x_ref.shape[0] != y_ref.shape[0]:
        raise ValueError(
            f"x/y row mismatch: x={x_ref.shape[0]} y={y_ref.shape[0]}"
        )

    def target_fn(x_received: np.ndarray) -> np.ndarray:
        x = as_float32_2d(x_received)
        if x.shape != x_ref.shape:
            raise ValueError(
                f"unexpected SymbolTorch input shape {x.shape}, expected {x_ref.shape}"
            )
        if not np.array_equal(x, x_ref):
            raise ValueError(
                "SymbolTorch changed input values or row order relative to bound residual targets"
            )
        return y_ref.copy()

    return target_fn
