"""Unit tests for bound target fn."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.bound_target import make_bound_target_fn  # noqa: E402


def test_bound_target_ok_and_rejects_reorder():
    X = np.arange(20, dtype=np.float32).reshape(5, 4)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    fn = make_bound_target_fn(X, y)
    out = fn(X.copy())
    assert out.shape == (5, 1)
    assert np.allclose(out[:, 0], y)

    try:
        fn(X[::-1].copy())
        raise AssertionError("expected ValueError on reorder")
    except ValueError:
        pass

    try:
        fn(X[:3].copy())
        raise AssertionError("expected ValueError on shape")
    except ValueError:
        pass
