"""Unit tests for linear fit / residual."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.linear_fit import (  # noqa: E402
    compute_teacher_residual,
    fit_linear_teacher,
    predict_linear,
)


def test_ridge_recovers_linear():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 30))
    true_coef = rng.normal(size=30)
    y = 1.5 + X @ true_coef + rng.normal(scale=0.01, size=80)
    fit = fit_linear_teacher(X, y, method="ridge", alpha=1e-3)
    pred = predict_linear(X, fit.intercept, fit.coefficients)
    resid = compute_teacher_residual(y, pred)
    assert fit.coefficients.shape == (30,)
    assert np.mean(np.abs(resid)) < 0.05
    # replay
    pred2 = fit.intercept + X @ fit.coefficients
    assert np.allclose(pred, pred2)


def test_ols_and_json_fields():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 30))
    y = X[:, 0] + 0.1 * X[:, 1]
    fit = fit_linear_teacher(X, y, method="ols")
    d = fit.to_json_dict([f"v{i}" for i in range(30)])
    assert len(d["coefficients"]) == 30
    assert "equation_rhs_raw" in d
