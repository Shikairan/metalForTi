"""Linear distillation of RGAT teacher outputs (OLS / Ridge)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
MIN_LINEAR_COEF_ABS = 1e-6


@dataclass
class LinearFitResult:
    intercept: float
    coefficients: np.ndarray  # [30]
    method: str
    alpha: Optional[float]
    rank: int
    singular_values: np.ndarray
    condition_number: float
    constant_columns: List[int]
    rank_deficient: bool
    coefficient_identifiability: bool
    rcond: Optional[float] = None
    cv_alphas: Optional[List[float]] = None
    cv_scores: Optional[List[float]] = None

    def to_json_dict(self, feature_names: Sequence[str]) -> Dict[str, Any]:
        coef = np.asarray(self.coefficients, dtype=np.float64).reshape(-1)
        if len(coef) != len(feature_names):
            raise ValueError("coefficients length != feature_names")
        rhs_raw, rhs_display = build_linear_equation_strings(
            self.intercept, coef, feature_names
        )
        return {
            "intercept": float(self.intercept),
            "coefficients": coef.tolist(),
            "feature_names": list(feature_names),
            "method": self.method,
            "alpha": None if self.alpha is None else float(self.alpha),
            "rank": int(self.rank),
            "singular_values": np.asarray(self.singular_values, dtype=np.float64).tolist(),
            "condition_number": float(self.condition_number),
            "constant_columns": list(self.constant_columns),
            "rank_deficient": bool(self.rank_deficient),
            "coefficient_identifiability": bool(self.coefficient_identifiability),
            "rcond": self.rcond,
            "cv_alphas": self.cv_alphas,
            "cv_scores": self.cv_scores,
            "equation_rhs_raw": rhs_raw,
            "equation_rhs_display": rhs_display,
        }


def enforce_min_coefficient_magnitude(
    coefficients: Sequence[float],
    *,
    min_abs: float = MIN_LINEAR_COEF_ABS,
) -> np.ndarray:
    """Ensure every linear coefficient has |c| >= min_abs (zero -> +min_abs)."""
    if min_abs <= 0:
        raise ValueError(f"min_abs must be > 0, got {min_abs}")
    coef = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    out = coef.copy()
    for i in range(out.size):
        if abs(out[i]) < min_abs:
            out[i] = min_abs if out[i] >= 0.0 else -min_abs
    return out


def build_linear_equation_strings(
    intercept: float,
    coefficients: Sequence[float],
    feature_names: Sequence[str],
) -> Tuple[str, str]:
    coef = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if len(coef) != len(feature_names):
        raise ValueError("coefficients length != feature_names")
    terms = [f"({c:.16g})*{name}" for c, name in zip(coef.tolist(), feature_names)]
    rhs_raw = " + ".join([f"({float(intercept):.16g})"] + terms)
    rhs_display = " + ".join(
        [f"{float(intercept):.8g}"]
        + [f"({c:.8g})*{n}" for c, n in zip(coef.tolist(), feature_names)]
    )
    return rhs_raw, rhs_display


def linear_json_with_min_coef(
    lin_json: Dict[str, Any],
    *,
    min_abs: float = MIN_LINEAR_COEF_ABS,
) -> Dict[str, Any]:
    """Copy linear JSON dict with floored coefficient magnitudes and rebuilt equations."""
    feature_names = list(lin_json["feature_names"])
    intercept = float(lin_json["intercept"])
    coef = enforce_min_coefficient_magnitude(lin_json["coefficients"], min_abs=min_abs)
    rhs_raw, rhs_display = build_linear_equation_strings(intercept, coef, feature_names)
    out = dict(lin_json)
    out["coefficients"] = coef.tolist()
    out["equation_rhs_raw"] = rhs_raw
    out["equation_rhs_display"] = rhs_display
    out["min_coefficient_abs"] = float(min_abs)
    return out


def _as_2d_float64(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError("X contains non-finite values")
    return a


def _as_1d_float64(y: np.ndarray) -> np.ndarray:
    a = np.asarray(y, dtype=np.float64).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError("y contains non-finite values")
    return a


def _design_diagnostics(X: np.ndarray) -> Tuple[List[int], np.ndarray, int, float]:
    const_cols = [
        int(j) for j in range(X.shape[1]) if np.allclose(X[:, j], X[0, j])
    ]
    X_aug = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float64), X], axis=1)
    # economy SVD for rank / condition
    s = np.linalg.svd(X_aug, compute_uv=False)
    eps = np.finfo(np.float64).eps
    tol = s.max() * max(X_aug.shape) * eps if s.size else 0.0
    rank = int(np.sum(s > tol)) if s.size else 0
    cond = float(s.max() / s.min()) if s.size and s.min() > 0 else float("inf")
    return const_cols, s, rank, cond


def _solve_ols(X: np.ndarray, y: np.ndarray, *, rcond: Optional[float] = None) -> Tuple[np.ndarray, Optional[float]]:
    X_aug = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float64), X], axis=1)
    if rcond is None:
        rcond = np.finfo(np.float64).eps * max(X_aug.shape)
    beta, *_ = np.linalg.lstsq(X_aug, y, rcond=rcond)
    return beta.reshape(-1), float(rcond)


def _solve_ridge(X: np.ndarray, y: np.ndarray, *, alpha: float) -> np.ndarray:
    if alpha <= 0:
        raise ValueError(f"Ridge requires alpha > 0, got {alpha}")
    n, d = X.shape
    X_aug = np.concatenate([np.ones((n, 1), dtype=np.float64), X], axis=1)
    ridge = np.sqrt(alpha) * np.eye(d + 1, dtype=np.float64)
    ridge[0, 0] = 0.0
    A = np.concatenate([X_aug, ridge], axis=0)
    b = np.concatenate([y, np.zeros(d + 1, dtype=np.float64)], axis=0)
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    return beta.reshape(-1)


def predict_linear(
    x: np.ndarray,
    intercept: float,
    coefficients: np.ndarray,
) -> np.ndarray:
    X = _as_2d_float64(x)
    coef = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if X.shape[1] != coef.shape[0]:
        raise ValueError(f"X cols {X.shape[1]} != coef {coef.shape[0]}")
    out = intercept + X @ coef
    return np.asarray(out, dtype=np.float64).reshape(-1)


def compute_teacher_residual(
    teacher: np.ndarray,
    linear: np.ndarray,
) -> np.ndarray:
    t = _as_1d_float64(teacher)
    l = _as_1d_float64(linear)
    if t.shape != l.shape:
        raise ValueError(f"shape mismatch teacher {t.shape} vs linear {l.shape}")
    return t - l


def _mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def select_ridge_alpha_cv(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[float, List[float], List[float]]:
    X = _as_2d_float64(X)
    y = _as_1d_float64(y)
    n = X.shape[0]
    k = min(int(n_splits), n)
    if k < 2:
        return float(alphas[0]), list(map(float, alphas)), [float("nan")] * len(alphas)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    scores: List[float] = []
    for alpha in alphas:
        fold_maes: List[float] = []
        for i in range(k):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            beta = _solve_ridge(X[train_idx], y[train_idx], alpha=float(alpha))
            pred = predict_linear(X[val_idx], float(beta[0]), beta[1:])
            fold_maes.append(_mae(pred, y[val_idx]))
        scores.append(float(np.mean(fold_maes)))
    best_i = int(np.nanargmin(scores))
    return float(alphas[best_i]), list(map(float, alphas)), scores


def fit_linear_teacher(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    method: str = "ridge",
    alpha: Optional[float] = None,
    alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
    cv_splits: int = 5,
    seed: int = 42,
    rcond: Optional[float] = None,
) -> LinearFitResult:
    X = _as_2d_float64(x_train)
    y = _as_1d_float64(y_train)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"n_samples mismatch X={X.shape[0]} y={y.shape[0]}")
    if X.shape[0] < 2:
        raise ValueError("need at least 2 training samples")

    const_cols, singular_values, rank, cond = _design_diagnostics(X)
    rank_def = rank < (X.shape[1] + 1)
    method = method.lower().strip()
    cv_alphas: Optional[List[float]] = None
    cv_scores: Optional[List[float]] = None
    used_rcond: Optional[float] = None

    if method == "ols":
        beta, used_rcond = _solve_ols(X, y, rcond=rcond)
        used_alpha: Optional[float] = None
    elif method == "ridge":
        if alpha is None:
            used_alpha, cv_alphas, cv_scores = select_ridge_alpha_cv(
                X, y, alphas=alphas, n_splits=cv_splits, seed=seed
            )
        else:
            used_alpha = float(alpha)
            if used_alpha <= 0:
                raise ValueError(f"Ridge alpha must be > 0, got {used_alpha}")
        beta = _solve_ridge(X, y, alpha=float(used_alpha))
    else:
        raise ValueError(f"Unknown method: {method} (expected ols|ridge)")

    return LinearFitResult(
        intercept=float(beta[0]),
        coefficients=np.asarray(beta[1:], dtype=np.float64),
        method=method,
        alpha=used_alpha if method == "ridge" else None,
        rank=rank,
        singular_values=np.asarray(singular_values, dtype=np.float64),
        condition_number=cond,
        constant_columns=const_cols,
        rank_deficient=rank_def,
        coefficient_identifiability=not rank_def,
        rcond=used_rcond,
        cv_alphas=cv_alphas,
        cv_scores=cv_scores,
    )
