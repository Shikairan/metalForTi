"""30-dim Ridge linear basis for linResExp (linear + PySR residual)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.linear_model import Ridge


@dataclass
class LinearBasis:
    intercept: float
    coefficients: np.ndarray
    feature_names: List[str]
    alpha: float

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return (x @ self.coefficients + self.intercept).astype(np.float32)

    def predict_torch(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.as_tensor(self.coefficients, device=x.device, dtype=x.dtype)
        b = torch.as_tensor(self.intercept, device=x.device, dtype=x.dtype)
        return x @ w + b

    def to_equation(self, *, include_all_features: bool = True) -> str:
        parts: List[str] = [f"{self.intercept:.8g}"]
        for name, coef in zip(self.feature_names, self.coefficients):
            if not include_all_features and abs(coef) < 1e-12:
                continue
            sign = "+" if coef >= 0 else "-"
            parts.append(f"{sign} {abs(coef):.8g}*{name}")
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intercept": float(self.intercept),
            "ridge_alpha": float(self.alpha),
            "feature_names": list(self.feature_names),
            "coefficients": {n: float(c) for n, c in zip(self.feature_names, self.coefficients)},
            "linear_equation": self.to_equation(include_all_features=True),
            "n_features": len(self.feature_names),
        }


def fit_linear_basis(
    x: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: List[str],
    alpha: float = 1.0,
) -> LinearBasis:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape[1] != len(feature_names):
        raise ValueError(f"x has {x.shape[1]} cols but {len(feature_names)} feature names")
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(x, y)
    return LinearBasis(
        intercept=float(model.intercept_),
        coefficients=np.asarray(model.coef_, dtype=np.float64),
        feature_names=list(feature_names),
        alpha=float(alpha),
    )


def save_linear_basis(basis: LinearBasis, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(basis.to_dict(), f, indent=2, ensure_ascii=False)


def load_linear_basis(path: Path) -> LinearBasis:
    data = json.loads(path.read_text(encoding="utf-8"))
    names = list(data["feature_names"])
    coefs = np.array([data["coefficients"][n] for n in names], dtype=np.float64)
    return LinearBasis(
        intercept=float(data["intercept"]),
        coefficients=coefs,
        feature_names=names,
        alpha=float(data.get("ridge_alpha", 1.0)),
    )


def count_vars_in_expression(expression: Optional[str], feature_names: List[str]) -> int:
    if not expression or str(expression).strip() in ("", "None"):
        return 0
    text = str(expression)
    return sum(1 for n in feature_names if n in text)


def export_linres_json(
    basis: LinearBasis,
    residual_equation: str,
    out_path: Path,
    *,
    block_name: str,
) -> Dict[str, Any]:
    """Serialize linear basis + residual symbolic equation."""
    payload: Dict[str, Any] = {
        "block_name": block_name,
        "model": "linear_basis_plus_pysr_residual",
        "ridge_alpha": basis.alpha,
        "linear": basis.to_dict(),
        "residual": {"0": residual_equation},
        "equations": {
            "linear": basis.to_equation(include_all_features=True),
            "residual": residual_equation,
        },
        "feature_coverage": {
            "linear_n": len(basis.feature_names),
            "residual_n": count_vars_in_expression(residual_equation, basis.feature_names),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload
