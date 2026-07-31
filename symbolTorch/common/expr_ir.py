"""Controlled SymPy expression IR for linear + residual combination."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import sympy
from sympy.core.function import AppliedUndef

from .constants import FEATURE_NAMES

EVALUATOR_VERSION = "expr_ir_v1"

ALLOWED_FUNCS = {
    sympy.sin,
    sympy.exp,
    sympy.log,
    sympy.Add,
    sympy.Mul,
    sympy.Pow,
}


def build_linear_sympy_expr(
    intercept: float,
    coefficients: Sequence[float],
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> sympy.Expr:
    if len(coefficients) != len(feature_names):
        raise ValueError("coefficients length != feature_names")
    terms: List[sympy.Expr] = [sympy.Float(float(intercept))]
    for c, name in zip(coefficients, feature_names):
        sym = sympy.Symbol(name)
        terms.append(sympy.Float(float(c)) * sym)
    return sympy.Add(*terms, evaluate=False)


def _collect_symbols(expr: sympy.Expr) -> Set[sympy.Symbol]:
    return {s for s in expr.free_symbols if isinstance(s, sympy.Symbol)}


def validate_expression_tree(
    expr: sympy.Expr,
    *,
    allowed_symbols: Optional[Iterable[str]] = None,
    allow_sin: bool = True,
) -> None:
    allowed = set(allowed_symbols) if allowed_symbols is not None else set(FEATURE_NAMES)
    for s in _collect_symbols(expr):
        if str(s) not in allowed:
            raise ValueError(f"disallowed symbol in expression: {s}")

    for node in sympy.preorder_traversal(expr):
        if isinstance(node, sympy.Symbol):
            continue
        if isinstance(node, (sympy.Integer, sympy.Float, sympy.Rational, sympy.Number)):
            continue
        if node.is_Number:
            continue
        if isinstance(node, (sympy.Add, sympy.Mul, sympy.Pow)):
            continue
        if isinstance(node, sympy.Abs):
            raise ValueError("abs is not in the allowed operator set")
        if isinstance(node, sympy.sin):
            if not allow_sin:
                raise ValueError("sin not allowed")
            continue
        if isinstance(node, (sympy.exp, sympy.log)):
            continue
        # inv often appears as Pow(x, -1) after sympy; custom Function named inv
        if isinstance(node, AppliedUndef) and node.func.__name__ == "inv":
            continue
        if isinstance(node, sympy.Function):
            name = type(node).__name__
            if name in {"sin", "exp", "log", "inv"}:
                continue
            raise ValueError(f"disallowed function in expression: {name}")
        # UnaryMinus is Mul(-1, x)
        if isinstance(node, sympy.Basic):
            # Allow relational? no
            if isinstance(node, (sympy.Rel, sympy.Tuple, sympy.MatrixBase)):
                raise ValueError(f"disallowed node type: {type(node)}")
            continue
        raise ValueError(f"disallowed node: {node!r}")


def serialize_expression_ir(expr: sympy.Expr) -> Dict[str, Any]:
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "sympy_str": str(expr),
        "srepr": sympy.srepr(expr),
        "free_symbols": sorted(str(s) for s in _collect_symbols(expr)),
    }


def combine_linear_residual(
    linear_expr: sympy.Expr,
    residual_expr: sympy.Expr,
) -> sympy.Expr:
    return sympy.Add(linear_expr, residual_expr, evaluate=False)


def evaluate_expression_numpy(
    expr: sympy.Expr,
    X: np.ndarray,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> np.ndarray:
    """Evaluate a validated expression on [N, 30] float matrix."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(feature_names):
        raise ValueError(f"X shape {X.shape} incompatible with {len(feature_names)} features")

    inv = sympy.Function("inv")
    local_dict = {name: sympy.Symbol(name) for name in feature_names}
    local_dict["inv"] = inv

    # Rebuild from string if needed safety — use expr directly
    symbols = [sympy.Symbol(n) for n in feature_names]
    modules = [
        {
            "sin": np.sin,
            "exp": np.exp,
            "log": np.log,
            "inv": lambda z: 1.0 / z,
            "Abs": np.abs,
        },
        "numpy",
    ]
    fn = sympy.lambdify(symbols, expr, modules=modules)
    cols = [X[:, i] for i in range(X.shape[1])]
    out = fn(*cols)
    out = np.asarray(out, dtype=np.float64).reshape(-1)
    if out.shape[0] != X.shape[0]:
        # constant expression broadcasts
        out = np.broadcast_to(out, (X.shape[0],)).astype(np.float64).copy()
    return out


def parse_sympy_expr(raw: str, feature_names: Sequence[str] = FEATURE_NAMES) -> sympy.Expr:
    local = {n: sympy.Symbol(n) for n in feature_names}
    local["inv"] = sympy.Function("inv")
    local["sin"] = sympy.sin
    local["exp"] = sympy.exp
    local["log"] = sympy.log
    return sympy.sympify(raw, locals=local)
