"""SymTorch distillation helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from .constants import FEATURE_NAMES

logger = logging.getLogger(__name__)


def require_symtorch():
    try:
        from symtorch import SymbolicModel  # noqa: WPS433
    except ImportError as e:
        raise ImportError(
            "symtorch (torch-symbolic) is required. Use Python >= 3.11:\n"
            "  pip install torch-symbolic\n"
            f"Original error: {e}"
        ) from e
    return SymbolicModel


def build_sr_params(
    *,
    niterations: int,
    quick: bool,
    maxsize: int | None = None,
) -> Dict[str, Any]:
    from .constants import DEFAULT_SR_PARAMS, QUICK_SR_PARAMS

    params = dict(DEFAULT_SR_PARAMS)
    params["niterations"] = int(niterations)
    if maxsize is not None:
        params["maxsize"] = int(maxsize)
    elif "maxsize" not in params:
        params["maxsize"] = 40
    if quick:
        params.update(QUICK_SR_PARAMS)
        params.setdefault("maxsize", 20)
    return params


def _equation_string(regressor) -> str:
    try:
        if hasattr(regressor, "sympy"):
            return str(regressor.sympy())
    except Exception:
        pass
    try:
        df = regressor.equations_
        if df is not None and len(df) > 0:
            idx = getattr(regressor, "equation_selection_idx", 0)
            row = df.iloc[int(idx)]
            return str(row.get("equation", row))
    except Exception:
        pass
    return str(regressor)


def distill_block_on_numpy_io(
    block: Callable,
    x_np: np.ndarray,
    *,
    block_name: str,
    sr_params: Optional[Dict[str, Any]] = None,
    variable_names: Optional[List[str]] = None,
    save_path: Optional[Path | str] = None,
):
    """Model-agnostic distill when inputs are not parent-model graph features.

    save_path: PySR 输出根目录。SymTorch 会写入 ``{save_path}/{block_name}/``。
    应传入本次运行目录下的路径（如 ``runs/<run>/SR_output``），避免不同 ckpt
    共用 ``tabular_fs`` 等目录互相覆盖。
    """
    SymbolicModel = require_symtorch()
    sym = SymbolicModel(block, block_name=block_name)
    fit_params = {}
    if variable_names:
        fit_params["variable_names"] = variable_names
    distill_kwargs: Dict[str, Any] = {
        "sr_params": sr_params,
        "fit_params": fit_params or None,
    }
    if save_path is not None:
        distill_kwargs["save_path"] = str(Path(save_path))
    sym.distill(x_np, **distill_kwargs)
    sym.switch_to_symbolic()
    return sym


def export_equations_json(sym, out_path: Path, *, target: Optional[str] = None) -> Dict[str, Any]:
    """Serialize equation strings；并写出完整可读方程字段。"""
    from .equation_format import enrich_and_write_equation_json

    reg = sym.pysr_regressor
    # 取第 0 维（lowExp 单输出）
    raw = ""
    for dim, model in sorted(reg.items(), key=lambda kv: kv[0]):
        if int(dim) == 0 or raw == "":
            raw = _equation_string(model)
            if int(dim) == 0:
                break
    tgt = target or ("YS" if "ys" in str(sym.block_name).lower() else "FS")
    return enrich_and_write_equation_json(
        out_path,
        target=tgt,
        block_name=str(sym.block_name),
        raw_expr=raw,
    )


def save_symbolic_module(sym, path: Path, *, target: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    json_path = path.with_name(path.stem + ".json")
    export_equations_json(sym, json_path, target=target)
    try:
        import dill  # noqa: WPS433

        with path.open("wb") as f:
            dill.dump(sym, f)
    except Exception as exc:
        logger.warning("Could not pickle %s (%s). Equations saved to %s", path, exc, json_path)


def make_tabular_lookup_fn(x_ref: np.ndarray, y_ref: np.ndarray) -> Callable:
    """Row-wise lookup for teacher labels (used during lowExp IO collection)."""
    x_ref = np.asarray(x_ref, dtype=np.float32)
    y_ref = np.asarray(y_ref, dtype=np.float32).reshape(-1, 1)

    def f(x_np: np.ndarray) -> np.ndarray:
        x_np = np.asarray(x_np, dtype=np.float32)
        out = np.zeros((x_np.shape[0], 1), dtype=np.float32)
        for i, row in enumerate(x_np):
            j = int(np.argmin(np.linalg.norm(x_ref - row, axis=1)))
            out[i, 0] = y_ref[j, 0]
        return out

    return f


