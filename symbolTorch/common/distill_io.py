"""SymTorch distillation helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from .constants import DEFAULT_SLIME_PARAMS, FEATURE_NAMES

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
    low_exp: bool = False,
) -> Dict[str, Any]:
    from .constants import DEFAULT_SR_PARAMS, DEFAULT_SR_PARAMS_LOW, QUICK_SR_PARAMS

    base = DEFAULT_SR_PARAMS_LOW if low_exp else DEFAULT_SR_PARAMS
    params = dict(base)
    params["niterations"] = int(niterations)
    if quick:
        params.update(QUICK_SR_PARAMS)
    return params


def build_fit_params(variable_names: Optional[List[str]] = None) -> Dict[str, Any]:
    names = variable_names if variable_names is not None else list(FEATURE_NAMES)
    return {"variable_names": names}


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


def distill_block(
    block: Union[nn.Module, Callable],
    inputs: torch.Tensor,
    *,
    block_name: str,
    parent_model: Optional[nn.Module] = None,
    sr_params: Optional[Dict[str, Any]] = None,
    max_output_dim: Optional[int] = None,
    variable_names: Optional[List[str]] = None,
    slime: bool = False,
    slime_params: Optional[Dict[str, Any]] = None,
    save_path: Optional[Path] = None,
    resume_pt: Optional[Path] = None,
):
    """
    Run SymTorch distill on a block. For multi-output modules, optionally limit dims.
    Pass ``resume_pt`` to continue from a partially distilled SymbolicModel.
    """
    SymbolicModel = require_symtorch()
    if isinstance(block, nn.Module):
        block = block.cpu()
    fit_params = build_fit_params(variable_names)
    inp = inputs.detach().cpu().float()

    sym: Any
    if resume_pt is not None and resume_pt.is_file():
        logger.info("Resuming symbolic block from %s", resume_pt)
        sym = load_symbolic_module(resume_pt)
    else:
        sym = SymbolicModel(block, block_name=block_name)

    _ = parent_model  # GNN teachers: always distill block I/O directly, not via parent hooks

    n_dims = int(max_output_dim) if max_output_dim is not None else None
    if n_dims is not None:
        reg = sym.SLIME_pysr_regressor if slime else sym.pysr_regressor
        start = max(reg.keys()) + 1 if reg else 0
        for dim in range(start, n_dims):
            logger.info("Distilling %s output_dim=%d", block_name, dim)
            sym.distill(
                inp,
                output_dim=dim,
                parent_model=None,
                sr_params=sr_params,
                fit_params=fit_params,
                SLIME=slime,
                slime_params=slime_params,
                save_path=str(save_path) if save_path else None,
            )
    else:
        sym.distill(
            inp,
            parent_model=None,
            sr_params=sr_params,
            fit_params=fit_params,
            SLIME=slime,
            slime_params=slime_params,
            save_path=str(save_path) if save_path else None,
        )

    sym.switch_to_symbolic(SLIME=slime)
    return sym


def distill_block_on_numpy_io(
    block: Callable,
    x_np: np.ndarray,
    *,
    block_name: str,
    sr_params: Optional[Dict[str, Any]] = None,
    variable_names: Optional[List[str]] = None,
    slime: bool = False,
    slime_params: Optional[Dict[str, Any]] = None,
):
    """Model-agnostic distill when inputs are not parent-model graph features."""
    SymbolicModel = require_symtorch()
    sym = SymbolicModel(block, block_name=block_name)
    fit_params = {}
    if variable_names:
        fit_params["variable_names"] = variable_names
    sym.distill(
        x_np,
        sr_params=sr_params,
        fit_params=fit_params or None,
        SLIME=slime,
        slime_params=slime_params,
    )
    sym.switch_to_symbolic(SLIME=slime)
    return sym


def export_equations_json(sym, out_path: Path, *, slime: bool = False) -> Dict[str, Any]:
    """Serialize equation strings per output dimension."""
    reg = sym.SLIME_pysr_regressor if slime else sym.pysr_regressor
    payload: Dict[str, Any] = {"block_name": sym.block_name, "slime": slime, "equations": {}}
    for dim, model in sorted(reg.items(), key=lambda kv: kv[0]):
        payload["equations"][str(dim)] = _equation_string(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def save_symbolic_module(sym, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    json_path = path.with_name(path.stem + ".json")
    export_equations_json(sym, json_path)
    try:
        import dill  # noqa: WPS433

        with path.open("wb") as f:
            dill.dump(sym, f)
    except Exception as exc:
        logger.warning("Could not pickle %s (%s). Equations saved to %s", path, exc, json_path)


def load_symbolic_module(path: Path):
    import dill  # noqa: WPS433

    if not path.is_file():
        alt = path.with_suffix(".json")
        raise FileNotFoundError(
            f"Missing symbolic checkpoint: {path}\n"
            f"Re-run distillation or provide equations-only file: {alt}"
        )
    with path.open("rb") as f:
        return dill.load(f)


def try_load_symbolic_module(path: Path):
    if path.is_file():
        return load_symbolic_module(path)
    return None


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


def build_slime_params(x0: np.ndarray, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = dict(DEFAULT_SLIME_PARAMS)
    params["x"] = np.asarray(x0, dtype=np.float32)
    if overrides:
        params.update(overrides)
    return params
