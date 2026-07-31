"""Evaluation metrics aligned with r-gatDouble training scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


def mae_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    return F.l1_loss(pred[mask], target[mask]).item()


def evaluate_predictions(
    pred_ys: torch.Tensor,
    pred_fs: torch.Tensor,
    ys: torch.Tensor,
    fs: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
) -> Dict[str, float]:
    return {
        "train_mae_ys": mae_masked(pred_ys, ys, train_mask),
        "train_mae_fs": mae_masked(pred_fs, fs, train_mask),
        "val_mae_ys": mae_masked(pred_ys, ys, val_mask),
        "val_mae_fs": mae_masked(pred_fs, fs, val_mask),
    }


def _to_numpy(a) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy().astype(np.float64).reshape(-1)
    return np.asarray(a, dtype=np.float64).reshape(-1)


def _to_mask(m) -> np.ndarray:
    if isinstance(m, torch.Tensor):
        return m.detach().cpu().numpy().astype(bool).reshape(-1)
    return np.asarray(m, dtype=bool).reshape(-1)


def mae_np(pred, target, mask=None) -> float:
    p = _to_numpy(pred)
    t = _to_numpy(target)
    if mask is not None:
        m = _to_mask(mask)
        if int(m.sum()) == 0:
            return float("nan")
        p, t = p[m], t[m]
    return float(np.mean(np.abs(p - t)))


def rmse_np(pred, target, mask=None) -> float:
    p = _to_numpy(pred)
    t = _to_numpy(target)
    if mask is not None:
        m = _to_mask(mask)
        if int(m.sum()) == 0:
            return float("nan")
        p, t = p[m], t[m]
    return float(np.sqrt(np.mean((p - t) ** 2)))


def r2_np(pred, target, mask=None) -> float:
    p = _to_numpy(pred)
    t = _to_numpy(target)
    if mask is not None:
        m = _to_mask(mask)
        if int(m.sum()) == 0:
            return float("nan")
        p, t = p[m], t[m]
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def split_mae_bundle(
    pred,
    target,
    train_mask,
    val_mask,
    *,
    prefix: str,
) -> Dict[str, float]:
    return {
        f"train_mae_{prefix}": mae_np(pred, target, train_mask),
        f"val_mae_{prefix}": mae_np(pred, target, val_mask),
        f"train_rmse_{prefix}": rmse_np(pred, target, train_mask),
        f"val_rmse_{prefix}": rmse_np(pred, target, val_mask),
        f"train_r2_{prefix}": r2_np(pred, target, train_mask),
        f"val_r2_{prefix}": r2_np(pred, target, val_mask),
    }


def distillation_metrics(
    *,
    teacher,
    linear,
    residual_true,
    residual_pred: Optional[Any] = None,
    combined: Optional[Any] = None,
    label: Optional[Any] = None,
    train_mask,
    val_mask,
    head: str,
) -> Dict[str, float]:
    """Layered MAE for one prediction head."""
    out: Dict[str, float] = {}
    out.update(split_mae_bundle(linear, teacher, train_mask, val_mask, prefix=f"linear_vs_teacher_{head}"))
    out.update(
        split_mae_bundle(
            np.zeros_like(_to_numpy(residual_true)),
            residual_true,
            train_mask,
            val_mask,
            prefix=f"zero_residual_{head}",
        )
    )
    if residual_pred is not None:
        out.update(
            split_mae_bundle(
                residual_pred, residual_true, train_mask, val_mask, prefix=f"sym_residual_{head}"
            )
        )
    if combined is not None:
        out.update(
            split_mae_bundle(combined, teacher, train_mask, val_mask, prefix=f"combined_vs_teacher_{head}")
        )
        out["distillation_mae_train_" + head] = mae_np(combined, teacher, train_mask)
        out["distillation_mae_val_" + head] = mae_np(combined, teacher, val_mask)
        if label is not None:
            out.update(split_mae_bundle(combined, label, train_mask, val_mask, prefix=f"combined_vs_label_{head}"))
            out.update(split_mae_bundle(teacher, label, train_mask, val_mask, prefix=f"teacher_vs_label_{head}"))
            out[f"label_mae_delta_val_{head}"] = (
                mae_np(combined, label, val_mask) - mae_np(teacher, label, val_mask)
            )
    # linear explanation style R2: 1 - SSE_linear / SST_teacher
    for split_name, mask in (("train", train_mask), ("val", val_mask)):
        m = _to_mask(mask)
        if int(m.sum()) == 0:
            out[f"{split_name}_r2_linear_explains_teacher_{head}"] = float("nan")
            continue
        t = _to_numpy(teacher)[m]
        l = _to_numpy(linear)[m]
        sse = float(np.sum((t - l) ** 2))
        sst = float(np.sum((t - np.mean(t)) ** 2))
        out[f"{split_name}_r2_linear_explains_teacher_{head}"] = (
            float("nan") if sst <= 0 else 1.0 - sse / sst
        )
        r = _to_numpy(residual_true)[m]
        out[f"{split_name}_residual_std_{head}"] = float(np.std(r))
        out[f"{split_name}_cov_linear_residual_{head}"] = float(np.cov(l, r)[0, 1]) if len(l) > 1 else float("nan")
    return out


def save_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def write_summary_md(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join([f"# {title}", ""] + lines)
    path.write_text(body + "\n", encoding="utf-8")


def assert_finite(name: str, arr) -> None:
    a = _to_numpy(arr)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains NaN/Inf")
