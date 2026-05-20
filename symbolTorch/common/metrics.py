"""Evaluation metrics aligned with r-gatDouble training scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F


def inverse_fs(y: torch.Tensor) -> torch.Tensor:
    return torch.exp(y)


def rel_pct_fs(pred: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    pred_raw = inverse_fs(pred)
    tgt_raw = inverse_fs(target)
    return (pred_raw - tgt_raw).abs() / tgt_raw.abs().clamp_min(eps) * 100.0


def rel_pct_ys(pred: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    return (pred - target).abs() / target.abs().clamp_min(eps) * 100.0


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


def worst_val_fs_node_idx(
    pred_fs: torch.Tensor,
    fs: torch.Tensor,
    val_mask: torch.Tensor,
) -> int:
    rel = rel_pct_fs(pred_fs, fs)
    v_idx = torch.where(val_mask)[0]
    worst_pos = int(torch.argmax(rel[v_idx]).item())
    return int(v_idx[worst_pos].item())


def save_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def write_summary_md(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join([f"# {title}", ""] + lines)
    path.write_text(body + "\n", encoding="utf-8")
