"""Per-round train/val mask snapshots for loop_train_swap_* scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_round_mask_snapshot(
    history_dir: Path,
    round_idx: int,
    train_during: torch.Tensor,
    val_during: torch.Tensor,
    train_after: torch.Tensor,
    val_after: torch.Tensor,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write one .pt file for this outer-loop round.

    - ``*_during``: masks used for all inner epochs in this round (before curate/swap).
    - ``*_after``: masks written to ``train_mask.pt`` / ``val_mask.pt`` after the round.

    Inactive (curated-out) nodes satisfy ``~(train_mask | val_mask)``.
    """
    history_dir = Path(history_dir).resolve()
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / f"round_{int(round_idx):06d}.pt"
    payload: Dict[str, Any] = {
        "round": int(round_idx),
        "train_mask_during": train_during.bool().clone(),
        "val_mask_during": val_during.bool().clone(),
        "train_mask_after": train_after.bool().clone(),
        "val_mask_after": val_after.bool().clone(),
    }
    if meta:
        payload["meta"] = meta
    torch.save(payload, path)
    # Human-readable companion (small): index lists only
    side = path.with_suffix(".json")
    n = int(train_during.numel())
    inactive_during = ~(payload["train_mask_during"] | payload["val_mask_during"])
    inactive_after = ~(payload["train_mask_after"] | payload["val_mask_after"])
    summary = {
        "round": int(round_idx),
        "num_nodes": n,
        "train_count_during": int(payload["train_mask_during"].sum().item()),
        "val_count_during": int(payload["val_mask_during"].sum().item()),
        "inactive_count_during": int(inactive_during.sum().item()),
        "train_count_after": int(payload["train_mask_after"].sum().item()),
        "val_count_after": int(payload["val_mask_after"].sum().item()),
        "inactive_count_after": int(inactive_after.sum().item()),
        "meta": meta or {},
    }
    with side.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return path


def load_round_snapshot(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


def indices_for_split(train_mask: torch.Tensor, val_mask: torch.Tensor) -> Dict[str, list[int]]:
    """Return node indices for train / val / inactive from boolean masks (1-D)."""
    tm = train_mask.bool().flatten()
    vm = val_mask.bool().flatten()
    inactive = ~(tm | vm)
    return {
        "train_idx": torch.where(tm)[0].tolist(),
        "val_idx": torch.where(vm)[0].tolist(),
        "inactive_idx": torch.where(inactive)[0].tolist(),
    }
