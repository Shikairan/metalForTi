"""Load and validate liner run artifacts for lowExp / comb."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .manifest import assert_liner_run_ready, load_manifest


@dataclass
class LinerPayload:
    run_dir: Path
    manifest: Dict[str, Any]
    X_all: np.ndarray
    teacher_head0: np.ndarray
    teacher_fs: np.ndarray
    linear_head0: np.ndarray
    linear_fs: np.ndarray
    residual_head0: np.ndarray
    residual_fs: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    sample_id: np.ndarray
    feature_names: list
    head0_name: str
    head1_name: str
    label_head0: np.ndarray | None = None
    label_fs: np.ndarray | None = None


def _np1(t) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy().astype(np.float64).reshape(-1)
    return np.asarray(t, dtype=np.float64).reshape(-1)


def _np2(t) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy().astype(np.float64)
    return np.asarray(t, dtype=np.float64)


def load_and_validate_liner_run(liner_run: Path) -> LinerPayload:
    liner_run = assert_liner_run_ready(Path(liner_run))
    manifest = load_manifest(liner_run / "manifest.json")
    blob = torch.load(liner_run / "residual_targets.pt", map_location="cpu", weights_only=False)

    required = [
        "X_all",
        "teacher_head0",
        "teacher_fs",
        "linear_head0",
        "linear_fs",
        "residual_head0",
        "residual_fs",
        "train_mask",
        "val_mask",
        "sample_id",
        "feature_names",
        "head0_name",
        "head1_name",
        "manifest_fingerprint",
    ]
    for k in required:
        if k not in blob:
            raise ValueError(f"residual_targets.pt missing key: {k}")

    if blob["manifest_fingerprint"] != manifest.get("fingerprint"):
        raise ValueError(
            "residual_targets.pt fingerprint != manifest.json fingerprint"
        )
    if list(blob["feature_names"]) != list(manifest.get("feature_names", [])):
        raise ValueError("feature_names mismatch between residual_targets and manifest")
    if blob["head0_name"] != manifest.get("head0_name") or blob["head1_name"] != manifest.get(
        "head1_name"
    ):
        raise ValueError("head names mismatch between residual_targets and manifest")

    X_all = _np2(blob["X_all"])
    if X_all.ndim != 2 or X_all.shape[1] != 30:
        raise ValueError(f"X_all expected [N,30], got {X_all.shape}")
    n = X_all.shape[0]
    for key in (
        "teacher_head0",
        "teacher_fs",
        "linear_head0",
        "linear_fs",
        "residual_head0",
        "residual_fs",
        "sample_id",
    ):
        arr = _np1(blob[key]) if key != "sample_id" else np.asarray(blob[key]).reshape(-1)
        if arr.shape[0] != n:
            raise ValueError(f"{key} length {arr.shape[0]} != N={n}")

    train_mask = np.asarray(blob["train_mask"]).reshape(-1).astype(bool)
    val_mask = np.asarray(blob["val_mask"]).reshape(-1).astype(bool)
    if train_mask.shape[0] != n or val_mask.shape[0] != n:
        raise ValueError("mask length mismatch vs X_all")

    # residual identity
    r0 = _np1(blob["residual_head0"])
    rf = _np1(blob["residual_fs"])
    if not np.allclose(
        r0, _np1(blob["teacher_head0"]) - _np1(blob["linear_head0"]), rtol=1e-10, atol=1e-10
    ):
        raise ValueError("residual_head0 != teacher_head0 - linear_head0")
    if not np.allclose(
        rf, _np1(blob["teacher_fs"]) - _np1(blob["linear_fs"]), rtol=1e-10, atol=1e-10
    ):
        raise ValueError("residual_fs != teacher_fs - linear_fs")

    label_h0 = _np1(blob["label_head0"]) if "label_head0" in blob else None
    label_fs = _np1(blob["label_fs"]) if "label_fs" in blob else None

    return LinerPayload(
        run_dir=liner_run.resolve(),
        manifest=manifest,
        X_all=X_all,
        teacher_head0=_np1(blob["teacher_head0"]),
        teacher_fs=_np1(blob["teacher_fs"]),
        linear_head0=_np1(blob["linear_head0"]),
        linear_fs=_np1(blob["linear_fs"]),
        residual_head0=r0,
        residual_fs=rf,
        train_mask=train_mask,
        val_mask=val_mask,
        sample_id=np.asarray(blob["sample_id"], dtype=np.int64).reshape(-1),
        feature_names=list(blob["feature_names"]),
        head0_name=str(blob["head0_name"]),
        head1_name=str(blob["head1_name"]),
        label_head0=label_h0,
        label_fs=label_fs,
    )
