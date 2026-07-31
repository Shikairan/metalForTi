"""Run manifest + fingerprint helpers for liner / lowExp / comb."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


SUCCESS_MARKER = "_SUCCESS"


def _abspath(p: Path | str | None) -> Optional[str]:
    if p is None:
        return None
    return str(Path(p).resolve())


def mask_hash(mask: np.ndarray) -> str:
    m = np.asarray(mask, dtype=np.uint8).reshape(-1)
    return hashlib.sha256(m.tobytes()).hexdigest()[:16]


def feature_order_hash(names: Sequence[str]) -> str:
    blob = "\0".join(names).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def compute_fingerprint(fields: Mapping[str, Any]) -> str:
    """Stable fingerprint from selected manifest fields."""
    payload = json.dumps(dict(fields), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def build_liner_manifest(
    *,
    csv: Path | str,
    data_dir: Path | str,
    ckpt: Path | str,
    head0_name: str,
    head1_name: str,
    feature_names: Sequence[str],
    n_samples: int,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    method: str,
    alpha: Optional[float],
    seed: int,
    run_id: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    train_h = mask_hash(train_mask)
    val_h = mask_hash(val_mask)
    feat_h = feature_order_hash(feature_names)
    base = {
        "module": "liner",
        "run_id": run_id,
        "csv": _abspath(csv),
        "data_dir": _abspath(data_dir),
        "ckpt": _abspath(ckpt),
        "head0_name": head0_name,
        "head1_name": head1_name,
        "n_samples": int(n_samples),
        "feature_names": list(feature_names),
        "feature_order_hash": feat_h,
        "train_mask_hash": train_h,
        "val_mask_hash": val_h,
        "n_train": int(np.asarray(train_mask).sum()),
        "n_val": int(np.asarray(val_mask).sum()),
        "method": method,
        "alpha": None if alpha is None else float(alpha),
        "seed": int(seed),
    }
    if extra:
        base.update(extra)
    base["fingerprint"] = compute_fingerprint(
        {
            "csv": base["csv"],
            "data_dir": base["data_dir"],
            "ckpt": base["ckpt"],
            "head0_name": base["head0_name"],
            "head1_name": base["head1_name"],
            "feature_order_hash": feat_h,
            "train_mask_hash": train_h,
            "val_mask_hash": val_h,
            "method": base["method"],
            "alpha": base["alpha"],
            "seed": base["seed"],
        }
    )
    return base


def save_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(manifest), f, indent=2, ensure_ascii=False)


def load_manifest(path: Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_success_marker(out_dir: Path) -> None:
    (Path(out_dir) / SUCCESS_MARKER).write_text("ok\n", encoding="utf-8")


def has_success_marker(out_dir: Path) -> bool:
    return (Path(out_dir) / SUCCESS_MARKER).is_file()


def assert_compatible_manifests(
    liner: Mapping[str, Any],
    other: Mapping[str, Any],
    *,
    require_keys: Sequence[str] = (
        "csv",
        "data_dir",
        "ckpt",
        "head0_name",
        "head1_name",
        "feature_order_hash",
        "train_mask_hash",
        "val_mask_hash",
        "fingerprint",
    ),
) -> None:
    for k in require_keys:
        if k not in liner:
            raise ValueError(f"liner manifest missing key: {k}")
        if k not in other:
            raise ValueError(f"other manifest missing key: {k}")
        if liner[k] != other[k]:
            raise ValueError(
                f"manifest mismatch on '{k}': liner={liner[k]!r} other={other[k]!r}"
            )


def assert_liner_run_ready(liner_run: Path) -> Path:
    liner_run = Path(liner_run)
    if not liner_run.is_dir():
        raise FileNotFoundError(f"liner run not found: {liner_run}")
    if not has_success_marker(liner_run):
        raise FileNotFoundError(f"liner run missing {SUCCESS_MARKER}: {liner_run}")
    residual = liner_run / "residual_targets.pt"
    if not residual.is_file():
        raise FileNotFoundError(f"missing residual_targets.pt under {liner_run}")
    man = liner_run / "manifest.json"
    if not man.is_file():
        raise FileNotFoundError(f"missing manifest.json under {liner_run}")
    return liner_run
