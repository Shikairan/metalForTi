"""Per-sample prediction and error export for comb runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd


def _as1d(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).reshape(-1)


def export_per_sample_errors(
    out_dir: Path,
    *,
    sample_id: np.ndarray,
    head0_name: str,
    head1_name: str,
    teacher_head0: np.ndarray,
    teacher_fs: np.ndarray,
    linear_head0: np.ndarray,
    linear_fs: np.ndarray,
    residual_true_head0: np.ndarray,
    residual_true_fs: np.ndarray,
    residual_pred_head0: np.ndarray,
    residual_pred_fs: np.ndarray,
    combined_head0: np.ndarray,
    combined_fs: np.ndarray,
    label_head0: Optional[np.ndarray] = None,
    label_fs: Optional[np.ndarray] = None,
    full_train_mode: bool = False,
) -> Dict[str, Any]:
    """Write per_sample_errors.csv and per_sample_errors.json."""
    out_dir = Path(out_dir)
    n = int(sample_id.shape[0])
    h0, h1 = head0_name, head1_name

    rows: Dict[str, Any] = {
        "sample_id": np.asarray(sample_id, dtype=np.int64).reshape(-1),
        f"teacher_{h0}": _as1d(teacher_head0),
        f"linear_{h0}": _as1d(linear_head0),
        f"residual_true_{h0}": _as1d(residual_true_head0),
        f"residual_pred_{h0}": _as1d(residual_pred_head0),
        f"combined_{h0}": _as1d(combined_head0),
        f"abs_err_linear_vs_teacher_{h0}": np.abs(_as1d(linear_head0) - _as1d(teacher_head0)),
        f"abs_err_residual_pred_vs_true_{h0}": np.abs(
            _as1d(residual_pred_head0) - _as1d(residual_true_head0)
        ),
        f"abs_err_combined_vs_teacher_{h0}": np.abs(_as1d(combined_head0) - _as1d(teacher_head0)),
        f"teacher_{h1}": _as1d(teacher_fs),
        f"linear_{h1}": _as1d(linear_fs),
        f"residual_true_{h1}": _as1d(residual_true_fs),
        f"residual_pred_{h1}": _as1d(residual_pred_fs),
        f"combined_{h1}": _as1d(combined_fs),
        f"abs_err_linear_vs_teacher_{h1}": np.abs(_as1d(linear_fs) - _as1d(teacher_fs)),
        f"abs_err_residual_pred_vs_true_{h1}": np.abs(
            _as1d(residual_pred_fs) - _as1d(residual_true_fs)
        ),
        f"abs_err_combined_vs_teacher_{h1}": np.abs(_as1d(combined_fs) - _as1d(teacher_fs)),
    }
    if label_head0 is not None:
        rows[f"label_{h0}"] = _as1d(label_head0)
        rows[f"abs_err_combined_vs_label_{h0}"] = np.abs(
            _as1d(combined_head0) - _as1d(label_head0)
        )
        rows[f"abs_err_teacher_vs_label_{h0}"] = np.abs(
            _as1d(teacher_head0) - _as1d(label_head0)
        )
    if label_fs is not None:
        rows[f"label_{h1}"] = _as1d(label_fs)
        rows[f"abs_err_combined_vs_label_{h1}"] = np.abs(
            _as1d(combined_fs) - _as1d(label_fs)
        )
        rows[f"abs_err_teacher_vs_label_{h1}"] = np.abs(
            _as1d(teacher_fs) - _as1d(label_fs)
        )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "per_sample_errors.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "n_samples": n,
        "head0_name": h0,
        "head1_name": h1,
        "full_train_mode": bool(full_train_mode),
        "mean_abs_err_linear_vs_teacher": {
            h0: float(df[f"abs_err_linear_vs_teacher_{h0}"].mean()),
            h1: float(df[f"abs_err_linear_vs_teacher_{h1}"].mean()),
        },
        "mean_abs_err_combined_vs_teacher": {
            h0: float(df[f"abs_err_combined_vs_teacher_{h0}"].mean()),
            h1: float(df[f"abs_err_combined_vs_teacher_{h1}"].mean()),
        },
        "mean_abs_err_residual_pred_vs_true": {
            h0: float(df[f"abs_err_residual_pred_vs_true_{h0}"].mean()),
            h1: float(df[f"abs_err_residual_pred_vs_true_{h1}"].mean()),
        },
    }
    json_path = out_dir / "per_sample_errors.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {"csv": str(csv_path), "json": str(json_path), "summary": summary}
