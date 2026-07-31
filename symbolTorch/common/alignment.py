"""CSV ↔ graph feature alignment and mask checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .constants import FEATURE_NAMES

CSV_MODEL_COL_PREFIXES = (
    [f"element_{i}" for i in range(10)]
    + ["testenv_0", "testenv_1"]
    + [f"coldway_{i}" for i in range(18)]
)


@dataclass(frozen=True)
class AlignedTabular:
    X_all: np.ndarray  # float64 [N, 30]
    sample_id: np.ndarray  # int64 [N]
    feature_names: Tuple[str, ...]
    csv_path: Path
    n: int


def load_csv_model_matrix(csv_path: Path) -> Tuple[np.ndarray, list[str]]:
    """Load first 30 model-space columns from datagnnUts-style CSV."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.shape[1] < 30:
        raise ValueError(f"CSV needs >= 30 columns, got {df.shape[1]}: {csv_path}")

    cols = list(df.columns[:30])
    if all(c in df.columns for c in CSV_MODEL_COL_PREFIXES):
        cols = list(CSV_MODEL_COL_PREFIXES)
        X = df[cols].to_numpy(dtype=np.float64)
    else:
        X = df.iloc[:, :30].to_numpy(dtype=np.float64)

    if X.ndim != 2 or X.shape[1] != 30:
        raise ValueError(f"Expected [N, 30] model matrix, got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite values in CSV model matrix: {csv_path}")
    return X, cols


def graph_sample_x(graph) -> np.ndarray:
    x = graph["sample"].x
    if hasattr(x, "detach"):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    return np.asarray(arr, dtype=np.float64)


def assert_csv_matches_graph(
    X_csv: np.ndarray,
    graph,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    X_graph = graph_sample_x(graph)
    n_sample = (
        int(graph["sample"].num_nodes)
        if hasattr(graph["sample"], "num_nodes")
        else X_graph.shape[0]
    )
    if X_csv.shape[0] != n_sample:
        raise ValueError(
            f"CSV rows ({X_csv.shape[0]}) != graph sample nodes ({n_sample})"
        )
    if X_graph.shape != X_csv.shape:
        raise ValueError(f"graph.x shape {X_graph.shape} != CSV shape {X_csv.shape}")
    if not np.isfinite(X_graph).all():
        raise ValueError("Non-finite values in graph['sample'].x")
    if not np.allclose(X_csv, X_graph, rtol=rtol, atol=atol):
        diff = np.max(np.abs(X_csv - X_graph))
        raise ValueError(
            f"CSV vs graph['sample'].x mismatch (max abs diff={diff}). "
            "Refuse to generate formulas."
        )


def assert_masks(
    train_mask: np.ndarray | torch.Tensor,
    val_mask: np.ndarray | torch.Tensor,
    n: int,
    *,
    allow_empty_val: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    tm = np.asarray(train_mask, dtype=bool).reshape(-1)
    vm = np.asarray(val_mask, dtype=bool).reshape(-1)
    if tm.shape[0] != n or vm.shape[0] != n:
        raise ValueError(
            f"mask length mismatch: train={tm.shape[0]} val={vm.shape[0]} n={n}"
        )
    if int(tm.sum()) == 0:
        raise ValueError("train_mask is empty")
    if np.any(tm & vm):
        if not np.array_equal(tm, vm):
            raise ValueError(
                "train_mask and val_mask overlap but differ; "
                "only identical full-set masks (e.g. utsFsAll) are allowed"
            )
        # utsFsAll: train == val == all samples; metrics are full-set, not held-out val
        return tm, vm
    if int(vm.sum()) == 0 and not allow_empty_val:
        raise ValueError(
            "val_mask is empty; pass allow_empty_val=True for full-train mode"
        )
    return tm, vm


def is_full_train_masks(train_mask, val_mask) -> bool:
    """True when train and val masks are identical (no held-out split)."""
    tm = np.asarray(train_mask, dtype=bool).reshape(-1)
    vm = np.asarray(val_mask, dtype=bool).reshape(-1)
    return bool(np.array_equal(tm, vm))


def load_aligned_tabular(
    csv_path: Path,
    graph,
    *,
    feature_names: Optional[Sequence[str]] = None,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> AlignedTabular:
    X, _cols = load_csv_model_matrix(csv_path)
    assert_csv_matches_graph(X, graph, rtol=rtol, atol=atol)
    names = tuple(feature_names) if feature_names is not None else tuple(FEATURE_NAMES)
    if len(names) != 30:
        raise ValueError(f"feature_names must have length 30, got {len(names)}")
    n = int(X.shape[0])
    sample_id = np.arange(n, dtype=np.int64)
    return AlignedTabular(
        X_all=X,
        sample_id=sample_id,
        feature_names=names,
        csv_path=Path(csv_path).resolve(),
        n=n,
    )


def duplicate_input_groups(
    X: np.ndarray,
    *,
    decimals: int = 8,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Group identical (rounded) rows; return group_id, counts, summary dict."""
    Xr = np.round(np.asarray(X, dtype=np.float64), decimals=decimals)
    _, inverse, counts = np.unique(Xr, axis=0, return_inverse=True, return_counts=True)
    n_dup_groups = int(np.sum(counts > 1))
    summary = {
        "n_rows": int(X.shape[0]),
        "n_unique": int(len(counts)),
        "n_duplicate_groups": n_dup_groups,
        "max_group_size": int(counts.max()) if len(counts) else 0,
    }
    return inverse.astype(np.int64), counts.astype(np.int64), summary
