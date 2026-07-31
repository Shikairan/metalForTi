"""Unit tests for CSV/graph alignment helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.alignment import (  # noqa: E402
    assert_masks,
    duplicate_input_groups,
    load_csv_model_matrix,
    load_aligned_tabular,
)


class _Sample:
    def __init__(self, x):
        self.x = x
        self.num_nodes = int(x.shape[0])


class _Graph(dict):
    pass


def test_load_csv_and_align(tmp_path: Path):
    X = np.random.randn(12, 30).astype(np.float64)
    cols = (
        [f"element_{i}" for i in range(10)]
        + ["testenv_0", "testenv_1"]
        + [f"coldway_{i}" for i in range(18)]
    )
    df = pd.DataFrame(X, columns=cols)
    df["YS"] = 1.0
    csv_path = tmp_path / "t.csv"
    df.to_csv(csv_path, index=False)

    loaded, _ = load_csv_model_matrix(csv_path)
    assert loaded.shape == (12, 30)

    g = _Graph()
    g["sample"] = _Sample(torch.from_numpy(X.astype(np.float32)))
    aligned = load_aligned_tabular(csv_path, g)
    assert aligned.n == 12
    assert aligned.sample_id.tolist() == list(range(12))


def test_mask_and_duplicates():
    n = 10
    tm = np.zeros(n, dtype=bool)
    tm[:6] = True
    vm = np.zeros(n, dtype=bool)
    vm[6:] = True
    assert_masks(tm, vm, n)

    bad = tm.copy()
    bad[6] = True
    try:
        assert_masks(bad, vm, n)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    X = np.vstack([np.ones((3, 30)), np.zeros((2, 30))])
    _, counts, summary = duplicate_input_groups(X)
    assert summary["n_unique"] == 2
    assert summary["max_group_size"] == 3
