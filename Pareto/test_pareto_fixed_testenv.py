"""固定 tem/sr（data1123）锁定 testenv 维。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from Pareto.ga_testenv_lock import physical_testenv_to_z_np, resolve_fixed_testenv
from preprocess.preprocess_datagnn_repro import DEFAULT_TESTENV_MEAN, DEFAULT_TESTENV_STD


class _Bounds:
    def __init__(self, lo, hi):
        self.testenv_lower = np.array(lo, dtype=np.float32)
        self.testenv_upper = np.array(hi, dtype=np.float32)


def test_resolve_both_none() -> None:
    b = _Bounds([-2.0, -2.0], [2.0, 2.0])
    assert resolve_fixed_testenv(None, None, DEFAULT_TESTENV_MEAN, DEFAULT_TESTENV_STD, b) is None


def test_resolve_only_one_raises() -> None:
    b = _Bounds([-2.0, -2.0], [2.0, 2.0])
    try:
        resolve_fixed_testenv(25.0, None, DEFAULT_TESTENV_MEAN, DEFAULT_TESTENV_STD, b)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "同时提供" in str(e)


def test_physical_to_z() -> None:
    z = physical_testenv_to_z_np(25.0, 0.001, DEFAULT_TESTENV_MEAN, DEFAULT_TESTENV_STD)
    assert z.shape == (2,)
    assert np.isfinite(z).all()


def test_operators_lock_testenv() -> None:
    import torch
    from grd.feature_layout import INPUT_DIM, TESTENV_SLICE
    from Pareto.ga_operators import GAConfig, mutate_genome, segmented_crossover
    from Pareto.ga_testenv_lock import FixedTestenvContext

    fixed_z = torch.tensor([0.5, -0.3], dtype=torch.float32)
    lock = FixedTestenvContext(25.0, 0.001, fixed_z.numpy())
    cfg = GAConfig(fixed_testenv=lock, p_mut=1.0)
    rng = torch.Generator().manual_seed(0)
    p1 = torch.randn(INPUT_DIM)
    p2 = torch.randn(INPUT_DIM)
    p1[TESTENV_SLICE] = torch.tensor([9.0, 9.0])
    p2[TESTENV_SLICE] = torch.tensor([-9.0, -9.0])

    c1, c2 = segmented_crossover(p1, p2, rng, lock_testenv=True)
    assert torch.allclose(c1[TESTENV_SLICE], p1[TESTENV_SLICE])
    assert torch.allclose(c2[TESTENV_SLICE], p2[TESTENV_SLICE])

    g = p1.clone()
    for _ in range(20):
        g = mutate_genome(g, torch.stack([p1, p2]), rng, cfg)
    assert torch.allclose(g[TESTENV_SLICE], p1[TESTENV_SLICE])


if __name__ == "__main__":
    test_resolve_both_none()
    test_resolve_only_one_raises()
    test_physical_to_z()
    try:
        import torch  # noqa: F401

        test_operators_lock_testenv()
    except ImportError:
        print("(skip operator tests: no torch)")
    print("test_pareto_fixed_testenv: OK")
