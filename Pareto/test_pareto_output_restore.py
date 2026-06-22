"""Pareto 输出物理量纲还原单元测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess.preprocess_datagnn_repro import (
    DEFAULT_FS_MEAN,
    DEFAULT_TESTENV_MEAN,
    DEFAULT_YS_MEAN,
    denormalize_genome_to_data1123,
    denormalize_targets_to_data1123,
    inverse_coldway_flat_18,
    inverse_testenv_z,
    load_testenv_stats_np,
    normalize_targets_from_data1123,
)


def test_inverse_testenv_roundtrip() -> None:
    te_mean, te_std = load_testenv_stats_np(ROOT / "gnnDir" / "datacsv" / "testenv_stats.csv")
    tem, fcr = 25.0, 0.001
    z = np.array([(tem - te_mean[0]) / te_std[0], (fcr - te_mean[1]) / te_std[1]])
    out = inverse_testenv_z(z, te_mean=te_mean, te_std=te_std)
    assert abs(out[0] - tem) < 1e-6 and abs(out[1] - fcr) < 1e-9


def test_inverse_coldway_shape() -> None:
    cw = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8])
    out = inverse_coldway_flat_18(cw)
    assert out.shape == (18,)


def test_denormalize_genome_30d() -> None:
    te_mean, te_std = load_testenv_stats_np(ROOT / "gnnDir" / "datacsv" / "testenv_stats.csv")
    g = np.zeros(30)
    out = denormalize_genome_to_data1123(g, te_mean=te_mean, te_std=te_std)
    assert out.shape == (30,)
    assert abs(out[10] - te_mean[0]) < 1e-4
    assert abs(out[11] - te_mean[1]) < 1e-4


def test_ys_fs_denorm_roundtrip() -> None:
    ys_model, fs_model = normalize_targets_from_data1123(1014.8, 0.147, ys_mean=DEFAULT_YS_MEAN, fs_mean=DEFAULT_FS_MEAN)
    ys_phys, fs_phys = denormalize_targets_to_data1123(
        ys_model, fs_model, ys_mean=DEFAULT_YS_MEAN, fs_mean=DEFAULT_FS_MEAN
    )
    assert abs(ys_phys - 1014.8) < 0.01
    assert abs(fs_phys - 0.147) < 1e-6


if __name__ == "__main__":
    test_inverse_testenv_roundtrip()
    test_inverse_coldway_shape()
    test_denormalize_genome_30d()
    test_ys_fs_denorm_roundtrip()
    print("test_pareto_output_restore: OK")
