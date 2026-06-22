"""Pareto 标签均值：必须用物理量均值做逆变换。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess.preprocess_datagnn_repro import (
    DEFAULT_FS_MEAN,
    DEFAULT_YS_MEAN,
    denormalize_targets_to_data1123,
    physical_label_means_for_targets,
    resolve_user_targets,
)


def test_physical_means_not_pt_tensor_means() -> None:
    """误用 ys.pt 全表均值≈1 时，逆变换几乎不变（用户日志中的错误）。"""
    ys_wrong, fs_wrong = denormalize_targets_to_data1123(2.0218, 0.5402, ys_mean=1.0, fs_mean=1.0)
    assert abs(ys_wrong - 2.0218) < 1e-4
    assert abs(fs_wrong - 0.005402) < 1e-5

    ys_mean, fs_mean = physical_label_means_for_targets(ROOT / "preprocess" / "data1123.csv")
    assert abs(ys_mean - DEFAULT_YS_MEAN) < 1e-3
    assert abs(fs_mean - DEFAULT_FS_MEAN) < 1e-3
    ys_ok, fs_ok = denormalize_targets_to_data1123(
        2.0218, 0.5402, ys_mean=ys_mean, fs_mean=fs_mean
    )
    assert abs(ys_ok - 1952.6) < 1.0
    assert abs(fs_ok - 0.152) < 0.01


def test_user_log_scenario() -> None:
    """节点 #124 标签 2.0218/0.5402 还原后应与目标 1201/0.2 同量级。"""
    ys_mean, fs_mean = physical_label_means_for_targets(ROOT / "preprocess" / "data1123.csv")
    resolved = resolve_user_targets(1201, 0.2, ys_mean, fs_mean)
    ys_phys, fs_phys = denormalize_targets_to_data1123(2.0218, 0.5402, ys_mean=ys_mean, fs_mean=fs_mean)
    assert abs(ys_phys - resolved.ys_physical) < 800
    assert abs(fs_phys - resolved.fs_data1123) < 0.05


if __name__ == "__main__":
    test_physical_means_not_pt_tensor_means()
    test_user_log_scenario()
    print("test_pareto_label_means: OK")
