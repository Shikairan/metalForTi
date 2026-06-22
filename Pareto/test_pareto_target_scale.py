"""Pareto 目标预处理：仅 data1123 输入格式。"""
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
    resolve_user_targets,
)


def test_fs_20_and_0_2_not_equivalent() -> None:
    """FS=20 与 FS=0.2 在 data1123 语义下不等价，模型量纲不同。"""
    r20 = resolve_user_targets(1201, 20, DEFAULT_YS_MEAN, DEFAULT_FS_MEAN)
    r02 = resolve_user_targets(1201, 0.2, DEFAULT_YS_MEAN, DEFAULT_FS_MEAN)
    assert r20.fs_data1123 == 20.0
    assert r02.fs_data1123 == 0.2
    assert r20.fs_model != r02.fs_model
    assert r20.fs_model > r02.fs_model


def test_log_display_same_scale() -> None:
    """模型标签还原后与 data1123 目标同量级。"""
    resolved = resolve_user_targets(1201, 0.2, DEFAULT_YS_MEAN, DEFAULT_FS_MEAN)
    ys_phys, fs_phys = denormalize_targets_to_data1123(
        2.0218, 0.5402, ys_mean=DEFAULT_YS_MEAN, fs_mean=DEFAULT_FS_MEAN
    )
    assert abs(resolved.ys_model - 1.24355) < 0.001
    assert abs(ys_phys - 1952.6) < 1.0
    assert abs(fs_phys - 0.152) < 0.01
    f1 = abs(ys_phys - resolved.ys_physical)
    assert f1 < 800


if __name__ == "__main__":
    test_fs_20_and_0_2_not_equivalent()
    test_log_display_same_scale()
    print("test_pareto_target_scale: OK")
