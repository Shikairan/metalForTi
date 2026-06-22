"""Pareto 目标预处理：原始输入 → 模型量纲 → 逆变换展示。"""
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


def test_dataori2_fs_auto_detect() -> None:
    """FS=20（dataOri2）应得到与 FS=0.2（data1123）相同的模型量纲。"""
    r20 = resolve_user_targets(1201, 20, DEFAULT_YS_MEAN, DEFAULT_FS_MEAN)
    r02 = resolve_user_targets(1201, 0.2, DEFAULT_YS_MEAN, DEFAULT_FS_MEAN)
    assert r20.fs_input_scale == "dataori2"
    assert r02.fs_input_scale == "data1123"
    assert abs(r20.fs_model - r02.fs_model) < 1e-9
    assert abs(r20.fs_data1123 - 0.2) < 1e-9


def test_log_display_same_scale_as_user_example() -> None:
    """用户示例：标签 YS≈2.02（模型）应还原为 ~1950 MPa，与目标 1201 同量级。"""
    resolved = resolve_user_targets(1201, 20, DEFAULT_YS_MEAN, DEFAULT_FS_MEAN)
    ys_phys, fs_phys = denormalize_targets_to_data1123(2.0218, 0.5402, ys_mean=DEFAULT_YS_MEAN, fs_mean=DEFAULT_FS_MEAN)
    assert abs(resolved.ys_model - 1.24355) < 0.001
    assert abs(ys_phys - 1952.6) < 1.0
    assert abs(fs_phys - 0.152) < 0.01
    f1 = abs(ys_phys - resolved.ys_physical)
    assert f1 < 800  # 同量级，而非 1e3 量级错误


if __name__ == "__main__":
    test_dataori2_fs_auto_detect()
    test_log_display_same_scale_as_user_example()
    print("test_pareto_target_scale: OK")
