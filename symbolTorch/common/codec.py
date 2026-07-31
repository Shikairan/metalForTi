"""Thin codec wrappers around preprocess_datagnn_repro for symbolTorch."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .constants import METAL_FOR_TI_ROOT

_PREPROCESS = METAL_FOR_TI_ROOT / "preprocess"
if str(_PREPROCESS) not in sys.path:
    sys.path.insert(0, str(_PREPROCESS))

import preprocess_datagnn_repro as pre  # noqa: E402


def codec_stats_dict(
    *,
    te_mean: Optional[np.ndarray] = None,
    te_std: Optional[np.ndarray] = None,
    ys_mean: Optional[float] = None,
    fs_mean: Optional[float] = None,
    uts_mean: Optional[float] = None,
) -> Dict[str, Any]:
    te_m = np.asarray(
        te_mean if te_mean is not None else pre.DEFAULT_TESTENV_MEAN, dtype=np.float64
    )
    te_s = np.asarray(
        te_std if te_std is not None else pre.DEFAULT_TESTENV_STD, dtype=np.float64
    )
    return {
        "T_DIV": float(pre.T_DIV),
        "testenv_mean": te_m.tolist(),
        "testenv_std": te_s.tolist(),
        "mean_YS": float(ys_mean if ys_mean is not None else pre.DEFAULT_YS_MEAN),
        "mean_FS_dataOri2": float(fs_mean if fs_mean is not None else pre.DEFAULT_FS_MEAN),
        "mean_UTS": float(uts_mean if uts_mean is not None else pre.DEFAULT_UTS_MEAN),
        "FS_DATA1123_TO_DATAORI2_SCALE": float(pre.FS_DATA1123_TO_DATAORI2_SCALE),
    }


def load_testenv_stats(stats_path: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
    if stats_path is not None and Path(stats_path).is_file():
        return pre.load_testenv_stats_np(Path(stats_path))
    return (
        np.asarray(pre.DEFAULT_TESTENV_MEAN, dtype=np.float64),
        np.asarray(pre.DEFAULT_TESTENV_STD, dtype=np.float64),
    )


def model_to_physical_30(
    genome_30d: np.ndarray,
    *,
    te_mean: Optional[np.ndarray] = None,
    te_std: Optional[np.ndarray] = None,
) -> np.ndarray:
    te_m = te_mean if te_mean is not None else pre.DEFAULT_TESTENV_MEAN
    te_s = te_std if te_std is not None else pre.DEFAULT_TESTENV_STD
    return pre.denormalize_genome_to_data1123(genome_30d, te_mean=te_m, te_std=te_s)


def decode_head0_model(
    value_model: float,
    *,
    head0_name: str,
    ys_mean: float = pre.DEFAULT_YS_MEAN,
    uts_mean: float = pre.DEFAULT_UTS_MEAN,
) -> Dict[str, float]:
    name = head0_name.upper()
    if name == "YS":
        phys = float(value_model) * float(ys_mean)
        return {"YS_MPa": phys}
    if name == "UTS":
        phys = float(value_model) * float(uts_mean)
        return {"UTS_MPa": phys}
    raise ValueError(f"unsupported head0_name: {head0_name}")


def decode_fs_model(
    fs_model: float,
    *,
    fs_mean: float = pre.DEFAULT_FS_MEAN,
) -> Dict[str, float]:
    fs_dataori2 = float(fs_model) * float(fs_mean)
    fs_data1123 = pre.fs_dataori2_to_data1123(fs_dataori2)
    return {
        "FS_dataOri2": fs_dataori2,
        "FS_data1123": fs_data1123,
    }


def physical_docs_notes() -> list[str]:
    return [
        "元素 10 维保持 wt% 原值进入模型空间。",
        "testenv: tem/fcr 为 z-score；人类输入中 data1123 的 sr 与 fcr 同源管线，不能只改变量名。",
        f"coldway: active * ln(T/{pre.T_DIV}) 与 active * ln(t)；[0,0] 可能表示未激活或 T=800、t=1，不可无歧义逆推。",
        "YS_MPa = YS_model * mean_YS；UTS_MPa = UTS_model * mean_UTS；"
        "FS_dataOri2 = FS_model * mean_FS；FS_data1123 = FS_dataOri2 / 100。",
    ]
