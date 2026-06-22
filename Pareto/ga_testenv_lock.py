"""
ga_testenv_lock.py — 用户固定 tem/sr（data1123）时锁定基因组 testenv 维。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from preprocess.preprocess_datagnn_repro import inverse_testenv_z

if TYPE_CHECKING:
    import torch

    from grd.feature_layout import FeatureBounds

_TESTENV_SLICE = slice(10, 12)

logger = logging.getLogger("Pareto.ga_testenv_lock")


@dataclass(frozen=True)
class FixedTestenvContext:
    """data1123 物理 tem/sr → 模型 z-score，供评估与育种前写入基因组。"""

    tem_physical: float
    sr_physical: float
    testenv_z_np: np.ndarray  # (2,) float32

    @property
    def enabled(self) -> bool:
        return True

    def testenv_z_tensor(self) -> torch.Tensor:
        import torch

        return torch.tensor(self.testenv_z_np, dtype=torch.float32)

    def apply(self, genome: torch.Tensor) -> torch.Tensor:
        g = genome.clone()
        g[_TESTENV_SLICE] = self.testenv_z_tensor()
        return g

    def apply_inplace(self, genome: torch.Tensor) -> None:
        genome[_TESTENV_SLICE] = self.testenv_z_tensor()

    def to_summary_dict(self) -> dict:
        return {
            "testenv_locked": True,
            "fixed_tem": self.tem_physical,
            "fixed_sr": self.sr_physical,
            "fixed_testenv_z": self.testenv_z_np.tolist(),
        }


def physical_testenv_to_z_np(
    tem: float,
    sr: float,
    te_mean: np.ndarray,
    te_std: np.ndarray,
) -> np.ndarray:
    p = np.array([tem, sr], dtype=np.float64)
    std_safe = np.where(te_std == 0, 1.0, te_std)
    return ((p - te_mean) / std_safe).astype(np.float32)


def _to_numpy_1d(arr) -> np.ndarray:
    if hasattr(arr, "detach"):
        return arr.detach().cpu().numpy().astype(np.float64)
    return np.asarray(arr, dtype=np.float64)


def resolve_fixed_testenv(
    fixed_tem: Optional[float],
    fixed_sr: Optional[float],
    te_mean: np.ndarray,
    te_std: np.ndarray,
    bounds: FeatureBounds,
) -> Optional[FixedTestenvContext]:
    """解析 CLI；二者均省略则返回 None，只提供一个则报错。"""
    if fixed_tem is None and fixed_sr is None:
        return None
    if fixed_tem is None or fixed_sr is None:
        raise ValueError(
            "必须同时提供 --fixed-tem 与 --fixed-sr，或均省略以允许 testenv 参与遗传"
        )

    z_np = physical_testenv_to_z_np(fixed_tem, fixed_sr, te_mean, te_std)
    lo = _to_numpy_1d(bounds.testenv_lower)
    hi = _to_numpy_1d(bounds.testenv_upper)
    if (z_np < lo).any() or (z_np > hi).any():
        phys_lo = inverse_testenv_z(lo.astype(np.float32), te_mean=te_mean, te_std=te_std)
        phys_hi = inverse_testenv_z(hi.astype(np.float32), te_mean=te_mean, te_std=te_std)
        logger.warning(
            "固定试验环境 tem=%.4f sr=%.6f（data1123）超出训练 box "
            "tem∈[%.4f, %.4f] sr∈[%.6f, %.6f]，将继续优化（模型外推风险）",
            fixed_tem,
            fixed_sr,
            float(phys_lo[0]),
            float(phys_hi[0]),
            float(phys_lo[1]),
            float(phys_hi[1]),
        )

    logger.info(
        "试验环境已锁定：tem=%.4f sr=%.6f（data1123）→ z=[%.6f, %.6f]；"
        "优化仅演化成分与 coldway（父本/子代评估前覆盖 testenv，604 档案保留原始值）",
        fixed_tem,
        fixed_sr,
        float(z_np[0]),
        float(z_np[1]),
    )
    return FixedTestenvContext(
        tem_physical=float(fixed_tem),
        sr_physical=float(fixed_sr),
        testenv_z_np=z_np,
    )
