"""
datagnn / material_graph 节点特征布局（in_dim=30）与默认约束配置。

组分 A 模式：T_total=100 wt%，Ti = 100 - sum(element_0..9)；Ti 不在 x 向量内。
element_0..9 = Al,Zr,Sn,Mo,Cr,Nb,Si,V,Ta,Fe（显式合金化元素）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch

from grd.masked_projector import (
    DEFAULT_TOTAL_WT,
    FeatureSliceSpec,
    MaskedCompositeProjector,
)

ELEMENT_DIM = 10
TESTENV_DIM = 2
COLDWAY_DIM = 18
INPUT_DIM = ELEMENT_DIM + TESTENV_DIM + COLDWAY_DIM  # 30

ELEMENT_NAMES = ["Al", "Zr", "Sn", "Mo", "Cr", "Nb", "Si", "V", "Ta", "Fe"]

ELEMENT_SLICE = slice(0, ELEMENT_DIM)
TESTENV_SLICE = slice(ELEMENT_DIM, ELEMENT_DIM + TESTENV_DIM)
COLDWAY_SLICE = slice(ELEMENT_DIM + TESTENV_DIM, INPUT_DIM)


@dataclass
class FeatureBounds:
    """与训练张量同空间的上下界（testenv 为 z-score）。"""

    testenv_lower: torch.Tensor  # (2,)
    testenv_upper: torch.Tensor  # (2,)
    coldway_lower: torch.Tensor  # (18,)
    coldway_upper: torch.Tensor  # (18,)


def compute_ti_balance(
    x: torch.Tensor,
    total_wt: float = DEFAULT_TOTAL_WT,
) -> torch.Tensor:
    """Ti(wt%) = total_wt - sum(element_0..9)，形状 (N,)。"""
    return float(total_wt) - x[:, ELEMENT_SLICE].sum(dim=1)


def load_testenv_stats(stats_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """读取 testenv_stats.csv，返回 mean/std，列顺序 tem, fcr。"""
    df = pd.read_csv(stats_path)
    if "col" in df.columns:
        df = df.set_index("col")
        mean = torch.tensor([df.loc["tem", "mean"], df.loc["fcr", "mean"]], dtype=torch.float32)
        std = torch.tensor([df.loc["tem", "std"], df.loc["fcr", "std"]], dtype=torch.float32)
    else:
        mean = torch.tensor(df["mean"].tolist()[:2], dtype=torch.float32)
        std = torch.tensor(df["std"].tolist()[:2], dtype=torch.float32)
    std = std.clamp(min=1e-8)
    return mean, std


def physical_testenv_to_z(
    physical: Union[float, List[float], torch.Tensor],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """物理量 -> 与训练一致的 z-score。"""
    p = torch.as_tensor(physical, dtype=torch.float32).reshape(-1)
    return (p - mean) / std


def bounds_from_train_x(
    x: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    margin: float = 0.05,
    q_lo: float = 0.01,
    q_hi: float = 0.99,
) -> FeatureBounds:
    """用训练节点分位数估计 testenv / coldway 的 box。"""
    xt = x[train_mask]
    te = xt[:, TESTENV_SLICE]
    cw = xt[:, COLDWAY_SLICE]
    te_lo = torch.quantile(te, q_lo, dim=0)
    te_hi = torch.quantile(te, q_hi, dim=0)
    cw_lo = torch.quantile(cw, q_lo, dim=0)
    cw_hi = torch.quantile(cw, q_hi, dim=0)

    span_te = (te_hi - te_lo).clamp(min=1e-6)
    span_cw = (cw_hi - cw_lo).clamp(min=1e-6)
    te_lo = te_lo - margin * span_te
    te_hi = te_hi + margin * span_te
    cw_lo = cw_lo - margin * span_cw
    cw_hi = cw_hi + margin * span_cw

    return FeatureBounds(
        testenv_lower=te_lo,
        testenv_upper=te_hi,
        coldway_lower=cw_lo,
        coldway_upper=cw_hi,
    )


def bounds_with_physical_testenv(
    x: torch.Tensor,
    train_mask: torch.Tensor,
    stats_path: Path,
    *,
    tem_lower: Optional[float] = None,
    tem_upper: Optional[float] = None,
    fcr_lower: Optional[float] = None,
    fcr_upper: Optional[float] = None,
    margin: float = 0.05,
) -> FeatureBounds:
    base = bounds_from_train_x(x, train_mask, margin=margin)
    mean, std = load_testenv_stats(stats_path)
    te_lo = base.testenv_lower.clone()
    te_hi = base.testenv_upper.clone()
    phys_lo = [tem_lower, fcr_lower]
    phys_hi = [tem_upper, fcr_upper]
    for i, (plo, phi) in enumerate(zip(phys_lo, phys_hi)):
        if plo is not None:
            te_lo[i] = physical_testenv_to_z(plo, mean[i], std[i])
        if phi is not None:
            te_hi[i] = physical_testenv_to_z(phi, mean[i], std[i])
    return FeatureBounds(
        testenv_lower=te_lo,
        testenv_upper=te_hi,
        coldway_lower=base.coldway_lower,
        coldway_upper=base.coldway_upper,
    )


def build_default_slice_specs(
    bounds: FeatureBounds,
    total_wt: float = DEFAULT_TOTAL_WT,
) -> List[FeatureSliceSpec]:
    return [
        FeatureSliceSpec(
            name="element",
            start=0,
            end=ELEMENT_DIM,
            kind="ti_balance",
            total_wt=total_wt,
        ),
        FeatureSliceSpec(
            name="testenv",
            start=ELEMENT_DIM,
            end=ELEMENT_DIM + TESTENV_DIM,
            kind="box",
            box_lower=bounds.testenv_lower.tolist(),
            box_upper=bounds.testenv_upper.tolist(),
        ),
        FeatureSliceSpec(
            name="coldway",
            start=ELEMENT_DIM + TESTENV_DIM,
            end=INPUT_DIM,
            kind="box",
            box_lower=bounds.coldway_lower.tolist(),
            box_upper=bounds.coldway_upper.tolist(),
        ),
    ]


def build_projector(
    anchor_x: torch.Tensor,
    bounds: FeatureBounds,
    total_wt: float = DEFAULT_TOTAL_WT,
) -> MaskedCompositeProjector:
    return MaskedCompositeProjector(
        slices=build_default_slice_specs(bounds, total_wt=total_wt),
        anchor=anchor_x,
    )
