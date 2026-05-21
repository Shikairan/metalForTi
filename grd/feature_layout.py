"""
feature_layout.py — 30 维节点特征布局与反推约束配置

与 gnnDir/datagnn 及 material_graph.pt 对齐：
  [0:10]   element — Al,Zr,Sn,Mo,Cr,Nb,Si,V,Ta,Fe（wt%）
  [10:12]  testenv — tem、fcr 的 z-score
  [12:30]  coldway — 18 维工艺特征

组分采用 A 模式：T_total=100 wt%，Ti = T_total - sum(element_0..9)（Ti 不在 x 内）。
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

# ---------- 维段常量 ----------
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
    """
    testenv 与 coldway 在训练张量空间中的 box 上下界。

    testenv 为 z-score；coldway 与 x[:, 12:30] 同尺度。
    element 由 ti_balance 投影约束，不在此 dataclass 中。
    """

    testenv_lower: torch.Tensor  # (2,) tem, fcr
    testenv_upper: torch.Tensor  # (2,)
    coldway_lower: torch.Tensor  # (18,)
    coldway_upper: torch.Tensor  # (18,)


def compute_ti_balance(
    x: torch.Tensor,
    total_wt: float = DEFAULT_TOTAL_WT,
) -> torch.Tensor:
    """
    按 A 模式计算每节点钛余量（wt%）。

    公式: Ti = total_wt - sum(element_0..9)

    参数:
        x: (N, 30) 或至少含前 10 列的元素特征。
        total_wt: 总量标尺，默认 100。

    返回:
        (N,) 浮点张量，每节点一个 Ti 含量。
    """
    return float(total_wt) - x[:, ELEMENT_SLICE].sum(dim=1)


def load_testenv_stats(stats_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    读取 build_datagnn 导出的 testenv_stats.csv。

    参数:
        stats_path: 含 col/mean/std 或列式 mean/std 的 CSV。

    返回:
        mean: (2,) tem、fcr 的原始空间均值。
        std: (2,) 标准差（下限 clamp 1e-8 防除零）。
    """
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
    """
    将物理单位的 tem/fcr 转为与训练图一致的 z-score。

    公式: z = (x - mean) / std

    参数:
        physical: 标量或长度为 2 的序列（tem, fcr）。
        mean, std: load_testenv_stats 的返回值。

    返回:
        (2,) z-score 张量。
    """
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
    """
    用训练节点上的分位数估计 testenv、coldway 的 box 界，并外扩 margin 比例跨度。

    参数:
        x: 全图特征 (N, 30)。
        train_mask: 训练节点掩码。
        margin: 在 [q_lo, q_hi] 区间宽度上外扩的比例，默认 5%。
        q_lo, q_hi: 分位数，默认 1% 与 99%。

    返回:
        FeatureBounds，可直接用于 build_default_slice_specs。
    """
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
    """
    testenv 界：若提供物理上下界则转为 z-score；否则沿用训练分位数。
    coldway 界：始终来自 bounds_from_train_x。

    参数:
        x, train_mask: 同 bounds_from_train_x。
        stats_path: testenv_stats.csv 路径。
        tem_lower/upper, fcr_lower/upper: 物理单位，None 表示该端用分位数。
        margin: 分位数外扩比例。

    返回:
        FeatureBounds。
    """
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
    """
    构造 grd 默认的三段投影规格：element(ti_balance) + testenv(box) + coldway(box)。

    参数:
        bounds: 训练估计的 testenv/coldway 界。
        total_wt: Ti A 模式总量，默认 100 wt%。

    返回:
        传给 MaskedCompositeProjector 的 FeatureSliceSpec 列表。
    """
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
    """
    一键构建默认的按维段组合投影器。

    参数:
        anchor_x: 训练真值 x，作 scaled_simplex 锚定（ti_balance 不强制依赖 anchor）。
        bounds: testenv/coldway 的 box 界。
        total_wt: 组分总量上限。

    返回:
        可用于 GNNInverter 的 MaskedCompositeProjector 实例。
    """
    return MaskedCompositeProjector(
        slices=build_default_slice_specs(bounds, total_wt=total_wt),
        anchor=anchor_x,
    )
