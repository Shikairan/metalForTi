"""
masked_projector.py — 按特征维段的硬约束投影

对 x 的不同列区间施加不同投影（钛合金组分 / 试验环境 / 工艺），
供 GNNInverter 在每次优化步后调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import torch

from grd.gnn_inverter import Projector, SimplexProjector

# A 模式默认总量：Ti(wt%) = TOTAL_WT - sum(element_0..9)
DEFAULT_TOTAL_WT = 100.0


@dataclass
class FeatureSliceSpec:
    """
    单段特征的投影规格。

    属性:
        name: 段名称（日志用），如 element / testenv / coldway。
        start: 闭区间起始列索引（含）。
        end: 开区间结束列索引（不含）。
        kind: 投影类型（见 MaskedCompositeProjector）。
        box_lower / box_upper: box 类型时的下/上界，标量或逐维列表。
        total_wt: ti_balance 类型时的合金总量标尺（wt%）。
    """

    name: str
    start: int
    end: int
    kind: Literal[
        "simplex",
        "scaled_simplex",
        "ti_balance",
        "box",
        "nonnegative",
        "none",
    ] = "box"
    box_lower: Optional[Union[float, List[float], torch.Tensor]] = None
    box_upper: Optional[Union[float, List[float], torch.Tensor]] = None
    total_wt: float = DEFAULT_TOTAL_WT


def _project_simplex_rows(block: torch.Tensor) -> torch.Tensor:
    """
    对 (N, d) 的每一行做概率单纯形投影：非负且行和为 1。

    参数:
        block: 待投影子矩阵，不原地修改。

    返回:
        投影后的新张量，形状与 block 相同。
    """
    return SimplexProjector().project(block.clone())


def _project_scaled_simplex_rows(block: torch.Tensor, target_sums: torch.Tensor) -> torch.Tensor:
    """
    先单纯形（行和=1），再按 target_sums 缩放到目标总量（如 wt% 行和）。

    参数:
        block: (N, d) 子矩阵。
        target_sums: (N,) 每行目标总和。

    返回:
        缩放后的子矩阵。
    """
    proj = _project_simplex_rows(block)
    scale = target_sums.view(-1, 1).to(device=proj.device, dtype=proj.dtype)
    return proj * scale


def _project_ti_balance_rows(block: torch.Tensor, total_wt: float = DEFAULT_TOTAL_WT) -> torch.Tensor:
    """
    钛合金 A 模式组分投影。

    - 各元素非负；
    - 行和不超过 total_wt；超出则等比缩放到 total_wt；
    - Ti 含量不在 block 内，由 feature_layout.compute_ti_balance 后处理。

    参数:
        block: (N, 10) 对应 element_0..9。
        total_wt: 合金元素总量上限（wt%），默认 100。

    返回:
        满足约束的子矩阵。
    """
    block = block.clamp(min=0)
    total = float(total_wt)
    s = block.sum(dim=1, keepdim=True)
    over = s > total
    if bool(over.any()):
        block = torch.where(over, block * (total / s.clamp(min=1e-12)), block)
    return block


def _resolve_bound(
    bound: Optional[Union[float, List[float], torch.Tensor]],
    d: int,
    block: torch.Tensor,
    *,
    default: float,
) -> torch.Tensor:
    """
    将标量/列表/张量界统一为 (d,) 张量，与 block 同 device/dtype。

    参数:
        bound: 下界或上界；None 时用 default 填充各维。
        d: 该段维度数。
        block: 用于对齐 device/dtype 的参考张量。
        default: bound 为 None 时的填充值。

    返回:
        形状 (d,) 的界张量。
    """
    if bound is None:
        return torch.full((d,), float(default), device=block.device, dtype=block.dtype)
    if isinstance(bound, torch.Tensor):
        t = bound.reshape(-1).to(device=block.device, dtype=block.dtype)
    elif isinstance(bound, (list, tuple)):
        t = torch.tensor(bound, device=block.device, dtype=block.dtype)
    else:
        t = torch.full((d,), float(bound), device=block.device, dtype=block.dtype)
    if t.numel() != d:
        raise ValueError(f"Bound length {t.numel()} != slice dim {d}")
    return t


class MaskedCompositeProjector(Projector):
    """
    按 FeatureSliceSpec 列表对 x 的各列段依次投影。

    scaled_simplex 段可用 anchor 提供每行目标总和；ti_balance 为钛合金默认组分约束。
    """

    def __init__(
        self,
        slices: List[FeatureSliceSpec],
        anchor: Optional[torch.Tensor] = None,
    ) -> None:
        """
        参数:
            slices: 从左到右按列索引划分的投影规格列表。
            anchor: 训练真值 x (N, d_in)，用于 scaled_simplex 的行总和标尺。
        """
        self.slices = slices
        self.anchor = anchor

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        对整图节点特征 x (N, d_in) 逐段投影并写回。

        参数:
            x: 当前优化中的输入特征。

        返回:
            满足各段约束的 x（与输入同形状，可能原地更新列切片）。
        """
        out = x
        for spec in self.slices:
            if spec.kind == "none":
                continue
            block = out[:, spec.start : spec.end]
            if spec.kind == "simplex":
                out[:, spec.start : spec.end] = _project_simplex_rows(block)
            elif spec.kind == "scaled_simplex":
                if self.anchor is None:
                    target_sums = block.sum(dim=1).clamp(min=1e-12)
                else:
                    target_sums = self.anchor[:, spec.start : spec.end].sum(dim=1).clamp(min=1e-12)
                out[:, spec.start : spec.end] = _project_scaled_simplex_rows(block, target_sums)
            elif spec.kind == "ti_balance":
                out[:, spec.start : spec.end] = _project_ti_balance_rows(
                    block, total_wt=spec.total_wt
                )
            elif spec.kind == "nonnegative":
                out[:, spec.start : spec.end] = block.clamp(min=0)
            elif spec.kind == "box":
                d = spec.end - spec.start
                lower = _resolve_bound(spec.box_lower, d, block, default=-1e6)
                upper = _resolve_bound(spec.box_upper, d, block, default=1e6)
                out[:, spec.start : spec.end] = block.clamp(min=lower, max=upper)
            else:
                raise ValueError(f"Unknown slice kind: {spec.kind}")
        return out
