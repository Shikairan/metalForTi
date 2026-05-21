"""按特征维段的硬约束投影（element / testenv / coldway）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import torch

from grd.gnn_inverter import Projector, SimplexProjector

# A 模式：Ti 余量 = TOTAL_WT - sum(element_0..9)，element 非负且总和不超过 TOTAL_WT
DEFAULT_TOTAL_WT = 100.0


@dataclass
class FeatureSliceSpec:
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
    return SimplexProjector().project(block.clone())


def _project_scaled_simplex_rows(block: torch.Tensor, target_sums: torch.Tensor) -> torch.Tensor:
    proj = _project_simplex_rows(block)
    scale = target_sums.view(-1, 1).to(device=proj.device, dtype=proj.dtype)
    return proj * scale


def _project_ti_balance_rows(block: torch.Tensor, total_wt: float = DEFAULT_TOTAL_WT) -> torch.Tensor:
    """
    钛合金 A 模式：element_0..9 非负，且 sum <= total_wt（wt%）。
    超出时按行等比缩放到 total_wt；Ti(%) = total_wt - sum 由后处理计算，不在 x 向量内。
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
    def __init__(
        self,
        slices: List[FeatureSliceSpec],
        anchor: Optional[torch.Tensor] = None,
    ) -> None:
        self.slices = slices
        self.anchor = anchor

    def project(self, x: torch.Tensor) -> torch.Tensor:
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
