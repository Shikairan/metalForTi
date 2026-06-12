"""
ga_compile.py — 基因组编译：element / testenv / coldway(3×3×2) 约束修复。
"""

from __future__ import annotations

from typing import Optional

import torch

from grd.feature_layout import (
    COLDWAY_DIM,
    COLDWAY_SLICE,
    DEFAULT_TOTAL_WT,
    ELEMENT_DIM,
    ELEMENT_SLICE,
    FeatureBounds,
    TESTENV_SLICE,
)
from grd.masked_projector import MaskedCompositeProjector

_COLDWAY_STAGES = 3
_COLDWAY_METHODS = 3
_COLDWAY_VALS = 2
_ROW_DIM = _COLDWAY_METHODS * _COLDWAY_VALS  # 6


def _row_active(row6: torch.Tensor, eps: float = 1e-8) -> bool:
    return bool(row6.abs().sum().item() > eps)


def _pick_one_method(row62: torch.Tensor) -> torch.Tensor:
    """3 选 1：保留 L2 范数最大的方式，其余 (0,0)。"""
    out = torch.zeros_like(row62)
    norms = torch.stack([row62[m].pow(2).sum() for m in range(_COLDWAY_METHODS)])
    if norms.max().item() <= 1e-12:
        return out
    best = int(torch.argmax(norms).item())
    out[best] = row62[best]
    return out


def compile_coldway(
    cold: torch.Tensor,
    bounds: FeatureBounds,
    train_coldway: Optional[torch.Tensor],
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    cold: (18,) — 3 阶段 × (3 方式 × 2 数值)，行累计、至少阶段 0 非空。
    """
    cw = cold.clone().float().reshape(_COLDWAY_STAGES, _ROW_DIM)
    lo = bounds.coldway_lower.float()
    hi = bounds.coldway_upper.float()

    rows = []
    for i in range(_COLDWAY_STAGES):
        row62 = cw[i].reshape(_COLDWAY_METHODS, _COLDWAY_VALS)
        row62 = _pick_one_method(row62)
        flat = row62.reshape(_ROW_DIM)
        flat = flat.clamp(
            min=lo[i * _ROW_DIM : (i + 1) * _ROW_DIM],
            max=hi[i * _ROW_DIM : (i + 1) * _ROW_DIM],
        )
        rows.append(flat)

    mat = torch.stack(rows, dim=0)

    def _fill_stage_row(stage_i: int) -> None:
        if train_coldway is not None and train_coldway.numel() > 0:
            j = int(torch.randint(0, train_coldway.shape[0], (1,)).item())
            mat[stage_i] = train_coldway[j].reshape(_COLDWAY_STAGES, _ROW_DIM)[stage_i]
        else:
            mat[stage_i] = ((lo + hi) / 2)[stage_i * _ROW_DIM : (stage_i + 1) * _ROW_DIM]

    if not _row_active(mat[0], eps=eps):
        _fill_stage_row(0)

    last_active = -1
    for i in range(_COLDWAY_STAGES):
        if _row_active(mat[i], eps=eps):
            last_active = i
    if last_active < 0:
        last_active = 0

    for i in range(_COLDWAY_STAGES):
        if i > last_active:
            mat[i] = torch.zeros(_ROW_DIM, dtype=mat.dtype)
        elif i <= last_active and not _row_active(mat[i], eps=eps):
            _fill_stage_row(i)

    for i in range(_COLDWAY_STAGES):
        row62 = mat[i].reshape(_COLDWAY_METHODS, _COLDWAY_VALS)
        mat[i] = _pick_one_method(row62).reshape(_ROW_DIM)

    return mat.reshape(COLDWAY_DIM)


def compile_element(elem: torch.Tensor, total_wt: float = DEFAULT_TOTAL_WT) -> torch.Tensor:
    e = elem.clamp(min=0)
    s = e.sum()
    if s.item() > float(total_wt):
        e = e * (float(total_wt) / s.clamp(min=1e-12))
    return e


def compile_testenv(te: torch.Tensor, bounds: FeatureBounds) -> torch.Tensor:
    return te.clamp(min=bounds.testenv_lower, max=bounds.testenv_upper)


def compile_genome(
    genome: torch.Tensor,
    bounds: FeatureBounds,
    projector: MaskedCompositeProjector,
    train_bank: Optional[torch.Tensor] = None,
    *,
    total_wt: float = DEFAULT_TOTAL_WT,
) -> torch.Tensor:
    """
  编译单个个体的 30 维基因组。

  参数:
      genome: (30,)
      bounds: testenv/coldway box
      projector: MaskedCompositeProjector（含 ti_balance）
      train_bank: (M, 30) 训练样本，用于 coldway 种子

  返回:
      (30,) 编译后基因组
  """
    g = genome.clone().float()
    train_cw = None
    if train_bank is not None and train_bank.numel() > 0:
        train_cw = train_bank[:, COLDWAY_SLICE]

    g[ELEMENT_SLICE] = compile_element(g[ELEMENT_SLICE], total_wt=total_wt)
    g[TESTENV_SLICE] = compile_testenv(g[TESTENV_SLICE], bounds)
    g[COLDWAY_SLICE] = compile_coldway(g[COLDWAY_SLICE], bounds, train_cw)

    row = g.unsqueeze(0)
    row = projector.project(row)
    g = row.squeeze(0)
    g[COLDWAY_SLICE] = compile_coldway(g[COLDWAY_SLICE], bounds, train_cw)
    return g


def _self_test() -> None:
    from grd.feature_layout import bounds_from_train_x, build_projector

    x = torch.rand(20, 30)
    mask = torch.ones(20, dtype=torch.bool)
    bounds = bounds_from_train_x(x, mask)
    proj = build_projector(x, bounds)
    g = x[0].clone()
    g[COLDWAY_SLICE] = torch.randn(COLDWAY_DIM) * 0.1
    out = compile_genome(g, bounds, proj, x)
    assert out.shape == (30,)
    assert out[ELEMENT_SLICE].sum() <= 100.0 + 1e-4
    cw = out[COLDWAY_SLICE].reshape(3, 6)
    for i in range(3):
        m = cw[i].reshape(3, 2)
        nonzero = (m.pow(2).sum(dim=1) > 1e-10).sum().item()
        assert nonzero <= 1
    print("[OK] ga_compile self-test passed")


if __name__ == "__main__":
    _self_test()
