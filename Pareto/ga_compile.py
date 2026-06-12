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


def _sample_train_stage_row(
    train_coldway: torch.Tensor,
    stage_i: int,
    rng: Optional[torch.Generator],
) -> Optional[torch.Tensor]:
    """从训练库中抽取「该阶段非空」的 6 维行块。"""
    active_rows: list[torch.Tensor] = []
    for j in range(train_coldway.shape[0]):
        row6 = train_coldway[j].reshape(_COLDWAY_STAGES, _ROW_DIM)[stage_i]
        if _row_active(row6):
            active_rows.append(row6.clone())
    if not active_rows:
        return None
    idx = int(torch.randint(0, len(active_rows), (1,), generator=rng).item())
    return active_rows[idx]


def _enforce_cumulative_stages(
    mat: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    train_coldway: Optional[torch.Tensor],
    rng: Optional[torch.Generator],
    eps: float,
) -> None:
    """行累计：若启用阶段 k，则 0..k 均非空；禁止阶段 1+3 跳阶段 2。"""
    if not _row_active(mat[0], eps=eps):
        mid = ((lo + hi) / 2)[:_ROW_DIM]
        mat[0] = mid
        row62 = mat[0].reshape(_COLDWAY_METHODS, _COLDWAY_VALS)
        mat[0] = _pick_one_method(row62).reshape(_ROW_DIM)

    raw_last = -1
    for i in range(_COLDWAY_STAGES):
        if _row_active(mat[i], eps=eps):
            raw_last = i

    if raw_last < 0:
        return

    mid_row = ((lo + hi) / 2)[:_ROW_DIM]
    for i in range(raw_last + 1):
        if _row_active(mat[i], eps=eps):
            continue
        filled: Optional[torch.Tensor] = None
        if train_coldway is not None and train_coldway.numel() > 0:
            filled = _sample_train_stage_row(train_coldway, i, rng)
        if filled is None:
            filled = mid_row.clone()
        row62 = filled.reshape(_COLDWAY_METHODS, _COLDWAY_VALS)
        mat[i] = _pick_one_method(row62).reshape(_ROW_DIM)

    for i in range(raw_last + 1, _COLDWAY_STAGES):
        mat[i].zero_()

    for i in range(raw_last + 1):
        row62 = mat[i].reshape(_COLDWAY_METHODS, _COLDWAY_VALS)
        mat[i] = _pick_one_method(row62).reshape(_ROW_DIM)


def compile_coldway(
    cold: torch.Tensor,
    bounds: FeatureBounds,
    train_coldway: Optional[torch.Tensor],
    *,
    rng: Optional[torch.Generator] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    cold: (18,) — 3 阶段 × (3 方式 × 2 数值)，行累计、至少阶段 0 非空。

    rng: 随机数生成器；传入后保证 --seed 可复现性。
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
    _enforce_cumulative_stages(mat, lo, hi, train_coldway, rng, eps)
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
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    编译单个个体的 30 维基因组。

    参数:
        genome: (30,)
        bounds: testenv/coldway box
        projector: MaskedCompositeProjector（含 ti_balance）
        train_bank: (M, 30) 训练样本，用于 coldway 种子
        rng: 随机数生成器；传入后保证 --seed 可复现性

    返回:
        (30,) 编译后基因组
    """
    g = genome.clone().float()
    train_cw = None
    if train_bank is not None and train_bank.numel() > 0:
        train_cw = train_bank[:, COLDWAY_SLICE]

    g[ELEMENT_SLICE] = compile_element(g[ELEMENT_SLICE], total_wt=total_wt)
    g[TESTENV_SLICE] = compile_testenv(g[TESTENV_SLICE], bounds)
    g[COLDWAY_SLICE] = compile_coldway(g[COLDWAY_SLICE], bounds, train_cw, rng=rng)

    row = g.unsqueeze(0)
    row = projector.project(row)
    g = row.squeeze(0)
    g[COLDWAY_SLICE] = compile_coldway(g[COLDWAY_SLICE], bounds, train_cw, rng=rng)
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
    # 阶段 1+3 无阶段 2 的非法模式应被修复
    g_bad = x[0].clone()
    g_bad[COLDWAY_SLICE] = 0.0
    g_bad[COLDWAY_SLICE].reshape(3, 6)[0, 4] = 0.5
    g_bad[COLDWAY_SLICE].reshape(3, 6)[2, 0] = -0.1
    out_bad = compile_genome(g_bad, bounds, proj, x)
    cw2 = out_bad[COLDWAY_SLICE].reshape(3, 6)
    assert cw2[1].abs().sum().item() > 1e-8
    print("[OK] ga_compile self-test passed")


if __name__ == "__main__":
    _self_test()
