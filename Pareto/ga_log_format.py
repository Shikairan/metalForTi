"""
ga_log_format.py — 每代最优个体的人类可读日志排版。
"""

from __future__ import annotations

from typing import Optional

import torch

from grd.feature_layout import (
    COLDWAY_SLICE,
    DEFAULT_TOTAL_WT,
    ELEMENT_NAMES,
    ELEMENT_SLICE,
    TESTENV_SLICE,
    compute_ti_balance,
)
from Pareto.ga_evaluate import FitnessResult

_LINE = "─" * 72
_COLDWAY_STAGES = 3
_COLDWAY_METHODS = 3


def _fmt(v: float, nd: int = 4) -> str:
    if abs(v) < 1e-12:
        return "—"
    if abs(v) >= 1000 or (abs(v) < 1e-2 and v != 0):
        return f"{v:.3e}"
    return f"{v:.{nd}f}"


def _fmt_cell(v: float, width: int = 8) -> str:
    """表格列：固定宽度，避免科学计数法挤在一起。"""
    if abs(v) < 1e-12:
        return f"{'—':>{width}}"
    if abs(v) >= 1000 or (abs(v) < 1e-2 and v != 0):
        return f"{v:>{width}.2e}"
    return f"{v:>{width}.4f}"


def format_coldway_table(cold: torch.Tensor) -> list[str]:
    """3 阶段 × 3 方式 × 2 参数；每阶段仅展示激活的一种方式。"""
    mat = cold.reshape(_COLDWAY_STAGES, 3, 2)
    lines: list[str] = []
    last_active = -1
    for s in range(_COLDWAY_STAGES):
        row = mat[s]
        if row.abs().sum().item() <= 1e-10:
            continue
        last_active = s
        norms = [float(row[m].pow(2).sum().sqrt().item()) for m in range(_COLDWAY_METHODS)]
        m_best = int(max(range(_COLDWAY_METHODS), key=lambda m: norms[m]))
        a, b = float(row[m_best, 0].item()), float(row[m_best, 1].item())
        lines.append(
            f"    阶段{s + 1}  方式{m_best + 1}  ({_fmt(a)}, {_fmt(b)})"
        )
    if last_active < 0:
        lines.append("    (无冷加工)")
    elif last_active < _COLDWAY_STAGES - 1:
        s0 = last_active + 2
        s1 = _COLDWAY_STAGES
        if s0 == s1:
            lines.append(f"    阶段{s0}: 未启用")
        else:
            lines.append(f"    阶段{s0}~{s1}: 未启用")
    return lines


def format_generation_block(
    gen_label: str,
    front_size: int,
    genome: torch.Tensor,
    fitness: FitnessResult,
    *,
    target_ys: Optional[float] = None,
    target_fs: Optional[float] = None,
    ys_fs_from_labels: bool = False,
    archive_size: Optional[int] = None,
    new_virtual_count: Optional[int] = None,
    gene_source: Optional[str] = None,
) -> str:
    """生成单代最优个体的多行文本块（不含 [INFO] 前缀）。"""
    g = genome.detach().cpu().float()
    ti = float(compute_ti_balance(g.unsqueeze(0), DEFAULT_TOTAL_WT)[0].item())
    elem_sum = float(g[ELEMENT_SLICE].sum().item())

    pred_label = "标签 YS/FS（未 GNN forward）" if ys_fs_from_labels else "预测"
    meta_parts = [f"帕累托前沿 {front_size} 个体"]
    if archive_size is not None:
        meta_parts.append(f"基因库 {archive_size}")
    if new_virtual_count is not None and new_virtual_count > 0:
        meta_parts.append(f"本代新增虚拟 {new_virtual_count}")

    lines = [
        _LINE,
        f"  {gen_label}  |  " + "  |  ".join(meta_parts),
        _LINE,
    ]
    if gene_source:
        lines.append(f"  【基因来源】{gene_source}")
        lines.append("")
    lines.extend([
        "  【目标与预测】",
        f"    目标    YS = {_fmt(target_ys) if target_ys is not None else '—':>10}    FS = {_fmt(target_fs) if target_fs is not None else '—':>10}",
        f"    {pred_label:<6}  YS = {_fmt(fitness.ys_pred):>10}    FS = {_fmt(fitness.fs_pred):>10}",
        f"    误差    |ΔYS| = {_fmt(fitness.f1):>8}    |ΔFS| = {_fmt(fitness.f2):>8}    锚定L2 = {_fmt(fitness.f3):>8}",
        "",
        "  【合金成分 wt%】",
    ])

    hdr = "    " + " ".join(f"{n:>8}" for n in ELEMENT_NAMES) + f"  | {'Ti':>8}"
    val = "    " + " ".join(
        _fmt_cell(float(g[i].item())) for i in range(10)
    ) + f"  | {_fmt_cell(ti)}"
    lines.extend([hdr, val, f"    (10元合计 {_fmt(elem_sum)} wt%)", ""])

    te = g[TESTENV_SLICE]
    lines.extend([
        "  【试验环境 z-score】",
        f"    tem = {_fmt(float(te[0].item())):>10}    fcr = {_fmt(float(te[1].item())):>10}",
        "",
        "  【冷加工 coldway】",
    ])
    lines.extend(format_coldway_table(g[COLDWAY_SLICE]))
    lines.append(_LINE)
    return "\n".join(lines)
