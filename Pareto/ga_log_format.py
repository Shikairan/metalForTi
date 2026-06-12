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
# 与 datagnn 编码一致：方式1=(T,t)，方式2/3=(C_a,C_b)
_COLDWAY_PARAM_NAMES = (
    ("T", "t"),
    ("C_a", "C_b"),
    ("C_a", "C_b"),
)


def _fmt(v: float, nd: int = 4) -> str:
    if abs(v) < 1e-12:
        return "—"
    if abs(v) >= 1000 or (abs(v) < 1e-2 and v != 0):
        return f"{v:.3e}"
    return f"{v:.{nd}f}"


def _fmt_coldway_val(v: float, nd: int = 4) -> str:
    """coldway 参数：0 也显式打印，避免与「缺失」混淆。"""
    if v == 0.0:
        return "0"
    return _fmt(v, nd=nd)


def _fmt_cell(v: float, width: int = 8) -> str:
    """表格列：固定宽度，避免科学计数法挤在一起。"""
    if abs(v) < 1e-12:
        return f"{'—':>{width}}"
    if abs(v) >= 1000 or (abs(v) < 1e-2 and v != 0):
        return f"{v:>{width}.2e}"
    return f"{v:>{width}.4f}"


def _row_active(row62: torch.Tensor, eps: float = 1e-8) -> bool:
    return bool(row62.abs().sum().item() > eps)


def format_coldway_table(cold: torch.Tensor) -> list[str]:
    """3 阶段 × 3 方式 × 2 参数；每阶段仅展示激活的一种方式。"""
    mat = cold.reshape(_COLDWAY_STAGES, 3, 2)
    lines: list[str] = []

    active_flags = [_row_active(mat[s]) for s in range(_COLDWAY_STAGES)]
    if any(active_flags):
        last_raw = max(i for i, a in enumerate(active_flags) if a)
        for i in range(last_raw):
            if not active_flags[i]:
                lines.append(
                    f"    [警告] 阶段{i + 1} 为空但后续阶段有值（非法行累计，应经 compile 修复）"
                )
                break

    last_active = -1
    for s in range(_COLDWAY_STAGES):
        row = mat[s]
        if not _row_active(row):
            continue
        last_active = s
        norms = [float(row[m].pow(2).sum().sqrt().item()) for m in range(_COLDWAY_METHODS)]
        m_best = int(max(range(_COLDWAY_METHODS), key=lambda m: norms[m]))
        a, b = float(row[m_best, 0].item()), float(row[m_best, 1].item())
        p0, p1 = _COLDWAY_PARAM_NAMES[m_best]
        lines.append(
            f"    阶段{s + 1}  方式{m_best + 1}  "
            f"{p0}={_fmt_coldway_val(a)}  {p1}={_fmt_coldway_val(b)}"
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


def _format_one_solution(
    section_title: str,
    genome: torch.Tensor,
    fitness: FitnessResult,
    *,
    target_ys: Optional[float],
    target_fs: Optional[float],
    ys_fs_from_labels: bool,
    gene_source: Optional[str],
    same_as_note: Optional[str] = None,
) -> list[str]:
    """单个解的详情段落（不含外层分隔线）。"""
    g = genome.detach().cpu().float()
    ti = float(compute_ti_balance(g.unsqueeze(0), DEFAULT_TOTAL_WT)[0].item())
    elem_sum = float(g[ELEMENT_SLICE].sum().item())
    pred_label = "标签 YS/FS（未 GNN forward）" if ys_fs_from_labels else "预测"

    lines = [f"  【{section_title}】"]
    if same_as_note:
        lines.append(f"    {same_as_note}")
    if gene_source:
        lines.append(f"    基因来源：{gene_source}")
    lines.extend([
        "    【目标与预测】",
        f"      目标    YS = {_fmt(target_ys) if target_ys is not None else '—':>10}    FS = {_fmt(target_fs) if target_fs is not None else '—':>10}",
        f"      {pred_label:<6}  YS = {_fmt(fitness.ys_pred):>10}    FS = {_fmt(fitness.fs_pred):>10}",
        f"      误差    |ΔYS| = {_fmt(fitness.f1):>8}    |ΔFS| = {_fmt(fitness.f2):>8}    锚定L2 = {_fmt(fitness.f3):>8}",
        "",
        "    【合金成分 wt%】",
    ])

    hdr = "      " + " ".join(f"{n:>8}" for n in ELEMENT_NAMES) + f"  | {'Ti':>8}"
    val = "      " + " ".join(
        _fmt_cell(float(g[i].item())) for i in range(10)
    ) + f"  | {_fmt_cell(ti)}"
    lines.extend([hdr, val, f"      (10元合计 {_fmt(elem_sum)} wt%)", ""])

    te = g[TESTENV_SLICE]
    lines.extend([
        "    【试验环境 z-score】",
        f"      tem = {_fmt(float(te[0].item())):>10}    fcr = {_fmt(float(te[1].item())):>10}",
        "",
        "    【冷加工 coldway】",
    ])
    for cw_line in format_coldway_table(g[COLDWAY_SLICE]):
        lines.append("  " + cw_line)
    lines.append("")
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
    virtual_genome: Optional[torch.Tensor] = None,
    virtual_fitness: Optional[FitnessResult] = None,
    virtual_gene_source: Optional[str] = None,
    virtual_same_as_overall: bool = False,
) -> str:
    """生成单代日志块：全库最优 + 历史虚拟最优（不含 [INFO] 前缀）。"""
    meta_parts = [f"帕累托前沿 {front_size} 个体"]
    if archive_size is not None:
        meta_parts.append(f"基因库 {archive_size}")
    if new_virtual_count is not None and new_virtual_count > 0:
        meta_parts.append(f"本代新增虚拟 {new_virtual_count}")

    lines = [
        _LINE,
        f"  {gen_label}  |  " + "  |  ".join(meta_parts),
        _LINE,
        "",
    ]
    lines.extend(
        _format_one_solution(
            "全库最优",
            genome,
            fitness,
            target_ys=target_ys,
            target_fs=target_fs,
            ys_fs_from_labels=ys_fs_from_labels,
            gene_source=gene_source,
        )
    )

    if virtual_genome is not None and virtual_fitness is not None:
        lines.extend(
            _format_one_solution(
                "历史虚拟最优",
                virtual_genome,
                virtual_fitness,
                target_ys=target_ys,
                target_fs=target_fs,
                ys_fs_from_labels=False,
                gene_source=virtual_gene_source,
                same_as_note="（与全库最优为同一个体）" if virtual_same_as_overall else None,
            )
        )
    else:
        lines.extend(["  【历史虚拟最优】", "    （尚无虚拟个体）", ""])

    lines.append(_LINE)
    return "\n".join(lines)
