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
from Pareto.coldway_display import format_coldway_lines
from Pareto.ga_objectives import target_shortfall
from Pareto.ga_evaluate import FitnessResult
from Pareto.ga_report import OutputRestoreContext
from preprocess.preprocess_datagnn_repro import (
    denormalize_genome_to_data1123,
    denormalize_targets_to_data1123,
    inverse_coldway_flat_18,
)

_LINE = "─" * 72
_COLDWAY_STAGES = 3


def _fmt(v: float, nd: int = 4) -> str:
    if abs(v) < 1e-12:
        return "—"
    if abs(v) >= 1000 or (abs(v) < 1e-2 and v != 0):
        return f"{v:.3e}"
    return f"{v:.{nd}f}"


def _fmt_objective(v: float, nd: int = 4) -> str:
    """NSGA-II 目标值：0 也显式打印，避免与「未启用」混淆。"""
    if abs(v) < 1e-12:
        return f"{0.0:.{nd}f}"
    return _fmt(v, nd=nd)


def _fmt_coldway_val(v: float, nd: int = 4) -> str:
    """兼容旧引用；请优先使用 Pareto.coldway_display.fmt_coldway_val。"""
    from Pareto.coldway_display import fmt_coldway_val

    return fmt_coldway_val(v, nd=nd)


def _fmt_cell(v: float, width: int = 8) -> str:
    """表格列：固定宽度，避免科学计数法挤在一起。"""
    if abs(v) < 1e-12:
        return f"{'—':>{width}}"
    if abs(v) >= 1000 or (abs(v) < 1e-2 and v != 0):
        return f"{v:>{width}.2e}"
    return f"{v:>{width}.4f}"


def _coldway_for_display(cold: torch.Tensor, restore: OutputRestoreContext | None) -> torch.Tensor:
    """模型空间 18 维 → 物理 T/t（layout 不变）。"""
    cw = cold.detach().cpu().float().reshape(-1)
    if restore is None:
        return cw
    return torch.as_tensor(
        inverse_coldway_flat_18(cw.numpy()),
        dtype=torch.float32,
    )


def format_coldway_table(
    cold: torch.Tensor,
    *,
    restore: OutputRestoreContext | None = None,
) -> list[str]:
    """3 阶段 × 3 方式 × 2 参数；每阶段仅展示激活的一种方式，参数名统一为 T、t。"""
    cw = _coldway_for_display(cold, restore).numpy()
    return format_coldway_lines(cw)


def _genome_for_display(
    genome: torch.Tensor,
    restore: OutputRestoreContext | None,
) -> torch.Tensor:
    g = genome.detach().cpu().float()
    if restore is None:
        return g
    return torch.as_tensor(
        denormalize_genome_to_data1123(
            g.numpy(),
            te_mean=restore.te_mean,
            te_std=restore.te_std,
        ),
        dtype=torch.float32,
    )


def _format_one_solution(
    section_title: str,
    genome: torch.Tensor,
    fitness: FitnessResult,
    *,
    target_ys: Optional[float],
    target_fs: Optional[float],
    ys_fs_from_labels: bool,
    gene_source: Optional[str],
    restore: OutputRestoreContext | None = None,
    same_as_note: Optional[str] = None,
) -> list[str]:
    """单个解的详情段落（不含外层分隔线）。"""
    g_disp = _genome_for_display(genome, restore)
    ti = float(compute_ti_balance(g_disp.unsqueeze(0), DEFAULT_TOTAL_WT)[0].item())
    elem_sum = float(g_disp[ELEMENT_SLICE].sum().item())
    pred_label = "标签 YS/FS（未 GNN forward）" if ys_fs_from_labels else "预测"

    if restore is None:
        raise ValueError(
            "日志输出需要 OutputRestoreContext：目标/预测必须在同一物理量纲下展示，"
            "请传入 restore（原始输入经预处理后，展示时做逆变换）"
        )

    ys_pred, fs_pred = denormalize_targets_to_data1123(
        fitness.ys_pred,
        fitness.fs_pred,
        ys_mean=restore.ys_mean,
        fs_mean=restore.fs_mean,
    )
    f1_phys = target_shortfall(ys_pred, restore.target_ys_physical)
    f2_phys = target_shortfall(fs_pred, restore.target_fs_physical)
    tgt_ys = restore.target_ys_physical
    tgt_fs = restore.target_fs_physical
    unit_note = "data1123"

    lines = [f"  【{section_title}】"]
    if same_as_note:
        lines.append(f"    {same_as_note}")
    if gene_source:
        lines.append(f"    基因来源：{gene_source}")
    lines.extend([
        f"    【目标与预测（{unit_note}，仅展示）】",
        f"      目标    YS = {_fmt(tgt_ys):>10}    FS = {_fmt(tgt_fs):>10}",
        f"      {pred_label:<6}  YS = {_fmt(ys_pred):>10}    FS = {_fmt(fs_pred):>10}",
        f"      欠达标（展示） YS = {_fmt(f1_phys):>8}    FS = {_fmt(f2_phys):>8}",
        f"      误差（NSGA-II 优化，模型量纲） f1={_fmt_objective(fitness.f1):>8}  f2={_fmt_objective(fitness.f2):>8}  f3={_fmt_objective(fitness.f3):>8}",
        "",
        "    【合金成分 wt%】",
    ])

    hdr = "      " + " ".join(f"{n:>8}" for n in ELEMENT_NAMES) + f"  | {'Ti':>8}"
    val = "      " + " ".join(
        _fmt_cell(float(g_disp[i].item())) for i in range(10)
    ) + f"  | {_fmt_cell(ti)}"
    lines.extend([hdr, val, f"      (10元合计 {_fmt(elem_sum)} wt%)", ""])

    te = g_disp[TESTENV_SLICE]
    env_title = "试验环境（tem, sr）"
    te_label = ("tem", "sr")
    lines.extend([
        f"    【{env_title}】",
        f"      {te_label[0]} = {_fmt(float(te[0].item())):>10}    {te_label[1]} = {_fmt(float(te[1].item())):>10}",
        "",
        "    【冷加工 coldway】",
    ])
    for cw_line in format_coldway_table(genome[COLDWAY_SLICE], restore=restore):
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
    restore: OutputRestoreContext | None = None,
) -> str:
    """生成单代日志块：种群帕累托代表（不含 [INFO] 前缀）。"""
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
            "种群帕累托代表",
            genome,
            fitness,
            target_ys=target_ys,
            target_fs=target_fs,
            ys_fs_from_labels=ys_fs_from_labels,
            gene_source=gene_source,
            restore=restore,
        )
    )

    if virtual_genome is not None and virtual_fitness is not None:
        lines.extend(
            _format_one_solution(
                "基因库最优虚拟个体",
                virtual_genome,
                virtual_fitness,
                target_ys=target_ys,
                target_fs=target_fs,
                ys_fs_from_labels=False,
                gene_source=virtual_gene_source,
                restore=restore,
                same_as_note="（与帕累托代表为同一个体）" if virtual_same_as_overall else None,
            )
        )

    lines.append(_LINE)
    return "\n".join(lines)
