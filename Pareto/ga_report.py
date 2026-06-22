"""
ga_report.py — 帕累托前沿 JSON/TXT 报告与散点图。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger("Pareto.ga_report")

from grd.feature_layout import (
    COLDWAY_SLICE,
    DEFAULT_TOTAL_WT,
    ELEMENT_NAMES,
    ELEMENT_SLICE,
    TESTENV_SLICE,
    compute_ti_balance,
)
from preprocess.preprocess_datagnn_repro import (
    denormalize_genome_to_data1123,
    denormalize_targets_to_data1123,
)
from Pareto.ga_archive import ArchiveEntry, GeneArchive, weighted_score
from Pareto.ga_objectives import target_shortfall
from Pareto.ga_evaluate import FitnessResult
from Pareto.ga_nsga2 import Individual


@dataclass
class OutputRestoreContext:
    """Pareto 输出数值还原为 data1123 物理量纲。"""

    ys_mean: float
    fs_mean: float
    te_mean: np.ndarray
    te_std: np.ndarray
    target_ys_physical: float
    target_fs_physical: float

FIELD_DESCRIPTIONS_CN: Dict[str, str] = {
    "generated_at_utc": "报告生成时间（UTC）",
    "target_ys": "用户目标屈服强度 YS（模型量纲，与 ys.pt 一致）",
    "target_fs": "用户目标 FS（模型量纲，与 fs.pt 一致）",
    "target_ys_physical": "用户输入的 YS（与 data1123.csv 同量纲，MPa）",
    "target_fs_physical": "用户输入的 FS（与 data1123.csv 同量纲，如 0.147）",
    "targets_physical_input": "CLI 是否将 target 视为物理量",
    "label_means": "全表 YS/FS 均值（归一化用）",
    "objectives": "优化目标模式：three 或 two",
    "population_size": "最终种群规模",
    "archive_size": "基因库总条目数",
    "num_original": "原始图节点条目数",
    "num_virtual": "累积虚拟个体条目数",
    "offspring_per_generation": "每代新增子代数",
    "selection_method": "父本/存活选择算法（NSGA-II）",
    "pareto_front_size": "第一非支配层个体数",
    "individuals": "帕累托前沿个体列表",
    "genome_30d": "30 维基因组（data1123 物理量纲：element 10 + tem/sr 2 + coldway 18）",
    "element_wt_pct": "10 元含量 wt%",
    "ti_balance_wt_pct": "钛余量 wt%",
    "testenv": "试验环境 tem、sr（物理量，与 data1123 一致）",
    "coldway_3x6": "coldway 18 维重塑为 3×6（物理 T/t，0 表示未激活槽）",
    "f1_ys_shortfall": "max(0, 目标 YS − 预测 YS)（data1123 量纲，仅罚欠达标）",
    "f2_fs_shortfall": "max(0, 目标 FS − 预测 FS)（data1123 量纲，仅罚欠达标）",
    "f3_anchor_l2": "与训练集最近邻的 L2 距离",
    "ys_pred": "GNN 预测 YS（data1123 量纲，MPa）",
    "fs_pred": "GNN 预测 FS（data1123 量纲）",
    "nearest_train_idx": "最近邻训练样本在原始图中的节点 id（0 基准，对应 material_graph 节点序号）",
    "knee_index": "加权和折中解在 individuals 中的索引",
    "pareto_representative": "最终种群帕累托代表（拥挤距离最大）",
    "best_virtual": "基因库历史最优虚拟个体（加权分 f1+f2+0.1*f3 最小）",
    "fixed_testenv": "用户锁定的试验环境（data1123 tem/sr）；省略表示 testenv 参与遗传",
    "gene_source": "基因来源（原始/杂交虚拟）",
    "field_descriptions": "字段中文说明",
}


def _individual_to_dict(
    genome: torch.Tensor,
    fit: FitnessResult,
    restore: OutputRestoreContext,
) -> Dict[str, Any]:
    g_model = genome.detach().cpu().float()
    g_phys = torch.as_tensor(
        denormalize_genome_to_data1123(
            g_model.numpy(),
            te_mean=restore.te_mean,
            te_std=restore.te_std,
        ),
        dtype=torch.float32,
    )
    elem = {name: float(g_phys[i].item()) for i, name in enumerate(ELEMENT_NAMES)}
    ti = float(compute_ti_balance(g_phys.unsqueeze(0), DEFAULT_TOTAL_WT)[0].item())
    cw = g_phys[COLDWAY_SLICE].reshape(3, 6).tolist()
    te = g_phys[TESTENV_SLICE].tolist()
    ys_pred, fs_pred = denormalize_targets_to_data1123(
        fit.ys_pred,
        fit.fs_pred,
        ys_mean=restore.ys_mean,
        fs_mean=restore.fs_mean,
    )
    f1 = target_shortfall(ys_pred, restore.target_ys_physical)
    f2 = target_shortfall(fs_pred, restore.target_fs_physical)
    return {
        "genome_30d": g_phys.tolist(),
        "element_wt_pct": elem,
        "ti_balance_wt_pct": ti,
        "testenv": {"tem": te[0], "sr": te[1]},
        "coldway_3x6": cw,
        "f1_ys_shortfall": f1,
        "f2_fs_shortfall": f2,
        "f3_anchor_l2": fit.f3,
        "ys_pred": ys_pred,
        "fs_pred": fs_pred,
        "nearest_train_idx": fit.nearest_train_idx,
    }


def _entry_to_dict(entry: ArchiveEntry, restore: OutputRestoreContext) -> Dict[str, Any]:
    d = _individual_to_dict(entry.genome, entry.fitness, restore)
    d["gene_source"] = entry.source_label()
    d["is_original"] = entry.is_original
    d["weighted_score"] = weighted_score(entry.fitness)
    return d


def _append_best_dict_lines(lines: List[str], title: str, rec: Optional[Dict[str, Any]]) -> None:
    lines.append(f"【{title}】")
    if not rec:
        lines.append("  （无）")
        lines.append("")
        return
    lines.extend([
        f"  来源: {rec.get('gene_source', '—')}",
        f"  加权分: {rec.get('weighted_score', 0):.6f}",
        f"  f1 (欠达标 YS): {rec['f1_ys_shortfall']:.6f}",
        f"  f2 (欠达标 FS): {rec['f2_fs_shortfall']:.6f}",
        f"  f3 (锚定 L2): {rec.get('f3_anchor_l2', 0):.6f}",
        f"  预测 YS/FS（data1123）: {rec['ys_pred']:.4f} / {rec['fs_pred']:.6f}",
        f"  tem / sr: {rec['testenv']['tem']:.4f} / {rec['testenv']['sr']:.4f}",
        f"  Ti 余量 wt%: {rec['ti_balance_wt_pct']:.4f}",
        "",
    ])


def find_knee_index(individuals: List[Dict[str, Any]]) -> int:
    """返回加权和最小个体的索引（f1 + f2 + 0.1*f3 最小）。

    注意：这是加权标量折中解，而非几何意义上的 knee point。
    JSON 输出中的 knee_index 字段沿用此语义。
    """
    if not individuals:
        return 0
    scores = [
        ind["f1_ys_shortfall"] + ind["f2_fs_shortfall"] + 0.1 * ind.get("f3_anchor_l2", 0.0)
        for ind in individuals
    ]
    return int(min(range(len(scores)), key=lambda i: scores[i]))


def build_archive_summary(
    archive: GeneArchive,
    front: List[Individual],
    *,
    target_ys: float,
    target_fs: float,
    objectives: str,
    offspring_per_generation: int,
    generations: int,
    device: str,
    paths: Dict[str, str],
    selection_method: str = "NSGA-II",
    target_ys_physical: Optional[float] = None,
    target_fs_physical: Optional[float] = None,
    targets_physical: bool = False,
    label_means: Optional[Dict[str, float]] = None,
    restore: Optional[OutputRestoreContext] = None,
    fixed_testenv: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """从最终种群帕累托前沿构建报告。"""
    if restore is None:
        raise ValueError("restore context is required for physical output")
    individuals = []
    for ind in front:
        if ind.fitness is None:
            continue
        individuals.append(_individual_to_dict(ind.genome, ind.fitness, restore))
    knee = find_knee_index(individuals) if individuals else 0

    from Pareto.ga_nsga2 import pareto_representative

    rep = pareto_representative(front) if front else None
    rep_dict = (
        _individual_to_dict(rep.genome, rep.fitness, restore)
        if rep is not None and rep.fitness is not None
        else None
    )

    best_v = archive.best_virtual_entry()
    best_virtual_dict = (
        _entry_to_dict(best_v, restore) if best_v is not None else None
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_ys": target_ys,
        "target_fs": target_fs,
        "target_ys_physical": target_ys_physical,
        "target_fs_physical": target_fs_physical,
        "targets_physical_input": targets_physical,
        "label_means": label_means,
        "objectives": objectives,
        "selection_method": selection_method,
        "population_size": offspring_per_generation,
        "offspring_per_generation": offspring_per_generation,
        "generations": generations,
        "device": device,
        "archive_size": archive.size(),
        "num_original": archive.num_original(),
        "num_virtual": archive.num_virtual(),
        "pareto_representative": rep_dict,
        "best_virtual": best_virtual_dict,
        "fixed_testenv": fixed_testenv,
        "pareto_front_size": len(individuals),
        "knee_index": knee,
        "individuals": individuals,
        "paths": paths,
        "field_descriptions": FIELD_DESCRIPTIONS_CN,
    }


def write_pareto_json(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def write_ga_summary_txt(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "=" * 72,
        "Pareto 遗传逆设计结果汇总",
        "=" * 72,
        "",
        f"生成时间（UTC）: {summary.get('generated_at_utc', '')}",
        f"目标 YS（模型量纲）: {summary.get('target_ys')}",
        f"目标 FS（模型量纲）: {summary.get('target_fs')}",
        f"目标 YS（data1123）: {summary.get('target_ys_physical', '—')}",
        f"目标 FS（data1123）: {summary.get('target_fs_physical', '—')}",
        f"优化目标: {summary.get('objectives')}",
        f"选择算法: {summary.get('selection_method', 'NSGA-II')}",
        f"设备: {summary.get('device')}",
        f"进化代数: {summary.get('generations')}",
        f"种群规模: {summary.get('offspring_per_generation', summary.get('population_size'))}",
        f"基因库规模: {summary.get('archive_size', '—')}（原始 {summary.get('num_original', '—')} + 虚拟 {summary.get('num_virtual', '—')}）",
        f"帕累托前沿个体数: {summary.get('pareto_front_size')}",
        "",
    ]
    fx = summary.get("fixed_testenv")
    if fx and fx.get("testenv_locked"):
        lines.extend([
            f"试验环境锁定: tem={fx.get('fixed_tem')}  sr={fx.get('fixed_sr')}（data1123）",
            "",
        ])
    _append_best_dict_lines(lines, "种群帕累托代表", summary.get("pareto_representative"))
    _append_best_dict_lines(lines, "基因库最优虚拟个体", summary.get("best_virtual"))
    inds = summary.get("individuals", [])
    knee = summary.get("knee_index", 0)
    if inds:
        k = inds[knee]
        lines.extend([
            "【折中解（knee）】",
            f"  f1 (欠达标 YS): {k['f1_ys_shortfall']:.6f}",
            f"  f2 (欠达标 FS): {k['f2_fs_shortfall']:.6f}",
            f"  f3 (锚定 L2): {k.get('f3_anchor_l2', 0):.6f}",
            f"  预测 YS/FS（data1123）: {k['ys_pred']:.4f} / {k['fs_pred']:.6f}",
            f"  tem / sr: {k['testenv']['tem']:.4f} / {k['testenv']['sr']:.4f}",
            f"  Ti 余量 wt%: {k['ti_balance_wt_pct']:.4f}",
            "",
        ])
    lines.append("【输出文件】")
    for key, val in summary.get("paths", {}).items():
        lines.append(f"  {key}: {val}")
    lines.extend(["", "=" * 72])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_pareto_scatter(path: Path, summary: Dict[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 未安装，跳过散点图输出（%s）", path)
        return

    inds = summary.get("individuals", [])
    if not inds:
        logger.warning("帕累托前沿为空，跳过散点图输出（%s）", path)
        return
    f1 = [d["f1_ys_shortfall"] for d in inds]
    f2 = [d["f2_fs_shortfall"] for d in inds]
    plt.figure(figsize=(6, 5))
    plt.scatter(f1, f2, c="steelblue", alpha=0.8)
    knee = summary.get("knee_index", 0)
    if inds:
        plt.scatter([f1[knee]], [f2[knee]], c="red", marker="*", s=120, label="knee")
    plt.xlabel("欠达标 YS (f1)")
    plt.ylabel("欠达标 FS (f2)")
    plt.title("Pareto front (f1 vs f2)")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120)
    plt.close()
