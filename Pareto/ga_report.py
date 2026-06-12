"""
ga_report.py — 帕累托前沿 JSON/TXT 报告与散点图。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from Pareto.ga_archive import GeneArchive, weighted_score
from Pareto.ga_evaluate import FitnessResult
from Pareto.ga_nsga2 import Individual

FIELD_DESCRIPTIONS_CN: Dict[str, str] = {
    "generated_at_utc": "报告生成时间（UTC）",
    "target_ys": "用户目标屈服强度 YS",
    "target_fs": "用户目标 FS",
    "objectives": "优化目标模式：three 或 two",
    "population_size": "最终种群规模",
    "archive_size": "基因库总条目数",
    "num_original": "原始图节点条目数",
    "num_virtual": "累积虚拟个体条目数",
    "offspring_per_generation": "每代新增虚拟子代数",
    "pareto_front_size": "第一非支配层个体数",
    "individuals": "帕累托前沿个体列表",
    "genome_30d": "30 维基因组（element+testenv+coldway）",
    "element_wt_pct": "10 元含量 wt%",
    "ti_balance_wt_pct": "钛余量 wt%",
    "testenv_z": "试验环境 tem/fcr z-score",
    "coldway_3x6": "coldway 重塑为 3×6",
    "f1_ys_abs_err": "|预测 YS - 目标 YS|",
    "f2_fs_abs_err": "|预测 FS - 目标 FS|",
    "f3_anchor_l2": "与训练集最近邻的 L2 距离",
    "ys_pred": "GNN 预测 YS",
    "fs_pred": "GNN 预测 FS",
    "nearest_train_idx": "最近邻训练样本在原始图中的节点 id（0 基准，对应 material_graph 节点序号）",
    "knee_index": "加权和折中解在 individuals 中的索引",
    "best_virtual_weighted_score": "虚拟（GNN 预测）节点中的最优加权分",
    "virtual_pareto_front_size": "仅含 GA 设计（虚拟）节点的帕累托前沿个体数",
    "virtual_knee_index": "虚拟节点折中解在 virtual_individuals 中的索引",
    "virtual_individuals": "仅含 GA 设计（虚拟）节点的帕累托前沿列表",
    "field_descriptions": "字段中文说明",
}


def _individual_to_dict(
    genome: torch.Tensor,
    fit: FitnessResult,
) -> Dict[str, Any]:
    g = genome.detach().cpu().float()
    elem = {name: float(g[i].item()) for i, name in enumerate(ELEMENT_NAMES)}
    ti = float(compute_ti_balance(g.unsqueeze(0), DEFAULT_TOTAL_WT)[0].item())
    cw = g[COLDWAY_SLICE].reshape(3, 6).tolist()
    return {
        "genome_30d": g.tolist(),
        "element_wt_pct": elem,
        "ti_balance_wt_pct": ti,
        "testenv_z": g[TESTENV_SLICE].tolist(),
        "coldway_3x6": cw,
        "f1_ys_abs_err": fit.f1,
        "f2_fs_abs_err": fit.f2,
        "f3_anchor_l2": fit.f3,
        "ys_pred": fit.ys_pred,
        "fs_pred": fit.fs_pred,
        "nearest_train_idx": fit.nearest_train_idx,
    }


def find_knee_index(individuals: List[Dict[str, Any]]) -> int:
    """返回加权和最小个体的索引（f1 + f2 + 0.1*f3 最小）。

    注意：这是加权标量折中解，而非几何意义上的 knee point。
    JSON 输出中的 knee_index 字段沿用此语义。
    """
    if not individuals:
        return 0
    scores = [
        ind["f1_ys_abs_err"] + ind["f2_fs_abs_err"] + 0.1 * ind.get("f3_anchor_l2", 0.0)
        for ind in individuals
    ]
    return int(min(range(len(scores)), key=lambda i: scores[i]))


def build_archive_summary(
    archive: GeneArchive,
    front: List[Individual],
    *,
    virtual_front: Optional[List[Individual]] = None,
    target_ys: float,
    target_fs: float,
    objectives: str,
    offspring_per_generation: int,
    generations: int,
    device: str,
    paths: Dict[str, str],
) -> Dict[str, Any]:
    """从基因库与帕累托前沿构建最终报告。

    virtual_front: 仅含虚拟节点的帕累托前沿（GA 设计出的新合金）。
    当原始节点因标签精确而霸占全库前沿时，virtual_front 提供独立视图，
    让用户看到 GA 真正探索到的新合金设计。
    """
    individuals = []
    for ind in front:
        if ind.fitness is None:
            continue
        individuals.append(_individual_to_dict(ind.genome, ind.fitness))
    knee = find_knee_index(individuals) if individuals else 0

    best = archive.best_entry()
    best_weighted = weighted_score(best.fitness) if best is not None else None

    best_virt = archive.best_virtual_entry()
    best_virtual_weighted = weighted_score(best_virt.fitness) if best_virt is not None else None

    virtual_individuals: List[Dict[str, Any]] = []
    if virtual_front:
        for ind in virtual_front:
            if ind.fitness is None:
                continue
            virtual_individuals.append(_individual_to_dict(ind.genome, ind.fitness))
    virtual_knee = find_knee_index(virtual_individuals) if virtual_individuals else 0

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_ys": target_ys,
        "target_fs": target_fs,
        "objectives": objectives,
        "population_size": offspring_per_generation,
        "offspring_per_generation": offspring_per_generation,
        "generations": generations,
        "device": device,
        "archive_size": archive.size(),
        "num_original": archive.num_original(),
        "num_virtual": archive.num_virtual(),
        "best_weighted_score": best_weighted,
        "best_virtual_weighted_score": best_virtual_weighted,
        "pareto_front_size": len(individuals),
        "knee_index": knee,
        "individuals": individuals,
        "virtual_pareto_front_size": len(virtual_individuals),
        "virtual_knee_index": virtual_knee,
        "virtual_individuals": virtual_individuals,
        "paths": paths,
        "field_descriptions": FIELD_DESCRIPTIONS_CN,
    }


def write_pareto_json(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _fmt_individual_block(ind: Dict[str, Any], title: str) -> List[str]:
    return [
        title,
        f"  f1 (|ΔYS|): {ind['f1_ys_abs_err']:.6f}",
        f"  f2 (|ΔFS|): {ind['f2_fs_abs_err']:.6f}",
        f"  f3 (锚定 L2): {ind.get('f3_anchor_l2', 0):.6f}",
        f"  预测 YS/FS: {ind['ys_pred']:.6f} / {ind['fs_pred']:.6f}",
        f"  Ti 余量 wt%: {ind['ti_balance_wt_pct']:.4f}",
        "",
    ]


def write_ga_summary_txt(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "=" * 72,
        "Pareto 遗传逆设计结果汇总",
        "=" * 72,
        "",
        f"生成时间（UTC）: {summary.get('generated_at_utc', '')}",
        f"目标 YS: {summary.get('target_ys')}",
        f"目标 FS: {summary.get('target_fs')}",
        f"优化目标: {summary.get('objectives')}",
        f"设备: {summary.get('device')}",
        f"进化代数: {summary.get('generations')}",
        f"每代子代数: {summary.get('offspring_per_generation', summary.get('population_size'))}",
        f"基因库规模: {summary.get('archive_size', '—')}（原始 {summary.get('num_original', '—')} + 虚拟 {summary.get('num_virtual', '—')}）",
        f"帕累托前沿（全库）: {summary.get('pareto_front_size')} 个体",
        f"帕累托前沿（仅虚拟）: {summary.get('virtual_pareto_front_size', '—')} 个体",
        "",
    ]
    if summary.get("best_weighted_score") is not None:
        lines.append(f"全库最优加权分: {summary['best_weighted_score']:.6f}")
    if summary.get("best_virtual_weighted_score") is not None:
        lines.append(f"虚拟节点最优加权分: {summary['best_virtual_weighted_score']:.6f}")
    lines.append("")

    inds = summary.get("individuals", [])
    knee = summary.get("knee_index", 0)
    if inds:
        lines.extend(_fmt_individual_block(inds[knee], "【全库折中解（knee）】"))

    virt_inds = summary.get("virtual_individuals", [])
    virt_knee = summary.get("virtual_knee_index", 0)
    if virt_inds:
        lines.extend(_fmt_individual_block(virt_inds[virt_knee], "【虚拟节点折中解（GA 新设计）】"))

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
    f1 = [d["f1_ys_abs_err"] for d in inds]
    f2 = [d["f2_fs_abs_err"] for d in inds]
    plt.figure(figsize=(6, 5))
    plt.scatter(f1, f2, c="steelblue", alpha=0.8)
    knee = summary.get("knee_index", 0)
    if inds:
        plt.scatter([f1[knee]], [f2[knee]], c="red", marker="*", s=120, label="knee")
    plt.xlabel("|ΔYS| (f1)")
    plt.ylabel("|ΔFS| (f2)")
    plt.title("Pareto front (f1 vs f2)")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120)
    plt.close()
