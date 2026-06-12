#!/usr/bin/env python3
"""
run_ga_design.py — Pareto 遗传逆设计 CLI 入口。

用法（metalForTi 根目录）:
  python -m Pareto.run_ga_design --target-ys <float> --target-fs <float>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import torch

from grd.feature_layout import bounds_from_train_x, build_projector
from grd.io_utils import load_dual_rgat, load_graph_bundle, merge_hetero_edges
from Pareto.ga_archive import ArchiveEntry, GeneArchive, weighted_tournament_select
from Pareto.ga_evaluate import FitnessEvaluator
from Pareto.ga_log_format import format_generation_block
from Pareto.ga_graph import GraphContext
from Pareto.ga_nsga2 import Individual, get_pareto_front
from Pareto.ga_operators import GAConfig, crossover_and_mutate
from Pareto.ga_report import (
    build_archive_summary,
    write_ga_summary_txt,
    write_pareto_json,
    write_pareto_scatter,
)

logger = logging.getLogger("Pareto.run_ga_design")


def _entries_to_individuals(
    entries: List[ArchiveEntry],
    evaluator: FitnessEvaluator,
) -> List[Individual]:
    """将 ArchiveEntry 列表转换为带 objectives 的 Individual 列表。"""
    return [
        Individual(
            genome=e.genome.clone(),
            fitness=e.fitness,
            objectives=evaluator.objectives_tensor(e.fitness),
        )
        for e in entries
    ]


def _log_generation(
    archive: GeneArchive,
    gen_label: str,
    evaluator: FitnessEvaluator,
    *,
    target_ys: float,
    target_fs: float,
    ys_fs_from_labels: bool = False,
    new_virtual_count: int = 0,
    front_candidates: Optional[List[ArchiveEntry]] = None,
) -> None:
    """打印当前代最优个体的规整文本块，并单独显示虚拟节点最优。

    front_candidates: 用于计算帕累托前沿的候选集（混合父本池）。
    传入子集而非全库，将 O(n²) 帕累托排序限制在精英池规模。
    """
    best_entry = archive.best_entry()
    if best_entry is None:
        logger.warning("%s: 无有效个体", gen_label)
        return

    if front_candidates is not None:
        individuals = _entries_to_individuals(front_candidates, evaluator)
    else:
        individuals = archive.to_individuals(evaluator)
    front = get_pareto_front(individuals)

    block = format_generation_block(
        gen_label,
        len(front),
        best_entry.genome,
        best_entry.fitness,
        target_ys=target_ys,
        target_fs=target_fs,
        ys_fs_from_labels=ys_fs_from_labels,
        archive_size=archive.size(),
        new_virtual_count=new_virtual_count,
        gene_source=best_entry.source_label(),
    )
    logger.info("%s", block)

    # 单独显示虚拟节点中的最优，让 GA 设计进展独立可见
    best_virt = archive.best_virtual_entry()
    if best_virt is not None:
        virt_block = format_generation_block(
            f"{gen_label} [虚拟最优]",
            front_size=0,
            genome=best_virt.genome,
            fitness=best_virt.fitness,
            target_ys=target_ys,
            target_fs=target_fs,
            ys_fs_from_labels=False,
            gene_source=best_virt.source_label(),
        )
        logger.info("%s", virt_block)


def _resolve_device(requested: str, force_cpu: bool) -> str:
    if force_cpu:
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，已回退到 CPU")
        return "cpu"
    return requested


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Pareto 基因库累积遗传逆设计")
    p.add_argument("--target-ys", type=float, required=True, help="目标 YS（与 ys.pt 同量纲）")
    p.add_argument("--target-fs", type=float, required=True, help="目标 FS（与 fs.pt 同量纲）")
    p.add_argument("--data-dir", type=Path, default=root / "gnnDir" / "gnndataPT" / "r-gatPT")
    p.add_argument(
        "--ckpt",
        type=Path,
        default=root / "gnnDir" / "gnn" / "r-gatDouble" / "runs" / "best_ysfs_gat.pt",
    )
    p.add_argument("--rgat-dir", type=Path, default=root / "gnnDir" / "gnn" / "r-gatDouble")
    p.add_argument("--out-dir", type=Path, default=root / "Pareto" / "outputs_ga")
    p.add_argument("--pop-size", type=int, default=60, help="每代子代数 / 父本池规模")
    p.add_argument("--generations", type=int, default=150)
    p.add_argument("--objectives", choices=["two", "three"], default="three")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--p-cross", type=float, default=0.9)
    p.add_argument("--p-mut", type=float, default=0.15)
    p.add_argument("--element-thr", type=float, default=0.8)
    p.add_argument("--testenv-thr", type=float, default=0.8)
    p.add_argument("--coldway-thr", type=float, default=0.8)
    p.add_argument(
        "--orig-breeder-ratio",
        type=float,
        default=0.3,
        help=(
            "父本池中原始节点的占比（0~1）。"
            "原始节点以实验标签评估（无预测误差），天然优于 GNN 预测的虚拟节点，"
            "设置此比例可保证虚拟节点有繁殖配额，让 GA 真正进化出新设计。"
            "设为 1.0 等价于原全原始选择；设为 0.0 则仅用虚拟节点（第 0 代除外）。"
        ),
    )
    p.add_argument("--force-cpu", action="store_true")
    return p.parse_args()


def _make_offspring(
    breeders: List[ArchiveEntry],
    pop_size: int,
    x_train: torch.Tensor,
    bounds,
    projector,
    rng: torch.Generator,
    ga_cfg: GAConfig,
) -> List[torch.Tensor]:
    children: List[torch.Tensor] = []
    while len(children) < pop_size:
        p1 = weighted_tournament_select(breeders, rng)
        # 当父本池规模 > 1 时，至多重试一次以避免自交
        p2 = weighted_tournament_select(breeders, rng)
        if len(breeders) > 1 and p2 is p1:
            p2 = weighted_tournament_select(breeders, rng)
        c1, c2 = crossover_and_mutate(
            p1.genome,
            p2.genome,
            x_train,
            bounds,
            projector,
            rng,
            ga_cfg,
        )
        children.append(c1)
        if len(children) < pop_size:
            children.append(c2)
    return children


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    device = _resolve_device(args.device, args.force_cpu)
    rng = torch.Generator().manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("使用设备: %s", device)
    x, ys, fs, train_mask, _ = load_graph_bundle(args.data_dir)
    graph = torch.load(args.data_dir / "material_graph.pt", map_location="cpu", weights_only=False)
    edge_index, edge_type = merge_hetero_edges(graph)
    ctx = GraphContext.from_tensors(x, edge_index, edge_type)

    model, _ = load_dual_rgat(args.ckpt, args.rgat_dir, device)
    bounds = bounds_from_train_x(x, train_mask)
    projector = build_projector(x, bounds)
    x_train = x[train_mask].clone()

    ga_cfg = GAConfig(p_cross=args.p_cross, p_mut=args.p_mut)
    use_anchor = args.objectives == "three"
    evaluator = FitnessEvaluator(
        model,
        ctx,
        x_train,
        args.target_ys,
        args.target_fs,
        device,
        use_anchor=use_anchor,
        element_thr=args.element_thr,
        testenv_thr=args.testenv_thr,
        coldway_thr=args.coldway_thr,
    )

    orig_ratio = max(0.0, min(1.0, args.orig_breeder_ratio))
    archive = GeneArchive.from_graph(x, ys, fs, evaluator)
    # 代 0 尚无虚拟节点，select_top_k_mixed 自动回退到全原始
    breeders = archive.select_top_k_mixed(args.pop_size, orig_ratio)
    logger.info(
        "基因库已初始化：%d 原始节点，代 0 选 top %d 父本（原始占比 %.0f%%，无 GNN forward）",
        archive.size(),
        len(breeders),
        orig_ratio * 100,
    )
    _log_generation(
        archive,
        "代 0（标签选优）",
        evaluator,
        target_ys=args.target_ys,
        target_fs=args.target_fs,
        ys_fs_from_labels=True,
        front_candidates=breeders,
    )

    for gen in range(1, args.generations + 1):
        children_genomes = _make_offspring(
            breeders,
            args.pop_size,
            x_train,
            bounds,
            projector,
            rng,
            ga_cfg,
        )
        fitness_list = [evaluator.evaluate_one(g) for g in children_genomes]
        archive.add_virtual_batch(children_genomes, fitness_list, generation=gen)
        # 混合父本池：orig_ratio 比例来自原始节点，其余来自虚拟节点
        breeders = archive.select_top_k_mixed(args.pop_size, orig_ratio)
        _log_generation(
            archive,
            f"代 {gen}",
            evaluator,
            target_ys=args.target_ys,
            target_fs=args.target_fs,
            new_virtual_count=args.pop_size,
            front_candidates=breeders,
        )

    # 全库帕累托前沿（含原始 + 虚拟）
    individuals = archive.to_individuals(evaluator)
    front = get_pareto_front(individuals)

    # 虚拟节点专属帕累托前沿（GA 真正设计出的新合金）
    virtual_entries = [e for e in archive.entries if not e.is_original]
    virtual_individuals = _entries_to_individuals(virtual_entries, evaluator)
    virtual_front = get_pareto_front(virtual_individuals) if virtual_individuals else []

    logger.info(
        "完成。基因库 %d（原始 %d + 虚拟 %d），全库帕累托 %d，虚拟专属帕累托 %d",
        archive.size(),
        archive.num_original(),
        archive.num_virtual(),
        len(front),
        len(virtual_front),
    )

    paths = {
        "pareto_json": str((args.out_dir / "pareto_front.json").resolve()),
        "virtual_pareto_json": str((args.out_dir / "virtual_pareto_front.json").resolve()),
        "summary_txt": str((args.out_dir / "ga_summary.txt").resolve()),
        "scatter_png": str((args.out_dir / "pareto_scatter.png").resolve()),
        "virtual_scatter_png": str((args.out_dir / "virtual_pareto_scatter.png").resolve()),
    }
    summary = build_archive_summary(
        archive,
        front,
        virtual_front=virtual_front,
        target_ys=args.target_ys,
        target_fs=args.target_fs,
        objectives=args.objectives,
        offspring_per_generation=args.pop_size,
        generations=args.generations,
        device=device,
        paths=paths,
    )
    write_pareto_json(args.out_dir / "pareto_front.json", summary)
    write_ga_summary_txt(args.out_dir / "ga_summary.txt", summary)
    write_pareto_scatter(args.out_dir / "pareto_scatter.png", summary)

    # 仅含虚拟节点的报告与散点图
    if virtual_front:
        virtual_summary = build_archive_summary(
            archive,
            virtual_front,
            virtual_front=virtual_front,
            target_ys=args.target_ys,
            target_fs=args.target_fs,
            objectives=args.objectives,
            offspring_per_generation=args.pop_size,
            generations=args.generations,
            device=device,
            paths=paths,
        )
        write_pareto_json(args.out_dir / "virtual_pareto_front.json", virtual_summary)
        write_pareto_scatter(args.out_dir / "virtual_pareto_scatter.png", virtual_summary)

    logger.info("已写入 %s", args.out_dir)


if __name__ == "__main__":
    main()
