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
from Pareto.ga_archive import ArchiveEntry, GeneArchive, random_pair_select
from Pareto.ga_evaluate import FitnessEvaluator
from Pareto.ga_log_format import format_generation_block
from Pareto.ga_graph import GraphContext
from Pareto.ga_nsga2 import Individual, get_pareto_front
from Pareto.ga_compile import compile_genome
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
    """打印当前代全库最优个体的规整文本块。

    front_candidates: 用于计算帕累托前沿的候选集（默认为当前 breeders 精英池）。
    传入子集而非全库，将 O(n²) 帕累托排序限制在精英池规模，避免每代全库重排序。
    """
    best_entry = archive.best_entry()
    if best_entry is None:
        logger.warning("%s: 无有效个体", gen_label)
        return

    best_virtual = archive.best_virtual_entry()

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
        virtual_genome=best_virtual.genome if best_virtual is not None else None,
        virtual_fitness=best_virtual.fitness if best_virtual is not None else None,
        virtual_gene_source=best_virtual.source_label() if best_virtual is not None else None,
        virtual_same_as_overall=(
            best_virtual is not None and best_virtual is best_entry
        ),
    )
    logger.info("%s", block)


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
        default=root / "modelAll" / "runs" / "best_rgat_full.pt",
    )
    p.add_argument("--rgat-dir", type=Path, default=root / "modelAll")
    p.add_argument("--out-dir", type=Path, default=root / "Pareto" / "outputs_ga")
    p.add_argument("--pop-size", type=int, default=604, help="每代杂交子代数（GNN 评估数）")
    p.add_argument(
        "--virtual-pool-size",
        type=int,
        default=604,
        help="虚拟精英池规模（历史虚拟加权 top-k，与原始池合并为父本池）",
    )
    p.add_argument("--generations", type=int, default=150)
    p.add_argument("--objectives", choices=["two", "three"], default="three")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--p-cross", type=float, default=0.9)
    p.add_argument("--p-mut", type=float, default=0.15)
    p.add_argument("--element-thr", type=float, default=0.8)
    p.add_argument("--testenv-thr", type=float, default=0.8)
    p.add_argument("--coldway-thr", type=float, default=0.8)
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
        p1, p2 = random_pair_select(breeders, rng)
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
    train_node_indices = torch.where(train_mask)[0]

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
        train_node_indices=train_node_indices,
    )

    archive = GeneArchive.from_graph(x, ys, fs, evaluator)

    def _repair_genome(g: torch.Tensor) -> torch.Tensor:
        return compile_genome(g, bounds, projector, x_train, rng=rng)

    n_repaired = archive.repair_all_genomes(_repair_genome)
    logger.info("已对基因库 %d 条基因组执行 coldway/约束 compile 修复", n_repaired)

    n_orig = archive.num_original()
    logger.info(
        "基因库已初始化：%d 原始节点（固定父本池），虚拟精英池上限 %d",
        n_orig,
        args.virtual_pool_size,
    )
    _log_generation(
        archive,
        "代 0（原始池就绪）",
        evaluator,
        target_ys=args.target_ys,
        target_fs=args.target_fs,
        ys_fs_from_labels=True,
        front_candidates=archive.original_entries(),
    )

    for gen in range(1, args.generations + 1):
        breeder_pool = archive.build_breeder_pool(args.virtual_pool_size)
        n_virt_elite = len(breeder_pool) - n_orig
        logger.info(
            "代 %d 父本池：原始 %d + 虚拟精英 %d = %d（随机配对杂交）",
            gen,
            n_orig,
            n_virt_elite,
            len(breeder_pool),
        )
        children_genomes = _make_offspring(
            breeder_pool,
            args.pop_size,
            x_train,
            bounds,
            projector,
            rng,
            ga_cfg,
        )
        fitness_list = [evaluator.evaluate_one(g) for g in children_genomes]
        archive.add_virtual_batch(children_genomes, fitness_list, generation=gen)
        offspring_batch = archive.latest_virtual_batch(args.pop_size)
        _log_generation(
            archive,
            f"代 {gen}（本代 {args.pop_size} 子代已排序入库）",
            evaluator,
            target_ys=args.target_ys,
            target_fs=args.target_fs,
            new_virtual_count=args.pop_size,
            front_candidates=offspring_batch,
        )

    individuals = archive.to_individuals(evaluator)
    front = get_pareto_front(individuals)
    logger.info(
        "完成。基因库 %d（原始 %d + 虚拟 %d），帕累托前沿 %d 个体",
        archive.size(),
        archive.num_original(),
        archive.num_virtual(),
        len(front),
    )

    paths = {
        "pareto_json": str((args.out_dir / "pareto_front.json").resolve()),
        "summary_txt": str((args.out_dir / "ga_summary.txt").resolve()),
        "scatter_png": str((args.out_dir / "pareto_scatter.png").resolve()),
    }
    summary = build_archive_summary(
        archive,
        front,
        target_ys=args.target_ys,
        target_fs=args.target_fs,
        objectives=args.objectives,
        offspring_per_generation=args.pop_size,
        virtual_pool_size=args.virtual_pool_size,
        generations=args.generations,
        device=device,
        paths=paths,
    )
    write_pareto_json(args.out_dir / "pareto_front.json", summary)
    write_ga_summary_txt(args.out_dir / "ga_summary.txt", summary)
    write_pareto_scatter(args.out_dir / "pareto_scatter.png", summary)
    logger.info("已写入 %s", args.out_dir)


if __name__ == "__main__":
    main()
