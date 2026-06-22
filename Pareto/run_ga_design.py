#!/usr/bin/env python3
"""
run_ga_design.py — NSGA-II 帕累托遗传逆设计 CLI 入口。

用法（metalForTi 根目录）:
  python -m Pareto.run_ga_design --target-ys <float> --target-fs <float>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import torch

from preprocess.preprocess_datagnn_repro import (
    label_means_from_arrays,
    load_testenv_stats_np,
    resolve_user_targets,
)
from grd.feature_layout import bounds_from_train_x, build_projector
from grd.io_utils import load_dual_rgat, load_graph_bundle, merge_hetero_edges
from Pareto.ga_archive import ArchiveEntry, GeneArchive
from Pareto.ga_evaluate import FitnessEvaluator, FitnessResult
from Pareto.ga_log_format import format_generation_block
from Pareto.ga_graph import GraphContext
from Pareto.ga_nsga2 import (
    Individual,
    assign_rank_and_crowding,
    environmental_selection,
    get_pareto_front,
    pareto_representative,
    tournament_select,
)
from Pareto.ga_compile import compile_genome
from Pareto.ga_operators import GAConfig, crossover_and_mutate
from Pareto.ga_report import (
    OutputRestoreContext,
    build_archive_summary,
    write_ga_summary_txt,
    write_pareto_json,
    write_pareto_scatter,
)

logger = logging.getLogger("Pareto.run_ga_design")


def _entry_to_individual(entry: ArchiveEntry, evaluator: FitnessEvaluator) -> Individual:
    return Individual(
        genome=entry.genome.clone(),
        fitness=entry.fitness,
        objectives=evaluator.objectives_tensor(entry.fitness),
    )


def _entries_to_individuals(
    entries: List[ArchiveEntry],
    evaluator: FitnessEvaluator,
) -> List[Individual]:
    return [_entry_to_individual(entry, evaluator) for entry in entries]


def _find_archive_entry(archive: GeneArchive, ind: Individual) -> Optional[ArchiveEntry]:
    g = ind.genome.detach().cpu()
    for e in archive.entries:
        if torch.allclose(e.genome.cpu(), g, atol=1e-5, rtol=0):
            return e
    return None


def _log_generation(
    archive: GeneArchive,
    gen_label: str,
    evaluator: FitnessEvaluator,
    population: List[Individual],
    *,
    target_ys: float,
    target_fs: float,
    restore: OutputRestoreContext | None = None,
    ys_fs_from_labels: bool = False,
    new_virtual_count: int = 0,
) -> None:
    """打印当前代种群帕累托代表与前沿规模。"""
    if not population:
        logger.warning("%s: 种群为空", gen_label)
        return

    rep = pareto_representative(population)
    if rep is None or rep.fitness is None:
        return

    front = get_pareto_front(population)
    entry = _find_archive_entry(archive, rep)
    gene_source = entry.source_label() if entry is not None else "当前种群帕累托代表"

    block = format_generation_block(
        gen_label,
        len(front),
        rep.genome,
        rep.fitness,
        target_ys=target_ys,
        target_fs=target_fs,
        ys_fs_from_labels=ys_fs_from_labels,
        archive_size=archive.size(),
        new_virtual_count=new_virtual_count,
        gene_source=gene_source,
        virtual_genome=None,
        virtual_fitness=None,
        restore=restore,
    )
    logger.info("%s", block)


def _resolve_device(requested: str, force_cpu: bool) -> str:
    if force_cpu:
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，已回退到 CPU")
        return "cpu"
    return requested


def _resolve_targets(
    target_ys: float,
    target_fs: float,
    ys: torch.Tensor,
    fs: torch.Tensor,
    *,
    targets_physical: bool,
):
    """data1123 原始目标 → 模型量纲 + 物理量。"""
    ys_mean, fs_mean = label_means_from_arrays(ys.cpu().numpy(), fs.cpu().numpy())
    resolved = resolve_user_targets(
        target_ys,
        target_fs,
        ys_mean,
        fs_mean,
        targets_physical=targets_physical,
    )
    return (
        resolved.ys_model,
        resolved.fs_model,
        resolved.ys_physical,
        resolved.fs_data1123,
        resolved.ys_mean,
        resolved.fs_mean,
    )


def _resolve_testenv_stats_path(data_dir: Path, root: Path) -> Path:
    for p in (
        data_dir.parent.parent / "datacsv" / "testenv_stats.csv",
        root / "gnnDir" / "datacsv" / "testenv_stats.csv",
    ):
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"testenv_stats.csv not found near {data_dir}. "
        "Run gnnDir/build_datagnn.py first."
    )


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="NSGA-II 帕累托遗传逆设计")
    p.add_argument("--target-ys", type=float, required=True, help="目标 YS（与 data1123.csv 同量纲，MPa）")
    p.add_argument("--target-fs", type=float, required=True, help="目标 FS（与 data1123.csv 的 FS 列同量纲）")
    p.add_argument(
        "--targets-physical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="target-ys/fs 为 data1123 原始量纲，自动换算为模型量纲（默认开启；--no-targets-physical 表示已是 ys.pt/fs.pt 量纲）",
    )
    p.add_argument("--data-dir", type=Path, default=root / "gnnDir" / "gnndataPT" / "r-gatPT")
    p.add_argument(
        "--ckpt",
        type=Path,
        default=root / "modelAll" / "runs" / "best_rgat_full.pt",
    )
    p.add_argument("--rgat-dir", type=Path, default=root / "modelAll")
    p.add_argument("--out-dir", type=Path, default=root / "Pareto" / "outputs_ga")
    p.add_argument("--pop-size", type=int, default=604, help="NSGA-II 种群规模（每代子代数）")
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
    parents: List[Individual],
    pop_size: int,
    x_train: torch.Tensor,
    bounds,
    projector,
    rng: torch.Generator,
    ga_cfg: GAConfig,
) -> List[torch.Tensor]:
    """NSGA-II 二元锦标赛选父，产出 pop_size 个子代基因组。"""
    assign_rank_and_crowding(parents)
    children: List[torch.Tensor] = []
    while len(children) < pop_size:
        p1 = tournament_select(parents, rng)
        p2 = tournament_select(parents, rng)
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


def _evaluate_offspring(
    genomes: List[torch.Tensor],
    evaluator: FitnessEvaluator,
) -> List[Individual]:
    out: List[Individual] = []
    for g in genomes:
        fit = evaluator.evaluate_one(g)
        out.append(
            Individual(
                genome=g,
                fitness=fit,
                objectives=evaluator.objectives_tensor(fit),
            )
        )
    return out


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    device = _resolve_device(args.device, args.force_cpu)
    rng = torch.Generator().manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("使用设备: %s | 算法: NSGA-II（非支配排序 + 拥挤距离）", device)
    root = Path(__file__).resolve().parents[1]
    x, ys, fs, train_mask, _ = load_graph_bundle(args.data_dir)
    targets_physical = args.targets_physical
    (
        target_ys,
        target_fs,
        target_ys_phys,
        target_fs_phys,
        ys_mean,
        fs_mean,
    ) = _resolve_targets(
        args.target_ys,
        args.target_fs,
        ys,
        fs,
        targets_physical=targets_physical,
    )
    te_mean, te_std = load_testenv_stats_np(_resolve_testenv_stats_path(args.data_dir, root))
    restore = OutputRestoreContext(
        ys_mean=ys_mean,
        fs_mean=fs_mean,
        te_mean=te_mean,
        te_std=te_std,
        target_ys_physical=target_ys_phys,
        target_fs_physical=target_fs_phys,
    )
    if targets_physical:
        logger.info(
            "目标（data1123）YS=%.4f FS=%.6f → 模型量纲 YS=%.6f FS=%.6f "
            "(全表均值 YS=%.4f FS_dataOri2=%.4f)",
            target_ys_phys,
            target_fs_phys,
            target_ys,
            target_fs,
            ys_mean,
            fs_mean,
        )
    else:
        logger.info(
            "目标（模型量纲）YS=%.6f FS=%.6f | 对应 data1123 YS=%.4f FS=%.6f",
            target_ys,
            target_fs,
            target_ys_phys,
            target_fs_phys,
        )
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
        target_ys,
        target_fs,
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
    original_parents = _entries_to_individuals(archive.original_entries(), evaluator)
    population: List[Individual] = []

    logger.info(
        "基因库已初始化：%d 原始节点 | 种群规模 %d | 目标 %s",
        n_orig,
        args.pop_size,
        args.objectives,
    )
    _log_generation(
        archive,
        "代 0（原始池就绪，标签适应度）",
        evaluator,
        original_parents[: args.pop_size] if len(original_parents) > args.pop_size else original_parents,
        target_ys=target_ys,
        target_fs=target_fs,
        restore=restore,
        ys_fs_from_labels=True,
    )

    for gen in range(1, args.generations + 1):
        parent_pool = original_parents if not population else population
        pool_label = "604 原始（标签）" if not population else f"种群 {len(population)}"
        logger.info("代 %d 父本: %s | 二元锦标赛选父", gen, pool_label)

        children_genomes = _make_offspring(
            parent_pool,
            args.pop_size,
            x_train,
            bounds,
            projector,
            rng,
            ga_cfg,
        )
        offspring = _evaluate_offspring(children_genomes, evaluator)
        fitness_list: List[FitnessResult] = [ind.fitness for ind in offspring if ind.fitness is not None]
        archive.add_virtual_batch(children_genomes, fitness_list, generation=gen)

        if not population:
            population = environmental_selection(offspring, args.pop_size)
        else:
            population = environmental_selection(population + offspring, args.pop_size)

        front = get_pareto_front(population)
        logger.info(
            "代 %d 环境选择完成：种群 %d，帕累托前沿 %d",
            gen,
            len(population),
            len(front),
        )
        _log_generation(
            archive,
            f"代 {gen}（NSGA-II 环境选择后）",
            evaluator,
            population,
            target_ys=target_ys,
            target_fs=target_fs,
            restore=restore,
            new_virtual_count=args.pop_size,
        )

    front = get_pareto_front(population)
    logger.info(
        "完成。基因库 %d（原始 %d + 虚拟 %d），最终种群帕累托前沿 %d 个体",
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
        target_ys=target_ys,
        target_fs=target_fs,
        target_ys_physical=target_ys_phys,
        target_fs_physical=target_fs_phys,
        targets_physical=targets_physical,
        label_means={"ys": ys_mean, "fs": fs_mean},
        restore=restore,
        objectives=args.objectives,
        offspring_per_generation=args.pop_size,
        generations=args.generations,
        device=device,
        paths=paths,
        selection_method="NSGA-II",
    )
    write_pareto_json(args.out_dir / "pareto_front.json", summary)
    write_ga_summary_txt(args.out_dir / "ga_summary.txt", summary)
    write_pareto_scatter(args.out_dir / "pareto_scatter.png", summary)
    logger.info("已写入 %s", args.out_dir)


if __name__ == "__main__":
    main()
