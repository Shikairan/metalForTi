#!/usr/bin/env python3
"""
run_combo_test.py — 父本池随机杂交，对合法子代 GNN forward，只看 ys / fs。

用法（metalForTi 根目录）:
  python -m Pareto.testnode.run_combo_test
  python -m Pareto.testnode.run_combo_test --no-forward   # 仅合法性校验
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from grd.feature_layout import bounds_from_train_x, build_projector
from grd.io_utils import load_dual_rgat, load_graph_bundle, merge_hetero_edges
from Pareto.ga_archive import GeneArchive
from Pareto.ga_compile import compile_genome
from Pareto.ga_evaluate import FitnessEvaluator, FitnessResult
from Pareto.ga_graph import GraphContext
from Pareto.ga_operators import GAConfig
from Pareto.testnode.breed_combo import build_parent_pool, make_offspring_from_pool
from Pareto.testnode.validate_genome import validate_genome, validate_batch

logger = logging.getLogger("Pareto.testnode.run_combo_test")


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="随机杂交子代 YS/FS 预测")
    p.add_argument("--data-dir", type=Path, default=root / "gnnDir" / "gnndataPT" / "r-gatPT")
    p.add_argument(
        "--ckpt",
        type=Path,
        default=root / "modelAll" / "runs" / "best_rgat_full.pt",
    )
    p.add_argument("--rgat-dir", type=Path, default=root / "modelAll")
    p.add_argument("--out-dir", type=Path, default=root / "Pareto" / "testnode" / "outputs")
    p.add_argument("--num-offspring", type=int, default=604)
    p.add_argument("--virtual-pool-size", type=int, default=604)
    p.add_argument("--mock-virtual", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--p-cross", type=float, default=0.9)
    p.add_argument("--p-mut", type=float, default=0.15)
    p.add_argument("--no-forward", action="store_true")
    p.add_argument("--max-forward", type=int, default=0, help="0=全部合法子代")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force-cpu", action="store_true")
    return p.parse_args()


def _resolve_device(requested: str, force_cpu: bool) -> str:
    if force_cpu or not torch.cuda.is_available():
        return "cpu"
    return requested


def _inject_mock_virtual(
    archive: GeneArchive,
    n: int,
    x_train: torch.Tensor,
    bounds,
    projector,
    rng: torch.Generator,
) -> None:
    genomes: List[torch.Tensor] = []
    for _ in range(n):
        j = int(torch.randint(0, x_train.shape[0], (1,), generator=rng).item())
        g = x_train[j].clone() + torch.randn_like(x_train[j]) * 0.08
        genomes.append(compile_genome(g, bounds, projector, x_train, rng=rng))
    fits = [
        FitnessResult(f1=0.0, f2=0.0, f3=0.0, ys_pred=0.0, fs_pred=0.0, nearest_train_idx=0)
        for _ in range(n)
    ]
    archive.add_virtual_batch(genomes, fits, generation=0)


def _stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    t = torch.tensor(list(values), dtype=torch.float64)
    return {
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "median": float(t.median().item()),
    }


def _forward_ys_fs(
    evaluator: FitnessEvaluator,
    offspring: List[torch.Tensor],
    parent_pairs: List[tuple[str, str]],
    indices: List[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    total = len(indices)
    for k, i in enumerate(indices):
        fit = evaluator.evaluate_one(offspring[i])
        rows.append(
            {
                "index": i,
                "ys": fit.ys_pred,
                "fs": fit.fs_pred,
                "parent_a": parent_pairs[i][0],
                "parent_b": parent_pairs[i][1],
            }
        )
        if total >= 100 and (k + 1) % 100 == 0:
            logger.info("  forward %d / %d", k + 1, total)
    return rows


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    device = _resolve_device(args.device, args.force_cpu)
    rng = torch.Generator().manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    x, ys, fs, train_mask, _ = load_graph_bundle(args.data_dir)
    bounds = bounds_from_train_x(x, train_mask)
    projector = build_projector(x, bounds)
    x_train = x[train_mask].clone()

    graph = torch.load(args.data_dir / "material_graph.pt", map_location="cpu", weights_only=False)
    edge_index, edge_type = merge_hetero_edges(graph)
    ctx = GraphContext.from_tensors(x, edge_index, edge_type)

    class _LabelOnlyModel(torch.nn.Module):
        def forward(self, x_aug, ei, et):
            raise RuntimeError("testnode 不用于标签 forward")

    archive = GeneArchive.from_graph(
        x,
        ys,
        fs,
        FitnessEvaluator(_LabelOnlyModel(), ctx, x_train, 0.0, 0.0, "cpu"),
    )
    archive.repair_all_genomes(
        lambda g: compile_genome(g, bounds, projector, x_train, rng=rng)
    )

    if args.mock_virtual > 0:
        _inject_mock_virtual(archive, args.mock_virtual, x_train, bounds, projector, rng)

    pool = build_parent_pool(archive, args.virtual_pool_size)
    offspring, parent_pairs = make_offspring_from_pool(
        pool,
        args.num_offspring,
        x_train,
        bounds,
        projector,
        rng,
        GAConfig(p_cross=args.p_cross, p_mut=args.p_mut),
    )

    n_ok, n_bad, bad_idx = validate_batch(offspring, bounds)
    logger.info("子代 %d 条：合法 %d，非法 %d", len(offspring), n_ok, n_bad)
    if bad_idx:
        for i in bad_idx[:5]:
            logger.error("  非法 #%d: %s", i, validate_genome(offspring[i], bounds).summary())
        raise SystemExit(1)

    valid_indices = [i for i in range(len(offspring)) if i not in bad_idx]
    if args.max_forward > 0:
        valid_indices = valid_indices[: args.max_forward]

    rows: List[Dict[str, Any]] = []
    if not args.no_forward:
        model, _ = load_dual_rgat(args.ckpt, args.rgat_dir, device)
        evaluator = FitnessEvaluator(model, ctx, x_train, 0.0, 0.0, device, use_anchor=False)
        logger.info("GNN forward %d 条合法子代（%s）", len(valid_indices), device)
        rows = _forward_ys_fs(evaluator, offspring, parent_pairs, valid_indices)

        ys_s = _stats([r["ys"] for r in rows])
        fs_s = _stats([r["fs"] for r in rows])
        logger.info("YS  mean=%.4f  min=%.4f  max=%.4f", ys_s["mean"], ys_s["min"], ys_s["max"])
        logger.info("FS  mean=%.4f  min=%.4f  max=%.4f", fs_s["mean"], fs_s["min"], fs_s["max"])

    out_json = args.out_dir / "combo_ys_fs.json"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "count": len(rows),
        "ys": _stats([r["ys"] for r in rows]),
        "fs": _stats([r["fs"] for r in rows]),
        "nodes": rows,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已写入 %s", out_json)

    if rows:
        out_csv = args.out_dir / "combo_ys_fs.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["index", "ys", "fs", "parent_a", "parent_b"])
            w.writeheader()
            w.writerows(rows)
        logger.info("已写入 %s", out_csv)


if __name__ == "__main__":
    main()
