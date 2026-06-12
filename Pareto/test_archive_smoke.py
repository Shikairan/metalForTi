#!/usr/bin/env python3
"""冒烟：代 0 零 forward，每代仅子代 forward。"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from grd.feature_layout import bounds_from_train_x, build_projector
from grd.io_utils import load_dual_rgat, load_graph_bundle, merge_hetero_edges
from Pareto.ga_archive import GeneArchive, weighted_tournament_select
from Pareto.ga_evaluate import FitnessEvaluator
from Pareto.ga_graph import GraphContext
from Pareto.ga_operators import GAConfig, crossover_and_mutate


class ArchiveForwardCountTest(unittest.TestCase):
    def test_forward_only_on_offspring(self) -> None:
        root = Path(__file__).resolve().parents[1]
        data_dir = root / "gnnDir" / "gnndataPT" / "r-gatPT"
        ckpt = root / "gnnDir" / "gnn" / "r-gatDouble" / "runs" / "best_ysfs_gat.pt"
        rgat_dir = root / "gnnDir" / "gnn" / "r-gatDouble"

        x, ys, fs, train_mask, _ = load_graph_bundle(data_dir)
        graph = torch.load(data_dir / "material_graph.pt", map_location="cpu", weights_only=False)
        edge_index, edge_type = merge_hetero_edges(graph)
        ctx = GraphContext.from_tensors(x, edge_index, edge_type)

        model, _ = load_dual_rgat(ckpt, rgat_dir, "cpu")
        bounds = bounds_from_train_x(x, train_mask)
        projector = build_projector(x, bounds)
        x_train = x[train_mask].clone()

        pop_size = 10
        generations = 2
        target_ys = float(ys.median().item())
        target_fs = float(fs.median().item())

        evaluator = FitnessEvaluator(
            model, ctx, x_train, target_ys, target_fs, "cpu", use_anchor=True
        )
        archive = GeneArchive.from_graph(x, ys, fs, evaluator)
        breeders = archive.select_top_k(pop_size)
        self.assertEqual(archive.size(), 604)

        forward_count = 0
        original_eval = evaluator.evaluate_one

        def counting_eval(genome: torch.Tensor):
            nonlocal forward_count
            forward_count += 1
            return original_eval(genome)

        rng = torch.Generator().manual_seed(42)
        ga_cfg = GAConfig()

        with patch.object(evaluator, "evaluate_one", side_effect=counting_eval):
            self.assertEqual(forward_count, 0)

            for gen in range(1, generations + 1):
                children = []
                while len(children) < pop_size:
                    p1 = weighted_tournament_select(breeders, rng)
                    p2 = weighted_tournament_select(breeders, rng)
                    c1, c2 = crossover_and_mutate(
                        p1.genome, p2.genome, x_train, bounds, projector, rng, ga_cfg
                    )
                    children.append(c1)
                    if len(children) < pop_size:
                        children.append(c2)
                fits = [evaluator.evaluate_one(g) for g in children]
                archive.add_virtual_batch(children, fits, generation=gen)
                breeders = archive.select_top_k(pop_size)

        self.assertEqual(forward_count, pop_size * generations)
        self.assertEqual(archive.size(), 604 + pop_size * generations)


if __name__ == "__main__":
    unittest.main()
