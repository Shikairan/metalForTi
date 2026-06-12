"""
breed_combo.py — 从父本池随机配对杂交生成子代基因组。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from grd.feature_layout import FeatureBounds
from grd.masked_projector import MaskedCompositeProjector
from Pareto.ga_archive import ArchiveEntry, GeneArchive, random_pair_select
from Pareto.ga_operators import GAConfig, crossover_and_mutate


def make_offspring_from_pool(
    breeder_pool: List[ArchiveEntry],
    num_offspring: int,
    x_train: torch.Tensor,
    bounds: FeatureBounds,
    projector: MaskedCompositeProjector,
    rng: torch.Generator,
    ga_cfg: Optional[GAConfig] = None,
) -> Tuple[List[torch.Tensor], List[Tuple[str, str]]]:
    """
    父本池内随机配对杂交，返回子代基因组及父本来源标签对。

    返回:
        genomes: 长度 num_offspring
        parent_pairs: 与 genomes 等长的 (parent1_label, parent2_label)
    """
    if not breeder_pool:
        raise ValueError("父本池为空")
    cfg = ga_cfg or GAConfig()
    genomes: List[torch.Tensor] = []
    pairs: List[Tuple[str, str]] = []

    while len(genomes) < num_offspring:
        p1, p2 = random_pair_select(breeder_pool, rng)
        c1, c2 = crossover_and_mutate(
            p1.genome,
            p2.genome,
            x_train,
            bounds,
            projector,
            rng,
            cfg,
        )
        genomes.append(c1)
        pairs.append((p1.source_label(), p2.source_label()))
        if len(genomes) < num_offspring:
            genomes.append(c2)
            pairs.append((p1.source_label(), p2.source_label()))

    return genomes, pairs


def build_parent_pool(
    archive: GeneArchive,
    virtual_pool_size: int,
) -> List[ArchiveEntry]:
    """双池父本：604 原始（固定）+ 历史虚拟 top-k。"""
    return archive.build_breeder_pool(virtual_pool_size)
