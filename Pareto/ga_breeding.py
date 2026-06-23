"""
ga_breeding.py — 拓展育种池 + 随机移民（--breeder-pool expanded）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from Pareto.ga_breeding_plan import (
    DEFAULT_IMMIGRANT_FRAC,
    DEFAULT_RANDOM_MATE_FRAC,
    DEFAULT_VIRTUAL_BREED_SAMPLE,
    IMMIGRANT_MUT_SCALE,
    split_offspring_counts,
)
from Pareto.ga_archive import GeneArchive, weighted_score
from Pareto.ga_compile import compile_genome
from Pareto.ga_nsga2 import Individual, assign_rank_and_crowding, tournament_select
from Pareto.ga_operators import GAConfig, crossover_and_mutate, mutate_genome
from Pareto.ga_testenv_lock import FixedTestenvContext
from grd.feature_layout import INPUT_DIM, FeatureBounds
from grd.masked_projector import MaskedCompositeProjector

@dataclass
class OffspringPlan:
    genomes: List[torch.Tensor]
    offspring_kinds: List[str]
    is_immigrant: List[bool]
    immigrant_sources: List[Optional[str]]
    n_tournament: int
    n_random_mate: int
    n_immigrant: int
    breeder_pool_size: int


def _genome_fingerprint(genome: torch.Tensor, ndigits: int = 5) -> Tuple[float, ...]:
    return tuple(round(float(v), ndigits) for v in genome.detach().cpu().tolist())


def _dedupe_genomes(genomes: List[torch.Tensor]) -> List[torch.Tensor]:
    seen: set[Tuple[float, ...]] = set()
    out: List[torch.Tensor] = []
    for g in genomes:
        key = _genome_fingerprint(g)
        if key in seen:
            continue
        seen.add(key)
        out.append(g)
    return out


def sample_virtual_genomes_for_breeder(
    archive: GeneArchive,
    n_sample: int,
    rng: torch.Generator,
) -> List[torch.Tensor]:
    """50% 加权 top + 50% 均匀随机（在虚拟条目不足时取全部）。"""
    virtuals = [e for e in archive.entries if not e.is_original]
    if not virtuals or n_sample <= 0:
        return []
    n_sample = min(n_sample, len(virtuals))
    n_top = n_sample // 2
    n_rand = n_sample - n_top
    ranked = sorted(virtuals, key=lambda e: weighted_score(e.fitness))
    top = ranked[:n_top]
    top_set = {id(e) for e in top}
    remaining = [e for e in virtuals if id(e) not in top_set]
    random_pick: List = []
    if n_rand > 0 and remaining:
        perm = torch.randperm(len(remaining), generator=rng)
        take = min(n_rand, len(remaining))
        random_pick = [remaining[int(perm[i].item())] for i in range(take)]
    return [e.genome.clone() for e in top + random_pick]


def build_expanded_breeder_genomes(
    population: List[Individual],
    archive: GeneArchive,
    virtual_sample_size: int,
    rng: torch.Generator,
) -> List[torch.Tensor]:
    """population ∪ 604 原始 ∪ 虚拟抽样（去重，population 优先）。"""
    genomes: List[torch.Tensor] = [ind.genome.clone() for ind in population]
    genomes.extend(e.genome.clone() for e in archive.original_entries())
    genomes.extend(sample_virtual_genomes_for_breeder(archive, virtual_sample_size, rng))
    return _dedupe_genomes(genomes)


def _apply_fixed_to_genome(
    genome: torch.Tensor,
    fixed_testenv: Optional[FixedTestenvContext],
) -> torch.Tensor:
    if fixed_testenv is None:
        return genome
    return fixed_testenv.apply(genome)


def _immigrant_cfg(base: GAConfig) -> GAConfig:
    return GAConfig(
        p_cross=base.p_cross,
        p_mut=min(1.0, base.p_mut * IMMIGRANT_MUT_SCALE),
        p_mut_delta=base.p_mut_delta,
        p_mut_zero=base.p_mut_zero,
        p_mut_activate=base.p_mut_activate,
        mutate_by_stage=base.mutate_by_stage,
        delta_frac=base.delta_frac * IMMIGRANT_MUT_SCALE,
        fixed_testenv=base.fixed_testenv,
    )


def make_immigrant_genome(
    archive: GeneArchive,
    x_train: torch.Tensor,
    bounds: FeatureBounds,
    projector: MaskedCompositeProjector,
    rng: torch.Generator,
    ga_cfg: GAConfig,
) -> Tuple[torch.Tensor, str]:
    """随机移民：40% 原始 / 40% 虚拟 / 20% 训练节点。"""
    imm_cfg = _immigrant_cfg(ga_cfg)
    r = float(torch.rand(1, generator=rng).item())
    originals = archive.original_entries()
    virtuals = [e for e in archive.entries if not e.is_original]

    if r < 0.4 and originals:
        idx = int(torch.randint(0, len(originals), (1,), generator=rng).item())
        g = originals[idx].genome.clone()
        source = "original"
    elif r < 0.8 and virtuals:
        idx = int(torch.randint(0, len(virtuals), (1,), generator=rng).item())
        g = virtuals[idx].genome.clone()
        source = "virtual"
    else:
        j = int(torch.randint(0, x_train.shape[0], (1,), generator=rng).item())
        g = x_train[j].clone()
        source = "train"

    g = _apply_fixed_to_genome(g, ga_cfg.fixed_testenv)
    g = mutate_genome(g, x_train, rng, imm_cfg)
    g = compile_genome(
        g,
        bounds,
        projector,
        x_train,
        rng=rng,
        fixed_testenv=ga_cfg.fixed_testenv,
    )
    return g, source


def _pick_random_pair(
    breeder_genomes: List[torch.Tensor],
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(breeder_genomes)
    if n == 0:
        raise ValueError("育种池为空")
    i1 = int(torch.randint(0, n, (1,), generator=rng).item())
    i2 = int(torch.randint(0, n, (1,), generator=rng).item())
    return breeder_genomes[i1], breeder_genomes[i2]


def make_offspring_expanded(
    parent_pool: List[Individual],
    archive: GeneArchive,
    pop_size: int,
    x_train: torch.Tensor,
    bounds: FeatureBounds,
    projector: MaskedCompositeProjector,
    rng: torch.Generator,
    ga_cfg: GAConfig,
    *,
    fixed_testenv: Optional[FixedTestenvContext] = None,
    virtual_breed_sample: int = DEFAULT_VIRTUAL_BREED_SAMPLE,
    immigrant_frac: float = DEFAULT_IMMIGRANT_FRAC,
    random_mate_frac: float = DEFAULT_RANDOM_MATE_FRAC,
) -> OffspringPlan:
    """拓展育种：锦标赛 + 拓展池随机配对 + 移民，合计 pop_size。"""
    n_tournament, n_random, n_immigrant = split_offspring_counts(
        pop_size,
        immigrant_frac=immigrant_frac,
        random_mate_frac=random_mate_frac,
    )

    genomes: List[torch.Tensor] = []
    kinds: List[str] = []
    is_immigrant: List[bool] = []
    immigrant_sources: List[Optional[str]] = []

    parents_prep = parent_pool
    if fixed_testenv is not None:
        parents_prep = [
            Individual(
                genome=fixed_testenv.apply(ind.genome),
                fitness=ind.fitness,
                objectives=ind.objectives,
                rank=ind.rank,
                crowding=ind.crowding,
            )
            for ind in parent_pool
        ]
    assign_rank_and_crowding(parents_prep)

    tournament_done = 0
    while tournament_done < n_tournament:
        p1 = tournament_select(parents_prep, rng)
        p2 = tournament_select(parents_prep, rng)
        c1, c2 = crossover_and_mutate(
            p1.genome,
            p2.genome,
            x_train,
            bounds,
            projector,
            rng,
            ga_cfg,
        )
        for child in (c1, c2):
            if tournament_done >= n_tournament:
                break
            genomes.append(child)
            kinds.append("tournament")
            is_immigrant.append(False)
            immigrant_sources.append(None)
            tournament_done += 1

    breeder_genomes = build_expanded_breeder_genomes(
        parent_pool,
        archive,
        virtual_breed_sample,
        rng,
    )
    if fixed_testenv is not None:
        breeder_genomes = [fixed_testenv.apply(g) for g in breeder_genomes]

    random_done = 0
    while random_done < n_random:
        g1, g2 = _pick_random_pair(breeder_genomes, rng)
        c1, c2 = crossover_and_mutate(
            g1,
            g2,
            x_train,
            bounds,
            projector,
            rng,
            ga_cfg,
        )
        for child in (c1, c2):
            if random_done >= n_random:
                break
            genomes.append(child)
            kinds.append("random_mate")
            is_immigrant.append(False)
            immigrant_sources.append(None)
            random_done += 1

    for _ in range(n_immigrant):
        g, source = make_immigrant_genome(
            archive,
            x_train,
            bounds,
            projector,
            rng,
            ga_cfg,
        )
        genomes.append(g)
        kinds.append("immigrant")
        is_immigrant.append(True)
        immigrant_sources.append(source)

    assert len(genomes) == pop_size, (len(genomes), pop_size)
    return OffspringPlan(
        genomes=genomes,
        offspring_kinds=kinds,
        is_immigrant=is_immigrant,
        immigrant_sources=immigrant_sources,
        n_tournament=n_tournament,
        n_random_mate=n_random,
        n_immigrant=n_immigrant,
        breeder_pool_size=len(breeder_genomes),
    )
