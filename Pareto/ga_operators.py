"""
ga_operators.py — 分段交叉、三种标量变异、种群初始化。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from grd.feature_layout import (
    COLDWAY_DIM,
    COLDWAY_SLICE,
    ELEMENT_DIM,
    ELEMENT_SLICE,
    INPUT_DIM,
    TESTENV_DIM,
    TESTENV_SLICE,
)
from Pareto.ga_compile import compile_genome
from grd.feature_layout import FeatureBounds
from grd.masked_projector import MaskedCompositeProjector

_ROW_DIM = 6
_COLDWAY_STAGES = 3


@dataclass
class GAConfig:
    p_cross: float = 0.9
    p_mut: float = 0.15
    p_mut_delta: float = 0.6
    p_mut_zero: float = 0.2
    p_mut_activate: float = 0.2
    mutate_by_stage: bool = True
    delta_frac: float = 0.1


def _uniform_crossover_slice(
    a: torch.Tensor,
    b: torch.Tensor,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = torch.rand(a.shape, generator=rng) < 0.5
    c1 = torch.where(mask, a, b)
    c2 = torch.where(mask, b, a)
    return c1, c2


def segmented_crossover(
    parent1: torch.Tensor,
    parent2: torch.Tensor,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """element / testenv / coldway 各段内交叉；coldway 按阶段行块。"""
    c1, c2 = parent1.clone(), parent2.clone()

    e1, e2 = _uniform_crossover_slice(
        parent1[ELEMENT_SLICE], parent2[ELEMENT_SLICE], rng
    )
    c1[ELEMENT_SLICE], c2[ELEMENT_SLICE] = e1, e2

    t1, t2 = _uniform_crossover_slice(
        parent1[TESTENV_SLICE], parent2[TESTENV_SLICE], rng
    )
    c1[TESTENV_SLICE], c2[TESTENV_SLICE] = t1, t2

    for stage in range(_COLDWAY_STAGES):
        sl = slice(COLDWAY_SLICE.start + stage * _ROW_DIM, COLDWAY_SLICE.start + (stage + 1) * _ROW_DIM)
        r1, r2 = _uniform_crossover_slice(parent1[sl], parent2[sl], rng)
        c1[sl], c2[sl] = r1, r2

    return c1, c2


def _sample_nonzero(
    train_bank: torch.Tensor,
    dim_index: int,
    rng: torch.Generator,
) -> float:
    col = train_bank[:, dim_index]
    nz = col[col.abs() > 1e-8]
    if nz.numel() == 0:
        return float(torch.randn(1, generator=rng).item() * 0.01)
    j = int(torch.randint(0, nz.numel(), (1,), generator=rng).item())
    return float(nz[j].item())


def mutate_scalar(
    v: float,
    dim_index: int,
    train_bank: torch.Tensor,
    rng: torch.Generator,
    cfg: GAConfig,
) -> float:
    r = torch.rand(1, generator=rng).item()
    if r < cfg.p_mut_delta:
        # delta 变异：对零值直接激活，对非零值按比例扰动
        if abs(v) < 1e-12:
            return _sample_nonzero(train_bank, dim_index, rng)
        eps = (torch.rand(1, generator=rng).item() * 2 - 1) * cfg.delta_frac
        return float(v * (1.0 + eps))
    if r < cfg.p_mut_delta + cfg.p_mut_zero:
        # zero 变异：将当前值归零
        return 0.0
    # activate 变异（p_mut_activate）：将零值激活为非零值；非零值保持不变
    if abs(v) < 1e-12:
        return _sample_nonzero(train_bank, dim_index, rng)
    return v


def mutate_genome(
    genome: torch.Tensor,
    train_bank: torch.Tensor,
    rng: torch.Generator,
    cfg: GAConfig,
) -> torch.Tensor:
    g = genome.clone()
    if cfg.mutate_by_stage and torch.rand(1, generator=rng).item() < 0.5:
        stage = int(torch.randint(0, _COLDWAY_STAGES, (1,), generator=rng).item())
        base = COLDWAY_SLICE.start + stage * _ROW_DIM
        candidate_dims = list(range(ELEMENT_DIM + TESTENV_DIM)) + list(
            range(base, base + _ROW_DIM)
        )
    else:
        candidate_dims = list(range(INPUT_DIM))

    for i in candidate_dims:
        if torch.rand(1, generator=rng).item() < cfg.p_mut:
            val = float(g[i].item())
            g[i] = mutate_scalar(val, i, train_bank, rng, cfg)

    return g


def init_population(
    pop_size: int,
    train_bank: torch.Tensor,
    bounds: FeatureBounds,
    projector: MaskedCompositeProjector,
    rng: torch.Generator,
    cfg: GAConfig,
) -> List[torch.Tensor]:
    pop: List[torch.Tensor] = []
    n_train = train_bank.shape[0]
    for _ in range(pop_size):
        j = int(torch.randint(0, n_train, (1,), generator=rng).item())
        g = train_bank[j].clone()
        g = g + torch.randn(INPUT_DIM, generator=rng) * 0.05
        g = compile_genome(g, bounds, projector, train_bank, rng=rng)
        g = mutate_genome(g, train_bank, rng, cfg)
        g = compile_genome(g, bounds, projector, train_bank, rng=rng)
        pop.append(g)
    return pop


def crossover_and_mutate(
    parent1: torch.Tensor,
    parent2: torch.Tensor,
    train_bank: torch.Tensor,
    bounds: FeatureBounds,
    projector: MaskedCompositeProjector,
    rng: torch.Generator,
    cfg: GAConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if torch.rand(1, generator=rng).item() < cfg.p_cross:
        c1, c2 = segmented_crossover(parent1, parent2, rng)
    else:
        c1, c2 = parent1.clone(), parent2.clone()
    c1 = mutate_genome(c1, train_bank, rng, cfg)
    c2 = mutate_genome(c2, train_bank, rng, cfg)
    c1 = compile_genome(c1, bounds, projector, train_bank, rng=rng)
    c2 = compile_genome(c2, bounds, projector, train_bank, rng=rng)
    return c1, c2


def _self_test() -> None:
    from grd.feature_layout import bounds_from_train_x, build_projector

    rng = torch.Generator().manual_seed(0)
    x = torch.rand(30, INPUT_DIM)
    mask = torch.ones(30, dtype=torch.bool)
    bounds = bounds_from_train_x(x, mask)
    proj = build_projector(x, bounds)
    cfg = GAConfig()
    p1, p2 = x[0], x[1]
    c1, c2 = segmented_crossover(p1, p2, rng)
    assert c1.shape == (INPUT_DIM,)
    child1, child2 = crossover_and_mutate(p1, p2, x, bounds, proj, rng, cfg)
    assert child1.sum() >= 0
    pop = init_population(5, x, bounds, proj, rng, cfg)
    assert len(pop) == 5
    print("[OK] ga_operators self-test passed")


if __name__ == "__main__":
    _self_test()
