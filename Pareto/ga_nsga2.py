"""
ga_nsga2.py — 手写 NSGA-II：非支配排序、拥挤距离、环境选择。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch

from Pareto.ga_evaluate import FitnessResult


@dataclass
class Individual:
    genome: torch.Tensor
    fitness: Optional[FitnessResult] = None
    objectives: Optional[torch.Tensor] = None
    rank: int = 0
    crowding: float = 0.0


def _dominates(a: torch.Tensor, b: torch.Tensor) -> bool:
    """最小化目标：a 支配 b。"""
    le = (a <= b).all()
    lt = (a < b).any()
    return bool(le and lt)


def fast_non_dominated_sort(
    objectives: List[torch.Tensor],
) -> List[List[int]]:
    n = len(objectives)
    domination_count = [0] * n
    dominated_set: List[List[int]] = [[] for _ in range(n)]
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objectives[p], objectives[q]):
                dominated_set[p].append(q)
            elif _dominates(objectives[q], objectives[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in dominated_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    if not fronts[-1]:
        fronts.pop()
    return fronts


def _crowding_distance(objectives: List[torch.Tensor]) -> List[float]:
    """objectives: 当前前沿上各个体的目标向量列表（与前沿局部顺序对齐）。"""
    n = len(objectives)
    if n <= 2:
        return [float("inf")] * n
    m = objectives[0].numel()
    dist = [0.0] * n
    local = list(range(n))
    for obj_idx in range(m):
        order = sorted(local, key=lambda i: float(objectives[i][obj_idx].item()))
        dist[order[0]] = float("inf")
        dist[order[-1]] = float("inf")
        vals = [float(objectives[i][obj_idx].item()) for i in order]
        vmin, vmax = vals[0], vals[-1]
        span = vmax - vmin
        if span < 1e-12:
            continue
        for k in range(1, n - 1):
            prev_v = float(objectives[order[k - 1]][obj_idx].item())
            next_v = float(objectives[order[k + 1]][obj_idx].item())
            dist[order[k]] += (next_v - prev_v) / span
    return dist


def assign_rank_and_crowding(
    population: List[Individual],
) -> None:
    objs = [ind.objectives for ind in population]
    assert all(o is not None for o in objs), "所有个体必须已设置 objectives 张量"
    objectives: List[torch.Tensor] = objs  # type: ignore[assignment]
    fronts = fast_non_dominated_sort(objectives)
    for rank, front in enumerate(fronts):
        front_objs = [objectives[i] for i in front]
        crowding = _crowding_distance(front_objs)
        for local_i, pop_i in enumerate(front):
            population[pop_i].rank = rank
            population[pop_i].crowding = crowding[local_i]


def tournament_select(
    population: List[Individual],
    rng: torch.Generator,
    k: int = 2,
) -> Individual:
    idx = torch.randint(0, len(population), (k,), generator=rng).tolist()
    candidates = [population[i] for i in idx]
    best = candidates[0]
    for c in candidates[1:]:
        if c.rank < best.rank:
            best = c
        elif c.rank == best.rank and c.crowding > best.crowding:
            best = c
        elif c.rank == best.rank and c.crowding == best.crowding:
            # 随机打破完全平局，避免总选第一个候选
            if torch.rand(1, generator=rng).item() < 0.5:
                best = c
    return best


def environmental_selection(
    combined: List[Individual],
    pop_size: int,
) -> List[Individual]:
    assign_rank_and_crowding(combined)
    # 直接使用 combined 的完整 objectives 列表，避免过滤后索引与 combined 错位
    fronts = fast_non_dominated_sort([ind.objectives for ind in combined])
    next_pop: List[Individual] = []
    for front in fronts:
        if len(next_pop) + len(front) <= pop_size:
            next_pop.extend(combined[i] for i in front)
        else:
            remaining = pop_size - len(next_pop)
            front_objs = [combined[i].objectives for i in front]
            assert all(o is not None for o in front_objs)
            crowding = _crowding_distance(front_objs)
            order = sorted(range(len(front)), key=lambda k: crowding[k], reverse=True)
            for k in order[:remaining]:
                next_pop.append(combined[front[k]])
            break
    return next_pop


def get_pareto_front(population: List[Individual]) -> List[Individual]:
    assign_rank_and_crowding(population)
    return [ind for ind in population if ind.rank == 0]


def nsga2_evolve(
    population: List[Individual],
    pop_size: int,
    generations: int,
    make_offspring: Callable[[], List[Individual]],
    on_generation: Optional[Callable[[int, List[Individual]], None]] = None,
) -> List[Individual]:
    """运行 NSGA-II 主循环；make_offspring 返回 pop_size 个子代。"""
    for gen in range(generations):
        offspring = make_offspring()
        for ind in population + offspring:
            if ind.objectives is None and ind.fitness is not None:
                raise ValueError("Individual missing objectives tensor")
        combined = population + offspring
        population = environmental_selection(combined, pop_size)
        if on_generation:
            on_generation(gen, population)
    return population
