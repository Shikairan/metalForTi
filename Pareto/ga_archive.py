"""
ga_archive.py — 基因库累积：604 原始节点 + 每代虚拟个体。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

import torch

from Pareto.ga_evaluate import FitnessResult

if TYPE_CHECKING:
    from Pareto.ga_evaluate import FitnessEvaluator


def weighted_score(fit: FitnessResult) -> float:
    """折中目标：f1 + f2 + 0.1*f3（越小越好）。"""
    return fit.f1 + fit.f2 + 0.1 * fit.f3


@dataclass
class ArchiveEntry:
    genome: torch.Tensor
    fitness: FitnessResult
    generation: int
    is_original: bool
    source_node_id: Optional[int] = None
    virtual_id: Optional[int] = None

    def source_label(self) -> str:
        """人类可读的基因来源说明。"""
        if self.is_original:
            node = self.source_node_id if self.source_node_id is not None else "?"
            return f"604 原始基因（图节点 #{node}）"
        vid = self.virtual_id if self.virtual_id is not None else "?"
        return f"杂交虚拟基因（第 {self.generation} 代，虚拟 #{vid}）"


class GeneArchive:
    def __init__(self) -> None:
        self._entries: List[ArchiveEntry] = []
        self._next_virtual_id = 0

    @property
    def entries(self) -> List[ArchiveEntry]:
        return self._entries

    def size(self) -> int:
        return len(self._entries)

    def num_original(self) -> int:
        return sum(1 for e in self._entries if e.is_original)

    def num_virtual(self) -> int:
        return sum(1 for e in self._entries if not e.is_original)

    @classmethod
    def from_graph(
        cls,
        x: torch.Tensor,
        ys: torch.Tensor,
        fs: torch.Tensor,
        evaluator: FitnessEvaluator,
    ) -> GeneArchive:
        """将全部图节点以标签适应度写入档案（不 GNN forward）。"""
        archive = cls()
        n = x.shape[0]
        for i in range(n):
            genome = x[i].clone()
            fit = evaluator.fitness_from_labels(
                genome,
                float(ys[i].item()),
                float(fs[i].item()),
            )
            archive._entries.append(
                ArchiveEntry(
                    genome=genome,
                    fitness=fit,
                    generation=0,
                    is_original=True,
                    source_node_id=i,
                )
            )
        return archive

    def add_virtual_batch(
        self,
        genomes: List[torch.Tensor],
        fitness_list: List[FitnessResult],
        generation: int,
    ) -> None:
        if len(genomes) != len(fitness_list):
            raise ValueError("genomes 与 fitness_list 长度不一致")
        for genome, fit in zip(genomes, fitness_list):
            self._entries.append(
                ArchiveEntry(
                    genome=genome.clone(),
                    fitness=fit,
                    generation=generation,
                    is_original=False,
                    virtual_id=self._next_virtual_id,
                )
            )
            self._next_virtual_id += 1

    def best_entry(self) -> Optional[ArchiveEntry]:
        if not self._entries:
            return None
        return min(self._entries, key=lambda e: weighted_score(e.fitness))

    def best_virtual_entry(self) -> Optional[ArchiveEntry]:
        """历史虚拟个体中加权分最优者。"""
        virtuals = [e for e in self._entries if not e.is_original]
        if not virtuals:
            return None
        return min(virtuals, key=lambda e: weighted_score(e.fitness))

    def select_top_k(self, k: int) -> List[ArchiveEntry]:
        """全库加权 top-k（遗留；父本池请用 build_breeder_pool）。"""
        if k <= 0:
            return []
        ranked = sorted(self._entries, key=lambda e: weighted_score(e.fitness))
        return ranked[: min(k, len(ranked))]

    def original_entries(self) -> List[ArchiveEntry]:
        """原始节点池：604 条，组成后不再增删。"""
        return [e for e in self._entries if e.is_original]

    def select_top_virtual_k(self, k: int) -> List[ArchiveEntry]:
        """虚拟精英池：全部历史虚拟中加权 top-k，每代轮换。"""
        if k <= 0:
            return []
        virtuals = [e for e in self._entries if not e.is_original]
        if not virtuals:
            return []
        ranked = sorted(virtuals, key=lambda e: weighted_score(e.fitness))
        return ranked[: min(k, len(ranked))]

    def build_breeder_pool(self, virtual_pool_size: int) -> List[ArchiveEntry]:
        """父本池 = 固定原始池 + 历史最优虚拟池。"""
        return self.original_entries() + self.select_top_virtual_k(virtual_pool_size)

    def repair_all_genomes(
        self,
        repair_fn,
    ) -> int:
        """对库内全部条目应用 compile/repair（返回修复条数）。"""
        n = 0
        for entry in self._entries:
            entry.genome = repair_fn(entry.genome)
            n += 1
        return n

    def latest_virtual_batch(self, batch_size: int) -> List[ArchiveEntry]:
        """最近入库的虚拟批次（用于当代子代统计）。"""
        if batch_size <= 0:
            return []
        virtuals = [e for e in self._entries if not e.is_original]
        return virtuals[-batch_size:] if virtuals else []

    def to_individuals(self, evaluator: Optional[FitnessEvaluator] = None) -> List:
        from Pareto.ga_nsga2 import Individual

        out: List[Individual] = []
        for e in self._entries:
            ind = Individual(genome=e.genome.clone(), fitness=e.fitness)
            if evaluator is not None:
                ind.objectives = evaluator.objectives_tensor(e.fitness)
            out.append(ind)
        return out


def weighted_tournament_select(
    entries: List[ArchiveEntry],
    rng: torch.Generator,
    k: int = 2,
) -> ArchiveEntry:
    """加权锦标赛：分数低者胜（遗留 API）。"""
    if not entries:
        raise ValueError("entries 为空")
    if len(entries) == 1:
        return entries[0]
    idx = torch.randint(0, len(entries), (k,), generator=rng).tolist()
    candidates = [entries[i] for i in idx]
    return min(candidates, key=lambda e: weighted_score(e.fitness))


def random_pair_select(
    entries: List[ArchiveEntry],
    rng: torch.Generator,
) -> Tuple[ArchiveEntry, ArchiveEntry]:
    """从历史精英池中均匀随机选取一对父本（尽量不自交）。"""
    if not entries:
        raise ValueError("entries 为空")
    if len(entries) == 1:
        return entries[0], entries[0]
    n = len(entries)
    i1 = int(torch.randint(0, n, (1,), generator=rng).item())
    i2 = int(torch.randint(0, n - 1, (1,), generator=rng).item())
    if i2 >= i1:
        i2 += 1
    return entries[i1], entries[i2]


def _self_test() -> None:
    from Pareto.ga_evaluate import FitnessEvaluator
    from Pareto.ga_graph import GraphContext

    n = 604
    x = torch.randn(n, 30)
    ys = torch.linspace(400, 900, n)
    fs = torch.linspace(500, 1000, n)
    ctx = GraphContext.from_tensors(x, torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, dtype=torch.long))

    class _DummyModel(torch.nn.Module):
        def forward(self, x_aug, ei, et):
            return torch.zeros(x_aug.shape[0]), torch.zeros(x_aug.shape[0])

    ev = FitnessEvaluator(
        _DummyModel(),
        ctx,
        x[:100],
        target_ys=700.0,
        target_fs=800.0,
        device="cpu",
    )
    archive = GeneArchive.from_graph(x, ys, fs, ev)
    assert archive.size() == 604
    assert archive.num_original() == 604
    assert archive.num_virtual() == 0

    assert len(archive.original_entries()) == 604

    genomes = [torch.randn(30) for _ in range(700)]
    fits = [
        FitnessResult(f1=float(i % 10), f2=2.0, f3=0.5, ys_pred=699.0, fs_pred=798.0, nearest_train_idx=0)
        for i in range(700)
    ]
    archive.add_virtual_batch(genomes, fits, generation=1)
    assert archive.num_virtual() == 700

    elite = archive.select_top_virtual_k(604)
    assert len(elite) == 604
    scores = [weighted_score(e.fitness) for e in elite]
    assert scores == sorted(scores)

    pool = archive.build_breeder_pool(604)
    assert len(pool) == 604 + 604

    best = archive.best_entry()
    assert best is not None
    best_v = archive.best_virtual_entry()
    assert best_v is not None
    assert not best_v.is_original

    p1, p2 = random_pair_select(pool, torch.Generator().manual_seed(0))
    assert any(p1 is e for e in pool)
    assert any(p2 is e for e in pool)

    print("[OK] ga_archive self-test passed")


if __name__ == "__main__":
    _self_test()
