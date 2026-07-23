"""
ga_multi_gpu_eval.py — 多卡并行 GNN 适应度评估（与单卡 FitnessEvaluator 解耦）。

将基因组切分到多张 GPU，每卡一份模型副本，线程池并行 evaluate_one，按原序汇总。
默认不启用；由 CLI ``--eval-devices`` 打开。
"""

from __future__ import annotations

import copy
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn

from Pareto.ga_evaluate import FitnessEvaluator, FitnessResult

logger = logging.getLogger("Pareto.ga_multi_gpu_eval")


def parse_eval_devices(
    spec: Optional[str],
    *,
    force_cpu: bool = False,
    fallback_device: str = "cpu",
) -> List[str]:
    """
    解析评估设备列表。

    - force_cpu → ``["cpu"]``
    - spec 为 None / 空 → ``[fallback_device]``（单卡/单设备，与旧行为一致）
    - ``all`` → 当前进程可见的全部 ``cuda:i``
    - ``0,1,2,3`` 或 ``cuda:0,cuda:1`` → 显式列表
    """
    if force_cpu:
        return ["cpu"]
    if spec is None or str(spec).strip() == "":
        return [fallback_device]

    raw = str(spec).strip().lower()
    if raw == "all":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            logger.warning("--eval-devices all 但 CUDA 不可用，回退到 %s", fallback_device)
            return [fallback_device]
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return devices

    devices: List[str] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        if p.isdigit():
            devices.append(f"cuda:{p}")
        elif p.startswith("cuda"):
            devices.append(p)
        elif p == "cpu":
            devices.append("cpu")
        else:
            raise ValueError(f"无法解析设备项: {part!r}（期望 all / 0,1,2 / cuda:0,cuda:1）")
    if not devices:
        raise ValueError(f"--eval-devices 解析结果为空: {spec!r}")

    # 校验 cuda 下标未越界
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        for d in devices:
            if d.startswith("cuda:"):
                idx = int(d.split(":", 1)[1])
                if idx < 0 or idx >= n:
                    raise ValueError(
                        f"设备 {d} 超出可见 GPU 范围 [0, {n})；"
                        "注意 CUDA_VISIBLE_DEVICES 会重映射编号"
                    )
    elif any(d.startswith("cuda") for d in devices):
        logger.warning("请求了 CUDA 设备但不可用，全部回退到 cpu")
        return ["cpu"]
    return devices


def shard_ranges(n_items: int, n_shards: int) -> List[tuple[int, int]]:
    """将 [0, n_items) 尽量均匀切成 n_shards 段，返回 (start, end) 列表。"""
    if n_shards <= 0:
        raise ValueError("n_shards must be positive")
    if n_items <= 0:
        return [(0, 0)] * n_shards
    base, rem = divmod(n_items, n_shards)
    ranges: List[tuple[int, int]] = []
    start = 0
    for i in range(n_shards):
        size = base + (1 if i < rem else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


def _clone_model_to_device(model: nn.Module, device: str) -> nn.Module:
    """深拷贝模型到目标设备并 eval。"""
    cloned = copy.deepcopy(model).to(device)
    cloned.eval()
    return cloned


def build_worker_evaluators(
    model: nn.Module,
    devices: Sequence[str],
    *,
    ctx,
    x_train: torch.Tensor,
    target_ys: float,
    target_fs: float,
    use_anchor: bool = True,
    element_thr: float = 0.8,
    testenv_thr: float = 0.8,
    coldway_thr: float = 0.8,
    train_node_indices: Optional[torch.Tensor] = None,
) -> List[FitnessEvaluator]:
    """每张卡一份模型副本 + FitnessEvaluator。"""
    workers: List[FitnessEvaluator] = []
    # 先落到 CPU 再分别拷到各卡，避免从已占用显存的源卡连环 deepcopy
    cpu_model = copy.deepcopy(model).cpu()
    for d in devices:
        m = _clone_model_to_device(cpu_model, d)
        workers.append(
            FitnessEvaluator(
                m,
                ctx,
                x_train,
                target_ys,
                target_fs,
                d,
                use_anchor=use_anchor,
                element_thr=element_thr,
                testenv_thr=testenv_thr,
                coldway_thr=coldway_thr,
                train_node_indices=train_node_indices,
            )
        )
    del cpu_model
    return workers


def parallel_evaluate_genomes(
    workers: Sequence[FitnessEvaluator],
    genomes: Sequence[torch.Tensor],
) -> List[FitnessResult]:
    """
    将 genomes 按段切到各 worker，线程池并行评估，按原序返回。
    单 worker 或空列表时退化为串行。
    """
    n = len(genomes)
    if n == 0:
        return []
    if len(workers) == 1:
        return workers[0].evaluate_population(list(genomes))

    ranges = shard_ranges(n, len(workers))
    results: List[Optional[FitnessResult]] = [None] * n

    def _run_shard(worker_idx: int, start: int, end: int) -> List[tuple[int, FitnessResult]]:
        worker = workers[worker_idx]
        device = worker.device
        out: List[tuple[int, FitnessResult]] = []
        # 绑定当前线程默认 CUDA 设备，减少隐式跨卡
        ctx_mgr = torch.cuda.device(device) if str(device).startswith("cuda") else nullcontext()
        with ctx_mgr:
            for i in range(start, end):
                out.append((i, worker.evaluate_one(genomes[i])))
        return out

    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        futs = []
        for wi, (start, end) in enumerate(ranges):
            if start >= end:
                continue
            futs.append(pool.submit(_run_shard, wi, start, end))
        for fut in as_completed(futs):
            for i, fit in fut.result():
                results[i] = fit

    if any(r is None for r in results):
        missing = [i for i, r in enumerate(results) if r is None]
        raise RuntimeError(f"多卡评估结果缺失索引: {missing[:20]}")
    return results  # type: ignore[return-value]


class ShardedMultiGpuEvaluator:
    """
    多卡评估门面：接口对齐 FitnessEvaluator 的常用方法，
    ``evaluate_population`` 走分片并行，其余委托主 worker。
    """

    def __init__(self, workers: List[FitnessEvaluator]) -> None:
        if not workers:
            raise ValueError("workers 不能为空")
        self.workers = workers
        self.primary = workers[0]
        self.devices = [w.device for w in workers]

    @classmethod
    def create(
        cls,
        model: nn.Module,
        devices: Sequence[str],
        *,
        ctx,
        x_train: torch.Tensor,
        target_ys: float,
        target_fs: float,
        use_anchor: bool = True,
        element_thr: float = 0.8,
        testenv_thr: float = 0.8,
        coldway_thr: float = 0.8,
        train_node_indices: Optional[torch.Tensor] = None,
    ) -> "ShardedMultiGpuEvaluator":
        workers = build_worker_evaluators(
            model,
            devices,
            ctx=ctx,
            x_train=x_train,
            target_ys=target_ys,
            target_fs=target_fs,
            use_anchor=use_anchor,
            element_thr=element_thr,
            testenv_thr=testenv_thr,
            coldway_thr=coldway_thr,
            train_node_indices=train_node_indices,
        )
        return cls(workers)

    @property
    def device(self) -> str:
        return self.primary.device

    @property
    def use_anchor(self) -> bool:
        return self.primary.use_anchor

    @property
    def model(self) -> nn.Module:
        return self.primary.model

    def evaluate_one(self, genome: torch.Tensor) -> FitnessResult:
        return self.primary.evaluate_one(genome)

    def evaluate_population(self, population: List[torch.Tensor]) -> List[FitnessResult]:
        return parallel_evaluate_genomes(self.workers, population)

    def objectives_tensor(self, fit: FitnessResult) -> torch.Tensor:
        return self.primary.objectives_tensor(fit)

    def fitness_from_labels(
        self,
        genome: torch.Tensor,
        ys_label: float,
        fs_label: float,
    ) -> FitnessResult:
        return self.primary.fitness_from_labels(genome, ys_label, fs_label)


EvaluatorLike = Union[FitnessEvaluator, ShardedMultiGpuEvaluator]


def evaluate_genomes(
    evaluator: EvaluatorLike,
    genomes: Sequence[torch.Tensor],
) -> List[FitnessResult]:
    """统一入口：多卡走分片并行，单卡走 FitnessEvaluator.evaluate_population。"""
    return evaluator.evaluate_population(list(genomes))
