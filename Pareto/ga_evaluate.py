"""
ga_evaluate.py — 增广图 forward 与三目标适应度评估。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from Pareto.ga_graph import GraphContext, build_augmented_graph


@dataclass
class FitnessResult:
    f1: float  # |dYS|
    f2: float  # |dFS|
    f3: float  # anchor L2 (0 if two-objective)
    ys_pred: float
    fs_pred: float
    nearest_train_idx: int


class FitnessEvaluator:
    def __init__(
        self,
        model: nn.Module,
        ctx: GraphContext,
        x_train: torch.Tensor,
        target_ys: float,
        target_fs: float,
        device: str,
        *,
        use_anchor: bool = True,
        element_thr: float = 0.8,
        testenv_thr: float = 0.8,
        coldway_thr: float = 0.8,
        train_node_indices: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model
        self.model.eval()  # 防御性确保推理模式，避免 Dropout 激活
        self.ctx = ctx
        self.x_train = x_train.detach().cpu().float()
        self.target_ys = float(target_ys)
        self.target_fs = float(target_fs)
        self.device = device
        self.use_anchor = use_anchor
        self.element_thr = element_thr
        self.testenv_thr = testenv_thr
        self.coldway_thr = coldway_thr
        # train_node_indices[i] = 当前 x_train[i] 在原始全图中的节点 id (0..N-1)
        # 用于将 _anchor_distance 返回的局部索引映射回图节点 id
        self.train_node_indices: Optional[torch.Tensor] = (
            train_node_indices.detach().cpu().long() if train_node_indices is not None else None
        )

    def _anchor_distance(self, genome: torch.Tensor) -> Tuple[float, int]:
        g = genome.detach().cpu().float().unsqueeze(0)
        diff = self.x_train - g
        dist = diff.pow(2).sum(dim=1)
        local_idx = int(torch.argmin(dist).item())
        # 映射回原始图节点 id；若未提供 train_node_indices 则返回局部索引（兼容旧接口）
        graph_idx = (
            int(self.train_node_indices[local_idx].item())
            if self.train_node_indices is not None
            else local_idx
        )
        return float(dist[local_idx].sqrt().item()), graph_idx

    @torch.no_grad()
    def evaluate_one(self, genome: torch.Tensor) -> FitnessResult:
        x_aug, ei, et, d_idx = build_augmented_graph(
            genome,
            self.ctx,
            element_thr=self.element_thr,
            testenv_thr=self.testenv_thr,
            coldway_thr=self.coldway_thr,
        )
        x_aug = x_aug.to(self.device)
        ei = ei.to(self.device)
        et = et.to(self.device)

        ys_pred, fs_pred = self.model(x_aug, ei, et)
        yp = float(ys_pred[d_idx].item())
        fp = float(fs_pred[d_idx].item())
        f1 = abs(yp - self.target_ys)
        f2 = abs(fp - self.target_fs)
        if self.use_anchor:
            f3, nn_idx = self._anchor_distance(genome)
        else:
            f3, nn_idx = 0.0, -1
        return FitnessResult(
            f1=f1,
            f2=f2,
            f3=f3,
            ys_pred=yp,
            fs_pred=fp,
            nearest_train_idx=nn_idx,
        )

    def evaluate_population(
        self,
        population: List[torch.Tensor],
    ) -> List[FitnessResult]:
        return [self.evaluate_one(g) for g in population]

    def objectives_tensor(self, fit: FitnessResult) -> torch.Tensor:
        if self.use_anchor:
            return torch.tensor([fit.f1, fit.f2, fit.f3], dtype=torch.float32)
        return torch.tensor([fit.f1, fit.f2], dtype=torch.float32)

    def fitness_from_labels(
        self,
        genome: torch.Tensor,
        ys_label: float,
        fs_label: float,
    ) -> FitnessResult:
        """用节点标签 YS/FS 计算适应度，不调用 GNN forward。"""
        yp = float(ys_label)
        fp = float(fs_label)
        f1 = abs(yp - self.target_ys)
        f2 = abs(fp - self.target_fs)
        if self.use_anchor:
            f3, nn_idx = self._anchor_distance(genome)
        else:
            f3, nn_idx = 0.0, -1
        return FitnessResult(
            f1=f1,
            f2=f2,
            f3=f3,
            ys_pred=yp,
            fs_pred=fp,
            nearest_train_idx=nn_idx,
        )
