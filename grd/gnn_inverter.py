"""
GNNInverter: 工业级梯度反推模块
=================================
支持全图联合优化、多正则策略、硬约束投影、多初始点并行。
兼容 PyTorch Geometric 消息传递网络（如 RGAT、GCN 等）。

Author: Assistant
Date: 2026-05-21
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logger = logging.getLogger("gnn_inverter")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ===========================================================================
# 1. 配置数据类
# ===========================================================================

@dataclass
class GNNInverterConfig:
    """梯度反推的全局配置。"""

    # 优化器
    optimizer: Literal["adam", "lbfgs"] = "adam"
    lr: float = 0.05
    max_iters: int = 2000
    patience: int = 200          # 早停耐心值
    lr_patience: int = 100       # 学习率调度耐心值
    lr_factor: float = 0.5       # 学习率衰减因子
    grad_clip: float = 1.0       # 梯度裁剪范数

    # 正则权重
    lambda_smooth: float = 0.05  # 图平滑正则
    lambda_sparse: float = 1e-4  # L1 稀疏正则
    lambda_anchor: float = 0.0   # L2 锚定正则（>0 时启用）
    lambda_nonneg: float = 1e3   # 非负软惩罚（仅在无硬投影时生效）
    lambda_sum1: float = 1e3     # 质量和为 1 软惩罚（仅在无硬投影时生效）

    # 硬约束投影（每 step_projection_interval 步执行）
    projection_interval: int = 1
    projectors: List[str] = field(default_factory=lambda: ["simplex"])
    # projectors 可选: "nonnegative", "simplex", "box"
    box_lower: Optional[float] = 0.0
    box_upper: Optional[float] = 1.0

    # 多初始点
    n_restarts: int = 5          # 随机重启次数
    restart_noise_scale: float = 0.2

    # 收敛阈值
    recon_tol: float = 1e-5      # 重建误差容限（MSE per node）
    min_delta: float = 1e-8      # 损失下降最小阈值

    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ===========================================================================
# 2. 正则化器（策略模式）
# ===========================================================================

class Regularizer(ABC):
    """正则化接口。输入当前 x 和图结构，返回标量损失。"""

    @abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        anchor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """返回正则损失（标量）。"""
        ...


class SmoothnessRegularizer(Regularizer):
    """图平滑正则: 鼓励邻居节点特征相似。"""

    def __init__(self, weight: float = 0.05, aggr: Literal["mean", "sum"] = "mean") -> None:
        self.weight = weight
        self.aggr = aggr

    def __call__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        anchor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_index.numel() == 0 or self.weight == 0.0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        src, dst = edge_index[0], edge_index[1]
        diff = x[src] - x[dst]
        loss = diff.pow(2).sum(dim=-1)
        if self.aggr == "mean":
            loss = loss.mean()
        else:
            loss = loss.sum()
        return self.weight * loss


class SparsityRegularizer(Regularizer):
    """L1 稀疏正则。"""

    def __init__(self, weight: float = 1e-4) -> None:
        self.weight = weight

    def __call__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        anchor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.weight == 0.0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return self.weight * x.abs().mean()


class AnchorRegularizer(Regularizer):
    """L2 锚定正则：鼓励反推结果靠近先验锚定点（如训练集均值）。"""

    def __init__(self, weight: float = 0.0) -> None:
        self.weight = weight

    def __call__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        anchor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.weight == 0.0 or anchor is None:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return self.weight * (x - anchor).pow(2).mean()


class PhysicalPenaltyRegularizer(Regularizer):
    """软物理惩罚：非负 + 质量和为 1（当不启用硬投影时的后备方案）。"""

    def __init__(self, lambda_nonneg: float = 1e3, lambda_sum1: float = 1e3) -> None:
        self.lambda_nonneg = lambda_nonneg
        self.lambda_sum1 = lambda_sum1

    def __call__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        anchor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.lambda_nonneg > 0:
            loss = loss + self.lambda_nonneg * F.relu(-x).pow(2).mean()
        if self.lambda_sum1 > 0:
            # 逐节点质量和为 1
            sum_err = (x.sum(dim=-1) - 1.0).pow(2).mean()
            loss = loss + self.lambda_sum1 * sum_err
        return loss


# ===========================================================================
# 3. 硬约束投影器
# ===========================================================================

class Projector(ABC):
    """硬约束投影接口。原地修改 x。"""

    @abstractmethod
    def project(self, x: torch.Tensor) -> torch.Tensor:
        ...


class NonNegativeProjector(Projector):
    def project(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp_(min=0)


class BoxProjector(Projector):
    def __init__(self, lower: float = 0.0, upper: float = 1.0) -> None:
        self.lower = lower
        self.upper = upper

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp_(min=self.lower, max=self.upper)


class SimplexProjector(Projector):
    """
    投影到概率单纯形：x_i >= 0, sum(x_i) = 1。
    欧氏投影到单纯形（排序法；见 Duchi et al. ICML 2008 / Michelot JOTA 1986）。
    """

    def project(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, d_in)
        # 先截断非负
        x = x.clamp(min=0)
        # 逐节点投影到单纯形
        # 算法：对每行 u（已排序），找 ρ = max{j: u_j + (1 - Σ_{i=1}^j u_i)/j > 0}
        # 然后 x_i = max(u_i - τ, 0)
        # 这里用 PyTorch 向量化实现
        N, d = x.shape
        u, _ = torch.sort(x, dim=1, descending=True)  # (N, d)
        css = u.cumsum(dim=1) - 1.0                      # (N, d)
        idx = torch.arange(1, d + 1, device=x.device, dtype=x.dtype).view(1, d)
        cond = (u - css / idx) > 0                       # (N, d)
        # 找到每行最后一个 True 的位置
        rho = cond.sum(dim=1)                            # (N,)
        rho = rho.clamp(min=1)
        tau = (css.gather(1, (rho - 1).unsqueeze(1)).squeeze(1)) / rho
        x = (x - tau.unsqueeze(1)).clamp(min=0)
        # 数值误差修正：重新归一化
        s = x.sum(dim=1, keepdim=True)
        x = x / s.clamp(min=1e-12)
        return x


class CompositeProjector(Projector):
    """组合多个投影器顺序执行。"""

    def __init__(self, names: List[str], box_lower: float = 0.0, box_upper: float = 1.0) -> None:
        self.projectors: List[Projector] = []
        for name in names:
            if name == "nonnegative":
                self.projectors.append(NonNegativeProjector())
            elif name == "box":
                self.projectors.append(BoxProjector(box_lower, box_upper))
            elif name == "simplex":
                self.projectors.append(SimplexProjector())
            else:
                raise ValueError(f"Unknown projector: {name}")

    def project(self, x: torch.Tensor) -> torch.Tensor:
        for p in self.projectors:
            x = p.project(x)
        return x


# ===========================================================================
# 4. 初始化策略
# ===========================================================================

class Initializer(ABC):
    @abstractmethod
    def generate(
        self,
        shape: Tuple[int, int],
        anchor: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        """生成初始 x，shape=(N, d_in)。"""
        ...


class ZeroInitializer(Initializer):
    def generate(self, shape, anchor=None, device="cpu"):
        return torch.zeros(*shape, device=device)


class RandomNormalInitializer(Initializer):
    def __init__(self, scale: float = 0.1) -> None:
        self.scale = scale

    def generate(self, shape, anchor=None, device="cpu"):
        return torch.randn(*shape, device=device) * self.scale


class TrainingMeanInitializer(Initializer):
    """锚定在训练集均值附近加噪。"""

    def __init__(self, noise_scale: float = 0.1) -> None:
        self.noise_scale = noise_scale

    def generate(self, shape, anchor=None, device="cpu"):
        if anchor is None:
            return torch.randn(*shape, device=device) * 0.1
        # anchor: (N, d_in) 或 (d_in,)
        if anchor.dim() == 1:
            anchor = anchor.unsqueeze(0).expand(*shape)
        noise = torch.randn_like(anchor) * self.noise_scale
        return (anchor + noise).clamp(min=0).to(device)


class DirichletInitializer(Initializer):
    """从 Dirichlet 分布采样（天然在单纯形内）。"""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def generate(self, shape, anchor=None, device="cpu"):
        N, d = shape
        # numpy 采样后转 torch
        x = np.random.dirichlet(np.full(d, self.alpha), size=N).astype(np.float32)
        return torch.from_numpy(x).to(device)


# ===========================================================================
# 5. 结果封装
# ===========================================================================

@dataclass
class InversionResult:
    x_inv: torch.Tensor                # 反推结果 (N, d_in)
    loss_history: List[float]          # 损失曲线
    recon_history: List[float]       # 重建误差子曲线
    converged: bool                    # 是否收敛
    n_iters: int                       # 实际迭代次数
    final_recon_mse: float             # 最终重建 MSE
    final_ys_mae: float                # ys MAE
    final_fs_mae: float                # fs MAE
    runtime_sec: float                 # 耗时
    init_name: str = ""                # 初始化策略名


# ===========================================================================
# 6. 核心：GNNInverter
# ===========================================================================

class GNNInverter:
    """
    工业级 GNN 梯度反推器。

    兼容任意 PyTorch / PyG 模型，只要模型支持:
        model(x, edge_index, edge_type) -> (ys, fs)
    """

    def __init__(
        self,
        model: nn.Module,
        config: GNNInverterConfig,
        regularizers: Optional[List[Regularizer]] = None,
        projector: Optional[Projector] = None,
        anchor: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model.to(config.device).eval()
        self.cfg = config
        self.device = config.device

        # 冻结模型参数
        for p in self.model.parameters():
            p.requires_grad = False

        # 正则器
        if regularizers is None:
            self.regularizers = [
                SmoothnessRegularizer(config.lambda_smooth),
                SparsityRegularizer(config.lambda_sparse),
                AnchorRegularizer(config.lambda_anchor),
                PhysicalPenaltyRegularizer(config.lambda_nonneg, config.lambda_sum1),
            ]
        else:
            self.regularizers = regularizers

        # 投影器
        if projector is None and config.projectors:
            self.projector = CompositeProjector(
                config.projectors, config.box_lower, config.box_upper
            )
        else:
            self.projector = projector

        # 锚定点（如训练集均值）
        self.anchor = anchor.to(self.device) if anchor is not None else None

    # -----------------------------------------------------------------------
    # 单次反推（核心）
    # -----------------------------------------------------------------------
    def invert_single(
        self,
        target_ys: torch.Tensor,
        target_fs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        x_init: torch.Tensor,
        init_name: str = "custom",
    ) -> InversionResult:
        """
        从单个初始点出发进行梯度反推。
        """
        target_ys = target_ys.to(self.device)
        target_fs = target_fs.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        x_init = x_init.to(self.device).detach().clone().requires_grad_(True)

        N = target_ys.size(0)

        # 优化器
        if self.cfg.optimizer == "adam":
            optimizer = Adam([x_init], lr=self.cfg.lr)
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=self.cfg.lr_factor,
                patience=self.cfg.lr_patience, verbose=False,
            )
        elif self.cfg.optimizer == "lbfgs":
            optimizer = LBFGS(
                [x_init],
                lr=self.cfg.lr,
                max_iter=20,
                history_size=50,
                line_search_fn="strong_wolfe",
            )
            scheduler = None
        else:
            raise ValueError(f"Unsupported optimizer: {self.cfg.optimizer}")

        loss_history = []
        recon_history = []
        best_loss = float("inf")
        best_x = x_init.detach().clone()
        patience_counter = 0

        t0 = time.time()

        for step in range(self.cfg.max_iters):
            # LBFGS 需要闭包
            if self.cfg.optimizer == "lbfgs":
                def closure():
                    optimizer.zero_grad()
                    loss, recon = self._compute_loss(
                        x_init, target_ys, target_fs, edge_index, edge_type
                    )
                    loss.backward()
                    if self.cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(x_init, self.cfg.grad_clip)
                    return loss

                loss = optimizer.step(closure)
                # 重新计算 recon 用于日志
                with torch.no_grad():
                    _, recon = self._compute_loss(
                        x_init, target_ys, target_fs, edge_index, edge_type
                    )
            else:
                optimizer.zero_grad()
                loss, recon = self._compute_loss(
                    x_init, target_ys, target_fs, edge_index, edge_type
                )
                loss.backward()
                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(x_init, self.cfg.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step(loss)

            # 硬约束投影
            if self.projector is not None and (step + 1) % self.cfg.projection_interval == 0:
                with torch.no_grad():
                    x_init.data = self.projector.project(x_init.data)

            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
            recon_val = recon.item()
            loss_history.append(loss_val)
            recon_history.append(recon_val)

            # 早停与最优保存
            if loss_val < best_loss - self.cfg.min_delta:
                best_loss = loss_val
                best_x = x_init.detach().clone()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.cfg.patience:
                logger.info(f"[{init_name}] Early stopping at step {step}")
                break

            if recon_val < self.cfg.recon_tol:
                logger.info(f"[{init_name}] Reconstruction converged at step {step}")
                break

        runtime = time.time() - t0

        # 最终评估（用 best_x，防止最后几步过拟合正则）
        with torch.no_grad():
            ys_pred, fs_pred = self.model(best_x, edge_index, edge_type)
            final_recon_mse = (F.mse_loss(ys_pred, target_ys) + F.mse_loss(fs_pred, target_fs)).item()
            final_ys_mae = F.l1_loss(ys_pred, target_ys).item()
            final_fs_mae = F.l1_loss(fs_pred, target_fs).item()

        return InversionResult(
            x_inv=best_x.cpu(),
            loss_history=loss_history,
            recon_history=recon_history,
            converged=final_recon_mse < self.cfg.recon_tol or patience_counter < self.cfg.patience,
            n_iters=len(loss_history),
            final_recon_mse=final_recon_mse,
            final_ys_mae=final_ys_mae,
            final_fs_mae=final_fs_mae,
            runtime_sec=runtime,
            init_name=init_name,
        )

    def _compute_loss(
        self,
        x: torch.Tensor,
        target_ys: torch.Tensor,
        target_fs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (总损失, 纯重建损失)。"""
        ys_pred, fs_pred = self.model(x, edge_index, edge_type)
        recon = F.mse_loss(ys_pred, target_ys) + F.mse_loss(fs_pred, target_fs)

        reg = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for reg_fn in self.regularizers:
            reg = reg + reg_fn(x, edge_index, self.anchor)

        total = recon + reg
        return total, recon

    # -----------------------------------------------------------------------
    # 多初始点反推（推荐）
    # -----------------------------------------------------------------------
    def invert_multistart(
        self,
        target_ys: torch.Tensor,
        target_fs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        initializers: Optional[Dict[str, Initializer]] = None,
    ) -> InversionResult:
        """
        多初始点重启，返回重建误差最小的结果。
        默认包含: training_mean, dirichlet, random_normal, zero。
        """
        N = target_ys.size(0)
        d_in = self._infer_input_dim()

        if initializers is None:
            initializers = {
                "training_mean": TrainingMeanInitializer(self.cfg.restart_noise_scale),
                "dirichlet": DirichletInitializer(alpha=1.0),
                "random_normal": RandomNormalInitializer(scale=0.2),
                "zero": ZeroInitializer(),
            }

        results: List[InversionResult] = []

        for name, init_fn in initializers.items():
            x0 = init_fn.generate((N, d_in), self.anchor, self.device)
            logger.info(f"[MultiStart] Running inversion from init: {name}")
            res = self.invert_single(
                target_ys, target_fs, edge_index, edge_type, x0, init_name=name
            )
            results.append(res)
            logger.info(
                f"[MultiStart] {name}: recon_mse={res.final_recon_mse:.6f}, "
                f"ys_mae={res.final_ys_mae:.6f}, fs_mae={res.final_fs_mae:.6f}, "
                f"iters={res.n_iters}, time={res.runtime_sec:.2f}s"
            )

        # 选择重建误差最小的结果
        best = min(results, key=lambda r: r.final_recon_mse)
        logger.info(f"[MultiStart] Best init: {best.init_name}, recon_mse={best.final_recon_mse:.6f}")
        return best

    # -----------------------------------------------------------------------
    # 批量反推（全图节点独立目标）
    # -----------------------------------------------------------------------
    def invert_batch(
        self,
        target_ys: torch.Tensor,
        target_fs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch_size: int = 1,   # 目前只支持 batch_size=1（全图），因为图耦合
    ) -> InversionResult:
        """对全图进行一次性联合反推。"""
        if batch_size != 1:
            raise NotImplementedError("Graph-coupled inversion requires full-graph optimization.")
        return self.invert_multistart(target_ys, target_fs, edge_index, edge_type)

    # -----------------------------------------------------------------------
    # 辅助：推断输入维度
    # -----------------------------------------------------------------------
    def _infer_input_dim(self) -> int:
        """通过假前向推断模型的输入特征维度。"""
        # 尝试从模型的第一层推断
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
        raise RuntimeError("Cannot infer input dim from model.")


# ===========================================================================
# 7. 使用示例（与你的 DualRGAT 兼容）
# ===========================================================================

def demo():
    """
    演示如何对 SingleEncoder_DualRGAT 进行梯度反推。
    需要用户自行提供 model, x_train, edge_index, edge_type。
    """
    # 假设你已加载模型和数据
    # from model_gat_double import SingleEncoder_DualRGAT
    # model = SingleEncoder_DualRGAT(in_dim=5, hidden_dim=64, num_relations=3)
    # model.load_state_dict(torch.load("checkpoint.pt"))
    # x_train = ...  # (544, 5)
    # edge_index, edge_type = ...

    # 1. 构造配置
    cfg = GNNInverterConfig(
        optimizer="adam",
        lr=0.05,
        max_iters=1500,
        patience=200,
        lambda_smooth=0.1,
        lambda_sparse=1e-4,
        lambda_anchor=0.5,          # 锚定在训练分布附近
        projection_interval=1,
        projectors=["simplex"],     # 硬约束：概率单纯形
        n_restarts=5,
        recon_tol=1e-5,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 2. 计算锚定点（训练集均值）
    # anchor = x_train.mean(dim=0, keepdim=True).expand(544, -1)

    # 3. 实例化反推器
    # inverter = GNNInverter(model, cfg, anchor=anchor)

    # 4. 获取目标输出（例如从真实数据或期望目标）
    # with torch.no_grad():
    #     target_ys, target_fs = model(x_train, edge_index, edge_type)

    # 5. 执行多初始点反推
    # result = inverter.invert_multistart(target_ys, target_fs, edge_index, edge_type)

    # 6. 使用结果
    # x_inv = result.x_inv          # (544, 5)
    # print(f"Converged: {result.converged}, Final recon MSE: {result.final_recon_mse:.6f}")
    pass


if __name__ == "__main__":
    demo()


# ===========================================================================
# 扩展：批量场景对比（添加到 gnn_inverter.py 末尾即可）
# ===========================================================================

from dataclasses import asdict
from typing import Dict, List
import pandas as pd


@dataclass
class Scenario:
    """单个反推场景的定义。"""
    name: str
    config: GNNInverterConfig
    regularizers: Optional[List[Regularizer]] = None
    projector: Optional[Projector] = None


@dataclass
class ComparisonResult:
    """多场景对比结果。"""
    best_scenario: str
    results: Dict[str, InversionResult]
    df: pd.DataFrame


class GNNInversionBenchmark:
    """
    多场景批量对比器。
    一次性跑完材料组分/温度场/探索性分析等多种配置，自动选优。
    """

    def __init__(
        self,
        model: nn.Module,
        target_ys: torch.Tensor,
        target_fs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        anchor: Optional[torch.Tensor] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.target_ys = target_ys.to(device)
        self.target_fs = target_fs.to(device)
        self.edge_index = edge_index.to(device)
        self.edge_type = edge_type.to(device)
        self.anchor = anchor.to(device) if anchor is not None else None
        self.device = device

        # 冻结模型
        for p in self.model.parameters():
            p.requires_grad = False

    def compare(
        self,
        scenarios: List[Scenario],
        verbose: bool = True,
    ) -> ComparisonResult:
        """
        串行执行所有场景，收集结果，生成对比报告。

        示例:
            benchmark = GNNInversionBenchmark(model, target_ys, target_fs, edge_index, edge_type)
            scenarios = [
                Scenario("材料组分", cfg_A),
                Scenario("温度场", cfg_B),
                Scenario("探索性", cfg_C),
            ]
            comp = benchmark.compare(scenarios)
            print(comp.df)
        """
        results: Dict[str, InversionResult] = {}

        for sc in scenarios:
            if verbose:
                logger.info(f"\n[Benchmark] 开始场景: {sc.name}")
                logger.info(f"  Config: {asdict(sc.config)}")

            inverter = GNNInverter(
                model=self.model,
                config=sc.config,
                regularizers=sc.regularizers,
                projector=sc.projector,
                anchor=self.anchor,
            )

            res = inverter.invert_multistart(
                target_ys=self.target_ys,
                target_fs=self.target_fs,
                edge_index=self.edge_index,
                edge_type=self.edge_type,
            )
            results[sc.name] = res

            if verbose:
                logger.info(
                    f"[Benchmark] {sc.name} 完成: "
                    f"recon_mse={res.final_recon_mse:.6f}, "
                    f"ys_mae={res.final_ys_mae:.6f}, "
                    f"fs_mae={res.final_fs_mae:.6f}, "
                    f"time={res.runtime_sec:.2f}s"
                )

        # 构建对比 DataFrame
        records = []
        for name, res in results.items():
            records.append({
                "场景": name,
                "重建MSE": res.final_recon_mse,
                "YS_MAE": res.final_ys_mae,
                "FS_MAE": res.final_fs_mae,
                "迭代次数": res.n_iters,
                "耗时(s)": res.runtime_sec,
                "是否收敛": res.converged,
                "最优初始化": res.init_name,
            })
        df = pd.DataFrame(records)

        # 按重建 MSE 排序选最优
        best_name = df.loc[df["重建MSE"].idxmin(), "场景"]
        if verbose:
            logger.info(f"\n[Benchmark] 最优场景: {best_name} (最小重建 MSE)")

        return ComparisonResult(
            best_scenario=best_name,
            results=results,
            df=df,
        )

    def visualize_comparison(
        self,
        comp: ComparisonResult,
        save_path: str = "benchmark_comparison.png",
    ) -> None:
        """绘制多场景对比图。"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        names = list(comp.results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

        # 1. 重建 MSE 对比（柱状图）
        ax = axes[0, 0]
        mses = [comp.results[n].final_recon_mse for n in names]
        bars = ax.bar(names, mses, color=colors, edgecolor="k")
        ax.set_ylabel("Reconstruction MSE")
        ax.set_title("Reconstruction Error by Scenario")
        ax.set_yscale("log")
        for bar, val in zip(bars, mses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.1,
                    f"{val:.2e}", ha="center", va="bottom", fontsize=8)
        ax.grid(True, ls="--", alpha=0.3)

        # 2. YS/FS MAE 对比
        ax = axes[0, 1]
        x = np.arange(len(names))
        width = 0.35
        ys_maes = [comp.results[n].final_ys_mae for n in names]
        fs_maes = [comp.results[n].final_fs_mae for n in names]
        ax.bar(x - width/2, ys_maes, width, label="YS MAE", color="steelblue")
        ax.bar(x + width/2, fs_maes, width, label="FS MAE", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel("MAE")
        ax.set_title("Prediction Error by Scenario")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.3)

        # 3. 耗时对比
        ax = axes[0, 2]
        times = [comp.results[n].runtime_sec for n in names]
        ax.bar(names, times, color=colors, edgecolor="k")
        ax.set_ylabel("Time (s)")
        ax.set_title("Runtime by Scenario")
        for bar, val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.05,
                    f"{val:.1f}s", ha="center", va="bottom", fontsize=8)
        ax.grid(True, ls="--", alpha=0.3)

        # 4. 损失曲线对比
        ax = axes[1, 0]
        for name, color in zip(names, colors):
            hist = comp.results[name].loss_history
            ax.semilogy(hist, label=name, color=color, alpha=0.8)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Total Loss (log)")
        ax.set_title("Optimization Trajectory")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.3)

        # 5. 反推组成分布（箱线图，所有场景叠加）
        ax = axes[1, 1]
        all_data = []
        all_labels = []
        for name in names:
            x_inv = comp.results[name].x_inv.numpy()
            for d in range(x_inv.shape[1]):
                all_data.append(x_inv[:, d])
                all_labels.append(f"{name}\nx{d}")
        ax.boxplot(all_data, labels=all_labels)
        ax.set_ylabel("Value")
        ax.set_title("Inverted Composition Distribution")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
        ax.grid(True, ls="--", alpha=0.3)

        # 6. 收敛迭代次数对比
        ax = axes[1, 2]
        iters = [comp.results[n].n_iters for n in names]
        ax.bar(names, iters, color=colors, edgecolor="k")
        ax.set_ylabel("Iterations")
        ax.set_title("Convergence Speed")
        ax.grid(True, ls="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        logger.info(f"[Benchmark] 对比图已保存: {save_path}")


# ===========================================================================
# 快速使用示例
# ===========================================================================

def demo_benchmark():
    """
    一次性对比三种场景，自动输出最优方案。
    """
    # 假设 model, target_ys, target_fs, edge_index, edge_type, anchor 已准备好
    # benchmark = GNNInversionBenchmark(model, target_ys, target_fs, edge_index, edge_type, anchor)

    # 定义三种场景
    scenarios = [
        Scenario(
            name="材料组分(simplex)",
            config=GNNInverterConfig(
                projectors=["simplex"],
                lambda_smooth=0.1,
                lambda_anchor=0.2,
                n_restarts=3,
            ),
        ),
        Scenario(
            name="温度场(box)",
            config=GNNInverterConfig(
                projectors=["box"],
                box_lower=-50,
                box_upper=200,
                lambda_smooth=0.05,
                lambda_anchor=0.0,
                n_restarts=3,
            ),
        ),
        Scenario(
            name="探索性(无硬约束)",
            config=GNNInverterConfig(
                projectors=[],
                lambda_nonneg=1e3,
                lambda_sum1=1e3,
                lambda_anchor=0.0,
                n_restarts=3,
            ),
        ),
    ]

    # 执行对比（串行，544 节点很快）
    # comp = benchmark.compare(scenarios)
    # print(comp.df.to_string(index=False))
    # print(f"\n最优方案: {comp.best_scenario}")
    # benchmark.visualize_comparison(comp, "scenario_comparison.png")
    pass
