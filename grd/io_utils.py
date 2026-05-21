"""
io_utils.py — 数据与模型 I/O

从 gnnDir 导出的 PT 包加载图特征、标签、mask，并将异质图边合并为 RGAT 所需的
edge_index / edge_type；加载 DualRGAT 检查点。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


def _ensure_rgat_double_on_path(gnn_rgat_dir: Path) -> None:
    """
    将 r-gatDouble 目录加入 sys.path，以便动态导入 model_gat_double。

    参数:
        gnn_rgat_dir: 含 model_gat_double.py 的目录（通常为 gnnDir/gnn/r-gatDouble）。
    """
    p = str(gnn_rgat_dir.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def load_graph_bundle(data_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从 PT 数据目录加载反推所需的全部张量。

    参数:
        data_dir: 含 material_graph.pt、ys.pt、fs.pt、train_mask.pt、val_mask.pt 的目录。

    返回:
        x: 节点特征 (N, 30)，与 HeteroData['sample'].x 一致。
        ys: 屈服强度标签 (N,)。
        fs: 抗拉/疲劳强度标签 (N,)。
        train_mask: 训练节点布尔掩码 (N,)。
        val_mask: 验证节点布尔掩码 (N,)。

    异常:
        FileNotFoundError: 任一必需文件缺失。
    """
    graph_path = data_dir / "material_graph.pt"
    for name in ("material_graph.pt", "ys.pt", "fs.pt", "train_mask.pt", "val_mask.pt"):
        if not (data_dir / name).is_file():
            raise FileNotFoundError(f"Missing {data_dir / name}")

    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    x = graph["sample"].x.float()
    ys = torch.load(data_dir / "ys.pt", map_location="cpu", weights_only=False).reshape(-1).float()
    fs = torch.load(data_dir / "fs.pt", map_location="cpu", weights_only=False).reshape(-1).float()
    train_mask = torch.load(data_dir / "train_mask.pt", map_location="cpu", weights_only=False).reshape(-1).bool()
    val_mask = torch.load(data_dir / "val_mask.pt", map_location="cpu", weights_only=False).reshape(-1).bool()
    return x, ys, fs, train_mask, val_mask


def merge_hetero_edges(graph) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将异质图中的三种关系边合并为单图 edge_index 与 edge_type。

    关系映射:
        comp_sim -> 0（元素相似）
        env_sim  -> 1（试验环境相似）
        heat_sim -> 2（工艺 coldway 相似）

    参数:
        graph: torch.load 得到的 HeteroData 或兼容字典。

    返回:
        edge_index: (2, E) 长整型边索引。
        edge_type: (E,) 每条边的关系类型 id。
    """
    rels = [("comp_sim", 0), ("env_sim", 1), ("heat_sim", 2)]
    edge_indexes = []
    edge_types = []
    for rel_name, rel_id in rels:
        ei = graph[("sample", rel_name, "sample")].edge_index.long()
        edge_indexes.append(ei)
        edge_types.append(torch.full((ei.shape[1],), rel_id, dtype=torch.long))
    edge_index = torch.cat(edge_indexes, dim=1)
    edge_type = torch.cat(edge_types, dim=0)
    return edge_index, edge_type


def load_dual_rgat(
    ckpt_path: Path,
    gnn_rgat_dir: Path,
    device: str,
) -> Tuple[nn.Module, dict]:
    """
    加载 SingleEncoder_DualRGAT 检查点并置于 eval 模式。

    参数:
        ckpt_path: best_ysfs_gat.pt 等，须含 model_state_dict、in_dim、hidden_dim 等元数据。
        gnn_rgat_dir: 模型定义所在目录。
        device: 'cuda' 或 'cpu'。

    返回:
        model: 已 load_state_dict 且 .eval() 的 nn.Module。
        ckpt: 完整 checkpoint 字典（含训练元信息）。
    """
    _ensure_rgat_double_on_path(gnn_rgat_dir)
    from model_gat_double import SingleEncoder_DualRGAT  # noqa: WPS433

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    in_dim = int(ckpt["in_dim"])
    hidden_dim = int(ckpt["hidden_dim"])
    num_relations = int(ckpt.get("edge_dim", 3))
    model = SingleEncoder_DualRGAT(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_relations=num_relations,
        heads=int(ckpt.get("gat_heads", 4)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model, ckpt
