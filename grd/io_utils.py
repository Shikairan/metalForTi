"""
加载 gnnDir 图数据、checkpoint 与边结构。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

# 将 r-gatDouble 加入 path 以导入 model_gat_double
def _ensure_rgat_double_on_path(gnn_rgat_dir: Path) -> None:
    p = str(gnn_rgat_dir.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def load_graph_bundle(data_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回 x, ys, fs, train_mask, val_mask。
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
