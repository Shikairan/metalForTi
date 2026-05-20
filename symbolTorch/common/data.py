"""Load material graph bundle (same layout as r-gatDouble training)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch


def load_graph_bundle(data_dir: Path):
    graph_path = data_dir / "material_graph.pt"
    ys_path = data_dir / "ys.pt"
    fs_path = data_dir / "fs.pt"
    train_mask_path = data_dir / "train_mask.pt"
    val_mask_path = data_dir / "val_mask.pt"

    missing = [p for p in (graph_path, ys_path, fs_path) if not p.is_file()]
    if missing:
        hint = (
            f"Missing data under {data_dir}.\n"
            "Generate with:\n"
            "  cd metalForTi/gnnDir\n"
            "  python regenerate_rgnnpt.py\n"
            "  # or: python rgcn_dataloader.py --csv datagnn.csv "
            "--out-graph gnndataPT/r-gatPT/material_graph.pt ...\n"
        )
        raise FileNotFoundError(f"{missing[0]}\n{hint}")

    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    ys = torch.load(ys_path, map_location="cpu", weights_only=False).reshape(-1).float()
    fs = torch.load(fs_path, map_location="cpu", weights_only=False).reshape(-1).float()

    if train_mask_path.is_file() and val_mask_path.is_file():
        train_mask = torch.load(train_mask_path, map_location="cpu", weights_only=False).reshape(-1).bool()
        val_mask = torch.load(val_mask_path, map_location="cpu", weights_only=False).reshape(-1).bool()
    elif hasattr(graph["sample"], "train_mask") and hasattr(graph["sample"], "val_mask"):
        train_mask = graph["sample"].train_mask.reshape(-1).bool()
        val_mask = graph["sample"].val_mask.reshape(-1).bool()
    else:
        raise FileNotFoundError(
            f"Missing {train_mask_path} and {val_mask_path}, and graph has no embedded masks.\n"
            "Run: python gen_masks.py --out-train-mask ... --out-val-mask ..."
        )
    return graph, ys, fs, train_mask, val_mask


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


def bundle_to_device(
    graph,
    ys: torch.Tensor,
    fs: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    device: torch.device,
):
    x = graph["sample"].x.float().to(device)
    ys = ys.to(device)
    fs = fs.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    edge_index, edge_type = merge_hetero_edges(graph)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    return x, ys, fs, train_mask, val_mask, edge_index, edge_type
