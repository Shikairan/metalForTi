#!/usr/bin/env python3
"""
全量 RGAT 训练：604 节点、train:val=1:0.2，交替 MSE(YS)/MSE(FS)。
模型与 checkpoint 保存在 modelAll/runs/。
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from model_rgat import RGAT_Dual

_NUM_RELATIONS = 3
_GAT_HEADS = 4
BEST_CKPT_NAME = "best_rgat_full.pt"
_ROOT = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_bundle(data_dir: Path):
    for name in ("material_graph.pt", "ys.pt", "fs.pt", "train_mask.pt", "val_mask.pt"):
        if not (data_dir / name).is_file():
            raise FileNotFoundError(f"缺少 {data_dir / name}，请先运行 build_data.py")

    graph = torch.load(data_dir / "material_graph.pt", map_location="cpu", weights_only=False)
    ys = torch.load(data_dir / "ys.pt", map_location="cpu", weights_only=False).reshape(-1).float()
    fs = torch.load(data_dir / "fs.pt", map_location="cpu", weights_only=False).reshape(-1).float()
    train_mask = torch.load(data_dir / "train_mask.pt", map_location="cpu", weights_only=False).reshape(-1).bool()
    val_mask = torch.load(data_dir / "val_mask.pt", map_location="cpu", weights_only=False).reshape(-1).bool()
    return graph, ys, fs, train_mask, val_mask


def _merge_hetero_edges(graph) -> Tuple[torch.Tensor, torch.Tensor]:
    rels = [("comp_sim", 0), ("env_sim", 1), ("heat_sim", 2)]
    edge_indexes, edge_types = [], []
    for rel_name, rel_id in rels:
        ei = graph[("sample", rel_name, "sample")].edge_index.long()
        edge_indexes.append(ei)
        edge_types.append(torch.full((ei.shape[1],), rel_id, dtype=torch.long))
    return torch.cat(edge_indexes, dim=1), torch.cat(edge_types, dim=0)


def _eval_metrics(
    model: RGAT_Dual,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    ys: torch.Tensor,
    fs: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
) -> Tuple[float, float, float, float]:
    model.eval()
    with torch.no_grad():
        pred_ys, pred_fs = model(x, edge_index, edge_type)
        train_mae_ys = F.l1_loss(pred_ys[train_mask], ys[train_mask]).item()
        val_mae_ys = F.l1_loss(pred_ys[val_mask], ys[val_mask]).item()
        train_mae_fs = F.l1_loss(pred_fs[train_mask], fs[train_mask]).item()
        val_mae_fs = F.l1_loss(pred_fs[val_mask], fs[val_mask]).item()
    return train_mae_ys, val_mae_ys, train_mae_fs, val_mae_fs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="modelAll 全量 RGAT 训练")
    p.add_argument("--data-dir", type=Path, default=_ROOT / "data")
    p.add_argument("--out-dir", type=Path, default=_ROOT / "runs")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    graph, ys, fs, train_mask, val_mask = _load_bundle(args.data_dir)
    x = graph["sample"].x.float()
    edge_index, edge_type = _merge_hetero_edges(graph)

    x = x.to(device)
    ys = ys.to(device)
    fs = fs.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)

    n_train = int(train_mask.sum())
    n_val = int(val_mask.sum())
    print(f"[INFO] nodes={x.shape[0]}  train={n_train}  val={n_val}  device={device}")

    model = RGAT_Dual(
        in_dim=int(x.shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_relations=_NUM_RELATIONS,
        heads=_GAT_HEADS,
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )

    best_val_score = float("inf")
    best_epoch = -1
    best_path = args.out_dir / BEST_CKPT_NAME
    log_path = args.out_dir / "train_log.csv"

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_mse_ys",
                "train_mse_fs",
                "train_mae_ys",
                "val_mae_ys",
                "train_mae_fs",
                "val_mae_fs",
                "val_score",
            ]
        )

        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            optimizer.zero_grad()
            ys_pred, _ = model(x, edge_index, edge_type)
            loss_ys = F.mse_loss(ys_pred[train_mask], ys[train_mask])
            loss_ys.backward()
            optimizer.step()

            optimizer.zero_grad()
            _, fs_pred = model(x, edge_index, edge_type)
            loss_fs = F.mse_loss(fs_pred[train_mask], fs[train_mask])
            loss_fs.backward()
            optimizer.step()

            train_mae_ys, val_mae_ys, train_mae_fs, val_mae_fs = _eval_metrics(
                model, x, edge_index, edge_type, ys, fs, train_mask, val_mask
            )
            val_score = val_mae_ys + val_mae_fs
            writer.writerow(
                [
                    epoch,
                    f"{loss_ys.item():.8f}",
                    f"{loss_fs.item():.8f}",
                    f"{train_mae_ys:.8f}",
                    f"{val_mae_ys:.8f}",
                    f"{train_mae_fs:.8f}",
                    f"{val_mae_fs:.8f}",
                    f"{val_score:.8f}",
                ]
            )

            if val_score < best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model": "RGAT_Dual",
                        "model_class": "SingleEncoder_DualRGAT",
                        "backbone": "RGATConv",
                        "best_val_score": best_val_score,
                        "best_epoch": best_epoch,
                        "train_loss": "alternating_mse_ys_then_fs",
                        "in_dim": int(x.shape[1]),
                        "hidden_dim": int(args.hidden_dim),
                        "gat_heads": _GAT_HEADS,
                        "edge_dim": _NUM_RELATIONS,
                        "dropout": float(args.dropout),
                        "train_nodes": n_train,
                        "val_nodes": n_val,
                        "data_dir": str(args.data_dir.resolve()),
                    },
                    best_path,
                )

            if epoch == 1 or epoch % int(args.log_interval) == 0 or epoch == int(args.epochs):
                print(
                    f"epoch={epoch} "
                    f"train_mse_ys={loss_ys.item():.6f} train_mse_fs={loss_fs.item():.6f} "
                    f"val_mae_ys={val_mae_ys:.6f} val_mae_fs={val_mae_fs:.6f} "
                    f"val_score={val_score:.6f} best={best_val_score:.6f}@{best_epoch}"
                )

    print(f"[OK] best checkpoint: {best_path}")
    print(f"[OK] train log: {log_path}")


if __name__ == "__main__":
    main()
