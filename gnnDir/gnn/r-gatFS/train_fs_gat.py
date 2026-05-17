from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from model_gat import FSGAT

_NUM_RELATIONS = 3
_GAT_HEADS = 4


def _inverse_fs(y: torch.Tensor) -> torch.Tensor:
    # Keep consistent with train_symbolic_findworst.py target restore behavior.
    return torch.exp(y)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_required(data_dir: Path):
    graph_path = data_dir / "material_graph.pt"
    fs_path = data_dir / "fs.pt"
    train_mask_path = data_dir / "train_mask.pt"
    val_mask_path = data_dir / "val_mask.pt"

    for p in [graph_path, fs_path, train_mask_path, val_mask_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Required file not found: {p}")

    graph = torch.load(graph_path, map_location="cpu")
    fs = torch.load(fs_path, map_location="cpu").reshape(-1).float()
    train_mask = torch.load(train_mask_path, map_location="cpu").reshape(-1).bool()
    val_mask = torch.load(val_mask_path, map_location="cpu").reshape(-1).bool()
    return graph, fs, train_mask, val_mask


def _merge_hetero_edges(graph) -> Tuple[torch.Tensor, torch.Tensor]:
    rels = [
        ("comp_sim", 0),
        ("env_sim", 1),
        ("heat_sim", 2),
    ]
    edge_indexes = []
    edge_types = []
    for rel_name, rel_id in rels:
        ei = graph[("sample", rel_name, "sample")].edge_index.long()
        edge_indexes.append(ei)
        edge_types.append(torch.full((ei.shape[1],), rel_id, dtype=torch.long))
    edge_index = torch.cat(edge_indexes, dim=1)
    edge_type = torch.cat(edge_types, dim=0)
    return edge_index, edge_type


def _validate_inputs(
    x: torch.Tensor,
    fs: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    edge_type: torch.Tensor,
    *,
    allow_inactive: bool = False,
) -> None:
    n = x.shape[0]
    if fs.numel() != n or train_mask.numel() != n or val_mask.numel() != n:
        raise ValueError("x, fs, train_mask, val_mask length mismatch")
    if bool((train_mask & val_mask).any()):
        raise ValueError("train_mask and val_mask overlap")
    if not allow_inactive:
        if not bool((train_mask | val_mask).all()):
            miss = int((~(train_mask | val_mask)).sum().item())
            raise ValueError(
                f"train_mask and val_mask do not cover all nodes (inactive_count={miss}). "
                "Pass --allow-inactive if masks were produced by loop_train_swap curate (or similar)."
            )
    if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0:
        raise ValueError("train_mask and val_mask must both be non-empty")
    allowed = {0, 1, 2}
    got = set(edge_type.unique().tolist())
    if not got.issubset(allowed):
        raise ValueError(f"edge_type contains invalid relation IDs: {got}")


def _eval_metrics(
    model: FSGAT,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    fs: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> Tuple[float, float, float, float, int, float, float, int]:
    model.eval()
    with torch.no_grad():
        pred = model(x, edge_index, edge_type)

        train_mae = F.l1_loss(pred[train_mask], fs[train_mask]).item()
        val_mae = F.l1_loss(pred[val_mask], fs[val_mask]).item()

        pred_raw = _inverse_fs(pred[val_mask])
        fs_raw = _inverse_fs(fs[val_mask])
        rel_pct = (pred_raw - fs_raw).abs() / fs_raw.abs().clamp_min(eps) * 100.0
        n_val = int(rel_pct.numel())
        k = max(1, (n_val + 1) // 2)
        sorted_rel, _ = torch.sort(rel_pct)
        val_half_rel_pct = float(sorted_rel[k - 1].item())
        lt10 = sorted_rel < 10.0
        if bool(lt10.any()):
            val_last_lt10_sorted_idx = int(torch.where(lt10)[0][-1].item())
        else:
            val_last_lt10_sorted_idx = -1
        worst_pos = int(torch.argmax(rel_pct).item())
        worst_pct = float(rel_pct[worst_pos].item())
        worst_idx = int(torch.where(val_mask)[0][worst_pos].item())
        worst_pred_raw = float(pred_raw[worst_pos].item())
        worst_true_raw = float(fs_raw[worst_pos].item())

    return (
        train_mae,
        val_mae,
        val_half_rel_pct,
        worst_pct,
        worst_idx,
        worst_pred_raw,
        worst_true_raw,
        val_last_lt10_sorted_idx,
    )


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Train FS-only RGAT (FSGAT) on material graph data")
    p.add_argument("--data-dir", type=Path, default=base / "gnndataPT" / "r-gatPT")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    p.add_argument(
        "--epochs",
        "--epoch",
        type=int,
        default=1000,
        dest="epochs",
        help="Training epochs (singular --epoch is accepted).",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization strength.")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=5)
    p.add_argument(
        "--allow-inactive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow nodes in neither train nor val (default: True; typical after loop_train_swap curate). "
            "Use --no-allow-inactive to require full coverage."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    graph, fs, train_mask, val_mask = _load_required(args.data_dir)
    x = graph["sample"].x.float()
    edge_index, edge_type = _merge_hetero_edges(graph)
    _validate_inputs(
        x, fs, train_mask, val_mask, edge_type, allow_inactive=args.allow_inactive
    )

    x = x.to(device)
    fs = fs.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)

    model = FSGAT(
        in_dim=int(x.shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_relations=_NUM_RELATIONS,
        heads=_GAT_HEADS,
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    best_val_mae = float("inf")
    best_epoch = -1
    best_path = args.out_dir / "best_fs_gat.pt"
    log_path = args.out_dir / "train_log.csv"

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_mae",
                "val_mae",
                "val_half_rel_pct",
                "val_worst_rel_pct",
                "val_worst_node_idx",
                "val_worst_pred_raw_fs",
                "val_worst_true_raw_fs",
                "val_last_lt10_sorted_idx",
            ]
        )

        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            optimizer.zero_grad()
            pred = model(x, edge_index, edge_type)
            train_loss = F.l1_loss(pred[train_mask], fs[train_mask])
            train_loss.backward()
            optimizer.step()

            (
                train_mae,
                val_mae,
                val_half_rel_pct,
                val_worst_pct,
                val_worst_idx,
                val_worst_pred_raw,
                val_worst_true_raw,
                val_last_lt10_sorted_idx,
            ) = _eval_metrics(
                model,
                x,
                edge_index,
                edge_type,
                fs,
                train_mask,
                val_mask,
            )
            writer.writerow(
                [
                    epoch,
                    f"{train_loss.item():.8f}",
                    f"{train_mae:.8f}",
                    f"{val_mae:.8f}",
                    f"{val_half_rel_pct:.8f}",
                    f"{val_worst_pct:.8f}",
                    val_worst_idx,
                    f"{val_worst_pred_raw:.8f}",
                    f"{val_worst_true_raw:.8f}",
                    val_last_lt10_sorted_idx,
                ]
            )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model": "FSGAT",
                        "best_val_mae": best_val_mae,
                        "best_epoch": best_epoch,
                        "in_dim": int(x.shape[1]),
                        "hidden_dim": int(args.hidden_dim),
                        "gat_heads": _GAT_HEADS,
                        "edge_dim": _NUM_RELATIONS,
                        "backbone": "RGATConv",
                    },
                    best_path,
                )

            if epoch == 1 or epoch % int(args.log_interval) == 0 or epoch == int(args.epochs):
                print(
                    f"epoch={epoch} train_loss={train_loss.item():.6f} "
                    f"train_mae={train_mae:.6f} val_mae={val_mae:.6f} "
                    f"val_half_rel_pct={val_half_rel_pct:.2f}% "
                    f"val_worst_rel_pct={val_worst_pct:.2f}% val_worst_node_idx={val_worst_idx} "
                    f"val_last_lt10_sorted_idx={val_last_lt10_sorted_idx} "
                    f"val_worst_pred_raw_fs={val_worst_pred_raw:.4f} val_worst_true_raw_fs={val_worst_true_raw:.4f} "
                    f"best_val_mae={best_val_mae:.6f} best_epoch={best_epoch}"
                )

    print(f"[OK] best checkpoint: {best_path}")
    print(f"[OK] train log: {log_path}")


if __name__ == "__main__":
    main()
