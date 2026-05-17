from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from model_gat import RGAT_Dual

_RUNS_DIR = Path(__file__).resolve().parent / "runs"


def _require_under_runs(path: Path, *, label: str) -> Path:
    p = path.resolve()
    base = _RUNS_DIR.resolve()
    if not p.is_relative_to(base):
        raise ValueError(f"{label} must be under {base} (got {p})")
    return p


_NUM_RELATIONS = 3
_GAT_HEADS = 4
BEST_CKPT_NAME = "best_ysfs_gat.pt"


def _inverse_fs(y: torch.Tensor) -> torch.Tensor:
    # Keep consistent with train_symbolic_findworst.py target restore behavior.
    return torch.exp(y)


def _rel_pct_vec(pred: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    pred_raw = _inverse_fs(pred)
    tgt_raw = _inverse_fs(target)
    return (pred_raw - tgt_raw).abs() / tgt_raw.abs().clamp_min(eps) * 100.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_required(data_dir: Path):
    graph_path = data_dir / "material_graph.pt"
    ys_path = data_dir / "ys.pt"
    fs_path = data_dir / "fs.pt"
    train_mask_path = data_dir / "train_mask.pt"
    val_mask_path = data_dir / "val_mask.pt"

    for p in [graph_path, ys_path, fs_path, train_mask_path, val_mask_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Required file not found: {p}")

    graph = torch.load(graph_path, map_location="cpu")
    ys = torch.load(ys_path, map_location="cpu").reshape(-1).float()
    fs = torch.load(fs_path, map_location="cpu").reshape(-1).float()
    train_mask = torch.load(train_mask_path, map_location="cpu").reshape(-1).bool()
    val_mask = torch.load(val_mask_path, map_location="cpu").reshape(-1).bool()
    return graph, ys, fs, train_mask, val_mask


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
    ys: torch.Tensor,
    fs: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    edge_type: torch.Tensor,
    *,
    allow_inactive: bool = False,
) -> None:
    n = x.shape[0]
    if ys.numel() != n or fs.numel() != n or train_mask.numel() != n or val_mask.numel() != n:
        raise ValueError("x, ys, fs, train_mask, val_mask length mismatch")
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
    model: RGAT_Dual,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    ys: torch.Tensor,
    fs: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> Tuple[float, float, float, float, float, float, float, float, int, float, float, int]:
    """MAEs; val half YS/FS separately on val; worst rel%% = max YS and max FS on val (any node); FS worst idx for FS row."""
    model.eval()
    with torch.no_grad():
        pred_ys, pred_fs = model(x, edge_index, edge_type)

        train_mae_fs = F.l1_loss(pred_fs[train_mask], fs[train_mask]).item()
        val_mae_fs = F.l1_loss(pred_fs[val_mask], fs[val_mask]).item()
        train_mae_ys = F.l1_loss(pred_ys[train_mask], ys[train_mask]).item()
        val_mae_ys = F.l1_loss(pred_ys[val_mask], ys[val_mask]).item()

        rel_ys = _rel_pct_vec(pred_ys, ys, eps=eps)
        rel_fs = _rel_pct_vec(pred_fs, fs, eps=eps)
        v_idx = torch.where(val_mask)[0]
        rel_yv = rel_ys[v_idx]
        rel_fv = rel_fs[v_idx]
        n_val = int(rel_yv.numel())
        k = max(1, (n_val + 1) // 2)
        sorted_y, _ = torch.sort(rel_yv)
        sorted_f, _ = torch.sort(rel_fv)
        val_half_rel_pct_ys = float(sorted_y[k - 1].item())
        val_half_rel_pct_fs = float(sorted_f[k - 1].item())
        val_worst_rel_pct_ys = float(rel_yv.max().item())
        val_worst_rel_pct_fs = float(rel_fv.max().item())

        worst_pos = int(torch.argmax(rel_fv).item())
        worst_idx = int(v_idx[worst_pos].item())
        pred_raw_v = _inverse_fs(pred_fs[val_mask])
        fs_raw_v = _inverse_fs(fs[val_mask])
        worst_pred_raw = float(pred_raw_v[worst_pos].item())
        worst_true_raw = float(fs_raw_v[worst_pos].item())

        val_last_lt10_numb = int(((rel_yv < 10.0) & (rel_fv < 10.0)).sum().item())

    return (
        train_mae_fs,
        val_mae_fs,
        train_mae_ys,
        val_mae_ys,
        val_half_rel_pct_ys,
        val_half_rel_pct_fs,
        val_worst_rel_pct_ys,
        val_worst_rel_pct_fs,
        worst_idx,
        worst_pred_raw,
        worst_true_raw,
        val_last_lt10_numb,
    )


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Train YS+FS dual-head RGAT on material graph data")
    p.add_argument("--data-dir", type=Path, default=base / "gnndataPT" / "r-gatPT")
    p.add_argument("--out-dir", type=Path, default=_RUNS_DIR)
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
    args.out_dir = _require_under_runs(Path(args.out_dir), label="--out-dir")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    graph, ys, fs, train_mask, val_mask = _load_required(args.data_dir)
    x = graph["sample"].x.float()
    edge_index, edge_type = _merge_hetero_edges(graph)
    _validate_inputs(
        x, ys, fs, train_mask, val_mask, edge_type, allow_inactive=args.allow_inactive
    )

    x = x.to(device)
    ys = ys.to(device)
    fs = fs.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)

    model = RGAT_Dual(
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

    def _val_score(vm_fs: float, vm_ys: float) -> float:
        return vm_fs + vm_ys

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
                "train_mae_fs",
                "val_mae_fs",
                "train_mae_ys",
                "val_mae_ys",
                "val_score",
                "val_half_rel_pct_ys",
                "val_half_rel_pct_fs",
                "val_worst_rel_pct_ys",
                "val_worst_rel_pct_fs",
                "val_worst_node_idx",
                "val_worst_pred_raw_fs",
                "val_worst_true_raw_fs",
                "val_last_lt10_sorted_numb",
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

            (
                train_mae_fs,
                val_mae_fs,
                train_mae_ys,
                val_mae_ys,
                val_half_rel_pct_ys,
                val_half_rel_pct_fs,
                val_worst_rel_pct_ys,
                val_worst_rel_pct_fs,
                val_worst_idx,
                val_worst_pred_raw,
                val_worst_true_raw,
                val_last_lt10_numb,
            ) = _eval_metrics(
                model,
                x,
                edge_index,
                edge_type,
                ys,
                fs,
                train_mask,
                val_mask,
            )
            val_score = _val_score(val_mae_fs, val_mae_ys)
            writer.writerow(
                [
                    epoch,
                    f"{loss_ys.item():.8f}",
                    f"{loss_fs.item():.8f}",
                    f"{train_mae_fs:.8f}",
                    f"{val_mae_fs:.8f}",
                    f"{train_mae_ys:.8f}",
                    f"{val_mae_ys:.8f}",
                    f"{val_score:.8f}",
                    f"{val_half_rel_pct_ys:.8f}",
                    f"{val_half_rel_pct_fs:.8f}",
                    f"{val_worst_rel_pct_ys:.8f}",
                    f"{val_worst_rel_pct_fs:.8f}",
                    val_worst_idx,
                    f"{val_worst_pred_raw:.8f}",
                    f"{val_worst_true_raw:.8f}",
                    val_last_lt10_numb,
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
                    },
                    best_path,
                )

            if epoch == 1 or epoch % int(args.log_interval) == 0 or epoch == int(args.epochs):
                print(
                    f"epoch={epoch} train_mse_ys={loss_ys.item():.6f} train_mse_fs={loss_fs.item():.6f} "
                    f"train_mae_fs={train_mae_fs:.6f} val_mae_fs={val_mae_fs:.6f} "
                    f"train_mae_ys={train_mae_ys:.6f} val_mae_ys={val_mae_ys:.6f} "
                    f"val_score={val_score:.6f} "
                    f"val_half_rel_pct=[{val_half_rel_pct_ys:.2f},{val_half_rel_pct_fs:.2f}] "
                    f"val_worst_rel_pct=[{val_worst_rel_pct_ys:.2f},{val_worst_rel_pct_fs:.2f}] "
                    f"val_worst_node_idx={val_worst_idx} "
                    f"val_last_lt10_sorted_numb={val_last_lt10_numb} "
                    f"val_worst_pred_raw_fs={val_worst_pred_raw:.4f} val_worst_true_raw_fs={val_worst_true_raw:.4f} "
                    f"best_val_score={best_val_score:.6f} best_epoch={best_epoch}"
                )

    print(f"[OK] best checkpoint: {best_path}")
    print(f"[OK] train log: {log_path}")


if __name__ == "__main__":
    main()
