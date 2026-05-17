from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

# This file lives in gnnDir/gnn/r-gatDymDoub/ — add that dir for local imports.
_RGAT_DIR = Path(__file__).resolve().parent
_RUNS_DIR = _RGAT_DIR / "runs"
if str(_RGAT_DIR) not in sys.path:
    sys.path.insert(0, str(_RGAT_DIR))

# gnnDir/ is two levels above r-gatDymDoub/ (r-gatDymDoub -> gnn -> gnnDir).
_GNN_ROOT = _RGAT_DIR.parent.parent

_GNN_PKG_DIR = _RGAT_DIR.parent
if str(_GNN_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_GNN_PKG_DIR))
from mask_history import save_round_mask_snapshot  # noqa: E402


def _require_under_runs(path: Path, *, label: str) -> Path:
    """Run artifacts (.pt / .csv / .json) must be read/written only under this package's runs/."""
    p = path.resolve()
    base = _RUNS_DIR.resolve()
    if not p.is_relative_to(base):
        raise ValueError(f"{label} must be under {base} (got {p})")
    return p

from model_gat import RGAT_Dual  # noqa: E402
from train_fs_gat import (  # noqa: E402
    BEST_CKPT_NAME,
    _NUM_RELATIONS,
    _GAT_HEADS,
    _inverse_fs,
    _load_required,
    _merge_hetero_edges,
    _validate_inputs,
    set_seed,
)


STATE_NAME = "gat_mask_loop_state.json"
HISTORY_NAME = "gat_mask_swap_history.csv"


def _default_data_dir() -> Path:
    return _GNN_ROOT / "gnndataPT" / "r-gatPT"


def _default_out_dir() -> Path:
    return _RUNS_DIR


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"total_swaps": 0, "last_round": 0}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _init_history_csv(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "timestamp",
                "round",
                "epochs_done",
                "train_mae_fs",
                "val_mae_fs",
                "train_mae_ys",
                "val_mae_ys",
                "worst_val_node",
                "best_train_node",
                "worst_rel_pct",
                "best_rel_pct",
                "curate_count_after",
                "phase",
                "mask_op",
                "batch_k",
                "effective_k",
                "worst_nodes",
                "best_nodes",
                "elapsed_sec",
            ]
        )


def _append_history_csv(path: Path, row: list[Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def _rel_pct_vec(pred: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    pred_raw = _inverse_fs(pred)
    tgt_raw = _inverse_fs(target)
    return (pred_raw - tgt_raw).abs() / tgt_raw.abs().clamp_min(eps) * 100.0


def _difficulty_weights(
    val_mae_ys: float,
    val_mae_fs: float,
    best_mae_ys: float,
    best_mae_fs: float,
    *,
    eps: float = 1e-8,
) -> Tuple[float, float]:
    """d = current / historical_best; larger d => harder or regressed task => higher weight (dym.md)."""
    d_ys = 1.0 if best_mae_ys == float("inf") else val_mae_ys / (best_mae_ys + eps)
    d_fs = 1.0 if best_mae_fs == float("inf") else val_mae_fs / (best_mae_fs + eps)
    tot = d_ys + d_fs
    if tot < 1e-12:
        return 0.5, 0.5
    return d_ys / tot, d_fs / tot


def _difficulty_weighted_comb_rel_pct(
    rel_ys: torch.Tensor,
    rel_fs: torch.Tensor,
    val_mae_ys: float,
    val_mae_fs: float,
    best_mae_ys: float,
    best_mae_fs: float,
) -> Tuple[torch.Tensor, float, float]:
    w_ys, w_fs = _difficulty_weights(val_mae_ys, val_mae_fs, best_mae_ys, best_mae_fs)
    rel_ys = rel_ys/(rel_ys+rel_fs)
    rel_fs = rel_fs/(rel_ys+rel_fs)

    comb = w_ys * rel_ys + w_fs * rel_fs
    #comb = 0.34*rel_ys + 0.66*rel_fs
    return comb, w_ys, w_fs


def _loop_dual_metrics(
    pred_ys: torch.Tensor,
    pred_fs: torch.Tensor,
    ys: torch.Tensor,
    fs: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    best_mae_ys: float,
    best_mae_fs: float,
    *,
    eps: float = 1e-6,
) -> Tuple[float, float, float, float, List[float], List[float], int, int, float, float]:
    """MAEs; val_half / val_worst pair unchanged; val_worst_node uses difficulty-weighted comb rel%% (dym.md)."""
    train_mae_ys = F.l1_loss(pred_ys[train_mask], ys[train_mask]).item()
    train_mae_fs = F.l1_loss(pred_fs[train_mask], fs[train_mask]).item()
    val_mae_ys = F.l1_loss(pred_ys[val_mask], ys[val_mask]).item()
    val_mae_fs = F.l1_loss(pred_fs[val_mask], fs[val_mask]).item()

    rel_ys = _rel_pct_vec(pred_ys, ys, eps=eps)
    rel_fs = _rel_pct_vec(pred_fs, fs, eps=eps)

    v_idx = torch.where(val_mask)[0]
    if v_idx.numel() == 0:
        raise RuntimeError("empty val_mask")

    rel_yv = rel_ys[v_idx]
    rel_fv = rel_fs[v_idx]
    n_v = int(rel_yv.numel())
    k_med = max(1, (n_v + 1) // 2)
    sorted_y, _ = torch.sort(rel_yv)
    sorted_f, _ = torch.sort(rel_fv)
    val_half_rel_pct_pair = [
        float(sorted_y[k_med - 1].item()),
        float(sorted_f[k_med - 1].item()),
    ]

    val_worst_rel_pct_pair = [
        float(rel_yv.max().item()),
        float(rel_fv.max().item()),
    ]

    comb, w_ys, w_fs = _difficulty_weighted_comb_rel_pct(
        rel_ys, rel_fs, val_mae_ys, val_mae_fs, best_mae_ys, best_mae_fs
    )
    comb_v = comb[v_idx]
    worst_local = int(torch.argmax(comb_v).item())
    node_worst = int(v_idx[worst_local].item())

    val_lt10_both = int(((rel_yv < 10.0) & (rel_fv < 10.0)).sum().item())

    return (
        train_mae_ys,
        train_mae_fs,
        val_mae_ys,
        val_mae_fs,
        val_half_rel_pct_pair,
        val_worst_rel_pct_pair,
        node_worst,
        val_lt10_both,
        w_ys,
        w_fs,
    )


def _pick_topk_worst_val_best_train_dual(
    pred_ys: torch.Tensor,
    pred_fs: torch.Tensor,
    ys: torch.Tensor,
    fs: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    k: int,
    val_mae_ys: float,
    val_mae_fs: float,
    best_mae_ys: float,
    best_mae_fs: float,
    *,
    eps: float = 1e-6,
) -> Tuple[List[int], List[float], List[int], List[float], int]:
    """Val: largest difficulty-weighted comb rel%%; train: smallest same comb (dym.md)."""
    rel_ys = _rel_pct_vec(pred_ys, ys, eps=eps)
    rel_fs = _rel_pct_vec(pred_fs, fs, eps=eps)
    comb, _, _ = _difficulty_weighted_comb_rel_pct(
        rel_ys, rel_fs, val_mae_ys, val_mae_fs, best_mae_ys, best_mae_fs
    )

    v_idx = torch.where(val_mask)[0]
    t_idx = torch.where(train_mask)[0]
    if v_idx.numel() == 0 or t_idx.numel() == 0:
        raise RuntimeError("empty val_mask or train_mask")

    n_v = int(v_idx.numel())
    n_t = int(t_idx.numel())
    k_req = max(1, int(k))
    k_eff = min(k_req, n_v, n_t)

    comb_v = comb[v_idx]
    comb_t = comb[t_idx]
    _, order_v = torch.sort(comb_v, descending=True)
    take_v = order_v[:k_eff]
    worst_nodes = [int(v_idx[int(i)].item()) for i in take_v]
    worst_pcts = [float(comb_v[int(i)].item()) for i in take_v]

    _, order_t = torch.sort(comb_t, descending=False)
    take_t = order_t[:k_eff]
    best_nodes = [int(t_idx[int(i)].item()) for i in take_t]
    best_pcts = [float(comb_t[int(i)].item()) for i in take_t]

    return worst_nodes, worst_pcts, best_nodes, best_pcts, k_eff


def _apply_mask_swap(
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    i_worst_val: int,
    i_best_train: int,
) -> None:
    if i_worst_val == i_best_train:
        raise ValueError(f"swap nodes collide: idx={i_worst_val}")
    if not bool(val_mask[i_worst_val].item()):
        raise ValueError(f"worst_val node {i_worst_val} is not in val_mask")
    if not bool(train_mask[i_best_train].item()):
        raise ValueError(f"best_train node {i_best_train} is not in train_mask")

    val_mask[i_worst_val] = False
    train_mask[i_worst_val] = True
    train_mask[i_best_train] = False
    val_mask[i_best_train] = True


def _apply_mask_curate(
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    i_worst_val: int,
    i_best_train: int,
) -> None:
    """Remove worst from val (inactive: neither train nor val); promote best train to val. |val| unchanged, |train| -= 1."""
    if i_worst_val == i_best_train:
        raise ValueError(f"curate nodes collide: idx={i_worst_val}")
    if not bool(val_mask[i_worst_val].item()):
        raise ValueError(f"worst_val node {i_worst_val} is not in val_mask")
    if not bool(train_mask[i_best_train].item()):
        raise ValueError(f"best_train node {i_best_train} is not in train_mask")

    val_mask[i_worst_val] = False
    train_mask[i_worst_val] = False
    train_mask[i_best_train] = False
    val_mask[i_best_train] = True


def _apply_mask_swap_batch(
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    worst_nodes: List[int],
    best_nodes: List[int],
) -> None:
    if len(worst_nodes) != len(best_nodes):
        raise ValueError("swap batch length mismatch")
    for w, b in zip(worst_nodes, best_nodes):
        _apply_mask_swap(train_mask, val_mask, w, b)


def _apply_mask_curate_batch(
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    worst_nodes: List[int],
    best_nodes: List[int],
) -> None:
    if len(worst_nodes) != len(best_nodes):
        raise ValueError("curate batch length mismatch")
    for w, b in zip(worst_nodes, best_nodes):
        _apply_mask_curate(train_mask, val_mask, w, b)


def _fmt_id_list(ids: List[int], max_show: int = 8) -> str:
    if len(ids) <= max_show:
        return ",".join(map(str, ids))
    head = ",".join(map(str, ids[:max_show]))
    return f"{head},...(+{len(ids) - max_show})"


def _load_model_from_checkpoint(
    ckpt_path: Path,
    *,
    in_dim: int,
    hidden_dim: int,
    num_relations: int,
    heads: int,
    dropout: float,
    device: torch.device,
) -> RGAT_Dual:
    model = RGAT_Dual(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_relations=num_relations,
        heads=heads,
        dropout=dropout,
    ).to(device)
    if ckpt_path.is_file():
        blob = torch.load(ckpt_path, map_location=device)
        if isinstance(blob, dict) and "model_state_dict" in blob:
            try:
                model.load_state_dict(blob["model_state_dict"], strict=True)
            except RuntimeError as e:
                print(
                    f"[WARN] could not load checkpoint (e.g. old FSGAT / wrong arch): {e}\n"
                    f"[WARN] training from scratch this round; delete or replace {ckpt_path} to avoid this."
                )
        else:
            try:
                model.load_state_dict(blob, strict=False)
            except RuntimeError as e:
                print(f"[WARN] could not load raw state_dict from {ckpt_path}: {e}")
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Loop train RGAT_Dual (YS+FS): one forward per epoch; dynamic priority L1 backward (dym.md); "
            "curate/swap uses difficulty-weighted comb rel%% (current val MAE / hist-best val MAE). "
            "Checkpoint/state/csv only under this dir's runs/. Writes masks under --data-dir; "
            "optional per-round mask snapshots (see mask_history.py)."
        )
    )
    p.add_argument(
        "--swap-batch-size",
        type=int,
        default=1,
        help=(
            "How many val/train pairs to move per round (curate or exchange). "
            "Clipped to min(|val|,|train|). Use 1 for legacy single-pair behavior."
        ),
    )
    p.add_argument("--data-dir", type=Path, default=_default_data_dir())
    p.add_argument("--out-dir", type=Path, default=_default_out_dir())
    p.add_argument("--epochs-per-round", type=int, default=100)
    p.add_argument(
        "--log-interval",
        type=int,
        default=5,
        help="Print dual YS/FS val metrics (half/worst pair + diff_w + worst node) every N epochs.",
    )
    p.add_argument(
        "--max-curate",
        type=int,
        default=60,
        help="Rounds 1..N: curate (worst val -> inactive, best train -> val). Rounds N+1..: swap val/train pair.",
    )
    p.add_argument("--max-rounds", type=int, default=0, help="0 = infinite until Ctrl+C.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help="Default: out-dir / gat_mask_loop_state.json",
    )
    p.add_argument(
        "--history-csv",
        type=Path,
        default=None,
        help="Default: out-dir / gat_mask_swap_history.csv",
    )
    p.add_argument(
        "--mask-history-dir",
        type=Path,
        default=None,
        help="Per-round mask snapshots (default: out-dir/mask_round_history; must stay under runs/).",
    )
    p.add_argument(
        "--no-mask-history",
        action="store_true",
        help="Disable per-round mask snapshot files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    data_dir = Path(args.data_dir).resolve()
    out_dir = _require_under_runs(Path(args.out_dir), label="--out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = (
        _require_under_runs(Path(args.state_path), label="--state-path")
        if args.state_path
        else out_dir / STATE_NAME
    )
    history_csv = (
        _require_under_runs(Path(args.history_csv), label="--history-csv")
        if args.history_csv
        else out_dir / HISTORY_NAME
    )
    mask_hist_dir: Path | None
    if args.no_mask_history:
        mask_hist_dir = None
    else:
        mask_hist_dir = (
            _require_under_runs(Path(args.mask_history_dir), label="--mask-history-dir")
            if args.mask_history_dir is not None
            else out_dir / "mask_round_history"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = out_dir / BEST_CKPT_NAME
    _require_under_runs(ckpt_path, label="checkpoint path")

    graph, ys_cpu, fs_cpu, train_mask_cpu, val_mask_cpu = _load_required(data_dir)
    x = graph["sample"].x.float()
    edge_index_cpu, edge_type_cpu = _merge_hetero_edges(graph)
    _validate_inputs(
        x,
        ys_cpu,
        fs_cpu,
        train_mask_cpu,
        val_mask_cpu,
        edge_type_cpu,
        allow_inactive=True,
    )

    _init_history_csv(history_csv)
    state = _load_state(state_path)

    hb_ys = float(state["hist_best_val_mae_ys"]) if state.get("hist_best_val_mae_ys") is not None else float("inf")
    hb_fs = float(state["hist_best_val_mae_fs"]) if state.get("hist_best_val_mae_fs") is not None else float("inf")

    print(f"[INIT] data_dir={data_dir}")
    print(f"[INIT] out_dir={out_dir} device={device}")
    print(
        f"[INIT] epochs_per_round={args.epochs_per_round} log_interval={args.log_interval} "
        f"swap_batch_size={args.swap_batch_size} max_curate={args.max_curate} max_rounds={args.max_rounds}"
    )
    print(f"[INIT] state_path={state_path} history_csv={history_csv}")
    print(f"[INIT] mask_history_dir={mask_hist_dir if mask_hist_dir is not None else '(disabled)'}")

    round_idx = int(state.get("last_round", 0))

    while True:
        round_idx += 1
        if int(args.max_rounds) > 0 and round_idx > int(args.max_rounds):
            break

        t_round = time.time()
        train_mask = train_mask_cpu.to(device)
        val_mask = val_mask_cpu.to(device)
        x_d = x.to(device)
        ys = ys_cpu.to(device)
        fs = fs_cpu.to(device)
        edge_index = edge_index_cpu.to(device)
        edge_type = edge_type_cpu.to(device)

        model = _load_model_from_checkpoint(
            ckpt_path,
            in_dim=int(x_d.shape[1]),
            hidden_dim=int(args.hidden_dim),
            num_relations=_NUM_RELATIONS,
            heads=_GAT_HEADS,
            dropout=float(args.dropout),
            device=device,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )

        print("\n" + "=" * 80)
        print(f"[ROUND {round_idx}] train {int(args.epochs_per_round)} epochs")
        print("=" * 80)

        def _round_val_score(vm_fs: float, vm_ys: float) -> float:
            return vm_fs + vm_ys

        best_val_score = float("inf")
        best_epoch = -1
        prev_val_mae_ys: float | None = None
        prev_val_mae_fs: float | None = None
        for epoch in range(1, int(args.epochs_per_round) + 1):
            model.train()
            optimizer.zero_grad()
            pred_ys, pred_fs = model(x_d, edge_index, edge_type)
            loss_ys = F.l1_loss(pred_ys[train_mask], ys[train_mask])
            loss_fs = F.l1_loss(pred_fs[train_mask], fs[train_mask])
            if prev_val_mae_ys is None:
                loss_ys.backward(retain_graph=True)
                loss_fs.backward()
            else:
                eps_s = 1e-8
                ry = prev_val_mae_ys / (hb_ys + eps_s) if hb_ys < float("inf") else 1.0
                rf = prev_val_mae_fs / (hb_fs + eps_s) if hb_fs < float("inf") else 1.0
                if ry >= rf:
                    loss_ys.backward()
                else:
                    loss_fs.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                pred_ys_e, pred_fs_e = model(x_d, edge_index, edge_type)
                hb_before_ys, hb_before_fs = hb_ys, hb_fs
                (
                    train_mae_ys,
                    train_mae_fs,
                    val_mae_ys,
                    val_mae_fs,
                    val_half_pair,
                    val_worst_pair,
                    val_worst_node_idx,
                    val_last_lt10_sorted_numb,
                    w_ys,
                    w_fs,
                ) = _loop_dual_metrics(
                    pred_ys_e,
                    pred_fs_e,
                    ys,
                    fs,
                    train_mask,
                    val_mask,
                    hb_before_ys,
                    hb_before_fs,
                )
                hb_ys = min(hb_ys, val_mae_ys)
                hb_fs = min(hb_fs, val_mae_fs)
                prev_val_mae_ys = val_mae_ys
                prev_val_mae_fs = val_mae_fs

            val_score = _round_val_score(val_mae_fs, val_mae_ys)
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
                        "train_loss": "single_forward_dual_l1_dynamic_priority_backward",
                        "loop_round": round_idx,
                        "in_dim": int(x_d.shape[1]),
                        "hidden_dim": int(args.hidden_dim),
                        "gat_heads": _GAT_HEADS,
                        "edge_dim": _NUM_RELATIONS,
                    },
                    ckpt_path,
                )

            if (
                epoch == 1
                or epoch % int(args.log_interval) == 0
                or epoch == int(args.epochs_per_round)
            ):
                print(
                    f"epoch={epoch} train_l1_ys={loss_ys.item():.6f} train_l1_fs={loss_fs.item():.6f} "
                    f"train_mae=[{train_mae_ys:.6f},{train_mae_fs:.6f}] "
                    f"val_mae=[{val_mae_ys:.6f},{val_mae_fs:.6f}] "
                    f"val_half_rel_pct=[{val_half_pair[0]:.2f},{val_half_pair[1]:.2f}] "
                    f"val_worst_rel_pct=[{val_worst_pair[0]:.2f},{val_worst_pair[1]:.2f}] "
                    f"diff_w=[{w_ys:.4f},{w_fs:.4f}] "
                    f"val_worst_node_idx={val_worst_node_idx} "
                    f"\n\tval_last_lt10_sorted_numb={val_last_lt10_sorted_numb} "
                    f"val_score={val_score:.6f} best_val_score={best_val_score:.6f} best_epoch={best_epoch}"
                )

        model.eval()
        with torch.no_grad():
            pred_ys, pred_fs = model(x_d, edge_index, edge_type)
            hb_post_before_ys, hb_post_before_fs = hb_ys, hb_fs
            (
                train_mae_ys,
                train_mae_fs,
                val_mae_ys,
                val_mae_fs,
                _half_p,
                _worst_p,
                _worst_n,
                val_last_lt10_sorted_numb,
                w_ys_p,
                w_fs_p,
            ) = _loop_dual_metrics(
                pred_ys,
                pred_fs,
                ys,
                fs,
                train_mask,
                val_mask,
                hb_post_before_ys,
                hb_post_before_fs,
            )
            worst_nodes, worst_pcts, best_nodes, best_pcts, k_eff = (
                _pick_topk_worst_val_best_train_dual(
                    pred_ys,
                    pred_fs,
                    ys,
                    fs,
                    train_mask,
                    val_mask,
                    int(args.swap_batch_size),
                    val_mae_ys,
                    val_mae_fs,
                    hb_post_before_ys,
                    hb_post_before_fs,
                )
            )
            hb_ys = min(hb_ys, val_mae_ys)
            hb_fs = min(hb_fs, val_mae_fs)

        phase = "curate" if round_idx <= int(args.max_curate) else "exchange"
        curate_after = min(round_idx, int(args.max_curate))
        use_curate = round_idx <= int(args.max_curate)
        mask_op = "curate" if use_curate else "exchange"
        k_req = max(1, int(args.swap_batch_size))

        if k_eff < k_req:
            print(
                f"[ROUND {round_idx}] batch clipped: requested swap_batch_size={k_req} "
                f"effective_k={k_eff} (min of val/train pool)"
            )

        if use_curate and int(train_mask.sum()) < k_eff + 1:
            raise RuntimeError(
                "curate needs train_count >= effective_k+1 after batch (train must stay non-empty); "
                f"train_count={int(train_mask.sum())} effective_k={k_eff}"
            )

        i_worst, i_best = worst_nodes[0], best_nodes[0]
        worst_pct, best_pct = worst_pcts[0], best_pcts[0]
        worst_nodes_str = "|".join(map(str, worst_nodes))
        best_nodes_str = "|".join(map(str, best_nodes))

        _post_common = (
            f"[ROUND {round_idx}] post_train mode={mask_op} "
            f"train_mae=[{train_mae_ys:.6f},{train_mae_fs:.6f}] "
            f"val_mae=[{val_mae_ys:.6f},{val_mae_fs:.6f}] "
            f"diff_w=[{w_ys_p:.4f},{w_fs_p:.4f}] "
            f"val_last_lt10_sorted_numb={val_last_lt10_sorted_numb} "
            f"batch_k={k_req} effective_k={k_eff} "
        )
        if use_curate:
            print(
                f"{_post_common}"
                f"drop_val_worst_ids={_fmt_id_list(worst_nodes)} "
                f"(diff_weighted_rel%% head {worst_pcts[0]:.2f}..{worst_pcts[-1]:.2f}) "
                f"promote_to_val_ids={_fmt_id_list(best_nodes)} "
                f"(diff_weighted_rel%% head {best_pcts[0]:.2f}..{best_pcts[-1]:.2f}) phase={phase}"
            )
        else:
            print(
                f"{_post_common}"
                f"swap worst_val_ids={_fmt_id_list(worst_nodes)} "
                f"(diff_weighted_rel%% head {worst_pcts[0]:.2f}..{worst_pcts[-1]:.2f}) "
                f"<-> best_train_ids={_fmt_id_list(best_nodes)} "
                f"(diff_weighted_rel%% head {best_pcts[0]:.2f}..{best_pcts[-1]:.2f}) phase={phase}"
            )

        train_during = train_mask.detach().cpu().bool().clone()
        val_during = val_mask.detach().cpu().bool().clone()

        if use_curate:
            _apply_mask_curate_batch(train_mask, val_mask, worst_nodes, best_nodes)
        else:
            _apply_mask_swap_batch(train_mask, val_mask, worst_nodes, best_nodes)
        _validate_inputs(
            x_d, ys, fs, train_mask, val_mask, edge_type, allow_inactive=True
        )

        train_mask_cpu = train_mask.detach().cpu().bool()
        val_mask_cpu = val_mask.detach().cpu().bool()
        torch.save(train_mask_cpu, data_dir / "train_mask.pt")
        torch.save(val_mask_cpu, data_dir / "val_mask.pt")

        train_after = train_mask_cpu.clone()
        val_after = val_mask_cpu.clone()
        if mask_hist_dir is not None:
            snap_path = save_round_mask_snapshot(
                mask_hist_dir,
                round_idx,
                train_during,
                val_during,
                train_after,
                val_after,
                meta={
                    "mask_op": mask_op,
                    "phase": phase,
                    "effective_k": k_eff,
                    "data_dir": str(data_dir),
                },
            )
            print(f"[ROUND {round_idx}] mask snapshot -> {snap_path}")

        state["total_swaps"] = int(state.get("total_swaps", 0)) + 1
        state["last_round"] = round_idx
        state["last_worst_val_node"] = i_worst
        state["last_best_train_node"] = i_best
        state["last_swap_batch_size"] = k_req
        state["last_effective_k"] = k_eff
        state["last_worst_val_nodes"] = worst_nodes
        state["last_best_train_nodes"] = best_nodes
        state["curate_phase_complete"] = round_idx >= int(args.max_curate)
        if hb_ys < float("inf"):
            state["hist_best_val_mae_ys"] = hb_ys
        else:
            state.pop("hist_best_val_mae_ys", None)
        if hb_fs < float("inf"):
            state["hist_best_val_mae_fs"] = hb_fs
        else:
            state.pop("hist_best_val_mae_fs", None)
        _save_state(state_path, state)

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        _append_history_csv(
            history_csv,
            [
                ts,
                round_idx,
                int(args.epochs_per_round),
                f"{train_mae_fs:.8f}",
                f"{val_mae_fs:.8f}",
                f"{train_mae_ys:.8f}",
                f"{val_mae_ys:.8f}",
                i_worst,
                i_best,
                f"{worst_pct:.8f}",
                f"{best_pct:.8f}",
                curate_after,
                phase,
                mask_op,
                k_req,
                k_eff,
                worst_nodes_str,
                best_nodes_str,
                f"{time.time() - t_round:.4f}",
            ],
        )
        print(f"[ROUND {round_idx}] saved masks + state; logged {history_csv}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user (Ctrl+C).")
        raise SystemExit(0)
