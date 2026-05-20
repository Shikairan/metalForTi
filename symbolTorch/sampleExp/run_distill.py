#!/usr/bin/env python3
"""sampleExp: SLIME local symbolic explanations at selected alloy nodes."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.cli import add_common_args, experiment_header, resolve_device, resolve_out_dir, set_seed  # noqa: E402
from common.data import bundle_to_device, load_graph_bundle  # noqa: E402
from common.distill_io import (  # noqa: E402
    build_slime_params,
    build_sr_params,
    distill_block_on_numpy_io,
    export_equations_json,
)
from common.metrics import rel_pct_fs, rel_pct_ys, worst_val_fs_node_idx, write_summary_md  # noqa: E402
from common.teacher import load_teacher, teacher_forward  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("sampleExp")


class NodeBranchPredictor:
    """Predict YS or FS at a fixed node while perturbing only that node's features."""

    def __init__(
        self,
        teacher: torch.nn.Module,
        x_base: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_idx: int,
        branch: str,
    ) -> None:
        self.teacher = teacher
        self.x_base = x_base
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.node_idx = int(node_idx)
        self.branch = branch

    def __call__(self, x_np: np.ndarray) -> np.ndarray:
        x_np = np.asarray(x_np, dtype=np.float32)
        out = np.zeros((x_np.shape[0], 1), dtype=np.float32)
        self.teacher.eval()
        with torch.no_grad():
            for i in range(x_np.shape[0]):
                x_full = self.x_base.clone()
                x_full[self.node_idx] = torch.from_numpy(x_np[i]).to(x_full.device)
                ys, fs = self.teacher(x_full, self.edge_index, self.edge_type)
                val = ys[self.node_idx] if self.branch == "ys" else fs[self.node_idx]
                out[i, 0] = float(val.item())
        return out


def _select_nodes(
    pred_ys: torch.Tensor,
    pred_fs: torch.Tensor,
    ys: torch.Tensor,
    fs: torch.Tensor,
    val_mask: torch.Tensor,
    node_idx: int | None,
    top_k: int,
) -> List[int]:
    if node_idx is not None:
        return [int(node_idx)]
    v_idx = torch.where(val_mask)[0]
    rel = rel_pct_fs(pred_fs, fs)[v_idx] + rel_pct_ys(pred_ys, ys)[v_idx]
    k = min(int(top_k), int(rel.numel()))
    order = torch.argsort(rel, descending=True)[:k]
    return [int(v_idx[i].item()) for i in order]


def _explain_node(
    teacher,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    x_pool_np: np.ndarray,
    node_idx: int,
    out_dir: Path,
    sr_params: dict,
    slime_params_base: dict,
) -> None:
    node_dir = out_dir / f"node_{node_idx:05d}"
    node_dir.mkdir(parents=True, exist_ok=True)
    x0 = x[node_idx].detach().cpu().numpy()
    slime_params = build_slime_params(x0, slime_params_base)

    for branch in ("ys", "fs"):
        predictor = NodeBranchPredictor(teacher, x, edge_index, edge_type, node_idx, branch)
        sym = distill_block_on_numpy_io(
            predictor,
            x_pool_np,
            block_name=f"{branch}_node{node_idx}",
            sr_params=sr_params,
            slime=True,
            slime_params=slime_params,
        )
        payload = export_equations_json(sym, node_dir / f"{branch}_formula.json", slime=True)
        expr = payload.get("equations", {}).get("0", "(see json)")
        (node_dir / f"{branch}_formula.txt").write_text(str(expr) + "\n", encoding="utf-8")

    meta = {
        "node_idx": node_idx,
        "x0": x0.tolist(),
        "slime_params": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in slime_params.items()},
    }
    with (node_dir / "local_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    experiment_header("sampleExp")
    p = argparse.ArgumentParser(description="sampleExp: SLIME per-node explanations")
    add_common_args(p)
    p.add_argument("--node-idx", type=int, default=None, help="Explain this node index")
    p.add_argument("--top-k", type=int, default=1, help="If no --node-idx, explain top-k val nodes by combined rel%%")
    p.add_argument("--slime-nn", type=int, default=10, help="SLIME J_nn nearest neighbors")
    p.add_argument("--slime-synthetic", type=int, default=100, help="SLIME synthetic samples")
    args = p.parse_args()
    if args.quick:
        args.sr_niterations = min(args.sr_niterations, 80)
        args.slime_synthetic = min(args.slime_synthetic, 30)

    set_seed(args.seed)
    out_dir = resolve_out_dir(args, Path(__file__).resolve().parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    graph, ys, fs, train_mask, val_mask = load_graph_bundle(args.data_dir)
    x, ys, fs, train_mask, val_mask, edge_index, edge_type = bundle_to_device(
        graph, ys, fs, train_mask, val_mask, device
    )
    in_dim = int(x.shape[1])

    teacher = load_teacher(
        args.ckpt,
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        device=device,
        dropout=args.dropout,
    )

    with torch.no_grad():
        pred_ys, pred_fs = teacher_forward(teacher, x, edge_index, edge_type)

    default_worst = worst_val_fs_node_idx(pred_fs, fs, val_mask)
    nodes = _select_nodes(pred_ys, pred_fs, ys, fs, val_mask, args.node_idx, args.top_k)
    if args.node_idx is None and args.top_k == 1 and default_worst not in nodes:
        nodes = [default_worst]

    x_pool = x[train_mask].detach().cpu().numpy()
    sr_params = build_sr_params(niterations=args.sr_niterations, quick=args.quick)
    slime_base = {"J_nn": args.slime_nn, "num_synthetic": args.slime_synthetic}

    for ni in nodes:
        logger.info("SLIME explain node %d", ni)
        _explain_node(teacher, x, edge_index, edge_type, x_pool, ni, out_dir, sr_params, slime_base)

    summary = {
        "experiment": "sampleExp",
        "nodes": nodes,
        "default_worst_val_fs_node": default_worst,
        "slime_nn": args.slime_nn,
        "slime_synthetic": args.slime_synthetic,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [f"Explained nodes: {nodes}", f"Default worst val FS node: {default_worst}"]
    for ni in nodes:
        lines.append(f"- node_{ni:05d}/ys_formula.txt, fs_formula.txt")
    write_summary_md(out_dir / "summary.md", "sampleExp", lines)
    logger.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
