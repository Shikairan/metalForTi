#!/usr/bin/env python3
"""lowExp: tabular symbolic x -> YS/FS (no graph at inference)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.cli import add_common_args, experiment_header, resolve_device, resolve_out_dir, set_seed  # noqa: E402
from common.constants import FEATURE_NAMES  # noqa: E402
from common.data import bundle_to_device, load_graph_bundle  # noqa: E402
from common.distill_io import (  # noqa: E402
    build_sr_params,
    distill_block_on_numpy_io,
    export_equations_json,
    make_tabular_lookup_fn,
    save_symbolic_module,
)
from common.hybrid_models import TabularSymbolicModel  # noqa: E402
from common.metrics import evaluate_predictions, save_metrics, write_summary_md  # noqa: E402
from common.teacher import load_teacher, teacher_forward  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("lowExp")


def main() -> None:
    experiment_header("lowExp")
    p = argparse.ArgumentParser(description="lowExp: tabular symbolic model (graph-free)")
    add_common_args(p)
    args = p.parse_args()
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

    sample_mask = train_mask | val_mask if args.include_val else train_mask
    x_sample = x[sample_mask]
    x_np = x_sample.detach().cpu().numpy()

    with torch.no_grad():
        t_ys, t_fs = teacher_forward(teacher, x, edge_index, edge_type)
    ys_np = t_ys[sample_mask].detach().cpu().numpy()
    fs_np = t_fs[sample_mask].detach().cpu().numpy()

    torch.save(
        {"ys_teacher": t_ys.cpu(), "fs_teacher": t_fs.cpu(), "x": x.cpu()},
        out_dir / "teacher_predictions.pt",
    )

    sr_params = build_sr_params(niterations=args.sr_niterations, quick=args.quick, low_exp=True)

    fn_ys = make_tabular_lookup_fn(x_np, ys_np)
    fn_fs = make_tabular_lookup_fn(x_np, fs_np)

    logger.info("Distilling tabular YS")
    sym_ys = distill_block_on_numpy_io(
        fn_ys,
        x_np,
        block_name="tabular_ys",
        sr_params=sr_params,
        variable_names=list(FEATURE_NAMES),
    )
    export_equations_json(sym_ys, out_dir / "ys_tabular_sym.json")
    save_symbolic_module(sym_ys, out_dir / "ys_tabular_sym.pt")

    logger.info("Distilling tabular FS")
    sym_fs = distill_block_on_numpy_io(
        fn_fs,
        x_np,
        block_name="tabular_fs",
        sr_params=sr_params,
        variable_names=list(FEATURE_NAMES),
    )
    export_equations_json(sym_fs, out_dir / "fs_tabular_sym.json")
    save_symbolic_module(sym_fs, out_dir / "fs_tabular_sym.pt")

    tabular = TabularSymbolicModel(sym_ys, sym_fs).eval()
    with torch.no_grad():
        s_ys, s_fs = tabular(x)

    teacher_metrics = evaluate_predictions(t_ys, t_fs, ys, fs, train_mask, val_mask)
    tabular_metrics = evaluate_predictions(s_ys, s_fs, ys, fs, train_mask, val_mask)
    metrics = {
        "experiment": "lowExp",
        "graph_at_inference": False,
        "teacher": teacher_metrics,
        "tabular_symbolic": tabular_metrics,
        "graph_info_loss_val_mae_ys": tabular_metrics["val_mae_ys"] - teacher_metrics["val_mae_ys"],
        "graph_info_loss_val_mae_fs": tabular_metrics["val_mae_fs"] - teacher_metrics["val_mae_fs"],
    }
    save_metrics(out_dir / "metrics.json", metrics)
    write_summary_md(
        out_dir / "summary.md",
        "lowExp",
        [
            "Pure tabular symbolic: x (30 features) -> YS / FS",
            "No graph structure at inference",
            f"Val MAE tabular YS/FS: {tabular_metrics['val_mae_ys']:.4f} / {tabular_metrics['val_mae_fs']:.4f}",
            f"Graph info loss (tabular - teacher) val MAE: "
            f"{metrics['graph_info_loss_val_mae_ys']:.4f} / {metrics['graph_info_loss_val_mae_fs']:.4f}",
        ],
    )
    logger.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
