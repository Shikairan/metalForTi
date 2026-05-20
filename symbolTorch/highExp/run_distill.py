#!/usr/bin/env python3
"""highExp: symbolize ys/fs encoders only; keep RGAT + heads neural."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.cli import (  # noqa: E402
    add_common_args,
    effective_max_output_dim,
    experiment_header,
    resolve_device,
    resolve_out_dir,
    set_seed,
)
from common.data import bundle_to_device, load_graph_bundle  # noqa: E402
from common.distill_io import (  # noqa: E402
    build_sr_params,
    distill_block,
    export_equations_json,
    save_symbolic_module,
    try_load_symbolic_module,
)
from common.hybrid_models import HighExpHybrid  # noqa: E402
from common.metrics import evaluate_predictions, save_metrics, write_summary_md  # noqa: E402
from common.teacher import load_teacher, teacher_forward  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("highExp")


def main() -> None:
    experiment_header("highExp")
    p = argparse.ArgumentParser(description="highExp: symbolic encoders + neural RGAT/heads")
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

    sr_params = build_sr_params(niterations=args.sr_niterations, quick=args.quick)
    max_od = effective_max_output_dim(args)

    ys_pt = out_dir / "ys_encoder_sym.pt"
    fs_pt = out_dir / "fs_encoder_sym.pt"
    sym_ys_enc = try_load_symbolic_module(ys_pt)
    if sym_ys_enc is None:
        logger.info("Distilling ys_encoder (max_output_dim=%s)", max_od)
        sym_ys_enc = distill_block(
            teacher.ys_encoder,
            x_sample,
            block_name="ys_encoder",
            sr_params=sr_params,
            max_output_dim=max_od,
            save_path=out_dir / "sr_cache" / "ys_encoder",
            resume_pt=ys_pt,
        )
        save_symbolic_module(sym_ys_enc, ys_pt)
    else:
        logger.info("Loaded cached ys_encoder from %s", ys_pt)
    export_equations_json(sym_ys_enc, out_dir / "ys_encoder_sym.json")

    sym_fs_enc = try_load_symbolic_module(fs_pt)
    if sym_fs_enc is None:
        logger.info("Distilling fs_encoder (max_output_dim=%s)", max_od)
        sym_fs_enc = distill_block(
            teacher.fs_encoder,
            x_sample,
            block_name="fs_encoder",
            sr_params=sr_params,
            max_output_dim=max_od,
            save_path=out_dir / "sr_cache" / "fs_encoder",
            resume_pt=fs_pt,
        )
        save_symbolic_module(sym_fs_enc, fs_pt)
    else:
        logger.info("Loaded cached fs_encoder from %s", fs_pt)
    export_equations_json(sym_fs_enc, out_dir / "fs_encoder_sym.json")

    hybrid = HighExpHybrid(teacher, sym_ys_enc, sym_fs_enc).eval()
    hybrid.to(device)
    torch.save({"hybrid_class": "HighExpHybrid", "teacher_ckpt": str(args.ckpt)}, out_dir / "hybrid_meta.pt")
    torch.save(hybrid.state_dict(), out_dir / "hybrid_state.pt")

    with torch.no_grad():
        t_ys, t_fs = teacher_forward(teacher, x, edge_index, edge_type)
        h_ys, h_fs = hybrid(x, edge_index, edge_type)

    metrics = {
        "experiment": "highExp",
        "quick": args.quick,
        "max_output_dim": max_od,
        "teacher": evaluate_predictions(t_ys, t_fs, ys, fs, train_mask, val_mask),
        "hybrid": evaluate_predictions(h_ys, h_fs, ys, fs, train_mask, val_mask),
    }
    save_metrics(out_dir / "metrics.json", metrics)
    write_summary_md(
        out_dir / "summary.md",
        "highExp",
        [
            "Symbolic: `ys_encoder`, `fs_encoder`",
            "Neural: RGAT stacks + MLP heads",
            f"Formulas: `{out_dir / 'ys_encoder_sym.json'}`, `{out_dir / 'fs_encoder_sym.json'}`",
            f"Val MAE hybrid YS/FS: {metrics['hybrid']['val_mae_ys']:.4f} / {metrics['hybrid']['val_mae_fs']:.4f}",
        ],
    )
    logger.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
