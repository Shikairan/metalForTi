#!/usr/bin/env python3
"""medExp: symbolic encoders + symbolic heads; neural RGAT in the middle."""

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
from common.constants import head_feature_names  # noqa: E402
from common.data import bundle_to_device, load_graph_bundle  # noqa: E402
from common.distill_io import (  # noqa: E402
    build_sr_params,
    distill_block,
    export_equations_json,
    save_symbolic_module,
    try_load_symbolic_module,
)
from common.hybrid_models import MedExpHybrid  # noqa: E402
from common.metrics import evaluate_predictions, save_metrics, write_summary_md  # noqa: E402
from common.teacher import collect_branch_hidden, load_teacher, teacher_forward  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("medExp")


def _load_or_distill_encoder(teacher, x_sample, enc_dir, out_dir, branch: str, sr_params, max_od):
    for base in (enc_dir, out_dir):
        sym_path = base / f"{branch}_encoder_sym.pt"
        sym = try_load_symbolic_module(sym_path)
        if sym is not None:
            logger.info("Loading cached %s encoder from %s", branch, sym_path)
            return sym

    enc = teacher.ys_encoder if branch == "ys" else teacher.fs_encoder
    sym_path = out_dir / f"{branch}_encoder_sym.pt"
    sym = distill_block(
        enc,
        x_sample,
        block_name=f"{branch}_encoder",
        sr_params=sr_params,
        max_output_dim=max_od,
        save_path=out_dir / "sr_cache" / f"{branch}_encoder",
        resume_pt=sym_path,
    )
    export_equations_json(sym, out_dir / f"{branch}_encoder_sym.json")
    save_symbolic_module(sym, sym_path)
    return sym


def main() -> None:
    experiment_header("medExp")
    p = argparse.ArgumentParser(description="medExp: symbolic encoder+head, neural RGAT")
    add_common_args(p)
    p.add_argument(
        "--encoder-sym-dir",
        type=Path,
        default=None,
        help="Reuse highExp runs dir (ys/fs_encoder_sym.pt). Default: medExp out-dir",
    )
    args = p.parse_args()
    set_seed(args.seed)

    exp_dir = Path(__file__).resolve().parent
    out_dir = resolve_out_dir(args, exp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enc_dir = args.encoder_sym_dir or out_dir

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
    h_names = head_feature_names(args.hidden_dim)

    sym_ys_enc = _load_or_distill_encoder(teacher, x_sample, enc_dir, out_dir, "ys", sr_params, max_od)
    sym_fs_enc = _load_or_distill_encoder(teacher, x_sample, enc_dir, out_dir, "fs", sr_params, max_od)

    with torch.no_grad():
        h_ys, h_fs = collect_branch_hidden(teacher, x, edge_index, edge_type)
    h_ys_sample = h_ys[sample_mask]
    h_fs_sample = h_fs[sample_mask]

    logger.info("Distilling ys_head")
    sym_ys_head = distill_block(
        teacher.ys_head,
        h_ys_sample,
        block_name="ys_head",
        parent_model=None,
        sr_params=sr_params,
        max_output_dim=None,
        variable_names=h_names,
        save_path=out_dir / "sr_cache" / "ys_head",
    )
    export_equations_json(sym_ys_head, out_dir / "ys_head_sym.json")
    save_symbolic_module(sym_ys_head, out_dir / "ys_head_sym.pt")

    logger.info("Distilling fs_head")
    sym_fs_head = distill_block(
        teacher.fs_head,
        h_fs_sample,
        block_name="fs_head",
        parent_model=None,
        sr_params=sr_params,
        max_output_dim=None,
        variable_names=h_names,
        save_path=out_dir / "sr_cache" / "fs_head",
    )
    export_equations_json(sym_fs_head, out_dir / "fs_head_sym.json")
    save_symbolic_module(sym_fs_head, out_dir / "fs_head_sym.pt")

    hybrid = MedExpHybrid(teacher, sym_ys_enc, sym_fs_enc, sym_ys_head, sym_fs_head).eval()
    hybrid.to(device)
    torch.save(hybrid.state_dict(), out_dir / "hybrid_state.pt")

    with torch.no_grad():
        t_ys, t_fs = teacher_forward(teacher, x, edge_index, edge_type)
        h_ys_pred, h_fs_pred = hybrid(x, edge_index, edge_type)

    metrics = {
        "experiment": "medExp",
        "quick": args.quick,
        "encoder_sym_dir": str(enc_dir),
        "teacher": evaluate_predictions(t_ys, t_fs, ys, fs, train_mask, val_mask),
        "hybrid": evaluate_predictions(h_ys_pred, h_fs_pred, ys, fs, train_mask, val_mask),
    }
    save_metrics(out_dir / "metrics.json", metrics)
    write_summary_md(
        out_dir / "summary.md",
        "medExp",
        [
            "Symbolic: encoders + heads",
            "Neural: RGAT only",
            f"Formulas under `{out_dir}`",
            f"Val MAE hybrid YS/FS: {metrics['hybrid']['val_mae_ys']:.4f} / {metrics['hybrid']['val_mae_fs']:.4f}",
        ],
    )
    logger.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
