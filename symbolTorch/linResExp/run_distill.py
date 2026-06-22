#!/usr/bin/env python3
"""linResExp: 30-dim Ridge linear basis + PySR residual for YS / FS (no graph at inference)."""

from __future__ import annotations

import os

# symtorch 会对符号 forward 做 torch.compile，CPU 上需要 g++；无编译器时禁用 dynamo
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

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
    _equation_string,
    build_sr_params,
    distill_block_on_numpy_io,
    export_equations_json,
    make_tabular_lookup_fn,
    save_symbolic_module,
)
from common.hybrid_models import TabularLinearOnlyModel, TabularLinResModel  # noqa: E402
from common.linear_basis import export_linres_json, fit_linear_basis, save_linear_basis  # noqa: E402
from common.metrics import evaluate_predictions, save_metrics, write_summary_md  # noqa: E402
from common.teacher import load_teacher, teacher_forward  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("linResExp")


def _residual_equation(sym) -> str:
    reg = sym.pysr_regressor
    if not reg:
        return "None"
    return _equation_string(reg[min(reg.keys())])


def main() -> None:
    experiment_header("linResExp")
    p = argparse.ArgumentParser(
        description="linResExp: 30-dim Ridge linear basis + PySR residual (graph-free inference)"
    )
    add_common_args(p)
    p.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge regularization for the 30-dim linear basis (default: 1.0)",
    )
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

    feature_names = list(FEATURE_NAMES)
    logger.info("Fitting 30-dim Ridge linear basis (alpha=%.4g)", args.ridge_alpha)
    basis_ys = fit_linear_basis(x_np, ys_np, feature_names=feature_names, alpha=args.ridge_alpha)
    basis_fs = fit_linear_basis(x_np, fs_np, feature_names=feature_names, alpha=args.ridge_alpha)
    save_linear_basis(basis_ys, out_dir / "ys_linear_basis.json")
    save_linear_basis(basis_fs, out_dir / "fs_linear_basis.json")

    ys_lin_np = basis_ys.predict_numpy(x_np).reshape(-1, 1)
    fs_lin_np = basis_fs.predict_numpy(x_np).reshape(-1, 1)
    ys_res_np = (ys_np.reshape(-1, 1) - ys_lin_np).astype(np.float32)
    fs_res_np = (fs_np.reshape(-1, 1) - fs_lin_np).astype(np.float32)

    sr_params = build_sr_params(niterations=args.sr_niterations, quick=args.quick, low_exp=True)

    fn_ys_res = make_tabular_lookup_fn(x_np, ys_res_np)
    fn_fs_res = make_tabular_lookup_fn(x_np, fs_res_np)

    logger.info("Distilling YS residual (PySR)")
    sym_res_ys = distill_block_on_numpy_io(
        fn_ys_res,
        x_np,
        block_name="tabular_ys_residual",
        sr_params=sr_params,
        variable_names=feature_names,
    )
    export_equations_json(sym_res_ys, out_dir / "ys_residual_sym.json")
    save_symbolic_module(sym_res_ys, out_dir / "ys_residual_sym.pt")

    logger.info("Distilling FS residual (PySR)")
    sym_res_fs = distill_block_on_numpy_io(
        fn_fs_res,
        x_np,
        block_name="tabular_fs_residual",
        sr_params=sr_params,
        variable_names=feature_names,
    )
    export_equations_json(sym_res_fs, out_dir / "fs_residual_sym.json")
    save_symbolic_module(sym_res_fs, out_dir / "fs_residual_sym.pt")

    export_linres_json(
        basis_ys,
        _residual_equation(sym_res_ys),
        out_dir / "ys_linres.json",
        block_name="tabular_ys_linres",
    )
    export_linres_json(
        basis_fs,
        _residual_equation(sym_res_fs),
        out_dir / "fs_linres.json",
        block_name="tabular_fs_linres",
    )

    linear_only = TabularLinearOnlyModel(basis_ys, basis_fs).eval()
    lin_res = TabularLinResModel(basis_ys, basis_fs, sym_res_ys, sym_res_fs).eval()

    with torch.no_grad():
        l_ys, l_fs = linear_only(x)
        s_ys, s_fs = lin_res(x)

    teacher_metrics = evaluate_predictions(t_ys, t_fs, ys, fs, train_mask, val_mask)
    linear_metrics = evaluate_predictions(l_ys, l_fs, ys, fs, train_mask, val_mask)
    linres_metrics = evaluate_predictions(s_ys, s_fs, ys, fs, train_mask, val_mask)
    metrics = {
        "experiment": "linResExp",
        "graph_at_inference": False,
        "ridge_alpha": float(args.ridge_alpha),
        "linear_feature_count": len(feature_names),
        "teacher": teacher_metrics,
        "linear_only": linear_metrics,
        "tabular_linres": linres_metrics,
        "graph_info_loss_val_mae_ys": linres_metrics["val_mae_ys"] - teacher_metrics["val_mae_ys"],
        "graph_info_loss_val_mae_fs": linres_metrics["val_mae_fs"] - teacher_metrics["val_mae_fs"],
        "linres_gain_vs_linear_val_mae_ys": linear_metrics["val_mae_ys"] - linres_metrics["val_mae_ys"],
        "linres_gain_vs_linear_val_mae_fs": linear_metrics["val_mae_fs"] - linres_metrics["val_mae_fs"],
    }
    save_metrics(out_dir / "metrics.json", metrics)
    write_summary_md(
        out_dir / "summary.md",
        "linResExp",
        [
            "30-dim Ridge linear basis (all features) + PySR residual",
            "No graph structure at inference",
            f"Ridge alpha: {args.ridge_alpha}",
            f"Val MAE linear-only YS/FS: {linear_metrics['val_mae_ys']:.4f} / {linear_metrics['val_mae_fs']:.4f}",
            f"Val MAE lin+res YS/FS: {linres_metrics['val_mae_ys']:.4f} / {linres_metrics['val_mae_fs']:.4f}",
            f"Graph info loss (lin+res - teacher) val MAE: "
            f"{metrics['graph_info_loss_val_mae_ys']:.4f} / {metrics['graph_info_loss_val_mae_fs']:.4f}",
        ],
    )
    logger.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
