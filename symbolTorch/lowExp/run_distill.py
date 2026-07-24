#!/usr/bin/env python3
"""lowExp: tabular symbolic x -> YS/FS (no graph at inference)."""

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
    build_sr_params,
    distill_block_on_numpy_io,
    export_equations_json,
    make_tabular_lookup_fn,
    save_symbolic_module,
)
from common.equation_format import build_equation_record, write_equations_markdown  # noqa: E402
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
    logger.info("本次输出目录: %s", out_dir.resolve())
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

    sr_params = build_sr_params(
        niterations=args.sr_niterations,
        quick=args.quick,
        maxsize=(20 if args.quick else args.sr_maxsize),
    )

    data_dir = Path(args.data_dir)
    head0 = args.head0_name or ("UTS" if (data_dir / "uts.pt").is_file() and not (data_dir / "ys.pt").is_file() else "YS")
    head1 = args.head1_name or "FS"
    logger.info("方程目标名: %s / %s | SR niterations=%s maxsize=%s", head0, head1, sr_params.get("niterations"), sr_params.get("maxsize"))

    fn_ys = make_tabular_lookup_fn(x_np, ys_np)
    fn_fs = make_tabular_lookup_fn(x_np, fs_np)
    sr_out_root = out_dir / "SR_output"
    sr_out_root.mkdir(parents=True, exist_ok=True)
    logger.info("PySR 输出目录: %s/<block_name>/", sr_out_root.resolve())

    logger.info("Distilling tabular %s", head0)
    sym_ys = distill_block_on_numpy_io(
        fn_ys,
        x_np,
        block_name=f"tabular_{head0.lower()}",
        sr_params=sr_params,
        variable_names=list(FEATURE_NAMES),
        save_path=sr_out_root,
    )
    ys_payload = export_equations_json(sym_ys, out_dir / f"{head0.lower()}_tabular_sym.json", target=head0)
    save_symbolic_module(sym_ys, out_dir / f"{head0.lower()}_tabular_sym.pt", target=head0)

    logger.info("Distilling tabular %s", head1)
    sym_fs = distill_block_on_numpy_io(
        fn_fs,
        x_np,
        block_name=f"tabular_{head1.lower()}",
        sr_params=sr_params,
        variable_names=list(FEATURE_NAMES),
        save_path=sr_out_root,
    )
    fs_payload = export_equations_json(sym_fs, out_dir / f"{head1.lower()}_tabular_sym.json", target=head1)
    save_symbolic_module(sym_fs, out_dir / f"{head1.lower()}_tabular_sym.pt", target=head1)

    eq_records = [
        build_equation_record(head0, ys_payload["equation_raw"], block_name=f"tabular_{head0.lower()}"),
        build_equation_record(head1, fs_payload["equation_raw"], block_name=f"tabular_{head1.lower()}"),
    ]
    write_equations_markdown(
        out_dir / "equations.md",
        eq_records,
        notes=[
            f"量纲与标签 pt 一致（模型量纲）；第一头={head0}，第二头={head1}。",
            "推理仅使用节点 30 维特征，不使用图邻居。",
            f"SR: niterations={sr_params.get('niterations')} maxsize={sr_params.get('maxsize')}。",
            f"ckpt={args.ckpt} | data-dir={args.data_dir}",
        ],
    )

    tabular = TabularSymbolicModel(sym_ys, sym_fs).eval()
    with torch.no_grad():
        s_ys, s_fs = tabular(x)

    teacher_metrics = evaluate_predictions(t_ys, t_fs, ys, fs, train_mask, val_mask)
    tabular_metrics = evaluate_predictions(s_ys, s_fs, ys, fs, train_mask, val_mask)
    metrics = {
        "experiment": "lowExp",
        "graph_at_inference": False,
        "head0_name": head0,
        "head1_name": head1,
        "ckpt": str(Path(args.ckpt).resolve()),
        "data_dir": str(data_dir.resolve()),
        "sr": {"niterations": sr_params.get("niterations"), "maxsize": sr_params.get("maxsize"), "quick": bool(args.quick)},
        "teacher": teacher_metrics,
        "tabular_symbolic": tabular_metrics,
        "graph_info_loss_val_mae_head0": tabular_metrics["val_mae_ys"] - teacher_metrics["val_mae_ys"],
        "graph_info_loss_val_mae_head1": tabular_metrics["val_mae_fs"] - teacher_metrics["val_mae_fs"],
        # 兼容旧字段名
        "graph_info_loss_val_mae_ys": tabular_metrics["val_mae_ys"] - teacher_metrics["val_mae_ys"],
        "graph_info_loss_val_mae_fs": tabular_metrics["val_mae_fs"] - teacher_metrics["val_mae_fs"],
        "equations": {
            head0: ys_payload["equation"],
            head1: fs_payload["equation"],
        },
    }
    save_metrics(out_dir / "metrics.json", metrics)
    write_summary_md(
        out_dir / "summary.md",
        "lowExp",
        [
            f"Tabular symbolic: x (30) -> {head0} / {head1}",
            "No graph structure at inference",
            f"ckpt: {args.ckpt}",
            f"Val MAE tabular {head0}/{head1}: {tabular_metrics['val_mae_ys']:.4f} / {tabular_metrics['val_mae_fs']:.4f}",
            f"Graph info loss (tabular - teacher) val MAE: "
            f"{metrics['graph_info_loss_val_mae_ys']:.4f} / {metrics['graph_info_loss_val_mae_fs']:.4f}",
            "",
            "## 完整方程（亦见 equations.md）",
            "",
            "```text",
            ys_payload["equation"],
            fs_payload["equation"],
            "```",
        ],
    )
    logger.info("Done. Outputs in %s (see equations.md)", out_dir)


if __name__ == "__main__":
    main()
