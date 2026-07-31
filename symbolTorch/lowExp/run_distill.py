#!/usr/bin/env python3
"""lowExp: symbolic regression on liner teacher residuals (graph-free)."""

from __future__ import annotations

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.alignment import duplicate_input_groups  # noqa: E402
from common.cli import (  # noqa: E402
    add_common_args,
    experiment_header,
    finalize_run_success,
    resolve_device,
    resolve_out_dir,
    set_seed,
)
from common.constants import FEATURE_NAMES  # noqa: E402
from common.distill_io import (  # noqa: E402
    build_sr_params,
    distill_residual_targets,
    evaluate_symbolic_numpy,
    export_equations_json,
    get_selected_sympy_expr,
    save_symbolic_module,
)
from common.equation_format import build_equation_record, write_equations_markdown  # noqa: E402
from common.expr_ir import validate_expression_tree  # noqa: E402
from common.liner_io import load_and_validate_liner_run  # noqa: E402
from common.manifest import save_manifest  # noqa: E402
from common.metrics import (  # noqa: E402
    assert_finite,
    distillation_metrics,
    save_metrics,
    write_summary_md,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("lowExp")


def run_lowexp(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    experiment_dir = Path(__file__).resolve().parent
    out_dir = resolve_out_dir(args, experiment_dir)
    logger.info("本次输出目录: %s", out_dir.resolve())
    _ = resolve_device(args.device)  # reserved for future GPU symbolic eval

    if not getattr(args, "liner_run", None):
        raise ValueError("lowExp residual mode requires --liner-run <liner/runs/...>")

    payload = load_and_validate_liner_run(Path(args.liner_run))
    head0 = payload.head0_name
    head1 = payload.head1_name
    X_all = payload.X_all
    train_mask = payload.train_mask
    val_mask = payload.val_mask
    X_train = X_all[train_mask]

    if getattr(args, "include_val", False):
        logger.warning(
            "--include-val is ignored in residual mode; fitting uses train_mask only. "
            "Results remain independently validated on val_mask."
        )

    sr_params = build_sr_params(
        niterations=args.sr_niterations,
        quick=args.quick,
        maxsize=(20 if args.quick else args.sr_maxsize),
    )
    sr_out_root = out_dir / "SR_output"
    sr_out_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Residual SR %s/%s | niterations=%s maxsize=%s | liner=%s",
        head0,
        head1,
        sr_params.get("niterations"),
        sr_params.get("maxsize"),
        payload.run_dir,
    )

    logger.info("Distilling residual %s", head0)
    sym_h0 = distill_residual_targets(
        X_train,
        payload.residual_head0[train_mask],
        block_name=f"residual_{head0.lower()}",
        variable_names=list(FEATURE_NAMES),
        sr_params=sr_params,
        save_path=sr_out_root,
    )
    h0_payload = export_equations_json(
        sym_h0, out_dir / f"{head0.lower()}_residual_sym.json", target=f"R_{head0}"
    )
    save_symbolic_module(
        sym_h0, out_dir / f"{head0.lower()}_residual_sym.pt", target=f"R_{head0}"
    )

    logger.info("Distilling residual %s", head1)
    sym_fs = distill_residual_targets(
        X_train,
        payload.residual_fs[train_mask],
        block_name=f"residual_{head1.lower()}",
        variable_names=list(FEATURE_NAMES),
        sr_params=sr_params,
        save_path=sr_out_root,
    )
    fs_payload = export_equations_json(
        sym_fs, out_dir / f"{head1.lower()}_residual_sym.json", target=f"R_{head1}"
    )
    save_symbolic_module(
        sym_fs, out_dir / f"{head1.lower()}_residual_sym.pt", target=f"R_{head1}"
    )

    # validate sympy trees
    expr_h0 = get_selected_sympy_expr(sym_h0)
    expr_fs = get_selected_sympy_expr(sym_fs)
    validate_expression_tree(expr_h0, allowed_symbols=FEATURE_NAMES)
    validate_expression_tree(expr_fs, allowed_symbols=FEATURE_NAMES)

    rhat_h0 = evaluate_symbolic_numpy(sym_h0, X_all)
    rhat_fs = evaluate_symbolic_numpy(sym_fs, X_all)
    assert_finite("symbolic_residual_head0", rhat_h0)
    assert_finite("symbolic_residual_fs", rhat_fs)
    if rhat_h0.shape[0] != X_all.shape[0] or rhat_fs.shape[0] != X_all.shape[0]:
        raise ValueError("symbolic residual shape mismatch")

    # train/val/all finite already via assert_finite on all
    for split, mask in (
        ("train", train_mask),
        ("val", val_mask),
        ("all", np.ones(X_all.shape[0], dtype=bool)),
    ):
        assert_finite(f"rhat_h0_{split}", rhat_h0[mask])
        assert_finite(f"rhat_fs_{split}", rhat_fs[mask])

    group_id, counts, dup_summary = duplicate_input_groups(X_all)
    irr = {}
    for name, resid in ((head0, payload.residual_head0), (head1, payload.residual_fs)):
        vars_ = []
        for g in range(int(counts.shape[0])):
            if counts[g] <= 1:
                continue
            idx = np.where(group_id == g)[0]
            vars_.append(float(np.var(resid[idx])))
        irr[name] = {
            "mean_within_group_residual_var": float(np.mean(vars_)) if vars_ else 0.0,
            "max_within_group_residual_var": float(np.max(vars_)) if vars_ else 0.0,
        }

    eq_records = [
        build_equation_record(
            f"R_{head0}", h0_payload["equation_raw"], block_name=f"residual_{head0.lower()}"
        ),
        build_equation_record(
            f"R_{head1}", fs_payload["equation_raw"], block_name=f"residual_{head1.lower()}"
        ),
    ]
    write_equations_markdown(
        out_dir / "residual_equations.md",
        eq_records,
        title="lowExp residual equations",
        notes=[
            "拟合目标为 teacher - linear（liner 残差），不是完整教师输出或真实标签。",
            "推理仅使用节点 30 维特征，不使用图邻居。",
            f"SLIME=False | SR niterations={sr_params.get('niterations')} maxsize={sr_params.get('maxsize')}",
            f"liner_run={payload.run_dir}",
            f"liner_fingerprint={payload.manifest.get('fingerprint')}",
        ],
    )

    torch.save(
        {
            "symbolic_residual_head0": torch.from_numpy(rhat_h0),
            "symbolic_residual_fs": torch.from_numpy(rhat_fs),
            "sample_id": torch.from_numpy(payload.sample_id),
            "head0_name": head0,
            "head1_name": head1,
            "liner_fingerprint": payload.manifest["fingerprint"],
        },
        out_dir / "residual_predictions.pt",
    )

    lowexp_manifest: Dict[str, Any] = {
        "module": "lowExp",
        "run_id": out_dir.name,
        "liner_run": str(payload.run_dir),
        "liner_fingerprint": payload.manifest["fingerprint"],
        "csv": payload.manifest.get("csv"),
        "data_dir": payload.manifest.get("data_dir"),
        "ckpt": payload.manifest.get("ckpt"),
        "head0_name": head0,
        "head1_name": head1,
        "feature_names": list(FEATURE_NAMES),
        "feature_order_hash": payload.manifest.get("feature_order_hash"),
        "train_mask_hash": payload.manifest.get("train_mask_hash"),
        "val_mask_hash": payload.manifest.get("val_mask_hash"),
        "fingerprint": payload.manifest["fingerprint"],
        "seed": int(args.seed),
        "sr": {
            "niterations": sr_params.get("niterations"),
            "maxsize": sr_params.get("maxsize"),
            "quick": bool(args.quick),
            "SLIME": False,
        },
        "duplicate_inputs": dup_summary,
        "irreducible_residual": irr,
        "residual_equation_raw": {
            head0: h0_payload["equation_raw"],
            head1: fs_payload["equation_raw"],
        },
    }
    save_manifest(out_dir / "manifest.json", lowexp_manifest)

    metrics: Dict[str, Any] = {
        "experiment": "lowExp",
        "mode": "residual",
        "graph_at_inference": False,
        "head0_name": head0,
        "head1_name": head1,
        "liner_run": str(payload.run_dir),
        "liner_fingerprint": payload.manifest["fingerprint"],
        "sr": lowexp_manifest["sr"],
        "equations": {
            f"R_{head0}": h0_payload["equation"],
            f"R_{head1}": fs_payload["equation"],
        },
        "variables_used": {
            head0: h0_payload.get("variables_used"),
            head1: fs_payload.get("variables_used"),
        },
    }
    metrics.update(
        distillation_metrics(
            teacher=payload.teacher_head0,
            linear=payload.linear_head0,
            residual_true=payload.residual_head0,
            residual_pred=rhat_h0,
            combined=payload.linear_head0 + rhat_h0,
            label=payload.label_head0,
            train_mask=train_mask,
            val_mask=val_mask,
            head="head0",
        )
    )
    metrics.update(
        distillation_metrics(
            teacher=payload.teacher_fs,
            linear=payload.linear_fs,
            residual_true=payload.residual_fs,
            residual_pred=rhat_fs,
            combined=payload.linear_fs + rhat_fs,
            label=payload.label_fs,
            train_mask=train_mask,
            val_mask=val_mask,
            head="fs",
        )
    )
    save_metrics(out_dir / "metrics.json", metrics)
    write_summary_md(
        out_dir / "summary.md",
        "lowExp residual",
        [
            f"Symbolic residual: x(30) -> R_{head0} / R_{head1}",
            f"liner: {payload.run_dir}",
            f"Val MAE sym residual vs teacher residual: "
            f"{metrics.get('val_mae_sym_residual_head0')} / "
            f"{metrics.get('val_mae_sym_residual_fs')}",
            f"Val distillation MAE (linear+R vs teacher): "
            f"{metrics.get('distillation_mae_val_head0')} / "
            f"{metrics.get('distillation_mae_val_fs')}",
            "",
            "## Residual equations",
            "",
            "```text",
            h0_payload["equation"],
            fs_payload["equation"],
            "```",
        ],
    )

    runs_root = experiment_dir / "runs"
    finalize_run_success(runs_root if out_dir.parent == runs_root else None, out_dir)
    logger.info("Done. Outputs in %s", out_dir)
    return out_dir


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="lowExp: symbolic residual model (requires --liner-run)"
    )
    add_common_args(p)
    p.add_argument(
        "--liner-run",
        type=Path,
        required=True,
        help="Path to successful liner/runs/<id> (must contain residual_targets.pt + _SUCCESS)",
    )
    return p


def main(argv: Optional[list] = None) -> None:
    experiment_header("lowExp")
    args = build_argparser().parse_args(argv)
    run_lowexp(args)


if __name__ == "__main__":
    main()
