#!/usr/bin/env python3
"""liner: linear Ridge/OLS distillation of RGAT teacher outputs + residual export."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.alignment import (  # noqa: E402
    assert_masks,
    duplicate_input_groups,
    is_full_train_masks,
    load_aligned_tabular,
)
from common.cli import (  # noqa: E402
    add_common_args,
    experiment_header,
    finalize_run_success,
    resolve_device,
    resolve_out_dir,
    set_seed,
)
from common.codec import codec_stats_dict  # noqa: E402
from common.constants import FEATURE_NAMES  # noqa: E402
from common.data import bundle_to_device, load_graph_bundle  # noqa: E402
from common.linear_fit import (  # noqa: E402
    compute_teacher_residual,
    fit_linear_teacher,
    predict_linear,
)
from common.manifest import build_liner_manifest, save_manifest  # noqa: E402
from common.metrics import (  # noqa: E402
    assert_finite,
    distillation_metrics,
    mae_np,
    save_metrics,
    write_summary_md,
)
from common.teacher import load_teacher, teacher_forward  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("liner")

DEFAULT_CSV = ROOT.parent / "gnnDir" / "datacsv" / "datagnnUts.csv"


def _resolve_head_names(args, data_dir: Path) -> tuple[str, str]:
    head0 = args.head0_name or (
        "UTS"
        if (data_dir / "uts.pt").is_file() and not (data_dir / "ys.pt").is_file()
        else "YS"
    )
    head1 = args.head1_name or "FS"
    if head0 not in ("YS", "UTS"):
        raise ValueError(f"--head0-name must be YS or UTS, got {head0}")
    if head1 != "FS":
        raise ValueError(f"--head1-name must be FS, got {head1}")
    return head0, head1


def run_liner(args: argparse.Namespace) -> Path:
    """Execute liner pipeline; return output directory. Raises on failure (no _SUCCESS)."""
    set_seed(args.seed)
    experiment_dir = Path(__file__).resolve().parent
    out_dir = resolve_out_dir(args, experiment_dir)
    logger.info("本次输出目录: %s", out_dir.resolve())
    device = resolve_device(args.device)

    csv_path = Path(args.csv)
    data_dir = Path(args.data_dir)
    graph, label0, fs, train_mask_t, val_mask_t = load_graph_bundle(data_dir)
    aligned = load_aligned_tabular(csv_path, graph, feature_names=FEATURE_NAMES)
    train_mask, val_mask = assert_masks(
        train_mask_t,
        val_mask_t,
        aligned.n,
        allow_empty_val=bool(getattr(args, "allow_empty_val", False)),
    )

    head0, head1 = _resolve_head_names(args, data_dir)
    x_dev, ys_dev, fs_dev, tm_dev, vm_dev, edge_index, edge_type = bundle_to_device(
        graph, label0, fs, train_mask_t, val_mask_t, device
    )
    in_dim = int(x_dev.shape[1])
    teacher = load_teacher(
        Path(args.ckpt),
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        device=device,
        dropout=args.dropout,
    )

    with torch.no_grad():
        t_h0, t_fs = teacher_forward(teacher, x_dev, edge_index, edge_type)
    teacher_head0 = t_h0.detach().cpu().numpy().astype(np.float64).reshape(-1)
    teacher_fs = t_fs.detach().cpu().numpy().astype(np.float64).reshape(-1)
    label_head0 = ys_dev.detach().cpu().numpy().astype(np.float64).reshape(-1)
    label_fs = fs_dev.detach().cpu().numpy().astype(np.float64).reshape(-1)
    assert_finite("teacher_head0", teacher_head0)
    assert_finite("teacher_fs", teacher_fs)

    X_all = aligned.X_all
    X_train = X_all[train_mask]
    method = str(args.method).lower()
    alpha_arg = None if args.alpha is None else float(args.alpha)

    fit_h0 = fit_linear_teacher(
        X_train,
        teacher_head0[train_mask],
        method=method,
        alpha=alpha_arg,
        seed=int(args.seed),
    )
    fit_fs = fit_linear_teacher(
        X_train,
        teacher_fs[train_mask],
        method=method,
        alpha=alpha_arg,
        seed=int(args.seed),
    )

    linear_head0 = predict_linear(X_all, fit_h0.intercept, fit_h0.coefficients)
    linear_fs = predict_linear(X_all, fit_fs.intercept, fit_fs.coefficients)
    residual_head0 = compute_teacher_residual(teacher_head0, linear_head0)
    residual_fs = compute_teacher_residual(teacher_fs, linear_fs)
    assert_finite("linear_head0", linear_head0)
    assert_finite("linear_fs", linear_fs)
    assert_finite("residual_head0", residual_head0)
    assert_finite("residual_fs", residual_fs)

    # replay check from JSON dict
    for name, fit, pred in (
        (head0, fit_h0, linear_head0),
        (head1, fit_fs, linear_fs),
    ):
        d = fit.to_json_dict(FEATURE_NAMES)
        replay = predict_linear(X_all, d["intercept"], np.asarray(d["coefficients"]))
        if not np.allclose(replay, pred, rtol=1e-10, atol=1e-10):
            raise ValueError(f"linear JSON replay mismatch for {name}")

    _, _, dup_summary = duplicate_input_groups(X_all)
    # irreducible error lower bound: within-dup-group residual variance
    group_id, counts, _ = duplicate_input_groups(X_all)
    irr = {}
    for head, resid in ((head0, residual_head0), (head1, residual_fs)):
        vars_ = []
        for g in range(int(counts.shape[0])):
            if counts[g] <= 1:
                continue
            idx = np.where(group_id == g)[0]
            vars_.append(float(np.var(resid[idx])))
        irr[head] = {
            "n_dup_groups_with_var": len(vars_),
            "mean_within_group_residual_var": float(np.mean(vars_)) if vars_ else 0.0,
            "max_within_group_residual_var": float(np.max(vars_)) if vars_ else 0.0,
        }

    run_id = out_dir.name
    codec = codec_stats_dict()
    manifest = build_liner_manifest(
        csv=csv_path,
        data_dir=data_dir,
        ckpt=args.ckpt,
        head0_name=head0,
        head1_name=head1,
        feature_names=FEATURE_NAMES,
        n_samples=aligned.n,
        train_mask=train_mask,
        val_mask=val_mask,
        method=method,
        alpha=fit_h0.alpha if method == "ridge" else None,
        seed=int(args.seed),
        run_id=run_id,
        extra={
            "alpha_head0": fit_h0.alpha,
            "alpha_fs": fit_fs.alpha,
            "cv_alphas_head0": fit_h0.cv_alphas,
            "cv_scores_head0": fit_h0.cv_scores,
            "cv_alphas_fs": fit_fs.cv_alphas,
            "cv_scores_fs": fit_fs.cv_scores,
            "rank_deficient_head0": fit_h0.rank_deficient,
            "rank_deficient_fs": fit_fs.rank_deficient,
            "duplicate_inputs": dup_summary,
            "irreducible_residual": irr,
            "codec": codec,
            "independent_val": not is_full_train_masks(train_mask, val_mask),
            "full_train_mode": is_full_train_masks(train_mask, val_mask),
        },
    )
    save_manifest(out_dir / "manifest.json", manifest)

    h0_json = fit_h0.to_json_dict(FEATURE_NAMES)
    h0_json["target"] = head0
    fs_json = fit_fs.to_json_dict(FEATURE_NAMES)
    fs_json["target"] = head1
    with (out_dir / "linear_head0.json").open("w", encoding="utf-8") as f:
        json.dump(h0_json, f, indent=2, ensure_ascii=False)
    with (out_dir / "linear_fs.json").open("w", encoding="utf-8") as f:
        json.dump(fs_json, f, indent=2, ensure_ascii=False)

    torch.save(
        {
            "teacher_head0": torch.from_numpy(teacher_head0),
            "teacher_fs": torch.from_numpy(teacher_fs),
            "head0_name": head0,
            "head1_name": head1,
            "sample_id": torch.from_numpy(aligned.sample_id),
        },
        out_dir / "teacher_predictions.pt",
    )
    torch.save(
        {
            "linear_head0": torch.from_numpy(linear_head0),
            "linear_fs": torch.from_numpy(linear_fs),
            "sample_id": torch.from_numpy(aligned.sample_id),
        },
        out_dir / "linear_predictions.pt",
    )
    torch.save(
        {
            "X_all": torch.from_numpy(X_all),
            "teacher_head0": torch.from_numpy(teacher_head0),
            "teacher_fs": torch.from_numpy(teacher_fs),
            "linear_head0": torch.from_numpy(linear_head0),
            "linear_fs": torch.from_numpy(linear_fs),
            "residual_head0": torch.from_numpy(residual_head0),
            "residual_fs": torch.from_numpy(residual_fs),
            "train_mask": torch.from_numpy(train_mask),
            "val_mask": torch.from_numpy(val_mask),
            "sample_id": torch.from_numpy(aligned.sample_id),
            "feature_names": list(FEATURE_NAMES),
            "head0_name": head0,
            "head1_name": head1,
            "manifest_fingerprint": manifest["fingerprint"],
            "label_head0": torch.from_numpy(label_head0),
            "label_fs": torch.from_numpy(label_fs),
        },
        out_dir / "residual_targets.pt",
    )

    eq_lines = [
        "# liner linear equations",
        "",
        f"- method: `{method}`",
        f"- alpha head0/fs: `{fit_h0.alpha}` / `{fit_fs.alpha}`",
        f"- fingerprint: `{manifest['fingerprint']}`",
        "",
        f"## {head0}",
        "",
        "```text",
        f"L_{head0}(x) = {h0_json['equation_rhs_display']}",
        "```",
        "",
        f"## {head1}",
        "",
        "```text",
        f"L_{head1}(x) = {fs_json['equation_rhs_display']}",
        "```",
        "",
        "机器可读 RHS 见 linear_*.json 的 equation_rhs_raw（高精度）。",
        "公式结构包含全部 30 项，不代表每项具有显著物理贡献。",
        "",
    ]
    (out_dir / "linear_equations.md").write_text("\n".join(eq_lines), encoding="utf-8")

    metrics: Dict[str, Any] = {
        "experiment": "liner",
        "head0_name": head0,
        "head1_name": head1,
        "method": method,
        "fingerprint": manifest["fingerprint"],
        "teacher_vs_label": {
            "train_mae_head0": mae_np(teacher_head0, label_head0, train_mask),
            "val_mae_head0": mae_np(teacher_head0, label_head0, val_mask),
            "train_mae_fs": mae_np(teacher_fs, label_fs, train_mask),
            "val_mae_fs": mae_np(teacher_fs, label_fs, val_mask),
        },
        "linear_vs_label": {
            "train_mae_head0": mae_np(linear_head0, label_head0, train_mask),
            "val_mae_head0": mae_np(linear_head0, label_head0, val_mask),
            "train_mae_fs": mae_np(linear_fs, label_fs, train_mask),
            "val_mae_fs": mae_np(linear_fs, label_fs, val_mask),
        },
    }
    metrics.update(
        distillation_metrics(
            teacher=teacher_head0,
            linear=linear_head0,
            residual_true=residual_head0,
            label=label_head0,
            train_mask=train_mask,
            val_mask=val_mask,
            head="head0",
        )
    )
    metrics.update(
        distillation_metrics(
            teacher=teacher_fs,
            linear=linear_fs,
            residual_true=residual_fs,
            label=label_fs,
            train_mask=train_mask,
            val_mask=val_mask,
            head="fs",
        )
    )
    save_metrics(out_dir / "metrics.json", metrics)
    write_summary_md(
        out_dir / "summary.md",
        "liner",
        [
            f"Linear distillation of RGAT teacher → {head0}/{head1}",
            f"method={method} alpha_h0={fit_h0.alpha} alpha_fs={fit_fs.alpha}",
            f"val MAE linear vs teacher: "
            f"{metrics.get('val_mae_linear_vs_teacher_head0')} / "
            f"{metrics.get('val_mae_linear_vs_teacher_fs')}",
            f"fingerprint={manifest['fingerprint']}",
        ],
    )

    runs_root = experiment_dir / "runs"
    finalize_run_success(runs_root if out_dir.parent == runs_root else None, out_dir)
    logger.info("Done. Outputs in %s", out_dir)
    return out_dir


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="liner: linear RGAT teacher distillation")
    add_common_args(p)
    p.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Model-space CSV (default datagnnUts.csv)",
    )
    p.add_argument(
        "--method",
        type=str,
        default="ridge",
        choices=["ols", "ridge"],
        help="Linear method (default ridge)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Ridge alpha; if omitted, choose via train-set CV",
    )
    p.add_argument(
        "--allow-empty-val",
        action="store_true",
        help="Allow empty val_mask (full-train mode; metrics not independent val)",
    )
    return p


def main(argv: Optional[list] = None) -> None:
    experiment_header("liner")
    args = build_argparser().parse_args(argv)
    run_liner(args)


if __name__ == "__main__":
    main()
