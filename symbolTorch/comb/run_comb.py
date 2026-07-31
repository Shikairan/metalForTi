#!/usr/bin/env python3
"""comb: joint liner + lowExp entry; combine linear + residual formulas."""

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

from common.cli import (  # noqa: E402
    add_common_args,
    experiment_header,
    finalize_run_success,
    resolve_device,
    resolve_out_dir,
    set_seed,
)
from common.codec import codec_stats_dict, physical_docs_notes  # noqa: E402
from common.constants import FEATURE_NAMES  # noqa: E402
from common.distill_io import get_selected_sympy_expr  # noqa: E402
from common.equation_format import format_full_equation  # noqa: E402
from common.expr_ir import (  # noqa: E402
    EVALUATOR_VERSION,
    build_linear_sympy_expr,
    combine_linear_residual,
    evaluate_expression_numpy,
    parse_sympy_expr,
    serialize_expression_ir,
    validate_expression_tree,
)
from common.linear_fit import linear_json_with_min_coef  # noqa: E402
from common.manifest import (  # noqa: E402
    assert_compatible_manifests,
    has_success_marker,
    load_manifest,
    save_manifest,
)
from common.metrics import (  # noqa: E402
    assert_finite,
    distillation_metrics,
    save_metrics,
    write_summary_md,
)
from common.per_sample_export import export_per_sample_errors  # noqa: E402
from common.alignment import is_full_train_masks  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("comb")

DEFAULT_CSV = ROOT.parent / "gnnDir" / "datacsv" / "datagnnUts.csv"


def _load_lowexp_run(lowexp_run: Path) -> Dict[str, Any]:
    lowexp_run = Path(lowexp_run)
    if not has_success_marker(lowexp_run):
        raise FileNotFoundError(f"lowExp run missing _SUCCESS: {lowexp_run}")
    man = load_manifest(lowexp_run / "manifest.json")
    pred = torch.load(lowexp_run / "residual_predictions.pt", map_location="cpu", weights_only=False)
    return {"run_dir": lowexp_run.resolve(), "manifest": man, "predictions": pred}


def _sympy_from_lowexp_json(lowexp_run: Path, head: str, feature_names):
    path = lowexp_run / f"{head.lower()}_residual_sym.json"
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    raw = payload.get("equation_raw") or payload.get("equations", {}).get("0")
    if not raw:
        raise ValueError(f"missing equation_raw in {path}")
    return parse_sympy_expr(str(raw), feature_names), str(raw), payload


def _try_load_dill_sym(path: Path):
    try:
        import dill

        with path.open("rb") as f:
            return dill.load(f)
    except Exception:
        return None


def run_comb(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    experiment_dir = Path(__file__).resolve().parent
    out_dir = resolve_out_dir(args, experiment_dir)
    logger.info("本次输出目录: %s", out_dir.resolve())
    resolve_device(args.device)

    liner_run = Path(args.liner_run) if args.liner_run else None
    lowexp_run = Path(args.lowexp_run) if args.lowexp_run else None

    # --- liner ---
    if liner_run is None:
        from liner.run_liner import build_argparser as liner_parser
        from liner.run_liner import run_liner

        liner_args = liner_parser().parse_args([])
        # overlay from comb args
        liner_args.csv = args.csv
        liner_args.data_dir = args.data_dir
        liner_args.ckpt = args.ckpt
        liner_args.head0_name = args.head0_name
        liner_args.head1_name = args.head1_name or "FS"
        liner_args.method = args.linear_method
        liner_args.alpha = args.alpha
        liner_args.hidden_dim = args.hidden_dim
        liner_args.dropout = args.dropout
        liner_args.seed = args.seed
        liner_args.device = args.device
        liner_args.out_dir = out_dir / "liner_stage"
        liner_args.run_name = None
        liner_args.allow_empty_val = bool(getattr(args, "allow_empty_val", False))
        logger.info("Running liner stage -> %s", liner_args.out_dir)
        liner_run = run_liner(liner_args)
    else:
        liner_run = liner_run.resolve()

    liner = load_and_validate_liner_run(liner_run)

    # --- lowExp ---
    if lowexp_run is None:
        from lowExp.run_distill import build_argparser as low_parser
        from lowExp.run_distill import run_lowexp

        low_args = low_parser().parse_args(["--liner-run", str(liner_run)])
        low_args.liner_run = liner_run
        low_args.sr_niterations = args.sr_niterations
        low_args.sr_maxsize = args.sr_maxsize
        low_args.quick = args.quick
        low_args.seed = args.seed
        low_args.device = args.device
        low_args.out_dir = out_dir / "lowexp_stage"
        low_args.run_name = None
        low_args.data_dir = args.data_dir
        low_args.ckpt = args.ckpt
        low_args.hidden_dim = args.hidden_dim
        low_args.dropout = args.dropout
        low_args.head0_name = liner.head0_name
        low_args.head1_name = liner.head1_name
        logger.info("Running lowExp stage -> %s", low_args.out_dir)
        lowexp_run = run_lowexp(low_args)
    else:
        lowexp_run = lowexp_run.resolve()

    low = _load_lowexp_run(lowexp_run)
    assert_compatible_manifests(liner.manifest, low["manifest"])

    head0 = liner.head0_name
    head1 = liner.head1_name
    X_all = liner.X_all
    train_mask = liner.train_mask
    val_mask = liner.val_mask

    rhat_h0 = (
        low["predictions"]["symbolic_residual_head0"].detach().cpu().numpy().astype(np.float64).reshape(-1)
    )
    rhat_fs = (
        low["predictions"]["symbolic_residual_fs"].detach().cpu().numpy().astype(np.float64).reshape(-1)
    )
    if not np.array_equal(low["predictions"]["sample_id"].numpy(), liner.sample_id):
        # allow tensor vs numpy
        sid = np.asarray(low["predictions"]["sample_id"]).reshape(-1)
        if not np.array_equal(sid, liner.sample_id):
            raise ValueError("sample_id mismatch between liner and lowExp")

    combined_h0 = liner.linear_head0 + rhat_h0
    combined_fs = liner.linear_fs + rhat_fs
    assert_finite("combined_head0", combined_h0)
    assert_finite("combined_fs", combined_fs)

    # linear json
    with (liner_run / "linear_head0.json").open("r", encoding="utf-8") as f:
        lin_h0 = json.load(f)
    with (liner_run / "linear_fs.json").open("r", encoding="utf-8") as f:
        lin_fs = json.load(f)

    lin_h0_export = linear_json_with_min_coef(lin_h0)
    lin_fs_export = linear_json_with_min_coef(lin_fs)

    linear_expr_h0 = build_linear_sympy_expr(
        lin_h0["intercept"], lin_h0["coefficients"], FEATURE_NAMES
    )
    linear_expr_fs = build_linear_sympy_expr(
        lin_fs["intercept"], lin_fs["coefficients"], FEATURE_NAMES
    )

    # prefer dill sympy if available else JSON
    sym_h0 = _try_load_dill_sym(lowexp_run / f"{head0.lower()}_residual_sym.pt")
    sym_fs = _try_load_dill_sym(lowexp_run / f"{head1.lower()}_residual_sym.pt")
    if sym_h0 is not None:
        resid_expr_h0 = get_selected_sympy_expr(sym_h0)
        raw_h0 = str(resid_expr_h0)
    else:
        resid_expr_h0, raw_h0, _ = _sympy_from_lowexp_json(lowexp_run, head0, FEATURE_NAMES)
    if sym_fs is not None:
        resid_expr_fs = get_selected_sympy_expr(sym_fs)
        raw_fs = str(resid_expr_fs)
    else:
        resid_expr_fs, raw_fs, _ = _sympy_from_lowexp_json(lowexp_run, head1, FEATURE_NAMES)

    validate_expression_tree(resid_expr_h0, allowed_symbols=FEATURE_NAMES)
    validate_expression_tree(resid_expr_fs, allowed_symbols=FEATURE_NAMES)

    combined_expr_h0 = combine_linear_residual(linear_expr_h0, resid_expr_h0)
    combined_expr_fs = combine_linear_residual(linear_expr_fs, resid_expr_fs)

    formula_h0 = evaluate_expression_numpy(combined_expr_h0, X_all, FEATURE_NAMES)
    formula_fs = evaluate_expression_numpy(combined_expr_fs, X_all, FEATURE_NAMES)
    assert_finite("formula_head0", formula_h0)
    assert_finite("formula_fs", formula_fs)

    if not np.allclose(formula_h0, combined_h0, rtol=1e-5, atol=1e-6):
        raise ValueError(
            f"combined formula vs modules mismatch head0 "
            f"(max abs={np.max(np.abs(formula_h0 - combined_h0))})"
        )
    if not np.allclose(formula_fs, combined_fs, rtol=1e-5, atol=1e-6):
        raise ValueError(
            f"combined formula vs modules mismatch fs "
            f"(max abs={np.max(np.abs(formula_fs - combined_fs))})"
        )

    export_linear_expr_h0 = build_linear_sympy_expr(
        lin_h0_export["intercept"], lin_h0_export["coefficients"], FEATURE_NAMES
    )
    export_linear_expr_fs = build_linear_sympy_expr(
        lin_fs_export["intercept"], lin_fs_export["coefficients"], FEATURE_NAMES
    )
    export_combined_expr_h0 = combine_linear_residual(export_linear_expr_h0, resid_expr_h0)
    export_combined_expr_fs = combine_linear_residual(export_linear_expr_fs, resid_expr_fs)

    codec = codec_stats_dict()
    final_equations = {
        "evaluator_version": EVALUATOR_VERSION,
        "min_linear_coefficient_abs": lin_h0_export["min_coefficient_abs"],
        "feature_names": list(FEATURE_NAMES),
        "heads": {},
        "codec": codec,
        "model_to_physical_input_transform": {
            "element": "wt% unchanged",
            "testenv": "z-score with testenv_mean/std; fcr == sr pipeline for data1123",
            "coldway": "active * ln(T/T_DIV), active * ln(t); [0,0] ambiguous",
        },
        "model_to_physical_output_transform": {
            "YS_MPa": "YS_model * mean_YS",
            "UTS_MPa": "UTS_model * mean_UTS",
            "FS_dataOri2": "FS_model * mean_FS",
            "FS_data1123": "FS_dataOri2 / 100",
        },
        "units": {"YS": "MPa", "UTS": "MPa", "FS_dataOri2": "% elongation scale", "FS_data1123": "data1123 FS"},
    }
    for head, resid_raw, comb_expr, lin_json in (
        (head0, raw_h0, export_combined_expr_h0, lin_h0_export),
        (head1, raw_fs, export_combined_expr_fs, lin_fs_export),
    ):
        final_equations["heads"][head] = {
            "target": head,
            "feature_names": list(FEATURE_NAMES),
            "linear_intercept": lin_json["intercept"],
            "linear_coefficients": lin_json["coefficients"],
            "linear_equation_raw": lin_json["equation_rhs_raw"],
            "residual_equation_raw": resid_raw,
            "combined_equation_raw": str(comb_expr),
            "combined_equation_display": format_full_equation(head, str(comb_expr)),
            "expression_ir": serialize_expression_ir(comb_expr),
            "model_to_physical_input_transform": final_equations["model_to_physical_input_transform"],
            "model_to_physical_output_transform": final_equations["model_to_physical_output_transform"],
            "units": final_equations["units"],
        }

    with (out_dir / "final_equations.json").open("w", encoding="utf-8") as f:
        json.dump(final_equations, f, indent=2, ensure_ascii=False)

    # model space md
    ms_lines = [
        "# Final equations (model space)",
        "",
        "可直接作用于 CSV 前 30 列（模型空间）。",
        "",
        f"## {head0}",
        "",
        "```text",
        final_equations["heads"][head0]["combined_equation_display"],
        "```",
        "",
        "### Linear",
        "",
        "```text",
        f"L = {lin_h0_export['equation_rhs_display']}",
        "```",
        "",
        "### Residual",
        "",
        "```text",
        f"R = {raw_h0}",
        "```",
        "",
        f"## {head1}",
        "",
        "```text",
        final_equations["heads"][head1]["combined_equation_display"],
        "```",
        "",
        "### Linear",
        "",
        "```text",
        f"L = {lin_fs_export['equation_rhs_display']}",
        "```",
        "",
        "### Residual",
        "",
        "```text",
        f"R = {raw_fs}",
        "```",
        "",
    ]
    (out_dir / "final_equations_model_space.md").write_text("\n".join(ms_lines), encoding="utf-8")

    phys_lines = [
        "# Final equations (physical interpretation)",
        "",
        "完整模型公式仍在模型空间求值。以下为输入/输出编解码规则（依据 preprocess_datagnn_repro.py）。",
        "",
        "## Codec stats",
        "",
        "```json",
        json.dumps(codec, indent=2),
        "```",
        "",
        "## Notes",
        "",
    ]
    for n in physical_docs_notes():
        phys_lines.append(f"- {n}")
    phys_lines.extend(
        [
            "",
            "## Model-space formulas (do not treat tem/fcr/coldway as raw physical fields)",
            "",
            "```text",
            final_equations["heads"][head0]["combined_equation_display"],
            final_equations["heads"][head1]["combined_equation_display"],
            "```",
            "",
            "coldway 编码不完全可逆：不生成虚假唯一逆解。",
            "",
        ]
    )
    (out_dir / "final_equations_physical.md").write_text("\n".join(phys_lines), encoding="utf-8")

    torch.save(
        {
            "linear_head0": torch.from_numpy(liner.linear_head0),
            "linear_fs": torch.from_numpy(liner.linear_fs),
            "symbolic_residual_head0": torch.from_numpy(rhat_h0),
            "symbolic_residual_fs": torch.from_numpy(rhat_fs),
            "combined_head0": torch.from_numpy(combined_h0),
            "combined_fs": torch.from_numpy(combined_fs),
            "teacher_head0": torch.from_numpy(liner.teacher_head0),
            "teacher_fs": torch.from_numpy(liner.teacher_fs),
            "sample_id": torch.from_numpy(liner.sample_id),
            "head0_name": head0,
            "head1_name": head1,
        },
        out_dir / "predictions.pt",
    )

    (out_dir / "liner_run.txt").write_text(str(liner_run) + "\n", encoding="utf-8")
    (out_dir / "lowexp_run.txt").write_text(str(lowexp_run) + "\n", encoding="utf-8")

    # Copy linear / residual equation docs for convenience
    import shutil

    for src_name, dst_name in (
        (liner_run / "linear_equations.md", "linear_equations.md"),
        (lowexp_run / "residual_equations.md", "residual_equations.md"),
    ):
        if Path(src_name).is_file():
            shutil.copy2(src_name, out_dir / dst_name)

    full_train = is_full_train_masks(train_mask, val_mask)
    per_sample_info = export_per_sample_errors(
        out_dir,
        sample_id=liner.sample_id,
        head0_name=head0,
        head1_name=head1,
        teacher_head0=liner.teacher_head0,
        teacher_fs=liner.teacher_fs,
        linear_head0=liner.linear_head0,
        linear_fs=liner.linear_fs,
        residual_true_head0=liner.residual_head0,
        residual_true_fs=liner.residual_fs,
        residual_pred_head0=rhat_h0,
        residual_pred_fs=rhat_fs,
        combined_head0=combined_h0,
        combined_fs=combined_fs,
        label_head0=liner.label_head0,
        label_fs=liner.label_fs,
        full_train_mode=full_train,
    )

    comb_manifest: Dict[str, Any] = {
        "module": "comb",
        "run_id": out_dir.name,
        "liner_run": str(liner_run),
        "lowexp_run": str(lowexp_run),
        "fingerprint": liner.manifest["fingerprint"],
        "csv": liner.manifest.get("csv"),
        "data_dir": liner.manifest.get("data_dir"),
        "ckpt": liner.manifest.get("ckpt"),
        "head0_name": head0,
        "head1_name": head1,
        "feature_order_hash": liner.manifest.get("feature_order_hash"),
        "train_mask_hash": liner.manifest.get("train_mask_hash"),
        "val_mask_hash": liner.manifest.get("val_mask_hash"),
        "seed": int(args.seed),
        "codec": codec,
        "independent_val": not full_train,
        "full_train_mode": full_train,
        "per_sample_errors": per_sample_info["summary"],
    }
    save_manifest(out_dir / "manifest.json", comb_manifest)

    metrics: Dict[str, Any] = {
        "experiment": "comb",
        "head0_name": head0,
        "head1_name": head1,
        "liner_run": str(liner_run),
        "lowexp_run": str(lowexp_run),
        "fingerprint": liner.manifest["fingerprint"],
    }
    metrics.update(
        distillation_metrics(
            teacher=liner.teacher_head0,
            linear=liner.linear_head0,
            residual_true=liner.residual_head0,
            residual_pred=rhat_h0,
            combined=combined_h0,
            label=liner.label_head0,
            train_mask=train_mask,
            val_mask=val_mask,
            head="head0",
        )
    )
    metrics.update(
        distillation_metrics(
            teacher=liner.teacher_fs,
            linear=liner.linear_fs,
            residual_true=liner.residual_fs,
            residual_pred=rhat_fs,
            combined=combined_fs,
            label=liner.label_fs,
            train_mask=train_mask,
            val_mask=val_mask,
            head="fs",
        )
    )
    # linear-only baseline for comparison
    metrics["linear_only_distillation_mae_val_head0"] = float(
        np.mean(np.abs(liner.linear_head0[val_mask] - liner.teacher_head0[val_mask]))
    ) if int(val_mask.sum()) else float("nan")
    metrics["linear_only_distillation_mae_val_fs"] = float(
        np.mean(np.abs(liner.linear_fs[val_mask] - liner.teacher_fs[val_mask]))
    ) if int(val_mask.sum()) else float("nan")
    save_metrics(out_dir / "metrics.json", metrics)

    write_summary_md(
        out_dir / "summary.md",
        "comb",
        [
            f"Combined: L + R for {head0}/{head1}",
            f"liner: {liner_run}",
            f"lowExp: {lowexp_run}",
            f"Val distillation MAE (combined vs teacher): "
            f"{metrics.get('distillation_mae_val_head0')} / {metrics.get('distillation_mae_val_fs')}",
            f"Val linear-only MAE vs teacher: "
            f"{metrics.get('linear_only_distillation_mae_val_head0')} / "
            f"{metrics.get('linear_only_distillation_mae_val_fs')}",
            f"Per-sample errors: per_sample_errors.csv ({per_sample_info['summary']['n_samples']} rows)",
            "",
            "See final_equations_model_space.md, linear_equations.md, residual_equations.md",
        ],
    )

    runs_root = experiment_dir / "runs"
    finalize_run_success(runs_root if out_dir.parent == runs_root else None, out_dir)
    logger.info("Done. Outputs in %s", out_dir)
    return out_dir


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="comb: liner + lowExp joint pipeline")
    add_common_args(p)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--liner-run", type=Path, default=None, help="Reuse existing liner run")
    p.add_argument("--lowexp-run", type=Path, default=None, help="Reuse existing lowExp run")
    p.add_argument(
        "--linear-method",
        type=str,
        default="ridge",
        choices=["ols", "ridge"],
    )
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--allow-empty-val", action="store_true")
    return p


def main(argv: Optional[list] = None) -> None:
    experiment_header("comb")
    args = build_argparser().parse_args(argv)
    if args.lowexp_run is not None and args.liner_run is None:
        # infer liner from lowexp manifest
        man = load_manifest(Path(args.lowexp_run) / "manifest.json")
        if not man.get("liner_run"):
            raise ValueError("--lowexp-run requires liner_run in its manifest or pass --liner-run")
        args.liner_run = Path(man["liner_run"])
    run_comb(args)


if __name__ == "__main__":
    main()
