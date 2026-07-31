#!/usr/bin/env python3
"""Patch comb run final_equations.json: floor linear |coef| to min 1e-6."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.codec import codec_stats_dict, physical_docs_notes  # noqa: E402
from common.constants import FEATURE_NAMES  # noqa: E402
from common.equation_format import format_full_equation  # noqa: E402
from common.expr_ir import (  # noqa: E402
    build_linear_sympy_expr,
    combine_linear_residual,
    parse_sympy_expr,
    serialize_expression_ir,
)
from common.linear_fit import (  # noqa: E402
    MIN_LINEAR_COEF_ABS,
    enforce_min_coefficient_magnitude,
    linear_json_with_min_coef,
)


def _head_export_block(
    head: str,
    head_payload: Dict[str, Any],
    *,
    min_abs: float,
) -> Dict[str, Any]:
    feature_names = head_payload.get("feature_names") or list(FEATURE_NAMES)
    lin_json = {
        "intercept": head_payload["linear_intercept"],
        "coefficients": head_payload["linear_coefficients"],
        "feature_names": feature_names,
    }
    lin_export = linear_json_with_min_coef(lin_json, min_abs=min_abs)
    resid_raw = str(head_payload["residual_equation_raw"])
    resid_expr = parse_sympy_expr(resid_raw, feature_names)
    linear_expr = build_linear_sympy_expr(
        lin_export["intercept"], lin_export["coefficients"], feature_names
    )
    combined_expr = combine_linear_residual(linear_expr, resid_expr)
    out = dict(head_payload)
    out["linear_intercept"] = lin_export["intercept"]
    out["linear_coefficients"] = lin_export["coefficients"]
    out["linear_equation_raw"] = lin_export["equation_rhs_raw"]
    out["residual_equation_raw"] = resid_raw
    out["combined_equation_raw"] = str(combined_expr)
    out["combined_equation_display"] = format_full_equation(head, str(combined_expr))
    out["expression_ir"] = serialize_expression_ir(combined_expr)
    out["min_linear_coefficient_abs"] = float(min_abs)
    return out


def patch_final_equations_run(run_dir: Path, *, min_abs: float = MIN_LINEAR_COEF_ABS) -> List[str]:
    run_dir = Path(run_dir).resolve()
    json_path = run_dir / "final_equations.json"
    if not json_path.is_file():
        raise FileNotFoundError(json_path)

    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    heads = payload.get("heads") or {}
    if not heads:
        raise ValueError(f"no heads in {json_path}")

    reports: List[str] = []
    new_heads: Dict[str, Any] = {}
    for head, block in heads.items():
        old_coef = block["linear_coefficients"]
        new_block = _head_export_block(head, block, min_abs=min_abs)
        new_coef = new_block["linear_coefficients"]
        changed = [
            (i, old_coef[i], new_coef[i])
            for i in range(len(old_coef))
            if abs(float(old_coef[i]) - float(new_coef[i])) > 0
        ]
        for i, o, n in changed:
            reports.append(f"{run_dir.name}/{head}: idx={i} ({FEATURE_NAMES[i]}) {o} -> {n}")
        new_heads[head] = new_block

    payload["heads"] = new_heads
    payload["min_linear_coefficient_abs"] = float(min_abs)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    head_names = list(new_heads.keys())
    h0, h1 = head_names[0], head_names[1]

    from common.linear_fit import build_linear_equation_strings

    _, disp_h0 = build_linear_equation_strings(
        new_heads[h0]["linear_intercept"],
        new_heads[h0]["linear_coefficients"],
        FEATURE_NAMES,
    )
    _, disp_h1 = build_linear_equation_strings(
        new_heads[h1]["linear_intercept"],
        new_heads[h1]["linear_coefficients"],
        FEATURE_NAMES,
    )
    ms_lines = [
        "# Final equations (model space)",
        "",
        "可直接作用于 CSV 前 30 列（模型空间）。",
        "",
        f"## {h0}",
        "",
        "```text",
        new_heads[h0]["combined_equation_display"],
        "```",
        "",
        "### Linear",
        "",
        "```text",
        f"L = {disp_h0}",
        "```",
        "",
        "### Residual",
        "",
        "```text",
        f"R = {new_heads[h0]['residual_equation_raw']}",
        "```",
        "",
        f"## {h1}",
        "",
        "```text",
        new_heads[h1]["combined_equation_display"],
        "```",
        "",
        "### Linear",
        "",
        "```text",
        f"L = {disp_h1}",
        "```",
        "",
        "### Residual",
        "",
        "```text",
        f"R = {new_heads[h1]['residual_equation_raw']}",
        "```",
        "",
    ]
    (run_dir / "final_equations_model_space.md").write_text("\n".join(ms_lines), encoding="utf-8")

    codec = payload.get("codec") or codec_stats_dict()
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
            new_heads[h0]["combined_equation_display"],
            new_heads[h1]["combined_equation_display"],
            "```",
            "",
            "coldway 编码不完全可逆：不生成虚假唯一逆解。",
            "",
        ]
    )
    (run_dir / "final_equations_physical.md").write_text("\n".join(phys_lines), encoding="utf-8")

    for head, block in new_heads.items():
        coef = enforce_min_coefficient_magnitude(block["linear_coefficients"], min_abs=min_abs)
        if float(np.min(np.abs(coef))) + 1e-15 < min_abs:
            raise AssertionError(f"{head}: coefficient below min_abs after patch")

    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        default=[
            Path(__file__).parent / "runs" / "full_ysfs_it400_ms40",
            Path(__file__).parent / "runs" / "full_utsFsAll_it400_ms40",
        ],
        help="comb run directories containing final_equations.json",
    )
    parser.add_argument("--min-abs", type=float, default=MIN_LINEAR_COEF_ABS)
    args = parser.parse_args()

    all_reports: List[str] = []
    for run_dir in args.run_dirs:
        reports = patch_final_equations_run(run_dir, min_abs=args.min_abs)
        all_reports.extend(reports)
        print(f"Patched {run_dir} ({len(reports)} coefficient updates)")

    if all_reports:
        print("\nChanges:")
        for line in all_reports:
            print(f"  {line}")
    else:
        print("No coefficient changes needed (all already >= min_abs).")


if __name__ == "__main__":
    main()
