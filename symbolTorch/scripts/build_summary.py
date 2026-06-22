#!/usr/bin/env python3
"""汇总四档 symbolTorch 实验结果 → SYMBOLTORCH_SUMMARY.{json,md}"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
EXPS = ("highExp", "medExp", "lowExp", "sampleExp")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _pick_equations(sym_json: Optional[Dict[str, Any]], max_n: int = 3) -> List[str]:
    if not sym_json:
        return []
    eqs = sym_json.get("equations", {})
    out: List[str] = []
    for k in sorted(eqs.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
        v = eqs[k]
        if v is None or str(v).strip() in ("", "None"):
            continue
        out.append(f"[{k}] {v}")
        if len(out) >= max_n:
            break
    return out


def _collect_exp(name: str) -> Dict[str, Any]:
    runs = ROOT / name / "runs"
    block: Dict[str, Any] = {
        "experiment": name,
        "runs_dir": str(runs.resolve()),
        "metrics": _read_json(runs / "metrics.json"),
        "summary_md": _read_text(runs / "summary.md"),
        "artifacts": [],
        "equations_preview": {},
    }

    patterns = {
        "highExp": ["ys_encoder_sym.json", "fs_encoder_sym.json"],
        "medExp": ["ys_encoder_sym.json", "fs_encoder_sym.json", "ys_head_sym.json", "fs_head_sym.json"],
        "lowExp": ["ys_tabular_sym.json", "fs_tabular_sym.json"],
        "sampleExp": [],
    }
    for fname in patterns.get(name, []):
        p = runs / fname
        if p.is_file():
            block["artifacts"].append(str(p))
            block["equations_preview"][fname] = _pick_equations(_read_json(p))

    if name == "sampleExp":
        nodes = sorted(runs.glob("node_*/ys_formula.json"))
        block["artifacts"].extend(str(p) for p in nodes[:5])
        for p in nodes[:3]:
            ni = p.parent.name
            ys = _read_json(p)
            fs = _read_json(p.parent / "fs_formula.json")
            block["equations_preview"][ni] = {
                "ys": _pick_equations(ys, max_n=1),
                "fs": _pick_equations(fs, max_n=1),
            }

    block["ok"] = block["metrics"] is not None
    return block


def _format_metrics_table(metrics: Optional[Dict[str, Any]]) -> List[str]:
    if not metrics:
        return ["  （无 metrics.json）"]
    lines: List[str] = []
    for key in ("teacher", "hybrid", "tabular_symbolic"):
        if key not in metrics:
            continue
        m = metrics[key]
        lines.append(
            f"  {key}: val_mae_ys={m.get('val_mae_ys', '—'):.4f}  "
            f"val_mae_fs={m.get('val_mae_fs', '—'):.4f}"
            if isinstance(m.get("val_mae_ys"), (int, float))
            else f"  {key}: {m}"
        )
    if "nodes" in metrics:
        lines.append(f"  解释节点: {metrics['nodes']}")
    return lines or ["  （metrics 无标准字段）"]


def build_summary(ckpt: str, quick: bool) -> Dict[str, Any]:
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from common.constants import DEFAULT_CKPT, DEFAULT_DATA_DIR

    experiments = [_collect_exp(n) for n in EXPS]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_ckpt": ckpt or str(DEFAULT_CKPT.resolve()),
        "data_dir": str(DEFAULT_DATA_DIR.resolve()),
        "quick_mode": quick,
        "experiments": experiments,
        "all_ok": all(e["ok"] for e in experiments),
    }


def write_markdown(summary: Dict[str, Any], path: Path) -> None:
    lines = [
        "# symbolTorch 实验汇总",
        "",
        f"生成时间（UTC）: {summary['generated_at_utc']}",
        f"教师模型: `{summary['teacher_ckpt']}`",
        f"数据目录: `{summary['data_dir']}`",
        f"快速模式 (--quick): {summary['quick_mode']}",
        f"全部完成: {'是' if summary['all_ok'] else '否'}",
        "",
    ]
    for exp in summary["experiments"]:
        name = exp["experiment"]
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"状态: {'完成' if exp['ok'] else '未完成或失败'}")
        lines.append(f"输出目录: `{exp['runs_dir']}`")
        lines.append("")
        if exp["summary_md"]:
            lines.append("### 实验摘要")
            lines.append("")
            lines.append(exp["summary_md"])
            lines.append("")
        lines.append("### 指标")
        lines.append("")
        lines.extend(_format_metrics_table(exp.get("metrics")))
        lines.append("")
        if exp.get("equations_preview"):
            lines.append("### 公式预览（节选）")
            lines.append("")
            for fname, eqs in exp["equations_preview"].items():
                lines.append(f"**{fname}**")
                if isinstance(eqs, dict):
                    for branch, arr in eqs.items():
                        lines.append(f"- {branch}: {arr[0] if arr else '—'}")
                elif eqs:
                    for e in eqs:
                        lines.append(f"- `{e}`")
                else:
                    lines.append("- （无有效公式）")
                lines.append("")
        lines.append("---")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--quick", action="store_true", default=True)
    args = p.parse_args()

    summary = build_summary(args.ckpt, args.quick)
    out_json = ROOT / "SYMBOLTORCH_SUMMARY.json"
    out_md = ROOT / "SYMBOLTORCH_SUMMARY.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, out_md)
    print(f"[OK] {out_json}")
    print(f"[OK] {out_md}")
    print(f"all_ok={summary['all_ok']}")


if __name__ == "__main__":
    main()
