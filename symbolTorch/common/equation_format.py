"""Human-readable equation formatting for lowExp exports."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .constants import COLDWAY_FLAT_COLS, ELEMENT_FEATURE_NAMES, FEATURE_NAMES, TESTENV_COLS

# 展示用中文说明（公式内仍用符号名，避免破坏可解析性）
FEATURE_GLOSSARY_CN: Dict[str, str] = {
    **{n: f"元素 {n.replace('el_', '')} 含量 wt%" for n in ELEMENT_FEATURE_NAMES},
    "tem": "试验温度（模型空间 / z-score，对应 data 中 tem）",
    "fcr": "试验环境第二维（模型空间 / z-score；图特征名 fcr，与 data1123 的 sr 同源管线）",
    **{
        f"coldway_{i}": f"工艺 coldway 第 {i} 维（18 维展平，模型空间）"
        for i in range(18)
    },
}


def _round_floats_in_expr(expr: str, ndigits: int = 8) -> str:
    """把表达式里的浮点数字面量收成固定位数，便于阅读。"""

    def _repl(m: re.Match[str]) -> str:
        s = m.group(0)
        try:
            v = float(s)
        except ValueError:
            return s
        if abs(v - round(v)) < 1e-12:
            return str(int(round(v)))
        formatted = f"{v:.{ndigits}g}"
        return formatted

    return re.sub(r"(?<![A-Za-z_])[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", _repl, expr)


def _prettify_ops(expr: str) -> str:
    s = expr.strip()
    s = s.replace("**", "^")
    # 在 + - * / = 两侧补空格（避免破坏科学计数法与一元负号的粗糙处理）
    s = re.sub(r"\s*\+\s*", " + ", s)
    s = re.sub(r"\s*\*\s*", " * ", s)
    s = re.sub(r"\s*/\s*", " / ", s)
    s = re.sub(r"(?<=\w)\s*-\s*(?=\w)", " - ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def variables_used_in_expr(expr: str, names: Sequence[str] = FEATURE_NAMES) -> List[str]:
    used: List[str] = []
    for name in sorted(names, key=len, reverse=True):
        # 词边界：避免 Al 匹配到占位
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", expr):
            used.append(name)
    return used


def format_full_equation(
    target: str,
    raw_expr: str,
    *,
    ndigits: int = 8,
) -> str:
    """生成完整单行方程：TARGET = <expr>。"""
    body = _prettify_ops(_round_floats_in_expr(str(raw_expr).strip(), ndigits=ndigits))
    return f"{target} = {body}"


def build_equation_record(
    target: str,
    raw_expr: str,
    *,
    block_name: str,
) -> Dict[str, Any]:
    full = format_full_equation(target, raw_expr)
    used = variables_used_in_expr(raw_expr)
    glossary = {k: FEATURE_GLOSSARY_CN.get(k, k) for k in used}
    return {
        "block_name": block_name,
        "target": target,
        "equation": full,
        "equation_rhs": _prettify_ops(_round_floats_in_expr(str(raw_expr).strip())),
        "equation_raw": str(raw_expr).strip(),
        "variables_used": used,
        "variable_glossary": glossary,
    }


def write_equations_markdown(
    path: Path,
    records: Sequence[Dict[str, Any]],
    *,
    title: str = "lowExp 符号方程",
    notes: Optional[Sequence[str]] = None,
) -> None:
    """写出人类可读的完整方程 Markdown。"""
    lines: List[str] = [f"# {title}", ""]
    if notes:
        lines.append("## 说明")
        lines.append("")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")

    for rec in records:
        target = rec["target"]
        lines.append(f"## {target}")
        lines.append("")
        lines.append("完整方程：")
        lines.append("")
        lines.append("```text")
        lines.append(rec["equation"])
        lines.append("```")
        lines.append("")
        used = rec.get("variables_used") or []
        gloss = rec.get("variable_glossary") or {}
        if used:
            lines.append("出现的变量：")
            lines.append("")
            lines.append("| 符号 | 含义 |")
            lines.append("|------|------|")
            for name in used:
                lines.append(f"| `{name}` | {gloss.get(name, FEATURE_GLOSSARY_CN.get(name, ''))} |")
            lines.append("")
        lines.append(f"- block: `{rec.get('block_name', '')}`")
        lines.append(f"- 原始 RHS: `{rec.get('equation_raw', '')}`")
        lines.append("")

    lines.append("## 全部 30 维特征名（参考）")
    lines.append("")
    lines.append("| 下标 | 符号 | 含义 |")
    lines.append("|------|------|------|")
    for i, name in enumerate(FEATURE_NAMES):
        lines.append(f"| {i} | `{name}` | {FEATURE_GLOSSARY_CN.get(name, '')} |")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def enrich_and_write_equation_json(
    path: Path,
    *,
    target: str,
    block_name: str,
    raw_expr: str,
) -> Dict[str, Any]:
    """写入可读增强后的单目标 JSON。"""
    rec = build_equation_record(target, raw_expr, block_name=block_name)
    payload = {
        "block_name": block_name,
        "target": target,
        "slime": False,
        "equation": rec["equation"],
        "equation_rhs": rec["equation_rhs"],
        "equation_raw": rec["equation_raw"],
        "variables_used": rec["variables_used"],
        "variable_glossary": rec["variable_glossary"],
        # 兼容旧字段
        "equations": {"0": rec["equation_raw"]},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload
