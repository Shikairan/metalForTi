"""coldway 人类可读展示（纯 NumPy，无 torch 依赖）。"""
from __future__ import annotations

import numpy as np

_COLDWAY_STAGES = 3
_COLDWAY_METHODS = 3
_COLDWAY_PARAM_NAMES = ("T", "t")


def fmt_coldway_val(v: float, nd: int = 4) -> str:
    """物理 T/t：0 显式打印，固定小数位。"""
    if abs(v) < 1e-12:
        return f"{0.0:.{nd}f}"
    return f"{float(v):.{nd}f}"


def _row_active(row62: np.ndarray, eps: float = 1e-8) -> bool:
    return bool(np.abs(row62).sum() > eps)


def format_coldway_lines(mat: np.ndarray) -> list[str]:
    """由 (3,3,2) 物理 coldway 矩阵生成日志行；各阶段参数名统一为 T、t。"""
    mat = np.asarray(mat, dtype=np.float32).reshape(_COLDWAY_STAGES, 3, 2)
    lines: list[str] = []

    active_flags = [_row_active(mat[s]) for s in range(_COLDWAY_STAGES)]
    if any(active_flags):
        last_raw = max(i for i, a in enumerate(active_flags) if a)
        for i in range(last_raw):
            if not active_flags[i]:
                lines.append(
                    f"    [警告] 阶段{i + 1} 为空但后续阶段有值（非法行累计，应经 compile 修复）"
                )
                break

    last_active = -1
    p0, p1 = _COLDWAY_PARAM_NAMES
    for s in range(_COLDWAY_STAGES):
        row = mat[s]
        if not _row_active(row):
            continue
        last_active = s
        norms = [float(np.linalg.norm(row[m])) for m in range(_COLDWAY_METHODS)]
        m_best = int(max(range(_COLDWAY_METHODS), key=lambda m: norms[m]))
        a, b = float(row[m_best, 0]), float(row[m_best, 1])
        lines.append(
            f"    阶段{s + 1}  方式{m_best + 1}  "
            f"{p0}={fmt_coldway_val(a):>10}  {p1}={fmt_coldway_val(b):>10}"
        )
    if last_active < 0:
        lines.append("    (无冷加工)")
    elif last_active < _COLDWAY_STAGES - 1:
        s0 = last_active + 2
        s1 = _COLDWAY_STAGES
        if s0 == s1:
            lines.append(f"    阶段{s0}: 未启用")
        else:
            lines.append(f"    阶段{s0}~{s1}: 未启用")
    return lines
