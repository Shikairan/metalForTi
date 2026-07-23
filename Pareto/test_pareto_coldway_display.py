"""Pareto 日志 coldway 展示：统一 T/t 符号与物理量格式。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Pareto.coldway_display import format_coldway_lines
from preprocess.preprocess_datagnn_repro import inverse_coldway_flat_18


def test_coldway_unified_t_t_labels() -> None:
    """冷却方式显示为 水冷/空冷/炉冷；参数统一为 T、t，并还原为物理量。"""
    cold = np.zeros(18, dtype=np.float32)
    cold[0], cold[1] = 0.0, float(np.log(0.25))
    cold[10], cold[11] = -0.3567, 2.7726

    phys = inverse_coldway_flat_18(cold)
    lines = format_coldway_lines(phys.reshape(3, 3, 2))
    text = "\n".join(lines)
    assert "C_a" not in text and "C_b" not in text
    assert "方式1" not in text and "方式2" not in text and "方式3" not in text
    assert "T=" in text and "t=" in text
    assert "阶段1  水冷" in text
    assert "阶段2  炉冷" in text
    assert "阶段3: 未启用" in text
    assert "800.0000" in text
    assert "0.2500" in text


if __name__ == "__main__":
    test_coldway_unified_t_t_labels()
    print("test_pareto_coldway_display: OK")
