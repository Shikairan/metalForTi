"""方案 A：单边欠达标目标 f1/f2。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Pareto.ga_objectives import target_shortfall


def test_shortfall_below_target() -> None:
    assert target_shortfall(900.0, 1000.0) == 100.0
    assert abs(target_shortfall(0.10, 0.15) - 0.05) < 1e-9


def test_shortfall_at_or_above_target() -> None:
    assert target_shortfall(1000.0, 1000.0) == 0.0
    assert target_shortfall(1100.0, 1000.0) == 0.0
    assert target_shortfall(0.20, 0.15) == 0.0


if __name__ == "__main__":
    test_shortfall_below_target()
    test_shortfall_at_or_above_target()
    print("test_pareto_objective_shortfall: OK")
