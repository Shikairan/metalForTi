"""拓展育种池与移民子代拆分。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Pareto.ga_breeding_plan import split_offspring_counts


def test_split_pop_604() -> None:
    t, r, i = split_offspring_counts(604)
    assert t + r + i == 604
    assert i == 60
    assert r == 109
    assert t == 435


def test_split_pop_10() -> None:
    t, r, i = split_offspring_counts(10)
    assert t + r + i == 10


if __name__ == "__main__":
    test_split_pop_604()
    test_split_pop_10()
    print("test_pareto_breeding: OK")
