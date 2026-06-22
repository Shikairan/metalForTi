"""
ga_objectives.py — NSGA-II 目标函数（方案 A：单边欠达标）。
"""


def target_shortfall(pred: float, target: float) -> float:
    """单边欠达标惩罚：仅当 pred < target 时返回差值，达标或超额为 0。"""
    return max(0.0, float(target) - float(pred))
