"""
validate_genome.py — 30 维基因组合法性检查（不调用 GNN）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from grd.feature_layout import (
    COLDWAY_SLICE,
    DEFAULT_TOTAL_WT,
    ELEMENT_SLICE,
    FeatureBounds,
    TESTENV_SLICE,
    compute_ti_balance,
)

_COLDWAY_STAGES = 3
_COLDWAY_METHODS = 3
_COLDWAY_PARAM_NAMES = (
    ("T", "t"),
    ("C_a", "C_b"),
    ("C_a", "C_b"),
)
_EPS = 1e-6


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]

    def summary(self) -> str:
        if self.ok:
            return "合法"
        return "; ".join(self.errors)


def _row_active(row6: torch.Tensor) -> bool:
    return bool(row6.abs().sum().item() > _EPS)


def validate_genome(
    genome: torch.Tensor,
    bounds: FeatureBounds,
    *,
    total_wt: float = DEFAULT_TOTAL_WT,
) -> ValidationResult:
    """检查单条 30 维基因组是否满足 element / testenv / coldway 约束。"""
    errors: List[str] = []
    g = genome.detach().cpu().float()

    elem = g[ELEMENT_SLICE]
    if (elem < -_EPS).any():
        errors.append("element 存在负值")
    elem_sum = float(elem.sum().item())
    if elem_sum > float(total_wt) + 1e-3:
        errors.append(f"10 元合计 {elem_sum:.4f} 超过 {total_wt} wt%")

    ti = float(compute_ti_balance(g.unsqueeze(0), total_wt)[0].item())
    if ti < -1e-3:
        errors.append(f"Ti 余量 {ti:.4f} 为负")

    te = g[TESTENV_SLICE]
    lo_te = bounds.testenv_lower.float()
    hi_te = bounds.testenv_upper.float()
    if (te < lo_te - 1e-5).any() or (te > hi_te + 1e-5).any():
        errors.append("testenv 超出训练 box 界")

    cw = g[COLDWAY_SLICE].reshape(_COLDWAY_STAGES, 6)
    lo_cw = bounds.coldway_lower.float()
    hi_cw = bounds.coldway_upper.float()
    for s in range(_COLDWAY_STAGES):
        sl = slice(s * 6, (s + 1) * 6)
        row = cw[s]
        row_lo = lo_cw[sl]
        row_hi = hi_cw[sl]
        for m in range(_COLDWAY_METHODS):
            for p in range(2):
                j = m * 2 + p
                v = float(row[j].item())
                lo_v = float(row_lo[j].item())
                hi_v = float(row_hi[j].item())
                if v < lo_v - 1e-5 or v > hi_v + 1e-5:
                    pname = _COLDWAY_PARAM_NAMES[m][p]
                    errors.append(
                        f"coldway 阶段{s + 1} 方式{m + 1} {pname}={v:.4f} "
                        f"超出 [{lo_v:.4f}, {hi_v:.4f}]"
                    )

    active = [_row_active(cw[s]) for s in range(_COLDWAY_STAGES)]
    if not active[0]:
        errors.append("coldway 阶段 1 必须非空")

    if any(active):
        last_raw = max(i for i, a in enumerate(active) if a)
        for i in range(last_raw):
            if not active[i]:
                errors.append(f"coldway 行累计违规：阶段{i + 1} 空但阶段{last_raw + 1} 有值")
                break

    for s in range(_COLDWAY_STAGES):
        if not active[s]:
            continue
        m = cw[s].reshape(_COLDWAY_METHODS, 2)
        norms = [float(m[k].pow(2).sum().item()) for k in range(_COLDWAY_METHODS)]
        n_active = sum(1 for n in norms if n > _EPS * _EPS)
        if n_active == 0:
            errors.append(f"coldway 阶段{s + 1} 标记为激活但三方式均为零")
        elif n_active > 1:
            errors.append(f"coldway 阶段{s + 1} 违反 3 选 1（{n_active} 种方式同时非零）")

    return ValidationResult(ok=len(errors) == 0, errors=errors)


def validate_batch(
    genomes: List[torch.Tensor],
    bounds: FeatureBounds,
) -> tuple[int, int, List[int]]:
    """返回 (合法数, 非法数, 非法索引列表)。"""
    bad_idx: List[int] = []
    for i, g in enumerate(genomes):
        if not validate_genome(g, bounds).ok:
            bad_idx.append(i)
    n_bad = len(bad_idx)
    return len(genomes) - n_bad, n_bad, bad_idx


def _self_test() -> None:
    from grd.feature_layout import bounds_from_train_x, build_projector
    from Pareto.ga_compile import compile_genome

    x = torch.rand(40, 30).abs()
    mask = torch.ones(40, dtype=torch.bool)
    bounds = bounds_from_train_x(x, mask)
    proj = build_projector(x, bounds)
    g = compile_genome(x[0], bounds, proj, x)
    assert validate_genome(g, bounds).ok
    g_bad = g.clone()
    g_bad[COLDWAY_SLICE].reshape(3, 6)[2, 0] = 1.0
    g_bad[COLDWAY_SLICE].reshape(3, 6)[1] = 0.0
    assert not validate_genome(g_bad, bounds).ok
    print("[OK] validate_genome self-test passed")


if __name__ == "__main__":
    _self_test()
