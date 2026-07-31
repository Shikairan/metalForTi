# lowExp 概述

## 1. 目标

`lowExp` 拟合 **liner 产生的教师残差**（不是完整教师输出、也不是真实标签）：

```text
R(x) ≈ teacher(x, graph) - L_ridge(x)
```

再用 **SymTorch + PySR** 得到无图残差公式。最终预测为 `L(x) + R(x)`，由 `comb` 组合。

## 2. 流水线

```text
liner/runs/<id>/residual_targets.pt
        │
        ▼
  绑定 X_train 与 residual_train（make_bound_target_fn，SLIME=False）
        │
        ▼
  PySR 蒸馏 residual_head0 / residual_fs
        │
        ▼
  *_residual_sym.json/.pt
  residual_equations.md
  residual_predictions.pt
  metrics.json / summary.md
```

必须提供 `--liner-run`。默认只用 `train_mask` 拟合；`val_mask` 仅评估。

## 3. 联合入口

一键流水线见 `comb/run_comb.py`。线性阶段见 `liner/run_liner.py`。
