# `masked_projector.py` — 按维段硬约束投影

## 文件路径

`grd/masked_projector.py`

## 作用

实现 **投影梯度法（Projected Gradient / Projected Adam）** 中的投影算子 \(\Pi_\mathcal{C}(\cdot)\)：  
在每次梯度更新后，将 \(x\) 的不同列段投影到各自可行集 \(\mathcal{C}_k\)。

## 核心类与函数

### `FeatureSliceSpec`

描述 `x[:, start:end]` 的投影类型：

| `kind` | 可行集 | 用途 |
|--------|--------|------|
| `ti_balance` | 非负，行和 ≤ `total_wt` | 钛合金 element 段（默认） |
| `box` | 轴对齐区间 | testenv、coldway |
| `simplex` | 非负，行和 = 1 | 概率组分（本仓库默认不用） |
| `scaled_simplex` | 单纯形 × 行缩放 | 按 anchor 总量缩放 |
| `nonnegative` | \(x \ge 0\) | 仅非负 |
| `none` | 不投影 | 跳过 |

### `_project_ti_balance_rows(block, total_wt)`

1. `clamp(min=0)`
2. 若 `sum(row) > total_wt`，整行等比缩放：`x <- x * (total_wt / sum)`

保证 **合金化元素之和不超过 100 wt%**，Ti 余量由 `compute_ti_balance` 补足到 100。

### `_project_simplex_rows` / `_project_scaled_simplex_rows`

调用 `gnn_inverter.SimplexProjector`（Duchi 排序算法），再按需乘以 `target_sums`。

### `MaskedCompositeProjector.project(x)`

按 `slices` 顺序逐段写回 `x`，不改变其他列段。

## 优化中的位置

```text
x ← x - η ∇L(x)          # Adam / LBFGS
x ← Π_C(x)               # 本模块
```

- **Adam**：通常每 `projection_interval` 步投影一次  
- **LBFGS**：步前 + 步后投影（见 `gnn_inverter.invert_single`）

## 算法与参考文献

| 主题 | 说明 | 参考 |
|------|------|------|
| 欧氏单纯形投影 | \(O(d \log d)\) 排序法 | Duchi et al., JMLR 2008（见总 README） |
| 投影梯度法 | 非光滑约束优化的标准框架 | Bertsekas, *Nonlinear Programming* |
| Box 约束 | 逐维 `clamp` | 盒约束投影为分量级闭式解 |

## 相关文档

- [feature_layout.md](./feature_layout.md) — 默认三段配置
- [gnn_inverter.md](./gnn_inverter.md) — 何时调用投影器
