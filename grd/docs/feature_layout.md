# `feature_layout.py` — 特征布局与约束配置

## 文件路径

`grd/feature_layout.py`

## 作用

定义与 **datagnn / material_graph** 一致的 30 维布局，并负责：

1. 计算 **Ti 余量**（A 模式，不在 `x` 向量内）
2. 从训练集估计 **testenv / coldway** 的 box 界
3. 可选：物理 tem/fcr → z-score 界
4. 组装默认 **`MaskedCompositeProjector`**

## 常量

| 常量 | 值 | 含义 |
|------|-----|------|
| `ELEMENT_DIM` | 10 | Al…Fe |
| `TESTENV_DIM` | 2 | tem, fcr（z-score） |
| `COLDWAY_DIM` | 18 | 工艺 |
| `INPUT_DIM` | 30 | 总输入维 |
| `DEFAULT_TOTAL_WT` | 100.0 | Ti A 模式总量 wt% |

切片：`ELEMENT_SLICE`、`TESTENV_SLICE`、`COLDWAY_SLICE`。

## 函数说明

### `compute_ti_balance(x, total_wt=100)`

```text
Ti(wt%) = total_wt - sum(x[:, 0:10])
```

仅用于**后处理与报告**；优化中 Ti 由「元素行和 ≤ total_wt」间接约束。

### `load_testenv_stats(stats_path)`

读 `testenv_stats.csv`，得到 `tem`/`fcr` 的 `mean`、`std`，用于 z-score 反变换或物理界转换。

### `physical_testenv_to_z(physical, mean, std)`

```text
z = (x_phys - mean) / std
```

与 `build_datagnn.py` 标准化一致。

### `bounds_from_train_x(x, train_mask, margin=0.05, q_lo=0.01, q_hi=0.99)`

在**训练节点**上对 testenv、coldway 各维取分位数，再按区间宽度外扩 `margin`（默认 5%）。  
这是 **box 约束**，不是高斯假设，属于非参数可行域估计。

### `bounds_with_physical_testenv(...)`

在分位数界基础上，若用户提供 `tem_lower/upper` 等物理值，则覆盖对应维的 z 空间上下界；coldway 仍用分位数。

### `build_default_slice_specs(bounds, total_wt)`

生成三段 `FeatureSliceSpec`：

| 段 | kind | 说明 |
|----|------|------|
| element | `ti_balance` | 非负 + 行和 ≤ total_wt |
| testenv | `box` | 逐维 clamp |
| coldway | `box` | 逐维 clamp |

### `build_projector(anchor_x, bounds, total_wt)`

返回配置好的 `MaskedCompositeProjector`。

## 算法要点

- **A 模式组分**：不将 Ti 作为优化变量，避免维数冗余；可行域为「单纯形缩放」的变体（见 [masked_projector.md](./masked_projector.md)）。
- **分位数 box**：等价于用训练分布支撑集近似约束，防止反推跑到训练未见区域。

## 相关文档

- [masked_projector.md](./masked_projector.md) — `ti_balance` 投影实现
- [../README.md#算法原理](../README.md#算法原理) — 数学形式与论文
