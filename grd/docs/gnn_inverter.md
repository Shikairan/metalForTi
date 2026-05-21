# `gnn_inverter.py` — 核心梯度反推引擎

## 文件路径

`grd/gnn_inverter.py`

## 作用

在 **GNN 参数冻结** 的前提下，把反推表述为无约束/约束优化问题，提供：

- 配置类 `GNNInverterConfig`
- 正则项策略（图平滑、L1、锚定、软物理惩罚）
- 硬投影器（非负、box、单纯形及组合）
- 多初始点策略
- 主类 `GNNInverter`：`invert_single` / `invert_multistart`
- 可选 `GNNInversionBenchmark` 多场景对比

## 优化问题形式

记可学习变量为全图输入 $X \in \mathbb{R}^{N \times d}$，冻结的 GNN 为 $f_\theta$，目标输出为 $y^{*}, z^{*}$（YS/FS）。



$$
\min_{X}\; \mathcal{L}_{\text{recon}}(X)
+ \lambda_s \mathcal{R}_{\text{smooth}}(X)
+ \lambda_a \mathcal{R}_{\text{anchor}}(X)
+ \cdots
$$





$$
\mathcal{L}_{\text{recon}}
= \frac{1}{|\mathcal{M}|}\sum_{i\in\mathcal{M}}
\bigl(\|f_y(X)_i - y^{*}_i\|^2 + \|f_z(X)_i - z^{*}_i\|^2\bigr)
$$



$\mathcal{M}$ 为节点掩码（`recon_mask`）；默认全图。

每次迭代：对 $X$ 反向传播 $\nabla_X \mathcal{L}$，再 $X \leftarrow \Pi_{\mathcal{C}}(X)$。

## 主要类

### `GNNInverterConfig`

超参：优化器（`adam`/`lbfgs`）、学习率、迭代与早停、正则权重、`projectors` 名称列表、设备等。

### 正则器

| 类 | $\mathcal{R}$ 形式 | 默认在 run_inversion |
|----|---------------------|----------------------|
| `SmoothnessRegularizer` | $\sum_{(i,j)\in E_k} \|X_i - X_j\|^2$ | 是，`target_edge_types=[0]` 仅 comp_sim |
| `SparsityRegularizer` | $\|X\|_1$ | 权重极小 1e-5 |
| `AnchorRegularizer` | $\|X - X_{\text{anchor}}\|^2$ | 是 |
| `PhysicalPenaltyRegularizer` | 负值惩罚 + 行和偏离 | 权重 0（关闭） |

`SmoothnessRegularizer` 传入 `edge_type`，避免 env/heat 边误平滑组分。

### 投影器

| 类 | 说明 |
|----|------|
| `SimplexProjector` | Duchi 算法，行投影到 $\Delta^{d-1}$ |
| `BoxProjector` | 分量 clamp |
| `CompositeProjector` | 按名称串联 |
| （推荐在 grd 中用 `MaskedCompositeProjector`） | 见 `masked_projector.py` |

### 初始化器

`ZeroInitializer`、`RandomNormalInitializer`、`TrainingMeanInitializer`、`DirichletInitializer` — 用于 `invert_multistart` 逃离局部极小。

### `GNNInverter`

| 方法 | 作用 |
|------|------|
| `invert_single` | 单初始点优化；支持 `recon_mask` |
| `invert_multistart` | 多初始点，取最小 `final_recon_mse` |
| `_compute_loss` | 前向 + 重建 + 正则；正则接收 `edge_type` |
| `_infer_input_dim` | 从 `nn.Linear` 推断 $d$（建议显式 `input_dim`） |

#### Adam 路径

`loss.backward()` → `clip_grad_norm_`（可选）→ `optimizer.step()` → 按间隔 `projector.project`。

#### LBFGS 路径

步前投影 → `closure` 内 `backward`（**不**在 closure 内 clip，避免破坏曲率）→ `step` → 步后投影 → 重算 loss 对齐 `best_x`。

### `GNNInversionBenchmark`

批量对比不同 `Scenario`（材料/温度/探索配置），输出 `pandas.DataFrame` 与可选 PNG。

## 与正向 GNN 的关系

正向模型需满足接口：

```python
ys, fs = model(x, edge_index, edge_type)
```

本仓库默认 **DualRGAT**（共享编码思想 + 双 RGAT 头），关系卷积与 **R-GCN / RGAT** 同类。

## 相关文档

- [masked_projector.md](./masked_projector.md) — $\Pi_{\mathcal{C}}$ 实现
- [run_inversion.md](./run_inversion.md) — 默认正则与 recon_mask 接线
- [../README.md#算法原理](../README.md#算法原理) — 全文公式与论文链接
