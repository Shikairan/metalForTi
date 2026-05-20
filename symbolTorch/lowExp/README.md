# lowExp — 最强可解释（纯表格，推理无图）

## 思路

1. 教师 `RGAT_Dual` 在全图上推理，得到每个节点的 `YS` / `FS` 软标签；
2. 用 SymTorch 对 **仅节点特征 `x`（30 维）** 拟合符号式 `x → YS`、`x → FS`；
3. 推理时 **不使用** `edge_index` / `edge_type`，即牺牲图结构先验。

适合作为「可写进论文的闭式近似」，但不能等价复现 GNN 的邻域效应。

## 运行

```bash
cd metalForTi/symbolTorch/lowExp
/root/miniconda3/bin/python3.13 run_distill.py --quick
```

`lowExp` 默认 PySR 运算符含 `log`，便于拟合 log 空间的 FS 目标。

## 输出（`runs/`）

| 文件 | 含义 |
|------|------|
| `ys_tabular_sym.json` / `fs_tabular_sym.json` | 全局表格公式 |
| `teacher_predictions.pt` | 教师在全图上的预测缓存 |
| `metrics.json` | 含 `graph_info_loss_*`：tabular MAE − teacher MAE |

`graph_info_loss` 为正表示去掉图后误差变大。

## 示例指标解读

`summary.md` 中若出现负的 graph info loss，表示在该次快速调试运行中，表格符号模型在 val 上略优于教师（多为过拟合或 `--quick` 样本过少，完整运行请以 `metrics.json` 为准）。
