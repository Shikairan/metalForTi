# grd：GNN 全特征梯度反推

对 `gnnDir/gnn/r-gatDouble` 训练的 `SingleEncoder_DualRGAT` 反演输入 `x`（30 维），默认用 **真实 ys/fs** 做还原验证。

## 特征与约束

| 维段 | 索引 | 约束 |
|------|------|------|
| element | 0–9 | **A 模式**：非负，`sum ≤ 100` wt%；`Ti = 100 − sum`（Ti 不在 x 内，见输出 `ti_balance_*`） |
| testenv | 10–11 | 训练集 **1%/99% 分位数** box（z-score，与训练一致） |
| coldway | 12–29 | 训练集分位数 box |

元素列顺序：`Al, Zr, Sn, Mo, Cr, Nb, Si, V, Ta, Fe`。

## 运行

```bash
cd /path/to/repo
python -m grd.run_inversion \
  --data-dir gnnDir/gnndataPT/r-gatPT \
  --ckpt gnnDir/gnn/r-gatDouble/runs/best_ysfs_gat.pt \
  --rgat-dir gnnDir/gnn/r-gatDouble \
  --out-dir grd/outputs
```

- **GPU**：默认 `--device cuda`（不可用则自动 CPU）
- **CPU 测试**：`--force-cpu`（默认仅 `training_mean` 初始化，降低内存）
- 可选物理温度界：`--tem-lower` / `--tem-upper`（需 `testenv_stats.csv` 转 z 空间）

## 输出

- `grd/outputs/x_inv.pt`：`x_inv`（与 `material_graph` 同格式）、`ti_balance_inv` / `ti_balance_true`
- `grd/outputs/inversion_summary.json`：汇总指标 + `field_descriptions` 字段中文说明
- `grd/outputs/inversion_summary.txt`：可读 TXT 报告，每项带中文解释

## 依赖

`torch`、`torch-geometric`、`pandas`、`numpy`（见 `gnnDir/requirements.txt`）。

全图约 27 万条边，**强烈建议 GPU**；CPU 上完整 multistart 可能 OOM。
