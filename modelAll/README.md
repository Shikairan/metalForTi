# modelAll — 全量 RGAT 训练

在 **604 条全量样本**上训练真实 RGAT 双头模型（YS + FS）。数据来自 `gnnDir/datagnn.csv`，划分 **train:val = 1:0.2**（约 503 / 101），不做 mask loop 剔除。

## 目录

```
modelAll/
├── model_rgat.py      # RGAT 模型（与 r-gatDouble 同结构）
├── build_data.py        # 从 gnnDir 构建 PT 数据包
├── train.py             # 训练脚本
├── data/                # 图 + 标签 + mask（build_data 生成）
└── runs/                # best_rgat_full.pt、train_log.csv
```

## 用法

在 **metalForTi 根目录**：

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# 1. 构建数据（默认读 gnnDir/datagnn.csv）
python modelAll/build_data.py --sanity

# 2. 训练（默认 1000 epoch，按 val MAE 选最优）
python modelAll/train.py

# 快速冒烟
python modelAll/train.py --epochs 5 --log-interval 1
```

## 与 r-gatDouble 的区别

| 项 | r-gatDouble | modelAll |
|----|-------------|----------|
| 数据 | gnndataPT/r-gatPT（可能经 mask loop） | 全量 604，新鲜 1:0.2 划分 |
| 代码/权重 | gnnDir/gnn/r-gatDouble/ | modelAll/ 自包含 |
| 模型结构 | SingleEncoder_DualRGAT | 相同 |

checkpoint 格式与 `grd/io_utils.load_dual_rgat` 兼容，加载时把 `gnn_rgat_dir` 指向 `modelAll/` 即可。
