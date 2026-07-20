# modelAll/ysFs — 全量 RGAT 训练（YS + FS）

在 **604 条全量样本**上训练双头模型（YS + FS）。数据来自 `gnnDir/datagnn.csv`，划分 **train:val = 1:0.2**（约 503 / 101）。

## 目录

```
modelAll/ysFs/
├── model_rgat.py      # RGAT 模型（与 r-gatDouble 同结构）
├── build_data.py        # 从 gnnDir 构建 PT 数据包
├── train.py             # 训练脚本
├── data/                # 图 + ys.pt/fs.pt + mask
└── runs/                # best_rgat_full.pt、train_log.csv
```

## 用法

在 **metalForTi 根目录**：

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"

python modelAll/ysFs/build_data.py --sanity
python modelAll/ysFs/train.py

# 快速冒烟
python modelAll/ysFs/train.py --epochs 5 --log-interval 1
```

checkpoint 格式与 `grd/io_utils.load_dual_rgat` 兼容：

```bash
--ckpt modelAll/ysFs/runs/best_rgat_full.pt \
--rgat-dir modelAll/ysFs
```

同级还有 [`../utsFs/`](../utsFs/)（UTS + FS 训练）。
