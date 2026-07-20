# modelAll/utsFs — 全量 RGAT 训练（UTS + FS）

在 **604 条全量样本**上训练双头模型（UTS + FS）。数据来自 [`gnnDir/datacsv/datagnnUts.csv`](../../gnnDir/datacsv/datagnnUts.csv)，划分 **train:val = 1:0.2**。

模型结构与 [`../ysFs/`](../ysFs/) 相同（`SingleEncoder_DualRGAT`）；第一头参数名仍为 `ys_*`（兼容 `load_dual_rgat`），语义为 **UTS**。

## 目录

```
modelAll/utsFs/
├── model_rgat.py
├── build_data.py        # 读 datagnnUts.csv → uts.pt / fs.pt
├── train.py
├── data/                # build_data 生成
└── runs/                # best_rgat_uts_fs.pt
```

## 用法

在 **metalForTi 根目录**（需先有 `datagnnUts.csv`）：

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# 若尚未生成 datagnnUts.csv：
python -m preprocess.preprocess_datagnn_repro forward-uts \
  --input preprocess/dataOri2.csv \
  --output gnnDir/datacsv/datagnnUts.csv

python modelAll/utsFs/build_data.py --sanity
python modelAll/utsFs/train.py

# 快速冒烟
python modelAll/utsFs/train.py --epochs 5 --log-interval 1
```

加载示例：

```bash
--ckpt modelAll/utsFs/runs/best_rgat_uts_fs.pt \
--rgat-dir modelAll/utsFs
```

注意：第一头输出为 **UTS**（非 YS），Pareto/grd 若仍按 YS 语义使用需另行适配。
