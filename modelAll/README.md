# modelAll — 全量 RGAT 训练入口

按预测目标 / 划分策略分成独立子目录，互不覆盖：

| 子目录 | 目标 | 划分 | 默认数据 | checkpoint |
|--------|------|------|----------|------------|
| [`ysFs/`](ysFs/) | YS + FS | train:val≈1:0.2 | `gnnDir/datagnn.csv` | `ysFs/runs/best_rgat_full.pt` |
| [`utsFs/`](utsFs/) | UTS + FS | train:val≈1:0.2 | `gnnDir/datacsv/datagnnUts.csv` | `utsFs/runs/best_rgat_uts_fs.pt` |
| [`utsFsAll/`](utsFsAll/) | UTS + FS | **604 全部训练** | 同上 | `utsFsAll/runs/best_rgat_uts_fs_all.pt` |

## 快速开始

```bash
cd metalForTi
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# YS + FS
python modelAll/ysFs/build_data.py --sanity
python modelAll/ysFs/train.py

# UTS + FS（有验证集）
python modelAll/utsFs/build_data.py --sanity
python modelAll/utsFs/train.py

# UTS + FS（604 全训练）
python modelAll/utsFsAll/build_data.py --sanity
python modelAll/utsFsAll/train.py
```

Pareto / grd / symbolTorch 默认仍指向 **YS+FS** 权重：`modelAll/ysFs/`。
