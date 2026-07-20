# modelAll — 全量 RGAT 训练入口

按预测目标分成独立子目录，互不覆盖：

| 子目录 | 目标 | 默认数据 | checkpoint |
|--------|------|----------|------------|
| [`ysFs/`](ysFs/) | YS + FS | `gnnDir/datagnn.csv` | `ysFs/runs/best_rgat_full.pt` |
| [`utsFs/`](utsFs/) | UTS + FS | `gnnDir/datacsv/datagnnUts.csv` | `utsFs/runs/best_rgat_uts_fs.pt` |

## 快速开始

```bash
cd metalForTi
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# YS + FS（原 modelAll 流程）
python modelAll/ysFs/build_data.py --sanity
python modelAll/ysFs/train.py

# UTS + FS
python modelAll/utsFs/build_data.py --sanity
python modelAll/utsFs/train.py
```

Pareto / grd / symbolTorch 默认仍指向 **YS+FS** 权重：`modelAll/ysFs/`。
