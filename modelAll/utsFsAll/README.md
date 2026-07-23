# modelAll/utsFsAll — 604 全量训练（UTS + FS，无 held-out）

与 [`../utsFs/`](../utsFs/) 相同模型结构与数据源（`datagnnUts.csv`），区别是：

| 项 | utsFs | utsFsAll |
|----|-------|----------|
| 划分 | train:val ≈ 503:101 | **604 全部训练** |
| 选优 | val MAE | 全表 MAE（`full_score`） |
| checkpoint | `best_rgat_uts_fs.pt` | `best_rgat_uts_fs_all.pt` |

## 目录

```
modelAll/utsFsAll/
├── model_rgat.py
├── build_data.py
├── train.py
├── data/                 # build_data 生成；train_mask=val_mask=全 True
└── runs/                 # best_rgat_uts_fs_all.pt
```

## 用法

```bash
cd metalForTi
export PYTHONPATH="$(pwd):${PYTHONPATH}"

python modelAll/utsFsAll/build_data.py --sanity
python modelAll/utsFsAll/train.py
# 冒烟
python modelAll/utsFsAll/train.py --epochs 5 --log-interval 1
```
