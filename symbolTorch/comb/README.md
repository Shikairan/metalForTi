# comb

联合入口：调用 `liner` + `lowExp`，组合 `L(x)+R(x)`，导出模型空间公式与物理解释文档。

## 一键（复用已有阶段）

```bash
cd /home/data/metalgnn/metalForTi/symbolTorch/comb
python run_comb.py \
  --liner-run ../liner/runs/smoke_liner \
  --lowexp-run ../lowExp/runs/smoke_lowexp \
  --run-name smoke_comb
```

## 全量流水线

```bash
python run_comb.py \
  --csv /home/data/metalgnn/metalForTi/gnnDir/datacsv/datagnnUts.csv \
  --data-dir /home/data/metalgnn/metalForTi/modelAll/ysFs/data \
  --ckpt /home/data/metalgnn/metalForTi/modelAll/ysFs/runs/best_rgat_full.pt \
  --head0-name YS \
  --quick
```
