# liner

线性蒸馏 RGAT 教师输出（默认 Ridge），导出完整 30 项公式与教师残差，供 `lowExp` / `comb` 使用。

## 运行

```bash
cd /home/data/metalgnn/metalForTi/symbolTorch/liner
python run_liner.py \
  --csv /home/data/metalgnn/metalForTi/gnnDir/datacsv/datagnnUts.csv \
  --data-dir /home/data/metalgnn/metalForTi/modelAll/ysFs/data \
  --ckpt /home/data/metalgnn/metalForTi/modelAll/ysFs/runs/best_rgat_full.pt \
  --head0-name YS \
  --method ridge
```

产物见 `runs/<run_id>/`（成功后有 `_SUCCESS`，`runs/latest` 指向本次成功 run）。
