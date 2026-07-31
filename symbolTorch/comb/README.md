# comb

联合入口：调用 `liner` + `lowExp`，组合 `L(x)+R(x)`，导出模型空间公式与物理解释文档。

## PyTorch 直接推理

无图组合公式模块（默认 `full_utsFsAll_it400_ms40`）：

```bash
cd /home/data/metalgnn/metalForTi/symbolTorch
python comb/utsfsall_combined_torch.py --check
```

```python
from comb.utsfsall_combined_torch import load_from_run_dir

model = load_from_run_dir(device="cuda")  # 或 cpu
x = ...  # float tensor [N, 30]，列顺序同 datagnnUts.csv
uts, fs = model(x)                        # 模型量纲
uts_mpa, fs_phys = model.to_physical(uts, fs)
```


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
