# lowExp（残差模式）

对 `liner` 产出的 **教师残差**（`teacher - linear`）做 SymTorch/PySR 符号回归。推理无图。

必须提供 `--liner-run`（含 `_SUCCESS` 与 `residual_targets.pt`）。

## 运行

```bash
cd /home/data/metalgnn/metalForTi/symbolTorch/lowExp
python run_distill.py \
  --liner-run ../liner/runs/latest \
  --quick
```

正式搜索去掉 `--quick`，可调 `--sr-niterations` / `--sr-maxsize`。

联合入口见 `../comb/run_comb.py`。
