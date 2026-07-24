# lowExp

表格符号蒸馏：`x(30) → YS / FS`，推理不用图。

**完整文档**（接口、调用、QA）→ [`../doc/`](../doc/README.md)

## 最短命令

```bash
cd /home/data/metalgnn/metalForTi/symbolTorch/lowExp
python run_distill.py --quick
python run_distill.py --ckpt ../../modelAll/ysFs/runs/best_rgat_full.pt
```

结果默认写在 `runs/<时间戳或--run-name>/`；`runs/latest` 指向最近一次。
