# lowExp 调用指南

## 1. 环境

- **Python ≥ 3.11**（`torch-symbolic` 要求）
- 建议独立环境，安装：

```bash
cd /home/data/metalgnn/metalForTi/symbolTorch
pip install -r requirements.txt
```

另需 **Julia**（PySR 首次运行常自动安装）。可选：`dill`（更稳地保存 `*_sym.pt`）。

自检：

```bash
cd /home/data/metalgnn/metalForTi/symbolTorch
python scripts/check_env.py
```

## 2. 推荐命令

lowExp **必须**先有成功的 `liner` 运行（`--liner-run`）。

### 2.1 快速冒烟

```bash
cd /home/data/metalgnn/metalForTi/symbolTorch/liner
python run_liner.py --head0-name YS --method ridge --run-name demo

cd ../lowExp
python run_distill.py --liner-run ../liner/runs/demo --quick
```

### 2.2 联合入口

```bash
cd /home/data/metalgnn/metalForTi/symbolTorch/comb
python run_comb.py --quick --head0-name YS \
  --data-dir /home/data/metalgnn/metalForTi/modelAll/ysFs/data \
  --ckpt /home/data/metalgnn/metalForTi/modelAll/ysFs/runs/best_rgat_full.pt
```

或复用阶段：

```bash
python run_comb.py --liner-run ../liner/runs/demo --lowexp-run ../lowExp/runs/latest
```

## 3. 成功标记

仅成功运行会写入 `_SUCCESS` 并更新 `runs/latest`。失败目录可保留，但不会被 `latest` 指向。
