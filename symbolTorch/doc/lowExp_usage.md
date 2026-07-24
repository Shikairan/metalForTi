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

在 `metalForTi` 仓库根或 `lowExp` 目录均可；下列以仓库内绝对路径为例。

### 2.1 快速冒烟（必加 `--quick`）

```bash
cd /home/data/metalgnn/metalForTi/symbolTorch/lowExp
python run_distill.py --quick
```

GPU：

```bash
python run_distill.py --quick --device cuda
```

### 2.2 指定教师权重与数据

```bash
python run_distill.py \
  --ckpt /home/data/metalgnn/metalForTi/modelAll/ysFs/runs/best_rgat_full.pt \
  --data-dir /home/data/metalgnn/metalForTi/gnnDir/gnndataPT/r-gatPT \
  --run-name ysFs_it800 \
  --device cuda
# 输出 → lowExp/runs/ysFs_it800/
# 或看快捷方式：lowExp/runs/latest/
```

换其它已训 Dual 头 ckpt：

```bash
python run_distill.py --ckpt /path/to/other_rgat_dual.pt --hidden-dim 64
```

`hidden_dim` 必须与该 ckpt 一致。

### 2.3 蒸馏样本含验证集

```bash
python run_distill.py --include-val --sr-niterations 400
```

### 2.4 批量脚本

```bash
cd /home/data/metalgnn/metalForTi/symbolTorch
./run_all.sh
# 每次写入新的 runs/<时间戳>_quick_s42/，不删除历史
# PYTHON=/path/to/python3.13 CKPT=/path/to.pt ./run_all.sh
```

日志：`symbolTorch/runs_all.log`、`symbolTorch/logs/lowExp.log`。

## 3. 查看结果

```bash
cd /home/data/metalgnn/metalForTi/symbolTorch/lowExp/runs
ls            # 各次运行子目录
cat latest/equations.md   # 最近一次
```

## 4. 程序内复用（可选）

蒸馏完成后可用 `TabularSymbolicModel` 对 `(N,30)` 张量推理（内部会把输入放到 CPU 跑符号模块）：

```python
import torch
from common.hybrid_models import TabularSymbolicModel
# 需自行用 distill_io / dill 加载 ys_tabular_sym.pt、fs_tabular_sym.pt
# tabular = TabularSymbolicModel(sym_ys, sym_fs).eval()
# ys_hat, fs_hat = tabular(x)  # x: (N, 30)
```

日常以 JSON 公式为主即可。

## 5. 与仓库其它模块的关系

| 模块 | 关系 |
|------|------|
| `modelAll/ysFs` | 默认教师来源 |
| `gnnDir/.../r-gatPT` | 默认图数据 |
| `Pareto` / `grd` | **不依赖** lowExp；lowExp 是独立解释链路 |
| `utsFs` / `utsFsAll` | 可作 `--ckpt`，但第一头是 UTS 语义，见 [QA](lowExp_qa.md) |
