# symbolTorch

将教师 GNN（YS/FS）蒸馏为**可读符号公式**。当前仅保留 **lowExp**（表格 30 维 → YS/FS，推理无图）。

## 文档

完整说明见 **[doc/](doc/README.md)**：

- [概述](doc/lowExp_overview.md)
- [接口输入输出](doc/lowExp_io.md)
- [调用](doc/lowExp_usage.md)
- [QA](doc/lowExp_qa.md)

## 快速开始

```bash
cd /home/data/metalgnn/metalForTi/symbolTorch
pip install -r requirements.txt   # Python >= 3.11
python scripts/check_env.py

cd lowExp
python run_distill.py --quick
# 指定教师：
# python run_distill.py --ckpt /path/to/rgat_dual.pt --hidden-dim 64
```

默认教师：`../modelAll/ysFs/runs/best_rgat_full.pt`  
默认数据：`../gnnDir/gnndataPT/r-gatPT/`  
默认输出：`lowExp/runs/<run-name>/`（不覆盖历史；`latest` → 最近一次）
