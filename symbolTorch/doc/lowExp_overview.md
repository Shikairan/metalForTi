# lowExp 概述

## 1. 目标

用已训练的 **RGAT 双头教师**（YS + FS）在训练图上做一次前向，得到每个节点的教师预测；再用 **SymTorch + PySR** 拟合：

```text
x ∈ R^{30}  →  YS_hat, FS_hat
```

**推理阶段不再使用边 / 图结构**，得到人类可读的公式（写在 JSON 里）。

与教师相比，公式通常更易解释，但可能损失「邻居信息」带来的精度；`metrics.json` 里的 `graph_info_loss_*` 量化这种差距。

## 2. 流水线

```text
data-dir（material_graph + ys/fs + masks）
        +
ckpt（RGAT_Dual / 兼容 Dual 头权重）
        │
        ▼
  教师全图 forward → teacher_predictions.pt
        │
        ▼
  在 sample_mask 节点上收集 (x, teacher_ys/fs)
        │
        ▼
  PySR 蒸馏 tabular_ys / tabular_fs
        │
        ▼
  ys_tabular_sym.json/.pt
  fs_tabular_sym.json/.pt
  metrics.json / summary.md
```

默认 `sample_mask = train_mask`；加 `--include-val` 时为 `train | val`。

## 3. 目录结构（symbolTorch）

```text
symbolTorch/
  doc/                 ← 本文档
  common/              ← CLI、数据、教师、蒸馏、指标
  lowExp/
    run_distill.py     ← 唯一实验入口
    runs/
      <run-name>/
        equations.md / metrics.json / …
        SR_output/          ← 本 run 的 PySR 临时文件（按 block 分子目录）
          tabular_uts/ 或 tabular_ys/
          tabular_fs/
      latest -> …
  scripts/check_env.py
  run_all.sh
  requirements.txt
```

## 4. 默认路径（代码常量）

| 项 | 默认值 |
|----|--------|
| 数据 | `metalForTi/gnnDir/gnndataPT/r-gatPT/` |
| 教师权重 | `metalForTi/modelAll/ysFs/runs/best_rgat_full.pt` |
| 输出 | `symbolTorch/lowExp/runs/<run-name>/`（每次新建；`latest` → 最近一次） |

均可被 CLI 覆盖，见 [lowExp_io.md](lowExp_io.md)、[lowExp_usage.md](lowExp_usage.md)。
