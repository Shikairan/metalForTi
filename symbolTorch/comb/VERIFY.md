# RGAT 线性+残差符号蒸馏 — 运行记录

## 最新正式长程运行（推荐引用）

### YS/FS（utsFs，503/101 划分）

| 字段 | 值 |
|------|-----|
| Run ID | `full_ysfs_it400_ms40` |
| 路径 | [`comb/runs/full_ysfs_it400_ms40/`](runs/full_ysfs_it400_ms40/) |
| 报告 | [`RUN_REPORT.md`](runs/full_ysfs_it400_ms40/RUN_REPORT.md) |
| 逐样本误差 | `per_sample_errors.csv`（需 comb 更新后重跑才有；YS run 在 plan 实施前完成则无此文件） |

**Val 蒸馏 MAE**：YS 0.0249 / FS 0.0447

### UTS/FS（utsFsAll，604 全量训练）

| 字段 | 值 |
|------|-----|
| Run ID | `full_utsFsAll_it400_ms40` |
| 路径 | [`comb/runs/full_utsFsAll_it400_ms40/`](runs/full_utsFsAll_it400_ms40/) |
| 报告 | [`RUN_REPORT.md`](runs/full_utsFsAll_it400_ms40/RUN_REPORT.md) |
| 逐样本误差 | [`per_sample_errors.csv`](runs/full_utsFsAll_it400_ms40/per_sample_errors.csv)（604 行） |

**全表蒸馏 MAE**：UTS 0.0895 / FS 0.2189（linear only：0.1142 / 0.2665）

## 冒烟运行（`--quick`）

| Run | Head0 | L only | L+R |
|-----|-------|--------|-----|
| smoke_comb | YS | 0.0559 | 0.0501 |
| smoke_comb | FS | 0.0689 | 0.0571 |
| smoke_comb_uts | UTS | 0.0953 | 0.0817 |
| smoke_comb_uts | FS | 0.2337 | 0.2151 |

## 契约检查（均已通过）

- CSV ↔ `graph.x` 对齐阻断
- `teacher - linear = residual`
- 组合公式数值 ≡ 模块 `L+R`
- 失败 run 不写 `_SUCCESS` / 不更新 `latest`

## 一键复现正式运行

```bash
cd metalForTi/symbolTorch/comb
python run_comb.py \
  --csv ../gnnDir/datacsv/datagnnUts.csv \
  --data-dir ../modelAll/ysFs/data \
  --ckpt ../modelAll/ysFs/runs/best_rgat_full.pt \
  --head0-name YS \
  --sr-niterations 400 --sr-maxsize 40 \
  --run-name full_ysfs_it400_ms40
```
