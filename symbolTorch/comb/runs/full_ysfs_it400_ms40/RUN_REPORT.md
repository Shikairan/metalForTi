# 长程运行记录：full_ysfs_it400_ms40

**完成时间**：2026-07-30（UTC+8 约 17:31）  
**状态**：成功（`_SUCCESS`）  
**总耗时**：约 78 分钟（含 liner + 双头 PySR 400 迭代）

## 命令

```bash
cd /home/data/metalgnn/metalForTi/symbolTorch/comb
python run_comb.py \
  --csv /home/data/metalgnn/metalForTi/gnnDir/datacsv/datagnnUts.csv \
  --data-dir /home/data/metalgnn/metalForTi/modelAll/ysFs/data \
  --ckpt /home/data/metalgnn/metalForTi/modelAll/ysFs/runs/best_rgat_full.pt \
  --head0-name YS --head1-name FS \
  --linear-method ridge \
  --sr-niterations 400 --sr-maxsize 40 \
  --device cpu --seed 42 \
  --run-name full_ysfs_it400_ms40
```

## 配置摘要

| 项 | 值 |
|----|-----|
| 样本数 N | 604（train 503 / val 101） |
| 线性方法 | Ridge（CV 选 alpha） |
| alpha YS / FS | 1.0 / 10.0 |
| PySR | niterations=400, maxsize=40, SLIME=False |
| 指纹 | `e3eda163cc76d22eb8d9c4a155856f0e` |

## 验证集蒸馏 MAE（相对 RGAT 教师）

| 目标 | 仅线性 L | 线性+残差 L+R | 相对 quick 冒烟 |
|------|----------|---------------|-----------------|
| **YS** | 0.0559 | **0.0249** | 0.0501 → **-50%** |
| **FS** | 0.0689 | **0.0447** | 0.0571 → **-22%** |

符号残差对教师残差的 val MAE：YS **0.0249**，FS **0.0447**。

## 验证集相对真实标签 MAE

| 目标 | RGAT 教师 | L+R 组合 | Δ（组合−教师） |
|------|-----------|----------|----------------|
| YS | 0.1514 | 0.1548 | +0.0034 |
| FS | 0.3311 | 0.3353 | +0.0042 |

组合公式相对标签略差于教师（预期：无图近似），但蒸馏教师误差显著低于仅线性。

## 产物路径

```text
comb/runs/full_ysfs_it400_ms40/
├── _SUCCESS
├── manifest.json
├── metrics.json
├── predictions.pt
├── final_equations.json
├── final_equations_model_space.md   # 完整 L+R 公式
├── final_equations_physical.md      # 编解码说明
├── summary.md
├── liner_stage/                     # 线性 + residual_targets.pt
├── lowexp_stage/                    # 残差符号 + SR_output/
├── full_ysfs_it400_ms40.log         # 控制台日志（若 tee）
```

`comb/runs/latest` → `full_ysfs_it400_ms40`

## 最终公式（模型空间，节选）

**YS** = L_YS(x) + R_YS(x)

- 线性：30 项 Ridge（截距 1.0644，含 tem/fcr/coldway 等）
- 残差：`R_YS` 含 `fcr`, `tem`, `Nb`, `coldway_2` 等非线性项

**FS** = L_FS(x) + R_FS(x)

- 残差：`sin(...) * inv(tem + ...)` 形式，使用 `fcr`, `tem`, `Zr`, `el_Si`

完整式见 `final_equations_model_space.md`；机器可读见 `final_equations.json`。

## 结论

1. 正式 PySR（400 iter）相对 `--quick`（40 iter）在 val 蒸馏 MAE 上明显改善。
2. 残差符号层有效吸收图教师在线性基线之外的信号。
3. 流水线契约（对齐、指纹、公式回放、`_SUCCESS`）在长程运行中保持稳定。

## 复现

```bash
python comb/run_comb.py \
  --liner-run comb/runs/full_ysfs_it400_ms40/liner_stage \
  --lowexp-run comb/runs/full_ysfs_it400_ms40/lowexp_stage \
  --run-name replay_comb
```
