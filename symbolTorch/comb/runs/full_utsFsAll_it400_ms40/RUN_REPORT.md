# 长程运行记录：full_utsFsAll_it400_ms40

**模型**：`modelAll/utsFsAll`（604 全量训练，UTS+FS，无 held-out val）  
**完成时间**：2026-07-30  
**状态**：成功（`_SUCCESS`）  
**耗时**：约 68 分钟

## 命令

```bash
cd /home/data/metalgnn/metalForTi/symbolTorch/comb
python run_comb.py \
  --csv /home/data/metalgnn/metalForTi/gnnDir/datacsv/datagnnUts.csv \
  --data-dir /home/data/metalgnn/metalForTi/modelAll/utsFsAll/data \
  --ckpt /home/data/metalgnn/metalForTi/modelAll/utsFsAll/runs/best_rgat_uts_fs_all.pt \
  --head0-name UTS --head1-name FS \
  --linear-method ridge \
  --sr-niterations 400 --sr-maxsize 40 \
  --device cpu --seed 42 \
  --run-name full_utsFsAll_it400_ms40
```

## 配置

| 项 | 值 |
|----|-----|
| N | 604（`full_train_mode=true`，train=val=全表） |
| Ridge alpha UTS/FS | 0.1 / 0.1 |
| PySR | 400 iter, maxsize 40 |
| 指纹 | `19881268f0e9667fcdf863d6fc60dfbb` |

> **注意**：本 run 无独立验证集；下列 MAE 为**全表**蒸馏误差，不可与 utsFs 的 held-out val 直接对比。

## 全表蒸馏 MAE（相对 RGAT 教师）

| 目标 | 仅线性 L | 线性+残差 L+R |
|------|----------|---------------|
| **UTS** | 0.1142 | **0.0895** |
| **FS** | 0.2665 | **0.2189** |

## 公式（模型空间）

### 线性 L(x)（30 项 Ridge）

见 [`linear_equations.md`](linear_equations.md) 或 [`final_equations_model_space.md`](final_equations_model_space.md) 的 Linear 小节。

**UTS 线性（节选）**：

```text
L_UTS = 0.8350171 + 0.040941132*Al - 0.061953447*Zr + ... + 0.031604143*fcr + ...（共 30 项）
```

**FS 线性（节选）**：

```text
L_FS = 1.4511466 - 0.084648437*Al + ... + 0.16405049*tem - 0.13409522*fcr + ...（共 30 项）
```

### 非线性残差 R(x)（PySR）

见 [`residual_equations.md`](residual_equations.md)。

**R_UTS**：

```text
R_UTS = fcr * tem * (fcr / (Al + 0.08873339)) * (0.10387048 - 0.097857766*Zr) * (Mo + el_Si)
```

**R_FS**：

```text
R_FS = (tem - 0.5413826) * (tem - (-1)*0.65531564 / (fcr*(fcr - (-Al - 1.2845563)))) * 0.3372095 - 0.2900931
```

### 组合 Y = L + R

见 [`final_equations_model_space.md`](final_equations_model_space.md) 与 [`final_equations.json`](final_equations.json)。

## 每条数据误差

**主文件**：[`per_sample_errors.csv`](per_sample_errors.csv)（604 行 × 23 列）

| 列组 | 含义 |
|------|------|
| `sample_id` | 0..603，与 CSV/图节点行序一致 |
| `teacher_*`, `linear_*`, `residual_true_*`, `residual_pred_*`, `combined_*` | 各阶段预测值（模型量纲） |
| `abs_err_linear_vs_teacher_*` | \|L − teacher\| |
| `abs_err_residual_pred_vs_true_*` | \|R_pred − (teacher−L)\| |
| `abs_err_combined_vs_teacher_*` | \|L+R − teacher\|（蒸馏误差） |
| `label_*`, `abs_err_*_vs_label_*` | 相对真实标签 UTS/FS 的误差 |

汇总统计：[`per_sample_errors.json`](per_sample_errors.json)

机器可读预测张量：[`predictions.pt`](predictions.pt)

## 产物清单

```text
comb/runs/full_utsFsAll_it400_ms40/
├── _SUCCESS
├── linear_equations.md          # 线性公式
├── residual_equations.md        # 残差公式
├── final_equations_model_space.md
├── final_equations.json
├── per_sample_errors.csv        # ★ 逐样本误差
├── per_sample_errors.json
├── metrics.json
├── predictions.pt
├── liner_stage/
└── lowexp_stage/
```

`comb/runs/latest` → `full_utsFsAll_it400_ms40`
