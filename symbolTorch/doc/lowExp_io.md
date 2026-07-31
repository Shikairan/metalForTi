# lowExp 接口：输入 / 输出

## 1. 输入

### 1.1 必需 `--liner-run`

指向成功的 `liner/runs/<id>/`（含 `_SUCCESS`、`manifest.json`、`residual_targets.pt`）。

残差目标：

```text
residual = teacher - linear
```

由 liner 预计算；lowExp **不再**重新跑 RGAT 或线性拟合。

### 1.2 可选 SR 参数

| 参数 | 说明 |
|------|------|
| `--sr-niterations` | PySR 迭代（默认 400） |
| `--sr-maxsize` | 复杂度上限（默认 40） |
| `--quick` | 少迭代冒烟 |
| `--seed` | 随机种子 |

`--include-val` 在残差模式下被忽略（仅 train 拟合）。

## 2. 输出（`lowExp/runs/<id>/`）

| 文件 | 含义 |
|------|------|
| `manifest.json` | 绑定 liner fingerprint / 头名称 / SR 配置 |
| `{head0,fs}_residual_sym.json/.pt` | 残差符号公式 |
| `residual_equations.md` | 可读残差方程 |
| `residual_predictions.pt` | 全体样本符号残差预测 |
| `metrics.json` | 零残差基线、符号残差 MAE、组合蒸馏 MAE 等 |
| `SR_output/` | PySR 原始输出 |
| `_SUCCESS` | 仅成功时写入 |

特征名仍为 `FEATURE_NAMES`（`el_Si`、`tem`、`fcr`、`coldway_*`）。
