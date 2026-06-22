# linResExp — 30 维线性基底 + PySR 残差

> **定位**：在 [lowExp](../lowExp/README.md)（纯 PySR）与 [highExp](../highExp/README.md)（图+符号）之间，提供**保证 30 个输入都出现在线性主项**、再用 PySR 补非线性的折中方案。

---

## 1. 做什么

```text
YS = 线性主项(30 维全变量 Ridge)  +  PySR(残差)
FS = 同上
```

1. 用教师 GNN（含图）在训练样本上产生软标签；
2. **Ridge 回归**拟合 `x(30维) → YS/FS`，线性式子**包含全部 30 个特征名**；
3. 对残差 `r = y_teacher − y_linear` 跑 **PySR**，学非线性修正；
4. 推理时**只用 x**，不用图。

---

## 2. 和 lowExp 对比

| | lowExp | linResExp |
|--|--------|-----------|
| 线性 30 维 | 无 | **有（Ridge，全变量）** |
| 非线性 | PySR 直接拟合教师 | PySR 只拟合残差 |
| 公式中变量数 | 通常远少于 30 | 线性部分固定 30 |
| 推理用图 | 否 | 否 |

---

## 3. 快速运行

```bash
cd symbolTorch/linResExp
python3.13 run_distill.py --quick --device cuda
```

常用参数：

```bash
--ridge-alpha 1.0      # Ridge 正则（默认 1.0）
--sr-niterations 400   # PySR 迭代（完整跑）
--include-val          # 蒸馏样本含验证集
--ckpt ../../modelAll/runs/best_rgat_full.pt
```

---

## 4. 输出文件（`linResExp/runs/`）

| 文件 | 说明 |
|------|------|
| `ys_linres.json` / `fs_linres.json` | **主看**：线性式 + 残差式 + 特征覆盖统计 |
| `ys_linear_basis.json` | 30 维 Ridge 系数 |
| `ys_residual_sym.json` | PySR 残差公式 |
| `metrics.json` | `teacher` / `linear_only` / `tabular_linres` 三路 MAE |
| `summary.md` | 两三行摘要 |

### `metrics.json` 字段

| 字段 | 含义 |
|------|------|
| `linear_only` | 仅 30 维线性基底的误差 |
| `tabular_linres` | 线性 + 残差合并后的误差 |
| `linres_gain_vs_linear_*` | 残差带来的 MAE 改善（正=变好） |
| `graph_info_loss_*` | lin+res 与教师的差距 |

---

## 5. 读公式

`ys_linres.json` 示例结构：

```json
{
  "equations": {
    "linear": "0.5 + 0.01*Al + ... + 0.002*coldway_17",
    "residual": "tem * 0.03 + ..."
  },
  "feature_coverage": {
    "linear_n": 30,
    "residual_n": 3
  }
}
```

- `linear`：**必定含 30 个特征名**（系数可为 0 量级）；
- `residual`：PySR 自由搜索，变量数不保证；
- 最终预测 = 两项之和。

---

## 6. 实现入口

- 脚本：[`run_distill.py`](run_distill.py)
- 线性基底：[`common/linear_basis.py`](../common/linear_basis.py)
- 混合模型：[`common/hybrid_models.py`](../common/hybrid_models.py) 中 `TabularLinResModel`
