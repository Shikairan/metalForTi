# symbolTorch

将 RGAT 教师蒸馏为 **线性基线 + 非线性符号残差**：

```text
Y = L(x) + R(x)
```

| 模块 | 职责 |
|------|------|
| `liner/` | Ridge/OLS 蒸馏教师 → 线性公式 + 残差 |
| `lowExp/` | PySR 拟合残差（需 `--liner-run`） |
| `comb/` | 一键串联并导出最终公式 |

## 快速开始

```bash
cd /home/data/metalgnn/metalForTi/symbolTorch
# 1) 线性
cd liner && python run_liner.py --head0-name YS --method ridge
# 2) 残差符号（冒烟）
cd ../lowExp && python run_distill.py --liner-run ../liner/runs/latest --quick
# 3) 组合
cd ../comb && python run_comb.py --liner-run ../liner/runs/latest --lowexp-run ../lowExp/runs/latest
```

或：`python comb/run_comb.py --quick ...` 一键跑通。

文档见 [doc/](doc/README.md)、[comb/VERIFY.md](comb/VERIFY.md) 与 `comb/runs/full_ysfs_it400_ms40/RUN_REPORT.md`。
