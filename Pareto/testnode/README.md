# testnode — 随机杂交子代 YS/FS

父本池随机配对杂交，校验合法基因后对子代做 GNN forward，**只输出 ys 和 fs 预测**。

## 用法

在 **metalForTi 根目录**：

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"

python -m Pareto.testnode.run_combo_test
python -m Pareto.testnode.run_combo_test --num-offspring 604 --seed 0
python -m Pareto.testnode.run_combo_test --no-forward   # 仅合法性
```

## 输出

| 文件 | 内容 |
|------|------|
| `outputs/combo_ys_fs.json` | 每条子代的 `ys`、`fs`，以及汇总 min/max/mean/median |
| `outputs/combo_ys_fs.csv` | 同上，CSV 格式 |
