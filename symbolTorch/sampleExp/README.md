# sampleExp — 样本级 SLIME 解释

## 思路

对**某一个合金节点**（默认：验证集 FS 相对误差最大的节点）：

1. 固定整张图与边；
2. 只扰动该节点的 30 维特征；
3. 用 SLIME 在训练集特征邻域 + 高斯合成样本上，拟合局部符号式 `x → YS` 与 `x → FS`。

回答：「这个配方为什么预测成这样？」——公式**只对附近样本有效**，不是全局定律。

## 运行

```bash
cd metalForTi/symbolTorch/sampleExp

# 默认：val 上 FS 最差节点
/root/miniconda3/bin/python3.13 run_distill.py --quick

# 指定节点
/root/miniconda3/bin/python3.13 run_distill.py --node-idx 575 --quick

# 多个 val 难例
/root/miniconda3/bin/python3.13 run_distill.py --top-k 5 --quick
```

### SLIME 相关参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--slime-nn` | 10 | 训练集中最近邻个数 `J_nn` |
| `--slime-synthetic` | 100 | 目标点周围合成样本数 |

## 输出（`runs/`）

```
runs/
  node_00575/
    ys_formula.txt      # 人类可读公式
    ys_formula.json
    fs_formula.txt
    fs_formula.json
    local_meta.json     # 节点索引、x0、SLIME 参数
  metrics.json          # 解释的节点列表
  summary.md
```

**注意**：每个节点需多次全图前向（扰动样本），`--top-k` 较大时较慢；`--quick` 会减少合成样本与 SR 迭代。
