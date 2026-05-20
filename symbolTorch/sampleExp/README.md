# sampleExp 初学者教例 — 解释「这一条配方」为什么预测成这样

> 适合：某条合金在验证集上误差特别大，你想写「案例分析」而不是全局公式。  
> **不适合**：第一次接触 symbolTorch（请先跑 [lowExp](../lowExp/README.md)）。

---

## 0. 本教例结束后你会得到什么

在 `runs/node_XXXXX/` 文件夹里（`XXXXX` 是节点编号）：

- `ys_formula.txt` — 局部 YS 公式（纯文本，记事本可开）
- `fs_formula.txt` — 局部 FS 公式
- `local_meta.json` — 这条配方的 30 个原始特征值
- 根目录 `metrics.json` — 解释了哪几个节点

**重要**：这些公式**只对这一条配方附近有效**，不能当成全数据集定律。

---

## 1. 用故事理解 SLIME / sampleExp

全局公式（lowExp）像「全国平均菜谱」。  
但某家饭馆的菜特别咸（**难例节点**），平均菜谱解释不了。

**sampleExp** 的做法：

1. 选定这家饭馆（`node_idx`）；  
2. 在「训练集里配方相似」的样本附近，稍微改动配方成分做实验；  
3. 看教师 GNN 预测怎么变；  
4. 拟合一个**只在这个小区间里**成立的公式。

技术上叫 **SLIME**（SymTorch 内置），比老方法 LIME 更适合弯曲的预测面。

---

## 2. 和另外三档的区别（初学者表）

| | sampleExp | lowExp | highExp |
|--|-----------|--------|---------|
| 公式数量 | 每节点 2 个（YS+FS） | 全局 2 个 | 全局很多 |
| 适用范围 | **局部** | 全局 | 全局 |
| 是否用图 | 是（每次改一个节点要全图算） | 否 | 是 |
| 运行速度 | 慢（多次全图前向） | 较快 | 很慢 |

---

## 3. 开始前检查清单

- [ ] 教师和数据就绪（同总教程）
- [ ] 知道「节点编号」是 0 到 N-1 的整数，对应 `material_graph` 里节点顺序（与 datagnn 行序一致）
- [ ] 第一次加 `--quick`（减少合成样本数，快一些）

---

## 4. 手把手运行

### 4.1 默认：自动选「最难」的验证集节点

```bash
cd /home/data/metalTi/metalForTi/symbolTorch/sampleExp

/root/miniconda3/bin/python3.13 run_distill.py --quick --device cuda
```

脚本会：

1. 用教师在验证集上算预测；  
2. 找 FS 相对误差最大的那个节点（与训练脚本里的 worst 逻辑类似）；  
3. 对该节点跑 SLIME，生成 `runs/node_XXXXX/`。

### 4.2 指定某一条配方

若你知道节点号是 `575`（举例）：

```bash
/root/miniconda3/bin/python3.13 run_distill.py --node-idx 575 --quick --device cuda
```

### 4.3 一次解释多个难例

```bash
/root/miniconda3/bin/python3.13 run_distill.py --top-k 3 --quick --device cuda
```

会生成 `node_xxx`、`node_yyy`、`node_zzz` 三个文件夹，**时间约 ×3**。

### 4.4 查看结果

```bash
cat runs/summary.md
ls runs/
cat runs/node_00575/ys_formula.txt    # 编号以实际为准
cat runs/node_00575/local_meta.json
```

---

## 5. 输入 / 输出详解

### 5.1 输入

| 输入 | 说明 |
|------|------|
| 教师 + 全图 | 每次试探配方都要整张图前向一次 |
| `x_pool` | 训练集节点的 30 维特征，作 SLIME 的「邻居池」 |
| `x0` | 目标节点自己的 30 维特征 |
| SLIME 参数 | `--slime-nn`（默认 10 个最近邻）、`--slime-synthetic`（默认 100 个合成点） |

### 5.2 输出目录结构

```text
sampleExp/runs/
├── metrics.json
├── summary.md
└── node_00575/              ← 示例编号
    ├── ys_formula.txt       ← 【给同事看】
    ├── fs_formula.txt
    ├── ys_formula.json
    ├── fs_formula.json
    └── local_meta.json      ← 含 x0 配方数值
```

### 5.3 `local_meta.json` 怎么用

里面的 `x0` 是长度 30 的数组，按顺序对应：

`Al, Zr, Sn, Mo, Cr, Nb, el_Si, V, Ta, Fe, tem, fcr, coldway_0…17`

写报告时可以把 `x0` 和公式对照：**当时这条配方各元素是多少，公式如何组合它们。**

### 5.4 `metrics.json`

```json
{
  "nodes": [575],
  "default_worst_val_fs_node": 575,
  "slime_nn": 10,
  "slime_synthetic": 100
}
```

只记录「解释了谁」，**不像 lowExp 那样有 MAE**（局部解释侧重可读性，不侧重全局误差）。

---

## 6. 实现原理（初学者版）

核心类：`NodeBranchPredictor`（在 [`run_distill.py`](run_distill.py)）

```text
对于每一个试探配方 x'（30 个数）：
    复制整张图的 x
    只把【目标节点】那一行换成 x'
    教师全图前向 → 读出该节点的 YS 或 FS
```

SymTorch 收集很多 `(x', 预测值)` 对，用 SLIME 搜局部公式。

**为何慢？** 每个试探点都是一次完整 GNN 前向；`num_synthetic=100` 就要上百次。

---

## 7. 写案例分析报告的建议模板

```markdown
### 节点 575 预测偏差分析

- 实测 FS vs 预测 FS：…
- 该配方特征（见 local_meta.json）：Al=…, tem=…
- 局部 YS 解释公式：…（粘贴 ys_formula.txt）
- 局部 FS 解释公式：…
- 说明：公式为 SLIME 局部近似，仅用于该配方邻域，不可外推至其他成分设计。
```

---

## 8. 常用参数

| 参数 | 默认 | 初学者建议 |
|------|------|------------|
| `--node-idx` | 自动 | 已知难例时指定 |
| `--top-k` | 1 | 不要一次设很大 |
| `--slime-synthetic` | 100 | quick 时自动减小 |
| `--slime-nn` | 10 | 一般不用改 |

---

## 9. 常见错误

| 现象 | 原因 | 处理 |
|------|------|------|
| 极慢 | 全图前向太多 | `--quick`、减小 `--slime-synthetic` |
| 公式很奇怪 | 局部拟合/教师本身差 | 结合教师误差；勿过度解读 |
| 节点号不知道 | 未记录 | 从 `train_log.csv` 或 `metrics` 找 worst_node_idx |

---

## 10. 下一步

- 全局公式 → [lowExp](../lowExp/README.md)  
- 成分层公式 → [highExp](../highExp/README.md)  
- 总教程 → [../README.md](../README.md)
