# lowExp 初学者教例 — 用一个总公式预测 YS / FS（不用图）

> **建议第一次跑 symbolTorch 就从本文件夹开始。**  
> 难度最低、耗时相对最短、最容易在 JSON 里看到「像公式一样」的结果。

---

## 0. 本教例结束后你会得到什么

- 两个 JSON 文件，里面是类似初中学公式的表达式，例如：  
  `YS ≈ 0.48 + Nb * (-0.001) + tem * 0.02 + …`
- 一个 `metrics.json`，告诉你这个公式和「黑盒 GNN」差多少
- **不需要**理解图网络，**不需要** 64 维隐变量

---

## 1. 用生活例子理解 lowExp

想象班里有一个很厉害的同桌（**教师 GNN**），他不仅会算题，还会看「你和谁坐得近」（**图**）来改答案。

**lowExp 做的事**：

1. 让同桌先把每道题的答案抄下来（教师预测）；  
2. 你只根据「题目条件」（30 个配方数字）找规律，写成一个**固定公式**；  
3. 以后算新题**只看题目、不看邻座**（推理不用图）。

所以：公式最好懂，但可能不如同桌本人准——尤其当「邻座影响」很重要时。

---

## 2. 和另外三档比一比（初学者表）

| | lowExp | highExp | sampleExp |
|--|--------|---------|-----------|
| 公式个数 | 全局 2 个（YS、FS） | encoder 很多维 | 每个配方一套 |
| 要不要图 | **不要** | 要 | 要（只改一个点） |
| 变量名 | Al, tem, coldway | 同左 | 同左 |
| 第一次跑 | **推荐** | 第二推荐 | 第三推荐 |

---

## 3. 开始前检查清单

在终端逐项打勾：

- [ ] Python ≥ 3.11，已 `pip install -r requirements.txt`
- [ ] `python3.13 scripts/check_env.py` 通过
- [ ] 存在 `gnnDir/gnndataPT/r-gatPT/material_graph.pt`
- [ ] 存在 `gnnDir/gnn/r-gatDouble/runs/best_ysfs_gat.pt`
- [ ] 你知道结果会写在 `lowExp/runs/`（第一次跑会自动创建）

---

## 4. 手把手：三步跑通

### 步骤 1：进入目录

```bash
cd /home/data/metalTi/metalForTi/symbolTorch/lowExp
```

### 步骤 2：运行（初学者务必加 `--quick`）

```bash
/root/miniconda3/bin/python3.13 run_distill.py --quick
```

有 NVIDIA 显卡可加：

```bash
/root/miniconda3/bin/python3.13 run_distill.py --quick --device cuda
```

**终端里你会看到**：大量 PySR / Julia 日志、`Running SR on output dimension 0`、最后 `Done. Outputs in ...`。

### 步骤 3：打开结果

```bash
# 人类可读总结
cat runs/summary.md

# YS 公式（核心）
cat runs/ys_tabular_sym.json

# 误差数字
cat runs/metrics.json
```

---

## 5. 输入说明（脚本吃了什么）

### 5.1 必需输入（默认路径，不用改）

| 输入 | 路径 | 含义 |
|------|------|------|
| 图数据目录 | `../../gnnDir/gnndataPT/r-gatPT` | 通过 `--data-dir` 指定 |
| 教师权重 | `../../gnnDir/gnn/r-gatDouble/runs/best_ysfs_gat.pt` | 通过 `--ckpt` 指定 |

### 5.2 内部数据流（给好奇的初学者）

```text
1. 读取 N 条合金，每条 30 个特征 → 矩阵 x
2. 教师在全图上算一遍 → 得到 ys_teacher, fs_teacher（软标签）
3. 只用「训练集」那些行 (train_mask=True)：
      输入 x_train  ,  目标 ys_teacher_train
4. PySR 搜索公式，使公式(x_train) ≈ ys_teacher_train
5. 对「验证集」和公式算误差，写入 metrics.json
```

**注意**：公式学的是**教师的输出**，不是直接抄 CSV 里的真值；教师本身已经学过真值了。

### 5.3 可选参数

```bash
# 看全部参数
/root/miniconda3/bin/python3.13 run_distill.py --help

# 常用：
--quick              # 第一次必加
--device cuda        # 有 GPU 时加速教师
--include-val        # 蒸馏时也用验证集样本（样本更多，更慢）
--sr-niterations 600 # 完整跑时加大（默认 400）
--out-dir ./my_runs  # 换输出目录
```

---

## 6. 输出说明（逐项解释）

默认目录：`lowExp/runs/`

| 文件 | 要不要看 | 内容 |
|------|----------|------|
| `ys_tabular_sym.json` | **必看** | YS 公式 |
| `fs_tabular_sym.json` | **必看** | FS 公式 |
| `metrics.json` | **必看** | 误差对比 |
| `summary.md` | 建议看 | 两三行总结 |
| `teacher_predictions.pt` | 可忽略 | 教师全图预测备份 |
| `ys_tabular_sym.pt` | 可忽略 | 程序内部用，较大 |
| `SR_output/` | 可删 | PySR 临时文件 |

### 6.1 `ys_tabular_sym.json` 逐字段

```json
{
  "block_name": "tabular_ys",
  "slime": false,
  "equations": {
    "0": "这里是一大段公式字符串"
  }
}
```

- `equations["0"]`：只有一个，因为 YS 是标量。  
- 公式里的 `Nb`、`coldway_2` 等，对应 [根 README 第 3 章](../README.md#第-3-章输入数据长什么样) 的 30 个名字。

### 6.2 `metrics.json` 怎么判断好不好

```json
"teacher": {
  "val_mae_ys": 0.58,
  "val_mae_fs": 0.87
},
"tabular_symbolic": {
  "val_mae_ys": 0.59,
  "val_mae_fs": 0.88
},
"graph_info_loss_val_mae_ys": 0.01,
"graph_info_loss_val_mae_fs": 0.01
```

| 字段 | 意思 |
|------|------|
| `val_mae_ys` | 验证集上，预测 YS 和真值平均差多少（越小越好） |
| `teacher` | 黑盒 GNN 的误差（上限参考） |
| `tabular_symbolic` | **你的公式**的误差 |
| `graph_info_loss_*` | 公式误差 − 教师误差；**为正**表示去掉图后更差，说明构图有用 |

初学者：**tabular 和 teacher 在同一数量级（例如都在 0.5～1.0）就算不错**；`--quick` 仅作演示，正式结果要去掉 `--quick`。

---

## 7. 背后怎么实现的（简化版代码导读）

脚本文件：[`run_distill.py`](run_distill.py)

```python
# ① 加载数据和教师（和训练时同一个 GNN）
teacher = load_teacher(ckpt, ...)
t_ys, t_fs = teacher_forward(teacher, x, edge_index, edge_type)  # 这里用了图

# ② 只取训练节点的 x 和教师预测当「教材」
fn_ys = make_tabular_lookup_fn(x_np, ys_teacher_on_train)

# ③ SymTorch + PySR 搜公式
sym_ys = distill_block_on_numpy_io(fn_ys, x_np, variable_names=FEATURE_NAMES)

# ④ 推理：只用 x，不用 edge_index
tabular = TabularSymbolicModel(sym_ys, sym_fs)
s_ys, s_fs = tabular(x)
```

关键：**蒸馏时**教师用过图；**你以后用公式**时只代入 30 个数即可。

实现细节见 [`common/distill_io.py`](../common/distill_io.py)、[`common/hybrid_models.py`](../common/hybrid_models.py)。

---

## 8. 完整版怎么跑（写论文用）

```bash
/root/miniconda3/bin/python3.13 run_distill.py \
  --sr-niterations 400 \
  --device cuda
```

不加 `--quick`，PySR 更认真搜公式，时间更长。  
跑前确认磁盘空间充足（`SR_output` 会变大）。

---

## 9. 常见错误与解决

| 现象 | 可能原因 | 怎么办 |
|------|----------|--------|
| `No module named 'symtorch'` | 环境不对 | 用 Python 3.11+ 重装 requirements |
| 找不到 `material_graph.pt` | 没生成数据 | `gnnDir` 里跑 `regenerate_rgnnpt.py` |
| 找不到 `best_ysfs_gat.pt` | 没训练教师 | 跑 `train_fs_gat.py` |
| 一直装 Julia | 首次正常 | 等待完成 |
| 公式特别长 | PySR 复杂度过高 | 正常；可后期手工筛短公式 |
| `graph_info_loss` 很大 | 图很重要 | 说明应用 highExp，不要只信 lowExp |

---

## 10. 学完之后建议

- 想保留图、又要成分公式 → 去 [highExp](../highExp/README.md)  
- 想解释某一个难例 → 去 [sampleExp](../sampleExp/README.md)  
- 总索引 → [symbolTorch 入门教程](../README.md)
