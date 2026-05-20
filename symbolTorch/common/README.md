# common 文件夹 — 初学者导读（公共工具箱）

> 你**一般不需要改**这里的代码。  
> 本 README 帮你理解：四个实验文件夹为什么能共用同一套逻辑。

---

## 1. 这个文件夹是干什么的？

四个教例（highExp、medExp、lowExp、sampleExp）都要做重复的事：

- 读 `material_graph.pt` 和标签  
- 加载 `best_ysfs_gat.pt` 教师  
- 调用 SymTorch 搜公式  
- 算误差、写 `metrics.json`  

这些重复代码都放在 **`common/`**，避免 copy-paste。

```text
highExp/run_distill.py  ──┐
medExp/run_distill.py   ──┼──→  import common.*
lowExp/run_distill.py   ──┤
sampleExp/run_distill.py──┘
```

---

## 2. 每个文件干什么（一句话版）

| 文件 | 初学者理解 |
|------|------------|
| `constants.py` | 默认路径、30 个特征名字、PySR 默认设置 |
| `data.py` | 读 .pt 数据、拼边 |
| `teacher.py` | 加载 GNN 教师、跑预测、抽中间层 |
| `distill_io.py` | 调用 SymTorch 搜公式、保存 json |
| `hybrid_models.py` | 把「公式块」和「神经网络块」拼成新模型 |
| `metrics.py` | 算 MAE、写误差 json |
| `cli.py` | 命令行参数 `--quick` 等 |

---

## 3. 数据是怎么读进来的？（`data.py`）

### 你需要提供的目录里要有：

```text
data-dir/
  material_graph.pt
  ys.pt
  fs.pt
  train_mask.pt    # 可选：也可嵌在 graph 里
  val_mask.pt
```

### 读完后得到什么

| 变量 | 形状 | 含义 |
|------|------|------|
| `x` | (N, 30) | 每条合金特征 |
| `ys`, `fs` | (N,) | 真值标签 |
| `train_mask` | (N,) | True=用来蒸馏 |
| `edge_index` | (2, E) | 边的起点终点 |
| `edge_type` | (E,) | 边类型 0/1/2 |

**初学者常问**：N 是多少？—— 你的 `datagnn.csv` 有多少行，一般就是多少节点。

---

## 4. 教师模型（`teacher.py`）

```python
teacher = load_teacher("best_ysfs_gat.pt", in_dim=30, hidden_dim=64, device=...)
ys, fs = teacher_forward(teacher, x, edge_index, edge_type)
```

- 结构和 [`model_gat_double.py`](../../gnnDir/gnn/r-gatDouble/model_gat_double.py) 一致。  
- `collect_branch_hidden`：medExp 用来取 RGAT 后面、head 前面的 64 维向量。

---

## 5. 搜公式（`distill_io.py`）— 核心

### 两种蒸馏方式

| 函数 | 用在 | 通俗理解 |
|------|------|----------|
| `distill_block` | highExp、medExp 的神经网络层 | 给一层网络很多输入，看它输出什么，再搜公式模仿 |
| `distill_block_on_numpy_io` | lowExp、sampleExp | 给任意「输入→输出」规则（含黑盒教师）搜公式 |

### 保存了什么

- **一定生成** `xxx.json`：人类可读的公式。  
- **尽量生成** `xxx.pt`：下次可 `load` 继续（失败时只留 json 也行）。

### `--quick` 在这里的意思

`constants.py` 里：

```python
QUICK_SR_PARAMS = {"niterations": 40}
QUICK_MAX_OUTPUT_DIM = 4
```

即：少搜几代、encoder 只搜 4 个输出维。

---

## 6. 混合模型（`hybrid_models.py`）

把 SymTorch 得到的「公式模块」嵌回流水线：

| 类 | 用于教例 | 结构 |
|----|----------|------|
| `HighExpHybrid` | highExp | 公式 encoder + 神经 RGAT/head |
| `MedExpHybrid` | medExp | 公式 encoder + 神经 RGAT + 公式 head |
| `TabularSymbolicModel` | lowExp | 公式 x→YS，公式 x→FS |

**符号部分在 CPU 算**，结果再送回 GPU 给 RGAT 用（避免显卡报错）。初学者只需知道：**不影响你读 json 公式**。

---

## 7. 误差怎么算（`metrics.py`）

```python
evaluate_predictions(pred_ys, pred_fs, ys, fs, train_mask, val_mask)
```

返回 train/val 上的平均绝对误差 MAE。

FS 的「相对误差 %」在 sampleExp 选难例时会用到（`rel_pct_fs`）。

---

## 8. 命令行（`cli.py`）

所有 `run_distill.py` 共享：

- `--data-dir`、`--ckpt`、`--out-dir`  
- `--quick`、`--device`、`--seed`  
- `--sr-niterations`、`--include-val`  

所以四个教例**参数名字一样**，换目录跑即可。

---

## 9. 初学者需要改代码吗？

| 需求 | 建议 |
|------|------|
| 换数据路径 | 用命令行 `--data-dir`，别改代码 |
| 换教师 | `--ckpt` |
| 改特征名 | 改 `constants.py` 的 `FEATURE_NAMES`（高级） |
| 改 PySR 运算符 | 改 `constants.py` 的 `DEFAULT_SR_PARAMS`（高级） |

---

## 10. 回到教例

- [总入门教程](../README.md)  
- [lowExp 教例](../lowExp/README.md) — 建议第一个跑  
- [highExp](../highExp/README.md) / [medExp](../medExp/README.md) / [sampleExp](../sampleExp/README.md)
