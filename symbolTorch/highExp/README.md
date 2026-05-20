# highExp 初学者教例 — 把「配方→中间层」变成公式（图网络保留）

> 适合已经跑过 [lowExp](../lowExp/README.md)、想进一步理解「GNN 里面第一层在干什么」的初学者。

---

## 0. 本教例结束后你会得到什么

- `ys_encoder_sym.json`、`fs_encoder_sym.json`：用 **Al、tem、coldway** 等写成的公式（每个隐层维度可能有一条，完整版很多）
- `metrics.json`：对比「只把 encoder 换成公式、后面仍用神经网络」时，预测还准不准
- 理解：**图网络（RGAT）仍然在起作用**，不是 lowExp 那种单公式走天下

---

## 1. 用图理解「教师」和「highExp」

教师模型 [`RGAT_Dual`](../../gnnDir/gnn/r-gatDouble/model_gat_double.py) 可以想象成四条流水线：

```text
配方 x (30个数)
    ↓  【ys_encoder】 小神经网络  →  64 维向量
    ↓  【ys_gat1 / ys_gat2】 图网络（看相似配方）  →  还是 64 维
    ↓  【ys_head】 小神经网络  →  一个数 YS

FS 分支完全再来一套（fs_encoder → fs_gat → fs_head）
```

**highExp 只把方框【encoder】换成公式**，后面【图网络 + head】仍是原来的神经网络权重。

```text
配方 x
    ↓  【公式 encoder】  ← 本教例蒸馏这里
    ↓  【RGAT + head】   ← 仍是神经网络
    →  YS / FS
```

---

## 2. 为什么值得做（初学者版好处）

| 好处 | 解释 |
|------|------|
| 公式好读 | 变量是成分、温度、工艺，不是抽象的 h0 |
| 保留「相似配方互影响」 | RGAT 还在，比 lowExp 更贴真实预测流程 |
| 可写进论文 | 「成分–工艺对隐表示的非线性映射」 |
| 给 medExp 省时间 | medExp 可直接复用本目录的 encoder 公式 |

| 代价 | 解释 |
|------|------|
| 很慢 | encoder 输出 64 个数，完整版要对每个数搜公式（×2 分支） |
| 公式很多 | 不是一个式子，而是 64 组（quick 模式只有 4 组） |

---

## 3. 开始前检查清单

- [ ] 已完成 [根 README 第 4～5 章](../README.md#第-4-章从零开始--环境安装手把手)
- [ ] 建议先跑通 [lowExp](../lowExp/README.md)（熟悉流程）
- [ ] 磁盘预留充足（`runs/sr_cache/` 可能很大）
- [ ] 第一次务必使用 `--quick`

---

## 4. 手把手运行

```bash
cd /home/data/metalTi/metalForTi/symbolTorch/highExp

# 第一次（quick：每个 encoder 只蒸馏 4 个输出维）
/root/miniconda3/bin/python3.13 run_distill.py --quick --device cuda

# 看结果
cat runs/summary.md
cat runs/ys_encoder_sym.json
cat runs/metrics.json
```

### 跑很久 / 中断了怎么办？

highExp 最容易因为 **PySR 太慢** 被中断。目录里可能已有 `runs/sr_cache/ys_encoder/.../dim0..dimN`。

- **续跑**：保留 `runs/ys_encoder_sym.pt`（若有），再次执行同一命令，会从下一维继续。  
- **重来（quick）**：删除 `runs/sr_cache/` 和 `runs/*_encoder_sym.pt` 后加 `--quick` 重跑。  
- **不要混用**：曾用「非 quick」跑到 dim25，又用 `--quick` 期望只有 4 维——需删缓存重来，否则维度不一致。

---

## 5. 输入 / 输出（详细）

### 5.1 输入

与 lowExp 相同的数据和教师，额外逻辑：

| 项目 | 说明 |
|------|------|
| 蒸馏样本 | 默认 `train_mask=True` 的节点；`--include-val` 可加入验证集 |
| 蒸馏对象 | `teacher.ys_encoder` 和 `teacher.fs_encoder` 两个 `nn.Sequential` |
| 每个 encoder 输入形状 | `(样本数, 30)` |
| 每个 encoder 输出形状 | `(样本数, 64)` |

### 5.2 输出文件

| 文件 | 初学者要不要打开 |
|------|------------------|
| `ys_encoder_sym.json` | **要** — YS 分支 encoder 公式 |
| `fs_encoder_sym.json` | **要** — FS 分支 |
| `metrics.json` | **要** — teacher vs hybrid |
| `hybrid_state.pt` | 暂不用 — 混合模型权重 |
| `hybrid_meta.pt` | 记录用了哪个 ckpt |
| `sr_cache/` | 可删 — PySR 缓存 |

### 5.3 `ys_encoder_sym.json` 长什么样

```json
{
  "block_name": "ys_encoder",
  "equations": {
    "0": "关于 Al, tem, ... 的公式",
    "1": "第 2 个隐层维的公式",
    "2": "...",
    "3": "..."
  }
}
```

- **quick 模式**：通常只有 `"0"`～`"3"` 四个键。  
- **完整模式**：应有 `"0"`～`"63"` 共 64 个键（每个都要跑 PySR，极慢）。

初学者：**不必读懂每一条**，先看 `"0"` 是否含有合理的成分项即可。

### 5.4 `metrics.json` 怎么读

```json
"teacher": { "val_mae_ys": ..., "val_mae_fs": ... },
"hybrid":  { "val_mae_ys": ..., "val_mae_fs": ... }
```

- `teacher`：原始 GNN。  
- `hybrid`：encoder 用公式、RGAT+head 用神经网络。  
- **hybrid 接近 teacher** → encoder 公式化成功，可以写报告。  
- **hybrid 差很多** → 需要更多 `sr-niterations`、或完整 64 维、或检查是否只跑了 quick 的 4 维。

---

## 6. 实现步骤（对应 `run_distill.py`）

```text
步骤 1  加载 graph、ys、fs、mask
步骤 2  加载教师 RGAT_Dual
步骤 3  取 x_train = x[train_mask]
步骤 4  distill_block(teacher.ys_encoder, x_train)  → sym_ys_enc
步骤 5  distill_block(teacher.fs_encoder, x_train)  → sym_fs_enc
步骤 6  组装 HighExpHybrid(教师, sym_ys_enc, sym_fs_enc)
步骤 7  全图预测：对比 teacher 与 hybrid
步骤 8  写 metrics.json、summary.md
```

**为什么不把整个 GNN 一次公式化？**  
因为中间 RGAT 要查「邻居节点」，不是 30 个数能表达的，SymTorch 无法简单 hook。

**为什么 encoder 在 CPU 上蒸馏？**  
PySR 用 NumPy/CPU；代码里 `block.cpu()` 避免设备不一致报错。

---

## 7. 和 medExp 怎么配合

medExp 可以**不重复**跑 encoder：

```bash
cd ../medExp
/root/miniconda3/bin/python3.13 run_distill.py --quick \
  --encoder-sym-dir ../highExp/runs
```

这样 medExp 只新跑 `head` 的公式，省时。

---

## 8. 常见初学者问题

**Q：公式里是 coldway_0 不是 T1？**  
冷加工在图数据里是 18 维展平后的名字，和 CSV 列 T1、t1 对应关系见 `gnnDir/build_datagnn.py`。

**Q：FS 和 YS 为何两套 encoder？**  
教师设计就是两条独立分支，强度机理不同，分开公式化更合理。

**Q：能用 GPU 吗？**  
能。`--device cuda` 加速教师和 RGAT；搜公式仍在 CPU。见 [根 README 第 10 章](../README.md#第-10-章gpu-还是-cpu)。

---

## 9. 下一步

- 补全 head 公式 → [medExp/README.md](../medExp/README.md)  
- 回到最简单的总公式 → [lowExp/README.md](../lowExp/README.md)  
- 总教程 → [../README.md](../README.md)
