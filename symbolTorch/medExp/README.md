# medExp 初学者教例 — encoder + 预测头都变成公式（中间保留图网络）

> **建议**：先跑通 [highExp](../highExp/README.md)，再用本教例。  
> 本档是四档里**综合度最高**、**概念最多**的一档，不适合作为第一次接触 symbolTorch 的入口。

---

## 0. 本教例结束后你会得到什么

- encoder 公式（和 highExp 相同或在本目录重新生成）
- **head 公式**：从 64 维隐变量 `h0…h63` 映射到 YS、FS 的式子
- 一个混合模型：只有 **中间 RGAT 仍是神经网络**，两头都是公式
- `metrics.json` 告诉你这种「三明治」结构离教师还差多少

---

## 1. 三明治结构（初学者必看图）

```text
        【公式 encoder】     ← 输入：Al, tem, coldway...
              ↓
        【神经网络 RGAT】    ← 黑盒：看相似配方、传消息（本教例不公式化）
              ↓
           h0…h63           ← 64 个中间数（初学者可当成「浓缩信息」）
              ↓
        【公式 head】        ← 输出：一个 YS 或 FS 数
```

和 [highExp](../highExp/README.md) 比：highExp 只公式化了最上面一块 encoder。  
和 [lowExp](../lowExp/README.md) 比：lowExp 不要中间的 RGAT。

---

## 2. 为什么要多此一举（好处 / 代价）

| 好处 | 说明 |
|------|------|
| 更接近「端到端」解释 | 从配方到 YS/FS 都有公式参与（中间夹一层图） |
| head 只有 1 个输出 | 每个 head 只跑 1 次 PySR，比 encoder 64 次省 |
| 可复用 highExp | `--encoder-sym-dir` 跳过重复的 encoder 蒸馏 |

| 代价 | 说明 |
|------|------|
| head 公式难读 | 变量是 h0、h1…，不是 Al |
| RGAT 仍不透明 | 写报告时要说明「图网络段未公式化」 |

---

## 3. 开始前检查清单

- [ ] highExp 的 `runs/ys_encoder_sym.pt` 已存在（推荐），或愿意在本目录从头蒸馏 encoder
- [ ] 环境、数据、教师 ckpt 同 [根 README](../README.md)
- [ ] 理解 `h0` 不是成分，是 RGAT 之后的抽象特征

---

## 4. 手把手运行（推荐命令）

```bash
cd /home/data/metalTi/metalForTi/symbolTorch/medExp

# 推荐：encoder 用 highExp 的结果，本脚本只蒸馏 head
/root/miniconda3/bin/python3.13 run_distill.py --quick \
  --encoder-sym-dir ../highExp/runs \
  --device cuda

# 查看
cat runs/summary.md
cat runs/ys_head_sym.json
cat runs/metrics.json
```

若还没跑 highExp，也可以（会更慢）：

```bash
/root/miniconda3/bin/python3.13 run_distill.py --quick --device cuda
```

---

## 5. 输入说明

| 输入 | 来源 | 形状/含义 |
|------|------|-----------|
| `x` | 图数据 | `(N, 30)` 配方特征 |
| `edge_index`, `edge_type` | 图数据 | 谁和谁相连 |
| 教师 ckpt | r-gatDouble | 提供 RGAT、head 的神经网络权重 |
| `h_ys`, `h_fs` | 脚本内部计算 | RGAT 之后、head 之前的向量，`(N, 64)` |
| 蒸馏 head 用的行 | `h[train_mask]` | 只拿训练集行去搜公式 |

### 专用参数

| 参数 | 含义 |
|------|------|
| `--encoder-sym-dir ../highExp/runs` | 到该目录找 `ys_encoder_sym.pt`、`fs_encoder_sym.pt` |

---

## 6. 输出说明

| 文件 | 内容 |
|------|------|
| `ys_head_sym.json` | YS 的 head 公式，变量 `h0…h63` |
| `fs_head_sym.json` | FS 的 head 公式 |
| `ys_encoder_sym.json` | 若本目录新跑了 encoder 才有 |
| `hybrid_state.pt` | MedExpHybrid 权重 |
| `metrics.json` | teacher vs hybrid |

### head 公式示例（示意）

```json
{
  "equations": {
    "0": "0.52 + h3 * (-0.01) + h7 * 0.02 + ..."
  }
}
```

初学者写报告时可以这样表述：

> 「在图网络输出的 64 维特征 h 上，YS 主要由 h3、h7 等组合决定；而 h 本身由成分–工艺公式（highExp）与 RGAT 共同得到。」

---

## 7. 实现流程（`run_distill.py` 在干什么）

```text
1. 加载教师
2. 加载或蒸馏 ys/fs encoder（先搜 --encoder-sym-dir，再搜本目录 out_dir）
3. collect_branch_hidden()：
      跑一遍教师，在 gat2+norm+gelu 后截取 h_ys, h_fs
4. distill_block(ys_head, h_ys_train)   → 1 个公式
5. distill_block(fs_head, h_fs_train)   → 1 个公式
6. MedExpHybrid = 符号 encoder + 符号 head + 教师里的 RGAT 权重
7. 评估并保存 metrics
```

代码位置：[`run_distill.py`](run_distill.py)、[`common/teacher.py`](../common/teacher.py) 的 `collect_branch_hidden`。

---

## 8. 怎么判断结果好不好

1. 看 `metrics.json` 里 `hybrid` 与 `teacher` 的 val MAE 是否接近。  
2. 看 `ys_head_sym.json` 是否过于冗长（可接受，PySR 会倾向复杂式）。  
3. 若 hybrid 很差但 highExp hybrid 很好 → 问题可能在 head 公式，可加大 `--sr-niterations`。

---

## 9. 常见错误

| 现象 | 处理 |
|------|------|
| 找不到 encoder 的 pt | 先跑 highExp，或去掉 `--encoder-sym-dir` |
| head 公式全是 h 看不懂 | 正常；配合 highExp 的 encoder 公式一起讲 |
| 比 highExp 还慢 | head 不慢；慢在 encoder——务必用 `--encoder-sym-dir` |

---

## 10. 下一步

- 要最简单总公式 → [lowExp](../lowExp/README.md)  
- 要单配方案例 → [sampleExp](../sampleExp/README.md)  
- 总教程 → [../README.md](../README.md)
