# medExp — 中等可解释（Encoder + Head 符号化）

## 模型结构

```
x ──► [符号 encoder] ──► [神经网络 RGAT] ──► h ──► [符号 head] ──► YS/FS
```

- **符号化**：encoder + `ys_head` / `fs_head`
- **神经网络**：仅中间双路 RGAT（消息传递 + 注意力）

Head 的输入变量在 PySR 中记为 `h0 … h63`（隐层维），而非原始成分名。

## 运行

```bash
cd metalForTi/symbolTorch/medExp

# 推荐：复用 highExp 已蒸馏的 encoder，只新跑 head
/root/miniconda3/bin/python3.13 run_distill.py --quick \
  --encoder-sym-dir ../highExp/runs

# 独立全流程（encoder + head 都在本目录 runs/ 下蒸馏）
/root/miniconda3/bin/python3.13 run_distill.py --quick
```

## 输出（`runs/`）

| 文件 | 含义 |
|------|------|
| `ys_head_sym.json` / `fs_head_sym.json` | Head 符号公式（变量 h0…hN） |
| `ys_encoder_sym.json` | 若未指定 `--encoder-sym-dir`，也会在本目录生成 |
| `hybrid_state.pt` | 混合模型权重 |
| `metrics.json` | Teacher vs Hybrid MAE |

## 与 highExp 的关系

`--encoder-sym-dir` 指向 [highExp/runs](../highExp/runs) 时，会优先加载其中的 `ys_encoder_sym.pt` / `fs_encoder_sym.pt`，避免重复 PySR。
