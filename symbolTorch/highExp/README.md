# highExp — 高可解释（仅 Encoder 符号化）

## 模型结构

```
x (30维) ──► [符号 ys_encoder] ──► RGAT_YS ──► Head_YS ──► YS
         └──► [符号 fs_encoder] ──► RGAT_FS ──► Head_FS ──► FS
```

- **符号化**：`ys_encoder`、`fs_encoder`（MLP：30 → 64）
- **保持神经网络**：双路 `RGATConv`、LayerNorm、MLP head

公式可直接阅读为「成分 + 测试环境 + 冷加工」对隐向量的代数关系；图上的邻域聚合仍在 RGAT 黑盒中。

## 运行

```bash
cd metalForTi/symbolTorch/highExp
/root/miniconda3/bin/python3.13 run_distill.py --quick
# 完整：去掉 --quick（将对每个 encoder 的 64 个输出维各跑一次 PySR，很慢）
```

可选：`--device cpu`（蒸馏本身在 CPU；评估可用 GPU）

## 输出（`runs/`）

| 文件 | 含义 |
|------|------|
| `ys_encoder_sym.json` / `fs_encoder_sym.json` | 符号公式 |
| `ys_encoder_sym.pt` | 可续跑的 SymbolicModel（本地） |
| `hybrid_state.pt` | 混合模型 RGAT/Head 权重 |
| `hybrid_meta.pt` | 元数据（教师 ckpt 路径等） |
| `metrics.json` | `teacher` vs `hybrid` 的 train/val MAE |
| `summary.md` | 一行摘要 |

## 中断恢复

若蒸馏中断，保留 `runs/sr_cache/` 与已有 `*_encoder_sym.pt`，再次执行 `run_distill.py` 会从已有输出维继续。

若要重来：删除 `runs/sr_cache` 与 `*_encoder_sym.pt` 后重跑。
