# lowExp 接口：输入 / 输出

## 1. 输入

### 1.1 数据目录 `--data-dir`

须包含（或图内嵌 masks）：

| 文件 | 形状 / 含义 |
|------|-------------|
| `material_graph.pt` | PyG HeteroData；`sample.x` 为 `(N, 30)` |
| `ys.pt` | `(N,)` YS 标签（模型量纲） |
| `fs.pt` | `(N,)` FS 标签（模型量纲） |
| `train_mask.pt` | `(N,)` bool |
| `val_mask.pt` | `(N,)` bool |

图边关系（教师前向需要）：`comp_sim` / `env_sim` / `heat_sim` → 合并为 `edge_index` + `edge_type`（0/1/2）。

默认：`gnnDir/gnndataPT/r-gatPT`。

### 1.2 教师权重 `--ckpt`

| 要求 | 说明 |
|------|------|
| 类 | `RGAT_Dual`（或 ckpt 元数据为 `SingleEncoder_DualRGAT` 且 state 兼容时按 Dual 加载） |
| 格式 | `{"model_state_dict": ..., "hidden_dim": ...}` 或纯 `state_dict` |
| 输入维 | 与 `x.shape[1]` 一致（通常 30） |
| `hidden_dim` | 须与 CLI `--hidden-dim` 一致（默认 64） |

默认：`modelAll/ysFs/runs/best_rgat_full.pt`。  
换用其它 Dual 头权重时显式传 `--ckpt /path/to.pt`。

> **注意**：`modelAll/utsFs` / `utsFsAll` 为 UTS+FS，第一头语义不是 YS。若强行蒸馏，公式学的是教师第一头，**不要**当 YS 解释。

### 1.3 特征布局（30 维）

与训练图一致；公式变量名见 `common/constants.py` 的 `FEATURE_NAMES`：

| 下标 | 名称示例 | 含义 |
|------|----------|------|
| 0–9 | `Al`…`Fe`（`Si` 写作 `el_Si`，避免与 SymPy 冲突） | 元素 wt% |
| 10–11 | `tem`, `fcr` | 试验环境（训练用 z-score 空间） |
| 12–29 | `coldway_0` … `coldway_17` | 工艺 18 维 |

### 1.4 CLI 参数（`add_common_args`）

| 参数 | 默认 | 含义 |
|------|------|------|
| `--data-dir` | 见上 | 图与标签目录 |
| `--ckpt` | 见上 | 教师 `.pt` |
| `--out-dir` | （空） | 若指定则使用该完整路径 |
| `--run-name` | 自动时间戳 | 写入 `lowExp/runs/<run-name>/`；默认 `YYYYMMDD_HHMMSS[_quick]_s{seed}[_itN]`，**不覆盖**历史 |
| `--hidden-dim` | `64` | 须匹配 ckpt |
| `--dropout` | `0.2` | 构建教师结构用 |
| `--sr-niterations` | `400` | PySR 迭代（`--quick` 时改为 40） |
| `--seed` | `42` | 随机种子 |
| `--include-val` | off | 蒸馏样本是否含验证节点 |
| `--quick` | off | 快速调试（少迭代） |
| `--device` | `auto` | `auto` / `cpu` / `cuda` |

---

## 2. 输出（默认写在 `lowExp/runs/<本次运行名>/`）

每次运行新建子目录，互不覆盖。`runs/latest` 符号链接指向最近一次。

也可：`--run-name my_exp` → `runs/my_exp/`；或 `--out-dir /任意路径`。

| 文件 | 内容 |
|------|------|
| `ys_tabular_sym.json` | YS 完整方程（含 `equation` = `YS = …`、变量表） |
| `fs_tabular_sym.json` | FS 完整方程 |
| `equations.md` | **人类可读主文件**：完整 YS/FS 方程 + 变量说明 + 30 维对照 |
| `ys_tabular_sym.pt` / `fs_tabular_sym.pt` | SymTorch 模块 |
| `teacher_predictions.pt` | 全图教师预测 |
| `metrics.json` | MAE，并含 `equations.YS/FS` 完整式 |
| `summary.md` | 摘要（内嵌完整方程） |

### 2.1 公式 JSON 结构

```json
{
  "block_name": "tabular_ys",
  "target": "YS",
  "equation": "YS = fcr * 0.14712471 + tem * (-0.21390598) + 0.9633571",
  "equation_rhs": "fcr * 0.14712471 + tem * (-0.21390598) + 0.9633571",
  "equation_raw": "fcr*0.14712471 + tem*(-0.21390598) + 0.9633571",
  "variables_used": ["fcr", "tem"],
  "variable_glossary": { "tem": "...", "fcr": "..." },
  "slime": false,
  "equations": { "0": "..." }
}
```

优先阅读 **`equation`（含 `YS =` / `FS =` 的完整式）**，或同目录 **`equations.md`**。
`slime` 为历史字段，可忽略。

### 2.2 `metrics.json` 字段

| 键 | 含义 |
|----|------|
| `experiment` | `"lowExp"` |
| `graph_at_inference` | `false`（公式推理无图） |
| `teacher.train/val_mae_ys/fs` | 教师相对**标签**的 MAE |
| `tabular_symbolic.train/val_mae_ys/fs` | 公式相对**标签**的 MAE |
| `graph_info_loss_val_mae_ys/fs` | `tabular_val − teacher_val`（正值 ≈ 去掉图信息后变差多少） |

量纲与 `ys.pt` / `fs.pt` 一致（模型量纲，不是 data1123 物理打印值）。

### 2.3 运行时副产物

- `runs/<run>/SR_output/<block_name>/`：PySR 工作目录（按次运行隔离；可删，已 gitignore）
- 旧的共享目录 `lowExp/SR_output/` 已弃用，可手动删除以免与历史结果混淆
- 环境变量：脚本默认 `TORCHDYNAMO_DISABLE=1`（无 g++ 时避免 torch.compile 失败）
