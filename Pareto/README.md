# Pareto — NSGA-II 帕累托遗传逆设计

基于冻结 `SingleEncoder_DualRGAT`，在 **604 节点背景图**上插入 **1 个虚拟设计节点**，以用户给定 **(YS, FS)** 为目标进化 30 维配方基因组。

## 算法流程（标准 NSGA-II）

1. **原始池（档案）**：604 图节点一次性入库，作为历史档案与第 1 代父本来源（标签适应度，无 GNN forward）。
2. **父本选择**：二元锦标赛——优先非支配 rank 更小，同层取拥挤距离更大。
3. **子代**：交叉 + `compile_genome` 变异，每代 **pop_size** 条，各做 GNN forward。
4. **环境选择**：`父代种群 ∪ 子代` → 非支配排序 + 拥挤距离截断，保留 **pop_size** 条构成下一代种群。
5. **档案累积**：全部评估过的虚拟子代写入基因库（604 + 604×代）。

## 快速开始

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# 输入物理量 YS/FS（MPa 等，预处理前数值，默认与 data1123.csv 同量纲）
python -m Pareto.run_ga_design --target-ys 1014.8 --target-fs 0.147

# 若 target 已与 ys.pt/fs.pt 同量纲（均值归一化后）
python -m Pareto.run_ga_design --target-ys 1.05 --target-fs 0.52 --no-targets-physical
```

**量纲说明**（输入与输出均对齐 **data1123.csv**）：

1. **CLI 输入**：与 `data1123.csv` 列刻度一致
   - `--target-ys`：`YS` 列（MPa）
   - `--target-fs`：`FS` 列（直接读表内数值，如 `0.147`；**勿**使用 dataOri2 的 `14.7`/`20`）
2. **内部 Pareto**：自动换算为 `ys.pt` / `fs.pt` 模型量纲后优化
3. **日志 / JSON 输出**：逆变换为 data1123 物理量

```bash
python -m Pareto.run_ga_design --target-ys 1201 --target-fs 0.2
```

若手头为 dataOri2 的 FS（如 `20`），请先 ÷100 得到 data1123 值（`0.2`）再传入。

- 公式见 [`preprocess/PREPROCESS_datagnn.md`](../preprocess/PREPROCESS_datagnn.md)

## 默认参数

| 参数 | 默认 |
|------|------|
| `--pop-size` | `604`（种群 / 每代子代数） |
| `--generations` | `150` |
| `--objectives` | `three`（f1=\|ΔYS\|, f2=\|ΔFS\|, f3=锚定 L2） |

CPU 冒烟：`--force-cpu --pop-size 10 --generations 2`

完整 150 代约 **604×150 = 90600** 次 GNN forward（仅子代）。

## 输出

- `Pareto/outputs_ga/pareto_front.json` — 最终种群第一非支配层（**数值已还原为 data1123 物理量纲**）
- `Pareto/outputs_ga/ga_summary.txt` — 文本摘要（同上）
- `Pareto/outputs_ga/pareto_scatter.png` — f1/f2 散点（物理量纲误差）

### 输出字段（物理量纲，30 维结构不变）

| 字段 | 含义 |
|------|------|
| `genome_30d` | 30 维基因组：元素 10 + 测试条件 2 + 热处理 18，**仅数值还原** |
| `testenv.tem` / `testenv.sr` | 测试温度（℃）、应变速率（与 data1123 的 `sr` 一致） |
| `ys_pred` / `fs_pred` | GNN 预测 YS/FS（**data1123 量纲**） |
| `f1_ys_abs_err` / `f2_fs_abs_err` | 物理量纲下的 \|预测 − 目标\| |

内部 NSGA-II 仍在**模型量纲**（`ys.pt`/`fs.pt` 刻度）下以 f1/f2/f3 优化；日志与 JSON 中的 data1123 数值及「展示误差」仅便于阅读，**不参与**非支配排序与环境选择。

每代日志会同时打印：
- **误差（展示）**：物理量纲下的 |ΔYS|/|ΔFS|
- **f1/f2/f3（NSGA-II 优化）**：模型量纲，与实际帕累托排序一致

每代日志中 coldway 各阶段**统一显示物理量 `T`、`t`**（不再对方式 2/3 误标为 C_a/C_b），数值固定 4 位小数。

每代日志展示：**当前种群帕累托代表** + **前沿个体数**。

## 自检

```bash
python -m Pareto.ga_archive
python -m Pareto.test_archive_smoke
python -m Pareto.test_pareto_output_restore
python -m Pareto.test_pareto_coldway_display
python -m Pareto.test_pareto_label_means
python -m Pareto.test_pareto_target_scale
```
