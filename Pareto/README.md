# Pareto — 基因库累积遗传逆设计

基于冻结 `SingleEncoder_DualRGAT`，在 **604 节点背景图**上插入 **1 个虚拟设计节点**，以用户给定 **(YS, FS)** 为目标，用 **基因库累积 + 加权选父** 进化 30 维配方基因组。

## 算法流程

1. **代 0**：将全部 604 图节点以标签 YS/FS 写入基因库，按 `f1+f2+0.1*f3` 加权选 top 60 父本，**不 GNN forward**。
2. **代 1 起**：每代从全库加权 top 60 父本杂交产生 60 个子代，**仅子代**做 604+1 forward，子代入库。
3. **基因库规模**：604 原始 + 每代 +60 虚拟 → 第 k 代共 **604 + 60k** 条。
4. **回交**：父本可来自原始节点或任意历史虚拟个体。

## 快速开始

在 **metalForTi 根目录**、Python 3.10 主环境：

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"

python -m Pareto.run_ga_design --target-ys <float> --target-fs <float>
```

仅需 `--target-ys` 与 `--target-fs`；其余参数均有默认值（见下表）。

## 默认参数

| 参数 | 默认 |
|------|------|
| `--data-dir` | `gnnDir/gnndataPT/r-gatPT` |
| `--ckpt` | `gnnDir/gnn/r-gatDouble/runs/best_ysfs_gat.pt` |
| `--rgat-dir` | `gnnDir/gnn/r-gatDouble` |
| `--out-dir` | `Pareto/outputs_ga` |
| `--pop-size` | `60`（每代子代数 / 父本池规模） |
| `--generations` | `150` |
| `--objectives` | `three`（\|ΔYS\|、\|ΔFS\|、训练锚定 L2） |
| `--device` | `cuda`（不可用则 cpu） |

CPU 冒烟：`--force-cpu --pop-size 10 --generations 2`（约 20 次 forward）

完整 150 代约 **60×150 = 9000** 次 GNN forward（代 0 无 forward）。

## 输出

- `Pareto/outputs_ga/pareto_front.json` — 全库帕累托前沿 + 档案统计
- `Pareto/outputs_ga/ga_summary.txt` — 中文摘要
- `Pareto/outputs_ga/pareto_scatter.png` — f1–f2 散点图

## 基因组结构（30 维）

| 段 | 索引 | 说明 |
|----|------|------|
| element | 0–9 | Al…Fe（wt%），Ti = 100 − sum |
| testenv | 10–11 | tem、fcr（z-score） |
| coldway | 12–29 | 3 阶段 × (3 方式 × 2 数值)，存储为 3×6 展平 |

coldway 约束：每阶段 **3 选 1** 方式；阶段 **行累计**（无中间空行）；至少 **1 次** coldway（阶段 0 非空）。

## 遗传算子

- **父本选择**：全库 `f1+f2+0.1*f3` 加权 top 60
- **交叉**：仅在同基因座内（element / testenv / coldway 按阶段行）
- **变异**：±10%、归零、0→非零（从训练库采样），变异后 `compile_genome`

## 依赖

复用 [`grd`](../grd/)：`io_utils`、`feature_layout`、`masked_projector`。不修改 grd 反推核心。

## GPU 说明

RTX 50 系（sm_120）需 PyTorch cu128（≥2.7）；否则使用 `--force-cpu`。

## 自检

```bash
python -m Pareto.ga_compile
python -m Pareto.ga_operators
python -m Pareto.ga_archive
```
