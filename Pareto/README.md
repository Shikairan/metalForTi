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
python -m Pareto.run_ga_design --target-ys <float> --target-fs <float>
```

## 默认参数

| 参数 | 默认 |
|------|------|
| `--pop-size` | `604`（种群 / 每代子代数） |
| `--generations` | `150` |
| `--objectives` | `three`（f1=\|ΔYS\|, f2=\|ΔFS\|, f3=锚定 L2） |

CPU 冒烟：`--force-cpu --pop-size 10 --generations 2`

完整 150 代约 **604×150 = 90600** 次 GNN forward（仅子代）。

## 输出

- `Pareto/outputs_ga/pareto_front.json` — 最终种群第一非支配层
- `Pareto/outputs_ga/ga_summary.txt`
- `Pareto/outputs_ga/pareto_scatter.png`

每代日志展示：**当前种群帕累托代表** + **前沿个体数**。

## 自检

```bash
python -m Pareto.ga_archive
python -m Pareto.test_archive_smoke
```
