# Pareto — 双池父本 + 基因库累积遗传逆设计

基于冻结 `SingleEncoder_DualRGAT`，在 **604 节点背景图**上插入 **1 个虚拟设计节点**，以用户给定 **(YS, FS)** 为目标进化 30 维配方基因组。

## 算法流程

1. **原始池（固定）**：604 图节点一次性入库，**永不增删**，始终参与父本配对。
2. **虚拟精英池（轮换）**：每代从**全部历史虚拟**中按 `f1+f2+0.1*f3` 取加权 **top 604**，与原始池合并为父本池。
3. **杂交**：父本池内**均匀随机配对**（原始 × 原始、原始 × 虚拟、虚拟 × 虚拟均可），每代产出 **604** 个子代。
4. **评估入库**：604 子代各做 604+1 GNN forward，写入基因库；全库规模 **604 + 604×代**。

## 快速开始

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python -m Pareto.run_ga_design --target-ys <float> --target-fs <float>
```

## 默认参数

| 参数 | 默认 |
|------|------|
| `--pop-size` | `604`（每代杂交子代数） |
| `--virtual-pool-size` | `604`（虚拟精英池 top-k） |
| `--generations` | `150` |
| `--objectives` | `three` |

父本池规模：代 0 仅 604 原始；代 k（k≥1 且虚拟足够）最多 **604 + 604 = 1208**。

CPU 冒烟：`--force-cpu --pop-size 10 --virtual-pool-size 10 --generations 2`

完整 150 代约 **604×150 = 90600** 次 GNN forward（仅子代）。

## 输出

- `Pareto/outputs_ga/pareto_front.json`
- `Pareto/outputs_ga/ga_summary.txt`
- `Pareto/outputs_ga/pareto_scatter.png`

每代日志展示：**全库最优** + **历史虚拟最优**；帕累托前沿统计基于**当代 604 子代批次**。

## 遗传算子

- **父本池**：604 原始（固定）+ 历史虚拟 top 604（轮换）
- **配对**：父本池内均匀随机
- **交叉 / 变异**：分段交叉 + `compile_genome`

## 自检

```bash
python -m Pareto.ga_archive
```
