# `summary_report.py` — 反推结果汇总报告

## 文件路径

`grd/summary_report.py`

## 作用

反推结束后，将数值指标与元数据整理为：

1. **`inversion_summary.json`** — 机器可读，含 `field_descriptions` 中英文字段说明  
2. **`inversion_summary.txt`** — 人类可读，分章节 + 每项中文解释  

不负责优化，仅做**后处理与文档化**。

## 常量 `FIELD_DESCRIPTIONS_CN`

字典：JSON 键 → 中文含义。写入 `summary["field_descriptions"]`，并在 TXT 第八章索引列出。

## 函数说明

### `_fmt_float(v, nd=6)`

大数/小数用科学计数法，否则固定小数位，供 TXT 排版。

### `build_summary_dict(...) -> dict`

从一次反推运行收集：

| 字段组 | 内容 |
|--------|------|
| 运行元数据 | device、target_mode、total_wt、data_dir、ckpt |
| 优化结果 | converged、final_recon_mse、ys/fs MAE、n_iters、best_init |
| 特征 MAE | element / ti_balance / testenv_z / coldway |
| 统计量 | Ti 均值、元素行和均值/最大值 |
| bounds | testenv/coldway 投影界 |
| paths | 三个输出文件绝对路径 |

### `write_summary_json(path, summary)`

`json.dumps(..., ensure_ascii=False, indent=2)`。

### `write_summary_txt(path, summary, *, x_inv, x_true, ti_inv, ti_true, sample_node_indices)`

生成固定结构的 TXT：

1. 运行配置  
2. 优化与重建指标  
3. 分段 MAE  
4. Ti/元素统计  
5. 投影上下界  
6. 输出文件说明  
7. 样例节点对比（最多 3 个）  
8. JSON 字段中文索引  

## 调用位置

仅由 `run_inversion.main()` 在保存 `x_inv.pt` 之后调用。

## 相关文档

- [run_inversion.md](./run_inversion.md) — 何时生成报告
- [../README.md#输出文件](../README.md#输出文件) — 字段表
