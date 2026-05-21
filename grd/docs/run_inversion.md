# `run_inversion.py` — 命令行入口

## 文件路径

`grd/run_inversion.py`

## 作用

将 `grd` 各模块串联为**一键反推流水线**，适合实验与生产复现：

1. 解析命令行参数  
2. 加载数据与模型  
3. 构建分位数 box + `MaskedCompositeProjector`  
4. `GNNInverter.invert_multistart`  
5. 写 `x_inv.pt`、`inversion_summary.json`、`inversion_summary.txt`

## 主函数 `main()` 流程

```text
_parse_args()
  → load_graph_bundle + merge_hetero_edges
  → load_dual_rgat
  → bounds_from_train_x（或 bounds_with_physical_testenv）
  → build_projector(x, bounds, total_wt)
  → GNNInverter(..., regularizers=_build_regularizers, projector=...)
  → 选择 initializers（CPU 默认仅 training_mean）
  → 设置 opt_recon_mask（--recon-mask-mode）
  → invert_multistart(..., recon_mask=opt_recon_mask)
  → 计算分段 MAE、Ti 余量
  → torch.save(x_inv.pt)
  → build_summary_dict + write_summary_json/txt
```

## 辅助函数

| 函数 | 作用 |
|------|------|
| `_resolve_device` | cuda 不可用时回退 cpu |
| `_parse_args` | 全部 CLI 参数，默认路径指向 `gnnDir/...` |
| `_build_regularizers` | 平滑（仅边类型 0）+ 极小 L1 + 锚定 |

## 关键默认策略

| 项 | 默认值 | 含义 |
|----|--------|------|
| `target_mode` | `ground_truth` | 用真实 ys/fs 做还原验证 |
| `recon_mask_mode` | `none` | 重建损失全图；可改 `val` |
| `node_mask` | `val` | **仅报告**特征 MAE 的子集 |
| `projectors` in cfg | `[]` | 硬约束由外部 `MaskedCompositeProjector` 负责 |
| `lambda_nonneg/sum1` | 0 | 不用全局软 sum=1 惩罚 |
| `input_dim` | 30 | 显式维数，避免误推断 |

## 运行示例

```bash
# 仓库根目录
python -m grd.run_inversion

python -m grd.run_inversion \
  --recon-mask-mode val \
  --max-iters 1500 \
  --device cuda
```

## 模块依赖图

```text
run_inversion.py
  ├── io_utils
  ├── feature_layout
  ├── gnn_inverter
  └── summary_report
```

## 相关文档

- [../README.md](../README.md) — 总览与算法原理
- [summary_report.md](./summary_report.md) — 输出报告格式
- [io_utils.md](./io_utils.md) — 输入数据路径
