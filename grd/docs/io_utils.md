# `io_utils.py` — 数据与模型 I/O

## 文件路径

`grd/io_utils.py`

## 作用

衔接 **gnnDir 导出的 PT 数据包** 与 **r-gatDouble 模型代码**，为反推提供：

- 节点特征与标签张量
- 合并后的同质图边
- 已加载权重的 `SingleEncoder_DualRGAT`

本模块**不参与优化**，只做 I/O 与路径配置。

## 函数说明

### `_ensure_rgat_double_on_path(gnn_rgat_dir)`

将 `gnnDir/gnn/r-gatDouble` 插入 `sys.path`，以便 `from model_gat_double import SingleEncoder_DualRGAT`。

### `load_graph_bundle(data_dir) -> (x, ys, fs, train_mask, val_mask)`

| 输出 | 形状 | 来源文件 |
|------|------|----------|
| `x` | (N, 30) | `material_graph.pt` → `graph["sample"].x` |
| `ys` | (N,) | `ys.pt` |
| `fs` | (N,) | `fs.pt` |
| `train_mask` | (N,) bool | `train_mask.pt` |
| `val_mask` | (N,) bool | `val_mask.pt` |

缺任一文件则 `FileNotFoundError`。

### `merge_hetero_edges(graph) -> (edge_index, edge_type)`

将 HeteroData 中三种关系边拼成 RGAT 所需格式：

| 关系名 | `edge_type` id | 构图语义（gnnDir） |
|--------|----------------|-------------------|
| `comp_sim` | 0 | 元素向量余弦相似 |
| `env_sim` | 1 | 试验环境相似 |
| `heat_sim` | 2 | 工艺 coldway 相似 |

返回 `edge_index` 形状 `(2, E)`，双向边已展开。

### `load_dual_rgat(ckpt_path, gnn_rgat_dir, device) -> (model, ckpt)`

1. 读取 `best_ysfs_gat.pt` 元数据（`in_dim`, `hidden_dim`, `gat_heads`, `edge_dim`）
2. 实例化 `SingleEncoder_DualRGAT` 并 `load_state_dict`
3. `.to(device).eval()`，反推时模型参数冻结

## 数据流位置

```text
material_graph.pt + ys/fs/mask
        ↓ load_graph_bundle
        x, ys, fs, masks
        ↓ merge_hetero_edges
        edge_index, edge_type
        ↓ load_dual_rgat
        model (frozen)
```

## 算法关联

- 图结构来自 **异质图相似度构图**（见 gnnDir `rgcn_dataloader.py`），与 R-GCN / 关系 GNN 文献中的 k-NN/阈值图一致思想。
- 模型为 **RGAT 双头回归**，见 [gnn_inverter.md](./gnn_inverter.md) 参考文献中的 GAT / R-GCN。

## 相关文档

- [feature_layout.md](./feature_layout.md) — `x` 的 30 维含义
- [run_inversion.md](./run_inversion.md) — 默认路径参数
