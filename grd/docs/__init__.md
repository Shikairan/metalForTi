# `__init__.py` — 包入口与导出

## 文件路径

`grd/__init__.py`

## 作用

将 `grd` 声明为可导入的 Python 包，并统一导出对外 API，避免调用方记忆子模块路径。

## 导出符号

| 名称 | 来源模块 | 用途 |
|------|----------|------|
| `GNNInverter`, `GNNInverterConfig`, `InversionResult` | `gnn_inverter` | 核心反推 |
| `MaskedCompositeProjector`, `FeatureSliceSpec` | `masked_projector` | 按维硬投影 |
| `FeatureBounds`, `build_projector`, `compute_ti_balance` | `feature_layout` | 特征布局与约束 |
| `INPUT_DIM`, `ELEMENT_DIM`, `ELEMENT_NAMES` 等 | `feature_layout` | 维段常量 |

## 典型用法

```python
from grd import GNNInverter, GNNInverterConfig, build_projector, compute_ti_balance
```

命令行入口不经过本文件，请使用：

```bash
python -m grd.run_inversion
```

## 依赖关系

```
__init__.py
  ├── feature_layout
  ├── gnn_inverter
  └── masked_projector
```

## 相关文档

- [io_utils.md](./io_utils.md) — 数据加载
- [run_inversion.md](./run_inversion.md) — CLI 完整流程
