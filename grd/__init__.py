"""
grd 包：基于已训练 GNN 的节点输入特征梯度反推。

子模块职责：
- gnn_inverter：优化循环、正则、投影、多初始点反推
- feature_layout：30 维特征布局与 Ti 余量（A 模式）约束配置
- masked_projector：按维段（element/testenv/coldway）硬投影
- io_utils：加载 material_graph、checkpoint、异质边
- run_inversion：命令行入口
- summary_report：生成带中文说明的 JSON/TXT 汇总报告
"""

from grd.feature_layout import (
    COLDWAY_DIM,
    DEFAULT_TOTAL_WT,
    ELEMENT_DIM,
    ELEMENT_NAMES,
    INPUT_DIM,
    TESTENV_DIM,
    FeatureBounds,
    build_projector,
    compute_ti_balance,
)
from grd.gnn_inverter import GNNInverter, GNNInverterConfig, InversionResult
from grd.masked_projector import FeatureSliceSpec, MaskedCompositeProjector

__all__ = [
    "GNNInverter",
    "GNNInverterConfig",
    "InversionResult",
    "MaskedCompositeProjector",
    "FeatureSliceSpec",
    "FeatureBounds",
    "build_projector",
    "compute_ti_balance",
    "DEFAULT_TOTAL_WT",
    "ELEMENT_NAMES",
    "INPUT_DIM",
    "ELEMENT_DIM",
    "TESTENV_DIM",
    "COLDWAY_DIM",
]
