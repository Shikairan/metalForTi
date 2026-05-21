"""梯度反推包 grd：GNN 输入特征联合反演。"""

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
