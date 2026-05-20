"""Shared paths and feature names for symbolTorch distillation."""

from __future__ import annotations

from pathlib import Path

SYMBOL_TORCH_ROOT = Path(__file__).resolve().parents[1]
METAL_FOR_TI_ROOT = SYMBOL_TORCH_ROOT.parent
GNN_DIR = METAL_FOR_TI_ROOT / "gnnDir"
RGAT_DOUBLE_DIR = GNN_DIR / "gnn" / "r-gatDouble"

DEFAULT_DATA_DIR = GNN_DIR / "gnndataPT" / "r-gatPT"
DEFAULT_CKPT = RGAT_DOUBLE_DIR / "runs" / "best_ysfs_gat.pt"

NUM_RELATIONS = 3
GAT_HEADS = 4
HIDDEN_DIM_DEFAULT = 64
IN_DIM = 30

# PySR/SymPy rejects bare "Si" (conflicts with sympy function names).
ELEMENT_COLS = ["Al", "Zr", "Sn", "Mo", "Cr", "Nb", "Si", "V", "Ta", "Fe"]
ELEMENT_FEATURE_NAMES = [f"el_{c}" if c == "Si" else c for c in ELEMENT_COLS]
TESTENV_COLS = ["tem", "fcr"]
# Raw coldway CSV columns (15); graph node x uses 18-d flattened 3x6 seq (coldway_0..17).
COLDWAY_RAW_COLS = [
    "T1", "t1", "T2", "t2", "T3", "t3",
    "C1_1", "C1_2", "C1_3", "C2_1", "C2_2", "C2_3", "C3_1", "C3_2", "C3_3",
]
COLDWAY_FLAT_COLS = [f"coldway_{i}" for i in range(18)]

FEATURE_NAMES = ELEMENT_FEATURE_NAMES + TESTENV_COLS + COLDWAY_FLAT_COLS


def head_feature_names(hidden_dim: int) -> list[str]:
    return [f"h{i}" for i in range(int(hidden_dim))]

DEFAULT_SR_PARAMS = {
    "niterations": 400,
    "binary_operators": ["+", "*", "-", "/"],
    "unary_operators": ["inv(x) = 1/x", "sin", "exp"],
    "extra_sympy_mappings": {"inv": lambda x: 1 / x},
    "complexity_of_operators": {"sin": 3, "exp": 3},
}

DEFAULT_SR_PARAMS_LOW = {
    **DEFAULT_SR_PARAMS,
    "unary_operators": ["inv(x) = 1/x", "sin", "exp", "log"],
}

DEFAULT_SLIME_PARAMS = {
    "J_nn": 10,
    "num_synthetic": 100,
    "real_weighting": 1.0,
    "nn_metric": "euclidean",
}

QUICK_SR_PARAMS = {"niterations": 40}
QUICK_MAX_OUTPUT_DIM = 4

PYTHON_BIN_HINT = "/root/miniconda3/bin/python3.13"
