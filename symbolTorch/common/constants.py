"""Shared paths and feature names for symbolTorch distillation."""

from __future__ import annotations

from pathlib import Path

SYMBOL_TORCH_ROOT = Path(__file__).resolve().parents[1]
METAL_FOR_TI_ROOT = SYMBOL_TORCH_ROOT.parent
GNN_DIR = METAL_FOR_TI_ROOT / "gnnDir"
MODEL_ALL_DIR = METAL_FOR_TI_ROOT / "modelAll" / "ysFs"
RGAT_DOUBLE_DIR = GNN_DIR / "gnn" / "r-gatDouble"

DEFAULT_DATA_DIR = GNN_DIR / "gnndataPT" / "r-gatPT"
DEFAULT_CKPT = MODEL_ALL_DIR / "runs" / "best_rgat_full.pt"
NUM_RELATIONS = 3
GAT_HEADS = 4
HIDDEN_DIM_DEFAULT = 64

# PySR/SymPy rejects bare "Si" (conflicts with sympy function names).
ELEMENT_COLS = ["Al", "Zr", "Sn", "Mo", "Cr", "Nb", "Si", "V", "Ta", "Fe"]
ELEMENT_FEATURE_NAMES = [f"el_{c}" if c == "Si" else c for c in ELEMENT_COLS]
TESTENV_COLS = ["tem", "fcr"]
COLDWAY_FLAT_COLS = [f"coldway_{i}" for i in range(18)]

FEATURE_NAMES = ELEMENT_FEATURE_NAMES + TESTENV_COLS + COLDWAY_FLAT_COLS

DEFAULT_SR_PARAMS = {
    "niterations": 400,
    "binary_operators": ["+", "*", "-", "/"],
    "unary_operators": ["inv(x) = 1/x", "sin", "exp", "log"],
    "extra_sympy_mappings": {"inv": lambda x: 1 / x},
    "complexity_of_operators": {"sin": 3, "exp": 3},
}

QUICK_SR_PARAMS = {"niterations": 40}

PYTHON_BIN_HINT = "/root/miniconda3/bin/python3.13"
