"""Dual-head model: shared encoder + independent YS/FS RGAT stacks (see model_gat_double.py)."""

from model_gat_double import SingleEncoder_DualRGAT

RGAT_Dual = SingleEncoder_DualRGAT

__all__ = ["RGAT_Dual", "SingleEncoder_DualRGAT"]
