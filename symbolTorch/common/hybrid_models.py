"""Hybrid neural-symbolic RGAT models."""

from __future__ import annotations

import torch
import torch.nn as nn


def _sym_forward(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """SymbolicModel on CPU; clone to avoid CUDAGraph overwrite on repeated calls."""
    dev = x.device
    x_cpu = x.detach().cpu().float()
    out = module(x_cpu)
    if isinstance(out, torch.Tensor):
        return out.to(dev).clone()
    return torch.as_tensor(out, device=dev, dtype=x.dtype)


class TabularSymbolicModel(nn.Module):
    """Pure tabular YS/FS (no graph)."""

    def __init__(self, sym_ys, sym_fs) -> None:
        super().__init__()
        self.sym_ys = sym_ys
        self.sym_fs = sym_fs

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ys = _sym_forward(self.sym_ys, x).reshape(-1)
        fs = _sym_forward(self.sym_fs, x).reshape(-1)
        return ys, fs


