"""Hybrid neural-symbolic RGAT models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HighExpHybrid(nn.Module):
    """Symbolic encoders + neural RGAT + neural heads."""

    def __init__(
        self,
        teacher: nn.Module,
        sym_ys_encoder: nn.Module,
        sym_fs_encoder: nn.Module,
    ) -> None:
        super().__init__()
        self.sym_ys_encoder = sym_ys_encoder
        self.sym_fs_encoder = sym_fs_encoder
        self.ys_gat1 = teacher.ys_gat1
        self.ys_gat2 = teacher.ys_gat2
        self.ys_norm1 = teacher.ys_norm1
        self.ys_norm2 = teacher.ys_norm2
        self.ys_head = teacher.ys_head
        self.fs_gat1 = teacher.fs_gat1
        self.fs_gat2 = teacher.fs_gat2
        self.fs_norm1 = teacher.fs_norm1
        self.fs_norm2 = teacher.fs_norm2
        self.fs_head = teacher.fs_head
        self.dropout = teacher.dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat_ys = _sym_forward(self.sym_ys_encoder, x)
        feat_fs = _sym_forward(self.sym_fs_encoder, x)

        h = self.ys_gat1(feat_ys, edge_index, edge_type)
        h = self.dropout(F.gelu(self.ys_norm1(h)))
        h = self.ys_gat2(h, edge_index, edge_type)
        h = self.dropout(F.gelu(self.ys_norm2(h)))
        ys = self.ys_head(h).squeeze(-1)

        h = self.fs_gat1(feat_fs, edge_index, edge_type)
        h = self.dropout(F.gelu(self.fs_norm1(h)))
        h = self.fs_gat2(h, edge_index, edge_type)
        h = self.dropout(F.gelu(self.fs_norm2(h)))
        fs = self.fs_head(h).squeeze(-1)

        return ys, fs


class MedExpHybrid(nn.Module):
    """Symbolic encoders + neural RGAT + symbolic heads."""

    def __init__(
        self,
        teacher: nn.Module,
        sym_ys_encoder: nn.Module,
        sym_fs_encoder: nn.Module,
        sym_ys_head: nn.Module,
        sym_fs_head: nn.Module,
    ) -> None:
        super().__init__()
        self.sym_ys_encoder = sym_ys_encoder
        self.sym_fs_encoder = sym_fs_encoder
        self.ys_gat1 = teacher.ys_gat1
        self.ys_gat2 = teacher.ys_gat2
        self.ys_norm1 = teacher.ys_norm1
        self.ys_norm2 = teacher.ys_norm2
        self.fs_gat1 = teacher.fs_gat1
        self.fs_gat2 = teacher.fs_gat2
        self.fs_norm1 = teacher.fs_norm1
        self.fs_norm2 = teacher.fs_norm2
        self.sym_ys_head = sym_ys_head
        self.sym_fs_head = sym_fs_head
        self.dropout = teacher.dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat_ys = _sym_forward(self.sym_ys_encoder, x)
        feat_fs = _sym_forward(self.sym_fs_encoder, x)

        h = self.ys_gat1(feat_ys, edge_index, edge_type)
        h = self.dropout(F.gelu(self.ys_norm1(h)))
        h = self.ys_gat2(h, edge_index, edge_type)
        h = self.dropout(F.gelu(self.ys_norm2(h)))
        ys = _sym_forward(self.sym_ys_head, h).reshape(-1)

        h = self.fs_gat1(feat_fs, edge_index, edge_type)
        h = self.dropout(F.gelu(self.fs_norm1(h)))
        h = self.fs_gat2(h, edge_index, edge_type)
        h = self.dropout(F.gelu(self.fs_norm2(h)))
        fs = _sym_forward(self.sym_fs_head, h).reshape(-1)

        return ys, fs


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
