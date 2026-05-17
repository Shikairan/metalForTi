from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv


class SingleEncoder_DualRGAT(nn.Module):
    """
    共享 Encoder → 两条独立 RGAT（YS / FS），各含双 RGATConv + Norm + MLP head。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_relations: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")
        self.hidden_dim = hidden_dim

        self.fs_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.ys_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.ys_gat1 = RGATConv(
            hidden_dim,
            hidden_dim // heads,
            num_relations,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        self.ys_gat2 = RGATConv(
            hidden_dim,
            hidden_dim,
            num_relations,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.ys_norm1 = nn.LayerNorm(hidden_dim)
        self.ys_norm2 = nn.LayerNorm(hidden_dim)
        self.ys_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.fs_gat1 = RGATConv(
            hidden_dim,
            hidden_dim // heads,
            num_relations,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        self.fs_gat2 = RGATConv(
            hidden_dim,
            hidden_dim,
            num_relations,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.fs_norm1 = nn.LayerNorm(hidden_dim)
        self.fs_norm2 = nn.LayerNorm(hidden_dim)
        self.fs_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat_ys = self.ys_encoder(x)
        feat_fs = self.fs_encoder(x)

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
