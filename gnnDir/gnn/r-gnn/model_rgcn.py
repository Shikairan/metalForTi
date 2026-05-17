from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class FSRGCN(nn.Module):
    """R-GCN for FS-only node regression."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_relations: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations=num_relations, num_bases=3)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations, num_bases=3)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        h1 = self.conv1(x, edge_index, edge_type)
        h1 = self.norm1(h1)
        h1 = F.gelu(h1)
        h1 = self.dropout(h1)

        h2 = self.conv2(h1, edge_index, edge_type)
        h2 = self.norm2(h2)
        h2 = F.gelu(h2)
        h2 = self.dropout(h2)

        h = h1 + h2
        out = self.head(h).squeeze(-1)
        return out

