import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv


class FSGAT(nn.Module):
    """
    关系图注意力 RGAT（多关系）+ LayerNorm/GELU/dropout + 残差 + 加深回归头；FS 节点回归。
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

        self.conv1 = RGATConv(
            in_channels=in_dim,
            out_channels=hidden_dim // heads,
            num_relations=num_relations,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        # 第二层 heads=1，out_channels=hidden_dim，与 h1 同维才能 h1+h2（勿用 hidden_dim//第一层heads）
        self.conv2 = RGATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_relations=num_relations,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
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
