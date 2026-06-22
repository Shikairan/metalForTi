"""Load RGAT_Dual teacher checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .constants import GAT_HEADS, MODEL_ALL_DIR, NUM_RELATIONS, RGAT_DOUBLE_DIR


def _ensure_rgat_import() -> None:
    for d in (MODEL_ALL_DIR, RGAT_DOUBLE_DIR):
        p = str(d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_rgat_dual(in_dim: int, hidden_dim: int, num_relations: int, heads: int, dropout: float, device):
    _ensure_rgat_import()
    try:
        from model_rgat import RGAT_Dual  # noqa: WPS433
    except ImportError:
        from model_gat import RGAT_Dual  # noqa: WPS433
    return RGAT_Dual(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_relations=num_relations,
        heads=heads,
        dropout=dropout,
    ).to(device)


def load_teacher(
    ckpt_path: Path,
    *,
    in_dim: int,
    hidden_dim: int,
    device: torch.device,
    num_relations: int = NUM_RELATIONS,
    heads: int = GAT_HEADS,
    dropout: float = 0.2,
) -> torch.nn.Module:
    if not ckpt_path.is_file():
        hint = (
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train first:\n"
            f"  cd {MODEL_ALL_DIR}\n"
            "  python build_data.py && python train.py\n"
        )
        raise FileNotFoundError(hint)

    model = _load_rgat_dual(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_relations=num_relations,
        heads=heads,
        dropout=dropout,
        device=device,
    )

    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        meta = blob.get("model") or blob.get("model_class")
        if meta and str(meta) not in ("RGAT_Dual", "SingleEncoder_DualRGAT"):
            raise ValueError(f"Unexpected checkpoint model tag: {meta}")
        model.load_state_dict(blob["model_state_dict"], strict=True)
        ckpt_hidden = blob.get("hidden_dim")
        if ckpt_hidden is not None and int(ckpt_hidden) != int(hidden_dim):
            raise ValueError(f"hidden_dim mismatch: ckpt={ckpt_hidden} arg={hidden_dim}")
    else:
        model.load_state_dict(blob, strict=False)

    model.eval()
    return model


@torch.no_grad()
def teacher_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
):
    return model(x, edge_index, edge_type)


@torch.no_grad()
def collect_branch_hidden(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hidden states after gat2 + norm2 + gelu (before head)."""
    import torch.nn.functional as F

    feat_ys = model.ys_encoder(x)
    feat_fs = model.fs_encoder(x)

    h = model.ys_gat1(feat_ys, edge_index, edge_type)
    h = model.dropout(F.gelu(model.ys_norm1(h)))
    h = model.ys_gat2(h, edge_index, edge_type)
    h_ys = model.dropout(F.gelu(model.ys_norm2(h)))

    h = model.fs_gat1(feat_fs, edge_index, edge_type)
    h = model.dropout(F.gelu(model.fs_norm1(h)))
    h = model.fs_gat2(h, edge_index, edge_type)
    h_fs = model.dropout(F.gelu(model.fs_norm2(h)))

    return h_ys, h_fs
