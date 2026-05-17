from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch


def _read_index_file(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Index file not found: {path}")
    values = []
    text = path.read_text(encoding="utf-8")
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        # allow comma separated or whitespace separated
        for token in s.replace(",", " ").split():
            values.append(int(token))
    if not values:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(values, dtype=np.int64)


def _build_random_masks(num_nodes: int, train_ratio: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(num_nodes)
    n_train = int(round(num_nodes * float(train_ratio)))
    n_train = max(1, min(num_nodes - 1, n_train))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[torch.from_numpy(train_idx)] = True
    val_mask[torch.from_numpy(val_idx)] = True
    return train_mask, val_mask


def _build_index_masks(num_nodes: int, train_idx: np.ndarray, val_idx: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[torch.from_numpy(train_idx)] = True
    val_mask[torch.from_numpy(val_idx)] = True
    return train_mask, val_mask


def _validate_index_range(name: str, idx: np.ndarray, num_nodes: int) -> None:
    if idx.ndim != 1:
        raise ValueError(f"{name} should be a 1D index array")
    if idx.size == 0:
        raise ValueError(f"{name} is empty")
    bad = idx[(idx < 0) | (idx >= num_nodes)]
    if bad.size > 0:
        show = bad[:10].tolist()
        raise ValueError(f"{name} has out-of-range indices (show up to 10): {show}")


def _validate_masks(train_mask: torch.Tensor, val_mask: torch.Tensor) -> None:
    if train_mask.dtype != torch.bool or val_mask.dtype != torch.bool:
        raise ValueError("Masks must be torch.bool")
    if train_mask.ndim != 1 or val_mask.ndim != 1:
        raise ValueError("Masks must be 1D")
    if train_mask.numel() != val_mask.numel():
        raise ValueError("train_mask and val_mask size mismatch")
    if bool((train_mask & val_mask).any()):
        raise ValueError("train_mask and val_mask overlap")
    cover = train_mask | val_mask
    if not bool(cover.all()):
        miss = int((~cover).sum().item())
        raise ValueError(f"train_mask and val_mask do not cover all nodes, missing={miss}")
    if int(train_mask.sum().item()) == 0 or int(val_mask.sum().item()) == 0:
        raise ValueError("train_mask and val_mask must both be non-empty")


def _save_mask(path: Path, mask: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mask, path)


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    default_out_dir = base / "gnndataPT" / "r-gnnPT"
    p = argparse.ArgumentParser(description="Generate train/val mask PT files (random or index mode).")
    p.add_argument("--mode", choices=["random", "index"], required=True)
    p.add_argument("--num-nodes", type=int, default=604)

    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--train-idx-path", type=Path, default=None)
    p.add_argument("--val-idx-path", type=Path, default=None)

    p.add_argument("--out-train-mask", type=Path, default=default_out_dir / "train_mask.pt")
    p.add_argument("--out-val-mask", type=Path, default=default_out_dir / "val_mask.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_nodes <= 1:
        raise SystemExit("--num-nodes must be > 1")

    if args.mode == "random":
        train_mask, val_mask = _build_random_masks(
            num_nodes=int(args.num_nodes),
            train_ratio=float(args.train_ratio),
            seed=int(args.seed),
        )
    else:
        if args.train_idx_path is None or args.val_idx_path is None:
            raise SystemExit("--mode index requires --train-idx-path and --val-idx-path")
        train_idx = _read_index_file(args.train_idx_path)
        val_idx = _read_index_file(args.val_idx_path)
        _validate_index_range("train_idx", train_idx, int(args.num_nodes))
        _validate_index_range("val_idx", val_idx, int(args.num_nodes))
        train_mask, val_mask = _build_index_masks(int(args.num_nodes), train_idx, val_idx)

    _validate_masks(train_mask, val_mask)
    _save_mask(args.out_train_mask, train_mask)
    _save_mask(args.out_val_mask, val_mask)

    print(f"Saved train_mask: {args.out_train_mask} sum={int(train_mask.sum().item())}")
    print(f"Saved val_mask:   {args.out_val_mask} sum={int(val_mask.sum().item())}")


if __name__ == "__main__":
    main()

