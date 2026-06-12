#!/usr/bin/env python3
"""
从 gnnDir/datagnn.csv 构建全量图数据，train:val = 1:0.2，输出到 modelAll/data/。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent
_GNN_DIR = _ROOT.parent / "gnnDir"
if str(_GNN_DIR) not in sys.path:
    sys.path.insert(0, str(_GNN_DIR))

from rgcn_dataloader import build_material_heterodata, print_sanity  # noqa: E402

TRAIN_PART = 1.0
VAL_PART = 0.2
TRAIN_RATIO = TRAIN_PART / (TRAIN_PART + VAL_PART)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="modelAll 全量 RGAT 数据构建（1:0.2 划分）")
    p.add_argument(
        "--csv",
        type=Path,
        default=_GNN_DIR / "datagnn.csv",
        help="gnnDir 特征表（默认 datagnn.csv）",
    )
    p.add_argument("--out-dir", type=Path, default=_ROOT / "data")
    p.add_argument("--element-thr", type=float, default=0.8)
    p.add_argument("--testenv-thr", type=float, default=0.8)
    p.add_argument("--coldway-thr", type=float, default=0.8)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--sanity", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data, ys, fs = build_material_heterodata(
        args.csv,
        element_thr=float(args.element_thr),
        testenv_thr=float(args.testenv_thr),
        coldway_thr=float(args.coldway_thr),
        train_mask_path=None,
        val_mask_path=None,
        train_ratio=TRAIN_RATIO,
        split_seed=int(args.split_seed),
    )

    graph_path = args.out_dir / "material_graph.pt"
    ys_path = args.out_dir / "ys.pt"
    fs_path = args.out_dir / "fs.pt"
    train_path = args.out_dir / "train_mask.pt"
    val_path = args.out_dir / "val_mask.pt"

    torch.save(data, graph_path)
    torch.save(ys, ys_path)
    torch.save(fs, fs_path)
    torch.save(data["sample"].train_mask, train_path)
    torch.save(data["sample"].val_mask, val_path)

    n = int(data["sample"].x.shape[0])
    n_train = int(data["sample"].train_mask.sum())
    n_val = int(data["sample"].val_mask.sum())
    print(f"[OK] nodes={n}  train={n_train}  val={n_val}  ratio≈1:{n_val / max(n_train, 1):.3f}")
    print(f"[OK] graph  -> {graph_path}")
    print(f"[OK] labels -> {ys_path}, {fs_path}")
    print(f"[OK] masks  -> {train_path}, {val_path}")
    if args.sanity:
        print_sanity(data, ys, fs)


if __name__ == "__main__":
    main()
