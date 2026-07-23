#!/usr/bin/env python3
"""
从 gnnDir/datacsv/datagnnUts.csv 构建图数据：604 节点全部参与训练（无 train/val 划分）。
输出到 modelAll/utsFsAll/data/。标签：UTS + FS。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent
_METAL_ROOT = _ROOT.parent.parent
_GNN_DIR = _METAL_ROOT / "gnnDir"
if str(_GNN_DIR) not in sys.path:
    sys.path.insert(0, str(_GNN_DIR))

from rgcn_dataloader import build_material_heterodata, print_sanity  # noqa: E402

# 仅用于调用 build_material_heterodata（其要求 0<ratio<1）；随后会覆盖为全 True
_DUMMY_TRAIN_RATIO = 503 / 604


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="modelAll/utsFsAll 数据构建：604 全部训练（UTS+FS，无 held-out val）"
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=_GNN_DIR / "datacsv" / "datagnnUts.csv",
        help="特征表（默认 gnnDir/datacsv/datagnnUts.csv）",
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
    if not args.csv.is_file():
        raise SystemExit(f"--csv not found: {args.csv}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data, _ys_unused, fs = build_material_heterodata(
        args.csv,
        element_thr=float(args.element_thr),
        testenv_thr=float(args.testenv_thr),
        coldway_thr=float(args.coldway_thr),
        train_mask_path=None,
        val_mask_path=None,
        train_ratio=_DUMMY_TRAIN_RATIO,
        split_seed=int(args.split_seed),
    )
    df = pd.read_csv(args.csv)
    if "UTS" not in df.columns:
        raise SystemExit(f"CSV missing UTS column: {args.csv}")
    uts = torch.tensor(df["UTS"].to_numpy(dtype="float32"), dtype=torch.float32)

    n = int(data["sample"].x.shape[0])
    # 604 全部参与训练；monitor_mask 与 train 相同，供日志/选优（无 held-out）
    all_mask = torch.ones(n, dtype=torch.bool)
    data["sample"].train_mask = all_mask
    data["sample"].val_mask = all_mask.clone()
    data["sample"].num_nodes = n

    graph_path = args.out_dir / "material_graph.pt"
    uts_path = args.out_dir / "uts.pt"
    fs_path = args.out_dir / "fs.pt"
    train_path = args.out_dir / "train_mask.pt"
    val_path = args.out_dir / "val_mask.pt"

    torch.save(data, graph_path)
    torch.save(uts, uts_path)
    torch.save(fs, fs_path)
    torch.save(all_mask, train_path)
    torch.save(all_mask.clone(), val_path)

    n_train = int(all_mask.sum())
    print(f"[OK] nodes={n}  train={n_train}  val=monitor_full_set={n_train}  (no held-out split)")
    print(f"[OK] graph  -> {graph_path}")
    print(f"[OK] labels -> {uts_path}, {fs_path}")
    print(f"[OK] masks  -> {train_path}, {val_path}")
    if args.sanity:
        print_sanity(data, uts, fs)


if __name__ == "__main__":
    main()
