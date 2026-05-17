#!/usr/bin/env python3
"""对照 datagnn.csv 列出 val_mask 对应的行号与 element 特征，并统计验证集中不同 element 向量种数。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

ELEM_COLS = [f"element_{i}" for i in range(10)]


def main() -> None:
    base = Path(__file__).resolve().parent.parent
    default_mask = base / "r-gatPT" / "val_mask.pt"
    default_csv = base.parent / "datagnn.csv"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val-mask", type=Path, default=default_mask, help="val_mask.pt 路径")
    p.add_argument("--csv", type=Path, default=default_csv, help="datagnn.csv 路径")
    p.add_argument(
        "--element-round",
        type=int,
        default=6,
        help="统计「几种 element」时对每个维度四舍五入的小数位（避免浮点噪声）",
    )
    args = p.parse_args()

    val_mask = torch.load(args.val_mask, map_location="cpu")
    val_mask = val_mask.reshape(-1).bool()
    if not args.csv.is_file():
        raise FileNotFoundError(f"CSV 不存在: {args.csv}")
    if not args.val_mask.is_file():
        raise FileNotFoundError(f"val_mask 不存在: {args.val_mask}")

    df = pd.read_csv(args.csv)
    miss = [c for c in ELEM_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"CSV 缺少列: {miss}")

    n = len(df)
    if val_mask.numel() != n:
        raise ValueError(f"长度不一致: val_mask={val_mask.numel()}, CSV 行数={n}")

    val_idx = torch.where(val_mask)[0].tolist()
    rnd = args.element_round
    unique_keys: set[tuple[float, ...]] = set()

    print(f"val_mask: {args.val_mask}")
    print(f"CSV: {args.csv}")
    print(f"验证集样本数: {len(val_idx)}")
    print()
    # 表头占文件第 1 行；第 i 条数据（0-based iloc=i）在文件中为第 (i+2) 行
    print("节点索引(0-based)\t文件中行号(1-based,含表头)\t数据行号(1-based,不含表头)\telement_0..9")
    for i in val_idx:
        file_line = i + 2
        data_row_1based = i + 1
        row = df.iloc[i]
        elems = tuple(row[c] for c in ELEM_COLS)
        key = tuple(round(float(x), rnd) for x in elems)
        unique_keys.add(key)
        elem_str = ", ".join(f"{float(x):.6g}" for x in elems)
        print(f"{i}\t{file_line}\t{data_row_1based}\t{elem_str}")

    print()
    print(f"验证集中不同 element 向量种数（各维四舍五入到 {rnd} 位小数后去重）: {len(unique_keys)}")


if __name__ == "__main__":
    main()
