"""Shared CLI for all symbolTorch experiments."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from .constants import (
    DEFAULT_CKPT,
    DEFAULT_DATA_DIR,
    HIDDEN_DIM_DEFAULT,
    PYTHON_BIN_HINT,
    QUICK_MAX_OUTPUT_DIM,
    SYMBOL_TORCH_ROOT,
)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory with material_graph.pt, ys.pt, fs.pt, masks",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=DEFAULT_CKPT,
        help="RGAT_Dual teacher checkpoint",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <experiment>/runs)",
    )
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM_DEFAULT)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--sr-niterations", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-val",
        action="store_true",
        help="Include validation nodes when collecting distillation samples",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Fast debug mode: fewer SR iterations, max_output_dim={QUICK_MAX_OUTPUT_DIM}",
    )
    parser.add_argument(
        "--max-output-dim",
        type=int,
        default=None,
        help="Limit SymTorch output dimensions (encoder). Default: all",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_out_dir(args: argparse.Namespace, experiment_dir: Path) -> Path:
    if args.out_dir is not None:
        return args.out_dir
    return experiment_dir / "runs"


def effective_max_output_dim(args: argparse.Namespace):
    if args.max_output_dim is not None:
        return args.max_output_dim
    if args.quick:
        from .constants import QUICK_MAX_OUTPUT_DIM

        return QUICK_MAX_OUTPUT_DIM
    return None


def print_python_hint() -> str:
    return f"Recommended interpreter: {PYTHON_BIN_HINT}"


def experiment_header(name: str) -> None:
    print(f"\n=== symbolTorch / {name} ===")
    print(print_python_hint())
