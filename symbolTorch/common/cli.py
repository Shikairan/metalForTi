"""Shared CLI for all symbolTorch experiments."""

from __future__ import annotations

import argparse
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .constants import (
    DEFAULT_CKPT,
    DEFAULT_DATA_DIR,
    HIDDEN_DIM_DEFAULT,
    PYTHON_BIN_HINT,
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
        help="完整输出目录（指定则不再自动建 runs/<run-name>/）",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="写入 runs/<run-name>/；默认按时间戳自动生成，不覆盖历史结果",
    )
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM_DEFAULT)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--sr-niterations", type=int, default=400)
    parser.add_argument(
        "--sr-maxsize",
        type=int,
        default=40,
        help="PySR maxsize（表达式复杂度上限，默认 40；--quick 时用较小值）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--head0-name",
        type=str,
        default=None,
        help="第一头方程名（默认 YS；uts 数据可用 UTS）",
    )
    parser.add_argument(
        "--head1-name",
        type=str,
        default=None,
        help="第二头方程名（默认 FS）",
    )
    parser.add_argument(
        "--include-val",
        action="store_true",
        help="Include validation nodes when collecting distillation samples",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast debug mode with fewer symbolic-regression iterations",
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


def _sanitize_run_name(name: str) -> str:
    s = re.sub(r"[^\w.\-]+", "_", name.strip())
    s = s.strip("._") or "run"
    return s[:120]


def make_default_run_name(args: argparse.Namespace) -> str:
    """UTC 时间戳 + 可选 quick/seed 标签，保证各次运行目录不冲突。"""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parts = [stamp]
    if getattr(args, "quick", False):
        parts.append("quick")
    seed = getattr(args, "seed", None)
    if seed is not None:
        parts.append(f"s{seed}")
    niter = getattr(args, "sr_niterations", None)
    if niter is not None and not getattr(args, "quick", False):
        parts.append(f"it{niter}")
    return "_".join(parts)


def update_latest_symlink(runs_root: Path, run_dir: Path) -> None:
    """维护 runs/latest → 本次输出目录（相对链接）。"""
    link = runs_root / "latest"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(run_dir.name, target_is_directory=True)
    except OSError:
        pass


def resolve_out_dir(args: argparse.Namespace, experiment_dir: Path) -> Path:
    """
    输出路径规则：
    - 若指定 --out-dir：原样使用
    - 否则：experiment/runs/<run-name>/
      run-name 来自 --run-name，或自动生成时间戳目录（不覆盖旧结果）
    """
    if args.out_dir is not None:
        return Path(args.out_dir)

    runs_root = experiment_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    name = args.run_name if getattr(args, "run_name", None) else make_default_run_name(args)
    name = _sanitize_run_name(str(name))
    out = runs_root / name
    if out.exists():
        k = 2
        while True:
            cand = runs_root / f"{name}_{k}"
            if not cand.exists():
                out = cand
                break
            k += 1
    out.mkdir(parents=True, exist_ok=True)
    update_latest_symlink(runs_root, out)
    return out


def print_python_hint() -> str:
    return f"Recommended interpreter: {PYTHON_BIN_HINT}"


def experiment_header(name: str) -> None:
    print(f"\n=== symbolTorch / {name} ===")
    print(print_python_hint())
